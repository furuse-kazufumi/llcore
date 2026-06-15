# SPDX-License-Identifier: Apache-2.0
"""Head-to-head utilities for GPT vs recurrent constant-state LMs."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias, cast

import torch

from llcore.lm.data import encode_corpus, train_val_split
from llcore.lm.eval import held_out_report_any, passes_gate
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import TrainConfig, Trainer


ConstantStateLM: TypeAlias = RecurrentLM | RWKVLM


@dataclass
class CompareConfig:
    """Small comparison recipe intended for CPU smoke runs."""

    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 64
    state_size: int = 64
    max_iters: int = 120
    batch_size: int = 12
    eval_iters: int = 4
    throughput_prompt_lens: tuple[int, ...] | None = None
    throughput_new_tokens: int = 16
    throughput_repeats: int = 3
    seed: int = 1337

    def __post_init__(self) -> None:
        if self.n_head <= 0:
            raise ValueError(f"n_head must be > 0, got {self.n_head}")
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )
        if self.throughput_new_tokens < 1:
            raise ValueError(
                f"throughput_new_tokens must be >= 1, got {self.throughput_new_tokens}"
            )
        if self.throughput_repeats < 1:
            raise ValueError(
                f"throughput_repeats must be >= 1, got {self.throughput_repeats}"
            )
        if self.throughput_prompt_lens is not None and any(
            prompt_len < 1 for prompt_len in self.throughput_prompt_lens
        ):
            raise ValueError("throughput_prompt_lens must contain only positive lengths")


def build_models(vocab_size: int, cfg: CompareConfig) -> tuple[CharGPT, RecurrentLM, RWKVLM]:
    gpt = CharGPT(
        GPTConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_head=cfg.n_head,
            n_embd=cfg.n_embd,
        )
    )
    recurrent = RecurrentLM(
        RecurrentConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_embd=cfg.n_embd,
            state_size=cfg.state_size,
        )
    )
    rwkv = RWKVLM(
        RWKVConfig(
            vocab_size=vocab_size,
            block_size=cfg.block_size,
            n_layer=cfg.n_layer,
            n_embd=cfg.n_embd,
        )
    )
    return gpt, recurrent, rwkv


def gpt_kv_bytes(model: CharGPT, prompt_len: int) -> int:
    return 2 * model.config.n_layer * prompt_len * model.config.n_embd * 4


def constant_state_bytes(model: ConstantStateLM) -> int:
    if isinstance(model, RecurrentLM):
        recurrent_state = model.init_state(batch_size=1)
        return model.state_bytes(recurrent_state)
    rwkv_state = model.init_state(batch_size=1)
    return model.state_bytes(rwkv_state)


def _default_prompt_lens(block_size: int) -> tuple[int, ...]:
    return (1, 16, block_size, block_size * 4)


def _measure_generate_seconds(
    model: CharGPT | RecurrentLM | RWKVLM,
    prompt: torch.Tensor,
    *,
    new_tokens: int,
    repeats: int,
    seed: int,
) -> float | None:
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        _ = model.generate(prompt.clone(), max_new_tokens=new_tokens)
        best = float("inf")
        for rep in range(repeats):
            torch.manual_seed(seed + rep + 1)
            start = time.perf_counter()
            _ = model.generate(prompt.clone(), max_new_tokens=new_tokens)
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)
    finally:
        torch.set_num_threads(old_threads)
    if best == float("inf"):
        return None
    return best


def _measure_generate_tok_s(
    model: CharGPT | RecurrentLM | RWKVLM,
    prompt_len: int,
    *,
    new_tokens: int,
    repeats: int,
    seed: int,
) -> dict[str, float | int | bool | str | None]:
    if isinstance(model, CharGPT) and prompt_len > model.config.block_size:
        return {
            "executable": False,
            "prefill_s": None,
            "decode_tok_per_s": None,
            "total_tok_per_s": None,
            "effective_prompt_len": model.config.block_size,
            "note": "prompt_len exceeds block_size; GPT would crop context in generate(), so exact throughput is omitted",
        }
    prompt = torch.zeros((1, prompt_len), dtype=torch.long)
    total_short_s = _measure_generate_seconds(
        model, prompt, new_tokens=new_tokens, repeats=repeats, seed=seed
    )
    total_long_s = _measure_generate_seconds(
        model, prompt, new_tokens=new_tokens * 2, repeats=repeats, seed=seed + 1000
    )
    decode_tok_per_s: float | None = None
    prefill_s: float | None = None
    if total_short_s is not None and total_long_s is not None:
        decode_delta_s = total_long_s - total_short_s
        if decode_delta_s > 0:
            decode_tok_per_s = new_tokens / decode_delta_s
            prefill_s = max(0.0, total_short_s - (new_tokens / decode_tok_per_s))
    total_tok_per_s = None if total_short_s is None or total_short_s <= 0 else new_tokens / total_short_s
    return {
        "executable": True,
        "prefill_s": prefill_s,
        "decode_tok_per_s": decode_tok_per_s,
        "total_tok_per_s": total_tok_per_s,
        "effective_prompt_len": prompt_len,
        "note": "warmup + min-of-repeats; decode_tok_per_s is marginal from N vs 2N new tokens with torch.set_num_threads(1)",
    }


def _format_metric(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _render_ppl_table(result: dict[str, object]) -> str:
    reports = cast(dict[str, dict[str, object]], result["reports"])
    verdict = cast(dict[str, dict[str, object]], result["verdict"])
    lines = [
        "# Recurrent LM Head-to-Head",
        "",
        "| Model | PPL | Unigram PPL | Ratio vs GPT | Passes gate |",
        "| --- | ---: | ---: | ---: | :---: |",
    ]
    for name in ("gpt", "recurrent", "rwkv"):
        report = reports[name]
        verdict_row = verdict[name]
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _format_metric(report["model_ppl"]),
                    _format_metric(report["unigram_ppl"]),
                    _format_metric(verdict_row["ppl_ratio_vs_gpt"]),
                    "yes" if verdict_row["passes_unigram_gate"] else "no",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Caveats",
            "",
        ]
    )
    for caveat in cast(list[str], result["caveats"]):
        lines.append(f"- {caveat}")
    return "\n".join(lines) + "\n"


def _render_memory_curve_svg(result: dict[str, object]) -> str:
    memory = cast(dict[str, object], result["memory"])
    gpt_kv_bytes_by_t = cast(dict[str, int], memory["gpt_kv_bytes"])
    points = sorted((int(t), int(v)) for t, v in gpt_kv_bytes_by_t.items())
    widths = [t for t, _ in points]
    gpt_values = [v for _, v in points]
    recurrent_state_bytes = cast(int, memory["recurrent_state_bytes"])
    rwkv_state_bytes = cast(int, memory["rwkv_state_bytes"])
    recurrent_values = [recurrent_state_bytes] * len(points)
    rwkv_values = [rwkv_state_bytes] * len(points)
    all_values = gpt_values + recurrent_values + rwkv_values
    max_x = max(widths)
    max_y = max(all_values)

    width = 640
    height = 360
    left = 64
    right = 20
    top = 24
    bottom = 48
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: int) -> float:
        if max_x <= 1:
            return float(left)
        return left + (value - 1) * plot_width / (max_x - 1)

    def sy(value: int) -> float:
        if max_y <= 0:
            return float(height - bottom)
        return top + plot_height - (value * plot_height / max_y)

    def polyline(values: list[int]) -> str:
        return " ".join(f"{sx(t):.1f},{sy(v):.1f}" for t, v in zip(widths, values, strict=True))

    y_ticks = [0, max_y // 4, max_y // 2, (3 * max_y) // 4, max_y]
    x_labels = "".join(
        f'<text x="{sx(t):.1f}" y="{height - 20}" text-anchor="middle">{t}</text>' for t in widths
    )
    y_labels = "".join(
        f'<text x="{left - 8}" y="{sy(v) + 4:.1f}" text-anchor="end">{v}</text>'
        f'<line x1="{left}" y1="{sy(v):.1f}" x2="{width - right}" y2="{sy(v):.1f}" stroke="#e5e7eb" />'
        for v in y_ticks
    )

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">
<title id="title">LM memory at prompt length T</title>
<desc id="desc">GPT KV bytes rise linearly with prompt length while recurrent and RWKV stay constant.</desc>
<rect width="{width}" height="{height}" fill="#ffffff" />
<text x="{left}" y="16" font-family="Segoe UI, sans-serif" font-size="14" fill="#111827">Memory at prompt length T (bytes)</text>
<text x="{left}" y="{height - 4}" font-family="Segoe UI, sans-serif" font-size="11" fill="#6b7280">GPT beyond block_size is analytic projection only</text>
<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#111827" />
<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#111827" />
{y_labels}
{x_labels}
<polyline fill="none" stroke="#2563eb" stroke-width="3" points="{polyline(gpt_values)}" />
<polyline fill="none" stroke="#059669" stroke-width="3" points="{polyline(recurrent_values)}" />
<polyline fill="none" stroke="#dc2626" stroke-width="3" points="{polyline(rwkv_values)}" />
<text x="{width - 140}" y="26" font-family="Segoe UI, sans-serif" font-size="12" fill="#2563eb">GPT KV</text>
<text x="{width - 140}" y="44" font-family="Segoe UI, sans-serif" font-size="12" fill="#059669">Recurrent state</text>
<text x="{width - 140}" y="62" font-family="Segoe UI, sans-serif" font-size="12" fill="#dc2626">RWKV state</text>
</svg>
"""


def _write_compare_sidecars(result: dict[str, object], out_path: Path) -> None:
    out_path.with_suffix(".md").write_text(_render_ppl_table(result), encoding="utf-8")
    out_path.with_suffix(".svg").write_text(_render_memory_curve_svg(result), encoding="utf-8")


def compare_on_text(
    text: str,
    *,
    cfg: CompareConfig | None = None,
    out_path: Path | None = None,
) -> dict[str, object]:
    recipe = cfg or CompareConfig()
    torch.manual_seed(recipe.seed)
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=0.1)
    gpt, recurrent, rwkv = build_models(tok.vocab_size, recipe)
    train_cfg = TrainConfig(
        max_iters=recipe.max_iters,
        warmup_iters=max(1, recipe.max_iters // 10),
        lr_decay_iters=recipe.max_iters,
        batch_size=recipe.batch_size,
        eval_interval=recipe.max_iters,
        eval_iters=recipe.eval_iters,
        seed=recipe.seed,
    )

    Trainer(gpt, train_cfg).train(train_ids, val_ids)
    Trainer(recurrent, train_cfg).train(train_ids, val_ids)
    Trainer(rwkv, train_cfg).train(train_ids, val_ids)
    reports = {
        "gpt": held_out_report_any(gpt, train_ids, val_ids, tok.vocab_size, block_size=recipe.block_size),
        "recurrent": held_out_report_any(
            recurrent, train_ids, val_ids, tok.vocab_size, block_size=recipe.block_size
        ),
        "rwkv": held_out_report_any(rwkv, train_ids, val_ids, tok.vocab_size, block_size=recipe.block_size),
    }

    lengths = list(_default_prompt_lens(recipe.block_size))
    gpt_slope = gpt_kv_bytes(gpt, 2) - gpt_kv_bytes(gpt, 1)
    recurrent_bytes = constant_state_bytes(recurrent)
    rwkv_bytes = constant_state_bytes(rwkv)
    memory = {
        "notes": {
            "gpt_kv_bytes": (
                "Analytic projection 2*L*T*D*4 bytes. Values for T > block_size are not executable "
                "for this GPT config; they are counterfactual projection points only."
            )
        },
        "gpt_kv_bytes": {str(t): gpt_kv_bytes(gpt, t) for t in lengths},
        "gpt_kv_slope_bytes_per_token": gpt_slope,
        "recurrent_state_bytes": recurrent_bytes,
        "rwkv_state_bytes": rwkv_bytes,
        "recurrent_state_slope_bytes_per_token": 0,
        "rwkv_state_slope_bytes_per_token": 0,
    }
    throughput_prompt_lens = (
        recipe.throughput_prompt_lens
        if recipe.throughput_prompt_lens is not None
        else _default_prompt_lens(recipe.block_size)
    )
    throughput = {
        "notes": {
            "method": "warmup + min-of-repeats with torch.set_num_threads(1)",
            "new_tokens": recipe.throughput_new_tokens,
            "decode_estimator": "difference of total generate() time for N vs 2N new tokens on the same prompt",
        },
        "gpt": {
            str(t): _measure_generate_tok_s(
                gpt,
                t,
                new_tokens=recipe.throughput_new_tokens,
                repeats=recipe.throughput_repeats,
                seed=recipe.seed,
            )
            for t in throughput_prompt_lens
        },
        "recurrent": {
            str(t): _measure_generate_tok_s(
                recurrent,
                t,
                new_tokens=recipe.throughput_new_tokens,
                repeats=recipe.throughput_repeats,
                seed=recipe.seed,
            )
            for t in throughput_prompt_lens
        },
        "rwkv": {
            str(t): _measure_generate_tok_s(
                rwkv,
                t,
                new_tokens=recipe.throughput_new_tokens,
                repeats=recipe.throughput_repeats,
                seed=recipe.seed,
            )
            for t in throughput_prompt_lens
        },
    }
    caveats = []
    any_fail = any(not passes_gate(report["model_ppl"], report["unigram_ppl"]) for report in reports.values())
    all_fail = all(not passes_gate(report["model_ppl"], report["unigram_ppl"]) for report in reports.values())
    if any_fail and not all_fail:
        caveats.append(
            "At least one model fails the unigram gate; treat this run as undertrained unless a longer schedule confirms the ranking."
        )
    if all_fail:
        caveats.append(
            "All compared models fail the unigram gate; the capability ranking is not yet a publishable head-to-head verdict."
        )
    caveats.append(
        "GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements."
    )
    recurrent_break_even_prompt_len = max(1, (recurrent_bytes + gpt_slope - 1) // gpt_slope)
    rwkv_break_even_prompt_len = max(1, (rwkv_bytes + gpt_slope - 1) // gpt_slope)
    pareto = {
        "x_axis": "model_ppl",
        "y_axis": "memory_bytes_and_slope",
        "gpt_kv_slope_bytes_per_token": gpt_slope,
        "recurrent_state_slope_bytes_per_token": 0,
        "rwkv_state_slope_bytes_per_token": 0,
        "memory_winner_by_slope": "recurrent/rwkv",
        "recurrent_break_even_prompt_len_vs_gpt": recurrent_break_even_prompt_len,
        "rwkv_break_even_prompt_len_vs_gpt": rwkv_break_even_prompt_len,
        "capability_reference": "gpt",
        "capability_reference_note": "reference axis only; if unigram gate fails, this is not a publishable capability winner",
    }
    result: dict[str, object] = {
        "config": asdict(recipe),
        "reports": reports,
        "memory": memory,
        "throughput": throughput,
        "pareto": pareto,
        "caveats": caveats,
        "verdict": {
            name: {
                "passes_unigram_gate": passes_gate(report["model_ppl"], report["unigram_ppl"]),
                "ppl_ratio_vs_gpt": report["model_ppl"] / reports["gpt"]["model_ppl"],
            }
            for name, report in reports.items()
        },
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_compare_sidecars(result, out_path)
    return result
