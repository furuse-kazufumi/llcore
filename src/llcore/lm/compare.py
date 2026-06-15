# SPDX-License-Identifier: Apache-2.0
"""Head-to-head utilities for GPT vs recurrent constant-state LMs."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TypeAlias

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
    throughput_prompt_lens: tuple[int, ...] = (1, 16, 64, 256)
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
            "tok_per_s": None,
            "effective_prompt_len": model.config.block_size,
            "note": "prompt_len exceeds block_size; GPT would crop context in generate(), so exact throughput is omitted",
        }
    prompt = torch.zeros((1, prompt_len), dtype=torch.long)
    old_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        best = float("inf")
        for rep in range(repeats):
            torch.manual_seed(seed + rep)
            start = time.perf_counter()
            _ = model.generate(prompt.clone(), max_new_tokens=new_tokens)
            elapsed = time.perf_counter() - start
            best = min(best, elapsed)
    finally:
        torch.set_num_threads(old_threads)
    return {
        "executable": True,
        "tok_per_s": new_tokens / best if best > 0 else None,
        "effective_prompt_len": prompt_len,
        "note": "min-of-repeats, torch.set_num_threads(1)",
    }


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

    lengths = [1, 16, recipe.block_size, recipe.block_size * 4]
    gpt_slope = 2 * recipe.n_layer * recipe.n_embd * 4
    memory = {
        "notes": {
            "gpt_kv_bytes": (
                "Analytic projection 2*L*T*D*4 bytes. Values for T > block_size are not executable "
                "for this GPT config; they are counterfactual projection points only."
            )
        },
        "gpt_kv_bytes": {str(t): gpt_kv_bytes(gpt, t) for t in lengths},
        "gpt_kv_slope_bytes_per_token": gpt_slope,
        "recurrent_state_bytes": constant_state_bytes(recurrent),
        "rwkv_state_bytes": constant_state_bytes(rwkv),
        "recurrent_state_slope_bytes_per_token": 0,
        "rwkv_state_slope_bytes_per_token": 0,
    }
    throughput = {
        "notes": {
            "method": "generate() min-of-repeats with torch.set_num_threads(1)",
            "new_tokens": recipe.throughput_new_tokens,
        },
        "gpt": {
            str(t): _measure_generate_tok_s(
                gpt,
                t,
                new_tokens=recipe.throughput_new_tokens,
                repeats=recipe.throughput_repeats,
                seed=recipe.seed,
            )
            for t in recipe.throughput_prompt_lens
        },
        "recurrent": {
            str(t): _measure_generate_tok_s(
                recurrent,
                t,
                new_tokens=recipe.throughput_new_tokens,
                repeats=recipe.throughput_repeats,
                seed=recipe.seed,
            )
            for t in recipe.throughput_prompt_lens
        },
        "rwkv": {
            str(t): _measure_generate_tok_s(
                rwkv,
                t,
                new_tokens=recipe.throughput_new_tokens,
                repeats=recipe.throughput_repeats,
                seed=recipe.seed,
            )
            for t in recipe.throughput_prompt_lens
        },
    }
    caveats = []
    if any(not passes_gate(report["model_ppl"], report["unigram_ppl"]) for report in reports.values()):
        caveats.append(
            "At least one model fails the unigram gate; treat this run as undertrained unless a longer schedule confirms the ranking."
        )
    if all(not passes_gate(report["model_ppl"], report["unigram_ppl"]) for report in reports.values()):
        caveats.append(
            "All compared models fail the unigram gate; the capability ranking is not yet a publishable head-to-head verdict."
        )
    caveats.append(
        "GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements."
    )
    pareto = {
        "x_axis": "model_ppl",
        "y_axis": "memory_bytes_and_slope",
        "gpt_kv_slope_bytes_per_token": gpt_slope,
        "recurrent_state_slope_bytes_per_token": 0,
        "rwkv_state_slope_bytes_per_token": 0,
        "memory_winner": "recurrent/rwkv",
        "capability_reference": "gpt",
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
    return result
