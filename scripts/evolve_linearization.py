# SPDX-License-Identifier: Apache-2.0
"""PoC: evolve a per-layer linearization mask on a real Qwen2 model (evolution × structure).

Applies the preserved evolutionary substrate to the analyzed model structure. Genome = one bit per
layer (linearize to constant-state attention | keep softmax). Fitness = number of layers linearized
(the memory objective: each linearized layer trades an O(T) KV cache for an O(1) state) minus a hard
penalty when the held-out quality loss exceeds a budget. The real-model Δnll is the fitness signal,
so the GA accounts for layer interactions that a per-layer greedy ranking misses. We report the
evolved mask vs a greedy-by-tolerance baseline at the same quality budget.

Honest scope: zero-shot linearization (no per-candidate distillation — that would shift the whole
frontier favorably and is the natural follow-up); tiny CPU model; perplexity proxy on one corpus.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from llcore.runtime.evolve_linearize import Genome, evolve
from llcore.runtime.linearize import LinearAttention
from llcore.runtime.loader import load_qwen2
from llcore.runtime.qwen2 import Qwen2Attention, Qwen2LM


def _set_linear(model: Qwen2LM, idxs: list[int]) -> dict[int, Qwen2Attention]:
    saved: dict[int, Qwen2Attention] = {}
    for i in idxs:
        attn = model.model.layers[i].self_attn
        if isinstance(attn, Qwen2Attention):
            saved[i] = attn
            model.model.layers[i].self_attn = LinearAttention.from_attention(attn, model.params)
    return saved


def _restore(model: Qwen2LM, saved: dict[int, Qwen2Attention]) -> None:
    for i, attn in saved.items():
        model.model.layers[i].self_attn = attn


@torch.no_grad()
def _nll(model: Qwen2LM, ids: torch.Tensor) -> float:
    logits = model(ids)
    assert isinstance(logits, torch.Tensor)
    return float(
        torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
        ).item()
    )


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="evolve a per-layer linearization mask")
    ap.add_argument("--model-dir", default="D:/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--text-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--budget", type=float, default=0.10, help="max held-out Δnll (nats) allowed")
    ap.add_argument("--penalty", type=float, default=50.0)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--generations", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/evolve_linearization")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model, tok, p = load_qwen2(args.model_dir)
    n_layer = p.n_layer
    print(f"[load] {n_layer} layers in {time.perf_counter()-t0:.0f}s", flush=True)

    text = Path(args.text_file).read_text(encoding="utf-8")[50000:]
    ids = tok(text, return_tensors="pt").input_ids[:, : args.n_tokens]  # type: ignore[operator]
    base = _nll(model, ids)
    print(f"[base] softmax nll={base:.4f} ppl={torch.tensor(base).exp():.2f}  eval_tokens={ids.size(1)}", flush=True)

    # per-layer single-linearization Δ (for the greedy baseline + reporting)
    per_layer = []
    for i in range(n_layer):
        saved = _set_linear(model, [i])
        per_layer.append(_nll(model, ids) - base)
        _restore(model, saved)
    order = sorted(range(n_layer), key=lambda i: per_layer[i])  # most tolerant first

    cache: dict[Genome, float] = {}
    evals = {"n": 0}

    def fitness(genome: Genome) -> float:
        if genome in cache:
            return cache[genome]
        idxs = [i for i, b in enumerate(genome) if b]
        saved = _set_linear(model, idxs)
        dn = _nll(model, ids) - base
        _restore(model, saved)
        f = float(len(idxs)) - args.penalty * max(0.0, dn - args.budget)
        cache[genome] = f
        evals["n"] += 1
        return f

    # greedy baseline: add layers in tolerance order while cumulative Δnll <= budget
    greedy: list[int] = []
    for i in order:
        saved = _set_linear(model, greedy + [i])
        dn = _nll(model, ids) - base
        _restore(model, saved)
        if dn <= args.budget:
            greedy.append(i)
    saved = _set_linear(model, greedy)
    greedy_dn = _nll(model, ids) - base
    _restore(model, saved)
    print(f"[greedy] linearized {len(greedy)} layers {sorted(greedy)}  Δnll={greedy_dn:+.4f}", flush=True)

    # evolutionary search
    te = time.perf_counter()
    res = evolve(fitness, n_layer, pop_size=args.pop, generations=args.generations, seed=args.seed)
    best = res["best_genome"]
    assert isinstance(best, tuple)
    evolved_idx = [i for i, b in enumerate(best) if b]
    saved = _set_linear(model, evolved_idx)
    evolved_dn = _nll(model, ids) - base
    _restore(model, saved)
    print(
        f"[evolve] linearized {len(evolved_idx)} layers {sorted(evolved_idx)}  Δnll={evolved_dn:+.4f}  "
        f"({evals['n']} real evals, {time.perf_counter()-te:.0f}s)",
        flush=True,
    )

    verdict = (
        "evolution linearized MORE layers at budget (exploited interactions)"
        if len(evolved_idx) > len(greedy)
        else ("tie with greedy" if len(evolved_idx) == len(greedy) else "greedy won (evolution under-searched)")
    )
    report = {
        "model_dir": args.model_dir,
        "n_layer": n_layer,
        "eval_tokens": int(ids.size(1)),
        "base_nll": base,
        "budget_nll": args.budget,
        "per_layer_delta": per_layer,
        "tolerance_order": order,
        "greedy": {"layers": sorted(greedy), "n": len(greedy), "delta_nll": greedy_dn},
        "evolved": {"layers": sorted(evolved_idx), "n": len(evolved_idx), "delta_nll": evolved_dn},
        "history": res["history"],
        "real_evals": evals["n"],
        "verdict": verdict,
    }
    (out / "evolve_linearization.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] {verdict}", flush=True)
    print(f"[done] wrote {out}/evolve_linearization.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
