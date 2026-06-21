# SPDX-License-Identifier: Apache-2.0
"""Level-2 NAS: evolve a per-layer mixer choice (softmax / sliding-window / linear) on a real model.

The binary linearization mask was greedy-friendly (near-additive cost, monotone budget). Level-2
gives each layer THREE mixers with different memory/quality profiles — full softmax (O(T) KV, best
quality), sliding-window softmax (O(window) KV, local quality), linear attention (O(d²) state,
O(1) in T). The genome is one categorical gene per layer; fitness = % of attention KV/state memory
saved at a reference context length, minus a penalty when held-out Δnll exceeds a budget. This 3^L
space with interacting choices is where evolution can beat a greedy assignment. We report the
evolved configuration vs a budget-greedy baseline.

Honest scope: zero-shot (no per-candidate distillation), tiny CPU model, perplexity proxy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from llcore.runtime.evolve_linearize import CatGenome, evolve_categorical
from llcore.runtime.linearize import LinearAttention, SlidingWindowAttention
from llcore.runtime.loader import load_qwen2
from llcore.runtime.qwen2 import Qwen2Attention

MIXERS = ("softmax", "sliding", "linear")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="Level-2 NAS over per-layer mixer choice")
    ap.add_argument("--model-dir", default="D:/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--text-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--ref-context", type=int, default=2048, help="context length for the memory model")
    ap.add_argument("--budget", type=float, default=0.15, help="max held-out Δnll (nats)")
    ap.add_argument("--penalty", type=float, default=5.0)
    ap.add_argument("--pop", type=int, default=16)
    ap.add_argument("--generations", type=int, default=14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seed-greedy", action="store_true", help="inject the greedy solution into the initial population (memetic)")
    ap.add_argument("--out", default="out/nas_level2")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, p = load_qwen2(args.model_dir)
    n_layer = p.n_layer
    originals = {i: model.model.layers[i].self_attn for i in range(n_layer)}
    assert all(isinstance(a, Qwen2Attention) for a in originals.values())

    # per-mixer per-layer memory (bytes) at the reference context length
    Tref = args.ref_context
    mem_softmax = 2 * p.n_kv_head * Tref * p.head_dim * 4
    mem_sliding = 2 * p.n_kv_head * min(args.window, Tref) * p.head_dim * 4
    mem_linear = p.n_head * p.head_dim * p.head_dim * 4 + p.n_head * p.head_dim * 4
    mem_opt = (mem_softmax, mem_sliding, mem_linear)
    mem_all_softmax = mem_softmax * n_layer

    def set_genome(genome: CatGenome) -> None:
        for i, opt in enumerate(genome):
            src = originals[i]
            assert isinstance(src, Qwen2Attention)
            if opt == 0:
                model.model.layers[i].self_attn = src
            elif opt == 1:
                model.model.layers[i].self_attn = SlidingWindowAttention.from_attention(src, p, window=args.window)
            else:
                model.model.layers[i].self_attn = LinearAttention.from_attention(src, p)

    def restore() -> None:
        for i in range(n_layer):
            model.model.layers[i].self_attn = originals[i]

    @torch.no_grad()
    def nll(ids: torch.Tensor) -> float:
        logits = model(ids)
        assert isinstance(logits, torch.Tensor)
        return float(
            torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)).item()
        )

    text = Path(args.text_file).read_text(encoding="utf-8")[50000:]
    ids = tok(text, return_tensors="pt").input_ids[:, : args.n_tokens]
    base = nll(ids)
    print(f"[base] all-softmax nll={base:.4f} ppl={torch.tensor(base).exp():.2f}  mem/layer={mem_softmax}B @ctx{Tref}", flush=True)

    cache: dict[CatGenome, tuple[float, float]] = {}
    evals = {"n": 0}

    def evaluate(genome: CatGenome) -> tuple[float, float]:
        """Return (pct_mem_saved, delta_nll)."""
        if genome in cache:
            return cache[genome]
        set_genome(genome)
        dn = nll(ids) - base
        restore()
        mem = sum(mem_opt[o] for o in genome)
        pct_saved = 100.0 * (mem_all_softmax - mem) / mem_all_softmax
        cache[genome] = (pct_saved, dn)
        evals["n"] += 1
        return pct_saved, dn

    def fitness(genome: CatGenome) -> float:
        pct, dn = evaluate(genome)
        return float(pct - args.penalty * 100.0 * max(0.0, dn - args.budget))

    # budget-greedy baseline: per layer (tolerance order) pick the cheapest mixer that keeps
    # cumulative Δnll <= budget (try linear, then sliding, then softmax).
    # tolerance order from single-layer linear Δ
    single = []
    for i in range(n_layer):
        set_genome(tuple(2 if j == i else 0 for j in range(n_layer)))
        single.append(nll(ids) - base)
        restore()
    order = sorted(range(n_layer), key=lambda i: single[i])
    greedy = [0] * n_layer
    for i in order:
        for opt in (2, 1):  # cheapest memory first: linear, then sliding
            trial = list(greedy)
            trial[i] = opt
            _, dn = evaluate(tuple(trial))
            if dn <= args.budget:
                greedy[i] = opt
                break
    g_pct, g_dn = evaluate(tuple(greedy))
    print(f"[greedy] {MIXERS}-counts={[greedy.count(k) for k in range(3)]}  mem_saved={g_pct:.1f}%  Δnll={g_dn:+.4f}", flush=True)

    # evolutionary search (optionally memetic: seed with the greedy solution)
    te = time.perf_counter()
    seeds = [tuple(greedy)] if args.seed_greedy else None
    res = evolve_categorical(
        fitness, n_layer, 3, pop_size=args.pop, generations=args.generations, seed=args.seed, seed_genomes=seeds
    )
    best = res["best_genome"]
    assert isinstance(best, tuple)
    e_pct, e_dn = evaluate(best)
    counts = [list(best).count(k) for k in range(3)]
    print(
        f"[evolve] counts(softmax/sliding/linear)={counts}  mem_saved={e_pct:.1f}%  Δnll={e_dn:+.4f}  "
        f"({evals['n']} real evals, {time.perf_counter()-te:.0f}s)",
        flush=True,
    )

    verdict = (
        f"evolution saved {e_pct - g_pct:+.1f}pp more memory than greedy at the same budget"
        if e_pct > g_pct + 0.5
        else ("evolution tied greedy" if abs(e_pct - g_pct) <= 0.5 else f"greedy won by {g_pct - e_pct:.1f}pp")
    )
    report = {
        "model_dir": args.model_dir,
        "n_layer": n_layer,
        "mixers": MIXERS,
        "window": args.window,
        "ref_context": Tref,
        "mem_bytes_per_layer": {"softmax": mem_softmax, "sliding": mem_sliding, "linear": mem_linear},
        "budget_nll": args.budget,
        "base_nll": base,
        "greedy": {"genome": greedy, "counts": [greedy.count(k) for k in range(3)], "pct_mem_saved": g_pct, "delta_nll": g_dn},
        "evolved": {"genome": list(best), "counts": counts, "pct_mem_saved": e_pct, "delta_nll": e_dn},
        "history": res["history"],
        "real_evals": evals["n"],
        "verdict": verdict,
    }
    (out / "nas_level2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[verdict] {verdict}", flush=True)
    print(f"[done] wrote {out}/nas_level2.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
