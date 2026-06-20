# SPDX-License-Identifier: Apache-2.0
"""Pareto-frontier Level-2 NAS: evolve the whole memory<->quality tradeoff curve.

``nas_level2`` optimized a single scalar (memory saved at one Delta-nll budget). Real
deployment wants the *frontier*: for every quality budget, the cheapest mixer
assignment. This script builds that frontier two ways and compares them by 2-D
hypervolume:

1. **Greedy frontier (baseline)** -- run the budget-greedy assignment of
   ``nas_level2`` at a sweep of Delta-nll budgets; each budget yields one
   ``(% memory saved, Delta-nll)`` point.
2. **Memetic NSGA-II frontier** -- seed a multi-objective GA with those greedy
   points and let it refine the whole front (:func:`evolve_multiobjective`).

The memetic front should dominate or match the greedy one. If it merely ties,
the landscape is separable and greedy already traces the frontier -- an honest
negative, the same lens as ``project_llcore_evolvable_llm_replan``.

Honest scope: zero-shot (no per-candidate distillation), tiny CPU model,
perplexity proxy. Per-layer mixer in {softmax, sliding-window, linear}.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import cast

import torch

from llcore.runtime.evolve_linearize import CatGenome, evolve_multiobjective, pareto_front
from llcore.runtime.linearize import LinearAttention, SlidingWindowAttention
from llcore.runtime.loader import load_qwen2
from llcore.runtime.qwen2 import Qwen2Attention

MIXERS = ("softmax", "sliding", "linear")


def hypervolume_2d(points: list[tuple[float, float]], ref: tuple[float, float]) -> float:
    """2-D hypervolume (area dominated) for a **maximization** front; ``ref`` is the
    shared lower-left reference point so two fronts are directly comparable."""
    hv = 0.0
    y_prev = ref[1]
    for x, y in sorted(points, key=lambda p: -p[0]):
        if y > y_prev:
            hv += (x - ref[0]) * (y - y_prev)
            y_prev = y
    return hv


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="Pareto-frontier Level-2 NAS (memory vs quality)")
    ap.add_argument("--model-dir", default="D:/models/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--text-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--n-tokens", type=int, default=256)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--ref-context", type=int, default=2048)
    ap.add_argument("--budgets", default="0.02,0.05,0.10,0.15,0.25,0.50",
                    help="comma Delta-nll budgets for the greedy baseline frontier (also GA seeds)")
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--generations", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out/nas_pareto")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tok, p = load_qwen2(args.model_dir)
    n_layer = p.n_layer
    originals = {i: model.model.layers[i].self_attn for i in range(n_layer)}
    assert all(isinstance(a, Qwen2Attention) for a in originals.values())

    tref = args.ref_context
    mem_softmax = 2 * p.n_kv_head * tref * p.head_dim * 4
    mem_sliding = 2 * p.n_kv_head * min(args.window, tref) * p.head_dim * 4
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
            torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1)
            ).item()
        )

    text = Path(args.text_file).read_text(encoding="utf-8")[50000:]
    ids = tok(text, return_tensors="pt").input_ids[:, : args.n_tokens]  # type: ignore[operator]
    base = nll(ids)
    print(f"[base] all-softmax nll={base:.4f} ppl={torch.tensor(base).exp():.2f}", flush=True)

    cache: dict[CatGenome, tuple[float, float]] = {}

    def measure(genome: CatGenome) -> tuple[float, float]:
        """Return ``(pct_mem_saved, delta_nll)`` for a genome (memoized)."""
        if genome in cache:
            return cache[genome]
        set_genome(genome)
        dn = nll(ids) - base
        restore()
        mem = sum(mem_opt[o] for o in genome)
        pct = 100.0 * (mem_all_softmax - mem) / mem_all_softmax
        cache[genome] = (pct, dn)
        return pct, dn

    # single-layer linear tolerance order (shared by every greedy budget)
    single: list[float] = []
    for i in range(n_layer):
        set_genome(tuple(2 if j == i else 0 for j in range(n_layer)))
        single.append(nll(ids) - base)
        restore()
    order = sorted(range(n_layer), key=lambda i: single[i])

    def greedy_at(budget: float) -> CatGenome:
        g = [0] * n_layer
        for i in order:
            for opt in (2, 1):  # cheapest memory first: linear, then sliding
                trial = list(g)
                trial[i] = opt
                if measure(tuple(trial))[1] <= budget:
                    g[i] = opt
                    break
        return tuple(g)

    budgets = [float(b) for b in str(args.budgets).split(",") if b.strip()]
    greedy_seeds: list[CatGenome] = []
    greedy_points: list[tuple[float, float]] = []
    for b in budgets:
        gg = greedy_at(b)
        if gg not in greedy_seeds:
            greedy_seeds.append(gg)
        greedy_points.append(measure(gg))
    print(f"[greedy] {len(greedy_seeds)} distinct configs over {len(budgets)} budgets", flush=True)

    # memetic NSGA-II: maximize (pct_saved, -delta_nll), seeded with the greedy configs
    te = time.perf_counter()
    res = evolve_multiobjective(
        lambda g: (measure(g)[0], -measure(g)[1]),
        n_layer, 3, pop_size=args.pop, generations=args.generations,
        seed=args.seed, seed_genomes=greedy_seeds,
    )
    front = cast("list[tuple[CatGenome, tuple[float, ...]]]", res["front"])
    evolved_points = [measure(g) for g, _ in front]
    print(f"[evolve] {len(front)} Pareto configs ({len(cache)} real evals, {time.perf_counter() - te:.0f}s)",
          flush=True)

    # compare the two fronts by 2-D hypervolume with a shared reference point
    all_dn = [dn for _, dn in greedy_points + evolved_points] or [0.0]
    ref = (0.0, -(max(all_dn) + 1e-6))
    g_hv = hypervolume_2d([(pc, -dn) for pc, dn in greedy_points], ref)
    e_hv = hypervolume_2d([(pc, -dn) for pc, dn in evolved_points], ref)
    if e_hv > g_hv * 1.005:
        verdict = f"memetic frontier dominates greedy: hypervolume +{100 * (e_hv - g_hv) / max(g_hv, 1e-9):.1f}%"
    elif e_hv < g_hv * 0.995:
        verdict = (f"greedy frontier wins by {100 * (g_hv - e_hv) / max(g_hv, 1e-9):.1f}% hypervolume "
                   f"(separable landscape)")
    else:
        verdict = "memetic frontier ties greedy (separable: greedy already traces the frontier)"

    def fmt(points: list[tuple[float, float]]) -> list[dict[str, float]]:
        items: list[tuple[int, tuple[float, ...]]] = [(i, (pc, -dn)) for i, (pc, dn) in enumerate(points)]
        keep = pareto_front(items)
        return sorted(
            ({"pct_mem_saved": points[i][0], "delta_nll": points[i][1]} for i, _ in keep),
            key=lambda d: d["pct_mem_saved"],
        )

    greedy_fmt = fmt(greedy_points)
    evolved_fmt = fmt(evolved_points)
    report = {
        "model_dir": args.model_dir,
        "n_layer": n_layer,
        "mixers": MIXERS,
        "ref_context": tref,
        "base_nll": base,
        "budgets": budgets,
        "greedy_frontier": greedy_fmt,
        "evolved_frontier": evolved_fmt,
        "hypervolume": {"greedy": g_hv, "evolved": e_hv},
        "real_evals": len(cache),
        "history": res["history"],
        "verdict": verdict,
    }
    (out / "nas_pareto.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[greedy frontier]")
    for d in greedy_fmt:
        print(f"  saved {d['pct_mem_saved']:5.1f}%  delta_nll {d['delta_nll']:+.4f}")
    print("[evolved frontier]")
    for d in evolved_fmt:
        print(f"  saved {d['pct_mem_saved']:5.1f}%  delta_nll {d['delta_nll']:+.4f}")
    print(f"\n[verdict] {verdict}")
    print(f"[done] wrote {out}/nas_pareto.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
