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

**Distillation-aware frontier (``--distill``)** -- step ② of the conversational-llcore
line. Instead of the *zero-shot* linear attention, the linear mixer option uses a
per-layer **distilled** student (a small learnable feature map trained to match the
softmax teacher's output on a held-out calibration window; see
:func:`llcore.runtime.distill.distill_all_layers`). Each converted layer keeps the same
constant-state memory, so distillation can only improve the *quality* axis -- the frontier
shifts out. The script then reports the 2-D hypervolume **right-shift** of the distilled
memetic frontier vs the zero-shot one (:func:`frontier_right_shift`). The calibration window
is **disjoint** from the eval window so the recovery is measured on unseen text, and
per-layer distillation is independent (joint multi-layer distillation -- where errors may
compound -- is the separate step ②(ii)).

Honest scope: tiny CPU model, perplexity proxy. Per-layer mixer in {softmax, sliding-window,
linear}. Without ``--distill`` the output JSON is unchanged (backward compatible).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import cast

import numpy as np
import torch

from llcore.runtime.distill import distill_all_layers
from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache
from llcore.runtime.eval_proxy import (
    base_window_losses,
    bootstrap_hv_gain,
    build_passkey_prompt,
    build_proxy_v2_report,
    context_sweep,
    genome_attn_kl,
    honest_verdict,
    kendall_tau,
    make_windows,
    needle_horizon,
    paired_window_deltas,
    reeval_frontier,
    right_shift_ci,
    score_needle,
)
from llcore.runtime.evolve_linearize import CatGenome, dominates, evolve_multiobjective
from llcore.runtime.linearize import LinearAttention, SlidingWindowAttention
from llcore.runtime.loader import load_qwen2
from llcore.runtime.pareto_metrics import frontier_right_shift, hypervolume_2d
from llcore.runtime.qwen2 import Qwen2Attention, Qwen2LM

MIXERS = ("softmax", "sliding", "linear")

SetGenome = Callable[[CatGenome, bool], None]


def _proxy_v2_rigorous(
    model: Qwen2LM,
    tok: object,
    set_genome: SetGenome,
    restore: Callable[[], None],
    ids_all: torch.Tensor,
    args: argparse.Namespace,
    mem_opt: tuple[int, int, int],
    mem_all_softmax: int,
    fast_delta_cache: dict[tuple[CatGenome, bool], np.ndarray],
    zs_front: list[CatGenome],
    zs_greedy: list[CatGenome],
    ds_front: list[CatGenome] | None,
) -> dict[str, object]:
    """The proxy-v2 rigorous tier: runs ONCE on the handful of frontier genomes (never in the search).

    Re-evaluates greedy + memetic frontiers on a FRESH disjoint holdout pool (removes winner's curse),
    puts a paired-bootstrap CI on the memetic-vs-greedy hypervolume gain, sweeps the context length on
    the most aggressive genome (regime dependence), runs the attention-KL diagnostic, and optionally a
    cross-corpus holdout + a long-context needle probe. :func:`honest_verdict` turns every guard into
    one disclosed verdict. The headline uses HOLDOUT numbers only.
    """

    def pct_of(g: CatGenome) -> float:
        return 100.0 * (mem_all_softmax - sum(mem_opt[o] for o in g)) / mem_all_softmax

    def _pw(row: dict[str, object]) -> np.ndarray:
        return cast(np.ndarray, row["per_window_holdout"])

    def _g(row: dict[str, object]) -> CatGenome:
        return cast(CatGenome, row["genome"])

    n_tok = int(ids_all.size(1))
    used = args.fast_windows * args.inner_context
    hoff = args.holdout_offset if args.holdout_offset is not None else used
    holdout = make_windows(ids_all, args.inner_context, args.holdout_windows, offset=hoff)
    if not holdout:  # corpus exhausted after the fast pool — wrap with half-stride rather than fail
        holdout = make_windows(
            ids_all, args.inner_context, args.holdout_windows,
            offset=min(hoff, max(0, n_tok - args.inner_context)), stride=max(1, args.inner_context // 2),
        )
    hbase = base_window_losses(model, restore, holdout)
    k_hold = len(holdout)

    mem_genomes = [g for g in dict.fromkeys(zs_front) if (g, False) in fast_delta_cache]
    greedy_genomes = [g for g in dict.fromkeys(zs_greedy) if (g, False) in fast_delta_cache]
    search_m = {g: fast_delta_cache[(g, False)] for g in mem_genomes}
    search_g = {g: fast_delta_cache[(g, False)] for g in greedy_genomes}
    rows_m = reeval_frontier(model, set_genome, restore, mem_genomes, pct_of, search_m, holdout, hbase)
    rows_g = reeval_frontier(model, set_genome, restore, greedy_genomes, pct_of, search_g, holdout, hbase)

    pw: dict[CatGenome, np.ndarray] = {}
    pct: dict[CatGenome, float] = {}
    for row in rows_m + rows_g:
        pw[_g(row)] = _pw(row)
        pct[_g(row)] = float(cast(float, row["pct"]))
    hv_gain = bootstrap_hv_gain(
        {repr(k): v for k, v in pw.items()},
        {repr(k): v for k, v in pct.items()},
        greedy_ids=[repr(_g(r)) for r in rows_g],
        memetic_ids=[repr(_g(r)) for r in rows_m],
    )

    tau: float | None = None
    if len(rows_m) >= 3:
        tau = kendall_tau(
            np.array([float(cast(float, r["delta_nll_selection"])) for r in rows_m]),
            np.array([float(cast(float, r["delta_nll_heldout"])) for r in rows_m]),
        )

    sweep_lengths = tuple(int(x) for x in str(args.context_sweep).split(",") if x.strip())
    agg = max(mem_genomes, key=pct_of) if mem_genomes else None
    sweep: dict[int, dict[str, float]] = {}
    attn_kl: dict[str, object] | None = None
    if agg is not None:
        sweep = context_sweep(
            model, set_genome, restore, agg, ids_all,
            lengths=sweep_lengths, K=args.holdout_windows, offset=hoff,
        )
        attn_kl = genome_attn_kl(
            model, set_genome, restore, agg, ids_all[:, hoff : hoff + 256], t_kl=256
        )

    rshift_ci: dict[str, object] | None = None
    if ds_front is not None:
        ds_genomes = [g for g in dict.fromkeys(ds_front) if (g, True) in fast_delta_cache]
        ds_search = {g: fast_delta_cache[(g, True)] for g in ds_genomes}
        ds_rows = reeval_frontier(
            model, set_genome, restore, ds_genomes, pct_of, ds_search, holdout, hbase, use_distill=True
        )
        if rows_m and ds_rows:
            rshift_ci = right_shift_ci(
                {repr(_g(r)): _pw(r) for r in rows_m},
                {repr(_g(r)): _pw(r) for r in ds_rows},
                {repr(_g(r)): float(cast(float, r["pct"])) for r in rows_m},
                {repr(_g(r)): float(cast(float, r["pct"])) for r in ds_rows},
            )

    cross: dict[str, object] | None = None
    if args.cross_corpus:
        ctext = Path(args.cross_corpus).read_text(encoding="utf-8")[50000:]
        cids = tok(ctext, return_tensors="pt").input_ids  # type: ignore[operator]
        cwin = make_windows(cids, args.inner_context, args.holdout_windows, offset=0)
        if cwin:
            cbase = base_window_losses(model, restore, cwin)
            crows_m = reeval_frontier(model, set_genome, restore, mem_genomes, pct_of, search_m, cwin, cbase)
            crows_g = reeval_frontier(model, set_genome, restore, greedy_genomes, pct_of, search_g, cwin, cbase)
            cgain = bootstrap_hv_gain(
                {repr(_g(r)): _pw(r) for r in crows_m + crows_g},
                {repr(_g(r)): float(cast(float, r["pct"])) for r in crows_m + crows_g},
                greedy_ids=[repr(_g(r)) for r in crows_g],
                memetic_ids=[repr(_g(r)) for r in crows_m],
            )
            cross = {
                "corpus": args.cross_corpus,
                "n_windows": float(len(cwin)),
                "hv_gain_ci": cgain,
                "frontier_holdout": [
                    {k: v for k, v in r.items() if k != "per_window_holdout"} for r in crows_m
                ],
            }

    needle: dict[str, object] | None = None
    if args.needle and agg is not None:
        nlens = tuple(int(x) for x in str(args.needle_lengths).split(",") if x.strip())
        filler = ids_all[:, hoff : hoff + max(nlens) + 64]
        depths = (0.0, 0.5, 0.9)
        base_acc: dict[tuple[int, float], float] = {}
        all_softmax = tuple(0 for _ in agg)
        set_genome(all_softmax, False)
        for length in nlens:
            for depth in depths:
                try:
                    pid, span = build_passkey_prompt(tok, filler, length, depth_frac=depth)
                except ValueError:
                    continue
                base_acc[(length, depth)] = score_needle(model, pid, span)["argmax_acc"]
        restore()
        needle = needle_horizon(
            model, set_genome, restore, agg, tok, filler, base_acc,
            lengths=nlens, depths=depths,
        )

    halfwidths = [
        abs(float(cast(float, r["ci_hi"])) - float(cast(float, r["ci_lo"]))) / 2.0 for r in rows_m
    ]
    floor = float(np.median(halfwidths)) if halfwidths else 0.0
    verdict = honest_verdict(hv_gain, rows_m, tau, floor)
    if k_hold < 12:
        verdict = {**verdict, "ci_reliability": f"point estimate, CI unreliable (K={k_hold}<12)"}

    report = build_proxy_v2_report(
        inner_context=args.inner_context,
        context_sweep=sweep,
        frontier_holdout=rows_m,
        hv_gain_ci=hv_gain,
        right_shift_ci=rshift_ci,
        needle=needle,
        attention_kl=attn_kl,
        proxy_vs_judge_tau=tau,
        cross_corpus=cross,
    )
    report["holdout_offset"] = hoff
    report["holdout_windows"] = float(k_hold)
    report["fast_windows"] = float(args.fast_windows)
    report["aggressive_genome_pct"] = pct_of(agg) if agg is not None else 0.0
    report["verdict"] = verdict
    return report


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
    ap.add_argument("--distill", action="store_true",
                    help="also build a distillation-aware frontier (linear layers use a per-layer "
                         "distilled student) and report the hypervolume right-shift vs zero-shot")
    ap.add_argument("--distill-steps", type=int, default=200)
    ap.add_argument("--distill-lr", type=float, default=5e-2)
    ap.add_argument("--distill-tokens", type=int, default=256,
                    help="held-out calibration tokens for distillation (window AFTER the eval window)")
    ap.add_argument("--out", default="out/nas_pareto")
    # --- proxy-v2 (all default OFF => v1 JSON byte-identical) ---
    ap.add_argument("--proxy-v2", action="store_true",
                    help="use the statistically honest proxy: paired multi-window bootstrap-CI inner "
                         "loop + frontier-only holdout re-eval (winner's-curse removed), context sweep, "
                         "needle probe, attention-KL diagnostic. Adds report['proxy_v2'].")
    ap.add_argument("--inner-context", type=int, default=1024,
                    help="proxy-v2 inner-loop context length (right of the 227-tok crossover and the "
                         "degradation onset)")
    ap.add_argument("--fast-windows", type=int, default=8, help="proxy-v2 inner-loop paired windows (K)")
    ap.add_argument("--holdout-windows", type=int, default=12,
                    help="proxy-v2 holdout/context-sweep windows; <12 downgrades CIs to unreliable")
    ap.add_argument("--holdout-offset", type=int, default=None,
                    help="token offset for the fresh holdout pool (default: just after the fast pool)")
    ap.add_argument("--context-sweep", default="256,512,1024,2048",
                    help="proxy-v2 context lengths swept on the most aggressive frontier genome")
    ap.add_argument("--cross-corpus", default=None,
                    help="proxy-v2 cross-corpus generalization holdout (a DISJOINT corpus file)")
    ap.add_argument("--needle", action="store_true",
                    help="proxy-v2 long-context passkey/needle retrieval probe (slow; frontier only)")
    ap.add_argument("--needle-lengths", default="2048,4096", help="proxy-v2 needle context lengths")
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

    # per-layer distilled students, filled by --distill (referenced by set_genome's linear branch)
    distilled: dict[int, LinearAttention] = {}

    def set_genome(genome: CatGenome, use_distill: bool = False) -> None:
        for i, opt in enumerate(genome):
            src = originals[i]
            assert isinstance(src, Qwen2Attention)
            if opt == 0:
                model.model.layers[i].self_attn = src
            elif opt == 1:
                model.model.layers[i].self_attn = SlidingWindowAttention.from_attention(src, p, window=args.window)
            else:
                model.model.layers[i].self_attn = (
                    distilled[i] if use_distill else LinearAttention.from_attention(src, p)
                )

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
    ids = tok(text, return_tensors="pt").input_ids[:, : args.n_tokens]
    base = nll(ids)
    print(f"[base] all-softmax nll={base:.4f} ppl={torch.tensor(base).exp():.2f}", flush=True)

    cache: dict[tuple[CatGenome, bool], tuple[float, float]] = {}
    # proxy-v2 side cache: the per-window Δnll VECTOR behind each genome's scalar (reused by the
    # rigorous tier so the holdout re-eval and hypervolume-gain CI need no extra inner-loop forwards).
    fast_delta_cache: dict[tuple[CatGenome, bool], np.ndarray] = {}

    ids_all = tok(text, return_tensors="pt").input_ids
    fast_windows: list[torch.Tensor] = []
    fast_base: list[torch.Tensor] = []
    if args.proxy_v2:
        fast_windows = make_windows(ids_all, args.inner_context, args.fast_windows, offset=0)
        if not fast_windows:
            raise SystemExit(
                f"corpus too short for {args.fast_windows} non-overlapping windows of "
                f"{args.inner_context} tokens ({int(ids_all.size(1))} tokens available)"
            )
        fast_base = base_window_losses(model, restore, fast_windows)
        print(
            f"[proxy-v2] fast inner-loop pool: {len(fast_windows)} paired windows x "
            f"{args.inner_context} tok (mean Δnll selects; bootstrap CI rides along)",
            flush=True,
        )

    def measure(genome: CatGenome, use_distill: bool = False) -> tuple[float, float]:
        """Return ``(pct_mem_saved, delta_nll)`` for a genome (memoized per (genome, mode)).

        With ``--proxy-v2`` the quality scalar is the mean of a PAIRED multi-window Δnll at
        ``inner_context`` (the per-window vector is stashed for the rigorous tier); the scalar contract
        is unchanged so the GA / greedy / Pareto machinery is untouched. Without it, the v1 single
        256-tok Δnll is used verbatim (byte-identical output).
        """
        key = (genome, use_distill)
        if key in cache:
            return cache[key]
        if args.proxy_v2:
            dv = paired_window_deltas(
                model, set_genome, restore, genome, fast_windows, fast_base, use_distill
            )
            fast_delta_cache[key] = dv
            dn = float(dv.mean())
        else:
            set_genome(genome, use_distill)
            dn = nll(ids) - base
            restore()
        mem = sum(mem_opt[o] for o in genome)
        pct = 100.0 * (mem_all_softmax - mem) / mem_all_softmax
        cache[key] = (pct, dn)
        return pct, dn

    # build per-layer distilled students from a HELD-OUT calibration window (disjoint from eval ids)
    distill_mse_log: list[dict[str, float]] = []
    calib_n = 0
    if args.distill:
        calib_ids = tok(text, return_tensors="pt").input_ids[:, args.n_tokens : args.n_tokens + args.distill_tokens]
        calib_n = int(calib_ids.size(1))
        print(f"[distill] calibrating {n_layer} layers on held-out tokens "
              f"[{args.n_tokens}:{args.n_tokens + calib_n}] (eval=[0:{args.n_tokens}]), "
              f"{args.distill_steps} steps lr={args.distill_lr}", flush=True)

        def _log(i: int, info: dict[str, float]) -> None:
            distill_mse_log.append(
                {"layer": float(i), "mse_before": info["mse_before"], "mse_after": info["mse_after"]}
            )
            print(f"[distill] layer {i:2d}: mse {info['mse_before']:.4e} -> {info['mse_after']:.4e}", flush=True)

        distilled.update(
            distill_all_layers(model, calib_ids, steps=args.distill_steps, lr=args.distill_lr, on_layer=_log)
        )
        restore()  # distill_all_layers already restores; be explicit before measuring

    budgets = [float(b) for b in str(args.budgets).split(",") if b.strip()]

    def build_frontier(
        use_distill: bool,
    ) -> tuple[
        list[tuple[float, float]], list[tuple[float, float]], object, list[CatGenome], list[CatGenome]
    ]:
        """Greedy + memetic NSGA-II frontier for one mode.

        Returns ``(greedy_points, evolved_points, history, evolved_genomes, greedy_genomes)`` — the
        last two (the genomes behind the points) feed the proxy-v2 rigorous tier; ignored by v1.
        """
        # single-layer linear tolerance order (mode-specific: distilled layers reorder)
        single: list[float] = []
        for i in range(n_layer):
            g_single = tuple(2 if j == i else 0 for j in range(n_layer))
            if args.proxy_v2:
                single.append(measure(g_single, use_distill)[1])  # fast-pool signal, cached
            else:
                set_genome(g_single, use_distill)
                single.append(nll(ids) - base)  # v1 path unchanged (byte-identical output)
                restore()
        order = sorted(range(n_layer), key=lambda i: single[i])

        def greedy_at(budget: float) -> CatGenome:
            g = [0] * n_layer
            for i in order:
                for opt in (2, 1):  # cheapest memory first: linear, then sliding
                    trial = list(g)
                    trial[i] = opt
                    if measure(tuple(trial), use_distill)[1] <= budget:
                        g[i] = opt
                        break
            return tuple(g)

        greedy_seeds: list[CatGenome] = []
        greedy_points: list[tuple[float, float]] = []
        for b in budgets:
            gg = greedy_at(b)
            if gg not in greedy_seeds:
                greedy_seeds.append(gg)
            greedy_points.append(measure(gg, use_distill))

        res = evolve_multiobjective(
            lambda g: (measure(g, use_distill)[0], -measure(g, use_distill)[1]),
            n_layer, 3, pop_size=args.pop, generations=args.generations,
            seed=args.seed, seed_genomes=greedy_seeds,
        )
        front = cast("list[tuple[CatGenome, tuple[float, ...]]]", res["front"])
        evolved_points = [measure(g, use_distill) for g, _ in front]
        return greedy_points, evolved_points, res["history"], [g for g, _ in front], greedy_seeds

    def greedy_vs_memetic(
        greedy_points: list[tuple[float, float]], evolved_points: list[tuple[float, float]]
    ) -> tuple[float, float, str]:
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
        return g_hv, e_hv, verdict

    def fmt(points: list[tuple[float, float]]) -> list[dict[str, float]]:
        objs = [(pc, -dn) for pc, dn in points]  # maximization objectives (saved %, -delta_nll)
        keep = [i for i in range(len(objs))
                if not any(dominates(objs[j], objs[i]) for j in range(len(objs)) if j != i)]
        return sorted(
            ({"pct_mem_saved": points[i][0], "delta_nll": points[i][1]} for i in keep),
            key=lambda d: d["pct_mem_saved"],
        )

    te = time.perf_counter()
    zs_greedy, zs_evolved, zs_hist, zs_front_g, zs_greedy_g = build_frontier(False)
    zs_g_hv, zs_e_hv, zs_verdict = greedy_vs_memetic(zs_greedy, zs_evolved)
    zs_greedy_fmt = fmt(zs_greedy)
    zs_evolved_fmt = fmt(zs_evolved)
    print(f"[zero-shot] {len(zs_evolved)} Pareto configs ({len(cache)} real evals)", flush=True)

    report: dict[str, object] = {
        "model_dir": args.model_dir,
        "n_layer": n_layer,
        "mixers": MIXERS,
        "ref_context": tref,
        "base_nll": base,
        "budgets": budgets,
        "greedy_frontier": zs_greedy_fmt,
        "evolved_frontier": zs_evolved_fmt,
        "hypervolume": {"greedy": zs_g_hv, "evolved": zs_e_hv},
        "real_evals": len(cache),
        "history": zs_hist,
        "verdict": zs_verdict,
    }

    ds_evolved_fmt: list[dict[str, float]] = []
    rshift: dict[str, float | str] | None = None
    ds_front_g: list[CatGenome] = []
    if args.distill:
        ds_greedy, ds_evolved, ds_hist, ds_front_g, ds_greedy_g = build_frontier(True)
        ds_g_hv, ds_e_hv, ds_verdict = greedy_vs_memetic(ds_greedy, ds_evolved)
        ds_evolved_fmt = fmt(ds_evolved)
        rshift = frontier_right_shift(
            [(pc, -dn) for pc, dn in zs_evolved],
            [(pc, -dn) for pc, dn in ds_evolved],
        )
        report["distill"] = {
            "calib_tokens": calib_n,
            "steps": args.distill_steps,
            "lr": args.distill_lr,
            "layer_mse": distill_mse_log,
            "greedy_frontier": fmt(ds_greedy),
            "evolved_frontier": ds_evolved_fmt,
            "hypervolume": {"greedy": ds_g_hv, "evolved": ds_e_hv},
            "history": ds_hist,
            "verdict": ds_verdict,
        }
        report["right_shift"] = rshift

    if args.proxy_v2:
        report["proxy_v2"] = _proxy_v2_rigorous(
            model, tok, set_genome, restore, ids_all, args, mem_opt, mem_all_softmax,
            fast_delta_cache, zs_front_g, zs_greedy_g, ds_front_g if args.distill else None,
        )
        pv = cast("dict[str, object]", report["proxy_v2"])
        print(f"\n[proxy-v2 verdict] {cast('dict[str, str]', pv['verdict'])['memetic_vs_greedy']} "
              f"(confidence={cast('dict[str, str]', pv['verdict'])['confidence']})", flush=True)

    elapsed = time.perf_counter() - te
    report["elapsed_s"] = elapsed
    (out / "nas_pareto.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[zero-shot greedy frontier]")
    for d in zs_greedy_fmt:
        print(f"  saved {d['pct_mem_saved']:5.1f}%  delta_nll {d['delta_nll']:+.4f}")
    print("[zero-shot evolved frontier]")
    for d in zs_evolved_fmt:
        print(f"  saved {d['pct_mem_saved']:5.1f}%  delta_nll {d['delta_nll']:+.4f}")
    print(f"[zero-shot verdict] {zs_verdict}")
    if args.distill:
        print("\n[distilled evolved frontier]")
        for d in ds_evolved_fmt:
            print(f"  saved {d['pct_mem_saved']:5.1f}%  delta_nll {d['delta_nll']:+.4f}")
        assert rshift is not None
        print(f"[right-shift] {rshift['verdict']}")
    print(f"[done] wrote {out}/nas_pareto.json ({len(cache)} real evals, {elapsed:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
