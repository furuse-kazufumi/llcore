# SPDX-License-Identifier: Apache-2.0
"""Main experiment runner for the SDP-gated coupled evolution (G0-G5).

exp1 — landscape attribution (NON-circular mechanism evidence): do high-fitness
       genes concentrate in the inf-norm-REJECTED-but-(2norm/SDP)-certified region?
exp2 — gated evolution: 4 gates x 3 tasks x n seeds; best fitness, winner region,
       gate bookkeeping, admitted-divergent soundness check.

Usage:
  py -3.11 exp_runner.py exp1
  py -3.11 exp_runner.py exp2 [--seeds N] [--pop P] [--gens G]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

from coupled_components import (
    BenignDecayObjective,
    CoupledGeneCodec,
    NonNormalObjective,
    RotationObjective,
    classify_region,
    empirical_spectral_radius,
    make_verifier,
)
from evolvable_core import EvolveConfig, evolve

_HERE = os.path.dirname(os.path.abspath(__file__))
CODEC = CoupledGeneCodec()
OBJECTIVES = {"rotation": RotationObjective(), "benign": BenignDecayObjective(),
              "nonnormal": NonNormalObjective()}
GATES = ("none", "inf_norm", "two_norm", "sdp")
REGIONS = ("inf", "two_norm_only", "sdp_only", "non_certified")


# --------------------------------------------------------------------------- #
# paired comparison (project standard: one-sided Wilcoxon + paired-sign delta)
# --------------------------------------------------------------------------- #


def paired_compare(a: list[float], b: list[float], alternative: str = "greater") -> dict:
    """Test a > b (paired). Returns p (one/two-sided Wilcoxon) + paired_sign_delta."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = a - b
    n = len(d)
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    psd = (n_pos - n_neg) / n  # in [-1, 1]
    p = None
    try:
        from scipy.stats import wilcoxon
        nz = d[d != 0]
        if len(nz) >= 1:
            p = float(wilcoxon(a[d != 0], b[d != 0], alternative=alternative).pvalue)
    except Exception as exc:  # scipy missing or degenerate
        p = f"unavailable:{type(exc).__name__}"
    return {"n": n, "mean_delta": float(np.mean(d)), "median_delta": float(np.median(d)),
            "n_pos": n_pos, "n_neg": n_neg, "paired_sign_delta": float(psd),
            "wilcoxon_p": p, "alternative": alternative}


# --------------------------------------------------------------------------- #
# exp1 — landscape attribution
# --------------------------------------------------------------------------- #


def exp1(n_random: int = 8000, n_ga_visited: int = 4000, seed: int = 1) -> dict:
    """Pool = random genes + genes a no-gate GA visits; classify each by region
    and fitness on each task. Reports per-region fitness stats + the region of
    the top-K fittest genes per task (does payoff need the better verifier?)."""
    t0 = time.time()
    rng = np.random.default_rng(seed)
    pool = [CODEC.clip(CODEC.random(rng)) for _ in range(n_random)]

    # add genes a no-gate GA visits on each task (where fitness pressure pushes)
    for obj in OBJECTIVES.values():
        r = evolve(CODEC, obj, make_verifier("none"),
                   EvolveConfig(pop_size=30, n_generations=25), rng=np.random.default_rng(seed + 1))
        pool.extend(r.admitted_genotypes[: n_ga_visited // 3])

    genes = [CODEC.to_gene(g) for g in pool]
    regions = [classify_region(gn) for gn in genes]
    emp_rho = [empirical_spectral_radius(gn, n_samples=2000) for gn in genes]

    result = {"n_pool": len(pool), "region_counts": {r: regions.count(r) for r in REGIONS}}
    for tname, obj in OBJECTIVES.items():
        fits = np.array([obj.fitness(gn) for gn in genes])
        # per-region fitness stats (contracting genes only: emp_rho<1)
        per_region = {}
        for reg in REGIONS:
            mask = np.array([(regions[i] == reg) for i in range(len(genes))])
            if mask.sum() > 0:
                per_region[reg] = {"count": int(mask.sum()),
                                   "max_fitness": float(fits[mask].max()),
                                   "mean_fitness": float(fits[mask].mean())}
        # region membership of the top-K fittest CONTRACTING genes
        contracting = np.array([emp_rho[i] < 1.0 for i in range(len(genes))])
        idx_sorted = np.argsort(-fits)
        topk_regions = []
        for i in idx_sorted:
            if contracting[i]:
                topk_regions.append(regions[i])
            if len(topk_regions) >= 50:
                break
        topk_counts = {r: topk_regions.count(r) for r in REGIONS}
        result[tname] = {
            "per_region_fitness": per_region,
            "top50_contracting_region_counts": topk_counts,
            "best_overall_region": regions[int(idx_sorted[0])],
            "best_overall_fitness": float(fits[int(idx_sorted[0])]),
        }
    result["elapsed_s"] = round(time.time() - t0, 1)
    out = os.path.join(_HERE, "exp1_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"exp1 -> {out} ({result['elapsed_s']}s)")
    return result


# --------------------------------------------------------------------------- #
# exp2 — gated evolution
# --------------------------------------------------------------------------- #


def exp2(n_seeds: int = 15, pop: int = 24, gens: int = 30, base_seed: int = 1000) -> dict:
    t0 = time.time()
    cfg = EvolveConfig(pop_size=pop, n_generations=gens, resample_cap=40)
    seeds = [base_seed + i for i in range(n_seeds)]
    # runs[task][gate] = list over seeds of dict(best, winner_region, admit_div_frac, rej, fb)
    runs: dict = {t: {g: [] for g in GATES} for t in OBJECTIVES}

    for tname, obj in OBJECTIVES.items():
        for s in seeds:
            for gate in GATES:
                r = evolve(CODEC, obj, make_verifier(gate), cfg, rng=np.random.default_rng(s))
                # soundness: any divergent gene in final pop? (sound gates must be 0)
                div = sum(empirical_spectral_radius(CODEC.to_gene(g), n_samples=4000) >= 1.0
                          for g in r.final_population)
                runs[tname][gate].append({
                    "seed": s,
                    "best": r.best_fitness,
                    "winner_region": classify_region(r.best_gene),
                    "final_divergent": int(div),
                    "n_rejections": r.n_rejections,
                    "fallback": r.fallback_count,
                })
        print(f"  exp2 {tname} done ({round(time.time()-t0,1)}s)")

    # --- gate verdicts ---
    verdicts = {}
    # G2 load-bearing: ungated final-pop divergent fraction (avg over seeds), pop size
    g2 = {}
    for tname in OBJECTIVES:
        none_div = np.mean([x["final_divergent"] for x in runs[tname]["none"]]) / pop
        gated_div = {g: int(np.sum([x["final_divergent"] for x in runs[tname][g]]))
                     for g in ("inf_norm", "two_norm", "sdp")}
        g2[tname] = {"ungated_divergent_frac": float(none_div), "gated_total_divergent": gated_div}
    verdicts["G2_load_bearing"] = g2

    # G1 soundness: total divergent admitted by sound gates across all tasks/seeds
    g1 = {g: int(np.sum([x["final_divergent"] for t in OBJECTIVES for x in runs[t][g]]))
          for g in ("inf_norm", "two_norm", "sdp")}
    verdicts["G1_soundness_total_divergent"] = g1

    # G4 payoff (rotation + nonnormal): sdp vs inf_norm, paired
    # G5 null (benign): sdp vs inf_norm, two-sided
    payoff = {}
    for tname in OBJECTIVES:
        sdp_best = [x["best"] for x in runs[tname]["sdp"]]
        inf_best = [x["best"] for x in runs[tname]["inf_norm"]]
        two_best = [x["best"] for x in runs[tname]["two_norm"]]
        none_best = [x["best"] for x in runs[tname]["none"]]
        alt = "two-sided" if tname == "benign" else "greater"
        payoff[tname] = {
            "sdp_vs_inf": paired_compare(sdp_best, inf_best, alt),
            "two_vs_inf": paired_compare(two_best, inf_best, alt),
            "sdp_vs_two": paired_compare(sdp_best, two_best, alt),
            "none_vs_sdp": paired_compare(none_best, sdp_best, "greater"),
            "mean_best": {g: float(np.mean([x["best"] for x in runs[tname][g]])) for g in GATES},
            "winner_regions": {g: {r: sum(1 for x in runs[tname][g] if x["winner_region"] == r)
                                   for r in REGIONS} for g in GATES},
        }
    verdicts["payoff"] = payoff

    result = {"config": {"n_seeds": n_seeds, "pop": pop, "gens": gens, "base_seed": base_seed},
              "runs": runs, "verdicts": verdicts, "elapsed_s": round(time.time() - t0, 1)}
    out = os.path.join(_HERE, "exp2_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"exp2 -> {out} ({result['elapsed_s']}s)")
    return result


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "exp2"
    kw = {}
    for i, a in enumerate(sys.argv):
        if a == "--seeds":
            kw["n_seeds"] = int(sys.argv[i + 1])
        if a == "--pop":
            kw["pop"] = int(sys.argv[i + 1])
        if a == "--gens":
            kw["gens"] = int(sys.argv[i + 1])
    if which == "exp1":
        exp1()
    else:
        exp2(**kw)
