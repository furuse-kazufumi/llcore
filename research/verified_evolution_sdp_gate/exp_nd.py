# SPDX-License-Identifier: Apache-2.0
"""Dimension-scaling experiment: does the SDP/2-norm-vs-inf payoff GROW with dimension?

For n ∈ {2,3,4}, run gated evolution (none / inf_norm / two_norm / sdp) on the block-rotation
objective and measure best fitness per gate. Hypothesis: as n grows, the ∞-norm over-rejects
more of the rotational optimum, so (sdp − inf) and (two − inf) payoffs grow with n.

Usage: py -3.11 exp_nd.py [--seeds N] [--pop P] [--gens G]
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from coupled_nd import (
    CoupledNDGeneCodec,
    RotationNDObjective,
    classify_region,
    empirical_rho,
    make_nd_verifier,
)
from evolvable_core import EvolveConfig, evolve
from exp_runner import paired_compare

_HERE = os.path.dirname(os.path.abspath(__file__))
DIMS = (2, 3, 4)
# Main scaling run uses the CHEAP sound gates (no cvxpy): the headline question is whether
# the conservative ∞-norm's evolutionary cost grows with dimension, recovered by a better
# (2-norm) sound verifier. The SDP gate (cvxpy, 2^n vertices) is checked separately/small —
# its unique gain over 2-norm is the thin shell already established at n=2.
GATES = ("none", "inf_norm", "two_norm")
REGIONS = ("inf", "two_norm_only", "sdp_only", "non_certified")


def run(n_seeds: int = 10, pop: int = 40, gens: int = 50, base: int = 4000,
        gates: tuple = GATES, dims: tuple = DIMS, tag: str = "") -> dict:
    t0 = time.time()
    cfg = EvolveConfig(pop_size=pop, n_generations=gens, resample_cap=18)
    out = {"config": {"n_seeds": n_seeds, "pop": pop, "gens": gens, "gates": list(gates)},
           "by_dim": {}}
    for n in dims:
        codec = CoupledNDGeneCodec(n)
        obj = RotationNDObjective(n)
        per_gate = {g: [] for g in gates}
        for s in range(base, base + n_seeds):
            for g in gates:
                r = evolve(codec, obj, make_nd_verifier(g), cfg, rng=np.random.default_rng(s))
                div = sum(empirical_rho(codec.to_gene(gt), n_samples=4000) >= 1.0
                          for gt in r.final_population)
                per_gate[g].append({"seed": s, "best": r.best_fitness,
                                     "winner_region": classify_region(r.best_gene),
                                     "final_divergent": int(div), "fallback": r.fallback_count})
        mean_best = {g: float(np.mean([x["best"] for x in per_gate[g]])) for g in gates}
        inf_b = [x["best"] for x in per_gate["inf_norm"]]
        rec = {"mean_best": mean_best,
               "g1_final_divergent": {g: int(np.sum([x["final_divergent"] for x in per_gate[g]]))
                                      for g in gates if g != "none"},
               "winner_regions": {g: {rg: sum(1 for x in per_gate[g] if x["winner_region"] == rg)
                                      for rg in REGIONS} for g in gates},
               "runs": per_gate}
        if "two_norm" in gates:
            rec["two_vs_inf"] = paired_compare([x["best"] for x in per_gate["two_norm"]], inf_b, "greater")
            rec["two_minus_inf"] = mean_best["two_norm"] - mean_best["inf_norm"]
        if "sdp" in gates:
            rec["sdp_vs_inf"] = paired_compare([x["best"] for x in per_gate["sdp"]], inf_b, "greater")
            rec["sdp_minus_inf"] = mean_best["sdp"] - mean_best["inf_norm"]
        out["by_dim"][n] = rec
        print(f"  n={n} done ({round(time.time()-t0,1)}s): mean_best="
              f"{ {g: round(v,3) for g,v in mean_best.items()} }", flush=True)
    key = "two_minus_inf" if "two_norm" in gates else "sdp_minus_inf"
    out["scaling"] = {f"{key}_by_n": {n: round(out['by_dim'][n][key], 3) for n in dims}}
    out["elapsed_s"] = round(time.time() - t0, 1)
    p = os.path.join(_HERE, f"exp_nd_results{tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"exp_nd -> {p} ({out['elapsed_s']}s)", flush=True)
    print("SCALING", key, "by n:", out["scaling"][f"{key}_by_n"], flush=True)
    return out


if __name__ == "__main__":
    kw = {}
    for i, a in enumerate(sys.argv):
        if a == "--seeds":
            kw["n_seeds"] = int(sys.argv[i + 1])
        if a == "--pop":
            kw["pop"] = int(sys.argv[i + 1])
        if a == "--gens":
            kw["gens"] = int(sys.argv[i + 1])
    run(**kw)
