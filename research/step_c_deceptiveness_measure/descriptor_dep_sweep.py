# SPDX-License-Identifier: Apache-2.0
"""Descriptor-dependence sweep (Phase B verify lens).

Question: does flip_flop's below/above verdict (vs the committed threshold
metric_at_dstar=0.015281642817850516) flip when we change the BEHAVIOR
DESCRIPTOR settings of the SAME elite_dip metric -- specifically:
  (A) n_bins (binning of the projected behavior axis): 8 / 12 / 16 / 24 / 32
  (B) the behavior descriptor itself:
        - "full"      = make_behavior(res) = (eff_mem_norm, std(leak))  [committed]
        - "dim0"      = first behavior component only (1D)
        - "dim1"      = second behavior component only (1D)
        - "constant"  = degenerate (collapses -> metric must return 0)
At a fixed, feasible budget (n_samples, honest_n_trials, n_seeds), so the only
thing changing is the descriptor/binning. If the below/above verdict flips, the
"smooth / below" reading is a descriptor artifact.

Honest disclosure:
- Budget is reduced vs the committed run (n_samples=400, honest_n_trials=3,
  n_seeds=3) to fit the harness time limit (~100ms/eval). This RAISES envelope
  bias and widens noise; that is itself part of the finding (the metric value is
  budget-sensitive). We report raw scores + which side of the threshold.
- We compare each score to the SAME committed threshold 0.015282 (operational,
  same-metric). RANK transfers, magnitude does not (per the artifacts).
src / task modules are read-only import. numpy only. CPU.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (_HERE,
          _HERE.parents[0] / "step_c_memory_tasks",
          _HERE.parents[0] / "ea_multitask",
          _HERE.parents[0] / "ea_multitask" / "candidates",
          _HERE.parents[0] / "step6_real_proxy",
          _HERE.parents[1] / "src"):
    sys.path.insert(0, str(p))

from metric_behavior_elite_dip import deceptiveness

METRIC_AT_DSTAR = 0.015281642817850516

# budget (reduced but multi-seed; disclosed). Kept small to fit harness limit
# (~100ms/eval). The point is the FLIP of the verdict across descriptors/bins,
# not a magnitude claim; small budget is itself part of the artifact finding.
N_SAMPLES = 220
N_TRIALS = 2
N_SEEDS = 2
BASE_SEED = 20260531


def _est(eval_fn, behavior_fn, bounds, dim, n_bins):
    vals = []
    for s in range(N_SEEDS):
        rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, s, n_bins]))
        vals.append(deceptiveness(eval_fn, behavior_fn, bounds, dim, rng,
                                  n_samples=N_SAMPLES, n_bins=n_bins,
                                  honest_n_trials=N_TRIALS, min_per_bin=4))
    arr = np.array(vals)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0, vals


def main() -> int:
    from reservoir import (LeakyDelayLineReservoir, gene_bounds, make_behavior,
                           make_eval_once)
    from memory_tasks import FlipFlopTask

    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    bounds = gene_bounds(res)
    full_behavior = make_behavior(res)
    dim = res.gene_dim

    # build descriptor variants (wrap the committed behavior)
    def dim0(g):
        return np.atleast_1d(np.asarray(full_behavior(g), dtype=np.float64))[0:1]

    def dim1(g):
        b = np.atleast_1d(np.asarray(full_behavior(g), dtype=np.float64))
        return b[1:2] if b.size > 1 else b[0:1]

    def const(g):
        return np.array([0.5])

    descriptors = {"full": full_behavior, "dim0": dim0, "dim1": dim1, "constant": const}

    ev_ff = make_eval_once(res, FlipFlopTask(), n_train=48, n_eval=48)

    out = {
        "lens": "descriptor_dependence",
        "task": "flip_flop",
        "metric": "behavior_elite_fitness_dip",
        "metric_at_dstar": METRIC_AT_DSTAR,
        "committed_value": 0.016092457495001964,
        "committed_below_threshold": False,
        "budget": {"n_samples": N_SAMPLES, "honest_n_trials": N_TRIALS, "n_seeds": N_SEEDS,
                   "note": "reduced vs committed (1600x10x5); raises envelope bias (disclosed)"},
        "sweep": {},
    }

    print("=== descriptor x n_bins sweep on flip_flop (elite_dip) ===", flush=True)
    print(f"threshold metric_at_dstar={METRIC_AT_DSTAR:.6f}; budget n={N_SAMPLES} trials={N_TRIALS} seeds={N_SEEDS}", flush=True)
    t_all = time.time()
    for dname, dfn in descriptors.items():
        out["sweep"][dname] = {}
        for n_bins in (8, 16, 32):
            t0 = time.time()
            mean, std, vals = _est(ev_ff, dfn, bounds, dim, n_bins)
            below = bool(mean < METRIC_AT_DSTAR)
            out["sweep"][dname][str(n_bins)] = {
                "mean": round(mean, 6), "std": round(std, 6),
                "per_seed": [round(v, 6) for v in vals],
                "below_threshold": below,
            }
            print(f"  descr={dname:9s} n_bins={n_bins:2d} mean={mean:.5f} "
                  f"std={std:.5f} vs thr -> below={below}  ({time.time()-t0:.0f}s)",
                  flush=True)

    # summarize flips
    summary = {}
    for dname in descriptors:
        verds = {out["sweep"][dname][str(b)]["below_threshold"] for b in (8, 12, 16, 24, 32)}
        summary[dname] = {"verdicts_seen": sorted(str(v) for v in verds),
                          "flips_across_bins": len(verds) > 1}
    # across all descriptor+bins, are both below and above achievable?
    all_cells = [out["sweep"][d][str(b)]["below_threshold"]
                 for d in descriptors for b in (8, 12, 16, 24, 32)]
    out["flip_summary"] = {
        "per_descriptor": summary,
        "any_below_achievable": any(all_cells),
        "any_above_achievable": any(not c for c in all_cells),
        "verdict_flips_with_descriptor_or_binning": (any(all_cells) and any(not c for c in all_cells)),
    }

    path = _HERE / "descriptor_dep_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}  (total {time.time()-t_all:.0f}s)", flush=True)
    print(json.dumps(out["flip_summary"], ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
