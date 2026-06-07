# SPDX-License-Identifier: Apache-2.0
# -*- coding: utf-8 -*-
"""
verify_margin.py -- Phase B adversarial verification, lens=sampling_threshold_margin.

The real-task EA evals are ~100ms each; re-measuring at full budget over many seed
banks is multi-hour (the downhill cross-metric author hit the same wall). So this
script splits the work:

 PART 1 (cheap, RE-MEASURED): threshold (metric_at_dstar) stability on the SYNTHETIC
   corridor across base_seed / n_seeds / n_samples / n_bins. The threshold is the
   quantity the whole below/above verdict hangs on; if it is seed-fragile the verdict
   is fragile. Synthetic eval is fast (deterministic ramp), so we can sweep widely.

 PART 2 (from existing per-seed data, NO re-run): margin analysis of the on-disk
   real-task estimates. We recompute, from the stored per_seed arrays, the margin to
   the threshold in std units, the seed-noise CV, and whether the 95% CI brackets the
   threshold -- i.e. whether the below/above verdict could flip under resampling.

ASCII-only. Writes verify_margin_results.json.
"""
from __future__ import annotations
import json, sys, math
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "step_c_applicability"))
sys.path.insert(0, str(HERE.parents[1] / "src"))

for s in (sys.stdout, sys.stderr):
    rc = getattr(s, "reconfigure", None)
    if rc:
        try: rc(encoding="utf-8")
        except Exception: pass

from metric_behavior_elite_dip import deceptiveness_estimate  # noqa
from exp_knob_sweep import D as SYN_D, behavior_mean, make_corridor_eval  # noqa

THR = 0.015281642817850516  # stored metric_at_dstar (d=0.16)
SBOUNDS = (np.zeros(SYN_D), np.ones(SYN_D))
EV_DSTAR = make_corridor_eval(0.16)


def t_mult(n):
    return {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
            7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)


def ci95(per_seed):
    a = np.asarray(per_seed, float)
    m = float(a.mean()); s = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    se = s / math.sqrt(len(a)) if len(a) > 1 else 0.0
    t = t_mult(len(a))
    return m, s, m - t * se, m + t * se


def main():
    out = {"lens": "sampling_threshold_margin"}

    # =========================================================================
    # PART 1 -- threshold (d*) raw-value stability, RE-MEASURED on synthetic.
    # =========================================================================
    p1 = []
    base = dict(n_samples=4000, n_bins=24, honest_n_trials=1)
    for bs in (20260531, 1, 777, 424242):
        est = deceptiveness_estimate(EV_DSTAR, behavior_mean, SBOUNDS, SYN_D,
                                     n_seeds=5, base_seed=bs, **base)
        p1.append({"vary": "base_seed", "val": bs, "mean": round(est.mean, 5),
                   "std": round(est.std, 5),
                   "per_seed": [round(x, 4) for x in est.per_seed]})
    for ns in (3, 5, 10):
        est = deceptiveness_estimate(EV_DSTAR, behavior_mean, SBOUNDS, SYN_D,
                                     n_seeds=ns, base_seed=20260531, **base)
        p1.append({"vary": "n_seeds", "val": ns, "mean": round(est.mean, 5),
                   "std": round(est.std, 5)})
    for nsamp in (2000, 4000, 8000):
        est = deceptiveness_estimate(EV_DSTAR, behavior_mean, SBOUNDS, SYN_D,
                                     n_seeds=5, base_seed=20260531,
                                     n_samples=nsamp, n_bins=24, honest_n_trials=1)
        p1.append({"vary": "n_samples", "val": nsamp, "mean": round(est.mean, 5),
                   "std": round(est.std, 5)})
    for nb in (16, 24, 32):
        est = deceptiveness_estimate(EV_DSTAR, behavior_mean, SBOUNDS, SYN_D,
                                     n_seeds=5, base_seed=20260531,
                                     n_samples=4000, n_bins=nb, honest_n_trials=1)
        p1.append({"vary": "n_bins", "val": nb, "mean": round(est.mean, 5),
                   "std": round(est.std, 5)})
    out["P1_threshold_stability"] = p1
    means = [r["mean"] for r in p1]
    out["P1_summary"] = {
        "stored_threshold": THR,
        "remeasured_min": round(min(means), 5),
        "remeasured_max": round(max(means), 5),
        "remeasured_spread_factor": round(max(means) / max(min(means), 1e-9), 2),
        "stored_threshold_std_5seed": 0.01015,
        "stored_threshold_cv": round(0.01015 / THR, 2),
        "note": ("threshold raw value at d=0.16 moves by this factor under seed/sample/bin "
                 "perturbation; its own 5-seed std is ~67% of its value (CV>0.6)."),
    }

    # =========================================================================
    # PART 2 -- margin analysis of the EXISTING real-task estimates (no re-run).
    # =========================================================================
    res = json.loads((HERE / "measure_real_tasks_results.json").read_text(encoding="utf-8"))
    tasks = res["real_tasks"]
    p2 = {}
    for name, rec in tasks.items():
        per = rec["per_seed"]
        m, s, lo, hi = ci95(per)
        margin = m - THR
        p2[name] = {
            "per_seed": [round(x, 4) for x in per],
            "mean": round(m, 5), "std": round(s, 5),
            "cv_seed_noise": round(s / m, 3) if m else None,
            "ci95": [round(lo, 5), round(hi, 5)],
            "threshold": round(THR, 5),
            "margin_to_threshold": round(margin, 5),
            "margin_in_std_units": round(margin / (s + 1e-12), 2),
            "below_threshold_pointest": bool(m < THR),
            "ci_brackets_threshold": bool(lo < THR < hi),
            "verdict_could_flip_under_resample": bool(lo < THR < hi),
        }
    out["P2_real_margin"] = p2

    # =========================================================================
    # PART 3 -- combined honest summary.
    # =========================================================================
    below = {k: v["below_threshold_pointest"] for k, v in p2.items()}
    flippable = {k: v["verdict_could_flip_under_resample"] for k, v in p2.items()}
    out["P3_summary"] = {
        "on_disk_conclusion_all_below_threshold": res["conclusion"]["all_below_threshold"],
        "on_disk_below_per_task": res["conclusion"]["below_threshold_per_task"],
        "recomputed_below_per_task": below,
        "verdict_flippable_per_task": flippable,
        "brief_premise_was": "all below, max 0.041, d*=0.1234",
        "brief_premise_matches_disk": False,
        "actual_real_values": {k: round(ci95(tasks[k]["per_seed"])[0], 4) for k in tasks},
        "actual_threshold": round(THR, 4),
    }

    (HERE / "verify_margin_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("WROTE verify_margin_results.json")
    print(json.dumps(out["P1_summary"], indent=2))
    print(json.dumps(out["P3_summary"], indent=2))


if __name__ == "__main__":
    main()
