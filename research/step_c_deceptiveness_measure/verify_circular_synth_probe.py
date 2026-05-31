# SPDX-License-Identifier: Apache-2.0
"""verify_circular_synth_probe.py (NEW, read-only on existing files)

circular_reasoning lens — quantify the structural circularity + CLT-shadow that
the metric/JSON authors disclosed only qualitatively. Writes results to
verify_circular_synth_probe_out.json (avoids console encoding issues).

P1. corr(fitness, behavior=mean) on synthetic corridor ~ |1| (tautology).
P2. CLT shadow: mean(U(0,1)^24) concentrates near 0.5; global peak (b>=0.95)
    essentially never sampled -> envelope never reaches global peak bin.
P3. measured synthetic dip at d in {0.16,0.5,1.0} vs geometric 0.8*d (attenuation).
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np

BASE = r"D:\projects\llcore\research"
sys.path.insert(0, os.path.join(BASE, "step_c_deceptiveness_measure"))
sys.path.insert(0, os.path.join(BASE, "step_c_applicability"))

from exp_knob_sweep import make_corridor_eval, behavior_mean, D
from metric_behavior_elite_dip import deceptiveness_estimate


def main() -> int:
    out = {"D": int(D)}

    # P1: tautology
    rng = np.random.default_rng(0)
    G = rng.random((5000, D))
    b = G.mean(axis=1)
    erng = np.random.default_rng(1)
    p1 = {}
    for d in (0.0, 0.16, 0.5, 1.0):
        ef = make_corridor_eval(d)
        f = np.array([ef(g, erng) for g in G])
        p1[f"{d:.2f}"] = float(np.corrcoef(f, b)[0, 1])
    out["P1_corr_fitness_vs_behavior_mean"] = p1

    # P2: CLT shadow
    b2 = np.random.default_rng(2).random((200000, D)).mean(axis=1)
    out["P2_clt"] = {
        "behavior_mean": float(b2.mean()),
        "behavior_std": float(b2.std()),
        "behavior_min": float(b2.min()),
        "behavior_max": float(b2.max()),
        "P_in_well_0.60_0.70": float(np.mean((b2 >= 0.60) & (b2 <= 0.70))),
        "P_ge_0.90": float(np.mean(b2 >= 0.90)),
        "P_ge_0.95_global_peak": float(np.mean(b2 >= 0.95)),
    }

    # P3: attenuation
    bounds = (np.zeros(D), np.ones(D))
    p3 = {}
    for d in (0.16, 0.5, 1.0):
        ef = make_corridor_eval(d)
        est = deceptiveness_estimate(ef, behavior_mean, bounds, D,
                                     n_seeds=3, n_samples=4000, n_bins=24, honest_n_trials=1)
        expect = 0.8 * d
        p3[f"{d:.2f}"] = {
            "measured_dip": float(est.mean),
            "ci95_lo": float(est.ci95_lo),
            "ci95_hi": float(est.ci95_hi),
            "geometric_0.8d": float(expect),
            "attenuation_x": float(expect / est.mean) if est.mean > 0 else None,
        }
    out["P3_attenuation"] = p3

    path = os.path.join(BASE, "step_c_deceptiveness_measure",
                        "verify_circular_synth_probe_out.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
