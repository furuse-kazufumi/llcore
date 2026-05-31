# SPDX-License-Identifier: Apache-2.0
"""verify_circular_synth_probe.py (NEW, read-only on existing files)

circular_reasoning lens — quantify the structural circularity and the CLT-shadow
that the metric/JSON authors disclosed but only argued qualitatively.

P1. On the synthetic corridor, fitness IS a function of behavior=mean(g) by
    construction -> corr(fitness, behavior) over random genes should be ~|1|.
    That is the tautology: the metric "recovers d" only because behavior == the
    very axis the knob is carved on.
P2. CLT shadow: mean of D=24 U(0,1) draws concentrates near 0.5 (std~0.0589).
    The well center (0.65), and especially the global peak (0.95), are almost
    never sampled -> the behavior-elite envelope never reaches the global peak
    bin -> the measured dip is a heavily attenuated shadow of the true d.
    Quantify the sampled-behavior range and how often b>=0.95 / b in well.
P3. Compare measured synthetic dip at d=0.16 to the geometric expectation 0.8*d
    (the metric docstring's own prediction) to show the ~8x attenuation.

Pure numpy. CPU. Writes nothing.
"""
from __future__ import annotations
import os, sys
import numpy as np

BASE = r"D:\projects\llcore\research"
sys.path.insert(0, os.path.join(BASE, "step_c_deceptiveness_measure"))
sys.path.insert(0, os.path.join(BASE, "step_c_applicability"))

from exp_knob_sweep import make_corridor_eval, behavior_mean, D
from metric_behavior_elite_dip import deceptiveness_estimate


def p1_tautology():
    print("=== P1: synthetic fitness is a function of behavior=mean (tautology) ===")
    rng = np.random.default_rng(0)
    G = rng.random((5000, D))
    b = G.mean(axis=1)
    erng = np.random.default_rng(1)
    for d in (0.0, 0.16, 0.5, 1.0):
        ef = make_corridor_eval(d)
        # corridor_eval(gene, rng); noise is tiny (_NOISE=0.008) so corr ~ structural.
        f = np.array([ef(g, erng) for g in G])
        c = np.corrcoef(f, b)[0, 1]
        print(f"  d={d:.2f}: corr(fitness, behavior=mean) = {c:+.4f}")
    print("  -> |corr|~1 confirms fitness is determined by the behavior axis itself.")
    print("     The metric 'recovers d' on synthetic because behavior == the knob axis = CIRCULAR.")


def p2_clt_shadow():
    print("\n=== P2: CLT shadow — sampled behavior never reaches well/global peak ===")
    rng = np.random.default_rng(1)
    b = rng.random((200000, D)).mean(axis=1)
    print(f"  behavior=mean(U(0,1)^{D}): mean={b.mean():.4f} std={b.std():.4f} "
          f"range=({b.min():.4f},{b.max():.4f})")
    in_well = np.mean((b >= 0.60) & (b <= 0.70))
    at_peak = np.mean(b >= 0.95)
    near_peak = np.mean(b >= 0.90)
    print(f"  P(b in well[0.60,0.70]) = {in_well:.5f}")
    print(f"  P(b >= 0.90)            = {near_peak:.8f}")
    print(f"  P(b >= 0.95 global peak)= {at_peak:.8f}")
    print("  -> global-peak bin essentially never sampled; behavior-elite envelope's 'global'")
    print("     bin is a near-0.5 sample, so the measured dip is an attenuated shadow of true d.")


def p3_attenuation():
    print("\n=== P3: measured synthetic dip vs geometric expectation 0.8*d (attenuation) ===")
    bounds = (np.zeros(D), np.ones(D))
    for d in (0.16, 0.5, 1.0):
        ef = make_corridor_eval(d)
        est = deceptiveness_estimate(ef, behavior_mean, bounds, D,
                                     n_seeds=3, n_samples=4000, n_bins=24, honest_n_trials=1)
        expect = 0.8 * d
        atten = expect / est.mean if est.mean > 0 else float("inf")
        print(f"  d={d:.2f}: measured dip={est.mean:.4f} (95%CI[{est.ci95_lo:.4f},{est.ci95_hi:.4f}])"
              f"  geometric 0.8*d={expect:.3f}  attenuation~{atten:.1f}x")
    print("  -> measured << 0.8*d confirms the metric_at_dstar=0.0153 is a CLT-attenuated value,")
    print("     not the true d=0.16 magnitude. Real-task raw dips are compared to THIS shadow.")


if __name__ == "__main__":
    for s in (sys.stdout, sys.stderr):
        rc = getattr(s, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except Exception:
                pass
    p1_tautology()
    p2_clt_shadow()
    p3_attenuation()
