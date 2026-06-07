# SPDX-License-Identifier: Apache-2.0
"""STRONG soundness test of the lifted degree-4/6 certifier — via product expansion (JSR), not
just pointwise ρ.

Why this is the right test (skeptic's foundational concern): the lifted LMI imposes Lyapunov
decrease only at the t-box VERTICES {J_v}. The nonlinear map's mean Jacobian J̄(s)=∫₀¹J(τs)dτ lies
in conv{J_v}. For a QUADRATIC Lyapunov (k=1), vertex feasibility ⇒ hull feasibility by matrix
convexity (Aᵀ P A convex in A). For degree≥2 the lift A^[k] is non-convex in A, so vertex
feasibility does NOT directly imply hull decrease of THIS V. Soundness instead rests on:
  common vertex Lyapunov ⇒ JSR{J_v} < 1  ⇒ (JSR convex-hull theorem) JSR{conv J_v} < 1
  ⇒ the nonlinear map (Jacobians in the hull) contracts.
So a NECESSARY consequence of a SOUND certificate is: every certified gene has JSR{J_v} < 1, hence
JSR_lower_bound (max over products of ρ(∏)^{1/k}) < 1. If ANY deg4/deg6-certified gene has
JSR_lb ≥ 1, a product of vertex Jacobians expands ⇒ the certifier is UNSOUND (a real bug).

This is strictly stronger than the pointwise-ρ@50k oracle (which can miss switched expansion).
"""
from __future__ import annotations

import itertools
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coupled_map import CoupledGene  # noqa: E402
from verifier_deg4 import _vertices_n2  # noqa: E402


def jsr_lb(vertices, max_len=6) -> float:
    V = [np.asarray(M, dtype=np.float64) for M in vertices]
    best = 0.0
    for k in range(1, max_len + 1):
        for combo in itertools.product(range(len(V)), repeat=k):
            P = np.eye(V[0].shape[0])
            for i in combo:
                P = V[i] @ P
            best = max(best, float(np.max(np.abs(np.linalg.eigvals(P)))) ** (1.0 / k))
    return best


def run(max_len: int = 6) -> dict:
    t0 = time.time()
    with open(os.path.join(_HERE, "exp_deg6_residual_genes.json"), encoding="utf-8") as f:
        data = json.load(f)
    deg = data["deg_certified"]      # genes deg4/deg6 certified (quad-rejected) — must have JSR<1
    uncert = data["residual_uncert"]  # not certified — for the finite-gap attribution refresh

    # (A) certifier soundness: every certified gene must have JSR_lb < 1.
    cert_jsr = []
    bad = []
    for rec in deg:
        g = CoupledGene.make(decay=np.asarray(rec["decay"]), W=np.asarray(rec["W"]).reshape(2, 2))
        j = jsr_lb(_vertices_n2(g), max_len=max_len)
        cert_jsr.append(j)
        if j >= 1.0 - 1e-9:
            bad.append({**rec, "jsr_lb": j})
    cert_jsr = np.array(cert_jsr) if cert_jsr else np.array([0.0])

    # (B) refresh the uncertified attribution with the longer product length.
    n_exp = n_gap = 0
    for rec in uncert:
        g = CoupledGene.make(decay=np.asarray(rec["decay"]), W=np.asarray(rec["W"]).reshape(2, 2))
        j = jsr_lb(_vertices_n2(g), max_len=max_len)
        if j >= 1.0 - 1e-9:
            n_exp += 1
        else:
            n_gap += 1

    out = {
        "max_len": max_len,
        "n_certified_genes": len(deg),
        "certified_jsr_lb_max": round(float(cert_jsr.max()), 4),
        "certified_jsr_lb_mean": round(float(cert_jsr.mean()), 4),
        "n_certified_with_jsr_ge_1": len(bad),
        "CERTIFIER_SOUND_by_JSR": len(bad) == 0,
        "unsound_examples": bad,
        "uncertified_refresh": {
            "n_uncertified": len(uncert),
            "switched_expansive_correct_reject": n_exp,
            "finite_degree_gap_candidate": n_gap,
            "frac_finite_gap": round(n_gap / len(uncert), 3) if uncert else None,
        },
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(_HERE, "verify_certifier_jsr_soundness_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "unsound_examples"}, indent=2), flush=True)
    return out


if __name__ == "__main__":
    ml = int(sys.argv[sys.argv.index("--max-len") + 1]) if "--max-len" in sys.argv else 6
    run(max_len=ml)
