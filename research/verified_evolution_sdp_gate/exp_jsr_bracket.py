# SPDX-License-Identifier: Apache-2.0
"""Phase 2 — JSR bracket: how close does CPU SOS get to exact-JSR on the deg6-residual?

For each deg6-residual gene, bracket the true JSR of the t-box vertex set:
  lower = jsr_lb (Gripenberg, max over length-≤K products of ρ(∏)^{1/k}),
  upper = γ*_d (smallest γ s.t. {J_v/γ} admits a common degree-(2d) Lyapunov), d=2,3,4 (deg4/6/8).
A gene is certified contracting at the SOS-degree-(2d) limit iff γ*_d < 1.

Measures (DEG8_JSR_PREREGISTRATION): G-JSR (fraction with γ*_8<1), G-tight (mean bracket width
γ*_d−jsr_lb shrinks with d), and the near-boundary tail (high jsr_lb) that stays uncertified.
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

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coupled_map import CoupledGene  # noqa: E402
from verifier_deg4 import _vertices_n2  # noqa: E402
from verifier_jsr import jsr_lower_bound, jsr_upper_bound_sos  # noqa: E402


def _gene(rec):
    return CoupledGene.make(decay=np.asarray(rec["decay"]), W=np.asarray(rec["W"]).reshape(2, 2))


def run(max_len: int = 6) -> dict:
    t0 = time.time()
    with open(os.path.join(_HERE, "exp_deg6_residual_genes.json"), encoding="utf-8") as f:
        residual = json.load(f)["residual_uncert"]

    rows = []
    closed_d = {2: 0, 3: 0, 4: 0}        # γ*_d < 1 counts (deg4 / deg6 / deg8)
    width_sum = {2: 0.0, 3: 0.0, 4: 0.0}
    width_n = {2: 0, 3: 0, 4: 0}
    bracket_invalid = 0                  # jsr_lb > γ*_d (lower>upper) — would be a bug
    monotone_violations = 0              # γ*_2 < γ*_3 or γ*_3 < γ*_4 (upper must be non-increasing)
    finite_gap = 0
    for rec in residual:
        g = _gene(rec)
        V = _vertices_n2(g)
        lb = jsr_lower_bound(V, max_len=max_len)
        if lb < 1.0:
            finite_gap += 1
        gstar = {}
        for d in (2, 3, 4):
            ub = jsr_upper_bound_sos(V, d, lo=lb)
            gstar[d] = ub
            if np.isfinite(ub):
                if ub < 1.0:
                    closed_d[d] += 1
                if ub + 1e-6 < lb:
                    bracket_invalid += 1
                width_sum[d] += (ub - lb)
                width_n[d] += 1
        # upper bound monotone non-increasing in d
        finite = [gstar[d] for d in (2, 3, 4) if np.isfinite(gstar[d])]
        for a, b in zip(finite, finite[1:]):
            if b > a + 1e-4:
                monotone_violations += 1
        rows.append({"jsr_lb": round(lb, 4),
                     "gstar_deg4": round(gstar[2], 4) if np.isfinite(gstar[2]) else None,
                     "gstar_deg6": round(gstar[3], 4) if np.isfinite(gstar[3]) else None,
                     "gstar_deg8": round(gstar[4], 4) if np.isfinite(gstar[4]) else None})

    n = len(residual)
    mean_width = {f"deg{2*d}": (round(width_sum[d] / width_n[d], 4) if width_n[d] else None)
                  for d in (2, 3, 4)}
    # near-boundary tail = finite-gap genes deg8 still cannot close
    tail = [r for r in rows if r["jsr_lb"] < 1.0 and (r["gstar_deg8"] is None or r["gstar_deg8"] >= 1.0)]
    out = {
        "n_deg6_residual": n, "finite_gap": finite_gap,
        "closed_by_gamma_star": {"deg4": closed_d[2], "deg6": closed_d[3], "deg8": closed_d[4]},
        "frac_finite_gap_closed_by_deg8": round(closed_d[4] / finite_gap, 3) if finite_gap else None,
        "mean_bracket_width": mean_width,         # should shrink deg4 -> deg6 -> deg8
        "bracket_tightens_with_degree": (mean_width["deg8"] is not None and mean_width["deg4"] is not None
                                         and mean_width["deg8"] <= mean_width["deg4"]),
        "bracket_invalid_lb_gt_ub": bracket_invalid,     # must be 0
        "upper_bound_monotone_violations": monotone_violations,  # must be 0
        "near_boundary_tail_count": len(tail),
        "near_boundary_tail_jsr_lb": sorted([r["jsr_lb"] for r in tail], reverse=True)[:10],
        "G_JSR_majority_closed": (closed_d[4] / finite_gap > 0.5) if finite_gap else False,
        "rows": rows,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(_HERE, "exp_jsr_bracket_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2), flush=True)
    return out


if __name__ == "__main__":
    run()
