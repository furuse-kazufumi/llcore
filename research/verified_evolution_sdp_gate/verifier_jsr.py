# SPDX-License-Identifier: Apache-2.0
"""JSR bracket — the theoretical limit of the SOS Lyapunov coverage ladder (CPU).

The verifier ladder certifies a common lifted Lyapunov over the t-box vertex set {J_v}; this
certifies the joint spectral radius JSR{J_v} < 1 (the exact contraction rate of the box-switched
linearisation, which bounds the nonlinear map by the convex-hull theorem). The degree-d SOS
Lyapunov is a CPU approximation of JSR; this module brackets the true JSR per gene:

  * LOWER bound (Gripenberg)      jsr_lb = max over length-≤K vertex products of ρ(∏)^{1/k}.
  * UPPER bound (SOS-degree-d)    gamma_star_d = smallest γ such that {J_v/γ} admits a common
    degree-d lifted Lyapunov (bisection; `sym_power(J/γ,k)=sym_power(J,k)/γ^k` ⇒ monotone in γ).

True JSR ∈ [jsr_lb, gamma_star_d]. A gene is **certified contracting at the SOS-degree-d limit**
iff gamma_star_d < 1. As d grows the upper bound decreases toward JSR (Parrilo–Jadbabaie), so the
ladder closes the residual up to its near-boundary tail. "Exact JSR" is NP-hard — we report a
tight *bracket*, never an exact value, and never call a gene expansive unless jsr_lb ≥ 1.
"""
from __future__ import annotations

import itertools

import numpy as np

from verifier_deg6 import certify_degN  # general degree-d lifted Lyapunov (degree=2→deg4, 3→deg6, 4→deg8)


def jsr_lower_bound(vertices, max_len: int = 6) -> float:
    V = [np.asarray(M, dtype=np.float64) for M in vertices]
    best = 0.0
    for k in range(1, max_len + 1):
        for combo in itertools.product(range(len(V)), repeat=k):
            P = np.eye(V[0].shape[0])
            for i in combo:
                P = V[i] @ P
            best = max(best, float(np.max(np.abs(np.linalg.eigvals(P)))) ** (1.0 / k))
    return best


def _certifies_scaled(vertices, degree: int, gamma: float, margin: float = 1e-7) -> bool:
    """Common degree-(2*degree) Lyapunov for the γ-scaled vertices {J_v/γ}."""
    scaled = [np.asarray(J, dtype=np.float64) / gamma for J in vertices]
    return certify_degN(scaled, degree, margin=margin)


def jsr_upper_bound_sos(vertices, degree: int, lo: float | None = None,
                        hi: float = 3.0, tol: float = 1e-3, margin: float = 1e-7) -> float:
    """gamma_star_d = smallest γ∈[lo,hi] s.t. {J_v/γ} admits a common degree-(2*degree) Lyapunov.

    Monotone: larger γ ⇒ smaller scaled matrices ⇒ easier to certify. Bisection. lo defaults to the
    spectral-radius floor max_v ρ(J_v) (JSR ≥ that). Returns float('inf') if not certifiable at hi.
    """
    V = [np.asarray(J, dtype=np.float64) for J in vertices]
    rho_floor = max(float(np.max(np.abs(np.linalg.eigvals(J)))) for J in V)
    lo = rho_floor if lo is None else max(lo, rho_floor)
    # guard: must be certifiable at hi, else the bound is > hi.
    if not _certifies_scaled(V, degree, hi, margin):
        return float("inf")
    lo_g, hi_g = lo, hi
    # ensure lo is infeasible (push down a touch below rho_floor where pre-screen fails)
    if _certifies_scaled(V, degree, lo_g, margin):
        return lo_g  # already certifiable at the floor (γ* ≈ rho_floor)
    while hi_g - lo_g > tol:
        mid = 0.5 * (lo_g + hi_g)
        if _certifies_scaled(V, degree, mid, margin):
            hi_g = mid
        else:
            lo_g = mid
    return hi_g


def jsr_bracket(vertices, degree: int = 4, max_len: int = 6) -> dict:
    """Bracket the JSR of {J_v}: [jsr_lb, gamma_star_d]. degree=4 ⇒ degree-8 SOS upper bound."""
    lb = jsr_lower_bound(vertices, max_len=max_len)
    ub = jsr_upper_bound_sos(vertices, degree, lo=lb)
    return {"jsr_lb": lb, "gamma_star": ub, "degree_lift": degree,
            "certified_contracting": ub < 1.0, "bracket_width": (ub - lb) if np.isfinite(ub) else None}


# ---- n=2 adapter -------------------------------------------------------------- #


def _vertices_n2(gene):
    from verifier_deg4 import _vertices_n2 as _v
    return _v(gene)


def jsr_certifies_n2(gene, degree: int = 4) -> bool:
    """Certify contraction of an n=2 gene at the SOS-degree-(2*degree) JSR limit (γ*_d < 1)."""
    return jsr_upper_bound_sos(_vertices_n2(gene), degree, lo=None) < 1.0


if __name__ == "__main__":
    # smoke: a clearly-contracting gene brackets below 1; the bracket tightens with degree.
    import os
    import sys
    _HERE = os.path.dirname(os.path.abspath(__file__))
    for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
        sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", _d)))
    from coupled_map import CoupledGene
    g = CoupledGene.make(decay=[0.7, 0.7], W=[[0.0, 0.5], [-0.5, 0.0]])
    v = _vertices_n2(g)
    for deg in (1, 2, 3, 4):
        b = jsr_bracket(v, degree=deg)
        print(f"degree-{2*deg}: jsr_lb={b['jsr_lb']:.4f} gamma*={b['gamma_star']:.4f} "
              f"width={b['bracket_width']:.4f} certified={b['certified_contracting']}")
