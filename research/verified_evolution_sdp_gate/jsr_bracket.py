# SPDX-License-Identifier: Apache-2.0
"""JSR honesty oracle — attribute the deg6-residual.

The verifier ladder certifies a COMMON Lyapunov over the t-box vertex Jacobian set {J_v}, i.e. it
proves the *switched* contraction (joint spectral radius JSR{J_v} < 1) — strictly stronger than
the pointwise ρ(J)<1 that ``empirical_spectral_radius`` falsifies from below. So a gene with
pointwise ρ<1 at every vertex can STILL be switched-expansive (JSR ≥ 1), in which case NO finite-
degree Lyapunov over the vertex set can ever certify it: deg6 rejecting it is CORRECT, not a
degree limitation.

For each deg6-residual gene (ρ<1 but L4 fails), compute a JSR LOWER bound over the vertex set
(max over length-k products of ρ(∏)^{1/k}). Split the residual:
  * JSR_lb ≥ 1  → vertex relaxation provably non-contracting ⇒ correct rejection (not a degree gap)
  * JSR_lb < 1  → inconclusive: a genuine finite-degree gap candidate (degree-8+/exact-JSR rung)
(honest caveat: the vertex set is a SOUND OVER-APPROX of the achievable-t set, so JSR_lb≥1 means
the *relaxation* is expansive; the tighter achievable set could differ — the certifiers use the
same relaxation, so the attribution is internally consistent.)
"""
from __future__ import annotations

import json
import os
import sys

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


def _gene(rec) -> CoupledGene:
    return CoupledGene.make(decay=np.asarray(rec["decay"]),
                            W=np.asarray(rec["W"]).reshape(2, 2))


def jsr_lower_bound(vertices: list[np.ndarray], max_len: int = 5) -> dict:
    """JSR lower bound = max over products of length 1..max_len of ρ(∏)^{1/len}.
    Enumerate all |V|^len products (|V|=4, max_len=5 ⇒ ≤1364 products)."""
    import itertools
    V = [np.asarray(M, dtype=np.float64) for M in vertices]
    best = 0.0
    best_len = 1
    for k in range(1, max_len + 1):
        for combo in itertools.product(range(len(V)), repeat=k):
            P = np.eye(V[0].shape[0])
            for i in combo:
                P = V[i] @ P
            rho = float(np.max(np.abs(np.linalg.eigvals(P))))
            val = rho ** (1.0 / k)
            if val > best:
                best, best_len = val, k
    return {"jsr_lb": best, "argmax_len": best_len}


def run() -> dict:
    src = os.path.join(_HERE, "exp_deg6_residual_genes.json")
    with open(src, encoding="utf-8") as f:
        data = json.load(f)
    residual = data["residual_uncert"]  # ρ<1 (pointwise) but L4 (deg6) fails

    n_expansive = 0       # JSR_lb >= 1 -> correct rejection (switched-expansive over the box)
    n_finite_gap = 0      # JSR_lb < 1  -> candidate genuine finite-degree gap
    per_gene = []
    for rec in residual:
        g = _gene(rec)
        verts = _vertices_n2(g)
        pointwise_max_rho = max(float(np.max(np.abs(np.linalg.eigvals(J)))) for J in verts)
        jb = jsr_lower_bound(verts, max_len=5)
        cls = "switched_expansive_correct_reject" if jb["jsr_lb"] >= 1.0 - 1e-9 \
            else "finite_degree_gap_candidate"
        if jb["jsr_lb"] >= 1.0 - 1e-9:
            n_expansive += 1
        else:
            n_finite_gap += 1
        per_gene.append({"pointwise_max_rho_vertices": round(pointwise_max_rho, 4),
                         "jsr_lb": round(jb["jsr_lb"], 4), "argmax_len": jb["argmax_len"],
                         "class": cls})

    n = len(residual)
    out = {
        "n_deg6_residual": n,
        "switched_expansive_correct_reject": n_expansive,
        "finite_degree_gap_candidate": n_finite_gap,
        "frac_correct_reject": round(n_expansive / n, 3) if n else None,
        "frac_finite_degree_gap": round(n_finite_gap / n, 3) if n else None,
        "per_gene": per_gene,
    }
    with open(os.path.join(_HERE, "jsr_bracket_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_gene"}, indent=2), flush=True)
    return out


if __name__ == "__main__":
    run()
