# SPDX-License-Identifier: Apache-2.0
"""Phase 1 — does the degree-8 SOS Lyapunov rung recover the deg6-residual? (DEG8_JSR_PREREGISTRATION)

Targets the 53 deg6-residual genes (rho<1 pointwise but degree-6 cannot certify) from
`exp_deg6_residual_genes.json`. degree-8 = `certify_degN(vertices, degree=4)` (symmetric 4th
Kronecker power). Measures:
  G-8A recovery: of the finite-gap subset (jsr_lb<1), how many does degree-8 newly certify?
  G-8B complementarity: deg6∖deg8 and deg8∖deg6 over the residual pool (still non-nested?).
  G-8C soundness: every degree-8-certified gene has jsr_lb<1 (0 unsound) AND none of the
       6 switched-expansive (jsr_lb>=1) genes is certified.
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
from verifier_deg6 import certify_degN, mono_basis, sym_power  # noqa: E402
from verifier_jsr import jsr_lower_bound  # noqa: E402


def _gene(rec):
    return CoupledGene.make(decay=np.asarray(rec["decay"]), W=np.asarray(rec["W"]).reshape(2, 2))


def run(max_len: int = 6) -> dict:
    t0 = time.time()
    # sym_power(.,4) correctness vs brute force (the new lift).
    rng = np.random.default_rng(4)
    sym_ok = True
    for n in (2, 3, 4):
        A = rng.standard_normal((n, n)); z = rng.standard_normal(n)
        basis = mono_basis(n, 4)
        m = np.array([np.prod([z[i] for i in b]) for b in basis])
        Az = A @ z
        m_true = np.array([np.prod([Az[i] for i in b]) for b in basis])
        if float(np.max(np.abs(sym_power(A, 4) @ m - m_true))) >= 1e-8:
            sym_ok = False

    with open(os.path.join(_HERE, "exp_deg6_residual_genes.json"), encoding="utf-8") as f:
        residual = json.load(f)["residual_uncert"]   # rho<1 but deg6 fails

    deg6_recover = deg8_recover = 0       # of the residual, how many each NEWLY certifies
    deg6_only = deg8_only = both = neither = 0
    deg8_unsound = 0                      # deg8-certified with jsr_lb>=1 (must be 0)
    expansive_certified = 0              # a jsr_lb>=1 gene wrongly certified by deg8 (must be 0)
    finite_gap = 0
    per = []
    for rec in residual:
        g = _gene(rec)
        V = _vertices_n2(g)
        c6 = certify_degN(V, 3)          # degree-6 (should be False — these are deg6-residual)
        c8 = certify_degN(V, 4)          # degree-8
        jlb = jsr_lower_bound(V, max_len=max_len)
        if jlb < 1.0:
            finite_gap += 1
        if c6:
            deg6_recover += 1
        if c8:
            deg8_recover += 1
            if jlb >= 1.0:
                deg8_unsound += 1
                expansive_certified += 1
        if c6 and not c8:
            deg6_only += 1
        elif c8 and not c6:
            deg8_only += 1
        elif c6 and c8:
            both += 1
        else:
            neither += 1
        per.append({"jsr_lb": round(jlb, 4), "deg6": c6, "deg8": c8})

    n = len(residual)
    out = {
        "sym_power_deg4_correct": sym_ok,
        "n_deg6_residual": n, "finite_gap_in_residual": finite_gap,
        "deg6_recovers_of_residual": deg6_recover,   # ~0 by construction (these are deg6-residual)
        "deg8_recovers_of_residual": deg8_recover,
        "complementarity": {"deg6_only": deg6_only, "deg8_only": deg8_only,
                            "both": both, "neither": neither},
        "deg8_unsound_certs": deg8_unsound,
        "expansive_certified_by_deg8": expansive_certified,
        # gates
        "G_8A_deg8_advances": deg8_recover >= 1,
        "G_8B_non_nested": (deg6_only > 0 and deg8_only > 0),
        "G_8C_soundness": (deg8_unsound == 0 and expansive_certified == 0),
        "per_gene": per,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(_HERE, "exp_deg8_ladder_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_gene"}, indent=2), flush=True)
    return out


if __name__ == "__main__":
    run()
