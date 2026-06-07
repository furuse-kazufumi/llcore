# SPDX-License-Identifier: Apache-2.0
"""#2 — does the degree-4 (non-quadratic) Lyapunov backend reach the Track-D D4 residual?

Sample n=2 genes, classify by the cheapest sufficient certificate, isolate the D4 RESIDUAL
(empirically contracting ρ<1, but NOT certified by inf-norm / 2-norm / common-QUADRATIC SDP),
and measure how many the degree-4 certifier recovers — and that it stays SOUND (every
deg4-certified gene has empirical ρ<1).
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
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coupled_map import CoupledGene  # noqa: E402
from coupled_components import _inf_certifies, _two_certifies, _sdp_certifies, empirical_spectral_radius  # noqa: E402
from verifier_deg4 import cert_deg4_n2  # noqa: E402


def run(n_samples: int = 4000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n_inf = n_two = n_sdp = n_deg4_extra = 0
    residual = []          # rho<1 but inf/two/sdp all fail
    residual_recovered = 0
    deg4_unsound = 0       # deg4-certified but emp rho>=1 (must be 0)
    n_contracting = 0
    for _ in range(n_samples):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        ci, ct = _inf_certifies(g), _two_certifies(g)
        if ci:
            n_inf += 1
        elif ct:
            n_two += 1
        cs = _sdp_certifies(g)
        if cs and not ci and not ct:
            n_sdp += 1
        # any sound quadratic-class certificate?
        quad = ci or ct or cs
        rho = empirical_spectral_radius(g, n_samples=4000)
        if rho < 1.0:
            n_contracting += 1
        # degree-4 certificate
        cd = cert_deg4_n2(g)
        if cd and rho >= 1.0:
            deg4_unsound += 1
        if cd and not quad:
            n_deg4_extra += 1          # deg4 certifies something the quadratic class does not
        # D4 residual: empirically contracting but no quadratic-class certificate
        if rho < 1.0 and not quad:
            residual.append(1)
            if cd:
                residual_recovered += 1

    n_res = len(residual)
    out = {
        "n_samples": n_samples,
        "n_inf": n_inf, "n_two_only": n_two, "n_sdp_only": n_sdp,
        "n_empirically_contracting": n_contracting,
        "d4_residual_count": n_res,
        "d4_residual_recovered_by_deg4": residual_recovered,
        "d4_residual_recovery_rate": (residual_recovered / n_res) if n_res else None,
        "deg4_extra_over_quadratic": n_deg4_extra,
        "deg4_unsound_certs": deg4_unsound,  # MUST be 0
    }
    p = os.path.join(_HERE, "exp_deg4_results.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    print(f"-> {p}", flush=True)
    return out


if __name__ == "__main__":
    ns = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 4000
    run(n_samples=ns)
