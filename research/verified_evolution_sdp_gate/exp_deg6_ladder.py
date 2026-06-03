# SPDX-License-Identifier: Apache-2.0
"""EXP-A — the COVERAGE frontier of the verifier ladder + degree-4/6 complementarity.

On a pool of empirically-contracting n=2 genes, count how many each cumulative union certifies:

    L0=inf  ⊆  L1=inf∪2norm  ⊆  L2=sdp  ⊆  L3=sdp∪deg4  ⊆  L4=sdp∪deg4∪deg6

and whether the lifted degree-4 / degree-6 certificates are non-nested (deg4∖deg6 and deg6∖deg4
both non-empty). Soundness: every deg-certified gene must be empirically contracting. Residual
genes (ρ<1 but not even L4) are saved for the JSR honesty oracle.

Pre-registered gates (DEG6_PREREGISTRATION.md): G-A1 L4−L2≥+5 ; G-A2 both complements>0 ; G-A3 0 unsound.
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
from coupled_components import (  # noqa: E402
    _inf_certifies, _two_certifies, _sdp_certifies, empirical_spectral_radius,
)
from verifier_deg4 import cert_deg4_n2  # noqa: E402
from verifier_deg6 import cert_deg6_n2  # noqa: E402


def run(n_target: int = 300, seed: int = 2024, time_cap: float = 420.0,
        sound_samples: int = 20000) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    counts = dict(inf=0, two_only=0, sdp_only=0, deg4_only=0, deg6_only=0,
                  deg4_and_deg6=0, residual_uncert=0)
    deg_certified = []     # residual genes deg4/deg6 certifies (for soundness + audit)
    residual_uncert = []   # rho<1 but L4 fails (for JSR oracle)
    unsound = 0
    n_contr = 0
    scanned = 0
    while n_contr < n_target and time.time() - t0 < time_cap:
        scanned += 1
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if empirical_spectral_radius(g, n_samples=4000) >= 1.0:
            continue
        n_contr += 1
        if _inf_certifies(g):
            counts["inf"] += 1
            continue
        if _two_certifies(g):
            counts["two_only"] += 1
            continue
        if _sdp_certifies(g):
            counts["sdp_only"] += 1
            continue
        # residual region (no quadratic-class certificate). test deg4 / deg6.
        d4 = cert_deg4_n2(g)
        d6 = cert_deg6_n2(g)
        rec = {"decay": g.decay.tolist(), "W": g.W.reshape(-1).tolist(), "deg4": d4, "deg6": d6}
        if d4 or d6:
            # soundness re-check (independent, higher-sample from-below oracle)
            rho_hi = empirical_spectral_radius(g, n_samples=sound_samples)
            rec["rho_sound"] = rho_hi
            if rho_hi >= 1.0:
                unsound += 1
            deg_certified.append(rec)
            if d4 and not d6:
                counts["deg4_only"] += 1
            elif d6 and not d4:
                counts["deg6_only"] += 1
            else:
                counts["deg4_and_deg6"] += 1
        else:
            counts["residual_uncert"] += 1
            residual_uncert.append(rec)

    L0 = counts["inf"]
    L1 = L0 + counts["two_only"]
    L2 = L1 + counts["sdp_only"]
    L3 = L2 + counts["deg4_only"] + counts["deg4_and_deg6"]
    L4 = L3 + counts["deg6_only"]
    deg4_minus_deg6 = counts["deg4_only"]
    deg6_minus_deg4 = counts["deg6_only"]

    out = {
        "seed": seed, "scanned": scanned, "n_contracting": n_contr,
        "counts": counts,
        "cumulative": {"L0_inf": L0, "L1_two": L1, "L2_sdp": L2, "L3_deg4": L3, "L4_deg6": L4},
        "complementarity": {"deg4_minus_deg6": deg4_minus_deg6,
                            "deg6_minus_deg4": deg6_minus_deg4},
        "deg6_beyond_deg4": L4 - L3,
        "deg_union_beyond_sdp": L4 - L2,
        "n_unsound_deg_certs": unsound,
        # pre-registered gates
        "G_A1_coverage_advances": (L4 - L2) >= 5,
        "G_A2_ladder_non_nested": (deg4_minus_deg6 > 0 and deg6_minus_deg4 > 0),
        "G_A3_soundness": unsound == 0,
        "elapsed_s": round(time.time() - t0, 1),
    }
    # persist residual gene lists for downstream oracles
    with open(os.path.join(_HERE, "exp_deg6_ladder_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(_HERE, "exp_deg6_residual_genes.json"), "w", encoding="utf-8") as f:
        json.dump({"deg_certified": deg_certified, "residual_uncert": residual_uncert}, f, indent=2)
    print(json.dumps(out, indent=2), flush=True)
    return out


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300
    run(n_target=n)
