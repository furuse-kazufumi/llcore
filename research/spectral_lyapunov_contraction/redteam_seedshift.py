# SPDX-License-Identifier: Apache-2.0
"""Seed-fragility attack on the D2/D3/D4 comparison gates.

The gate counts hinge on the rho<1 classification, computed by the implementer with the empirical
oracle at seed 777, n=6000. If a DIFFERENT seed + denser sample reclassifies genes near rho=1, the
'21% / 254 / 33%' fractions could move. Here we recompute the rho<1 set with the INDEPENDENT dense
oracle (seed 20260602, n=40000 + corners + grid) and re-derive D2/D3/D4 against the SAME certifier
admit sets. If the headline numbers are robust to this reclassification, the comparison is not
seed-fragile.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import warnings
warnings.filterwarnings("ignore")

_HERE = Path(__file__).resolve().parent
_C = (_HERE.parent / "coupled_z3_contraction").resolve()
for p in (str(_HERE), str(_C)):
    if p not in sys.path:
        sys.path.insert(0, p)

from coupled_map import CoupledGene, t_min_per_coord, infnorm_over_box_freeT
from redteam_fast import build_population
from two_norm_vertex_certifier import certify_2norm_contraction
from lyapunov_sdp_certifier import certify_common_lyapunov, CVXPY_AVAILABLE
from redteam_track_d import emp_norms_dense  # independent dense oracle

TOL = 1e-9


def main():
    genes, n_grid = build_population(3000, 0)
    dom = "tmin1"
    inf_a, two_a, sdp_a, rho_lt1 = set(), set(), set(), set()
    for gi, (decay, W) in enumerate(genes):
        decay = np.clip(decay, 0, 1); W = np.clip(W, -2, 2)
        g = CoupledGene.make(decay=decay, W=W)
        t_lo = t_min_per_coord(g)
        if infnorm_over_box_freeT(g, t_lo=t_lo) < 1.0:
            inf_a.add(gi)
        if certify_2norm_contraction(g, t_domain=dom).certified:
            two_a.add(gi)
        if CVXPY_AVAILABLE and certify_common_lyapunov(g, t_domain=dom).certified:
            sdp_a.add(gi)
        # INDEPENDENT rho classification (fresh seed, dense)
        _, _, emp_rho = emp_norms_dense(decay, W)
        if emp_rho < 1.0 - TOL:
            rho_lt1.add(gi)
        if (gi + 1) % 1000 == 0:
            print(f"  ...{gi+1}/{len(genes)}", flush=True)

    n_rho = len(rho_lt1)
    inf_reject_rho = (rho_lt1 - inf_a)             # 850-analogue under fresh oracle
    n_850 = len(inf_reject_rho)
    two_gain = (two_a - inf_a) & rho_lt1
    sdp_gain = (sdp_a - two_a) & rho_lt1
    residual = rho_lt1 - inf_a - two_a - sdp_a
    out = {
        "oracle": "INDEPENDENT dense (seed 20260602, n=40000+corners+grid)",
        "n_rho_lt1": n_rho,
        "n_inf_reject_rho_lt1 (850-analogue)": n_850,
        "D2 two_gain_over_inf (rho<1)": len(two_gain),
        "D2 fraction_of_850": len(two_gain) / n_850 if n_850 else None,
        "D3 sdp_gain_over_two (rho<1)": len(sdp_gain),
        "D3 fraction_of_850": len(sdp_gain) / n_850 if n_850 else None,
        "D3 two_beats_sdp": len(two_a - sdp_a),
        "D4 residual_no_certifier": len(residual),
        "D4 fraction_of_rho_lt1": len(residual) / n_rho if n_rho else None,
        "union_all3": len(inf_a | two_a | sdp_a),
        "inf_unique_vs_sdp": len(inf_a - sdp_a),
    }
    print(json.dumps(out, indent=2))
    (_HERE / "redteam_seedshift_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
