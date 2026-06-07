# SPDX-License-Identifier: Apache-2.0
"""AUDIT: did swapping SCS -> CLARABEL introduce SDP FALSE POSITIVES?

The audit claim is that CLARABEL fixes SCS FALSE NEGATIVES (under-certification). The opposite
error of a solver swap is a FALSE POSITIVE: certifying a gene that is actually switched-EXPANSIVE.

Test (skeptic's strongest oracle): take the genes CLARABEL admits via the SDP gate that SCS rejects
(the NEW admits), and for each independently compute:
  * empirical_rho  : pointwise sup of rho(J) over the (s,x) box  (must be < 1)
  * jsr_lb         : Gripenberg lower bound on JSR{J_v} (max over length-<=K vertex products of
                     rho(prod)^{1/k})  (must be < 1)
If ANY new CLARABEL admit has jsr_lb >= 1 it is SWITCHED-EXPANSIVE => the SDP common-Lyapunov
certificate it received is UNSOUND (a real soundness bug), because a common vertex Lyapunov implies
JSR{J_v} < 1, which forces jsr_lb < 1. jsr_lb is computed WITHOUT the solver, so it cannot be
fooled by the solver.

Runs on the EXACT same 3270-gene Track-D population (build_population(3000,0)).
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

import cvxpy as cp  # noqa: E402
from coupled_map import CoupledGene  # noqa: E402
from redteam_fast import build_population, emp_rho_min_fast  # noqa: E402
from lyapunov_sdp_certifier import certify_common_lyapunov  # noqa: E402
from verifier_deg4 import _vertices_n2  # noqa: E402

EMP_SEED = 777
EMP_N = 6000
TOL = 1e-9


def jsr_lb(vertices, max_len: int = 6) -> float:
    """Gripenberg lower bound on JSR{vertices}. Solver-independent product oracle."""
    V = [np.asarray(M, dtype=np.float64) for M in vertices]
    best = 0.0
    for k in range(1, max_len + 1):
        for combo in itertools.product(range(len(V)), repeat=k):
            P = np.eye(V[0].shape[0])
            for i in combo:
                P = V[i] @ P
            best = max(best, float(np.max(np.abs(np.linalg.eigvals(P)))) ** (1.0 / k))
    return best


def sdp_admit(gene: CoupledGene, dom: str, solver) -> bool:
    """Bare SDP common-Lyapunov admit (NO inf/2-norm short-circuit, NO rho pre-screen).

    We deliberately test the SOLVER's verdict directly so the comparison is purely SCS vs CLARABEL.
    Note certify_common_lyapunov already re-checks the returned P's eigenvalues independently, so a
    True here means the solver returned a P that PASSES the independent PD re-check.
    """
    r = certify_common_lyapunov(gene, t_domain=dom, solver=solver)
    return r.certified is True


def main():
    t0 = time.time()
    genes_raw, n_grid = build_population(3000, 0)
    n_total = len(genes_raw)
    print(f"population total={n_total} grid={n_grid}", flush=True)
    print(f"cvxpy {cp.__version__} solvers={[s for s in ('SCS', 'CLARABEL') if s in cp.installed_solvers()]}", flush=True)

    new_admits = {"free01": [], "tmin1": []}  # CLARABEL admits, SCS rejects
    lost_admits = {"free01": [], "tmin1": []}  # SCS admits, CLARABEL rejects (info)
    counts = {"free01": {"scs": 0, "clarabel": 0}, "tmin1": {"scs": 0, "clarabel": 0}}

    for gi, (decay, W) in enumerate(genes_raw):
        decay = np.clip(decay, 0.0, 1.0)
        W = np.clip(W, -2.0, 2.0)
        g = CoupledGene.make(decay=decay, W=W)
        for dom in ("free01", "tmin1"):
            a_scs = sdp_admit(g, dom, cp.SCS)
            a_cla = sdp_admit(g, dom, cp.CLARABEL)
            counts[dom]["scs"] += int(a_scs)
            counts[dom]["clarabel"] += int(a_cla)
            if a_cla and not a_scs:
                new_admits[dom].append((gi, decay, W))
            if a_scs and not a_cla:
                lost_admits[dom].append((gi, decay, W))
        if (gi + 1) % 500 == 0:
            print(f"  ...{gi + 1}/{n_total} ({time.time() - t0:.1f}s)", flush=True)

    out = {"meta": {"emp_seed": EMP_SEED, "emp_n": EMP_N, "n_genes": n_total,
                    "cvxpy": cp.__version__, "max_jsr_len": 6}}
    overall_bad = 0
    for dom in ("free01", "tmin1"):
        recs = []
        worst_rho = 0.0
        worst_jsr = 0.0
        n_bad_rho = 0
        n_bad_jsr = 0
        for (gi, decay, W) in new_admits[dom]:
            g = CoupledGene.make(decay=decay, W=W)
            erho = emp_rho_min_fast(decay, W, n_samples=EMP_N, seed=EMP_SEED)
            jl = jsr_lb(_vertices_n2(g), max_len=6)
            worst_rho = max(worst_rho, erho)
            worst_jsr = max(worst_jsr, jl)
            bad_rho = erho >= 1.0 - TOL
            bad_jsr = jl >= 1.0 - TOL
            n_bad_rho += int(bad_rho)
            n_bad_jsr += int(bad_jsr)
            recs.append({"index": gi, "decay": decay.tolist(), "W": W.tolist(),
                         "emp_rho": erho, "jsr_lb": jl,
                         "bad_rho": bad_rho, "bad_jsr": bad_jsr})
        overall_bad += n_bad_jsr
        # sort worst-jsr first for inspection
        recs.sort(key=lambda r: -r["jsr_lb"])
        out[dom] = {
            "scs_admits": counts[dom]["scs"],
            "clarabel_admits": counts[dom]["clarabel"],
            "n_new_admits_clarabel_not_scs": len(new_admits[dom]),
            "n_lost_admits_scs_not_clarabel": len(lost_admits[dom]),
            "worst_new_admit_emp_rho": worst_rho,
            "worst_new_admit_jsr_lb": worst_jsr,
            "n_new_admit_rho_ge_1": n_bad_rho,
            "n_new_admit_jsr_ge_1": n_bad_jsr,
            "SOUND_no_false_positive": (n_bad_rho == 0 and n_bad_jsr == 0),
            "new_admit_examples_worst_jsr_first": recs[:15],
        }
        print(f"[{dom}] SCS_admits={counts[dom]['scs']} CLARABEL_admits={counts[dom]['clarabel']} "
              f"NEW(cla\\scs)={len(new_admits[dom])} LOST(scs\\cla)={len(lost_admits[dom])}", flush=True)
        print(f"[{dom}] worst NEW-admit emp_rho={worst_rho:.6f} jsr_lb={worst_jsr:.6f} "
              f"bad_rho={n_bad_rho} bad_jsr={n_bad_jsr} "
              f"SOUND={out[dom]['SOUND_no_false_positive']}", flush=True)

    out["OVERALL_SOUND_no_clarabel_false_positive"] = (overall_bad == 0)
    out["meta"]["elapsed_s"] = round(time.time() - t0, 1)
    op = os.path.join(_HERE, "_audit_clarabel_false_positive_results.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {op}", flush=True)
    print(f"OVERALL_SOUND_no_clarabel_false_positive = {out['OVERALL_SOUND_no_clarabel_false_positive']}", flush=True)
    return out


if __name__ == "__main__":
    main()
