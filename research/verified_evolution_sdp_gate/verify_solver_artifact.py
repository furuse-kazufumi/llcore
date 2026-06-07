# SPDX-License-Identifier: Apache-2.0
"""VERIFY the adversarial-review keystone IN OUR OWN HANDS (discipline: never adopt an external
finding without first-party reproduction).

Claim under test: the deg4/deg6 "complementarity" (deg4∖deg6=23, deg6∖deg4=13) is an SCS
solver-instability artifact — under the accurate CLARABEL solver the ladder is NESTED.

Also: does CLARABEL recover residual_uncert genes that SCS (default) failed to certify (false
negatives)? If yes, the SCS coverage numbers UNDER-count.

FIXTURE NOTE (2026-06-03 audit fix): this keystone demonstration reads the IMMUTABLE SCS-era
snapshot ``exp_deg6_residual_genes_scs_era.json`` (deg_certified=54, residual_uncert=53), NOT the
live ``exp_deg6_residual_genes.json``. The live file was OVERWRITTEN by the later CLARABEL ladder
run (it now holds the post-correction 4 deg_certified / 10 residual_uncert pool), which silently
destroyed the SCS-era battery this demonstration depends on. The snapshot was restored verbatim
from git commit ``cd400ef`` and is treated as a fixed regression fixture so the solver-swap keystone
in DEG6_VERDICT.md §0 stays PERMANENTLY reproducible. See also
``test_solver_swap_regression.py`` (the standing pytest guard) and
``AUDIT_SCS_CLARABEL_2026-06-03.md`` (Reproducibility note).
"""
from __future__ import annotations

import itertools
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
    sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", _d)))

import cvxpy as cp  # noqa: E402
from coupled_map import CoupledGene  # noqa: E402
from verifier_deg4 import _vertices_n2  # noqa: E402
from verifier_deg6 import sym_power  # noqa: E402


def certify(vertices, degree: int, solver, margin: float = 1e-7) -> bool:
    for J in vertices:
        if float(np.max(np.abs(np.linalg.eigvals(J)))) >= 1.0:
            return False
    lifts = [sym_power(J, degree) for J in vertices]
    m = lifts[0].shape[0]
    P = cp.Variable((m, m), symmetric=True)
    I = np.eye(m)
    cons = [P >> I] + [P - L.T @ P @ L >> margin * I for L in lifts]
    try:
        cp.Problem(cp.Minimize(cp.trace(P)), cons).solve(solver=solver)
    except Exception:
        return False
    if P.value is None:
        return False
    Pv = 0.5 * (P.value + P.value.T)
    if float(np.min(np.linalg.eigvalsh(Pv))) <= 0.0:
        return False
    for L in lifts:
        M = Pv - L.T @ Pv @ L
        if float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T)))) <= 0.0:
            return False
    return True


def _gene(rec):
    return CoupledGene.make(decay=np.asarray(rec["decay"]), W=np.asarray(rec["W"]).reshape(2, 2))


print("cvxpy", cp.__version__, "installed solvers:", [s for s in ("SCS", "CLARABEL") if s in cp.installed_solvers()])

# Read the IMMUTABLE SCS-era snapshot, NOT the live exp_deg6_residual_genes.json (which was
# overwritten by the later CLARABEL ladder run -> only 4 deg_certified / 10 residual_uncert remain
# there now). The snapshot preserves the original SCS-era battery (54 SCS-"deg-certified" residual
# genes + 53 SCS-"uncertified" residual genes) so this solver-swap keystone stays reproducible.
# Restored verbatim from git cd400ef:research/verified_evolution_sdp_gate/exp_deg6_residual_genes.json.
_SNAPSHOT = os.path.join(_HERE, "exp_deg6_residual_genes_scs_era.json")
with open(_SNAPSHOT, encoding="utf-8") as f:
    data = json.load(f)
deg = data["deg_certified"]        # 54 genes SCS labelled "deg-certified" (its inflated residual)
uncert = data["residual_uncert"]   # 53 genes SCS could not certify (its false negatives)
assert len(deg) == 54 and len(uncert) == 53, (
    f"SCS-era snapshot must be 54/53, got {len(deg)}/{len(uncert)} — fixture corrupted?"
)
print(f"snapshot: {os.path.basename(_SNAPSHOT)}  deg_certified={len(deg)}  residual_uncert={len(uncert)}")

for solver_name, solver in (("SCS", cp.SCS), ("CLARABEL", cp.CLARABEL)):
    d4o = d6o = both = neither = 0
    for rec in deg:
        V = _vertices_n2(_gene(rec))
        c4 = certify(V, 2, solver)
        c6 = certify(V, 3, solver)
        if c4 and not c6:
            d4o += 1
        elif c6 and not c4:
            d6o += 1
        elif c4 and c6:
            both += 1
        else:
            neither += 1
    print(f"[{solver_name:9s}] on 54 SCS-deg_certified: deg4_only={d4o} deg6_only={d6o} "
          f"both={both} neither={neither}  (complementarity={'YES' if d4o>0 and d6o>0 else 'NO -> nested'})")

# CLARABEL false-negative recovery on the 53 SCS-uncertified residual
rec4 = rec6 = 0
for rec in uncert:
    V = _vertices_n2(_gene(rec))
    if certify(V, 2, cp.CLARABEL):
        rec4 += 1
    if certify(V, 3, cp.CLARABEL):
        rec6 += 1
print(f"[CLARABEL] recovers of 53 SCS-uncertified residual: deg4={rec4} deg6={rec6} "
      f"(SCS false negatives)")
