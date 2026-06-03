# SPDX-License-Identifier: Apache-2.0
"""STANDING regression guard for the SCS->CLARABEL solver-swap keystone (DEG6_VERDICT.md §0).

WHAT THIS PROTECTS
------------------
The arc's most valuable methodology finding: cvxpy's default first-order **SCS** solver returns
FALSE NEGATIVES near the SDP feasibility boundary, fabricating an apparent deg4/deg6 Lyapunov
"complementarity". Under the accurate interior-point **CLARABEL** solver the lifted ladder is
NESTED (deg4 ⊆ deg6) and SCS yields a strictly different admit set. If the certifiers ever silently
revert to the SCS default, the fabricated structure recurs. These tests permanently fail-closed on
that regression.

FIXTURE
-------
The IMMUTABLE SCS-era battery ``exp_deg6_residual_genes_scs_era.json`` (deg_certified=54,
residual_uncert=53) — restored verbatim from git commit ``cd400ef`` after the live
``exp_deg6_residual_genes.json`` was overwritten by the later CLARABEL ladder run. The keystone
demonstration (``verify_solver_artifact.py``) and these tests both read the snapshot so the result
stays reproducible regardless of the live file's contents.

THREE GUARDS
------------
(a) Under CLARABEL the deg4/deg6 ladder is NESTED on the battery: deg4_only == 0
    (no gene is degree-4-certified yet degree-6-rejected).
(b) The SCS vs CLARABEL admit sets DIVERGE — the solver-swap detector still fires: SCS yields a
    strictly different (and on this battery, complementary deg4_only>0 ∧ deg6_only>0) set, proving
    the artifact is reproducible and would be re-detected.
(c) Every CLARABEL-certified gene passes the JSR soundness oracle: jsr_lb < 1 (a product of vertex
    Jacobians never expands ⇒ the certifier is SOUND, not merely the eigen re-check).

Deterministic and fast (sub-minute): pure linear algebra + small SDPs over a fixed 54-gene battery.

Run: py -3.11 -m pytest test_solver_swap_regression.py -q
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import warnings

import numpy as np
import pytest

# SCS emits "Solution may be inaccurate" warnings by design here (that IS the artifact); silence so
# the test output stays clean.
warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

cp = pytest.importorskip("cvxpy", reason="cvxpy required for the solver-swap regression guard")

from coupled_map import CoupledGene  # noqa: E402
from verifier_deg4 import _vertices_n2  # noqa: E402
from verifier_deg6 import sym_power  # noqa: E402

_SNAPSHOT = os.path.join(_HERE, "exp_deg6_residual_genes_scs_era.json")

if "CLARABEL" not in cp.installed_solvers():
    pytest.skip("CLARABEL solver not installed — keystone requires the accurate solver",
                allow_module_level=True)


def _certify(vertices, degree: int, solver, margin: float = 1e-7) -> bool:
    """Common degree-(2*degree) lifted Lyapunov over {J_v} with an EXPLICIT solver, plus the
    independent eigenvalue re-check of P and every decrease-LMI (never solver-blind). Mirrors
    ``verify_solver_artifact.certify`` exactly so the regression matches the keystone."""
    for J in vertices:
        if float(np.max(np.abs(np.linalg.eigvals(J)))) >= 1.0:
            return False
    lifts = [sym_power(J, degree) for J in vertices]
    m = lifts[0].shape[0]
    P = cp.Variable((m, m), symmetric=True)
    eye = np.eye(m)
    cons = [P >> eye] + [P - L.T @ P @ L >> margin * eye for L in lifts]
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


def _jsr_lb(vertices, max_len: int = 4) -> float:
    """JSR lower bound (Gripenberg): max over length-≤max_len vertex products of ρ(∏)^{1/k}.
    A SOUND common-Lyapunov certificate implies JSR{J_v} < 1, hence this lower bound < 1."""
    V = [np.asarray(M, dtype=np.float64) for M in vertices]
    best = 0.0
    for k in range(1, max_len + 1):
        for combo in itertools.product(range(len(V)), repeat=k):
            prod = np.eye(V[0].shape[0])
            for i in combo:
                prod = V[i] @ prod
            best = max(best, float(np.max(np.abs(np.linalg.eigvals(prod)))) ** (1.0 / k))
    return best


def _gene(rec):
    return CoupledGene.make(decay=np.asarray(rec["decay"]), W=np.asarray(rec["W"]).reshape(2, 2))


@pytest.fixture(scope="module")
def battery():
    """The immutable SCS-era residual battery (deg_certified=54, residual_uncert=53)."""
    with open(_SNAPSHOT, encoding="utf-8") as f:
        data = json.load(f)
    deg = data["deg_certified"]
    uncert = data["residual_uncert"]
    assert len(deg) == 54 and len(uncert) == 53, (
        f"SCS-era snapshot must be 54/53 (restored from git cd400ef), got "
        f"{len(deg)}/{len(uncert)} — fixture corrupted, the keystone is not reproducible"
    )
    return deg, uncert


@pytest.fixture(scope="module")
def admit_sets(battery):
    """Per-solver (deg4_only, deg6_only, both, neither) split over the 54-gene battery, plus the
    raw (deg4, deg6) admit sets keyed by gene index. Computed once, shared across tests."""
    deg, _ = battery
    verts = [_vertices_n2(_gene(rec)) for rec in deg]
    out = {}
    for name, solver in (("SCS", cp.SCS), ("CLARABEL", cp.CLARABEL)):
        d4 = {i for i, V in enumerate(verts) if _certify(V, 2, solver)}
        d6 = {i for i, V in enumerate(verts) if _certify(V, 3, solver)}
        out[name] = {
            "deg4": d4, "deg6": d6,
            "deg4_only": len(d4 - d6), "deg6_only": len(d6 - d4),
            "both": len(d4 & d6), "neither": len(set(range(len(verts))) - (d4 | d6)),
        }
    out["_verts"] = verts
    return out


def test_clarabel_ladder_is_nested(admit_sets):
    """(a) Under the accurate CLARABEL solver the deg4/deg6 ladder is NESTED: no gene is
    degree-4-certified yet degree-6-rejected (deg4_only == 0). The 23/13 SCS "complementarity"
    is retracted as a solver false-negative artifact."""
    c = admit_sets["CLARABEL"]
    assert c["deg4_only"] == 0, (
        f"CLARABEL ladder must be nested (deg4 ⊆ deg6); got deg4_only={c['deg4_only']} — "
        f"the SCS artifact has recurred or the solver pinning broke"
    )
    # deg4 admit set must be a subset of deg6 (full nesting statement).
    assert admit_sets["CLARABEL"]["deg4"] <= admit_sets["CLARABEL"]["deg6"], (
        "every CLARABEL-deg4-certified gene must also be CLARABEL-deg6-certified (nested)"
    )


def test_scs_vs_clarabel_admit_sets_diverge(admit_sets):
    """(b) The solver-swap detector still FIRES: SCS yields a strictly different admit set than
    CLARABEL. On this battery the difference is decisive — SCS fabricates complementarity
    (deg4_only>0 ∧ deg6_only>0) where CLARABEL sees a single nested both-class. If SCS and
    CLARABEL ever agreed here, the keystone demonstration would be silently dead."""
    scs, cla = admit_sets["SCS"], admit_sets["CLARABEL"]
    # The admit sets must differ on at least one gene at some degree.
    diverges = (scs["deg4"] != cla["deg4"]) or (scs["deg6"] != cla["deg6"])
    assert diverges, (
        "SCS and CLARABEL produced identical admit sets — the solver-swap artifact is no longer "
        "reproducible (SCS may have been silently swapped out, or the battery changed)"
    )
    # And specifically: SCS fabricates complementarity (the §0 keystone) while CLARABEL does not.
    assert scs["deg4_only"] > 0 and scs["deg6_only"] > 0, (
        f"SCS must still fabricate complementarity on the battery; got "
        f"deg4_only={scs['deg4_only']} deg6_only={scs['deg6_only']}"
    )
    assert not (cla["deg4_only"] > 0 and cla["deg6_only"] > 0), (
        "CLARABEL must NOT show complementarity (it is nested) — see (a)"
    )


def test_every_clarabel_certified_gene_is_jsr_sound(admit_sets, battery):
    """(c) Soundness oracle: every CLARABEL-certified gene (deg4 ∪ deg6 over the battery) has
    jsr_lb < 1 — a product of its vertex Jacobians never expands, so the lifted certificate is
    SOUND (the artifact was false *negatives*; soundness was never at risk, and this guards it)."""
    verts = admit_sets["_verts"]
    certified = sorted(admit_sets["CLARABEL"]["deg4"] | admit_sets["CLARABEL"]["deg6"])
    assert certified, "expected a non-empty CLARABEL-certified set on the 54-gene battery"
    unsound = []
    max_jsr = 0.0
    for i in certified:
        j = _jsr_lb(verts[i], max_len=4)
        max_jsr = max(max_jsr, j)
        if j >= 1.0 - 1e-9:
            unsound.append((i, j))
    assert not unsound, (
        f"UNSOUND: CLARABEL-certified genes with jsr_lb >= 1 (expansive vertex product): {unsound}"
    )
    # Sanity: the worst certified gene is still strictly contracting with margin.
    assert max_jsr < 1.0, f"max jsr_lb over certified battery = {max_jsr} (must be < 1)"
