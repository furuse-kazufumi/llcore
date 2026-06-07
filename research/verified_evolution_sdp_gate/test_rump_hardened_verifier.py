# SPDX-License-Identifier: Apache-2.0
"""Tests for the ADDITIVE Rump+OR hardened verifier factory (PART 2).

Invariants under test:
  (a) the factory admits the solver-independent fast-path genes (inf/2-norm) with no solver;
  (b) it admits an SDP-only gene (the non-normal reference) when CLARABEL is present;
  (c) ADMIT-SET PRESERVED-OR-GROWN: on a small contracting battery, every gene the EXISTING
      float-recheck default certifier (`coupled_components._sdp_certifies`) admits is ALSO admitted
      by the Rump+OR factory (Rump+OR ⊇ float — never shrinks);
  (d) SOUNDNESS: any gene the Rump+OR factory admits has empirical ρ < 1 (no false positive);
  (e) the recheck matches the FLOAT PATH's no-margin matrices (margin=0 delegation), so a P that
      the float `eigvalsh(>0)` test on `P` and `P - JᵀPJ` accepts is also Rump-accepted when
      comfortably PD.

Self-contained; reuses validated rump_pd; does NOT touch src/ or change any default certifier.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.normpath(os.path.join(_HERE, ".."))
for _d in (_HERE,
           os.path.join(_RESEARCH, "coupled_z3_contraction"),
           os.path.join(_RESEARCH, "spectral_lyapunov_contraction")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

pytest.importorskip("cvxpy")
import cvxpy as cp  # noqa: E402
from coupled_map import CoupledGene  # noqa: E402
from coupled_components import _sdp_certifies, empirical_spectral_radius  # noqa: E402
from rump_hardened_verifier import (  # noqa: E402
    make_sdp_verifier_rump_or, rump_recheck_matches_float_path,
)
from rump_pd import _vertex_jacobians  # noqa: E402

_CLARABEL = "CLARABEL" in cp.installed_solvers()

BENIGN = CoupledGene.make(decay=[0.8, 0.8], W=[[0.0, 0.0], [0.0, 0.0]])          # inf/2-norm
ROTATION = CoupledGene.make(decay=[0.3, 0.3], W=[[0.5, 0.9], [-0.9, 0.5]])       # 2-norm region
NONNORMAL = CoupledGene.make(decay=[0.5, 0.5], W=[[0.0, 0.0], [1.6, 0.0]])       # sdp_only


def test_a_fast_path_genes_admitted():
    v = make_sdp_verifier_rump_or()
    assert v.certifies(BENIGN) is True


@pytest.mark.skipif(not _CLARABEL, reason="requires CLARABEL")
def test_b_sdp_only_gene_admitted_with_clarabel():
    v = make_sdp_verifier_rump_or()
    # genuine SDP solve required; Rump+OR admits it.
    assert v.certifies(NONNORMAL) is True


@pytest.mark.skipif(not _CLARABEL, reason="requires CLARABEL")
def test_c_admit_set_preserved_or_grown_and_sound():
    """On a small contracting battery: Rump+OR ⊇ float (never shrinks) and every Rump+OR admit is
    empirically contracting (sound)."""
    v = make_sdp_verifier_rump_or()
    rng = np.random.default_rng(2024)
    n_checked = 0
    lost = []
    unsound = []
    while n_checked < 40:
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if empirical_spectral_radius(g, n_samples=2000) >= 1.0:
            continue
        n_checked += 1
        f = bool(_sdp_certifies(g))
        r = bool(v.certifies(g))
        if f and not r:
            lost.append((g.clipped().decay.tolist(), g.clipped().W.reshape(-1).tolist()))
        if r:
            # soundness: an admitted gene must be empirically contracting at a higher sample count.
            if empirical_spectral_radius(g, n_samples=20000) >= 1.0:
                unsound.append((g.clipped().decay.tolist(), g.clipped().W.reshape(-1).tolist()))
    assert lost == [], f"Rump+OR shrank the admit set (lost float-admitted genes): {lost}"
    assert unsound == [], f"Rump+OR admitted a non-contracting gene (false positive): {unsound}"
    assert n_checked == 40


def test_e_recheck_matches_float_no_margin():
    """`rump_recheck_matches_float_path` verifies the no-margin matrices P and P - JᵀPJ. A
    comfortably-PD common certificate must pass; a fabricated indefinite P must be rejected."""
    verts = _vertex_jacobians(BENIGN, t_domain="tmin1")
    # P = I is a valid common-Lyapunov for the benign gene (it is a contraction in the 2-norm).
    assert rump_recheck_matches_float_path(np.eye(2), verts) is True
    # an indefinite P must be rejected (soundness; no false positive).
    bad_P = np.diag([1.0, -1.0])
    assert rump_recheck_matches_float_path(bad_P, verts) is False


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
