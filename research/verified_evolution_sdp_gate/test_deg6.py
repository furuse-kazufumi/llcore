# SPDX-License-Identifier: Apache-2.0
"""TDD for the degree-6 (non-quadratic) Lyapunov VerifierBackend (verifier_deg6.py).

NOTE (post adversarial review): the certifiers are pinned to the accurate CLARABEL solver.
cvxpy's SCS default produced FALSE NEGATIVES near the feasibility boundary, which fabricated an
apparent deg4/deg6 "complementarity". Under CLARABEL the lifted ladder is NESTED (deg4 ⊆ deg6),
the quadratic SDP already certifies ~95% of contracting genes, and the D4 residual is tiny. These
tests assert the SOUND, NESTED behaviour with a deterministic residual-gene fixture.

Run: py -3.11 -m pytest test_deg6.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from coupled_components import _inf_certifies, _sdp_certifies, _two_certifies, empirical_spectral_radius
from coupled_map import CoupledGene
from verifier_deg4 import cert_deg4_n2, sym2_power
from verifier_deg6 import cert_deg6_n2, make_deg6_verifier_n2, mono_basis, sym_power


def _quad(g) -> bool:
    """Any sound quadratic-class certificate (inf / 2-norm / common-quadratic SDP)."""
    return _inf_certifies(g) or _two_certifies(g) or _sdp_certifies(g)


# A deterministic D4-RESIDUAL gene (from the CLARABEL EXP-A residual pool): quadratic-class REJECTED
# but degree-4/6 certified, empirically contracting (rho≈0.88). Used as a stable fixture instead of
# a flaky random search (the residual is rare under an accurate solver).
_RESIDUAL_GENE = CoupledGene.make(
    decay=[0.585387227007845, 0.42030016089787947],
    W=[[-0.3862125356196242, 1.7757712667750045],
       [-1.807150485416289, -0.6957048379417996]])


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("degree", [2, 3])
def test_sym_power_correct(n, degree):
    rng = np.random.default_rng(10 * n + degree)
    basis = mono_basis(n, degree)
    for _ in range(15):
        A = rng.standard_normal((n, n))
        z = rng.standard_normal(n)
        m = np.array([np.prod([z[i] for i in b]) for b in basis])
        Az = A @ z
        m_true = np.array([np.prod([Az[i] for i in b]) for b in basis])
        assert np.max(np.abs(sym_power(A, degree) @ m - m_true)) < 1e-9


def test_sym_power_degree2_matches_deg4_sym2():
    rng = np.random.default_rng(0)
    for _ in range(20):
        A = rng.standard_normal((2, 2))
        assert np.max(np.abs(sym_power(A, 2) - sym2_power(A))) < 1e-12


def test_deg6_rejects_expansive_and_admits_safe():
    safe = CoupledGene.make(decay=[0.9, 0.9], W=[[0.1, 0.05], [0.05, 0.1]])
    expansive = CoupledGene.make(decay=[0.1, 0.1], W=[[1.5, 0.9], [0.9, 1.5]])
    assert cert_deg6_n2(safe) is True
    assert cert_deg6_n2(expansive) is False


def test_deg6_sound_on_random_sample():
    """Every gene deg6 certifies must be empirically contracting (consistency oracle)."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(120):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if cert_deg6_n2(g):
            checked += 1
            assert empirical_spectral_radius(g, n_samples=4000) < 1.0
    assert checked > 0


def test_deg6_recovers_the_residual_fixture():
    """The fixture is a genuine D4-residual gene: quad-REJECTED, deg6-certified, contracting."""
    assert _quad(_RESIDUAL_GENE) is False            # no quadratic-class certificate
    assert cert_deg6_n2(_RESIDUAL_GENE) is True       # but the degree-6 Lyapunov certifies it
    assert empirical_spectral_radius(_RESIDUAL_GENE, n_samples=6000) < 1.0  # genuinely contracting


def test_ladder_nested_deg4_subset_deg6():
    """Under the accurate solver the lifted ladder is NESTED: a deg4-certified gene is deg6-certified
    (the prior 'complementarity' was an SCS false-negative artifact, now retracted)."""
    assert cert_deg4_n2(_RESIDUAL_GENE) is True
    assert cert_deg6_n2(_RESIDUAL_GENE) is True
    # spot-check nesting on a random sample: no gene is deg4-certified yet deg6-rejected.
    rng = np.random.default_rng(20260603)
    for _ in range(150):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if cert_deg4_n2(g):
            assert cert_deg6_n2(g) is True, "deg4-certified gene must be deg6-certified (nested)"


def test_union_backend_superset_of_sdp():
    """The sdp_deg4_deg6 union backend admits everything sdp admits."""
    v = make_deg6_verifier_n2()
    rng = np.random.default_rng(3)
    for _ in range(60):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if _sdp_certifies(g):
            assert v.certifies(g) is True
