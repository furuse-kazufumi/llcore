# SPDX-License-Identifier: Apache-2.0
"""TDD for the degree-6 (non-quadratic) Lyapunov VerifierBackend (verifier_deg6.py).

Covers: symmetric-3rd-power correctness, soundness (cert ⇒ empirically contracting), residual
recovery (deg6 certifies genes the quadratic class rejects), the NON-OBVIOUS complementarity
(deg6∖deg4 and deg4∖deg6 both non-empty ⇒ the lifted ladder is non-nested), and that the union
backend is a superset of sdp.

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
    """The general sym_power(.,2) reproduces verifier_deg4.sym2_power exactly."""
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


def test_deg6_recovers_a_residual_gene():
    """deg6 certifies at least one genuinely-contracting gene the quadratic class rejects."""
    rng = np.random.default_rng(20260603)
    found = False
    for _ in range(400):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if not _quad(g) and cert_deg6_n2(g):
            assert empirical_spectral_radius(g, n_samples=6000) < 1.0
            found = True
            break
    assert found, "deg6 recovered no residual gene in the sample"


def test_complementarity_deg6_not_deg4():
    """NON-OBVIOUS: a gene deg6 certifies but deg4 does NOT (lifted ladder is non-nested)."""
    rng = np.random.default_rng(20260603)
    found = False
    for _ in range(700):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if _quad(g):
            continue
        if cert_deg6_n2(g) and not cert_deg4_n2(g):
            assert empirical_spectral_radius(g, n_samples=6000) < 1.0
            found = True
            break
    assert found, "no deg6∖deg4 gene found (expected the ladder to be non-nested)"


def test_complementarity_deg4_not_deg6():
    """NON-OBVIOUS (the surprising direction): a gene deg4 certifies but deg6 does NOT."""
    rng = np.random.default_rng(20260603)
    found = False
    for _ in range(700):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if _quad(g):
            continue
        if cert_deg4_n2(g) and not cert_deg6_n2(g):
            assert empirical_spectral_radius(g, n_samples=6000) < 1.0
            found = True
            break
    assert found, "no deg4∖deg6 gene found (expected the ladder to be non-nested)"


def test_union_backend_superset_of_sdp():
    """The sdp_deg4_deg6 union backend admits everything sdp admits."""
    v = make_deg6_verifier_n2()
    rng = np.random.default_rng(3)
    for _ in range(60):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if _sdp_certifies(g):
            assert v.certifies(g) is True
