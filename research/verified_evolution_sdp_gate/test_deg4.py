# SPDX-License-Identifier: Apache-2.0
"""TDD for the degree-4 (non-quadratic) Lyapunov VerifierBackend (verifier_deg4.py).

Run: py -3.11 -m pytest test_deg4.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from coupled_components import _inf_certifies, _sdp_certifies, _two_certifies, empirical_spectral_radius
from coupled_map import CoupledGene
from verifier_deg4 import cert_deg4_n2, make_deg4_verifier_n2, sym2_power


@pytest.mark.parametrize("n", [2, 3, 4])
def test_sym2_power_correct(n):
    rng = np.random.default_rng(n)
    basis = [(i, j) for i in range(n) for j in range(i, n)]
    for _ in range(15):
        A = rng.standard_normal((n, n))
        z = rng.standard_normal(n)
        m = np.array([z[i] * z[j] for i, j in basis])
        Az = A @ z
        m2 = np.array([Az[i] * Az[j] for i, j in basis])
        assert np.max(np.abs(sym2_power(A) @ m - m2)) < 1e-10


def test_deg4_rejects_expansive_and_admits_safe():
    safe = CoupledGene.make(decay=[0.9, 0.9], W=[[0.1, 0.05], [0.05, 0.1]])
    expansive = CoupledGene.make(decay=[0.1, 0.1], W=[[1.5, 0.9], [0.9, 1.5]])
    assert cert_deg4_n2(safe) is True
    assert cert_deg4_n2(expansive) is False


def test_deg4_sound_on_random_sample():
    """Every gene deg4 certifies must be empirically contracting (consistency)."""
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(120):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if cert_deg4_n2(g):
            checked += 1
            assert empirical_spectral_radius(g, n_samples=4000) < 1.0
    assert checked > 0  # the sample contained at least some certified genes


def test_deg4_recovers_a_d4_residual_gene():
    """deg4 certifies at least one gene the inf/2-norm/quadratic-SDP class rejects."""
    rng = np.random.default_rng(7)
    found = False
    for _ in range(400):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        quad = _inf_certifies(g) or _two_certifies(g) or _sdp_certifies(g)
        if not quad and cert_deg4_n2(g):
            assert empirical_spectral_radius(g, n_samples=4000) < 1.0  # genuinely contracting
            found = True
            break
    assert found, "deg4 recovered no D4-residual gene in the sample"


def test_combined_backend_superset_of_sdp():
    """The sdp_deg4 backend admits everything sdp admits (it is sdp OR deg4)."""
    v = make_deg4_verifier_n2()
    rng = np.random.default_rng(3)
    for _ in range(60):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if _sdp_certifies(g):
            assert v.certifies(g) is True
