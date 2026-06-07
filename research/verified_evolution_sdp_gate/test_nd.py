# SPDX-License-Identifier: Apache-2.0
"""TDD for the n-dim coupled GeneCodec extension (coupled_nd.py).

Run: py -3.11 -m pytest test_nd.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from coupled_nd import (
    CoupledNDGene,
    CoupledNDGeneCodec,
    RotationNDObjective,
    cert_inf,
    cert_sdp,
    cert_two,
    classify_region,
    empirical_rho,
    make_nd_verifier,
)
from evolvable_core import EvolveConfig, evolve


def test_codec_dim_and_roundtrip():
    for n in (2, 3, 4):
        c = CoupledNDGeneCodec(n)
        assert c.dim == n + n * n
        rng = np.random.default_rng(0)
        g = c.clip(c.random(rng))
        gene = c.to_gene(g)
        assert gene.n == n
        assert gene.decay.shape == (n,) and gene.W.shape == (n, n)


def test_n2_regression_matches_existing_certifiers():
    """n-dim certifiers must agree with the validated n=2 Track-C/D certifiers."""
    from coupled_components import _inf_certifies, _two_certifies
    from coupled_map import CoupledGene
    rng = np.random.default_rng(3)
    dis = 0
    for _ in range(300):
        decay = rng.uniform(0, 1, 2)
        W = rng.uniform(-2, 2, (2, 2))
        g2 = CoupledGene.make(decay=decay, W=W)
        gn = CoupledNDGene.make(decay=decay, W=W)
        if bool(_inf_certifies(g2)) != bool(cert_inf(gn)):
            dis += 1
        if bool(_two_certifies(g2)) != bool(cert_two(gn)):
            dis += 1
    assert dis == 0


@pytest.mark.parametrize("n", [3, 4])
def test_known_gene_certifier_decisions(n):
    safe = CoupledNDGene.make(decay=np.full(n, 0.9), W=0.05 * np.ones((n, n)))
    expansive = CoupledNDGene.make(decay=np.full(n, 0.1), W=1.5 * np.eye(n) + 0.5)
    for cert in (cert_inf, cert_two, cert_sdp):
        assert cert(safe) is True
        assert cert(expansive) is False


def test_empirical_oracle_separates():
    n = 3
    safe = CoupledNDGene.make(decay=np.full(n, 0.9), W=0.05 * np.ones((n, n)))
    expansive = CoupledNDGene.make(decay=np.full(n, 0.1), W=2.0 * np.eye(n))
    assert empirical_rho(safe, n_samples=3000) < 1.0
    assert empirical_rho(expansive, n_samples=3000) > 1.0


@pytest.mark.parametrize("n", [3])
def test_evolution_climbs_nd(n):
    codec = CoupledNDGeneCodec(n)
    r = evolve(codec, RotationNDObjective(n), make_nd_verifier("sdp"),
               EvolveConfig(pop_size=30, n_generations=30, resample_cap=15),
               rng=np.random.default_rng(5))
    assert r.best_fitness_curve[-1] > r.best_fitness_curve[0] + 0.1


@pytest.mark.parametrize("gate", ["inf_norm", "two_norm", "sdp"])
def test_nd_gate_admits_only_contractions(gate):
    n = 3
    codec = CoupledNDGeneCodec(n)
    r = evolve(codec, RotationNDObjective(n), make_nd_verifier(gate),
               EvolveConfig(pop_size=20, n_generations=12, resample_cap=12),
               rng=np.random.default_rng(11), gate_initial=True)
    for gt in r.final_population:
        assert empirical_rho(codec.to_gene(gt), n_samples=2500) < 1.0 + 1e-6
