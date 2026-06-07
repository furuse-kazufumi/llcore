# SPDX-License-Identifier: Apache-2.0
"""TDD tests for the Verified Evolution skeleton (research/, src untouched).

Run: py -3.11 -m pytest research/verified_evolution_sdp_gate/test_skeleton.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from coupled_components import (
    BenignDecayObjective,
    CoupledGeneCodec,
    NonNormalObjective,
    RotationObjective,
    _NONNORMAL_REF,
    classify_region,
    empirical_spectral_radius,
    make_verifier,
)
from coupled_map import CoupledGene
from evolvable_core import EvolveConfig, evolve

CODEC = CoupledGeneCodec()
SMALL = EvolveConfig(pop_size=12, n_generations=10, resample_cap=20)


# --------------------------------------------------------------------------- #
# codec
# --------------------------------------------------------------------------- #


def test_codec_random_within_box_and_dim():
    rng = np.random.default_rng(0)
    for _ in range(200):
        g = CODEC.random(rng)
        assert g.shape == (6,)
        c = CODEC.clip(g)
        assert np.all(c[:2] >= 0) and np.all(c[:2] <= 1)
        assert np.all(c[2:] >= -2) and np.all(c[2:] <= 2)


def test_codec_operators_preserve_dim():
    rng = np.random.default_rng(1)
    a, b = CODEC.random(rng), CODEC.random(rng)
    assert CODEC.crossover(a, b, rng).shape == (6,)
    assert CODEC.mutate(a, 0.15, rng).shape == (6,)
    gene = CODEC.to_gene(a)
    assert isinstance(gene, CoupledGene)


# --------------------------------------------------------------------------- #
# determinism + none-gate control (G0)
# --------------------------------------------------------------------------- #


def test_determinism_same_seed_identical():
    obj, ver = RotationObjective(), make_verifier("none")
    r1 = evolve(CODEC, obj, ver, SMALL, rng=np.random.default_rng(42))
    r2 = evolve(CODEC, obj, ver, SMALL, rng=np.random.default_rng(42))
    assert r1.best_fitness_curve == r2.best_fitness_curve
    assert r1.diversity_curve == r2.diversity_curve
    assert np.allclose(r1.best_genotype, r2.best_genotype)


def test_none_gate_admits_all_no_rejection():
    r = evolve(CODEC, RotationObjective(), make_verifier("none"), SMALL,
               rng=np.random.default_rng(7))
    assert r.n_rejections == 0
    assert r.fallback_count == 0


# --------------------------------------------------------------------------- #
# EVOLUTION ACTUALLY WORKS (the realization)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("objname,objcls", [("rotation", RotationObjective),
                                            ("benign", BenignDecayObjective)])
def test_evolution_improves_fitness(objname, objcls):
    """best fitness at the end strictly exceeds the start (evolution climbs)."""
    r = evolve(CODEC, objcls(), make_verifier("sdp"),
               EvolveConfig(pop_size=20, n_generations=25, resample_cap=30),
               rng=np.random.default_rng(3))
    assert r.best_fitness_curve[-1] > r.best_fitness_curve[0] + 1e-3, \
        f"{objname}: no improvement {r.best_fitness_curve[0]} -> {r.best_fitness_curve[-1]}"


# --------------------------------------------------------------------------- #
# gate soundness on known genes (G1 unit-level)
# --------------------------------------------------------------------------- #


def test_known_gene_gate_decisions():
    safe = CoupledGene.make(decay=[0.9, 0.9], W=[[0.1, 0.05], [0.05, 0.1]])
    expansive = CoupledGene.make(decay=[0.3, 0.3], W=[[0.5, 0.9], [0.9, 0.5]])
    for name in ("inf_norm", "two_norm", "sdp"):
        v = make_verifier(name)
        assert v.certifies(safe) is True, f"{name} should admit safe gene"
        assert v.certifies(expansive) is False, f"{name} should reject expansive gene"


def test_reference_nonnormal_is_sdp_only():
    assert classify_region(_NONNORMAL_REF) == "sdp_only"
    assert make_verifier("inf_norm").certifies(_NONNORMAL_REF) is False
    assert make_verifier("two_norm").certifies(_NONNORMAL_REF) is False
    assert make_verifier("sdp").certifies(_NONNORMAL_REF) is True


def test_empirical_oracle_separates_contracting():
    safe = CoupledGene.make(decay=[0.9, 0.9], W=[[0.1, 0.05], [0.05, 0.1]])
    expansive = CoupledGene.make(decay=[0.1, 0.1], W=[[1.5, 0.9], [0.9, 1.5]])
    assert empirical_spectral_radius(safe, n_samples=3000) < 1.0
    assert empirical_spectral_radius(expansive, n_samples=3000) > 1.0


# --------------------------------------------------------------------------- #
# verified evolution never admits a divergent gene (G1 loop-level, small scale)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("gate", ["inf_norm", "two_norm", "sdp"])
def test_verified_evolution_admits_only_contractions(gate):
    r = evolve(CODEC, RotationObjective(), make_verifier(gate), SMALL,
               rng=np.random.default_rng(11))
    # spot-check the final population (cheaper than every admitted child)
    for g in r.final_population:
        gene = CODEC.to_gene(g)
        assert empirical_spectral_radius(gene, n_samples=2000) < 1.0 + 1e-6, \
            f"{gate} admitted a non-contracting gene into final pop"
