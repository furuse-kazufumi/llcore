# SPDX-License-Identifier: Apache-2.0
"""TDD for the extra evolution-direction Objectives (objectives_extra.py).

Run: py -3.11 -m pytest test_objectives_extra.py -q
"""
from __future__ import annotations

import numpy as np
import pytest

from coupled_components import CoupledGeneCodec, empirical_spectral_radius, make_verifier
from evolvable_core import EvolveConfig, evolve
from objectives_extra import ALL_EXTRA

CODEC = CoupledGeneCodec()


@pytest.mark.parametrize("cls", ALL_EXTRA)
def test_objective_is_deterministic(cls):
    obj = cls()
    rng = np.random.default_rng(0)
    g = CODEC.to_gene(CODEC.clip(CODEC.random(rng)))
    assert obj.fitness(g) == obj.fitness(g)  # no per-eval RNG
    assert obj._target().shape[1] == 2


@pytest.mark.parametrize("cls", ALL_EXTRA)
def test_skeleton_evolves_new_direction(cls):
    """The unchanged evolve() climbs each new direction under a sound (2-norm) gate."""
    r = evolve(CODEC, cls(), make_verifier("two_norm"),
               EvolveConfig(pop_size=24, n_generations=25, resample_cap=15),
               rng=np.random.default_rng(5), gate_initial=True)
    start, final = r.best_fitness_curve[0], r.best_fitness_curve[-1]
    # the skeleton evolves the direction: it reaches a good fitness OR climbs meaningfully
    # (easy directions like dual_rate_decay start already high; elitism keeps the curve monotone)
    assert final >= start and (final > 0.5 or final - start > 0.03)
    # verified: gate_initial keeps the whole population contracting
    for g in r.final_population:
        assert empirical_spectral_radius(CODEC.to_gene(g), n_samples=2500) < 1.0 + 1e-6
