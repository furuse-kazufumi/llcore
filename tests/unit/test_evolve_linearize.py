# SPDX-License-Identifier: Apache-2.0
"""Tests for the evolutionary search over per-layer linearization masks.

This applies the preserved evolutionary substrate to the *analyzed structure*: a model's layers
become a genome (per layer: keep softmax | linearize to constant-state attention), and a multi-
objective fitness (memory saved vs quality retained) is optimized by a GA. Because layers interact
(cumulative linearization cost is super-linear), evolution can find a better subset than greedily
linearizing the most-tolerant layers. The GA core is a pure, injected-fitness function so it is
deterministic and testable without the heavy model in the loop.
"""
from __future__ import annotations


def test_mutate_flips_bits_deterministically() -> None:
    from llcore.runtime.evolve_linearize import mutate
    import random

    g = (False,) * 20
    m1 = mutate(g, 0.5, random.Random(1))
    m2 = mutate(g, 0.5, random.Random(1))
    assert m1 == m2  # same seed -> same result
    assert len(m1) == 20
    assert sum(m1) > 0  # at rate 0.5, some bits flipped


def test_crossover_takes_from_both_parents() -> None:
    from llcore.runtime.evolve_linearize import crossover
    import random

    a = (True,) * 16
    b = (False,) * 16
    child = crossover(a, b, random.Random(0))
    assert len(child) == 16
    assert any(child) and not all(child)  # mixes both


def test_evolve_maximizes_simple_fitness() -> None:
    from llcore.runtime.evolve_linearize import evolve

    # fitness = number of linearized layers -> optimum is all True
    res = evolve(lambda g: float(sum(g)), n_genes=12, pop_size=16, generations=20, seed=0)
    assert res["best_fitness"] == 12.0
    assert all(res["best_genome"])


def test_evolve_respects_quality_constraint() -> None:
    from llcore.runtime.evolve_linearize import evolve

    # maximize linearized layers but a hard penalty above 3 -> optimum linearizes exactly 3
    def fit(g: tuple[bool, ...]) -> float:
        n = sum(g)
        return float(n) - 100.0 * max(0, n - 3)

    res = evolve(fit, n_genes=10, pop_size=20, generations=30, seed=1)
    assert sum(res["best_genome"]) == 3


def test_evolve_history_tracks_improvement() -> None:
    from llcore.runtime.evolve_linearize import evolve

    res = evolve(lambda g: float(sum(g)), n_genes=8, pop_size=12, generations=15, seed=2)
    hist = res["history"]
    assert len(hist) == 15
    assert hist[-1] >= hist[0]  # best-so-far is monotonic non-decreasing
