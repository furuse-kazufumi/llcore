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


# --- categorical GA (Level-2 NAS: per-layer mixer choice among >2 options) ---


def test_mutate_categorical_valid_and_deterministic() -> None:
    from llcore.runtime.evolve_linearize import mutate_categorical
    import random

    g = (0,) * 12
    m1 = mutate_categorical(g, 3, 0.5, random.Random(1))
    m2 = mutate_categorical(g, 3, 0.5, random.Random(1))
    assert m1 == m2
    assert len(m1) == 12 and all(0 <= x < 3 for x in m1)
    assert any(x != 0 for x in m1)  # at rate 0.5 some genes changed


def test_evolve_categorical_maximizes() -> None:
    from llcore.runtime.evolve_linearize import evolve_categorical

    # fitness = number of genes set to option 2 -> optimum is all 2s
    res = evolve_categorical(
        lambda g: float(sum(1 for x in g if x == 2)),
        n_genes=10, n_options=3, pop_size=20, generations=25, seed=0,
    )
    assert res["best_fitness"] == 10.0
    assert all(x == 2 for x in res["best_genome"])


def test_evolve_categorical_seeding_never_worse_than_seed() -> None:
    from llcore.runtime.evolve_linearize import evolve_categorical

    # a deceptive fitness where a good seed should never be lost (elitism + seeding)
    good = (2, 2, 2, 2, 2, 0, 0, 0)  # fitness 5
    fit = lambda g: float(sum(1 for x in g if x == 2))  # noqa: E731
    res = evolve_categorical(fit, n_genes=8, n_options=3, pop_size=10, generations=5, seed=0, seed_genomes=[good])
    assert res["best_fitness"] >= fit(good)  # memetic: never regress below the seed


# --- multi-objective Pareto NAS (memory saved vs quality loss frontier) ---


def test_dominates_maximization() -> None:
    from llcore.runtime.evolve_linearize import dominates

    assert dominates((2.0, 2.0), (1.0, 1.0))      # strictly better on both
    assert dominates((2.0, 1.0), (1.0, 1.0))      # better on one, equal on the other
    assert not dominates((2.0, 0.0), (1.0, 1.0))  # better on one, worse on the other
    assert not dominates((1.0, 1.0), (1.0, 1.0))  # equal -> no domination


def test_pareto_front_keeps_only_nondominated() -> None:
    from llcore.runtime.evolve_linearize import pareto_front

    items = [("a", (3.0, 1.0)), ("b", (1.0, 3.0)), ("c", (2.0, 2.0)), ("d", (1.0, 1.0))]
    keys = {g for g, _ in pareto_front(items)}
    assert keys == {"a", "b", "c"}  # d is dominated by all three; a/b/c mutually non-dominated


def test_non_dominated_sort_layers() -> None:
    from llcore.runtime.evolve_linearize import non_dominated_sort

    objs = [(3.0, 1.0), (1.0, 3.0), (2.0, 2.0), (1.0, 1.0)]  # 0,1,2 = front0 ; 3 = front1
    fronts = non_dominated_sort(objs)
    assert set(fronts[0]) == {0, 1, 2}
    assert fronts[1] == [3]


def test_crowding_distance_boundary_infinite() -> None:
    from llcore.runtime.evolve_linearize import crowding_distance

    objs = [(0.0, 3.0), (1.0, 2.0), (3.0, 0.0)]
    cd = crowding_distance(objs, [0, 1, 2])
    assert cd[0] == float("inf") and cd[2] == float("inf")  # extremes are boundary points
    assert cd[1] < float("inf")  # interior point is finite


def test_evolve_multiobjective_recovers_tradeoff_front() -> None:
    from llcore.runtime.evolve_linearize import dominates, evolve_multiobjective

    # obj1 = #genes set to 2, obj2 = #genes set to 0. A gene can't be both, so the two
    # objectives trade off; the true Pareto front is all configs with no "1" genes
    # (obj1 + obj2 == n_genes).
    n = 6

    def ev(g: tuple[int, ...]) -> tuple[float, ...]:
        return (float(sum(1 for x in g if x == 2)), float(sum(1 for x in g if x == 0)))

    res = evolve_multiobjective(ev, n_genes=n, n_options=3, pop_size=24, generations=30, seed=0)
    front = res["front"]
    assert isinstance(front, list) and front  # non-empty
    objs = [o for _, o in front]
    # internal non-domination: no front point dominates another
    for i, oi in enumerate(objs):
        assert not any(dominates(oj, oi) for j, oj in enumerate(objs) if j != i)
    # every front point is Pareto-optimal (no wasted "1" genes)
    assert all(abs(o[0] + o[1] - n) < 1e-9 for o in objs)
    # good spread along the frontier (recovers most of the n+1 ideal points)
    assert len({(o[0], o[1]) for o in objs}) >= n - 1


def test_evolve_multiobjective_deterministic() -> None:
    from llcore.runtime.evolve_linearize import evolve_multiobjective

    def ev(g: tuple[int, ...]) -> tuple[float, ...]:
        return (float(sum(1 for x in g if x == 2)), float(sum(1 for x in g if x == 0)))

    a = evolve_multiobjective(ev, n_genes=6, n_options=3, pop_size=16, generations=10, seed=3)
    b = evolve_multiobjective(ev, n_genes=6, n_options=3, pop_size=16, generations=10, seed=3)
    fa, fb = a["front"], b["front"]
    assert isinstance(fa, list) and isinstance(fb, list)
    assert {(g, o) for g, o in fa} == {(g, o) for g, o in fb}


def test_evolve_multiobjective_seeding_preserved() -> None:
    from llcore.runtime.evolve_linearize import evolve_multiobjective

    n = 6
    seed_genome = (2,) * n  # objective (6, 0) — a non-dominated extreme of the frontier

    def ev(g: tuple[int, ...]) -> tuple[float, ...]:
        return (float(sum(1 for x in g if x == 2)), float(sum(1 for x in g if x == 0)))

    res = evolve_multiobjective(
        ev, n_genes=n, n_options=3, pop_size=12, generations=8, seed=0, seed_genomes=[seed_genome]
    )
    objs = [o for _, o in res["front"]]  # type: ignore[union-attr]
    # the seeded extreme (6, 0) must survive on the final Pareto front
    assert any(abs(o[0] - 6.0) < 1e-9 and abs(o[1] - 0.0) < 1e-9 for o in objs)
