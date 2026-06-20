# SPDX-License-Identifier: Apache-2.0
"""A small, deterministic GA over per-layer linearization masks (the evolutionary substrate
applied to the analyzed model structure).

A genome is one bit per layer: ``True`` = linearize that layer to constant-state attention,
``False`` = keep softmax. The fitness (injected) trades memory saved against quality retained — the
same footprint-as-fitness + quality-gate philosophy as ``llcore.fitness.memory_objective``, now
searching the architecture of a *real pretrained model*. Evolution (not greedy ranking) matters
because layers interact: the cumulative cost of linearizing a set is super-linear in the set size,
so the best subset is not simply the k most individually-tolerant layers.

The GA core here is pure and fitness-agnostic (so it is fast to test); the expensive real-model
fitness lives in ``scripts/evolve_linearization.py``.
"""
from __future__ import annotations

import random
from collections.abc import Callable

Genome = tuple[bool, ...]
FitnessFn = Callable[[Genome], float]


def mutate(genome: Genome, rate: float, rng: random.Random) -> Genome:
    """Independently flip each bit with probability ``rate``."""
    return tuple((not b) if rng.random() < rate else b for b in genome)


def crossover(a: Genome, b: Genome, rng: random.Random) -> Genome:
    """Uniform crossover: each gene comes from parent ``a`` or ``b`` with equal probability."""
    return tuple(a[i] if rng.random() < 0.5 else b[i] for i in range(len(a)))


def _tournament(pop: list[Genome], fits: list[float], k: int, rng: random.Random) -> Genome:
    idxs = [rng.randrange(len(pop)) for _ in range(k)]
    return pop[max(idxs, key=lambda i: fits[i])]


def evolve(
    fitness_fn: FitnessFn,
    n_genes: int,
    *,
    pop_size: int = 16,
    generations: int = 12,
    mutation_rate: float = 0.1,
    elitism: int = 2,
    tournament_k: int = 3,
    seed: int = 0,
) -> dict[str, object]:
    """Evolve a per-layer linearization mask maximizing ``fitness_fn``.

    Returns ``{best_genome, best_fitness, history}`` where ``history[g]`` is the best fitness
    found through generation ``g`` (monotonic non-decreasing).
    """
    rng = random.Random(seed)
    pop: list[Genome] = [tuple(rng.random() < 0.5 for _ in range(n_genes)) for _ in range(pop_size)]
    fits = [fitness_fn(g) for g in pop]
    best_i = max(range(pop_size), key=lambda i: fits[i])
    best_g, best_f = pop[best_i], fits[best_i]
    history: list[float] = []
    for _ in range(generations):
        order = sorted(range(pop_size), key=lambda i: fits[i], reverse=True)
        new_pop: list[Genome] = [pop[order[i]] for i in range(min(elitism, pop_size))]
        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fits, tournament_k, rng)
            p2 = _tournament(pop, fits, tournament_k, rng)
            new_pop.append(mutate(crossover(p1, p2, rng), mutation_rate, rng))
        pop = new_pop
        fits = [fitness_fn(g) for g in pop]
        gi = max(range(pop_size), key=lambda i: fits[i])
        if fits[gi] > best_f:
            best_f, best_g = fits[gi], pop[gi]
        history.append(best_f)
    return {"best_genome": best_g, "best_fitness": best_f, "history": history}
