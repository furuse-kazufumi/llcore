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
from typing import TypeVar

Genome = tuple[bool, ...]
CatGenome = tuple[int, ...]
FitnessFn = Callable[[Genome], float]
CatFitnessFn = Callable[[CatGenome], float]

_G = TypeVar("_G", bound=tuple[object, ...])


def mutate(genome: Genome, rate: float, rng: random.Random) -> Genome:
    """Independently flip each bit with probability ``rate``."""
    return tuple((not b) if rng.random() < rate else b for b in genome)


def mutate_categorical(genome: CatGenome, n_options: int, rate: float, rng: random.Random) -> CatGenome:
    """Independently re-roll each gene to a uniform option in ``[0, n_options)`` with prob ``rate``."""
    return tuple(rng.randrange(n_options) if rng.random() < rate else g for g in genome)


def crossover(a: _G, b: _G, rng: random.Random) -> _G:
    """Uniform crossover: each gene comes from parent ``a`` or ``b`` with equal probability."""
    return tuple(a[i] if rng.random() < 0.5 else b[i] for i in range(len(a)))  # type: ignore[return-value]


def _tournament(pop: list[_G], fits: list[float], k: int, rng: random.Random) -> _G:
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


def evolve_categorical(
    fitness_fn: CatFitnessFn,
    n_genes: int,
    n_options: int,
    *,
    pop_size: int = 16,
    generations: int = 12,
    mutation_rate: float = 0.1,
    elitism: int = 2,
    tournament_k: int = 3,
    seed: int = 0,
    seed_genomes: list[CatGenome] | None = None,
) -> dict[str, object]:
    """Evolve a per-gene categorical assignment (e.g. per-layer mixer choice) maximizing fitness.

    Genes take integer values in ``[0, n_options)``. ``seed_genomes`` injects known solutions
    (e.g. a greedy baseline) into the initial population, turning the search memetic: if the GA
    cannot improve on a seeded greedy solution, that is strong evidence the landscape is separable
    and greedy is near-optimal; if it can, evolution adds value.
    """
    rng = random.Random(seed)
    pop: list[CatGenome] = list(seed_genomes or [])[:pop_size]
    while len(pop) < pop_size:
        pop.append(tuple(rng.randrange(n_options) for _ in range(n_genes)))
    fits = [fitness_fn(g) for g in pop]
    best_i = max(range(pop_size), key=lambda i: fits[i])
    best_g, best_f = pop[best_i], fits[best_i]
    history: list[float] = []
    for _ in range(generations):
        order = sorted(range(pop_size), key=lambda i: fits[i], reverse=True)
        new_pop: list[CatGenome] = [pop[order[i]] for i in range(min(elitism, pop_size))]
        while len(new_pop) < pop_size:
            p1 = _tournament(pop, fits, tournament_k, rng)
            p2 = _tournament(pop, fits, tournament_k, rng)
            new_pop.append(mutate_categorical(crossover(p1, p2, rng), n_options, mutation_rate, rng))
        pop = new_pop
        fits = [fitness_fn(g) for g in pop]
        gi = max(range(pop_size), key=lambda i: fits[i])
        if fits[gi] > best_f:
            best_f, best_g = fits[gi], pop[gi]
        history.append(best_f)
    return {"best_genome": best_g, "best_fitness": best_f, "history": history}


# --- multi-objective Pareto search (the memory↔quality frontier, not a single budget) ---

Objectives = tuple[float, ...]
MOFitnessFn = Callable[[CatGenome], Objectives]


def dominates(a: Objectives, b: Objectives) -> bool:
    """Pareto domination for **maximization**: ``a`` dominates ``b`` iff ``a`` is >=
    ``b`` on every objective and strictly > on at least one."""
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def pareto_front(items: list[tuple[_G, Objectives]]) -> list[tuple[_G, Objectives]]:
    """Return the non-dominated subset of ``(genome, objectives)`` pairs (the frontier)."""
    objs = [o for _, o in items]
    return [
        (g, oi)
        for i, (g, oi) in enumerate(items)
        if not any(dominates(objs[j], oi) for j in range(len(objs)) if j != i)
    ]


def non_dominated_sort(objs: list[Objectives]) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al. NSGA-II). Returns fronts as index lists,
    best (rank 0) first. Empty trailing fronts are dropped."""
    n = len(objs)
    dominated: list[list[int]] = [[] for _ in range(n)]  # indices that p dominates
    dom_count = [0] * n  # how many solutions dominate p
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(objs[p], objs[q]):
                dominated[p].append(q)
            elif dominates(objs[q], objs[p]):
                dom_count[p] += 1
        if dom_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: list[int] = []
        for p in fronts[i]:
            for q in dominated[p]:
                dom_count[q] -= 1
                if dom_count[q] == 0:
                    nxt.append(q)
        fronts.append(nxt)
        i += 1
    return [f for f in fronts if f]


def crowding_distance(objs: list[Objectives], front: list[int]) -> dict[int, float]:
    """NSGA-II crowding distance for one front; boundary points get ``inf`` so the
    extremes of the frontier are always preserved."""
    dist: dict[int, float] = {i: 0.0 for i in front}
    if not front:
        return dist
    m = len(objs[front[0]])
    for k in range(m):
        order = sorted(front, key=lambda i: objs[i][k])
        dist[order[0]] = float("inf")
        dist[order[-1]] = float("inf")
        span = objs[order[-1]][k] - objs[order[0]][k]
        if span <= 0:
            continue
        for r in range(1, len(order) - 1):
            if dist[order[r]] != float("inf"):
                dist[order[r]] += (objs[order[r + 1]][k] - objs[order[r - 1]][k]) / span
    return dist


def evolve_multiobjective(
    fitness_fn: MOFitnessFn,
    n_genes: int,
    n_options: int,
    *,
    pop_size: int = 24,
    generations: int = 20,
    mutation_rate: float = 0.1,
    tournament_k: int = 2,
    seed: int = 0,
    seed_genomes: list[CatGenome] | None = None,
) -> dict[str, object]:
    """NSGA-II over per-gene categorical assignments, maximizing a **vector** of
    objectives (e.g. ``(% memory saved, -Delta nll)``) to trace the whole tradeoff
    frontier rather than one budget point.

    Returns ``{front, history, evaluations}`` where ``front`` is the final Pareto
    front as ``(genome, objectives)`` pairs (deduped by genome) and ``history[g]``
    is the front size after generation ``g``. ``fitness_fn`` is memoized, so an
    expensive real-model evaluation runs at most once per distinct genome.

    Memetic: ``seed_genomes`` injects known solutions (e.g. budget-greedy points
    across a sweep) so the GA refines a good frontier instead of rediscovering it
    — the same rationale as :func:`evolve_categorical`, applied to the full curve.
    """
    rng = random.Random(seed)
    cache: dict[CatGenome, Objectives] = {}

    def ev(g: CatGenome) -> Objectives:
        o = cache.get(g)
        if o is None:
            o = tuple(fitness_fn(g))
            cache[g] = o
        return o

    pop: list[CatGenome] = list(seed_genomes or [])[:pop_size]
    while len(pop) < pop_size:
        pop.append(tuple(rng.randrange(n_options) for _ in range(n_genes)))

    def rank_and_crowd(genomes: list[CatGenome]) -> tuple[dict[int, int], dict[int, float]]:
        objs = [ev(g) for g in genomes]
        ranks: dict[int, int] = {}
        crowd: dict[int, float] = {}
        for r, f in enumerate(non_dominated_sort(objs)):
            cd = crowding_distance(objs, f)
            for i in f:
                ranks[i] = r
                crowd[i] = cd[i]
        return ranks, crowd

    def make_offspring(parents: list[CatGenome], ranks: dict[int, int], crowd: dict[int, float]) -> list[CatGenome]:
        def pick() -> CatGenome:
            best = rng.randrange(len(parents))
            for _ in range(tournament_k - 1):
                i = rng.randrange(len(parents))
                if ranks[i] < ranks[best] or (ranks[i] == ranks[best] and crowd[i] > crowd[best]):
                    best = i
            return parents[best]

        kids: list[CatGenome] = []
        while len(kids) < pop_size:
            kids.append(mutate_categorical(crossover(pick(), pick(), rng), n_options, mutation_rate, rng))
        return kids

    history: list[int] = []
    ranks, crowd = rank_and_crowd(pop)
    for _ in range(generations):
        combined = pop + make_offspring(pop, ranks, crowd)
        cobjs = [ev(g) for g in combined]
        new_pop: list[CatGenome] = []
        new_ranks: dict[int, int] = {}
        new_crowd: dict[int, float] = {}
        for r, f in enumerate(non_dominated_sort(cobjs)):
            cd = crowding_distance(cobjs, f)
            chosen = f if len(new_pop) + len(f) <= pop_size else sorted(
                f, key=lambda i: cd[i], reverse=True)[: pop_size - len(new_pop)]
            for i in chosen:
                new_ranks[len(new_pop)] = r
                new_crowd[len(new_pop)] = cd[i]
                new_pop.append(combined[i])
            if len(new_pop) >= pop_size:
                break
        pop, ranks, crowd = new_pop, new_ranks, new_crowd
        history.append(len(pareto_front(list(zip(pop, [ev(g) for g in pop])))))

    front = pareto_front(list(zip(pop, [ev(g) for g in pop])))
    seen: set[CatGenome] = set()
    deduped: list[tuple[CatGenome, Objectives]] = []
    for g, o in front:
        if g not in seen:
            seen.add(g)
            deduped.append((g, o))
    return {"front": deduped, "history": history, "evaluations": len(cache)}
