# SPDX-License-Identifier: Apache-2.0
"""llcore — Verified Evolution skeleton (CPU, additive research; src/ untouched).

This is the **骨組み (skeleton)** for llcore's founding thesis: *evolve the core
dynamics of an AI substrate without letting the verifier let them break*. It is
deliberately small and extensible — once evolution works here, adding a new
**evolution direction** = a new :class:`Objective`, and **feature extension** =
a new :class:`GeneCodec` / :class:`VerifierBackend` / kernel. Nothing else changes.

Three pluggable interfaces are all you need to extend:

* :class:`GeneCodec`      — how a genotype (flat real vector) maps to a domain
  gene, plus random/clip/crossover/mutate. Swap to evolve a different substrate
  (higher-dim coupled map, multi-kernel union, learning-rule gene, ...).
* :class:`Objective`      — a **deterministic** fitness (higher = better). This
  is the *direction* of evolution. Add a new task = add an Objective.
* :class:`VerifierBackend`— a fail-closed admission gate: ``certifies(gene)``
  returns True iff the gene is provably safe (here: contraction-certified). Swap
  the backend (``none`` / induced-norm / SDP-Lyapunov / future JSR) without
  touching the GA.

The GA itself (:func:`evolve`) is a minimal mirror of
``src/llcore/evolution/minimal_ga.py`` (tournament + uniform crossover + Gaussian
mutation + elitism) generalised to an arbitrary genotype vector, with the
verifier inserted at child admission (fail-closed: reject -> resample up to a cap
-> known-safe fallback). With ``VerifierBackend`` = none it is an ordinary GA.

Design rules honoured:
* **src/ is never imported-and-modified**; this file has zero src dependency
  (codecs may reuse research modules). It is a standalone skeleton.
* Determinism: a fixed ``rng`` seed fully determines the run (every random draw
  is threaded through ``rng``), so paired / common-random-number comparisons work.
* The verifier is **fail-closed**: an uncertified child is never admitted.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

import numpy as np

# --------------------------------------------------------------------------- #
# Pluggable interfaces (the three extension points).
# --------------------------------------------------------------------------- #


@runtime_checkable
class GeneCodec(Protocol):
    """Maps a flat genotype vector <-> a domain gene, with GA operators.

    A genotype is a 1-D float64 vector of length :attr:`dim`. ``to_gene`` decodes
    it into whatever object the :class:`Objective` and :class:`VerifierBackend`
    consume (e.g. a ``CoupledGene``). All operators take/return genotype vectors.
    """

    dim: int

    def random(self, rng: np.random.Generator) -> np.ndarray: ...
    def clip(self, g: np.ndarray) -> np.ndarray: ...
    def to_gene(self, g: np.ndarray) -> Any: ...
    def crossover(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray: ...
    def mutate(self, g: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray: ...


@runtime_checkable
class Objective(Protocol):
    """A deterministic fitness over decoded genes. Higher is better.

    Determinism (no per-eval RNG) is required so a flat landscape is genuinely
    flat, not measurement noise (llcore Step D lesson). The *direction* of
    evolution is entirely captured here — add a direction by adding an Objective.
    """

    name: str

    def fitness(self, gene: Any) -> float: ...


@runtime_checkable
class VerifierBackend(Protocol):
    """Fail-closed admission gate. ``certifies(gene)`` True => safe to admit.

    Backends differ only in *which* sound certificate they use. The GA is
    agnostic to the backend, which is the whole point of the verifier-backend
    plugin (llcore arc: the right backend is SDP-Lyapunov, not Z3/SMT)."""

    name: str

    def certifies(self, gene: Any) -> bool: ...


# --------------------------------------------------------------------------- #
# Result / config records.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EvolveConfig:
    pop_size: int = 30
    n_generations: int = 40
    tournament_k: int = 3
    crossover_rate: float = 0.5
    mutation_sigma: float = 0.15
    elitism: int = 1
    resample_cap: int = 50


@dataclass(frozen=True)
class EvolveResult:
    """Outcome of one :func:`evolve` run."""

    best_genotype: np.ndarray
    best_gene: Any
    best_fitness: float
    best_fitness_curve: tuple[float, ...]
    mean_fitness_curve: tuple[float, ...]
    diversity_curve: tuple[float, ...]
    final_population: tuple[np.ndarray, ...]   # genotype vectors
    # gate bookkeeping
    verifier_name: str
    objective_name: str
    n_rejections: int = 0
    n_resamples: int = 0
    fallback_count: int = 0
    n_children_generated: int = 0
    admitted_genotypes: tuple[np.ndarray, ...] = field(default_factory=tuple)


# --------------------------------------------------------------------------- #
# GA operators on a flat genotype (codec supplies domain-specific clip).
# --------------------------------------------------------------------------- #


def _tournament(pop: list[np.ndarray], fits: np.ndarray, k: int,
                rng: np.random.Generator) -> np.ndarray:
    """Pick the fittest of ``k`` uniformly-sampled individuals. Returns a copy."""
    idx = rng.integers(0, len(pop), size=k)
    best = idx[int(np.argmax(fits[idx]))]
    return pop[best].copy()


def _make_child(pop: list[np.ndarray], fits: np.ndarray, codec: GeneCodec,
                cfg: EvolveConfig, rng: np.random.Generator) -> np.ndarray:
    """One child via tournament(+crossover)+mutate. Draw order is fixed."""
    parent_a = _tournament(pop, fits, cfg.tournament_k, rng)
    if rng.random() < cfg.crossover_rate:
        parent_b = _tournament(pop, fits, cfg.tournament_k, rng)
        child = codec.crossover(parent_a, parent_b, rng)
    else:
        child = parent_a
    child = codec.mutate(child, cfg.mutation_sigma, rng)
    return codec.clip(child)


def evolve(codec: GeneCodec, objective: Objective, verifier: VerifierBackend,
           cfg: EvolveConfig | None = None, *,
           rng: np.random.Generator | None = None,
           initial_genotypes: list[np.ndarray] | None = None) -> EvolveResult:
    """Run verified evolution. Returns an :class:`EvolveResult`.

    The verifier is inserted at **child admission** (fail-closed): a child whose
    decoded gene the verifier does not certify is rejected and resampled up to
    ``cfg.resample_cap`` times; if the cap is hit, a known-safe fallback genotype
    is admitted instead (``codec`` supplies it via ``getattr(codec, 'fallback')``
    if present, else the clipped all-zero vector). With ``verifier.name == 'none'``
    the gate admits unconditionally and this is an ordinary GA.
    """
    cfg = cfg or EvolveConfig()
    if rng is None:
        rng = np.random.default_rng()
    if cfg.elitism > cfg.pop_size:
        raise ValueError("elitism must be <= pop_size")
    if cfg.tournament_k > cfg.pop_size:
        raise ValueError("tournament_k must be <= pop_size")

    fallback = getattr(codec, "fallback_genotype", None)
    if fallback is None:
        fallback = codec.clip(np.zeros(codec.dim))

    def _admits(genotype: np.ndarray) -> bool:
        return verifier.certifies(codec.to_gene(genotype))

    def _eval(genotypes: list[np.ndarray]) -> np.ndarray:
        return np.array([objective.fitness(codec.to_gene(g)) for g in genotypes],
                        dtype=np.float64)

    # initial population (not gated: the gate acts on CHILDREN, where the
    # admission decision of evolution lives; mirrors src minimal_ga).
    if initial_genotypes is None:
        pop = [codec.clip(codec.random(rng)) for _ in range(cfg.pop_size)]
    else:
        if len(initial_genotypes) != cfg.pop_size:
            raise ValueError("initial_genotypes size != pop_size")
        pop = [codec.clip(g.copy()) for g in initial_genotypes]
    fits = _eval(pop)

    def _div(p: list[np.ndarray]) -> float:
        return float(np.var(np.stack(p)))

    best_curve = [float(fits.max())]
    mean_curve = [float(fits.mean())]
    div_curve = [_div(pop)]

    n_rej = n_res = n_fb = n_child = 0
    admitted: list[np.ndarray] = []

    for _gen in range(cfg.n_generations):
        order = np.argsort(-fits)
        elites = [pop[i].copy() for i in order[: cfg.elitism]]

        children: list[np.ndarray] = []
        while len(children) < cfg.pop_size - cfg.elitism:
            child = _make_child(pop, fits, codec, cfg, rng)
            if _admits(child):
                children.append(child)
                admitted.append(child)
                n_child += 1
                continue
            n_rej += 1
            admitted_after = False
            for _ in range(cfg.resample_cap):
                n_res += 1
                child = _make_child(pop, fits, codec, cfg, rng)
                if _admits(child):
                    children.append(child)
                    admitted.append(child)
                    n_child += 1
                    admitted_after = True
                    break
                n_rej += 1
            if not admitted_after:
                children.append(fallback.copy())
                admitted.append(fallback.copy())
                n_child += 1
                n_fb += 1

        pop = elites + children
        fits = _eval(pop)
        best_curve.append(float(fits.max()))
        mean_curve.append(float(fits.mean()))
        div_curve.append(_div(pop))

    best_i = int(np.argmax(fits))
    return EvolveResult(
        best_genotype=pop[best_i].copy(),
        best_gene=codec.to_gene(pop[best_i]),
        best_fitness=float(fits[best_i]),
        best_fitness_curve=tuple(best_curve),
        mean_fitness_curve=tuple(mean_curve),
        diversity_curve=tuple(div_curve),
        final_population=tuple(p.copy() for p in pop),
        verifier_name=verifier.name,
        objective_name=objective.name,
        n_rejections=n_rej,
        n_resamples=n_res,
        fallback_count=n_fb,
        n_children_generated=n_child,
        admitted_genotypes=tuple(admitted),
    )
