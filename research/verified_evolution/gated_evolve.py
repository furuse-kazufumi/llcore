# SPDX-License-Identifier: Apache-2.0
"""Track B — verifier-gated GA loop (additive; src untouched).

`gated_evolve` re-implements the minimal GA child-admission loop of
`src/llcore/evolution/minimal_ga.py::evolve`, reusing its operators, and inserts
an optional Z3 soundness gate at child admission. Gate modes:

- ``"none"``        — accept every child unconditionally. **Byte-identical** to
  ``src`` ``evolve()`` (same RNG draw order: tournament(+crossover)+mutate, no
  extra draws). This is the control: any difference vs ``evolve()`` would be a
  re-implementation bug, so we assert equality in :func:`assert_none_matches_src`.
- ``"state_norm"``  — admit only if ``verify_gene_safe(gene).ok`` (Z3 |s|<=1 gate).
- ``"contraction"`` — admit only if
  ``verify_lipschitz_contraction(gene).contraction is True`` (Z3 L<1 gate).

Fail-closed: a rejected child is RESAMPLED (regenerate from tournament parents)
up to ``resample_cap`` times; if the cap is hit, fall back to a known-safe gene
and increment ``fallback_count``. RNG is threaded through every resample so a
fixed seed is fully deterministic.

honest note: in ``"none"`` mode we deliberately DO NOT call the gate and DO NOT
consume any extra RNG, so the draw stream matches ``src`` ``evolve()`` exactly.
In gated modes the resampling consumes additional draws — that is the whole point
(the gate changes the search trajectory).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.evolution.minimal_ga import (  # noqa: E402
    EvolutionResult,
    Individual,
    Population,
    crossover_uniform,
    evaluate_population,
    initialize_random_population,
    tournament_select,
    uniform_mutate,
)
from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.verifier.invariants import (  # noqa: E402
    verify_gene_safe,
    verify_lipschitz_contraction,
)

# Known-safe fallback gene: decay=0.5, mix=0.0, gate_str=0.0.
# Closed-form Lipschitz upper bound = max(|0.5|, |0.5 + 0.5*0.0|) = 0.5 < 1
# (state-direction contraction), and convex-combination => |s|<=1 (state_norm).
FALLBACK_GENE = StateUpdateGene(decay=0.5, mix=0.0, gate_str=0.0)

GateMode = str  # "none" | "state_norm" | "contraction"


def _gate_admits(gene: StateUpdateGene, mode: GateMode) -> bool:
    """Return True iff the gate admits ``gene`` under ``mode`` (fail-closed)."""
    if mode == "none":
        return True
    if mode == "state_norm":
        return bool(verify_gene_safe(gene).ok)
    if mode == "contraction":
        # fail-closed: only a hard True (Z3 unsat / certified) admits.
        return verify_lipschitz_contraction(gene).contraction is True
    raise ValueError(f"unknown gate mode {mode!r}")


@dataclass(frozen=True)
class GatedEvolutionResult:
    """:class:`EvolutionResult` plus gate bookkeeping."""

    result: EvolutionResult
    gate_mode: GateMode
    n_rejections: int          # total children rejected (across all resamples)
    n_resamples: int           # total resample attempts that produced a child
    fallback_count: int        # times the resample cap was hit -> fallback gene
    n_children_generated: int  # total children admitted into populations
    admitted_genes: tuple[StateUpdateGene, ...]  # every admitted (non-elite) child gene


def _make_child(
    pop: Population,
    *,
    tournament_k: int,
    crossover_rate: float,
    mutation_sigma: float,
    rng: np.random.Generator,
) -> StateUpdateGene:
    """One child via tournament(+crossover)+mutate.

    RNG draw order matches ``src`` ``evolve()`` exactly:
    tournament_select -> rng.random() -> [tournament_select + crossover] -> mutate.
    """
    parent_a = tournament_select(pop, tournament_k, rng)
    if rng.random() < crossover_rate:
        parent_b = tournament_select(pop, tournament_k, rng)
        child = crossover_uniform(parent_a.gene, parent_b.gene, rng)
    else:
        child = parent_a.gene
    child = uniform_mutate(child, mutation_sigma, rng)
    return child


def gated_evolve(
    fitness_func,
    *,
    gate_mode: GateMode = "none",
    pop_size: int = 10,
    n_generations: int = 10,
    tournament_k: int = 3,
    mutation_sigma: float = 0.15,
    crossover_rate: float = 0.5,
    elitism: int = 1,
    resample_cap: int = 50,
    rng: np.random.Generator | None = None,
    initial_pop: list[StateUpdateGene] | None = None,
) -> GatedEvolutionResult:
    """Verifier-gated minimal GA (RWKV gene only; reuses src operators).

    See module docstring. Returns an :class:`EvolutionResult`-equivalent plus
    gate bookkeeping. With ``gate_mode="none"`` the RNG stream and results are
    byte-identical to ``src`` ``evolve()``.
    """
    if rng is None:
        rng = np.random.default_rng()
    if elitism > pop_size:
        raise ValueError(f"elitism={elitism} must be <= pop_size={pop_size}")
    if tournament_k > pop_size:
        raise ValueError(f"tournament_k={tournament_k} must be <= pop_size={pop_size}")

    def _div(p: Population) -> float:
        return float(p.gene_matrix.var())

    # initial population: src uses initialize_random_population (same draws).
    # NOTE: the initial population is NOT gated in src; to stay byte-identical in
    # "none" mode we also do not gate it. (The gate acts on CHILDREN, which is
    # where evolution's admission decision lives.)
    if initial_pop is None:
        initial_pop = initialize_random_population(pop_size, rng)
    elif len(initial_pop) != pop_size:
        raise ValueError(f"initial_pop size {len(initial_pop)} != pop_size {pop_size}")

    pop = evaluate_population(initial_pop, fitness_func, rng)
    generations: list[Population] = [pop]
    best_curve: list[float] = [pop.best.fitness]
    diversity_curve: list[float] = [_div(pop)]

    n_rejections = 0
    n_resamples = 0
    fallback_count = 0
    n_children = 0
    admitted: list[StateUpdateGene] = []

    for _gen in range(n_generations):
        sorted_inds = sorted(pop.individuals, key=lambda ind: -ind.fitness)
        elites: list[Individual] = list(sorted_inds[:elitism])

        new_genes: list[StateUpdateGene] = []
        while len(new_genes) < pop_size - elitism:
            child = _make_child(
                pop,
                tournament_k=tournament_k,
                crossover_rate=crossover_rate,
                mutation_sigma=mutation_sigma,
                rng=rng,
            )
            if _gate_admits(child, gate_mode):
                new_genes.append(child)
                admitted.append(child)
                n_children += 1
                continue
            # rejected: resample (fail-closed) up to resample_cap.
            n_rejections += 1
            admitted_after_resample = False
            for _ in range(resample_cap):
                n_resamples += 1
                child = _make_child(
                    pop,
                    tournament_k=tournament_k,
                    crossover_rate=crossover_rate,
                    mutation_sigma=mutation_sigma,
                    rng=rng,
                )
                if _gate_admits(child, gate_mode):
                    new_genes.append(child)
                    admitted.append(child)
                    n_children += 1
                    admitted_after_resample = True
                    break
                n_rejections += 1
            if not admitted_after_resample:
                # cap hit -> known-safe fallback (counts as admitted, no RNG draw).
                new_genes.append(FALLBACK_GENE)
                admitted.append(FALLBACK_GENE)
                n_children += 1
                fallback_count += 1

        new_pop = evaluate_population(new_genes, fitness_func, rng)
        pop = Population(individuals=tuple(elites) + new_pop.individuals)
        generations.append(pop)
        best_curve.append(pop.best.fitness)
        diversity_curve.append(_div(pop))

    result = EvolutionResult(
        generations=tuple(generations),
        best_fitness_curve=tuple(best_curve),
        diversity_curve=tuple(diversity_curve),
    )
    return GatedEvolutionResult(
        result=result,
        gate_mode=gate_mode,
        n_rejections=n_rejections,
        n_resamples=n_resamples,
        fallback_count=fallback_count,
        n_children_generated=n_children,
        admitted_genes=tuple(admitted),
    )


def assert_none_matches_src(seeds=(1000, 1001, 1002), **evolve_kwargs) -> bool:
    """Sanity: gated_evolve(gate_mode="none") == src evolve() byte-identically.

    Returns True if best_fitness_curve and diversity_curve match exactly across
    the given seeds. Used by exp_b_runner as a control before the real cells.
    """
    from llcore.evolution import evolve as src_evolve

    # a tiny deterministic fitness that exercises the gene without external state.
    def _ff(gene: StateUpdateGene, rng: np.random.Generator) -> float:
        # uses rng so the stochastic-fitness draw order is exercised too.
        x = rng.uniform(-1, 1, size=(4, 3))
        g = gene.clipped()
        return float(np.tanh(g.decay + g.mix - g.gate_str + x.sum()))

    kw = dict(pop_size=10, n_generations=8, tournament_k=3, mutation_sigma=0.15,
              crossover_rate=0.5, elitism=1)
    kw.update(evolve_kwargs)
    for s in seeds:
        r_src = src_evolve(_ff, rng=np.random.default_rng(s), **kw)
        r_gat = gated_evolve(_ff, gate_mode="none", rng=np.random.default_rng(s), **kw)
        if r_src.best_fitness_curve != r_gat.result.best_fitness_curve:
            return False
        if r_src.diversity_curve != r_gat.result.diversity_curve:
            return False
    return True
