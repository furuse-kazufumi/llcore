# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 0c minimal GA.

破綻ゲート G1-G7 を pytest 経由で独立検証。実行::

    pytest tests/unit/test_poc_0c_minimal_ga.py
"""
from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np
import pytest

from llcore.evolution import (
    Individual,
    Population,
    crossover_uniform,
    evolve,
    tournament_select,
    uniform_mutate,
)
from llcore.fitness import (
    AdditionTask,
    CopyTask,
    calibrate_baseline,
    evaluate_gene,
    make_fixed_readout,
)
from llcore.state_update import StateUpdateGene


# ---------------------------------------------------------------------------
# fixtures (module scope to amortize calibration cost)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def task_setup():
    readout_c = make_fixed_readout(8, 8, seed=1001)
    readout_a = make_fixed_readout(8, 1, seed=1002)
    copy0 = CopyTask(state_dim=8, out_dim=8, delay=0)
    add = AdditionTask(state_dim=8, out_dim=1)
    mse_c = calibrate_baseline(copy0, readout_c)
    mse_a = calibrate_baseline(add, readout_a)
    return (
        dc_replace(copy0, baseline_mse=mse_c),
        dc_replace(add, baseline_mse=mse_a),
        readout_c,
        readout_a,
    )


def _make_fitness(task, readout):
    def f(gene: StateUpdateGene, rng: np.random.Generator) -> float:
        return evaluate_gene(gene, task, readout, rng, n_trials=3)
    return f


# ---------------------------------------------------------------------------
# Basic API
# ---------------------------------------------------------------------------


def test_uniform_mutate_within_clip() -> None:
    """mutation が clip 範囲内に収まる."""
    rng = np.random.default_rng(0)
    g = StateUpdateGene(decay=0.5, mix=0.0, gate_str=0.0)
    for _ in range(20):
        g2 = uniform_mutate(g, sigma=0.5, rng=rng)
        assert 0.0 <= g2.decay <= 1.0
        assert -1.0 <= g2.mix <= 1.0
        assert -2.0 <= g2.gate_str <= 2.0


def test_crossover_uniform_returns_valid_gene() -> None:
    """crossover で valid な子 gene が出る."""
    rng = np.random.default_rng(0)
    a = StateUpdateGene(decay=0.1, mix=-0.5, gate_str=1.0)
    b = StateUpdateGene(decay=0.9, mix=0.5, gate_str=-1.0)
    child = crossover_uniform(a, b, rng)
    # 各 parameter は a か b のどちらか
    assert child.decay in (a.decay, b.decay)
    assert child.mix in (a.mix, b.mix)
    assert child.gate_str in (a.gate_str, b.gate_str)


def test_tournament_select_returns_max() -> None:
    """tournament k=size = greedy (常に best 選ぶ)."""
    rng = np.random.default_rng(0)
    inds = (
        Individual(gene=StateUpdateGene(0.1, 0.1, 0.1), fitness=0.3),
        Individual(gene=StateUpdateGene(0.5, 0.5, 0.5), fitness=0.7),
        Individual(gene=StateUpdateGene(0.9, 0.9, 0.9), fitness=0.5),
    )
    pop = Population(individuals=inds)
    selected = tournament_select(pop, k=3, rng=rng)
    assert selected.fitness == 0.7


# ---------------------------------------------------------------------------
# G1-G7
# ---------------------------------------------------------------------------


def test_g1_evolve_completes_no_nan(task_setup) -> None:
    """[G1] 10x10 evolve が完走、NaN/Inf 出ない."""
    copy_task, _, readout_c, _ = task_setup
    result = evolve(
        _make_fitness(copy_task, readout_c),
        pop_size=10, n_generations=10,
        rng=np.random.default_rng(1001),
    )
    assert len(result.generations) == 11
    assert all(np.isfinite(f) for f in result.best_fitness_curve)


def test_g2_no_extinction(task_setup) -> None:
    """[G2] 全 generation で size 一定."""
    copy_task, _, readout_c, _ = task_setup
    result = evolve(
        _make_fitness(copy_task, readout_c),
        pop_size=10, n_generations=10,
        rng=np.random.default_rng(1002),
    )
    assert all(g.size == 10 for g in result.generations)


def test_g3_best_monotonic_with_elitism(task_setup) -> None:
    """[G3] elitism=1 で best fitness が単調非減少 (elite 前世代 fitness 保持)."""
    copy_task, _, readout_c, _ = task_setup
    result = evolve(
        _make_fitness(copy_task, readout_c),
        pop_size=10, n_generations=10, elitism=1,
        rng=np.random.default_rng(1003),
    )
    curve = result.best_fitness_curve
    for i in range(len(curve) - 1):
        assert curve[i + 1] >= curve[i] - 1e-9, f"monotonic violated at gen {i}: {curve[i]} -> {curve[i+1]}"


def test_g4_diversity_maintained(task_setup) -> None:
    """[G4] gene 多様性が維持される (variance > 1e-6)."""
    copy_task, _, readout_c, _ = task_setup
    result = evolve(
        _make_fitness(copy_task, readout_c),
        pop_size=10, n_generations=10,
        rng=np.random.default_rng(1004),
    )
    assert min(result.diversity_curve) > 1e-6


def test_g5_evolution_competitive_with_random(task_setup) -> None:
    """[G5] 進化が random search baseline と互角以上 (0.9 比)."""
    copy_task, _, readout_c, _ = task_setup
    fit = _make_fitness(copy_task, readout_c)
    rng = np.random.default_rng(1005)
    # 200 random
    best_random = 0.0
    for _ in range(200):
        g = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        best_random = max(best_random, fit(g, rng))
    # 進化 3 seed 中 best (compute 軽量化)
    best_evolved = 0.0
    for seed in (2001, 2002, 2003):
        r = evolve(fit, pop_size=10, n_generations=10, rng=np.random.default_rng(seed))
        best_evolved = max(best_evolved, r.final_best.fitness)
    assert best_evolved >= best_random * 0.9


def test_g6_determinism_same_seed(task_setup) -> None:
    """[G6] 同 seed で 2 回完全一致."""
    copy_task, _, readout_c, _ = task_setup
    fit = _make_fitness(copy_task, readout_c)
    r1 = evolve(fit, pop_size=10, n_generations=10, rng=np.random.default_rng(2020))
    r2 = evolve(fit, pop_size=10, n_generations=10, rng=np.random.default_rng(2020))
    assert r1.best_fitness_curve == r2.best_fitness_curve
    assert r1.final_best.fitness == r2.final_best.fitness


def test_g7_specialist_emerges(task_setup) -> None:
    """[G7] copy と add で異なる best gene (specialist) — dist > 0.1."""
    copy_task, add_task, readout_c, readout_a = task_setup
    r_c = evolve(
        _make_fitness(copy_task, readout_c),
        pop_size=10, n_generations=10,
        rng=np.random.default_rng(3001),
    )
    r_a = evolve(
        _make_fitness(add_task, readout_a),
        pop_size=10, n_generations=10,
        rng=np.random.default_rng(3001),
    )
    dist = float(np.linalg.norm(
        r_c.final_best.gene.as_array() - r_a.final_best.gene.as_array()
    ))
    assert dist > 0.1, f"specialist not emerged: dist={dist}"
