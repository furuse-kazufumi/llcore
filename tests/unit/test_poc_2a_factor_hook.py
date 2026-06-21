# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 2a factor_hook × RWKV mock."""
from __future__ import annotations

from dataclasses import replace as dc_replace

import numpy as np

from llcore.evolution import evolve
from llcore.factor_hook import (
    FACTOR_NAMES,
    FactorSnapshot,
    HeuristicFactorHook,
    NoopFactorHook,
    ThoughtFactorDeltaHook,
    apply_hook_to_gene,
)
from llcore.fitness import (
    CopyTask,
    calibrate_baseline,
    evaluate_gene,
    make_fixed_readout,
)
from llcore.state_update import StateUpdateGene


def test_factor_names_canonical_10() -> None:
    """10 因子の canonical 名前."""
    assert len(FACTOR_NAMES) == 10
    expected = {
        "structurize", "reconstruct", "closed_loop", "self_extension",
        "uncertainty", "exploration", "integrate", "provenance",
        "perspective", "reality_contact",
    }
    assert set(FACTOR_NAMES) == expected


def test_g1_snapshot_default_and_clamp() -> None:
    """[G1] 未指定因子 default 0.5、範囲外 clamp [0,1]."""
    snap = FactorSnapshot(values={"uncertainty": 1.5, "exploration": -0.3})
    assert snap.get("structurize") == 0.5  # default
    assert snap.get("uncertainty") == 1.0  # clamp upper
    assert snap.get("exploration") == 0.0  # clamp lower
    assert len(snap.vector()) == 10


def test_g2_noop_always_one() -> None:
    """[G2] NoopFactorHook delta_for = 1.0 常時."""
    h = NoopFactorHook()
    assert h.delta_for(FactorSnapshot()) == 1.0
    assert h.delta_for(FactorSnapshot(values={"uncertainty": 0.99})) == 1.0


def test_g3_heuristic_directionality() -> None:
    """[G3] uncertainty 高 → Δ<1, integrate 高 → Δ>1, 単調."""
    h = HeuristicFactorHook(sensitivity=1.0)
    d_unc = h.delta_for(FactorSnapshot(values={"uncertainty": 1.0, "integrate": 0.0, "structurize": 0.0}))
    d_int = h.delta_for(FactorSnapshot(values={"uncertainty": 0.0, "integrate": 1.0, "structurize": 1.0}))
    d_neu = h.delta_for(FactorSnapshot())
    assert d_unc < d_neu < d_int
    assert d_unc < 1.0 < d_int


def test_g3_heuristic_clamp() -> None:
    """extreme snapshot でも Δ ∈ [0.25, 4.0] に clamp."""
    h = HeuristicFactorHook(sensitivity=100.0)  # 超 large sensitivity
    d_unc = h.delta_for(FactorSnapshot(values={"uncertainty": 1.0}))
    d_int = h.delta_for(FactorSnapshot(values={"integrate": 1.0, "structurize": 1.0}))
    assert 0.25 <= d_unc <= 4.0
    assert 0.25 <= d_int <= 4.0


def test_g4_apply_hook_modifies_decay() -> None:
    """[G4] heuristic hook で decay が変化、noop で不変."""
    gene = StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5)
    snap = FactorSnapshot(values={"uncertainty": 0.9})
    g_noop = apply_hook_to_gene(gene, NoopFactorHook(), snap)
    g_heu = apply_hook_to_gene(gene, HeuristicFactorHook(sensitivity=1.5), snap)
    assert abs(g_noop.decay - gene.decay) < 1e-9  # noop: 不変
    assert abs(g_heu.decay - gene.decay) > 0.01  # heuristic: 変化


def test_g5_snapshot_distinguishability() -> None:
    """[G5] 異なる snapshot で effective gene が異なる."""
    gene = StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5)
    h = HeuristicFactorHook(sensitivity=1.5)
    g_a = apply_hook_to_gene(gene, h, FactorSnapshot(values={"uncertainty": 0.9}))
    g_b = apply_hook_to_gene(gene, h, FactorSnapshot(values={"integrate": 0.9, "structurize": 0.9}))
    assert abs(g_a.decay - g_b.decay) > 0.05


def test_g6_determinism() -> None:
    """[G6] 同 hook + snapshot で完全一致."""
    h = HeuristicFactorHook(sensitivity=1.0)
    snap = FactorSnapshot(values={"uncertainty": 0.7, "integrate": 0.3})
    assert h.delta_for(snap) == h.delta_for(snap)
    gene = StateUpdateGene(decay=0.6, mix=0.4, gate_str=0.7)
    assert apply_hook_to_gene(gene, h, snap) == apply_hook_to_gene(gene, h, snap)


def test_g7_evolution_smoke_with_hook() -> None:
    """[G7] 進化ループに factor_hook 注入で 10x10 完走 + monotonic."""
    readout = make_fixed_readout(8, 8, seed=1001)
    base_task = CopyTask(state_dim=8, out_dim=8, delay=0)
    task = dc_replace(base_task, baseline_mse=calibrate_baseline(base_task, readout))
    hook = HeuristicFactorHook(sensitivity=0.8)
    snap = FactorSnapshot(values={"uncertainty": 0.6, "integrate": 0.4, "structurize": 0.4})

    def hooked_fitness(gene: StateUpdateGene, rng: np.random.Generator) -> float:
        return evaluate_gene(apply_hook_to_gene(gene, hook, snap), task, readout, rng, n_trials=3)

    result = evolve(hooked_fitness, pop_size=10, n_generations=10, rng=np.random.default_rng(2024))
    assert all(np.isfinite(f) for f in result.best_fitness_curve)
    curve = result.best_fitness_curve
    for i in range(len(curve) - 1):
        assert curve[i + 1] >= curve[i] - 1e-9


def test_protocol_runtime_check() -> None:
    """Noop / Heuristic とも ThoughtFactorDeltaHook プロトコルを満たす."""
    assert isinstance(NoopFactorHook(), ThoughtFactorDeltaHook)
    assert isinstance(HeuristicFactorHook(), ThoughtFactorDeltaHook)
