# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 2b — persona-indexed specialist × verifier × open-ended evolution.

各 gate (G1-G8) に対し最低 1 test を持ち、加えて persona prior / adaptive floor /
lineage reservoir / modes meter の API 単体 test を併設する。
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from llcore.evolution import (
    AdaptiveFloorGate,
    LineageReservoir,
    ModesMeter,
    pairwise_l2_diversity,
)
from llcore.fitness import CopyTask, calibrate_baseline, make_fixed_readout
from llcore.persona import (
    NUM_PERSONAS,
    PERSONA_LABELS,
    PERSONA_PRIORS,
    PersonaPrior,
    persona_sample_gene,
)
from llcore.state_update import StateUpdateGene


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def task_and_readout():
    readout = make_fixed_readout(8, 8, seed=2001)
    base = CopyTask(state_dim=8, out_dim=8, delay=0)
    mse = calibrate_baseline(base, readout)
    return replace(base, baseline_mse=mse), readout


# ---------------------------------------------------------------------------
# Persona prior 単体
# ---------------------------------------------------------------------------


def test_persona_priors_count_and_labels() -> None:
    """8 persona prior + label が一貫している."""
    assert NUM_PERSONAS == 8
    assert len(PERSONA_PRIORS) == 8
    assert len(PERSONA_LABELS) == 8
    ids = [p.persona_id for p in PERSONA_PRIORS]
    assert ids == list(range(8))


def test_persona_priors_are_distinct() -> None:
    """各 persona の mean 位置は最低 1 軸で他と区別できる (identifiability)."""
    means = np.array([p.mean_array() for p in PERSONA_PRIORS])
    # 全 persona 対の mean 距離が 1e-3 以上
    for i in range(8):
        for j in range(i + 1, 8):
            d = float(np.linalg.norm(means[i] - means[j]))
            assert d > 1e-3, f"persona {i} and {j} identical means: {means[i]} vs {means[j]}"


def test_persona_sample_gene_within_clip_range() -> None:
    """8 persona prior すべてのサンプル gene が clip 範囲内."""
    rng = np.random.default_rng(20260530)
    for pid in range(8):
        for _ in range(20):
            gene = persona_sample_gene(pid, rng)
            assert 0.0 <= gene.decay <= 1.0
            assert -1.0 <= gene.mix <= 1.0
            assert -2.0 <= gene.gate_str <= 2.0


def test_persona_sample_gene_invalid_id() -> None:
    rng = np.random.default_rng(42)
    with pytest.raises(ValueError):
        persona_sample_gene(8, rng)
    with pytest.raises(ValueError):
        persona_sample_gene(-1, rng)


# ---------------------------------------------------------------------------
# AdaptiveFloorGate 単体
# ---------------------------------------------------------------------------


def test_adaptive_floor_ratchet_monotonic() -> None:
    """[G5 core] ratchet=True で floor が単調非減少."""
    gate = AdaptiveFloorGate(percentile=30.0, ratchet=True)
    gate.update([0.1, 0.2, 0.3, 0.4, 0.5])
    f1 = gate.floor
    gate.update([0.05, 0.05, 0.05, 0.05, 0.05])  # 退化
    f2 = gate.floor
    gate.update([0.3, 0.4, 0.5, 0.6, 0.7])  # 改善
    f3 = gate.floor
    assert f1 > 0  # 初期 floor > 0
    assert f2 >= f1 - 1e-12  # ratchet で退化しない
    assert f3 >= f2 - 1e-12  # 改善で上昇 (or 等しい)
    assert gate.is_monotonic()


def test_adaptive_floor_no_ratchet_can_drop() -> None:
    """ratchet=False では floor が下がりうる."""
    gate = AdaptiveFloorGate(percentile=50.0, ratchet=False)
    gate.update([1.0, 1.0, 1.0, 1.0])
    f1 = gate.floor
    gate.update([0.0, 0.0, 0.0, 0.0])
    f2 = gate.floor
    assert f2 < f1


def test_adaptive_floor_survivors_protects_top_1() -> None:
    """全員 fail でも top-1 が保護されて全滅回避."""
    gate = AdaptiveFloorGate(percentile=50.0, ratchet=True)
    gate.floor = 100.0  # 異常な高 floor
    survivors = gate.survivors([0.1, 0.5, 0.2])
    assert len(survivors) == 1
    assert survivors[0] == 1  # argmax


def test_adaptive_floor_invalid_percentile() -> None:
    with pytest.raises(ValueError):
        AdaptiveFloorGate(percentile=-1.0)
    with pytest.raises(ValueError):
        AdaptiveFloorGate(percentile=101.0)


# ---------------------------------------------------------------------------
# LineageReservoir 単体
# ---------------------------------------------------------------------------


def test_lineage_reservoir_best_overwrite() -> None:
    """高 fitness で上書きされる."""
    res = LineageReservoir()
    g1 = StateUpdateGene(0.5, 0.5, 0.5)
    g2 = StateUpdateGene(0.6, 0.5, 0.5)
    g3 = StateUpdateGene(0.7, 0.5, 0.5)
    assert res.update_best(0, g1, 0.3)
    assert res.update_best(0, g2, 0.5)  # 上書き
    assert not res.update_best(0, g3, 0.4)  # 低 fitness なので維持
    best = res.get_best(0)
    assert best is not None
    assert best[0] == 0.5
    assert best[1] == g2


def test_lineage_reservoir_reinject_extinct() -> None:
    """[G3 core] 絶滅 persona の best-ever を再投入リストで返す."""
    res = LineageReservoir()
    for pid in range(8):
        res.update_best(pid, StateUpdateGene(0.5, 0.5, 0.5), 0.1 * pid)
    # present に 3 つだけ残し、5 つは絶滅
    present = {0, 3, 7}
    revive = res.reinject_extinct(present)
    revived_ids = {r[0] for r in revive}
    assert revived_ids == {1, 2, 4, 5, 6}


def test_lineage_reservoir_reinject_protected_only() -> None:
    res = LineageReservoir()
    for pid in range(8):
        res.update_best(pid, StateUpdateGene(0.5, 0.5, 0.5), 0.1)
    present = {0, 1, 2}
    revive = res.reinject_extinct(present, protected={3, 4, 5})  # 6, 7 は保護外
    assert {r[0] for r in revive} == {3, 4, 5}


# ---------------------------------------------------------------------------
# ModesMeter 単体
# ---------------------------------------------------------------------------


def test_modes_meter_a_new_first_obs_all_new() -> None:
    """[G6 core] 最初の observe では全 descriptor が新規."""
    meter = ModesMeter(n_bins=16)
    rng = np.random.default_rng(123)
    genes = [
        StateUpdateGene(
            decay=float(rng.uniform(0, 1)),
            mix=float(rng.uniform(-1, 1)),
            gate_str=float(rng.uniform(-2, 2)),
        )
        for _ in range(20)
    ]
    a_new, div = meter.observe(genes)
    assert a_new > 0  # 全 descriptor が新規
    assert div > 0  # 多様な集団なので pairwise > 0


def test_modes_meter_a_new_decreases_on_repeat() -> None:
    """同じ集団を再 observe すると A_new = 0 (新規なし)."""
    meter = ModesMeter(n_bins=8)
    genes = [
        StateUpdateGene(decay=0.5, mix=0.0, gate_str=0.0),
        StateUpdateGene(decay=0.3, mix=0.2, gate_str=0.5),
    ]
    a1, _ = meter.observe(genes)
    a2, _ = meter.observe(genes)
    assert a1 > 0
    assert a2 == 0


def test_modes_meter_regime_adaptive() -> None:
    """常に新規 gene を投入すると adaptive 判定."""
    meter = ModesMeter(n_bins=32)
    rng = np.random.default_rng(456)
    for _ in range(20):
        genes = [
            StateUpdateGene(
                decay=float(rng.uniform(0, 1)),
                mix=float(rng.uniform(-1, 1)),
                gate_str=float(rng.uniform(-2, 2)),
            )
            for _ in range(10)
        ]
        meter.observe(genes)
    assert meter.regime() == "adaptive"


def test_modes_meter_is_adaptive_active_and_gate_pass() -> None:
    """[G6 AND-gate] A_new active + diversity 維持で adaptive_active=True (Codex Q4 対応)."""
    meter = ModesMeter(n_bins=32)
    rng = np.random.default_rng(789)
    for _ in range(20):
        genes = [
            StateUpdateGene(
                decay=float(rng.uniform(0, 1)),
                mix=float(rng.uniform(-1, 1)),
                gate_str=float(rng.uniform(-2, 2)),
            )
            for _ in range(10)
        ]
        meter.observe(genes)
    ok, info = meter.is_adaptive_active(
        active_threshold=0.9, require_no_diversity_collapse=True
    )
    assert ok is True
    assert info["a_new_active_frac"] >= 0.9
    assert info["diversity_collapsed"] is False


def test_modes_meter_is_adaptive_active_and_gate_blocks_diversity_collapse() -> None:
    """[G6 AND-gate negative] diversity 崩壊時は A_new active でも adaptive を主張しない.

    判定ロジックの単体検査として ``a_new_history`` / ``diversity_history`` を
    直接設定し、AND gate の左 (A_new active) と右 (NOT diversity_collapsed) の
    両方が要求されることを確かめる (data-driven 不安定性を回避).
    """
    meter = ModesMeter(n_bins=32)
    # A_new active 維持 (常に >0 → frac=1.0)
    meter.a_new_history = [5] * 20
    # diversity head 1.0 → tail 0.01 (崩壊: tail/head = 1%, threshold 0.05 下回る)
    meter.diversity_history = [1.0] * 5 + [0.5] * 10 + [0.01] * 5

    ok_strict, info_strict = meter.is_adaptive_active(
        active_threshold=0.9, require_no_diversity_collapse=True
    )
    assert info_strict["a_new_active_frac"] == 1.0
    assert info_strict["diversity_collapsed"] is True
    assert ok_strict is False  # 左 OK / 右 NG → AND False

    # require_no_diversity_collapse=False なら A_new 単独で True に戻る
    ok_lax, _ = meter.is_adaptive_active(
        active_threshold=0.9, require_no_diversity_collapse=False
    )
    assert ok_lax is True


def test_pairwise_l2_diversity_zero_singleton() -> None:
    """1 個体だけなら diversity = 0."""
    assert pairwise_l2_diversity([StateUpdateGene(0.5, 0.0, 0.0)]) == 0.0


def test_pairwise_l2_diversity_nonzero_distinct() -> None:
    """異なる gene 集団なら diversity > 0."""
    genes = [
        StateUpdateGene(0.1, 0.0, 0.0),
        StateUpdateGene(0.9, 0.0, 0.0),
    ]
    d = pairwise_l2_diversity(genes)
    assert d > 0.5  # decay 差 0.8


# ---------------------------------------------------------------------------
# Integration / smoke test (小規模 run で G1, G2, G3, G7, G8 を smoke 検証)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_evolution_traces(task_and_readout):
    """小規模 (8 persona × 2 ind = 16, 10 世代) で specialist + control を実行."""
    # 遅延 import で test collection 高速化
    import sys
    from pathlib import Path

    scripts_dir = Path(__file__).resolve().parents[2] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        from poc_2b_persona_indexed_verified_evolution import run_persona_evolution
    finally:
        sys.path.remove(str(scripts_dir))

    task, readout = task_and_readout
    spec = run_persona_evolution(
        n_personas=8, pop_per_persona=2, n_generations=10,
        task=task, readout=readout,
        rng=np.random.default_rng(99),
        floor_percentile=30.0,
        use_reservoir=True, use_floor=True, use_verifier=True,
    )
    ctrl = run_persona_evolution(
        n_personas=1, pop_per_persona=16, n_generations=10,
        task=task, readout=readout,
        rng=np.random.default_rng(99),
        floor_percentile=30.0,
        use_reservoir=False, use_floor=True, use_verifier=True,
    )
    return spec, ctrl


def test_g1_kernel_coverage_smoke(small_evolution_traces) -> None:
    """[G1] specialist 集団は control より多軸で広い coverage を持つ (smoke)."""
    spec, ctrl = small_evolution_traces
    spec_union = np.array([
        ind.gene.as_array() for pop in spec.populations for ind in pop
    ])
    ctrl_union = np.array([
        ind.gene.as_array() for pop in ctrl.populations for ind in pop
    ])
    spec_var = float(spec_union.var(axis=0).sum())
    ctrl_var = float(ctrl_union.var(axis=0).sum())
    # specialist が control より広いはず (8 persona prior による)
    assert spec_var > ctrl_var, f"specialist var={spec_var:.4f} not > control={ctrl_var:.4f}"


def test_g2_verifier_differentiation_smoke(small_evolution_traces) -> None:
    """[G2] verifier rejection rate が persona 間で std > 0 で差別化 (smoke)."""
    spec, _ = small_evolution_traces
    rates = []
    for pid, rejects in spec.verifier_reject_by_persona.items():
        if rejects:
            rates.append(sum(1 for r in rejects if r) / len(rejects))
    assert len(rates) >= 4  # 最低半数の persona でサンプルあり
    std = float(np.std(rates))
    assert std > 0.0, f"verifier rejection rates identical across personas: {rates}"


def test_g3_majority_personas_survive_smoke(small_evolution_traces) -> None:
    """[G3 smoke] 8 persona × 2 ind = 16 (極小集団) で過半 5/8 以上の persona 生存.

    本番 G3 (32 ind / 50 gen) では full 8/8 を実証する (poc_2b verdict)。
    smoke は極小集団 + verifier 過酷 (state_bound=0.4) で reservoir 救済が遅延する
    可能性があるため過半を要件とする (honest 留保: smoke スケール限定)。
    """
    spec, _ = small_evolution_traces
    final = spec.populations[-1]
    present = {ind.persona_id for ind in final}
    # smoke では reservoir の救済が遅れる極小集団のため、過半 (5/8) 以上を要求
    assert len(present) >= 5, f"only {len(present)} personas survived: {present}"


def test_g4_best_fitness_monotonic_smoke(small_evolution_traces) -> None:
    """[G4] best fitness curve が単調非減少 (smoke)."""
    spec, _ = small_evolution_traces
    curve = spec.best_fitness_curve
    for i in range(len(curve) - 1):
        assert curve[i + 1] >= curve[i] - 1e-9, f"non-monotonic at {i}: {curve[i]} → {curve[i + 1]}"


def test_g5_floor_monotonic_smoke(small_evolution_traces) -> None:
    """[G5] floor history 単調非減少 (smoke)."""
    spec, _ = small_evolution_traces
    valid = [f for f in spec.floor_history if f != float("-inf")]
    for i in range(len(valid) - 1):
        assert valid[i + 1] >= valid[i] - 1e-12


def test_g6_a_new_active_smoke(small_evolution_traces) -> None:
    """[G6] A_new > 0 を 80% 以上の世代で維持 (smoke は緩和 80%)."""
    spec, _ = small_evolution_traces
    active = sum(1 for a in spec.a_new_history if a > 0) / len(spec.a_new_history)
    assert active >= 0.8, f"A_new active frac={active}"


def test_g7_no_extinction_smoke(small_evolution_traces) -> None:
    """[G7] 集団サイズが min_pop=8 以上を全世代で."""
    spec, _ = small_evolution_traces
    sizes = [len(p) for p in spec.populations]
    assert min(sizes) >= 8


def test_g8_verifier_latency_smoke(small_evolution_traces) -> None:
    """[G8 smoke] verifier latency mean < 20 ms / call.

    smoke は 16 個体 × 10 世代 = ~200 verifier call で warm-up 不足のため
    本番 PoC (32 × 50, mean=6.07ms) より per-call が遅くなる (~10-15ms 観測).
    smoke の役割は trace 構造の sanity check で、production claim は PoC
    スクリプト本走の数値 (verdict doc G8) に従う. **honest**: 20ms 閾値は
    "verifier 健在 + Z3 timeout / hang なし" を意味するに留め、
    "< 10ms" 主張は本走数値が根拠.
    """
    spec, _ = small_evolution_traces
    arr = np.array(spec.verifier_latencies_ms)
    assert arr.size > 0
    assert arr.mean() < 20.0, f"smoke verifier latency mean={arr.mean():.2f}ms"
