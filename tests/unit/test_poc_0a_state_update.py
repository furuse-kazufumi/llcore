# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 0a v2 state update genes (RWKV-style leak integrator).

履歴:
- v1: ``decay*s + mix*x*tanh(gate_str*s)`` → zero attractor degenerate
- v2 (2026-05-29): 2 reviewer verdict で RWKV-style に再設計
  ``decay*s + (1-decay)*tanh(mix*x + gate_str*s)``

破綻ゲート G1-G10 を pytest 経由でも独立検証。実行::

    pytest tests/unit/test_poc_0a_state_update.py
"""
from __future__ import annotations

import numpy as np
import pytest

from llcore.state_update import StateUpdateGene, eval_step, run_sequence


# ---------------------------------------------------------------------------
# 基本 API
# ---------------------------------------------------------------------------


def test_gene_array_roundtrip() -> None:
    """as_array → from_array で gene が完全復元される."""
    g = StateUpdateGene(decay=0.85, mix=-0.4, gate_str=1.3)
    g2 = StateUpdateGene.from_array(g.as_array())
    assert g == g2


def test_gene_clipped_v2_ranges() -> None:
    """v2 で拡張された clip 範囲 (mix:-1..1, gate_str:-2..2) を検証."""
    g = StateUpdateGene(decay=-0.5, mix=2.0, gate_str=5.0).clipped()
    assert g.decay == 0.0
    assert g.mix == 1.0
    assert g.gate_str == 2.0
    g2 = StateUpdateGene(decay=1.5, mix=-3.0, gate_str=-5.0).clipped()
    assert g2.decay == 1.0
    assert g2.mix == -1.0
    assert g2.gate_str == -2.0


def test_shape_mismatch_raises() -> None:
    """state と x の shape 不一致は ValueError."""
    gene = StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.5)
    with pytest.raises(ValueError):
        eval_step(np.zeros(4), np.zeros(5), gene)


def test_run_sequence_shape() -> None:
    """run_sequence の出力形状は (L+1, dim)."""
    L, dim = 32, 4
    inputs = np.random.default_rng(1).normal(0, 0.5, size=(L, dim))
    gene = StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.5)
    states = run_sequence(inputs, gene)
    assert states.shape == (L + 1, dim)
    assert np.allclose(states[0], 0.0)  # initial state = zero


# ---------------------------------------------------------------------------
# G1-G5 (v1 から継続)
# ---------------------------------------------------------------------------


def test_g1_single_step_finite() -> None:
    """[G1] 単一 step で NaN/Inf 出ない."""
    rng = np.random.default_rng(0)
    state = rng.normal(0, 0.5, 8)
    x = rng.normal(0, 0.5, 8)
    gene = StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.5)
    out = eval_step(state, x, gene)
    assert np.all(np.isfinite(out))
    assert out.shape == (8,)


def test_g2_bounded_norm() -> None:
    """[G2] L=256 で state norm が input norm の 10 倍以内 (v2 で K=100→10)."""
    rng = np.random.default_rng(2)
    L, dim = 256, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    gene = StateUpdateGene(decay=0.95, mix=0.5, gate_str=0.5)
    states = run_sequence(inputs, gene)
    state_norms = np.linalg.norm(states, axis=1)
    input_norm = float(np.linalg.norm(inputs, axis=1).mean())
    assert np.all(np.isfinite(state_norms))
    assert state_norms.max() < 10 * input_norm


def test_g3_determinism() -> None:
    """[G3] 同 seed で完全一致."""
    L, dim = 64, 4
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    inputs_a = rng_a.normal(0, 0.5, size=(L, dim))
    inputs_b = rng_b.normal(0, 0.5, size=(L, dim))
    gene = StateUpdateGene(decay=0.8, mix=0.5, gate_str=0.5)
    s1 = run_sequence(inputs_a, gene)
    s2 = run_sequence(inputs_b, gene)
    assert np.array_equal(s1, s2)


@pytest.mark.parametrize(
    "decay,mix,gate_str,name",
    [
        (0.0, 0.5, 0.5, "decay=0"),
        (1.0, 0.5, 0.5, "decay=1"),
        (0.9, 0.0, 0.5, "mix=0"),
        (0.9, -0.5, 0.5, "mix_neg"),  # v2 で負許容
        (0.9, 0.5, 0.0, "gate_str=0"),
        (0.9, 0.5, -1.0, "gate_str_neg"),  # v2 で負許容 (抑制性)
    ],
)
def test_g4_degenerate_values(decay: float, mix: float, gate_str: float, name: str) -> None:
    """[G4] 極端値で finite + bounded."""
    rng = np.random.default_rng(3)
    inputs = rng.uniform(-0.5, 0.5, size=(32, 4))
    gene = StateUpdateGene(decay=decay, mix=mix, gate_str=gate_str)
    states = run_sequence(inputs, gene)
    assert np.all(np.isfinite(states)), f"NaN/Inf in {name}"
    assert np.linalg.norm(states, axis=1).max() < 1e6, f"unbounded in {name}"


def test_g5_random_population_v2() -> None:
    """[G5] random 個体集団 N=20 (v2 拡張) が全員 finite + bounded."""
    rng = np.random.default_rng(4)
    L, dim = 256, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    input_norm = float(np.linalg.norm(inputs, axis=1).mean())
    for _ in range(20):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        states = run_sequence(inputs, gene)
        assert np.all(np.isfinite(states))
        assert np.linalg.norm(states, axis=1).max() < 10 * input_norm


# ---------------------------------------------------------------------------
# G6-G10 (v2 追加)
# ---------------------------------------------------------------------------


def test_g6_nontrivial_activation() -> None:
    """[G6] 非自明性: 非ゼロ入力で state variance > 0 (zero attractor 弾く)."""
    rng = np.random.default_rng(5)
    inputs = rng.uniform(-1, 1, size=(256, 8))
    gene = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    states = run_sequence(inputs, gene)
    state_norms = np.linalg.norm(states[10:], axis=1)
    input_norm = float(np.linalg.norm(inputs, axis=1).mean())
    assert state_norms.mean() > 0.01 * input_norm
    assert state_norms.var() > 1e-6


def test_g7_input_distinguishability() -> None:
    """[G7] 入力区別性: 異なる入力列で最終 state の相対距離 > 0.1."""
    rng = np.random.default_rng(6)
    inputs_a = rng.uniform(-1, 1, size=(256, 8))
    rng2 = np.random.default_rng(7)
    inputs_b = rng2.uniform(-1, 1, size=(256, 8))
    gene = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    s_a = run_sequence(inputs_a, gene)[-1]
    s_b = run_sequence(inputs_b, gene)[-1]
    eps = 1e-10
    rel = float(np.linalg.norm(s_a - s_b) / (np.linalg.norm(s_a) + np.linalg.norm(s_b) + eps))
    assert rel > 0.1


def test_g8_memory_persistence() -> None:
    """[G8] 記憶持続性: zero-input phase で state が即落ちしない."""
    rng = np.random.default_rng(8)
    L1, L2, dim = 50, 50, 8
    inputs = np.concatenate([rng.uniform(-1, 1, size=(L1, dim)), np.zeros((L2, dim))])
    gene = StateUpdateGene(decay=0.95, mix=0.5, gate_str=0.5)
    states = run_sequence(inputs, gene)
    norm_at_50 = float(np.linalg.norm(states[L1]))
    norm_at_100 = float(np.linalg.norm(states[L1 + L2]))
    assert norm_at_50 > 1e-6
    assert norm_at_100 > 0.01 * norm_at_50


def test_g9_zero_state_escape() -> None:
    """[G9] zero-state escape: state=0 初期で N step 以内に norm > eps (v1 degenerate 撃退)."""
    rng = np.random.default_rng(9)
    L, dim = 16, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    gene = StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5)
    states = run_sequence(inputs, gene, initial_state=np.zeros(dim))
    state_norms = np.linalg.norm(states, axis=1)
    # state[0] = 0 必須
    assert state_norms[0] == 0.0
    # 1-5 step 以内に escape
    escape = next((t for t in range(1, L + 1) if state_norms[t] > 1e-3), -1)
    assert 1 <= escape <= 5


def test_g10_parameter_sensitivity() -> None:
    """[G10] parameter sensitivity: 各 gene perturbation で出力距離 > 0.01."""
    rng = np.random.default_rng(10)
    inputs = rng.uniform(-1, 1, size=(128, 8))
    base = StateUpdateGene(decay=0.7, mix=0.5, gate_str=0.5)
    s_base = run_sequence(inputs, base)[-1]
    perturbed = [
        ("decay+0.2", StateUpdateGene(decay=0.9, mix=0.5, gate_str=0.5)),
        ("mix+0.3", StateUpdateGene(decay=0.7, mix=0.8, gate_str=0.5)),
        ("gate+0.5", StateUpdateGene(decay=0.7, mix=0.5, gate_str=1.0)),
    ]
    for name, gene in perturbed:
        s_p = run_sequence(inputs, gene)[-1]
        dist = float(np.linalg.norm(s_p - s_base))
        assert dist > 0.01, f"{name} dead: dist={dist}"


# ---------------------------------------------------------------------------
# v2 regression: v1 の zero attractor が再発しないことを直接確認
# ---------------------------------------------------------------------------


def test_v1_zero_attractor_regression() -> None:
    """v1 では state=0 初期で永久に 0 だった (degenerate)。
    v2 では同条件で state が動くことを確認."""
    inputs = np.random.default_rng(11).uniform(-1, 1, size=(64, 8))
    gene = StateUpdateGene(decay=0.95, mix=0.05, gate_str=0.5)  # v1 で degenerate だった gene
    states = run_sequence(inputs, gene, initial_state=np.zeros(8))
    # v2: 必ず非ゼロに動く
    assert np.linalg.norm(states[-1]) > 1e-3
    assert np.linalg.norm(states[10:], axis=1).var() > 1e-6
