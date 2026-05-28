# SPDX-License-Identifier: Apache-2.0
"""Unit tests for PoC 0a state update genes.

破綻ゲート G1-G5 を pytest 経由でも独立検証できるよう、scripts/poc_0a_*.py の
gate 関数と並走する unit test。実行: ``pytest tests/unit/test_poc_0a_state_update.py``.
"""
from __future__ import annotations

import numpy as np
import pytest

from llcore.state_update import StateUpdateGene, eval_step, run_sequence


def test_gene_array_roundtrip() -> None:
    """as_array → from_array で gene が完全復元される."""
    g = StateUpdateGene(decay=0.85, mix=0.15, gate_str=1.3)
    g2 = StateUpdateGene.from_array(g.as_array())
    assert g == g2


def test_gene_clipped_within_range() -> None:
    """範囲外の値が clip される."""
    g = StateUpdateGene(decay=-0.5, mix=1.5, gate_str=3.0).clipped()
    assert 0.0 <= g.decay <= 1.0
    assert 0.0 <= g.mix <= 1.0
    assert 0.0 <= g.gate_str <= 2.0


def test_single_step_finite() -> None:
    """[G1] 単一 step で NaN/Inf が出ない."""
    rng = np.random.default_rng(0)
    state = rng.normal(0, 0.5, 8)
    x = rng.normal(0, 0.5, 8)
    gene = StateUpdateGene(decay=0.9, mix=0.1, gate_str=1.0)
    out = eval_step(state, x, gene)
    assert np.all(np.isfinite(out))
    assert out.shape == (8,)


def test_shape_mismatch_raises() -> None:
    """state と x の shape 不一致は ValueError."""
    gene = StateUpdateGene(decay=0.9, mix=0.1, gate_str=1.0)
    with pytest.raises(ValueError):
        eval_step(np.zeros(4), np.zeros(5), gene)


def test_run_sequence_shape() -> None:
    """run_sequence の出力形状は (L+1, dim)."""
    L, dim = 32, 4
    inputs = np.random.default_rng(1).normal(0, 0.5, size=(L, dim))
    gene = StateUpdateGene(decay=0.9, mix=0.1, gate_str=1.0)
    states = run_sequence(inputs, gene)
    assert states.shape == (L + 1, dim)
    assert np.allclose(states[0], 0.0)  # initial state = zero


def test_bounded_norm_g2() -> None:
    """[G2] L=256 で state norm が input norm の 100 倍以内."""
    rng = np.random.default_rng(2)
    L, dim = 256, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    gene = StateUpdateGene(decay=0.95, mix=0.05, gate_str=0.5)
    states = run_sequence(inputs, gene)
    state_norms = np.linalg.norm(states, axis=1)
    input_norm = float(np.linalg.norm(inputs, axis=1).mean())
    assert np.all(np.isfinite(state_norms))
    assert state_norms.max() < 100 * input_norm


def test_determinism_g3() -> None:
    """[G3] 同 seed で完全一致 (decay/mix/gate_str 数値演算の決定論性)."""
    L, dim = 64, 4
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    inputs_a = rng_a.normal(0, 0.5, size=(L, dim))
    inputs_b = rng_b.normal(0, 0.5, size=(L, dim))
    gene = StateUpdateGene(decay=0.8, mix=0.2, gate_str=1.2)
    s1 = run_sequence(inputs_a, gene)
    s2 = run_sequence(inputs_b, gene)
    assert np.array_equal(s1, s2)


@pytest.mark.parametrize(
    "decay,mix,gate_str",
    [
        (0.0, 0.5, 1.0),  # decay=0
        (0.9, 0.0, 1.0),  # mix=0
        (0.9, 0.1, 0.0),  # gate_str=0
        (0.0, 0.0, 0.0),  # all zero
        (1.0, 1.0, 2.0),  # all max
    ],
)
def test_degenerate_values_g4(decay: float, mix: float, gate_str: float) -> None:
    """[G4] 極端値で NaN/Inf 出ず、norm が有限."""
    rng = np.random.default_rng(3)
    inputs = rng.uniform(-0.5, 0.5, size=(32, 4))
    gene = StateUpdateGene(decay=decay, mix=mix, gate_str=gate_str)
    states = run_sequence(inputs, gene)
    assert np.all(np.isfinite(states))
    assert np.linalg.norm(states, axis=1).max() < 1e6


def test_random_population_g5() -> None:
    """[G5] random 個体集団 N=5 全てが finite + bounded."""
    rng = np.random.default_rng(4)
    L, dim = 256, 8
    inputs = rng.uniform(-1, 1, size=(L, dim))
    input_norm = float(np.linalg.norm(inputs, axis=1).mean())
    for _ in range(5):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(0.0, 1.0)),
            gate_str=float(rng.uniform(0.0, 2.0)),
        )
        states = run_sequence(inputs, gene)
        assert np.all(np.isfinite(states))
        assert np.linalg.norm(states, axis=1).max() < 100 * input_norm
