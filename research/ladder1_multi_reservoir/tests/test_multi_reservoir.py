# SPDX-License-Identifier: Apache-2.0
"""DeepReservoir 基質の sanity テスト.

- gene_dim / total_taps の計算正しさ
- run の出力 shape と finite 保証
- random_gene が bounds 内
- 単一層 DeepReservoir((N,)) が step_c の LeakyDelayLineReservoir と数値一致
  (= 多層化が単一層を真に一般化している回帰保証)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # multi_reservoir
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))  # 単一版 reservoir

from multi_reservoir import DeepReservoir, gene_bounds, make_behavior, make_eval_once
from reservoir import LeakyDelayLineReservoir
from reservoir import gene_bounds as single_gene_bounds
from memory_tasks import DelayedParityTask


def test_gene_dim_single_layer():
    # 1 層 8 taps, in_dim 1: leak(8) + w_in(8*1) = 16
    res = DeepReservoir(layer_taps=(8,), in_dim=1)
    assert res.gene_dim == 16
    assert res.total_taps == 8
    assert res.n_layers == 1


def test_gene_dim_multi_layer():
    # 層0: leak(8)+w_in(8*1)=16, 層1: leak(8)+w_in(8*8)=72 → 88
    res = DeepReservoir(layer_taps=(8, 8), in_dim=1)
    assert res.gene_dim == 16 + 72
    assert res.total_taps == 16
    # in_dim=2 の場合 層0 入力次元が 2: leak(8)+w_in(8*2)=24, 層1: 72 → 96
    res2 = DeepReservoir(layer_taps=(8, 8), in_dim=2)
    assert res2.gene_dim == 24 + 72


def test_run_shape_and_finite():
    res = DeepReservoir(layer_taps=(6, 5, 4), in_dim=2)
    rng = np.random.default_rng(0)
    gene = res.random_gene(rng)
    inputs = rng.standard_normal((20, 2))
    states = res.run(gene, inputs)
    assert states.shape == (20, res.total_taps)
    assert np.all(np.isfinite(states))


def test_random_gene_within_bounds():
    res = DeepReservoir(layer_taps=(8, 8), in_dim=1)
    lo, hi = gene_bounds(res)
    rng = np.random.default_rng(1)
    for _ in range(20):
        g = res.random_gene(rng)
        assert g.shape == (res.gene_dim,)
        assert np.all(g >= lo - 1e-9) and np.all(g <= hi + 1e-9)


def test_invalid_layer_taps_raise():
    with pytest.raises(ValueError):
        DeepReservoir(layer_taps=(), in_dim=1)
    with pytest.raises(ValueError):
        DeepReservoir(layer_taps=(8, 0), in_dim=1)


def test_single_layer_matches_legacy_reservoir():
    """単一層 DeepReservoir が step_c の単一 reservoir と数値一致する.

    gene レイアウトと run 数式が同一なので、同じ gene で同じ状態列を出すべき。
    これが成立すれば多層化は単一層の真の一般化 (回帰なし)。
    """
    n_taps, in_dim = 8, 1
    deep = DeepReservoir(layer_taps=(n_taps,), in_dim=in_dim)
    legacy = LeakyDelayLineReservoir(n_taps=n_taps, in_dim=in_dim)

    # gene_dim・bounds が一致することを先に確認
    assert deep.gene_dim == legacy.gene_dim
    lo_d, hi_d = gene_bounds(deep)
    lo_l, hi_l = single_gene_bounds(legacy)
    assert np.allclose(lo_d, lo_l) and np.allclose(hi_d, hi_l)

    rng = np.random.default_rng(42)
    gene = deep.random_gene(rng)
    inputs = rng.standard_normal((25, in_dim))
    s_deep = deep.run(gene, inputs)
    s_legacy = legacy.run(gene, inputs)
    assert np.allclose(s_deep, s_legacy, atol=1e-12)


def test_eval_once_returns_unit_interval():
    res = DeepReservoir(layer_taps=(8, 8), in_dim=1)
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    eval_once = make_eval_once(res, task, n_train=32, n_eval=32)
    rng = np.random.default_rng(3)
    for _ in range(5):
        r2 = eval_once(res.random_gene(rng), rng)
        assert 0.0 <= r2 <= 1.0


def test_behavior_shape():
    res = DeepReservoir(layer_taps=(8, 8), in_dim=1)
    behavior = make_behavior(res)
    rng = np.random.default_rng(4)
    b = behavior(res.random_gene(rng))
    assert b.shape == (2,)
    assert np.all(np.isfinite(b))
