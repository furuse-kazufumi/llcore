# SPDX-License-Identifier: Apache-2.0
"""ParallelGatedReservoir 基質の sanity テスト (機構 parallel_gated).

検証項目 (基質 sanity):
- gene_dim / per_reservoir_gene_dim / feature_dim / n_pairs の計算正しさ
- features の出力 shape と finite 保証
- random_gene が bounds 内 & shape 一致
- gene 分割の整合 (K サブ gene に等分割され reservoir 数と一致)
- 積項が実際に「reservoir 間の要素積」になっていること (機構の核の検算)
- 不正な構成パラメータで ValueError
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # mech_parallel_gated
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))  # memory_tasks, reservoir

from mech_parallel_gated import (  # noqa: E402
    ParallelGatedReservoir,
    gene_bounds,
    make_eval_once,
)
from reservoir import LeakyDelayLineReservoir  # noqa: E402
from memory_tasks import DelayedParityTask  # noqa: E402


def test_gene_dim_default_k4_taps6():
    # K=4, n_taps=6, in_dim=1: 各 reservoir gene = 6 + 6*1 = 12 → 全体 48
    res = ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=1)
    assert res.per_reservoir_gene_dim == 12
    assert res.gene_dim == 4 * 12
    assert res.n_pairs == 6  # C(4,2)
    # feature: 線形 4*6=24 + 積 C(4,2)*6 = 6*6=36 → 60
    assert res.feature_dim == 24 + 36


def test_gene_dim_in_dim2():
    # in_dim=2: 各 reservoir gene = 6 + 6*2 = 18 → K=3 で 54
    res = ParallelGatedReservoir(n_reservoirs=3, n_taps=6, in_dim=2)
    assert res.per_reservoir_gene_dim == 18
    assert res.gene_dim == 3 * 18
    assert res.n_pairs == 3  # C(3,2)
    assert res.feature_dim == 3 * 6 + 3 * 6  # 線形 18 + 積 18


def test_features_shape_and_finite():
    res = ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=1)
    rng = np.random.default_rng(0)
    gene = res.random_gene(rng)
    inputs = rng.standard_normal((20, 1))
    feats = res.features(gene, inputs)
    assert feats.shape == (res.feature_dim,)
    assert np.all(np.isfinite(feats))


def test_features_finite_under_extreme_gene():
    # bounds 端 (極端な leak/w_in) でも tanh 飽和で finite を保証する。
    res = ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=1)
    lo, hi = gene_bounds(res)
    for gene in (lo, hi, (lo + hi) / 2.0):
        feats = res.features(gene, np.ones((30, 1)) * 5.0)
        assert feats.shape == (res.feature_dim,)
        assert np.all(np.isfinite(feats))


def test_random_gene_within_bounds():
    res = ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=1)
    lo, hi = gene_bounds(res)
    assert lo.shape == (res.gene_dim,)
    assert hi.shape == (res.gene_dim,)
    rng = np.random.default_rng(1)
    for _ in range(20):
        g = res.random_gene(rng)
        assert g.shape == (res.gene_dim,)
        assert np.all(g >= lo - 1e-9) and np.all(g <= hi + 1e-9)


def test_product_terms_are_pairwise_elementwise_products():
    """機構の核: 特徴後半が reservoir ペアの要素積であることを直接検算する."""
    res = ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=1)
    rng = np.random.default_rng(7)
    gene = res.random_gene(rng)
    inputs = rng.standard_normal((15, 1))

    # 各 reservoir の最終状態を独立に再計算 (proto を流用)。
    proto = LeakyDelayLineReservoir(n_taps=res.n_taps, in_dim=res.in_dim)
    d = res.per_reservoir_gene_dim
    finals = [proto.run(gene[k * d:(k + 1) * d], inputs)[-1] for k in range(res.n_reservoirs)]

    feats = res.features(gene, inputs)
    # 線形項 = 全 reservoir 状態の連結。
    linear = feats[: res.n_reservoirs * res.n_taps]
    assert np.allclose(linear, np.concatenate(finals))

    # 積項 = ペア (i<j) の要素積を順に連結。
    off = res.n_reservoirs * res.n_taps
    for i, j in combinations(range(res.n_reservoirs), 2):
        block = feats[off: off + res.n_taps]
        assert np.allclose(block, finals[i] * finals[j])
        off += res.n_taps
    assert off == res.feature_dim


def test_eval_once_returns_unit_interval():
    res = ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=1)
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    eval_once = make_eval_once(res, task, n_train=32, n_eval=32)
    rng = np.random.default_rng(3)
    for _ in range(5):
        r2 = eval_once(res.random_gene(rng), rng)
        assert 0.0 <= r2 <= 1.0


def test_split_gene_consistency():
    res = ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=1)
    rng = np.random.default_rng(11)
    gene = res.random_gene(rng)
    subs = res._split_gene(gene)
    assert len(subs) == res.n_reservoirs
    assert all(s.shape == (res.per_reservoir_gene_dim,) for s in subs)
    # 連結すると元の gene に戻る (等分割の整合)。
    assert np.allclose(np.concatenate(subs), gene)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        ParallelGatedReservoir(n_reservoirs=0, n_taps=6, in_dim=1)
    with pytest.raises(ValueError):
        ParallelGatedReservoir(n_reservoirs=4, n_taps=0, in_dim=1)
    with pytest.raises(ValueError):
        ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=0)
    # gene 次元不一致は features/_split_gene で ValueError。
    res = ParallelGatedReservoir(n_reservoirs=4, n_taps=6, in_dim=1)
    with pytest.raises(ValueError):
        res.features(np.zeros(res.gene_dim - 1), np.ones((10, 1)))
