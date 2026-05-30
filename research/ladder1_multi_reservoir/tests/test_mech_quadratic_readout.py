# SPDX-License-Identifier: Apache-2.0
"""機構 quadratic_readout の基質 sanity テスト.

- quadratic_features の出力 shape / finite / 線形項保存 / 二次項正しさ
- feature_dim と gene_dim の整合 (gene は基底 reservoir と同一, feature は二次展開)
- run が基底単層 reservoir と数値一致 (= reservoir 本体を変えていない回帰保証)
- make_eval_once が held-out R² を [0,1] で返し finite
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # mech_quadratic_readout, multi_reservoir
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))  # reservoir, memory_tasks

from mech_quadratic_readout import (  # noqa: E402
    QuadraticReadoutReservoir,
    gene_bounds,
    make_eval_once,
    quadratic_features,
)
from reservoir import LeakyDelayLineReservoir  # noqa: E402
from reservoir import gene_bounds as single_gene_bounds  # noqa: E402
from memory_tasks import DelayedParityTask  # noqa: E402


def test_quadratic_features_single_vector():
    h = np.array([2.0, -3.0, 0.5])
    phi = quadratic_features(h)
    # D = N + N*(N+1)/2 = 3 + 6 = 9
    assert phi.shape == (9,)
    assert np.all(np.isfinite(phi))
    # 線形項 (先頭 N) は h を保存
    assert np.allclose(phi[:3], h)
    # 二次項 (i<=j): (0,0)(0,1)(0,2)(1,1)(1,2)(2,2)
    expected_quad = np.array([4.0, -6.0, 1.0, 9.0, -1.5, 0.25])
    assert np.allclose(phi[3:], expected_quad)


def test_quadratic_features_batch():
    rng = np.random.default_rng(0)
    H = rng.standard_normal((5, 4))
    phi = quadratic_features(H)
    # D = 4 + 4*5/2 = 4 + 10 = 14
    assert phi.shape == (5, 14)
    assert np.all(np.isfinite(phi))
    # 各行が単一展開と一致
    for i in range(5):
        assert np.allclose(phi[i], quadratic_features(H[i]))


def test_feature_dim_and_gene_dim_consistency():
    res = QuadraticReadoutReservoir(n_taps=12, in_dim=1)
    # gene_dim は基底単層 reservoir と同一 (readout 拡張は gene を増やさない)
    base = LeakyDelayLineReservoir(n_taps=12, in_dim=1)
    assert res.gene_dim == base.gene_dim == 12 + 12 * 1
    # feature_dim = N + N(N+1)/2 = 12 + 78 = 90
    assert res.feature_dim == 12 + 12 * 13 // 2 == 90


def test_run_matches_base_reservoir():
    """run は基底単層 reservoir と完全一致 (reservoir 本体を一切変えていない保証)."""
    n_taps, in_dim = 12, 1
    res = QuadraticReadoutReservoir(n_taps=n_taps, in_dim=in_dim)
    base = LeakyDelayLineReservoir(n_taps=n_taps, in_dim=in_dim)
    rng = np.random.default_rng(7)
    gene = res.random_gene(rng)
    inputs = rng.standard_normal((20, in_dim))
    assert np.allclose(res.run(gene, inputs), base.run(gene, inputs), atol=1e-12)


def test_feature_dim_matches_run_then_feature():
    """run 最終状態を feature に通すと feature_dim になる (eval 経路の shape 整合)."""
    res = QuadraticReadoutReservoir(n_taps=8, in_dim=1)
    rng = np.random.default_rng(11)
    gene = res.random_gene(rng)
    inputs = rng.standard_normal((15, 1))
    final = res.run(gene, inputs)[-1]
    assert final.shape == (8,)
    phi = res.feature(final)
    assert phi.shape == (res.feature_dim,)
    assert np.all(np.isfinite(phi))


def test_gene_bounds_match_base():
    res = QuadraticReadoutReservoir(n_taps=12, in_dim=1)
    lo, hi = gene_bounds(res)
    lo_b, hi_b = single_gene_bounds(LeakyDelayLineReservoir(n_taps=12, in_dim=1))
    assert lo.shape == (res.gene_dim,)
    assert np.allclose(lo, lo_b) and np.allclose(hi, hi_b)


def test_random_gene_within_bounds():
    res = QuadraticReadoutReservoir(n_taps=12, in_dim=1)
    lo, hi = gene_bounds(res)
    rng = np.random.default_rng(2)
    for _ in range(20):
        g = res.random_gene(rng)
        assert g.shape == (res.gene_dim,)
        assert np.all(g >= lo - 1e-9) and np.all(g <= hi + 1e-9)


def test_eval_once_returns_finite_unit_interval():
    res = QuadraticReadoutReservoir(n_taps=12, in_dim=1)
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    eval_once = make_eval_once(res, task, n_train=32, n_eval=32)
    rng = np.random.default_rng(3)
    for _ in range(5):
        r2 = eval_once(res.random_gene(rng), rng)
        assert np.isfinite(r2)
        assert 0.0 <= r2 <= 1.0
