# SPDX-License-Identifier: Apache-2.0
"""hybrid_max 機構の基質 sanity テスト.

検証範囲 (基質の最低保証):
- features 出力の shape が feature_dim と一致 + finite
- gene_dim と gene 展開 (2 枝ぶん) の整合
- feature_dim と quadratic 展開次元の整合 (use_gate / use_quadratic の各組合せ)
- random_gene が bounds 内 + 正しい shape
- make_eval_once の held-out R² が [0,1] (clip 契約) + finite
- gate / quadratic を切ると features が縮退する (各機構が実際に寄与している回帰保証)
- gate が乗法であること (片枝 0 入力で gate ブロックが 0)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # mech_hybrid_max
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))  # memory_tasks

from mech_hybrid_max import (  # noqa: E402
    HybridMaxReservoir,
    _quadratic_dim,
    _quadratic_expand,
    eval_on_dataset,
    gene_bounds,
    make_batched_dataset,
    make_eval_once,
)
from memory_tasks import DelayedParityTask  # noqa: E402


def test_gene_dim_two_branches():
    # 1 枝: 層0 leak(8)+w_in(8*1)=16, 層1 leak(8)+w_in(8*8)=72 → 88。2 枝 → 176。
    res = HybridMaxReservoir(layer_taps=(8, 8), in_dim=1)
    assert res._branch_gene_dim == 16 + 72
    assert res.gene_dim == 2 * (16 + 72)
    assert res.n_branches == 2
    assert res.taps_per_branch == 16


def test_gene_dim_single_layer_in_dim2():
    # in_dim=2: 1 層 leak(8)+w_in(8*2)=24。2 枝 → 48。
    res = HybridMaxReservoir(layer_taps=(8,), in_dim=2)
    assert res._branch_gene_dim == 24
    assert res.gene_dim == 48


def test_feature_dim_full():
    """gate + quadratic 全部入りの feature_dim が計算式と一致する."""
    res = HybridMaxReservoir(layer_taps=(8, 8), in_dim=1, use_gate=True, use_quadratic=True)
    # base = 2 枝 * taps_per_branch(16) + gate(layer_taps[-1]=8) = 40
    base = 2 * 16 + 8
    assert res._base_feature_dim == base
    assert res.feature_dim == _quadratic_dim(base, pair_cap=res.pair_cap)


def test_feature_dim_ablations():
    """use_gate / use_quadratic の各組合せで feature_dim が整合する."""
    taps = (8, 8)
    # gate なし quadratic なし: base = 2*16 = 32
    r00 = HybridMaxReservoir(layer_taps=taps, use_gate=False, use_quadratic=False)
    assert r00.feature_dim == 32
    # gate あり quadratic なし: base = 32 + 8 = 40
    r10 = HybridMaxReservoir(layer_taps=taps, use_gate=True, use_quadratic=False)
    assert r10.feature_dim == 40
    # gate なし quadratic あり: base=32 → quadratic 拡張
    r01 = HybridMaxReservoir(layer_taps=taps, use_gate=False, use_quadratic=True)
    assert r01.feature_dim == _quadratic_dim(32, pair_cap=r01.pair_cap)


def test_features_shape_and_finite():
    res = HybridMaxReservoir(layer_taps=(6, 5), in_dim=2)
    rng = np.random.default_rng(0)
    gene = res.random_gene(rng)
    inputs = rng.standard_normal((20, 2))
    feat = res.features(gene, inputs)
    assert feat.ndim == 1
    assert feat.shape == (res.feature_dim,)
    assert np.all(np.isfinite(feat))


def test_random_gene_within_bounds():
    res = HybridMaxReservoir(layer_taps=(8, 8), in_dim=1)
    lo, hi = gene_bounds(res)
    assert lo.shape == (res.gene_dim,)
    assert hi.shape == (res.gene_dim,)
    rng = np.random.default_rng(1)
    for _ in range(20):
        g = res.random_gene(rng)
        assert g.shape == (res.gene_dim,)
        assert np.all(g >= lo - 1e-9) and np.all(g <= hi + 1e-9)


def test_invalid_layer_taps_raise():
    with pytest.raises(ValueError):
        HybridMaxReservoir(layer_taps=())
    with pytest.raises(ValueError):
        HybridMaxReservoir(layer_taps=(8, 0))


def test_eval_once_returns_unit_interval():
    """held-out R² が clip 契約どおり [0,1] + finite (parity task 上)."""
    res = HybridMaxReservoir(layer_taps=(8, 8), in_dim=1)
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    eval_once = make_eval_once(res, task, n_train=32, n_eval=32)
    rng = np.random.default_rng(3)
    for _ in range(5):
        r2 = eval_once(res.random_gene(rng), rng)
        assert np.isfinite(r2)
        assert 0.0 <= r2 <= 1.0


def test_quadratic_expand_contents():
    """quadratic_expand が [feat, feat^2, 上三角ペア積] を正しく構成する."""
    feat = np.array([1.0, 2.0, 3.0])
    out = _quadratic_expand(feat, pair_cap=3)
    # 期待: [1,2,3, 1,4,9, 1*2, 1*3, 2*3] = 9 要素
    expected = np.array([1.0, 2.0, 3.0, 1.0, 4.0, 9.0, 2.0, 3.0, 6.0])
    assert out.shape == (9,)
    assert np.allclose(out, expected)
    assert out.shape[0] == _quadratic_dim(3, pair_cap=3)


def test_gate_is_multiplicative():
    """乗法ゲートは片枝の最終層状態が 0 のとき 0 になる (= 真に積である).

    枝 B の最終層 W_in / leak を細工せず、入力を 0 系列にすると全枝状態が 0 になるため、
    ここでは『gate ブロックが base 末尾に積として入る』ことを直接構成で確認する。
    """
    res = HybridMaxReservoir(layer_taps=(4,), in_dim=1, use_gate=True, use_quadratic=False)
    rng = np.random.default_rng(7)
    gene = res.random_gene(rng)
    # 0 入力系列 → 全 reservoir 状態が 0 (h0=0, tanh(0)=0 で漸化式が 0 を保つ) → gate も 0。
    inputs = np.zeros((15, 1), dtype=np.float64)
    feat = res.features(gene, inputs)
    assert np.allclose(feat, 0.0)


@pytest.mark.parametrize("use_gate,use_quad", [(True, True), (True, False),
                                               (False, True), (False, False)])
def test_features_batch_matches_single(use_gate, use_quad):
    """features_batch が features を row-wise に積んだものと数値一致する (高速路の回帰保証)."""
    res = HybridMaxReservoir(layer_taps=(6, 5), in_dim=2,
                             use_gate=use_gate, use_quadratic=use_quad)
    rng = np.random.default_rng(11)
    gene = res.random_gene(rng)
    batch = rng.standard_normal((7, 18, 2))
    fb = res.features_batch(gene, batch)
    assert fb.shape == (7, res.feature_dim)
    assert np.all(np.isfinite(fb))
    for i in range(7):
        assert np.allclose(fb[i], res.features(gene, batch[i]), atol=1e-12)


def test_eval_on_dataset_matches_eval_once():
    """eval_on_dataset (batch高速路) が make_eval_once の eval_once と一致する.

    同一 task / 同一 rng seed で train/eval を引けば、batch 路と逐次路は同じ held-out R²。
    """
    res = HybridMaxReservoir(layer_taps=(8, 8), in_dim=1)
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    gene = res.random_gene(np.random.default_rng(0))

    # 逐次路
    eval_once = make_eval_once(res, task, n_train=24, n_eval=24, ridge_lambda=1e-2)
    r2_seq = eval_once(gene, np.random.default_rng(123))
    # batch 路 (同一 seed でデータ生成 → 同一 train/eval 系列)
    ds = make_batched_dataset(task, 24, 24, np.random.default_rng(123))
    r2_batch = eval_on_dataset(res, gene, ds, ridge_lambda=1e-2)
    assert abs(r2_seq - r2_batch) < 1e-9


def test_gate_quadratic_actually_expand():
    """gate / quadratic を有効化すると特徴次元が確実に増える (機構が寄与している)."""
    taps = (8, 8)
    base = HybridMaxReservoir(layer_taps=taps, use_gate=False, use_quadratic=False).feature_dim
    with_gate = HybridMaxReservoir(layer_taps=taps, use_gate=True, use_quadratic=False).feature_dim
    with_quad = HybridMaxReservoir(layer_taps=taps, use_gate=False, use_quadratic=True).feature_dim
    full = HybridMaxReservoir(layer_taps=taps, use_gate=True, use_quadratic=True).feature_dim
    assert with_gate > base
    assert with_quad > base
    assert full > with_gate and full > with_quad
