# SPDX-License-Identifier: Apache-2.0
"""機構 wide_single 基質の sanity テスト.

- gene_dim / total_taps の計算正しさ (幅 n_taps と整合)
- run の出力 shape と finite 保証
- random_gene が bounds 内
- eval_once が held-out R² を [0, 1] で返す (finite)
- WideSingleConfig が流用基質 LeakyDelayLineReservoir と数値一致
  (= 薄いラッパが基質を改変せず素通ししている回帰保証)
- 不正引数で ValueError
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # mech_wide_single
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))  # reservoir, memory_tasks

from mech_wide_single import (  # noqa: E402
    DEFAULT_WIDTHS,
    WideSingleConfig,
    random_search_ceiling,
    width_ceiling_curve,
)
from reservoir import LeakyDelayLineReservoir  # noqa: E402
from memory_tasks import DelayedParityTask  # noqa: E402


def test_gene_dim_and_total_taps():
    # 幅 8, in_dim 1: leak(8) + w_in(8*1) = 16, total_taps=8
    cfg = WideSingleConfig(n_taps=8, in_dim=1)
    assert cfg.gene_dim == 16
    assert cfg.total_taps == 8
    # 幅 48: leak(48)+w_in(48*1)=96, total_taps=48
    cfg48 = WideSingleConfig(n_taps=48, in_dim=1)
    assert cfg48.gene_dim == 96
    assert cfg48.total_taps == 48
    # in_dim=2: leak(24)+w_in(24*2)=72
    cfg2 = WideSingleConfig(n_taps=24, in_dim=2)
    assert cfg2.gene_dim == 24 + 24 * 2


def test_label():
    assert WideSingleConfig(n_taps=24).label == "1L-24wide"
    assert WideSingleConfig(n_taps=64).label == "1L-64wide"


def test_run_shape_and_finite():
    cfg = WideSingleConfig(n_taps=48, in_dim=1)
    rng = np.random.default_rng(0)
    gene = cfg.random_gene(rng)
    inputs = rng.standard_normal((20, 1))
    states = cfg.run(gene, inputs)
    assert states.shape == (20, cfg.total_taps)
    assert np.all(np.isfinite(states))


def test_random_gene_within_bounds():
    cfg = WideSingleConfig(n_taps=64, in_dim=1)
    lo, hi = cfg.gene_bounds()
    rng = np.random.default_rng(1)
    for _ in range(20):
        g = cfg.random_gene(rng)
        assert g.shape == (cfg.gene_dim,)
        assert np.all(g >= lo - 1e-9) and np.all(g <= hi + 1e-9)


def test_eval_once_returns_finite_unit_interval():
    cfg = WideSingleConfig(n_taps=24, in_dim=1)
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    eval_once = cfg.make_eval_once(task, n_train=32, n_eval=32)
    rng = np.random.default_rng(3)
    for _ in range(5):
        r2 = eval_once(cfg.random_gene(rng), rng)
        assert np.isfinite(r2)
        assert 0.0 <= r2 <= 1.0


def test_wrapper_matches_legacy_reservoir():
    """WideSingleConfig が流用基質 LeakyDelayLineReservoir と数値一致する.

    ラッパは run/gene/bounds を基質へ素通しするだけなので、同じ gene で同じ
    状態列を出すべき。成立すれば「基質を改変せず流用している」回帰保証になる。
    """
    n_taps, in_dim = 48, 1
    cfg = WideSingleConfig(n_taps=n_taps, in_dim=in_dim)
    legacy = LeakyDelayLineReservoir(n_taps=n_taps, in_dim=in_dim)

    assert cfg.gene_dim == legacy.gene_dim
    lo_c, hi_c = cfg.gene_bounds()
    from reservoir import gene_bounds as legacy_bounds  # noqa: E402
    lo_l, hi_l = legacy_bounds(legacy)
    assert np.allclose(lo_c, lo_l) and np.allclose(hi_c, hi_l)

    rng = np.random.default_rng(42)
    gene = cfg.random_gene(rng)
    inputs = rng.standard_normal((25, in_dim))
    assert np.allclose(cfg.run(gene, inputs), legacy.run(gene, inputs), atol=1e-12)


def test_invalid_args_raise():
    with pytest.raises(ValueError):
        WideSingleConfig(n_taps=0, in_dim=1)
    with pytest.raises(ValueError):
        WideSingleConfig(n_taps=8, in_dim=0)


def test_random_search_ceiling_determinism():
    """同 seed_idx で random_search_ceiling が決定論的に同じ値を返す (再現性)."""
    cfg = WideSingleConfig(n_taps=8, in_dim=1)
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    a = random_search_ceiling(cfg, task, n_random=20, seed_idx=0, n_train=24, n_eval=24)
    b = random_search_ceiling(cfg, task, n_random=20, seed_idx=0, n_train=24, n_eval=24)
    assert a == b
    assert np.isfinite(a)
    assert 0.0 <= a <= 1.0


def test_width_ceiling_curve_shapes():
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    curve = width_ceiling_curve(
        task, widths=(8, 24), n_random=10, n_seeds=3, n_train=24, n_eval=24,
    )
    assert set(curve.keys()) == {8, 24}
    for w, vals in curve.items():
        assert vals.shape == (3,)
        assert np.all(np.isfinite(vals))
        assert np.all((vals >= 0.0) & (vals <= 1.0))


def test_default_widths_includes_baseline():
    # 床基準 n_taps=8 が必ず含まれていること (baseline 比較の前提)。
    assert 8 in DEFAULT_WIDTHS
    assert DEFAULT_WIDTHS[0] == 8
