# SPDX-License-Identifier: Apache-2.0
"""ea_lab の MAP-E / randselect ablation / 汎化 runner の機構テスト.

合成 eval_once (reservoir 不使用) で高速に機構のみ検証する。優位性の主張は exp スクリプト側。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ea_lab import (  # noqa: E402
    EAMethodScores,
    map_elites_full,
    map_elites_randselect,
    run_ea_methods_over_seeds,
)

DIM = 4
BOUNDS = (np.full(DIM, -1.0), np.full(DIM, 1.0))
BEHAVIOR_BOUNDS = (np.full(2, -1.0), np.full(2, 1.0))
GRID = (5, 5)


def _behavior(gene: np.ndarray) -> np.ndarray:
    """behavior = 先頭 2 次元 (連続)."""
    return gene[:2].copy()


def _eval_train(gene: np.ndarray, rng: np.random.Generator) -> float:
    """合成 fitness: 原点に近いほど高い (-||g||^2 を [0,1] 風に)。微小ノイズ付き."""
    base = 1.0 - float(np.mean(gene ** 2))
    return base + 0.01 * float(rng.standard_normal())


def _eval_test(gene: np.ndarray, rng: np.random.Generator) -> float:
    """test regime = train と同型だがオフセット (汎化測定用の別 draw)。"""
    base = 1.0 - float(np.mean((gene - 0.1) ** 2))
    return base + 0.01 * float(rng.standard_normal())


def _run_me(fn, n_evals=120, seed=0):
    return fn(
        _eval_train, _behavior, dim=DIM, bounds=BOUNDS, behavior_bounds=BEHAVIOR_BOUNDS,
        grid_shape=GRID, n_evals=n_evals, init_batch=20, sigma=0.2,
        rng=np.random.default_rng(seed),
    )


def test_map_elites_full_respects_budget() -> None:
    r = _run_me(map_elites_full)
    assert r.n_evals == 120
    assert r.best_gene.shape == (DIM,)
    assert np.all(np.isfinite(r.best_gene))
    assert 1 <= r.n_filled_cells <= GRID[0] * GRID[1]


def test_map_elites_randselect_respects_budget() -> None:
    r = _run_me(map_elites_randselect)
    assert r.n_evals == 120
    assert r.best_gene.shape == (DIM,)
    assert np.all(np.isfinite(r.best_gene))


def test_full_beats_randselect_on_structured_landscape() -> None:
    """構造のある landscape では full (選択あり) の best fitness >= randselect (選択なし).

    機構の sanity: fitness ゲート (③) を持つ full は、無条件上書きの randselect より
    best が悪くならない (複数 seed の中央値で比較、ノイズ吸収)。"""
    full = np.array([_run_me(map_elites_full, seed=s).best_fitness for s in range(7)])
    rand = np.array([_run_me(map_elites_randselect, seed=s).best_fitness for s in range(7)])
    assert np.median(full) >= np.median(rand)


def test_randselect_has_no_fitness_gate_even_in_init_batch() -> None:
    """回帰 (Codex F-High): randselect は init batch でも fitness ゲートを持たない.

    単一 cell (grid=(1,1)) かつ higher-gene=higher-fitness の決定論 landscape で、
    全 eval を init_batch に含めて実行。**archive 占有者**を probe で直接検証する: elite は
    cell に **max** gene を残す (③ ratchet) が、random は fitness ゲート無しで **最後に置いた
    gene** を残す。よって elite 占有者 fitness >= random 占有者 fitness が常に成立し、両者は
    一般に一致しない (= ③ が randselect の archive 動態から完全に除去されている)。

    注: best_fitness は Codex F2 修正で **global best-of-budget** に変えたため両 mode で一致する
    (= 読み出しの公平化)。③除去の witness は **archive 占有者** に移した。"""
    dim = 1
    bounds = (np.full(dim, 0.0), np.full(dim, 1.0))
    bbounds = (np.full(2, 0.0), np.full(2, 1.0))
    CELL = (0, 0)

    def behave(g):  # 全 gene を 1 cell に写像
        return np.array([0.0, 0.0])

    def ev(g, rng):  # ノイズ無し: fitness = gene 値 (単調)
        return float(g[0])

    kw = dict(dim=dim, bounds=bounds, behavior_bounds=bbounds, grid_shape=(1, 1),
              n_evals=12, init_batch=12, sigma=0.3)

    def _occupant(mode, seed):
        from ea_lab import _map_elites_core
        arch: dict = {}
        _map_elites_core(ev, behave, **kw, rng=np.random.default_rng(seed),
                         selection_mode=mode, archive_out=arch)
        return arch[CELL][1]  # 占有者 fitness

    # elite 占有者 = max gene ≥ random 占有者 (= 最後に置いた gene)
    assert _occupant("elite", 0) >= _occupant("random", 0)
    # global-best 読み出しは両 mode で一致する (公平化の確認)
    elite = _map_elites_core_elite(ev, behave, **kw)
    rand = _map_elites_core_rand(ev, behave, **kw)
    assert elite.best_fitness == pytest.approx(rand.best_fitness)
    # archive 占有者は一般に一致しない (max != last): 複数 seed で少なくとも 1 回は厳密に上回る
    strictly = [_occupant("elite", s) > _occupant("random", s) for s in range(5)]
    assert any(strictly)


def _map_elites_core_elite(ev, behave, *, _seed=0, **kw):
    from ea_lab import _map_elites_core
    return _map_elites_core(ev, behave, **kw, rng=np.random.default_rng(_seed),
                            selection_mode="elite")


def _map_elites_core_rand(ev, behave, *, _seed=0, **kw):
    from ea_lab import _map_elites_core
    return _map_elites_core(ev, behave, **kw, rng=np.random.default_rng(_seed),
                            selection_mode="random")


def test_invalid_selection_mode_raises() -> None:
    from ea_lab import _map_elites_core
    with pytest.raises(ValueError, match="selection_mode"):
        _map_elites_core(
            _eval_train, _behavior, dim=DIM, bounds=BOUNDS, behavior_bounds=BEHAVIOR_BOUNDS,
            grid_shape=GRID, n_evals=40, init_batch=10, sigma=0.2,
            rng=np.random.default_rng(0), selection_mode="bogus",
        )


def test_run_ea_methods_shapes_and_finite() -> None:
    out = run_ea_methods_over_seeds(
        _eval_train, _eval_test, _behavior, dim=DIM, bounds=BOUNDS,
        behavior_bounds=BEHAVIOR_BOUNDS, grid_shape=GRID, n_evals=80,
        n_seeds=3, honest_n_trials=4, sigma=0.2,
    )
    assert set(out.keys()) == {"map_elites", "map_elites_randselect", "panmictic_ga", "random"}
    for m, sc in out.items():
        assert isinstance(sc, EAMethodScores)
        assert sc.test.shape == (3,) and sc.train.shape == (3,)
        assert np.all(np.isfinite(sc.test)) and np.all(np.isfinite(sc.train))


def test_run_ea_methods_deterministic() -> None:
    kw = dict(dim=DIM, bounds=BOUNDS, behavior_bounds=BEHAVIOR_BOUNDS, grid_shape=GRID,
              n_evals=80, n_seeds=2, honest_n_trials=4, sigma=0.2, base_seed=123)
    a = run_ea_methods_over_seeds(_eval_train, _eval_test, _behavior, **kw)
    b = run_ea_methods_over_seeds(_eval_train, _eval_test, _behavior, **kw)
    for m in a:
        assert np.array_equal(a[m].test, b[m].test)
        assert np.array_equal(a[m].train, b[m].train)
