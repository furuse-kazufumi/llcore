# SPDX-License-Identifier: Apache-2.0
"""honest_eval (公正な評価 + 反証ハーネス) の回帰テスト.

CPU 手順 1 (EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md §7b)。狙い:
- **機構健全 regression**: 構造のある landscape では進化が同予算 random を有意に上回る
  (診断 [GA健全性] の unimodal δ=+0.97 を回帰として固定)。
- **選択機構実在 regression**: 定数 fitness では進化の改善が消える。
- ハーネス自体 (honest_reevaluate / equal_budget / 統計量) の正しさ。
"""
from __future__ import annotations

import numpy as np

from llcore.evolution.honest_eval import (
    FalsificationResult,
    equal_budget,
    evolution_vs_random,
    honest_reevaluate,
)
from llcore.state_update import StateUpdateGene


# ---------------------------------------------------------------------------
# eval_once コールバック (task 非依存テスト用)
# ---------------------------------------------------------------------------


def _unimodal(gene, rng) -> float:
    """構造のある決定論的 landscape: target gene 周りの単峰 (rng 不使用)."""
    a = gene.as_array()
    target = np.array([0.6, 0.3, 0.4])
    return float(np.exp(-np.sum((a - target) ** 2)))


def _constant(gene, rng) -> float:
    """定数 fitness: 選択圧ゼロ (改善が消えるはず)."""
    return 0.5


# ---------------------------------------------------------------------------
# ハーネス単体
# ---------------------------------------------------------------------------


def test_equal_budget() -> None:
    assert equal_budget(10, 10, 1) == 100
    assert equal_budget(12, 15, 1) == 12 + 15 * 11


def test_honest_reevaluate_deterministic_is_rng_independent() -> None:
    g = StateUpdateGene(0.6, 0.3, 0.4)
    v1 = honest_reevaluate(_unimodal, g, n_trials=5, rng=np.random.default_rng(1))
    v2 = honest_reevaluate(_unimodal, g, n_trials=5, rng=np.random.default_rng(999))
    assert v1 == v2  # 決定論的 eval は rng/n_trials に依存しない
    assert abs(v1 - 1.0) < 1e-12  # target gene そのものは fitness=1.0


# ---------------------------------------------------------------------------
# 回帰: 機構は健全 (構造のある landscape で進化が同予算 random を有意に上回る)
# ---------------------------------------------------------------------------


def test_machinery_healthy_on_structured_landscape() -> None:
    """[機構健全 regression] 単峰 landscape で 進化 > 同予算 random (有意)."""
    r = evolution_vs_random(
        _unimodal, pop_size=10, n_generations=10, n_seeds=15,
        honest_n_trials=1, base_seed=20260530,
    )
    assert isinstance(r, FalsificationResult)
    assert r.diff > 0.0, f"進化が random を上回らない: diff={r.diff}"
    assert r.win_rate >= 0.8, f"勝率が低い: {r.win_rate}"
    assert r.passes, (
        f"機構健全の合格条件 (diff>0 & 片側 p<0.05 & n_seeds>=15 & |δ|>=0.147) 未達: "
        f"p={r.wilcoxon_p}, δ={r.paired_sign_delta}"
    )


# ---------------------------------------------------------------------------
# 回帰: 選択機構の実在 (定数 fitness で改善が消える)
# ---------------------------------------------------------------------------


def test_selection_exists_constant_fitness_no_improvement() -> None:
    """[選択機構実在 regression] 定数 fitness では 進化 と random が同値 (差ゼロ)."""
    r = evolution_vs_random(
        _constant, pop_size=10, n_generations=10, n_seeds=10,
        honest_n_trials=1, base_seed=7,
    )
    assert abs(r.diff) < 1e-9, f"定数 fitness で差が出た (選択圧の捏造疑い): diff={r.diff}"
    assert r.ga_mean == 0.5 and r.random_mean == 0.5
    assert not r.passes  # 定数では成立してはならない


# ---------------------------------------------------------------------------
# 現状の honest disclosure: ハーネスが falsify 判定を返せること自体の確認
# ---------------------------------------------------------------------------


def test_result_fields_consistent() -> None:
    r = evolution_vs_random(
        _unimodal, pop_size=8, n_generations=8, n_seeds=12,
        honest_n_trials=1, base_seed=42,
    )
    assert r.n_seeds == 12
    assert -1.0 <= r.cliff_delta <= 1.0
    assert 0.0 <= r.win_rate <= 1.0
    assert 0.0 <= r.wilcoxon_p <= 1.0
    assert abs(r.diff - (r.ga_mean - r.random_mean)) < 1e-12
