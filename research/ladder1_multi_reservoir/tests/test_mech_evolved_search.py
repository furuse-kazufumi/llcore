# SPDX-License-Identifier: Apache-2.0
"""mech_evolved_search 基質 sanity テスト.

検証対象 (機構コードの基質的健全性のみ。研究の主張 = 床外しは exp スクリプト側で測る):
- EvolvedSearchConfig のバリデーション / budget 計算
- evolve_search 出力の finite / 値域 / shape / gene_dim 整合
- best_gene が gene_bounds 内 (clip が効いている)
- honest 再評価が進化と独立な seed を使う (artifact 排除契約)
- n_evals が config.budget と一致 (探索予算が宣言通り)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent))  # mech_evolved_search, multi_reservoir
sys.path.insert(0, str(_HERE.parents[1] / "step_c_memory_tasks"))  # memory_tasks

from mech_evolved_search import (  # noqa: E402
    EvolvedSearchConfig,
    EvolvedSearchResult,
    evolve_search,
    make_deep_eval,
)
from multi_reservoir import DeepReservoir, gene_bounds  # noqa: E402
from memory_tasks import DelayedParityTask  # noqa: E402


# --- config バリデーション ---


def test_config_defaults_match_spec():
    """タスク指定の既定 ((μ+λ)-ES: pop 30, gen 40, k=3, elitism 1, σ=0.15) を満たす."""
    cfg = EvolvedSearchConfig()
    assert cfg.pop_size == 30
    assert cfg.n_generations == 40
    assert cfg.tournament_k == 3
    assert cfg.elitism == 1
    assert cfg.mutation_sigma == pytest.approx(0.15)


def test_config_budget_formula():
    # budget = pop + gen * (pop - elitism)
    cfg = EvolvedSearchConfig(pop_size=30, n_generations=40, elitism=1)
    assert cfg.budget == 30 + 40 * (30 - 1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pop_size": 0},
        {"n_generations": -1},
        {"elitism": 31},  # > pop_size
        {"tournament_k": 0},
        {"tournament_k": 31},  # > pop_size
        {"honest_n_trials": 0},
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        EvolvedSearchConfig(**kwargs)


# --- evolve_search 基質 sanity (小予算で高速に) ---


def _small_setup():
    res = DeepReservoir(layer_taps=(8, 8, 8), in_dim=1)
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    eval_once = make_deep_eval(res, task, n_train=24, n_eval=24)
    cfg = EvolvedSearchConfig(
        pop_size=6, n_generations=3, tournament_k=2, elitism=1,
        mutation_sigma=0.15, honest_n_trials=4,
    )
    return res, eval_once, cfg


def test_result_type_and_finite():
    res, eval_once, cfg = _small_setup()
    out = evolve_search(eval_once, res, config=cfg, seed=0)
    assert isinstance(out, EvolvedSearchResult)
    assert np.isfinite(out.honest_max_r2)
    assert 0.0 <= out.honest_max_r2 <= 1.0  # make_eval_once が R² を [0,1] にクリップ
    assert all(np.isfinite(v) for v in out.best_fitness_curve)


def test_best_gene_shape_and_gene_dim():
    res, eval_once, cfg = _small_setup()
    out = evolve_search(eval_once, res, config=cfg, seed=1)
    assert out.best_gene.shape == (res.gene_dim,)
    assert np.all(np.isfinite(out.best_gene))


def test_best_gene_within_bounds():
    """gaussian mutation 後も clip が効いて gene_bounds 内に収まる (数値安定)."""
    res, eval_once, cfg = _small_setup()
    lo, hi = gene_bounds(res)
    out = evolve_search(eval_once, res, config=cfg, seed=2)
    assert np.all(out.best_gene >= lo - 1e-9)
    assert np.all(out.best_gene <= hi + 1e-9)


def test_curve_length_matches_generations():
    """best_fitness_curve は初期世代 + n_generations の長さ."""
    res, eval_once, cfg = _small_setup()
    out = evolve_search(eval_once, res, config=cfg, seed=3)
    assert len(out.best_fitness_curve) == cfg.n_generations + 1


def test_n_evals_matches_budget():
    """消費した fitness 評価回数が config.budget と一致 (宣言通りの探索予算)."""
    res, eval_once, cfg = _small_setup()
    out = evolve_search(eval_once, res, config=cfg, seed=4)
    assert out.n_evals == cfg.budget


def test_elitism_monotone_best_curve():
    """elitism>=1 なら進化中 best fitness は単調非減少 (凍結持越しの構造的保証).

    NOTE: これは進化中 (artifact 込み) 曲線の性質。honest_max_r2 とは別物
    (honest 再評価は fresh seed なので curve の最終値より下がりうる)。
    """
    res, eval_once, cfg = _small_setup()
    out = evolve_search(eval_once, res, config=cfg, seed=5)
    curve = np.array(out.best_fitness_curve)
    assert np.all(np.diff(curve) >= -1e-12)


def test_determinism_same_seed():
    """同 seed なら honest_max_r2 と best_gene が再現する (rng 受け渡しの決定論)."""
    res, eval_once, cfg = _small_setup()
    a = evolve_search(eval_once, res, config=cfg, seed=7)
    b = evolve_search(eval_once, res, config=cfg, seed=7)
    assert a.honest_max_r2 == pytest.approx(b.honest_max_r2)
    assert np.allclose(a.best_gene, b.best_gene)
