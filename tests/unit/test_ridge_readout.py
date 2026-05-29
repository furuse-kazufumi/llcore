# SPDX-License-Identifier: Apache-2.0
"""per-gene ridge readout (CPU 手順 2) の回帰テスト.

EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md §7b の CPU 手順 2 (fixed readout →
per-gene least-squares(ridge, held-out) 置換で landscape un-flatten)。

ゲート (falsifiable):
- **fit 正しさ**: 線形に生成した target を ridge が回復する。
- **leakage なし**: 入力と無相関な random target は held-out R²≈0 (暗記しない)。
- **un-flatten (mechanism)**: copy delay=0 で ridge は fixed readout より spread が広く、
  最良 gene が R²>0.9 に到達 (fixed は ~0.63 が上限) = readout が各 gene に適応する。
- **honest regime 発見 (反証ゲート)**: delay≥4 では leak integrator + 線形 readout が
  遅延入力を復元できず全 gene ~0 / addition も線形デコード不可で ~0
  = 「ridge un-flatten だけでは ③(選択)が立つ構造的-かつ-難しい landscape は作れない」を固定。
- **harness 結線**: make_ridge_eval_once が evolution_vs_random にそのまま渡せる。
"""
from __future__ import annotations

import numpy as np

from llcore.evolution.honest_eval import FalsificationResult, evolution_vs_random
from llcore.fitness import (
    AdditionTask,
    CopyTask,
    FixedReadout,
    fit_ridge_readout,
    make_fixed_readout,
    make_ridge_eval_once,
    ridge_fitness,
)
from llcore.fitness.ridge_readout import RidgeReadout
from llcore.fitness.tasks import calibrate_baseline, evaluate_gene
from llcore.state_update import StateUpdateGene

import dataclasses


def _random_genes(n: int, seed: int) -> list[StateUpdateGene]:
    rng = np.random.default_rng(seed)
    return [
        StateUpdateGene(
            float(rng.uniform(0, 1)), float(rng.uniform(-1, 1)), float(rng.uniform(-2, 2))
        )
        for _ in range(n)
    ]


# ---------------------------------------------------------------------------
# fit の正しさ
# ---------------------------------------------------------------------------


def test_fit_ridge_readout_recovers_linear_map() -> None:
    """線形に生成した target を ridge (small λ) が回復する."""
    rng = np.random.default_rng(0)
    n, d, p = 200, 5, 3
    states = rng.standard_normal((n, d))
    w_true = rng.standard_normal((d, p))
    b_true = rng.standard_normal(p)
    targets = states @ w_true + b_true
    readout = fit_ridge_readout(states, targets, ridge_lambda=1e-6)
    assert isinstance(readout, RidgeReadout)
    pred = readout(states)
    assert pred.shape == (n, p)
    assert np.mean((pred - targets) ** 2) < 1e-6  # ほぼ完全回復


def test_ridge_readout_call_single_and_batch() -> None:
    rng = np.random.default_rng(1)
    states = rng.standard_normal((50, 4))
    targets = rng.standard_normal((50, 2))
    readout = fit_ridge_readout(states, targets, ridge_lambda=1e-2)
    single = readout(states[0])
    assert single.shape == (2,)
    batch = readout(states)
    assert batch.shape == (50, 2)
    assert np.allclose(batch[0], single)


def test_fit_ridge_readout_invalid_shapes() -> None:
    import pytest

    with pytest.raises(ValueError):
        fit_ridge_readout(np.zeros((3,)), np.zeros((3, 1)))  # states 1D
    with pytest.raises(ValueError):
        fit_ridge_readout(np.zeros((3, 2)), np.zeros((4, 1)))  # N mismatch
    with pytest.raises(ValueError):
        fit_ridge_readout(np.zeros((3, 2)), np.zeros((3, 1)), ridge_lambda=-1.0)


# ---------------------------------------------------------------------------
# leakage なし (held-out)
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class _RandomTargetTask:
    """入力と無相関な random target を返す task (leakage 検出用)."""

    name: str = "random_target"
    seq_len: int = 12
    state_dim: int = 6
    out_dim: int = 3

    def generate(self, rng):
        inputs = rng.uniform(-1, 1, size=(self.seq_len, self.state_dim))
        target = rng.standard_normal(self.out_dim)  # 入力と無相関
        return inputs, target


def test_ridge_fitness_no_leakage_on_random_target() -> None:
    """入力と無相関な target は held-out R²≈0 (train で暗記しても held-out に出ない)."""
    task = _RandomTargetTask()
    gene = StateUpdateGene(0.5, 0.5, 0.5)
    f = ridge_fitness(gene, task, n_train=64, n_eval=64, rng=np.random.default_rng(3))
    assert 0.0 <= f < 0.15, f"無相関 target で fitness が高い (leakage 疑い): {f}"


# ---------------------------------------------------------------------------
# un-flatten (mechanism): ridge は fixed より spread が広く高 R² に到達
# ---------------------------------------------------------------------------


def test_ridge_unflattens_vs_fixed_readout_copy_delay0() -> None:
    """copy delay=0: ridge readout は fixed readout より gene 間 spread が広く、
    最良 gene が R²>0.9 に届く (fixed は ~0.63 上限)。"""
    copy = CopyTask(state_dim=6, out_dim=6, seq_len=16, delay=0)
    genes = _random_genes(24, seed=0)

    # fixed readout (baseline calibration 済) の score
    fr = make_fixed_readout(6, 6, seed=123)
    task_fixed = dataclasses.replace(copy, baseline_mse=calibrate_baseline(copy, fr))
    fixed = np.array(
        [evaluate_gene(g, task_fixed, fr, np.random.default_rng(7), n_trials=5) for g in genes]
    )

    # ridge readout の fitness
    ridge = np.array(
        [ridge_fitness(g, copy, n_train=48, n_eval=48, rng=np.random.default_rng(7)) for g in genes]
    )

    assert ridge.std() > fixed.std(), f"ridge spread {ridge.std():.3f} <= fixed {fixed.std():.3f}"
    assert ridge.max() > 0.9, f"ridge 最良 gene が高 R² に届かない: {ridge.max():.3f}"
    assert fixed.max() < ridge.max(), "fixed readout の上限が ridge を超えた (un-flatten 否定)"


# ---------------------------------------------------------------------------
# honest regime 発見 (反証ゲート): readout 修正だけでは構造的-難 landscape は作れない
# ---------------------------------------------------------------------------


def test_ridge_fitness_delayed_copy_no_selection_signal() -> None:
    """delay≥4 の copy: clip 後 fitness は全 gene 0 (GA に選択信号なし).

    honest 注 (Codex High finding 2026-05-30): clip=False の raw R² は **負** (mean 予測
    以下) であり「raw=0 の信号皆無」とは別物。'原理的に不能' でなく『この評価設定では
    線形 readout が有用信号を出さず、clip 後 fitness が平坦 = 選択に使えない』が正確。
    """
    copy = CopyTask(state_dim=6, out_dim=6, seq_len=16, delay=4)
    genes = _random_genes(16, seed=1)
    clipped = np.array(
        [ridge_fitness(g, copy, n_train=48, n_eval=48, rng=np.random.default_rng(7)) for g in genes]
    )
    raw = np.array(
        [ridge_fitness(g, copy, n_train=48, n_eval=48, rng=np.random.default_rng(7), clip=False)
         for g in genes]
    )
    assert clipped.max() == 0.0, f"clip 後 fitness が非ゼロ: max={clipped.max():.3f}"
    assert raw.max() < 0.05, f"raw R² に有用信号 (clip 検閲の可能性): max={raw.max():.3f}"


def test_ridge_fitness_addition_no_selection_signal() -> None:
    """addition (||sum x||): clip 後 fitness は全 gene ~0 (診断再現, 線形デコード弱).

    honest 注: raw R² は負 (mean 予測以下) で、clip により 0 化。'原理的に不能' でなく
    この評価設定での観測 (state は tanh 非線形の出力なので数学的不能の証明ではない)。
    """
    add = AdditionTask(state_dim=6, out_dim=1, seq_len=16)
    genes = _random_genes(16, seed=2)
    clipped = np.array(
        [ridge_fitness(g, add, n_train=48, n_eval=48, rng=np.random.default_rng(7)) for g in genes]
    )
    raw = np.array(
        [ridge_fitness(g, add, n_train=48, n_eval=48, rng=np.random.default_rng(7), clip=False)
         for g in genes]
    )
    assert clipped.max() < 0.15, f"addition が線形デコードできてしまった: max={clipped.max():.3f}"
    assert raw.max() < 0.1, f"raw R² に有用信号: max={raw.max():.3f}"


# ---------------------------------------------------------------------------
# harness 結線
# ---------------------------------------------------------------------------


def test_make_ridge_eval_once_integrates_with_harness() -> None:
    """make_ridge_eval_once が evolution_vs_random にそのまま渡せ、整合した結果を返す.

    注: copy delay=0 は ridge un-flatten 後『容易な単峰』になり GA≈random (passes は
    主張しない)。ここで固定するのは harness 結線と field 整合のみ (honest)。
    """
    copy = CopyTask(state_dim=6, out_dim=6, seq_len=16, delay=0)
    eval_once = make_ridge_eval_once(copy, n_train=24, n_eval=24)
    r = evolution_vs_random(
        eval_once, pop_size=6, n_generations=4, n_seeds=6, honest_n_trials=3, base_seed=42
    )
    assert isinstance(r, FalsificationResult)
    assert r.n_seeds == 6
    assert 0.0 <= r.ga_mean <= 1.0 and 0.0 <= r.random_mean <= 1.0
    assert abs(r.diff - (r.ga_mean - r.random_mean)) < 1e-12
    assert -1.0 <= r.cliff_delta <= 1.0
