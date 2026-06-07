# SPDX-License-Identifier: Apache-2.0
"""Synthetic sequence tasks + fitness evaluation (PoC 0b の核).

設計:
- :class:`SyntheticTask` ABC: ``generate(rng)`` で (inputs, target) を返し、
  ``score(state_trajectory, target, readout)`` で fitness ∈ [0, 1] を返す。
- :class:`CopyTask`: 入力を delay 遅延で再現するタスク (RNN 標準 toy)。
- :class:`AdditionTask`: 入力 sequence の累積和の絶対値を予測。

Readout:
- :func:`make_fixed_readout` で seed 固定 random 線形写像を作る (gene 進化対象外)。
- copy: state → input dim 復元 (matrix W_copy)
- addition: state → scalar (matrix W_add)

honest 留保:
- readout 固定 = state update gene の表現力に注目した PoC、
  本来 readout も進化対象だが Stage 0b の主役は state update gene。
- fitness は MSE ベース ``1 - clip(MSE / baseline, 0, 1)``、
  baseline は random gene 集団の MSE 中央値 (task ごと固定 calibration)。
- Stage 0c (進化 10x10) で本 fitness を使う前提。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from llcore.state_update import StateUpdateGene, run_sequence


# ---------------------------------------------------------------------------
# Readout (gene 進化対象外、seed 固定)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FixedReadout:
    """seed 固定 random 線形写像.

    Attributes
    ----------
    matrix : np.ndarray
        shape (out_dim, state_dim). ``state @ matrix.T`` で readout を得る。
    seed : int
        生成 seed (再現性のため保持)。
    """

    matrix: np.ndarray
    seed: int

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """state を readout する: ``state @ matrix.T``."""
        return state @ self.matrix.T


def make_fixed_readout(state_dim: int, out_dim: int, *, seed: int) -> FixedReadout:
    """seed 固定の線形 readout を作る.

    standard normal × (1/sqrt(state_dim)) で正規化 (Xavier 風)。
    """
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal(size=(out_dim, state_dim)) / np.sqrt(state_dim)
    return FixedReadout(matrix=matrix, seed=seed)


# ---------------------------------------------------------------------------
# Task ABC
# ---------------------------------------------------------------------------


class SyntheticTask(Protocol):
    """合成 sequence task の interface.

    score / raw_error の役割分離 (Codex 指摘 2026-05-29 fix):
    - :meth:`raw_error` — 生 MSE 風 error (calibration / 単純比較で使う基本量)
    - :meth:`score`     — baseline 正規化済 fitness ∈ [0, 1] (進化の選択圧)

    両者は task ごとに**同じ error 定義**を共有する (addition の abs 不変等
    の task-specific error は raw_error 内で定義し、score もそれを使う)。
    """

    name: str
    seq_len: int
    state_dim: int
    out_dim: int

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """入力列 (L, state_dim) と target (..., out_dim) を返す."""

    def raw_error(
        self,
        state_trajectory: np.ndarray,
        target: np.ndarray,
        readout: FixedReadout,
    ) -> float:
        """task-specific raw MSE 風 error (calibration baseline と score が共有)."""

    def score(
        self,
        state_trajectory: np.ndarray,
        target: np.ndarray,
        readout: FixedReadout,
    ) -> float:
        """fitness ∈ [0, 1] を返す (1.0 = 完全再現)."""


# ---------------------------------------------------------------------------
# Copy Task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CopyTask:
    """入力 sequence を delay 遅延で再現するタスク (RNN 標準).

    入力 ``x[0..T-1]`` を読み込み、最終 state から readout で ``x[T - 1 - delay]``
    を復元する。delay=0 なら直近入力の再現、delay>0 で memory horizon を要求する。

    fitness = 1 - clip(MSE(pred, target) / baseline_mse, 0, 1).
    baseline_mse は random gene 集団 (N=20, seed 固定) の中央値で事前 calibration。

    honest 留保: これは "**fixed-readout probe-based fitness**" (Codex 2026-05-29
    指摘) — gene の state が seed-pinned linear probe で読み出し可能な信号を
    保持しているかの代理指標。gene 純粋 fitness ではない。
    """

    name: str = "copy"
    seq_len: int = 32
    state_dim: int = 8
    out_dim: int = 8  # 入力 dim と同じ
    delay: int = 0
    baseline_mse: float = 1.0  # calibrate_baseline で更新

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """入力列 (L, state_dim) を生成、target は x[L-1-delay]."""
        inputs = rng.uniform(-1, 1, size=(self.seq_len, self.state_dim))
        target_idx = self.seq_len - 1 - self.delay
        target = inputs[target_idx].copy()
        return inputs, target

    def raw_error(
        self,
        state_trajectory: np.ndarray,
        target: np.ndarray,
        readout: FixedReadout,
    ) -> float:
        """raw MSE (calibration / score 共有)."""
        final_state = state_trajectory[-1]
        pred = readout(final_state)
        return float(np.mean((pred - target) ** 2))

    def score(
        self,
        state_trajectory: np.ndarray,
        target: np.ndarray,
        readout: FixedReadout,
    ) -> float:
        """raw_error を baseline で正規化、fitness ∈ [0, 1] を返す."""
        mse = self.raw_error(state_trajectory, target, readout)
        return float(np.clip(1.0 - mse / max(self.baseline_mse, 1e-9), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Addition Task
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AdditionTask:
    """累積和の L2 norm regression task.

    入力 ``x[0..T-1]`` の cumsum の L2 norm (非負スカラ) を target に。
    readout は state → 1-dim、abs(pred) と target を比較 (sign 不変評価)。

    honest 留保 (Codex 2026-05-29 指摘): 実体は "addition" でなく ``||sum_t x_t||_2``
    の regression。命名は便宜上 addition だが claim wording は注意。
    """

    name: str = "addition"
    seq_len: int = 32
    state_dim: int = 8
    out_dim: int = 1
    baseline_mse: float = 1.0

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """入力列と target = ||sum(inputs)|| を生成."""
        inputs = rng.uniform(-1, 1, size=(self.seq_len, self.state_dim))
        cumsum = inputs.sum(axis=0)
        target = np.array([float(np.linalg.norm(cumsum))], dtype=np.float64)
        return inputs, target

    def raw_error(
        self,
        state_trajectory: np.ndarray,
        target: np.ndarray,
        readout: FixedReadout,
    ) -> float:
        """sign 不変 MSE: ``(|pred| - target)^2`` 平均.

        calibration と score が同じ error を共有 (Codex 指摘 fix)。
        """
        final_state = state_trajectory[-1]
        pred = readout(final_state)
        return float(np.mean((np.abs(pred) - target) ** 2))

    def score(
        self,
        state_trajectory: np.ndarray,
        target: np.ndarray,
        readout: FixedReadout,
    ) -> float:
        mse = self.raw_error(state_trajectory, target, readout)
        return float(np.clip(1.0 - mse / max(self.baseline_mse, 1e-9), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_gene(
    gene: StateUpdateGene,
    task: SyntheticTask,
    readout: FixedReadout,
    rng: np.random.Generator,
    *,
    n_trials: int = 5,
) -> float:
    """gene を task で n_trials 平均評価し fitness を返す.

    各 trial で task.generate(rng) → run_sequence(inputs, gene) → task.score。
    n_trials 平均で noise 低減。決定論性 (gene + task + readout + seed 固定)
    は外部呼出側で seed 管理する。
    """
    scores: list[float] = []
    for _ in range(n_trials):
        inputs, target = task.generate(rng)
        trajectory = run_sequence(inputs, gene)
        s = task.score(trajectory, target, readout)
        scores.append(s)
    return float(np.mean(scores))


def calibrate_baseline(
    task: SyntheticTask,
    readout: FixedReadout,
    *,
    n_individuals: int = 20,
    n_trials: int = 5,
    seed: int = 20260529,
) -> float:
    """random gene 集団 (N=20) で raw_error の中央値を計算し baseline として返す.

    重要 (Codex 2026-05-29 指摘 fix): task.raw_error を直接呼び、score と error
    定義を一致させる。v1 で AdditionTask の abs/non-abs 不整合があったが解消。

    score 関数の baseline_mse を更新する用途。task instance は frozen dataclass
    なので呼出側で dataclasses.replace を使う。
    """
    rng = np.random.default_rng(seed)
    errors: list[float] = []
    for _ in range(n_individuals):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        for _ in range(n_trials):
            inputs, target = task.generate(rng)
            trajectory = run_sequence(inputs, gene)
            errors.append(task.raw_error(trajectory, target, readout))
    return float(np.median(errors))


def calibrate_baseline_robust(
    task: SyntheticTask,
    readout: FixedReadout,
    *,
    seeds: tuple[int, ...] = (20260529, 20260530, 20260531),
    n_individuals: int = 20,
    n_trials: int = 5,
) -> tuple[float, float, float]:
    """seed sweep で baseline の頑健性を測る (Codex Q5 対応, 2026-05-29).

    3 seed の median を返し、最小・最大も併せて返す (variability 観測)。

    Returns
    -------
    median_of_medians : float
        seed 別 median の中央値 (採用 baseline_mse)
    min_median : float
        seed 別 median の最小
    max_median : float
        seed 別 median の最大
    """
    medians: list[float] = []
    for s in seeds:
        m = calibrate_baseline(
            task, readout,
            n_individuals=n_individuals,
            n_trials=n_trials,
            seed=s,
        )
        medians.append(m)
    return float(np.median(medians)), float(min(medians)), float(max(medians))
