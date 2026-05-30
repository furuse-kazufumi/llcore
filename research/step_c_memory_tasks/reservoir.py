# SPDX-License-Identifier: Apache-2.0
"""leaky-delay-line reservoir 基質 — 連続 gene でダイナミクスを進化させる.

固定 ESN と違い tap ごとの leak rate / 入力重みを gene で決める。異なる leak が
異なる時間スケールの記憶を担い、長期依存タスクで「正しい時定数配分」を要求する。
fitness は per-gene ridge readout (src の fit_ridge_readout 流用) の held-out R²。
research/ 隔離。src は read-only 流用のみ (非変更)。

NOTE: プランのコードは ``parents[3] / "src"`` と書いてあるが、実際のディレクトリ構造では
  reservoir.py の parents[2] が ``D:/projects/llcore`` であり、
  parents[3] は ``D:/projects`` になる。よって ``parents[2] / "src"`` が正しいパス。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# src/llcore への正しい相対パス (reservoir.py → step_c_memory_tasks → research → llcore/src)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数値安定な sigmoid。大きな正値でクリップして exp overflow を防ぐ。"""
    # np.clip で [-500, 500] に収めると exp(-500)≈0、exp(500)≈inf を防ぐ。
    # また float32 でも安全だが、gene は float64 前提。
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


@dataclass(frozen=True)
class LeakyDelayLineReservoir:
    """leaky-integrator bank — tap ごとに固有の leak rate と入力重みを gene で決める.

    ダイナミクス::
        h_i[t] = (1 - a_i) * h_i[t-1] + a_i * tanh(w_i · x[t] + h_i[t-1])

    a_i が小さい (≈0) ほど前状態を強く保持 → 長時間スケール記憶。
    a_i が大きい (≈1) ほど現入力に敏感 → 短時間スケール。
    gene の各 a_i が異なる時定数を担うことで「記憶バンク」を実現する。

    Attributes
    ----------
    n_taps : int
        隠れユニット数 (= reservoir のサイズ)。
    in_dim : int
        入力次元。
    """

    n_taps: int = 8
    in_dim: int = 1

    @property
    def gene_dim(self) -> int:
        """gene ベクトルの次元 = n_taps (leak_raw) + n_taps * in_dim (w_in)."""
        return self.n_taps + self.n_taps * self.in_dim

    def _unpack(self, gene: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """gene を (leak, w_in) に展開する.

        Returns
        -------
        leak : np.ndarray
            shape (n_taps,)、値域 (0, 1)。sigmoid 変換後。
        w_in : np.ndarray
            shape (n_taps, in_dim)。生の入力重み。
        """
        leak = _sigmoid(gene[: self.n_taps])
        w_in = gene[self.n_taps :].reshape(self.n_taps, self.in_dim)
        return leak, w_in

    def run(self, gene: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """gene が定めるダイナミクスで入力系列を流し、各時刻の隠れ状態を返す.

        Parameters
        ----------
        gene : np.ndarray
            shape (gene_dim,)。
        inputs : np.ndarray
            shape (T, in_dim)。

        Returns
        -------
        states : np.ndarray
            shape (T, n_taps)。全時刻の隠れ状態。finite 保証 (tanh で飽和)。
        """
        gene = np.asarray(gene, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        leak, w_in = self._unpack(gene)

        T = inputs.shape[0]
        h = np.zeros(self.n_taps, dtype=np.float64)
        states = np.empty((T, self.n_taps), dtype=np.float64)

        for t in range(T):
            # drive: (n_taps,) — 各 tap の入力刺激 + 前状態フィードバック
            drive = w_in @ inputs[t]  # (n_taps,)
            # leaky integrator: 前状態を (1-a) で保持、新状態を a で混合
            h = (1.0 - leak) * h + leak * np.tanh(drive + h)
            states[t] = h

        # tanh で飽和するため NaN/inf は原理的に発生しないが念のため assert は省き
        # テストで np.all(np.isfinite(states)) を確認する設計。
        return states

    def random_gene(self, rng: np.random.Generator) -> np.ndarray:
        """gene_bounds の範囲で一様乱数 gene を生成する."""
        lo, hi = gene_bounds(self)
        return lo + (hi - lo) * rng.random(self.gene_dim)


def gene_bounds(res: LeakyDelayLineReservoir) -> tuple[np.ndarray, np.ndarray]:
    """gene の探索範囲を返す.

    - leak_raw: [-4, 4] → sigmoid で (0.018, 0.982) に変換。
      -4 → leak≈0.018 (長時間スケール)、+4 → leak≈0.982 (短時間スケール)。
    - w_in: [-2, 2]。±2 の tanh 引数でほぼ飽和する程度。

    Returns
    -------
    lo, hi : np.ndarray
        それぞれ shape (gene_dim,)。
    """
    lo = np.concatenate([
        np.full(res.n_taps, -4.0),
        np.full(res.n_taps * res.in_dim, -2.0),
    ])
    hi = np.concatenate([
        np.full(res.n_taps, 4.0),
        np.full(res.n_taps * res.in_dim, 2.0),
    ])
    return lo, hi


def make_eval_once(
    res: LeakyDelayLineReservoir,
    task: object,
    *,
    n_train: int = 64,
    n_eval: int = 64,
    ridge_lambda: float = 1e-2,
):
    """ridge readout による held-out R² 評価コールバックを作る.

    train/eval を分離して引くため readout の leakage (暗記) は構造的に起こらない。
    fitness = clip([0, 1]) の held-out R²。

    Returns
    -------
    eval_once : Callable[[np.ndarray, np.random.Generator], float]
        ``(gene, rng) -> float`` の評価関数。
    """
    def _collect(gene: np.ndarray, n: int, rng: np.random.Generator):
        """n 本の sequence を生成し (final_state, target) を収集する."""
        states: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for _ in range(n):
            inputs, target = task.generate(rng)
            states.append(res.run(gene, inputs)[-1])  # 最終時刻の状態のみ使用
            targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
        return np.array(states, dtype=np.float64), np.array(targets, dtype=np.float64)

    def eval_once(gene: np.ndarray, rng: np.random.Generator) -> float:
        # train で readout を fit
        s_tr, y_tr = _collect(gene, n_train, rng)
        readout = fit_ridge_readout(s_tr, y_tr, ridge_lambda=ridge_lambda)

        # held-out で R² を評価 (train と同じ rng を継続 → independent な系列)
        s_ev, y_ev = _collect(gene, n_eval, rng)
        pred = np.atleast_2d(readout(s_ev))  # (n_eval, out_dim)
        mse = float(np.mean((pred - y_ev) ** 2))
        var = float(np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
        r2 = 1.0 - mse / max(var, 1e-12)
        return float(np.clip(r2, 0.0, 1.0))

    return eval_once


def make_behavior(res: LeakyDelayLineReservoir):
    """behavior descriptor = (平均実効記憶長, leak の分散) を計算するコールバックを作る.

    - 実効記憶長: leak が小さいほど長い。1/leak の平均を tanh で [0,1] に圧縮。
    - leak の分散: 時定数の多様性。均一=0、多様=大。

    NOTE (Codex pair-review BLOCKER 2026-05-30): この descriptor は leak のみを使い
    ``w_in`` を捨てている。実際のダイナミクスは leak + w_in の両方に依存するため、
    MAP-Elites の niche 軸として使うと異なる記憶戦略が同セルに衝突しやすい。
    また leak の 1次・2次モーメントだけでは時定数分布の形状を弁別できない。
    仕様 (プラン Task 2) には「behavior descriptor = (平均実効記憶長, leak の分散)」と
    明記されているため現 task スコープでは仕様通りに実装し、niche 軸の改良は
    後続タスク (MAP-Elites 統合フェーズ) で行う。

    Returns
    -------
    behavior : Callable[[np.ndarray], np.ndarray]
        ``(gene,) -> np.ndarray shape (2,)`` の behavior descriptor。
    """
    def behavior(gene: np.ndarray) -> np.ndarray:
        leak = _sigmoid(gene[: res.n_taps])
        # 実効記憶長 = 1/leak の平均 (leak→0 で∞)、tanh で [0,1] に正規化。
        # 分母を max(leak, 1e-3) で保護して 0除算を防ぐ。
        eff_mem = np.mean(1.0 / np.maximum(leak, 1e-3))
        eff_mem_norm = float(np.tanh(eff_mem / 50.0))
        return np.array([eff_mem_norm, float(np.std(leak))], dtype=np.float64)

    return behavior


__all__ = [
    "LeakyDelayLineReservoir",
    "gene_bounds",
    "make_eval_once",
    "make_behavior",
]
