# SPDX-License-Identifier: Apache-2.0
"""freq_tracking — 周波数 regime を軸にした遅延エコー追従タスク族 (leak niche 候補).

E-A フェーズ (③検定の土俵) の候補タスク。FlipFlop が too-easy (全 regime R²≈0.95,
汎化ギャップ負) と判明したのを受け、**異なる時定数を要求する regime 構造**を持つ
タスクを設計する。

狙い (niche 構造の出どころ)::
    入力は regime ごとの周波数 f の正弦波 x[t] = sin(2π f t + φ) + 微小ノイズ。
    target は系列終端から D ステップ前の **クリーンな** 入力値 x_clean[T-1-D]。
    過去を保持しないと最終 state から答えられない = 記憶必須。

    低周波 f (ゆっくり波) は長時間スケール (小 leak, 長記憶) のタップが追従しやすく、
    高周波 f (速い波) は短時間スケール (大 leak) が有利 — つまり **regime ごとに
    最適な leak (時定数) が違う**。これが behavior descriptor の leak 軸上に niche を
    生み、③(選択圧/分離) が load-bearing になりうる土俵を作る。

honest 留保:
- 純正弦の D ステップ遅延は位相シフトした正弦であり、leaky-integrator state の線形
  readout で原理的にデコード可能。難易度は D (遅延量) とノイズと f で連続調整する。
  飽和/床なら正直にそう報告する (honest negative も価値)。
- in_dim は全 regime で同一 (=1)。TaskMixture / split_regimes の前提を満たす。
- 地形は手で作らない (位相 φ とノイズは draw ごとの乱数のみ)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FreqTrackingTask:
    """周波数 f の正弦波を入力し、終端から D ステップ前のクリーン値を答える遅延エコー.

    ダイナミクス契約 (記憶必須)::
        x_clean[t] = sin(2π f t + φ)             (φ は draw ごと一様乱数)
        inputs[t]  = x_clean[t] + N(0, noise_std)  (in_dim=1)
        target     = x_clean[T - 1 - delay]       (out_dim=1, クリーン値)

    target は終端より delay ステップ前の値なので、最終 state は過去 delay ステップ前の
    位相を保持していなければ答えられない (=記憶タスク)。クリーン値を target にするのは
    「観測ノイズを除いた真の遅延信号を復元できるか」を測るため (ノイズは入力側のみ)。

    Attributes
    ----------
    freq : float
        正弦波の周波数 f (cycles/step)。regime 軸。
    seq_len : int
        系列長。delay より十分長くする。
    delay : int
        終端から何ステップ前の値を答えるか (遅延量)。難易度の主ノブ。
    noise_std : float
        入力に加える観測ノイズの標準偏差。難易度の副ノブ。
    in_dim : int
        入力次元 (全 regime 同一でなければならない)。既定 1。
    out_dim : int
        出力次元。既定 1。
    """

    freq: float = 0.1
    seq_len: int = 40
    delay: int = 5
    noise_std: float = 0.05
    in_dim: int = 1
    out_dim: int = 1

    def __post_init__(self) -> None:
        if self.delay < 0:
            raise ValueError(f"delay must be >= 0, got {self.delay}")
        if self.delay >= self.seq_len:
            raise ValueError(
                f"delay={self.delay} must be < seq_len={self.seq_len}"
            )
        if self.freq <= 0:
            raise ValueError(f"freq must be > 0, got {self.freq}")
        if self.noise_std < 0:
            raise ValueError(f"noise_std must be >= 0, got {self.noise_std}")
        if self.in_dim != 1:
            raise ValueError(
                f"FreqTrackingTask requires in_dim=1 (single sine channel), "
                f"got {self.in_dim}"
            )

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """周波数 f の正弦波 (位相ランダム) を 1 本生成し、遅延エコー target を返す.

        Returns
        -------
        inputs : np.ndarray
            shape (seq_len, 1)、観測ノイズ込みの正弦波。
        target : np.ndarray
            shape (1,)、終端から delay ステップ前のクリーン正弦値。
        """
        t = np.arange(self.seq_len, dtype=np.float64)
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        clean = np.sin(2.0 * np.pi * self.freq * t + phase)
        noise = rng.normal(0.0, self.noise_std, size=self.seq_len)
        inputs = (clean + noise).reshape(self.seq_len, 1)
        target = np.array([clean[self.seq_len - 1 - self.delay]], dtype=np.float64)
        return inputs.astype(np.float64), target


def make_regimes(
    freqs: tuple[float, ...] = (0.05, 0.1, 0.2, 0.4),
    *,
    seq_len: int = 40,
    delay: int = 5,
    noise_std: float = 0.05,
) -> list[FreqTrackingTask]:
    """周波数 regime 群を作る (全て同一 in_dim/out_dim/seq_len/delay).

    regime 軸は周波数 f のみ。delay / noise_std / seq_len は共通に固定し、
    「時定数の最適点だけが f で変わる」状況を作る (niche を leak 軸に集中させる)。
    """
    return [
        FreqTrackingTask(
            freq=f, seq_len=seq_len, delay=delay, noise_std=noise_std
        )
        for f in freqs
    ]


__all__ = ["FreqTrackingTask", "make_regimes"]
