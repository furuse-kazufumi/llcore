# SPDX-License-Identifier: Apache-2.0
"""temporal_xor2_delay — 2 ビット XOR を遅延 D 越しに保持する medium 難タスク族.

E-A フェーズ (E_A_DESIGN_multitask_generalization.md) の③検定候補。狙いは parity の
degree 床に縛られず、**2 ビットを遅延越しに保持して degree-2 readout で解く**土俵。

タスク仕様:
- t=0, t=1 に ±1 のビット b0, b1 を 1 チャネル (in_dim=1) に直列投入する。
- t=2..(1+D) は無情報 (0) の遅延区間。
- target = b0 * b1 (= ±1 XOR の積表現) を **最終時刻** に出す。
- seq_len = 2 + D。過去 (b0, b1) を遅延越しに保持しないと最終 state から解けない = 記憶必須。

degree-2 性 (positive control の理屈):
- 基質の readout は最終 state の **線形** ridge。b0*b1 は線形項では復元できない (degree-2)。
- 復元は reservoir の ``tanh`` 非線形が積項を state に焼くことに依存する
  (tanh(w·b0 + w·b1 + h) の Taylor 展開に b0*b1 の交差項が現れる)。
- よって「線形 readout のみでは床、reservoir tanh があれば解ける」= positive control 型。
  解けるかは leak/w_in の配分 (gene) 次第 → 進化の探索余地がある。

regime 軸 = 遅延 D ∈ {10, 20, 30, 40}。D が大きいほど 2 ビットを長く保持する必要があり
難化する。全 regime で in_dim/out_dim は同一 (= 1/1) なので 1 つの reservoir / TaskMixture に
入れられる。hold-out = 未学習 D での外挿汎化。

honest: 地形は手で作らない (b0,b1 は一様 ±1、target は決定論的積)。reservoir 改変なし。
``generate(rng) -> (inputs (seq_len, 1), target (1,))`` 契約厳守。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TemporalXor2DelayTask:
    """2 ビット (t=0,1) の XOR (積) を遅延 D 後の最終時刻に答える記憶タスク.

    Attributes
    ----------
    delay : int
        ビット投入後の無情報遅延ステップ数 D。seq_len = 2 + delay。
        D が大きいほど 2 ビット保持が長く必要 = 難化。regime 軸。
    in_dim : int
        入力次元。全 regime で 1 固定 (ビットを直列投入)。
    out_dim : int
        出力次元。1 (XOR 積 ±1)。
    """

    delay: int = 10
    in_dim: int = 1
    out_dim: int = 1

    def __post_init__(self) -> None:
        if self.delay < 0:
            raise ValueError(f"delay must be >= 0, got {self.delay}")
        if self.in_dim != 1:
            raise ValueError(f"in_dim must be 1 (serial bits), got {self.in_dim}")
        if self.out_dim != 1:
            raise ValueError(f"out_dim must be 1, got {self.out_dim}")

    @property
    def seq_len(self) -> int:
        """系列長 = 2 (ビット投入) + delay (無情報遅延)."""
        return 2 + self.delay

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """1 本の系列を生成する.

        Returns
        -------
        inputs : np.ndarray
            shape (seq_len, 1)。inputs[0]=b0, inputs[1]=b1, 以降 0。
        target : np.ndarray
            shape (1,)。b0 * b1 (= ±1)。
        """
        b0 = float(rng.choice([-1.0, 1.0]))
        b1 = float(rng.choice([-1.0, 1.0]))
        inputs = np.zeros((self.seq_len, 1), dtype=np.float64)
        inputs[0, 0] = b0
        inputs[1, 0] = b1
        target = np.array([b0 * b1], dtype=np.float64)
        return inputs, target


__all__ = ["TemporalXor2DelayTask"]
