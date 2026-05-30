# SPDX-License-Identifier: Apache-2.0
"""記憶タスク3種 — 長期依存×非線形で本来的に欺瞞的になりうる標準難タスク.

いずれも task.generate(rng) -> (inputs, target):
- inputs: shape (seq_len, in_dim) の系列
- target: 最終時刻に出すべき答え (1D ndarray)
過去を保持しないと最終 state から解けない = 記憶必須。地形は手で作らない (人工注入なし)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DelayedParityTask:
    """系列先頭 window 個の ±1 のパリティ (XOR) を最終時刻に答える."""
    seq_len: int = 20
    window: int = 5
    in_dim: int = 1
    out_dim: int = 1

    def __post_init__(self) -> None:
        if self.window > self.seq_len:
            raise ValueError(f"window={self.window} exceeds seq_len={self.seq_len}")

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        bits = rng.choice([-1.0, 1.0], size=self.seq_len)
        inputs = bits.reshape(self.seq_len, 1)
        n_neg = int(np.sum(bits[: self.window] < 0))
        target = 1.0 if n_neg % 2 == 0 else -1.0
        return inputs.astype(np.float64), np.array([target], dtype=np.float64)


@dataclass(frozen=True)
class FlipFlopTask:
    """ch0=set(+1)/ch1=reset(+1) のパルス列。最後に set/reset された値 ±1 を保持して答える."""
    seq_len: int = 30
    in_dim: int = 2
    out_dim: int = 1
    pulse_prob: float = 0.2

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.zeros((self.seq_len, 2), dtype=np.float64)
        state = 0.0
        first = int(rng.integers(0, max(1, self.seq_len // 4)))
        for t in range(self.seq_len):
            if t == first:
                ch = int(rng.integers(0, 2))
                inputs[t, ch] = 1.0
            elif rng.random() < self.pulse_prob:
                ch = int(rng.integers(0, 2))
                inputs[t, ch] = 1.0
            if inputs[t, 0] > 0:
                state = 1.0
            elif inputs[t, 1] > 0:
                state = -1.0
        return inputs, np.array([state], dtype=np.float64)


@dataclass(frozen=True)
class DelayedRecallTask:
    """t=0 の cue (±1) を、無情報な遅延区間の後、最終時刻に思い出して答える (T-maze 風)."""
    seq_len: int = 25
    in_dim: int = 1
    out_dim: int = 1

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.zeros((self.seq_len, 1), dtype=np.float64)
        cue = rng.choice([-1.0, 1.0])
        inputs[0, 0] = cue
        return inputs, np.array([float(cue)], dtype=np.float64)
