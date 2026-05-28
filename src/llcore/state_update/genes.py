# SPDX-License-Identifier: Apache-2.0
"""Minimal state update gene — PoC 0a の核.

falsifiable 命題 (PoC 0a):
    **decay/mix/gate_str の 3 パラメータで RNN-like state update を表現でき、
    入力長 L=256, dim=8 の有界入力に対し state が NaN/Inf にならず、
    state_norm が ``K * input_norm`` で抑えられる (K は decay/gate_str の関数)。**

破綻ゲート (PoC 0a 命題が成立しない条件):
- state に NaN/Inf が混入
- state_norm が monotonic に発散 (有界性違反)
- 同じ gene/入力で run 2 回の結果が一致しない (決定論性違反)

設計判断:
- numpy のみ依存 (PoC 0a レベルでは llive lldarwin_v2 を呼ばない = 分離テスト容易)
- 数値は float64 (Stage 5 までは精度より honest を優先)
- gate 関数は ``tanh(gate_str * s)`` (有界 [-1, 1] = 数値安定性の構造保証)
- decay は ``clip(decay, 0, 1)`` で物理的に意味ある範囲に制限

PoC 0c (進化 10×10) で初めて llive lldarwin_v2 を import 依存とする予定。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateUpdateGene:
    """3 パラメータ minimal state update kernel.

    Attributes
    ----------
    decay : float
        state 減衰率 (clipped to [0, 1] in :func:`eval_step`).
    mix : float
        新規入力混合率 (clipped to [0, 1]).
    gate_str : float
        tanh gating 強度 (clipped to [0, 2.0]).
    """

    decay: float
    mix: float
    gate_str: float

    def as_array(self) -> np.ndarray:
        """numpy へ落とす (進化集団管理用)."""
        return np.array([self.decay, self.mix, self.gate_str], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> StateUpdateGene:
        if arr.shape != (3,):
            raise ValueError(f"expected shape (3,), got {arr.shape}")
        return cls(decay=float(arr[0]), mix=float(arr[1]), gate_str=float(arr[2]))

    def clipped(self) -> StateUpdateGene:
        """物理範囲にクリップした gene を返す (元の gene は不変)."""
        return StateUpdateGene(
            decay=float(np.clip(self.decay, 0.0, 1.0)),
            mix=float(np.clip(self.mix, 0.0, 1.0)),
            gate_str=float(np.clip(self.gate_str, 0.0, 2.0)),
        )


def eval_step(
    state: np.ndarray, x: np.ndarray, gene: StateUpdateGene
) -> np.ndarray:
    """1 step update: ``state' = decay * state + mix * x * tanh(gate_str * state)``.

    Parameters
    ----------
    state : np.ndarray
        前 step の state, shape (dim,).
    x : np.ndarray
        新規入力, shape (dim,).
    gene : StateUpdateGene
        update parameters (clip 済みでなくても内部で clip される).

    Returns
    -------
    new_state : np.ndarray
        次 step の state, shape (dim,).

    Notes
    -----
    - tanh は有界 [-1, 1] = 数値安定性を構造的に保証
    - decay/mix/gate_str は内部で clip
    - state 形状チェック実施 (PoC は早期 fail-loud)
    """
    if state.shape != x.shape:
        raise ValueError(f"state {state.shape} and x {x.shape} must match")
    g = gene.clipped()
    gated = np.tanh(g.gate_str * state)
    return g.decay * state + g.mix * x * gated


def run_sequence(
    inputs: np.ndarray, gene: StateUpdateGene, initial_state: np.ndarray | None = None
) -> np.ndarray:
    """L step の sequence を回し、state 軌跡を返す.

    Parameters
    ----------
    inputs : np.ndarray
        shape (L, dim) — L step の入力列.
    gene : StateUpdateGene
        update parameters.
    initial_state : np.ndarray | None
        shape (dim,) — None なら zero 初期化.

    Returns
    -------
    states : np.ndarray
        shape (L+1, dim) — initial を含む全 step の state.
    """
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be 2D (L, dim), got shape {inputs.shape}")
    L, dim = inputs.shape
    state = np.zeros(dim, dtype=np.float64) if initial_state is None else initial_state.copy()
    states = np.empty((L + 1, dim), dtype=np.float64)
    states[0] = state
    for t in range(L):
        state = eval_step(state, inputs[t], gene)
        states[t + 1] = state
    return states
