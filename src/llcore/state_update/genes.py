# SPDX-License-Identifier: Apache-2.0
"""Minimal state update gene — PoC 0a v2 の核 (RWKV-style leak integrator).

履歴:
- v1 (2026-05-29 初版): ``decay*s + mix*x*tanh(gate_str*s)``
  → state=0 が fixed point の zero attractor で degenerate。
  G1-G5 形式 PASS だが情報伝達ゼロ。
- v2 (2026-05-29 同日): 2 reviewer (gem-critic + gpt-5.4 codex) 独立 verdict で
  「設計問題」「RWKV-style 推奨」を確定し、本式に差替え。
  詳細: docs/poc/poc_0a_verdict.md の「reviewer verdict + v1→v2 redesign」。

falsifiable 命題 (PoC 0a v2):
    **decay/mix/gate_str の 3 パラメータで RNN-like leak integrator + recurrent
    nonlinear coupling を表現でき、入力長 L=256, dim=8 の有界入力に対し
    (a) state が NaN/Inf にならず、
    (b) state_norm が ``K * input_norm`` 以下で抑えられ (K=10 緩い上界),
    (c) 非ゼロ入力で state が非自明 (variance > 0) に動き、
    (d) 異なる入力列で state 軌跡が区別できる。**

更新式 (RWKV-style leak integrator, gpt-5.4 reviewer 推奨):

    state[t+1] = decay * state[t]
               + (1 - decay) * tanh(mix * x[t] + gate_str * state[t])

役割分離:
- ``decay``    — memory timescale (1=完全記憶 / 0=完全忘却)
- ``mix``      — input gain (新規入力の影響強度)
- ``gate_str`` — recurrent contribution (state 自己フィードバック強度)

設計判断:
- numpy のみ依存 (PoC 0a レベルでは llive lldarwin_v2 を呼ばない = 分離テスト容易)
- 数値は float64 (Stage 5 までは精度より honest を優先)
- tanh が更新項全体を抑えるため数値安定性が構造的に保証
- convex combination ``decay*s + (1-decay)*phi`` で有界性を自動確保
- state=0, x!=0 でも ``tanh(mix*x) != 0`` で動き始める → zero attractor 回避
- clip 範囲拡張 (reviewer 指摘):
    * ``decay`` ∈ [0, 1] (これは memory timescale なので非負限定が物理的に妥当)
    * ``mix`` ∈ [-1, 1] (負入力 / 反転 / over-relaxation を許容)
    * ``gate_str`` ∈ [-2, 2] (抑制性 recurrent も探索空間に含める)

PoC 0c (進化 10×10) で自前 minimal GA を実装し、llive lldarwin_v2 は比較のみ。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateUpdateGene:
    """3 パラメータ minimal state update kernel (v2 RWKV-style 範囲).

    Attributes
    ----------
    decay : float
        memory timescale (clipped to [0, 1] in :meth:`clipped`).
        1=完全記憶 / 0=完全忘却.
    mix : float
        input gain (clipped to **[-1, 1]** in v2).
        v1 では [0, 1] だったが、reviewer 指摘で負入力 / 反転 / over-relaxation を
        探索空間に含めるため拡張。
    gate_str : float
        recurrent contribution (clipped to **[-2, 2]** in v2).
        v1 では [0, 2] だったが、reviewer 指摘で抑制性 recurrent 結合も探索空間に
        含めるため拡張 (生物学的にも一般的)。
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
        """物理範囲にクリップした gene を返す (元の gene は不変).

        v2 範囲 (2 reviewer 指摘で拡張):
        - decay    ∈ [0, 1]    (memory timescale, 非負限定が物理的に妥当)
        - mix      ∈ [-1, 1]   (負入力・反転・over-relaxation 許容)
        - gate_str ∈ [-2, 2]   (抑制性 recurrent も探索空間に)
        """
        return StateUpdateGene(
            decay=float(np.clip(self.decay, 0.0, 1.0)),
            mix=float(np.clip(self.mix, -1.0, 1.0)),
            gate_str=float(np.clip(self.gate_str, -2.0, 2.0)),
        )


def eval_step(
    state: np.ndarray, x: np.ndarray, gene: StateUpdateGene
) -> np.ndarray:
    """1 step update (RWKV-style leak integrator):

        ``state' = decay * state + (1 - decay) * tanh(mix * x + gate_str * state)``

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
    - tanh 内側で ``mix * x + gate_str * state`` を足し、state=0/x!=0 でも非ゼロ
      (zero attractor 回避; v1 の degenerate を v2 で構造的に解消)
    - ``decay * s + (1 - decay) * phi`` は convex combination で有界性自動確保
    - decay/mix/gate_str は内部で clip
    - state 形状チェック実施 (PoC は早期 fail-loud)
    """
    if state.shape != x.shape:
        raise ValueError(f"state {state.shape} and x {x.shape} must match")
    g = gene.clipped()
    preactivation = g.mix * x + g.gate_str * state
    update = np.tanh(preactivation)
    return g.decay * state + (1.0 - g.decay) * update


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
