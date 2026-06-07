# SPDX-License-Identifier: Apache-2.0
"""発散しうる基質 3 種 (R-endo viability PoC).

なぜなぜ分析の根本原因 = 「無条件有界な基質では証明対象 (収縮) と生存がデカップル」。
→ 生存を収縮に再結合する 3 つの実現法 (各々 ×環境依存の死):

- ``LinearSubstrate``  (#1 飽和除去): ``s' = a·s + g·x``, a=decay+(1−decay)·gate_str。
  |a|≥1 で state~a^t 幾何発散=死。収縮=生存に再結合。tube r=G·w̄/(1−|a|) は線形で **厳密**。
- ``SoftSatSubstrate`` (#2 soft 飽和/高天井): ``s' = decay·s + (1−decay)·K·tanh(pre/K)`` (大 K)。
  horizon 内は事実上線形 (発散可)・遠方のみ |s|≤K で有界 (NaN なし) = TRIZ 条件分離。
- ``HighGainSubstrate`` (#3 高ゲイン観測): 既存有界 tanh 力学のまま、出力を高ゲイン M で観測し
  小さな state 誤差が致命的出力誤差になる。**既存 tracking_tube の L/G を再利用** (production verifier 直結)。

共通契約 (duck typing): ``name`` / ``obs_gain`` 属性 + ``step(s,x,gene)`` / ``L(gene)`` / ``G(gene)``。
- ``L`` = 状態方向 Lipschitz 上界 (収縮判定 L<1)。
- ``G`` = 入力方向ゲイン (obs_gain 込み = 致命誤差スケール)。
- tube r = G·w̄/(1−L) (runner が計算)。死 = 実測誤差包絡 (obs_gain 込み) > 生存閾 V or 非有限。
すべて gene は clip 済み前提 (GA operator が clip する)。新規 production 依存なし (additive)。
"""
from __future__ import annotations

import numpy as np

from llcore.state_update import StateUpdateGene


class LinearSubstrate:
    """#1 飽和除去: 線形 leak integrator。|a|≥1 で幾何発散=死。"""

    name = "linear"
    obs_gain = 1.0

    @staticmethod
    def _a(g: StateUpdateGene) -> float:
        return g.decay + (1.0 - g.decay) * g.gate_str

    @staticmethod
    def _gin(g: StateUpdateGene) -> float:
        return (1.0 - g.decay) * g.mix

    def step(self, s: np.ndarray, x: np.ndarray, g: StateUpdateGene) -> np.ndarray:
        return self._a(g) * s + self._gin(g) * x

    def L(self, g: StateUpdateGene) -> float:
        return abs(self._a(g))

    def G(self, g: StateUpdateGene) -> float:
        return abs(self._gin(g))


class SoftSatSubstrate:
    """#2 soft 飽和 (高天井 K): horizon 内は線形 (発散可)・遠方のみ有界。"""

    name = "softsat"
    obs_gain = 1.0
    K = 50.0

    def step(self, s: np.ndarray, x: np.ndarray, g: StateUpdateGene) -> np.ndarray:
        pre = g.mix * x + g.gate_str * s
        return g.decay * s + (1.0 - g.decay) * self.K * np.tanh(pre / self.K)

    def L(self, g: StateUpdateGene) -> float:
        # sound 上界: J = decay + (1−decay)·gate_str·tanh'(pre/K), |tanh'|≤1。
        return g.decay + (1.0 - g.decay) * abs(g.gate_str)

    def G(self, g: StateUpdateGene) -> float:
        return (1.0 - g.decay) * abs(g.mix)


class HighGainSubstrate:
    """#3 高ゲイン観測: 既存有界 tanh 力学。小 state 誤差が高ゲイン M で致命的出力誤差に。"""

    name = "highgain"
    obs_gain = 20.0   # 出力ゲイン: state 誤差 → obs_gain 倍の致命誤差

    def step(self, s: np.ndarray, x: np.ndarray, g: StateUpdateGene) -> np.ndarray:
        pre = g.mix * x + g.gate_str * s
        return g.decay * s + (1.0 - g.decay) * np.tanh(pre)   # 既存有界力学 (|s|≤1)

    def L(self, g: StateUpdateGene) -> float:
        # 既存 tanh substrate の L 上界 (tracking_tube と同型, |tanh'|≤1)。
        return g.decay + (1.0 - g.decay) * abs(g.gate_str)

    def G(self, g: StateUpdateGene) -> float:
        # 入力ゲイン × obs_gain (致命誤差スケール)。
        return self.obs_gain * (1.0 - g.decay) * abs(g.mix)


ALL_SUBSTRATES = [LinearSubstrate(), SoftSatSubstrate(), HighGainSubstrate()]
