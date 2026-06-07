# SPDX-License-Identifier: Apache-2.0
"""発散しうる基質 3 種 (R-endo viability PoC) — 環境 = recurrence ゲイン κ。

なぜなぜ分析の根本原因 = 「無条件有界な基質では証明対象 (収縮) と生存がデカップル」。さらに
スモークで判明: 外乱 (w̄) だけを環境にすると、進化は入力ゲイン g を小さくして「高 a (記憶) ×
低包絡 (安全)」に逃げられ、soft 境界が回避可能。一方 hard 発散 (a≥1) は環境非依存で内外 gate が一致。

→ **環境を recurrence ゲイン κ にする**: 実効収縮 = κ·a。κ↑ で以前は安定だった gene (a<1) が
κ·a≥1 で**発散** = 環境変化が viability を脅かす・g で回避不能・real divergence。これで内的 gate に
「自己保存の仕事」が生まれる (環境結合で κ を sense し再 gate)。

3 実現法 (各々 κ で発散境界が動く):
- ``LinearSubstrate``  (#1 飽和除去): ``s' = (κ·a)·s + g·x``。|κ·a|≥1 で幾何発散=死。tube 厳密。
- ``SoftSatSubstrate`` (#2 soft 飽和/高天井): horizon 内は線形 (発散可)・遠方のみ |s|≤K (NaN なし)。
- ``HighGainSubstrate`` (#3 高ゲイン観測): 既存有界 tanh 力学・高ゲイン M で小 state 誤差が致命的出力誤差に。
  発散はしないが κ↑ で出力包絡が生存閾を超える (soft death, 既存 tracking_tube 同型 L/G を再利用)。

共通契約 (duck typing): ``name`` / ``obs_gain`` 属性 + ``step(s,x,gene,kappa)`` / ``L(gene,kappa)`` / ``G(gene)``。
- ``L(gene,kappa)`` = 環境 κ 下の状態方向 Lipschitz 上界 (収縮判定 κ·... <1)。
- ``G(gene)`` = 入力方向ゲイン (κ 非依存; obs_gain 込み)。tube r = G·w̄/(1−L)。
すべて gene は clip 済み前提。新規 production 依存なし (additive)。
"""
from __future__ import annotations

import numpy as np

from llcore.state_update import StateUpdateGene


class LinearSubstrate:
    """#1 飽和除去: 線形。実効収縮 κ·a。|κ·a|≥1 で幾何発散=死。"""

    name = "linear"
    obs_gain = 1.0

    @staticmethod
    def _a0(g: StateUpdateGene) -> float:
        return g.decay + (1.0 - g.decay) * g.gate_str

    @staticmethod
    def _gin(g: StateUpdateGene) -> float:
        return (1.0 - g.decay) * g.mix

    def step(self, s, x, g, kappa: float):
        return (kappa * self._a0(g)) * s + self._gin(g) * x

    def L(self, g: StateUpdateGene, kappa: float) -> float:
        return abs(kappa * self._a0(g))

    def G(self, g: StateUpdateGene) -> float:
        return abs(self._gin(g))


class SoftSatSubstrate:
    """#2 soft 飽和 (高天井 K): horizon 内は線形 (κ で発散可)・遠方のみ有界。"""

    name = "softsat"
    obs_gain = 1.0
    K = 50.0

    def step(self, s, x, g, kappa: float):
        pre = g.mix * x + kappa * g.gate_str * s
        return g.decay * s + (1.0 - g.decay) * self.K * np.tanh(pre / self.K)

    def L(self, g: StateUpdateGene, kappa: float) -> float:
        # sound 上界: J = decay + (1−decay)·κ·gate_str·tanh'(·), |tanh'|≤1。
        return g.decay + (1.0 - g.decay) * kappa * abs(g.gate_str)

    def G(self, g: StateUpdateGene) -> float:
        return (1.0 - g.decay) * abs(g.mix)


class HighGainSubstrate:
    """#3 高ゲイン観測: 既存有界 tanh 力学。小 state 誤差が高ゲイン M で致命的出力誤差に。

    発散はしない (tanh 有界) が、κ↑ で state 応答が増し出力包絡 (obs_gain 倍) が生存閾 V を超える。
    既存 tracking_tube と同型の L/G を再利用 (production verifier 直結の soft-death 版)。
    """

    name = "highgain"
    obs_gain = 20.0

    def step(self, s, x, g, kappa: float):
        pre = g.mix * x + kappa * g.gate_str * s
        return g.decay * s + (1.0 - g.decay) * np.tanh(pre)

    def L(self, g: StateUpdateGene, kappa: float) -> float:
        return g.decay + (1.0 - g.decay) * kappa * abs(g.gate_str)

    def G(self, g: StateUpdateGene) -> float:
        return self.obs_gain * (1.0 - g.decay) * abs(g.mix)


ALL_SUBSTRATES = [LinearSubstrate(), SoftSatSubstrate(), HighGainSubstrate()]
