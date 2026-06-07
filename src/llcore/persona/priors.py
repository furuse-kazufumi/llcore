# SPDX-License-Identifier: Apache-2.0
"""8 persona priors over (decay, mix, gate_str) kernel space.

falsifiable 命題 (PoC 2b の一部):
    8 persona × kernel prior は parameter identifiability を持ち、
    集団全体として control (p7 only) より kernel coverage が広い
    (G1 で convex hull / variance 比 を測定)。

設計:
- 各 persona は (decay, mix, gate_str) の 3 軸それぞれに **mean** と **sigma** を持つ。
- gene サンプリングは ``mean + sigma * standard_normal()``、clip 範囲で抑える。
- p0..p7 は kernel 空間の異なる領域に bias (重複を避けるため mean 位置を分散):

    | p   | decay      | mix         | gate_str    | 説明               |
    |-----|------------|-------------|-------------|--------------------|
    | p0  | 高 (0.85)  | 低 (0.20)   | 中 (0.50)   | 強記憶+小入力      |
    | p1  | 低 (0.20)  | 高 (0.80)   | 中 (0.50)   | 速応答+大入力      |
    | p2  | 中 (0.50)  | 中 (0.50)   | 拡張 (±1.5) | recurrent 拡張     |
    | p3  | 高 (0.85)  | 高 (0.80)   | 低 (0.20)   | 強記憶+大入力      |
    | p4  | 低 (0.20)  | 低 (0.20)   | 高 (1.50)   | 速応答+強帰還      |
    | p5  | 中 (0.50)  | 拡張 (±1.0) | 中 (0.50)   | mix 拡張 (負含む)  |
    | p6  | 拡張(0.50) | 中 (0.50)   | 中 (0.50)   | decay 拡張 (broad) |
    | p7  | 中 (0.50)  | 中 (0.50)   | 中 (0.50)   | control (uniform)  |

- p2/p5/p6 は **拡張** = sigma を広めに取り、軸方向の多様性を担う。
- p7 = control: 全軸 mean=中央, sigma=広い (= uniform 近似)。

honest 留保:
- mean/sigma は手動設計 (auto-learning でない)。
- p7 control の sigma は他 persona より大きく、coverage 比較は「mean 分散」観点と
  「sigma 拡張」観点の 2 つを同時に見る必要がある。
- 8 persona の重複は 4D 図上で部分的にあり (例: p3 と p0 は decay 高で近い)。
  G1 で convex hull volume 比較を行うことで重複の影響を honest に観察する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from llcore.state_update import StateUpdateGene


NUM_PERSONAS: int = 8

PERSONA_LABELS: Tuple[str, ...] = (
    "p0_high_decay_low_mix",
    "p1_low_decay_high_mix",
    "p2_mid_gate_extended",
    "p3_high_decay_high_mix",
    "p4_low_decay_high_gate",
    "p5_mid_mix_extended",
    "p6_decay_broad",
    "p7_control_uniform",
)


@dataclass(frozen=True)
class PersonaPrior:
    """1 persona の kernel sampling prior.

    Attributes
    ----------
    persona_id : int
        0..7 の persona index.
    label : str
        human-readable label.
    decay_mean : float
        decay parameter の事前平均 (clip 範囲 [0,1] 内).
    decay_sigma : float
        decay parameter の事前 sigma (gaussian).
    mix_mean : float
        mix parameter の事前平均 (clip 範囲 [-1,1] 内).
    mix_sigma : float
        mix parameter の事前 sigma.
    gate_mean : float
        gate_str parameter の事前平均 (clip 範囲 [-2,2] 内).
    gate_sigma : float
        gate_str parameter の事前 sigma.
    """

    persona_id: int
    label: str
    decay_mean: float
    decay_sigma: float
    mix_mean: float
    mix_sigma: float
    gate_mean: float
    gate_sigma: float

    def sample(self, rng: np.random.Generator) -> StateUpdateGene:
        """この prior に従い 1 個体 gene をサンプル (clip 適用済).

        各軸独立 gaussian。clip で範囲外を抑える。
        """
        decay = self.decay_mean + self.decay_sigma * rng.standard_normal()
        mix = self.mix_mean + self.mix_sigma * rng.standard_normal()
        gate = self.gate_mean + self.gate_sigma * rng.standard_normal()
        gene = StateUpdateGene(decay=float(decay), mix=float(mix), gate_str=float(gate))
        return gene.clipped()

    def mean_array(self) -> np.ndarray:
        """prior の mean を 3-vector で返す (G1 で coverage 比較に使用)."""
        return np.array([self.decay_mean, self.mix_mean, self.gate_mean], dtype=np.float64)

    def sigma_array(self) -> np.ndarray:
        """prior の sigma を 3-vector で返す."""
        return np.array([self.decay_sigma, self.mix_sigma, self.gate_sigma], dtype=np.float64)


# ----------------------------------------------------------------------
# 8 persona prior 定義 (mean / sigma 手動設計)
# ----------------------------------------------------------------------
# 設計指針:
# - p0..p6 は各軸の局所領域に絞る (sigma=0.10 程度)
# - p2/p5/p6/p7 は拡張 prior (sigma=0.30 以上)
# - p7 control: 全軸 mean=中央, sigma=広 (= 一様 baseline)
# - 軸別 sigma 上限: decay 0.30, mix 0.40, gate 0.80 程度

PERSONA_PRIORS: Tuple[PersonaPrior, ...] = (
    # p0: 高 decay (強記憶) + 低 mix (小入力) + 中 gate
    PersonaPrior(0, PERSONA_LABELS[0],
                 decay_mean=0.85, decay_sigma=0.08,
                 mix_mean=0.20,   mix_sigma=0.15,
                 gate_mean=0.50,  gate_sigma=0.25),
    # p1: 低 decay (速応答) + 高 mix (大入力) + 中 gate
    PersonaPrior(1, PERSONA_LABELS[1],
                 decay_mean=0.20, decay_sigma=0.08,
                 mix_mean=0.80,   mix_sigma=0.15,
                 gate_mean=0.50,  gate_sigma=0.25),
    # p2: 中 decay/mix + gate 拡張 (sigma 広)
    PersonaPrior(2, PERSONA_LABELS[2],
                 decay_mean=0.50, decay_sigma=0.10,
                 mix_mean=0.50,   mix_sigma=0.15,
                 gate_mean=0.00,  gate_sigma=1.20),
    # p3: 高 decay + 高 mix + 低 gate (典型的 "強記憶+大入力")
    PersonaPrior(3, PERSONA_LABELS[3],
                 decay_mean=0.85, decay_sigma=0.08,
                 mix_mean=0.80,   mix_sigma=0.15,
                 gate_mean=0.20,  gate_sigma=0.20),
    # p4: 低 decay + 低 mix + 高 gate (強帰還 / 振動傾向)
    PersonaPrior(4, PERSONA_LABELS[4],
                 decay_mean=0.20, decay_sigma=0.08,
                 mix_mean=0.20,   mix_sigma=0.15,
                 gate_mean=1.50,  gate_sigma=0.30),
    # p5: 中 decay + mix 拡張 (負入力含む 拡張) + 中 gate
    PersonaPrior(5, PERSONA_LABELS[5],
                 decay_mean=0.50, decay_sigma=0.10,
                 mix_mean=0.00,   mix_sigma=0.60,
                 gate_mean=0.50,  gate_sigma=0.25),
    # p6: decay 拡張 (broad) + 中 mix/gate
    PersonaPrior(6, PERSONA_LABELS[6],
                 decay_mean=0.50, decay_sigma=0.30,
                 mix_mean=0.50,   mix_sigma=0.15,
                 gate_mean=0.50,  gate_sigma=0.25),
    # p7: control = 全軸中央 + 広 sigma (= uniform baseline)
    PersonaPrior(7, PERSONA_LABELS[7],
                 decay_mean=0.50, decay_sigma=0.25,
                 mix_mean=0.00,   mix_sigma=0.50,
                 gate_mean=0.00,  gate_sigma=1.00),
)


def persona_sample_gene(
    persona_id: int, rng: np.random.Generator
) -> StateUpdateGene:
    """指定 persona_id の prior に従い 1 個体 gene をサンプル."""
    if not 0 <= persona_id < NUM_PERSONAS:
        raise ValueError(f"persona_id must be in [0, {NUM_PERSONAS}), got {persona_id}")
    return PERSONA_PRIORS[persona_id].sample(rng)


__all__ = [
    "NUM_PERSONAS",
    "PERSONA_LABELS",
    "PERSONA_PRIORS",
    "PersonaPrior",
    "persona_sample_gene",
]
