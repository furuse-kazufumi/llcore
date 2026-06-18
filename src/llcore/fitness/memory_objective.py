# SPDX-License-Identifier: Apache-2.0
"""branch A — メモリ効率を進化の適応度にする scalarized objective (PoC P1).

設計 = ``docs/BRANCH_A_MEMORY_FITNESS_GATE_DESIGN.md``。北極星 pivot(capability→メモリ効率,
2026-06-16)で作った機構資産(GA + verified-plasticity gate + falsification harness)を
勝ち筋(メモリ効率)へ再配線する「B コンポーネント」= **本設計で新規に作る唯一の実質部品**。

honest 位置づけ(``docs/POSITIONING_VS_LLAMACPP.md`` §2(c)準拠):

- 「メモリ指標を適応度にする」「accuracy×memory のスカラ化」自体は **HW-NAS(MnasNet 2018)/
  多目的 NAS の再導出(既知)** であり、新規アルゴリズムではない。
- **P1 スコープ(最重要 honest 留保)**: 本 objective が測る "memory" は、実 char-LM の RSS footprint
  (``MEMORY_EFFICIENCY_FINDINGS.md`` の 539→149MB 等)ではなく **state-boundedness proxy**
  (状態更新 gene の収縮率 ``L`` 由来の代理指標)。「メモリ実測を進化適応度にした」とは **書けない**。
  実 footprint を fitness にする P2 は将来(``lm/quant.py`` の ``int8_footprint_bytes`` と GA を結合する
  大改変)。
- retention も「fixed-readout probe fitness」(``tasks.py``)であって LLM accuracy ではない。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from llcore.state_update import StateUpdateGene

from .tasks import FixedReadout, SyntheticTask, evaluate_gene


def state_boundedness_footprint(gene: StateUpdateGene, *, l_cap: float = 2.0) -> float:
    """状態有界性 proxy ∈ [0, 1]。収縮率 ``L`` が小さい(=有界・メモリ効率的)ほど 0 に近い。

    ``L = max(|decay|, |decay + (1 − decay)·gate_str|)`` は状態方向ヤコビ
    ``J(t) = decay + (1 − decay)·gate_str·t`` (t∈[0,1]) の sup で、verifier の閉形式上界
    ``llcore.verifier.invariants._lipschitz_upper_bound`` と **同式**(test で同値性を保証)。

    ``L < 1`` = 収縮的(Banach 固定点・有界状態=定数状態の勝ち筋)、``L ≥ 1`` = 発散しうる
    (状態が膨らむ=メモリ非効率)。``[0, l_cap]`` で正規化して footprint とする
    (``L=0→0`` / ``L=1→0.5`` / ``L≥l_cap→1``、``l_cap`` 既定 2.0 = gate_str の物理上界由来)。

    honest: これは実 RSS footprint ではなく「定数サイズの有界状態を保てる gene か」の代理(P1)。
    """
    g = gene.clipped()
    l_bound = max(abs(g.decay), abs(g.decay + (1.0 - g.decay) * g.gate_str))
    return float(np.clip(l_bound / l_cap, 0.0, 1.0))


@dataclass(frozen=True)
class MemoryEfficiencyObjective:
    """accuracy(retention)× memory(state-boundedness)を重み付き和でスカラ化した適応度。

    ``fitness(gene) = (w_acc·retention + w_mem·(1 − footprint)) / (w_acc + w_mem) ∈ [0, 1]``

    - ``retention = evaluate_gene(gene, base_task, readout, rng)`` ∈ [0, 1](fixed-readout probe)
    - ``footprint = state_boundedness_footprint(gene)`` ∈ [0, 1](収縮率 proxy)

    ``w_mem=0`` で純 retention(capability ベースライン)に縮退する(floor 包含, 設計 §1.4 /
    ARTICLE_SEEDS #11)。MnasNet 流スカラ化なので既存 ``evolve()`` のスカラ fitness 前提に改変ゼロで載る。
    """

    base_task: SyntheticTask
    w_acc: float = 0.7
    w_mem: float = 0.3
    l_cap: float = 2.0
    n_trials: int = 5

    def __post_init__(self) -> None:
        if self.w_acc < 0.0 or self.w_mem < 0.0 or (self.w_acc + self.w_mem) <= 0.0:
            raise ValueError("w_acc/w_mem must be non-negative and sum to > 0")

    def fitness(
        self,
        gene: StateUpdateGene,
        readout: FixedReadout,
        rng: np.random.Generator,
    ) -> float:
        """gene のメモリ効率適応度 ∈ [0, 1] を返す。"""
        retention = evaluate_gene(gene, self.base_task, readout, rng, n_trials=self.n_trials)
        footprint = state_boundedness_footprint(gene, l_cap=self.l_cap)
        total = self.w_acc + self.w_mem
        return float((self.w_acc * retention + self.w_mem * (1.0 - footprint)) / total)


__all__ = ["MemoryEfficiencyObjective", "state_boundedness_footprint"]
