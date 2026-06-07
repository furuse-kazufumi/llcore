# SPDX-License-Identifier: Apache-2.0
"""Lineage reservoir (中立貯蔵庫) — persona 別 best-ever 保持 + 絶滅時 re-inject.

llive ``src/llive/perf/evolutionary/lineage_reservoir.py`` を参考に llcore 自前で
minimal 実装 (llive import 禁止、Read 参照のみ)。

目的:
    novelty / ε-lexicase は **行動多様性** を保つが **系統固定 (lineage fixation)** を
    止められない (既存個体の保存のみで絶滅系統を復活できない)。本 reservoir は
    persona 別 best-ever 個体を保持し、絶滅 persona は次世代で best-ever を re-inject
    することで **系統多様性を構造的に保証** する。

honest 留保 (llive 既出指摘の踏襲):
- 貯蔵庫は frozen elite を再投入するため、復活系統の「生存」は能動進化でなく
  代表の生命維持 (中立貯蔵庫の定義通り)。再結合の素材を残すのが目的。
- 集団 fitness と「persona 多様性」は orthogonal な指標であり、reservoir が
  active progress を阻害する可能性がある (frozen elite が選択圧の足を引っ張る)。
  G3 で「persona 全生存」を測りつつ、G4 で集団 best 単調性を測ることで両側を観察。
- minimal API: update_best(persona_id, gene, fitness) と reinject_extinct(present_personas)
  のみ。llive 版 (lineage_of マップ + Population 接続) は省略。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from llcore.state_update import StateUpdateGene


@dataclass
class LineageReservoir:
    """persona 別 best-ever gene + fitness を保持し絶滅 persona を再投入する貯蔵庫.

    使い方::

        reservoir = LineageReservoir()
        # 各世代:
        for ind in population:
            reservoir.update_best(ind.persona_id, ind.gene, ind.fitness)
        present = {ind.persona_id for ind in next_population}
        revived = reservoir.reinject_extinct(present)
        # revived: list[(persona_id, gene, fitness)] を bred 集団に挿入

    Attributes
    ----------
    best_by_persona : dict[int, tuple[float, StateUpdateGene]]
        persona_id → (best_fitness, best_gene). 高 fitness で上書き。
    reinject_history : list[set[int]]
        各世代で revive した persona_id 集合 (G3 検査用).
    """

    best_by_persona: dict[int, tuple[float, StateUpdateGene]] = field(default_factory=dict)
    reinject_history: list[set[int]] = field(default_factory=list)

    def update_best(
        self, persona_id: int, gene: StateUpdateGene, fitness: float
    ) -> bool:
        """persona の best-ever を更新 (高 fitness なら上書き).

        Returns
        -------
        bool
            True なら新 best として上書き、False なら既存維持。
        """
        prev = self.best_by_persona.get(persona_id)
        if prev is None or fitness > prev[0]:
            self.best_by_persona[persona_id] = (float(fitness), gene)
            return True
        return False

    def has(self, persona_id: int) -> bool:
        """persona の best-ever が貯蔵庫にあるか."""
        return persona_id in self.best_by_persona

    def get_best(self, persona_id: int) -> tuple[float, StateUpdateGene] | None:
        return self.best_by_persona.get(persona_id)

    def reinject_extinct(
        self, present_personas: Iterable[int], protected: Iterable[int] | None = None
    ) -> list[tuple[int, StateUpdateGene, float]]:
        """present_personas に居ない保護 persona の best-ever を再投入リストで返す.

        Parameters
        ----------
        present_personas : Iterable[int]
            この世代に既に居る persona_id 集合.
        protected : Iterable[int] | None
            保護対象 persona_id (None なら貯蔵庫全 persona 保護).

        Returns
        -------
        list[tuple[int, StateUpdateGene, float]]
            再投入する (persona_id, gene, best_fitness) のリスト.
        """
        present_set = set(int(p) for p in present_personas)
        if protected is None:
            protected_set = set(self.best_by_persona.keys())
        else:
            protected_set = set(int(p) for p in protected)
        extinct = sorted(
            p for p in protected_set if p not in present_set and p in self.best_by_persona
        )
        result: list[tuple[int, StateUpdateGene, float]] = []
        for pid in extinct:
            fit, gene = self.best_by_persona[pid]
            result.append((pid, gene, fit))
        self.reinject_history.append(set(extinct))
        return result

    def num_persona_stored(self) -> int:
        return len(self.best_by_persona)


__all__ = ["LineageReservoir"]
