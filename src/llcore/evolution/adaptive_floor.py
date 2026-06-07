# SPDX-License-Identifier: Apache-2.0
"""Adaptive percentile gate (適応難易度ゲート) for llcore PoC 2b.

llive ``src/llive/perf/evolutionary/pressures.py:AdaptivePercentileGate`` を参考に
llcore 自前で minimal 実装 (llive import 禁止、Read 参照のみ)。

目的:
    固定 fitness 上限 1.0 飽和で選択圧が消失して 12h ラン失敗を起こす病理を回避する。
    集団分位 floor を毎世代再計算し、ratchet で **単調非減少** に保つ。floor を超える
    個体だけ繁殖候補に残す = 集団が改善すると floor も追従して上がるため「ものさし」が
    飽和しない (llive で実証済の飽和回避レシピ ① の最小実装)。

honest 留保:
- llive 版は (a) 多 axes 同時 floor (b) Population 署名で世代 1 回計算 (c) 全員 fail で
  selector が gate 無視, など機能が豊富。llcore minimal 版は **単一 axis (fitness)** ・
  外部呼出で update する素朴実装。十分。
- ratchet=True で floor は単調非減少。これは「集団分位下限の上昇」であって fitness 本体の
  単調性ではない (G5 で測る)。
- 全員 fail を回避するため、:meth:`survivors` は最低 1 個体を保証 (top-1 保護)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class AdaptiveFloorGate:
    """集団 fitness 分位 floor を毎世代再計算する適応難易度ゲート (ratchet 単調非減少).

    使い方::

        gate = AdaptiveFloorGate(percentile=30.0, ratchet=True)
        for gen in range(n_generations):
            ...
            gate.update(fitness_scores)              # floor 更新
            survivors_idx = gate.survivors(fitness_scores)  # floor 以上の個体 idx
            # survivors で繁殖

    Attributes
    ----------
    percentile : float
        floor に使う集団分位 [0, 100]。30 = 下位 30% を切り捨て、70% が survivors。
    ratchet : bool
        True なら floor は単調非減少 (集団が一時的に退化しても floor は緩まない)。
    floor : float
        現在の floor 値 (update 前は -inf = 全通過)。
    floor_history : list[float]
        各 update 呼出時の floor 値時系列 (G5 単調性検査用)。
    """

    percentile: float = 30.0
    ratchet: bool = True
    floor: float = float("-inf")
    floor_history: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not (0.0 <= self.percentile <= 100.0):
            raise ValueError(f"percentile must be in [0, 100], got {self.percentile}")

    def update(self, fitness_scores: Sequence[float]) -> float:
        """集団 fitness から floor を再計算 (ratchet 適用).

        Parameters
        ----------
        fitness_scores : Sequence[float]
            この世代の全個体 fitness。

        Returns
        -------
        float
            更新後の floor 値。
        """
        arr = np.asarray(list(fitness_scores), dtype=np.float64)
        if arr.size == 0:
            # empty population: 無更新で履歴に現 floor を残す (単調性検査用)
            self.floor_history.append(self.floor)
            return self.floor
        pct = float(np.percentile(arr, self.percentile))
        if self.ratchet:
            # 単調非減少: 集団が退化しても floor は前回値を保持
            self.floor = max(self.floor, pct) if self.floor != float("-inf") else pct
        else:
            self.floor = pct
        self.floor_history.append(self.floor)
        return self.floor

    def survivors(self, fitness_scores: Sequence[float]) -> list[int]:
        """floor 以上の個体 index を返す (全員 fail なら top-1 を保護).

        Parameters
        ----------
        fitness_scores : Sequence[float]
            この世代の全個体 fitness。

        Returns
        -------
        list[int]
            繁殖候補となる個体の index リスト (sorted, 最低 1 件)。
        """
        arr = np.asarray(list(fitness_scores), dtype=np.float64)
        if arr.size == 0:
            return []
        passing = [int(i) for i in range(len(arr)) if arr[i] >= self.floor]
        if not passing:
            # 全員 fail = floor 過剰: top-1 を保護して全滅回避
            return [int(arr.argmax())]
        return passing

    def is_monotonic(self) -> bool:
        """floor_history が単調非減少か (G5 検査用)."""
        if len(self.floor_history) < 2:
            return True
        # -inf を初期値で許容
        valid = [f for f in self.floor_history if f != float("-inf")]
        return all(valid[i + 1] >= valid[i] - 1e-12 for i in range(len(valid) - 1))


__all__ = ["AdaptiveFloorGate"]
