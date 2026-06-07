# SPDX-License-Identifier: Apache-2.0
"""MODES-inspired open-endedness 計器 (A_new + diversity) for llcore PoC 2b.

llive ``scripts/poc_evolutionary_activity_modes.py`` の Bedau 指標を minimal 実装。
A_new (新規 behavior 採用件数) と diversity 崩壊検出 の AND gate で 3 regime
(adaptive / saturated / neutral) を弁別する。

定義 (minimal):
- behavior descriptor = 個体の kernel 3 次元 (decay, mix, gate_str) を quantize した
  tuple (持続的形質の代理). 32 bins per axis = 32^3 = 32768 の最大語彙.
- A_new(t) = 世代 t で **初めて出現した descriptor 数** (新規行動の採用).
- diversity(t) = 集団 gene matrix の pairwise L2 平均 (連続値多様性).
- 3 regime:
    * adaptive : A_new > 0 を継続維持 (90% 世代以上で A_new > 0)
    * saturated: A_new = 0 が支配 (新規が止まる) + diversity 低下
    * neutral  : A_new = 0 だが diversity は崩れない (浮動のみ)

honest 留保:
- llive 版は Bedau cumulative activity + neutral shadow による supra-neutral 弁別を
  実装しているが、llcore minimal 版はそれを **A_new (新規 descriptor 採用件数) と
  diversity** の 2 指標に簡約。3 regime 弁別の精度はやや劣るが PoC 2b の AND gate
  には十分。
- descriptor quantization の bin 幅は 32 で固定 (recipe より) — bin が粗いと A_new が
  早期飽和する artifact があり、bin が細かいと neutral でも A_new が出る. 32 は中庸.
- 「saturated 判定」は A_new tail (末尾世代 average) が閾値以下、かつ diversity
  decline > threshold の AND. tuning は PoC 2b verdict で議論.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from llcore.state_update import StateUpdateGene


_DECAY_BIN_LOW, _DECAY_BIN_HIGH = 0.0, 1.0
_MIX_BIN_LOW, _MIX_BIN_HIGH = -1.0, 1.0
_GATE_BIN_LOW, _GATE_BIN_HIGH = -2.0, 2.0


def _quantize_gene(gene: StateUpdateGene, n_bins: int) -> tuple[int, int, int]:
    """gene → discrete descriptor tuple (decay_bin, mix_bin, gate_bin)."""
    g = gene.clipped()
    db = int(np.clip((g.decay - _DECAY_BIN_LOW) / (_DECAY_BIN_HIGH - _DECAY_BIN_LOW) * n_bins, 0, n_bins - 1))
    mb = int(np.clip((g.mix - _MIX_BIN_LOW) / (_MIX_BIN_HIGH - _MIX_BIN_LOW) * n_bins, 0, n_bins - 1))
    gb = int(np.clip((g.gate_str - _GATE_BIN_LOW) / (_GATE_BIN_HIGH - _GATE_BIN_LOW) * n_bins, 0, n_bins - 1))
    return (db, mb, gb)


def pairwise_l2_diversity(genes: Sequence[StateUpdateGene]) -> float:
    """gene 集団の pairwise L2 距離の平均 (連続値多様性)."""
    if len(genes) < 2:
        return 0.0
    arr = np.array([g.clipped().as_array() for g in genes], dtype=np.float64)
    diffs = arr[:, np.newaxis, :] - arr[np.newaxis, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    n = len(arr)
    iu = np.triu_indices(n, k=1)
    return float(dists[iu].mean()) if len(iu[0]) > 0 else 0.0


@dataclass
class ModesMeter:
    """A_new (新規 descriptor 採用) + diversity (連続値多様性) 時系列計器.

    Attributes
    ----------
    n_bins : int
        descriptor quantization bin 数 (各軸). 32 = recipe 既定.
    seen_descriptors : set[tuple[int, int, int]]
        既出 descriptor 集合 (累積).
    a_new_history : list[int]
        各世代の新規 descriptor 採用数 A_new(t).
    diversity_history : list[float]
        各世代の pairwise L2 diversity.
    """

    n_bins: int = 32
    seen_descriptors: set[tuple[int, int, int]] = field(default_factory=set)
    a_new_history: list[int] = field(default_factory=list)
    diversity_history: list[float] = field(default_factory=list)

    def observe(self, genes: Sequence[StateUpdateGene]) -> tuple[int, float]:
        """1 世代の集団を観測し A_new と diversity を記録.

        Returns
        -------
        (a_new, diversity) : tuple[int, float]
            この世代の新規 descriptor 数 と pairwise L2 diversity.
        """
        descriptors = {_quantize_gene(g, self.n_bins) for g in genes}
        new_descriptors = descriptors - self.seen_descriptors
        a_new = len(new_descriptors)
        self.seen_descriptors |= new_descriptors

        div = pairwise_l2_diversity(genes)
        self.a_new_history.append(a_new)
        self.diversity_history.append(div)
        return a_new, div

    def a_new_fraction_active(self) -> float:
        """A_new > 0 を満たす世代の割合 (G6 検査用)."""
        if not self.a_new_history:
            return 0.0
        return sum(1 for a in self.a_new_history if a > 0) / len(self.a_new_history)

    def diversity_collapsed(self, threshold: float = 0.05) -> bool:
        """末尾世代の diversity が初期に比べ閾値以下に崩壊しているか."""
        if len(self.diversity_history) < 4:
            return False
        head_n = max(1, len(self.diversity_history) // 4)
        tail_n = max(1, len(self.diversity_history) // 4)
        head_mean = float(np.mean(self.diversity_history[:head_n]))
        tail_mean = float(np.mean(self.diversity_history[-tail_n:]))
        if head_mean < 1e-12:
            return False
        return tail_mean < threshold * head_mean

    def regime(self) -> str:
        """3 regime 弁別: adaptive / saturated / neutral.

        - adaptive : A_new > 0 を 70% 以上の世代で維持 (活発な新規採用)
        - saturated: A_new ≤ 0 が支配 (90% 世代で A_new=0) + diversity 崩壊
        - neutral  : それ以外 (A_new ≈ 0 だが diversity は維持)
        """
        active = self.a_new_fraction_active()
        if active >= 0.7:
            return "adaptive"
        if active <= 0.1 and self.diversity_collapsed():
            return "saturated"
        return "neutral"

    def is_adaptive_active(
        self,
        active_threshold: float = 0.9,
        require_no_diversity_collapse: bool = True,
        diversity_collapse_threshold: float = 0.05,
    ) -> tuple[bool, dict]:
        """**AND gate** 形式の adaptive regime 判定 (Codex Q4 finding 対応).

        falsifiable に adaptive を主張するには A_new 単独でなく
        diversity 崩壊との **AND** が必要 (saturated 誤判定回避).

        Parameters
        ----------
        active_threshold : float
            A_new > 0 を満たす世代の割合の下限 (既定 0.9 = 90% 以上).
        require_no_diversity_collapse : bool
            True なら ``diversity_collapsed`` 判定の AND を要求.
        diversity_collapse_threshold : float
            ``diversity_collapsed()`` の threshold (末尾/先頭比率).

        Returns
        -------
        ok : bool
            AND gate を通過したか.
        info : dict
            ``a_new_active_frac``, ``diversity_collapsed``, ``head_div``, ``tail_div``
            の数値 (verdict doc 報告用).
        """
        active = self.a_new_fraction_active()
        collapsed = self.diversity_collapsed(threshold=diversity_collapse_threshold)
        info: dict = {
            "a_new_active_frac": active,
            "a_new_threshold": active_threshold,
            "diversity_collapsed": collapsed,
            "require_no_diversity_collapse": require_no_diversity_collapse,
        }
        if self.diversity_history:
            head_n = max(1, len(self.diversity_history) // 4)
            tail_n = max(1, len(self.diversity_history) // 4)
            info["head_div_mean"] = float(np.mean(self.diversity_history[:head_n]))
            info["tail_div_mean"] = float(np.mean(self.diversity_history[-tail_n:]))
        ok = active >= active_threshold
        if require_no_diversity_collapse:
            ok = ok and not collapsed
        return ok, info


__all__ = ["ModesMeter", "pairwise_l2_diversity"]
