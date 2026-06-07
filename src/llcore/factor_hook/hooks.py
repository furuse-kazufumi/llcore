# SPDX-License-Identifier: Apache-2.0
"""ThoughtFactorDeltaHook Protocol + reference 実装 (llive 非依存 minimal 版).

10 思考因子 (llive と同命名):
    structurize / reconstruct / closed_loop / self_extension / uncertainty /
    exploration / integrate / provenance / perspective / reality_contact

各因子は [0, 1] スケール (0=なし, 1=飽和). FactorSnapshot は dict ベースで
未指定因子は 0.5 (中立) を返す。

設計:
- hook の戻り値は **multiplicative scaling factor** Δ (1.0=変更なし、<1=減速、>1=加速)
- HeuristicFactorHook では `exp(weighted_sum / sensitivity)` で連続化、
  hard clamp [0.25, 4.0] で runaway 防止 (llive 設計踏襲)
- apply_hook_to_gene で StateUpdateGene の decay を Δ で動的に scale
  (mix / gate_str は触らない — decay = memory timescale が認知状態に最も適切)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Protocol, runtime_checkable

from llcore.state_update import StateUpdateGene

#: 10 思考因子の canonical 名前 (llive と完全一致).
FACTOR_NAMES: tuple[str, ...] = (
    "structurize",      # 構造化
    "reconstruct",      # 再構成
    "closed_loop",      # 閉ループ
    "self_extension",   # 自己拡張
    "uncertainty",      # 不確実性
    "exploration",      # 探索
    "integrate",        # 整合
    "provenance",       # 来歴
    "perspective",      # 多視点
    "reality_contact",  # 現実接続
)


@dataclass(frozen=True)
class FactorSnapshot:
    """10 因子の point-in-time 値.

    未指定因子は 0.5 (中立) を返す。clamp [0, 1] で範囲外を抑える。

    Attributes
    ----------
    values : Mapping[str, float]
        factor name → value [0, 1].
    stage : str | None
        どの 6-stage step が snapshot を生成したか (optional, llive 互換)。
    """

    values: Mapping[str, float] = field(default_factory=dict)
    stage: str | None = None

    def get(self, name: str, default: float = 0.5) -> float:
        v = self.values.get(name, default)
        return max(0.0, min(1.0, float(v)))

    def vector(self) -> tuple[float, ...]:
        """canonical 順で値を返す."""
        return tuple(self.get(n) for n in FACTOR_NAMES)


@runtime_checkable
class ThoughtFactorDeltaHook(Protocol):
    """FactorSnapshot を multiplicative Δ に写像する Protocol.

    実装側は ``delta_for(snapshot) -> float`` を提供する。
    戻り値 ``1.0`` = 変更なし、``<1.0`` = 減速 (慎重)、``>1.0`` = 加速 (大胆)。
    """

    def delta_for(self, snapshot: FactorSnapshot) -> float: ...


class NoopFactorHook:
    """常に 1.0 を返す default hook (factor_hook OFF 等価)."""

    name = "noop"

    def delta_for(self, snapshot: FactorSnapshot) -> float:  # noqa: ARG002
        return 1.0


class HeuristicFactorHook:
    """reference 実装 (llive 設計踏襲).

    挙動:
    - uncertainty 高 → Δ < 1 (慎重に積分、SSM ステップを小さく)
    - integrate + structurize 高 → Δ > 1 (積極的に統合、ステップ大きく)
    - exploration 高 → 軽く Δ > 1 (大胆にサンプル)

    Hard clamp [0.25, 4.0] で runaway 防止。
    """

    name = "heuristic"

    def __init__(self, sensitivity: float = 1.0) -> None:
        self.sensitivity = float(sensitivity)

    def delta_for(self, snapshot: FactorSnapshot) -> float:
        unc = snapshot.get("uncertainty")
        integ = snapshot.get("integrate")
        struct = snapshot.get("structurize")
        explore = snapshot.get("exploration")
        # raw signal in roughly [-1, 1]
        signal = (integ + struct + 0.5 * explore - 1.5 * unc) / 2.0
        delta = math.exp(signal * self.sensitivity)
        return max(0.25, min(4.0, delta))


def apply_hook_to_gene(
    gene: StateUpdateGene, hook: ThoughtFactorDeltaHook, snapshot: FactorSnapshot
) -> StateUpdateGene:
    """Δ で gene の **decay** を動的に scale (mix/gate_str は触らない).

    decay = memory timescale が認知状態に最も適切な調整軸 (uncertainty 高 →
    decay 高 = state を保持しがち、integrate 高 → decay 低 = 新情報統合)。

    実装: ``decay' = clip(decay * (2 - Δ), 0, 1)`` で逆相関に。

    Follow-up debt (Codex 2026-05-29 指摘): Δ>2 で `2-Δ<0` → decay 完全 0 (clip)
    に潰れる。Δ の上側情報を捨てる。v0.2 で ``decay/Δ`` or ``decay + k*(1-Δ)``
    のような負値を作らない写像へ改修候補。

    返り値 gene は clip 後の値 (StateUpdateGene.clipped()).
    """
    delta = hook.delta_for(snapshot)
    # Δ > 1 (大胆) → decay 小 (新情報統合) / Δ < 1 (慎重) → decay 大 (記憶保持)
    # normalized delta in roughly [0.25, 4.0] → decay scale in [0.5, 1.75]
    # decay_new = decay * (2 - delta) で逆相関 (delta=1 で変化なし)
    new_decay = gene.decay * (2.0 - delta)
    return StateUpdateGene(
        decay=new_decay,
        mix=gene.mix,
        gate_str=gene.gate_str,
    ).clipped()
