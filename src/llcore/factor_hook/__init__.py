# SPDX-License-Identifier: Apache-2.0
"""Thought-factor → state update Δ bridge (Stage 2a, llive 非依存版).

llcore 核独自軸 #3 (事前調査で実装した先行未発見): 認知状態 (10 思考因子) を
state update kernel の Δ 動的化に直接結びつける Protocol。

llive `factor_hook.py` の Protocol を **参考**にしたが、llcore は llive
非依存路線のため自前 minimal 版を持つ (interface 互換は意図的に維持し、
将来 llive と相互運用したい場合の adapter 実装を容易に)。

API:
- :class:`FactorSnapshot` — 10 因子の dict (immutable, [0,1] clamp)
- :class:`ThoughtFactorDeltaHook` Protocol — delta_for(snapshot) -> float
- :class:`NoopFactorHook` — 常に 1.0 (Δ 無変化)
- :class:`HeuristicFactorHook` — uncertainty 高 → Δ 小 / integrate+structurize 高 → Δ 大
- :func:`apply_hook_to_gene` — Δ で gene の decay 等を動的調整

設計判断:
- 10 因子の命名は llive と完全一致 (将来融合可能性を保つ)
- pure numpy (z3-solver / llive 非依存)
- mock 環境で動作 (実 RWKV weight 不要、純数式 state update に作用)
"""

from .hooks import (
    FACTOR_NAMES,
    FactorSnapshot,
    HeuristicFactorHook,
    NoopFactorHook,
    ThoughtFactorDeltaHook,
    apply_hook_to_gene,
)

__all__ = [
    "FACTOR_NAMES",
    "FactorSnapshot",
    "HeuristicFactorHook",
    "NoopFactorHook",
    "ThoughtFactorDeltaHook",
    "apply_hook_to_gene",
]
