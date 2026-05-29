# SPDX-License-Identifier: Apache-2.0
"""Izhikevich SNN gene への llcore approach 移植 PoC (Stage 2.3).

LIF (Stage 2.2a) で確立した llcore approach (gene 化 + Z3 invariant + 進化 +
open-ended 4 機構) を **Izhikevich 2D model** へ一般化することで:

- Codex Q5 で指摘された LIF clip の表現幅狭さ (tonic spiking 周辺のみ) を解消
- RS / IB / CH / FS の 4 firing pattern type を 4 パラメータ (a, b, c, d) 内で
  単一 family として進化可能にする

公開 API:
- :class:`IzhikevichGene` — 4 パラメータ Izhikevich gene (a, b, c, d)
- :func:`simulate_izh` — forward Euler simulation (V/u trace + spike list)
- :func:`verify_v_bounded_per_gene` — Z3 v bounded invariant (assumed-input contract)
- :func:`verify_firing_rate_per_gene` — Z3 firing rate 上界 (dt-discretization bound)

honest 留保:
- Izhikevich model の v² 非線形は Z3 NRA (quantifier-free nonlinear real arithmetic)
  で扱える。float64 simulator とは exact rational vs 浮動小数で微差あり.
- forward Euler dt=0.25 ms。原論文の 2 段 half-step trick は本 PoC では使わず素朴 Euler.
- refractory なし: firing rate 上界は spike 間隔 >= dt 制約から導出 (1-step 不可分性).
- 既存 ``research/other_archs/snn/`` (LIF 関連) には触らず、本 dir は完全に隔離.
"""
from __future__ import annotations

from .izh_gene import IzhikevichGene, simulate_izh
from .izh_verifier import (
    verify_firing_rate_per_gene,
    verify_v_bounded_per_gene,
)

__all__ = [
    "IzhikevichGene",
    "simulate_izh",
    "verify_v_bounded_per_gene",
    "verify_firing_rate_per_gene",
]
