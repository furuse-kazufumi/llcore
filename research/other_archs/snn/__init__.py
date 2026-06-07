# SPDX-License-Identifier: Apache-2.0
"""SNN (Spiking Neural Network) への llcore approach 移植 PoC.

llcore approach = gene 化 + Z3 invariant + 進化 + open-ended 4 機構 が
discrete spike + 時間積分混在の LIF (Leaky Integrate-and-Fire) neuron でも
成立するかを **falsifiable** に検証する。

公開 API:
- :class:`LIFGene` — 4 パラメータ LIF gene (tau_m, V_th, V_reset, t_ref)
- :func:`simulate_lif` — forward Euler simulation (V trace + spike list)
- :func:`verify_firing_rate_bound` — Z3 firing rate 上界 invariant
- :func:`verify_membrane_bounded` — Z3 膜電位 bounded invariant
- :func:`verify_shielded_rl_hint` — Z3 shield 制約 (max rate <= R_safe) sketch

honest 留保:
- forward Euler dt=0.1 ms の数値積分。連続時間 ODE の保存性は近似。
- Shielded RL hint は **sketch のみ**。ProSh / Adaptive GR(1) shielding との
  本格統合は将来研究。
"""
from __future__ import annotations

from .snn_gene import LIFGene, simulate_lif
from .snn_verifier import (
    verify_firing_rate_bound,
    verify_membrane_bounded,
    verify_membrane_bounded_2step,
    verify_shielded_rl_hint,
)

__all__ = [
    "LIFGene",
    "simulate_lif",
    "verify_firing_rate_bound",
    "verify_membrane_bounded",
    "verify_membrane_bounded_2step",
    "verify_shielded_rl_hint",
]
