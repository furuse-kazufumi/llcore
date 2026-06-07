# SPDX-License-Identifier: Apache-2.0
"""State update 数式遺伝子表現 (PoC 0a → 3 の基盤).

3 パラメータ minimal kernel (PoC 0a):
- ``decay``     : 1 step ごとの state 減衰 (0.0-1.0)
- ``mix``       : 新規入力の混合率 (0.0-1.0)
- ``gate_str``  : tanh gating の強度 (0.0-2.0)

将来 (PoC 3a) で kernel 多様化:
- ``rwkv_v7`` style (delta-rule + vector gating)
- ``mamba_selective`` style (input-dependent Δ)
- ``hopfield_dense`` style (associative retrieval)
- ``linear_attention`` style (kernel feature map)

各 kernel は同一 :class:`StateUpdateGene` interface を介して
:mod:`llcore.verifier` の Z3 不変量 gate に供給される。
"""

from .genes import StateUpdateGene, eval_step, run_sequence

__all__ = ["StateUpdateGene", "eval_step", "run_sequence"]
