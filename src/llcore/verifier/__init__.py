# SPDX-License-Identifier: Apache-2.0
"""Z3-based state update invariant verifier (Stage 1a).

llcore の核独自軸: **進化ループ内 SMT online gate** (事前調査で先行未発見).
StateUpdateGene の clip 範囲下で「state norm 有界」が構造的に保たれることを
Z3 で satisfiability 検査し、進化中の gene を online で gate する基盤。

API:
- :func:`verify_state_norm_invariant` — clip 範囲下の有界性を Z3 で検証
- :func:`verify_gene_safe` — 単一 gene が安全 (invariant 違反なし) か検査
- :class:`InvariantResult` — 検査結果 (ok / counterexample / z3 unavailable 等)

honest 留保:
- tanh は Z3 で直接表現できないため、``|tanh(z)| <= min(|z|, 1)`` の上界で近似。
  これは保守的 (sound: 上界で OK なら実値も OK) だが完全ではない。
- Stage 1a は state_norm 有界 (Lyapunov-style 数値安定性) のみ。Lipschitz は Stage 1b、
  spectral radius は Stage 1c で別 PoC。
"""

from .invariants import (
    InvariantResult,
    is_z3_available,
    verify_gene_safe,
    verify_state_norm_invariant,
)

__all__ = [
    "InvariantResult",
    "is_z3_available",
    "verify_gene_safe",
    "verify_state_norm_invariant",
]
