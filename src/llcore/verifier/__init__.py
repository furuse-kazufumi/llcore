# SPDX-License-Identifier: Apache-2.0
"""Z3-based state update invariant verifier + Marabou bridge skeleton.

llcore の核独自軸:
- Stage 1a: 進化ループ内 SMT online gate (PoC 1a 完了)
- Stage 3a: Marabou Incremental NN Verification の **異構造拡張** refinement
  relation sound 拡張 + ChangeOp-MCC curriculum (PoC 3a)

API (Stage 1a):
- :func:`verify_state_norm_invariant` — clip 範囲下の有界性を Z3 で検証
- :func:`verify_gene_safe` — 単一 gene が安全 (invariant 違反なし) か検査
- :class:`InvariantResult` — 検査結果

API (Stage 3a):
- :mod:`llcore.verifier.changeop` — ChangeOp / ChangeOpSequence 型と合成
- :mod:`llcore.verifier.refinement` — sound 拡張 R(NN,NN',c) と Marabou bridge
- :mod:`llcore.verifier.curriculum` — MCC 風 ChangeOp 淘汰

honest 留保:
- tanh は Z3 で直接表現できないため、``|tanh(z)| <= 1`` の上界で近似 (sound).
- Marabou native 統合は Stage 5+。Stage 3a は **mock + Z3 で sound 性を再現** し
  refinement 拡張の機構実証に集中。
"""

from .invariants import (
    InvariantResult,
    is_z3_available,
    verify_gene_safe,
    verify_state_norm_invariant,
)
from .changeop import (
    ChangeOp,
    ChangeOpSequence,
    apply_changeop,
    apply_sequence,
    decay_shift,
    gate_shift,
    kernel_swap_mock,
    mix_shift,
    sequence_from_iter,
)
from .refinement import (
    E_BASE,
    K_INHERIT,
    KERNEL_SWAP_EXTRA,
    MarabouBridgeStatus,
    RefinementResult,
    SequenceCheckResult,
    epsilon_for,
    get_bridge_status,
    is_marabou_available,
    verify_composition,
    verify_refinement_single,
    verify_sequence_tolerance,
)
from .curriculum import (
    CurriculumGeneration,
    CurriculumState,
    evolve_one_generation,
    initial_population,
    is_saturated,
    run_curriculum,
)

__all__ = [
    # Stage 1a
    "InvariantResult",
    "is_z3_available",
    "verify_gene_safe",
    "verify_state_norm_invariant",
    # Stage 3a changeop
    "ChangeOp",
    "ChangeOpSequence",
    "apply_changeop",
    "apply_sequence",
    "decay_shift",
    "mix_shift",
    "gate_shift",
    "kernel_swap_mock",
    "sequence_from_iter",
    # Stage 3a refinement
    "E_BASE",
    "K_INHERIT",
    "KERNEL_SWAP_EXTRA",
    "MarabouBridgeStatus",
    "RefinementResult",
    "SequenceCheckResult",
    "epsilon_for",
    "get_bridge_status",
    "is_marabou_available",
    "verify_composition",
    "verify_refinement_single",
    "verify_sequence_tolerance",
    # Stage 3a curriculum
    "CurriculumGeneration",
    "CurriculumState",
    "evolve_one_generation",
    "initial_population",
    "is_saturated",
    "run_curriculum",
]
