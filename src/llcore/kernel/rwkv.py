# SPDX-License-Identifier: Apache-2.0
"""RWKV kernel — Kernel plugin Protocol の **最初の準拠例** (S1, dogfooding).

本流 RWKV (``state_update.genes`` + ``verifier.invariants``) を Kernel / VerifierBackend
Protocol に委譲 wrapper で適合させる。**既存実装は一切改変せず**、薄い adapter のみ
(設計 doc §4 S1, semver (D) src 既存挙動不変)。

これにより「3 抽象 Protocol が実在の本流コードで成立する」ことを実証し、SNN-LIF を
2 例目として載せる土台にする (設計 doc §5 (A))。
"""
from __future__ import annotations

import numpy as np

from llcore.state_update import StateUpdateGene, run_sequence
from llcore.verifier import ChangeOp, InvariantResult, apply_changeop, is_z3_available
from llcore.verifier.invariants import verify_gene_safe as _verify_gene_safe

from .protocol import Trajectory

# RWKV gene の物理範囲 (state_update.genes.clipped と一致):
#   decay ∈ [0, 1] / mix ∈ [-1, 1] / gate_str ∈ [-2, 2]
_RWKV_LOWER = np.array([0.0, -1.0, -2.0], dtype=np.float64)
_RWKV_UPPER = np.array([1.0, 1.0, 2.0], dtype=np.float64)


class RWKVCodec:
    """:class:`StateUpdateGene` 用 GeneCodec (3-dim).

    既存 ``as_array`` / ``from_array`` / ``clipped`` へ委譲する薄い層。
    """

    @property
    def dim(self) -> int:
        return 3

    @property
    def lower(self) -> np.ndarray:
        return _RWKV_LOWER.copy()

    @property
    def upper(self) -> np.ndarray:
        return _RWKV_UPPER.copy()

    def to_array(self, gene: StateUpdateGene) -> np.ndarray:
        return gene.as_array()

    def from_array(self, arr: np.ndarray) -> StateUpdateGene:
        return StateUpdateGene.from_array(arr)

    def clip(self, gene: StateUpdateGene) -> StateUpdateGene:
        return gene.clipped()


class RWKVKernel:
    """RWKV state update kernel plugin (Kernel Protocol 準拠例)."""

    name = "rwkv"
    # 既存 changeop.OP_TYPES と一致 (C4: kernel 別 op_type 宣言)
    change_op_types: tuple[str, ...] = (
        "decay_shift",
        "mix_shift",
        "gate_shift",
        "kernel_swap_mock",
    )

    def __init__(self) -> None:
        self.codec = RWKVCodec()

    def simulate(
        self,
        inputs: np.ndarray,
        gene: StateUpdateGene,
        initial_state: np.ndarray | None = None,
    ) -> Trajectory:
        """既存 ``run_sequence`` を Trajectory に正規化 (kind="state", events 空)."""
        states = run_sequence(inputs, gene, initial_state=initial_state)
        return Trajectory(primary=states, events=(), kind="state")

    def apply_change_op(
        self, gene: StateUpdateGene, op: ChangeOp
    ) -> StateUpdateGene:
        """既存 ``apply_changeop`` へ委譲 (ChangeOp 型をそのまま受ける, M2)."""
        return apply_changeop(gene, op)


class RWKVStateNormBackend:
    """RWKV state_norm invariant の VerifierBackend (per-gene 真正).

    既存 ``verifier.invariants.verify_gene_safe`` へ委譲。戻り値は既に本流
    :class:`InvariantResult` なので正規化 adapter は不要。

    per-gene 真正性: ``verify_gene_safe`` は ``z3.RealVal(g.decay)`` 等で **具体 gene 値を
    Z3 制約に投入**しており真の per-gene (box 流用ではない、設計 doc §2.3 表)。
    """

    name = "rwkv_state_norm"

    def __init__(
        self, *, max_input_abs: float = 1.0, state_bound: float = 1.0
    ) -> None:
        self._max_input_abs = max_input_abs
        self._state_bound = state_bound

    def verify_gene_safe(self, gene: StateUpdateGene) -> InvariantResult:
        return _verify_gene_safe(
            gene,
            max_input_abs=self._max_input_abs,
            state_bound=self._state_bound,
        )

    def is_available(self) -> bool:
        return is_z3_available()


__all__ = ["RWKVCodec", "RWKVKernel", "RWKVStateNormBackend"]
