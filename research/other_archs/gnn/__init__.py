# SPDX-License-Identifier: Apache-2.0
"""GNN への llcore approach 移植 PoC (research/other_archs/gnn).

llcore 本体 (src/llcore) には触れず、research/ 配下で隔離して実装する。
llcore.evolution.* (AdaptiveFloorGate / LineageReservoir / ModesMeter) は
import OK、それ以外の src/llcore 依存は最小化。

falsifiable 命題:
    GNN の message passing op (aggregation 重み α_sum/α_mean/α_max + update MLP の
    affine 係数) を低次元 gene 化し、Z3 で permutation equivariance +
    over-smoothing lower bound invariant を per-gene 検査することで、
    llcore approach が構造変化 ChangeOp を扱える mechanism を実証
    (CPU 完結, 32 個体 × 50 世代).
"""
from __future__ import annotations

from .gnn_gene import GnnGene, aggregate, forward_layer, update_node
from .gnn_verifier import (
    is_z3_available,
    verify_equivariance_structure,
    verify_oversmoothing_lower_bound,
)

__all__ = [
    "GnnGene",
    "aggregate",
    "forward_layer",
    "update_node",
    "is_z3_available",
    "verify_equivariance_structure",
    "verify_oversmoothing_lower_bound",
]
