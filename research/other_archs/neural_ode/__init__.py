# SPDX-License-Identifier: Apache-2.0
"""Neural ODE / LTC への llcore approach 移植 PoC.

llcore RWKV-style verified evolution を **連続時間 vector field** に移植する
研究 PoC。core algorithm gene 化 + Z3 invariant (Lipschitz ∧ Hurwitz) +
進化 + open-ended (適応難易度 + 中立貯蔵庫 + MODES 計器) の 4 機構が連続時間
ODE でも機能するかを検証する。

公開 API:
    NeuralODEGene, vector_field, forward_euler, empirical_lipschitz
    verify_lipschitz_bound, verify_gene_ode_safe (+ helpers)
    EvolutionTrace, run_neural_ode_evolution (poc.py から)

llive 非依存、llcore.evolution.* のみ依存 (自前 module).
"""
from .ode_gene import (
    A_HIGH,
    A_LOW,
    B_HIGH,
    B_LOW,
    DEFAULT_DIM,
    DEFAULT_DT,
    DEFAULT_N_STEP,
    DEFAULT_T,
    NeuralODEGene,
    W_HIGH,
    W_LOW,
    empirical_lipschitz,
    forward_euler,
    vector_field,
)
from .ode_verifier import (
    ODEInvariantResult,
    is_z3_available,
    verify_gene_hurwitz,
    verify_gene_lipschitz,
    verify_gene_ode_safe,
    verify_hurwitz_universal,
    verify_lipschitz_bound,
)

__all__ = [
    "A_HIGH",
    "A_LOW",
    "B_HIGH",
    "B_LOW",
    "DEFAULT_DIM",
    "DEFAULT_DT",
    "DEFAULT_N_STEP",
    "DEFAULT_T",
    "NeuralODEGene",
    "ODEInvariantResult",
    "W_HIGH",
    "W_LOW",
    "empirical_lipschitz",
    "forward_euler",
    "is_z3_available",
    "vector_field",
    "verify_gene_hurwitz",
    "verify_gene_lipschitz",
    "verify_gene_ode_safe",
    "verify_hurwitz_universal",
    "verify_lipschitz_bound",
]
