# SPDX-License-Identifier: Apache-2.0
"""Track C — the DIAGONAL-ONLY scalar contraction heuristic (the prior diagonal check).

This is the heuristic the two prior tracks (A/B) used: it inspects ONLY the diagonal Jacobian
entry of each coordinate and IGNORES the off-diagonal coupling W_ij. For a coupled (non-diagonal)
map this is UNSOUND -- it can ADMIT a gene whose coupled inf-norm (and even spectral radius) is
>= 1, i.e. an actually-EXPANSIVE map.

Per-coordinate diagonal Jacobian (ignoring coupling)::

    J_ii(t_i) = decay_i + (1 - decay_i)*t_i*W_ii,   t_i in [0,1] (or [t_min_i,1])

Closed-form sup over t_i in {t_lo_i, 1}::

    L_ii = max(|decay_i|, |decay_i + (1-decay_i)*W_ii|)        (free t in [0,1])

The scalar heuristic ADMITS iff max_i L_ii < 1. This is IDENTICAL in form to the diagonal-map
certifier the prior tracks proved Z3-decorative for: it is a closed-form scalar inequality, no
SMT discrimination. We deliberately reuse exactly that logic so the C2 contrast is apples-to-
apples: the only thing the scalar test omits vs the Z3 inf-norm certifier is the off-diagonal
abs-term (1-decay_i)*t_i*|W_ij| in each row sum.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from coupled_map import CoupledGene, t_min_per_coord


@dataclass(frozen=True)
class ScalarResult:
    """Diagonal-only scalar heuristic result."""

    admit: bool
    scalar_L: float  # max_i sup_t |J_ii|  (diagonal only)
    per_coord_L: list  # [L_00, L_11]
    t_domain: str
    reason: str


def scalar_diagonal_admit(
    gene: CoupledGene,
    *,
    t_domain: str = "free01",
    max_input_abs: float = 1.0,
    one_threshold: float = 1.0,
) -> ScalarResult:
    """Diagonal-only contraction admit: max_i sup_{t_i} |J_ii(t_i)| < one_threshold.

    IGNORES off-diagonal W_ij (the coupling). UNSOUND for coupled maps by construction; this is
    the point of the C2 comparison.
    """
    g = gene.clipped()
    if t_domain == "tmin1":
        t_lo = t_min_per_coord(g, max_input_abs=max_input_abs)
    elif t_domain == "free01":
        t_lo = np.zeros(2)
    else:  # pragma: no cover
        raise ValueError(f"unknown t_domain {t_domain!r}")

    per_coord = []
    for i in range(2):
        # |J_ii| is V-shaped in t_i (affine) -> sup at endpoints {t_lo_i, 1}.
        vals = []
        for ti in (t_lo[i], 1.0):
            vals.append(abs(g.decay[i] + (1.0 - g.decay[i]) * ti * g.W[i, i]))
        per_coord.append(float(max(vals)))

    scalar_L = float(max(per_coord))
    admit = scalar_L < one_threshold
    return ScalarResult(
        admit=admit,
        scalar_L=scalar_L,
        per_coord_L=per_coord,
        t_domain=t_domain,
        reason=(
            f"{'admit' if admit else 'reject'}: diagonal-only sup|J_ii| = {scalar_L:.6f} "
            f"{'<' if admit else '>='} {one_threshold} (off-diagonal coupling IGNORED)"
        ),
    )


if __name__ == "__main__":
    expansive = CoupledGene.make(decay=[0.3, 0.3], W=[[0.5, 0.9], [0.9, 0.5]])
    safe = CoupledGene.make(decay=[0.9, 0.9], W=[[0.1, 0.05], [0.05, 0.1]])
    for name, gn in [("expansive", expansive), ("safe", safe)]:
        r = scalar_diagonal_admit(gn)
        print(f"{name:10s} admit={r.admit} scalar_L={r.scalar_L:.4f} :: {r.reason}")
