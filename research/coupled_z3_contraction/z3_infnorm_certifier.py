# SPDX-License-Identifier: Apache-2.0
"""Track C — Z3 induced-inf-norm contraction certifier for the COUPLED n=2 map.

Certifies the SUFFICIENT, SMT-encodable condition::

    ||J||_inf = max_i sum_j |J_ij| < 1   over the box (free t in [0,1]^2, or [t_min,1]^2)

by searching for a counterexample t in the box such that SOME row's abs-sum >= 1. If Z3
returns ``unsat`` the gene is CERTIFIED ( ||J||_inf < 1 over the whole box => contraction in
inf-norm => unique fixed point + bounded state, Banach). ``sat`` => conservative reject (an
admissible t makes a row abs-sum reach 1). ``unknown``/timeout => fail-closed reject.

Why this is SOUND (over-approximation argument):
  t_i = sech^2(pre_i) is genuinely in (0,1], and the achievable set over the box is a proper
  closed subset. Freeing t in [0,1]^2 (or the tighter [t_min,1]^2) STRICTLY CONTAINS the
  achievable set, and each row abs-sum is monotone-increasing in t_i for the off-diagonal term
  and V-shaped (piecewise linear) for the diagonal term => the sup over the larger free box
  dominates the sup over the achievable set. Hence ``no row reaches 1 over free t`` (unsat) =>
  ``no row reaches 1 over achievable t`` => truly ||J||_inf < 1. (sat may use a non-achievable t
  => conservative, false-reject allowed, consistent with fail-closed discipline.)

This is the contrast with the prior DIAGONAL result: there, ||J||_inf collapses to a single
per-coordinate |J_ii| inequality and Z3 was provably identical to a closed-form scalar test
("decorative"). Here the row abs-sum mixes the diagonal AND off-diagonal magnitudes coupled
through the SAME t_i and the SAME (1-decay_i) factor, and the inf-norm takes a max over rows --
a genuinely multi-term feasibility query. We still cross-check against the closed-form endpoint
value (coupled_map.infnorm_over_box_freeT) for an honest "is Z3 doing anything the closed form
can't" audit.

Honest note on decorativeness (pre-committed): even for the coupled inf-norm, because each row's
abs-sum is piecewise-linear in t_i, the sup is attained at endpoints t_i in {t_lo_i, 1}, so a
closed-form endpoint enumeration ALSO computes ||J||_inf exactly. So Z3 is again not strictly
*necessary* for THIS particular sufficient condition. What changes vs the diagonal case is that
the SCALAR (diagonal-only) heuristic is now UNSOUND (C2) -- the value Z3 adds is over the SCALAR
heuristic, not over a full closed-form inf-norm computation. We report this distinction
explicitly rather than overclaiming "Z3 is now load-bearing in an absolute sense".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    import z3

    _HAS_Z3 = True
except ImportError:  # pragma: no cover
    _HAS_Z3 = False

from coupled_map import CoupledGene, t_min_per_coord


@dataclass(frozen=True)
class InfNormResult:
    """Result of the Z3 induced-inf-norm contraction certifier."""

    certified: bool | None  # True = unsat (||J||_inf<1), False = sat/timeout reject, None = no z3
    used_z3: bool
    solver_status: str  # "unsat" | "sat" | "unknown"
    t_domain: str  # "free01" | "tmin1"
    reason: str
    counterexample_t: list | None = None


def _abs_z3(expr):
    """|expr| as a Z3 expression."""
    return z3.If(expr >= 0, expr, -expr)


def certify_infnorm_contraction(
    gene: CoupledGene,
    *,
    t_domain: str = "free01",
    max_input_abs: float = 1.0,
    one_threshold: float = 1.0,
    timeout_ms: int = 2000,
) -> InfNormResult:
    """Z3-certify ||J(t)||_inf < one_threshold over t in the chosen box.

    Parameters
    ----------
    t_domain : "free01" | "tmin1"
        "free01": t_i in [0,1] (loosest sound over-approx).
        "tmin1" : t_i in [t_min_i, 1] (tighter sound floor from |pre_i| max over the box).
    one_threshold : float
        The contraction threshold (1.0 = strict contraction). The counterexample search is
        ``OR_i row_abs_sum_i >= one_threshold``; unsat => all rows < threshold.
    """
    g = gene.clipped()

    if t_domain == "tmin1":
        t_lo = t_min_per_coord(g, max_input_abs=max_input_abs)
    elif t_domain == "free01":
        t_lo = np.zeros(2)
    else:  # pragma: no cover
        raise ValueError(f"unknown t_domain {t_domain!r}")

    if not _HAS_Z3:  # pragma: no cover
        return InfNormResult(
            certified=None,
            used_z3=False,
            solver_status="unknown",
            t_domain=t_domain,
            reason="z3 not installed: contraction undecided (fail-closed: treat as unverified)",
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    t = [z3.Real(f"t{i}") for i in range(2)]
    for i in range(2):
        solver.add(t[i] >= z3.RealVal(float(t_lo[i])), t[i] <= 1)

    d = [z3.RealVal(float(g.decay[i])) for i in range(2)]
    W = [[z3.RealVal(float(g.W[i, j])) for j in range(2)] for i in range(2)]

    # Row abs-sums.  Row i: |J_ii| + |J_ij|, J_ii = d_i + (1-d_i)*t_i*W_ii, J_ij=(1-d_i)*t_i*W_ij
    row_sums = []
    for i in range(2):
        j = 1 - i
        one_minus_d = 1 - d[i]
        J_ii = d[i] + one_minus_d * t[i] * W[i][i]
        J_ij = one_minus_d * t[i] * W[i][j]
        row_sums.append(_abs_z3(J_ii) + _abs_z3(J_ij))

    thr = z3.RealVal(float(one_threshold))
    solver.add(z3.Or(*[rs >= thr for rs in row_sums]))

    result = solver.check()
    if result == z3.unsat:
        return InfNormResult(
            certified=True,
            used_z3=True,
            solver_status="unsat",
            t_domain=t_domain,
            reason=(
                f"unsat: ||J||_inf < {one_threshold} certified over t in [{t_domain}] "
                "(inf-norm contraction => unique fixed point + bounded state)"
            ),
        )
    if result == z3.sat:
        m = solver.model()

        def _f(var):
            v = m.eval(var, model_completion=True)
            try:
                return float(v.as_decimal(12).rstrip("?"))
            except Exception:
                return float(v.numerator_as_long()) / float(v.denominator_as_long())

        ce = [_f(t[i]) for i in range(2)]
        return InfNormResult(
            certified=False,
            used_z3=True,
            solver_status="sat",
            t_domain=t_domain,
            reason=(
                f"sat: some row abs-sum >= {one_threshold} reachable at t={ce} "
                "(no inf-norm contraction certified; conservative reject)"
            ),
            counterexample_t=ce,
        )
    return InfNormResult(
        certified=False,
        used_z3=True,
        solver_status="unknown",
        t_domain=t_domain,
        reason=f"z3 returned {result} (timeout/unknown) -- fail-closed reject",
    )


if __name__ == "__main__":
    # Smoke: a clearly expansive coupled gene (strong symmetric coupling) and a clearly safe one.
    expansive = CoupledGene.make(decay=[0.3, 0.3], W=[[0.5, 0.9], [0.9, 0.5]])
    safe = CoupledGene.make(decay=[0.9, 0.9], W=[[0.1, 0.05], [0.05, 0.1]])
    for name, gn in [("expansive", expansive), ("safe", safe)]:
        for dom in ("free01", "tmin1"):
            r = certify_infnorm_contraction(gn, t_domain=dom)
            print(f"{name:10s} [{dom}] certified={r.certified} status={r.solver_status} :: {r.reason}")
