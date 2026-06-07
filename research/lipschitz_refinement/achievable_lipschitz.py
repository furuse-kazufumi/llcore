# SPDX-License-Identifier: Apache-2.0
"""Track A — Achievable-t Lipschitz refinement (ADDITIVE; does not edit src).

The existing certifier in ``llcore.verifier.invariants.verify_lipschitz_contraction``
proves state-direction contraction ``L < 1`` by freeing the curvature term
``t = sech^2(pre)`` over the OVER-approximating interval ``[0, 1]``. Its own docstring
flags an "achievable-t refinement (t_min = sech^2(|mix|+|gate_str|))" as FUTURE WORK.

This module implements exactly that:

THE MATH
--------
Local state Jacobian:
    J(s,x) = ds'/ds = decay + (1-decay)*gate_str*t,   t = sech^2(pre), pre = mix*x + gate_str*s

Over the box ``|s| <= 1, |x| <= 1``, ``pre`` ranges exactly over ``[-M, M]`` with
``M = |mix| + |gate_str|`` (pre is linear in s,x; corners attain extremes). ``sech^2`` is
even and strictly decreasing in ``|pre|``, so the ACHIEVABLE set of ``t`` is exactly
``[t_min, 1]`` with ``t_min = sech^2(M) = 1 - tanh^2(M)``. (``t = 1`` is attained at
``pre = 0``, always reachable e.g. ``s = x = 0``.)

Since J is affine (monotone) in t, the EXACT local Lipschitz constant over the box is
    L_achievable = max(|J(t_min)|, |J(1)|)
                 = max(|decay + (1-decay)*gate_str*t_min|, |decay + (1-decay)*gate_str|).

The free-t bound used ``max`` over ``t in [0,1]`` (endpoints 0 and 1), which includes the
unreachable ``t < t_min`` and is therefore conservative (>= L_achievable).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# Reuse the EXISTING free-t certifier from llcore for comparison. Do NOT edit src.
from llcore.state_update import StateUpdateGene
from llcore.verifier.invariants import (
    LipschitzResult,
    _lipschitz_upper_bound,  # free-t closed-form: max(|decay|, |decay+(1-decay)*gate_str|)
    is_z3_available,
    verify_lipschitz_contraction,  # free-t Z3 certifier (for comparison)
)

try:
    import z3

    _HAS_Z3 = True
except ImportError:  # pragma: no cover
    _HAS_Z3 = False


# ---------------------------------------------------------------------------
# Closed-form achievable-t bound
# ---------------------------------------------------------------------------
def t_min_achievable(mix: float, gate_str: float) -> float:
    """Exact lower bound of the achievable curvature ``t = sech^2(pre)`` over the box.

    ``t_min = sech^2(M)`` with ``M = |mix| + |gate_str|``.
    Computed as ``1 - tanh(M)^2`` (numerically identical to ``sech^2`` but avoids overflow
    of ``cosh`` for large M; for the clipped box M <= 3 so either form is fine).
    """
    m = abs(mix) + abs(gate_str)
    return 1.0 - math.tanh(m) ** 2


def _jacobian(decay: float, gate_str: float, t: float) -> float:
    """J(t) = decay + (1 - decay) * gate_str * t (affine in t)."""
    return decay + (1.0 - decay) * gate_str * t


def L_achievable(gene: StateUpdateGene) -> float:
    """Closed-form EXACT box Lipschitz constant under achievable-t.

    L_achievable = max(|J(t_min)|, |J(1)|).
    """
    g = gene.clipped()
    tmin = t_min_achievable(g.mix, g.gate_str)
    j_tmin = _jacobian(g.decay, g.gate_str, tmin)
    j_one = _jacobian(g.decay, g.gate_str, 1.0)
    return max(abs(j_tmin), abs(j_one))


def L_free(gene: StateUpdateGene) -> float:
    """Free-t closed-form bound (reuse llcore helper): max(|decay|, |decay+(1-decay)*gate_str|)."""
    g = gene.clipped()
    return _lipschitz_upper_bound(g.decay, g.gate_str)


# ---------------------------------------------------------------------------
# Z3 achievable-t certifier
# ---------------------------------------------------------------------------
def verify_lipschitz_contraction_achievable(
    gene: StateUpdateGene,
    *,
    timeout_ms: int = 1000,
) -> LipschitzResult:
    """Z3 certifier that constrains ``t in [t_min, 1]`` (achievable set) instead of ``[0,1]``.

    Adds the constraint ``t >= t_min`` where ``t_min`` is computed in float64 from
    ``sech^2(|mix| + |gate_str|)`` and injected as ``z3.RealVal(t_min_float)`` (a rational).
    Searches for ``|J| >= 1``; ``unsat`` => certified (``L < 1`` over the achievable box).

    Mirrors the structure of the existing free-t certifier
    (:func:`llcore.verifier.invariants.verify_lipschitz_contraction`) but with the tighter
    lower bound on ``t``. ``L_upper_bound`` is populated with the achievable closed-form
    ``L_achievable`` (NOT the free bound) so callers can cross-check the Z3 verdict.

    fail-closed: timeout / unknown -> reject.
    """
    g = gene.clipped()
    l_ach = L_achievable(g)
    tmin = t_min_achievable(g.mix, g.gate_str)

    if not _HAS_Z3:
        # fail-safe: undecided without z3; expose closed-form bound but mark unverified.
        return LipschitzResult(
            contraction=None,
            L_upper_bound=l_ach,
            used_z3=False,
            solver_status="unknown",
            reason=(
                "z3 not installed: achievable-t contraction undecided "
                f"(closed-form L_achievable<={l_ach:.6f}; treat used_z3=False as unverified)"
            ),
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    t = z3.Real("t")
    # ACHIEVABLE refinement: t in [t_min, 1] (vs free-t [0, 1]).
    # float -> rational: t_min computed in float64, fed to z3 as exact RealVal.
    solver.add(t >= z3.RealVal(tmin), t <= 1)

    d = z3.RealVal(g.decay)
    gate = z3.RealVal(g.gate_str)
    j = d + (1 - d) * gate * t

    solver.add(z3.Or(j >= 1, j <= -1))

    result = solver.check()
    if result == z3.unsat:
        return LipschitzResult(
            contraction=True,
            L_upper_bound=l_ach,
            used_z3=True,
            solver_status="unsat",
            reason=(
                f"unsat: sup|J|<1 certified over achievable t in [{tmin:.6f}, 1] "
                f"(L_achievable<={l_ach:.6f}<1) for "
                f"d={g.decay:.3f}, m={g.mix:.3f}, g={g.gate_str:.3f}"
            ),
        )
    if result == z3.sat:
        return LipschitzResult(
            contraction=False,
            L_upper_bound=l_ach,
            used_z3=True,
            solver_status="sat",
            reason=(
                f"sat: |J|>=1 reachable for achievable t in [{tmin:.6f}, 1] "
                f"(L_achievable>={l_ach:.6f}; conservative reject) for "
                f"d={g.decay:.3f}, m={g.mix:.3f}, g={g.gate_str:.3f}"
            ),
        )
    return LipschitzResult(
        contraction=False,
        L_upper_bound=l_ach,
        used_z3=True,
        solver_status="unknown",
        reason=f"z3 returned {result} (timeout/unknown) — fail-closed reject",
    )


# Re-export the free-t certifier under a clear name for experiment scripts.
verify_lipschitz_contraction_free = verify_lipschitz_contraction


__all__ = [
    "t_min_achievable",
    "L_achievable",
    "L_free",
    "verify_lipschitz_contraction_achievable",
    "verify_lipschitz_contraction_free",
    "is_z3_available",
]


if __name__ == "__main__":
    # Smoke check: a few hand genes.
    samples = [
        StateUpdateGene(decay=0.5, mix=0.5, gate_str=1.0),
        StateUpdateGene(decay=0.9, mix=1.0, gate_str=2.0),
        StateUpdateGene(decay=0.0, mix=0.0, gate_str=0.0),
        StateUpdateGene(decay=0.2, mix=-1.0, gate_str=-2.0),
    ]
    print("z3 available:", is_z3_available())
    for s in samples:
        g = s.clipped()
        lf = L_free(g)
        la = L_achievable(g)
        free = verify_lipschitz_contraction_free(g)
        ach = verify_lipschitz_contraction_achievable(g)
        print(
            f"d={g.decay:.2f} m={g.mix:.2f} g={g.gate_str:.2f} | "
            f"L_free={lf:.4f} L_ach={la:.4f} | "
            f"free_cert={free.contraction} ach_cert={ach.contraction} | "
            f"t_min={t_min_achievable(g.mix, g.gate_str):.4f}"
        )
