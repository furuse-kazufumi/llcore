# SPDX-License-Identifier: Apache-2.0
"""Track D — common quadratic Lyapunov (SDP/LMI) contraction certificate via cvxpy.

Purely ADDITIVE research code (research/spectral_lyapunov_contraction/). Does NOT touch src/.

This is the TIGHTEST quadratic certificate for the affine-in-t Jacobian family. We seek a single
common Lyapunov matrix P >> 0 such that, at every t-box VERTEX J_v,
      J_v^T P J_v - P << 0       (discrete-time quadratic stability LMI).
Because J(t) is AFFINE in t over the box, the vertex set is the set of extreme matrices; quadratic
stability at all vertices implies stability for every J in the convex hull (the whole achievable-t
box), by convexity of the quadratic form V(z)=z^T P z and the standard polytopic-LMI argument.

If such P exists, the map is contractive in the P-weighted norm ||z||_P = sqrt(z^T P z): there is a
rate gamma < 1 with ||J_v z||_P <= gamma ||z||_P for all vertices, hence for all achievable J, hence
rho(J) < 1 over the box. This is a strictly RICHER class than any single induced norm (||.||_2 is the
special case P = I; the SDP can find a non-identity P that the 2-norm vertex test cannot).

SOLVER territory: this is a genuine SDP/LMI and needs cvxpy. If cvxpy is not importable, this module
degrades gracefully: CVXPY_AVAILABLE = False and certify_common_lyapunov(...) returns
available=False (NEVER raises, NEVER fabricates a verdict).
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_TRACK_C = os.path.normpath(os.path.join(_HERE, "..", "coupled_z3_contraction"))
if _TRACK_C not in sys.path:
    sys.path.insert(0, _TRACK_C)

from coupled_map import CoupledGene, t_min_per_coord  # noqa: E402

try:
    import cvxpy as cp  # noqa: F401

    CVXPY_AVAILABLE = True
    _CVXPY_VERSION = getattr(cp, "__version__", "unknown")
    _CVXPY_IMPORT_ERROR = None
    # FAIL-CLOSED default solver. cvxpy's bare default is SCS (first-order), which FALSE-NEGATIVES
    # near the SDP feasibility boundary (the 2026-06-03 audit artifact). We pin the accurate
    # CLARABEL as the default. CRITICALLY, if CLARABEL is NOT installed we do NOT fall back to
    # ``_DEFAULT_SOLVER = None`` (== cvxpy default == SCS): that would silently regress to the
    # artifact-prone solver. Instead ``_CLARABEL_OK`` is False, and when a caller does NOT pass an
    # explicit ``solver=`` the function REFUSES (returns available=False / certified=None) rather
    # than running under SCS. A caller may STILL pass ``solver="SCS"`` EXPLICITLY (the PART-2
    # OR-of-{CLARABEL,SCS} gate does this, then re-verifies the result with the Rump verified-PD
    # test) — an explicit choice is the caller's responsibility and is honoured. See memory
    # feedback_cvxpy_pin_accurate_solver.
    _CLARABEL_OK = "CLARABEL" in cp.installed_solvers()
    _DEFAULT_SOLVER = "CLARABEL" if _CLARABEL_OK else None
except Exception as exc:  # pragma: no cover - environment dependent
    CVXPY_AVAILABLE = False
    _CVXPY_VERSION = None
    _CVXPY_IMPORT_ERROR = repr(exc)
    _CLARABEL_OK = False
    _DEFAULT_SOLVER = None


def _box_vertices(t_lo: np.ndarray) -> list[np.ndarray]:
    t_lo = np.asarray(t_lo, dtype=np.float64).reshape(2)
    return [
        np.array([t_lo[0], t_lo[1]]),
        np.array([t_lo[0], 1.0]),
        np.array([1.0, t_lo[1]]),
        np.array([1.0, 1.0]),
    ]


def _jac_at_t(gene: CoupledGene, t: np.ndarray) -> np.ndarray:
    g = gene.clipped()
    t = np.asarray(t, dtype=np.float64).reshape(2)
    return np.diag(g.decay) + np.diag((1.0 - g.decay) * t) @ g.W


@dataclass(frozen=True)
class LyapResult:
    available: bool                 # was cvxpy importable AND did the solve run
    certified: bool | None          # True/False if available; None if not available / solver error
    domain: str
    solver_status: str | None       # cvxpy problem status (e.g. "optimal", "infeasible") or note
    P: list | None                  # the found P (2x2) as nested list, if certified
    min_eig_margin: float | None    # min eigenvalue of (P - J_v^T P J_v) over vertices (should be > 0)


def certify_common_lyapunov(gene: CoupledGene, t_domain: str = "tmin1",
                            max_input_abs: float = 1.0,
                            margin: float = 1e-7,
                            solver: str | None = None) -> LyapResult:
    """Solve the common-P vertex-LMI feasibility SDP. Degrades gracefully if cvxpy is absent.

    Feasibility encoded as an SDP: find P with P >> margin*I and
        P - J_v^T P J_v >> margin*I   at every vertex v.
    (Strict LMIs approximated by a fixed positive margin; reported in min_eig_margin.)
    """
    if not CVXPY_AVAILABLE:
        return LyapResult(available=False, certified=None, domain=t_domain,
                          solver_status="cvxpy_unavailable", P=None, min_eig_margin=None)

    # FAIL-CLOSED: if the caller did not pick a solver AND CLARABEL is not installed, REFUSE rather
    # than letting the solve fall through to cvxpy's SCS default (the artifact-prone first-order
    # solver). An EXPLICIT ``solver=`` from the caller is always honoured (e.g. the OR-of-solvers
    # gate that deliberately probes SCS and then Rump-re-verifies the result).
    if solver is None and not _CLARABEL_OK:
        return LyapResult(available=False, certified=None, domain=t_domain,
                          solver_status="clarabel_unavailable_fail_closed",
                          P=None, min_eig_margin=None)

    import cvxpy as cp

    g = gene.clipped()
    if t_domain == "free01":
        t_lo = np.zeros(2)
    elif t_domain == "tmin1":
        t_lo = t_min_per_coord(g, max_input_abs=max_input_abs)
    else:  # pragma: no cover
        raise ValueError(f"unknown t_domain {t_domain!r}")

    verts = _box_vertices(t_lo)
    Js = [_jac_at_t(g, v) for v in verts]

    P = cp.Variable((2, 2), symmetric=True)
    I2 = np.eye(2)
    # The LMI system is HOMOGENEOUS in P (scale-invariant), so a bare feasibility problem is
    # ill-posed: the conic solver treats the strict `>> margin*I` as `>=` and can return the
    # trivial P=0 (which our eigenvalue re-check would then reject) or report spurious infeasible
    # when paired with an upper cap. Fix the scale with `P >> I` (forces PD AND normalizes), and
    # minimize trace(P) to pull P toward the smallest valid Lyapunov matrix. This is well-posed.
    constraints = [P >> I2]
    for J in Js:
        # P - J^T P J >> margin I  (discrete Lyapunov decrease at this vertex)
        constraints.append(P - J.T @ P @ J >> margin * I2)
    prob = cp.Problem(cp.Minimize(cp.trace(P)), constraints)

    try:
        _eff_solver = solver if solver is not None else _DEFAULT_SOLVER  # fail-safe to CLARABEL
        if not _eff_solver:
            # FAIL-CLOSED: a FALSY solver value (e.g. an empty-string ``solver=""`` from a caller,
            # which slips past the ``solver is None`` guard above, or _DEFAULT_SOLVER=None when
            # CLARABEL is absent) must NEVER fall through to cvxpy's bare default (SCS, the
            # artifact-prone first-order solver that false-negatives near the feasibility boundary).
            # Refuse rather than run the bare ``prob.solve()``. An explicit accurate solver is the
            # only way to run the SDP. See memory feedback_cvxpy_pin_accurate_solver.
            return LyapResult(available=False, certified=None, domain=t_domain,
                              solver_status="no_accurate_solver_fail_closed",
                              P=None, min_eig_margin=None)
        prob.solve(solver=_eff_solver)
    except Exception as exc:  # solver hiccup -> report, do not crash the sweep
        return LyapResult(available=True, certified=None, domain=t_domain,
                          solver_status=f"solver_error:{type(exc).__name__}", P=None,
                          min_eig_margin=None)

    status = prob.status
    if status in ("optimal", "optimal_inaccurate") and P.value is not None:
        Pv = np.asarray(P.value, dtype=np.float64)
        Pv = 0.5 * (Pv + Pv.T)  # symmetrize numerically
        # Verify the certificate independently of the solver's claim.
        eig_P = float(np.min(np.linalg.eigvalsh(Pv)))
        min_dec = np.inf
        for J in Js:
            M = Pv - J.T @ Pv @ J
            M = 0.5 * (M + M.T)
            min_dec = min(min_dec, float(np.min(np.linalg.eigvalsh(M))))
        # Certified only if P is genuinely PD and every decrease LMI is genuinely PD.
        ok = (eig_P > 0.0) and (min_dec > 0.0)
        return LyapResult(available=True, certified=bool(ok), domain=t_domain,
                          solver_status=status, P=Pv.tolist(),
                          min_eig_margin=float(min(eig_P, min_dec)))
    else:
        # infeasible / unbounded / solver_error etc. -> no common quadratic P found
        return LyapResult(available=True, certified=False, domain=t_domain,
                          solver_status=status, P=None, min_eig_margin=None)


def availability_report() -> dict:
    return {
        "cvxpy_available": CVXPY_AVAILABLE,
        "cvxpy_version": _CVXPY_VERSION,
        "import_error": _CVXPY_IMPORT_ERROR,
    }


if __name__ == "__main__":
    print("availability:", availability_report())
    g = CoupledGene.make(decay=[0.3, 0.3], W=[[0.5, 0.9], [0.9, 0.5]])
    for dom in ("free01", "tmin1"):
        r = certify_common_lyapunov(g, t_domain=dom)
        print(f"[{dom}] available={r.available} certified={r.certified} status={r.solver_status} "
              f"min_eig_margin={r.min_eig_margin}")
