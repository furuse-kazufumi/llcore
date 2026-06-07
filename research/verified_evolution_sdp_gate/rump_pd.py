# SPDX-License-Identifier: Apache-2.0
"""Pilot A — Rump verified positive-definiteness + OR-of-{CLARABEL,SCS} standing gate.

Purely ADDITIVE research code (research/verified_evolution_sdp_gate/). Does NOT touch src/.
Zero new dependencies (numpy only). Does NOT change the behaviour of any existing committed
certifier — it is a new, stricter, SELF-CONTAINED gate layered *beside* them.

Motivation (see ``VERIFICATION_METHODS_SURVEY_2026-06-03.md`` Pilot A,
``AUDIT_SCS_CLARABEL_2026-06-03.md``, memory ``feedback_cvxpy_pin_accurate_solver``):

  * The arc's SDP certifiers re-check the solver's claimed Lyapunov matrix ``P`` (and each
    decrease LMI ``P - J^T P J``) with a FLOATING-POINT ``np.linalg.eigvalsh`` test
    ``min_eig > 0``. Near the feasibility boundary that 1e-7-scale test is float-fragile.
  * cvxpy's default SCS solver returns FALSE NEGATIVES near the boundary; CLARABEL recovers
    them. The audit pinned CLARABEL, but a *single* solver is still a single point of failure.

This module hardens BOTH:

  1. :func:`verified_pd` — a Rump-style VERIFIED positive-definiteness test. A successful
     floating-point Cholesky of ``M - alpha*I`` together with a RIGOROUS Cholesky backward-error
     bound PROVES ``lambda_min(M) >= (alpha - error_bound) > 0``. SOUND by construction: it
     returns ``ok=True`` only when ``M`` is provably PD. Under-certifying a barely-PD ``M`` is
     acceptable; a FALSE POSITIVE (certifying a non-PD ``M``) is forbidden.

  2. :func:`or_certifies_lyapunov` — solve the common-Lyapunov SDP under EACH solver
     (CLARABEL and SCS), and accept iff ANY solver returns a ``P`` whose ``P`` AND every
     ``P - J_v^T P J_v - margin*I`` pass the Rump verified-PD test. This is an OR (not a vote):
     a vote would preserve the SCS false negative; an OR bakes "solver-swap = decisive detector"
     into a standing gate.

----------------------------------------------------------------------------------------------
THE RIGOROUS CHOLESKY ERROR BOUND (soundness derivation)
----------------------------------------------------------------------------------------------
We use the classical floating-point Cholesky backward-error result (Wilkinson; Demmel,
*Applied Numerical Linear Algebra* Thm 10.7; Higham, *Accuracy and Stability of Numerical
Algorithms* Thm 10.3 / 10.4; Rump 2006, "Verification of positive definiteness", BIT).

If the floating-point Cholesky factorization of a symmetric ``n x n`` matrix ``A`` (unit roundoff
``u``) RUNS TO COMPLETION (no non-positive pivot, no nan/inf), then there exists a perturbation
``E`` with computed factor ``R`` such that ``R^T R = A + E`` EXACTLY and

        |E_ij| <= gamma_{n+1} * sqrt(A_ii) * sqrt(A_jj),      gamma_k := k*u / (1 - k*u),

(provided ``(n+1)*u < 1``). A standard, simple-to-state consequence (and the form the survey
quotes) bounds the spectral norm of ``E`` by

        ||E||_2 <= gamma_{n+1} * max_i A_ii * n        (loose, via |E_ij| <= gamma * maxdiag)

We use the slightly tighter, still-rigorous Frobenius/2-norm bound
        ||E||_2 <= ||E||_F <= gamma_{n+1} * sum_i A_ii  ==  gamma_{n+1} * trace(A)
because ``|E_ij| <= gamma_{n+1} sqrt(A_ii A_jj)`` gives
        ||E||_F^2 = sum_ij E_ij^2 <= gamma_{n+1}^2 (sum_i sqrt(A_ii))^2 ... -> we use the
plain, unambiguously-correct envelope ``||E||_2 <= gamma_{n+1} * n * max_i A_ii`` so the proof
needs no Cauchy-Schwarz subtlety. (For our 2x2 / 3x3 matrices the difference is immaterial; we
optimise SOUNDNESS and clarity over tightness, exactly as the spec permits.)

Since ``R^T R = A + E`` is a Cholesky product it is positive SEMI-definite, so
        lambda_min(A + E) >= 0  =>  lambda_min(A) >= -||E||_2 >= -bound(A).

Apply this to ``A = M - alpha*I``: a successful Cholesky proves
        lambda_min(M - alpha*I) >= -bound(M - alpha*I)
        => lambda_min(M) >= alpha - bound(M - alpha*I).
If ``alpha - bound(M - alpha*I) > 0`` then ``M`` is PROVABLY positive definite and that quantity
is a guaranteed lower bound on ``lambda_min(M)``.

The diagonal of ``A = M - alpha*I`` is ``M_ii - alpha``. ``bound`` depends on ``max_i A_ii``; to
keep the bound a sound OVER-estimate regardless of sign we use ``max(max_i A_ii, 0)`` clamped at
the original ``max_i M_ii`` (since ``alpha > 0`` shrinks the diagonal, ``max_i(M_ii - alpha) <=
max_i M_ii``; using ``max_i M_ii`` is therefore a sound upper envelope and is what we feed the
bound — never an under-estimate). This makes the guarantee hold for every alpha we try.
"""
from __future__ import annotations

import os
import sys

import numpy as np

# IEEE-754 double unit roundoff u = 2^-53.
_U = float(np.finfo(np.float64).eps) / 2.0  # eps = 2^-52, u = 2^-53


# --------------------------------------------------------------------------- #
# 1) Rump-style verified positive-definiteness.
# --------------------------------------------------------------------------- #


def _gamma(k: int) -> float:
    """gamma_k = k*u / (1 - k*u), the standard floating-point error accumulation factor.

    Returns +inf if k*u >= 1 (bound vacuous; caller treats as 'cannot certify')."""
    ku = k * _U
    if ku >= 1.0:
        return float("inf")
    return ku / (1.0 - ku)


def _cholesky_error_bound(maxdiag: float, n: int) -> float:
    """Rigorous upper bound on ||E||_2 for the computed Cholesky factor R with R^T R = A + E,
    where ``maxdiag`` is a sound upper bound on max_i A_ii and A is n x n.

    Uses the unambiguous envelope ||E||_2 <= gamma_{n+1} * n * max_i A_ii (see module docstring).
    ``maxdiag`` is clamped to be non-negative (a non-positive maxdiag means the Cholesky pivot is
    already non-positive, so Cholesky would fail anyway; the bound is then 0 and irrelevant)."""
    md = max(float(maxdiag), 0.0)
    g = _gamma(n + 1)
    if not np.isfinite(g):
        return float("inf")
    return g * n * md


def _spd_cholesky(A: np.ndarray) -> bool:
    """True iff floating-point Cholesky of A runs to completion (A numerically SPD).

    numpy raises LinAlgError on a non-positive pivot; we also reject any non-finite entry."""
    if not np.all(np.isfinite(A)):
        return False
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


def verified_pd(M: np.ndarray, *, n_alpha: int = 60) -> tuple[bool, float]:
    """Rump-style VERIFIED positive-definiteness of symmetric ``M``.

    Returns ``(ok, lam_min_lb)``:
      * ``ok``        True  => ``M`` is PROVABLY positive definite (sound; no false positives).
                      False => could not prove PD (M may be PD-but-too-close, indefinite, or
                               non-finite). Under-certification only; never a false positive.
      * ``lam_min_lb`` a GUARANTEED lower bound on ``lambda_min(M)`` when ``ok`` is True
                       (> 0); when ``ok`` is False it is the best (possibly <= 0) lower bound
                       found, for diagnostics only.

    Method (see module docstring for the soundness proof):
      symmetrize M; for a decreasing schedule of shifts ``alpha`` attempt a floating-point
      Cholesky of ``M - alpha*I``. A successful Cholesky + the rigorous error bound proves
      ``lambda_min(M) >= alpha - bound > 0``. We try the LARGEST alpha first and return the
      first (hence largest) guaranteed positive lower bound.
    """
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError("verified_pd expects a square matrix")
    n = M.shape[0]
    # Symmetrize (the caller may pass a numerically-asymmetric matrix; the verified statement is
    # about the symmetric part, which is what PD-ness of a quadratic form depends on).
    M = 0.5 * (M + M.T)
    if not np.all(np.isfinite(M)):
        return False, float("-inf")

    maxdiag_M = float(np.max(np.diag(M))) if n > 0 else 0.0
    # bound depends only on a sound upper envelope of max diag; since every alpha we try is >= 0,
    # max_i(M_ii - alpha) <= max_i M_ii, so feeding max_i M_ii is a SOUND over-estimate for all alpha.
    err_bound = _cholesky_error_bound(maxdiag_M, n)
    if not np.isfinite(err_bound):
        return False, float("-inf")

    # Scale reference for the alpha schedule: the spectral scale of M. ||M||_2 <= sqrt(n)*||M||_F-ish;
    # use the Frobenius norm as a cheap, safe upper proxy for choosing where to probe alpha.
    scale = float(np.linalg.norm(M)) + err_bound
    if scale <= 0.0:
        scale = 1.0

    I = np.eye(n)

    # Try alpha from large (~scale) down toward the error bound. The largest alpha whose shifted
    # Cholesky succeeds gives the largest guaranteed lower bound alpha - err_bound. We MUST require
    # alpha > err_bound for a positive guarantee.
    #
    # Geometric schedule between ~scale and just above err_bound, plus a fine tail near err_bound.
    alphas = []
    hi = scale
    lo = max(err_bound * (1.0 + 1e-12), err_bound + np.finfo(np.float64).tiny)
    if hi <= lo:
        hi = lo * 4.0
    # geometric descent
    for k in range(n_alpha):
        frac = k / max(n_alpha - 1, 1)
        alphas.append(hi * (lo / hi) ** frac)
    # ensure monotone-decreasing & unique
    alphas = sorted(set(alphas), reverse=True)

    best_lb = float("-inf")
    for alpha in alphas:
        if alpha <= err_bound:
            break  # cannot yield a positive guarantee
        A = M - alpha * I
        if _spd_cholesky(A):
            lb = alpha - err_bound
            if lb > 0.0:
                return True, float(lb)
            best_lb = max(best_lb, lb)

    # No positive-guaranteeing alpha succeeded. As a last, still-sound attempt: a bare Cholesky of
    # M itself proves lambda_min(M) >= -err_bound (only useful as a diagnostic LB, NOT a PD proof).
    if _spd_cholesky(M):
        best_lb = max(best_lb, -err_bound)
    return False, float(best_lb)


# --------------------------------------------------------------------------- #
# 2) OR-of-{CLARABEL,SCS} Rump-verified Lyapunov gate.
# --------------------------------------------------------------------------- #

# Wire in the sibling SDP certifier (Track D) for the cvxpy problem + vertex Jacobians.
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.normpath(os.path.join(_HERE, ".."))
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.join(_RESEARCH, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)


def _vertex_jacobians(gene, t_domain: str = "tmin1", max_input_abs: float = 1.0):
    """The t-box vertex Jacobians J(t) for a Track-C/D CoupledGene (n=2).

    Reuses the SAME vertex construction as ``coupled_components`` / ``lyapunov_sdp_certifier`` so
    the Rump recheck verifies exactly the LMIs the SDP solver was asked to satisfy."""
    from coupled_map import t_min_per_coord
    g = gene.clipped()
    if t_domain == "free01":
        t_lo = np.zeros(2)
    elif t_domain == "tmin1":
        t_lo = t_min_per_coord(g, max_input_abs=max_input_abs)
    else:  # pragma: no cover
        raise ValueError(f"unknown t_domain {t_domain!r}")
    verts = [np.array([t_lo[0], t_lo[1]]), np.array([t_lo[0], 1.0]),
             np.array([1.0, t_lo[1]]), np.array([1.0, 1.0])]
    return [np.diag(g.decay) + np.diag((1.0 - g.decay) * v) @ g.W for v in verts]


def rump_verify_certificate(P: np.ndarray, vertex_jacobians, *, margin: float = 1e-7) -> bool:
    """VERIFIED check of a claimed common-Lyapunov certificate ``P`` over a vertex set.

    Accept iff ``P`` AND every ``(P - J^T P J - margin*I)`` are Rump-verified positive definite.
    This is the SOUND replacement for the float ``eigvalsh(...) > 0`` recheck: a True here is a
    machine-checked guarantee, not a solver claim. SOUND: no false positives (verified_pd has none).
    """
    P = np.asarray(P, dtype=np.float64)
    P = 0.5 * (P + P.T)
    ok_P, _ = verified_pd(P)
    if not ok_P:
        return False
    for J in vertex_jacobians:
        J = np.asarray(J, dtype=np.float64)
        M = P - J.T @ P @ J - margin * np.eye(P.shape[0])
        M = 0.5 * (M + M.T)
        ok_M, _ = verified_pd(M)
        if not ok_M:
            return False
    return True


def or_certifies_lyapunov(gene, t_domain: str = "tmin1",
                          solvers: tuple = ("CLARABEL", "SCS"),
                          *, margin: float = 1e-7,
                          max_input_abs: float = 1.0,
                          return_detail: bool = False):
    """OR-of-solvers Rump-verified common-Lyapunov gate.

    For EACH solver in ``solvers``: solve the common-Lyapunov SDP (reusing
    ``lyapunov_sdp_certifier.certify_common_lyapunov``), take the returned ``P``, and accept the
    solver iff ``P`` AND every ``(P - J_v^T P J_v - margin*I)`` pass :func:`verified_pd` (Rump).

    Returns True if ANY solver yields a Rump-passing certificate (OR -- NOT a vote; a vote would
    preserve the SCS false negative). Fail-closed if cvxpy is unavailable.

    If ``return_detail`` is True, returns ``(ok, detail)`` where ``detail`` is a per-solver dict
    of {'solved': bool, 'status': str, 'rump_pd': bool}.
    """
    from lyapunov_sdp_certifier import CVXPY_AVAILABLE, certify_common_lyapunov

    detail: dict = {}
    if not CVXPY_AVAILABLE:
        return (False, detail) if return_detail else False

    import cvxpy as cp
    installed = set(cp.installed_solvers())

    verts = _vertex_jacobians(gene, t_domain=t_domain, max_input_abs=max_input_abs)

    any_ok = False
    for sv in solvers:
        d = {"solved": False, "status": None, "rump_pd": False}
        if sv not in installed:
            d["status"] = "not_installed"
            detail[sv] = d
            continue
        # Solve under this specific solver. certify_common_lyapunov already does its own float
        # eigvalsh recheck, but we IGNORE its certified flag and re-verify P with Rump ourselves so
        # the gate's verdict rests on the machine-checked test, not the float one.
        r = certify_common_lyapunov(gene, t_domain=t_domain, margin=margin,
                                    max_input_abs=max_input_abs, solver=sv)
        d["status"] = r.solver_status
        if r.P is not None:
            d["solved"] = True
            P = np.asarray(r.P, dtype=np.float64)
            d["rump_pd"] = bool(rump_verify_certificate(P, verts, margin=margin))
            if d["rump_pd"]:
                any_ok = True
        detail[sv] = d

    return (any_ok, detail) if return_detail else any_ok


if __name__ == "__main__":
    # Tiny smoke test (prints only; no run-gating asserts).
    rng = np.random.default_rng(0)
    A = rng.standard_normal((3, 3))
    spd = A @ A.T + 3.0 * np.eye(3)
    ok, lb = verified_pd(spd)
    print("clearly-PD:", ok, "lb=", lb, "true min eig=", float(np.min(np.linalg.eigvalsh(spd))))
    indef = np.diag([1.0, -1e-6, 2.0])
    ok2, lb2 = verified_pd(indef)
    print("indefinite:", ok2, "lb=", lb2)

    try:
        from coupled_map import CoupledGene
        g = CoupledGene.make(decay=[0.3, 0.3], W=[[0.5, 0.9], [0.9, 0.5]])
        ok3, det = or_certifies_lyapunov(g, return_detail=True)
        print("or_certifies_lyapunov:", ok3)
        print("detail:", det)
    except Exception as exc:  # pragma: no cover
        print("gene smoke skipped:", repr(exc))
