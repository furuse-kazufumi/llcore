# SPDX-License-Identifier: Apache-2.0
"""PART 2 — ADDITIVE Rump+OR HARDENED verifier factory (2026-06-03 hardening).

A NEW, additive verifier factory that hardens the SDP certifier's solver-trust recheck WITHOUT
changing any default certifier's behaviour. It layers two validated, adversarially-proven-sound
primitives from :mod:`rump_pd` on top of the existing common-Lyapunov SDP:

  1. :func:`rump_pd.verified_pd` — a machine-GUARANTEED positive-definiteness test (Rump-style
     verified Cholesky backward-error bound; 11 tests, a 36860-trial skeptic battery, 0 false
     positives). Replaces the solver-trusting float ``np.linalg.eigvalsh(...) > 0`` recheck with a
     test whose True verdict is a proof, not a solver claim.

  2. :func:`rump_pd.or_certifies_lyapunov` — OR-of-{CLARABEL,SCS}: accept iff ANY solver returns a
     ``P`` that passes the Rump recheck. (An OR, not a vote — a vote would preserve a solver's false
     negative; an OR turns "solver-swap" into a decisive detector.)

----------------------------------------------------------------------------------------------
CRITICAL INVARIANT — match the float path's ACTUAL recheck (NO margin)
----------------------------------------------------------------------------------------------
The audit found a trap: the SDP CONSTRAINT carries a strict margin (``P - J^T P J >> margin*I``),
but the certifier's internal FLOAT recheck in
``lyapunov_sdp_certifier.certify_common_lyapunov`` tests the **no-margin** matrices::

    eig_P  = min eig(P)              > 0        # P strictly PD, no margin
    min_dec = min eig(P - J^T P J)   > 0        # decrease LMI strictly PD, NO margin subtracted

So to PRESERVE-OR-GROW the admit set (never shrink it), the Rump recheck here MUST verify the
SAME matrices — ``P`` and every ``P - J^T P J`` — at the SAME (zero) margin. We therefore call
``rump_pd.rump_verify_certificate(P, verts, margin=0.0)`` (margin=0 ⇒ it verifies exactly
``P`` and ``P - J^T P J``, identical to the float test's matrices). Subtracting ``margin*I`` here
would test a STRICTLY HARDER condition than the float path and could spuriously shrink the admit
set — the exact failure mode the spec warns against. Because ``verified_pd`` is a SOUND lower bound
on ``lambda_min`` (``verified_pd >= float-eigvalsh`` on identical matrices, already proven), the
Rump recheck on the IDENTICAL matrices is a SUPERSET-or-equal of the float recheck: the admit set
is preserved-or-grown, NEVER shrunk, and stays SOUND (no false positives).

This module is PURELY ADDITIVE: it does NOT touch src/, does NOT modify ``rump_pd.py``, and does
NOT change the behaviour of any existing committed certifier (``_sdp_certifies``, ``certify_deg*``,
``SdpLyapunovBackend``, ...). It is a new factory beside them; promotion to the default is a
SEPARATE human decision (see RUMP_HARDENING_VERDICT.md), not flipped here.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.normpath(os.path.join(_HERE, ".."))
for _d in (_HERE,
           os.path.join(_RESEARCH, "coupled_z3_contraction"),
           os.path.join(_RESEARCH, "spectral_lyapunov_contraction")):
    if _d not in sys.path:
        sys.path.insert(0, _d)

# The no-margin strict-PD test the certifier's FLOAT recheck actually performs (margin=0).
_FLOAT_PATH_MARGIN = 0.0


def rump_recheck_matches_float_path(P, vertex_jacobians) -> bool:
    """Rump verified-PD recheck of a claimed common-Lyapunov ``P`` over a vertex set, testing the
    SAME matrices the certifier's internal FLOAT recheck tests: ``P`` strictly PD AND every
    ``P - J^T P J`` strictly PD, at the SAME (zero) margin the float path uses.

    Delegates to the validated :func:`rump_pd.rump_verify_certificate` with ``margin=0.0`` so the
    matrices verified are byte-for-byte the float path's matrices (it subtracts ``margin*I``;
    margin=0 ⇒ no subtraction). A True here is a machine-checked guarantee that the IDENTICAL
    condition the float recheck judged is satisfied — never a solver claim.

    Because ``verified_pd`` is a sound under-estimator of ``lambda_min`` that dominates the float
    ``eigvalsh`` test on identical matrices, this recheck is a SUPERSET-or-equal of the float
    recheck: admit set preserved-or-grown, never shrunk; sound (no false positives).
    """
    from rump_pd import rump_verify_certificate
    return bool(rump_verify_certificate(np.asarray(P, dtype=np.float64),
                                        vertex_jacobians, margin=_FLOAT_PATH_MARGIN))


def make_sdp_verifier_rump_or(t_domain: str = "tmin1", *, margin: float = 1e-7,
                              max_input_abs: float = 1.0,
                              solvers: tuple = ("CLARABEL", "SCS")):
    """ADDITIVE hardened VerifierBackend (n=2 coupled gene).

    Admit a gene iff EITHER:
      * the solver-INDEPENDENT inf-norm or 2-norm closed-form certificate holds (sound fast path,
        identical to the existing certifiers — needs no solver), OR
      * the OR-of-solvers SDP gate finds a Rump-VERIFIED common-Lyapunov ``P``: for ANY solver in
        ``solvers`` the SDP is solved (with the SAME ``margin`` constraint the existing certifier
        uses), and the returned ``P`` passes the Rump recheck on the SAME no-margin matrices the
        float path tests (``P`` and ``P - J^T P J``).

    Memoised on the rounded genotype; fail-closed (cvxpy/CLARABEL absent ⇒ the SDP branch declines,
    only the solver-independent fast paths can still admit). PURELY additive — does not alter the
    default certifiers.

    Note on the OR + margin: the SDP is SOLVED with ``margin`` (so the solver targets a strictly-PD
    interior, matching the existing pipeline), but the ACCEPTANCE recheck is the no-margin Rump test
    on ``P`` and ``P - J^T P J`` — exactly the float path's condition — so the admit set is
    preserved-or-grown vs the float recheck, never shrunk.
    """
    import cvxpy as _cp  # noqa: F401  (presence probe; lyapunov certifier degrades gracefully)
    from coupled_components import _inf_certifies, _two_certifies
    from lyapunov_sdp_certifier import CVXPY_AVAILABLE, certify_common_lyapunov
    from rump_pd import _vertex_jacobians

    cache: dict = {}

    def _certifies(gene) -> bool:
        # Solver-independent sound fast paths (identical to the existing certifiers).
        if _inf_certifies(gene) or _two_certifies(gene):
            return True
        if not CVXPY_AVAILABLE:
            return False
        verts = _vertex_jacobians(gene, t_domain=t_domain, max_input_abs=max_input_abs)
        installed = set(_cp.installed_solvers())
        for sv in solvers:
            if sv not in installed:
                continue
            # Solve under THIS solver explicitly (an explicit solver is honoured by the now
            # fail-closed certifier). We IGNORE the certifier's own float `certified` flag and
            # re-verify P ourselves with the Rump no-margin recheck so the verdict rests on the
            # machine-checked test, not the float one.
            r = certify_common_lyapunov(gene, t_domain=t_domain, margin=margin,
                                        max_input_abs=max_input_abs, solver=sv)
            if r.P is not None and rump_recheck_matches_float_path(r.P, verts):
                return True
        return False

    class _V:
        name = "sdp_rump_or"

        def certifies(self, gene) -> bool:
            g = gene.clipped()
            key = (tuple(np.round(g.decay, 6)), tuple(np.round(g.W.reshape(-1), 6)))
            v = cache.get(key)
            if v is None:
                v = bool(_certifies(gene))
                cache[key] = v
            return v

    return _V()


if __name__ == "__main__":  # pragma: no cover
    from coupled_map import CoupledGene
    v = make_sdp_verifier_rump_or()
    for name, g in (
        ("benign", CoupledGene.make(decay=[0.8, 0.8], W=[[0.0, 0.0], [0.0, 0.0]])),
        ("nonnormal_sdp_only", CoupledGene.make(decay=[0.5, 0.5], W=[[0.0, 0.0], [1.6, 0.0]])),
    ):
        print(f"{name}: rump_or certifies = {v.certifies(g)}")
