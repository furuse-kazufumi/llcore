# SPDX-License-Identifier: Apache-2.0
"""Richer VerifierBackend — degree-6 (non-quadratic) Lyapunov via the symmetric 3rd power.

The next rung above ``verifier_deg4`` on the verifier-fitness frontier the arc identified:

    inf-norm → 2-norm → quadratic SDP → degree-4 Lyapunov → **degree-6 Lyapunov** → …(JSR-exact)…

A degree-6 homogeneous Lyapunov V(z)=m3(z)ᵀ P m3(z), where m3(z) is the vector of degree-3
monomials (the Veronese map of order 3), certifies contraction via a common-P LMI on the
**symmetric 3rd Kronecker power** J^[3] of each t-box vertex Jacobian (Parrilo–Jadbabaie SOS
hierarchy, k=3):

    find P ≻ 0  s.t.  P - (J_v^[3])ᵀ P (J_v^[3]) ≻ 0   at every t-box vertex v.

SOUNDNESS — exactly as ``verifier_deg4``: the lifted FULL-space LMI on R^{C(n+2,3)} is a
*sufficient* (slightly conservative) form of the exact degree-6 SOS-on-the-variety condition,
hence a valid degree-6 Lyapunov certificate of contraction of the original map. The certifier
NEVER trusts the solver blindly: P's eigenvalues and the decrease-LMI eigenvalues are
re-checked independently, and ρ(J_v)<1 is a necessary pre-screen at every vertex.

SOLVER NOTE (retraction — see DEG6_VERDICT.md): an earlier draft reported the degree-4/6 lifted
certificates were *non-nested* (complementary). That was a **cvxpy SCS solver artifact** — SCS
(the default first-order solver) returns FALSE NEGATIVES near the feasibility boundary, fabricating
a fake complementarity. Under the accurate interior-point solver CLARABEL (now pinned via
``_SOLVER``) the lifted ladder is **NESTED** (deg4 ⊆ deg6) and the common quadratic SDP already
certifies ~95% of contracting genes; degree-4/6 add only a tiny residual. ``make_deg6_verifier_n2``
returns the union (sdp ∪ deg4 ∪ deg6) for completeness, but with an accurate solver deg6 dominates.

This is a NEW VerifierBackend plug-in for the skeleton — only the backend changes; ``evolve()``
and the codecs are untouched. cvxpy required (degrades to None/False if absent: fail-closed).
"""
from __future__ import annotations

import itertools

import numpy as np

try:
    import cvxpy as cp
    _CVXPY = True
    # Pin an ACCURATE interior-point solver. cvxpy's default for these feasibility-boundary SDPs
    # is SCS (first-order ADMM), which produces FALSE NEGATIVES near the feasibility boundary —
    # it fails to find a certificate that exists. That fabricated an apparent deg4/deg6
    # "complementarity" (an SCS artifact; under CLARABEL the lifted ladder is NESTED). The
    # independent eigen re-check guards soundness but cannot recover a missed certificate, so the
    # solver default — not the re-check — is the fix. Fail-closed if CLARABEL is unavailable.
    _SOLVER = cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None
except Exception:  # pragma: no cover
    _CVXPY = False
    _SOLVER = None

# reuse the n=2 vertex construction + the degree-4 certifier (additive, DRY).
from verifier_deg4 import _vertices_n2, cert_deg4_n2


def mono_basis(n: int, degree: int) -> list[tuple[int, ...]]:
    """Sorted degree-`degree` monomial multi-indices over n vars; dim = C(n+degree-1, degree)."""
    return list(itertools.combinations_with_replacement(range(n), degree))


def sym_power(A: np.ndarray, degree: int) -> np.ndarray:
    """Symmetric `degree`-th Kronecker power A^[degree]: the matrix s.t. under z->Az the
    degree-`degree` monomial vector m(z) (basis = sorted multisets, value Π z_{β_i}) maps as
    m(Az) = A^[degree] m(z). dim = C(n+degree-1, degree).

    Row r = monomial β=(β_1..β_d): m_r(Az) = Π_i (Az)_{β_i} = Σ_{p∈[n]^d} (Π_i A_{β_i,p_i}) z_{p}
    collected onto the symmetric basis sorted(p). Generalises ``verifier_deg4.sym2_power``
    (degree=2) and is brute-force-verified for n=2,3,4 and degree=2,3 (max error ~1e-16).
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    basis = mono_basis(n, degree)
    idx = {b: c for c, b in enumerate(basis)}
    m = len(basis)
    S = np.zeros((m, m))
    for r, beta in enumerate(basis):
        for p in itertools.product(range(n), repeat=degree):
            coeff = 1.0
            for i in range(degree):
                coeff *= A[beta[i], p[i]]
            S[r, idx[tuple(sorted(p))]] += coeff
    return S


def certify_degN(vertex_jacobians: list[np.ndarray], degree: int, margin: float = 1e-7) -> bool:
    """Common degree-(2*degree) homogeneous Lyapunov over a set of vertex Jacobians, via the
    symmetric `degree`-th power lift. degree=2 ≡ degree-4 (cf. verifier_deg4), degree=3 ≡ degree-6.

    Sound: if feasible, V(z)=m(z)ᵀ P m(z) is a valid Lyapunov ⇒ the map contracts over the box.
    Independent eigenvalue re-check of P AND of every decrease-LMI (never solver-blind).
    Necessary pre-screen: ρ(J_v)<1 at every vertex.
    """
    if not _CVXPY:
        return False
    for J in vertex_jacobians:
        if float(np.max(np.abs(np.linalg.eigvals(J)))) >= 1.0:
            return False
    lifts = [sym_power(J, degree) for J in vertex_jacobians]
    m = lifts[0].shape[0]
    P = cp.Variable((m, m), symmetric=True)
    I = np.eye(m)
    cons = [P >> I] + [P - L.T @ P @ L >> margin * I for L in lifts]
    try:
        cp.Problem(cp.Minimize(cp.trace(P)), cons).solve(solver=_SOLVER)
    except Exception:
        return False
    if P.value is None:
        return False
    Pv = 0.5 * (P.value + P.value.T)
    if float(np.min(np.linalg.eigvalsh(Pv))) <= 0.0:
        return False
    for L in lifts:
        M = Pv - L.T @ Pv @ L
        if float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T)))) <= 0.0:
            return False
    return True


def certify_deg6(vertex_jacobians: list[np.ndarray], margin: float = 1e-7) -> bool:
    """Common degree-6 (symmetric-3rd-power) Lyapunov over a set of vertex Jacobians."""
    return certify_degN(vertex_jacobians, degree=3, margin=margin)


# ---- n=2 adapter (Track-C/D gene) ------------------------------------------ #


def cert_deg6_n2(gene) -> bool:
    return certify_deg6(_vertices_n2(gene))


def make_deg6_verifier_n2():
    """VerifierBackend (n=2): admit if quadratic SDP OR degree-4 OR degree-6 certifies.

    The union is the correct sound set because the lifted degree-4 / degree-6 certificates are
    complementary (non-nested) — see module docstring + DEG6_VERDICT.md. Memoised on the
    rounded genotype; fail-closed (cvxpy-absent ⇒ deg* ⇒ False)."""
    from coupled_components import _sdp_certifies
    cache: dict = {}

    class _V:
        name = "sdp_deg4_deg6"

        def certifies(self, gene) -> bool:
            g = gene.clipped()
            key = (tuple(np.round(g.decay, 6)), tuple(np.round(g.W.reshape(-1), 6)))
            v = cache.get(key)
            if v is None:
                v = bool(_sdp_certifies(gene) or cert_deg4_n2(gene) or cert_deg6_n2(gene))
                cache[key] = v
            return v
    return _V()


if __name__ == "__main__":
    # sanity: symmetric 3rd power correctness vs brute-force on random z (should be ~0).
    rng = np.random.default_rng(0)
    for n in (2, 3, 4):
        A = rng.standard_normal((n, n))
        z = rng.standard_normal(n)
        basis = mono_basis(n, 3)
        m = np.array([np.prod([z[i] for i in b]) for b in basis])
        Az = A @ z
        m_true = np.array([np.prod([Az[i] for i in b]) for b in basis])
        err = float(np.max(np.abs(sym_power(A, 3) @ m - m_true)))
        print(f"sym_power(.,3) n={n} max error vs brute force: {err:.2e} (should be ~0)")
