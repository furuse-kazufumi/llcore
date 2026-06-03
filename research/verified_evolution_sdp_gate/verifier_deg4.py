# SPDX-License-Identifier: Apache-2.0
"""Richer VerifierBackend — degree-4 (non-quadratic) Lyapunov via the symmetric 2nd power.

Targets the **Track-D D4 residual**: genes with ρ(J)<1 over the box that NO common *quadratic*
Lyapunov P (and no fixed induced norm) certifies. A degree-4 homogeneous Lyapunov
V(z)=m(z)^T P m(z), where m(z) is the vector of degree-2 monomials (the Veronese map), certifies
a STRICTLY LARGER class (Parrilo-Jadbabaie SOS hierarchy, k=2). We use the *lifted* common-P
LMI on the symmetric 2nd Kronecker power J^[2] of each box-vertex Jacobian:

    find P ≻ 0  s.t.  P - (J_v^[2])^T P (J_v^[2]) ≻ 0   at every t-box vertex v.

Requiring this as a FULL matrix inequality on the lifted space R^{n(n+1)/2} is SOUND (sufficient
for contraction of the original map, since V(z) then strictly decreases) though slightly
conservative vs the exact SOS-on-the-variety condition — i.e. a valid degree-4 Lyapunov
certificate. When feasible where the plain quadratic SDP is not, it certifies a D4-residual gene.

This is a NEW VerifierBackend plug-in for the skeleton — only the backend changes; evolve() and
the codecs are untouched. cvxpy required (degrades to None if absent).
"""
from __future__ import annotations

import itertools

import numpy as np

try:
    import cvxpy as cp
    _CVXPY = True
except Exception:  # pragma: no cover
    _CVXPY = False


def _mono_basis(n: int) -> list[tuple[int, int]]:
    """Degree-2 monomial index pairs (i<=j), the Veronese basis of dim n(n+1)/2."""
    return [(i, j) for i in range(n) for j in range(i, n)]


def sym2_power(A: np.ndarray) -> np.ndarray:
    """Symmetric 2nd Kronecker power A^[2]: the matrix s.t. under z->Az the degree-2 monomial
    vector m(z) (basis i<=j, value z_i z_j) maps as m(Az) = A^[2] m(z). dim = n(n+1)/2.

    Row r = monomial (i,j): m_r(Az) = (Az)_i (Az)_j = Σ_{k,l} A_ik A_jl z_k z_l. Collect onto the
    symmetric basis (k,l)->(min,max): for an unordered pair {p,q} with p<q the coefficient is
    A_ip A_jq + A_iq A_jp; for p==p it is A_ip A_jp.
    """
    A = np.asarray(A, dtype=np.float64)
    n = A.shape[0]
    basis = _mono_basis(n)
    idx = {pair: c for c, pair in enumerate(basis)}
    m = len(basis)
    S = np.zeros((m, m))
    for r, (i, j) in enumerate(basis):
        for k in range(n):
            for l in range(n):
                p, q = (k, l) if k <= l else (l, k)
                S[r, idx[(p, q)]] += A[i, k] * A[j, l]
    return S


def certify_deg4(vertex_jacobians: list[np.ndarray], margin: float = 1e-7) -> bool:
    """Common degree-4 (symmetric-2nd-power) Lyapunov over a set of vertex Jacobians.

    Sound: if feasible, V(z)=m(z)^T P m(z) is a valid Lyapunov ⇒ the map contracts over the box.
    Independent eigenvalue re-check of P (never solver-blind). Necessary pre-screen: ρ(J_v)<1.
    """
    if not _CVXPY:
        return False
    for J in vertex_jacobians:
        if float(np.max(np.abs(np.linalg.eigvals(J)))) >= 1.0:
            return False
    lifts = [sym2_power(J) for J in vertex_jacobians]
    m = lifts[0].shape[0]
    P = cp.Variable((m, m), symmetric=True)
    I = np.eye(m)
    cons = [P >> I] + [P - L.T @ P @ L >> margin * I for L in lifts]
    try:
        cp.Problem(cp.Minimize(cp.trace(P)), cons).solve()
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


# ---- n=2 adapter (Track-C/D gene) + n-dim adapter (coupled_nd gene) ---------- #


def _vertices_n2(gene):
    """t-box vertex Jacobians for a Track-C/D CoupledGene (n=2)."""
    import os
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    for d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
        p = os.path.normpath(os.path.join(here, "..", d))
        if p not in sys.path:
            sys.path.insert(0, p)
    from coupled_map import t_min_per_coord
    g = gene.clipped()
    t_lo = t_min_per_coord(g, max_input_abs=1.0)
    verts = [np.array([t_lo[0], t_lo[1]]), np.array([t_lo[0], 1.0]),
             np.array([1.0, t_lo[1]]), np.array([1.0, 1.0])]
    return [np.diag(g.decay) + np.diag((1.0 - g.decay) * v) @ g.W for v in verts]


def cert_deg4_n2(gene) -> bool:
    return certify_deg4(_vertices_n2(gene))


def make_deg4_verifier_n2():
    """VerifierBackend (n=2): admit if the quadratic SDP OR the degree-4 Lyapunov certifies."""
    from coupled_components import _sdp_certifies
    cache: dict = {}

    class _V:
        name = "sdp_deg4"

        def certifies(self, gene) -> bool:
            g = gene.clipped()
            key = (tuple(np.round(g.decay, 6)), tuple(np.round(g.W.reshape(-1), 6)))
            v = cache.get(key)
            if v is None:
                v = bool(_sdp_certifies(gene) or cert_deg4_n2(gene))
                cache[key] = v
            return v
    return _V()


if __name__ == "__main__":
    # sanity: symmetric power correctness vs brute-force on random z, and a smoke certify
    rng = np.random.default_rng(0)
    A = rng.standard_normal((2, 2))
    z = rng.standard_normal(2)
    m = np.array([z[0] * z[0], z[0] * z[1], z[1] * z[1]])
    Az = A @ z
    m2 = np.array([Az[0] * Az[0], Az[0] * Az[1], Az[1] * Az[1]])
    err = np.max(np.abs(sym2_power(A) @ m - m2))
    print("sym2_power max error vs brute force:", err, "(should be ~0)")
