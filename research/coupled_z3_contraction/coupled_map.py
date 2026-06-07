# SPDX-License-Identifier: Apache-2.0
"""Track C — minimal COUPLED (non-diagonal) RWKV-style state map + Jacobian + empirical checks.

This module is purely ADDITIVE research code (research/coupled_z3_contraction/). It does
NOT touch src/. It defines the n=2 coupled map studied in Track C and the empirical oracles
the pre-registered gates (C1/C2/C3) need.

The map (n=2)::

    s' = decay (.) s + (1 - decay) (.) tanh(W @ s + V @ x)        s, x in R^2, (.) = elementwise

with::

    gene = (decay in [0,1]^2, W in [-2,2]^{2x2} off-diagonal allowed),  V fixed = identity.

State Jacobian (exact)::

    J = diag(decay) + diag((1-decay) (.) t) @ W,   t_i = sech^2(pre_i) in (0,1],  pre = W@s + V@x
    Row i: J_ii = decay_i + (1-decay_i)*t_i*W_ii
           J_ij = (1-decay_i)*t_i*W_ij     (j != i)

Contraction facts used by the gates:
  * ||J||_inf = max_i sum_j |J_ij| < 1  is SUFFICIENT for the map to be a contraction in the
    inf-norm over the box (Banach => unique fixed point + bounded state). This is what the Z3
    certifier proves (sound over-approximation, t free in [0,1]^2 or [t_min,1]).
  * spectral radius rho(J) < 1 over the box is the EXACT (necessary) local-contraction
    condition, but it is NOT a sound *single-point* bound for the trajectory (only inf/2-norm
    submultiplicativity guarantees the iterate stays bounded). rho < 1 with rho < inf-norm gives
    the C3 conservative-false-reject cases.

Honest caveats (pre-committed):
  * t_i = sech^2(pre_i) where pre depends on (s,x) AND on W (coupling) -- so t_i is NOT free;
    the achievable set is a proper subset of (0,1]^2. Freeing t in [0,1]^2 is a sound
    over-approximation (it can only make abs-sums larger), hence unsat => truly certified.
  * The [t_min, 1] variant uses a per-coordinate lower bound t_min_i = sech^2(M_i) with
    M_i = sum_j |W_ij| + |V_ii|*max_input_abs (the max possible |pre_i| over the box). This is
    a tighter-but-still-sound floor on t_i.
  * Empirical ||J|| is a from-BELOW estimate of the box sup (finite sample), so a small positive
    (certified_bound - empirical) gap is EXPECTED and is not unsoundness; only empirical > bound
    would alarm.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoupledGene:
    """A coupled n=2 RWKV-style gene.

    Attributes
    ----------
    decay : np.ndarray shape (2,)
        Per-coordinate decay in [0, 1].
    W : np.ndarray shape (2, 2)
        State-mixing matrix in [-2, 2], off-diagonal entries allowed (coupling).
    V : np.ndarray shape (2, 2)
        Input matrix; fixed to identity for Track C (kept explicit for clarity / future use).
    """

    decay: np.ndarray
    W: np.ndarray
    V: np.ndarray

    @staticmethod
    def make(decay, W, V=None) -> "CoupledGene":
        decay = np.asarray(decay, dtype=np.float64).reshape(2)
        W = np.asarray(W, dtype=np.float64).reshape(2, 2)
        if V is None:
            V = np.eye(2, dtype=np.float64)
        else:
            V = np.asarray(V, dtype=np.float64).reshape(2, 2)
        return CoupledGene(decay=decay, W=W, V=V)

    def clipped(self) -> "CoupledGene":
        """Project gene into the legal box (decay in [0,1], W in [-2,2]). V left as-is (identity)."""
        return CoupledGene(
            decay=np.clip(self.decay, 0.0, 1.0),
            W=np.clip(self.W, -2.0, 2.0),
            V=self.V,
        )

    def as_tuple(self):
        """Hashable / JSON-friendly representation."""
        return (
            tuple(round(float(v), 12) for v in self.decay),
            tuple(round(float(v), 12) for v in self.W.reshape(-1)),
        )


def step(gene: CoupledGene, s: np.ndarray, x: np.ndarray) -> np.ndarray:
    """One application of the coupled map. s, x shape (2,) -> s' shape (2,)."""
    g = gene.clipped()
    pre = g.W @ s + g.V @ x
    return g.decay * s + (1.0 - g.decay) * np.tanh(pre)


def jacobian(gene: CoupledGene, s: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Exact state Jacobian d s'/d s at (s, x). Returns shape (2,2).

    J = diag(decay) + diag((1-decay) * t) @ W,  t_i = sech^2(pre_i) = 1 - tanh^2(pre_i).
    """
    g = gene.clipped()
    pre = g.W @ s + g.V @ x
    t = 1.0 - np.tanh(pre) ** 2  # sech^2, in (0,1]
    J = np.diag(g.decay) + np.diag((1.0 - g.decay) * t) @ g.W
    return J


def t_min_per_coord(gene: CoupledGene, max_input_abs: float = 1.0) -> np.ndarray:
    """Sound per-coordinate lower bound on t_i = sech^2(pre_i) over the box (|s|<=1,|x|<=max).

    |pre_i| <= sum_j |W_ij|*|s_j| + sum_j |V_ij|*|x_j| <= sum_j |W_ij| + max_input_abs*sum_j |V_ij|.
    sech^2 is even and decreasing in |.|, so t_i >= sech^2(M_i) =: t_min_i.
    """
    g = gene.clipped()
    M = np.abs(g.W).sum(axis=1) + max_input_abs * np.abs(g.V).sum(axis=1)
    return 1.0 - np.tanh(M) ** 2


def infnorm_over_box_freeT(gene: CoupledGene, t_lo: np.ndarray | None = None) -> float:
    """Closed-form sup over t in [t_lo,1]^2 of ||J(t)||_inf for the coupled map.

    For each row i the abs-sum is::
        a_i(t_i) = |decay_i + (1-decay_i)*t_i*W_ii| + (1-decay_i)*t_i*|W_ij|   (j != i)
    With decay_i in [0,1] => (1-decay_i) >= 0 and t_i >= 0, the off-diagonal term is monotone
    increasing in t_i. The diagonal abs is piecewise-linear (V-shaped) in t_i, so the row max
    over t_i in [t_lo_i, 1] is attained at an endpoint t_i in {t_lo_i, 1}. We evaluate both
    endpoints per row and take the per-row max, then max over rows.

    This is the EXACT closed-form value the Z3 free-t certifier is testing against ( ||.||_inf < 1 ).
    """
    g = gene.clipped()
    n = 2
    if t_lo is None:
        t_lo = np.zeros(n)
    t_lo = np.asarray(t_lo, dtype=np.float64).reshape(n)
    best_row = 0.0
    for i in range(n):
        j = 1 - i
        row_endpoints = []
        for ti in (t_lo[i], 1.0):
            diag = abs(g.decay[i] + (1.0 - g.decay[i]) * ti * g.W[i, i])
            off = (1.0 - g.decay[i]) * ti * abs(g.W[i, j])
            row_endpoints.append(diag + off)
        best_row = max(best_row, max(row_endpoints))
    return float(best_row)


def empirical_box_norms(
    gene: CoupledGene,
    *,
    n_samples: int = 20000,
    max_input_abs: float = 1.0,
    seed: int = 0,
    include_corners: bool = True,
) -> dict:
    """Dense (s,x) box sample of operator norms of the EXACT Jacobian.

    Returns a dict with the from-below empirical sup of ||J||_inf, ||J||_2 (largest singular
    value), and the spectral radius rho(J) (max |eigenvalue|), plus the worst-case t encountered.
    These are the empirical oracles for C1 (soundness: emp_inf <= certified bound), C2 (an
    expansive coupled map has emp_inf > 1), and C3 (rho < 1 < inf-norm = conservative reject).

    include_corners: also evaluate the 4^2 = 16 sign-corners of (s,x) in {-1,1}^2 x {-max,max}^2,
    where sech^2 minima / abs-sum maxima often live, to reduce from-below under-sampling.
    """
    g = gene.clipped()
    rng = np.random.default_rng(seed)

    S = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    X = rng.uniform(-max_input_abs, max_input_abs, size=(n_samples, 2))

    if include_corners:
        import itertools

        s_corners = np.array(list(itertools.product([-1.0, 1.0], repeat=2)))
        x_corners = np.array(list(itertools.product([-max_input_abs, max_input_abs], repeat=2)))
        grid_s = np.repeat(s_corners, len(x_corners), axis=0)
        grid_x = np.tile(x_corners, (len(s_corners), 1))
        # also include s=0 / x=0 (t = 1 there -> often the abs-sum maximum for benign signs)
        zeros = np.zeros((1, 2))
        extra_s = np.vstack([grid_s, zeros, zeros])
        extra_x = np.vstack([grid_x, zeros, np.full((1, 2), max_input_abs)])
        S = np.vstack([S, extra_s])
        X = np.vstack([X, extra_x])

    max_inf = 0.0
    max_2 = 0.0
    max_rho = 0.0
    worst_t = None
    for k in range(S.shape[0]):
        s = S[k]
        x = X[k]
        pre = g.W @ s + g.V @ x
        t = 1.0 - np.tanh(pre) ** 2
        J = np.diag(g.decay) + np.diag((1.0 - g.decay) * t) @ g.W
        inf_n = float(np.abs(J).sum(axis=1).max())
        two_n = float(np.linalg.svd(J, compute_uv=False)[0])
        rho = float(np.max(np.abs(np.linalg.eigvals(J))))
        if inf_n > max_inf:
            max_inf = inf_n
            worst_t = t.tolist()
        max_2 = max(max_2, two_n)
        max_rho = max(max_rho, rho)
    return {
        "emp_infnorm": max_inf,
        "emp_2norm": max_2,
        "emp_spectral_radius": max_rho,
        "n_points": int(S.shape[0]),
        "worst_t": worst_t,
    }


def iterate_state_growth(
    gene: CoupledGene,
    *,
    n_steps: int = 2000,
    max_input_abs: float = 1.0,
    seed: int = 0,
    n_seqs: int = 8,
) -> dict:
    """Iterate the map with random bounded inputs and report whether the state stays bounded.

    A genuine contraction (in any norm) keeps |s| bounded; an expansive map under persistent
    excitation tends to push |s| toward the saturation envelope. Because tanh saturates at +-1,
    |s'| <= decay*|s| + (1-decay) <= max(|s|, 1) elementwise, so |s| can NEVER exceed 1 here
    (convex-combination structure). So state-growth is NOT a useful expansiveness signal for THIS
    map -- we report it honestly but rely on the empirical Jacobian inf-norm (>1 => expansive)
    as the C2 oracle. We additionally measure trajectory SEPARATION growth: start two nearby
    trajectories and track ||s_a - s_b|| (this DOES diverge for expansive local dynamics until
    saturation clamps it).
    """
    g = gene.clipped()
    rng = np.random.default_rng(seed)
    max_abs_state = 0.0
    any_nonfinite = False
    sep_growth_ratios = []
    for q in range(n_seqs):
        s = rng.uniform(-1.0, 1.0, size=2)
        s_pert = s + 1e-4 * rng.standard_normal(2)
        seq = rng.uniform(-max_input_abs, max_input_abs, size=(n_steps, 2))
        sep0 = float(np.linalg.norm(s - s_pert))
        max_sep_ratio = 0.0
        for k in range(n_steps):
            x = seq[k]
            s = step(g, s, x)
            s_pert = step(g, s_pert, x)
            if not (np.all(np.isfinite(s)) and np.all(np.isfinite(s_pert))):
                any_nonfinite = True
                break
            max_abs_state = max(max_abs_state, float(np.max(np.abs(s))))
            if sep0 > 0:
                ratio = float(np.linalg.norm(s - s_pert)) / sep0
                max_sep_ratio = max(max_sep_ratio, ratio)
        sep_growth_ratios.append(max_sep_ratio)
    return {
        "max_abs_state": max_abs_state,
        "any_nonfinite": any_nonfinite,
        "max_separation_growth_ratio": float(max(sep_growth_ratios)) if sep_growth_ratios else 0.0,
        "n_steps": n_steps,
        "n_seqs": n_seqs,
    }


if __name__ == "__main__":
    # Tiny smoke test (no asserts that gate the run -- just print).
    g = CoupledGene.make(decay=[0.3, 0.3], W=[[0.5, 0.9], [0.9, 0.5]])
    s = np.array([0.2, -0.4])
    x = np.array([0.5, 0.1])
    print("step:", step(g, s, x))
    print("J(s,x):", jacobian(g, s, x))
    print("freeT inf-norm sup:", infnorm_over_box_freeT(g))
    print("t_min:", t_min_per_coord(g))
    print("empirical:", empirical_box_norms(g, n_samples=3000))
