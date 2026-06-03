# SPDX-License-Identifier: Apache-2.0
"""Richer GeneCodec — n-dimensional coupled substrate (skeleton extension).

Demonstrates the llcore Verified-Evolution skeleton's extensibility: this file adds a
NEW `GeneCodec` (arbitrary dimension n) and matching n-dim `VerifierBackend`s, and plugs
them into the UNCHANGED `evolvable_core.evolve()`. Nothing in the skeleton changes — only
the substrate.

Substrate (generalises Track C's n=2 map to any n)::

    s' = decay ⊙ s + (1 - decay) ⊙ tanh(W s + V x),   s,x ∈ R^n, decay∈[0,1]^n, W∈[-2,2]^{n×n}, V=I
    J(t) = diag(decay) + diag((1-decay) ⊙ t) @ W,     t_i = sech^2(pre_i) ∈ (0,1]

Sound contraction certifiers over the achievable-t box [t_min,1]^n (2^n vertices):
  * inf_norm : closed-form sup ‖J‖_∞ < 1 (per-row abs-sum is V-shaped in t_i ⇒ endpoint max).
  * two_norm : max over the 2^n box vertices of σ_max(J) < 1 (σ_max convex, affine in t ⇒ vertex).
  * sdp      : common quadratic Lyapunov P ≻ 0 with P − J_v^T P J_v ≻ 0 at every vertex (cvxpy).

All three imply ρ(J)<1 over the box ⇒ contraction. cvxpy degrades gracefully (sdp → False if
absent). Purely additive; src/ untouched.

Scientific question this enables: does the SDP/2-norm advantage over the conservative ∞-norm
GROW with dimension? (More coordinates ⇒ more room for rotational/non-normal contractions the
∞-norm over-rejects ⇒ a better verifier should matter more as the core scales.)
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

try:
    import cvxpy as cp
    _CVXPY = True
    # Pin an accurate solver — SCS (cvxpy default) false-negatives near the feasibility boundary.
    _SOLVER = cp.CLARABEL if "CLARABEL" in cp.installed_solvers() else None
except Exception:  # pragma: no cover
    _CVXPY = False
    _SOLVER = None


# --------------------------------------------------------------------------- #
# Gene + dynamics (n-dim).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CoupledNDGene:
    decay: np.ndarray   # (n,) in [0,1]
    W: np.ndarray       # (n,n) in [-2,2]
    V: np.ndarray       # (n,n) = I

    @staticmethod
    def make(decay, W, V=None) -> "CoupledNDGene":
        decay = np.asarray(decay, dtype=np.float64).reshape(-1)
        n = decay.shape[0]
        W = np.asarray(W, dtype=np.float64).reshape(n, n)
        V = np.eye(n) if V is None else np.asarray(V, dtype=np.float64).reshape(n, n)
        return CoupledNDGene(decay=decay, W=W, V=V)

    @property
    def n(self) -> int:
        return self.decay.shape[0]

    def clipped(self) -> "CoupledNDGene":
        return CoupledNDGene(decay=np.clip(self.decay, 0.0, 1.0),
                             W=np.clip(self.W, -2.0, 2.0), V=self.V)


def step(g: CoupledNDGene, s: np.ndarray, x: np.ndarray) -> np.ndarray:
    gc = g.clipped()
    return gc.decay * s + (1.0 - gc.decay) * np.tanh(gc.W @ s + gc.V @ x)


def jacobian(g: CoupledNDGene, s: np.ndarray, x: np.ndarray) -> np.ndarray:
    gc = g.clipped()
    t = 1.0 - np.tanh(gc.W @ s + gc.V @ x) ** 2
    return np.diag(gc.decay) + np.diag((1.0 - gc.decay) * t) @ gc.W


def t_min_per_coord(g: CoupledNDGene, max_input_abs: float = 1.0) -> np.ndarray:
    gc = g.clipped()
    M = np.abs(gc.W).sum(axis=1) + max_input_abs * np.abs(gc.V).sum(axis=1)
    return 1.0 - np.tanh(M) ** 2


def _jac_at_t(g: CoupledNDGene, t: np.ndarray) -> np.ndarray:
    gc = g.clipped()
    return np.diag(gc.decay) + np.diag((1.0 - gc.decay) * t) @ gc.W


def _box_vertices(t_lo: np.ndarray) -> list[np.ndarray]:
    """The 2^n vertices of [t_lo_i, 1]^n."""
    n = t_lo.shape[0]
    return [np.array([(1.0 if b else t_lo[i]) for i, b in enumerate(combo)])
            for combo in itertools.product((0, 1), repeat=n)]


# --------------------------------------------------------------------------- #
# n-dim certifiers (sound contraction over the t-box).
# --------------------------------------------------------------------------- #


def infnorm_sup(g: CoupledNDGene, t_lo: np.ndarray) -> float:
    """Closed-form sup over t∈[t_lo,1]^n of ‖J(t)‖_∞. Each row i abs-sum is
    |decay_i + (1-decay_i)t_i W_ii| + (1-decay_i)t_i Σ_{j≠i}|W_ij|, V-shaped in t_i ⇒
    max at an endpoint t_i∈{t_lo_i,1}."""
    gc = g.clipped()
    n = gc.n
    best = 0.0
    for i in range(n):
        off = sum(abs(gc.W[i, j]) for j in range(n) if j != i)
        row = 0.0
        for ti in (t_lo[i], 1.0):
            diag = abs(gc.decay[i] + (1.0 - gc.decay[i]) * ti * gc.W[i, i])
            row = max(row, diag + (1.0 - gc.decay[i]) * ti * off)
        best = max(best, row)
    return float(best)


def cert_inf(g: CoupledNDGene, max_input_abs: float = 1.0) -> bool:
    return bool(infnorm_sup(g, t_min_per_coord(g, max_input_abs)) < 1.0)


def cert_two(g: CoupledNDGene, max_input_abs: float = 1.0) -> bool:
    t_lo = t_min_per_coord(g, max_input_abs)
    return all(float(np.linalg.svd(_jac_at_t(g, v), compute_uv=False)[0]) < 1.0
               for v in _box_vertices(t_lo))


def cert_sdp(g: CoupledNDGene, max_input_abs: float = 1.0, margin: float = 1e-7) -> bool:
    """Common quadratic Lyapunov vertex-LMI feasibility (cvxpy). Fast-path: cert_two ⇒ True
    (2-norm = P=I special case). Necessary pre-screen: ρ(J_v)<1 at every vertex. Independent
    eigenvalue re-check of the solver's P (never solver-blind)."""
    if cert_two(g, max_input_abs):
        return True
    if not _CVXPY:
        return False
    gc = g.clipped()
    n = gc.n
    verts = _box_vertices(t_min_per_coord(gc, max_input_abs))
    Js = [_jac_at_t(gc, v) for v in verts]
    for J in Js:
        if float(np.max(np.abs(np.linalg.eigvals(J)))) >= 1.0:
            return False  # LMI infeasible at this vertex
    P = cp.Variable((n, n), symmetric=True)
    I = np.eye(n)
    cons = [P >> I] + [P - J.T @ P @ J >> margin * I for J in Js]
    prob = cp.Problem(cp.Minimize(cp.trace(P)), cons)
    try:
        prob.solve(solver=_SOLVER)
    except Exception:
        return False
    if prob.status not in ("optimal", "optimal_inaccurate") or P.value is None:
        return False
    Pv = 0.5 * (P.value + P.value.T)
    if float(np.min(np.linalg.eigvalsh(Pv))) <= 0.0:
        return False
    for J in Js:
        M = Pv - J.T @ Pv @ J
        if float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T)))) <= 0.0:
            return False
    return True


def empirical_rho(g: CoupledNDGene, *, n_samples: int = 8000, seed: int = 0) -> float:
    """From-below empirical sup of ρ(J) over the (s,x) box (consistency oracle)."""
    gc = g.clipped()
    n = gc.n
    rng = np.random.default_rng(seed)
    S = rng.uniform(-1, 1, (n_samples, n))
    X = rng.uniform(-1, 1, (n_samples, n))
    mx = 0.0
    for k in range(n_samples):
        mx = max(mx, float(np.max(np.abs(np.linalg.eigvals(jacobian(gc, S[k], X[k]))))))
    return mx


def classify_region(g: CoupledNDGene) -> str:
    if cert_inf(g):
        return "inf"
    if cert_two(g):
        return "two_norm_only"
    if cert_sdp(g):
        return "sdp_only"
    return "non_certified"


# --------------------------------------------------------------------------- #
# GeneCodec (the new plug-in for evolvable_core).
# --------------------------------------------------------------------------- #


class CoupledNDGeneCodec:
    """Genotype (n + n^2 reals) <-> CoupledNDGene. Plug into evolvable_core.evolve()."""

    def __init__(self, n: int):
        self.n = n
        self.dim = n + n * n
        lo = np.concatenate([np.zeros(n), np.full(n * n, -2.0)])
        hi = np.concatenate([np.ones(n), np.full(n * n, 2.0)])
        self._lo, self._hi, self._half = lo, hi, 0.5 * (hi - lo)
        self.fallback_genotype = np.concatenate([np.full(n, 0.5), np.zeros(n * n)])  # ‖J‖_∞=0.5

    def random(self, rng):
        return self._lo + (self._hi - self._lo) * rng.random(self.dim)

    def clip(self, gtype):
        return np.clip(np.asarray(gtype, dtype=np.float64), self._lo, self._hi)

    def to_gene(self, gtype) -> CoupledNDGene:
        g = np.asarray(gtype, dtype=np.float64)
        return CoupledNDGene.make(decay=g[:self.n], W=g[self.n:].reshape(self.n, self.n))

    def crossover(self, a, b, rng):
        mask = rng.random(self.dim) < 0.5
        return np.where(mask, a, b)

    def mutate(self, g, sigma, rng):
        return g + sigma * self._half * rng.standard_normal(self.dim)


# --------------------------------------------------------------------------- #
# Objective: block-rotation free-response match (rewards coupled rotational dynamics).
# --------------------------------------------------------------------------- #


def _block_rotation_target(n: int, T: int, radius: float, omega: float, amp: float) -> np.ndarray:
    """Damped block-diagonal 2-D rotations (odd n: last coord pure decay). s0 = amp·e_0-ish."""
    s0 = np.zeros(n)
    for p in range(0, n - 1, 2):
        s0[p] = amp
    if n % 2 == 1:
        s0[-1] = amp
    R = np.zeros((n, n))
    c, s = np.cos(omega), np.sin(omega)
    for p in range(0, n - 1, 2):
        R[p, p] = c; R[p, p + 1] = -s; R[p + 1, p] = s; R[p + 1, p + 1] = c
    if n % 2 == 1:
        R[-1, -1] = 1.0
    traj = [s0.copy()]
    v = s0.copy()
    for _ in range(T):
        v = radius * (R @ v)
        traj.append(v.copy())
    return np.stack(traj)


@dataclass(frozen=True)
class RotationNDObjective:
    """Reproduce a damped n-dim block-rotation from a fixed start. As n grows, more rotation
    blocks ⇒ more rows with large abs-sums ⇒ the ∞-norm over-rejects more of the optimum."""

    n: int
    T: int = 30
    radius: float = 0.93
    period: float = 10.0
    amp: float = 0.4
    name: str = "rotation_nd"

    def _target(self) -> np.ndarray:
        return _block_rotation_target(self.n, self.T, self.radius, 2 * np.pi / self.period, self.amp)

    def _s0(self) -> np.ndarray:
        return self._target()[0]

    def fitness(self, gene: CoupledNDGene) -> float:
        tg = self._target()
        s = self._s0().copy()
        x0 = np.zeros(self.n)
        traj = [s.copy()]
        for _ in range(self.T):
            s = step(gene, s, x0)
            traj.append(s.copy())
        traj = np.stack(traj)
        tr, tt = traj[1:], tg[1:]
        ss_res = float(np.sum((tr - tt) ** 2))
        ss_tot = float(np.sum((tt - tt.mean(axis=0)) ** 2))
        return 0.0 if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def make_nd_verifier(name: str):
    """VerifierBackend for the n-dim substrate (memoised, fail-closed)."""
    fns = {"none": None, "inf_norm": cert_inf, "two_norm": cert_two, "sdp": cert_sdp}
    if name not in fns:
        raise ValueError(name)
    fn = fns[name]
    cache: dict = {}

    class _V:
        def __init__(self):
            self.name = name

        def certifies(self, gene: CoupledNDGene) -> bool:
            if fn is None:
                return True
            gc = gene.clipped()
            key = (tuple(np.round(gc.decay, 6)), tuple(np.round(gc.W.reshape(-1), 6)))
            v = cache.get(key)
            if v is None:
                v = bool(fn(gene))
                cache[key] = v
            return v
    return _V()


if __name__ == "__main__":
    for n in (2, 3, 4):
        codec = CoupledNDGeneCodec(n)
        obj = RotationNDObjective(n)
        rng = np.random.default_rng(0)
        fits = [obj.fitness(codec.to_gene(codec.clip(codec.random(rng)))) for _ in range(500)]
        print(f"n={n} dim={codec.dim} random R2: min={min(fits):.2f} max={max(fits):.3f} "
              f"target_shape={obj._target().shape}")
