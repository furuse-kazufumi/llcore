# SPDX-License-Identifier: Apache-2.0
"""Concrete plugins for the Verified Evolution skeleton (coupled n=2 substrate).

Provides, for :mod:`evolvable_core`:
  * :class:`CoupledGeneCodec`  — genotype <-> Track C ``CoupledGene`` (dim=6).
  * Objectives                 — RotationObjective / BenignDecayObjective /
                                 NonNormalObjective (deterministic free-response match).
  * VerifierBackends           — none / inf_norm / two_norm / sdp / union, wrapping
                                 the Track C/D certifiers, fail-closed + memoised.
  * empirical_spectral_radius  — dense (s,x)-box oracle (soundness check G1).
  * classify_region            — tightest sound certifier class for a gene.

All additive: reuses Track C/D research modules; **does NOT touch src/**.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# --- wire in sibling research modules (Track C / D) ------------------------- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_RESEARCH = os.path.normpath(os.path.join(_HERE, ".."))
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.join(_RESEARCH, _d)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coupled_map import (  # noqa: E402
    CoupledGene,
    infnorm_over_box_freeT,
    jacobian,
    step,
    t_min_per_coord,
)
from two_norm_vertex_certifier import certify_2norm_contraction  # noqa: E402
from lyapunov_sdp_certifier import (  # noqa: E402
    CVXPY_AVAILABLE,
    certify_common_lyapunov,
)

# Pin an accurate interior-point solver for the SDP. cvxpy's SCS default produces FALSE NEGATIVES
# near the feasibility boundary (it under-certifies, fabricating a too-small certified region and
# an inflated "residual"). CLARABEL if available, else cvxpy default. (Soundness is guarded by
# certify_common_lyapunov's own independent eigen re-check; this only fixes completeness.)
try:
    import cvxpy as _cp  # noqa: E402
    _SDP_SOLVER = "CLARABEL" if "CLARABEL" in _cp.installed_solvers() else None
except Exception:  # pragma: no cover
    _SDP_SOLVER = None

# Genotype layout: [decay0, decay1, W00, W01, W10, W11].
_BOX_LO = np.array([0.0, 0.0, -2.0, -2.0, -2.0, -2.0])
_BOX_HI = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 2.0])
_HALF = 0.5 * (_BOX_HI - _BOX_LO)  # [0.5,0.5,2,2,2,2] per-dim mutation scale


# --------------------------------------------------------------------------- #
# Gene codec.
# --------------------------------------------------------------------------- #


class CoupledGeneCodec:
    """Genotype (6 reals) <-> coupled n=2 gene. Extension point: swap for a
    higher-dim / multi-kernel codec to evolve a different substrate."""

    dim = 6
    fallback_genotype = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0])  # ||J||_inf=0.5<1

    def random(self, rng: np.random.Generator) -> np.ndarray:
        return _BOX_LO + (_BOX_HI - _BOX_LO) * rng.random(self.dim)

    def clip(self, g: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(g, dtype=np.float64), _BOX_LO, _BOX_HI)

    def to_gene(self, g: np.ndarray) -> CoupledGene:
        g = np.asarray(g, dtype=np.float64)
        return CoupledGene.make(decay=g[:2], W=g[2:6].reshape(2, 2))

    def crossover(self, a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        mask = rng.random(self.dim) < 0.5
        return np.where(mask, a, b)

    def mutate(self, g: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
        return g + sigma * _HALF * rng.standard_normal(self.dim)


# --------------------------------------------------------------------------- #
# Objectives (deterministic free-response matching). Higher fitness = better.
# --------------------------------------------------------------------------- #


def _free_response(gene: CoupledGene, s0: np.ndarray, T: int) -> np.ndarray:
    """Autonomous (x=0) trajectory s[0..T], shape (T+1, 2)."""
    s = np.asarray(s0, dtype=np.float64).copy()
    x0 = np.zeros(2)
    traj = [s.copy()]
    for _ in range(T):
        s = step(gene, s, x0)
        traj.append(s.copy())
    return np.stack(traj)


def _r2(traj: np.ndarray, target: np.ndarray) -> float:
    """Coefficient of determination over steps k=1..T (skip the trivial k=0 match).

    R^2 = 1 - SS_res/SS_tot in (-inf, 1]. Project-standard discriminative metric
    (Step C / E-A / ridge readout): a 'lazy decay-to-0' gene scores low/negative
    on a rotation target (does not explain its variance), so the landscape has
    headroom — no exp(-MSE) ceiling (the demo's 0.997 saturation, ③/llive trap)."""
    tr, tg = traj[1:], target[1:]
    ss_res = float(np.sum((tr - tg) ** 2))
    ss_tot = float(np.sum((tg - tg.mean(axis=0)) ** 2))
    if ss_tot < 1e-12:
        return 0.0
    return 1.0 - ss_res / ss_tot


@dataclass(frozen=True)
class RotationObjective:
    """Reproduce a damped 2-D rotation from a fixed start. Rewards COUPLED
    contracting dynamics near the stability boundary (complex Jacobian
    eigenvalues => antisymmetric W => ||J||_inf > 1 even when rho<1, sigma_max<1).
    Optimum lives in the 2-norm region the inf-norm OVER-REJECTS."""

    name: str = "rotation"
    T: int = 30
    radius: float = 0.93
    period: float = 10.0
    amp: float = 0.4
    s0: tuple = (0.4, 0.0)

    def _target(self) -> np.ndarray:
        w = 2.0 * np.pi / self.period
        ks = np.arange(self.T + 1)
        a = self.amp * self.radius ** ks
        return np.stack([a * np.cos(w * ks), a * np.sin(w * ks)], axis=1)

    def fitness(self, gene: CoupledGene) -> float:
        return _r2(_free_response(gene, np.array(self.s0), self.T), self._target())


@dataclass(frozen=True)
class BenignDecayObjective:
    """Reproduce a simple diagonal exponential decay (no rotation). Optimum lives
    INSIDE the inf-norm region (decay~rate, W~0) => honest-null control (G5)."""

    name: str = "benign"
    T: int = 30
    rate: float = 0.85
    amp: float = 0.4
    s0: tuple = (0.4, 0.4)

    def _target(self) -> np.ndarray:
        ks = np.arange(self.T + 1)
        a = self.amp * self.rate ** ks
        return np.stack([a, a], axis=1)

    def fitness(self, gene: CoupledGene) -> float:
        return _r2(_free_response(gene, np.array(self.s0), self.T), self._target())


# Reference NON-NORMAL contracting gene (strictly lower-triangular coupling):
# rho(J)=0.5<1 (contracting) but sigma_max>1 and ||J||_inf>1 (both fixed norms
# reject) => SDP-only region. The target is its free response; matching it needs
# a gene in the SDP-only region, so only the SDP gate can reach the optimum.
_NONNORMAL_REF = CoupledGene.make(decay=[0.5, 0.5], W=[[0.0, 0.0], [1.6, 0.0]])


@dataclass(frozen=True)
class NonNormalObjective:
    """Reproduce the free response of a non-normal contracting reference (transient
    amplification then decay). Designed so the optimum is SDP-CERTIFIABLE but
    inf-norm AND 2-norm REJECTED — a capability test for whether the SDP gate can
    reach dynamics the conservative gates forbid. (Stated honestly: this task is
    built to live in the SDP-only region; it is a reachability test, not a claim
    that natural tasks are non-normal.)"""

    name: str = "nonnormal"
    T: int = 30
    s0: tuple = (0.5, 0.5)

    def _target(self) -> np.ndarray:
        return _free_response(_NONNORMAL_REF, np.array(self.s0), self.T)

    def fitness(self, gene: CoupledGene) -> float:
        return _r2(_free_response(gene, np.array(self.s0), self.T), self._target())


# --------------------------------------------------------------------------- #
# Verifier backends (fail-closed contraction gates). Memoised on rounded genotype.
# --------------------------------------------------------------------------- #


def _key(gene: CoupledGene, nd: int = 6) -> tuple:
    g = gene.clipped()
    return (tuple(np.round(g.decay, nd)), tuple(np.round(g.W.reshape(-1), nd)))


def _inf_certifies(gene: CoupledGene) -> bool:
    """Closed-form ||J||_inf < 1 over the achievable-t box (== Z3 inf-norm)."""
    g = gene.clipped()
    t_lo = t_min_per_coord(g, max_input_abs=1.0)
    return bool(infnorm_over_box_freeT(g, t_lo=t_lo) < 1.0)


def _two_certifies(gene: CoupledGene) -> bool:
    return bool(certify_2norm_contraction(gene, t_domain="tmin1").certified)


def _vertex_jacobians(gene: CoupledGene):
    """The 4 t-box-vertex Jacobians J(t) = diag(decay)+diag((1-decay)t)W, t in
    {t_min,1}^2. The achievable-t set is inside this box (sound over-approx)."""
    g = gene.clipped()
    t_lo = t_min_per_coord(g, max_input_abs=1.0)
    verts = ((t_lo[0], t_lo[1]), (t_lo[0], 1.0), (1.0, t_lo[1]), (1.0, 1.0))
    return [np.diag(g.decay) + np.diag((1.0 - g.decay) * np.asarray(t)) @ g.W for t in verts]


def _sdp_certifies(gene: CoupledGene) -> bool:
    """SDP common-Lyapunov, BEHAVIOUR-PRESERVING fast path.

    Short-circuits (all sound, decisions identical to a bare solve):
      1. inf-norm or 2-norm certifies => contraction (subset of SDP, no solve).
      2. rho(J_v) >= 1 at ANY t-box vertex => the vertex LMI P - J_v^T P J_v >> 0
         is INFEASIBLE (discrete-Lyapunov necessity), so the common-P SDP cannot
         exist => reject WITHOUT a solve. This prunes the (many) non-contracting
         children that would otherwise each pay a full cvxpy canonicalisation.
      3. otherwise run the genuine SDP.
    Fail-closed if cvxpy unavailable/None."""
    if _inf_certifies(gene) or _two_certifies(gene):
        return True
    if not CVXPY_AVAILABLE:
        return False
    # necessary condition for common quadratic stability: rho<1 at every vertex.
    for J in _vertex_jacobians(gene):
        if float(np.max(np.abs(np.linalg.eigvals(J)))) >= 1.0:
            return False
    r = certify_common_lyapunov(gene, t_domain="tmin1", solver=_SDP_SOLVER)
    return r.certified is True


class _Backend:
    def __init__(self, name: str, fn):
        self.name = name
        self._fn = fn
        self._cache: dict[tuple, bool] = {}

    def certifies(self, gene: CoupledGene) -> bool:
        if self._fn is None:  # none gate
            return True
        k = _key(gene)
        v = self._cache.get(k)
        if v is None:
            v = bool(self._fn(gene))
            self._cache[k] = v
        return v


def make_verifier(name: str) -> _Backend:
    if name == "none":
        return _Backend("none", None)
    if name == "inf_norm":
        return _Backend("inf_norm", _inf_certifies)
    if name == "two_norm":
        return _Backend("two_norm", _two_certifies)
    if name == "sdp":
        return _Backend("sdp", _sdp_certifies)
    if name == "union":
        return _Backend("union", lambda g: _inf_certifies(g) or _sdp_certifies(g))
    raise ValueError(f"unknown verifier {name!r}")


VERIFIER_NAMES = ("none", "inf_norm", "two_norm", "sdp", "union")


# --------------------------------------------------------------------------- #
# Empirical oracle + region classifier.
# --------------------------------------------------------------------------- #


def empirical_spectral_radius(gene: CoupledGene, *, n_samples: int = 20000,
                              seed: int = 0, include_corners: bool = True) -> float:
    """From-below empirical sup of rho(J) over the (s,x) box. Soundness oracle:
    a gene admitted by a SOUND contraction gate must have this < 1."""
    g = gene.clipped()
    rng = np.random.default_rng(seed)
    S = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    X = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    pts_s, pts_x = [S], [X]
    if include_corners:
        import itertools
        c = np.array(list(itertools.product([-1.0, 1.0], repeat=2)))
        gs = np.repeat(c, len(c), axis=0)
        gx = np.tile(c, (len(c), 1))
        pts_s += [gs, np.zeros((1, 2))]
        pts_x += [gx, np.zeros((1, 2))]
    S = np.vstack(pts_s)
    X = np.vstack(pts_x)
    max_rho = 0.0
    for k in range(S.shape[0]):
        J = jacobian(g, S[k], X[k])
        max_rho = max(max_rho, float(np.max(np.abs(np.linalg.eigvals(J)))))
    return max_rho


def classify_region(gene: CoupledGene) -> str:
    """Tightest sound certifier class:
      'inf'           inf-norm certifies (smallest, conservative)
      'two_norm_only' 2-norm certifies but inf-norm does not
      'sdp_only'      SDP certifies but neither inf-norm nor 2-norm does
      'non_certified' no sound certificate (may still be contracting [D4 residual]
                      or genuinely expansive — disambiguate with empirical rho)
    """
    if _inf_certifies(gene):
        return "inf"
    if _two_certifies(gene):
        return "two_norm_only"
    if _sdp_certifies(gene):
        return "sdp_only"
    return "non_certified"


if __name__ == "__main__":
    codec = CoupledGeneCodec()
    print("cvxpy:", CVXPY_AVAILABLE)
    for obj in (RotationObjective(), BenignDecayObjective(), NonNormalObjective()):
        print(f"[{obj.name}] target[0..2]=\n{obj._target()[:3]}")
    # reference non-normal gene region:
    print("ref nonnormal region:", classify_region(_NONNORMAL_REF),
          "emp_rho=", round(empirical_spectral_radius(_NONNORMAL_REF, n_samples=4000), 4))
    # a benign decay gene region:
    benign = CoupledGene.make(decay=[0.8, 0.8], W=[[0.0, 0.0], [0.0, 0.0]])
    print("benign gene region:", classify_region(benign))
