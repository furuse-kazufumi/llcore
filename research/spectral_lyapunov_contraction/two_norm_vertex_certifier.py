# SPDX-License-Identifier: Apache-2.0
"""Track D — SOUND induced-2-norm contraction certificate via box-vertex enumeration.

Purely ADDITIVE research code (research/spectral_lyapunov_contraction/). Does NOT touch src/.

Math (verified, see PREREGISTRATION.md):
  The state Jacobian of the n=2 coupled map is
      J(t) = diag(decay) + diag((1 - decay) * t) @ W,   t = (t_1, t_2) in the box [t_lo, 1]^2,
  with t_i = sech^2(pre_i) the per-coordinate tanh derivative. Each entry of J is constant +
  linear in a single t_i, so J is an AFFINE function of t over the box.

  The induced 2-norm ||J||_2 = sigma_max(J) (largest singular value) is a CONVEX function of the
  matrix J (it is a matrix norm; norms are convex). A convex function composed with an affine map
  (t -> J(t)) is convex in t. A convex function on a box attains its maximum at a VERTEX of the box.
  Therefore
      sup_{t in [t_lo,1]^2} ||J(t)||_2  =  max over the 4 box vertices of sigma_max(J(vertex)).
  Computed with numpy SVD at 4 points -- NO solver needed.

Soundness: if max-over-vertices sigma_max(J) < 1, then ||J(t)||_2 < 1 for EVERY achievable t in the
box (the achievable t-set is a subset of the box). ||J||_2 < 1 over the reachable Jacobian set is a
sufficient condition for the map to be a contraction in the Euclidean (2) norm, which implies a
unique fixed point, bounded state, and (pointwise) spectral radius rho(J) <= ||J||_2 < 1.

This certifies a DIFFERENT (and, where W is near-symmetric, LARGER) set than the induced ∞-norm:
for a symmetric matrix sigma_max = rho, so the 2-norm can be tight against the exact contraction
condition exactly where the ∞-norm (which sums absolute row entries) is loose.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass

import numpy as np

# Reuse Track C's map / Jacobian / t_min exactly (additive import, no src/ touch).
_HERE = os.path.dirname(os.path.abspath(__file__))
_TRACK_C = os.path.normpath(os.path.join(_HERE, "..", "coupled_z3_contraction"))
if _TRACK_C not in sys.path:
    sys.path.insert(0, _TRACK_C)

from coupled_map import CoupledGene, t_min_per_coord  # noqa: E402


def jacobian_at_t(gene: CoupledGene, t: np.ndarray) -> np.ndarray:
    """J(t) = diag(decay) + diag((1-decay)*t) @ W, evaluated at an explicit t (NOT via (s,x)).

    This is the affine-in-t form the vertex argument relies on. t is a free 2-vector in the box.
    """
    g = gene.clipped()
    t = np.asarray(t, dtype=np.float64).reshape(2)
    return np.diag(g.decay) + np.diag((1.0 - g.decay) * t) @ g.W


def box_vertices(t_lo: np.ndarray) -> list[np.ndarray]:
    """The 4 vertices of the box [t_lo_0, 1] x [t_lo_1, 1]."""
    t_lo = np.asarray(t_lo, dtype=np.float64).reshape(2)
    return [
        np.array([t_lo[0], t_lo[1]]),
        np.array([t_lo[0], 1.0]),
        np.array([1.0, t_lo[1]]),
        np.array([1.0, 1.0]),
    ]


def sigma_max(J: np.ndarray) -> float:
    """Largest singular value (induced 2-norm) of J via numpy SVD."""
    return float(np.linalg.svd(np.asarray(J, dtype=np.float64), compute_uv=False)[0])


@dataclass(frozen=True)
class TwoNormResult:
    certified: bool
    sup_2norm: float          # max over the 4 box vertices of sigma_max(J)
    domain: str               # "free01" | "tmin1"
    vertex_argmax: tuple      # the t-vertex achieving the sup
    per_vertex: tuple         # (sigma_max at each of the 4 vertices)


def certify_2norm_contraction(gene: CoupledGene, t_domain: str = "tmin1",
                              max_input_abs: float = 1.0) -> TwoNormResult:
    """SOUND induced-2-norm contraction certificate over the t-box.

    t_domain:
      "free01" -> t in [0,1]^2          (looser over-approximation of achievable t)
      "tmin1"  -> t in [t_min,1]^2      (tighter sound floor; t_min = sech^2(max|pre|))

    Returns certified=True iff max-over-4-vertices sigma_max(J) < 1 (strict).
    """
    g = gene.clipped()
    if t_domain == "free01":
        t_lo = np.zeros(2)
    elif t_domain == "tmin1":
        t_lo = t_min_per_coord(g, max_input_abs=max_input_abs)
    else:  # pragma: no cover - guard
        raise ValueError(f"unknown t_domain {t_domain!r}")

    verts = box_vertices(t_lo)
    smaxes = [sigma_max(jacobian_at_t(g, v)) for v in verts]
    k = int(np.argmax(smaxes))
    sup = float(smaxes[k])
    return TwoNormResult(
        certified=bool(sup < 1.0),
        sup_2norm=sup,
        domain=t_domain,
        vertex_argmax=tuple(float(x) for x in verts[k]),
        per_vertex=tuple(float(x) for x in smaxes),
    )


def soundness_selfcheck(gene: CoupledGene, t_domain: str = "tmin1",
                        n_interior: int = 500, seed: int = 0) -> float:
    """Self-check the vertex argument for ONE gene: max(interior sigma_max - vertex sup).

    Returns interior_max - vertex_sup; should be <= ~0 (float noise) if the vertex-max is sound.
    A positive value would falsify the convexity/vertex claim for that gene.
    """
    g = gene.clipped()
    if t_domain == "free01":
        t_lo = np.zeros(2)
    else:
        t_lo = t_min_per_coord(g)
    res = certify_2norm_contraction(g, t_domain=t_domain)
    rng = np.random.default_rng(seed)
    interior_max = 0.0
    for _ in range(n_interior):
        t = np.array([rng.uniform(t_lo[0], 1.0), rng.uniform(t_lo[1], 1.0)])
        interior_max = max(interior_max, sigma_max(jacobian_at_t(g, t)))
    return float(interior_max - res.sup_2norm)


if __name__ == "__main__":
    # Smoke test (prints only, no gating asserts).
    g = CoupledGene.make(decay=[0.3, 0.3], W=[[0.5, 0.9], [0.9, 0.5]])
    for dom in ("free01", "tmin1"):
        r = certify_2norm_contraction(g, t_domain=dom)
        print(f"[{dom}] certified={r.certified} sup_2norm={r.sup_2norm:.6f} "
              f"argmax_t={r.vertex_argmax} per_vertex={tuple(round(v,4) for v in r.per_vertex)}")
    print("soundness self-check (interior - vertex_sup, tmin1):",
          soundness_selfcheck(g, "tmin1", n_interior=2000))
