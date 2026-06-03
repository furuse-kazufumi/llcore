# SPDX-License-Identifier: Apache-2.0
"""Verifier backend plugins (Stage 3b) — pluggable contraction soundness gates.

The llcore CPU-Verification arc concluded that the right contraction verifier is
**SDP/Lyapunov (LMI), not Z3/SMT**: for the scalar / diagonal RWKV gene the contraction
condition is closed-form (Z3 decorative), while for **coupled** (non-diagonal) genes a common
quadratic Lyapunov matrix certifies a strictly larger class than any fixed induced norm. This
module exposes that conclusion as **pluggable backends** so the evolution loop can choose its
soundness gate without hard-coding a certifier.

Backends (all *fail-closed*: an unavailable or erroring backend certifies nothing):

- :class:`ClosedFormScalarBackend` — the scalar RWKV-gene closed-form Lipschitz contraction
  (reuses :func:`llcore.verifier.verify_lipschitz_contraction`). No extra dependencies; the
  default backend.
- :class:`InfNormBackend` / :class:`TwoNormBackend` — coupled-gene induced-norm contraction
  over the achievable-t box (closed-form / vertex SVD; numpy only).
- :class:`SdpLyapunovBackend` — coupled-gene common quadratic Lyapunov LMI (cvxpy, optional
  ``[sdp]`` extra). Certifies P-weighted contractions no fixed norm captures. ``available``
  is False if cvxpy is absent, and :meth:`certifies` then returns False (fail-closed).

Gene contract (duck-typed):
- a **scalar** gene exposes ``decay``/``mix``/``gate_str`` floats (``StateUpdateGene``);
- a **coupled** gene exposes ``decay`` shape (n,) and ``W`` shape (n,n) arrays (and optional
  ``V``; defaults to identity). The coupled backends read these; they do not import any
  research module.

All coupled backends certify contraction over the *sound* achievable-t box
``t_i ∈ [t_min_i, 1]`` with ``t_min_i = sech²(Σ_j|W_ij| + max_input·Σ_j|V_ij|)`` — a tighter,
still-sound floor than the free ``[0,1]`` box. A True verdict implies ρ(J)<1 over the box.
"""
from __future__ import annotations

import itertools
from typing import Any, Protocol, runtime_checkable

import numpy as np

try:
    import cvxpy as _cp

    _CVXPY_AVAILABLE = True
except Exception:  # pragma: no cover - environment dependent
    _CVXPY_AVAILABLE = False


def cvxpy_available() -> bool:
    """True iff cvxpy (the SDP backend's solver) is importable."""
    return _CVXPY_AVAILABLE


# --------------------------------------------------------------------------- #
# Backend protocol.
# --------------------------------------------------------------------------- #


@runtime_checkable
class VerifierBackend(Protocol):
    """A soundness gate: ``certifies(gene)`` is True iff ``gene`` is provably contracting."""

    name: str

    @property
    def available(self) -> bool: ...

    def certifies(self, gene: Any) -> bool: ...


# --------------------------------------------------------------------------- #
# Coupled-gene helpers (numpy only).
# --------------------------------------------------------------------------- #


def _coupled_arrays(gene: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract (decay (n,), W (n,n), V (n,n)) from a duck-typed coupled gene, clipped to the
    legal box decay∈[0,1], W∈[-2,2]."""
    decay = np.clip(np.asarray(gene.decay, dtype=np.float64).reshape(-1), 0.0, 1.0)
    n = decay.shape[0]
    W = np.clip(np.asarray(gene.W, dtype=np.float64).reshape(n, n), -2.0, 2.0)
    V = getattr(gene, "V", None)
    V = np.eye(n) if V is None else np.asarray(V, dtype=np.float64).reshape(n, n)
    return decay, W, V


def _t_min(decay: np.ndarray, W: np.ndarray, V: np.ndarray, max_input_abs: float) -> np.ndarray:
    M = np.abs(W).sum(axis=1) + max_input_abs * np.abs(V).sum(axis=1)
    return 1.0 - np.tanh(M) ** 2


def _jac_at_t(decay: np.ndarray, W: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.diag(decay) + np.diag((1.0 - decay) * t) @ W


def _box_vertices(t_lo: np.ndarray) -> list[np.ndarray]:
    n = t_lo.shape[0]
    return [np.array([(1.0 if b else t_lo[i]) for i, b in enumerate(c)])
            for c in itertools.product((0, 1), repeat=n)]


def _infnorm_sup(decay: np.ndarray, W: np.ndarray, t_lo: np.ndarray) -> float:
    n = decay.shape[0]
    best = 0.0
    for i in range(n):
        off = sum(abs(W[i, j]) for j in range(n) if j != i)
        for ti in (t_lo[i], 1.0):
            diag = abs(decay[i] + (1.0 - decay[i]) * ti * W[i, i])
            best = max(best, diag + (1.0 - decay[i]) * ti * off)
    return float(best)


# --------------------------------------------------------------------------- #
# Backends.
# --------------------------------------------------------------------------- #


class ClosedFormScalarBackend:
    """Default backend: scalar RWKV-gene closed-form Lipschitz contraction (Stage 1b)."""

    name = "closed_form_scalar"

    @property
    def available(self) -> bool:
        from .invariants import is_z3_available  # noqa: F401 (z3 optional; closed form needs no z3)
        return True

    def certifies(self, gene: Any) -> bool:
        from .invariants import verify_lipschitz_contraction
        try:
            return verify_lipschitz_contraction(gene).contraction is True
        except Exception:
            return False


class InfNormBackend:
    """Coupled-gene induced-∞-norm contraction (closed-form, numpy only)."""

    name = "inf_norm"

    @property
    def available(self) -> bool:
        return True

    def certifies(self, gene: Any, max_input_abs: float = 1.0) -> bool:
        try:
            decay, W, V = _coupled_arrays(gene)
            return bool(_infnorm_sup(decay, W, _t_min(decay, W, V, max_input_abs)) < 1.0)
        except Exception:
            return False


class TwoNormBackend:
    """Coupled-gene induced-2-norm contraction (max σ_max over 2^n box vertices)."""

    name = "two_norm"

    @property
    def available(self) -> bool:
        return True

    def certifies(self, gene: Any, max_input_abs: float = 1.0) -> bool:
        try:
            decay, W, V = _coupled_arrays(gene)
            t_lo = _t_min(decay, W, V, max_input_abs)
            return all(float(np.linalg.svd(_jac_at_t(decay, W, v), compute_uv=False)[0]) < 1.0
                       for v in _box_vertices(t_lo))
        except Exception:
            return False


class SdpLyapunovBackend:
    """Coupled-gene common quadratic Lyapunov (LMI) contraction via cvxpy (optional [sdp]).

    Certifies a P ≻ 0 with P − J_vᵀ P J_v ≻ 0 at every t-box vertex ⇒ P-weighted contraction
    ⇒ ρ(J)<1 over the box. Strictly richer than any fixed induced norm. Fail-closed when cvxpy
    is unavailable. Fast-path: 2-norm ⇒ True (P=I); pre-screen ρ(J_v)<1 (necessary). The
    solver's P is re-verified by an independent eigenvalue check (never solver-blind)."""

    name = "sdp_lyapunov"

    @property
    def available(self) -> bool:
        return _CVXPY_AVAILABLE

    def certifies(self, gene: Any, max_input_abs: float = 1.0, margin: float = 1e-7) -> bool:
        try:
            decay, W, V = _coupled_arrays(gene)
            n = decay.shape[0]
            t_lo = _t_min(decay, W, V, max_input_abs)
            verts = _box_vertices(t_lo)
            Js = [_jac_at_t(decay, W, v) for v in verts]
            # 2-norm fast path (P=I) — also makes the backend usable without cvxpy for that subset.
            if all(float(np.linalg.svd(J, compute_uv=False)[0]) < 1.0 for J in Js):
                return True
            if not _CVXPY_AVAILABLE:
                return False
            for J in Js:  # necessary condition for the LMI
                if float(np.max(np.abs(np.linalg.eigvals(J)))) >= 1.0:
                    return False
            P = _cp.Variable((n, n), symmetric=True)
            I = np.eye(n)
            cons = [P >> I] + [P - J.T @ P @ J >> margin * I for J in Js]
            _cp.Problem(_cp.Minimize(_cp.trace(P)), cons).solve()
            if P.value is None:
                return False
            Pv = 0.5 * (P.value + P.value.T)
            if float(np.min(np.linalg.eigvalsh(Pv))) <= 0.0:
                return False
            for J in Js:
                M = Pv - J.T @ Pv @ J
                if float(np.min(np.linalg.eigvalsh(0.5 * (M + M.T)))) <= 0.0:
                    return False
            return True
        except Exception:
            return False


_REGISTRY = {
    "closed_form_scalar": ClosedFormScalarBackend,
    "inf_norm": InfNormBackend,
    "two_norm": TwoNormBackend,
    "sdp_lyapunov": SdpLyapunovBackend,
}


def get_verifier_backend(name: str = "closed_form_scalar") -> VerifierBackend:
    """Return a verifier backend by name. Names: ``closed_form_scalar`` (default),
    ``inf_norm``, ``two_norm``, ``sdp_lyapunov``. Raises KeyError for unknown names."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown verifier backend {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def available_backends() -> dict[str, bool]:
    """Map backend name -> whether its dependencies are present (sdp_lyapunov needs cvxpy)."""
    return {name: cls().available for name, cls in _REGISTRY.items()}
