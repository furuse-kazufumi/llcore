# SPDX-License-Identifier: Apache-2.0
"""Tests for the Rump verified-PD + OR-of-solvers gate (Pilot A).

Invariants under test (pre-registered in RUMP_OR_PREREGISTRATION.md):
  (a) clearly-PD matrices are certified by Rump;
  (b) indefinite & near-singular (lambda_min slightly negative) matrices are REJECTED;
  (c) coverage: the Rump-certified set is a SUPERSET of the float
      ``np.linalg.eigvalsh(min > 1e-9)``-certified set (no coverage loss) on a battery;
  (d) soundness: over 200+ random symmetric matrices spanning PD/indefinite there are ZERO
      Rump false positives vs exact ``eigvalsh`` ground truth.

Pure numpy; no new dependencies. Self-contained (does not touch src/ or existing certifiers).
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rump_pd import verified_pd  # noqa: E402


# --------------------------------------------------------------------------- #
# Helpers.
# --------------------------------------------------------------------------- #


def _true_min_eig(M: np.ndarray) -> float:
    M = 0.5 * (np.asarray(M, dtype=np.float64) + np.asarray(M, dtype=np.float64).T)
    return float(np.min(np.linalg.eigvalsh(M)))


def _random_symmetric(rng: np.random.Generator, n: int, shift: float) -> np.ndarray:
    """Random symmetric n x n matrix with a controllable diagonal shift (shift>0 pushes PD)."""
    A = rng.standard_normal((n, n))
    S = 0.5 * (A + A.T)
    return S + shift * np.eye(n)


# --------------------------------------------------------------------------- #
# (a) clearly-PD matrices are certified.
# --------------------------------------------------------------------------- #


def test_a_identity_certified():
    ok, lb = verified_pd(np.eye(3))
    assert ok is True
    assert lb > 0.0
    assert lb <= 1.0 + 1e-9  # sound lower bound on lambda_min == 1


def test_a_well_conditioned_spd_certified():
    rng = np.random.default_rng(1)
    for n in (2, 3, 4, 5):
        A = rng.standard_normal((n, n))
        M = A @ A.T + n * np.eye(n)  # min eig >= ~n
        ok, lb = verified_pd(M)
        assert ok is True, f"n={n} should certify"
        # lower bound must be sound: lb <= true min eig.
        assert lb <= _true_min_eig(M) + 1e-9
        assert lb > 0.0


def test_a_diagonal_pd_certified():
    M = np.diag([1.0, 5.0, 0.25, 10.0])
    ok, lb = verified_pd(M)
    assert ok is True
    # true min eig is 0.25; the guaranteed lower bound must not exceed it.
    assert 0.0 < lb <= 0.25 + 1e-9


# --------------------------------------------------------------------------- #
# (b) indefinite & near-singular are REJECTED (soundness, no false positive).
# --------------------------------------------------------------------------- #


def test_b_indefinite_rejected():
    M = np.diag([1.0, -1.0, 2.0])
    ok, _ = verified_pd(M)
    assert ok is False


def test_b_slightly_negative_min_eig_rejected():
    # lambda_min = -1e-6 (just below 0): MUST be rejected (false positive forbidden).
    M = np.diag([3.0, -1e-6, 2.0])
    ok, _ = verified_pd(M)
    assert ok is False


def test_b_exactly_singular_rejected():
    M = np.diag([1.0, 0.0, 2.0])  # lambda_min = 0 (PSD but not PD)
    ok, _ = verified_pd(M)
    assert ok is False


def test_b_near_singular_offdiag_rejected():
    # 2x2 with eigenvalues {2, -tiny}: [[a, b],[b, c]] indefinite by det < 0.
    M = np.array([[1.0, 1.0 + 1e-7], [1.0 + 1e-7, 1.0]])  # det = 1 - (1+1e-7)^2 < 0 => indefinite
    assert _true_min_eig(M) < 0.0
    ok, _ = verified_pd(M)
    assert ok is False


def test_b_nonfinite_rejected():
    M = np.array([[1.0, np.nan], [np.nan, 1.0]])
    ok, _ = verified_pd(M)
    assert ok is False


# --------------------------------------------------------------------------- #
# (c) coverage: Rump-certified ⊇ float-eigvalsh(min>1e-9)-certified (no coverage loss).
# --------------------------------------------------------------------------- #


def test_c_coverage_superset_of_float():
    """Every matrix the float test (min eig > 1e-9) accepts must also be Rump-accepted, OR have
    a min eig so small that float-acceptance was itself near the noise floor. We assert the strong
    form on matrices with a comfortable margin (min eig >= 1e-6), where Rump must not lose them."""
    rng = np.random.default_rng(7)
    n_float_certified = 0
    n_rump_also = 0
    losses = []
    COMFORT = 1e-6  # comfortably above the float test's 1e-9 floor and Rump's ~n*u*maxdiag bound
    for _ in range(400):
        n = int(rng.integers(2, 6))
        shift = float(rng.uniform(-1.0, 4.0))
        M = _random_symmetric(rng, n, shift)
        true_min = _true_min_eig(M)
        float_ok = true_min > 1e-9
        if not float_ok:
            continue
        n_float_certified += 1
        rump_ok, _ = verified_pd(M)
        if rump_ok:
            n_rump_also += 1
        elif true_min >= COMFORT:
            # A comfortably-PD matrix that Rump lost = a coverage regression. Record it.
            losses.append((n, true_min))
    # No coverage loss on comfortably-PD matrices.
    assert losses == [], f"Rump lost comfortably-PD matrices (coverage regression): {losses}"
    # Sanity: the battery actually exercised the float-certified branch.
    assert n_float_certified > 50
    # And Rump recovers the vast majority overall (the only ones it can lose are within the
    # tiny [1e-9, 1e-6] near-singular band, which the spec permits under-certifying).
    assert n_rump_also >= int(0.9 * n_float_certified)


# --------------------------------------------------------------------------- #
# (d) soundness battery: 200+ random symmetric matrices, ZERO Rump false positives.
# --------------------------------------------------------------------------- #


def test_d_soundness_zero_false_positives():
    """Over a broad battery spanning PD / PSD / indefinite, assert Rump NEVER certifies a matrix
    whose exact ``eigvalsh`` min eigenvalue is <= 0 (a false positive). This is the cardinal
    soundness invariant: under-certification is fine, a false positive is forbidden."""
    rng = np.random.default_rng(2024)
    n_total = 0
    n_pd = 0
    n_indef = 0
    n_rump_true = 0
    false_positives = []
    for _ in range(600):
        n = int(rng.integers(2, 7))
        # Span the boundary: shifts from clearly-indefinite to clearly-PD, plus deliberately
        # near-singular cases (shift chosen to put min eig within +-1e-7 of 0).
        mode = rng.integers(0, 3)
        S = 0.5 * (lambda A: A + A.T)(rng.standard_normal((n, n)))
        if mode == 0:
            M = S + float(rng.uniform(-2.0, 4.0)) * np.eye(n)
        elif mode == 1:
            # force min eig into a tiny band around 0 (the adversarial near-boundary case).
            mn = _true_min_eig(S)
            target = float(rng.uniform(-1e-6, 1e-6))
            M = S + (target - mn) * np.eye(n)
        else:
            # scaled indefinite (mixed sign eigenvalues likely).
            M = S * float(rng.uniform(0.1, 3.0))
        n_total += 1
        true_min = _true_min_eig(M)
        rump_ok, lb = verified_pd(M)
        if true_min > 0:
            n_pd += 1
        else:
            n_indef += 1
        if rump_ok:
            n_rump_true += 1
            # FALSE POSITIVE check: Rump said PD but exact min eig <= 0.
            if true_min <= 0.0:
                false_positives.append({"true_min": true_min, "lb": lb, "n": n})
            # When Rump certifies, its returned lower bound must be SOUND (<= true min eig).
            assert lb <= true_min + 1e-12, (
                f"unsound lower bound: lb={lb} > true_min={true_min}")
    assert n_total >= 200, "battery must exercise 200+ matrices"
    assert false_positives == [], f"Rump FALSE POSITIVES (forbidden): {false_positives}"
    # Sanity: the battery actually spanned both PD and indefinite, and Rump certified a healthy
    # fraction of the genuinely-PD ones (otherwise the soundness test is vacuous).
    assert n_pd > 30 and n_indef > 30
    assert n_rump_true > 0


def test_d_lower_bound_always_sound_when_certified():
    """Focused: whenever verified_pd returns ok=True, the returned lb is a sound underestimate of
    the true minimum eigenvalue."""
    rng = np.random.default_rng(99)
    checked = 0
    for _ in range(300):
        n = int(rng.integers(2, 6))
        A = rng.standard_normal((n, n))
        M = A @ A.T + float(rng.uniform(0.0, 2.0)) * np.eye(n)
        ok, lb = verified_pd(M)
        if ok:
            checked += 1
            assert lb > 0.0
            assert lb <= _true_min_eig(M) + 1e-12
    assert checked > 50


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
