# SPDX-License-Identifier: Apache-2.0
"""Sharper boundary hunt: drive verified_pd into its ok=True path with matrices whose true
lambda_min sits just below 0, where a false positive would live. Also stress the bound's
assumptions (small maxdiag + tiny negative eig; Cholesky-succeeds-but-indefinite scenarios).
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rump_pd import verified_pd, _cholesky_error_bound, _spd_cholesky  # noqa: E402

try:
    import mpmath as mp
    mp.mp.dps = 80
    HAVE_MP = True
except Exception:
    HAVE_MP = False


def true_min_eig(M):
    Ms = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    return float(np.min(np.linalg.eigvalsh(Ms)))


def mp_min_eig(M):
    if not HAVE_MP:
        return None
    Ms = 0.5 * (np.asarray(M, float) + np.asarray(M, float).T)
    n = Ms.shape[0]
    A = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = mp.mpf(repr(float(Ms[i, j])))
    E, _ = mp.eigsy(A)
    return float(min(E[i] for i in range(n)))


false_positives = []
lb_violations = []
trials = 0
certified = 0


def check(M, desc):
    global trials, certified
    trials += 1
    ok, lb = verified_pd(M)
    if not ok:
        return
    certified += 1
    tmin = true_min_eig(M)
    if tmin <= 0.0:
        rec = {"desc": desc, "true_min_eig_float": tmin, "returned_lb": lb,
               "n": int(np.asarray(M).shape[0])}
        if HAVE_MP:
            try:
                rec["mp_min_eig"] = mp_min_eig(M)
            except Exception as e:
                rec["mp_min_eig"] = f"err:{e}"
        false_positives.append(rec)
    if lb > tmin + 1e-12 * max(1.0, abs(tmin)):
        lb_violations.append({"desc": desc, "lb": lb, "true_min_eig_float": tmin})


rng = np.random.default_rng(7777)

# ---- A) Boundary bisection: for each random base, find the shift where verified_pd flips
#         ok True->False, then probe matrices straddling it and check their true min eig. ----
for trial in range(2000):
    n = int(rng.integers(2, 8))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    scale = 10 ** rng.uniform(-3, 6)
    base_evs = (np.abs(rng.standard_normal(n)) + 0.1) * scale

    def make(min_eig):
        evs = base_evs.copy()
        evs[0] = min_eig
        return (Q * evs) @ Q.T

    # bisect on min_eig in a tiny window around 0 to find verified_pd's accept threshold
    lo, hi = -1e-3 * scale, 1e-3 * scale
    # ensure lo rejects, hi accepts (hi PD enough)
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        ok, _ = verified_pd(make(mid))
        if ok:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-300 or (hi != 0 and abs(hi - lo) < 1e-15 * abs(hi)):
            break
    # probe a fine grid of min_eig values around the discovered threshold `hi`
    thr = hi
    for delta in (0.0, -1e-18, -1e-16, -1e-14, -1e-12, 1e-16, 1e-14, -thr * 1e-9 if thr else 0.0):
        me = thr + delta
        check(make(me), f"boundary-bisect n={n} scale={scale:.2e} min_eig~{me:.3e}")
    # also directly probe matrices with exact tiny-negative target min eig at this scale
    for me in (-1e-300, -np.finfo(float).tiny, -1e-15 * scale, 0.0):
        check(make(me), f"boundary-explicit n={n} scale={scale:.2e} me={me:.3e}")

# ---- B) Small-maxdiag + tiny negative eig: err_bound = gamma_{n+1}*n*maxdiag is tiny,
#         so alpha need only clear a tiny bound; does a non-PD matrix slip through? ----
for trial in range(2000):
    n = int(rng.integers(2, 8))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    # keep entries small so maxdiag (and thus err_bound) is tiny
    small = 10 ** rng.uniform(-8, -1)
    evs = (np.abs(rng.standard_normal(n)) + 0.1) * small
    evs[0] = -10 ** rng.uniform(-300, -1) * small  # tiny negative relative to scale
    M = (Q * evs) @ Q.T
    check(M, f"small-scale-tinyNeg n={n} small={small:.2e} negeig={evs[0]:.3e}")

# ---- C) Matrices where float Cholesky might succeed despite indefiniteness:
#         strongly diagonally weighted with a near-cancelling negative direction. ----
for trial in range(2000):
    n = int(rng.integers(2, 6))
    # build A = M - alpha*I implicitly: choose M near the cliff where chol(M-alpha I) just passes
    A = rng.standard_normal((n, n))
    M0 = A @ A.T
    tmin = true_min_eig(M0)
    # subtract just past tmin so true min eig is a hair negative; vary the hair across scales
    over = tmin + 10 ** rng.uniform(-20, -8) * (abs(tmin) + 1.0)
    M = M0 - over * np.eye(n)
    check(M, f"chol-cliff n={n} tmin0={tmin:.3e}")

# ---- D) Repeated tiny-negative eigenvalue (multiplicity) + huge positive block ----
for trial in range(1000):
    n = int(rng.integers(3, 9))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    big = 10 ** rng.uniform(2, 8)
    evs = np.full(n, big)
    k = int(rng.integers(1, n))  # k tiny-negative eigenvalues
    evs[:k] = -10 ** rng.uniform(-12, -2)
    M = (Q * evs) @ Q.T
    check(M, f"multi-negeig n={n} k={k} big={big:.2e}")

# ---- E) Asymmetric input whose symmetric part is indefinite (verified_pd symmetrizes) ----
for trial in range(1000):
    n = int(rng.integers(2, 7))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    evs = (np.abs(rng.standard_normal(n)) + 0.1) * 10 ** rng.uniform(-2, 4)
    evs[0] = -10 ** rng.uniform(-12, -3)
    S = (Q * evs) @ Q.T  # symmetric, indefinite
    skew = rng.standard_normal((n, n))
    skew = skew - skew.T
    M = S + 50.0 * skew  # large skew part; sym part still indefinite
    check(M, f"asym-symIndef n={n} negeig={evs[0]:.3e}")

# ---- F) Direct exact-zero eigenvalue via Gram of orthonormal-deficient set ----
for trial in range(1000):
    n = int(rng.integers(2, 9))
    Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
    evs = (np.abs(rng.standard_normal(n)) + 0.1) * 10 ** rng.uniform(-1, 5)
    evs[0] = 0.0
    M = (Q * evs) @ Q.T
    M = 0.5 * (M + M.T)
    check(M, f"exact-zero-eig n={n}")


print(f"TRIALS={trials}")
print(f"CERTIFIED_OK_TRUE={certified}")
print(f"FALSE_POSITIVES={len(false_positives)}")
print(f"LB_VIOLATIONS={len(lb_violations)}")
print(f"HAVE_MP={HAVE_MP}")
if false_positives:
    print("---- FALSE POSITIVE RECORDS (up to 30) ----")
    for r in false_positives[:30]:
        print(r)
if lb_violations:
    print("---- LB VIOLATIONS (up to 30) ----")
    for r in lb_violations[:30]:
        print(r)
