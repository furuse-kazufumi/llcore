# SPDX-License-Identifier: Apache-2.0
"""Adversarial soundness audit for verified_pd. NOT committed to src/. Read-only on rump_pd.py.

Goal: find a symmetric M with exact lambda_min(M) <= 0 (per numpy.linalg.eigvalsh float64,
confirmed by mpmath on borderline cases) for which verified_pd(M) returns ok=True. That is a
FALSE POSITIVE = unsoundness. Also verify returned lam_min_lb is a true lower bound on certified
cases.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from rump_pd import verified_pd  # noqa: E402

try:
    import mpmath as mp
    mp.mp.dps = 60
    HAVE_MP = True
except Exception:
    HAVE_MP = False


def true_min_eig(M):
    M = np.asarray(M, dtype=np.float64)
    Ms = 0.5 * (M + M.T)
    return float(np.min(np.linalg.eigvalsh(Ms)))


def mp_min_eig(M):
    """High-precision min eigenvalue via mpmath (symmetric)."""
    if not HAVE_MP:
        return None
    M = np.asarray(M, dtype=np.float64)
    Ms = 0.5 * (M + M.T)
    n = Ms.shape[0]
    A = mp.matrix(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = mp.mpf(float(Ms[i, j]))
    try:
        E, _ = mp.eigsy(A)
        return float(min(E[i] for i in range(n)))
    except Exception:
        # fallback: char poly roots for small n
        return None


class Audit:
    def __init__(self):
        self.trials = 0
        self.false_positives = []   # records
        self.lb_violations = []     # records where ok=True but lb > true_min
        self.certified = 0

    def check(self, M, desc):
        self.trials += 1
        M = np.asarray(M, dtype=np.float64)
        ok, lb = verified_pd(M)
        tmin = true_min_eig(M)
        if ok:
            self.certified += 1
            # soundness: ok=True requires tmin > 0 (true PD)
            if tmin <= 0.0:
                rec = {"desc": desc, "true_min_eig_float": tmin,
                       "returned_ok": True, "returned_lb": lb, "n": int(M.shape[0])}
                if HAVE_MP:
                    mm = mp_min_eig(M)
                    rec["mp_min_eig"] = mm
                self.false_positives.append(rec)
            # lower-bound soundness: lb must be <= true min eig
            # allow a tiny float tolerance on the eig computation itself
            if lb > tmin + 1e-12 * max(1.0, abs(tmin)):
                self.lb_violations.append({"desc": desc, "lb": lb,
                                           "true_min_eig_float": tmin,
                                           "n": int(M.shape[0])})
        return ok, lb, tmin


def hilbert(n):
    return np.array([[1.0 / (i + j + 1) for j in range(n)] for i in range(n)])


def pascal(n):
    P = np.zeros((n, n))
    for i in range(n):
        P[i, 0] = 1.0
        P[0, i] = 1.0
    for i in range(1, n):
        for j in range(1, n):
            P[i, j] = P[i - 1, j] + P[i, j - 1]
    return P


def run():
    a = Audit()
    rng = np.random.default_rng(20260603)

    # ---- 1) near-singular: lambda_min in [-1e-12, 0] via eigendecomposition reconstruction ----
    for _ in range(400):
        n = int(rng.integers(2, 8))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        # eigenvalues: one tiny (<=0), rest positive O(1..1e3)
        target = float(rng.uniform(-1e-12, 0.0))
        rest = np.abs(rng.standard_normal(n - 1)) * 10 ** rng.uniform(0, 3) + 1e-3
        evs = np.concatenate([[target], rest])
        M = (Q * evs) @ Q.T
        M = 0.5 * (M + M.T)
        a.check(M, f"near-singular reconstruct n={n} target={target:.3e}")

    # ---- 2) exactly-zero min eig (PSD but singular) ----
    for _ in range(300):
        n = int(rng.integers(2, 8))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        evs = np.abs(rng.standard_normal(n)) * 10 ** rng.uniform(-2, 3) + 1e-6
        evs[0] = 0.0  # exact zero
        M = (Q * evs) @ Q.T
        M = 0.5 * (M + M.T)
        a.check(M, f"PSD-singular zero-eig n={n}")

    # diagonal PSD-singular (zero on diagonal) - easy exact-zero cases
    for _ in range(200):
        n = int(rng.integers(2, 8))
        d = np.abs(rng.standard_normal(n)) * 10 ** rng.uniform(-3, 6) + 1e-3
        d[rng.integers(0, n)] = 0.0
        a.check(np.diag(d), f"diag PSD-singular n={n}")

    # ---- 3) tiny-negative diagonal entry buried among huge positive ones (scaling attack) ----
    for _ in range(300):
        n = int(rng.integers(2, 8))
        d = np.abs(rng.standard_normal(n)) * 10 ** rng.uniform(3, 9) + 1.0  # huge positives
        idx = int(rng.integers(0, n))
        d[idx] = float(rng.uniform(-1e-6, -1e-15))  # tiny negative
        M = np.diag(d)
        a.check(M, f"diag huge+tinyNeg n={n} neg={d[idx]:.3e} maxdiag={d.max():.3e}")

    # off-diagonal version: huge diag, one negative eigenvalue induced
    for _ in range(300):
        n = int(rng.integers(2, 8))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        big = 10 ** rng.uniform(3, 9)
        evs = np.abs(rng.standard_normal(n)) * big + big
        evs[0] = float(rng.uniform(-1.0, -1e-9))  # negative eig, but tiny vs scale
        M = (Q * evs) @ Q.T
        M = 0.5 * (M + M.T)
        a.check(M, f"bigscale-tinyNegEig n={n} neg={evs[0]:.3e} scale={big:.3e}")

    # ---- 4) badly-scaled mixed 1e6 / 1e-6 ----
    for _ in range(300):
        n = int(rng.integers(2, 8))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        evs = np.empty(n)
        for i in range(n):
            evs[i] = 10 ** rng.uniform(-6, 6)
        # make one slightly negative or zero
        if rng.random() < 0.5:
            evs[0] = -10 ** rng.uniform(-12, -6)
        else:
            evs[0] = 0.0
        M = (Q * evs) @ Q.T
        M = 0.5 * (M + M.T)
        a.check(M, f"badscale n={n} min_target={evs[0]:.3e}")

    # ---- 5) ill-conditioned (cond 1e8+) with min eig 0 or tiny negative ----
    for _ in range(300):
        n = int(rng.integers(2, 8))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        cond = 10 ** rng.uniform(8, 15)
        evs = np.geomspace(1.0, cond, n)[::-1]  # large to small
        evs = evs * 10 ** rng.uniform(-3, 3)
        evs[-1] = float(rng.choice([0.0, -evs[0] / cond / 10, -1e-14]))
        M = (Q * evs) @ Q.T
        M = 0.5 * (M + M.T)
        a.check(M, f"illcond n={n} cond={cond:.3e} min={evs[-1]:.3e}")

    # ---- 6) classic hard matrices with a tiny negative shift ----
    for n in range(2, 13):
        H = hilbert(n)
        tmin = true_min_eig(H)
        # shift to make exactly indefinite: subtract just over tmin
        for eps_factor in (1.0 + 1e-9, 1.0 + 1e-6, 1.0 + 1e-3, 2.0):
            M = H - (tmin * eps_factor) * np.eye(n)
            a.check(M, f"Hilbert{n} shifted x{eps_factor}")
        # also raw Hilbert (very small but positive min eig) — should be under-certified or fine
        a.check(H, f"Hilbert{n} raw")

    for n in range(2, 11):
        P = pascal(n)
        tmin = true_min_eig(P)
        for eps_factor in (1.0 + 1e-9, 1.0 + 1e-6, 1.0 + 1e-3, 2.0):
            M = P - (tmin * eps_factor) * np.eye(n)
            a.check(M, f"Pascal{n} shifted x{eps_factor}")
        a.check(P, f"Pascal{n} raw")

    # ---- 7) low-rank + tiny negative shift ----
    for _ in range(300):
        n = int(rng.integers(2, 8))
        r = int(rng.integers(1, n))  # rank deficient
        B = rng.standard_normal((n, r)) * 10 ** rng.uniform(-1, 3)
        L = B @ B.T  # rank r PSD, lambda_min = 0
        shift = float(rng.uniform(-1e-6, -1e-15))
        M = L + shift * np.eye(n)  # now indefinite (lambda_min = shift < 0)
        M = 0.5 * (M + M.T)
        a.check(M, f"lowrank+negshift n={n} r={r} shift={shift:.3e}")

    # low-rank exact (lambda_min == 0)
    for _ in range(200):
        n = int(rng.integers(2, 8))
        r = int(rng.integers(1, n))
        B = rng.standard_normal((n, r)) * 10 ** rng.uniform(-1, 4)
        M = B @ B.T
        M = 0.5 * (M + M.T)
        a.check(M, f"lowrank exact-singular n={n} r={r}")

    # ---- 8) random indefinite with one tiny negative eigenvalue (broad) ----
    for _ in range(500):
        n = int(rng.integers(2, 9))
        Q, _ = np.linalg.qr(rng.standard_normal((n, n)))
        evs = np.abs(rng.standard_normal(n)) * 10 ** rng.uniform(-2, 4) + 1e-4
        evs[0] = -10 ** rng.uniform(-15, -3)  # tiny negative across many scales
        M = (Q * evs) @ Q.T
        M = 0.5 * (M + M.T)
        a.check(M, f"rand-indef-tinyNeg n={n} neg={evs[0]:.3e}")

    # ---- 9) 2x2 closed-form indefinite (det<0) sweeping the boundary ----
    for _ in range(400):
        b = float(rng.uniform(0.1, 1e6))
        # [[a, b],[b, c]] with a*c just below b^2 => det<0 => one negative eig
        a_val = float(rng.uniform(0.1, 1e6))
        c_val = (b * b) / a_val * (1.0 - 10 ** rng.uniform(-15, -3))  # a*c < b^2
        M = np.array([[a_val, b], [b, c_val]])
        a.check(M, f"2x2 det<0 a={a_val:.3e} b={b:.3e}")

    # 2x2 det exactly 0 (PSD singular)
    for _ in range(200):
        b = float(rng.uniform(0.1, 1e6))
        a_val = float(rng.uniform(0.1, 1e6))
        c_val = (b * b) / a_val  # a*c == b^2 => det ~ 0 (one zero eig)
        M = np.array([[a_val, b], [b, c_val]])
        a.check(M, f"2x2 det~0 a={a_val:.3e} b={b:.3e}")

    # ---- 10) direct shift sweep: build PD then subtract exactly its min eig (and a hair more) ----
    for _ in range(400):
        n = int(rng.integers(2, 8))
        A = rng.standard_normal((n, n))
        M0 = A @ A.T + 10 ** rng.uniform(-3, 3) * np.eye(n)
        tmin = true_min_eig(M0)
        # subtract slightly more than tmin so result is indefinite by a hair
        over = tmin * (1.0 + 10 ** rng.uniform(-15, -6))
        M = M0 - over * np.eye(n)
        M = 0.5 * (M + M.T)
        a.check(M, f"shift-to-boundary n={n}")

    return a


if __name__ == "__main__":
    a = run()
    print(f"TRIALS={a.trials}")
    print(f"CERTIFIED_OK_TRUE={a.certified}")
    print(f"FALSE_POSITIVES={len(a.false_positives)}")
    print(f"LB_VIOLATIONS={len(a.lb_violations)}")
    print(f"HAVE_MP={HAVE_MP}")
    if a.false_positives:
        print("---- FALSE POSITIVE RECORDS (up to 20) ----")
        for r in a.false_positives[:20]:
            print(r)
    if a.lb_violations:
        print("---- LB VIOLATION RECORDS (up to 20) ----")
        for r in a.lb_violations[:20]:
            print(r)
