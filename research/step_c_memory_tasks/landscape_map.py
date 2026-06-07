# SPDX-License-Identifier: Apache-2.0
"""C1 多峰性診断 — 収束点間の中点が谷になるかで分離 peak の存在を測る (step4 C1 手法)."""
from __future__ import annotations

import numpy as np


def _hillclimb(eval_once, dim, bounds, n_evals, sigma, rng):
    lo, hi = bounds
    g = lo + (hi - lo) * rng.random(dim)
    f = eval_once(g, rng)
    for _ in range(n_evals - 1):
        cand = np.clip(g + rng.normal(0, sigma, size=dim), lo, hi)
        cf = eval_once(cand, rng)
        if cf >= f:
            g, f = cand, cf
    return g, f


def multimodality_report(eval_once, *, dim, bounds, n_restarts, n_evals, sigma, base_seed):
    """n_restarts 回 hill-climb し収束点を集め、ペアの中点が谷になる割合を測る."""
    optima = []
    for i in range(n_restarts):
        g, f = _hillclimb(eval_once, dim, bounds, n_evals, sigma,
                          np.random.default_rng(base_seed + i))
        optima.append((g, f))
    valley = 0
    pairs = 0
    for i in range(len(optima)):
        for j in range(i + 1, len(optima)):
            gi, fi = optima[i]
            gj, fj = optima[j]
            if np.allclose(gi, gj, atol=1e-2):
                continue
            mid = 0.5 * (gi + gj)
            fm = float(np.mean([eval_once(mid, np.random.default_rng(base_seed + 999 + k))
                                for k in range(3)]))
            pairs += 1
            if fm < min(fi, fj) - 0.05 * (abs(min(fi, fj)) + 1e-9):
                valley += 1
    frac = valley / pairs if pairs else 0.0
    return {
        "n_optima": len(optima),
        "valley_fraction": frac,
        "is_multimodal": frac >= 0.2,
    }
