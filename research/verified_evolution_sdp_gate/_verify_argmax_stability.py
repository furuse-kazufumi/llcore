# SPDX-License-Identifier: Apache-2.0
"""Confirm the n=4 T_residual_max gene is genuinely expansive (rho stable across seeds/samples)
and contrast against the project's own JSR/region-ceiling sound oracle expectation."""
from __future__ import annotations

import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from coupled_nd import (  # noqa: E402
    CoupledNDGene, step, cert_inf, cert_two, cert_sdp, empirical_rho,
)


def _transient_and_decay(g, n, rng, n_dir=10, s0_norm=0.25, T=40):
    max_tr = 0.0
    final_ratios = []
    x0 = np.zeros(n)
    for _ in range(n_dir):
        d = rng.standard_normal(n)
        s0 = s0_norm * d / (np.linalg.norm(d) + 1e-12)
        s = s0.copy()
        norm0 = np.linalg.norm(s0) + 1e-12
        peak = norm0
        for _t in range(T):
            s = step(g, s, x0)
            peak = max(peak, np.linalg.norm(s))
        max_tr = max(max_tr, peak / norm0)
        final_ratios.append(np.linalg.norm(s) / norm0)
    return max_tr, float(np.median(final_ratios))


def reproduce_argmax_n4(scan=1600, seed=99):
    n = 4
    rng = np.random.default_rng(seed + n)
    T_res = 0.0
    argmax_res = None
    scanned = 0
    while scanned < scan:
        scanned += 1
        decay = rng.uniform(0, 1, n)
        W = rng.uniform(-2, 2, (n, n))
        g = CoupledNDGene.make(decay=decay, W=W)
        tr, decay_ratio = _transient_and_decay(g, n, rng)
        if decay_ratio >= 0.6:
            continue
        quad = cert_inf(g) or cert_two(g) or cert_sdp(g)
        if quad:
            continue
        if tr > T_res:
            T_res = tr
            argmax_res = (g, tr, decay_ratio)
    return argmax_res


def main():
    g, tr, dr = reproduce_argmax_n4()
    print(f"argmax gene: transient={tr:.3f} decay_ratio={dr:.3f} "
          f"(JSON residual_example: transient=3.569 decay_ratio=0.0)", flush=True)
    print(f"quad-certified? inf={cert_inf(g)} two={cert_two(g)} sdp={cert_sdp(g)} "
          f"(should be all False = residual)", flush=True)
    print("empirical_rho across seeds/sample-sizes (from-below estimate, higher = more confident expansive):",
          flush=True)
    for ns in (2000, 8000, 20000):
        vals = [empirical_rho(g, n_samples=ns, seed=sd) for sd in range(3)]
        print(f"  n_samples={ns:>6}: rho = {[round(v,3) for v in vals]}  "
              f"all>=1? {all(v>=1.0 for v in vals)}", flush=True)


if __name__ == "__main__":
    main()
