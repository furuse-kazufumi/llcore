# SPDX-License-Identifier: Apache-2.0
"""Independent verification of the peer-review confound claim against exp_deg6_dimension.

Reproduces the EXACT residual-selection from exp_deg6_dimension.run() for n=4, then
applies the project's OWN sound consistency oracle (coupled_nd.empirical_rho) to every
"residual contracting" gene. Tests:
  (C2a) fraction of residual genes with empirical_rho >= 1 (locally expansive somewhere).
  (C2b) the empirical_rho of the gene that DEFINES T_residual_max (the argmax transient).
  (C2c) tanh-saturation artifact: does shrinking ||s0|| RAISE the max-gene transient?
  (C3)  pool sizes (residual vs quad) at each n -> max-over-sample asymmetry.
Mirrors the script logic verbatim (same seed, scan, gates) so the selection is identical.
"""
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
    """Verbatim copy of exp_deg6_dimension._transient_and_decay."""
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


def _transient_only(g, n, seed_for_dirs, s0_norm, n_dir=10, T=40):
    """Max transient at a chosen ||s0|| (re-uses the same direction RNG stream)."""
    rng = np.random.default_rng(seed_for_dirs)
    max_tr = 0.0
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
    return max_tr


def reproduce_n(n, scan=1600, seed=99):
    """Verbatim selection from exp_deg6_dimension.run() for a single n. Collect residual genes."""
    rng = np.random.default_rng(seed + n)
    residual = []  # (gene, transient, decay_ratio)
    n_quad = 0
    T_res = 0.0
    T_quad = 0.0
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
            n_quad += 1
            if tr > T_quad:
                T_quad = tr
        else:
            residual.append((g, tr, decay_ratio))
            if tr > T_res:
                T_res = tr
                argmax_res = (g, tr, decay_ratio)
    return residual, n_quad, T_res, T_quad, argmax_res


def main():
    print("=== Pool sizes per n (C3 sample-size confound) ===", flush=True)
    pools = {}
    for n in (2, 3, 4):
        residual, n_quad, T_res, T_quad, argmax = reproduce_n(n)
        pools[n] = (len(residual), n_quad, T_res, T_quad, argmax)
        ratio = len(residual) / max(n_quad, 1)
        print(f"n={n}: residual_pool={len(residual)} quad_pool={n_quad} "
              f"ratio={ratio:.1f}x  T_res={T_res:.3f} T_quad={T_quad:.3f} gap={T_res-T_quad:+.3f}",
              flush=True)

    # Focus on n=4 for the soundness confound (C2).
    n = 4
    residual, n_quad, T_res, T_quad, argmax = pools[n]
    print(f"\n=== n=4 proxy-soundness (C2) — applying empirical_rho oracle to {residual} residual genes ===",
          flush=True)
    residual_list, n_quad4, T_res4, T_quad4, argmax4 = reproduce_n(4)

    rhos = []
    expansive = 0
    for (g, tr, dr) in residual_list:
        r = empirical_rho(g, n_samples=4000, seed=0)
        rhos.append(r)
        if r >= 1.0:
            expansive += 1
    rhos = np.array(rhos)
    print(f"residual genes: {len(residual_list)}", flush=True)
    print(f"empirical_rho >= 1 (locally expansive somewhere): {expansive} "
          f"({100*expansive/len(residual_list):.0f}%)", flush=True)
    print(f"empirical_rho distribution: min={rhos.min():.3f} median={np.median(rhos):.3f} "
          f"max={rhos.max():.3f}", flush=True)

    # The argmax gene that DEFINES T_residual_max
    g_max, tr_max, dr_max = argmax4
    rho_max = empirical_rho(g_max, n_samples=8000, seed=0)
    print(f"\n=== The T_residual_max gene (defines the headline gap) ===", flush=True)
    print(f"  transient={tr_max:.3f} decay_ratio={dr_max:.3f} empirical_rho={rho_max:.3f}", flush=True)
    print(f"  EXPANSIVE? {rho_max >= 1.0}", flush=True)

    # C2c tanh-saturation artifact: shrink ||s0|| toward linear regime
    print(f"\n=== tanh-saturation artifact check (C2c) — argmax gene, shrink ||s0|| ===", flush=True)
    seed_dirs = np.random.default_rng(0)  # fixed direction stream for fair comparison
    for s0n in (0.25, 0.05, 0.01):
        tr = _transient_only(g_max, n, 12345, s0n)
        print(f"  ||s0||={s0n:.2f}: max_transient={tr:.3f}", flush=True)


if __name__ == "__main__":
    main()
