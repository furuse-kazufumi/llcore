# SPDX-License-Identifier: Apache-2.0
"""L2-lite PoC: a vertex-free, provably-CONSERVATIVE sound 2-norm contraction certifier.

Goal (research/verifier_cost_reduction/SKETCH.md, minimal first rung): replace cert_two's 2^n
t-box vertex enumeration with ONE interval-matrix spectral-norm bound (2 SVDs, O(n^3)), and measure
(1) soundness consistency, (2) cost vs the 2^n method, (3) tightness vs exact two_norm / inf.

Bound (sound by construction). J(t) = diag(decay) + diag((1-decay)⊙t)·W is AFFINE in t over the box
t ∈ [t_lo, 1]^n (t_lo = t_min_per_coord). Let M = J(t_mid) (midpoint) and R = entrywise half-width
matrix (R_ij = (1-decay_i)·((1-t_lo_i)/2)·|W_ij|, the constant decay diagonal contributes 0). For any
J in the box, |J - M| ≤ R entrywise, and spectral norm is monotone under nonnegative entrywise
domination (σ_max(B) ≤ σ_max(|B|) ≤ σ_max(R)), so

    σ_max(J(t)) ≤ σ_max(M) + σ_max(R)   for all t in the box.

Admit iff σ_max(M)+σ_max(R) < 1. This is an UPPER bound on the true sup, hence:
  L2lite-admit  ⊆  cert_two-admit  ⊆  {genes contracting over the box}.
So L2-lite can NEVER admit a gene cert_two rejects (check #1 must hold by construction; a violation
is a bug). The open questions are cost (vs 2^n) and tightness (does it collapse toward inf?).

Additive research only: reuses ../verified_evolution_sdp_gate/coupled_nd.py; src/ untouched.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SDP_GATE = os.path.normpath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate"))
if _SDP_GATE not in sys.path:
    sys.path.insert(0, _SDP_GATE)

from coupled_nd import (  # noqa: E402
    CoupledNDGene,
    _jac_at_t,
    cert_inf,
    cert_two,
    t_min_per_coord,
)

MAX_INPUT_ABS = 1.0


def sample_gene(rng, n: int) -> CoupledNDGene:
    """Region-populating sampler matching exp_landscape.sample_gene (generalized to any n).

    Uniform-box sampling (codec.random) puts ~100% of genes in non_certified (W up to ±2 ⇒ huge
    spectral radius), making any tightness comparison vacuous. This sampler — decay biased high +
    small Gaussian W scaled by 1/sqrt(n) — populates all four certifier regions, exactly as the
    landscape experiments do, so L2-lite's admit set can be compared against a real contracting pool.
    """
    decay = rng.uniform(rng.uniform(0.0, 0.5), 1.0, size=n)
    w_scale = rng.choice([0.15, 0.3, 0.6, 1.0, 1.5]) / np.sqrt(n)
    W = np.clip(rng.standard_normal((n, n)) * w_scale, -2.0, 2.0)
    return CoupledNDGene.make(decay=decay, W=W)


def l2lite_bound(g: CoupledNDGene, max_input_abs: float = MAX_INPUT_ABS) -> float:
    """Sound upper bound on sup_{t in box} sigma_max(J(t)) via interval-matrix midpoint+radius."""
    gc = g.clipped()
    t_lo = t_min_per_coord(gc, max_input_abs)          # t in [t_lo, 1]
    t_mid = 0.5 * (t_lo + 1.0)
    half = 0.5 * (1.0 - t_lo)                          # per-coord half-width of t_i
    M = _jac_at_t(gc, t_mid)                           # midpoint Jacobian (affine -> exact midpoint)
    coeff = (1.0 - gc.decay) * half                    # row scaling of the variable part
    R = coeff[:, None] * np.abs(gc.W)                  # entrywise half-width (decay diag is constant -> 0)
    sM = float(np.linalg.svd(M, compute_uv=False)[0])
    sR = float(np.linalg.svd(R, compute_uv=False)[0])
    return sM + sR


def cert_l2lite(g: CoupledNDGene, max_input_abs: float = MAX_INPUT_ABS) -> bool:
    return bool(l2lite_bound(g, max_input_abs) < 1.0)


def run_soundness_tightness(n: int, n_genes: int, seed: int) -> dict:
    """n=8 feasible exact two_norm: compare L2-lite vs exact cert_two and cert_inf on a gene pool."""
    rng = np.random.default_rng(seed)
    n_two = n_inf = n_l2 = 0
    l2_admit_two_reject = 0      # MUST be 0 (soundness consistency; violation = bug)
    l2_admit_inf_reject = 0      # L2-lite genes that inf rejects (= what L2-lite gains over inf)
    inf_admit_l2_reject = 0      # inf genes L2-lite rejects (should be ~0 if L2-lite >= inf coverage)
    for _ in range(n_genes):
        g = sample_gene(rng, n)
        is_inf = cert_inf(g, MAX_INPUT_ABS)
        is_two = cert_two(g, MAX_INPUT_ABS)
        is_l2 = cert_l2lite(g, MAX_INPUT_ABS)
        n_inf += is_inf
        n_two += is_two
        n_l2 += is_l2
        if is_l2 and not is_two:
            l2_admit_two_reject += 1
        if is_l2 and not is_inf:
            l2_admit_inf_reject += 1
        if is_inf and not is_l2:
            inf_admit_l2_reject += 1
    return {
        "n": n, "n_genes": n_genes, "seed": seed,
        "admit_counts": {"inf": n_inf, "l2lite": n_l2, "two_exact": n_two},
        "soundness_violations_l2_admit_two_reject": l2_admit_two_reject,  # expect 0
        "l2_captures_of_exact_two_pct": round(100.0 * n_l2 / n_two, 2) if n_two else None,
        "l2_gain_over_inf_genes": l2_admit_inf_reject,                    # how much L2-lite beats inf
        "inf_admit_l2_reject": inf_admit_l2_reject,                       # L2-lite < inf coverage?
    }


def run_cost(ns_genes: dict, seed: int) -> dict:
    """Wall-clock per gene: exact cert_two (2^n SVDs) vs L2-lite (2 SVDs)."""
    out = {}
    rng = np.random.default_rng(seed + 777)
    for n, ng in ns_genes.items():
        genes = [sample_gene(rng, n) for _ in range(ng)]
        t0 = time.perf_counter()
        for g in genes:
            cert_two(g, MAX_INPUT_ABS)
        t_two = (time.perf_counter() - t0) / ng
        t0 = time.perf_counter()
        for g in genes:
            cert_l2lite(g, MAX_INPUT_ABS)
        t_l2 = (time.perf_counter() - t0) / ng
        out[str(n)] = {
            "n_genes": ng, "vertices_2pow_n": 2 ** n,
            "sec_per_gene_two_exact": round(t_two, 6),
            "sec_per_gene_l2lite": round(t_l2, 6),
            "speedup_x": round(t_two / t_l2, 1) if t_l2 > 0 else None,
        }
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # cp932 console safety
    st = run_soundness_tightness(n=8, n_genes=3000, seed=20260606)
    print("--- soundness + tightness (n=8, 3000 genes) ---")
    print(json.dumps(st, ensure_ascii=False, indent=2))
    cost = run_cost({8: 300, 12: 60, 16: 8}, seed=20260606)
    print("--- cost (sec/gene; exact 2^n vs L2-lite 2 SVD) ---")
    print(json.dumps(cost, ensure_ascii=False, indent=2))
    result = {"soundness_tightness": st, "cost": cost, "max_input_abs": MAX_INPUT_ABS}
    with open(os.path.join(_HERE, "poc_l2lite_results.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print("wrote poc_l2lite_results.json")


if __name__ == "__main__":
    main()
