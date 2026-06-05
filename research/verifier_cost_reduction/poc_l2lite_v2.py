# SPDX-License-Identifier: Apache-2.0
"""PoC-2: do tighter *cheap* vertex-free sound 2-norm bounds beat cert_inf's coverage?

PoC-1 found the naive interval bound B1 = σ(M)+σ(R) is sound but MORE conservative than cert_inf
(captures 29.5% of exact two_norm; rejects 700 of inf's 1072). PoC-2 tests two more O(n^3) sound
bounds and their union, to decide whether *any* cheap (non-SDP) vertex-free certificate can beat inf
— or whether the genuine robust-LMI (SDP, R-LLM-1) is required.

All bounds are sound UPPER bounds on  sup_{t in [t_lo,1]^n} σ_max(J(t)),  J(t)=D+diag(c⊙t)W:
  B1 = σ(M) + σ(R)            (triangle split; 2 SVDs)          [PoC-1]
  B2 = σ(|M| + R)             (entrywise-abs domination; 1 SVD)
       sound because |J| ≤ |M|+R entrywise and σ_max is monotone under nonneg entrywise domination.
  Bmin = min(B1, B2)          (min of two valid upper bounds is a valid upper bound; admit = union)
where M=J(t_mid), R_ij=(1-decay_i)·((1-t_lo_i)/2)·|W_ij|  (the t-box midpoint+radius interval matrix).

admit_X = (B_X < 1). By construction every admit set ⊆ {genes contracting over the box} = cert_two
admit set, so 0 soundness violations is expected (a violation = bug). The open question is COVERAGE
vs cert_inf and vs exact cert_two. Reuses poc_l2lite.sample_gene; src/ untouched.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SDP_GATE = os.path.normpath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate"))
if _SDP_GATE not in sys.path:
    sys.path.insert(0, _SDP_GATE)

from coupled_nd import CoupledNDGene, _jac_at_t, cert_inf, cert_two, t_min_per_coord  # noqa: E402
from poc_l2lite import MAX_INPUT_ABS, sample_gene  # noqa: E402


def _MR(g: CoupledNDGene, max_input_abs: float = MAX_INPUT_ABS):
    gc = g.clipped()
    t_lo = t_min_per_coord(gc, max_input_abs)
    M = _jac_at_t(gc, 0.5 * (t_lo + 1.0))
    R = ((1.0 - gc.decay) * (0.5 * (1.0 - t_lo)))[:, None] * np.abs(gc.W)
    return M, R


def bound_b1(g) -> float:
    M, R = _MR(g)
    return float(np.linalg.svd(M, compute_uv=False)[0] + np.linalg.svd(R, compute_uv=False)[0])


def bound_b2(g) -> float:
    M, R = _MR(g)
    return float(np.linalg.svd(np.abs(M) + R, compute_uv=False)[0])


def run(n: int, n_genes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cnt = {"inf": 0, "two_exact": 0, "b1": 0, "b2": 0, "bmin": 0}
    viol = {"b1": 0, "b2": 0, "bmin": 0}          # admit but cert_two rejects -> MUST be 0
    gain_over_inf = {"b1": 0, "b2": 0, "bmin": 0}  # admit but inf rejects
    loss_vs_inf = {"b1": 0, "b2": 0, "bmin": 0}    # inf admits but bound rejects
    for _ in range(n_genes):
        g = sample_gene(rng, n)
        is_inf = cert_inf(g, MAX_INPUT_ABS)
        is_two = cert_two(g, MAX_INPUT_ABS)
        b1 = bound_b1(g) < 1.0
        b2 = bound_b2(g) < 1.0
        bmin = b1 or b2
        cnt["inf"] += is_inf; cnt["two_exact"] += is_two
        cnt["b1"] += b1; cnt["b2"] += b2; cnt["bmin"] += bmin
        for key, ok in (("b1", b1), ("b2", b2), ("bmin", bmin)):
            if ok and not is_two:
                viol[key] += 1
            if ok and not is_inf:
                gain_over_inf[key] += 1
            if is_inf and not ok:
                loss_vs_inf[key] += 1
    pct = {k: (round(100.0 * cnt[k] / cnt["two_exact"], 2) if cnt["two_exact"] else None)
           for k in ("inf", "b1", "b2", "bmin")}
    return {"n": n, "n_genes": n_genes, "seed": seed, "admit_counts": cnt,
            "pct_of_exact_two": pct, "soundness_violations": viol,
            "gain_over_inf": gain_over_inf, "loss_vs_inf": loss_vs_inf,
            "beats_inf_coverage": {k: cnt[k] > cnt["inf"] for k in ("b1", "b2", "bmin")}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    res = run(n=8, n_genes=3000, seed=20260606)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    with open(os.path.join(_HERE, "poc_l2lite_v2_results.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2)
    print("wrote poc_l2lite_v2_results.json")


if __name__ == "__main__":
    main()
