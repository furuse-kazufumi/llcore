# SPDX-License-Identifier: Apache-2.0
"""PoC-2.6: does the cheap vertex-free coverage hold as the state dimension n grows?

The cost-reduction thread concluded "scale n with inf ∪ B2, skip the SDP" — but coverage (B2 = 77.6%,
inf∪B2 = 87.2% of exact cert_two) was measured only at n=8. This PoC tests whether that coverage holds
(or degrades) at n=12 and n=16, where exact cert_two enumerates 2^12=4096 / 2^16=65536 t-box vertices
(still feasible for a modest pool, unlike n=32). Soundness must stay 0 violations at every n (B2 is a
provable upper bound, so admit ⊆ cert_two by construction; a nonzero count = bug).

If coverage holds, the "scale with inf∪B2" claim is backed beyond n=8. If it degrades, that is an honest
limit to disclose. Reuses poc_l2lite.sample_gene + poc_l2lite_v2.bound_b2; src/ untouched.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SDP_GATE = os.path.normpath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate"))
if _SDP_GATE not in sys.path:
    sys.path.insert(0, _SDP_GATE)

from coupled_nd import cert_inf, cert_two  # noqa: E402
from poc_l2lite import MAX_INPUT_ABS, sample_gene  # noqa: E402
from poc_l2lite_v2 import bound_b2  # noqa: E402


def run_n(n: int, n_genes: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cnt = {"inf": 0, "two_exact": 0, "b2": 0, "inf_or_b2": 0}
    viol = 0  # B2 admit but cert_two reject -> MUST be 0
    t0 = time.perf_counter()
    for _ in range(n_genes):
        g = sample_gene(rng, n)
        is_inf = cert_inf(g, MAX_INPUT_ABS)
        is_two = cert_two(g, MAX_INPUT_ABS)
        is_b2 = bound_b2(g) < 1.0
        cnt["inf"] += is_inf
        cnt["two_exact"] += is_two
        cnt["b2"] += is_b2
        cnt["inf_or_b2"] += (is_inf or is_b2)
        if is_b2 and not is_two:
            viol += 1
    two = cnt["two_exact"]
    pct = {k: (round(100.0 * cnt[k] / two, 2) if two else None) for k in ("inf", "b2", "inf_or_b2")}
    return {"n": n, "n_genes": n_genes, "vertices_2pow_n": 2 ** n,
            "admit_counts": cnt, "pct_of_exact_two": pct,
            "b2_soundness_violations": viol,
            "sec": round(time.perf_counter() - t0, 1)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    # pool sizes shrink as cert_two cost grows (2^n vertices): keep wall-clock bounded.
    plan = [(8, 3000), (12, 1200), (16, 200)]
    results = []
    for n, ng in plan:
        r = run_n(n, ng, seed=20260606 + n)
        print(json.dumps(r, ensure_ascii=False))
        results.append(r)
    out = {"plan": plan, "results": results}
    with open(os.path.join(_HERE, "poc_scale_results.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)
    print("wrote poc_scale_results.json")
    # honest one-line read
    b2 = [r["pct_of_exact_two"]["b2"] for r in results]
    u = [r["pct_of_exact_two"]["inf_or_b2"] for r in results]
    v = sum(r["b2_soundness_violations"] for r in results)
    print(f"B2 coverage by n: {b2}; inf∪B2: {u}; total soundness violations: {v}")


if __name__ == "__main__":
    main()
