# SPDX-License-Identifier: Apache-2.0
"""EXP-C — does the CAPABILITY potential of higher-degree verifiers grow with substrate dimension?

EXP-B finds the deg4/deg6 capability payoff is ~null at n=2 because the residual's achievable
transient amplification is weak (SDP-certified genes approximate residual targets). Mechanism
hypothesis (DEG6_PREREGISTRATION G-C): non-normal transient amplification grows with dimension,
so the *potential* gap a higher-degree verifier could unlock should grow with n.

Measure, per n∈{2,3,4} on coupled_nd: the MAX behavioural transient amplification achievable by
  * residual genes  (quad-REJECTED: not inf/2norm/sdp) that behaviourally contract, vs
  * quad-certified genes (inf/2norm/sdp) that behaviourally contract.
gap(n) = T_residual(n) − T_quad(n). If gap(n) is monotone increasing ⇒ the deg-verifier capability
payoff is dimension-gated (returns in high-dim / full-LLM regimes — motivates the GPU bet).

Behavioural contraction proxy (cheap, trajectory-based, no certifier): the autonomous free
response decays (final ‖s_T‖/‖s_0‖ < 0.6) across random start directions. Transient = max over
directions & steps of ‖s_t‖/‖s_0‖ in the near-linear regime (‖s_0‖=0.25).
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coupled_nd import (  # noqa: E402
    CoupledNDGene, step, cert_inf, cert_two, cert_sdp,
)


def _transient_and_decay(g, n, rng, n_dir=10, s0_norm=0.25, T=40):
    """Max transient ‖s_t‖/‖s_0‖ and median final-decay ratio over random start directions."""
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


def run(dims=(2, 3, 4), scan=1600, time_cap_per_n=200.0, seed=99) -> dict:
    results = {}
    for n in dims:
        t0 = time.time()
        rng = np.random.default_rng(seed + n)
        T_res = 0.0   # max transient among residual (quad-rejected) contracting genes
        T_quad = 0.0  # max transient among quad-certified contracting genes
        n_res = n_quad = 0
        res_example = None
        scanned = 0
        while scanned < scan and time.time() - t0 < time_cap_per_n:
            scanned += 1
            decay = rng.uniform(0, 1, n)
            W = rng.uniform(-2, 2, (n, n))
            g = CoupledNDGene.make(decay=decay, W=W)
            tr, decay_ratio = _transient_and_decay(g, n, rng)
            if decay_ratio >= 0.6:        # not behaviourally contracting -> skip
                continue
            quad = cert_inf(g) or cert_two(g) or cert_sdp(g)
            if quad:
                n_quad += 1
                if tr > T_quad:
                    T_quad = tr
            else:
                n_res += 1
                if tr > T_res:
                    T_res = tr
                    res_example = {"transient": round(tr, 3),
                                   "decay_ratio": round(decay_ratio, 3)}
        results[n] = {
            "dim": n + n * n, "scanned": scanned,
            "n_residual_contracting": n_res, "n_quad_contracting": n_quad,
            "T_residual_max": round(T_res, 3), "T_quad_max": round(T_quad, 3),
            "gap": round(T_res - T_quad, 3),
            "residual_example": res_example, "elapsed_s": round(time.time() - t0, 1),
        }
        print(f"n={n}: T_residual={T_res:.3f} T_quad={T_quad:.3f} gap={T_res-T_quad:+.3f} "
              f"(res={n_res} quad={n_quad}, {results[n]['elapsed_s']}s)", flush=True)

    gaps = [results[n]["gap"] for n in dims]
    t_res = [results[n]["T_residual_max"] for n in dims]
    monotone_gap = all(gaps[i + 1] >= gaps[i] for i in range(len(gaps) - 1))
    monotone_tres = all(t_res[i + 1] >= t_res[i] for i in range(len(t_res) - 1))
    out = {
        "dims": list(dims), "per_n": results,
        "T_residual_by_n": t_res, "gap_by_n": gaps,
        "G_C_gap_monotone_increasing": monotone_gap,
        "T_residual_monotone_increasing": monotone_tres,
    }
    with open(os.path.join(_HERE, "exp_deg6_dimension_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "per_n"}, indent=2), flush=True)
    return out


if __name__ == "__main__":
    run()
