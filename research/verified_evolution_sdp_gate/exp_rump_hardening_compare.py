# SPDX-License-Identifier: Apache-2.0
"""PART 2 comparison — float-recheck SDP admit count vs Rump+OR admit count on the EXP-A pool.

Regenerates the SAME 300-contracting n=2 gene pool ``exp_deg6_ladder.py`` uses (seed=2024,
empirical-contraction filter ρ<1 via ``empirical_spectral_radius``), then on that fixed pool counts:

  * FLOAT admit:  the EXISTING default SDP certifier ``coupled_components._sdp_certifies`` — which
    accepts on the inf/2-norm fast path OR a common-Lyapunov ``P`` accepted by the certifier's
    internal FLOAT ``eigvalsh`` recheck (no-margin on P and P - J^T P J).
  * RUMP+OR admit: the NEW additive factory ``rump_hardened_verifier.make_sdp_verifier_rump_or`` —
    same fast paths, but the SDP branch accepts iff ANY of {CLARABEL,SCS} returns a P passing the
    Rump verified-PD recheck on the SAME no-margin matrices the float path tests.

Reports BOTH counts + delta. By construction (verified_pd >= float-eigvalsh on identical matrices,
already proven) RUMP+OR >= FLOAT (admit set preserved-or-grown, never shrunk). Any EXTRA Rump+OR
admit (in Rump+OR but not FLOAT) is then checked with the solver-INDEPENDENT JSR lower-bound oracle
(the ``verify_certifier_jsr_soundness.py`` pattern): a sound extra admit MUST have jsr_lb < 1.

Writes ``rump_hardening_compare_results.json``. research/ only; src/ untouched; zero new deps.
"""
from __future__ import annotations

import itertools
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

from coupled_map import CoupledGene  # noqa: E402
from coupled_components import _sdp_certifies, empirical_spectral_radius  # noqa: E402
from rump_hardened_verifier import make_sdp_verifier_rump_or  # noqa: E402
from rump_pd import _vertex_jacobians  # noqa: E402

JSR_TOL = 1e-9


def jsr_lb(vertices, max_len: int = 6) -> float:
    """Gripenberg lower bound on JSR{vertices} — the solver-independent product oracle used by
    ``verify_certifier_jsr_soundness.py``. A sound certificate implies jsr_lb < 1; jsr_lb >= 1 on a
    certified gene would expose an unsound admit (a product of vertex Jacobians that expands)."""
    V = [np.asarray(M, dtype=np.float64) for M in vertices]
    best = 0.0
    for k in range(1, max_len + 1):
        for combo in itertools.product(range(len(V)), repeat=k):
            P = np.eye(V[0].shape[0])
            for i in combo:
                P = V[i] @ P
            best = max(best, float(np.max(np.abs(np.linalg.eigvals(P)))) ** (1.0 / k))
    return best


def regenerate_expA_pool(n_target: int = 300, seed: int = 2024, time_cap: float = 600.0):
    """Regenerate the EXACT EXP-A contracting pool the way exp_deg6_ladder.py does: draw genes from
    the SAME seeded stream and keep those with empirical ρ < 1 (n_samples=4000) until n_target."""
    rng = np.random.default_rng(seed)
    t0 = time.time()
    pool = []
    scanned = 0
    while len(pool) < n_target and time.time() - t0 < time_cap:
        scanned += 1
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if empirical_spectral_radius(g, n_samples=4000) >= 1.0:
            continue
        pool.append(g)
    return pool, scanned


def main(n_target: int = 300, seed: int = 2024, jsr_max_len: int = 6,
         sound_samples: int = 20000):
    t0 = time.time()
    print(f"regenerating EXP-A pool (n_target={n_target}, seed={seed}) ...", flush=True)
    pool, scanned = regenerate_expA_pool(n_target=n_target, seed=seed)
    print(f"pool: {len(pool)} contracting genes (scanned {scanned})", flush=True)

    rump_or = make_sdp_verifier_rump_or()

    float_admit = 0
    rump_admit = 0
    extra_records = []   # in Rump+OR but NOT float  -> must be sound (jsr_lb < 1)
    lost_records = []    # in float but NOT Rump+OR  -> should be EMPTY (admit set never shrinks)
    for i, g in enumerate(pool):
        f = bool(_sdp_certifies(g))
        r = bool(rump_or.certifies(g))
        float_admit += int(f)
        rump_admit += int(r)
        if r and not f:
            verts = _vertex_jacobians(g, t_domain="tmin1")
            j = jsr_lb(verts, max_len=jsr_max_len)
            rho = empirical_spectral_radius(g, n_samples=sound_samples)
            extra_records.append({"index": i, "decay": g.clipped().decay.tolist(),
                                  "W": g.clipped().W.reshape(-1).tolist(),
                                  "jsr_lb": j, "rho_sound": rho})
        if f and not r:
            verts = _vertex_jacobians(g, t_domain="tmin1")
            j = jsr_lb(verts, max_len=jsr_max_len)
            lost_records.append({"index": i, "decay": g.clipped().decay.tolist(),
                                 "W": g.clipped().W.reshape(-1).tolist(), "jsr_lb": j})

    delta = rump_admit - float_admit
    n_extra = len(extra_records)
    n_lost = len(lost_records)
    extra_unsound = [e for e in extra_records if e["jsr_lb"] >= 1.0 - JSR_TOL]
    extra_jsr_max = max((e["jsr_lb"] for e in extra_records), default=None)

    out = {
        "meta": {
            "seed": seed, "n_target": n_target, "pool_size": len(pool), "scanned": scanned,
            "jsr_max_len": jsr_max_len, "sound_samples": sound_samples,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "float_recheck_sdp_admit": float_admit,
        "rump_or_admit": rump_admit,
        "delta_rump_or_minus_float": delta,
        # invariant: Rump+OR >= float (admit set preserved-or-grown, never shrunk).
        "admit_set_preserved_or_grown": n_lost == 0,
        "n_extra_rump_or_only": n_extra,
        "n_lost_float_only": n_lost,
        # soundness of any EXTRA admits (solver-independent JSR oracle).
        "extra_admits_all_sound_jsr_lt_1": len(extra_unsound) == 0,
        "n_extra_unsound": len(extra_unsound),
        "extra_jsr_lb_max": extra_jsr_max,
        "extra_records": extra_records,
        "lost_records": lost_records,
        "unsound_extra_examples": extra_unsound,
        # decision-rule input (see RUMP_HARDENING_VERDICT.md).
        "counts_equal": delta == 0,
    }

    op = os.path.join(_HERE, "rump_hardening_compare_results.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    summary = {k: v for k, v in out.items()
               if k not in ("extra_records", "lost_records", "unsound_extra_examples")}
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nwrote {op}", flush=True)
    return out


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 300
    main(n_target=n)
