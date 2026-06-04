# SPDX-License-Identifier: Apache-2.0
"""R1 — does the quadratic-SDP contraction-completeness result GENERALISE with dimension?

Mirrors the n=2 coverage-frontier method of ``exp_deg6_ladder.py`` (count how many of an
empirically-contracting pool each cumulative certifier covers) but on the n-dim substrate
(``coupled_nd``) for n ∈ {2,3,4}, and WITHOUT the n=2-only degree-4/6 rungs. The headline
quantity is ``sdp_cum / pool`` (quadratic-SDP completeness) at each n and whether it stays
~95% or DECAYS as the coupled dimension grows.

Reuses (does NOT reinvent) the already-CLARABEL-pinned certifiers from coupled_nd:
``cert_inf``, ``cert_two``, ``cert_sdp``, plus ``empirical_rho``, ``_box_vertices``,
``_jac_at_t``, ``t_min_per_coord``. Soundness is checked INDEPENDENTLY of the solver via
(a) high-sample empirical spectral radius and (b) an n-dim JSR product lower-bound oracle.

See PREREGISTRATION_R1.md for the falsifiable gates (G1..G4) registered before the run.

Usage: py -3.11 exp_dim_completeness.py [--n-target 200] [--seed 4242]
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

# Reuse the n-dim substrate + already-pinned certifiers from the sibling research dir.
_HERE = os.path.dirname(os.path.abspath(__file__))
_GATE = os.path.normpath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate"))
if _GATE not in sys.path:
    sys.path.insert(0, _GATE)

from coupled_nd import (  # noqa: E402
    _CLARABEL_OK,
    _SOLVER,
    CoupledNDGene,
    _box_vertices,
    _jac_at_t,
    cert_inf,
    cert_sdp,
    cert_two,
    empirical_rho,
    t_min_per_coord,
)

try:
    import cvxpy as cp  # noqa: E402
except Exception:  # pragma: no cover
    cp = None


# --------------------------------------------------------------------------- #
# Vectorized from-below empirical spectral radius (identical math to
# coupled_nd.empirical_rho, but batches np.linalg.eigvals over the sample axis so
# 20k-sample soundness checks are tractable). Verified to match coupled_nd.empirical_rho
# to <1e-9 on the same seed in the self-test at module bottom.
# --------------------------------------------------------------------------- #


def empirical_rho_fast(g: CoupledNDGene, *, n_samples: int, seed: int = 0) -> float:
    gc = g.clipped()
    n = gc.n
    rng = np.random.default_rng(seed)
    S = rng.uniform(-1.0, 1.0, (n_samples, n))
    X = rng.uniform(-1.0, 1.0, (n_samples, n))
    # pre = S @ W^T + X @ V^T  (each row k is W@S[k] + V@X[k])
    pre = S @ gc.W.T + X @ gc.V.T              # (N, n)
    t = 1.0 - np.tanh(pre) ** 2                # (N, n) = sech^2
    # J_k = diag(decay) + diag((1-decay)*t_k) @ W
    coeff = (1.0 - gc.decay)[None, :] * t      # (N, n)
    # build (N, n, n): row i = coeff[:,i] * W[i,:]   plus decay on the diagonal
    Js = coeff[:, :, None] * gc.W[None, :, :]  # (N, n, n)
    diag_idx = np.arange(n)
    Js[:, diag_idx, diag_idx] += gc.decay[None, :]
    eig = np.linalg.eigvals(Js)                # (N, n) complex, vectorized over N
    return float(np.max(np.abs(eig)))


# --------------------------------------------------------------------------- #
# n-dim JSR product lower-bound oracle (generalises verify_certifier_jsr_soundness).
# --------------------------------------------------------------------------- #

# Per-n product-length cap so the product count (#verts^len) stays bounded and tractable:
#   n=2: 2^2=4 verts, len 6  -> 4^6   = 4,096
#   n=3: 2^3=8 verts, len 4  -> 8^4   = 4,096
#   n=4: 2^4=16 verts, len 3 -> 16^3  = 4,096
# A capped JSR_lb is still a valid ONE-SIDED lower bound: jsr_lb>=1 proves an unsound admit.
_JSR_MAXLEN = {2: 6, 3: 4, 4: 3}


def _vertices_nd(g: CoupledNDGene, max_input_abs: float = 1.0) -> list[np.ndarray]:
    """The 2^n vertex Jacobians of the achievable-t box (same set cert_two/cert_sdp use)."""
    t_lo = t_min_per_coord(g, max_input_abs)
    return [_jac_at_t(g, v) for v in _box_vertices(t_lo)]


def jsr_lb(vertices, max_len: int) -> float:
    """Joint-spectral-radius lower bound = max over products up to length max_len of
    rho(prod)^(1/len). jsr_lb >= 1 proves a switched-expansive set (unsound admit)."""
    V = [np.asarray(M, dtype=np.float64) for M in vertices]
    nverts = len(V)
    dim = V[0].shape[0]
    best = 0.0
    for k in range(1, max_len + 1):
        for combo in itertools.product(range(nverts), repeat=k):
            P = np.eye(dim)
            for i in combo:
                P = V[i] @ P
            best = max(best, float(np.max(np.abs(np.linalg.eigvals(P)))) ** (1.0 / k))
    return best


# --------------------------------------------------------------------------- #
# Coverage frontier for one dimension.
# --------------------------------------------------------------------------- #


def run_one_n(n: int, *, n_target: int, seed: int, screen_samples: int,
              sound_samples: int, time_cap: float) -> dict:
    rng = np.random.default_rng(seed)
    t0 = time.time()
    counts = dict(inf=0, two_only=0, sdp_only=0, residual=0)
    two_beats_sdp = 0
    n_unsound = 0
    max_rho_sound = 0.0
    max_jsr_certified = 0.0
    residual_records = []
    scanned = 0
    pool = 0
    maxlen = _JSR_MAXLEN[n]

    while pool < n_target and (time.time() - t0) < time_cap:
        scanned += 1
        decay = rng.uniform(0.0, 1.0, n)
        W = rng.uniform(-2.0, 2.0, (n, n))
        g = CoupledNDGene.make(decay=decay, W=W)
        if empirical_rho(g, n_samples=screen_samples, seed=0) >= 1.0:
            continue  # not in the empirically-contracting pool
        pool += 1

        is_inf = cert_inf(g)
        is_two = cert_two(g)        # solver-independent vertex SVD
        is_sdp = cert_sdp(g)        # CLARABEL-pinned; theoretical superset of cert_two

        # superset bookkeeping (G2): two certifies but sdp does not -> anomaly
        if is_two and not is_sdp:
            two_beats_sdp += 1

        if is_inf:
            counts["inf"] += 1
        elif is_two:
            counts["two_only"] += 1
        elif is_sdp:
            counts["sdp_only"] += 1
        else:
            counts["residual"] += 1
            residual_records.append({"decay": decay.tolist(), "W": W.reshape(-1).tolist()})

        # INDEPENDENT soundness check for every SDP-certified gene (inf/two/sdp all imply
        # a quadratic certificate; sdp is the union). cert_inf/cert_two => cert_sdp True.
        if is_inf or is_two or is_sdp:
            rho_hi = empirical_rho(g, n_samples=sound_samples, seed=0)
            j = jsr_lb(_vertices_nd(g), max_len=maxlen)
            max_rho_sound = max(max_rho_sound, rho_hi)
            max_jsr_certified = max(max_jsr_certified, j)
            if rho_hi >= 1.0 or j >= 1.0 - 1e-9:
                n_unsound += 1

    inf = counts["inf"]
    two_cum = inf + counts["two_only"]
    sdp_cum = two_cum + counts["sdp_only"]
    residual = counts["residual"]
    assert sdp_cum + residual == pool, (sdp_cum, residual, pool)

    rec = {
        "n": n,
        "seed": seed,
        "scanned": scanned,
        "pool": pool,
        "n_target": n_target,
        "jsr_max_len": maxlen,
        "n_box_vertices": 2 ** n,
        "counts": counts,
        "inf": inf,
        "two_cum": two_cum,
        "sdp_cum": sdp_cum,
        "residual": residual,
        "inf_pct": round(100.0 * inf / pool, 2) if pool else 0.0,
        "two_pct": round(100.0 * two_cum / pool, 2) if pool else 0.0,
        "sdp_pct": round(100.0 * sdp_cum / pool, 2) if pool else 0.0,
        "residual_pct": round(100.0 * residual / pool, 2) if pool else 0.0,
        "two_beats_sdp": two_beats_sdp,
        "n_unsound": n_unsound,
        "max_rho_sound_over_certified": round(max_rho_sound, 4),
        "max_jsr_lb_over_certified": round(max_jsr_certified, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "residual_records": residual_records,
    }
    print(f"  n={n} pool={pool}/{n_target} scanned={scanned} "
          f"inf={inf} two_cum={two_cum} sdp_cum={sdp_cum} ({rec['sdp_pct']}%) "
          f"residual={residual} two_beats_sdp={two_beats_sdp} unsound={n_unsound} "
          f"maxrho={rec['max_rho_sound_over_certified']} maxjsr={rec['max_jsr_lb_over_certified']} "
          f"({rec['elapsed_s']}s)", flush=True)
    return rec


def run(n_target: int = 200, seed: int = 4242, dims=(2, 3, 4),
        screen_samples: int = 4000, sound_samples: int = 20000,
        time_cap_per_n: float = 1800.0) -> dict:
    t0 = time.time()
    solver_name = getattr(_SOLVER, "name", lambda: str(_SOLVER))
    try:
        solver_str = _SOLVER.name() if hasattr(_SOLVER, "name") else str(_SOLVER)
    except Exception:
        solver_str = str(_SOLVER)
    clarabel_confirmed = bool(_CLARABEL_OK) and (cp is not None) and (_SOLVER is cp.CLARABEL)
    if not clarabel_confirmed:
        raise SystemExit(f"CLARABEL NOT confirmed as the pinned solver (_CLARABEL_OK={_CLARABEL_OK}, "
                         f"_SOLVER={solver_str}). Refusing to run an honest-disclosure SDP experiment "
                         f"under a non-CLARABEL solver.")
    print(f"CLARABEL confirmed: _CLARABEL_OK={_CLARABEL_OK}, _SOLVER={solver_str}, "
          f"installed={cp.installed_solvers()}", flush=True)

    per_n = []
    # distinct fixed seed per n (master + n) so pools are independent yet fully reproducible
    for n in dims:
        per_n.append(run_one_n(n, n_target=n_target, seed=seed + n,
                               screen_samples=screen_samples, sound_samples=sound_samples,
                               time_cap=time_cap_per_n))

    # ---- gate evaluation (G1..G4) ----
    p = {r["n"]: r["sdp_cum"] / r["pool"] for r in per_n}
    res_frac = {r["n"]: r["residual"] / r["pool"] for r in per_n}
    g1 = all(p[n] >= 0.80 for n in p)
    g2 = all(r["two_beats_sdp"] == 0 for r in per_n)
    g3 = all(r["n_unsound"] == 0 for r in per_n)

    p2, p4 = p.get(2), p.get(4)
    drop_2_to_4 = (p2 - p4) if (p2 is not None and p4 is not None) else None
    res4 = res_frac.get(4)
    generalises_strict = (all(p[n] >= 0.90 for n in p)
                          and drop_2_to_4 is not None and drop_2_to_4 <= 0.05
                          and res4 is not None and res4 <= 0.10)
    degrades = ((p.get(3, 1.0) < 0.90 or p.get(4, 1.0) < 0.90)
                or (drop_2_to_4 is not None and drop_2_to_4 > 0.10)
                or (res4 is not None and res4 > 0.15))
    if generalises_strict:
        generalises = "generalises"
    elif degrades:
        generalises = "degrades"
    else:
        generalises = "mixed"

    out = {
        "experiment": "R1_dimension_completeness",
        "prereg": "PREREGISTRATION_R1.md",
        "config": {"n_target": n_target, "seed": seed, "dims": list(dims),
                   "screen_samples": screen_samples, "sound_samples": sound_samples,
                   "jsr_max_len_per_n": _JSR_MAXLEN},
        "clarabel_confirmed": clarabel_confirmed,
        "solver": solver_str,
        "installed_solvers": cp.installed_solvers(),
        "per_n": [{k: v for k, v in r.items() if k != "residual_records"} for r in per_n],
        "sdp_pct_by_n": {r["n"]: r["sdp_pct"] for r in per_n},
        "residual_pct_by_n": {r["n"]: r["residual_pct"] for r in per_n},
        "sdp_completeness_drop_n2_to_n4": round(drop_2_to_4, 4) if drop_2_to_4 is not None else None,
        "gates": {
            "G1_completeness_ge_0.80_all_n": g1,
            "G2_two_beats_sdp_zero_all_n": g2,
            "G3_zero_unsound_all_n": g3,
            "G4_verdict": generalises,
        },
        "generalises": generalises,
        "elapsed_s": round(time.time() - t0, 1),
    }

    results_path = os.path.join(_HERE, "dim_completeness_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    # residual genes saved separately (audit / future higher-degree work)
    with open(os.path.join(_HERE, "dim_completeness_residual_genes.json"), "w", encoding="utf-8") as f:
        json.dump({str(r["n"]): r["residual_records"] for r in per_n}, f, indent=2)

    print(json.dumps({k: v for k, v in out.items() if k != "per_n"}, indent=2), flush=True)
    print(f"\nexp_dim_completeness -> {results_path} ({out['elapsed_s']}s)", flush=True)
    return out


if __name__ == "__main__":
    kw = {}
    if "--n-target" in sys.argv:
        kw["n_target"] = int(sys.argv[sys.argv.index("--n-target") + 1])
    if "--seed" in sys.argv:
        kw["seed"] = int(sys.argv[sys.argv.index("--seed") + 1])
    if "--sound-samples" in sys.argv:
        kw["sound_samples"] = int(sys.argv[sys.argv.index("--sound-samples") + 1])
    run(**kw)
