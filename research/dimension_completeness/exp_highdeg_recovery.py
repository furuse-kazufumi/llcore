# SPDX-License-Identifier: Apache-2.0
"""R2a — does the LIFTED higher-degree SOS hierarchy RECOVER the dimension-grown residual?

R1 (`exp_dim_completeness.py`) found the QUADRATIC common-P SDP contraction-completeness DEGRADES
with coupled dimension (n=2: 92% -> n=3: 79.5% -> n=4: 48%), and the contracting-but-uncertified
residual GROWS (16 -> 41 -> 104 genes). This experiment asks whether the LIFTED higher-degree SOS
Lyapunov ladder (degree-4 = symmetric-2nd power, degree-6 = sym-3, degree-8 = sym-4) RECOVERS the
R1 residual at n=3 and n=4, or whether the degradation is FUNDAMENTAL (residual beyond the CPU SOS
hierarchy: switched-expansive or needing exact-JSR).

Reuses (does NOT reinvent) the already-verified lift math:
  * verifier_deg6.sym_power(A, degree) / certify_degN(vertices, degree)  (CLARABEL-pinned, deg2/3/4)
  * coupled_nd._jac_at_t / _box_vertices / t_min_per_coord                (the 2^n t-box vertices)
  * exp_dim_completeness.empirical_rho_fast / jsr_lb / _JSR_MAXLEN        (n-dim soundness oracles)

See PREREGISTRATION_R2a.md for the falsifiable gates (RG-recover / RG-sound / RG-verdict)
registered before this run.

Usage: py -3.11 exp_highdeg_recovery.py [--max-deg 4] [--sound-samples 20000]
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
_GATE = os.path.normpath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate"))
if _GATE not in sys.path:
    sys.path.insert(0, _GATE)

# n-dim substrate + the 2^n t-box vertex machinery (UNCHANGED from R1).
from coupled_nd import (  # noqa: E402
    CoupledNDGene,
    _box_vertices,
    _jac_at_t,
    t_min_per_coord,
)

# Already-verified lift math (CLARABEL-pinned). degree=2 -> deg4, 3 -> deg6, 4 -> deg8.
from verifier_deg6 import (  # noqa: E402
    _CLARABEL_OK,
    _SOLVER,
    certify_degN,
    mono_basis,
    sym_power,
)

# n-dim soundness oracles reused verbatim from R1.
from exp_dim_completeness import (  # noqa: E402
    _JSR_MAXLEN,
    empirical_rho_fast,
    jsr_lb,
)

try:
    import cvxpy as cp  # noqa: E402
except Exception:  # pragma: no cover
    cp = None


# degree label -> sym_power order. deg4 = sym-2, deg6 = sym-3, deg8 = sym-4.
_DEG_ORDER = {4: 2, 6: 3, 8: 4}


def _gene(rec) -> CoupledNDGene:
    decay = np.asarray(rec["decay"], dtype=np.float64)
    n = decay.shape[0]
    return CoupledNDGene.make(decay=decay, W=np.asarray(rec["W"], dtype=np.float64).reshape(n, n))


def _vertices_nd(g: CoupledNDGene, max_input_abs: float = 1.0) -> list[np.ndarray]:
    """The 2^n vertex Jacobians of the achievable-t box (the SAME set R1's cert_sdp used)."""
    t_lo = t_min_per_coord(g, max_input_abs)
    return [_jac_at_t(g, v) for v in _box_vertices(t_lo)]


# --------------------------------------------------------------------------- #
# Lift-math sanity: sym_power(A,k) vs brute-force monomial transform on n=3/n=4.
# --------------------------------------------------------------------------- #


def sanity_check_lift(seed: int = 7) -> dict:
    """Verify m(Az) = sym_power(A,k) m(z) on random n=3/n=4 matrices for k in {2,3,4}."""
    rng = np.random.default_rng(seed)
    worst = 0.0
    per = []
    for n in (3, 4):
        for k in (2, 3, 4):
            kmax = 0.0
            for _ in range(3):
                A = rng.standard_normal((n, n))
                z = rng.standard_normal(n)
                basis = mono_basis(n, k)
                m = np.array([np.prod([z[i] for i in b]) for b in basis], dtype=np.float64)
                Az = A @ z
                m_true = np.array([np.prod([Az[i] for i in b]) for b in basis], dtype=np.float64)
                err = float(np.max(np.abs(sym_power(A, k) @ m - m_true)))
                kmax = max(kmax, err)
            worst = max(worst, kmax)
            per.append({"n": n, "degree_lift": 2 * k, "sym_order": k,
                        "lift_dim": len(mono_basis(n, k)), "max_err": kmax})
    return {"max_err": worst, "ok": worst < 1e-8, "per": per}


# --------------------------------------------------------------------------- #
# Recovery + soundness for one dimension.
# --------------------------------------------------------------------------- #


def run_one_n(n: int, residual_recs: list, *, degrees: list[int],
              sound_samples: int, time_cap: float) -> dict:
    t0 = time.time()
    maxlen = _JSR_MAXLEN[n]
    n_in = len(residual_recs)

    # per-gene record of the lowest degree that recovered it (None = still residual).
    recovered_at: list = [None] * n_in
    # cumulative recovered set (cumulative over the degree ladder).
    cum_recovered = set()
    deg_new = {d: 0 for d in degrees}     # NEW recoveries first achieved at degree d
    n_unsound = 0
    unsound_examples = []
    max_rho_recovered = 0.0
    max_jsr_recovered = 0.0
    per_gene = []

    for gi, rec in enumerate(residual_recs):
        g = _gene(rec)
        verts = _vertices_nd(g)
        rgi = {"idx": gi, "recovered_degree": None}
        for d in degrees:  # ascending: 4, 6, 8 -> cumulative by construction
            if (time.time() - t0) >= time_cap:
                rgi["time_capped"] = True
                break
            if certify_degN(verts, _DEG_ORDER[d]):
                recovered_at[gi] = d
                deg_new[d] += 1
                cum_recovered.add(gi)
                rgi["recovered_degree"] = d
                # INDEPENDENT soundness oracle for this newly-recovered gene.
                rho_hi = empirical_rho_fast(g, n_samples=sound_samples, seed=0)
                j = jsr_lb(verts, max_len=maxlen)
                rgi["rho_hi"] = round(rho_hi, 6)
                rgi["jsr_lb"] = round(j, 6)
                max_rho_recovered = max(max_rho_recovered, rho_hi)
                max_jsr_recovered = max(max_jsr_recovered, j)
                if rho_hi >= 1.0 or j >= 1.0 - 1e-9:
                    n_unsound += 1
                    unsound_examples.append({"idx": gi, "degree": d,
                                             "rho_hi": rho_hi, "jsr_lb": j, **rec})
                break  # lowest-degree recovery; cumulative ladder stops here
        per_gene.append(rgi)

    # cumulative counts down the ladder.
    cum = 0
    cum_by_deg = {}
    for d in degrees:
        cum += deg_new[d]
        cum_by_deg[d] = cum
    total_recovered = cum
    still_residual = n_in - total_recovered

    rec_out = {
        "n": n,
        "residual_in": n_in,
        "jsr_max_len": maxlen,
        "n_box_vertices": 2 ** n,
        "deg_new_recoveries": {str(d): deg_new[d] for d in degrees},          # first recovered at d
        "cum_recovered_by_deg": {str(d): cum_by_deg[d] for d in degrees},      # cumulative
        "deg4_rec": cum_by_deg.get(4, 0),
        "deg6_rec": cum_by_deg.get(6, 0),
        "deg8_rec": cum_by_deg.get(8, 0),
        "cum_recovered": total_recovered,
        "still_residual": still_residual,
        "recovery_frac": round(total_recovered / n_in, 4) if n_in else 0.0,
        "unsound": n_unsound,
        "max_rho_over_recovered": round(max_rho_recovered, 6),
        "max_jsr_lb_over_recovered": round(max_jsr_recovered, 6),
        "unsound_examples": unsound_examples,
        "elapsed_s": round(time.time() - t0, 1),
        "per_gene": per_gene,
    }
    print(f"  n={n} residual_in={n_in} "
          f"deg4_rec={rec_out['deg4_rec']} deg6_rec={rec_out['deg6_rec']} deg8_rec={rec_out['deg8_rec']} "
          f"cum={total_recovered} ({100.0*rec_out['recovery_frac']:.1f}%) "
          f"still={still_residual} unsound={n_unsound} "
          f"maxrho={rec_out['max_rho_over_recovered']} maxjsr={rec_out['max_jsr_lb_over_recovered']} "
          f"({rec_out['elapsed_s']}s)", flush=True)
    return rec_out


def run(max_deg: int = 4, sound_samples: int = 20000, time_cap_per_n: float = 3600.0) -> dict:
    t0 = time.time()

    # ---- CLARABEL confirmation (HARD guardrail) ----
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

    # ---- lift-math sanity BEFORE trusting any recovery ----
    sanity = sanity_check_lift()
    print(f"lift-math sanity: max_err={sanity['max_err']:.2e} ok={sanity['ok']}", flush=True)
    for p in sanity["per"]:
        print(f"    n={p['n']} deg{p['degree_lift']} (sym-{p['sym_order']}, dim {p['lift_dim']}): "
              f"err={p['max_err']:.2e}", flush=True)
    if not sanity["ok"]:
        raise SystemExit(f"Lift-math sanity FAILED (max_err={sanity['max_err']:.2e} >= 1e-8). "
                         f"Refusing to trust recoveries on a mis-constructed lift.")

    # degree ladder: 4 (sym-2), 6 (sym-3), 8 (sym-4 if max_deg>=4).
    degrees = [d for d in (4, 6, 8) if d <= 4 + 2 * (max_deg - 1)]  # max_deg=4 -> [4,6,8]
    print(f"degree ladder: {degrees}", flush=True)

    with open(os.path.join(_HERE, "dim_completeness_residual_genes.json"), encoding="utf-8") as f:
        residual = json.load(f)

    per_n = []
    for n in (3, 4):
        recs = residual[str(n)]
        per_n.append(run_one_n(n, recs, degrees=degrees, sound_samples=sound_samples,
                               time_cap=time_cap_per_n))

    # ---- gate evaluation (RG-recover / RG-sound / RG-verdict) ----
    by_n = {r["n"]: r for r in per_n}
    rg_sound = all(r["unsound"] == 0 for r in per_n)

    frac3 = by_n[3]["recovery_frac"]
    frac4 = by_n[4]["recovery_frac"]
    # verdict thresholds locked in prereg:
    #   recoverable: cum lifted (through deg8) closes MAJORITY (>50%) at BOTH n=3 and n=4, 0 unsound
    #   fundamental: closes only SMALL fraction (<25%) at n=4 (worst dimension), 0 unsound
    #   partial: in between
    if not rg_sound:
        verdict = "INVALID_unsound"
    elif frac3 > 0.50 and frac4 > 0.50:
        verdict = "recoverable"
    elif frac4 < 0.25:
        verdict = "fundamental"
    else:
        verdict = "partial"

    out = {
        "experiment": "R2a_highdeg_recovery",
        "prereg": "PREREGISTRATION_R2a.md",
        "r1_source": "dim_completeness_residual_genes.json",
        "config": {"degrees": degrees, "sound_samples": sound_samples,
                   "jsr_max_len_per_n": {str(k): v for k, v in _JSR_MAXLEN.items()},
                   "deg_to_sym_order": {str(k): v for k, v in _DEG_ORDER.items()}},
        "clarabel_confirmed": clarabel_confirmed,
        "solver": solver_str,
        "installed_solvers": cp.installed_solvers(),
        "lift_math_sanity": sanity,
        "lift_math_ok": sanity["ok"],
        "per_n": [{k: v for k, v in r.items() if k not in ("per_gene", "unsound_examples")}
                  for r in per_n],
        "per_n_detail": per_n,
        "gates": {
            "RG_recover_n3_frac": frac3,
            "RG_recover_n4_frac": frac4,
            "RG_sound_zero_unsound": rg_sound,
            "RG_verdict": verdict,
        },
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
    }

    results_path = os.path.join(_HERE, "highdeg_recovery_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("per_n_detail", "lift_math_sanity")}, indent=2), flush=True)
    print(f"\nexp_highdeg_recovery -> {results_path} ({out['elapsed_s']}s)", flush=True)
    return out


if __name__ == "__main__":
    kw = {}
    if "--max-deg" in sys.argv:
        kw["max_deg"] = int(sys.argv[sys.argv.index("--max-deg") + 1])
    if "--sound-samples" in sys.argv:
        kw["sound_samples"] = int(sys.argv[sys.argv.index("--sound-samples") + 1])
    run(**kw)
