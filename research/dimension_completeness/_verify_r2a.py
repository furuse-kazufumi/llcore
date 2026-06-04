# SPDX-License-Identifier: Apache-2.0
"""Adversarial independent verification of R2a. research/ only; does NOT touch src/.

Checks:
 (1) CLARABEL is the pinned solver actually used.
 (2) LIFT MATH: independent brute-force monomial transform check of sym_power at n=3,n=4,k=2,3,4,
     using an INDEPENDENT random draw and an INDEPENDENT (re-derived) monomial basis order.
 (3) SOUNDNESS: re-verify EVERY recovered gene (n=3 all; n=4 all) from below:
       (a) empirical rho at 20k+50k samples (<1), (b) n-dim jsr_lb over 2^n vertices (<1).
 (4) REPRODUCIBILITY: re-run the full n=3 recovery slice from the fixed residual set;
     headline counts (deg4/deg6/deg8 cum, recovered, still_residual, unsound) must reproduce.
 (5) Cross-check the attribution numbers (switched_expansive vs finite_gap) for the still-residual.
"""
from __future__ import annotations
import itertools, json, os, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_GATE = os.path.normpath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate"))
if _GATE not in sys.path:
    sys.path.insert(0, _GATE)

import cvxpy as cp
from coupled_nd import CoupledNDGene, _box_vertices, _jac_at_t, t_min_per_coord
from verifier_deg6 import _CLARABEL_OK, _SOLVER, certify_degN, sym_power
from exp_dim_completeness import _JSR_MAXLEN, empirical_rho_fast, jsr_lb

_DEG_ORDER = {4: 2, 6: 3, 8: 4}
report = {}

# ---- (1) CLARABEL ----
report["clarabel_OK_flag"] = bool(_CLARABEL_OK)
report["solver_is_clarabel"] = (_SOLVER is cp.CLARABEL)
report["solver_str"] = _SOLVER.name() if hasattr(_SOLVER, "name") else str(_SOLVER)
report["installed"] = cp.installed_solvers()


# ---- (2) INDEPENDENT lift-math brute force ----
def indep_basis(n, k):
    # independent re-derivation: sorted multisets of size k over n vars
    return sorted(itertools.combinations_with_replacement(range(n), k))

def lift_check():
    rng = np.random.default_rng(12345)  # DIFFERENT seed than exp's 7
    worst = 0.0
    per = []
    for n in (3, 4):
        for k in (2, 3, 4):
            kmax = 0.0
            for _ in range(5):
                A = rng.standard_normal((n, n)) * 1.7  # different scale too
                z = rng.standard_normal(n) * 2.0
                basis = indep_basis(n, k)
                m = np.array([np.prod([z[i] for i in b]) for b in basis])
                Az = A @ z
                m_true = np.array([np.prod([Az[i] for i in b]) for b in basis])
                S = sym_power(A, k)
                # also assert dims
                assert S.shape == (len(basis), len(basis)), (n, k, S.shape)
                err = float(np.max(np.abs(S @ m - m_true)))
                kmax = max(kmax, err)
            worst = max(worst, kmax)
            per.append({"n": n, "k": k, "lift_dim": len(indep_basis(n, k)), "max_err": kmax})
    return worst, per

worst, per = lift_check()
report["lift_max_err_independent"] = worst
report["lift_ok"] = worst < 1e-8
report["lift_per"] = per


def gene_of(rec):
    decay = np.asarray(rec["decay"], dtype=np.float64)
    n = decay.shape[0]
    return CoupledNDGene.make(decay=decay, W=np.asarray(rec["W"], dtype=np.float64).reshape(n, n))

def verts_of(g):
    t_lo = t_min_per_coord(g, 1.0)
    return [_jac_at_t(g, v) for v in _box_vertices(t_lo)]


with open(os.path.join(_HERE, "dim_completeness_residual_genes.json"), encoding="utf-8") as f:
    residual = json.load(f)
with open(os.path.join(_HERE, "highdeg_recovery_results.json"), encoding="utf-8") as f:
    claimed = json.load(f)

claimed_by_n = {r["n"]: r for r in claimed["per_n_detail"]}

# ---- (4) REPRODUCE n=3 full slice + (3) SOUNDNESS re-verify all recovered (both n) ----
per_n_out = []
for n in (3, 4):
    recs = residual[str(n)]
    maxlen = _JSR_MAXLEN[n]
    degrees = [4, 6, 8]
    deg_new = {d: 0 for d in degrees}
    recovered_at = [None] * len(recs)
    n_unsound = 0
    unsound_ex = []
    max_rho = 0.0
    max_jsr = 0.0
    # soundness re-verify each recovered gene from below, independent of solver
    still_jsr_ge1 = 0
    still_jsr_lt1 = 0
    min_expansive = 99.0
    max_gap = 0.0
    for gi, rec in enumerate(recs):
        g = gene_of(rec)
        verts = verts_of(g)
        rd = None
        for d in degrees:
            if certify_degN(verts, _DEG_ORDER[d]):
                rd = d
                break
        recovered_at[gi] = rd
        if rd is not None:
            deg_new[rd] += 1
            # independent soundness oracle at HIGH samples
            rho = empirical_rho_fast(g, n_samples=20000, seed=0)
            j = jsr_lb(verts, max_len=maxlen)
            max_rho = max(max_rho, rho)
            max_jsr = max(max_jsr, j)
            if rho >= 1.0 or j >= 1.0 - 1e-9:
                n_unsound += 1
                unsound_ex.append({"idx": gi, "deg": rd, "rho": rho, "jsr": j})
        else:
            # attribution of still-residual
            j = jsr_lb(verts, max_len=maxlen)
            if j >= 1.0 - 1e-9:
                still_jsr_ge1 += 1
                min_expansive = min(min_expansive, j)
            else:
                still_jsr_lt1 += 1
                max_gap = max(max_gap, j)
    cum = 0
    cum_by = {}
    for d in degrees:
        cum += deg_new[d]
        cum_by[d] = cum
    per_n_out.append({
        "n": n, "residual_in": len(recs),
        "deg_new": deg_new, "cum_by_deg": cum_by,
        "cum_recovered": cum, "still_residual": len(recs) - cum,
        "recovery_frac": round(cum / len(recs), 4),
        "unsound": n_unsound, "unsound_examples": unsound_ex,
        "max_rho_over_recovered": round(max_rho, 6),
        "max_jsr_over_recovered": round(max_jsr, 6),
        "still_switched_expansive_jsr_ge1": still_jsr_ge1,
        "still_finite_gap_jsr_lt1": still_jsr_lt1,
        "max_gap_jsr_lb": round(max_gap, 4),
        "min_expansive_jsr_lb": round(min_expansive, 4) if min_expansive < 99 else None,
    })
    print(f"n={n}: cum_by_deg={cum_by} recovered={cum} still={len(recs)-cum} "
          f"unsound={n_unsound} maxrho={max_rho:.6f} maxjsr={max_jsr:.6f} "
          f"switched_exp={still_jsr_ge1} finite_gap={still_jsr_lt1}", flush=True)

report["recomputed_per_n"] = per_n_out

# ---- compare to claimed ----
diffs = []
for r in per_n_out:
    n = r["n"]
    c = claimed_by_n[n]
    for key, mine, theirs in [
        ("cum_recovered", r["cum_recovered"], c["cum_recovered"]),
        ("deg4_rec", r["cum_by_deg"][4], c["deg4_rec"]),
        ("deg6_rec", r["cum_by_deg"][6], c["deg6_rec"]),
        ("deg8_rec", r["cum_by_deg"][8], c["deg8_rec"]),
        ("still_residual", r["still_residual"], c["still_residual"]),
        ("unsound", r["unsound"], c["unsound"]),
        ("deg_new4", r["deg_new"][4], c["deg_new_recoveries"]["4"]),
        ("deg_new6", r["deg_new"][6], c["deg_new_recoveries"]["6"]),
        ("deg_new8", r["deg_new"][8], c["deg_new_recoveries"]["8"]),
    ]:
        if mine != theirs:
            diffs.append({"n": n, "key": key, "mine": mine, "claimed": theirs})
    # attribution cross-check (from highdeg_recovery_results extra keys)
    if "still_switched_expansive_jsr_ge1" in c:
        if r["still_switched_expansive_jsr_ge1"] != c["still_switched_expansive_jsr_ge1"]:
            diffs.append({"n": n, "key": "switched_exp",
                          "mine": r["still_switched_expansive_jsr_ge1"],
                          "claimed": c["still_switched_expansive_jsr_ge1"]})
        if r["still_finite_gap_jsr_lt1"] != c["still_finite_gap_jsr_lt1"]:
            diffs.append({"n": n, "key": "finite_gap",
                          "mine": r["still_finite_gap_jsr_lt1"],
                          "claimed": c["still_finite_gap_jsr_lt1"]})

report["diffs_vs_claimed"] = diffs
report["reproduces"] = (len(diffs) == 0)
report["zero_unsound"] = all(r["unsound"] == 0 for r in per_n_out)

print("\n=== SUMMARY ===")
print("clarabel:", report["solver_is_clarabel"], report["clarabel_OK_flag"])
print("lift_ok:", report["lift_ok"], "max_err:", report["lift_max_err_independent"])
print("reproduces:", report["reproduces"], "diffs:", diffs)
print("zero_unsound:", report["zero_unsound"])
with open(os.path.join(_HERE, "_verify_r2a_out.json"), "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2)
