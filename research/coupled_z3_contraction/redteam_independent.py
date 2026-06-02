# SPDX-License-Identifier: Apache-2.0
"""INDEPENDENT RED-TEAM oracle for Track C (does NOT reuse the implementer's
empirical_box_norms / infnorm_over_box_freeT / runner). Re-derives everything from
the raw map definition to avoid trusting the implementer's helpers.

Goals:
  (R1) Independent soundness check: for every gene Z3 (free01 / tmin1) ADMITS,
       independently measure empirical ||J||_inf and ||J||_2 over a FRESH dense
       (s,x) sample (different seed) INCLUDING all 16 sign-corners + extra t-extremal
       points. Find ANY admitted gene with emp_infnorm > 1+tol (false admit => UNSOUND).
  (R2) Independent C2: count genes where the scalar diagonal heuristic ADMITS but the
       map is GENUINELY expansive (independent emp_infnorm > 1). Confirm Z3 rejects each.
  (R3) Over-claim probe: compare Z3 verdict against an INDEPENDENT closed-form inf-norm
       (max-row-sum over t-endpoints, re-derived) to test whether the over-the-box
       quantifier genuinely needs a solver, or Z3 == closed-form scalar max-row-sum.
  (R4) Independent t-floor soundness: verify the analytic free-t inf-norm I compute
       is actually >= every empirically-achieved inf-norm (the over-approx claim).

Determinism: numpy default_rng(seed); seeds reported. Z3 deterministic.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# We DO import the gene container + the two checkers under test (those ARE the
# subjects of the red-team), but NOT the implementer's empirical / closed-form oracles.
from coupled_map import CoupledGene  # noqa: E402
from scalar_heuristic import scalar_diagonal_admit  # noqa: E402
from z3_infnorm_certifier import certify_infnorm_contraction  # noqa: E402


# ----------------------------------------------------------------------------
# INDEPENDENT re-derivation of the map / Jacobian (no implementer helper reused).
# ----------------------------------------------------------------------------
def my_jacobian(decay, W, s, x):
    """J = diag(decay) + diag((1-decay)*t) @ W, t = sech^2(W@s + x) (V=I)."""
    decay = np.asarray(decay, float)
    W = np.asarray(W, float)
    pre = W @ s + x  # V = identity
    t = 1.0 - np.tanh(pre) ** 2
    return np.diag(decay) + np.diag((1.0 - decay) * t) @ W


def my_infnorm(J):
    return float(np.abs(J).sum(axis=1).max())


def my_2norm(J):
    return float(np.linalg.svd(J, compute_uv=False)[0])


def my_rho(J):
    return float(np.max(np.abs(np.linalg.eigvals(J))))


def independent_emp_norms(decay, W, *, n_samples, seed, max_input_abs=1.0):
    """Fresh dense box sample of the EXACT Jacobian. Returns (emp_inf, emp_2, emp_rho, worst).

    Includes: n_samples uniform random + all 16 sign corners of (s,x) +
    s=0 (t=1 everywhere) crossed with all x-corners and x=0 (the t=1 face, where the
    off-diagonal term is maximal because t is largest there). This is the most adversarial
    cheap sampling for max-row-sum.
    """
    decay = np.asarray(decay, float)
    W = np.asarray(W, float)
    rng = np.random.default_rng(seed)
    S = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    X = rng.uniform(-max_input_abs, max_input_abs, size=(n_samples, 2))

    s_corners = np.array(list(itertools.product([-1.0, 1.0], repeat=2)))
    x_corners = np.array(list(itertools.product([-max_input_abs, max_input_abs], repeat=2)))
    # full cartesian of s-corners x x-corners (16) + s=0 face x all x-corners + zero
    extra_s = []
    extra_x = []
    for sc in list(s_corners) + [np.zeros(2)]:
        for xc in list(x_corners) + [np.zeros(2)]:
            extra_s.append(sc)
            extra_x.append(xc)
    S = np.vstack([S, np.array(extra_s)])
    X = np.vstack([X, np.array(extra_x)])

    best_inf = -1.0
    best_2 = -1.0
    best_rho = -1.0
    worst = None
    for k in range(S.shape[0]):
        J = my_jacobian(decay, W, S[k], X[k])
        infn = my_infnorm(J)
        if infn > best_inf:
            best_inf = infn
            worst = (S[k].tolist(), X[k].tolist())
        b2 = my_2norm(J)
        if b2 > best_2:
            best_2 = b2
        r = my_rho(J)
        if r > best_rho:
            best_rho = r
    return best_inf, best_2, best_rho, worst


def my_t_min(decay, W, max_input_abs=1.0):
    W = np.asarray(W, float)
    # |pre_i| <= sum_j|W_ij| + max_input_abs*1 (V=I diagonal, so |V_ii|=1)
    M = np.abs(W).sum(axis=1) + max_input_abs * 1.0
    return 1.0 - np.tanh(M) ** 2


def my_closedform_infnorm(decay, W, t_lo):
    """Independent analytic sup over t in [t_lo,1]^2 of ||J(t)||_inf.

    Row i abs-sum a_i(t_i) = |d_i + (1-d_i)*t_i*W_ii| + (1-d_i)*t_i*|W_ij|.
    Off-diagonal term: monotone increasing in t_i (since (1-d_i)>=0). Diagonal abs: convex
    piecewise-linear (V-shaped) in t_i. Sum of convex + monotone-increasing is convex in t_i,
    so its max over [t_lo_i,1] is at an endpoint. Evaluate both endpoints, max per row, max over rows.
    """
    decay = np.asarray(decay, float)
    W = np.asarray(W, float)
    t_lo = np.asarray(t_lo, float)
    best = -1.0
    for i in range(2):
        j = 1 - i
        for ti in (t_lo[i], 1.0):
            diag = abs(decay[i] + (1.0 - decay[i]) * ti * W[i, i])
            off = (1.0 - decay[i]) * ti * abs(W[i, j])
            best = max(best, diag + off)
    return float(best)


# ----------------------------------------------------------------------------
# Gene population (re-derive the SAME grid + random the runner uses, independently).
# ----------------------------------------------------------------------------
def build_population(n_random=3000, seed=0):
    decays = [0.2, 0.5, 0.8]
    wdiag = [-0.5, 0.5, 0.9]
    woff = [-1.5, -0.9, 0.0, 0.9, 1.5]
    genes = []
    for d0, d1 in itertools.product(decays, decays):
        for wd in wdiag:
            for wo in woff:
                W_sym = [[wd, wo], [wo, wd]]
                W_asym = [[wd, wo], [-wo, wd]]
                for W in (W_sym, W_asym):
                    genes.append((np.array([d0, d1], float), np.array(W, float)))
    n_grid = len(genes)
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        decay = rng.uniform(0.0, 1.0, size=2)
        W = rng.uniform(-2.0, 2.0, size=(2, 2))
        genes.append((decay, W))
    return genes, n_grid


def main():
    t0 = time.time()
    # Independent seeds: emp seed 777 (NOT 0 used by the runner), closed-form is exact.
    EMP_SEED = 777
    EMP_N = 6000  # uniform points + 25 structured points; fresh seed
    HEAVY_N = 80000  # heavy re-confirm for any candidate false-admit or C2 driver
    TOL = 1e-9
    MAX_INPUT = 1.0

    genes, n_grid = build_population(n_random=3000, seed=0)
    print(f"[redteam] population: grid={n_grid} random={len(genes)-n_grid} total={len(genes)}", flush=True)

    # Aggregates
    c1 = {  # admitted-by-z3 then check emp_inf
        "free01": {"n_admit": 0, "false_admits": [], "worst_emp_inf": 0.0,
                   "max_cf_minus_emp": -9, "min_cf_minus_emp": 9},
        "tmin1": {"n_admit": 0, "false_admits": [], "worst_emp_inf": 0.0,
                  "max_cf_minus_emp": -9, "min_cf_minus_emp": 9},
    }
    c2 = {  # scalar admits AND independently expansive
        "free01": {"n_scalar_admit_expansive": 0, "z3_rejects_all": True,
                   "examples": [], "min_emp_inf": 9.9, "max_emp_inf": 0.0},
        "tmin1": {"n_scalar_admit_expansive": 0, "z3_rejects_all": True,
                  "examples": [], "min_emp_inf": 9.9, "max_emp_inf": 0.0},
    }
    # Over-claim probe: z3 verdict vs independent closed-form inf-norm < 1
    disagree = {"free01": 0, "tmin1": 0, "examples": []}
    # over-approx soundness: cf_free should be >= emp_inf ALWAYS
    overapprox_violations = []  # (cf < emp) would break the over-approximation argument

    scalar_admit_total = {"free01": 0, "tmin1": 0}
    z3_admit_total = {"free01": 0, "tmin1": 0}
    emp_expansive_total = 0

    for gi, (decay, W) in enumerate(genes):
        # clip to legal box (gene container does this; replicate independently)
        decay = np.clip(decay, 0.0, 1.0)
        W = np.clip(W, -2.0, 2.0)
        gene = CoupledGene.make(decay=decay, W=W)

        emp_inf, emp_2, emp_rho, worst = independent_emp_norms(
            decay, W, n_samples=EMP_N, seed=EMP_SEED, max_input_abs=MAX_INPUT)
        if emp_inf > 1.0 + TOL:
            emp_expansive_total += 1

        t_lo = {"free01": np.zeros(2), "tmin1": my_t_min(decay, W, MAX_INPUT)}

        for dom in ("free01", "tmin1"):
            z3r = certify_infnorm_contraction(gene, t_domain=dom)
            scr = scalar_diagonal_admit(gene, t_domain=dom)
            cf = my_closedform_infnorm(decay, W, t_lo[dom])
            if scr.admit:
                scalar_admit_total[dom] += 1
            if z3r.certified is True:
                z3_admit_total[dom] += 1

            # --- over-approx soundness: free-t cf must dominate emp_inf ---
            if dom == "free01" and cf + 1e-9 < emp_inf:
                overapprox_violations.append({
                    "index": gi, "decay": decay.tolist(), "W": W.tolist(),
                    "cf_free": cf, "emp_inf": emp_inf})

            # --- C1: z3 admits then check independent emp_inf ---
            if z3r.certified is True:
                d = c1[dom]
                d["n_admit"] += 1
                d["worst_emp_inf"] = max(d["worst_emp_inf"], emp_inf)
                d["max_cf_minus_emp"] = max(d["max_cf_minus_emp"], cf - emp_inf)
                d["min_cf_minus_emp"] = min(d["min_cf_minus_emp"], cf - emp_inf)
                if emp_inf > 1.0 + TOL:
                    d["false_admits"].append({
                        "index": gi, "decay": decay.tolist(), "W": W.tolist(),
                        "emp_inf": emp_inf, "emp_2": emp_2, "cf": cf, "worst": worst})

            # --- C2: scalar admits AND independently expansive ---
            if scr.admit and emp_inf > 1.0 + TOL:
                d = c2[dom]
                d["n_scalar_admit_expansive"] += 1
                d["min_emp_inf"] = min(d["min_emp_inf"], emp_inf)
                d["max_emp_inf"] = max(d["max_emp_inf"], emp_inf)
                if z3r.certified is not False:  # Z3 should REJECT (certified False)
                    d["z3_rejects_all"] = False
                if len(d["examples"]) < 12:
                    d["examples"].append({
                        "index": gi, "source": "grid" if gi < n_grid else "random",
                        "decay": decay.tolist(), "W": W.tolist(),
                        "scalar_L": scr.scalar_L, "emp_inf": emp_inf,
                        "emp_rho": emp_rho, "z3_certified": z3r.certified})

            # --- over-claim: z3 verdict vs closed-form inf-norm<1 ---
            cf_admit = cf < 1.0
            if cf_admit != (z3r.certified is True):
                disagree[dom] += 1
                if len(disagree["examples"]) < 12:
                    disagree["examples"].append({
                        "index": gi, "dom": dom, "decay": decay.tolist(), "W": W.tolist(),
                        "cf": cf, "cf_admit": cf_admit, "z3_certified": z3r.certified,
                        "z3_status": z3r.solver_status})

        if (gi + 1) % 500 == 0:
            print(f"  ...{gi+1}/{len(genes)} ({time.time()-t0:.1f}s)", flush=True)

    # Heavy re-confirm: any C1 false-admit or top C2 examples with HEAVY_N samples + fresh seed 999
    def heavy(decay, W):
        return independent_emp_norms(decay, W, n_samples=HEAVY_N, seed=999, max_input_abs=MAX_INPUT)

    heavy_c1 = {}
    for dom in ("free01", "tmin1"):
        fa = c1[dom]["false_admits"]
        confirmed = []
        for r in fa[:25]:
            hi, h2, hr, hw = heavy(np.array(r["decay"]), np.array(r["W"]))
            confirmed.append({"index": r["index"], "emp_inf_heavy": hi})
        heavy_c1[dom] = {"n_false_admit_light": len(fa),
                         "n_false_admit_heavy_confirmed": sum(1 for c in confirmed if c["emp_inf_heavy"] > 1.0 + TOL),
                         "samples": confirmed[:10]}

    heavy_c2 = {}
    for dom in ("free01", "tmin1"):
        exs = c2[dom]["examples"]
        still = 0
        for r in exs:
            hi, h2, hr, hw = heavy(np.array(r["decay"]), np.array(r["W"]))
            if hi > 1.0 + TOL:
                still += 1
        heavy_c2[dom] = {"n_examples": len(exs), "still_expansive_heavy": still}

    out = {
        "meta": {"emp_seed": EMP_SEED, "emp_n": EMP_N, "heavy_n": HEAVY_N,
                 "heavy_seed": 999, "tol": TOL, "n_genes": len(genes), "n_grid": n_grid,
                 "wall_seconds": round(time.time() - t0, 1), "python": sys.version.split()[0]},
        "population": {
            "emp_expansive_total": emp_expansive_total,
            "scalar_admit": scalar_admit_total,
            "z3_admit": z3_admit_total,
        },
        "R1_soundness_C1": {
            dom: {"n_admit": c1[dom]["n_admit"],
                  "n_false_admits": len(c1[dom]["false_admits"]),
                  "worst_admitted_emp_inf": c1[dom]["worst_emp_inf"],
                  "min_cf_minus_emp_over_admitted": c1[dom]["min_cf_minus_emp"],
                  "false_admit_examples": c1[dom]["false_admits"][:10]}
            for dom in ("free01", "tmin1")},
        "R1_heavy_confirm": heavy_c1,
        "R2_C2_value_over_scalar": {
            dom: {"n_scalar_admit_expansive": c2[dom]["n_scalar_admit_expansive"],
                  "z3_rejects_all_of_them": c2[dom]["z3_rejects_all"],
                  "min_emp_inf": (c2[dom]["min_emp_inf"] if c2[dom]["n_scalar_admit_expansive"] else None),
                  "max_emp_inf": (c2[dom]["max_emp_inf"] if c2[dom]["n_scalar_admit_expansive"] else None),
                  "examples": c2[dom]["examples"]}
            for dom in ("free01", "tmin1")},
        "R2_heavy_confirm": heavy_c2,
        "R3_overclaim_z3_vs_closedform": {
            "free01_disagree": disagree["free01"], "tmin1_disagree": disagree["tmin1"],
            "examples": disagree["examples"]},
        "R4_overapprox_violations_cf_lt_emp": {
            "n": len(overapprox_violations), "examples": overapprox_violations[:10]},
    }
    out_path = _HERE / "redteam_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[redteam] wrote {out_path} ({out['meta']['wall_seconds']}s)")
    print("=== SUMMARY ===")
    print(f"emp_expansive_total={emp_expansive_total}/{len(genes)}")
    for dom in ("free01", "tmin1"):
        r1 = out["R1_soundness_C1"][dom]
        r2 = out["R2_C2_value_over_scalar"][dom]
        print(f"[{dom}] C1: z3_admit={r1['n_admit']} false_admits={r1['n_false_admits']} "
              f"worst_admitted_emp_inf={r1['worst_admitted_emp_inf']:.6f} "
              f"min(cf-emp)={r1['min_cf_minus_emp_over_admitted']:.2e}")
        print(f"[{dom}] C2: scalar_admit_expansive={r2['n_scalar_admit_expansive']} "
              f"z3_rejects_all={r2['z3_rejects_all_of_them']} "
              f"emp_inf range=[{r2['min_emp_inf']},{r2['max_emp_inf']}]")
    print(f"R3 over-claim z3_vs_closedform disagree: free01={disagree['free01']} tmin1={disagree['tmin1']}")
    print(f"R4 over-approx violations (cf_free < emp_inf): {len(overapprox_violations)}")
    return out


if __name__ == "__main__":
    main()
