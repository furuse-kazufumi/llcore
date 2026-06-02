# SPDX-License-Identifier: Apache-2.0
"""FAST independent red-team sweep (vectorized inf-norm; SVD/eigvals only where needed).

Independent of the implementer's empirical_box_norms / runner. Re-derives the Jacobian and
inf-norm from scratch, vectorized over the sample for speed. ||J||_inf is the C1/C2 oracle;
||J||_2 and rho are computed ONLY at the inf-norm argmax point (+ corners) for context (C3-ish).

Seeds: emp uniform seed 777 (NOT the runner's 0); heavy reconfirm seed 999. Reported in JSON.
"""
from __future__ import annotations
import itertools, json, sys, time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coupled_map import CoupledGene
from scalar_heuristic import scalar_diagonal_admit
from z3_infnorm_certifier import certify_infnorm_contraction

TOL = 1e-9
MAX_INPUT = 1.0


def build_population(n_random=3000, seed=0):
    decays = [0.2, 0.5, 0.8]; wdiag = [-0.5, 0.5, 0.9]; woff = [-1.5, -0.9, 0.0, 0.9, 1.5]
    genes = []
    for d0, d1 in itertools.product(decays, decays):
        for wd in wdiag:
            for wo in woff:
                for W in ([[wd, wo], [wo, wd]], [[wd, wo], [-wo, wd]]):
                    genes.append((np.array([d0, d1], float), np.array(W, float)))
    n_grid = len(genes)
    rng = np.random.default_rng(seed)
    for _ in range(n_random):
        genes.append((rng.uniform(0, 1, 2), rng.uniform(-2, 2, (2, 2))))
    return genes, n_grid


# structured stress points reused for every gene (corners + t=1 faces)
_SC = list(itertools.product([-1.0, 1.0], repeat=2)) + [(0.0, 0.0)]
_XC = list(itertools.product([-MAX_INPUT, MAX_INPUT], repeat=2)) + [(0.0, 0.0)]
_STRUCT_S = np.array([s for s in _SC for _ in _XC], float)
_STRUCT_X = np.array([x for _ in _SC for x in _XC], float)


def emp_infnorm_fast(decay, W, *, n_samples, seed):
    """Vectorized empirical sup of ||J||_inf over uniform sample + structured stress points.

    Returns (emp_inf, argmax_s, argmax_x). J = diag(decay)+diag((1-decay)*t)@W, t=sech^2(W@s+x).
    """
    rng = np.random.default_rng(seed)
    S = np.vstack([rng.uniform(-1, 1, (n_samples, 2)), _STRUCT_S])
    X = np.vstack([rng.uniform(-MAX_INPUT, MAX_INPUT, (n_samples, 2)), _STRUCT_X])
    pre = S @ W.T + X                       # (N,2): pre_i = W[i,:]·s + x_i
    t = 1.0 - np.tanh(pre) ** 2             # (N,2)
    om = (1.0 - decay)                       # (2,)
    # J[k] = diag(decay) + diag(om*t[k]) @ W ; row i abs-sum = sum_j |Jij|
    # Jij = (i==j)*decay_i + om_i*t_ki*W_ij
    a = om[None, :] * t                      # (N,2)  scale per row
    # build |Jij| for all k, i, j
    # diag part adds decay_i to the (i,i) entry
    J = a[:, :, None] * W[None, :, :]        # (N,2,2): a_ki * W_ij
    J[:, 0, 0] += decay[0]
    J[:, 1, 1] += decay[1]
    rowsum = np.abs(J).sum(axis=2)           # (N,2)
    infn = rowsum.max(axis=1)                # (N,)
    k = int(np.argmax(infn))
    return float(infn[k]), S[k].copy(), X[k].copy()


def norms_at(decay, W, s, x):
    pre = W @ s + x; t = 1.0 - np.tanh(pre) ** 2
    J = np.diag(decay) + np.diag((1.0 - decay) * t) @ W
    two = float(np.linalg.svd(J, compute_uv=False)[0])
    rho = float(np.max(np.abs(np.linalg.eigvals(J))))
    return two, rho


def emp_rho_min_fast(decay, W, *, n_samples, seed):
    """Sup of spectral radius over the box (for C3). Cheap closed-form 2x2 eigen magnitude."""
    rng = np.random.default_rng(seed)
    S = np.vstack([rng.uniform(-1, 1, (n_samples, 2)), _STRUCT_S])
    X = np.vstack([rng.uniform(-MAX_INPUT, MAX_INPUT, (n_samples, 2)), _STRUCT_X])
    pre = S @ W.T + X; t = 1.0 - np.tanh(pre) ** 2
    a = (1.0 - decay)[None, :] * t
    J = a[:, :, None] * W[None, :, :]
    J[:, 0, 0] += decay[0]; J[:, 1, 1] += decay[1]
    # 2x2 eigen magnitude closed form: eig = (tr +- sqrt(tr^2-4det))/2
    tr = J[:, 0, 0] + J[:, 1, 1]
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    disc = tr * tr - 4 * det
    rho = np.where(
        disc >= 0,
        np.maximum(np.abs((tr + np.sqrt(np.abs(disc))) / 2), np.abs((tr - np.sqrt(np.abs(disc))) / 2)),
        np.sqrt(np.abs(det)),  # complex conjugate pair: |eig| = sqrt(det)
    )
    return float(rho.max())


def my_t_min(decay, W):
    M = np.abs(W).sum(axis=1) + MAX_INPUT
    return 1.0 - np.tanh(M) ** 2


def my_cf_infnorm(decay, W, t_lo):
    best = -1.0
    for i in range(2):
        j = 1 - i
        for ti in (t_lo[i], 1.0):
            best = max(best, abs(decay[i] + (1 - decay[i]) * ti * W[i, i]) + (1 - decay[i]) * ti * abs(W[i, j]))
    return float(best)


def main():
    t0 = time.time()
    EMP_N, HEAVY_N = 6000, 60000
    genes, n_grid = build_population(3000, 0)
    print(f"[fast] population grid={n_grid} random={len(genes)-n_grid} total={len(genes)}", flush=True)

    c1 = {d: {"n_admit": 0, "worst_emp_inf": 0.0, "min_cf_minus_emp": 9.0, "false_admits": []} for d in ("free01", "tmin1")}
    c2 = {d: {"n": 0, "z3_rejects_all": True, "min_emp": 9.9, "max_emp": 0.0, "examples": [], "by_off09": 0, "by_off15": 0} for d in ("free01", "tmin1")}
    c3 = {d: {"n_false_reject_rho_lt1": 0, "gaps": [], "n_empinf_lt1": 0, "examples": []} for d in ("free01", "tmin1")}
    disagree = {"free01": 0, "tmin1": 0, "examples": []}
    overapprox_viol = []
    pop = {"emp_expansive": 0, "scalar_admit": {"free01": 0, "tmin1": 0}, "z3_admit": {"free01": 0, "tmin1": 0}, "rho_lt1": 0}

    for gi, (decay, W) in enumerate(genes):
        decay = np.clip(decay, 0, 1); W = np.clip(W, -2, 2)
        g = CoupledGene.make(decay=decay, W=W)
        emp_inf, sa, xa = emp_infnorm_fast(decay, W, n_samples=EMP_N, seed=777)
        emp_rho_sup = emp_rho_min_fast(decay, W, n_samples=EMP_N, seed=777)
        if emp_inf > 1.0 + TOL:
            pop["emp_expansive"] += 1
        if emp_rho_sup < 1.0 - TOL:
            pop["rho_lt1"] += 1
        cf = {"free01": my_cf_infnorm(decay, W, np.zeros(2)), "tmin1": my_cf_infnorm(decay, W, my_t_min(decay, W))}
        if cf["free01"] + 1e-9 < emp_inf:
            overapprox_viol.append({"index": gi, "decay": decay.tolist(), "W": W.tolist(), "cf": cf["free01"], "emp": emp_inf})

        for dom in ("free01", "tmin1"):
            z3r = certify_infnorm_contraction(g, t_domain=dom)
            scr = scalar_diagonal_admit(g, t_domain=dom)
            if scr.admit: pop["scalar_admit"][dom] += 1
            if z3r.certified is True: pop["z3_admit"][dom] += 1

            if z3r.certified is True:
                c1[dom]["n_admit"] += 1
                c1[dom]["worst_emp_inf"] = max(c1[dom]["worst_emp_inf"], emp_inf)
                c1[dom]["min_cf_minus_emp"] = min(c1[dom]["min_cf_minus_emp"], cf[dom] - emp_inf)
                if emp_inf > 1.0 + TOL:
                    c1[dom]["false_admits"].append({"index": gi, "decay": decay.tolist(), "W": W.tolist(), "emp_inf": emp_inf, "cf": cf[dom]})

            if scr.admit and emp_inf > 1.0 + TOL:
                c2[dom]["n"] += 1
                c2[dom]["min_emp"] = min(c2[dom]["min_emp"], emp_inf)
                c2[dom]["max_emp"] = max(c2[dom]["max_emp"], emp_inf)
                if gi < n_grid:
                    woff = abs(W[0, 1])
                    if abs(woff - 0.9) < 1e-9: c2[dom]["by_off09"] += 1
                    elif abs(woff - 1.5) < 1e-9: c2[dom]["by_off15"] += 1
                if z3r.certified is not False:
                    c2[dom]["z3_rejects_all"] = False
                if len(c2[dom]["examples"]) < 12:
                    c2[dom]["examples"].append({"index": gi, "decay": decay.tolist(), "W": W.tolist(), "scalar_L": scr.scalar_L, "emp_inf": emp_inf, "z3_certified": z3r.certified})

            if z3r.certified is False and emp_rho_sup < 1.0 - TOL:
                c3[dom]["n_false_reject_rho_lt1"] += 1
                c3[dom]["gaps"].append(emp_inf - emp_rho_sup)
                if emp_inf < 1.0 - TOL:
                    c3[dom]["n_empinf_lt1"] += 1
                if len(c3[dom]["examples"]) < 10:
                    c3[dom]["examples"].append({"index": gi, "decay": decay.tolist(), "W": W.tolist(), "emp_inf": emp_inf, "emp_rho": emp_rho_sup})

            cf_admit = cf[dom] < 1.0
            if cf_admit != (z3r.certified is True):
                disagree[dom] += 1
                if len(disagree["examples"]) < 12:
                    disagree["examples"].append({"index": gi, "dom": dom, "decay": decay.tolist(), "W": W.tolist(), "cf": cf[dom], "cf_admit": cf_admit, "z3": z3r.certified, "status": z3r.solver_status})

        if (gi + 1) % 1000 == 0:
            print(f"  ...{gi+1}/{len(genes)} ({time.time()-t0:.1f}s)", flush=True)

    # heavy reconfirm C1 false admits + top C2 examples (fresh seed 999)
    heavy = {}
    for dom in ("free01", "tmin1"):
        hc1 = []
        for r in c1[dom]["false_admits"][:25]:
            hi, _, _ = emp_infnorm_fast(np.array(r["decay"]), np.array(r["W"]), n_samples=HEAVY_N, seed=999)
            hc1.append({"index": r["index"], "emp_inf_heavy": hi})
        hc2 = 0
        for r in c2[dom]["examples"]:
            hi, _, _ = emp_infnorm_fast(np.array(r["decay"]), np.array(r["W"]), n_samples=HEAVY_N, seed=999)
            if hi > 1.0 + TOL: hc2 += 1
        heavy[dom] = {"c1_false_admit_heavy_confirmed": sum(1 for c in hc1 if c["emp_inf_heavy"] > 1.0 + TOL),
                      "c1_samples": hc1[:8], "c2_examples_still_expansive_heavy": hc2}

    out = {
        "meta": {"emp_seed": 777, "emp_n": EMP_N, "heavy_seed": 999, "heavy_n": HEAVY_N, "tol": TOL,
                 "n_genes": len(genes), "n_grid": n_grid, "wall_seconds": round(time.time() - t0, 1), "python": sys.version.split()[0]},
        "population": pop,
        "R1_C1_soundness": {dom: {"n_admit": c1[dom]["n_admit"], "n_false_admits": len(c1[dom]["false_admits"]),
                                  "worst_admitted_emp_inf": c1[dom]["worst_emp_inf"], "min_cf_minus_emp": c1[dom]["min_cf_minus_emp"],
                                  "false_admit_examples": c1[dom]["false_admits"][:10]} for dom in ("free01", "tmin1")},
        "R2_C2_value_over_scalar": {dom: {"n_scalar_admit_expansive": c2[dom]["n"], "z3_rejects_all": c2[dom]["z3_rejects_all"],
                                          "emp_inf_min": (c2[dom]["min_emp"] if c2[dom]["n"] else None), "emp_inf_max": (c2[dom]["max_emp"] if c2[dom]["n"] else None),
                                          "grid_drivers_off09": c2[dom]["by_off09"], "grid_drivers_off15": c2[dom]["by_off15"],
                                          "examples": c2[dom]["examples"]} for dom in ("free01", "tmin1")},
        "R3_overclaim_z3_vs_closedform_disagree": {"free01": disagree["free01"], "tmin1": disagree["tmin1"], "examples": disagree["examples"]},
        "R4_overapprox_violations_cf_lt_emp": {"n": len(overapprox_viol), "examples": overapprox_viol[:10]},
        "C3_conservativeness": {dom: {"n_false_reject_rho_lt1": c3[dom]["n_false_reject_rho_lt1"], "n_of_those_empinf_lt1": c3[dom]["n_empinf_lt1"],
                                      "median_infnorm_minus_rho_gap": (float(np.median(c3[dom]["gaps"])) if c3[dom]["gaps"] else None),
                                      "max_infnorm_minus_rho_gap": (float(np.max(c3[dom]["gaps"])) if c3[dom]["gaps"] else None),
                                      "examples": c3[dom]["examples"]} for dom in ("free01", "tmin1")},
        "heavy_confirm": heavy,
    }
    (_HERE / "redteam_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n[fast] wrote redteam_results.json ({out['meta']['wall_seconds']}s)")
    print("=== SUMMARY ===")
    print("population:", json.dumps(pop))
    for dom in ("free01", "tmin1"):
        r1 = out["R1_C1_soundness"][dom]; r2 = out["R2_C2_value_over_scalar"][dom]
        print(f"[{dom}] C1 z3_admit={r1['n_admit']} false_admits={r1['n_false_admits']} worst_emp_inf={r1['worst_admitted_emp_inf']:.6f} min(cf-emp)={r1['min_cf_minus_emp']:.2e}")
        print(f"[{dom}] C2 scalar_admit_expansive={r2['n_scalar_admit_expansive']} z3_rejects_all={r2['z3_rejects_all']} emp_inf=[{r2['emp_inf_min']},{r2['emp_inf_max']}] grid_off09={r2['grid_drivers_off09']} grid_off15={r2['grid_drivers_off15']}")
    print(f"R3 z3_vs_closedform disagree: free01={disagree['free01']} tmin1={disagree['tmin1']}")
    print(f"R4 overapprox violations: {len(overapprox_viol)}")
    print(f"heavy: {json.dumps(heavy)}")
    return out


if __name__ == "__main__":
    main()
