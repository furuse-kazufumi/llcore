# SPDX-License-Identifier: Apache-2.0
"""Track D runner — compare ∞-norm vs 2-norm-vertex vs SDP-Lyapunov contraction certifiers
over the SAME 3270-gene population as Track C; compute pre-registered gates D1/D2/D3/D4.

Run: ``py -3.11 research/spectral_lyapunov_contraction/exp_d_runner.py``

Population, map, Jacobian, and empirical oracles are REUSED verbatim from Track C
(research/coupled_z3_contraction/) so genes/indices match bit-for-bit:
  - build_population(3000, 0)  -> 270 grid + 3000 random
  - emp_infnorm_fast (seed 777, n=6000 + structured corners)  -> empirical ||J||_inf sup
  - emp_rho_min_fast (seed 777)                                -> empirical rho(J) sup
  - infnorm_over_box_freeT                                     -> closed-form ∞-norm box sup (baseline certifier)

Track D adds:
  - certify_2norm_contraction (numpy SVD at 4 box vertices, sound)
  - certify_common_lyapunov   (cvxpy common-P vertex-LMI SDP; graceful if cvxpy absent)
  - a dense empirical ||J||_2 sup on the SAME (s,x) sample (for D1 soundness)

Determinism: numpy.random.default_rng(seed) everywhere; seeds reported in JSON + verdict.
Reports ONLY observed numbers. Negative outcomes are valid.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TRACK_C = (_HERE.parent / "coupled_z3_contraction").resolve()
for p in (str(_HERE), str(_TRACK_C)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Track C reuse (population + map + empirical oracles + baseline ∞-norm certifier).
from coupled_map import CoupledGene, infnorm_over_box_freeT, t_min_per_coord  # noqa: E402
from redteam_fast import (  # noqa: E402
    _STRUCT_S, _STRUCT_X, MAX_INPUT, build_population, emp_infnorm_fast, emp_rho_min_fast,
)

# Track D certifiers.
from two_norm_vertex_certifier import (  # noqa: E402
    certify_2norm_contraction, jacobian_at_t, sigma_max, soundness_selfcheck,
)
from lyapunov_sdp_certifier import (  # noqa: E402
    CVXPY_AVAILABLE, availability_report, certify_common_lyapunov,
)

# AUDIT (2026-06-03): the original Track D run used cvxpy's SCS default, which under-certifies the
# SDP near the feasibility boundary (false negatives). Pin CLARABEL to re-measure D3 (SDP vs 2-norm)
# honestly. See memory feedback_cvxpy_pin_accurate_solver.
try:
    import cvxpy as _cp  # noqa: E402
    _SOLVER = "CLARABEL" if CVXPY_AVAILABLE and "CLARABEL" in _cp.installed_solvers() else None
except Exception:  # pragma: no cover
    _SOLVER = None

TOL = 1e-9
SOUND_TOL = 1e-6          # from-below SVD float slack for the D1 soundness gate
EMP_SEED = 777            # SAME as Track C red-team
EMP_N = 6000
SELFCHECK_SEED = 0


def emp_2norm_sup_fast(decay: np.ndarray, W: np.ndarray, *, n_samples: int, seed: int) -> float:
    """Dense empirical sup of ||J||_2 (largest singular value) over the (s,x) box.

    Uses the SAME structured corners + uniform sample style as Track C's emp_infnorm_fast so the
    soundness comparison is apples-to-apples. Vectorized batched SVD over the sample.
    """
    rng = np.random.default_rng(seed)
    S = np.vstack([rng.uniform(-1, 1, (n_samples, 2)), _STRUCT_S])
    X = np.vstack([rng.uniform(-MAX_INPUT, MAX_INPUT, (n_samples, 2)), _STRUCT_X])
    pre = S @ W.T + X
    t = 1.0 - np.tanh(pre) ** 2                 # (N,2)
    a = (1.0 - decay)[None, :] * t              # (N,2)
    J = a[:, :, None] * W[None, :, :]           # (N,2,2)
    J[:, 0, 0] += decay[0]
    J[:, 1, 1] += decay[1]
    sv = np.linalg.svd(J, compute_uv=False)     # (N,2) singular values, descending
    return float(sv[:, 0].max())


def baseline_infnorm_admit(gene: CoupledGene, domain: str) -> tuple[bool, float]:
    """Track C's sound ∞-norm certifier (closed-form endpoint enumeration). Returns (admit, sup)."""
    g = gene.clipped()
    t_lo = None if domain == "free01" else t_min_per_coord(g)
    sup = infnorm_over_box_freeT(g, t_lo=t_lo)
    return bool(sup < 1.0), float(sup)


def emp_pnorm_sup_fast(decay: np.ndarray, W: np.ndarray, P: np.ndarray, *,
                       n_samples: int, seed: int) -> float:
    """Dense empirical sup over the (s,x) box of the P-weighted contraction gain of J.

    A common-quadratic-Lyapunov certificate (SDP) does NOT bound the identity 2-norm; it bounds the
    P-WEIGHTED norm. The correct soundness oracle for that certificate is therefore the gain in the
    P-metric: gain(J) = ||L J L^{-1}||_2 where P = L^T L (Cholesky), i.e. the largest singular value
    of the J expressed in the coordinates that make P the identity. gain < 1 over the box <=>
    contraction in ||.||_P. We report the empirical sup of that gain (from below).
    """
    P = np.asarray(P, dtype=np.float64)
    P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P)            # P = L L^T
    Linv = np.linalg.inv(L)
    rng = np.random.default_rng(seed)
    S = np.vstack([rng.uniform(-1, 1, (n_samples, 2)), _STRUCT_S])
    X = np.vstack([rng.uniform(-MAX_INPUT, MAX_INPUT, (n_samples, 2)), _STRUCT_X])
    pre = S @ W.T + X
    t = 1.0 - np.tanh(pre) ** 2
    a = (1.0 - decay)[None, :] * t
    J = a[:, :, None] * W[None, :, :]
    J[:, 0, 0] += decay[0]
    J[:, 1, 1] += decay[1]
    # P-metric gain: ||L^T J L^{-T}||_2  (since ||z||_P = ||L^T z||_2). Use M = L^T J L^{-T}.
    Lt = L.T
    Ltinv = Linv.T
    M = np.einsum("ij,njk,kl->nil", Lt, J, Ltinv)
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv[:, 0].max())


def main():
    t0 = time.time()
    genes_raw, n_grid = build_population(3000, 0)
    n_total = len(genes_raw)
    avail = availability_report()
    print(f"[track D] population grid={n_grid} random={n_total - n_grid} total={n_total}", flush=True)
    print(f"[track D] cvxpy: {avail}", flush=True)

    records = []
    selfcheck_max_excess = -np.inf  # max over sampled genes of (interior - vertex_sup); should be <= ~0
    for gi, (decay, W) in enumerate(genes_raw):
        decay = np.clip(decay, 0.0, 1.0)
        W = np.clip(W, -2.0, 2.0)
        g = CoupledGene.make(decay=decay, W=W)

        # Empirical oracles (Track C, seed 777) + new empirical 2-norm sup (same sample).
        emp_inf, _sa, _xa = emp_infnorm_fast(decay, W, n_samples=EMP_N, seed=EMP_SEED)
        emp_rho = emp_rho_min_fast(decay, W, n_samples=EMP_N, seed=EMP_SEED)
        emp_2 = emp_2norm_sup_fast(decay, W, n_samples=EMP_N, seed=EMP_SEED)

        rec = {
            "index": gi,
            "source": "grid" if gi < n_grid else "random",
            "decay": decay.tolist(),
            "W": W.tolist(),
            "emp_infnorm": emp_inf,
            "emp_2norm": emp_2,
            "emp_spectral_radius": emp_rho,
        }
        for dom in ("free01", "tmin1"):
            inf_admit, inf_sup = baseline_infnorm_admit(g, dom)
            r2 = certify_2norm_contraction(g, t_domain=dom)
            rec[f"inf_{dom}_admit"] = inf_admit
            rec[f"inf_{dom}_sup"] = inf_sup
            rec[f"two_{dom}_admit"] = r2.certified
            rec[f"two_{dom}_sup"] = r2.sup_2norm
            if CVXPY_AVAILABLE:
                rL = certify_common_lyapunov(g, t_domain=dom, solver=_SOLVER)
                rec[f"sdp_{dom}_admit"] = rL.certified
                rec[f"sdp_{dom}_status"] = rL.solver_status
                rec[f"sdp_{dom}_P"] = rL.P  # for the P-norm soundness oracle (None if not certified)
            else:
                rec[f"sdp_{dom}_admit"] = None
                rec[f"sdp_{dom}_status"] = "cvxpy_unavailable"
                rec[f"sdp_{dom}_P"] = None

        # Vertex-soundness self-check on a 1-in-50 subsample (cheap, re-verifies the math live).
        if gi % 50 == 0:
            exc = soundness_selfcheck(g, "tmin1", n_interior=400, seed=SELFCHECK_SEED)
            selfcheck_max_excess = max(selfcheck_max_excess, exc)

        records.append(rec)
        if (gi + 1) % 500 == 0:
            print(f"  ...{gi + 1}/{n_total} ({time.time() - t0:.1f}s)", flush=True)

    # ----- population counts -----
    rho_lt1 = [r for r in records if r["emp_spectral_radius"] < 1.0 - TOL]
    n_rho_lt1 = len(rho_lt1)

    def count_admit(key):
        return sum(1 for r in records if r.get(key) is True)

    population = {
        "n_genes": n_total,
        "n_grid": n_grid,
        "emp_rho_lt1": n_rho_lt1,
        "inf_free01_admit": count_admit("inf_free01_admit"),
        "inf_tmin1_admit": count_admit("inf_tmin1_admit"),
        "two_free01_admit": count_admit("two_free01_admit"),
        "two_tmin1_admit": count_admit("two_tmin1_admit"),
        "sdp_free01_admit": (count_admit("sdp_free01_admit") if CVXPY_AVAILABLE else None),
        "sdp_tmin1_admit": (count_admit("sdp_tmin1_admit") if CVXPY_AVAILABLE else None),
    }

    # ===== D1 soundness: every admit is empirically non-expansive in the RIGHT metric =====
    # IMPORTANT (honest correction, frozen in the verdict): the soundness ORACLE must match what
    # each certifier actually proves.
    #   * 2-norm-vertex certifies ||J||_2 < 1  => oracle: empirical ||J||_2 <= 1 AND rho <= 1.
    #   * SDP-Lyapunov certifies contraction in the P-WEIGHTED norm (=> rho < 1), and does NOT bound
    #     the identity ||J||_2. So its correct oracle is empirical rho <= 1 AND empirical P-norm
    #     gain <= 1 (computed with the certified P). Using ||J||_2 <= 1 here would be the WRONG test
    #     (it would flag genuinely-Lyapunov-contractive genes whose identity 2-norm exceeds 1).
    def d1_two(dom: str) -> dict:
        key = f"two_{dom}_admit"
        admitted = [r for r in records if r.get(key) is True]
        false_admits = [
            r for r in admitted
            if (r["emp_2norm"] > 1.0 + SOUND_TOL) or (r["emp_spectral_radius"] > 1.0 + SOUND_TOL)
        ]
        return {
            "certifier": "two_norm_vertex", "domain": dom, "oracle": "emp_2norm<=1 AND emp_rho<=1",
            "n_admitted": len(admitted),
            "n_false_admits": len(false_admits),
            "worst_admitted_emp_2norm": max((r["emp_2norm"] for r in admitted), default=0.0),
            "worst_admitted_emp_rho": max((r["emp_spectral_radius"] for r in admitted), default=0.0),
            "false_admit_examples": [
                {"index": r["index"], "decay": r["decay"], "W": r["W"],
                 "emp_2norm": r["emp_2norm"], "emp_rho": r["emp_spectral_radius"]}
                for r in false_admits[:10]
            ],
            "passed": len(false_admits) == 0,
        }

    def d1_sdp(dom: str) -> dict:
        key = f"sdp_{dom}_admit"
        admitted = [r for r in records if r.get(key) is True]
        false_admits = []
        worst_pnorm = 0.0
        for r in admitted:
            rho_bad = r["emp_spectral_radius"] > 1.0 + SOUND_TOL
            P = r.get(f"sdp_{dom}_P")
            pnorm_gain = None
            pnorm_bad = False
            if P is not None:
                pnorm_gain = emp_pnorm_sup_fast(
                    np.array(r["decay"]), np.array(r["W"]), np.array(P),
                    n_samples=EMP_N, seed=EMP_SEED)
                worst_pnorm = max(worst_pnorm, pnorm_gain)
                pnorm_bad = pnorm_gain > 1.0 + SOUND_TOL
            if rho_bad or pnorm_bad:
                false_admits.append({
                    "index": r["index"], "decay": r["decay"], "W": r["W"],
                    "emp_2norm": r["emp_2norm"], "emp_rho": r["emp_spectral_radius"],
                    "emp_pnorm_gain": pnorm_gain,
                })
        return {
            "certifier": "sdp_lyapunov", "domain": dom,
            "oracle": "emp_rho<=1 AND emp_P_norm_gain<=1 (P-weighted; identity ||J||_2 NOT required)",
            "n_admitted": len(admitted),
            "n_false_admits": len(false_admits),
            "worst_admitted_emp_rho": max((r["emp_spectral_radius"] for r in admitted), default=0.0),
            "worst_admitted_emp_pnorm_gain": worst_pnorm,
            "worst_admitted_emp_2norm_for_reference": max((r["emp_2norm"] for r in admitted), default=0.0),
            "false_admit_examples": false_admits[:10],
            "passed": len(false_admits) == 0,
        }

    d1 = {
        "two_free01": d1_two("free01"),
        "two_tmin1": d1_two("tmin1"),
    }
    if CVXPY_AVAILABLE:
        d1["sdp_free01"] = d1_sdp("free01")
        d1["sdp_tmin1"] = d1_sdp("tmin1")
    d1["vertex_soundness_selfcheck_max_excess"] = float(selfcheck_max_excess)
    d1["passed"] = all(v["passed"] for k, v in d1.items() if isinstance(v, dict))

    # ===== D2 tightness gain of 2-norm-vertex over ∞-norm =====
    def d2_for(dom: str) -> dict:
        # all genes 2-norm admits but ∞-norm rejects
        gain_all = [r for r in records if r.get(f"two_{dom}_admit") is True and r.get(f"inf_{dom}_admit") is False]
        # restricted to rho<1 (subset of Track C's 850 over-rejects)
        gain_rho = [r for r in gain_all if r["emp_spectral_radius"] < 1.0 - TOL]
        # the 850-analogue: ∞-norm rejects AND rho<1
        inf_rejects_rho_lt1 = [r for r in records if r.get(f"inf_{dom}_admit") is False and r["emp_spectral_radius"] < 1.0 - TOL]
        n_850 = len(inf_rejects_rho_lt1)
        return {
            "domain": dom,
            "n_two_admit_inf_reject_all": len(gain_all),
            "n_two_admit_inf_reject_rho_lt1": len(gain_rho),
            "n_inf_reject_rho_lt1_total": n_850,
            "fraction_of_850": (len(gain_rho) / n_850 if n_850 else None),
            "examples": [
                {"index": r["index"], "source": r["source"], "decay": r["decay"], "W": r["W"],
                 "inf_sup": r[f"inf_{dom}_sup"], "two_sup": r[f"two_{dom}_sup"],
                 "emp_2norm": r["emp_2norm"], "emp_rho": r["emp_spectral_radius"]}
                for r in sorted(gain_rho, key=lambda r: r["emp_spectral_radius"])[:15]
            ],
        }

    d2 = {"free01": d2_for("free01"), "tmin1": d2_for("tmin1")}

    # ===== D3 does the SOLVER (SDP) beat 2-norm-vertex =====
    if CVXPY_AVAILABLE:
        def d3_for(dom: str) -> dict:
            sdp_beats_two = [r for r in records if r.get(f"sdp_{dom}_admit") is True and r.get(f"two_{dom}_admit") is False]
            sdp_beats_two_rho = [r for r in sdp_beats_two if r["emp_spectral_radius"] < 1.0 - TOL]
            inf_rejects_rho_lt1 = sum(1 for r in records if r.get(f"inf_{dom}_admit") is False and r["emp_spectral_radius"] < 1.0 - TOL)
            # also: does 2-norm ever beat SDP? (would indicate solver tol issues) and full agreement
            two_beats_sdp = [r for r in records if r.get(f"two_{dom}_admit") is True and r.get(f"sdp_{dom}_admit") is False]
            return {
                "domain": dom,
                "n_sdp_admit_two_reject_all": len(sdp_beats_two),
                "n_sdp_admit_two_reject_rho_lt1": len(sdp_beats_two_rho),
                "fraction_of_850": (len(sdp_beats_two_rho) / inf_rejects_rho_lt1 if inf_rejects_rho_lt1 else None),
                "n_two_admit_sdp_reject": len(two_beats_sdp),
                "solver_earns_keep": len(sdp_beats_two_rho) > 0,
                "examples": [
                    {"index": r["index"], "source": r["source"], "decay": r["decay"], "W": r["W"],
                     "two_sup": r[f"two_{dom}_sup"], "emp_2norm": r["emp_2norm"], "emp_rho": r["emp_spectral_radius"],
                     "sdp_status": r[f"sdp_{dom}_status"]}
                    for r in sorted(sdp_beats_two, key=lambda r: r["emp_spectral_radius"])[:15]
                ],
                "two_beats_sdp_examples": [
                    {"index": r["index"], "two_sup": r[f"two_{dom}_sup"], "sdp_status": r[f"sdp_{dom}_status"]}
                    for r in two_beats_sdp[:10]
                ],
            }
        d3 = {"free01": d3_for("free01"), "tmin1": d3_for("tmin1"), "ran": True}
    else:
        d3 = {"ran": False, "note": "cvxpy unavailable; SDP not run. 2-norm-vertex (D2) is the partial answer."}

    # ===== D4 honest residual: rho<1 genes NO certifier admits =====
    def d4_for(dom: str) -> dict:
        residual = []
        for r in rho_lt1:
            inf_a = r.get(f"inf_{dom}_admit") is True
            two_a = r.get(f"two_{dom}_admit") is True
            sdp_a = (r.get(f"sdp_{dom}_admit") is True) if CVXPY_AVAILABLE else False
            if not (inf_a or two_a or sdp_a):
                residual.append(r)
        return {
            "domain": dom,
            "n_rho_lt1": n_rho_lt1,
            "n_residual_no_certifier": len(residual),
            "fraction_of_rho_lt1": (len(residual) / n_rho_lt1 if n_rho_lt1 else None),
            "examples": [
                {"index": r["index"], "source": r["source"], "decay": r["decay"], "W": r["W"],
                 "emp_2norm": r["emp_2norm"], "emp_rho": r["emp_spectral_radius"],
                 "inf_sup": r[f"inf_{dom}_sup"], "two_sup": r[f"two_{dom}_sup"]}
                for r in sorted(residual, key=lambda r: r["emp_2norm"])[:15]
            ],
        }

    d4 = {"free01": d4_for("free01"), "tmin1": d4_for("tmin1")}

    out = {
        "meta": {
            "emp_seed": EMP_SEED, "emp_n": EMP_N, "tol": TOL, "sound_tol": SOUND_TOL,
            "selfcheck_seed": SELFCHECK_SEED,
            "n_genes": n_total, "n_grid": n_grid,
            "cvxpy": avail,
            "wall_seconds": round(time.time() - t0, 1),
            "python": sys.version.split()[0],
        },
        "population": population,
        "D1_soundness": d1,
        "D2_twonorm_gain_over_infnorm": d2,
        "D3_sdp_vs_twonorm": d3,
        "D4_honest_residual": d4,
    }

    out_path = _HERE / "exp_d_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[track D] wrote {out_path} ({out['meta']['wall_seconds']}s)", flush=True)

    # ----- console summary -----
    print("\n=== SUMMARY ===")
    print("population:", json.dumps(population))
    print(f"D1 vertex-soundness selfcheck max excess = {d1['vertex_soundness_selfcheck_max_excess']:.3e} (<=~0 sound)")
    for dom in ("free01", "tmin1"):
        print(f"D1 two_{dom}: admitted={d1[f'two_{dom}']['n_admitted']} false_admits={d1[f'two_{dom}']['n_false_admits']} "
              f"worst_emp_2={d1[f'two_{dom}']['worst_admitted_emp_2norm']:.6f} PASS={d1[f'two_{dom}']['passed']}")
        if CVXPY_AVAILABLE:
            sd = d1[f'sdp_{dom}']
            print(f"D1 sdp_{dom}: admitted={sd['n_admitted']} false_admits={sd['n_false_admits']} "
                  f"worst_emp_rho={sd['worst_admitted_emp_rho']:.6f} worst_pnorm_gain={sd['worst_admitted_emp_pnorm_gain']:.6f} PASS={sd['passed']}")
        dd = d2[dom]
        print(f"D2 {dom}: two_admit&inf_reject(rho<1)={dd['n_two_admit_inf_reject_rho_lt1']}/{dd['n_inf_reject_rho_lt1_total']} "
              f"(frac {dd['fraction_of_850']}) all={dd['n_two_admit_inf_reject_all']}")
        if CVXPY_AVAILABLE:
            d3d = d3[dom]
            print(f"D3 {dom}: sdp_admit&two_reject(rho<1)={d3d['n_sdp_admit_two_reject_rho_lt1']} "
                  f"(all={d3d['n_sdp_admit_two_reject_all']}) two_beats_sdp={d3d['n_two_admit_sdp_reject']} "
                  f"solver_earns_keep={d3d['solver_earns_keep']}")
        d4d = d4[dom]
        print(f"D4 {dom}: residual(no certifier, rho<1)={d4d['n_residual_no_certifier']}/{d4d['n_rho_lt1']} "
              f"(frac {d4d['fraction_of_rho_lt1']})")
    return out


if __name__ == "__main__":
    main()
