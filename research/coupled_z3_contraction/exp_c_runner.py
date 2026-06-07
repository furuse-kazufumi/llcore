# SPDX-License-Identifier: Apache-2.0
"""Track C runner — sweep coupled genes through Z3 inf-norm certifier, scalar diagonal
heuristic, closed-form inf-norm, and the empirical Jacobian oracle; compute pre-registered
gates C1/C2/C3; write exp_c_results.json.

Run: ``py -3.11 research/coupled_z3_contraction/exp_c_runner.py``  (from repo root or this dir).

Determinism: numpy.random.default_rng(SEED) everywhere; Z3 exact. Reproduces identical JSON.
"""
from __future__ import annotations

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from coupled_map import (  # noqa: E402
    CoupledGene,
    empirical_box_norms,
    infnorm_over_box_freeT,
    iterate_state_growth,
    t_min_per_coord,
)
from scalar_heuristic import scalar_diagonal_admit  # noqa: E402
from z3_infnorm_certifier import certify_infnorm_contraction  # noqa: E402

SEED = 0
N_RANDOM = 3000
EMP_SAMPLES_SWEEP = 4000        # per-gene empirical sample during the sweep (from-below sup)
EMP_SAMPLES_CONFIRM = 40000     # heavier confirm pass for genes that drive a gate
TOL = 1e-9


def build_grid() -> list[CoupledGene]:
    """FROZEN grid (referenced by PREREGISTRATION.md).

    decay in {0.2,0.5,0.8}^2 ; W from diagonal in {-0.5,0.5,0.9} and off-diagonal in
    {-1.5,-0.9,0.0,0.9,1.5}, applied both symmetrically (W01=W10) and asymmetrically
    (W10 = -W01). Diagonal entries shared across both rows for compactness.
    """
    decays = [0.2, 0.5, 0.8]
    wdiag = [-0.5, 0.5, 0.9]
    woff = [-1.5, -0.9, 0.0, 0.9, 1.5]
    genes = []
    for d0, d1 in itertools.product(decays, decays):
        for wd in wdiag:
            for wo in woff:
                # symmetric coupling
                W_sym = [[wd, wo], [wo, wd]]
                # asymmetric coupling
                W_asym = [[wd, wo], [-wo, wd]]
                for W in (W_sym, W_asym):
                    genes.append(CoupledGene.make(decay=[d0, d1], W=W))
    return genes


def build_random(n: int, seed: int) -> list[CoupledGene]:
    rng = np.random.default_rng(seed)
    genes = []
    for _ in range(n):
        decay = rng.uniform(0.0, 1.0, size=2)
        W = rng.uniform(-2.0, 2.0, size=(2, 2))
        genes.append(CoupledGene.make(decay=decay, W=W))
    return genes


def eval_gene(gene: CoupledGene, emp_samples: int, emp_seed: int) -> dict:
    """Run all three checks + empirical oracle on a single gene."""
    g = gene.clipped()
    rec: dict = {
        "decay": [float(v) for v in g.decay],
        "W": [[float(g.W[i, j]) for j in range(2)] for i in range(2)],
        "t_min": [float(v) for v in t_min_per_coord(g)],
    }

    # Z3 inf-norm certifier (both domains)
    z3_free = certify_infnorm_contraction(g, t_domain="free01")
    z3_tmin = certify_infnorm_contraction(g, t_domain="tmin1")
    rec["z3_free01_certified"] = z3_free.certified
    rec["z3_free01_status"] = z3_free.solver_status
    rec["z3_tmin1_certified"] = z3_tmin.certified
    rec["z3_tmin1_status"] = z3_tmin.solver_status

    # Closed-form inf-norm sups (free + tmin)
    cf_free = infnorm_over_box_freeT(g, t_lo=None)
    cf_tmin = infnorm_over_box_freeT(g, t_lo=t_min_per_coord(g))
    rec["closedform_infnorm_free01"] = cf_free
    rec["closedform_infnorm_tmin1"] = cf_tmin
    rec["closedform_admit_free01"] = bool(cf_free < 1.0 - 0.0)  # strict < 1
    rec["closedform_admit_tmin1"] = bool(cf_tmin < 1.0 - 0.0)

    # Scalar diagonal heuristic (both domains)
    sc_free = scalar_diagonal_admit(g, t_domain="free01")
    sc_tmin = scalar_diagonal_admit(g, t_domain="tmin1")
    rec["scalar_free01_admit"] = sc_free.admit
    rec["scalar_free01_L"] = sc_free.scalar_L
    rec["scalar_tmin1_admit"] = sc_tmin.admit
    rec["scalar_tmin1_L"] = sc_tmin.scalar_L

    # Empirical oracle
    emp = empirical_box_norms(g, n_samples=emp_samples, seed=emp_seed)
    rec["emp_infnorm"] = emp["emp_infnorm"]
    rec["emp_2norm"] = emp["emp_2norm"]
    rec["emp_spectral_radius"] = emp["emp_spectral_radius"]
    rec["emp_n_points"] = emp["n_points"]
    return rec


def main():
    t0 = time.time()
    grid = build_grid()
    rnd = build_random(N_RANDOM, SEED)
    genes = grid + rnd
    n_grid = len(grid)
    n_rand = len(rnd)
    print(f"[track C] genes: grid={n_grid} random={n_rand} total={len(genes)}", flush=True)

    records = []
    for idx, g in enumerate(genes):
        rec = eval_gene(g, EMP_SAMPLES_SWEEP, emp_seed=SEED)
        rec["index"] = idx
        rec["source"] = "grid" if idx < n_grid else "random"
        records.append(rec)
        if (idx + 1) % 500 == 0:
            print(f"  ...{idx + 1}/{len(genes)} ({time.time() - t0:.1f}s)", flush=True)

    # ---- Gate C1: Z3 soundness (0 false admits) ----
    def c1_for_domain(domain: str) -> dict:
        key_cert = f"z3_{domain}_certified"
        admitted = [r for r in records if r[key_cert] is True]
        false_admits = [r for r in admitted if r["emp_infnorm"] > 1.0 + TOL]
        return {
            "domain": domain,
            "n_admitted": len(admitted),
            "n_false_admits": len(false_admits),
            "worst_admitted_emp_infnorm": max((r["emp_infnorm"] for r in admitted), default=0.0),
            "false_admit_examples": [
                {"index": r["index"], "decay": r["decay"], "W": r["W"], "emp_infnorm": r["emp_infnorm"]}
                for r in false_admits[:10]
            ],
            "passed": len(false_admits) == 0,
        }

    c1_free = c1_for_domain("free01")
    c1_tmin = c1_for_domain("tmin1")

    # Confirm C1 'worst admitted' genes with a heavier empirical pass (anti from-below illusion).
    def reconfirm_worst_admitted(domain: str, top: int = 25) -> dict:
        key_cert = f"z3_{domain}_certified"
        admitted = [r for r in records if r[key_cert] is True]
        admitted_sorted = sorted(admitted, key=lambda r: r["emp_infnorm"], reverse=True)[:top]
        worst = 0.0
        viol = 0
        for r in admitted_sorted:
            g = CoupledGene.make(decay=r["decay"], W=r["W"])
            emp = empirical_box_norms(g, n_samples=EMP_SAMPLES_CONFIRM, seed=SEED)
            worst = max(worst, emp["emp_infnorm"])
            if emp["emp_infnorm"] > 1.0 + TOL:
                viol += 1
        return {"domain": domain, "reconfirmed_top": len(admitted_sorted),
                "worst_emp_infnorm_heavy": worst, "violations_heavy": viol}

    c1_free_heavy = reconfirm_worst_admitted("free01")
    c1_tmin_heavy = reconfirm_worst_admitted("tmin1")

    # ---- Gate C2: scalar admits but map is expansive (Z3 not decorative) ----
    def c2_for_domain(domain: str) -> dict:
        key_scalar = f"scalar_{domain}_admit"
        key_z3 = f"z3_{domain}_certified"
        scalar_admit_expansive = [
            r for r in records if r[key_scalar] is True and r["emp_infnorm"] > 1.0 + TOL
        ]
        # of those, confirm Z3 (coupling-aware) correctly rejects
        z3_rejects = [r for r in scalar_admit_expansive if r[key_z3] is False]
        # also: do scalar and Z3 EVER disagree on the certify decision?
        scalar_vs_z3_disagree = [
            r for r in records if bool(r[key_scalar]) != bool(r[key_z3] is True)
        ]
        # closed-form vs Z3 agreement (honest "is Z3 doing more than closed form?")
        cf_key = f"closedform_admit_{domain}"
        cf_vs_z3_disagree = [
            r for r in records if bool(r[cf_key]) != bool(r[key_z3] is True)
        ]
        return {
            "domain": domain,
            "n_scalar_admit_expansive": len(scalar_admit_expansive),
            "fraction_scalar_admit_expansive": len(scalar_admit_expansive) / len(records),
            "n_of_those_z3_rejects": len(z3_rejects),
            "all_confirmed_expansive_and_z3_rejected": (
                len(z3_rejects) == len(scalar_admit_expansive) and len(scalar_admit_expansive) > 0
            ),
            "n_scalar_vs_z3_decision_disagree": len(scalar_vs_z3_disagree),
            "n_closedform_vs_z3_decision_disagree": len(cf_vs_z3_disagree),
            "examples": [
                {
                    "index": r["index"], "source": r["source"], "decay": r["decay"], "W": r["W"],
                    "scalar_L": r[f"scalar_{domain}_L"], "emp_infnorm": r["emp_infnorm"],
                    "emp_spectral_radius": r["emp_spectral_radius"],
                    "z3_certified": r[key_z3],
                }
                for r in scalar_admit_expansive[:15]
            ],
        }

    c2_free = c2_for_domain("free01")
    c2_tmin = c2_for_domain("tmin1")

    # Reconfirm C2 examples expansiveness with heavy sampling + trajectory separation growth.
    def reconfirm_c2(domain: str, top: int = 25) -> dict:
        key_scalar = f"scalar_{domain}_admit"
        cands = [r for r in records if r[key_scalar] is True and r["emp_infnorm"] > 1.0 + TOL]
        cands = sorted(cands, key=lambda r: r["emp_infnorm"], reverse=True)[:top]
        confirmed = 0
        sep_growth = []
        for r in cands:
            g = CoupledGene.make(decay=r["decay"], W=r["W"])
            emp = empirical_box_norms(g, n_samples=EMP_SAMPLES_CONFIRM, seed=SEED)
            it = iterate_state_growth(g, n_steps=1500, seed=SEED, n_seqs=6)
            if emp["emp_infnorm"] > 1.0 + TOL:
                confirmed += 1
            sep_growth.append(it["max_separation_growth_ratio"])
        return {
            "domain": domain, "reconfirmed_top": len(cands),
            "still_expansive_heavy": confirmed,
            "median_separation_growth_ratio": float(np.median(sep_growth)) if sep_growth else None,
            "max_separation_growth_ratio": float(np.max(sep_growth)) if sep_growth else None,
        }

    c2_free_heavy = reconfirm_c2("free01")
    c2_tmin_heavy = reconfirm_c2("tmin1")

    # ---- Gate C3: conservative false rejects (rho<1 but inf-norm certifier rejects) ----
    def c3_for_domain(domain: str) -> dict:
        key_z3 = f"z3_{domain}_certified"
        # truly locally contractive proxy: empirical spectral radius < 1 over box
        false_rejects = [
            r for r in records
            if r[key_z3] is False and r["emp_spectral_radius"] < 1.0 - TOL
        ]
        gaps = [r["emp_infnorm"] - r["emp_spectral_radius"] for r in false_rejects]
        # how many of those have emp_infnorm itself < 1 (truly contractive even in inf-norm
        # empirically, but Z3 rejected -> pure conservativeness of the free-t over-approx)?
        empinf_lt1 = [r for r in false_rejects if r["emp_infnorm"] < 1.0 - TOL]
        return {
            "domain": domain,
            "n_false_reject_rho_lt1": len(false_rejects),
            "fraction_false_reject_rho_lt1": len(false_rejects) / len(records),
            "n_of_those_emp_infnorm_lt1": len(empinf_lt1),
            "median_infnorm_minus_rho_gap": float(np.median(gaps)) if gaps else None,
            "max_infnorm_minus_rho_gap": float(np.max(gaps)) if gaps else None,
            "examples": [
                {
                    "index": r["index"], "source": r["source"], "decay": r["decay"], "W": r["W"],
                    "emp_infnorm": r["emp_infnorm"], "emp_2norm": r["emp_2norm"],
                    "emp_spectral_radius": r["emp_spectral_radius"],
                }
                for r in sorted(false_rejects, key=lambda r: r["emp_spectral_radius"])[:15]
            ],
        }

    c3_free = c3_for_domain("free01")
    c3_tmin = c3_for_domain("tmin1")

    # ---- Aggregate population stats ----
    n = len(records)
    z3_free_cert = sum(1 for r in records if r["z3_free01_certified"] is True)
    z3_tmin_cert = sum(1 for r in records if r["z3_tmin1_certified"] is True)
    scalar_free_admit = sum(1 for r in records if r["scalar_free01_admit"])
    scalar_tmin_admit = sum(1 for r in records if r["scalar_tmin1_admit"])
    emp_expansive = sum(1 for r in records if r["emp_infnorm"] > 1.0 + TOL)
    rho_lt1 = sum(1 for r in records if r["emp_spectral_radius"] < 1.0 - TOL)

    out = {
        "meta": {
            "seed": SEED,
            "n_genes": n,
            "n_grid": n_grid,
            "n_random": n_rand,
            "emp_samples_sweep": EMP_SAMPLES_SWEEP,
            "emp_samples_confirm": EMP_SAMPLES_CONFIRM,
            "tol": TOL,
            "wall_seconds": round(time.time() - t0, 1),
            "python": sys.version.split()[0],
        },
        "population": {
            "z3_free01_certified": z3_free_cert,
            "z3_tmin1_certified": z3_tmin_cert,
            "scalar_free01_admit": scalar_free_admit,
            "scalar_tmin1_admit": scalar_tmin_admit,
            "empirical_expansive": emp_expansive,
            "empirical_rho_lt1": rho_lt1,
        },
        "C1_soundness": {"free01": c1_free, "tmin1": c1_tmin,
                          "heavy_confirm_free01": c1_free_heavy, "heavy_confirm_tmin1": c1_tmin_heavy},
        "C2_value_over_scalar": {"free01": c2_free, "tmin1": c2_tmin,
                                  "heavy_confirm_free01": c2_free_heavy, "heavy_confirm_tmin1": c2_tmin_heavy},
        "C3_conservativeness": {"free01": c3_free, "tmin1": c3_tmin},
    }

    out_path = _HERE / "exp_c_results.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[track C] wrote {out_path} ({out['meta']['wall_seconds']}s)", flush=True)

    # Console summary
    print("\n=== SUMMARY ===")
    print(f"population: {json.dumps(out['population'])}")
    print(f"C1 free01: admitted={c1_free['n_admitted']} false_admits={c1_free['n_false_admits']} "
          f"worst_emp_inf={c1_free['worst_admitted_emp_infnorm']:.6f} PASS={c1_free['passed']}")
    print(f"C1 tmin1 : admitted={c1_tmin['n_admitted']} false_admits={c1_tmin['n_false_admits']} "
          f"worst_emp_inf={c1_tmin['worst_admitted_emp_infnorm']:.6f} PASS={c1_tmin['passed']}")
    print(f"C1 heavy free01: worst_emp_inf={c1_free_heavy['worst_emp_infnorm_heavy']:.6f} "
          f"violations={c1_free_heavy['violations_heavy']}")
    print(f"C2 free01: scalar_admit_expansive={c2_free['n_scalar_admit_expansive']} "
          f"(frac {c2_free['fraction_scalar_admit_expansive']:.4f}) z3_rejects={c2_free['n_of_those_z3_rejects']} "
          f"all_confirmed={c2_free['all_confirmed_expansive_and_z3_rejected']}")
    print(f"C2 free01: scalar_vs_z3_disagree={c2_free['n_scalar_vs_z3_decision_disagree']} "
          f"closedform_vs_z3_disagree={c2_free['n_closedform_vs_z3_decision_disagree']}")
    print(f"C2 heavy free01: still_expansive={c2_free_heavy['still_expansive_heavy']}/"
          f"{c2_free_heavy['reconfirmed_top']} max_sep_growth={c2_free_heavy['max_separation_growth_ratio']}")
    print(f"C3 free01: false_reject(rho<1)={c3_free['n_false_reject_rho_lt1']} "
          f"(frac {c3_free['fraction_false_reject_rho_lt1']:.4f}) of_those_empinf<1="
          f"{c3_free['n_of_those_emp_infnorm_lt1']} median_gap={c3_free['median_infnorm_minus_rho_gap']}")
    return out


if __name__ == "__main__":
    main()
