# SPDX-License-Identifier: Apache-2.0
"""Pre-registered experiment for Pilot A (RUMP_OR_PREREGISTRATION.md).

Runs the Rump verified-PD + OR-of-{CLARABEL,SCS} gate on sdp-certifiable coupled genes near the
SDP feasibility boundary (the "sdp_only thin-shell", ~1e-7 margin) and records REAL numbers:

  (i)  Rump-certified ⊇ float-eigvalsh-certified, with ZERO float-only false positives
       (P1). "float-only false positive" = a certificate the float `eigvalsh(>0)` recheck accepts
       whose vertex LMIs are NOT genuinely PD at the verified (Rump) level AND which is not
       rescued by the OR (i.e. a genuinely lost-coverage case attributable to Rump strictness on
       a comfortably-PD certificate, not a sound rejection of a non-PD matrix).
  (ii) how many genes CLARABEL certifies (Rump-verified) that SCS calls infeasible (P2).
  Plus OR-gate soundness: every OR-admitted gene has jsr_lb < 1 (solver-independent oracle).

Writes ``rump_or_experiment_results.json``. research/ only; src/ untouched; zero new deps.
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

import cvxpy as cp  # noqa: E402
from coupled_map import CoupledGene  # noqa: E402
from lyapunov_sdp_certifier import certify_common_lyapunov  # noqa: E402
from rump_pd import (  # noqa: E402
    _vertex_jacobians, or_certifies_lyapunov, rump_verify_certificate, verified_pd,
)
from coupled_components import _inf_certifies, _two_certifies  # noqa: E402

T_DOMAIN = "tmin1"
MARGIN = 1e-7
JSR_TOL = 1e-9


def jsr_lb(vertices, max_len: int = 6) -> float:
    """Gripenberg lower bound on JSR{vertices} (solver-independent product oracle).
    Identical to the audit's oracle: max over length-<=K vertex products of rho(prod)^{1/k}."""
    V = [np.asarray(M, dtype=np.float64) for M in vertices]
    best = 0.0
    for k in range(1, max_len + 1):
        for combo in itertools.product(range(len(V)), repeat=k):
            P = np.eye(V[0].shape[0])
            for i in combo:
                P = V[i] @ P
            best = max(best, float(np.max(np.abs(np.linalg.eigvals(P)))) ** (1.0 / k))
    return best


def _float_recheck(P: np.ndarray, verts, margin: float = MARGIN) -> bool:
    """The arc's ORIGINAL float recheck: P PD and every (P - J^T P J) PD by eigvalsh > 0.
    (This is exactly the test inside certify_common_lyapunov: min_dec > 0 on P - J^T P J, NOT
    subtracting margin — the strict `>> margin*I` is the solver's job. We replicate it faithfully
    so the comparison is apples-to-apples.)"""
    P = 0.5 * (np.asarray(P, dtype=np.float64) + np.asarray(P, dtype=np.float64).T)
    if float(np.min(np.linalg.eigvalsh(P))) <= 0.0:
        return False
    for J in verts:
        M = P - J.T @ P @ J
        M = 0.5 * (M + M.T)
        if float(np.min(np.linalg.eigvalsh(M))) <= 0.0:
            return False
    return True


def _float_recheck_same_matrix(P: np.ndarray, verts, margin: float = MARGIN,
                               floor: float = 1e-9) -> bool:
    """APPLES-TO-APPLES float baseline: the SAME matrices Rump verifies (``P`` and every
    ``P - J^T P J - margin*I``) accepted by the float test ``eigvalsh_min > floor`` (floor=1e-9,
    the arc's near-singular float floor). This is the correct baseline for the coverage invariant
    'Rump-certified superset of float-certified' (I3/P1): both tests judge the IDENTICAL matrices.
    """
    P = 0.5 * (np.asarray(P, dtype=np.float64) + np.asarray(P, dtype=np.float64).T)
    if float(np.min(np.linalg.eigvalsh(P))) <= floor:
        return False
    for J in verts:
        M = P - J.T @ P @ J - margin * np.eye(P.shape[0])
        M = 0.5 * (M + M.T)
        if float(np.min(np.linalg.eigvalsh(M))) <= floor:
            return False
    return True


def _is_sdp_only(gene) -> bool:
    """sdp_only candidate: neither inf-norm nor 2-norm certifies (so the certificate, if any,
    is a genuine SDP one — the thin-shell of interest)."""
    return not (_inf_certifies(gene) or _two_certifies(gene))


def find_thin_shell_genes(n_target: int = 60, seed: int = 777, scan_cap: int = 60000):
    """Deterministically scan for sdp-certifiable genes near the feasibility boundary.

    A gene qualifies if: (a) inf/2-norm REJECT (sdp_only candidate); (b) CLARABEL returns an
    'optimal' P with a SMALL certificate margin (thin shell, min_eig_margin < 1e-4) -- i.e. it
    sits near the SDP feasibility boundary where the SCS artifact lives. Independent of the Rump
    gate (anti-circularity)."""
    rng = np.random.default_rng(seed)
    found = []
    scanned = 0
    while len(found) < n_target and scanned < scan_cap:
        scanned += 1
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if not _is_sdp_only(g):
            continue
        r = certify_common_lyapunov(g, t_domain=T_DOMAIN, margin=MARGIN, solver="CLARABEL")
        if r.certified is True and r.min_eig_margin is not None and r.min_eig_margin < 1e-4:
            found.append(g)
    return found, scanned


def load_committed_residual_genes():
    """The committed near-boundary genes from exp_deg6_residual_genes.json (deg_certified +
    residual_uncert). These are the arc's recorded thin-shell / residual population."""
    path = os.path.join(_HERE, "exp_deg6_residual_genes.json")
    genes = []
    if not os.path.exists(path):
        return genes
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("deg_certified", "residual_uncert"):
        for rec in data.get(key, []):
            g = CoupledGene.make(decay=rec["decay"], W=np.asarray(rec["W"]).reshape(2, 2))
            genes.append((key, g))
    return genes


def analyse_gene(gene):
    """Per-gene record: per-solver certificate, float recheck, Rump recheck, OR verdict, jsr_lb."""
    verts = _vertex_jacobians(gene, t_domain=T_DOMAIN)
    rec = {
        "decay": gene.clipped().decay.tolist(),
        "W": gene.clipped().W.reshape(-1).tolist(),
        "jsr_lb": jsr_lb(verts, max_len=6),
        "solvers": {},
    }
    installed = set(cp.installed_solvers())
    for sv in ("CLARABEL", "SCS"):
        d = {"installed": sv in installed, "status": None,
             "has_P": False, "float_recheck": False, "rump_recheck": False}
        if sv in installed:
            r = certify_common_lyapunov(gene, t_domain=T_DOMAIN, margin=MARGIN, solver=sv)
            d["status"] = r.solver_status
            d["certifier_certified"] = bool(r.certified is True)
            if r.P is not None:
                d["has_P"] = True
                P = np.asarray(r.P, dtype=np.float64)
                d["float_recheck"] = bool(_float_recheck(P, verts))
                d["rump_recheck"] = bool(rump_verify_certificate(P, verts, margin=MARGIN))
        rec["solvers"][sv] = d
    rec["or_gate"] = bool(or_certifies_lyapunov(gene, t_domain=T_DOMAIN, margin=MARGIN))
    # Float-OR: any solver whose P passes the FLOAT recheck.
    rec["float_or_gate"] = any(rec["solvers"][sv]["float_recheck"] for sv in ("CLARABEL", "SCS"))
    return rec


def main(n_thin: int = 60):
    t0 = time.time()
    print(f"cvxpy {cp.__version__} solvers={[s for s in ('SCS','CLARABEL') if s in cp.installed_solvers()]}",
          flush=True)

    thin, scanned = find_thin_shell_genes(n_target=n_thin)
    print(f"thin-shell sdp_only genes found: {len(thin)} (scanned {scanned})", flush=True)
    committed = load_committed_residual_genes()
    print(f"committed residual genes loaded: {len(committed)}", flush=True)

    records = []
    for g in thin:
        r = analyse_gene(g)
        r["source"] = "thin_shell_scan"
        records.append(r)
    for tag, g in committed:
        r = analyse_gene(g)
        r["source"] = f"committed:{tag}"
        records.append(r)

    # ---- aggregate the pre-registered quantities -------------------------- #
    # Per-solver: float-certified set vs Rump-certified set.
    agg = {}
    for sv in ("CLARABEL", "SCS"):
        float_set = {i for i, r in enumerate(records) if r["solvers"][sv]["float_recheck"]}
        rump_set = {i for i, r in enumerate(records) if r["solvers"][sv]["rump_recheck"]}
        # float-only = float accepts but Rump rejects, for THIS solver's certificate.
        float_only = float_set - rump_set
        # Of those, how many are GENUINE coverage losses (the certificate's vertex LMIs are
        # actually PD with a comfortable margin) vs SOUND rejections (a vertex LMI is at/below 0)?
        genuine_losses = []
        for i in float_only:
            verts = _vertex_jacobians(_gene_of(records[i]), t_domain=T_DOMAIN)
            P = _P_of(records[i], sv)
            min_lmi = _min_decrease_lmi(P, verts)  # true min eig of (P - J^T P J - margin*I)
            # comfortable margin threshold: if the decrease LMI is comfortably PD (>=1e-6) yet
            # Rump rejected, THAT is a genuine coverage loss (a false-positive-of-float would be
            # the opposite: float accepts a non-PD => but float uses >0 so it cannot; we record
            # the sound-rejection vs loss split honestly).
            if min_lmi >= 1e-6:
                genuine_losses.append({"index": i, "min_decrease_lmi": min_lmi})
        agg[sv] = {
            "n_float_certified": len(float_set),
            "n_rump_certified": len(rump_set),
            "rump_superset_of_float": rump_set >= float_set,
            "n_float_only": len(float_only),
            "n_genuine_coverage_losses": len(genuine_losses),
            "genuine_coverage_losses": genuine_losses,
        }

    # CLARABEL recovers over SCS: CLARABEL Rump-certifies but SCS reports infeasible (no P).
    clarabel_recovers_over_scs = 0
    scs_recovers_over_clarabel = 0
    for r in records:
        cla = r["solvers"]["CLARABEL"]
        scs = r["solvers"]["SCS"]
        scs_infeasible = (not scs["has_P"]) or (str(scs["status"]).startswith("infeasible"))
        cla_infeasible = (not cla["has_P"]) or (str(cla["status"]).startswith("infeasible"))
        if cla["rump_recheck"] and scs_infeasible:
            clarabel_recovers_over_scs += 1
        if scs["rump_recheck"] and cla_infeasible:
            scs_recovers_over_clarabel += 1

    # OR-gate soundness: every OR-admitted gene must have jsr_lb < 1.
    or_admitted = [r for r in records if r["or_gate"]]
    or_unsound = [r for r in or_admitted if r["jsr_lb"] >= 1.0 - JSR_TOL]

    # Rump-OR ⊇ float-OR (coverage at the gate level), and OR ⊇ each single-solver Rump.
    rump_or_set = {i for i, r in enumerate(records) if r["or_gate"]}
    float_or_set = {i for i, r in enumerate(records) if r["float_or_gate"]}
    single_solver_rump_union = {
        i for i, r in enumerate(records)
        if r["solvers"]["CLARABEL"]["rump_recheck"] or r["solvers"]["SCS"]["rump_recheck"]
    }

    # Global "float-only false positive" count across BOTH solvers, in the spec's sense:
    # certificates float accepts but Rump cannot, that are GENUINE coverage losses (comfortable
    # margin) -- i.e. Rump under-certifying a comfortably-PD matrix. (Sound rejections of
    # at/below-zero LMIs are NOT counted; those are correct.)
    float_only_false_positives = sum(agg[sv]["n_genuine_coverage_losses"] for sv in ("CLARABEL", "SCS"))

    out = {
        "meta": {
            "cvxpy": cp.__version__,
            "t_domain": T_DOMAIN, "margin": MARGIN, "jsr_tol": JSR_TOL,
            "n_thin_shell": len(thin), "n_committed": len(committed),
            "n_records": len(records), "scanned": scanned,
            "elapsed_s": round(time.time() - t0, 1),
        },
        "per_solver": agg,
        # P1: coverage / no float-only false positives.
        "rump_superset_of_float_each_solver": all(agg[sv]["rump_superset_of_float"]
                                                  for sv in ("CLARABEL", "SCS")),
        "float_only_false_positives": float_only_false_positives,
        "rump_or_superset_of_float_or": rump_or_set >= float_or_set,
        "or_superset_of_single_solver_rump": rump_or_set >= single_solver_rump_union,
        "n_float_or_certified": len(float_or_set),
        "n_rump_or_certified": len(rump_or_set),
        # P2: standing artifact-class measure.
        "clarabel_recovers_over_scs": clarabel_recovers_over_scs,
        "scs_recovers_over_clarabel": scs_recovers_over_clarabel,
        # OR-gate soundness.
        "n_or_admitted": len(or_admitted),
        "n_or_unsound_jsr_ge_1": len(or_unsound),
        "or_gate_sound": len(or_unsound) == 0,
        "max_or_admitted_jsr_lb": max((r["jsr_lb"] for r in or_admitted), default=None),
        "records": records,
    }

    op = os.path.join(_HERE, "rump_or_experiment_results.json")
    with open(op, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    summary = {k: v for k, v in out.items() if k not in ("records", "per_solver")}
    summary["per_solver_summary"] = {
        sv: {k: agg[sv][k] for k in ("n_float_certified", "n_rump_certified",
                                     "rump_superset_of_float", "n_float_only",
                                     "n_genuine_coverage_losses")}
        for sv in ("CLARABEL", "SCS")
    }
    print(json.dumps(summary, indent=2), flush=True)
    print(f"\nwrote {op}", flush=True)
    return out


def _gene_of(rec):
    return CoupledGene.make(decay=rec["decay"], W=np.asarray(rec["W"]).reshape(2, 2))


def _P_of(rec, sv):
    g = _gene_of(rec)
    r = certify_common_lyapunov(g, t_domain=T_DOMAIN, margin=MARGIN, solver=sv)
    return np.asarray(r.P, dtype=np.float64) if r.P is not None else None


def _min_decrease_lmi(P, verts, margin: float = MARGIN) -> float:
    if P is None:
        return float("-inf")
    P = 0.5 * (P + P.T)
    mn = float(np.min(np.linalg.eigvalsh(P)))
    for J in verts:
        M = P - J.T @ P @ J - margin * np.eye(P.shape[0])
        M = 0.5 * (M + M.T)
        mn = min(mn, float(np.min(np.linalg.eigvalsh(M))))
    return mn


if __name__ == "__main__":
    n = int(sys.argv[sys.argv.index("--thin") + 1]) if "--thin" in sys.argv else 60
    main(n_thin=n)
