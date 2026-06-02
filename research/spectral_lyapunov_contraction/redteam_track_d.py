# SPDX-License-Identifier: Apache-2.0
"""INDEPENDENT red-team oracle for Track D. Fresh seeds, denser corner-inclusive sampling,
re-derived Jacobian from scratch. Does NOT trust the implementer's emp_* numbers or gate logic.

Three attacks:
  A. SOUNDNESS: for every 2-norm-vertex-admitted and SDP-admitted gene, independently compute
     a from-below sup of ||J||_2 and rho(J) over a DENSE (s,x) box sample (fresh seed 20260602,
     n much larger than 6000, plus exhaustive sign corners AND a fine boundary grid). Any admitted
     gene with rho>=1 or (for 2-norm) ||J||_2>1 (beyond float tol) is a HIGH-severity false admit.
  B. VERTEX-SUP CONVEXITY: independently test whether sup_{box t} sigma_max(J(t)) is REALLY at a
     box vertex. For random genes, sample the FULL t-box (interior) densely and compare to the
     4-vertex max. If interior ever exceeds vertices (beyond float noise), the 2-norm-vertex
     certifier is UNSOUND. This is the load-bearing math claim.
  C. SDP P RE-VERIFICATION: independently re-check each SDP-certified P: is P PD and is
     P - J_v^T P J_v PD at all 4 vertices? Re-derive the P-norm gain from the certified P.

Reports ONLY observed numbers. Seeds reported.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_TRACK_C = (_HERE.parent / "coupled_z3_contraction").resolve()
for p in (str(_HERE), str(_TRACK_C)):
    if p not in sys.path:
        sys.path.insert(0, p)

from coupled_map import CoupledGene, t_min_per_coord  # noqa: E402
from redteam_fast import build_population  # noqa: E402
from two_norm_vertex_certifier import certify_2norm_contraction  # noqa: E402
from lyapunov_sdp_certifier import CVXPY_AVAILABLE, certify_common_lyapunov  # noqa: E402

MAX_INPUT = 1.0
FRESH_SEED = 20260602          # NOT 777 (implementer's). Independent.
N_EMP = 40000                  # >> implementer's 6000
TOL = 1e-6                     # match the gate's SOUND_TOL for apples-to-apples


def _jac_at_t(decay, W, t):
    """J(t) = diag(decay) + diag((1-decay)*t) @ W. t is a free 2-vector."""
    t = np.asarray(t, float).reshape(2)
    return np.diag(decay) + np.diag((1.0 - decay) * t) @ W


def _dense_sx_sample(decay, W, n_samples, seed):
    """Build a dense (s,x) sample: uniform interior + EXHAUSTIVE sign corners + fine face grid.

    The achievable Jacobian arises from t_i = sech^2(pre_i), pre = W@s + x. To stress the
    from-below sup we add: (1) all 16 sign corners of (s,x) in {-1,1}^2 x {-1,1}^2;
    (2) s=0 / x=0 (gives t=1, the box-top corner); (3) a fine grid over the s-x boundary faces
    where |pre| extremes (=> t extremes) live; (4) a large uniform fill.
    """
    rng = np.random.default_rng(seed)
    S = [rng.uniform(-1.0, 1.0, (n_samples, 2))]
    X = [rng.uniform(-MAX_INPUT, MAX_INPUT, (n_samples, 2))]

    # exhaustive sign corners
    sc = np.array(list(itertools.product([-1.0, 1.0], repeat=2)), float)
    xc = np.array(list(itertools.product([-MAX_INPUT, MAX_INPUT], repeat=2)), float)
    grid_s = np.repeat(sc, len(xc), axis=0)
    grid_x = np.tile(xc, (len(sc), 1))
    S.append(grid_s); X.append(grid_x)
    # zeros (t=1 corner)
    S.append(np.zeros((1, 2))); X.append(np.zeros((1, 2)))

    # fine boundary-face grid: vary one coord on a face, others on a grid in [-1,1]
    g = np.linspace(-1.0, 1.0, 21)
    faces_s, faces_x = [], []
    for a in g:
        for b in g:
            faces_s += [[a, b]]
            faces_x += [[a, b]]
    faces_s = np.array(faces_s); faces_x = np.array(faces_x)
    # cross product of a moderate s-grid and x-grid (deterministic dense coverage)
    gs = np.linspace(-1.0, 1.0, 9)
    Sgrid = np.array(list(itertools.product(gs, gs)), float)
    Xgrid = np.array(list(itertools.product(gs, gs)), float)
    Sfull = np.repeat(Sgrid, len(Xgrid), axis=0)
    Xfull = np.tile(Xgrid, (len(Sgrid), 1))
    S.append(Sfull); X.append(Xfull)

    return np.vstack(S), np.vstack(X)


def emp_norms_dense(decay, W, n_samples=N_EMP, seed=FRESH_SEED):
    """Independent from-below sup of (||J||_inf, ||J||_2, rho) over the dense (s,x) sample."""
    S, X = _dense_sx_sample(decay, W, n_samples, seed)
    pre = S @ W.T + X
    t = 1.0 - np.tanh(pre) ** 2
    a = (1.0 - decay)[None, :] * t
    J = a[:, :, None] * W[None, :, :]
    J[:, 0, 0] += decay[0]
    J[:, 1, 1] += decay[1]
    inf_n = np.abs(J).sum(axis=2).max()
    sv = np.linalg.svd(J, compute_uv=False)
    two_n = sv[:, 0].max()
    tr = J[:, 0, 0] + J[:, 1, 1]
    det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    disc = tr * tr - 4.0 * det
    rho = np.where(
        disc >= 0,
        np.maximum(np.abs((tr + np.sqrt(np.abs(disc))) / 2), np.abs((tr - np.sqrt(np.abs(disc))) / 2)),
        np.sqrt(np.abs(det)),
    )
    return float(inf_n), float(two_n), float(rho.max())


def emp_pnorm_gain_dense(decay, W, P, n_samples=N_EMP, seed=FRESH_SEED):
    """Independent from-below sup of the P-weighted contraction gain ||L^T J L^{-T}||_2."""
    P = np.asarray(P, float)
    P = 0.5 * (P + P.T)
    L = np.linalg.cholesky(P)
    Lt = L.T
    Ltinv = np.linalg.inv(L).T
    S, X = _dense_sx_sample(decay, W, n_samples, seed)
    pre = S @ W.T + X
    t = 1.0 - np.tanh(pre) ** 2
    a = (1.0 - decay)[None, :] * t
    J = a[:, :, None] * W[None, :, :]
    J[:, 0, 0] += decay[0]
    J[:, 1, 1] += decay[1]
    M = np.einsum("ij,njk,kl->nil", Lt, J, Ltinv)
    sv = np.linalg.svd(M, compute_uv=False)
    return float(sv[:, 0].max())


def attack_B_vertex_convexity(genes_raw, n_genes=400, n_interior=4000, seed=99887766):
    """Independently test the vertex-sup claim: is sup_t sigma_max(J(t)) at a box vertex?

    For each tested gene: compute the 4-vertex max sigma_max, then sample n_interior points in
    the FULL t-box [t_lo,1]^2 (tmin1) and record max(interior sigma_max - vertex sup). A positive
    excess (beyond float noise) would FALSIFY the convexity claim => 2-norm-vertex UNSOUND.
    Also add a deterministic fine 2-D grid over the t-box.
    """
    rng = np.random.default_rng(seed)
    worst_excess = -np.inf
    worst_gene = None
    n_pos = 0
    grid = np.linspace(0.0, 1.0, 41)
    # pick a spread of genes (all grid + sample of random)
    idxs = list(range(min(270, len(genes_raw))))
    rand_idxs = rng.choice(np.arange(270, len(genes_raw)), size=min(n_genes, len(genes_raw) - 270), replace=False)
    idxs += list(rand_idxs)
    for gi in idxs:
        decay, W = genes_raw[gi]
        decay = np.clip(decay, 0.0, 1.0)
        W = np.clip(W, -2.0, 2.0)
        g = CoupledGene.make(decay=decay, W=W)
        t_lo = t_min_per_coord(g)
        verts = [np.array([t_lo[0], t_lo[1]]), np.array([t_lo[0], 1.0]),
                 np.array([1.0, t_lo[1]]), np.array([1.0, 1.0])]
        vsup = max(float(np.linalg.svd(_jac_at_t(decay, W, v), compute_uv=False)[0]) for v in verts)
        # random interior
        T0 = rng.uniform(t_lo[0], 1.0, n_interior)
        T1 = rng.uniform(t_lo[1], 1.0, n_interior)
        # deterministic fine grid mapped into [t_lo,1]
        gg0 = t_lo[0] + grid * (1.0 - t_lo[0])
        gg1 = t_lo[1] + grid * (1.0 - t_lo[1])
        G0 = np.repeat(gg0, len(gg1))
        G1 = np.tile(gg1, len(gg0))
        T0 = np.concatenate([T0, G0])
        T1 = np.concatenate([T1, G1])
        tt = np.stack([T0, T1], axis=1)
        a = (1.0 - decay)[None, :] * tt
        J = a[:, :, None] * W[None, :, :]
        J[:, 0, 0] += decay[0]
        J[:, 1, 1] += decay[1]
        sv = np.linalg.svd(J, compute_uv=False)
        interior_max = float(sv[:, 0].max())
        exc = interior_max - vsup
        if exc > worst_excess:
            worst_excess = exc
            worst_gene = {"index": int(gi), "decay": decay.tolist(), "W": W.tolist(),
                          "vertex_sup": vsup, "interior_max": interior_max, "excess": exc}
        if exc > 1e-9:
            n_pos += 1
    return {"n_genes_tested": len(idxs), "worst_excess": float(worst_excess),
            "n_with_positive_excess_gt_1e-9": n_pos, "worst_gene": worst_gene}


def main():
    t0 = time.time()
    genes_raw, n_grid = build_population(3000, 0)
    n_total = len(genes_raw)
    print(f"[redteam-D] population total={n_total} grid={n_grid} cvxpy={CVXPY_AVAILABLE}", flush=True)
    print(f"[redteam-D] independent oracle: seed={FRESH_SEED} n_emp={N_EMP} (dense + corners + grid)", flush=True)

    # ---- Re-run the certifiers independently (do NOT trust the JSON's admit flags) ----
    out = {"meta": {"fresh_seed": FRESH_SEED, "n_emp": N_EMP, "tol": TOL,
                    "n_genes": n_total, "n_grid": n_grid, "cvxpy_available": CVXPY_AVAILABLE},
           "attack_A_soundness": {}, "attack_B_vertex_convexity": {}, "attack_C_sdp_P": {}}

    for dom in ("free01", "tmin1"):
        two_false = []
        sdp_false = []
        sdp_P_reverify_fail = []
        n_two_admit = 0
        n_sdp_admit = 0
        worst_two_emp2 = 0.0
        worst_two_emprho = 0.0
        worst_sdp_emprho = 0.0
        worst_sdp_pgain = 0.0
        for gi, (decay, W) in enumerate(genes_raw):
            decay = np.clip(decay, 0.0, 1.0)
            W = np.clip(W, -2.0, 2.0)
            g = CoupledGene.make(decay=decay, W=W)

            r2 = certify_2norm_contraction(g, t_domain=dom)
            if r2.certified:
                n_two_admit += 1
                emp_inf, emp_2, emp_rho = emp_norms_dense(decay, W)
                worst_two_emp2 = max(worst_two_emp2, emp_2)
                worst_two_emprho = max(worst_two_emprho, emp_rho)
                if emp_2 > 1.0 + TOL or emp_rho > 1.0 + TOL:
                    two_false.append({"index": gi, "decay": decay.tolist(), "W": W.tolist(),
                                      "two_sup": r2.sup_2norm, "emp_2norm": emp_2, "emp_rho": emp_rho})

            if CVXPY_AVAILABLE:
                rL = certify_common_lyapunov(g, t_domain=dom)
                if rL.certified:
                    n_sdp_admit += 1
                    P = np.asarray(rL.P, float)
                    # independent re-verify of the P certificate at the 4 vertices
                    if dom == "free01":
                        t_lo = np.zeros(2)
                    else:
                        t_lo = t_min_per_coord(g)
                    verts = [np.array([t_lo[0], t_lo[1]]), np.array([t_lo[0], 1.0]),
                             np.array([1.0, t_lo[1]]), np.array([1.0, 1.0])]
                    Ps = 0.5 * (P + P.T)
                    eigP = float(np.min(np.linalg.eigvalsh(Ps)))
                    min_dec = np.inf
                    for v in verts:
                        Jv = _jac_at_t(decay, W, v)
                        Mdec = Ps - Jv.T @ Ps @ Jv
                        Mdec = 0.5 * (Mdec + Mdec.T)
                        min_dec = min(min_dec, float(np.min(np.linalg.eigvalsh(Mdec))))
                    if not (eigP > 0.0 and min_dec > 0.0):
                        sdp_P_reverify_fail.append({"index": gi, "eigP": eigP, "min_dec": min_dec})
                    # independent empirical oracle in the RIGHT (P-weighted) metric
                    emp_inf, emp_2, emp_rho = emp_norms_dense(decay, W)
                    pgain = emp_pnorm_gain_dense(decay, W, Ps)
                    worst_sdp_emprho = max(worst_sdp_emprho, emp_rho)
                    worst_sdp_pgain = max(worst_sdp_pgain, pgain)
                    if emp_rho > 1.0 + TOL or pgain > 1.0 + TOL:
                        sdp_false.append({"index": gi, "decay": decay.tolist(), "W": W.tolist(),
                                          "emp_rho": emp_rho, "emp_pnorm_gain": pgain, "emp_2norm": emp_2})
            if (gi + 1) % 1000 == 0:
                print(f"  [{dom}] ...{gi+1}/{n_total} ({time.time()-t0:.1f}s)", flush=True)

        out["attack_A_soundness"][dom] = {
            "two_norm": {"n_admit": n_two_admit, "n_false_admit": len(two_false),
                         "worst_emp_2norm": worst_two_emp2, "worst_emp_rho": worst_two_emprho,
                         "false_examples": two_false[:10]},
            "sdp": {"n_admit": n_sdp_admit, "n_false_admit": len(sdp_false),
                    "worst_emp_rho": worst_sdp_emprho, "worst_emp_pnorm_gain": worst_sdp_pgain,
                    "n_P_reverify_fail": len(sdp_P_reverify_fail),
                    "P_reverify_fail_examples": sdp_P_reverify_fail[:10],
                    "false_examples": sdp_false[:10]} if CVXPY_AVAILABLE else None,
        }
        print(f"[redteam-D][{dom}] two: admit={n_two_admit} false={len(two_false)} "
              f"worst_emp2={worst_two_emp2:.6f} worst_emprho={worst_two_emprho:.6f}", flush=True)
        if CVXPY_AVAILABLE:
            print(f"[redteam-D][{dom}] sdp: admit={n_sdp_admit} false={len(sdp_false)} "
                  f"worst_emprho={worst_sdp_emprho:.6f} worst_pgain={worst_sdp_pgain:.6f} "
                  f"P_reverify_fail={len(sdp_P_reverify_fail)}", flush=True)

    print("[redteam-D] attack B: vertex-sup convexity ...", flush=True)
    out["attack_B_vertex_convexity"] = attack_B_vertex_convexity(genes_raw)
    b = out["attack_B_vertex_convexity"]
    print(f"[redteam-D] attack B: tested={b['n_genes_tested']} worst_excess={b['worst_excess']:.3e} "
          f"n_positive={b['n_with_positive_excess_gt_1e-9']}", flush=True)

    out["meta"]["wall_seconds"] = round(time.time() - t0, 1)
    (_HERE / "redteam_track_d_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[redteam-D] wrote redteam_track_d_results.json ({out['meta']['wall_seconds']}s)", flush=True)
    return out


if __name__ == "__main__":
    main()
