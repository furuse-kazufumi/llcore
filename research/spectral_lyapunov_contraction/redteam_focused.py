# SPDX-License-Identifier: Apache-2.0
"""Focused independent probes (fast) while the full sweep runs.

(1) Analytic-vs-numeric vertex-convexity falsification on adversarially-chosen genes
    designed to make sigma_max NON-convex-looking (near-singular interior). If the box-sup is
    ever interior, the 2-norm-vertex certifier is unsound.
(2) Brute-force the single worst admitted gene (worst emp ||J||_2=0.999936) with a HUGE dense
    (s,x) AND free-t grid to see if its true ||J||_2 / rho actually exceeds 1.
(3) Count SDP optimal_inaccurate statuses among admitted genes (tol-fragility surface).
(4) Boundary check: is `<1` strict-admit fragile? how many admitted genes have two_sup in
    [0.999, 1.0)? how many residual genes have two_sup in (1.0, 1.001]? (near-miss census)
"""
from __future__ import annotations
import sys, json, itertools
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
_TRACK_C = (_HERE.parent / "coupled_z3_contraction").resolve()
for p in (str(_HERE), str(_TRACK_C)):
    if p not in sys.path:
        sys.path.insert(0, p)

from coupled_map import CoupledGene, t_min_per_coord
from redteam_fast import build_population
from two_norm_vertex_certifier import certify_2norm_contraction, jacobian_at_t, sigma_max
from lyapunov_sdp_certifier import CVXPY_AVAILABLE, certify_common_lyapunov

MAX_INPUT = 1.0


def jac(decay, W, t):
    t = np.asarray(t, float).reshape(2)
    return np.diag(decay) + np.diag((1.0 - decay) * t) @ W


def probe1_adversarial_convexity():
    """Hand-pick genes whose J is near-singular in the box interior -> sigma_max could dip and
    rise; brute a 1001x1001 t-grid and compare interior-max to 4-vertex max."""
    rng = np.random.default_rng(13572468)
    worst = -np.inf
    worst_g = None
    n_pos = 0
    cases = []
    # 2000 random + structured antisymmetric/symmetric extremes
    for _ in range(2000):
        cases.append((rng.uniform(0, 1, 2), rng.uniform(-2, 2, (2, 2))))
    for wd in (-0.9, 0.0, 0.9):
        for wo in (-1.5, -0.9, 0.9, 1.5):
            cases.append((np.array([0.2, 0.8]), np.array([[wd, wo], [-wo, wd]])))
            cases.append((np.array([0.5, 0.5]), np.array([[wd, wo], [wo, wd]])))
    grid = np.linspace(0.0, 1.0, 201)
    for decay, W in cases:
        decay = np.clip(decay, 0, 1); W = np.clip(W, -2, 2)
        g = CoupledGene.make(decay=decay, W=W)
        t_lo = t_min_per_coord(g)
        verts = [(t_lo[0], t_lo[1]), (t_lo[0], 1.0), (1.0, t_lo[1]), (1.0, 1.0)]
        vsup = max(sigma_max(jac(decay, W, v)) for v in verts)
        gg0 = t_lo[0] + grid * (1.0 - t_lo[0])
        gg1 = t_lo[1] + grid * (1.0 - t_lo[1])
        T0 = np.repeat(gg0, len(gg1)); T1 = np.tile(gg1, len(gg0))
        tt = np.stack([T0, T1], 1)
        a = (1.0 - decay)[None, :] * tt
        J = a[:, :, None] * W[None, :, :]
        J[:, 0, 0] += decay[0]; J[:, 1, 1] += decay[1]
        sv = np.linalg.svd(J, compute_uv=False)
        imax = float(sv[:, 0].max())
        exc = imax - vsup
        if exc > worst:
            worst = exc; worst_g = {"decay": decay.tolist(), "W": W.tolist(), "vsup": vsup, "imax": imax, "exc": exc}
        if exc > 1e-9:
            n_pos += 1
    return {"n_cases": len(cases), "worst_excess": float(worst), "n_positive_gt_1e-9": n_pos, "worst_gene": worst_g}


def probe2_brute_worst_admit():
    """The worst admitted gene per the JSON had emp ||J||_2 = 0.999936. Find it (max two_sup<1
    admit) and brute its TRUE box sup of ||J||_2 and rho via huge (s,x) sample + free-t grid."""
    genes_raw, n_grid = build_population(3000, 0)
    best = None
    for gi, (decay, W) in enumerate(genes_raw):
        decay = np.clip(decay, 0, 1); W = np.clip(W, -2, 2)
        g = CoupledGene.make(decay=decay, W=W)
        r = certify_2norm_contraction(g, t_domain="tmin1")
        if r.certified:
            if best is None or r.sup_2norm > best[1]:
                best = (gi, r.sup_2norm, decay, W)
    gi, sup, decay, W = best
    # huge (s,x) sample
    rng = np.random.default_rng(424242)
    S = np.vstack([rng.uniform(-1, 1, (500000, 2)),
                   np.array(list(itertools.product([-1.0, 1.0, 0.0], repeat=2)))])
    X = np.vstack([rng.uniform(-1, 1, (500000, 2)),
                   np.array(list(itertools.product([-1.0, 1.0, 0.0], repeat=2)))])
    pre = S @ W.T + X
    t = 1.0 - np.tanh(pre) ** 2
    a = (1.0 - decay)[None, :] * t
    J = a[:, :, None] * W[None, :, :]
    J[:, 0, 0] += decay[0]; J[:, 1, 1] += decay[1]
    sv = np.linalg.svd(J, compute_uv=False)
    two_sx = float(sv[:, 0].max())
    tr = J[:, 0, 0] + J[:, 1, 1]; det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
    disc = tr * tr - 4 * det
    rho = np.where(disc >= 0, np.maximum(np.abs((tr + np.sqrt(np.abs(disc))) / 2), np.abs((tr - np.sqrt(np.abs(disc))) / 2)), np.sqrt(np.abs(det)))
    rho_sx = float(rho.max())
    # free-t grid sup (the certified quantity)
    t_lo = t_min_per_coord(CoupledGene.make(decay=decay, W=W))
    grid = np.linspace(0, 1, 501)
    gg0 = t_lo[0] + grid * (1 - t_lo[0]); gg1 = t_lo[1] + grid * (1 - t_lo[1])
    T0 = np.repeat(gg0, len(gg1)); T1 = np.tile(gg1, len(gg0))
    tt = np.stack([T0, T1], 1)
    a2 = (1.0 - decay)[None, :] * tt
    J2 = a2[:, :, None] * W[None, :, :]
    J2[:, 0, 0] += decay[0]; J2[:, 1, 1] += decay[1]
    two_t = float(np.linalg.svd(J2, compute_uv=False)[:, 0].max())
    return {"index": gi, "decay": decay.tolist(), "W": W.tolist(), "certified_sup_2norm": sup,
            "brute_sx_2norm": two_sx, "brute_sx_rho": rho_sx, "brute_freeT_2norm": two_t}


def probe3_sdp_status_and_census():
    """Census of SDP statuses + near-boundary admits/residuals from the existing JSON."""
    jpath = _HERE / "exp_d_results.json"
    data = json.loads(jpath.read_text(encoding="utf-8"))
    # we don't have per-gene statuses in JSON beyond examples; recompute statuses live for tmin1
    genes_raw, n_grid = build_population(3000, 0)
    status_counts = {}
    inaccurate_admitted = 0
    near_admit = 0   # two_sup in [0.999,1.0)
    near_reject = 0  # two_sup in (1.0,1.001]
    if CVXPY_AVAILABLE:
        for gi, (decay, W) in enumerate(genes_raw):
            decay = np.clip(decay, 0, 1); W = np.clip(W, -2, 2)
            g = CoupledGene.make(decay=decay, W=W)
            rL = certify_common_lyapunov(g, t_domain="tmin1")
            st = rL.solver_status
            status_counts[st] = status_counts.get(st, 0) + 1
            if rL.certified and st == "optimal_inaccurate":
                inaccurate_admitted += 1
            r2 = certify_2norm_contraction(g, t_domain="tmin1")
            if 0.999 <= r2.sup_2norm < 1.0:
                near_admit += 1
            if 1.0 < r2.sup_2norm <= 1.001:
                near_reject += 1
    return {"sdp_status_counts_tmin1": status_counts,
            "sdp_inaccurate_admitted_tmin1": inaccurate_admitted,
            "two_sup_in_[0.999,1.0)_tmin1": near_admit,
            "two_sup_in_(1.0,1.001]_tmin1": near_reject}


if __name__ == "__main__":
    out = {}
    print("probe1 adversarial convexity ...", flush=True)
    out["probe1_convexity"] = probe1_adversarial_convexity()
    print(json.dumps(out["probe1_convexity"], indent=2)[:600], flush=True)
    print("probe2 brute worst admit ...", flush=True)
    out["probe2_worst_admit"] = probe2_brute_worst_admit()
    print(json.dumps(out["probe2_worst_admit"], indent=2), flush=True)
    print("probe3 sdp status census ...", flush=True)
    out["probe3_sdp_census"] = probe3_sdp_status_and_census()
    print(json.dumps(out["probe3_sdp_census"], indent=2), flush=True)
    (_HERE / "redteam_focused_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote redteam_focused_results.json", flush=True)
