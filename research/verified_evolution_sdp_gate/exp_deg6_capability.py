# SPDX-License-Identifier: Apache-2.0
"""EXP-B — the CAPABILITY frontier: does a stronger (deg4/deg6) gate let evolution REACH higher
fitness than SDP?

Nested gate ladder L0=inf ⊆ L1=inf∪2norm ⊆ L2=sdp ⊆ L3=sdp∪deg4 ⊆ L4=sdp∪deg4∪deg6, run through
``evolvable_core.evolve`` on two objectives, n_seeds paired (CRN):

  * rotation     — POSITIVE CONTROL (known inf→2norm payoff): validates the harness (G-B1).
  * residual_reach — target = free response of the best quad-rejected, deg4/deg6-certified
    reference gene (honest reachability framing, like NonNormalObjective). Tests G-B2.

Strict gate (project standard): one-sided Wilcoxon p<0.05 AND |paired_sign_delta|≥0.147 AND n≥15.
Honest-null pre-committed (DEG6_PREREGISTRATION): G-B2 is expected NULL (capability saturates at
SDP); a null here is the committed sharper finding, not a failure.
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
sys.path.insert(0, _HERE)
for _d in ("coupled_z3_contraction", "spectral_lyapunov_contraction"):
    _p = os.path.normpath(os.path.join(_HERE, "..", _d))
    if _p not in sys.path:
        sys.path.insert(0, _p)

from coupled_map import CoupledGene  # noqa: E402
from coupled_components import (  # noqa: E402
    CoupledGeneCodec, RotationObjective, _Backend,
    _free_response, _r2, _inf_certifies, _two_certifies, _sdp_certifies,
    empirical_spectral_radius, make_verifier,
)
from verifier_deg4 import cert_deg4_n2, make_deg4_verifier_n2  # noqa: E402
from verifier_deg6 import cert_deg6_n2, make_deg6_verifier_n2  # noqa: E402
from evolvable_core import EvolveConfig, evolve  # noqa: E402

try:
    from scipy.stats import wilcoxon
    _SCIPY = True
except Exception:
    _SCIPY = False


def _quad(g) -> bool:
    return _inf_certifies(g) or _two_certifies(g) or _sdp_certifies(g)


def _transient_amp(gene, s0, T=30) -> float:
    tr = _free_response(gene, np.asarray(s0, float), T)
    return float(np.max(np.linalg.norm(tr, axis=1)) / (np.linalg.norm(tr[0]) + 1e-12))


def find_residual_reference(seed: int = 777, scan: int = 3500, s0=(0.5, 0.3)):
    """Deterministic search for the best (max-transient) quad-rejected, deg4/deg6-certified gene.
    Returns (gene, info). Independent of any gate under test (anti-circularity)."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(scan):
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if _quad(g):
            continue
        ta = _transient_amp(g, s0)
        if ta < 1.2:
            continue
        if empirical_spectral_radius(g, n_samples=4000) >= 1.0:
            continue
        if not (cert_deg4_n2(g) or cert_deg6_n2(g)):
            continue
        if best is None or ta > best[1]["transient"]:
            best = (g, {"transient": ta, "decay": g.decay.tolist(),
                        "W": g.W.reshape(-1).tolist(),
                        "deg4": cert_deg4_n2(g), "deg6": cert_deg6_n2(g)})
    return best


class ResidualReachObjective:
    """Reproduce the free response of a residual (deg4/deg6-only) reference. Optimum is
    quad-REJECTED by construction → only L3/L4 can admit the exact optimum. Reachability test."""
    name = "residual_reach"

    def __init__(self, ref_gene, s0=(0.5, 0.3), T=30):
        self.s0 = np.asarray(s0, float)
        self.T = T
        self._target = _free_response(ref_gene, self.s0, T)

    def fitness(self, gene) -> float:
        return _r2(_free_response(gene, self.s0, self.T), self._target)


def _gate_ladder():
    """Nested admission sets L0..L4 (each ⊇ previous)."""
    return [
        ("L0_inf", make_verifier("inf_norm")),
        ("L1_two", _Backend("inf_two", lambda g: _inf_certifies(g) or _two_certifies(g))),
        ("L2_sdp", make_verifier("sdp")),
        ("L3_deg4", make_deg4_verifier_n2()),
        ("L4_deg6", make_deg6_verifier_n2()),
    ]


def _paired_sign_delta(a: np.ndarray, b: np.ndarray) -> float:
    d = a - b
    n_pos = int(np.sum(d > 1e-12))
    n_neg = int(np.sum(d < -1e-12))
    return (n_pos - n_neg) / len(d)


def _strict_gate(a: np.ndarray, b: np.ndarray) -> dict:
    """One-sided (a>b) Wilcoxon + paired_sign_delta + n. PASS iff p<0.05 & |psd|>=0.147 & n>=15."""
    psd = _paired_sign_delta(a, b)
    p = None
    if _SCIPY:
        d = a - b
        if np.any(np.abs(d) > 1e-12):
            try:
                p = float(wilcoxon(a, b, alternative="greater", zero_method="wilcox").pvalue)
            except Exception:
                p = None
    passes = (p is not None and p < 0.05) and abs(psd) >= 0.147 and len(a) >= 15 and psd > 0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "mean_delta": float((a - b).mean()), "paired_sign_delta": psd,
            "wilcoxon_p_greater": p, "n": len(a), "strict_pass": bool(passes)}


def run(n_seeds: int = 15, base_seed: int = 4000) -> dict:
    t0 = time.time()
    codec = CoupledGeneCodec()
    cfg = EvolveConfig(pop_size=24, n_generations=25, elitism=1, tournament_k=3,
                       crossover_rate=0.5, mutation_sigma=0.15, resample_cap=50)

    ref = find_residual_reference()
    if ref is None:
        raise RuntimeError("no residual reference gene found")
    ref_gene, ref_info = ref
    print("residual reference:", json.dumps(ref_info), flush=True)

    objectives = [RotationObjective(), ResidualReachObjective(ref_gene)]
    ladder = _gate_ladder()

    # reach[obj][gate] = array of per-seed best fitness (paired by seed index).
    reach: dict = {o.name: {name: np.zeros(n_seeds) for name, _ in ladder} for o in objectives}
    for obj in objectives:
        for si in range(n_seeds):
            for gname, gate in ladder:
                rng = np.random.default_rng(base_seed + si)  # CRN: same seed across gates
                res = evolve(codec, obj, gate, cfg, rng=rng, gate_initial=True)
                reach[obj.name][gname][si] = res.best_fitness
        print(f"[{obj.name}] reach means:",
              {g: round(float(reach[obj.name][g].mean()), 4) for g, _ in ladder}, flush=True)

    # Gates.
    rot = reach["rotation"]
    res = reach["residual_reach"]
    gb1 = _strict_gate(rot["L1_two"], rot["L0_inf"])           # positive control
    gb2_L4_L2 = _strict_gate(res["L4_deg6"], res["L2_sdp"])    # capability payoff (deg6 vs sdp)
    gb2_L3_L2 = _strict_gate(res["L3_deg4"], res["L2_sdp"])    # capability payoff (deg4 vs sdp)

    out = {
        "n_seeds": n_seeds, "base_seed": base_seed, "scipy": _SCIPY,
        "residual_reference": ref_info,
        "reach_means": {o.name: {g: round(float(reach[o.name][g].mean()), 4) for g, _ in ladder}
                        for o in objectives},
        "reach_raw": {o.name: {g: reach[o.name][g].tolist() for g, _ in ladder} for o in objectives},
        "G_B1_positive_control_rotation_L1_vs_L0": gb1,
        "G_B2_capability_L4deg6_vs_L2sdp": gb2_L4_L2,
        "G_B2_capability_L3deg4_vs_L2sdp": gb2_L3_L2,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(_HERE, "exp_deg6_capability_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "reach_raw"}, indent=2), flush=True)
    return out


if __name__ == "__main__":
    ns = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 15
    run(n_seeds=ns)
