# SPDX-License-Identifier: Apache-2.0
"""Red-team for the degree-6 ladder (DEG6_PREREGISTRATION §Red-team).

Lens 1 — soundness (authoritative): 50 000-sample independent from-below ρ oracle over EVERY
         deg4/deg6-certified gene from EXP-A; require 0 unsound.
Lens 2 — admission-size artifact: re-run the EXP-B capability ladder with a gene-keyed RANDOM
         fitness; a real capability payoff must vanish (reach ≈ tie across gates). If a "payoff"
         survives random fitness it was an artifact of how many genes each gate admits.
Lens 3 — circularity: the EXP-B residual reference must be quad-REJECTED (so no quad gate can
         admit the exact optimum) and self-R²=1, and is chosen by a gate-independent search.
Lens 4 — numerical robustness: deg6 coverage-advance + complementarity must hold across SDP
         margins 1e-6..1e-8 (the independent eigen re-check, not the solver status, is authority).
"""
from __future__ import annotations

import hashlib
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
    CoupledGeneCodec, _inf_certifies, _two_certifies, _sdp_certifies,
    empirical_spectral_radius, classify_region,
)
from verifier_deg4 import cert_deg4_n2, _vertices_n2  # noqa: E402
from verifier_deg6 import certify_degN, cert_deg6_n2  # noqa: E402
from evolvable_core import EvolveConfig, evolve  # noqa: E402
from exp_deg6_capability import _gate_ladder, find_residual_reference, _strict_gate  # noqa: E402


def lens1_soundness(sound_samples: int = 50000) -> dict:
    with open(os.path.join(_HERE, "exp_deg6_residual_genes.json"), encoding="utf-8") as f:
        deg = json.load(f)["deg_certified"]
    unsound = []
    for rec in deg:
        g = CoupledGene.make(decay=np.asarray(rec["decay"]), W=np.asarray(rec["W"]).reshape(2, 2))
        rho = empirical_spectral_radius(g, n_samples=sound_samples)
        if rho >= 1.0:
            unsound.append({**rec, "rho_50k": rho})
    return {"n_deg_certified": len(deg), "n_unsound_at_50k": len(unsound),
            "unsound": unsound, "PASS": len(unsound) == 0}


class _RandomFitness:
    """Deterministic gene-keyed pseudo-random fitness in [0,1] (independent of dynamics)."""
    name = "random_fitness"

    def fitness(self, gene) -> float:
        g = gene.clipped()
        key = np.round(np.concatenate([g.decay, g.W.reshape(-1)]), 4).tobytes()
        h = hashlib.sha256(key).digest()
        return int.from_bytes(h[:8], "big") / 2 ** 64


def lens2_admission_artifact(n_seeds: int = 8, base_seed: int = 9000) -> dict:
    codec = CoupledGeneCodec()
    cfg = EvolveConfig(pop_size=24, n_generations=25, elitism=1, tournament_k=3,
                       crossover_rate=0.5, mutation_sigma=0.15, resample_cap=50)
    ladder = _gate_ladder()
    obj = _RandomFitness()
    reach = {name: np.zeros(n_seeds) for name, _ in ladder}
    for si in range(n_seeds):
        for gname, gate in ladder:
            rng = np.random.default_rng(base_seed + si)
            reach[gname][si] = evolve(codec, obj, gate, cfg, rng=rng, gate_initial=True).best_fitness
    means = {g: round(float(reach[g].mean()), 4) for g, _ in ladder}
    # under random fitness, a stronger gate must NOT systematically reach higher.
    sg = _strict_gate(reach["L4_deg6"], reach["L2_sdp"])
    return {"reach_means": means, "L4_vs_L2_strict": sg,
            "PASS_no_artifact_payoff": (not sg["strict_pass"])}


def lens3_circularity() -> dict:
    ref = find_residual_reference()
    g, info = ref
    region = classify_region(g)
    quad = _inf_certifies(g) or _two_certifies(g) or _sdp_certifies(g)
    # self-R^2 on its own target is 1 by construction (sanity of the objective definition)
    return {"reference_region": region, "is_quad_rejected": (not quad),
            "deg4": bool(cert_deg4_n2(g)), "deg6": bool(cert_deg6_n2(g)),
            "transient": round(info["transient"], 3),
            "PASS_noncircular": (region == "non_certified" and not quad)}


def lens4_numerical(n_target: int = 80, seed: int = 5150, margins=(1e-6, 1e-7, 1e-8)) -> dict:
    rng = np.random.default_rng(seed)
    genes = []
    while len(genes) < n_target:
        g = CoupledGene.make(decay=rng.uniform(0, 1, 2), W=rng.uniform(-2, 2, (2, 2)))
        if _inf_certifies(g) or _two_certifies(g) or _sdp_certifies(g):
            continue
        if empirical_spectral_radius(g, n_samples=4000) < 1.0:
            genes.append(g)
    rows = {}
    for m in margins:
        d4o = d6o = both = 0
        for g in genes:
            v = _vertices_n2(g)
            c4 = certify_degN(v, 2, margin=m)
            c6 = certify_degN(v, 3, margin=m)
            if c4 and not c6:
                d4o += 1
            elif c6 and not c4:
                d6o += 1
            elif c4 and c6:
                both += 1
        rows[f"{m:.0e}"] = {"deg4_only": d4o, "deg6_only": d6o, "both": both,
                            "complement_both_pos": (d4o > 0 and d6o > 0)}
    robust = all(r["complement_both_pos"] for r in rows.values())
    return {"residual_pool": n_target, "by_margin": rows, "PASS_complement_robust": robust}


def run() -> dict:
    out = {}
    t0 = time.time()
    print("lens3 circularity ...", flush=True)
    out["lens3_circularity"] = lens3_circularity()
    print(json.dumps(out["lens3_circularity"]), flush=True)
    print("lens4 numerical robustness ...", flush=True)
    out["lens4_numerical"] = lens4_numerical()
    print(json.dumps(out["lens4_numerical"]), flush=True)
    print("lens2 admission-size artifact (random fitness) ...", flush=True)
    out["lens2_admission_artifact"] = lens2_admission_artifact()
    print(json.dumps(out["lens2_admission_artifact"]), flush=True)
    print("lens1 soundness @50k ...", flush=True)
    out["lens1_soundness"] = lens1_soundness()
    print(json.dumps({k: v for k, v in out["lens1_soundness"].items() if k != "unsound"}), flush=True)
    out["elapsed_s"] = round(time.time() - t0, 1)
    out["all_pass"] = bool(out["lens1_soundness"]["PASS"]
                           and out["lens2_admission_artifact"]["PASS_no_artifact_payoff"]
                           and out["lens3_circularity"]["PASS_noncircular"]
                           and out["lens4_numerical"]["PASS_complement_robust"])
    with open(os.path.join(_HERE, "redteam_deg6_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("ALL_PASS:", out["all_pass"], f"({out['elapsed_s']}s)", flush=True)
    return out


if __name__ == "__main__":
    run()
