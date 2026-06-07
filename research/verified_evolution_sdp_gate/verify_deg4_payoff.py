# SPDX-License-Identifier: Apache-2.0
"""DECISIVE (correct method) — is there a SOUND deg4 capability payoff over sdp on rotation?

Both prior measures were biased: random-sample region ceilings are density-biased (under-sample
small regions); a 2-seed gated GA reach is high-variance. The correct measure is REGION-CONSTRAINED
OPTIMISATION with winner attribution + soundness:

  for each nested gate L2=sdp, L3=sdp∪deg4, L4=sdp∪deg4∪deg6: run K strong GA seeds maximising
  rotation, take the GLOBAL BEST gene (the optimisation ceiling), classify its region, and verify
  it at 50 000 samples.

Decision: a REAL, SOUND deg4 payoff requires (a) L3 max-reach > L2 max-reach by a margin, (b) the
L3 best gene lives in the deg4-certified residual (not the sdp region — else it is just the sdp
ceiling the GA found and L2 under-searched), and (c) the L3 best gene is empirically contracting
@50k. Otherwise the smoke 0.9765 was GA noise / region-crossing, and the deg-rung capability
frontier is NULL at n=2 (the honest pre-registered expectation).
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
    CoupledGeneCodec, RotationObjective, make_verifier,
    _inf_certifies, _two_certifies, _sdp_certifies, empirical_spectral_radius,
)
from verifier_deg4 import cert_deg4_n2, make_deg4_verifier_n2  # noqa: E402
from verifier_deg6 import cert_deg6_n2, make_deg6_verifier_n2  # noqa: E402
from evolvable_core import EvolveConfig, evolve  # noqa: E402


def _region(g) -> str:
    if _inf_certifies(g):
        return "inf"
    if _two_certifies(g):
        return "two_only"
    if _sdp_certifies(g):
        return "sdp_only"
    d4, d6 = cert_deg4_n2(g), cert_deg6_n2(g)
    if d4 and d6:
        return "deg4_and_deg6"
    if d4:
        return "deg4_only"
    if d6:
        return "deg6_only"
    return "uncertified"


def run(n_seeds: int = 12, base_seed: int = 6000, sound_samples: int = 50000) -> dict:
    t0 = time.time()
    codec = CoupledGeneCodec()
    cfg = EvolveConfig(pop_size=24, n_generations=30, elitism=1, tournament_k=3,
                       crossover_rate=0.5, mutation_sigma=0.15, resample_cap=50)
    obj = RotationObjective()
    gates = [("L2_sdp", make_verifier("sdp")),
             ("L3_deg4", make_deg4_verifier_n2()),
             ("L4_deg6", make_deg6_verifier_n2())]

    out_gates = {}
    for gname, gate in gates:
        reaches = np.zeros(n_seeds)
        best_fit = -9.9
        best_gene = None
        for si in range(n_seeds):
            rng = np.random.default_rng(base_seed + si)
            res = evolve(codec, obj, gate, cfg, rng=rng, gate_initial=True)
            reaches[si] = res.best_fitness
            if res.best_fitness > best_fit:
                best_fit = res.best_fitness
                best_gene = res.best_gene
        # attribute + soundness of the optimisation ceiling gene
        region = _region(best_gene)
        rho = empirical_spectral_radius(best_gene, n_samples=sound_samples)
        out_gates[gname] = {
            "max_reach": round(float(reaches.max()), 4),
            "mean_reach": round(float(reaches.mean()), 4),
            "reaches": [round(float(x), 4) for x in reaches],
            "best_gene_region": region,
            "best_gene_rho50k": round(float(rho), 5),
            "best_gene_sound": bool(rho < 1.0),
            "best_gene": {"decay": best_gene.decay.tolist(), "W": best_gene.W.reshape(-1).tolist()},
        }
        print(f"[{gname}] max={out_gates[gname]['max_reach']} mean={out_gates[gname]['mean_reach']} "
              f"best_region={region} rho50k={out_gates[gname]['best_gene_rho50k']} "
              f"sound={out_gates[gname]['best_gene_sound']} ({time.time()-t0:.0f}s)", flush=True)

    l2 = out_gates["L2_sdp"]["max_reach"]
    l3 = out_gates["L3_deg4"]["max_reach"]
    l4 = out_gates["L4_deg6"]["max_reach"]
    l3_region = out_gates["L3_deg4"]["best_gene_region"]
    l3_sound = out_gates["L3_deg4"]["best_gene_sound"]
    real_payoff = (l3 - l2 >= 0.02) and (l3_region in ("deg4_only", "deg4_and_deg6", "deg6_only")) and l3_sound

    out = {
        "n_seeds": n_seeds, "config": "pop24/gens30/gate_initial",
        "gates": out_gates,
        "ceiling_L2_sdp": l2, "ceiling_L3_deg4": l3, "ceiling_L4_deg6": l4,
        "payoff_deg4_over_sdp_maxreach": round(l3 - l2, 4),
        "L3_best_in_residual_region": l3_region in ("deg4_only", "deg4_and_deg6", "deg6_only"),
        "L4_ge_L3_sanity": l4 >= l3 - 1e-9,
        "REAL_SOUND_deg4_payoff": bool(real_payoff),
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(_HERE, "verify_deg4_payoff_results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: v for k, v in out.items() if k != "gates"}, indent=2), flush=True)
    return out


if __name__ == "__main__":
    ns = int(sys.argv[sys.argv.index("--seeds") + 1]) if "--seeds" in sys.argv else 12
    run(n_seeds=ns)
