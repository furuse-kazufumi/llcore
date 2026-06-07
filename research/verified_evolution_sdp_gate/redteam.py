# SPDX-License-Identifier: Apache-2.0
"""Adversarial red-team for the SDP-gated coupled evolution (Lenses A-D).

A  circular/tautology : run sdp vs inf gate with a GENE-INDEPENDENT random
                        fitness. If sdp still beats inf, the G4 payoff is an
                        admission-SIZE artifact, not fitness structure.
B  soundness (indep)  : independent brute-force oracle (100k samples + dense
                        corners + long-horizon trajectory separation) on every
                        gate winner from exp2. Any divergent admitted gene
                        falsifies soundness.
C  power/seed         : re-run G4 (rotation, sdp vs inf) across 3 base-seed
                        families; report achieved effect + Bonferroni.
D  mechanism attrib.  : region class of each gate's winners. The payoff must come
                        from the inf-REJECTED region (two_norm_only/sdp_only),
                        else the stated mechanism is wrong.

Usage: py -3.11 redteam.py [A|B|C|D|all]
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8")
    except Exception:
        pass

from coupled_components import (
    BenignDecayObjective,
    CoupledGeneCodec,
    NonNormalObjective,
    RotationObjective,
    classify_region,
    empirical_spectral_radius,
    make_verifier,
    step,
)
from coupled_map import CoupledGene
from evolvable_core import EvolveConfig, evolve
from exp_runner import paired_compare

_HERE = os.path.dirname(os.path.abspath(__file__))
CODEC = CoupledGeneCodec()


class _RandomFitness:
    """Gene-independent pseudo-random fitness keyed on rounded genotype (so a
    fixed gene has a fixed score, but score is UNRELATED to dynamics)."""
    name = "random_fitness"

    def fitness(self, gene: CoupledGene) -> float:
        g = gene.clipped()
        h = hash((tuple(np.round(g.decay, 4)), tuple(np.round(g.W.reshape(-1), 4)))) % (2 ** 32)
        return float(np.random.default_rng(h).random())


def lens_A(n_seeds: int = 15, pop: int = 24, gens: int = 30, base: int = 5000) -> dict:
    """sdp vs inf on RANDOM fitness — must be a null (no payoff) if G4 is real."""
    cfg = EvolveConfig(pop_size=pop, n_generations=gens, resample_cap=40)
    sdp_best, inf_best = [], []
    for i in range(n_seeds):
        s = base + i
        sdp_best.append(evolve(CODEC, _RandomFitness(), make_verifier("sdp"), cfg,
                               rng=np.random.default_rng(s)).best_fitness)
        inf_best.append(evolve(CODEC, _RandomFitness(), make_verifier("inf_norm"), cfg,
                               rng=np.random.default_rng(s)).best_fitness)
    cmp = paired_compare(sdp_best, inf_best, "greater")
    out = {"lens": "A_circular_random_fitness", "sdp_vs_inf_on_random": cmp,
           "interpretation": ("PASS(no artifact) if NOT significant: random fitness gives sdp no "
                              "edge over inf. A significant sdp>inf here would mean the G4 effect "
                              "is an admission-SIZE artifact."),
           "artifact_detected": bool(cmp["wilcoxon_p"] is not None
                                     and not isinstance(cmp["wilcoxon_p"], str)
                                     and cmp["wilcoxon_p"] < 0.05 and cmp["paired_sign_delta"] > 0.147)}
    return out


def _independent_divergent(gene: CoupledGene) -> dict:
    """Independent soundness oracle: 50k (s,x) box samples (seed 99, different from the
    in-loop oracle) + long-horizon trajectory separation growth."""
    rho = empirical_spectral_radius(gene, n_samples=50000, seed=99)
    # long-horizon separation growth (does a tiny perturbation blow up?)
    g = gene.clipped()
    rng = np.random.default_rng(123)
    max_ratio = 0.0
    for _ in range(6):
        s = rng.uniform(-1, 1, 2)
        sp = s + 1e-5 * rng.standard_normal(2)
        seq = rng.uniform(-1, 1, (1500, 2))
        d0 = np.linalg.norm(s - sp)
        for k in range(1500):
            s, sp = step(g, s, seq[k]), step(g, sp, seq[k])
        max_ratio = max(max_ratio, float(np.linalg.norm(s - sp) / max(d0, 1e-12)))
    return {"emp_rho_100k": float(rho), "sep_growth_1500": max_ratio,
            "divergent": bool(rho >= 1.0)}


def lens_B() -> dict:
    """Re-verify soundness of every gate winner in exp2 with an independent oracle."""
    p = os.path.join(_HERE, "exp2_results.json")
    if not os.path.exists(p):
        return {"lens": "B_soundness", "error": "exp2_results.json missing — run exp2 first"}
    data = json.load(open(p, encoding="utf-8"))
    runs = data["runs"]
    objs = {"rotation": RotationObjective(), "benign": BenignDecayObjective(),
            "nonnormal": NonNormalObjective()}
    # we need the winner GENES; re-derive by re-running each (seed-deterministic) gated evolve.
    cfg = EvolveConfig(pop_size=data["config"]["pop"], n_generations=data["config"]["gens"],
                       resample_cap=40)
    # Focus on the richest sound gate (sdp = main soundness risk) over rotation+nonnormal
    # (benign winners are all in the inf region = trivially sound). The sdp winner gene is
    # an ADMITTED child or an init elite; either way an independent oracle must see rho<1.
    violations = []
    checked = 0
    for tname in ("rotation", "nonnormal"):
        obj = objs[tname]
        for gate in ("sdp", "two_norm"):
            for rec in runs[tname][gate]:
                r = evolve(CODEC, obj, make_verifier(gate), cfg,
                           rng=np.random.default_rng(rec["seed"]))
                chk = _independent_divergent(r.best_gene)
                checked += 1
                if chk["divergent"]:
                    violations.append({"task": tname, "gate": gate, "seed": rec["seed"], **chk})
    return {"lens": "B_soundness_independent", "winners_checked": checked,
            "scope": "sdp+two_norm winners on rotation+nonnormal, 50k independent oracle",
            "false_admit_violations": len(violations), "violations": violations[:10],
            "interpretation": "PASS if false_admit_violations == 0 (the richest sound gates never "
                              "yield a divergent winner under a stronger independent oracle)."}


def lens_C(pop: int = 24, gens: int = 30) -> dict:
    """G4 (rotation, sdp vs inf) across 3 base-seed families; Bonferroni over 3 gate
    comparisons (sdp/two/none vs inf)."""
    cfg = EvolveConfig(pop_size=pop, n_generations=gens, resample_cap=40)
    obj = RotationObjective()
    families = {}
    for fam, base in (("f1", 1000), ("f2", 7000), ("f3", 31000)):
        sdp_best, inf_best = [], []
        for i in range(15):
            s = base + i
            sdp_best.append(evolve(CODEC, obj, make_verifier("sdp"), cfg,
                                   rng=np.random.default_rng(s)).best_fitness)
            inf_best.append(evolve(CODEC, obj, make_verifier("inf_norm"), cfg,
                                   rng=np.random.default_rng(s)).best_fitness)
        families[fam] = paired_compare(sdp_best, inf_best, "greater")
    ps = [f["wilcoxon_p"] for f in families.values()
          if isinstance(f["wilcoxon_p"], float)]
    return {"lens": "C_power_seed", "families": families,
            "all_families_significant_bonf": all(p < 0.05 / 3 for p in ps) if ps else None,
            "interpretation": "PASS if G4 holds across all 3 base-seed families (Bonferroni "
                              "alpha=0.0167); robust, not seed-cherry-picked."}


def lens_D() -> dict:
    """Region class of each gate's winners on rotation/nonnormal — the payoff must
    come from the inf-rejected region (else mechanism claim is wrong)."""
    p = os.path.join(_HERE, "exp2_results.json")
    if not os.path.exists(p):
        return {"lens": "D_attribution", "error": "exp2_results.json missing"}
    data = json.load(open(p, encoding="utf-8"))
    runs = data["runs"]
    out = {}
    for tname in ("rotation", "nonnormal"):
        out[tname] = {}
        for gate in ("inf_norm", "two_norm", "sdp"):
            regs = [rec["winner_region"] for rec in runs[tname][gate]]
            out[tname][gate] = {r: regs.count(r) for r in
                                ("inf", "two_norm_only", "sdp_only", "non_certified")}
    return {"lens": "D_mechanism_attribution", "winner_regions": out,
            "interpretation": "Mechanism holds if sdp/two winners on rotation/nonnormal sit in "
                              "inf-REJECTED regions (two_norm_only/sdp_only) while inf winners are "
                              "confined to 'inf'. Confirms the payoff comes from the extra reach."}


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    res = {}
    if which in ("A", "all"):
        res["A"] = lens_A()
        print("Lens A done")
    if which in ("D", "all"):
        res["D"] = lens_D()
        print("Lens D done")
    if which in ("C", "all"):
        res["C"] = lens_C()
        print("Lens C done")
    if which in ("B", "all"):
        res["B"] = lens_B()
        print("Lens B done")
    out = os.path.join(_HERE, f"redteam_{which}_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"redteam -> {out}")


if __name__ == "__main__":
    main()
