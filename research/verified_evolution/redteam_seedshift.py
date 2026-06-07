# SPDX-License-Identifier: Apache-2.0
"""Seed-fragility attack: re-run Track B with a DIFFERENT seed block (2000..2019)
and an alternate readout seed, and see whether the B1 verdicts (COSTLY vs FREE)
and B2 load-bearing classification flip.

Reuses gated_evolve + the canonical wiring but with new seeds. Reports per-cell:
median_delta, one-sided Wilcoxon p, B2 ungated/gated contraction pathology rate,
B3 false admits. Pure additive, deterministic.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
for p in (str(_SRC), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from scipy.stats import wilcoxon  # noqa: E402

from llcore.fitness import (  # noqa: E402
    CopyTask, calibrate_baseline, evaluate_gene, make_fixed_readout,
)
from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.verifier.invariants import empirical_lipschitz  # noqa: E402
from gated_evolve import gated_evolve  # noqa: E402

GA_KW = dict(pop_size=10, n_generations=10, tournament_k=3, mutation_sigma=0.15,
             crossover_rate=0.5, elitism=1, resample_cap=50)
TRAIN_N_TRIALS = 5
TEST_N_TRIALS = 20


def run(seed_block_start: int, readout_seed: int, label: str) -> None:
    seeds = list(range(seed_block_start, seed_block_start + 20))
    readout = make_fixed_readout(8, 8, seed=readout_seed)
    tasks = {}
    for name, delay in (("copy_d0", 0), ("copy_d8", 8)):
        t = CopyTask(state_dim=8, out_dim=8, delay=delay)
        b = calibrate_baseline(t, readout)
        tasks[name] = replace(t, baseline_mse=b)

    print(f"\n##### {label}  (GA seeds {seeds[0]}..{seeds[-1]}, readout_seed={readout_seed}) #####")
    for task_name, task in tasks.items():
        def ff(gene, rng):
            return evaluate_gene(gene, task, readout, rng, n_trials=TRAIN_N_TRIALS)

        none_test = {}
        for s in seeds:
            gr = gated_evolve(ff, gate_mode="none", rng=np.random.default_rng(s), **GA_KW)
            none_test[s] = evaluate_gene(gr.result.final_best.gene, task,
                                         readout, np.random.default_rng(900000 + s),
                                         n_trials=TEST_N_TRIALS)
        for gate in ("contraction",):  # state_norm is structurally a no-op; skip
            deltas = []
            false_admits = 0
            adm = 0
            ung_viol = ung_tot = gat_viol = gat_tot = 0
            # ungated final-pop pathology
            for s in seeds:
                grn = gated_evolve(ff, gate_mode="none", rng=np.random.default_rng(s), **GA_KW)
                for ind in grn.result.generations[-1].individuals:
                    ung_tot += 1
                    if empirical_lipschitz(ind.gene) >= 1.0:
                        ung_viol += 1
            for s in seeds:
                gr = gated_evolve(ff, gate_mode=gate, rng=np.random.default_rng(s), **GA_KW)
                tf = evaluate_gene(gr.result.final_best.gene, task, readout,
                                   np.random.default_rng(900000 + s), n_trials=TEST_N_TRIALS)
                deltas.append(tf - none_test[s])
                false_admits += sum(1 for g in gr.admitted_genes
                                    if empirical_lipschitz(g) >= 1.0)
                adm += len(gr.admitted_genes)
                for ind in gr.result.generations[-1].individuals:
                    gat_tot += 1
                    if empirical_lipschitz(ind.gene) >= 1.0:
                        gat_viol += 1
            deltas = np.array(deltas)
            med = float(np.median(deltas))
            nz = deltas[deltas != 0.0]
            p_less = (float(wilcoxon(deltas, alternative="less", zero_method="wilcox").pvalue)
                      if nz.size else 1.0)
            verdict = ("COSTLY" if (p_less < 0.05 and med < -1e-9)
                       else ("FREE" if abs(med) < 0.02 or p_less >= 0.05 else "AMBIG"))
            print(f"  {task_name}/{gate}: median_delta={med:+.5f} p_less={p_less:.4g} "
                  f"n_nz={nz.size} -> {verdict} | B2 ungated={ung_viol}/{ung_tot}="
                  f"{ung_viol/ung_tot:.3f} gated={gat_viol}/{gat_tot} | "
                  f"B3 false_admits={false_admits}/{adm}")


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    # original (sanity) then two perturbed blocks
    run(1000, 1001, "ORIGINAL seeds (sanity)")
    run(2000, 1001, "SHIFTED GA seeds 2000..2019, same readout")
    run(3000, 4242, "SHIFTED GA seeds 3000..3019 + DIFFERENT readout 4242")
    return 0


if __name__ == "__main__":
    sys.exit(main())
