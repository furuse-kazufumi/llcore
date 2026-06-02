# SPDX-License-Identifier: Apache-2.0
"""exp_a3_soundness.py — Track A gate A3 (soundness vs reality).

For EVERY gene the ACHIEVABLE certifier admits (sample up to ~2000 admitted genes), compute
the empirical Lipschitz constant via central-difference over the box and check:

  A3a: empirical_L <= L_achievable + tol(1e-6)   -- MUST have 0 violations.
  A3b: L_achievable <= L_free for ALL genes      -- refinement never increases the bound.

The empirical estimator samples >= 4000 (s, x) points in the box [-1,1]x[-1,1] and takes the
max |ds'/ds| (a from-below approximation of the true box sup). We use both llcore's own
``empirical_lipschitz`` (n_samples >= 4000) AND an independent central-difference grid as a
cross-check.

Run: cd D:/projects/llcore && py -3.11 research/lipschitz_refinement/exp_a3_soundness.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from achievable_lipschitz import (  # noqa: E402
    L_achievable,
    L_free,
    is_z3_available,
    verify_lipschitz_contraction_achievable,
)
from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.verifier.invariants import empirical_lipschitz  # noqa: E402

SEED = 0
TOL = 1e-6
N_EMP = 5000  # >= 4000 (s,x) samples
OUT = Path(__file__).resolve().parent / "exp_a3_results.json"


def independent_empirical_L(gene: StateUpdateGene, n: int = N_EMP, seed: int = SEED) -> float:
    """Independent central-difference estimate of sup|ds'/ds| over the box (cross-check)."""
    g = gene.clipped()
    rng = np.random.default_rng(seed + 12345)  # different stream than llcore's
    h = 1e-6
    s = rng.uniform(-1.0, 1.0, size=n)
    x = rng.uniform(-1.0, 1.0, size=n)

    def step(state, inp):
        pre = g.mix * inp + g.gate_str * state
        return g.decay * state + (1.0 - g.decay) * np.tanh(pre)

    deriv = (step(s + h, x) - step(s - h, x)) / (2.0 * h)
    return float(np.max(np.abs(deriv)))


def build_all_genes() -> list[StateUpdateGene]:
    genes = []
    decays = np.linspace(0.0, 1.0, 16)
    mixes = np.linspace(-1.0, 1.0, 16)
    gates = np.linspace(-2.0, 2.0, 16)
    for d in decays:
        for m in mixes:
            for g in gates:
                genes.append(StateUpdateGene(float(d), float(m), float(g)).clipped())
    rng = np.random.default_rng(SEED)
    n_rand = 2000
    d = rng.uniform(0.0, 1.0, n_rand)
    m = rng.uniform(-1.0, 1.0, n_rand)
    g = rng.uniform(-2.0, 2.0, n_rand)
    for i in range(n_rand):
        genes.append(StateUpdateGene(float(d[i]), float(m[i]), float(g[i])).clipped())
    return genes


def main() -> int:
    if not is_z3_available():
        print("FATAL: z3 not available.")
        return 1

    all_genes = build_all_genes()
    n_total = len(all_genes)

    # A3b: L_achievable <= L_free for ALL genes.
    bound_increase_violations = 0
    max_increase = 0.0
    for ge in all_genes:
        delta = L_free(ge) - L_achievable(ge)  # must be >= 0
        if delta < -1e-9:
            bound_increase_violations += 1
            max_increase = max(max_increase, -delta)

    # Collect achievable-admitted genes, cap at ~2000.
    admitted = []
    for ge in all_genes:
        if verify_lipschitz_contraction_achievable(ge).contraction:
            admitted.append(ge)
        if len(admitted) >= 2000:
            break
    n_admitted = len(admitted)
    print(f"admitted genes sampled: {n_admitted} (of {n_total} total)")

    # A3a: empirical_L <= L_achievable + tol for every admitted gene.
    violations = []
    max_emp_over = -1.0  # worst (emp - L_ach)
    worst_gene = None
    for ge in admitted:
        l_ach = L_achievable(ge)
        emp_llcore = empirical_lipschitz(ge, n_samples=N_EMP, seed=SEED)
        emp_indep = independent_empirical_L(ge)
        emp = max(emp_llcore, emp_indep)
        over = emp - l_ach
        if over > max_emp_over:
            max_emp_over = over
            worst_gene = (ge.decay, ge.mix, ge.gate_str, emp, l_ach)
        if emp > l_ach + TOL:
            violations.append(
                {
                    "decay": ge.decay,
                    "mix": ge.mix,
                    "gate_str": ge.gate_str,
                    "empirical_L": emp,
                    "L_achievable": l_ach,
                    "excess": over,
                }
            )

    results = {
        "seed": SEED,
        "tol": TOL,
        "n_emp_samples": N_EMP,
        "n_total_genes": n_total,
        "n_admitted_sampled": n_admitted,
        "A3a_empirical_violations": len(violations),
        "A3a_pass": len(violations) == 0,
        "A3a_worst_excess_emp_minus_Lach": max_emp_over,
        "A3a_worst_gene": {
            "decay": worst_gene[0],
            "mix": worst_gene[1],
            "gate_str": worst_gene[2],
            "empirical_L": worst_gene[3],
            "L_achievable": worst_gene[4],
        }
        if worst_gene
        else None,
        "A3b_L_achievable_le_L_free_violations": bound_increase_violations,
        "A3b_max_increase": max_increase,
        "A3b_pass": bound_increase_violations == 0,
        "violations_detail": violations[:20],
    }

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== A3 RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
