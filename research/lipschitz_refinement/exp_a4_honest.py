# SPDX-License-Identifier: Apache-2.0
"""exp_a4_honest.py — Track A gate A4 (honest tightness).

Quantify the residual gap = L_achievable - empirical_sup over a sample, and report where
achievable-t STILL over-rejects: genes with empirical_L < 1 (truly contractive by the
from-below empirical estimate) but L_achievable >= 1 (refined certifier still rejects), if any.

Honest note (in the math): empirical_sup approximates the TRUE box sup FROM BELOW, so a small
positive gap (L_achievable - empirical_sup >= 0) is EXPECTED and is NOT unsoundness. An
over-rejection here (empirical_L < 1 but L_achievable >= 1) is not necessarily a refinement
bug either — it can mean the true box sup is genuinely >= 1 even though the finite empirical
sample missed it. We report counts and the largest gaps without claiming "exact".

We also directly compare the over-rejection counts of FREE vs ACHIEVABLE under the same
empirical_L<1 criterion, to show whether the refinement reduces over-rejection at all.

Run: cd D:/projects/llcore && py -3.11 research/lipschitz_refinement/exp_a4_honest.py
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
)
from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.verifier.invariants import empirical_lipschitz  # noqa: E402

SEED = 0
N_EMP = 5000
OUT = Path(__file__).resolve().parent / "exp_a4_results.json"


def build_genes() -> list[StateUpdateGene]:
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

    genes = build_genes()
    n_total = len(genes)

    gaps_ach = []  # L_achievable - empirical_sup
    gaps_free = []  # L_free - empirical_sup
    over_reject_ach = []  # empirical_L < 1 but L_achievable >= 1
    over_reject_free = []  # empirical_L < 1 but L_free >= 1

    for ge in genes:
        emp = empirical_lipschitz(ge, n_samples=N_EMP, seed=SEED)
        la = L_achievable(ge)
        lf = L_free(ge)
        gaps_ach.append(la - emp)
        gaps_free.append(lf - emp)
        if emp < 1.0 and la >= 1.0:
            over_reject_ach.append(
                {"decay": ge.decay, "mix": ge.mix, "gate_str": ge.gate_str,
                 "empirical_L": emp, "L_achievable": la, "L_free": lf}
            )
        if emp < 1.0 and lf >= 1.0:
            over_reject_free.append(
                {"decay": ge.decay, "mix": ge.mix, "gate_str": ge.gate_str,
                 "empirical_L": emp, "L_achievable": la, "L_free": lf}
            )

    gaps_ach = np.asarray(gaps_ach)
    gaps_free = np.asarray(gaps_free)

    # negative gap (L_achievable < empirical) would be a soundness alarm (unexpected).
    neg_gap_ach = int(np.sum(gaps_ach < -1e-6))

    results = {
        "seed": SEED,
        "n_emp_samples": N_EMP,
        "n_total": n_total,
        "residual_gap_L_achievable_minus_empirical_sup": {
            "mean": float(np.mean(gaps_ach)),
            "median": float(np.median(gaps_ach)),
            "max": float(np.max(gaps_ach)),
            "min": float(np.min(gaps_ach)),
            "p95": float(np.percentile(gaps_ach, 95)),
            "negative_gap_count_soundness_alarm": neg_gap_ach,
        },
        "residual_gap_L_free_minus_empirical_sup": {
            "mean": float(np.mean(gaps_free)),
            "median": float(np.median(gaps_free)),
            "max": float(np.max(gaps_free)),
        },
        "mean_gap_reduction_free_to_achievable": float(np.mean(gaps_free - gaps_ach)),
        "over_rejection": {
            "achievable_over_reject_count": len(over_reject_ach),
            "free_over_reject_count": len(over_reject_free),
            "over_reject_reduction": len(over_reject_free) - len(over_reject_ach),
            "achievable_examples": over_reject_ach[:10],
        },
        "honest_note": (
            "empirical_sup approximates the TRUE box sup FROM BELOW (finite (s,x) sample), "
            "so a small positive gap L_achievable - empirical_sup >= 0 is EXPECTED and is NOT "
            "unsoundness. L_achievable is the EXACT box sup under the achievable-t closed form, "
            "exact only w.r.t. the [-1,1]^2 input box and the per-coordinate diagonal map; "
            "t_min = sech^2(|mix|+|gate_str|) is computed in float64 then fed to Z3 as a rational."
        ),
    }

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== A4 RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
