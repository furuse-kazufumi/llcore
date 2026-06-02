# SPDX-License-Identifier: Apache-2.0
"""exp_a1_a2.py — Track A gates A1 (containment) and A2 (strict gain).

Over a deterministic grid AND a random sample (seed=0) of clipped genes covering the full
box (decay in [0,1], mix in [-1,1], gate_str in [-2,2]; >= 4000 genes total), compute for
each gene whether the FREE-t certifier and the ACHIEVABLE-t certifier each certify
contraction (L < 1). Then measure:

  A1 (containment): #(free-certified AND NOT achievable-certified)  -- MUST be 0.
  A2 (strict gain): #(achievable-certified AND NOT free-certified)  -- count + fraction.

Also reports the BOUND-VALUE refinement (L_achievable < L_free count + max delta), which is
the genuine, measurable contribution of the refinement even when the binary certification
decision is unchanged. (Honest disclosure: see A_VERDICT.md — both certifiers share the
J(1) endpoint, so binary certification gain is structurally 0; the win is the tighter bound.)

Run: cd D:/projects/llcore && py -3.11 research/lipschitz_refinement/exp_a1_a2.py
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
    verify_lipschitz_contraction_free,
)
from llcore.state_update import StateUpdateGene  # noqa: E402

SEED = 0
OUT = Path(__file__).resolve().parent / "exp_a1_a2_results.json"


def build_genes() -> list[StateUpdateGene]:
    """Deterministic grid + random sample (seed=0), >= 4000 genes total, full box."""
    genes: list[StateUpdateGene] = []

    # Deterministic grid: include exact boundaries (0,1 / -1,1 / -2,2) and zero-crossings.
    decays = np.linspace(0.0, 1.0, 16)
    mixes = np.linspace(-1.0, 1.0, 16)
    gates = np.linspace(-2.0, 2.0, 16)
    for d in decays:
        for m in mixes:
            for g in gates:
                genes.append(StateUpdateGene(float(d), float(m), float(g)).clipped())
    n_grid = len(genes)  # 16^3 = 4096

    # Random sample (seed=0) over the full box.
    rng = np.random.default_rng(SEED)
    n_rand = 2000
    d = rng.uniform(0.0, 1.0, n_rand)
    m = rng.uniform(-1.0, 1.0, n_rand)
    g = rng.uniform(-2.0, 2.0, n_rand)
    for i in range(n_rand):
        genes.append(StateUpdateGene(float(d[i]), float(m[i]), float(g[i])).clipped())

    return genes, n_grid, n_rand


def main() -> int:
    if not is_z3_available():
        print("FATAL: z3 not available — cannot run certifiers.")
        return 1

    genes, n_grid, n_rand = build_genes()
    n_total = len(genes)
    print(f"genes total = {n_total} (grid={n_grid}, random seed={SEED} ={n_rand})")

    free_cert = np.zeros(n_total, dtype=bool)
    ach_cert = np.zeros(n_total, dtype=bool)
    lfree = np.zeros(n_total)
    lach = np.zeros(n_total)

    for i, ge in enumerate(genes):
        rf = verify_lipschitz_contraction_free(ge)
        ra = verify_lipschitz_contraction_achievable(ge)
        free_cert[i] = bool(rf.contraction)
        ach_cert[i] = bool(ra.contraction)
        lfree[i] = L_free(ge)
        lach[i] = L_achievable(ge)
        if i % 1000 == 0:
            print(f"  ... {i}/{n_total}")

    # A1 containment: free-certified must be subset of achievable-certified.
    containment_violations = int(np.sum(free_cert & ~ach_cert))
    # A2 strict gain: achievable certifies strictly more.
    strict_gain = int(np.sum(ach_cert & ~free_cert))

    n_free = int(np.sum(free_cert))
    n_ach = int(np.sum(ach_cert))

    # Bound-value refinement (the genuine measurable win).
    bound_delta = lfree - lach  # >= 0 by construction (L_ach <= L_free)
    tighter_count = int(np.sum(bound_delta > 1e-9))
    max_bound_delta = float(np.max(bound_delta))
    mean_bound_delta_when_tighter = (
        float(np.mean(bound_delta[bound_delta > 1e-9])) if tighter_count else 0.0
    )
    # Sanity: L_ach <= L_free everywhere (refinement never increases bound).
    bound_increase_violations = int(np.sum(bound_delta < -1e-9))

    results = {
        "seed": SEED,
        "n_total": n_total,
        "n_grid": n_grid,
        "n_random": n_rand,
        "z3_version": str(__import__("z3").get_version_string()),
        "A1_containment_violations": containment_violations,
        "A1_pass": containment_violations == 0,
        "A2_strict_gain_count": strict_gain,
        "A2_strict_gain_fraction": strict_gain / n_total,
        "A2_pass": strict_gain > 0,
        "n_free_certified": n_free,
        "n_achievable_certified": n_ach,
        "certified_sets_identical": n_free == n_ach and strict_gain == 0 and containment_violations == 0,
        "bound_refinement": {
            "L_achievable_le_L_free_violations": bound_increase_violations,
            "tighter_bound_count": tighter_count,
            "tighter_bound_fraction": tighter_count / n_total,
            "max_bound_delta": max_bound_delta,
            "mean_bound_delta_when_tighter": mean_bound_delta_when_tighter,
        },
    }

    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n=== A1 / A2 RESULTS ===")
    print(json.dumps(results, indent=2))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
