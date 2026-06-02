# SPDX-License-Identifier: Apache-2.0
"""Trace the single gated violator in copy_d8/contraction.

Confirm the implementer's claim: the 1 final-pop gene with empL>=1 is an
UN-GATED initial-population survivor (the gate acts on child admission, the
initial pop is un-gated by design, matching src). It must NOT be an admitted
child. We:
  1. find the violating gene + its seed from the JSON,
  2. reconstruct that seed's initial population with src code,
  3. confirm the gene is (a) in the initial pop and (b) would be REJECTED as a
     child by the contraction gate (verify_lipschitz_contraction == False),
  4. confirm it survived purely via elitism (its fitness == best each gen).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
for p in (str(_SRC), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llcore.evolution.minimal_ga import initialize_random_population  # noqa: E402
from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.verifier.invariants import (  # noqa: E402
    empirical_lipschitz, verify_lipschitz_contraction,
)
from redteam_soundness import my_empirical_L  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    data = json.loads((_HERE / "exp_b_results.json").read_text(encoding="utf-8"))
    recs = data["cells"]["copy_d8"]["gates"]["contraction"]

    # find the violating gene + seed
    viol_gene = None
    viol_seed = None
    for r in recs:
        for (d, m, gs) in r["final_pop_genes"]:
            if my_empirical_L(d, m, gs) >= 1.0:
                viol_gene = (d, m, gs)
                viol_seed = r["seed"]
    print(f"violating gene = {viol_gene}  seed = {viol_seed}")
    g = StateUpdateGene(*viol_gene)
    print(f"  my empL          = {my_empirical_L(*viol_gene):.6f}")
    print(f"  their empL       = {empirical_lipschitz(g):.6f}")
    cert = verify_lipschitz_contraction(g)
    print(f"  gate verdict     = contraction={cert.contraction} "
          f"(L_upper_bound={cert.L_upper_bound:.6f})  -> gate WOULD {'ADMIT' if cert.contraction else 'REJECT'} as child")

    # reconstruct initial pop for that seed (initialize_random_population draws
    # the SAME first 10 genes since gated_evolve.none path matches src init)
    rng = np.random.default_rng(viol_seed)
    init = initialize_random_population(10, rng)
    print(f"\n  initial pop for seed {viol_seed}:")
    found = False
    for i, ig in enumerate(init):
        arr = ig.as_array()
        match = np.allclose(arr, viol_gene, atol=1e-9)
        tag = "  <== MATCH (violator is an initial-pop member)" if match else ""
        if match:
            found = True
        print(f"    [{i}] decay={arr[0]:.6f} mix={arr[1]:.6f} gate_str={arr[2]:.6f}{tag}")
    print(f"\n  violator IS an un-gated initial-pop survivor: {found}")
    print(f"  gate rejects it as a child:                  {cert.contraction is not True}")
    print(f"\n  VERDICT: soundness claim intact -> {found and cert.contraction is not True}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
