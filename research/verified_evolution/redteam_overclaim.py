# SPDX-License-Identifier: Apache-2.0
"""Over-claim / circularity attack.

(1) CONTROL on the REAL task fitness func (not the toy ff in assert_none_matches_src):
    confirm gated_evolve('none') == src evolve() byte-identical on copy_d0/copy_d8
    fitness, across several seeds. (The implementer's control used a toy ff.)

(2) CIRCULARITY: is the Z3 gate decorative? Show verify_lipschitz_contraction's
    verdict == (closed-form L_upper_bound < 1) for a large random sample, i.e. the
    "Z3 proof" is exactly the closed-form inequality. This is SOUND but means the
    soundness rests on the closed-form math + the over-approx argument, not on Z3
    discovering anything. Honest framing matters.

(3) OVER-APPROX TIGHTNESS: how often does the achievable empirical_L actually
    reach the certified L_upper_bound? If the bound is loose, "conservative
    safety cost" could be overstated; if tight, the cost is real. Report the
    distribution of (cert_bound - empirical_L) for admitted near-boundary genes,
    and the false-REJECT rate (genes rejected by gate but empirically L<1).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parents[1] / "src"
for p in (str(_SRC), str(_HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llcore.evolution import evolve as src_evolve  # noqa: E402
from llcore.fitness import (  # noqa: E402
    CopyTask, calibrate_baseline, evaluate_gene, make_fixed_readout,
)
from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.verifier.invariants import (  # noqa: E402
    verify_lipschitz_contraction, _lipschitz_upper_bound,
)
from gated_evolve import gated_evolve  # noqa: E402
from redteam_soundness import my_empirical_L  # noqa: E402


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    readout = make_fixed_readout(8, 8, seed=1001)
    GA_KW = dict(pop_size=10, n_generations=10, tournament_k=3, mutation_sigma=0.15,
                 crossover_rate=0.5, elitism=1)

    # ---- (1) control on REAL task fitness ----------------------------------
    print("=== (1) control gated('none')==src evolve() on REAL task fitness ===")
    for name, delay in (("copy_d0", 0), ("copy_d8", 8)):
        t = CopyTask(state_dim=8, out_dim=8, delay=delay)
        t = type(t)(state_dim=8, out_dim=8, delay=delay,
                    baseline_mse=calibrate_baseline(t, readout))

        def ff(gene, rng):
            return evaluate_gene(gene, t, readout, rng, n_trials=5)

        all_ok = True
        for s in (1000, 1007, 1013, 2000, 3000):
            r_src = src_evolve(ff, rng=np.random.default_rng(s), **GA_KW)
            r_gat = gated_evolve(ff, gate_mode="none", rng=np.random.default_rng(s),
                                 resample_cap=50, **GA_KW)
            ok = (r_src.best_fitness_curve == r_gat.result.best_fitness_curve
                  and r_src.diversity_curve == r_gat.result.diversity_curve)
            all_ok = all_ok and ok
        print(f"  {name}: byte-identical across 5 seeds = {all_ok}")

    # ---- (2) circularity: Z3 verdict == closed-form inequality -------------
    print("\n=== (2) Z3 contraction verdict vs closed-form (L_upper_bound<1) ===")
    rng = np.random.default_rng(11)
    mism = 0
    n = 20000
    for _ in range(n):
        d = float(rng.uniform(0, 1)); m = float(rng.uniform(-1, 1)); gs = float(rng.uniform(-2, 2))
        g = StateUpdateGene(d, m, gs)
        cert = verify_lipschitz_contraction(g)
        cf = _lipschitz_upper_bound(d, gs) < 1.0
        if (cert.contraction is True) != cf:
            mism += 1
    print(f"  mismatches Z3 vs closed-form over {n} genes: {mism}")
    print(f"  -> Z3 'proof' is the closed-form inequality L_upper_bound<1 "
          f"({'decorative wrt the bound' if mism == 0 else 'adds discrimination'})")

    # ---- (3) over-approx tightness + false-reject rate ----------------------
    print("\n=== (3) over-approx tightness & false-reject rate (gate conservatism) ===")
    rng = np.random.default_rng(13)
    gaps_admit = []
    false_reject = 0  # gate REJECTS but empirical_L < 1
    rejected = 0
    for _ in range(8000):
        d = float(rng.uniform(0, 1)); m = float(rng.uniform(-1, 1)); gs = float(rng.uniform(-2, 2))
        g = StateUpdateGene(d, m, gs)
        cert = verify_lipschitz_contraction(g)
        empL = my_empirical_L(d, m, gs)
        if cert.contraction is True:
            gaps_admit.append(cert.L_upper_bound - empL)
        else:
            rejected += 1
            if empL < 1.0:
                false_reject += 1
    gaps_admit = np.array(gaps_admit)
    print(f"  admitted gene gap (cert_bound - empirical_L): "
          f"median={np.median(gaps_admit):.4f} min={gaps_admit.min():.4f} "
          f"p95={np.percentile(gaps_admit,95):.4f}")
    print(f"  false-REJECT rate (rejected but empL<1): {false_reject}/{rejected} "
          f"= {false_reject/max(rejected,1):.3f}")
    print("  (false rejects are EXPECTED & sound: conservative over-approx; this")
    print("   quantifies how much 'safe but forbidden' fitness territory the gate cuts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
