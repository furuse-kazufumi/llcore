# SPDX-License-Identifier: Apache-2.0
"""Descriptor-dependence smoke test: confirm imports + cheap single estimate.

Read-only on src / task modules. Writes nothing except stdout.
Goal: prove the import chain resolves and one cheap elite_dip estimate runs,
so the full descriptor sweep (descriptor_dep_sweep.py) is feasible.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (_HERE,
          _HERE.parents[0] / "step_c_memory_tasks",
          _HERE.parents[0] / "ea_multitask",
          _HERE.parents[0] / "ea_multitask" / "candidates",
          _HERE.parents[0] / "step6_real_proxy",
          _HERE.parents[1] / "src"):
    sys.path.insert(0, str(p))

from metric_behavior_elite_dip import deceptiveness  # single-estimate (cheap)

def main() -> int:
    from reservoir import (LeakyDelayLineReservoir, gene_bounds, make_behavior,
                           make_eval_once)
    from memory_tasks import FlipFlopTask

    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim
    print(f"OK imports; reservoir gene_dim={dim}", flush=True)

    ev_ff = make_eval_once(res, FlipFlopTask(), n_train=48, n_eval=48)
    rng = np.random.default_rng(0)
    t0 = time.time()
    # tiny budget just to confirm it runs and time it
    v = deceptiveness(ev_ff, behavior, bounds, dim, rng,
                      n_samples=120, n_bins=16, honest_n_trials=2, min_per_bin=4)
    dt = time.time() - t0
    print(f"flip_flop elite_dip(n=120,bins=16,trials=2) = {v:.4f}  in {dt:.1f}s", flush=True)
    print(f"EST per-eval ~{dt/(120*2)*1000:.0f}ms", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
