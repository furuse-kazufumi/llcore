# SPDX-License-Identifier: Apache-2.0
"""実 task の dip メトリクスを軽量サンプリングで素早く測る (CI を必ず併記).

重い full 設定 (n_samples=1200, honest_n=10) は CPU で長時間かかるため、quick 設定
(n_samples, honest_n を絞る) で点推定 + CI を出す。ノイズが大きい点は CI で明示。
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "step_c_memory_tasks"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask" / "candidates"))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from metric_behavior_elite_dip import deceptiveness_estimate  # noqa: E402
from reservoir import (LeakyDelayLineReservoir, gene_bounds, make_behavior,  # noqa: E402
                       make_eval_once)
from memory_tasks import FlipFlopTask  # noqa: E402
from task_mixture import TaskMixture  # noqa: E402
from variable_delay_recall import VariableDelayRecallTask  # noqa: E402

# quick 設定。CI で noise を見せる前提で軽くする。
N_SAMPLES = 400
N_BINS = 12
N_SEEDS = 3
HONEST_N = 4

if __name__ == "__main__":
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim
    out = {}

    t0 = time.time()
    train_regimes = [VariableDelayRecallTask(seq_len=D, distractor_amp=0.2, in_dim=2)
                     for D in (15, 30)]
    ev_ea = make_eval_once(res, TaskMixture(train_regimes), n_train=24, n_eval=24)
    est_ea = deceptiveness_estimate(ev_ea, behavior, bounds, dim,
                                    n_seeds=N_SEEDS, n_samples=N_SAMPLES,
                                    n_bins=N_BINS, honest_n_trials=HONEST_N)
    out["ea_multitask_variable_delay_recall"] = est_ea.as_dict()
    print(f"E-A multitask: dip={est_ea.mean:.4f} ± {est_ea.std:.4f} "
          f"95%CI [{est_ea.ci95_lo:.4f},{est_ea.ci95_hi:.4f}]  per_seed={est_ea.per_seed} "
          f"({time.time()-t0:.0f}s)", flush=True)

    t1 = time.time()
    ev_ff = make_eval_once(res, FlipFlopTask(), n_train=24, n_eval=24)
    est_ff = deceptiveness_estimate(ev_ff, behavior, bounds, dim,
                                    n_seeds=N_SEEDS, n_samples=N_SAMPLES,
                                    n_bins=N_BINS, honest_n_trials=HONEST_N)
    out["flip_flop"] = est_ff.as_dict()
    print(f"flip_flop    : dip={est_ff.mean:.4f} ± {est_ff.std:.4f} "
          f"95%CI [{est_ff.ci95_lo:.4f},{est_ff.ci95_hi:.4f}]  per_seed={est_ff.per_seed} "
          f"({time.time()-t1:.0f}s)", flush=True)

    out["quick_config"] = {"n_samples": N_SAMPLES, "n_bins": N_BINS,
                           "n_seeds": N_SEEDS, "honest_n": HONEST_N,
                           "note": "QUICK light-sampling estimate; CIs are wide; "
                                   "treat as order-of-magnitude only."}
    (_HERE / "real_quick_results.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote real_quick_results.json", flush=True)
