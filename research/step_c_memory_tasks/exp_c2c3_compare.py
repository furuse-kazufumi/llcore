# exp_c2c3_compare.py
# SPDX-License-Identifier: Apache-2.0
"""C2/C3: 多峰タスクで MAP-E が 3 baseline を強化基準で上回るか.

C1 (exp_c1_landscape) で多峰だったタスク = delayed_parity (valley=1.000) /
flip_flop (valley=0.939) を対象にする。delayed_recall は滑らか (valley=0.000)
だったため除外。各 method は同一 n_evals 予算、best gene を進化と独立な fresh
seed で honest 再評価 (elitism artifact 排除)。判定は強化版 honest 基準 (片側
Wilcoxon p<0.05・|δ|≥0.147・n_seeds≥15) を strict_compare で適用。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))

from memory_tasks import DelayedParityTask, FlipFlopTask
from reservoir import LeakyDelayLineReservoir, gene_bounds, make_behavior, make_eval_once
from strict_compare import strict_compare
from selection_lab import run_methods_over_seeds

# C1 で多峰と確定したタスクのみ。
TASKS = {
    "delayed_parity": DelayedParityTask(seq_len=20, window=5, in_dim=1),
    "flip_flop": FlipFlopTask(seq_len=30, in_dim=2),
}


def main() -> None:
    for name, task in TASKS.items():
        res = LeakyDelayLineReservoir(n_taps=8, in_dim=task.in_dim)
        eval_once = make_eval_once(res, task, n_train=48, n_eval=48)
        behavior = make_behavior(res)
        lo, hi = gene_bounds(res)
        scores = run_methods_over_seeds(
            eval_once, behavior, dim=res.gene_dim, bounds=(lo, hi),
            behavior_bounds=(np.array([0.0, 0.0]), np.array([1.0, 0.5])),
            grid_shape=(12, 12), n_evals=2000, n_seeds=15, honest_n_trials=20,
            sigma=0.15, base_seed=20260530,
        )
        print(f"\n[{name}] mean honest R²: "
              f"MAP-E={np.mean(scores['map_elites']):.4f} "
              f"random={np.mean(scores['random']):.4f} "
              f"RR={np.mean(scores['rr_hillclimb']):.4f} "
              f"GA={np.mean(scores['panmictic_ga']):.4f}")
        all_pass = True
        for base in ("random", "rr_hillclimb", "panmictic_ga"):
            r = strict_compare(scores["map_elites"], scores[base], "map_elites", base)
            all_pass = all_pass and r.passes
            print(f"  MAP-E vs {base}: diff={r.diff:+.4f} p={r.wilcoxon_p:.4g} "
                  f"δ={r.paired_sign_delta:+.2f} passes={r.passes}")
        print(f"  => C3 ({name}): {'成立 (③ load-bearing)' if all_pass else '不成立'}")


if __name__ == "__main__":
    main()
