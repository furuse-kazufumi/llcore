# exp_c4_ablation.py
# SPDX-License-Identifier: Apache-2.0
"""C4: MAP-E の勝因が coverage でなく archive ratchet か (init_batch を変えて確認).

C1 で多峰だったタスク (delayed_parity / flip_flop) で init_batch を 30/200/1000 と
変えて honest 再評価到達を比較する。小 init_batch でも到達するなら勝因は archive
ratchet (= ③ の差し survival 経由)、大 init_batch でしか到達しないなら単なる
初期 coverage 由来。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))

from memory_tasks import DelayedParityTask, FlipFlopTask
from reservoir import LeakyDelayLineReservoir, gene_bounds, make_behavior, make_eval_once
from selection_lab import map_elites
from llcore.evolution.honest_eval import honest_reevaluate

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
        print(f"\n[{name}]")
        for init_batch in (30, 200, 1000):
            vals = []
            for s in range(10):
                r = map_elites(
                    eval_once, behavior, dim=res.gene_dim, bounds=(lo, hi),
                    behavior_bounds=(np.array([0.0, 0.0]), np.array([1.0, 0.5])),
                    grid_shape=(12, 12), n_evals=2000, init_batch=init_batch,
                    sigma=0.15, rng=np.random.default_rng(20260530 + s),
                )
                vals.append(honest_reevaluate(eval_once, r.best_gene, n_trials=20,
                                              rng=np.random.default_rng(99 + s)))
            print(f"  init_batch={init_batch}: mean honest R²={np.mean(vals):.4f}")
        # 小 init_batch でも到達 → ratchet 由来 (C4 成立) / 大でしか到達 → coverage 由来。


if __name__ == "__main__":
    main()
