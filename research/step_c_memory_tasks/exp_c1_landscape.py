# exp_c1_landscape.py
# SPDX-License-Identifier: Apache-2.0
"""C1: 各記憶タスクの landscape が多峰か (reservoir gene 空間で).

実行＝測定。3 タスク (delayed parity / flip-flop / delayed recall) × leaky-delay-line
reservoir 基質で、収束点間の中点が谷になる割合 (valley_fraction) を測り多峰性を判定する。

判断分岐:
- いずれかのタスクで is_multimodal=True → そのタスクを Task 6 (exp_c2c3) へ。
- 全タスクで False (滑らか) → STEP_C_VERDICT へ (③ は実タスク proxy では不要の証拠)。
"""
from __future__ import annotations

from memory_tasks import DelayedParityTask, DelayedRecallTask, FlipFlopTask
from landscape_map import multimodality_report
from reservoir import LeakyDelayLineReservoir, gene_bounds, make_eval_once

TASKS = {
    "delayed_parity": DelayedParityTask(seq_len=20, window=5, in_dim=1),
    "flip_flop": FlipFlopTask(seq_len=30, in_dim=2),
    "delayed_recall": DelayedRecallTask(seq_len=25, in_dim=1),
}


def main() -> None:
    any_multimodal = False
    for name, task in TASKS.items():
        in_dim = task.in_dim
        res = LeakyDelayLineReservoir(n_taps=8, in_dim=in_dim)
        eval_once = make_eval_once(res, task, n_train=48, n_eval=48)
        lo, hi = gene_bounds(res)
        rep = multimodality_report(
            eval_once, dim=res.gene_dim, bounds=(lo, hi),
            n_restarts=12, n_evals=400, sigma=0.15, base_seed=20260530,
        )
        any_multimodal = any_multimodal or rep["is_multimodal"]
        print(f"[{name}] n_optima={rep['n_optima']} "
              f"valley_fraction={rep['valley_fraction']:.3f} "
              f"is_multimodal={rep['is_multimodal']}")
    print(f"\n=> any_multimodal={any_multimodal} "
          f"({'Task 6 へ' if any_multimodal else 'STEP_C_VERDICT (撤退判定) へ'})")


if __name__ == "__main__":
    main()
