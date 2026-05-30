# exp_c4_ablation.py
# SPDX-License-Identifier: Apache-2.0
"""C4: MAP-E の勝因が coverage でなく archive ratchet か (init_batch を変えて確認).

C1 で多峰だったタスク (delayed_parity / flip_flop) で init_batch を 30/200/1000 と
変えて honest 再評価到達を比較する。

CONFOUND (Codex pair-review BLOCKER 2026-05-30): n_evals 固定のため init_batch を
増やすと ratchet 段数 (= n_evals - init_batch) が同時に減る。init_batch=1000 は
「random coverage 重・ratchet 軽 (1000 段)」、init_batch=30 は「ratchet 重 (1970 段)」
の対比。よって「小 init_batch=ratchet 由来 / 大=coverage 由来」と単純帰属はできない
(初期 coverage と ratchet 段数を分離した識別ではない)。

honest な読み方 (方向性のみ): init_batch=1000 は coverage が最も厚い (random 1000 点)
にもかかわらず到達が低いなら、その差分は ratchet 段数の寄与に帰属できる (coverage の
不足では説明不能)。逆に init_batch=1000 が 30 と同等以上なら ratchet の追加寄与は小さい。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Windows cp932 console で R²/日本語 を出力するため UTF-8 へ reconfigure
# (memory: feedback_cli_utf8_stdout_pattern)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

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
        # 方向性の読み (docstring CONFOUND 参照): init_batch=1000 (coverage 最厚・ratchet
        # 軽) が 30 (ratchet 重) より低ければ差分=ratchet 寄与。同等以上なら ratchet 寄与小。


if __name__ == "__main__":
    main()
