# SPDX-License-Identifier: Apache-2.0
"""memory_tasks の薄い再 export — sibling research dir (step_c_memory_tasks) から流用.

`research/step_c_memory_tasks/memory_tasks.py` の FlipFlop/DelayedRecall/DelayedParity を
import path に通して再 export するだけの helper。reimplement せず既存 task 定義を流用する
(STAGE_3B_DESIGN の「読んでインターフェースを把握し reimplement せず流用」規律)。

各 task は ``generate(rng) -> (inputs (L,in_dim), target (out_dim,))`` 契約を満たす。
in_dim/out_dim は task 属性。BG6 では smoke の小さめ既定 (memory_tasks の dataclass 既定値)
をそのまま使う。
"""
from __future__ import annotations

import sys
from pathlib import Path

# sibling research dir を import path に
_MEM = str(Path(__file__).resolve().parents[1] / "step_c_memory_tasks")
if _MEM not in sys.path:
    sys.path.insert(0, _MEM)

from memory_tasks import (  # noqa: E402
    DelayedParityTask,
    DelayedRecallTask,
    FlipFlopTask,
)


def flipflop_task() -> FlipFlopTask:
    """set/reset パルスの最終値を保持 (in_dim=2)。mamba selective が得意と仮説 (DESIGN §1)."""
    return FlipFlopTask()


def delayed_recall_task() -> DelayedRecallTask:
    """t=0 cue を遅延後に想起 (in_dim=1)。hopfield 連想が得意と仮説 (DESIGN §1)."""
    return DelayedRecallTask()


def delayed_parity_task() -> DelayedParityTask:
    """先頭 window の XOR (in_dim=1)。非線形長期依存。"""
    return DelayedParityTask()


# BG6 の task 一覧 (名前 -> ファクトリ)。順序固定 (再現性)。
MEMORY_TASKS = {
    "flipflop": flipflop_task,
    "delayed_recall": delayed_recall_task,
    "delayed_parity": delayed_parity_task,
}


__all__ = [
    "FlipFlopTask",
    "DelayedRecallTask",
    "DelayedParityTask",
    "flipflop_task",
    "delayed_recall_task",
    "delayed_parity_task",
    "MEMORY_TASKS",
]
