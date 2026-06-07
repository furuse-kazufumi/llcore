# SPDX-License-Identifier: Apache-2.0
"""Synthetic sequence fitness (PoC 0b の基盤).

State update gene を回した結果の state 軌跡から readout を生成し、
合成 sequence task で fitness (0..1) を計算する。CPU 完結、numpy のみ依存。

Task 設計:
- ``copy_task``     — 入力列 [x_0, ..., x_{T-1}] を delay step 遅延再現
- ``addition_task`` — 入力列の累積和を最終 readout から推定

各 task は :class:`SyntheticTask` で表現され、:func:`evaluate_gene` で
``StateUpdateGene`` 上の fitness を返す。

Readout 設計 (簡素化, PoC level):
- copy: 最終 state から linear projection (固定 random W) で復元
- addition: 最終 state の L2 norm から累積和を近似

honest 留保:
- linear readout は固定 (gene 進化対象外)。本来は readout も進化対象だが
  PoC 0b は state update gene 単体の評価力を測る目的に絞る。
- fitness は MSE ベース、1.0 が完全再現、0.0 が baseline error 以上。
- proxy task mechanism feasibility のみ、実 LLM scale ではない。
"""

from .ridge_readout import (
    RidgeEvalOnce,
    RidgeReadout,
    fit_ridge_readout,
    make_ridge_eval_once,
    ridge_fitness,
)
from .tasks import (
    AdditionTask,
    CopyTask,
    FixedReadout,
    SyntheticTask,
    calibrate_baseline,
    calibrate_baseline_robust,
    evaluate_gene,
    make_fixed_readout,
)

__all__ = [
    "AdditionTask",
    "CopyTask",
    "FixedReadout",
    "RidgeEvalOnce",
    "RidgeReadout",
    "SyntheticTask",
    "calibrate_baseline",
    "calibrate_baseline_robust",
    "evaluate_gene",
    "fit_ridge_readout",
    "make_fixed_readout",
    "make_ridge_eval_once",
    "ridge_fitness",
]
