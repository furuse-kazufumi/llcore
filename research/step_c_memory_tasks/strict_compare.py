# SPDX-License-Identifier: Apache-2.0
"""強化版 honest 基準で 2 スコア配列を判定 (selection_lab.compare の厳格版).

合格 = diff>0 ∧ 片側 Wilcoxon p<alpha ∧ n_seeds>=min_seeds ∧ |paired_sign_delta|>=min_effect。
honest_eval の片側 _paired_p / _paired_sign_delta を流用 (基準は監査 §5 = 強化版 passes と同一)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llcore.evolution.honest_eval import _paired_p, _paired_sign_delta  # noqa: E402


@dataclass(frozen=True)
class StrictComparison:
    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    diff: float
    win_rate: float
    wilcoxon_p: float
    paired_sign_delta: float
    n_seeds: int
    passes: bool


def strict_compare(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str,
    name_b: str,
    *,
    alpha: float = 0.05,
    min_seeds: int = 15,
    min_effect: float = 0.147,
) -> StrictComparison:
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    deltas = a - b
    diff = float(np.mean(deltas))
    p = _paired_p(a, b)
    delta = _paired_sign_delta(deltas)
    passes = bool(
        diff > 0.0 and p < alpha and len(a) >= min_seeds and abs(delta) >= min_effect
    )
    return StrictComparison(
        name_a=name_a, name_b=name_b,
        mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
        diff=diff, win_rate=float(np.mean(a > b)), wilcoxon_p=p,
        paired_sign_delta=delta, n_seeds=len(a), passes=passes,
    )
