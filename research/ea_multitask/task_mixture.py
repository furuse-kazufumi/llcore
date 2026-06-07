# SPDX-License-Identifier: Apache-2.0
"""TaskMixture — regime 構造を持つタスク族 (「風/天候」分布) と train/test 分離.

E-A フェーズ (E_A_DESIGN_multitask_generalization.md) の土俵。単一タスクでなく、
regime パラメータ (例: FlipFlop の pulse_prob/seq_len) で連続変化するタスク族を 1 つの
分布として扱う。③(選択圧/分離)の寄与を **未知 regime への hold-out 汎化**で測るため、
学習に使う regime と評価に使う regime を **draw レベルで完全分離** する。

設計 (honest):
- ``TaskMixture`` は既存タスクと同じ ``generate(rng) -> (inputs, target)`` 契約を満たす。
  → 既存 ``make_eval_once`` / reservoir を一切改変せずそのまま使える。
- 混合する regime は **同一 in_dim/out_dim** でなければならない (1 つの reservoir に
  入れるため)。異種 in_dim の混合は本スコープ外 (統一入力符号化が別途必要)。
- seq_len が regime ごとに違っても可 (reservoir は可変長を処理し最終状態を取る)。
- train regimes と test regimes は **互いに素**であることを ``split_regimes`` で保証する
  (汎化の水増し=リーク防止)。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np


class Task(Protocol):
    """generate(rng) -> (inputs (seq_len, in_dim), target (out_dim,)) を満たすタスク契約."""

    in_dim: int
    out_dim: int

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass(frozen=True)
class TaskMixture:
    """複数 regime を weights で混合する分布タスク (generate 契約は単一タスクと同一).

    Attributes
    ----------
    regimes : Sequence[Task]
        混合する regime 群。全て同一 in_dim/out_dim でなければならない。
    weights : np.ndarray | None
        各 regime の選択確率。None なら一様。和は 1 に正規化される。
    """

    regimes: Sequence[Task]
    weights: np.ndarray | None = None
    in_dim: int = field(init=False)
    out_dim: int = field(init=False)
    _probs: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if len(self.regimes) == 0:
            raise ValueError("TaskMixture requires at least one regime")
        in_dims = {int(r.in_dim) for r in self.regimes}
        out_dims = {int(r.out_dim) for r in self.regimes}
        if len(in_dims) != 1:
            raise ValueError(f"all regimes must share in_dim, got {sorted(in_dims)}")
        if len(out_dims) != 1:
            raise ValueError(f"all regimes must share out_dim, got {sorted(out_dims)}")
        object.__setattr__(self, "in_dim", in_dims.pop())
        object.__setattr__(self, "out_dim", out_dims.pop())

        if self.weights is None:
            probs = np.full(len(self.regimes), 1.0 / len(self.regimes), dtype=np.float64)
        else:
            w = np.asarray(self.weights, dtype=np.float64)
            if w.shape != (len(self.regimes),):
                raise ValueError(
                    f"weights shape {w.shape} != ({len(self.regimes)},)"
                )
            if np.any(w < 0):
                raise ValueError("weights must be non-negative")
            total = float(w.sum())
            if total <= 0:
                raise ValueError("weights must sum to a positive value")
            probs = w / total
        object.__setattr__(self, "_probs", probs)

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """weights に従って 1 regime を選び、その regime の系列を 1 本生成する."""
        idx = int(rng.choice(len(self.regimes), p=self._probs))
        return self.regimes[idx].generate(rng)


def split_regimes(
    regimes: Sequence[Task], test_idx: Sequence[int]
) -> tuple[list[Task], list[Task]]:
    """regime 群を train/test に互いに素分割する (hold-out 汎化用).

    Parameters
    ----------
    regimes : Sequence[Task]
        全 regime。
    test_idx : Sequence[int]
        test に回す regime のインデックス集合。残りが train。

    Returns
    -------
    (train_regimes, test_regimes) : tuple[list[Task], list[Task]]
        互いに素。どちらも空でないことを保証する。
    """
    test_set = set(int(i) for i in test_idx)
    n = len(regimes)
    if any(i < 0 or i >= n for i in test_set):
        raise ValueError(f"test_idx {sorted(test_set)} out of range [0,{n})")
    train = [r for i, r in enumerate(regimes) if i not in test_set]
    test = [r for i, r in enumerate(regimes) if i in test_set]
    if not train or not test:
        raise ValueError(
            f"split must leave both non-empty (n={n}, test={sorted(test_set)})"
        )
    return train, test


__all__ = ["Task", "TaskMixture", "split_regimes"]
