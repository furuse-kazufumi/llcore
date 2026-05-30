# SPDX-License-Identifier: Apache-2.0
"""TaskMixture / split_regimes の単体テスト (E-A フェーズ)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

# ea_multitask と step_c_memory_tasks を import path に追加
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "step_c_memory_tasks"))

from memory_tasks import DelayedRecallTask, FlipFlopTask  # noqa: E402
from task_mixture import TaskMixture, split_regimes  # noqa: E402


def _flipflop_regimes() -> list[FlipFlopTask]:
    """pulse_prob を振った同一 in_dim=2 の regime 群."""
    return [FlipFlopTask(seq_len=30, pulse_prob=p) for p in (0.1, 0.2, 0.3, 0.4)]


def test_generate_contract_shapes() -> None:
    mix = TaskMixture(_flipflop_regimes())
    rng = np.random.default_rng(0)
    inputs, target = mix.generate(rng)
    assert inputs.ndim == 2 and inputs.shape[1] == mix.in_dim == 2
    assert target.shape == (mix.out_dim,) == (1,)
    assert np.all(np.isfinite(inputs)) and np.all(np.isfinite(target))


def test_in_out_dim_inferred() -> None:
    mix = TaskMixture(_flipflop_regimes())
    assert mix.in_dim == 2
    assert mix.out_dim == 1


def test_mismatched_in_dim_raises() -> None:
    # FlipFlop(in_dim=2) と DelayedRecall(in_dim=1) の混合は禁止
    with pytest.raises(ValueError, match="in_dim"):
        TaskMixture([FlipFlopTask(), DelayedRecallTask()])


def test_empty_regimes_raises() -> None:
    with pytest.raises(ValueError, match="at least one"):
        TaskMixture([])


def test_uniform_weights_default() -> None:
    mix = TaskMixture(_flipflop_regimes())
    assert np.allclose(mix._probs, 0.25)


def test_weights_normalized() -> None:
    mix = TaskMixture(_flipflop_regimes(), weights=np.array([2.0, 2.0, 2.0, 2.0]))
    assert np.allclose(mix._probs, 0.25)
    assert pytest.approx(mix._probs.sum()) == 1.0


def test_weights_wrong_shape_raises() -> None:
    with pytest.raises(ValueError, match="weights shape"):
        TaskMixture(_flipflop_regimes(), weights=np.array([0.5, 0.5]))


def test_negative_weights_raise() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        TaskMixture(_flipflop_regimes(), weights=np.array([1.0, -1.0, 1.0, 1.0]))


def test_zero_sum_weights_raise() -> None:
    with pytest.raises(ValueError, match="positive value"):
        TaskMixture(_flipflop_regimes(), weights=np.zeros(4))


def test_zero_weight_regime_never_selected() -> None:
    # regime 1 のみ weight=1、他は 0 → 必ず regime 1 (pulse_prob=0.2) のみ
    regimes = _flipflop_regimes()
    mix = TaskMixture(regimes, weights=np.array([0.0, 1.0, 0.0, 0.0]))
    rng = np.random.default_rng(42)
    for _ in range(50):
        idx = int(rng.choice(len(mix.regimes), p=mix._probs))
        assert idx == 1


def test_generate_deterministic_with_seed() -> None:
    mix = TaskMixture(_flipflop_regimes())
    a_in, a_tg = mix.generate(np.random.default_rng(7))
    b_in, b_tg = mix.generate(np.random.default_rng(7))
    assert np.array_equal(a_in, b_in)
    assert np.array_equal(a_tg, b_tg)


def test_split_regimes_disjoint() -> None:
    regimes = _flipflop_regimes()
    train, test = split_regimes(regimes, test_idx=[1, 3])
    assert len(train) == 2 and len(test) == 2
    # train = idx 0,2 (pulse 0.1, 0.3) / test = idx 1,3 (pulse 0.2, 0.4)
    assert [r.pulse_prob for r in train] == [0.1, 0.3]
    assert [r.pulse_prob for r in test] == [0.2, 0.4]


def test_split_regimes_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="out of range"):
        split_regimes(_flipflop_regimes(), test_idx=[9])


def test_split_regimes_must_leave_both_nonempty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        split_regimes(_flipflop_regimes(), test_idx=[0, 1, 2, 3])


def test_split_regimes_empty_test_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        split_regimes(_flipflop_regimes(), test_idx=[])
