import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_tasks import DelayedParityTask, FlipFlopTask, DelayedRecallTask


def test_delayed_parity_shapes_and_label():
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    rng = np.random.default_rng(0)
    inputs, target = task.generate(rng)
    assert inputs.shape == (20, 1)
    assert set(np.unique(inputs)).issubset({-1.0, 1.0})
    window_bits = inputs[:5, 0]
    n_neg = int(np.sum(window_bits < 0))
    expected = 1.0 if n_neg % 2 == 0 else -1.0
    assert float(np.atleast_1d(target)[0]) == expected


def test_flipflop_holds_last_set():
    task = FlipFlopTask(seq_len=30, in_dim=2)
    rng = np.random.default_rng(1)
    inputs, target = task.generate(rng)
    assert inputs.shape == (30, 2)
    state = 0.0
    for t in range(30):
        if inputs[t, 0] > 0:
            state = 1.0
        elif inputs[t, 1] > 0:
            state = -1.0
    assert float(np.atleast_1d(target)[0]) == state
    assert float(np.atleast_1d(target)[0]) in {-1.0, 1.0}


def test_delayed_recall_returns_initial_cue():
    task = DelayedRecallTask(seq_len=25, in_dim=1)
    rng = np.random.default_rng(2)
    inputs, target = task.generate(rng)
    assert inputs.shape == (25, 1)
    assert float(np.atleast_1d(target)[0]) == float(np.sign(inputs[0, 0]))
    assert np.allclose(inputs[1:, 0], 0.0)
