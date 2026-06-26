# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/ttt_plateau_experiment.py`` — the past-block-gain plateau metric.

Pins the headline read of the L1 experiment: a positive ``past_block_gain`` means the arm keeps
lowering NLL past ``block_size`` (i.e. it actually uses long context), and the metric is ``None``
when the curve lacks the comparison points so the experiment can never silently report a fake gain.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "ttt_plateau_experiment.py"
_spec = importlib.util.spec_from_file_location("ttt_plateau_experiment", _SCRIPT)
assert _spec is not None and _spec.loader is not None
tpe = importlib.util.module_from_spec(_spec)
sys.modules["ttt_plateau_experiment"] = tpe
_spec.loader.exec_module(tpe)


def test_gain_positive_when_nll_keeps_dropping() -> None:
    curve = {64: 2.0, 128: 1.5, 256: 1.2, 512: 1.0}
    g = tpe.past_block_gain(curve, block_size=128)
    assert g is not None and abs(g - (1.5 - 1.0) / 1.5) < 1e-9


def test_gain_zero_when_flat_plateau() -> None:
    curve = {128: 1.5, 256: 1.5, 512: 1.5}
    assert tpe.past_block_gain(curve, block_size=128) == 0.0


def test_gain_negative_when_nll_rises_past_block() -> None:
    curve = {128: 1.0, 256: 1.2}  # degrades past training window (OOD)
    g = tpe.past_block_gain(curve, block_size=128)
    assert g is not None and g < 0


def test_none_when_block_or_beyond_missing() -> None:
    assert tpe.past_block_gain({64: 2.0, 128: 1.5}, block_size=128) is None  # nothing beyond
    assert tpe.past_block_gain({256: 1.2, 512: 1.0}, block_size=128) is None  # no block point
