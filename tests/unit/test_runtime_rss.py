# SPDX-License-Identifier: Apache-2.0
"""Tests for ``llcore.runtime.rss`` (shared process RSS measurement)."""
from __future__ import annotations

from llcore.runtime import rss


def test_peak_working_set_nonnegative() -> None:
    assert rss.peak_working_set_bytes() >= 0


def test_working_set_nonnegative() -> None:
    assert rss.working_set_bytes() >= 0


def test_pmc_struct_layout() -> None:
    # The two fields the harnesses read must exist and be sized as machine words.
    names = {name for name, _ in rss._PMC._fields_}
    assert {"PeakWorkingSetSize", "WorkingSetSize"} <= names


def test_counters_none_off_windows_is_tolerated(monkeypatch) -> None:
    # When the WinAPI path yields no counters, working_set_bytes still returns an int >= 0
    # (Linux /proc fallback or 0), never raising.
    monkeypatch.setattr(rss, "_process_memory_counters", lambda: None)
    assert rss.working_set_bytes() >= 0
    assert rss.peak_working_set_bytes() == 0


def test_working_set_mb_returns_float_or_none() -> None:
    v = rss.working_set_mb()
    assert v is None or (isinstance(v, float) and v > 0)


def test_working_set_mb_none_when_unmeasurable(monkeypatch) -> None:
    monkeypatch.setattr(rss, "working_set_bytes", lambda: 0)
    assert rss.working_set_mb() is None


def test_process_memory_shape_or_none() -> None:
    pm = rss.process_memory()
    if pm is not None:
        assert pm.working_set >= 0 and pm.peak_working_set >= 0
        assert pm.pagefile >= 0 and pm.peak_pagefile >= 0


def test_process_memory_none_off_windows(monkeypatch) -> None:
    monkeypatch.setattr(rss, "_process_memory_counters", lambda: None)
    assert rss.process_memory() is None
    assert rss.peak_mem_bytes() == (0, 0)


def test_peak_mem_bytes_pair() -> None:
    pair = rss.peak_mem_bytes()
    assert isinstance(pair, tuple) and len(pair) == 2
    assert pair[0] >= 0 and pair[1] >= 0
