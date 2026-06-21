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
