# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from collections.abc import Callable
from typing import Any

import pytest
import torch


def _load_harness() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "memory_footprint_harness.py"
    spec = importlib.util.spec_from_file_location("memory_footprint_harness", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_system_memory_snapshot_keeps_system_values_when_process_probe_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()

    class _FakeFunction:
        def __init__(self, fn: Callable[..., Any]) -> None:
            self._fn = fn
            self.argtypes = None
            self.restype = None

        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            return self._fn(*args, **kwargs)

    class _FakeKernel32:
        def __init__(self) -> None:
            self.GlobalMemoryStatusEx = _FakeFunction(self._global_memory_status_ex)
            self.GetCurrentProcess = _FakeFunction(self._get_current_process)

        def _global_memory_status_ex(self, ptr: Any) -> int:
            ms = ptr._obj
            ms.dwMemoryLoad = 77
            ms.ullTotalPhys = 16_000
            ms.ullAvailPhys = 4_000
            ms.ullTotalPageFile = 40_000
            ms.ullAvailPageFile = 12_000
            return 1

        def _get_current_process(self) -> int:
            return 0

    class _FakeWinDLL:
        def __init__(self, name: str, use_last_error: bool = True) -> None:
            self._name = name
            self._kernel = _FakeKernel32()

        def __getattr__(self, name: str) -> Any:
            return getattr(self._kernel, name)

    monkeypatch.setattr(harness.ctypes, "WinDLL", _FakeWinDLL)
    monkeypatch.setattr(harness, "_process_memory", lambda: None)

    snapshot = harness._system_memory_snapshot()
    assert snapshot is not None
    assert snapshot["memory_load_percent"] == 77
    assert snapshot["avail_phys_bytes"] == 4_000
    assert snapshot["avail_commit_bytes"] == 12_000
    assert "process_working_set_bytes" not in snapshot
    summary = harness._snapshot_summary(snapshot)
    assert summary is not None
    assert summary["avail_phys_mb"] == 0.0
    assert summary["avail_commit_mb"] == 0.0
    assert summary["process_working_set_mb"] is None
    assert summary["process_pagefile_mb"] is None
    assert summary["process_peak_pagefile_mb"] is None


def test_main_writes_json_with_system_before_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness = _load_harness()
    warmup_seen: dict[str, int] = {}

    class _DummyConfig:
        def __init__(self, vocab_size: int) -> None:
            self.vocab_size = vocab_size
            self.n_layer = 4
            self.n_embd = 128
            self.n_head = 4

    class _DummyModel(torch.nn.Module):
        def __init__(self, cfg: Any) -> None:
            super().__init__()
            self.config = _DummyConfig(cfg.vocab_size)
            self.p = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(harness, "CharGPT", _DummyModel)
    monkeypatch.setattr(harness, "RecurrentLM", _DummyModel)
    monkeypatch.setattr(harness, "RWKVLM", _DummyModel)

    def _fake_recurrent_state_bytes(model: Any, t: int, warmup: int = 1) -> tuple[int, float]:
        warmup_seen["recurrent"] = warmup
        return 2048, 0.0

    def _fake_gpt_context(model: Any, t: int, warmup: int = 1) -> tuple[int, int, float]:
        warmup_seen["gpt"] = warmup
        return 4096 * t, 1024 * t * t, 0.0

    monkeypatch.setattr(harness, "_recurrent_state_bytes", _fake_recurrent_state_bytes)
    monkeypatch.setattr(harness, "_gpt_context", _fake_gpt_context)

    snapshots = iter(
        [
            {
                "memory_load_percent": 70,
                "total_phys_bytes": 16_000_000,
                "avail_phys_bytes": 4_000_000,
                "total_commit_bytes": 32_000_000,
                "avail_commit_bytes": 12_000_000,
            },
            {
                "memory_load_percent": 75,
                "total_phys_bytes": 16_000_000,
                "avail_phys_bytes": 3_500_000,
                "total_commit_bytes": 32_000_000,
                "avail_commit_bytes": 11_000_000,
                "process_working_set_bytes": 200_000_000,
                "process_pagefile_bytes": 300_000_000,
                "process_peak_pagefile_bytes": 320_000_000,
            },
        ]
    )
    monkeypatch.setattr(harness, "_system_memory_snapshot", lambda: next(snapshots))

    out = tmp_path / "mem.json"
    rc = harness.main(["--lengths", "128,64,128", "--warmup", "3", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) >= {"config", "lengths_effective", "records", "system_before", "system_after"}
    assert payload["lengths_effective"] == [128, 64]
    assert payload["system_before"]["avail_commit_mb"] == 12.0
    assert payload["system_before"]["process_working_set_mb"] is None
    assert payload["system_after"]["avail_commit_mb"] == 11.0
    assert payload["system_after"]["process_working_set_mb"] == 200.0
    assert payload["system_after"]["process_pagefile_mb"] == 300.0
    assert payload["system_after"]["process_peak_pagefile_mb"] == 320.0
    assert warmup_seen == {"recurrent": 3, "gpt": 3}


def test_parse_lengths_preserves_input_order_while_deduping() -> None:
    harness = _load_harness()
    assert harness._parse_lengths("512,128,512,64") == [512, 128, 64]


def test_headline_uses_min_max_t_even_when_lengths_are_descending(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    harness = _load_harness()

    class _DummyConfig:
        def __init__(self, vocab_size: int) -> None:
            self.vocab_size = vocab_size
            self.n_layer = 4
            self.n_embd = 128
            self.n_head = 4

    class _DummyModel(torch.nn.Module):
        def __init__(self, cfg: Any) -> None:
            super().__init__()
            self.config = _DummyConfig(cfg.vocab_size)
            self.p = torch.nn.Parameter(torch.zeros(1))

    monkeypatch.setattr(harness, "CharGPT", _DummyModel)
    monkeypatch.setattr(harness, "RecurrentLM", _DummyModel)
    monkeypatch.setattr(harness, "RWKVLM", _DummyModel)
    monkeypatch.setattr(harness, "_recurrent_state_bytes", lambda model, t, warmup=1: (2048, 0.0))
    monkeypatch.setattr(harness, "_gpt_context", lambda model, t, warmup=1: (4096 * t, 1024 * t * t, 0.0))
    monkeypatch.setattr(harness, "_system_memory_snapshot", lambda: None)

    rc = harness.main(["--lengths", "512,128,64", "--json", str(tmp_path / "mem.json")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "[headline] T 64→512 (×8.00):" in out
    assert "GPT attn ×64 (QUADRATIC)." in out


def test_main_rejects_non_positive_lengths(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = _load_harness()
    rc = harness.main(["--lengths", "64,-1", "--json", str(tmp_path / "mem.json")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_main_rejects_non_integer_or_empty_length_items(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = _load_harness()
    rc = harness.main(["--lengths", "64,abc", "--json", str(tmp_path / "mem.json")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
    rc = harness.main(["--lengths", "64,,128", "--json", str(tmp_path / "mem.json")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_main_rejects_negative_warmup(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    harness = _load_harness()
    rc = harness.main(["--warmup", "-1", "--json", str(tmp_path / "mem.json")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
