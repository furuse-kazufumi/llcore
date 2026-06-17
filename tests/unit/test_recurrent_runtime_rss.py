# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/recurrent_runtime_rss.py`` (runtime peak RSS sweep).

Tiny configs / short lengths so the end-to-end run (which spawns 3 modes x N
lengths as subprocesses) is fast. Asserts structure + that every mode/length
produces a measurement; the headline growth ratios are validated by the real run.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def _load_script() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "recurrent_runtime_rss.py"
    spec = importlib.util.spec_from_file_location("recurrent_runtime_rss", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_lengths_sorts_dedups_validates() -> None:
    mod = _load_script()
    assert mod._parse_lengths("512,128,512,256") == [128, 256, 512]
    for bad in ("", "0", "-4", "abc", "8,"):
        with pytest.raises(ValueError):
            mod._parse_lengths(bad)


def test_run_worker_each_mode(tmp_path: Path) -> None:
    mod = _load_script()
    # Tiny config + short T so all three real workloads run in-process quickly.
    for mode in ("gpt", "recurrent", "rwkv"):
        rec = mod.run_worker(mode, t=8, n_embd=16, n_layer=1, n_head=2, vocab=32)
        assert rec["mode"] == mode
        assert rec["t"] == 8
        assert rec["peak_ws_mb"] >= 0.0


def test_worker_mode_prints_result_json(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    rc = mod.main(["--worker", "recurrent", "--t", "8", "--n-embd", "16",
                   "--n-layer", "1", "--n-head", "2", "--vocab", "32"])
    assert rc == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith(mod.RESULT_PREFIX))
    assert json.loads(line[len(mod.RESULT_PREFIX):])["mode"] == "recurrent"


def test_worker_requires_positive_t(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    assert mod.main(["--worker", "gpt"]) == 2
    assert "error:" in capsys.readouterr().err


def test_main_end_to_end_tiny(tmp_path: Path) -> None:
    mod = _load_script()
    out = tmp_path / "rss.json"
    rc = mod.main(["--lengths", "8,16", "--n-embd", "16", "--n-layer", "1",
                   "--n-head", "2", "--vocab", "32", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["records"]) == {"gpt", "recurrent", "rwkv"}
    # One measurement per requested length, per mode.
    for mode in ("gpt", "recurrent", "rwkv"):
        assert [r["t"] for r in payload["records"][mode]] == [8, 16]
    assert set(payload["growth_ratio"]) == {"gpt", "recurrent", "rwkv"}


def test_main_rejects_bad_lengths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    assert mod.main(["--lengths", "0", "--json", str(tmp_path / "x.json")]) == 2
    assert "error:" in capsys.readouterr().err
