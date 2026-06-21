# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/runtime_floor_rss.py`` (static runtime RSS floor sweep).

Tiny config / repeats=1 so the end-to-end run (3 stages x repeats subprocesses)
is fast. Asserts structure + that every stage produces an RSS reading; the headline
torch-tax / scaffold-ratio are validated by the real run.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def _load_script() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "runtime_floor_rss.py"
    spec = importlib.util.spec_from_file_location("runtime_floor_rss", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_working_set_bytes_nonnegative() -> None:
    mod = _load_script()
    assert mod._working_set_bytes() >= 0


def test_run_stage_python_and_torch() -> None:
    mod = _load_script()
    py = mod.run_stage("python", n_embd=16, n_layer=1, n_head=2, vocab=32)
    assert py["stage"] == "python"
    assert py["rss_mb"] >= 0.0
    assert "int8_mb" not in py
    tor = mod.run_stage("torch", n_embd=16, n_layer=1, n_head=2, vocab=32)
    assert tor["stage"] == "torch"
    assert tor["rss_mb"] >= 0.0


def test_run_stage_model_reports_footprint() -> None:
    mod = _load_script()
    rec = mod.run_stage("model", n_embd=16, n_layer=1, n_head=2, vocab=32)
    assert rec["stage"] == "model"
    assert rec["int8_mb"] >= 0.0
    assert rec["fp32_mb"] >= rec["int8_mb"]  # int8 should not exceed fp32
    assert rec["params_m"] >= 0.0


def test_worker_mode_prints_result_json(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    rc = mod.main(["--worker", "python", "--n-embd", "16", "--n-layer", "1",
                   "--n-head", "2", "--vocab", "32"])
    assert rc == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith(mod.RESULT_PREFIX))
    assert json.loads(line[len(mod.RESULT_PREFIX):])["stage"] == "python"


def test_main_rejects_bad_repeats(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    assert mod.main(["--repeats", "0", "--json", str(tmp_path / "x.json")]) == 2
    assert "error:" in capsys.readouterr().err


def test_main_end_to_end_tiny(tmp_path: Path) -> None:
    mod = _load_script()
    out = tmp_path / "floor.json"
    rc = mod.main(["--n-embd", "16", "--n-layer", "1", "--n-head", "2", "--vocab", "32",
                   "--repeats", "1", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["stages"]) == {"python", "torch", "model"}
    assert "torch_tax_mb" in payload
    assert "scaffold_ratio" in payload
    assert payload["stages"]["model"]["int8_mb"] >= 0.0
