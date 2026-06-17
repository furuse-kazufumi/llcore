# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/mmap_ram_exceed_poc.py`` (RAM-exceeding mmap PoC).

Uses a TINY model so the end-to-end parent run (which spawns real subprocesses)
is fast. The working-set cap default never constrains the tiny model, so the
tests assert structure + functional equivalence rather than the headline RSS
numbers (those are validated by the real ~500MB run).
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch


def _load_script() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "mmap_ram_exceed_poc.py"
    spec = importlib.util.spec_from_file_location("mmap_ram_exceed_poc", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_cfg(mod: Any) -> Any:
    # Smallest viable CharGPT: 1 layer, 1 head, tiny width/vocab.
    return mod.GPTConfig(vocab_size=64, block_size=16, n_layer=1, n_head=2, n_embd=32)


def test_quantize_per_channel_int8_shape() -> None:
    mod = _load_script()
    w = torch.randn(4, 8)
    q, scale = mod.quantize_per_channel_int8(w)
    assert q.dtype == torch.int8
    assert scale.shape == (4, 1)  # one fp32 scale per output row


def test_param_bytes_dedups_tied(tmp_path: Path) -> None:
    mod = _load_script()
    model = mod.build_large_model(_tiny_cfg(mod))
    # CharGPT ties lm_head to wte; _param_bytes must count the shared tensor once.
    nb = mod._param_bytes(model)
    naive = sum(p.numel() * p.element_size() for p in model.parameters())
    # parameters() already dedups shared tensors, so these match; the point is no
    # double counting blows it up.
    assert nb == naive
    assert nb > 0


def test_save_int8_checkpoint_is_smaller(tmp_path: Path) -> None:
    mod = _load_script()
    model = mod.build_large_model(_tiny_cfg(mod))
    fp32 = tmp_path / "fp32.pt"
    torch.save({"config": vars(model.config), "model_state": model.state_dict()}, fp32)
    int8 = tmp_path / "int8.pt"
    size = mod.save_int8_checkpoint(model, int8)
    assert size == int8.stat().st_size
    # int8 file must be meaningfully smaller than fp32 (2-D weights at 1 byte).
    assert size < fp32.stat().st_size
    blob = torch.load(int8, map_location="cpu", weights_only=True)
    assert blob["kind"] == "int8_per_channel"
    assert any(k.startswith("q::") for k in blob)  # quantized 2-D weights present


def test_set_working_set_cap_returns_bool() -> None:
    mod = _load_script()
    # Best-effort; on non-Windows or if unenforceable it must return a bool, not raise.
    assert isinstance(mod._set_working_set_cap(512 * 1024 * 1024), bool)


def _save_tiny(mod: Any, tmp_path: Path) -> Path:
    model = mod.build_large_model(_tiny_cfg(mod))
    path = tmp_path / "model_fp32.pt"
    torch.save({"config": vars(model.config), "model_state": model.state_dict()}, path)
    return path


def test_run_worker_capped_and_uncapped_agree(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt = _save_tiny(mod, tmp_path)
    # A high cap (512MB) never constrains the tiny model; both runs must produce
    # the SAME logits checksum (functional equivalence of mmap+assign load).
    uncapped = mod.run_worker(ckpt, None)
    capped = mod.run_worker(ckpt, 512 * 1024 * 1024)
    assert uncapped["mode"] == "uncapped" and capped["mode"] == "capped"
    assert uncapped["checksum"] == capped["checksum"]
    for rec in (uncapped, capped):
        assert rec["peak_ws_mb"] >= 0.0
        assert rec["post_load_delta_mb"] >= 0.0


def test_worker_mode_prints_result_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    ckpt = _save_tiny(mod, tmp_path)
    rc = mod.main(["--worker", "--checkpoint", str(ckpt)])
    assert rc == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith(mod.RESULT_PREFIX))
    assert json.loads(line[len(mod.RESULT_PREFIX):])["mode"] == "uncapped"


def test_main_end_to_end_tiny(tmp_path: Path) -> None:
    mod = _load_script()
    # Tiny config so the parent build + two spawned workers run fast.
    rc = mod.main([
        "--n-embd", "32", "--n-layer", "1", "--n-head", "2", "--vocab", "64", "--block", "16",
        "--out-dir", str(tmp_path / "art"), "--json", str(tmp_path / "poc.json"),
    ])
    assert rc == 0
    payload = json.loads((tmp_path / "poc.json").read_text(encoding="utf-8"))
    assert set(payload) >= {"n_params", "fp32_file_bytes", "int8_file_bytes", "uncapped", "capped"}
    assert payload["int8_over_fp32"] < 1.0  # int8 smaller than fp32
    assert payload["checksum_match"] is True  # capped run reproduces uncapped output


def test_worker_rejects_missing_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    rc = mod.main(["--worker", "--checkpoint", str(tmp_path / "nope.pt")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
