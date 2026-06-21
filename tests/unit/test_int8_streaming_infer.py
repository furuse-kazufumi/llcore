# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/int8_streaming_infer.py`` (int8 streaming-dequant inference).

Tiny models so the end-to-end parent (which spawns 3 subprocesses) is fast. The
headline RSS numbers are validated by the real ~500MB run; here we assert the
invariants: int8 resident < fp32 resident, and stream == dense output exactly.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F


def _load_script() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "int8_streaming_infer.py"
    spec = importlib.util.spec_from_file_location("int8_streaming_infer", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tiny_cfg(mod: Any) -> Any:
    return mod.GPTConfig(vocab_size=48, block_size=16, n_layer=2, n_head=2, n_embd=32)


def test_int8linear_from_linear_close_to_original() -> None:
    mod = _load_script()
    torch.manual_seed(0)
    lin = nn.Linear(16, 24)
    q = mod.Int8Linear.from_linear(lin)
    x = torch.randn(3, 16)
    out_q = q(x)
    out_ref = lin(x)
    assert out_q.shape == out_ref.shape
    # int8 (per-channel) should track the fp32 layer closely.
    assert (out_q - out_ref).abs().max().item() < 0.1


def test_stream_equals_dense_exactly() -> None:
    mod = _load_script()
    torch.manual_seed(1)
    lin = nn.Linear(16, 24)
    q = mod.Int8Linear.from_linear(lin)
    dense = q.to_fp32_linear()  # dequant once up front
    x = torch.randn(3, 16)
    # Streaming dequant (inside forward) and pre-materialized dense must match
    # bit-for-bit: same int8 source, same arithmetic, only timing differs.
    assert torch.equal(q(x), F.linear(x, dense.weight, dense.bias))


def test_convert_linears_roundtrip_and_resident_drop(tmp_path: Path) -> None:
    mod = _load_script()
    fp32 = mod.CharGPT(_tiny_cfg(mod))
    fp32_bytes = mod._resident_bytes(fp32)
    int8 = mod.convert_linears_to_int8(mod.CharGPT(_tiny_cfg(mod)))
    int8_bytes = mod._resident_bytes(int8)
    # Replacing Linear weights with int8 buffers must shrink the resident footprint.
    assert int8_bytes < fp32_bytes
    # No nn.Linear should remain after conversion; Int8Linear takes their place.
    assert not any(isinstance(m, nn.Linear) for m in int8.modules())
    assert any(isinstance(m, mod.Int8Linear) for m in int8.modules())
    # Converting back to dense restores plain nn.Linear modules.
    dense = mod.convert_int8_to_dense(int8)
    assert any(isinstance(m, nn.Linear) for m in dense.modules())
    assert not any(isinstance(m, mod.Int8Linear) for m in dense.modules())


def _save_tiny_int8(mod: Any, tmp_path: Path) -> Path:
    path = tmp_path / "model_int8.pt"
    mod._save_int8_model(_tiny_cfg(mod), path)
    return path


def test_run_worker_modes_agree(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt = _save_tiny_int8(mod, tmp_path)
    dense = mod.run_worker(ckpt, "dense", forward_repeats=2)
    stream = mod.run_worker(ckpt, "stream", forward_repeats=2)
    capped = mod.run_worker(ckpt, "stream", 512 * 1024 * 1024, forward_repeats=2)  # high cap: no constraint
    # All three start from the SAME int8 source -> identical logits.
    assert dense["checksum"] == stream["checksum"] == capped["checksum"]
    # int8 stream holds less resident than the dense fp32 materialization.
    assert stream["resident_mb"] < dense["resident_mb"]
    # Latency is measured (>= 0) and the repeat count is honored.
    for rec in (dense, stream, capped):
        assert rec["forward_ms_median"] >= 0.0
        assert rec["forward_repeats"] == 2


def test_main_end_to_end_tiny(tmp_path: Path) -> None:
    mod = _load_script()
    rc = mod.main([
        "--n-embd", "32", "--n-layer", "2", "--n-head", "2", "--vocab", "48", "--block", "16",
        "--out-dir", str(tmp_path / "art"), "--json", str(tmp_path / "s.json"),
    ])
    assert rc == 0
    payload = json.loads((tmp_path / "s.json").read_text(encoding="utf-8"))
    assert set(payload) >= {"dense", "stream", "stream_capped", "checksum_match", "latency_overhead_x"}
    assert payload["checksum_match"] is True
    assert payload["resident_reduction_pct"] > 0.0  # int8 stream resident < dense fp32
    assert payload["dense"]["forward_ms_median"] >= 0.0
    assert payload["stream"]["forward_ms_median"] >= 0.0


def test_worker_rejects_missing_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    rc = mod.main(["--worker", "stream", "--checkpoint", str(tmp_path / "nope.pt")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
