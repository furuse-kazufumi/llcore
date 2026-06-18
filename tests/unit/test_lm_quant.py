# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.quant` — int8 streaming-dequant + mmap inference."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.quant import (  # type: ignore[import-untyped]  # new module, untyped under standard mypy
    INT8_CKPT_KIND,
    Int8Linear,
    _dense_to_int8_state,
    convert_linears_to_int8,
    dequantize,
    int8_footprint_bytes,
    load_int8_model,
    quantize_per_channel_int8,
    save_int8_checkpoint,
)


def _cfg() -> GPTConfig:
    return GPTConfig(vocab_size=48, block_size=16, n_layer=2, n_head=2, n_embd=32)


def test_quantize_per_channel_shape_and_dtype() -> None:
    w = torch.randn(4, 8)
    q, scale = quantize_per_channel_int8(w)
    assert q.dtype == torch.int8
    assert scale.shape == (4, 1)
    # Dequant error bounded by half a step per row.
    assert torch.max(torch.abs(dequantize(q, scale) - w)).item() <= float(scale.max()) / 2 + 1e-6


def test_quantize_rejects_non_2d() -> None:
    with pytest.raises(ValueError):
        quantize_per_channel_int8(torch.zeros(4))


def test_int8linear_matches_original_closely() -> None:
    torch.manual_seed(0)
    lin = nn.Linear(16, 24)
    q = Int8Linear.from_linear(lin)
    x = torch.randn(3, 16)
    assert (q(x) - lin(x)).abs().max().item() < 0.1


def test_convert_linears_replaces_and_preserves_forward() -> None:
    torch.manual_seed(0)
    dense = CharGPT(_cfg())
    dense.eval()
    int8 = convert_linears_to_int8(CharGPT(_cfg()))
    # copy dense weights into the int8 skeleton via re-quantization
    int8.load_state_dict(_dense_to_int8_state(dense), assign=True)
    int8.eval()
    assert not any(type(m) is nn.Linear for m in int8.modules())
    assert any(isinstance(m, Int8Linear) for m in int8.modules())
    idx = torch.randint(0, 48, (1, 8))
    # int8 forward runs and produces finite logits of the right shape.
    out = int8.forward_logits(idx)
    assert out.shape == (1, 8, 48)
    assert torch.isfinite(out).all()


def test_save_load_roundtrip_matches_in_process(tmp_path: Path) -> None:
    torch.manual_seed(1)
    dense = CharGPT(_cfg())
    dense.eval()
    itos = [chr(65 + i) for i in range(48)]
    path = tmp_path / "int8.pt"
    save_int8_checkpoint(dense, path, itos)

    # In-process int8 reference: quantize the same weights directly.
    ref = convert_linears_to_int8(CharGPT(dense.config))
    ref.load_state_dict(_dense_to_int8_state(dense), assign=True)
    ref.eval()

    idx = torch.randint(0, 48, (1, 8))
    for use_mmap in (False, True):
        loaded, loaded_itos = load_int8_model(path, mmap=use_mmap)
        assert loaded_itos == itos
        # Same int8 source -> byte-identical logits to the in-process conversion.
        assert torch.equal(loaded.forward_logits(idx), ref.forward_logits(idx))


def test_load_rejects_non_int8_checkpoint(tmp_path: Path) -> None:
    path = tmp_path / "fp32.pt"
    dense = CharGPT(_cfg())
    torch.save({"config": vars(dense.config), "model_state": dense.state_dict(), "itos": []}, path)
    with pytest.raises(ValueError, match=INT8_CKPT_KIND):
        load_int8_model(path)


def test_footprint_int8_smaller_than_fp32() -> None:
    fp = int8_footprint_bytes(CharGPT(_cfg()))
    assert fp["int8_bytes"] < fp["fp32_bytes"]
    assert 0.2 < fp["int8_bytes"] / fp["fp32_bytes"] < 0.5
