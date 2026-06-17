# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/int8_quant_footprint.py`` (memory-efficiency pivot (b)).

The script is loaded as a module (same pattern as
``test_memory_footprint_harness.py``) so its pure helpers — quantization math,
footprint accounting, tied-parameter dedup, arg parsing — can be unit-tested
without a GPU and without touching the real checkpoints.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn


def _load_script() -> Any:
    # Resolve scripts/int8_quant_footprint.py relative to the repo root (two
    # parents up from tests/unit/) and import it as an ad-hoc module.
    script = Path(__file__).resolve().parents[2] / "scripts" / "int8_quant_footprint.py"
    spec = importlib.util.spec_from_file_location("int8_quant_footprint", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantize_per_tensor_scale_and_roundtrip() -> None:
    mod = _load_script()
    # amax = 4.0 -> per-tensor scale = 4/127. The exact-multiple entries (0, 4)
    # round-trip exactly; everything else is within one quantization step.
    w = torch.tensor([[0.0, 4.0], [-4.0, 2.0]])
    q, scale = mod.quantize_symmetric(w, per_channel=False)
    assert q.dtype == torch.int8
    assert scale.numel() == 1  # one scale for the whole tensor
    assert float(scale.item()) == pytest.approx(4.0 / 127.0)
    w_hat = mod.dequantize(q, scale)
    # Dequant error is bounded by half a step (scale/2) per element.
    assert torch.max(torch.abs(w_hat - w)).item() <= float(scale.item()) / 2 + 1e-6


def test_quantize_per_channel_scale_shape_is_per_row() -> None:
    mod = _load_script()
    # Row 0 amax=1, row 1 amax=10 -> per-channel scales must differ per row so
    # the small-magnitude row is not crushed by the large-magnitude row.
    w = torch.tensor([[1.0, -1.0], [10.0, -5.0]])
    q, scale = mod.quantize_symmetric(w, per_channel=True)
    assert scale.shape == (2, 1)  # one scale column per output row
    assert float(scale[0, 0]) == pytest.approx(1.0 / 127.0)
    assert float(scale[1, 0]) == pytest.approx(10.0 / 127.0)


def test_per_channel_has_lower_error_than_per_tensor() -> None:
    mod = _load_script()
    # A weight with a wide per-row dynamic range is exactly where per-channel
    # wins: per-tensor uses a single coarse step, per-channel adapts per row.
    torch.manual_seed(0)
    w = torch.randn(8, 8)
    w[0] *= 50.0  # one row with a much larger magnitude
    pt = mod.dequantize(*mod.quantize_symmetric(w, per_channel=False))
    pc = mod.dequantize(*mod.quantize_symmetric(w, per_channel=True))
    assert (pc - w).abs().mean().item() < (pt - w).abs().mean().item()


def test_quantize_rejects_non_2d() -> None:
    mod = _load_script()
    # Only 2-D weights are quantized; a 1-D bias must raise rather than silently
    # produce a wrong-shaped scale.
    with pytest.raises(ValueError):
        mod.quantize_symmetric(torch.zeros(4), per_channel=False)


class _TinyNet(nn.Module):
    """Minimal module: one 2-D weight (quantizable) + one 1-D bias (kept fp32)."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4, 8))  # 32 elems, 2-D -> quantized
        self.b = nn.Parameter(torch.zeros(4))     # 4 elems, 1-D -> fp32


def test_footprint_accounting_exact_numbers() -> None:
    mod = _load_script()
    net = _TinyNet()
    # per-tensor: w body=32B, w scale=4B, b fp32=16B -> 52B; fp32=(32+4)*4=144B.
    fp_pt = mod.footprint_bytes(net, per_channel=False)
    assert fp_pt["fp32_bytes"] == 144
    assert fp_pt["quantized_param_bytes"] == 32
    assert fp_pt["scale_bytes"] == 4
    assert fp_pt["unquantized_param_bytes"] == 16
    assert fp_pt["int8_bytes"] == 52
    assert fp_pt["ideal_int8_bytes"] == 36  # every param at 1 byte (no scales)
    assert fp_pt["n_quantized_params"] == 1
    assert fp_pt["n_unquantized_params"] == 1
    # per-channel: scale grows to one fp32 per row (4 rows * 4B = 16B).
    fp_pc = mod.footprint_bytes(net, per_channel=True)
    assert fp_pc["scale_bytes"] == 16
    assert fp_pc["int8_bytes"] == 64


def test_unique_named_params_dedups_tied_weights() -> None:
    mod = _load_script()
    net = _TinyNet()
    # Tie a second attribute to the same Parameter object (as CharGPT ties
    # lm_head.weight to wte.weight). It must be counted exactly once.
    net.w2 = net.w  # type: ignore[assignment]
    names = [name for name, _ in mod._unique_named_params(net)]
    assert names.count("w") + names.count("w2") == 1
    # Footprint must not double-count the shared weight.
    assert mod.footprint_bytes(net, per_channel=False)["quantized_param_bytes"] == 32


def test_apply_quantization_mutates_and_reports_error() -> None:
    mod = _load_script()
    torch.manual_seed(1)
    net = _TinyNet()
    net.w.data = torch.randn(4, 8)
    before = net.w.data.clone()
    stats = mod.apply_quantization(net, per_channel=True)
    # The 2-D weight is replaced by its dequantized approximation (changed but
    # close); the 1-D bias is untouched.
    assert not torch.equal(net.w.data, before)
    assert torch.allclose(net.w.data, before, atol=0.1)
    assert torch.equal(net.b.data, torch.zeros(4))
    assert 0.0 <= stats["weight_rel_rmse"] < 0.1
    assert stats["weight_max_abs_err"] >= 0.0


def test_parse_schemes_validates_and_dedups() -> None:
    mod = _load_script()
    assert mod._parse_schemes("per_channel,per_tensor,per_channel") == [
        "per_channel",
        "per_tensor",
    ]
    for bad in ("", "bogus", "per_tensor,", "per_tensor,nope"):
        with pytest.raises(ValueError):
            mod._parse_schemes(bad)


def _write_tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    """Build a minimal trained-shaped CharGPT checkpoint + matching corpus file."""
    from llcore.lm.model import CharGPT, GPTConfig
    from llcore.lm.tokenizer import CharTokenizer

    # Repeat a short phrase so train/val both exceed block_size after the split.
    text = "hello llcore world\n" * 60
    tok = CharTokenizer.from_text(text)
    model = CharGPT(
        GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=1, n_embd=16)
    )
    ckpt = tmp_path / "model.pt"
    # Same on-disk shape as scripts/_save_checkpoint: config + state + itos.
    torch.save({"config": vars(model.config), "model_state": model.state_dict(), "itos": tok.itos}, ckpt)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    return ckpt, corpus


def test_main_writes_report_with_both_schemes(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt, corpus = _write_tiny_checkpoint(tmp_path)
    out = tmp_path / "int8.json"
    rc = mod.main(
        [
            "--checkpoint", str(ckpt),
            "--corpus-file", str(corpus),
            "--val-frac", "0.2",
            "--json", str(out),
        ]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) >= {"config", "model", "fp32", "schemes"}
    assert [rec["scheme"] for rec in payload["schemes"]] == ["per_tensor", "per_channel"]
    for rec in payload["schemes"]:
        # int8 must be meaningfully smaller than fp32, and the report must carry
        # the PPL-delta + gate fields the headline depends on.
        assert 0.2 < rec["compression_ratio"] < 0.35
        assert rec["savings_pct"] > 50.0
        assert "delta_ppl_pct" in rec
        assert "ppl_gate_pass" in rec


def test_main_rejects_bad_args_and_missing_files(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt, corpus = _write_tiny_checkpoint(tmp_path)
    out = tmp_path / "int8.json"
    # Unknown scheme -> rc 2.
    assert mod.main(["--checkpoint", str(ckpt), "--corpus-file", str(corpus),
                     "--schemes", "nope", "--json", str(out)]) == 2
    # val-frac out of (0,1) -> rc 2.
    assert mod.main(["--checkpoint", str(ckpt), "--corpus-file", str(corpus),
                     "--val-frac", "1.5", "--json", str(out)]) == 2
    # Missing checkpoint / corpus -> rc 2 (fail-closed, no traceback).
    assert mod.main(["--checkpoint", str(tmp_path / "nope.pt"),
                     "--corpus-file", str(corpus), "--json", str(out)]) == 2
    assert mod.main(["--checkpoint", str(ckpt),
                     "--corpus-file", str(tmp_path / "nope.txt"), "--json", str(out)]) == 2
