# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/qat_train.py`` (QAT fake-quant + STE)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn


def _load_script() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "qat_train.py"
    spec = importlib.util.spec_from_file_location("qat_train", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fake_quant_ste_forward_is_quantized_backward_is_identity() -> None:
    mod = _load_script()
    w = torch.randn(4, 8, requires_grad=True)
    wq = mod.fake_quant_ste(w, bits=2)
    # forward equals the genuine quantize-dequantize of w.
    qmax = 1
    scale = w.detach().abs().amax(dim=1, keepdim=True) / qmax
    expected = torch.clamp(torch.round(w.detach() / scale), -qmax, qmax) * scale
    assert torch.allclose(wq.detach(), expected)
    # backward is the straight-through identity (grad of sum == ones).
    wq.sum().backward()
    assert w.grad is not None
    assert torch.allclose(w.grad, torch.ones_like(w))


def test_fakequant_linear_uses_quantized_weight() -> None:
    mod = _load_script()
    fq = mod.FakeQuantLinear(8, 4, bias=True)
    fq.bits = 3
    x = torch.randn(2, 8)
    out = fq(x)
    # Output matches F.linear with the fake-quantized weight.
    from torch.nn import functional as F

    expected = F.linear(x, mod.fake_quant_ste(fq.weight, 3), fq.bias)
    assert torch.allclose(out, expected)


def test_convert_to_fake_quant_replaces_all_linears() -> None:
    mod = _load_script()
    model = mod.CharGPT(mod.GPTConfig(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16))
    n_linear_before = sum(1 for m in model.modules()
                          if isinstance(m, nn.Linear) and not isinstance(m, mod.FakeQuantLinear))
    mod.convert_to_fake_quant(model, bits=2)
    # Every plain Linear is now a FakeQuantLinear with the requested bit-width.
    fq = [m for m in model.modules() if isinstance(m, mod.FakeQuantLinear)]
    assert len(fq) == n_linear_before
    assert all(m.bits == 2 for m in fq)
    assert not any(type(m) is nn.Linear for m in model.modules())


def test_main_end_to_end_short(tmp_path: Path) -> None:
    mod = _load_script()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello llcore world\n" * 400, encoding="utf-8")
    out = tmp_path / "qat.json"
    # 2 iters only (just exercises the train+eval pipeline); no fp32 ref provided.
    rc = mod.main(["--bits", "2", "--corpus-file", str(corpus),
                   "--fp32-checkpoint", str(tmp_path / "nope.pt"),
                   "--max-iters", "2", "--batch-size", "8", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "qat" in payload and "model_ppl" in payload["qat"]
    assert payload["fp32_reference"] is None  # no checkpoint -> no reference
    assert payload["capability_gate_pass"] is None


def test_main_rejects_bad_args(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    assert mod.main(["--bits", "1", "--corpus-file", str(tmp_path / "x.txt")]) == 2
    assert mod.main(["--bits", "2", "--corpus-file", str(tmp_path / "nope.txt")]) == 2


# --- LSQ (Learned Step Size Quantization) ---


def test_grad_scale_forward_identity_backward_scaled() -> None:
    mod = _load_script()
    x = torch.tensor([2.0, -3.0], requires_grad=True)
    y = mod.grad_scale(x, 0.25)
    assert torch.allclose(y.detach(), x.detach())  # forward is identity
    y.sum().backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.full_like(x, 0.25))  # backward multiplies by g


def test_lsq_init_scale_is_positive_1d_per_channel() -> None:
    mod = _load_script()
    w = torch.randn(5, 9)
    s = mod.lsq_init_scale(w, bits=2)
    assert s.dim() == 1 and s.shape == (5,)  # per output channel, 1-D
    assert torch.all(s > 0)


def test_lsq_quant_output_lies_on_the_quantization_grid() -> None:
    mod = _load_script()
    w = torch.randn(4, 8)
    scale = torch.rand(4) + 0.5  # positive per-channel
    out = mod.lsq_quant(w, scale, bits=2)
    levels = out / scale.unsqueeze(1)
    # integer levels within [-Q_N, Q_P] = [-2, 1] for 2-bit signed.
    assert torch.allclose(levels, levels.round(), atol=1e-5)
    assert float(levels.max()) <= 1 + 1e-5 and float(levels.min()) >= -2 - 1e-5


def test_lsq_quant_scale_and_weight_both_receive_gradient() -> None:
    mod = _load_script()
    w = torch.randn(4, 8, requires_grad=True)
    scale = torch.nn.Parameter(torch.rand(4) + 0.5)
    out = mod.lsq_quant(w, scale, bits=2)
    out.pow(2).sum().backward()
    assert w.grad is not None and scale.grad is not None
    assert torch.any(scale.grad != 0)  # the learnable step size actually moves


def test_lsq_linear_scale_is_1d_and_forward_uses_lsq_quant() -> None:
    mod = _load_script()
    lq = mod.LSQLinear(8, 4, bias=True)
    lq.bits = 2
    assert isinstance(lq.lsq_scale, torch.nn.Parameter)
    assert lq.lsq_scale.dim() == 1 and lq.lsq_scale.shape == (4,)  # 1-D => no weight-decay group
    x = torch.randn(2, 8)
    from torch.nn import functional as F

    expected = F.linear(x, mod.lsq_quant(lq.weight, lq.lsq_scale, 2), lq.bias)
    assert torch.allclose(lq(x), expected)


def test_convert_to_lsq_replaces_linears_and_inits_scale() -> None:
    mod = _load_script()
    model = mod.CharGPT(mod.GPTConfig(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16))
    n_linear_before = sum(1 for m in model.modules()
                          if isinstance(m, nn.Linear) and not isinstance(m, mod.LSQLinear))
    mod.convert_to_lsq(model, bits=2)
    lq = [m for m in model.modules() if isinstance(m, mod.LSQLinear)]
    assert len(lq) == n_linear_before
    assert all(m.bits == 2 for m in lq)
    # scale initialized from weights (LSQ init), not left at the constructor's ones.
    assert all(torch.all(m.lsq_scale > 0) for m in lq)
    assert any(not torch.allclose(m.lsq_scale, torch.ones_like(m.lsq_scale)) for m in lq)


def test_main_lsq_end_to_end_short(tmp_path: Path) -> None:
    mod = _load_script()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello llcore world\n" * 400, encoding="utf-8")
    out = tmp_path / "lsq.json"
    rc = mod.main(["--method", "lsq", "--bits", "2", "--corpus-file", str(corpus),
                   "--fp32-checkpoint", str(tmp_path / "nope.pt"),
                   "--max-iters", "2", "--batch-size", "8", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["config"]["method"] == "lsq"
    assert "qat" in payload and "model_ppl" in payload["qat"]
