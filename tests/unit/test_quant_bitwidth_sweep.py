# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/quant_bitwidth_sweep.py`` (cliff_then_flat 検証実験).

Loaded as a module (same pattern as the other harness tests). Covers the
generalized N-bit symmetric quantizer, footprint accounting, the top-1
capability proxy, and the end-to-end sweep on a tiny checkpoint.
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
    script = Path(__file__).resolve().parents[2] / "scripts" / "quant_bitwidth_sweep.py"
    spec = importlib.util.spec_from_file_location("quant_bitwidth_sweep", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_qmax_for_bits() -> None:
    mod = _load_script()
    # Symmetric range: 2^(bits-1) - 1 positive levels.
    assert mod.qmax_for_bits(8) == 127
    assert mod.qmax_for_bits(4) == 7
    assert mod.qmax_for_bits(3) == 3
    assert mod.qmax_for_bits(2) == 1
    # 1-bit symmetric has no nonzero level -> reject.
    with pytest.raises(ValueError):
        mod.qmax_for_bits(1)


def test_more_bits_means_lower_quant_error() -> None:
    mod = _load_script()
    torch.manual_seed(0)
    w = torch.randn(16, 16)

    def rel_err(bits: int) -> float:
        w_hat = mod.dequantize(*mod.quantize_symmetric(w, bits, per_channel=True))
        # Wrap the whole power expression in float() so mypy sees a concrete float.
        return float(float(((w_hat - w) ** 2).sum().item()) ** 0.5)

    # The cliff mechanism: fewer levels -> larger reconstruction error, monotonically.
    e8, e4, e3, e2 = rel_err(8), rel_err(4), rel_err(3), rel_err(2)
    assert e8 < e4 < e3 < e2


def test_quantize_rejects_non_2d() -> None:
    mod = _load_script()
    with pytest.raises(ValueError):
        mod.quantize_symmetric(torch.zeros(4), bits=4, per_channel=True)


class _TinyNet(nn.Module):
    """One 2-D weight (quantized) + one 1-D bias (kept fp32)."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.zeros(4, 8))  # 32 elems, 2-D
        self.b = nn.Parameter(torch.zeros(4))     # 4 elems, 1-D


def test_footprint_shrinks_with_fewer_bits() -> None:
    mod = _load_script()
    net = _TinyNet()
    # 4-bit: w body = 32*4/8 = 16B, scale = 4 rows*4B = 16B, b fp32 = 16B -> 48B; fp32 = 144B.
    fp4 = mod.footprint_bytes(net, bits=4, per_channel=True)
    assert fp4["fp32_bytes"] == 144
    assert fp4["quant_bytes"] == 48
    # Fewer bits -> strictly smaller footprint (int body halves from 4-bit to 2-bit).
    fp2 = mod.footprint_bytes(net, bits=2, per_channel=True)
    assert fp2["quant_bytes"] < fp4["quant_bytes"]
    assert fp2["compression_ratio"] < fp4["compression_ratio"]


def test_parse_bits_sorts_desc_dedups_and_validates() -> None:
    mod = _load_script()
    assert mod._parse_bits("4,8,3,8") == [8, 4, 3]
    for bad in ("", "1,4", "17", "abc", "4,"):
        with pytest.raises(ValueError):
            mod._parse_bits(bad)


def _write_tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    # llcore resolves to the installed (untyped) package under standard mypy.
    from llcore.lm.model import CharGPT, GPTConfig  # type: ignore[import-untyped]
    from llcore.lm.tokenizer import CharTokenizer  # type: ignore[import-untyped]

    text = "hello llcore world\n" * 60
    tok = CharTokenizer.from_text(text)
    model = CharGPT(
        GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=1, n_embd=16)
    )
    ckpt = tmp_path / "model.pt"
    torch.save({"config": vars(model.config), "model_state": model.state_dict(), "itos": tok.itos}, ckpt)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    return ckpt, corpus


def test_held_out_top1_bounds(tmp_path: Path) -> None:
    mod = _load_script()
    from llcore.lm.data import train_val_split  # type: ignore[import-untyped]

    ckpt, corpus = _write_tiny_checkpoint(tmp_path)
    model, tok = mod._load_checkpoint(ckpt)
    text = corpus.read_text(encoding="utf-8")
    ids = torch.tensor(tok.encode_safe(text), dtype=torch.long)
    _, val_ids = train_val_split(ids, val_frac=0.2)
    acc = mod.held_out_top1(model, val_ids, model.config.block_size)
    # top1 <= top5 <= 1.0, and both are valid probabilities.
    assert 0.0 <= acc["top1_acc"] <= acc["top5_acc"] <= 1.0
    assert acc["n_tokens"] > 0


def test_main_sweep_end_to_end(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt, corpus = _write_tiny_checkpoint(tmp_path)
    out = tmp_path / "sweep.json"
    rc = mod.main(
        ["--checkpoint", str(ckpt), "--corpus-file", str(corpus),
         "--val-frac", "0.2", "--bits", "8,4,2", "--json", str(out)]
    )
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) >= {"config", "fp32", "records", "ppl_cliff_bits", "ppl_knee_bits"}
    # Records are sorted descending by bit-width and carry PPL + capability fields.
    assert [r["bits"] for r in payload["records"]] == [8, 4, 2]
    for r in payload["records"]:
        assert "delta_ppl_pct" in r and "top1_acc" in r and "delta_top1_pp" in r
    # Footprint must shrink monotonically as bits drop.
    ratios = [r["compression_ratio"] for r in payload["records"]]
    assert ratios[0] > ratios[1] > ratios[2]


def test_main_rejects_bad_args(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt, corpus = _write_tiny_checkpoint(tmp_path)
    out = tmp_path / "sweep.json"
    # Out-of-range bits (1-bit) -> rc 2.
    assert mod.main(["--checkpoint", str(ckpt), "--corpus-file", str(corpus),
                     "--bits", "1", "--json", str(out)]) == 2
    # Missing checkpoint -> rc 2.
    assert mod.main(["--checkpoint", str(tmp_path / "nope.pt"),
                     "--corpus-file", str(corpus), "--bits", "4", "--json", str(out)]) == 2
