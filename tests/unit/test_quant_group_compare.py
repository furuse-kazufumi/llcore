# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/quant_group_compare.py`` (per-group vs per-channel quant)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn


def _load_script() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "quant_group_compare.py"
    spec = importlib.util.spec_from_file_location("quant_group_compare", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantize_grouped_full_equals_one_group() -> None:
    mod = _load_script()
    w = torch.randn(4, 64)
    # group_size 0 / >= in_features both mean per-channel = a single group per row.
    _, n0 = mod.quantize_grouped(w, bits=8, group_size=0)
    _, nbig = mod.quantize_grouped(w, bits=8, group_size=128)
    assert n0 == nbig == 4  # one scale per output row


def test_smaller_group_lowers_error_and_adds_scales() -> None:
    mod = _load_script()
    torch.manual_seed(0)
    w = torch.randn(4, 64)
    w[:, :8] *= 30.0  # a high-magnitude sub-range that per-channel must average over

    def mse(group_size: int) -> tuple[float, int]:
        w_hat, n = mod.quantize_grouped(w, bits=3, group_size=group_size)
        return float(((w_hat - w) ** 2).mean().item()), n

    e_full, n_full = mse(0)
    e_grp, n_grp = mse(8)
    # Finer groups: lower reconstruction error, more scales (footprint cost).
    assert e_grp < e_full
    assert n_grp > n_full


def test_quantize_grouped_rejects_non_2d() -> None:
    mod = _load_script()
    with pytest.raises(ValueError):
        mod.quantize_grouped(torch.zeros(8), bits=4, group_size=4)


def test_parse_groups_and_ints() -> None:
    mod = _load_script()
    assert mod._parse_groups("full,128,64,full") == [0, 128, 64]
    for bad in ("", "0", "-4", "abc"):
        with pytest.raises(ValueError):
            mod._parse_groups(bad)
    assert mod._parse_int_list("3,2,3", "bits") == [3, 2]
    with pytest.raises(ValueError):
        mod._parse_int_list("3,x", "bits")


class _TinyNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 64))
        self.b = nn.Parameter(torch.zeros(4))


def test_footprint_ratio_grows_as_group_shrinks() -> None:
    mod = _load_script()
    net = _TinyNet()
    # Same bit-width: finer groups carry more fp32 scales -> larger footprint ratio.
    r_full = mod.footprint_ratio(net, bits=4, total_scales=4)
    r_fine = mod.footprint_ratio(net, bits=4, total_scales=4 * 8)
    assert r_fine > r_full


def _write_tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    from llcore.lm.model import CharGPT, GPTConfig  # type: ignore[import-untyped]
    from llcore.lm.tokenizer import CharTokenizer  # type: ignore[import-untyped]

    text = "hello llcore world\n" * 60
    tok = CharTokenizer.from_text(text)
    model = CharGPT(
        GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=1, n_embd=32)
    )
    ckpt = tmp_path / "model.pt"
    torch.save({"config": vars(model.config), "model_state": model.state_dict(), "itos": tok.itos}, ckpt)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    return ckpt, corpus


def test_main_end_to_end(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt, corpus = _write_tiny_checkpoint(tmp_path)
    out = tmp_path / "g.json"
    rc = mod.main(["--checkpoint", str(ckpt), "--corpus-file", str(corpus),
                   "--val-frac", "0.2", "--bits", "4,2", "--groups", "full,16", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    # 2 bits x 2 groups = 4 records, each carrying both gate verdicts.
    assert len(payload["records"]) == 4
    for r in payload["records"]:
        assert "ppl_gate_pass" in r and "capability_gate_pass" in r


def test_main_rejects_bad_args(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt, corpus = _write_tiny_checkpoint(tmp_path)
    out = tmp_path / "g.json"
    assert mod.main(["--checkpoint", str(ckpt), "--corpus-file", str(corpus),
                     "--bits", "1", "--json", str(out)]) == 2
    assert mod.main(["--checkpoint", str(tmp_path / "nope.pt"), "--corpus-file", str(corpus),
                     "--json", str(out)]) == 2
