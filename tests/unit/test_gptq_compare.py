# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/gptq_compare.py`` (GPTQ vs RTN quantization)."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch


def _load_script() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "gptq_compare.py"
    spec = importlib.util.spec_from_file_location("gptq_compare", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quantize_rtn_shape() -> None:
    mod = _load_script()
    w = torch.randn(4, 8)
    q = mod.quantize_rtn(w, bits=4)
    assert q.shape == w.shape


def test_gptq_lowers_output_error_vs_rtn() -> None:
    mod = _load_script()
    # The defining property of GPTQ: it minimizes ||(W-Ŵ)X||^2 (output error),
    # so given a real input Hessian H = X Xᵀ it must beat RTN on OUTPUT error
    # even though its WEIGHT error may be larger.
    torch.manual_seed(0)
    w = torch.randn(32, 64)
    x = torch.randn(64, 256)
    h = x @ x.t()
    rtn = mod.quantize_rtn(w, bits=2)
    gptq = mod.quantize_gptq(w, h, bits=2)
    out_err_rtn = ((rtn - w) @ x).pow(2).mean().item()
    out_err_gptq = ((gptq - w) @ x).pow(2).mean().item()
    assert out_err_gptq < out_err_rtn


def test_parse_bits() -> None:
    mod = _load_script()
    assert mod._parse_bits("2,3,2") == [3, 2]
    for bad in ("", "1", "17", "x"):
        with pytest.raises(ValueError):
            mod._parse_bits(bad)


def _tiny_model(mod: Any) -> Any:
    return mod.CharGPT(mod.GPTConfig(vocab_size=48, block_size=16, n_layer=1, n_head=2, n_embd=32))


def test_capture_hessians_shapes() -> None:
    mod = _load_script()
    model = _tiny_model(mod)
    calib = torch.randint(0, 48, (4, 16))
    hess = mod.capture_hessians(model, calib, batch_size=2)
    names = mod._linear_names(model)
    assert set(hess) == set(names)
    for name in names:
        lin = mod._get_linear(model, name)
        # Hessian is [in_features, in_features].
        assert hess[name].shape == (lin.in_features, lin.in_features)


def test_quantize_model_changes_linear_weights() -> None:
    mod = _load_script()
    model = _tiny_model(mod)
    calib = torch.randint(0, 48, (4, 16))
    hess = mod.capture_hessians(model, calib, batch_size=2)
    before = {n: mod._get_linear(model, n).weight.data.clone() for n in mod._linear_names(model)}
    mod.quantize_model(model, bits=3, method="gptq", hessians=hess)
    assert any(not torch.equal(before[n], mod._get_linear(model, n).weight.data)
               for n in mod._linear_names(model))


def _write_tiny_checkpoint(tmp_path: Path) -> tuple[Path, Path]:
    from llcore.lm.tokenizer import CharTokenizer  # type: ignore[import-untyped]

    mod = _load_script()
    text = "hello llcore world\n" * 80
    tok = CharTokenizer.from_text(text)
    model = mod.CharGPT(
        mod.GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=2, n_embd=32)
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
    rc = mod.main(["--checkpoint", str(ckpt), "--corpus-file", str(corpus), "--val-frac", "0.2",
                   "--bits", "3,2", "--calib-windows", "4", "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    # 2 bits x 2 methods (rtn, gptq) = 4 records.
    assert len(payload["records"]) == 4
    assert {r["method"] for r in payload["records"]} == {"rtn", "gptq"}
    assert "gptq_crossed_capgate_bits" in payload


def test_main_rejects_bad_args(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt, corpus = _write_tiny_checkpoint(tmp_path)
    out = tmp_path / "g.json"
    assert mod.main(["--checkpoint", str(ckpt), "--corpus-file", str(corpus),
                     "--bits", "1", "--json", str(out)]) == 2
    assert mod.main(["--checkpoint", str(tmp_path / "nope.pt"), "--corpus-file", str(corpus),
                     "--json", str(out)]) == 2
