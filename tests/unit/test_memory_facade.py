# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.memory` — the memory-efficiency toolkit facade.

The facade is *packaging*: it re-exports the verified primitives (int8 quant,
capability gate, constant-state recurrent, GPT KV accounting) under one namespace
and adds :func:`measure_memory`, a one-call footprint + capability-retention report.
These tests pin the wiring (facade values are the same objects; ``measure_memory``
agrees with the underlying primitives) rather than re-deriving the numerics, which
are already covered by ``test_lm_quant`` / ``test_lm_eval``.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from llcore.lm.eval import held_out_top1_report, passes_capability_gate
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.quant import (
    Int8Linear,
    convert_linears_to_int8,
    int8_footprint_bytes,
    load_int8_model,
)
from llcore.lm.tokenizer import CharTokenizer

import llcore.memory as mem
from llcore.memory import MemoryReport, _should_promote, measure_memory


def _cfg() -> GPTConfig:
    return GPTConfig(vocab_size=48, block_size=16, n_layer=2, n_head=2, n_embd=32)


def _val_ids(n: int = 400) -> torch.Tensor:
    torch.manual_seed(7)
    return torch.randint(0, 48, (n,))


# --- facade re-exports -------------------------------------------------------


def test_facade_reexports_are_the_same_objects() -> None:
    """The facade must expose the *verified* primitives, not re-implementations."""
    from llcore.lm import compare as _compare
    from llcore.lm import eval as _eval
    from llcore.lm import quant as _quant
    from llcore.lm import recurrent as _recurrent

    assert mem.quantize_per_channel_int8 is _quant.quantize_per_channel_int8
    assert mem.Int8Linear is _quant.Int8Linear
    assert mem.convert_linears_to_int8 is _quant.convert_linears_to_int8
    assert mem.int8_footprint_bytes is _quant.int8_footprint_bytes
    assert mem.save_int8_checkpoint is _quant.save_int8_checkpoint
    assert mem.load_int8_model is _quant.load_int8_model
    assert mem.held_out_top1_report is _eval.held_out_top1_report
    assert mem.passes_capability_gate is _eval.passes_capability_gate
    assert mem.RecurrentLM is _recurrent.RecurrentLM
    assert mem.gpt_kv_bytes is _compare.gpt_kv_bytes


def test_facade_all_is_importable() -> None:
    for name in mem.__all__:
        assert hasattr(mem, name), f"{name} listed in __all__ but missing"


# --- measure_memory: footprint-only path ------------------------------------


def test_measure_memory_footprint_only_matches_primitive() -> None:
    model = CharGPT(_cfg())
    report = measure_memory(model)
    fp = int8_footprint_bytes(model)
    assert report.fp32_bytes == fp["fp32_bytes"]
    assert report.int8_bytes == fp["int8_bytes"]
    assert report.compression_ratio == report.int8_bytes / report.fp32_bytes
    assert report.percent_smaller == pytest.approx((1.0 - report.compression_ratio) * 100.0)
    # No eval data -> capability fields are absent, not fabricated.
    assert report.fp32_top1 is None
    assert report.int8_top1 is None
    assert report.retention is None
    assert report.capability_gate_pass is None
    assert report.n_eval_tokens is None


def test_measure_memory_does_not_mutate_the_input_model() -> None:
    """Footprint/retention must be read-only: the caller's fp32 model stays fp32."""
    model = CharGPT(_cfg())
    measure_memory(model, val_ids=_val_ids(), min_retention=0.97)
    assert any(type(m) is nn.Linear for m in model.modules())
    assert not any(isinstance(m, Int8Linear) for m in model.modules())


# --- measure_memory: capability-retention path ------------------------------


def test_measure_memory_retention_matches_underlying_primitives() -> None:
    model = CharGPT(_cfg())
    model.eval()
    val = _val_ids()
    block = model.config.block_size
    report = measure_memory(model, val_ids=val, block_size=block, min_retention=0.97)

    expected_fp32 = held_out_top1_report(model, val, block)["top1_acc"]
    int8_ref = copy.deepcopy(model)  # stays typed CharGPT; convert mutates in place
    convert_linears_to_int8(int8_ref)
    int8_ref.eval()
    expected_int8 = held_out_top1_report(int8_ref, val, block)["top1_acc"]

    assert report.fp32_top1 == expected_fp32
    assert report.int8_top1 == expected_int8
    assert report.capability_gate_pass == passes_capability_gate(expected_int8, expected_fp32, 0.97)
    assert report.n_eval_tokens is not None and report.n_eval_tokens > 0
    if expected_fp32 > 0:
        assert report.retention == pytest.approx(expected_int8 / expected_fp32)
    else:
        assert report.retention is None


def test_measure_memory_infers_block_size_from_model_config() -> None:
    model = CharGPT(_cfg())
    # No explicit block_size: should fall back to model.config.block_size.
    report = measure_memory(model, val_ids=_val_ids())
    assert report.n_eval_tokens is not None and report.n_eval_tokens > 0


def test_measure_memory_custom_min_retention_is_recorded_and_used() -> None:
    model = CharGPT(_cfg())
    val = _val_ids()
    strict = measure_memory(model, val_ids=val, min_retention=0.999)
    loose = measure_memory(model, val_ids=val, min_retention=0.0)
    assert strict.min_retention == 0.999
    assert loose.min_retention == 0.0
    # A zero floor can never fail the gate.
    assert loose.capability_gate_pass is True


# --- MemoryReport serialization ---------------------------------------------


def test_memory_report_to_dict_is_json_serializable() -> None:
    report = measure_memory(CharGPT(_cfg()), val_ids=_val_ids())
    d = report.to_dict()
    assert isinstance(d, dict)
    for key in (
        "fp32_bytes",
        "int8_bytes",
        "compression_ratio",
        "percent_smaller",
        "fp32_top1",
        "int8_top1",
        "retention",
        "capability_gate_pass",
        "min_retention",
        "n_eval_tokens",
    ):
        assert key in d
    # Round-trips through JSON without error.
    assert json.loads(json.dumps(d)) == d


# --- CLI: python -m llcore.memory report ------------------------------------


def _save_fp32_checkpoint(path: Path, text: str) -> CharTokenizer:
    tok = CharTokenizer.from_text(text)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size, block_size=16, n_layer=2, n_head=2, n_embd=32
    )
    model = CharGPT(cfg)
    torch.save({"config": vars(cfg), "model_state": model.state_dict(), "itos": tok.itos}, path)
    return tok


def test_cli_report_footprint_only(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    ckpt = tmp_path / "model.pt"
    _save_fp32_checkpoint(ckpt, "hello world " * 30)
    rc = mem.main(["report", str(ckpt)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "int8" in out.lower()
    assert "smaller" in out.lower()


def test_cli_report_json_footprint_only(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _save_fp32_checkpoint(ckpt, "hello world " * 30)
    out_json = tmp_path / "report.json"
    rc = mem.main(["report", str(ckpt), "--json", str(out_json)])
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["int8_bytes"] < payload["fp32_bytes"]
    assert payload["fp32_top1"] is None
    assert payload["capability_gate_pass"] is None


def test_cli_report_with_corpus_file_computes_retention(tmp_path: Path) -> None:
    text = "to be or not to be that is the question " * 20
    ckpt = tmp_path / "model.pt"
    _save_fp32_checkpoint(ckpt, text)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    out_json = tmp_path / "report.json"
    rc = mem.main(
        ["report", str(ckpt), "--corpus-file", str(corpus), "--json", str(out_json)]
    )
    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["fp32_top1"] is not None
    assert payload["int8_top1"] is not None
    assert payload["n_eval_tokens"] is not None and payload["n_eval_tokens"] > 0
    assert isinstance(payload["capability_gate_pass"], bool)


# --- fail-closed --save-int8 promotion gate ---------------------------------


def _report(gate: bool | None) -> MemoryReport:
    return MemoryReport(
        fp32_bytes=100,
        int8_bytes=30,
        compression_ratio=0.3,
        percent_smaller=70.0,
        min_retention=0.97,
        capability_gate_pass=gate,
    )


def test_should_promote_passes_only_when_gate_passed() -> None:
    assert _should_promote(_report(True), force=False)[0] is True


def test_should_promote_refuses_on_gate_fail() -> None:
    ok, reason = _should_promote(_report(False), force=False)
    assert ok is False
    assert "gate" in reason.lower()


def test_should_promote_refuses_without_capability_evidence() -> None:
    ok, reason = _should_promote(_report(None), force=False)
    assert ok is False
    assert "corpus" in reason.lower() or "evidence" in reason.lower()


def test_should_promote_force_overrides_both_refusals() -> None:
    assert _should_promote(_report(False), force=True)[0] is True
    assert _should_promote(_report(None), force=True)[0] is True


def test_cli_save_int8_promotes_when_gate_passes(tmp_path: Path) -> None:
    text = "to be or not to be that is the question " * 20
    ckpt = tmp_path / "model.pt"
    tok = _save_fp32_checkpoint(ckpt, text)
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(text, encoding="utf-8")
    out_int8 = tmp_path / "model_int8.pt"
    # min-retention 0.0 makes the gate pass deterministically.
    rc = mem.main(
        [
            "report", str(ckpt),
            "--corpus-file", str(corpus),
            "--min-retention", "0.0",
            "--save-int8", str(out_int8),
        ]
    )
    assert rc == 0
    assert out_int8.exists()
    loaded, loaded_itos = load_int8_model(out_int8)
    assert loaded_itos == tok.itos


def test_cli_save_int8_refused_without_corpus(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _save_fp32_checkpoint(ckpt, "hello world " * 30)
    out_int8 = tmp_path / "model_int8.pt"
    rc = mem.main(["report", str(ckpt), "--save-int8", str(out_int8)])
    assert rc != 0
    assert not out_int8.exists()


def test_cli_save_int8_force_writes_without_corpus(tmp_path: Path) -> None:
    ckpt = tmp_path / "model.pt"
    _save_fp32_checkpoint(ckpt, "hello world " * 30)
    out_int8 = tmp_path / "model_int8.pt"
    rc = mem.main(["report", str(ckpt), "--save-int8", str(out_int8), "--force"])
    assert rc == 0
    assert out_int8.exists()
