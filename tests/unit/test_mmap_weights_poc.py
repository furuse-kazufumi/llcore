# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/mmap_weights_poc.py`` (memory-efficiency pivot (a)).

Loaded as a module (same pattern as the other harness tests). The parent role
spawns real subprocesses (the script re-invokes itself), so the end-to-end test
exercises the full eager-vs-mmap measurement pipeline on a tiny checkpoint.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest
import torch


def _load_script() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "mmap_weights_poc.py"
    spec = importlib.util.spec_from_file_location("mmap_weights_poc", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_tiny_checkpoint(tmp_path: Path) -> Path:
    """Build a minimal CharGPT checkpoint with the {config, model_state, itos} shape."""
    # llcore is the installed (untyped) package under standard mypy.
    from llcore.lm.model import CharGPT, GPTConfig  # type: ignore[import-untyped]
    from llcore.lm.tokenizer import CharTokenizer  # type: ignore[import-untyped]

    text = "hello llcore world\n" * 40
    tok = CharTokenizer.from_text(text)
    model = CharGPT(
        GPTConfig(vocab_size=tok.vocab_size, block_size=16, n_layer=1, n_head=1, n_embd=16)
    )
    ckpt = tmp_path / "model.pt"
    # Default torch.save uses the zip serialization that torch.load(mmap=True) needs.
    torch.save(
        {"config": vars(model.config), "model_state": model.state_dict(), "itos": tok.itos},
        ckpt,
    )
    return ckpt


def test_state_dict_bytes_sums_tensor_storage() -> None:
    mod = _load_script()
    # 10 float32 (40B) + 4 int64 (32B) = 72B, regardless of key names.
    state = {"a": torch.zeros(10, dtype=torch.float32), "b": torch.zeros(4, dtype=torch.int64)}
    assert mod._state_dict_bytes(state) == 10 * 4 + 4 * 8


def test_load_model_state_eager_and_mmap_agree(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt = _write_tiny_checkpoint(tmp_path)
    eager = mod._load_model_state(ckpt, use_mmap=False)
    mmapd = mod._load_model_state(ckpt, use_mmap=True)
    # Same keys and identical values: mmap only changes *how* storage is backed,
    # not the tensor contents.
    assert set(eager) == set(mmapd)
    for key in eager:
        assert torch.equal(eager[key], mmapd[key])


def test_touch_all_returns_finite_checksum(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt = _write_tiny_checkpoint(tmp_path)
    state = mod._load_model_state(ckpt, use_mmap=True)
    acc = mod._touch_all(state)
    # The checksum just has to be a real number (its purpose is the page-fault
    # side effect, not the value).
    assert isinstance(acc, float)


def test_run_worker_reports_expected_fields(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt = _write_tiny_checkpoint(tmp_path)
    res = mod.run_worker(ckpt, use_mmap=True, touch=True)
    assert res["mode"] == "mmap"
    assert res["file_bytes"] == ckpt.stat().st_size
    assert res["n_state_tensors"] > 0
    assert res["state_param_mb"] >= 0.0
    # Deltas are floored at 0 (page-out noise must never report negative growth).
    assert res["post_load_delta_mb"] >= 0.0
    assert res["post_touch_delta_mb"] >= 0.0


def test_functional_check_matches_eager(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt = _write_tiny_checkpoint(tmp_path)
    func = mod.functional_check(ckpt)
    # assign=True keeps the mmap storage but must yield byte-identical logits.
    assert func["functional_match"] is True
    assert func["max_abs_logit_diff"] == 0.0


def test_worker_mode_prints_result_json(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    ckpt = _write_tiny_checkpoint(tmp_path)
    rc = mod.main(["--worker", "eager", "--checkpoint", str(ckpt)])
    assert rc == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith(mod.RESULT_PREFIX))
    payload = json.loads(line[len(mod.RESULT_PREFIX):])
    assert payload["mode"] == "eager"


def test_main_end_to_end_spawns_workers(tmp_path: Path) -> None:
    mod = _load_script()
    ckpt = _write_tiny_checkpoint(tmp_path)
    out = tmp_path / "mmap.json"
    # Parent role: this really spawns two subprocesses (eager + mmap) plus an
    # in-process functional check.
    rc = mod.main(["--checkpoint", str(ckpt), "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload) >= {"checkpoint", "eager", "mmap", "functional"}
    assert payload["eager"]["mode"] == "eager"
    assert payload["mmap"]["mode"] == "mmap"
    assert payload["functional"]["functional_match"] is True


def test_main_rejects_missing_checkpoint(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    rc = mod.main(["--checkpoint", str(tmp_path / "nope.pt"), "--json", str(tmp_path / "o.json")])
    assert rc == 2
    assert "error:" in capsys.readouterr().err
