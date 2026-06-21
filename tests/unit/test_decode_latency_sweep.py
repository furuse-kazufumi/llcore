# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/decode_latency_sweep.py`` (decode-step latency vs context age T).

Tiny configs / short context lengths / 1 repeat so the end-to-end run (3 modes x N
lengths as subprocesses) is fast. Asserts structure + that every mode/length produces
a timing; the headline scaling exponents (recurrent flat O(1) vs GPT growing) are
validated by the real run, not here.
"""
from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest


def _load_script() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "decode_latency_sweep.py"
    spec = importlib.util.spec_from_file_location("decode_latency_sweep", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_lengths_sorts_dedups_validates() -> None:
    mod = _load_script()
    assert mod._parse_lengths("512,128,512,256") == [128, 256, 512]
    for bad in ("", "0", "-4", "abc", "8,"):
        with pytest.raises(ValueError):
            mod._parse_lengths(bad)


def test_scaling_exponent_recovers_known_slope() -> None:
    mod = _load_script()
    lengths = [1, 2, 4, 8]
    flat = [3.0 for _ in lengths]                  # time ~ T^0 (O(1) decode)
    linear = [float(t) for t in lengths]           # time ~ T^1
    quad = [float(t * t) for t in lengths]         # time ~ T^2
    assert mod._scaling_exponent(lengths, flat) == pytest.approx(0.0, abs=1e-6)
    assert mod._scaling_exponent(lengths, linear) == pytest.approx(1.0, abs=1e-6)
    assert mod._scaling_exponent(lengths, quad) == pytest.approx(2.0, abs=1e-6)
    assert math.isnan(mod._scaling_exponent([1], [1.0]))


def test_run_worker_each_mode() -> None:
    mod = _load_script()
    for mode in ("gpt", "recurrent", "rwkv"):
        rec = mod.run_worker(mode, t=8, n_embd=16, n_layer=1, n_head=2, vocab=32,
                             repeats=1, warmup=0)
        assert rec["mode"] == mode
        assert rec["t"] == 8
        # 同一ランで prefill / decode の両方を報告する
        for key in ("prefill_median_ms", "prefill_min_ms", "decode_median_ms", "decode_min_ms"):
            assert rec[key] >= 0.0
        assert rec["repeats"] == 1


def test_time_prefill_decode_returns_pair() -> None:
    mod = _load_script()
    import torch
    from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
    model = RecurrentLM(RecurrentConfig(vocab_size=32, block_size=8, n_layer=1,
                                        n_embd=16, state_size=16))
    model.eval()
    with torch.no_grad():
        prefill, decode = mod._time_prefill_decode("recurrent", model, t=8, vocab=32)
    assert prefill >= 0.0 and decode >= 0.0


def test_worker_mode_prints_result_json(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    rc = mod.main(["--worker", "rwkv", "--t", "8", "--n-embd", "16",
                   "--n-layer", "1", "--n-head", "2", "--vocab", "32",
                   "--repeats", "1", "--warmup", "0"])
    assert rc == 0
    out = capsys.readouterr().out
    line = next(line for line in out.splitlines() if line.startswith(mod.RESULT_PREFIX))
    assert json.loads(line[len(mod.RESULT_PREFIX):])["mode"] == "rwkv"


def test_worker_requires_positive_t(capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    assert mod.main(["--worker", "gpt"]) == 2
    assert "error:" in capsys.readouterr().err


def test_main_end_to_end_tiny(tmp_path: Path) -> None:
    mod = _load_script()
    out = tmp_path / "decode.json"
    rc = mod.main(["--lengths", "8,16", "--n-embd", "16", "--n-layer", "1",
                   "--n-head", "2", "--vocab", "32", "--repeats", "1", "--warmup", "0",
                   "--json", str(out)])
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload["records"]) == {"gpt", "recurrent", "rwkv"}
    for mode in ("gpt", "recurrent", "rwkv"):
        assert [r["t"] for r in payload["records"][mode]] == [8, 16]
        for r in payload["records"][mode]:  # 各レコードが prefill/decode 両方を持つ
            assert "prefill_median_ms" in r and "decode_median_ms" in r
    assert set(payload["prefill_growth_ratio"]) == {"gpt", "recurrent", "rwkv"}
    assert set(payload["decode_growth_ratio"]) == {"gpt", "recurrent", "rwkv"}
    assert "axis" in payload and "honest" in payload  # amortization axis + honest caveats persisted


def test_main_rejects_bad_lengths(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    mod = _load_script()
    assert mod.main(["--lengths", "0", "--json", str(tmp_path / "x.json")]) == 2
    assert "error:" in capsys.readouterr().err
