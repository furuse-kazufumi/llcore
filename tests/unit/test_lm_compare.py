# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.compare`."""
from __future__ import annotations

from pathlib import Path

from llcore.lm.compare import CompareConfig, compare_on_text, gpt_kv_bytes
from llcore.lm.model import CharGPT, GPTConfig


def test_gpt_kv_bytes_formula() -> None:
    model = CharGPT(GPTConfig(vocab_size=8, block_size=16, n_layer=3, n_head=2, n_embd=32))
    assert gpt_kv_bytes(model, prompt_len=10) == 2 * 3 * 10 * 32 * 4


def test_compare_on_text_returns_reports_and_memory() -> None:
    text = ("0123456789" * 60) + "\n"
    result = compare_on_text(
        text,
        cfg=CompareConfig(
            block_size=16,
            max_iters=2,
            eval_iters=1,
            batch_size=4,
            throughput_prompt_lens=(1, 16),
            throughput_new_tokens=2,
            throughput_repeats=1,
        ),
    )
    assert set(result["reports"]) == {"gpt", "recurrent", "rwkv"}
    assert result["memory"]["recurrent_state_bytes"] > 0
    assert result["memory"]["rwkv_state_bytes"] > 0
    assert "counterfactual projection points only" in result["memory"]["notes"]["gpt_kv_bytes"]
    assert "throughput" in result
    assert "pareto" in result
    assert isinstance(result["caveats"], list)


def test_compare_on_text_creates_parent_output_dir(tmp_path: Path) -> None:
    text = ("0123456789" * 60) + "\n"
    out_path = tmp_path / "nested" / "result.json"
    compare_on_text(
        text,
        cfg=CompareConfig(
            block_size=16,
            max_iters=2,
            eval_iters=1,
            batch_size=4,
            throughput_prompt_lens=(1,),
            throughput_new_tokens=2,
            throughput_repeats=1,
        ),
        out_path=out_path,
    )
    assert out_path.exists()


def test_compare_config_validates_head_divisibility() -> None:
    try:
        CompareConfig(n_embd=65, n_head=4)
    except ValueError:
        pass
    else:
        raise AssertionError("expected CompareConfig to reject non-divisible n_embd")


def test_compare_marks_gpt_prompt_above_block_size_as_non_executable() -> None:
    text = ("0123456789" * 60) + "\n"
    result = compare_on_text(
        text,
        cfg=CompareConfig(
            block_size=16,
            max_iters=2,
            eval_iters=1,
            batch_size=4,
            throughput_prompt_lens=(32,),
            throughput_new_tokens=2,
            throughput_repeats=1,
        ),
    )
    assert result["throughput"]["gpt"]["32"]["executable"] is False
