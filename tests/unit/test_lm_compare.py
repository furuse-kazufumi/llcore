# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.compare`."""
from __future__ import annotations

import re
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
    assert "decode_tok_per_s" in result["throughput"]["gpt"]["1"]
    assert result["pareto"]["memory_winner_by_slope"] == "recurrent/rwkv"
    recurrent_be = result["pareto"]["recurrent_break_even_prompt_len_vs_gpt"]
    recurrent_bytes = result["memory"]["recurrent_state_bytes"]
    assert gpt_kv_bytes(CharGPT(GPTConfig(vocab_size=8, block_size=16, n_layer=2, n_head=4, n_embd=64)), recurrent_be) >= recurrent_bytes


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
    assert out_path.with_suffix(".md").exists()
    assert out_path.with_suffix(".svg").exists()
    assert "| Model | PPL | Unigram PPL | Ratio vs GPT | Passes gate |" in out_path.with_suffix(
        ".md"
    ).read_text(encoding="utf-8")
    svg_text = out_path.with_suffix(".svg").read_text(encoding="utf-8")
    assert "<svg" in svg_text
    assert "stroke-dasharray=\"8 6\"" in svg_text
    gpt_polylines = re.findall(r'<polyline fill="none" stroke="#2563eb"[^>]* points="([^"]+)"', svg_text)
    assert len(gpt_polylines) >= 2
    measured_points = gpt_polylines[0].split()
    projected_points = gpt_polylines[1].split()
    assert len(measured_points) == 2
    assert len(projected_points) == 2
    assert measured_points[-1] == projected_points[0]


def test_compare_svg_boundary_uses_block_size_not_throughput_sweep(tmp_path: Path) -> None:
    text = ("0123456789" * 60) + "\n"
    out_path = tmp_path / "boundary" / "result.json"
    compare_on_text(
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
        out_path=out_path,
    )
    svg_text = out_path.with_suffix(".svg").read_text(encoding="utf-8")
    gpt_polylines = re.findall(r'<polyline fill="none" stroke="#2563eb"[^>]* points="([^"]+)"', svg_text)
    assert len(gpt_polylines) >= 2
    measured_points = gpt_polylines[0].split()
    projected_points = gpt_polylines[1].split()
    assert len(measured_points) == 2
    assert len(projected_points) == 2
    assert measured_points[-1] == projected_points[0]


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


def test_compare_uses_block_size_relative_default_prompt_lengths() -> None:
    text = ("0123456789" * 60) + "\n"
    result = compare_on_text(
        text,
        cfg=CompareConfig(
            block_size=12,
            max_iters=2,
            eval_iters=1,
            batch_size=4,
            throughput_prompt_lens=None,
            throughput_new_tokens=2,
            throughput_repeats=1,
        ),
    )
    assert set(result["throughput"]["gpt"]) == {"1", "16", "12", "48"}


def test_compare_config_rejects_invalid_throughput_settings() -> None:
    invalid_cfgs = (
        dict(throughput_new_tokens=0),
        dict(throughput_repeats=0),
        dict(throughput_prompt_lens=(0,)),
    )
    for kwargs in invalid_cfgs:
        try:
            CompareConfig(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected ValueError for {kwargs}")


def test_compare_rejects_non_json_output_suffix(tmp_path: Path) -> None:
    text = ("0123456789" * 60) + "\n"
    try:
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
            out_path=tmp_path / "result.svg",
        )
    except ValueError as exc:
        assert ".json" in str(exc)
    else:
        raise AssertionError("expected compare_on_text to reject non-json out_path")
