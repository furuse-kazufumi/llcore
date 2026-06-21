# SPDX-License-Identifier: Apache-2.0
"""Tests for ``scripts/extract_needle_results.py`` against the verified proxy_v2 schema.

These pin the schema the GH Actions offload emits so the b2 integration cannot silently
KeyError on the two real traps: stringified context_sweep keys and ``"mean"`` (not
``"delta_nll"``) being the article's Δnll.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "extract_needle_results.py"
_spec = importlib.util.spec_from_file_location("extract_needle_results", _SCRIPT)
assert _spec is not None and _spec.loader is not None
extract_needle_results = importlib.util.module_from_spec(_spec)
sys.modules["extract_needle_results"] = extract_needle_results
_spec.loader.exec_module(extract_needle_results)


def _report(*, with_needle_horizon: int | None | str = "absent") -> dict[str, Any]:
    """Build a synthetic nas_pareto.json report mirroring eval_proxy.build_proxy_v2_report."""
    sweep_val = lambda mean: {  # noqa: E731 — terse fixture helper
        "mean": mean,
        "ci_lo": mean - 0.05,
        "ci_hi": mean + 0.05,
        "p_worse": 0.99,
        "pos_frac": 0.9,
        "p_sign": 0.01,
        "n_windows": 12.0,
    }
    proxy: dict[str, Any] = {
        "scope": "next_token_nll_proxy",
        # NOTE: keys are STRINGS, exactly as build_proxy_v2_report emits.
        "context_sweep": {
            "256": sweep_val(0.761),
            "512": sweep_val(1.012),
            "1024": sweep_val(1.182),
            "2048": sweep_val(1.534),
        },
    }
    if with_needle_horizon != "absent":
        proxy["needle"] = {"horizon": with_needle_horizon, "by_depth": {}}
    return {"proxy_v2": proxy}


def test_extract_uses_string_keys_and_mean_field() -> None:
    block = extract_needle_results.extract(_report())
    sweep = block["context_sweep"]
    # the real trap: keys are strings; mean (not delta_nll) is the Δnll
    assert sweep["2048"]["mean"] == pytest.approx(1.534)
    assert "delta_nll" not in sweep["2048"]


def test_render_sorts_lengths_numerically_and_shows_ci() -> None:
    out = extract_needle_results.render(extract_needle_results.extract(_report()))
    # 2048 must sort after 1024 (numeric, not lexicographic)
    assert out.index("256") < out.index("1024") < out.index("2048")
    assert "95% CI" in out
    assert "+1.5340" in out  # 2048 mean rendered


def test_needle_none_vs_failed() -> None:
    none_out = extract_needle_results.render(
        extract_needle_results.extract(_report(with_needle_horizon=None))
    )
    assert "horizon=None" in none_out and "never failed" in none_out
    failed_out = extract_needle_results.render(
        extract_needle_results.extract(_report(with_needle_horizon=2048))
    )
    assert "horizon=2048" in failed_out and "breaks" in failed_out


def test_missing_proxy_v2_raises() -> None:
    with pytest.raises(KeyError):
        extract_needle_results.extract({"frontier": []})
