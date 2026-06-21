# SPDX-License-Identifier: Apache-2.0
"""Regression tests for scripts/nas_pareto_report.py (the honest-disclosure Markdown renderer).

These pin the disclosure-critical invariants: the headline must quote HOLDOUT Δnll (not the
optimistically biased selection Δnll), the scope/conversational-claim refusal must always render,
missing optional probes (needle/cross-corpus) must be disclosed as gaps rather than silently dropped,
and the CLI must fail cleanly (rc=2) on a missing or malformed JSON instead of crashing.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_script(script_name: str) -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPORT = _load_script("nas_pareto_report.py")
SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "nas_pareto_report.py"


def _full_report() -> dict[str, Any]:
    """A complete proxy-v2 result with a 'suggestive' verdict, regime sweep, and a needle probe."""
    return {
        "model_dir": "D:/models/Qwen2.5-0.5B-Instruct",
        "n_layer": 24,
        "mixers": ["softmax", "sliding", "linear"],
        "base_nll": 4.4155,
        "real_evals": 380,
        "elapsed_s": 12345.0,
        "verdict": "memetic frontier ties greedy (separable)",
        "hypervolume": {"greedy": 12.3, "evolved": 12.5},
        "greedy_frontier": [{"pct_mem_saved": 40.0, "delta_nll": 0.2}],
        "evolved_frontier": [{"pct_mem_saved": 40.0, "delta_nll": 0.18}],
        "proxy_v2": {
            "scope": "next_token_nll_proxy",
            "conversational_claim": None,
            "inner_context": 1024,
            "holdout_offset": 8192,
            "holdout_windows": 8.0,
            "fast_windows": 8.0,
            "aggressive_genome_pct": 62.0,
            "proxy_vs_judge_tau": 0.6,
            "hv_gain_ci": {"gain_pct_mean": 1.6, "ci_lo": 0.5, "ci_hi": 3.9, "p_memetic_wins": 0.98},
            "frontier_holdout": [
                {
                    "pct": 40.0,
                    "delta_nll_selection": 0.18,
                    "delta_nll_heldout": 0.2,
                    "optimism_gap": -0.02,
                    "ci_lo": 0.1,
                    "ci_hi": 0.3,
                    "p_worse": 0.9,
                    "pos_frac": 0.8,
                    "p_sign": 0.05,
                }
            ],
            "context_sweep": {
                "1024": {"mean": 0.02, "ci_lo": 0.0, "ci_hi": 0.04, "p_worse": 0.9,
                         "pos_frac": 0.7, "p_sign": 0.1, "n_windows": 8.0},
                "2048": {"mean": 0.06, "ci_lo": 0.03, "ci_hi": 0.09, "p_worse": 0.99,
                         "pos_frac": 0.95, "p_sign": 0.01, "n_windows": 8.0},
            },
            "attention_kl": {"mean": 0.12, "max": 0.3, "sum": 1.4,
                             "per_layer": [{"layer": 5.0, "kl": 0.3}]},
            "needle": {"horizon": 4096,
                       "by_depth": {"4096:0.5": {"argmax_acc": 0.0, "control_acc": 1.0,
                                                 "mean_logprob": -9.1}}},
            "cross_corpus": None,
            "verdict": {
                "memetic_vs_greedy": "memetic beats greedy: +1.6% HV (95% CI 0.5..3.9%)",
                "confidence": "suggestive",
                "notes": "proxy-vs-judge tau 0.60 < 0.7",
                "ci_reliability": "point estimate, CI unreliable (K=8<12)",
            },
            "note": "next-token-nll proxy with paired multi-window bootstrap CIs",
        },
    }


def test_headline_uses_holdout_and_discloses_optimism_gap() -> None:
    md = REPORT.render(_full_report())
    assert "HEADLINE" in md and "holdout" in md.lower()
    # both the selection and the holdout numbers must be present (the gap is the disclosure)
    assert "optimism gap" in md.lower()
    assert "0.2000" in md and "0.1800" in md
    assert "-0.0200" in md  # the optimism_gap value itself


def test_scope_and_conversational_refusal_always_render() -> None:
    md = REPORT.render(_full_report())
    assert "next_token_nll_proxy" in md
    assert "NOT inferred from these perplexity proxies" in md


def test_prior_art_positioning_always_renders() -> None:
    # the RAD-grounded differentiation (proxy-noise honest-disclosure layer) must be disclosed
    # so the report cannot read as if a novel result were being over-claimed.
    md = REPORT.render(_full_report())
    assert "Prior-art positioning" in md
    assert "CI_lo > 0" in md
    assert "honest negative" in md


def test_ci_reliability_and_suggestive_downgrade_surface() -> None:
    md = REPORT.render(_full_report())
    assert "suggestive" in md
    assert "CI unreliable (K=8<12)" in md
    assert "p_memetic_wins" in md


def test_regime_sweep_rows_present() -> None:
    md = REPORT.render(_full_report())
    assert "1024" in md and "2048" in md
    assert "regime dependence" in md.lower()


def test_missing_needle_disclosed_as_gap_not_dropped() -> None:
    rep = _full_report()
    rep["proxy_v2"]["needle"] = None
    md = REPORT.render(rep)
    assert "UNTESTED" in md  # the gap is named, not silently omitted


def test_v1_report_without_proxy_v2_renders() -> None:
    rep = _full_report()
    del rep["proxy_v2"]
    md = REPORT.render(rep)
    assert "proxy-v2 rigorous tier absent" in md


def test_cli_missing_file_returns_rc2(tmp_path: Path) -> None:
    r = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(tmp_path / "nope.json")],
        capture_output=True, text=True,
    )
    assert r.returncode == 2


def test_cli_malformed_json_returns_rc2(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    r = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(bad)],
        capture_output=True, text=True,
    )
    assert r.returncode == 2


def test_cli_writes_markdown_file(tmp_path: Path) -> None:
    src = tmp_path / "nas_pareto.json"
    src.write_text(json.dumps(_full_report()), encoding="utf-8")
    out = tmp_path / "report.md"
    r = subprocess.run(
        [sys.executable, str(SCRIPT_PATH), str(src), "-o", str(out)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    md = out.read_text(encoding="utf-8")
    assert md.startswith("# NAS Pareto")
    assert "next_token_nll_proxy" in md


@pytest.mark.parametrize("dn", [0.0, -0.5, 1.25])
def test_frontier_table_signs_floats(dn: float) -> None:
    md = REPORT.render(
        {"model_dir": "m", "n_layer": 1, "mixers": [], "base_nll": 4.0, "real_evals": 1,
         "verdict": "x", "hypervolume": {"greedy": 1.0, "evolved": 1.0},
         "greedy_frontier": [{"pct_mem_saved": 10.0, "delta_nll": dn}],
         "evolved_frontier": []}
    )
    assert f"{dn:+.4f}" in md
