# SPDX-License-Identifier: Apache-2.0
"""Lightweight integrity checks for tracked LM comparison artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import cast
from xml.etree import ElementTree

from llcore.lm.compare import _render_memory_curve_svg, _render_ppl_table  # type: ignore[import-untyped]


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "docs" / "artifacts"
SUMMARY_PATH = ARTIFACTS_DIR / "lm_recurrent_interim_summary.md"
VERDICT_PATH = ARTIFACTS_DIR / "lm_recurrent_verdict.md"
_BLOCK64_IDENTICAL_SVG_STEMS = (
    "lm_recurrent_pilot120",
    "lm_recurrent_pilot160",
    "lm_recurrent_pilot160_seed2026",
    "lm_recurrent_pilot160_seed7",
    "lm_recurrent_pilot240",
    "lm_recurrent_pilot240_seed2026",
    "lm_recurrent_pilot240_seed7",
)
_CANONICAL_BLOCK64_SVG_STEM = "lm_recurrent_pilot160"
_SURVIVING_SVG_STEMS = (
    _CANONICAL_BLOCK64_SVG_STEM,
    "lm_recurrent_pilot256_40",
)


def _tracked_pilot_stems() -> list[str]:
    return sorted(path.stem for path in ARTIFACTS_DIR.glob("lm_recurrent_pilot*.json"))


def _summary_run_name(stem: str) -> str:
    return stem.removeprefix("lm_recurrent_")


def _summary_svg_target(stem: str) -> str:
    if stem in _BLOCK64_IDENTICAL_SVG_STEMS:
        return f"./{_CANONICAL_BLOCK64_SVG_STEM}.svg"
    return f"./{stem}.svg"


def _strict_gate_summary_label(result: dict[str, object]) -> str:
    verdict = cast(dict[str, dict[str, object]], result["verdict"])
    passed = [
        name
        for name in ("gpt", "recurrent", "rwkv")
        if bool(verdict[name]["passes_unigram_gate"])
    ]
    if not passed:
        return "all fail"
    return f"{'/'.join(passed)} pass"


def _tracked_seeded_stems(prefix: str) -> list[str]:
    return sorted(
        stem
        for stem in _tracked_pilot_stems()
        if stem == prefix or stem.startswith(f"{prefix}_seed")
    )


def _reproduction_block(text: str, stem: str) -> str:
    run_name = _summary_run_name(stem)
    pattern = rf"- `{re.escape(run_name)}`:\r?\n```powershell\r?\n(.*?)\r?\n```"
    match = re.search(pattern, text, re.DOTALL)
    assert match is not None
    return match.group(1)


def test_tracked_recurrent_svgs_are_well_formed_xml() -> None:
    for stem in _SURVIVING_SVG_STEMS:
        json_path = ARTIFACTS_DIR / f"{stem}.json"
        svg_path = ARTIFACTS_DIR / f"{stem}.svg"
        svg_text = svg_path.read_text(encoding="utf-8")
        result = json.loads(json_path.read_text(encoding="utf-8"))
        root = ElementTree.fromstring(svg_text)
        assert root.tag.endswith("svg")
        assert svg_text == _render_memory_curve_svg(result)
        assert 'stroke-dasharray="8 6"' in svg_text
        assert "GPT KV (measured)" in svg_text
        assert "GPT KV (projection)" in svg_text
        gpt_polylines = re.findall(r'<polyline fill="none" stroke="#2563eb"[^>]* points="([^"]+)"', svg_text)
        assert len(gpt_polylines) >= 2
        measured_points = gpt_polylines[0].split()
        projected_points = gpt_polylines[1].split()
        assert len(measured_points) >= 2
        assert len(projected_points) >= 1
        assert measured_points[-1] == projected_points[0]


def test_block64_memory_svg_hashes_match_and_256_proxy_differs() -> None:
    canonical_svg_text = (ARTIFACTS_DIR / f"{_CANONICAL_BLOCK64_SVG_STEM}.svg").read_text(encoding="utf-8")
    for stem in _BLOCK64_IDENTICAL_SVG_STEMS:
        result = json.loads((ARTIFACTS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        assert _render_memory_curve_svg(result) == canonical_svg_text
    proxy_result = json.loads((ARTIFACTS_DIR / "lm_recurrent_pilot256_40.json").read_text(encoding="utf-8"))
    assert _render_memory_curve_svg(proxy_result) != canonical_svg_text

def test_interim_summary_links_target_existing_tracked_artifacts() -> None:
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    for stem in _tracked_pilot_stems():
        targets = [f"./{stem}.json", f"./{stem}.md", _summary_svg_target(stem)]
        for target in targets:
            assert target in text
            resolved = (SUMMARY_PATH.parent / target.removeprefix("./")).resolve()
            assert resolved.exists()


def test_tracked_recurrent_markdown_matches_json_summary_values() -> None:
    for stem in _tracked_pilot_stems():
        json_path = ARTIFACTS_DIR / f"{stem}.json"
        md_path = ARTIFACTS_DIR / f"{stem}.md"
        result = json.loads(json_path.read_text(encoding="utf-8"))
        md_text = md_path.read_text(encoding="utf-8")
        assert md_text == _render_ppl_table(result)


def test_interim_summary_snapshot_rows_match_tracked_json() -> None:
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    model_labels = {"gpt": "GPT", "recurrent": "Recurrent", "rwkv": "RWKV"}
    for stem in _tracked_pilot_stems():
        result = json.loads((ARTIFACTS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        cfg = result["config"]
        reports = result["reports"]
        ordered = sorted(
            ("gpt", "recurrent", "rwkv"),
            key=lambda name: reports[name]["model_ppl"],
        )
        raw_order = " < ".join(model_labels[name] for name in ordered)
        expected_row = (
            f"| {_summary_run_name(stem)} | {cfg['block_size']} | {cfg['max_iters']} | {cfg['batch_size']} | "
            f"{reports['gpt']['model_ppl']:.3f} | {reports['recurrent']['model_ppl']:.3f} | "
            f"{reports['rwkv']['model_ppl']:.3f} | {reports['gpt']['unigram_ppl']:.3f} | "
            f"{raw_order} | {_strict_gate_summary_label(result)} |"
        )
        assert expected_row in text


def test_interim_summary_reproduction_blocks_reference_tracked_outputs() -> None:
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    for stem in _tracked_pilot_stems():
        result = json.loads((ARTIFACTS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        cfg = result["config"]
        block = _reproduction_block(text, stem)
        assert f'out_path=Path("docs/artifacts/{stem}.json")' in block
        assert f"block_size={cfg['block_size']}" in block
        assert f"max_iters={cfg['max_iters']}" in block
        assert f"batch_size={cfg['batch_size']}" in block
        assert f"seed={cfg['seed']}" in block


def test_verdict_doc_links_target_existing_artifacts() -> None:
    text = VERDICT_PATH.read_text(encoding="utf-8")
    json_targets = [f"./{stem}.json" for stem in _tracked_seeded_stems("lm_recurrent_pilot160")]
    json_targets += [f"./{stem}.json" for stem in _tracked_seeded_stems("lm_recurrent_pilot240")]
    json_targets.append("./lm_recurrent_pilot256_40.json")
    svg_targets = ("./lm_recurrent_pilot160.svg", "./lm_recurrent_pilot256_40.svg")
    for target in (*json_targets, *svg_targets):
        assert target in text
        resolved = (VERDICT_PATH.parent / target.removeprefix("./")).resolve()
        assert resolved.exists()


def test_verdict_doc_states_current_rwkv_claims() -> None:
    text = VERDICT_PATH.read_text(encoding="utf-8")
    assert "RWKV is the most reproducible current candidate" in text
    assert "raw PPL best in `6/6` tracked seeds" in text
    assert "unigram-floor pass in `6/6` tracked seeds" in text


def test_verdict_doc_recomputes_rwkv_claims_from_tracked_json() -> None:
    grouped_stems = _tracked_seeded_stems("lm_recurrent_pilot160") + _tracked_seeded_stems(
        "lm_recurrent_pilot240"
    )
    total = len(grouped_stems)
    rwkv_best = 0
    rwkv_floor = 0
    for stem in grouped_stems:
        result = json.loads((ARTIFACTS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        reports = result["reports"]
        verdict = result["verdict"]
        ordered = sorted(
            ("gpt", "recurrent", "rwkv"),
            key=lambda name: reports[name]["model_ppl"],
        )
        if ordered[0] == "rwkv":
            rwkv_best += 1
        if verdict["rwkv"]["passes_unigram_gate"]:
            rwkv_floor += 1
    assert total == 6
    assert rwkv_best == 6
    assert rwkv_floor == 6


def test_verdict_doc_representative_rows_match_tracked_json() -> None:
    text = VERDICT_PATH.read_text(encoding="utf-8")
    representative_stems = (
        "lm_recurrent_pilot160",
        "lm_recurrent_pilot160_seed2026",
        "lm_recurrent_pilot160_seed7",
        "lm_recurrent_pilot240",
        "lm_recurrent_pilot240_seed2026",
        "lm_recurrent_pilot240_seed7",
    )
    model_labels = {"gpt": "GPT", "recurrent": "Recurrent", "rwkv": "RWKV"}
    for stem in representative_stems:
        result = json.loads((ARTIFACTS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        reports = result["reports"]
        ordered = sorted(
            ("gpt", "recurrent", "rwkv"),
            key=lambda name: reports[name]["model_ppl"],
        )
        raw_order = " < ".join(model_labels[name] for name in ordered)
        expected_row = (
            f"| `{_summary_run_name(stem)}` | {reports['gpt']['model_ppl']:.3f} | "
            f"{reports['recurrent']['model_ppl']:.3f} | {reports['rwkv']['model_ppl']:.3f} | "
            f"`{raw_order}` | `{_strict_gate_summary_label(result)}` |"
        )
        assert expected_row in text
