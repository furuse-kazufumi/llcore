# SPDX-License-Identifier: Apache-2.0
"""Lightweight integrity checks for tracked LM comparison artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree

from llcore.lm.compare import _render_memory_curve_svg, _render_ppl_table


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "docs" / "artifacts"
SUMMARY_PATH = ARTIFACTS_DIR / "lm_recurrent_interim_summary.md"


def _tracked_pilot_stems() -> list[str]:
    return sorted(path.stem for path in ARTIFACTS_DIR.glob("lm_recurrent_pilot*.json"))


def _summary_run_name(stem: str) -> str:
    return stem.removeprefix("lm_recurrent_")


def test_tracked_recurrent_svgs_are_well_formed_xml() -> None:
    for stem in _tracked_pilot_stems():
        svg_name = f"{stem}.svg"
        json_path = ARTIFACTS_DIR / f"{stem}.json"
        svg_path = ARTIFACTS_DIR / svg_name
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


def test_interim_summary_links_target_existing_tracked_artifacts() -> None:
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    for stem in _tracked_pilot_stems():
        for suffix in (".json", ".md", ".svg"):
            target = f"./{stem}{suffix}"
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
            f"{raw_order} | all fail |"
        )
        assert expected_row in text


def test_interim_summary_reproduction_blocks_reference_tracked_outputs() -> None:
    text = SUMMARY_PATH.read_text(encoding="utf-8")
    for stem in _tracked_pilot_stems():
        result = json.loads((ARTIFACTS_DIR / f"{stem}.json").read_text(encoding="utf-8"))
        cfg = result["config"]
        assert f'out_path=Path("docs/artifacts/{stem}.json")' in text
        assert f"block_size={cfg['block_size']}" in text
        assert f"max_iters={cfg['max_iters']}" in text
        assert f"batch_size={cfg['batch_size']}" in text
        assert f"seed={cfg['seed']}" in text
