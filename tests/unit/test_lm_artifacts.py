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
    for target in (
        "./lm_recurrent_pilot120.json",
        "./lm_recurrent_pilot120.md",
        "./lm_recurrent_pilot120.svg",
        "./lm_recurrent_pilot256_40.json",
        "./lm_recurrent_pilot256_40.md",
        "./lm_recurrent_pilot256_40.svg",
    ):
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
