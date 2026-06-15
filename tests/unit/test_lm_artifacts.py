# SPDX-License-Identifier: Apache-2.0
"""Lightweight integrity checks for tracked LM comparison artifacts."""
from __future__ import annotations

import json
import re
from pathlib import Path
from xml.etree import ElementTree


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = REPO_ROOT / "docs" / "artifacts"
SUMMARY_PATH = ARTIFACTS_DIR / "lm_recurrent_interim_summary.md"


def test_tracked_recurrent_svgs_are_well_formed_xml() -> None:
    for svg_name in ("lm_recurrent_pilot120.svg", "lm_recurrent_pilot256_40.svg"):
        svg_path = ARTIFACTS_DIR / svg_name
        svg_text = svg_path.read_text(encoding="utf-8")
        root = ElementTree.fromstring(svg_text)
        assert root.tag.endswith("svg")
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
    for stem in ("lm_recurrent_pilot120", "lm_recurrent_pilot256_40"):
        json_path = ARTIFACTS_DIR / f"{stem}.json"
        md_path = ARTIFACTS_DIR / f"{stem}.md"
        result = json.loads(json_path.read_text(encoding="utf-8"))
        md_text = md_path.read_text(encoding="utf-8")

        reports = result["reports"]
        verdict = result["verdict"]
        for name in ("gpt", "recurrent", "rwkv"):
            report = reports[name]
            verdict_row = verdict[name]
            expected_row = (
                f"| {name} | {report['model_ppl']:.3f} | {report['unigram_ppl']:.3f} | "
                f"{verdict_row['ppl_ratio_vs_gpt']:.3f} | "
                f"{'yes' if verdict_row['passes_unigram_gate'] else 'no'} |"
            )
            assert expected_row in md_text

        for caveat in result["caveats"]:
            assert f"- {caveat}" in md_text
