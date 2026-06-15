# SPDX-License-Identifier: Apache-2.0
"""Lightweight integrity checks for tracked LM comparison artifacts."""
from __future__ import annotations

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
