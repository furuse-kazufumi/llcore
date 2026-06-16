# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`scripts.p1_corpus_probe`."""
from __future__ import annotations

from contextlib import redirect_stdout
from importlib import import_module
import io
import json
from pathlib import Path
import subprocess
import sys

p1_corpus_probe = import_module("scripts.p1_corpus_probe")


def test_analyze_corpus_mix_reports_new_chars_and_oov(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("cadx\n", encoding="utf-8")

    report = p1_corpus_probe.analyze_corpus_mix(base, [extra])

    assert report["base"]["chars"] == len("abca\n")
    assert report["base"]["vocab_size"] == 4
    assert report["combined"]["chars"] == len("abca\ncadx\n")
    assert report["combined"]["added_vocab_count_vs_base"] == 2
    assert report["combined"]["added_vocab_preview_vs_base"] == ["d", "x"]
    assert report["extras"][0]["new_char_count_vs_base"] == 2
    assert report["extras"][0]["new_char_preview_vs_base"] == ["d", "x"]
    assert report["extras"][0]["oov_chars_vs_base"] == 2
    assert report["extras"][0]["oov_rate_vs_base"] == 0.4


def test_analyze_corpus_mix_matches_training_normalization(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    base.write_text("abc\n\n", encoding="utf-8")
    extra.write_text("xyz\n\n\n", encoding="utf-8")

    report = p1_corpus_probe.analyze_corpus_mix(base, [extra])

    assert report["base"]["chars"] == len("abc\n")
    assert report["extras"][0]["chars"] == len("xyz\n")
    assert report["combined"]["chars"] == len("abc\nxyz\n")


def test_main_prints_markdown_and_writes_json(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    json_path = tmp_path / "nested" / "report.json"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("cadx\n", encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        rc = p1_corpus_probe.main([str(base), str(extra), "--json", str(json_path)])

    assert rc == 0
    text = out.getvalue()
    assert "| corpus | chars | vocab | new chars vs base (uniq) | OOV vs base (occurs) | sha256 |" in text
    assert "[headline] combined vocab delta vs base: 2 chars;" in text
    assert "extra sha256 hashes each individually normalized candidate file" in text
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["extras"][0]["oov_chars_vs_base"] == 2
    assert payload["combined_selected"] == payload["combined"]


def test_main_supports_extra_corpus_manifest(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extras" / "extra.txt"
    extra.parent.mkdir()
    manifest = tmp_path / "extras.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("cadx\n", encoding="utf-8")
    manifest.write_text("# comment\nextras/extra.txt\n", encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        rc = p1_corpus_probe.main(
            [str(base), "--extra-corpus-manifest", str(manifest)]
        )

    assert rc == 0
    text = out.getvalue()
    assert "extra:extra.txt" in text
    assert "[headline] combined vocab delta vs base: 2 chars;" in text


def test_preview_chars_shows_truncation_count() -> None:
    chars = {chr(ord("a") + i) for i in range(14)}

    preview = p1_corpus_probe._preview_chars(chars, limit=12)

    assert len(preview) == 13
    assert preview[-1] == "...(+2 more)"


def test_selected_extra_paths_filters_by_oov_rate_and_new_chars() -> None:
    report = {
        "extras": [
            {"path": "keep.txt", "oov_rate_vs_base_raw": 0.1, "oov_rate_vs_base": 0.1, "new_char_count_vs_base": 2},
            {"path": "drop_oov.txt", "oov_rate_vs_base_raw": 0.6, "oov_rate_vs_base": 0.6, "new_char_count_vs_base": 2},
            {"path": "drop_vocab.txt", "oov_rate_vs_base_raw": 0.1, "oov_rate_vs_base": 0.1, "new_char_count_vs_base": 5},
        ]
    }

    selected = p1_corpus_probe._selected_extra_paths(
        report,
        max_oov_rate=0.2,
        max_new_chars=3,
    )

    assert selected == ["keep.txt"]


def test_selected_extra_paths_uses_raw_oov_rate_not_rounded_display_value() -> None:
    report = {
        "extras": [
            {
                "path": "boundary.txt",
                "oov_rate_vs_base_raw": 1 / 3,
                "oov_rate_vs_base": 0.333333,
                "new_char_count_vs_base": 1,
            }
        ]
    }

    selected = p1_corpus_probe._selected_extra_paths(
        report,
        max_oov_rate=0.333333,
        max_new_chars=None,
    )

    assert selected == []


def test_main_writes_filtered_manifest(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    keep = tmp_path / "keep.txt"
    drop = tmp_path / "drop.txt"
    manifest = tmp_path / "nested" / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    keep.write_text("abcb\n", encoding="utf-8")
    drop.write_text("xyz\n", encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        rc = p1_corpus_probe.main(
            [
                str(base),
                str(keep),
                str(drop),
                "--max-oov-rate",
                "0.2",
                "--max-new-chars",
                "1",
                "--write-manifest",
                str(manifest),
            ]
        )

    assert rc == 0
    assert manifest.read_text(encoding="utf-8") == (
        "# Generated by scripts/p1_corpus_probe.py\n"
        "../keep.txt\n"
    )
    bundle_payload = json.loads(manifest.with_suffix(".txt.bundle.json").read_text(encoding="utf-8"))
    assert bundle_payload["generated_by"] == "scripts/p1_corpus_probe.py"
    assert bundle_payload["bundle"]["combined"]["chars"] == len("abca\nabcb\n")
    assert bundle_payload["bundle"]["combined"]["includes_base"] is True
    assert "[selection] 1/2 extras pass" in out.getvalue()
    assert "[selected subset]" in out.getvalue()


def test_main_json_includes_selected_combined_when_filtering(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    keep = tmp_path / "keep.txt"
    drop = tmp_path / "drop.txt"
    json_path = tmp_path / "report.json"
    base.write_text("abca\n", encoding="utf-8")
    keep.write_text("abcb\n", encoding="utf-8")
    drop.write_text("xyz\n", encoding="utf-8")

    rc = p1_corpus_probe.main(
        [
            str(base),
            str(keep),
            str(drop),
            "--max-oov-rate",
            "0.2",
            "--max-new-chars",
            "1",
            "--json",
            str(json_path),
        ]
    )

    assert rc == 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["combined"]["chars"] == len("abca\nabcb\nxyz\n")
    assert payload["combined_selected"]["chars"] == len("abca\nabcb\n")


def test_main_handles_empty_selection_without_bundle_metadata(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    drop = tmp_path / "drop.txt"
    manifest = tmp_path / "selected.txt"
    json_path = tmp_path / "report.json"
    base.write_text("abca\n", encoding="utf-8")
    drop.write_text("xyz\n", encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        rc = p1_corpus_probe.main(
            [
                str(base),
                str(drop),
                "--max-oov-rate",
                "0.0",
                "--write-manifest",
                str(manifest),
                "--json",
                str(json_path),
            ]
        )

    assert rc == 0
    assert manifest.read_text(encoding="utf-8") == "# Generated by scripts/p1_corpus_probe.py\n"
    assert not manifest.with_suffix(".txt.bundle.json").exists()
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["combined_selected"]["chars"] == len("abca\n")
    assert payload["combined_selected"]["sha256"] == payload["base"]["sha256"]
    assert "[selection] 0 extras selected; skipped manifest bundle metadata." in out.getvalue()


def test_script_runs_via_subprocess_entrypoint(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("cadx\n", encoding="utf-8")
    repo_root = Path(__file__).resolve().parents[2]

    proc = subprocess.run(
        [sys.executable, "scripts/p1_corpus_probe.py", str(base), str(extra)],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert proc.returncode == 0
    assert "new chars vs base (uniq)" in proc.stdout
    assert "extra sha256 hashes each individually normalized candidate file" in proc.stdout
