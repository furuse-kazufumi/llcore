# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`llcore.lm.corpus`."""
from __future__ import annotations

from pathlib import Path

import json
import shutil

from llcore.lm.corpus import (
    build_utf8_corpus_bundle,
    read_corpus_manifest,
    resolve_extra_corpus_files,
    sha256_text,
    verify_corpus_manifest_bundle,
)


def test_read_corpus_manifest_resolves_relative_paths_and_skips_comments(tmp_path: Path) -> None:
    corpora_dir = tmp_path / "corpora"
    corpora_dir.mkdir()
    extra_a = corpora_dir / "a.txt"
    extra_b = corpora_dir / "nested" / "b.txt"
    extra_b.parent.mkdir()
    extra_a.write_text("a\n", encoding="utf-8")
    extra_b.write_text("b\n", encoding="utf-8")
    manifest = tmp_path / "extras.txt"
    manifest.write_text("# comment\ncorpora/a.txt\n\ncorpora/nested/b.txt\n", encoding="utf-8")

    resolved = read_corpus_manifest(manifest)

    assert resolved == [str(extra_a.resolve()), str(extra_b.resolve())]


def test_resolve_extra_corpus_files_preserves_explicit_then_manifest_order(tmp_path: Path) -> None:
    explicit = tmp_path / "explicit.txt"
    explicit.write_text("x\n", encoding="utf-8")
    extra = tmp_path / "extra.txt"
    extra.write_text("y\n", encoding="utf-8")
    manifest = tmp_path / "extras.txt"
    manifest.write_text("extra.txt\n", encoding="utf-8")

    resolved = resolve_extra_corpus_files([explicit], [manifest])

    assert resolved == [str(explicit.resolve()), str(extra.resolve())]


def test_resolve_extra_corpus_files_deduplicates_explicit_files_and_skips_base_file(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    base.write_text("b\n", encoding="utf-8")
    extra.write_text("e\n", encoding="utf-8")

    resolved = resolve_extra_corpus_files([extra, base, extra], None, base_file=base)

    assert resolved == [str(extra.resolve())]


def test_read_corpus_manifest_reports_missing_entry_with_line_context(tmp_path: Path) -> None:
    manifest = tmp_path / "extras.txt"
    manifest.write_text("# ok\nmissing.txt\n", encoding="utf-8")

    try:
        read_corpus_manifest(manifest)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing manifest entry to fail closed")

    assert str(manifest) in message
    assert ":2:" in message
    assert "missing.txt" in message


def test_build_utf8_corpus_bundle_reports_ordered_file_and_combined_hashes(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("ab\n\n", encoding="utf-8")
    second.write_text("cd\n", encoding="utf-8")

    bundle = build_utf8_corpus_bundle([first, second])

    assert [Path(item["path"]).name for item in bundle["files"]] == ["first.txt", "second.txt"]
    assert bundle["files"][0]["chars"] == len("ab\n")
    assert bundle["files"][1]["chars"] == len("cd\n")
    assert bundle["combined"]["chars"] == len("ab\ncd\n")
    assert bundle["combined"]["includes_base"] is False
    assert len(bundle["bundle_sha256"]) == 64


def test_build_utf8_corpus_bundle_can_include_base_and_is_path_independent(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "nested" / "extra.txt"
    extra.parent.mkdir()
    base.write_text("ab\n", encoding="utf-8")
    extra.write_text("cd\n", encoding="utf-8")

    bundle_a = build_utf8_corpus_bundle([extra], base_file=base)
    bundle_b = build_utf8_corpus_bundle([Path(str(extra.resolve()))], base_file=base)

    assert bundle_a["combined"]["chars"] == len("ab\ncd\n")
    assert bundle_a["combined"]["includes_base"] is True
    assert bundle_a["bundle_sha256"] == bundle_b["bundle_sha256"]


def test_verify_corpus_manifest_bundle_accepts_matching_bundle(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "extras.txt"
    base.write_text("ab\n", encoding="utf-8")
    extra.write_text("cd\n", encoding="utf-8")
    manifest.write_text("extra.txt\n", encoding="utf-8")
    payload = {
        "generated_by": "test",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": sha256_text(manifest.read_text(encoding="utf-8")),
        "bundle": build_utf8_corpus_bundle([extra], base_file=base),
    }
    manifest.with_suffix(".txt.bundle.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = verify_corpus_manifest_bundle(manifest, [extra], base_file=base)

    assert summary is not None
    assert summary["entry_count"] == 1
    assert summary["generated_by"] == "test"
    assert summary["includes_base"] is True


def test_resolve_extra_corpus_files_rejects_manifest_with_base_or_duplicate_entries(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "extras.txt"
    base.write_text("ab\n", encoding="utf-8")
    extra.write_text("cd\n", encoding="utf-8")
    manifest.write_text("extra.txt\nbase.txt\nextra.txt\n", encoding="utf-8")
    payload = {
        "generated_by": "manual-test",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": sha256_text(manifest.read_text(encoding="utf-8")),
        "bundle": build_utf8_corpus_bundle([extra, base, extra], base_file=base),
    }
    manifest.with_suffix(".txt.bundle.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summaries: list[dict[str, object]] = []

    try:
        resolve_extra_corpus_files(
            None,
            [manifest],
            base_file=base,
            verified_bundle_summaries=summaries,
        )
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected manifest base/duplicate entries to fail closed")

    assert "collapse after base/duplicate filtering" in message
    assert summaries == []


def test_verify_corpus_manifest_bundle_rejects_drifted_manifest_or_base(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    other_base = tmp_path / "other_base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "extras.txt"
    base.write_text("ab\n", encoding="utf-8")
    other_base.write_text("xy\n", encoding="utf-8")
    extra.write_text("cd\n", encoding="utf-8")
    manifest.write_text("extra.txt\n", encoding="utf-8")
    payload = {
        "generated_by": "test",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": sha256_text(manifest.read_text(encoding="utf-8")),
        "bundle": build_utf8_corpus_bundle([extra], base_file=base),
    }
    bundle_path = manifest.with_suffix(".txt.bundle.json")
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest.write_text("extra.txt\n# drift\n", encoding="utf-8")
    try:
        verify_corpus_manifest_bundle(manifest, [extra], base_file=base)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected manifest hash drift to fail closed")
    assert "manifest bundle metadata drift" in message

    manifest.write_text("extra.txt\n", encoding="utf-8")
    payload["manifest_sha256"] = sha256_text(manifest.read_text(encoding="utf-8"))
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        verify_corpus_manifest_bundle(manifest, [extra], base_file=other_base)
    except ValueError as exc:
        mismatch = str(exc)
    else:
        raise AssertionError("expected base mismatch to fail closed")
    assert "manifest bundle contents no longer match" in mismatch


def test_verify_corpus_manifest_bundle_ignores_stale_absolute_paths_after_move(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    base = source / "base.txt"
    extra = source / "extra.txt"
    manifest = source / "extras.txt"
    base.write_text("ab\n", encoding="utf-8")
    extra.write_text("cd\n", encoding="utf-8")
    manifest.write_text("extra.txt\n", encoding="utf-8")
    payload = {
        "generated_by": "test",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": sha256_text(manifest.read_text(encoding="utf-8")),
        "bundle": build_utf8_corpus_bundle([extra], base_file=base),
    }
    bundle_path = manifest.with_suffix(".txt.bundle.json")
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    moved = tmp_path / "moved"
    moved.mkdir()
    moved_base = moved / "base.txt"
    moved_extra = moved / "extra.txt"
    moved_manifest = moved / "extras.txt"
    moved_bundle = moved / "extras.txt.bundle.json"
    shutil.copy2(base, moved_base)
    shutil.copy2(extra, moved_extra)
    shutil.copy2(manifest, moved_manifest)
    shutil.copy2(bundle_path, moved_bundle)

    verify_corpus_manifest_bundle(moved_manifest, [moved_extra], base_file=moved_base)
