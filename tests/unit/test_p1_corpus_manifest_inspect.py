# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`scripts.p1_manifest_inspect`."""
from __future__ import annotations

from contextlib import redirect_stdout
import hashlib
from importlib import import_module
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from llcore.lm.corpus import build_utf8_corpus_bundle

p1_corpus_probe = import_module("scripts.p1_corpus_probe")
p1_manifest_inspect = import_module("scripts.p1_manifest_inspect")


def test_main_inspects_probe_written_bundle_with_base(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    report_json = tmp_path / "inspect.json"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")

    rc = p1_corpus_probe.main(
        [
            str(base),
            str(extra),
            "--write-manifest",
            str(manifest),
        ]
    )
    assert rc == 0

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main(
            [
                str(manifest),
                "--base-corpus-file",
                str(base),
                "--require-verified",
                "--json",
                str(report_json),
            ]
        )

    assert inspect_rc == 0
    text = out.getvalue()
    assert "[bundle] generated_by=scripts/p1_corpus_probe.py includes_base=True verify=passed" in text
    assert "[manifest] selected.txt: entries=1 effective_entries=1" in text
    assert "[combined] chars=10 vocab=4" in text
    assert "[effective] entries=1" in text
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["verification"]["status"] == "passed"
    assert payload["bundle"]["combined"]["includes_base"] is True
    assert payload["manifest"]["effective_entry_count"] == 1
    assert payload["effective"]["entry_count"] == 1
    assert payload["manifest_verification"][0]["status"] == "verified"
    assert payload["manifest_verification"][0]["generated_by"] == "scripts/p1_corpus_probe.py"


def test_main_skips_probe_bundle_verification_without_base(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")

    rc = p1_corpus_probe.main(
        [
            str(base),
            str(extra),
            "--write-manifest",
            str(manifest),
        ]
    )
    assert rc == 0

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest)])

    assert inspect_rc == 0
    text = out.getvalue()
    assert "pass --base-corpus-file to verify" in text
    assert "(unverified)" in text
    payload, _ = p1_manifest_inspect.inspect_manifest(manifest)
    assert payload["manifest_verification"] == [
        {
            "status": "skipped",
            "manifest_path": str(manifest.resolve()),
            "reason": "bundle includes base corpus; pass --base-corpus-file to verify",
        }
    ]


def test_main_require_verified_fails_closed_on_skipped_bundle(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")

    rc = p1_corpus_probe.main(
        [
            str(base),
            str(extra),
            "--write-manifest",
            str(manifest),
        ]
    )
    assert rc == 0

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest), "--require-verified"])

    assert inspect_rc == 1
    assert "pass --base-corpus-file to verify" in out.getvalue()


def test_main_fails_closed_on_drifted_bundle(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")

    rc = p1_corpus_probe.main(
        [
            str(base),
            str(extra),
            "--write-manifest",
            str(manifest),
        ]
    )
    assert rc == 0
    extra.write_text("xyz\n", encoding="utf-8")

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main(
            [str(manifest), "--base-corpus-file", str(base)]
        )

    assert inspect_rc == 1
    assert "manifest bundle contents no longer match" in out.getvalue()


def test_main_handles_manifest_without_bundle(tmp_path: Path) -> None:
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "extras.txt"
    extra.write_text("abc\n", encoding="utf-8")
    manifest.write_text("extra.txt\n", encoding="utf-8")

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest)])

    assert inspect_rc == 0
    assert "[bundle] absent; only manifest paths are available." in out.getvalue()


def test_main_require_verified_fails_closed_when_bundle_is_absent(tmp_path: Path) -> None:
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "extras.txt"
    extra.write_text("abc\n", encoding="utf-8")
    manifest.write_text("extra.txt\n", encoding="utf-8")

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest), "--require-verified"])

    assert inspect_rc == 1
    assert "no sibling bundle metadata present" in out.getvalue()


def test_main_json_marks_absent_bundle_as_unverified(tmp_path: Path) -> None:
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "extras.txt"
    report_json = tmp_path / "inspect.json"
    extra.write_text("abc\n", encoding="utf-8")
    manifest.write_text("extra.txt\n", encoding="utf-8")

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest), "--json", str(report_json)])

    assert inspect_rc == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["manifest_verification"] == [
        {
            "status": "unverified",
            "manifest_path": str(manifest.resolve()),
            "reason": "no sibling bundle",
        }
    ]
    assert payload["verification"]["status"] == "unverified"


def test_inspect_manifest_rejects_invalid_bundle_payload(tmp_path: Path) -> None:
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "extras.txt"
    extra.write_text("abc\n", encoding="utf-8")
    manifest.write_text("extra.txt\n", encoding="utf-8")
    manifest.with_suffix(".txt.bundle.json").write_text('["bad"]', encoding="utf-8")

    with pytest.raises(ValueError, match="expected JSON object"):
        p1_manifest_inspect.inspect_manifest(manifest)


def test_main_returns_rc1_for_invalid_bundle_payload(tmp_path: Path) -> None:
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "extras.txt"
    extra.write_text("abc\n", encoding="utf-8")
    manifest.write_text("extra.txt\n", encoding="utf-8")
    manifest.with_suffix(".txt.bundle.json").write_text('["bad"]', encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest)])

    assert inspect_rc == 1
    assert "expected JSON object" in out.getvalue()


def test_main_returns_rc1_for_missing_manifest_entry(tmp_path: Path) -> None:
    manifest = tmp_path / "extras.txt"
    manifest.write_text("missing.txt\n", encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest)])

    assert inspect_rc == 1
    assert "manifest entry not found" in out.getvalue()


def test_main_returns_rc1_when_skip_path_bundle_files_are_invalid(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")
    rc = p1_corpus_probe.main([str(base), str(extra), "--write-manifest", str(manifest)])
    assert rc == 0
    bundle_path = manifest.with_suffix(".txt.bundle.json")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["bundle"]["files"] = "not-a-list"
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest)])

    assert inspect_rc == 1
    assert "invalid bundle.files" in out.getvalue()


def test_main_returns_rc1_when_skip_path_bundle_file_shape_is_invalid(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")
    rc = p1_corpus_probe.main([str(base), str(extra), "--write-manifest", str(manifest)])
    assert rc == 0
    bundle_path = manifest.with_suffix(".txt.bundle.json")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["bundle"]["files"][0].pop("path")
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest)])

    assert inspect_rc == 1
    assert "invalid bundle.files[0].path" in out.getvalue()


def test_main_returns_rc1_when_skip_path_bundle_sha_missing(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")
    rc = p1_corpus_probe.main([str(base), str(extra), "--write-manifest", str(manifest)])
    assert rc == 0
    bundle_path = manifest.with_suffix(".txt.bundle.json")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["bundle"].pop("bundle_sha256")
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest)])

    assert inspect_rc == 1
    assert "invalid bundle.bundle_sha256" in out.getvalue()


def test_main_returns_rc1_when_combined_payload_is_missing_fields(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")
    rc = p1_corpus_probe.main([str(base), str(extra), "--write-manifest", str(manifest)])
    assert rc == 0
    bundle_path = manifest.with_suffix(".txt.bundle.json")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["bundle"]["combined"].pop("chars")
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest), "--base-corpus-file", str(base)])

    assert inspect_rc == 1
    assert "invalid combined.chars" in out.getvalue()


def test_main_returns_rc1_when_combined_payload_uses_bool_for_chars(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")
    rc = p1_corpus_probe.main([str(base), str(extra), "--write-manifest", str(manifest)])
    assert rc == 0
    bundle_path = manifest.with_suffix(".txt.bundle.json")
    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    payload["bundle"]["combined"]["chars"] = True
    bundle_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    out = io.StringIO()

    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest)])

    assert inspect_rc == 1
    assert "invalid combined.chars" in out.getvalue()


def test_script_runs_via_subprocess_entrypoint(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")
    rc = p1_corpus_probe.main([str(base), str(extra), "--write-manifest", str(manifest)])
    assert rc == 0
    repo_root = Path(__file__).resolve().parents[2]

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/p1_manifest_inspect.py",
            str(manifest),
            "--base-corpus-file",
            str(base),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "[bundle] generated_by=scripts/p1_corpus_probe.py includes_base=True verify=passed" in proc.stdout
    assert "[combined] chars=10 vocab=4" in proc.stdout


def test_main_reports_effective_entries_after_dedup_and_base_skip(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    report_json = tmp_path / "effective.json"
    base.write_text("abca\n", encoding="utf-8")
    extra.write_text("abcb\n", encoding="utf-8")
    manifest.write_text("extra.txt\nbase.txt\nextra.txt\n", encoding="utf-8")
    bundle_payload = {
        "generated_by": "manual-test",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": p1_manifest_inspect.sha256_text(
            manifest.read_text(encoding="utf-8")
        ),
        "bundle": build_utf8_corpus_bundle([extra, base, extra], base_file=base),
    }
    manifest.with_suffix(".txt.bundle.json").write_text(
        json.dumps(bundle_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main(
            [
                str(manifest),
                "--base-corpus-file",
                str(base),
                "--json",
                str(report_json),
            ]
        )

    assert inspect_rc == 1
    text = out.getvalue()
    assert "collapse after base/duplicate filtering" in text
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["verification"]["status"] == "failed"
    assert payload["manifest_verification"] == [
        {
            "status": "failed",
            "manifest_path": str(manifest.resolve()),
            "reason": (
                "manifest entries collapse after base/duplicate filtering; "
                f"regenerate {manifest.resolve()} without base or duplicate entries"
            ),
        }
    ]


def test_main_rejects_manifest_with_only_base_entry(tmp_path: Path) -> None:
    base = tmp_path / "base.txt"
    manifest = tmp_path / "selected.txt"
    report_json = tmp_path / "empty_effective.json"
    base.write_text("abca\n", encoding="utf-8")
    manifest.write_text("base.txt\n", encoding="utf-8")
    bundle_payload = {
        "generated_by": "manual-test",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": p1_manifest_inspect.sha256_text(
            manifest.read_text(encoding="utf-8")
        ),
        "bundle": build_utf8_corpus_bundle([base]),
    }
    manifest.with_suffix(".txt.bundle.json").write_text(
        json.dumps(bundle_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main(
            [
                str(manifest),
                "--base-corpus-file",
                str(base),
                "--json",
                str(report_json),
            ]
        )

    assert inspect_rc == 1
    text = out.getvalue()
    assert "collapse after base/duplicate filtering" in text
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["verification"]["status"] == "failed"
    assert payload["effective"] is None
    assert payload["manifest_verification"] == [
        {
            "status": "failed",
            "manifest_path": str(manifest.resolve()),
            "reason": (
                "manifest entries collapse after base/duplicate filtering; "
                f"regenerate {manifest.resolve()} without base or duplicate entries"
            ),
        }
    ]


def test_main_returns_rc1_for_empty_manifest_when_bundle_excludes_base(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.txt"
    report_json = tmp_path / "empty_manifest.json"
    manifest.write_text("# no entries\n", encoding="utf-8")
    bundle_payload = {
        "generated_by": "manual-test",
        "manifest_path": str(manifest.resolve()),
        "manifest_sha256": p1_manifest_inspect.sha256_text(
            manifest.read_text(encoding="utf-8")
        ),
        "bundle": {
            "files": [],
            "combined": {
                "chars": 0,
                "vocab_size": 0,
                "sha256": p1_manifest_inspect.sha256_text(""),
                "includes_base": False,
            },
            "bundle_sha256": hashlib.sha256(b"[]").hexdigest(),
        },
    }
    manifest.with_suffix(".txt.bundle.json").write_text(
        json.dumps(bundle_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        inspect_rc = p1_manifest_inspect.main([str(manifest), "--json", str(report_json)])

    assert inspect_rc == 1
    assert "manifest contains no corpus entries" in out.getvalue()
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["verification"]["status"] == "failed"
    assert payload["manifest_verification"] == [
        {
            "status": "failed",
            "manifest_path": str(manifest.resolve()),
            "reason": "manifest contains no corpus entries; regenerate the manifest/bundle pair",
        }
    ]
