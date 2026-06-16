# SPDX-License-Identifier: Apache-2.0
"""Inspect one extra-corpus manifest and its sibling bundle metadata.

This is a cheap operator-facing check before expensive P1 train/eval runs:

- resolve the manifest exactly as ``llcore.lm`` would consume it;
- read ``<manifest>.bundle.json`` when present;
- optionally verify the bundle against the current files/base corpus;
- print a compact summary that makes ordering and combined-hash contracts visible.

This script inspects one manifest in isolation. If a later ``train``/``eval``
run also supplies ``--extra-corpus-file`` or multiple manifests, cross-source
deduplication in ``resolve_extra_corpus_files()`` can reduce the final consumed
set further than this script's per-manifest ``effective_entries`` view.

By default, ``skipped`` inspections (for example, ``includes_base=true`` with
no ``--base-corpus-file``) still exit 0 so operators can do cheap pre-checks
without forcing full verification. Use ``--require-verified`` when wiring this
tool into CI or other automation that should fail closed on anything except a
fully verified manifest/bundle pair.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llcore.lm.corpus import (
    _verified_manifest_summary,
    read_corpus_manifest,
    sha256_text,
    verify_corpus_manifest_bundle,
)


def _bundle_metadata_path(manifest_path: Path) -> Path:
    return manifest_path.with_suffix(manifest_path.suffix + ".bundle.json")


def _validated_display_bundle(bundle_path: Path, bundle: dict[str, Any]) -> dict[str, Any]:
    files = bundle.get("files")
    if not isinstance(files, list):
        raise ValueError(f"invalid bundle.files in {bundle_path!r}: expected list")
    normalized_files: list[dict[str, Any]] = []
    for idx, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError(
                f"invalid bundle.files[{idx}] in {bundle_path!r}: expected object"
            )
        path = item.get("path")
        if not isinstance(path, str) or not path:
            raise ValueError(f"invalid bundle.files[{idx}].path in {bundle_path!r}: expected string")
        chars = item.get("chars")
        if not isinstance(chars, int) or isinstance(chars, bool):
            raise ValueError(f"invalid bundle.files[{idx}].chars in {bundle_path!r}: expected int")
        vocab_size = item.get("vocab_size")
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise ValueError(
                f"invalid bundle.files[{idx}].vocab_size in {bundle_path!r}: expected int"
            )
        sha256 = item.get("sha256")
        if not isinstance(sha256, str) or not sha256:
            raise ValueError(
                f"invalid bundle.files[{idx}].sha256 in {bundle_path!r}: expected string"
            )
        normalized_files.append(item)
    bundle_sha256 = bundle.get("bundle_sha256")
    if not isinstance(bundle_sha256, str) or not bundle_sha256:
        raise ValueError(f"invalid bundle.bundle_sha256 in {bundle_path!r}: expected string")
    combined = bundle.get("combined")
    if not isinstance(combined, dict):
        raise ValueError(f"invalid combined bundle payload in {bundle_path!r}: expected object")
    chars = combined.get("chars")
    if not isinstance(chars, int) or isinstance(chars, bool):
        raise ValueError(f"invalid combined.chars in {bundle_path!r}: expected int")
    vocab_size = combined.get("vocab_size")
    if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
        raise ValueError(f"invalid combined.vocab_size in {bundle_path!r}: expected int")
    sha256 = combined.get("sha256")
    if not isinstance(sha256, str) or not sha256:
        raise ValueError(f"invalid combined.sha256 in {bundle_path!r}: expected string")
    return {
        "files": normalized_files,
        "bundle_sha256": bundle_sha256,
        "combined": {
            "chars": chars,
            "vocab_size": vocab_size,
            "sha256": sha256,
            "includes_base": combined.get("includes_base"),
        },
    }


def _effective_entries(entries: list[Path], *, base_corpus_file: Path | None) -> list[Path]:
    base_resolved = str(base_corpus_file.resolve()) if base_corpus_file is not None else None
    seen: set[str] = set()
    effective: list[Path] = []
    for entry in entries:
        resolved = str(entry.resolve())
        if resolved == base_resolved or resolved in seen:
            continue
        seen.add(resolved)
        effective.append(Path(resolved))
    return effective


def inspect_manifest(
    manifest_path: Path,
    *,
    base_corpus_file: Path | None = None,
) -> tuple[dict[str, Any], int]:
    entries = [Path(path) for path in read_corpus_manifest(manifest_path)]
    effective_entries = _effective_entries(entries, base_corpus_file=base_corpus_file)
    report: dict[str, Any] = {
        "manifest": {
            "path": str(manifest_path.resolve()),
            "entries": [str(path.resolve()) for path in entries],
            "entry_count": len(entries),
            "effective_entries": [str(path.resolve()) for path in effective_entries],
            "effective_entry_count": len(effective_entries),
            "sha256": sha256_text(manifest_path.read_text(encoding="utf-8")),
        },
        "bundle": None,
        "effective": None,
        "manifest_verification": [
            {
                "status": "unverified",
                "manifest_path": str(manifest_path.resolve()),
                "reason": "no sibling bundle",
            }
        ],
        "verification": {
            "status": "unverified",
            "message": "no sibling bundle metadata present",
        },
    }
    bundle_path = _bundle_metadata_path(manifest_path)
    if not bundle_path.exists():
        return report, 0

    payload = json.loads(bundle_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid bundle metadata in {bundle_path!r}: expected JSON object")
    bundle = payload.get("bundle")
    if not isinstance(bundle, dict):
        raise ValueError(f"invalid bundle payload in {bundle_path!r}: expected object")
    combined = bundle.get("combined")
    if not isinstance(combined, dict):
        raise ValueError(f"invalid combined bundle payload in {bundle_path!r}: expected object")
    includes_base = combined.get("includes_base")
    if not isinstance(includes_base, bool):
        raise ValueError(
            f"invalid combined.includes_base in {bundle_path!r}: expected boolean"
        )
    display_bundle = _validated_display_bundle(bundle_path, bundle)
    report["bundle"] = {
        "path": str(bundle_path.resolve()),
        "generated_by": payload.get("generated_by"),
        "manifest_sha256": payload.get("manifest_sha256"),
        "combined": display_bundle["combined"],
        "files": display_bundle["files"],
        "bundle_sha256": display_bundle["bundle_sha256"],
    }
    if includes_base and base_corpus_file is None:
        report["manifest_verification"] = [
            {
                "status": "skipped",
                "manifest_path": str(manifest_path.resolve()),
                "reason": "bundle includes base corpus; pass --base-corpus-file to verify",
            }
        ]
        report["verification"] = {
            "status": "skipped",
            "message": "bundle includes base corpus; pass --base-corpus-file to verify",
        }
        return report, 0
    if len(effective_entries) != len(entries):
        message = (
            "manifest entries collapse after base/duplicate filtering; "
            f"regenerate {manifest_path.resolve()} without base or duplicate entries"
        )
        report["manifest_verification"] = [
            {
                "status": "failed",
                "manifest_path": str(manifest_path.resolve()),
                "reason": message,
            }
        ]
        report["verification"] = {
            "status": "failed",
            "message": message,
        }
        return report, 1
    if not entries and not includes_base:
        message = "manifest contains no corpus entries; regenerate the manifest/bundle pair"
        report["manifest_verification"] = [
            {
                "status": "failed",
                "manifest_path": str(manifest_path.resolve()),
                "reason": message,
            }
        ]
        report["verification"] = {
            "status": "failed",
            "message": message,
        }
        return report, 1
    try:
        verify_corpus_manifest_bundle(
            manifest_path,
            entries,
            base_file=base_corpus_file,
        )
    except ValueError as exc:
        report["manifest_verification"] = [
            {
                "status": "failed",
                "manifest_path": str(manifest_path.resolve()),
                "reason": str(exc),
            }
        ]
        report["verification"] = {
            "status": "failed",
            "message": str(exc),
        }
        return report, 1
    report["verification"] = {
        "status": "passed",
        "message": "bundle metadata matches current manifest/files",
    }
    effective_summary = _verified_manifest_summary(
        manifest_path,
        generated_by=payload.get("generated_by"),
        includes_base=includes_base,
        effective_entries=effective_entries,
        base_file=base_corpus_file,
    )
    report["effective"] = {
        "entries": [str(path.resolve()) for path in effective_entries],
        "entry_count": len(effective_entries),
        "includes_base": includes_base,
        "combined_sha256": effective_summary["combined_sha256"],
        "bundle_sha256": effective_summary["bundle_sha256"],
    }
    report["manifest_verification"] = [effective_summary]
    return report, 0


def _markdown_table(bundle: dict[str, Any] | None, *, unverified: bool = False) -> str:
    rows = [
        "| file | chars | vocab | sha256 |",
        "|---|---|---|---|",
    ]
    if bundle is None:
        rows.append("| - | - | - | - |")
        return "\n".join(rows)
    files = bundle.get("files")
    if not isinstance(files, list) or not files:
        rows.append("| - | - | - | - |")
        return "\n".join(rows)
    for item in files:
        if not isinstance(item, dict):
            continue
        suffix = " (unverified)" if unverified else ""
        rows.append(
            f"| {Path(str(item.get('path', '?'))).name}{suffix} | "
            f"{item.get('chars', '?')} | "
            f"{item.get('vocab_size', '?')} | "
            f"{str(item.get('sha256', '?'))[:12]} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Inspect one manifest/bundle pair")
    ap.add_argument("manifest", help="extra-corpus manifest to inspect")
    ap.add_argument(
        "--base-corpus-file",
        default=None,
        help="optional base corpus file for verifying bundles with includes_base=true; required for full verification in strict mode",
    )
    ap.add_argument(
        "--require-verified",
        action="store_true",
        help="fail closed unless verification.status is passed",
    )
    ap.add_argument("--json", default=None, help="optional path to dump the inspection report")
    args = ap.parse_args(argv)

    try:
        report, rc = inspect_manifest(
            Path(args.manifest),
            base_corpus_file=Path(args.base_corpus_file) if args.base_corpus_file else None,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[verify] {exc}")
        return 1
    manifest = report["manifest"]
    bundle = report["bundle"]
    verification = report["verification"]
    print(
        f"[manifest] {Path(str(manifest['path'])).name}: "
        f"entries={manifest['entry_count']} "
        f"effective_entries={manifest['effective_entry_count']} "
        f"sha256={str(manifest['sha256'])[:12]}"
    )
    if bundle is None:
        print("[bundle] absent; only manifest paths are available.")
    else:
        combined = bundle["combined"]
        unverified = verification["status"] in {"skipped", "failed"}
        print(
            f"[bundle] generated_by={bundle['generated_by']} "
            f"includes_base={combined['includes_base']} "
            f"verify={verification['status']}"
        )
        print(_markdown_table(bundle, unverified=unverified))
        print(
            "[combined] "
            f"chars={combined['chars']} vocab={combined['vocab_size']} "
            f"sha256={str(combined['sha256'])[:12]}"
            f"{' (unverified)' if unverified else ''}"
        )
        effective = report["effective"]
        if isinstance(effective, dict):
            print(
                "[effective] "
                f"entries={effective['entry_count']} "
                f"combined_sha256={str(effective['combined_sha256'])[:12] if effective['combined_sha256'] else '-'} "
                f"bundle_sha256={str(effective['bundle_sha256'])[:12]}"
            )
    print(f"[verify] {verification['message']}")
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {json_path}")
    if args.require_verified and verification["status"] != "passed":
        return 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
