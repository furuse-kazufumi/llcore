# SPDX-License-Identifier: Apache-2.0
"""Pure helpers for corpus normalization, manifests, and UTF-8 file loading.

These functions are intentionally torch-free so lightweight tooling can share
the exact same text normalization semantics as the training CLI.
"""
from __future__ import annotations

from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


def normalize_corpus_text(text: str) -> str:
    """Normalize one corpus part to the training contract: exactly one trailing newline."""
    return text.rstrip("\n") + "\n"


def join_corpus_parts(parts: Sequence[str]) -> str:
    """Join normalized or raw corpus parts with a single blank-free boundary."""
    if not parts:
        raise ValueError("expected at least one corpus part")
    return "\n".join(part.rstrip("\n") for part in parts) + "\n"


def read_utf8_corpus_file(path: str | Path) -> str:
    """Read one UTF-8 corpus file and normalize it as a standalone part."""
    return normalize_corpus_text(Path(path).read_text(encoding="utf-8"))


def read_corpus_manifest(path: str | Path) -> list[str]:
    """Read a UTF-8 manifest of corpus paths.

    Blank lines and full-line ``#`` comments are ignored. Inline comments are not
    supported: ``path.txt  # note`` is treated as a literal path and therefore
    fails closed if that file does not exist.
    """
    manifest_path = Path(path)
    entries: list[str] = []
    for lineno, raw_line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        resolved = (manifest_path.parent / line).resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"{manifest_path}:{lineno}: manifest entry not found: {raw_line!r}"
            )
        entries.append(str(resolved))
    return entries


def _bundle_metadata_path(manifest_path: str | Path) -> Path:
    path = Path(manifest_path)
    return path.with_suffix(path.suffix + ".bundle.json")


def verify_corpus_manifest_bundle(
    manifest_path: str | Path,
    entries: Sequence[str | Path],
    *,
    base_file: str | Path | None = None,
) -> dict[str, Any] | None:
    """Verify sibling bundle metadata when present; ignore when absent.

    Returns a compact verified summary when sibling bundle metadata exists,
    otherwise ``None``.
    """
    manifest = Path(manifest_path)
    metadata_path = _bundle_metadata_path(manifest)
    if not metadata_path.exists():
        return None
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid bundle metadata in {metadata_path!r}: expected JSON object")
    if payload.get("manifest_sha256") != sha256_text(manifest.read_text(encoding="utf-8")):
        raise ValueError(f"manifest bundle metadata drift detected for {manifest!r}")
    bundle = payload.get("bundle")
    if not isinstance(bundle, dict):
        raise ValueError(f"invalid bundle payload in {metadata_path!r}: expected object")
    combined = bundle.get("combined")
    if not isinstance(combined, dict):
        raise ValueError(f"invalid combined bundle payload in {metadata_path!r}: expected object")
    includes_base = combined.get("includes_base")
    if not isinstance(includes_base, bool):
        raise ValueError(
            f"invalid combined.includes_base in {metadata_path!r}: expected boolean"
        )
    if includes_base and base_file is None:
        raise ValueError(
            f"bundle metadata for {manifest!r} requires a base corpus, but none was provided"
        )
    expected = build_utf8_corpus_bundle(
        entries,
        base_file=base_file if includes_base else None,
    )
    if _bundle_verification_view(bundle) != _bundle_verification_view(expected):
        raise ValueError(
            f"manifest bundle contents no longer match {manifest!r}; "
            "regenerate the manifest/bundle pair"
        )
    return {
        "manifest_path": str(manifest.resolve()),
        "metadata_path": str(metadata_path.resolve()),
        "entry_count": len(entries),
        "generated_by": payload.get("generated_by"),
        "includes_base": includes_base,
    }


def _verified_manifest_summary(
    manifest_path: str | Path,
    *,
    generated_by: Any,
    includes_base: bool,
    effective_entries: Sequence[str | Path],
    base_file: str | Path | None,
) -> dict[str, Any]:
    if effective_entries or includes_base:
        bundle = build_utf8_corpus_bundle(
            effective_entries,
            base_file=base_file if includes_base else None,
        )
        combined_sha256: str | None = str(bundle["combined"]["sha256"])
        bundle_sha256: str = str(bundle["bundle_sha256"])
    else:
        combined_sha256 = None
        bundle_sha256 = hashlib.sha256(b"[]").hexdigest()
    return {
        "status": "verified",
        "manifest_path": str(Path(manifest_path).resolve()),
        "entry_count": len(effective_entries),
        "generated_by": generated_by,
        "includes_base": includes_base,
        "combined_sha256": combined_sha256,
        "bundle_sha256": bundle_sha256,
    }


def _bundle_verification_view(bundle: dict[str, Any]) -> dict[str, Any]:
    files = bundle.get("files")
    if not isinstance(files, list):
        raise ValueError("bundle.files must be a list")
    comparable_files: list[dict[str, Any]] = []
    for idx, item in enumerate(files):
        if not isinstance(item, dict):
            raise ValueError("bundle.files entries must be objects")
        chars = item.get("chars")
        if not isinstance(chars, int) or isinstance(chars, bool):
            raise ValueError(f"bundle.files[{idx}].chars must be an int")
        vocab_size = item.get("vocab_size")
        if not isinstance(vocab_size, int) or isinstance(vocab_size, bool):
            raise ValueError(f"bundle.files[{idx}].vocab_size must be an int")
        comparable_files.append(
            {
                "chars": chars,
                "vocab_size": vocab_size,
                "sha256": item.get("sha256"),
            }
        )
    combined = bundle.get("combined")
    if not isinstance(combined, dict):
        raise ValueError("bundle.combined must be an object")
    combined_chars = combined.get("chars")
    if not isinstance(combined_chars, int) or isinstance(combined_chars, bool):
        raise ValueError("bundle.combined.chars must be an int")
    combined_vocab_size = combined.get("vocab_size")
    if not isinstance(combined_vocab_size, int) or isinstance(combined_vocab_size, bool):
        raise ValueError("bundle.combined.vocab_size must be an int")
    return {
        "files": comparable_files,
        "combined": {
            "chars": combined_chars,
            "vocab_size": combined_vocab_size,
            "sha256": combined.get("sha256"),
            "includes_base": combined.get("includes_base"),
        },
        "bundle_sha256": bundle.get("bundle_sha256"),
    }


def resolve_extra_corpus_files(
    extra_files: Sequence[str | Path] | None = None,
    manifest_files: Sequence[str | Path] | None = None,
    *,
    base_file: str | Path | None = None,
    verified_bundle_summaries: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Resolve explicit extra files plus manifest entries into one ordered file list."""
    base_resolved = str(Path(base_file).resolve()) if base_file is not None else None
    resolved: list[str] = []
    seen: set[str] = set()
    for path in extra_files or []:
        resolved_path = str(Path(path).resolve())
        if resolved_path == base_resolved or resolved_path in seen:
            continue
        resolved.append(resolved_path)
        seen.add(resolved_path)
    for manifest in manifest_files or []:
        entries = read_corpus_manifest(manifest)
        summary = verify_corpus_manifest_bundle(manifest, entries, base_file=base_file)
        effective_entries: list[str] = []
        for entry in entries:
            if entry == base_resolved or entry in seen:
                continue
            effective_entries.append(entry)
            seen.add(entry)
        if len(effective_entries) != len(entries):
            raise ValueError(
                "manifest entries collapse after base/duplicate filtering; "
                f"regenerate {Path(manifest).resolve()} and remove base/duplicate entries "
                "from the manifest or overlapping CLI extras"
            )
        resolved.extend(effective_entries)
        if verified_bundle_summaries is None:
            continue
        if summary is None:
            verified_bundle_summaries.append(
                {
                    "status": "unverified",
                    "manifest_path": str(Path(manifest).resolve()),
                    "reason": "no sibling bundle",
                }
            )
            continue
        verified_bundle_summaries.append(
            _verified_manifest_summary(
                manifest,
                generated_by=summary["generated_by"],
                includes_base=bool(summary["includes_base"]),
                effective_entries=effective_entries,
                base_file=base_file,
            )
        )
    return resolved


def load_utf8_corpus_files(
    base_file: str | Path,
    extra_files: Sequence[str | Path] | None = None,
) -> str:
    """Read and join a base UTF-8 corpus file plus optional extras."""
    parts = [read_utf8_corpus_file(base_file)]
    for path in extra_files or []:
        parts.append(read_utf8_corpus_file(path))
    return join_corpus_parts(parts)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def describe_utf8_corpus_files(paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Describe normalized UTF-8 corpus files in stable, ordered form."""
    descriptions: list[dict[str, Any]] = []
    for path in paths:
        resolved = Path(path).resolve()
        text = read_utf8_corpus_file(resolved)
        descriptions.append(
            {
                "path": str(resolved),
                "text": text,
                "chars": len(text),
                "vocab_size": len(set(text)),
                "sha256": sha256_text(text),
            }
        )
    return descriptions


def build_utf8_corpus_bundle(
    paths: Sequence[str | Path],
    *,
    base_file: str | Path | None = None,
) -> dict[str, Any]:
    """Summarize an ordered corpus bundle for reproducible manifests/reports."""
    files = describe_utf8_corpus_files(paths)
    combined_parts: list[str] = []
    if base_file is not None:
        combined_parts.append(read_utf8_corpus_file(base_file))
    combined_parts.extend(str(item["text"]) for item in files)
    combined = join_corpus_parts(combined_parts)
    bundle_fingerprint = hashlib.sha256(
        json.dumps(
            [item["sha256"] for item in files],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    stripped_files = [
        {
            "path": str(item["path"]),
            "chars": int(item["chars"]),
            "vocab_size": int(item["vocab_size"]),
            "sha256": str(item["sha256"]),
        }
        for item in files
    ]
    return {
        "files": stripped_files,
        "combined": {
            "chars": len(combined),
            "vocab_size": len(set(combined)),
            "sha256": sha256_text(combined),
            "includes_base": base_file is not None,
        },
        "bundle_sha256": bundle_fingerprint,
    }
