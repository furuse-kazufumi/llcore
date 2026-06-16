# SPDX-License-Identifier: Apache-2.0
"""Probe extra-corpus candidates before running expensive P1 training.

The goal is to surface the cheap signals that matter for the current P1 loop:

- how much text each extra corpus adds under the *same* normalization used by
  ``llcore.lm train``;
- how many new characters it introduces beyond the base corpus tokenizer;
- how much of the extra text would be OOV if evaluated with the base tokenizer.

This keeps the next heavy step honest: if an extra corpus mostly contributes new
characters, held-out changes may reflect tokenizer drift more than language
modeling gains. The per-extra ``sha256`` hashes the individually normalized
candidate file; it is not a hash of that file's exact byte slice inside the
combined training corpus. The probe can also emit a filtered manifest so the
same reviewed candidate set can flow directly into ``train`` / ``eval``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llcore.lm.corpus import (
    build_utf8_corpus_bundle,
    load_utf8_corpus_files,
    read_utf8_corpus_file,
    resolve_extra_corpus_files,
    sha256_text,
)
from llcore.lm.tokenizer import CharTokenizer


def _preview_chars(chars: set[str], *, limit: int = 12) -> list[str]:
    preview: list[str] = []
    sorted_chars = sorted(chars)
    for ch in sorted_chars[:limit]:
        preview.append(ch.encode("unicode_escape").decode("ascii"))
    if len(sorted_chars) > limit:
        preview.append(f"...(+{len(sorted_chars) - limit} more)")
    return preview


def analyze_corpus_mix(base_file: Path, extra_files: list[Path]) -> dict[str, Any]:
    """Describe the base corpus and candidate extra corpora under training semantics."""
    base_text = read_utf8_corpus_file(base_file)
    base_tok = CharTokenizer.from_text(base_text)
    combined_text = load_utf8_corpus_files(base_file, extra_files)
    combined_vocab = set(combined_text)
    base_vocab = set(base_text)
    extras: list[dict[str, Any]] = []
    for path in extra_files:
        text = read_utf8_corpus_file(path)
        vocab = set(text)
        new_chars = vocab - base_vocab
        oov_chars = sum(1 for ch in text if ch not in base_tok.stoi)
        extras.append(
            {
                "path": str(path),
                "chars": len(text),
                "vocab_size": len(vocab),
                "sha256": sha256_text(text),
                "new_char_count_vs_base": len(new_chars),
                "new_char_preview_vs_base": _preview_chars(new_chars),
                "oov_chars_vs_base": oov_chars,
                "oov_rate_vs_base_raw": oov_chars / max(1, len(text)),
                "oov_rate_vs_base": round(oov_chars / max(1, len(text)), 6),
            }
        )
    added_chars = combined_vocab - base_vocab
    return {
        "base": {
            "path": str(base_file),
            "chars": len(base_text),
            "vocab_size": len(base_vocab),
            "sha256": sha256_text(base_text),
        },
        "extras": extras,
        "combined": {
            "chars": len(combined_text),
            "vocab_size": len(combined_vocab),
            "sha256": sha256_text(combined_text),
            "added_vocab_count_vs_base": len(added_chars),
            "added_vocab_preview_vs_base": _preview_chars(added_chars),
        },
    }


def _markdown_table(report: dict[str, Any]) -> str:
    rows = [
        "| corpus | chars | vocab | new chars vs base (uniq) | OOV vs base (occurs) | sha256 |",
        "|" + "---|" * 6,
    ]
    base = report["base"]
    rows.append(
        f"| base:{Path(base['path']).name} | {base['chars']} | {base['vocab_size']} | 0 | 0 | "
        f"{base['sha256'][:12]} |"
    )
    for extra in report["extras"]:
        rows.append(
            f"| extra:{Path(extra['path']).name} | {extra['chars']} | {extra['vocab_size']} | "
            f"{extra['new_char_count_vs_base']} "
            f"({', '.join(extra['new_char_preview_vs_base']) or '-'}) | "
            f"{extra['oov_chars_vs_base']} ({extra['oov_rate_vs_base']:.6f}) | "
            f"{extra['sha256'][:12]} |"
        )
    combined = report["combined"]
    rows.append(
        f"| combined | {combined['chars']} | {combined['vocab_size']} | "
        f"{combined['added_vocab_count_vs_base']} "
        f"({', '.join(combined['added_vocab_preview_vs_base']) or '-'}) | n/a | "
        f"{combined['sha256'][:12]} |"
    )
    return "\n".join(rows)


def _selected_extra_paths(
    report: dict[str, Any],
    *,
    max_oov_rate: float | None,
    max_new_chars: int | None,
) -> list[str]:
    selected: list[str] = []
    for extra in report["extras"]:
        if max_oov_rate is not None and float(extra["oov_rate_vs_base_raw"]) > max_oov_rate:
            continue
        if max_new_chars is not None and int(extra["new_char_count_vs_base"]) > max_new_chars:
            continue
        selected.append(str(extra["path"]))
    return selected


def _write_manifest(path: Path, selected_paths: list[str]) -> None:
    lines = ["# Generated by scripts/p1_corpus_probe.py"]
    for selected in selected_paths:
        try:
            rel = os.path.relpath(selected, start=path.parent)
        except ValueError:
            rel = selected
        lines.append(Path(rel).as_posix())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest_bundle_metadata(
    path: Path,
    selected_paths: list[str],
    *,
    base_file: Path,
) -> Path:
    metadata_path = path.with_suffix(path.suffix + ".bundle.json")
    payload = {
        "generated_by": "scripts/p1_corpus_probe.py",
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha256_text(path.read_text(encoding="utf-8")),
        "bundle": build_utf8_corpus_bundle(
            [Path(selected) for selected in selected_paths],
            base_file=base_file,
        ),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Probe P1 extra-corpus candidates cheaply")
    ap.add_argument("base_corpus_file", help="UTF-8 base corpus file")
    ap.add_argument("extra_corpus_files", nargs="*", help="UTF-8 corpus files to append")
    ap.add_argument(
        "--extra-corpus-manifest",
        action="append",
        default=None,
        help="UTF-8 manifest listing extra corpus files (one path per line; # only as full-line comments)",
    )
    ap.add_argument("--json", default=None, help="optional path to dump the report as JSON")
    ap.add_argument(
        "--write-manifest",
        default=None,
        help="optional path to write the selected extra corpus paths as a manifest",
    )
    ap.add_argument(
        "--max-oov-rate",
        type=float,
        default=None,
        help="optional filter: only write extras whose OOV rate vs base is <= this value",
    )
    ap.add_argument(
        "--max-new-chars",
        type=int,
        default=None,
        help="optional filter: only write extras whose new-char count vs base is <= this value",
    )
    args = ap.parse_args(argv)
    extra_files = resolve_extra_corpus_files(
        [Path(path) for path in args.extra_corpus_files],
        [Path(path) for path in args.extra_corpus_manifest or []],
        base_file=args.base_corpus_file,
    )

    report = analyze_corpus_mix(
        Path(args.base_corpus_file),
        [Path(path) for path in extra_files],
    )
    print(_markdown_table(report))
    combined = report["combined"]
    print(
        "\n[headline] combined vocab delta vs base: "
        f"{combined['added_vocab_count_vs_base']} chars; "
        f"preview={combined['added_vocab_preview_vs_base'] or ['-']}."
    )
    print(
        "[note] extra sha256 hashes each individually normalized candidate file, "
        "not its exact combined-corpus byte slice. manifest bundle combined.sha256 "
        "tracks base+selected extras in train order."
    )
    selected_paths = _selected_extra_paths(
        report,
        max_oov_rate=args.max_oov_rate,
        max_new_chars=args.max_new_chars,
    )
    print(
        f"[selection] {len(selected_paths)}/{len(report['extras'])} extras pass "
        f"(max_oov_rate={args.max_oov_rate}, max_new_chars={args.max_new_chars})."
    )
    if len(selected_paths) != len(report["extras"]):
        selected_report = analyze_corpus_mix(
            Path(args.base_corpus_file),
            [Path(path) for path in selected_paths],
        )
        if selected_paths:
            print("\n[selected subset]")
            print(
                _markdown_table(
                    {"base": report["base"], "extras": [], "combined": selected_report["combined"]}
                )
            )
    else:
        selected_report = {
            "combined": report["combined"],
        }
    if args.json:
        payload = dict(report)
        payload["combined_selected"] = selected_report["combined"]
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    if args.write_manifest:
        manifest_path = Path(args.write_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        _write_manifest(manifest_path, selected_paths)
        print(f"wrote {args.write_manifest}")
        if selected_paths:
            bundle_path = _write_manifest_bundle_metadata(
                manifest_path,
                selected_paths,
                base_file=Path(args.base_corpus_file),
            )
            print(f"wrote {bundle_path}")
        else:
            print("[selection] 0 extras selected; skipped manifest bundle metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
