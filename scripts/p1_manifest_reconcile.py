# SPDX-License-Identifier: Apache-2.0
"""Compare pre-run manifest inspection output against runtime provenance artifacts.

This keeps the final audit step mechanical: given one or more saved JSON
reports from ``scripts/p1_manifest_inspect.py --json`` plus an explicit runtime
artifact (``verdict.json`` or ``train_state.pt``), verify that both sides
describe the same ``manifest_verification`` payload.

The content-bearing contract intentionally ignores absolute-path differences in
``manifest_path``. Those paths remain visible in the report for operator
diagnosis, but matching is driven by the same content-derived fields used by the
runtime provenance path.

Each inspect JSON is expected to describe exactly one manifest bundle and
therefore one ``manifest_verification`` entry. When multiple entries are
present, this CLI compares them positionally after concatenating the
inspect-side JSON files in argv order. Inspect-side and runtime-side lists must
therefore contain the same manifests in the same order. This is content-based
reconciliation: order swaps are only observable when the swapped entries differ
in ``_COMPARABLE_FIELDS``. Unknown provenance fields are ignored unless they
are added to ``_COMPARABLE_FIELDS``.

Entry validation is fail-closed. Every entry must carry a valid ``status`` and
``manifest_path``. ``verified`` entries must also carry the content-bearing
fields needed for reconciliation (`entry_count`, `generated_by`,
`includes_base`, `combined_sha256`, `bundle_sha256`). Degraded entries
(`unverified`, `skipped`, `failed`) must carry a human-readable ``reason``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch

_COMPARABLE_FIELDS = (
    "status",
    "entry_count",
    "generated_by",
    "includes_base",
    "combined_sha256",
    "bundle_sha256",
    "reason",
)
_VALID_STATUSES = {"verified", "unverified", "skipped", "failed"}


def _require_string(
    entry: dict[str, Any],
    field: str,
    *,
    path: Path,
    idx: int,
) -> str:
    value = entry.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"{path!r} manifest_verification[{idx}].{field} must be a non-empty string"
        )
    return value


def _require_int(
    entry: dict[str, Any],
    field: str,
    *,
    path: Path,
    idx: int,
) -> int:
    value = entry.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{path!r} manifest_verification[{idx}].{field} must be an int")
    return value


def _require_bool(
    entry: dict[str, Any],
    field: str,
    *,
    path: Path,
    idx: int,
) -> bool:
    value = entry.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{path!r} manifest_verification[{idx}].{field} must be a bool")
    return value


def _validate_manifest_entry(entry: dict[str, Any], *, path: Path, idx: int) -> dict[str, Any]:
    status = _require_string(entry, "status", path=path, idx=idx)
    if status not in _VALID_STATUSES:
        raise ValueError(
            f"{path!r} manifest_verification[{idx}].status must be one of "
            f"{sorted(_VALID_STATUSES)!r}"
        )
    _require_string(entry, "manifest_path", path=path, idx=idx)
    if status == "verified":
        _require_int(entry, "entry_count", path=path, idx=idx)
        _require_string(entry, "generated_by", path=path, idx=idx)
        _require_bool(entry, "includes_base", path=path, idx=idx)
        _require_string(entry, "combined_sha256", path=path, idx=idx)
        _require_string(entry, "bundle_sha256", path=path, idx=idx)
    else:
        _require_string(entry, "reason", path=path, idx=idx)
    return entry


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path!r} must contain a top-level JSON object")
    return payload


def _load_manifest_verification(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        payload = _load_json_object(path)
        entries = payload.get("manifest_verification")
        if not isinstance(entries, list):
            raise ValueError(f"{path!r} is missing manifest_verification[]")
        normalized: list[dict[str, Any]] = []
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                raise ValueError(f"{path!r} manifest_verification[{idx}] must be an object")
            normalized.append(_validate_manifest_entry(item, path=path, idx=idx))
        return normalized
    if suffix == ".pt":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:  # noqa: BLE001 - normalize torch/zip/pickle failures
            raise ValueError(
                f"failed to read checkpoint manifest_verification from {path!r}: {exc}"
            ) from exc
        if not isinstance(payload, dict):
            raise ValueError(f"{path!r} must contain a dict checkpoint payload")
        train_meta = payload.get("train_meta")
        if not isinstance(train_meta, dict):
            raise ValueError(f"{path!r} is missing train_meta")
        entries = train_meta.get("manifest_verification")
        if not isinstance(entries, list):
            raise ValueError(f"{path!r} train_meta is missing manifest_verification[]")
        normalized = []
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                raise ValueError(f"{path!r} manifest_verification[{idx}] must be an object")
            normalized.append(_validate_manifest_entry(item, path=path, idx=idx))
        return normalized
    raise ValueError(f"unsupported runtime artifact {path!r}; expected .json or .pt")


def _format_entry(entry: dict[str, Any]) -> str:
    status = entry.get("status", "?")
    manifest_path = str(entry.get("manifest_path", "?"))
    generated_by = entry.get("generated_by", "-")
    combined_sha = entry.get("combined_sha256")
    bundle_sha = entry.get("bundle_sha256")
    reason = entry.get("reason")
    parts = [
        f"status={status}",
        f"manifest={manifest_path}",
        f"generated_by={generated_by}",
    ]
    if combined_sha is not None:
        parts.append(f"combined_sha256={str(combined_sha)[:12]}")
    if bundle_sha is not None:
        parts.append(f"bundle_sha256={str(bundle_sha)[:12]}")
    if reason:
        parts.append(f"reason={reason}")
    return " ".join(parts)


def _comparable(entry: dict[str, Any]) -> dict[str, Any]:
    return {field: entry.get(field) for field in _COMPARABLE_FIELDS if field in entry}


def _diff_fields(left: dict[str, Any], right: dict[str, Any]) -> list[str]:
    diffs: list[str] = []
    for field in _COMPARABLE_FIELDS:
        left_has = field in left
        right_has = field in right
        if left_has != right_has:
            diffs.append(
                f"{field}: inspect={'<missing>' if not left_has else left.get(field)!r} "
                f"runtime={'<missing>' if not right_has else right.get(field)!r}"
            )
            continue
        if left_has and left.get(field) != right.get(field):
            diffs.append(f"{field}: inspect={left.get(field)!r} runtime={right.get(field)!r}")
    return diffs


def reconcile_manifest_verification(
    inspect_jsons: list[Path],
    runtime_artifact: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    inspect_entries: list[dict[str, Any]] = []
    for inspect_json in inspect_jsons:
        inspect_entries.extend(_load_manifest_verification(inspect_json))
    runtime_entries = _load_manifest_verification(runtime_artifact)
    return inspect_entries, runtime_entries


def _build_report(
    inspect_paths: list[Path],
    runtime_path: Path,
    inspect_entries: list[dict[str, Any]],
    runtime_entries: list[dict[str, Any]],
) -> dict[str, Any]:
    inspect_comparable = [_comparable(entry) for entry in inspect_entries]
    runtime_comparable = [_comparable(entry) for entry in runtime_entries]
    mismatches: list[dict[str, Any]] = []
    max_len = max(len(inspect_entries), len(runtime_entries))
    for idx in range(max_len):
        inspect_entry = inspect_entries[idx] if idx < len(inspect_entries) else None
        runtime_entry = runtime_entries[idx] if idx < len(runtime_entries) else None
        diffs: list[str] = []
        if inspect_entry is None:
            diffs.append("entry missing on inspect side")
        elif runtime_entry is None:
            diffs.append("entry missing on runtime side")
        else:
            diffs.extend(_diff_fields(inspect_entry, runtime_entry))
        if diffs:
            mismatches.append(
                {
                    "entry_index": idx,
                    "inspect": inspect_entry,
                    "runtime": runtime_entry,
                    "diffs": diffs,
                }
            )
    return {
        "inspect_jsons": [str(path.resolve()) for path in inspect_paths],
        "runtime_artifact": str(runtime_path.resolve()),
        "status": "matched" if inspect_comparable == runtime_comparable else "mismatch",
        "comparison_mode": "positional",
        "inspect_entry_count": len(inspect_entries),
        "runtime_entry_count": len(runtime_entries),
        "inspect_entries": inspect_entries,
        "runtime_entries": runtime_entries,
        "mismatches": mismatches,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Compare one or more inspect JSON files against runtime provenance. "
            "Inspect entries are concatenated in argv order and compared "
            "positionally against the runtime manifest_verification list."
        )
    )
    ap.add_argument(
        "inspect_jsons",
        nargs="+",
        help=(
            "one or more inspect JSON files emitted by "
            "p1_manifest_inspect.py --json; argv order must match train's "
            "--extra-corpus-manifest order"
        ),
    )
    ap.add_argument(
        "--runtime",
        required=True,
        help="runtime artifact with manifest_verification (verdict.json or trusted self-generated train_state.pt)",
    )
    ap.add_argument("--json", default=None, help="optional path to dump reconcile report")
    args = ap.parse_args(argv)

    inspect_paths = [Path(path) for path in args.inspect_jsons]
    runtime_path = Path(args.runtime)
    try:
        inspect_entries, runtime_entries = reconcile_manifest_verification(
            inspect_paths, runtime_path
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[reconcile] {exc}")
        return 1

    report = _build_report(inspect_paths, runtime_path, inspect_entries, runtime_entries)
    if args.json:
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {json_path}")

    inspect_comparable = [_comparable(entry) for entry in inspect_entries]
    runtime_comparable = [_comparable(entry) for entry in runtime_entries]
    if inspect_comparable != runtime_comparable:
        print("[reconcile] manifest_verification mismatch")
        if len(inspect_entries) != len(runtime_entries):
            print(
                "[reconcile] inspect/runtime entry counts differ; "
                "this CLI compares manifest_verification entries positionally"
            )
        print(f"[inspect] entries={len(inspect_entries)}")
        for idx, entry in enumerate(inspect_entries):
            print(f"[inspect] {_format_entry(entry)}")
            if idx < len(runtime_entries):
                for diff in _diff_fields(entry, runtime_entries[idx]):
                    print(f"[diff] entry[{idx}] {diff}")
        print(f"[runtime] entries={len(runtime_entries)}")
        for entry in runtime_entries:
            print(f"[runtime] {_format_entry(entry)}")
        return 1

    print(
        "[reconcile] manifest_verification matched "
        f"entries={len(inspect_entries)} source={runtime_path.name}"
    )
    for entry in inspect_entries:
        print(f"[entry] {_format_entry(entry)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
