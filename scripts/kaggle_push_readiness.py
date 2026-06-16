# SPDX-License-Identifier: Apache-2.0
"""Run local push-readiness checks for an existing Kaggle llcore bundle.

This script does not publish anything. It combines:

1. local bundle preflight (`kaggle_bundle_preflight.py`)
2. Kaggle push-credential presence + lightweight API probe
3. Kaggle quota fetch (GPU bundles only)
4. exact human-gated `kaggle kernels push -p ...` command emission

Honest disclosure for the GPU quota path:

- GPU readiness currently depends on `kaggle quota -v`.
- On this machine's Kaggle CLI 2.2.1, live `kaggle quota` / `kaggle quota -v`
  fails before CSV parsing with `not enough values to unpack`, so the live GPU
  quota path is presently unconfirmable here.
- The CSV schema assumptions (`remaining` / `used` / `total`) and GPU row
  matching logic are therefore parser coverage only, not a confirmed live
  compatibility claim.
- Even when `kaggle quota -v` works, this script does not verify
  `machine_shape`-specific capacity. A green GPU readiness result only means
  "credential probe ok + some GPU-like quota row shows remaining capacity".
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, cast


HERE = Path(__file__).resolve().parent
PREFLIGHT_SCRIPT = HERE / "kaggle_bundle_preflight.py"
DEFAULT_RUNNER_TIMEOUT = 300
DEFAULT_KAGGLE_TIMEOUT = 20
RC_VALIDATION = 2
RC_AUTH = 3
RC_QUOTA = 4
_TRUE_STR = "true"


def _load_script(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_kaggle(args: list[str], *, timeout_s: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["kaggle", *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )


def _kaggle_json_path() -> Path:
    config_root = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_root:
        return Path(config_root).expanduser() / "kaggle.json"
    return Path.home() / ".kaggle" / "kaggle.json"


def _credential_sources() -> list[str]:
    sources: list[str] = []
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        sources.append("env")
    if _kaggle_json_path().is_file():
        sources.append("kaggle.json")
    return sources


def _check_auth(*, timeout_s: int) -> dict[str, object]:
    sources = _credential_sources()
    if not sources:
        raise ValueError(
            "no Kaggle push credentials found; expected ~/.kaggle/kaggle.json, "
            "KAGGLE_CONFIG_DIR/kaggle.json, or KAGGLE_USERNAME+KAGGLE_KEY"
        )
    proc = _run_kaggle(["kernels", "list", "-m", "--page-size", "1", "--csv"], timeout_s=timeout_s)
    if proc.returncode != 0 or not proc.stdout.strip():
        detail = proc.stderr.strip() or proc.stdout.strip() or f"rc={proc.returncode}"
        raise ValueError(f"kaggle credential probe failed: {detail}")
    return {
        "authenticated": True,
        "credential_sources": sources,
        "probe_command": "kaggle kernels list -m --page-size 1 --csv",
    }


def _parse_quota_amount(value: str | None) -> float | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    text = text.replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if match is None:
        return None
    return float(match.group(0))


def _is_gpu_quota_row(row: dict[str, str]) -> bool:
    # This is a coarse resource-family match, not a machine-shape match.
    # For example, it cannot distinguish T4 vs P100 availability.
    for key in ("resource", "name", "quota", "type"):
        value = row.get(key)
        if value is not None and "gpu" in value.strip().lower():
            return True
    return False


def _check_quota(*, timeout_s: int, enable_gpu: bool) -> dict[str, object]:
    """Fetch and parse Kaggle quota rows for GPU bundles.

    This path is intentionally fail-closed, but its live compatibility is not
    fully established on this machine. Kaggle CLI 2.2.1 currently fails local
    `kaggle quota -v` calls before returning CSV, so the parser below is only
    schema-coverage tested. Column names (`remaining`, `used`, `total`) and the
    GPU row classifier are inferred from expected CLI output, not validated
    against a working live sample here.

    The result also does not enforce `machine_shape`-specific availability.
    Passing this check means only that a GPU-like quota row reported remaining
    capacity; it does not guarantee capacity for a specific shape such as T4.
    """
    proc = _run_kaggle(["quota", "-v"], timeout_s=timeout_s)
    csv_text = proc.stdout.strip()
    if proc.returncode != 0 or not csv_text:
        detail = proc.stderr.strip() or proc.stdout.strip() or f"rc={proc.returncode}"
        raise ValueError(f"kaggle quota check failed: {detail}")
    lines = [line for line in csv_text.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("kaggle quota check returned no data rows")
    reader = csv.DictReader(lines)
    rows = [
        {k.strip().lower(): (v.strip() if v is not None else "") for k, v in row.items() if k is not None}
        for row in reader
    ]
    if not rows:
        raise ValueError("kaggle quota check returned no parseable rows")
    parsed_rows: list[dict[str, object]] = []
    for row in rows:
        remaining = _parse_quota_amount(row.get("remaining"))
        if remaining is None:
            used = _parse_quota_amount(row.get("used"))
            total = _parse_quota_amount(row.get("total"))
            if used is not None and total is not None:
                remaining = max(0.0, total - used)
        parsed_rows.append({**row, "remaining": remaining})

    relevant_rows = [
        row
        for row in parsed_rows
        if isinstance(row, dict)
        and _is_gpu_quota_row({k: str(v) for k, v in row.items() if k != "remaining"})
    ]
    if not relevant_rows:
        if enable_gpu:
            raise ValueError("kaggle quota check returned no GPU quota rows")
        raise ValueError("kaggle quota check returned no GPU-like quota rows")
    remaining_positive = any(
        isinstance(row.get("remaining"), (int, float))
        and cast(float, row.get("remaining")) > 0
        for row in relevant_rows
    )
    if not remaining_positive:
        raise ValueError("kaggle quota check reported no remaining GPU quota")

    return {
        "csv": lines,
        "rows": parsed_rows,
        "checked_resource": "gpu",
        "resource_match_mode": "substring",
    }


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Run local preflight plus Kaggle auth/quota checks for a prepared bundle. "
            "Does not push or publish anything."
        )
    )
    ap.add_argument("--bundle-dir", required=True, help="existing Kaggle bundle directory")
    ap.add_argument("--json", help="optional combined report path")
    ap.add_argument("--run-runner", action="store_true", help="run bundled runner locally before readiness checks")
    ap.add_argument(
        "--runner-timeout",
        type=int,
        default=DEFAULT_RUNNER_TIMEOUT,
        help="runner timeout seconds for local preflight (default: 300)",
    )
    ap.add_argument(
        "--kaggle-timeout",
        type=int,
        default=DEFAULT_KAGGLE_TIMEOUT,
        help="timeout seconds for kaggle auth/quota checks (default: 20)",
    )
    return ap


def check_readiness(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.runner_timeout < 1:
        print("error: --runner-timeout must be >= 1", file=sys.stderr)
        return RC_VALIDATION
    if args.kaggle_timeout < 1:
        print("error: --kaggle-timeout must be >= 1", file=sys.stderr)
        return RC_VALIDATION

    preflight_mod = _load_script(PREFLIGHT_SCRIPT, "kaggle_bundle_preflight")
    bundle_dir = Path(args.bundle_dir).resolve()
    try:
        preflight_report = preflight_mod.preflight_bundle(
            bundle_dir,
            run_runner=args.run_runner,
            runner_timeout=args.runner_timeout,
        )
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return RC_VALIDATION
    checks = preflight_report.get("checks")
    metadata = checks.get("metadata") if isinstance(checks, dict) else None
    enable_tpu = isinstance(metadata, dict) and metadata.get("enable_tpu") == _TRUE_STR
    if enable_tpu:
        print(
            "error: TPU-enabled bundles are not supported by kaggle_push_readiness quota checks yet",
            file=sys.stderr,
        )
        return RC_VALIDATION
    try:
        auth_report = _check_auth(timeout_s=args.kaggle_timeout)
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return RC_AUTH
    enable_gpu = isinstance(metadata, dict) and metadata.get("enable_gpu") == _TRUE_STR
    if enable_gpu:
        try:
            quota_report = _check_quota(timeout_s=args.kaggle_timeout, enable_gpu=True)
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return RC_QUOTA
    else:
        quota_report = {
            "checked_resource": "cpu",
            "skipped": True,
            "reason": "cpu bundle does not require accelerator quota",
        }

    metadata = preflight_report["checks"]["metadata"]
    kernel_id = metadata["kernel_id"]
    report = {
        "bundle_dir": str(bundle_dir),
        "kernel_id": kernel_id,
        "push_command": f'kaggle kernels push -p "{bundle_dir}"',
        "preflight": preflight_report,
        "auth": auth_report,
        "quota": quota_report,
    }
    quota_rows = quota_report.get("rows")
    quota_row_count = len(quota_rows) if isinstance(quota_rows, list) else 0
    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(
        "[kaggle-ready]",
        f"dir={bundle_dir}",
        f"kernel_id={kernel_id}",
        f"runner={'yes' if args.run_runner else 'no'}",
        "auth=yes",
        f"quota_rows={quota_row_count}",
        f"quota_checked={quota_report['checked_resource']}",
    )
    print("[next]", report["push_command"])
    return 0


def main(argv: list[str] | None = None) -> int:
    return check_readiness(argv)


if __name__ == "__main__":
    raise SystemExit(main())
