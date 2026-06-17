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
- Auth/owner checks are limited to env + `kaggle.json` credential sources; if
  a Kaggle CLI setup authenticates through some other mechanism, this script
  may fail closed before push.
- The owner backstop from `kaggle kernels list -m --csv` assumes the CLI emits
  an `author` column whose content is comparable to the kernel owner slug. That
  column name/content has not been live-verified on this machine; if Kaggle
  emits a display name or different field, this check is advisory only and can
  fail to trigger as a hard owner validation.
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
from pathlib import PurePosixPath
from typing import Any, cast
import zipfile


HERE = Path(__file__).resolve().parent
PREFLIGHT_SCRIPT = HERE / "kaggle_bundle_preflight.py"
DEFAULT_RUNNER_TIMEOUT = 300
DEFAULT_KAGGLE_TIMEOUT = 20
RC_VALIDATION = 2
RC_AUTH = 3
RC_QUOTA = 4
_TRUE_STR = "true"
_KERNEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*$")
_KAGGLE_API_V1_TOKEN_ENV = "KAGGLE_API_V1_TOKEN"
_LICENSE_GUARD_EXTENSIONS = {".py", ".md", ".txt", ".json", ".toml", ".yaml", ".yml"}
_LICENSE_GUARD_ARCHIVE_NAMES = {"src_llcore.zip", "pkg_llcore.zip"}
# This marker scan intentionally covers bundled source-like text as well as
# redistribution docs. Adding these literals to src/llcore comments will make
# bundle readiness fail closed until wording is adjusted.
_COMMERCIAL_LICENSE_MARKERS = ("LICENSE-COMMERCIAL", "Commercial dual-license")
_KAGGLE_NOTEBOOK_RUN_TYPE_ENV = "KAGGLE_KERNEL_RUN_TYPE"
_PATHLIKE_TOKEN_SUFFIXES = (".txt", ".json")


class _KernelOwnerValidationError(ValueError):
    pass


class _KernelOwnerAuthError(ValueError):
    pass


class _KaggleConfigValidationError(ValueError):
    def __init__(self, message: str, *, token_advisory_ok: bool = False) -> None:
        super().__init__(message)
        self.token_advisory_ok = token_advisory_ok


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
    config_root = _kaggle_config_root()
    return config_root / "kaggle.json"


def _kaggle_config_root() -> Path:
    config_root = os.environ.get("KAGGLE_CONFIG_DIR")
    if config_root:
        return Path(config_root).expanduser()
    return Path.home() / ".kaggle"


def _kaggle_access_token_path() -> Path:
    return _kaggle_config_root() / "access_token"


def _kaggle_access_token_txt_path() -> Path:
    return _kaggle_config_root() / "access_token.txt"


def _kaggle_oauth_credentials_path() -> Path:
    return _kaggle_config_root() / "credentials.json"


def _nonempty_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value is None:
        return None
    value = value.strip()
    return value or None


def _nonempty_file(path: Path) -> bool:
    try:
        return path.is_file() and bool(path.read_text(encoding="utf-8").strip())
    except (OSError, UnicodeDecodeError):
        return False


def _looks_like_path(value: str) -> bool:
    if "/" in value or "\\" in value:
        return True
    if value.startswith(".") or value.endswith(_PATHLIKE_TOKEN_SUFFIXES):
        return True
    return bool(re.match(r"^[A-Za-z]:", value))


def _credential_sources() -> list[str]:
    sources: list[str] = []
    if _nonempty_env("KAGGLE_USERNAME") and _nonempty_env("KAGGLE_KEY"):
        sources.append("env")
    notebook_token_path = _nonempty_env(_KAGGLE_API_V1_TOKEN_ENV)
    if (
        os.environ.get(_KAGGLE_NOTEBOOK_RUN_TYPE_ENV)
        and notebook_token_path
        and _nonempty_file(Path(notebook_token_path).expanduser())
    ):
        sources.append("api_v1_token_file")
    api_token = _nonempty_env("KAGGLE_API_TOKEN")
    if api_token:
        api_token_path = Path(api_token).expanduser()
        if api_token_path.is_file():
            if _nonempty_file(api_token_path):
                sources.append("api_token_env")
        elif not _looks_like_path(api_token):
            sources.append("api_token_env")
    if _kaggle_json_path().is_file():
        sources.append("kaggle.json")
    if _nonempty_file(_kaggle_access_token_path()):
        sources.append("access_token")
    if _nonempty_file(_kaggle_access_token_txt_path()):
        sources.append("access_token")
    if _kaggle_oauth_credentials_path().is_file():
        sources.append("oauth_credentials")
    return list(dict.fromkeys(sources))


def _configured_kaggle_username() -> str | None:
    env_username = _nonempty_env("KAGGLE_USERNAME")
    env_key = _nonempty_env("KAGGLE_KEY")
    if env_username and env_key:
        return env_username
    json_path = _kaggle_json_path()
    if json_path.is_file():
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError) as exc:
            raise _KaggleConfigValidationError(f"kaggle.json present but unreadable: {exc}") from exc
        except json.JSONDecodeError as exc:
            raise _KaggleConfigValidationError(
                f"kaggle.json present but malformed JSON: {exc.msg}",
                token_advisory_ok=True,
            ) from exc
        if not isinstance(payload, dict):
            raise _KaggleConfigValidationError(
                "kaggle.json present but malformed: expected top-level object",
                token_advisory_ok=True,
            )
        username = payload.get("username")
        key = payload.get("key")
        if not isinstance(username, str) or not isinstance(key, str):
            raise _KaggleConfigValidationError(
                "kaggle.json present but malformed: expected non-empty string username and key",
                token_advisory_ok=True,
            )
        username = username.strip()
        key = key.strip()
        if not username or not key:
            raise _KaggleConfigValidationError(
                "kaggle.json present but malformed: expected non-empty string username and key",
                token_advisory_ok=True,
            )
        return username
    oauth_username = _configured_oauth_username()
    if oauth_username:
        return oauth_username
    return None


def _configured_oauth_username() -> str | None:
    credentials_path = _kaggle_oauth_credentials_path()
    if not credentials_path.is_file():
        return None
    try:
        payload = json.loads(credentials_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise _KaggleConfigValidationError(f"credentials.json present but unreadable: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise _KaggleConfigValidationError(
            f"credentials.json present but malformed JSON: {exc.msg}",
            token_advisory_ok=True,
        ) from exc
    if not isinstance(payload, dict):
        raise _KaggleConfigValidationError(
            "credentials.json present but malformed: expected top-level object",
            token_advisory_ok=True,
        )
    refresh_token = payload.get("refresh_token")
    username = payload.get("username")
    if not isinstance(refresh_token, str):
        raise _KaggleConfigValidationError(
            "credentials.json present but malformed: expected non-empty string refresh_token",
            token_advisory_ok=True,
        )
    refresh_token = refresh_token.strip()
    if not refresh_token:
        raise _KaggleConfigValidationError(
            "credentials.json present but malformed: expected non-empty string refresh_token",
            token_advisory_ok=True,
        )
    if username is None:
        return None
    if not isinstance(username, str):
        raise _KaggleConfigValidationError(
            "credentials.json present but malformed: username must be a string when provided",
            token_advisory_ok=True,
        )
    username = username.strip()
    return username or None


def _check_kernel_owner_config(kernel_id: str) -> str:
    owner, _, _slug = kernel_id.partition("/")
    if not owner:
        # `_KERNEL_ID_RE` validation at the call site makes this unreachable in
        # normal flow, but keep the guard for direct helper use.
        raise _KernelOwnerValidationError(f"kernel_id must be owner/slug, got: {kernel_id!r}")
    try:
        configured_username = _configured_kaggle_username()
    except _KaggleConfigValidationError as exc:
        raise _KernelOwnerValidationError(str(exc)) from exc
    if not configured_username:
        raise _KernelOwnerAuthError(
            "no Kaggle push credentials found; expected ~/.kaggle/kaggle.json, "
            "KAGGLE_CONFIG_DIR/kaggle.json, ~/.kaggle/credentials.json, "
            "KAGGLE_USERNAME+KAGGLE_KEY, or a token-based auth source"
        )
    # `kernel_id` owner is strict-lowercase by `_KERNEL_ID_RE`; `.lower()` only
    # normalizes the configured username loaded from env/json/oauth credentials.
    if owner.lower() != configured_username.lower():
        raise _KernelOwnerValidationError(
            f"kernel_id owner {owner!r} does not match configured Kaggle username {configured_username!r}"
        )
    return configured_username


def _extract_probe_author(csv_text: str) -> str | None:
    # Assumes the CLI emits an `author` column compatible with kernel owner
    # slugs. This is parser-covered but not live-schema-verified here.
    lines = [line for line in csv_text.splitlines() if line.strip()]
    if len(lines) < 2:
        return None
    reader = csv.DictReader(lines)
    rows = [
        {k.strip().lower(): (v.strip() if v is not None else "") for k, v in row.items() if k is not None}
        for row in reader
    ]
    if not rows:
        return None
    author = rows[0].get("author", "").strip()
    return author or None


def _check_auth(*, timeout_s: int) -> dict[str, object]:
    sources = _credential_sources()
    if not sources:
        raise ValueError(
            "no Kaggle push credentials found; expected KAGGLE_USERNAME+KAGGLE_KEY, "
            "KAGGLE_API_TOKEN, notebook KAGGLE_API_V1_TOKEN file, "
            "KAGGLE_CONFIG_DIR/credentials.json or kaggle.json, or "
            "~/.kaggle/{credentials.json,kaggle.json,access_token,access_token.txt}"
        )
    proc = _run_kaggle(["kernels", "list", "-m", "--page-size", "1", "--csv"], timeout_s=timeout_s)
    if proc.returncode != 0 or not proc.stdout.strip():
        detail = proc.stderr.strip() or proc.stdout.strip() or f"rc={proc.returncode}"
        raise ValueError(f"kaggle credential probe failed: {detail}")
    # A header-only CSV is intentional first-push compatibility: an account can
    # be authenticated yet legitimately have zero kernels listed.
    report = {
        "authenticated": True,
        "credential_sources": sources,
        "probe_command": "kaggle kernels list -m --page-size 1 --csv",
    }
    probe_author = _extract_probe_author(proc.stdout)
    if probe_author is not None:
        report["probe_author"] = probe_author
    return report


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


def _iter_bundle_text_files(bundle_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in bundle_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name == "input_corpus.txt":
            continue
        suffix = path.suffix.lower()
        if suffix not in _LICENSE_GUARD_EXTENSIONS and not path.name.startswith(("LICENSE", "NOTICE")):
            continue
        files.append(path)
    return files


def _iter_bundle_archive_files(bundle_dir: Path) -> list[Path]:
    files: list[Path] = []
    for path in bundle_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.name in _LICENSE_GUARD_ARCHIVE_NAMES:
            files.append(path)
    return files


def _iter_archive_text_members(bundle_dir: Path, archive_path: Path) -> list[tuple[str, str]]:
    try:
        with zipfile.ZipFile(archive_path) as zf:
            members: list[tuple[str, str]] = []
            for info in zf.infolist():
                if info.is_dir():
                    continue
                member = PurePosixPath(info.filename.replace("\\", "/"))
                if member.suffix.lower() not in _LICENSE_GUARD_EXTENSIONS:
                    continue
                try:
                    content = zf.read(info).decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise ValueError(
                        "bundle license guard failed: archive member is not valid UTF-8: "
                        f"{archive_path.name}:{member.as_posix()}"
                    ) from exc
                members.append((member.as_posix(), content))
            return members
    except OSError as exc:
        raise ValueError(
            f"bundle license guard failed: archive unreadable: {archive_path.relative_to(bundle_dir)}"
        ) from exc
    except zipfile.BadZipFile as exc:
        raise ValueError(f"bundle license guard failed: invalid zip archive: {archive_path.name}") from exc


def _check_bundle_license_policy(bundle_dir: Path) -> dict[str, object]:
    text_files = _iter_bundle_text_files(bundle_dir)
    archive_files = _iter_bundle_archive_files(bundle_dir)
    source_like_present = any(path.name == "runner.py" or "src" in path.parts for path in text_files) or bool(
        archive_files
    )
    if not source_like_present:
        return {
            "checked": False,
            "reason": "bundle has no source-like text payload; skipped license guard",
        }
    license_path = bundle_dir / "LICENSE"
    notice_path = bundle_dir / "NOTICE"
    if not license_path.is_file() or not notice_path.is_file():
        raise ValueError(
            "bundle license guard failed: candidate bundle must include LICENSE and NOTICE "
            "before push review"
        )
    commercial_license_path = bundle_dir / "LICENSE-COMMERCIAL"
    if commercial_license_path.exists():
        raise ValueError("bundle license guard failed: LICENSE-COMMERCIAL must not be bundled")
    findings: list[str] = []
    for path in text_files:
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValueError(
                f"bundle license guard failed: unreadable text file: {path.relative_to(bundle_dir)}"
            ) from exc
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"bundle license guard failed: non-UTF-8 text file: {path.relative_to(bundle_dir)}"
            ) from exc
        for marker in _COMMERCIAL_LICENSE_MARKERS:
            if marker in content:
                findings.append(f"{path.relative_to(bundle_dir)}: {marker}")
    for archive_path in archive_files:
        for member_name, content in _iter_archive_text_members(bundle_dir, archive_path):
            for marker in _COMMERCIAL_LICENSE_MARKERS:
                if marker in content:
                    findings.append(f"{archive_path.relative_to(bundle_dir)}:{member_name}: {marker}")
    if findings:
        joined = "; ".join(findings[:5])
        raise ValueError(
            "bundle license guard failed: commercial-license wording remains in candidate bundle "
            f"({joined})"
        )
    return {
        "checked": True,
        "license_path": str(license_path),
        "notice_path": str(notice_path),
        "commercial_markers_found": 0,
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
    if not isinstance(metadata, dict):
        print("error: preflight metadata missing", file=sys.stderr)
        return RC_VALIDATION
    try:
        license_report = _check_bundle_license_policy(bundle_dir)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return RC_VALIDATION
    kernel_id_raw = metadata.get("kernel_id")
    if not isinstance(kernel_id_raw, str) or not kernel_id_raw.strip():
        print("error: preflight metadata missing kernel_id", file=sys.stderr)
        return RC_VALIDATION
    kernel_id = kernel_id_raw.strip()
    if not _KERNEL_ID_RE.fullmatch(kernel_id):
        print(f"error: kernel_id must be owner/slug, got: {kernel_id!r}", file=sys.stderr)
        return RC_VALIDATION
    credential_sources = _credential_sources()
    token_auth_available = any(
        source in {"api_v1_token_file", "api_token_env", "access_token", "oauth_credentials"} for source in credential_sources
    )
    owner_check_status = "validated_local_config"
    try:
        configured_username = _check_kernel_owner_config(kernel_id)
        if token_auth_available:
            owner_check_status = "validated_local_config_token_present"
    except _KernelOwnerAuthError as exc:
        if token_auth_available:
            configured_username = None
            owner_check_status = "advisory_token_auth"
        else:
            print(f"error: {exc}", file=sys.stderr)
            return RC_AUTH
    except _KernelOwnerValidationError as exc:
        cause = exc.__cause__
        if (
            token_auth_available
            and isinstance(cause, _KaggleConfigValidationError)
            and cause.token_advisory_ok
        ):
            configured_username = None
            owner_check_status = "advisory_token_auth_malformed_local_config"
        else:
            print(f"error: {exc}", file=sys.stderr)
            if isinstance(cause, _KaggleConfigValidationError):
                return RC_VALIDATION
            # Keep owner mismatch fail-closed in both cases, but classify token-
            # backed setups as auth-adjacent because the live push identity may
            # come from token/OAuth state rather than the local config string.
            return RC_AUTH if token_auth_available else RC_VALIDATION
    try:
        auth_report = _check_auth(timeout_s=args.kaggle_timeout)
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return RC_AUTH
    probe_author = auth_report.get("probe_author")
    if isinstance(probe_author, str) and probe_author.strip():
        if kernel_id.partition("/")[0].lower() != probe_author.strip().lower():
            auth_report["probe_author_status"] = "advisory_owner_mismatch_unverified"
        else:
            auth_report["probe_author_status"] = "validated_against_owner"
    else:
        auth_report["probe_author_status"] = "advisory_unverified_empty_probe"
    auth_report["configured_username"] = configured_username
    auth_report["owner_check_status"] = owner_check_status
    owner_warning: str | None = None
    if isinstance(auth_report.get("probe_author_status"), str) and str(auth_report["probe_author_status"]).startswith("advisory_"):
        owner_warning = (
            "owner verification is advisory only; live probe author did not produce a verified owner match"
        )
    if isinstance(owner_check_status, str) and owner_check_status.startswith("advisory_"):
        owner_warning = (
            "owner verification is advisory only; local owner/auth configuration could not be fully validated"
        )
    if owner_warning is not None:
        print(f"warning: {owner_warning}", file=sys.stderr)
    enable_gpu = metadata.get("enable_gpu") == _TRUE_STR
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
    report = {
        "bundle_dir": str(bundle_dir),
        "kernel_id": kernel_id,
        "push_command": f'kaggle kernels push -p "{bundle_dir}"',
        "preflight": preflight_report,
        "license": license_report,
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
        f"owner={owner_check_status}",
        f"quota_rows={quota_row_count}",
        f"quota_checked={quota_report['checked_resource']}",
    )
    print("[next]", report["push_command"])
    return 0


def main(argv: list[str] | None = None) -> int:
    return check_readiness(argv)


if __name__ == "__main__":
    raise SystemExit(main())
