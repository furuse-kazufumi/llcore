# SPDX-License-Identifier: Apache-2.0
"""Run local preflight checks for a Kaggle llcore.lm.compare bundle.

This script never contacts Kaggle. It only validates local bundle structure and
optionally runs the bundled `runner.py` to confirm the package executes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from pathlib import PurePosixPath
import zipfile

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "src")))

from llcore.lm_compare_config import CompareConfig


REQUIRED_FILES = (
    "kernel-metadata.json",
    "bundle_manifest.json",
    "LICENSE",
    "NOTICE",
    "runner.py",
    "README.md",
)
EMBEDDED_COPIED_FILE_KEYS = (
    "corpus",
    "config",
    "metadata",
    "src_llcore",
    "pkg_llcore",
    "license",
    "notice",
)
DATASET_COPIED_FILE_KEYS = (
    "metadata",
    "runner",
    "license",
    "notice",
    "dataset_payload",
)
DATASET_PAYLOAD_DIRNAME = "dataset_payload"
DATASET_UNPACK_DIRNAME = ".dataset_payload_unpack"
KAGGLEIGNORE_NAME = ".kaggleignore"
DATASET_METADATA_NAME = "dataset-metadata.json"
DATASET_PAYLOAD_MANIFEST_NAME = "dataset_payload_manifest.json"
DATASET_SRC_ARCHIVE_NAME = "src_llcore.zip"
DATASET_PKG_ARCHIVE_NAME = "pkg_llcore.zip"
_DATASET_ARCHIVE_MAX_ENTRIES = 4096
_DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES = 256 * 1024 * 1024
_TEXT_SCAN_SUFFIXES = {".py", ".json", ".txt", ".md", ".cfg", ".ini", ".toml", ".yaml", ".yml", ".rst"}
_BOOL_TEXT_VALUES = {"true", "false"}
_PUBLISH_BLOCKLIST_ARCHIVE_SUFFIXES = {
    ".pem",
    ".key",
    ".p12",
    ".pfx",
    ".p8",
    ".der",
    ".crt",
    ".cer",
    ".csr",
    ".jks",
    ".keystore",
}
_PUBLISH_BLOCKLIST_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("private-key", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("openai-api-key-name", re.compile(r"\bOPENAI_API_KEY\b")),
    ("kaggle-key-name", re.compile(r"\bKAGGLE_KEY\b")),
    ("kaggle-api-token-name", re.compile(r"\bKAGGLE_API_TOKEN\b")),
    ("local-windows-path", re.compile(r"\b[A-Za-z]:(?:\\|/)+(?:Users|projects)\b")),
)
_EMBEDDED_COPIED_FILE_PATHS = {
    "corpus": "input_corpus.txt",
    "config": "config.json",
    "metadata": "kernel-metadata.json",
    "src_llcore": "src/llcore",
    "pkg_llcore": "llcore",
    "license": "LICENSE",
    "notice": "NOTICE",
}
_DATASET_COPIED_FILE_PATHS = {
    "metadata": "kernel-metadata.json",
    "runner": "runner.py",
    "license": "LICENSE",
    "notice": "NOTICE",
    "dataset_payload": DATASET_PAYLOAD_DIRNAME,
}
_DATASET_PAYLOAD_COPIED_FILE_KEYS = (
    "corpus",
    "config",
    "metadata",
    "src_llcore_zip",
    "pkg_llcore_zip",
    "license",
    "notice",
)
_DATASET_PAYLOAD_COPIED_FILE_PATHS = {
    "corpus": "input_corpus.txt",
    "config": "config.json",
    "metadata": DATASET_METADATA_NAME,
    "src_llcore_zip": DATASET_SRC_ARCHIVE_NAME,
    "pkg_llcore_zip": DATASET_PKG_ARCHIVE_NAME,
    "license": "LICENSE",
    "notice": "NOTICE",
}
_DATASET_REQUIRED_KAGGLEIGNORE_ENTRIES = (
    f"{DATASET_PAYLOAD_DIRNAME}/",
    f"{DATASET_UNPACK_DIRNAME}/",
    "artifacts/",
    "preflight_report.json",
    "prepare_report.json",
)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_ignored_source_path(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(
        (p for p in root.rglob("*") if p.is_file() and not _is_ignored_source_path(p)),
        key=lambda item: PurePosixPath(item.relative_to(root).as_posix()).parts,
    ):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _sha256_zip_tree(archive_path: Path, *, expected_prefix: str) -> str:
    digest = hashlib.sha256()
    seen_members: set[str] = set()
    file_members: set[PurePosixPath] = set()
    dir_members: set[PurePosixPath] = set()
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if len(infos) > _DATASET_ARCHIVE_MAX_ENTRIES:
            raise ValueError(
                f"{archive_path.name} contains too many members: {len(infos)} > {_DATASET_ARCHIVE_MAX_ENTRIES}"
            )
        total_uncompressed = 0
        for info in sorted(
            infos,
            key=lambda item: PurePosixPath(item.filename.replace("\\", "/")).parts,
        ):
            member_name = info.filename.replace("\\", "/")
            if not member_name:
                raise ValueError(f"{archive_path.name} contains an empty member name")
            if member_name in seen_members:
                raise ValueError(f"{archive_path.name} contains duplicate member name: {member_name}")
            seen_members.add(member_name)
            if member_name.startswith("/") or member_name.startswith("../"):
                raise ValueError(f"{archive_path.name} contains unsafe member path: {member_name}")
            if not member_name.startswith(expected_prefix):
                raise ValueError(
                    f"{archive_path.name} contains member outside expected prefix {expected_prefix!r}: {member_name}"
                )
            member_path = PurePosixPath(member_name)
            if any(part in {"", ".", ".."} for part in member_path.parts):
                raise ValueError(f"{archive_path.name} contains unsafe member path: {member_name}")
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise ValueError(f"{archive_path.name} contains forbidden symlink member: {member_name}")
            if member_path in file_members:
                raise ValueError(
                    f"{archive_path.name} reuses a file path as a directory: {member_name}"
                )
            if info.is_dir():
                dir_members.add(member_path)
                continue
            file_conflict = next((parent for parent in member_path.parents if parent in file_members), None)
            if file_conflict is not None:
                raise ValueError(
                    f"{archive_path.name} contains file/directory collision at: {file_conflict.as_posix()}"
                )
            if member_path in dir_members:
                raise ValueError(
                    f"{archive_path.name} reuses a directory path as a file: {member_name}"
                )
            file_members.add(member_path)
            dir_members.update(member_path.parents)
            total_uncompressed += info.file_size
            if total_uncompressed > _DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES:
                raise ValueError(
                    f"{archive_path.name} exceeds extracted size budget: "
                    f"{total_uncompressed} > {_DATASET_ARCHIVE_MAX_UNCOMPRESSED_BYTES}"
                )
            rel = member_name.removeprefix(expected_prefix).encode("utf-8")
            digest.update(rel)
            digest.update(b"\0")
            digest.update(zf.read(info))
            digest.update(b"\0")
    return digest.hexdigest()


def _text_publish_findings(label: str, text: str) -> list[str]:
    findings: list[str] = []
    for finding_label, pattern in _PUBLISH_BLOCKLIST_PATTERNS:
        if pattern.search(text):
            findings.append(f"{label}: blocked publish marker {finding_label!r} detected")
    return findings


def _archive_member_publish_findings(label: str, member_name: str) -> list[str]:
    findings: list[str] = []
    suffix = PurePosixPath(member_name.replace("\\", "/")).suffix.lower()
    if suffix in _PUBLISH_BLOCKLIST_ARCHIVE_SUFFIXES:
        findings.append(f"{label}: blocked publish archive member suffix {suffix!r} detected")
    return findings


def _bundle_dir_label(bundle_dir: Path) -> str:
    return bundle_dir.name


def _normalized_bundle_top_level_entry(path: Path, *, bundle_dir: Path) -> str:
    return path.relative_to(bundle_dir).as_posix()


def _sanitize_report_text(text: str, *, bundle_dir: Path) -> str:
    return text.replace(str(bundle_dir), "<bundle_dir>")


def _safe_extract_zip(
    archive_path: Path,
    dest_root: Path,
    *,
    expected_prefix: str,
    max_entries: int = 4096,
    max_uncompressed_bytes: int = 256 * 1024 * 1024,
) -> None:
    total_uncompressed = 0
    seen_members: set[str] = set()
    extracted_files = 0
    resolved_dest_root = dest_root.resolve()
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_entries:
            raise ValueError(
                f"archive has too many entries: {archive_path} ({len(infos)} > {max_entries})"
            )
        for info in infos:
            member_name = info.filename.replace("\\", "/")
            if not member_name:
                raise ValueError(f"archive contains an empty member name: {archive_path}")
            if member_name in seen_members:
                raise ValueError(f"archive contains a duplicate member name: {member_name}")
            seen_members.add(member_name)
            if member_name.startswith("/") or member_name.startswith("../"):
                raise ValueError(f"archive member escapes extraction root: {member_name}")
            member_path = PurePosixPath(member_name)
            if any(part in {"", ".", ".."} for part in member_path.parts):
                raise ValueError(f"archive member is not a safe relative path: {member_name}")
            if not member_name.startswith(expected_prefix):
                raise ValueError(
                    f"archive member does not stay under expected prefix {expected_prefix!r}: {member_name}"
                )
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise ValueError(f"symlinks are not allowed in dataset payload archive: {member_name}")
            target = (dest_root / member_path).resolve()
            if not target.is_relative_to(resolved_dest_root):
                raise ValueError(f"archive member resolves outside extraction root: {member_name}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total_uncompressed += info.file_size
            if total_uncompressed > max_uncompressed_bytes:
                raise ValueError(
                    f"archive exceeds extracted size budget: {archive_path} "
                    f"({total_uncompressed} > {max_uncompressed_bytes})"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files += 1
    if extracted_files < 1:
        raise ValueError(f"archive did not extract any files under expected prefix: {archive_path}")


def _build_simulated_dataset_mount(dataset_payload_dir: Path) -> Path:
    temp_root = Path(tempfile.mkdtemp(prefix="llcore-kaggle-dataset-smoke-"))
    for name in (
        "config.json",
        "input_corpus.txt",
        "LICENSE",
        "NOTICE",
        DATASET_METADATA_NAME,
        DATASET_PAYLOAD_MANIFEST_NAME,
    ):
        shutil.copyfile(dataset_payload_dir / name, temp_root / name)
    for archive_name in (DATASET_SRC_ARCHIVE_NAME, DATASET_PKG_ARCHIVE_NAME):
        archive_path = dataset_payload_dir / archive_name
        extract_dir = temp_root / Path(archive_name).stem
        extract_dir.mkdir(parents=True, exist_ok=True)
        expected_prefix = "src/llcore/" if archive_name == DATASET_SRC_ARCHIVE_NAME else "llcore/"
        _safe_extract_zip(archive_path, extract_dir, expected_prefix=expected_prefix)
    return temp_root


def _top_level_publish_safety_summary(bundle_dir: Path) -> dict[str, object]:
    findings: list[str] = []
    scanned_text_files: list[str] = []
    for path in sorted(bundle_dir.iterdir(), key=lambda item: item.name):
        if path.is_symlink() or not path.is_file():
            continue
        if path.suffix in _TEXT_SCAN_SUFFIXES or path.name in {
            "LICENSE",
            "NOTICE",
            "README.md",
            "kernel-metadata.json",
            "bundle_manifest.json",
            "config.json",
            "input_corpus.txt",
            "preflight_report.json",
            "prepare_report.json",
        }:
            scanned_text_files.append(path.name)
            findings.extend(
                _text_publish_findings(
                    _normalized_bundle_top_level_entry(path, bundle_dir=bundle_dir),
                    path.read_text(encoding="utf-8", errors="replace"),
                )
            )
    if findings:
        raise ValueError("bundle root secret/path scan failed: " + "; ".join(findings))
    return {
        "status": "passed",
        "scanned_text_files": scanned_text_files,
    }


def _validate_dataset_bundle_kaggleignore(kaggleignore_path: Path) -> None:
    if not kaggleignore_path.is_file():
        raise ValueError("dataset bundle must include .kaggleignore to exclude dataset_payload/ from kernel push")
    kaggleignore_text = kaggleignore_path.read_text(encoding="utf-8")
    kaggleignore_lines = {line.strip() for line in kaggleignore_text.splitlines()}
    required_entries = _DATASET_REQUIRED_KAGGLEIGNORE_ENTRIES
    missing_ignore_entries = [entry for entry in required_entries if entry not in kaggleignore_lines]
    if missing_ignore_entries:
        raise ValueError(".kaggleignore must exclude: " + ", ".join(missing_ignore_entries))
    protected_segments = {DATASET_PAYLOAD_DIRNAME, DATASET_UNPACK_DIRNAME}
    for raw_line in kaggleignore_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or not line.startswith("!"):
            continue
        target = line[1:].strip().lstrip("/").replace("\\", "/")
        if target.startswith("./"):
            target = target[2:]
        target_segments = [segment for segment in target.split("/") if segment not in {"", "."}]
        if any(segment in protected_segments for segment in target_segments):
            raise ValueError(
                ".kaggleignore must not re-include protected dataset paths: "
                + line
            )


def _validate_bundle_top_level_entries(bundle_dir: Path, *, data_mode: str) -> None:
    allowed_entries: dict[str, str] = {
        "kernel-metadata.json": "file",
        "bundle_manifest.json": "file",
        "LICENSE": "file",
        "NOTICE": "file",
        "runner.py": "file",
        "README.md": "file",
        "artifacts": "dir",
        "preflight_report.json": "file",
        "prepare_report.json": "file",
    }
    if data_mode == "embedded":
        allowed_entries.update(
            {
                "config.json": "file",
                "input_corpus.txt": "file",
                "src": "dir",
                "llcore": "dir",
            }
        )
    else:
        allowed_entries.update(
            {
                KAGGLEIGNORE_NAME: "file",
                DATASET_PAYLOAD_DIRNAME: "dir",
                DATASET_UNPACK_DIRNAME: "dir",
            }
        )
    unexpected_entries: list[str] = []
    invalid_typed_entries: list[str] = []
    for path in bundle_dir.iterdir():
        expected_kind = allowed_entries.get(path.name)
        if expected_kind is None:
            unexpected_entries.append(path.name)
            continue
        if path.is_symlink():
            invalid_typed_entries.append(f"{path.name} (symlink not allowed)")
            continue
        if expected_kind == "file" and not path.is_file():
            invalid_typed_entries.append(f"{path.name} (expected file)")
            continue
        if expected_kind == "dir" and not path.is_dir():
            invalid_typed_entries.append(f"{path.name} (expected directory)")
            continue
    if unexpected_entries:
        raise ValueError(
            "bundle contains unexpected top-level entries: " + ", ".join(unexpected_entries)
        )
    if invalid_typed_entries:
        raise ValueError(
            "bundle contains invalid top-level entry types: " + ", ".join(sorted(invalid_typed_entries))
        )


def _scan_dataset_payload_publish_safety(dataset_payload_dir: Path) -> dict[str, object]:
    findings: list[str] = []
    scanned_text_files: list[str] = []
    scanned_archive_text_members: list[str] = []
    known_top_level_files = {
        "config.json",
        DATASET_METADATA_NAME,
        DATASET_PAYLOAD_MANIFEST_NAME,
        "input_corpus.txt",
        "LICENSE",
        "NOTICE",
        DATASET_SRC_ARCHIVE_NAME,
        DATASET_PKG_ARCHIVE_NAME,
    }
    top_level_entries = sorted(dataset_payload_dir.iterdir(), key=lambda path: path.name)
    for entry in top_level_entries:
        if entry.is_dir():
            raise ValueError(
                "dataset payload secret/path scan failed: unexpected nested directory "
                f"under dataset_payload/: {entry.name}"
            )
        if not entry.is_file():
            raise ValueError(
                "dataset payload secret/path scan failed: unexpected non-file entry "
                f"under dataset_payload/: {entry.name}"
            )
        if entry.name not in known_top_level_files:
            raise ValueError(
                "dataset payload secret/path scan failed: unexpected unscanned file "
                f"under dataset_payload/: {entry.name}"
            )
    for path in top_level_entries:
        if path.suffix in _TEXT_SCAN_SUFFIXES or path.name in {
            "LICENSE",
            "NOTICE",
            DATASET_METADATA_NAME,
            DATASET_PAYLOAD_MANIFEST_NAME,
            "config.json",
            "input_corpus.txt",
        }:
            scanned_text_files.append(path.name)
            findings.extend(
                _text_publish_findings(
                    str(path.relative_to(dataset_payload_dir)),
                    path.read_text(encoding="utf-8", errors="replace"),
                )
            )
        if path.name not in {DATASET_SRC_ARCHIVE_NAME, DATASET_PKG_ARCHIVE_NAME}:
            continue
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                findings.extend(_archive_member_publish_findings(f"{path.name}:{info.filename}", info.filename))
                member = PurePosixPath(info.filename.replace("\\", "/"))
                if member.suffix not in _TEXT_SCAN_SUFFIXES:
                    continue
                scanned_archive_text_members.append(f"{path.name}:{member.as_posix()}")
                findings.extend(
                    _text_publish_findings(
                        f"{path.name}:{member.as_posix()}",
                        zf.read(info).decode("utf-8", errors="replace"),
                    )
                )
    if findings:
        raise ValueError("dataset payload secret/path scan failed: " + "; ".join(findings))
    return {
        "status": "passed",
        "dataset_payload_dir": dataset_payload_dir.name,
        "scanned_text_files": scanned_text_files,
        "scanned_top_level_files": [path.name for path in top_level_entries],
        "scanned_archive_text_member_count": len(scanned_archive_text_members),
        "scanned_archive_text_members_sample": scanned_archive_text_members[:8],
    }


def _require_bool_text(value: object, *, field_name: str) -> str:
    if not isinstance(value, str) or value not in _BOOL_TEXT_VALUES:
        raise ValueError(f"{field_name} must be one of: true, false")
    return value


def _validate_bundle_dir(bundle_dir: Path) -> dict[str, object]:
    missing = [name for name in REQUIRED_FILES if not (bundle_dir / name).is_file()]
    if missing:
        raise ValueError(f"bundle is missing required paths: {', '.join(missing)}")

    metadata = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))

    if not isinstance(metadata, dict):
        raise ValueError("kernel-metadata.json must contain an object")
    if not isinstance(manifest, dict):
        raise ValueError("bundle_manifest.json must contain an object")
    data_mode = manifest.get("data_mode", "embedded")
    if data_mode not in {"embedded", "dataset"}:
        raise ValueError("bundle_manifest.json.data_mode must be 'embedded' or 'dataset'")
    _validate_bundle_top_level_entries(bundle_dir, data_mode=data_mode)

    manifest_runner_sha256 = manifest.get("runner_sha256")
    manifest_license_sha256 = manifest.get("license_sha256")
    manifest_notice_sha256 = manifest.get("notice_sha256")
    manifest_is_private = manifest.get("is_private")
    manifest_enable_internet = manifest.get("enable_internet")
    manifest_enable_gpu = manifest.get("enable_gpu")
    if not isinstance(manifest_runner_sha256, str) or len(manifest_runner_sha256) != 64:
        raise ValueError("bundle_manifest.json.runner_sha256 must be a 64-char sha256 hex string")
    if not isinstance(manifest_license_sha256, str) or len(manifest_license_sha256) != 64:
        raise ValueError("bundle_manifest.json.license_sha256 must be a 64-char sha256 hex string")
    if not isinstance(manifest_notice_sha256, str) or len(manifest_notice_sha256) != 64:
        raise ValueError("bundle_manifest.json.notice_sha256 must be a 64-char sha256 hex string")
    actual_license_sha256 = _sha256_text(bundle_dir / "LICENSE")
    actual_notice_sha256 = _sha256_text(bundle_dir / "NOTICE")
    if actual_license_sha256 != manifest_license_sha256:
        raise ValueError("LICENSE sha256 does not match bundle_manifest.json.license_sha256")
    if actual_notice_sha256 != manifest_notice_sha256:
        raise ValueError("NOTICE sha256 does not match bundle_manifest.json.notice_sha256")

    metadata_id = metadata.get("id")
    manifest_kernel_id = manifest.get("kernel_id")
    if metadata_id != manifest_kernel_id:
        raise ValueError("kernel-metadata.json id does not match bundle_manifest.json kernel_id")
    if not isinstance(metadata_id, str) or not metadata_id.strip():
        raise ValueError("kernel-metadata.json id must be a non-empty string")
    if metadata.get("code_file") != "runner.py":
        raise ValueError("kernel-metadata.json code_file must be 'runner.py'")
    if manifest.get("runner") != "runner.py":
        raise ValueError("bundle_manifest.json runner must be 'runner.py'")
    metadata_title = metadata.get("title")
    manifest_title = manifest.get("title")
    if metadata_title != manifest_title:
        raise ValueError("kernel-metadata.json title does not match bundle_manifest.json title")
    if not isinstance(metadata_title, str) or not metadata_title.strip():
        raise ValueError("kernel-metadata.json title must be a non-empty string")
    actual_runner_sha256 = _sha256_text(bundle_dir / "runner.py")
    if actual_runner_sha256 != manifest_runner_sha256:
        raise ValueError("runner.py sha256 does not match bundle_manifest.json.runner_sha256")
    metadata_enable_gpu = metadata.get("enable_gpu")
    metadata_enable_internet = metadata.get("enable_internet")
    metadata_is_private = metadata.get("is_private")
    metadata_enable_tpu = metadata.get("enable_tpu")
    metadata_machine_shape = metadata.get("machine_shape")
    manifest_is_private = _require_bool_text(manifest_is_private, field_name="bundle_manifest.json is_private")
    manifest_enable_internet = _require_bool_text(
        manifest_enable_internet, field_name="bundle_manifest.json enable_internet"
    )
    manifest_enable_gpu = _require_bool_text(manifest_enable_gpu, field_name="bundle_manifest.json enable_gpu")
    manifest_enable_tpu = _require_bool_text(
        manifest.get("enable_tpu"), field_name="bundle_manifest.json enable_tpu"
    )
    metadata_is_private = _require_bool_text(metadata_is_private, field_name="kernel-metadata.json is_private")
    metadata_enable_internet = _require_bool_text(
        metadata_enable_internet, field_name="kernel-metadata.json enable_internet"
    )
    metadata_enable_gpu = _require_bool_text(metadata_enable_gpu, field_name="kernel-metadata.json enable_gpu")
    metadata_enable_tpu = _require_bool_text(metadata_enable_tpu, field_name="kernel-metadata.json enable_tpu")
    if metadata_is_private != manifest_is_private:
        raise ValueError("kernel-metadata.json is_private does not match bundle_manifest.json is_private")
    if metadata_enable_internet != manifest_enable_internet:
        raise ValueError(
            "kernel-metadata.json enable_internet does not match bundle_manifest.json enable_internet"
        )
    if metadata_enable_gpu != manifest_enable_gpu:
        raise ValueError("kernel-metadata.json enable_gpu does not match bundle_manifest.json enable_gpu")
    if metadata_enable_tpu != manifest_enable_tpu:
        raise ValueError("kernel-metadata.json enable_tpu does not match bundle_manifest.json enable_tpu")
    if metadata_enable_gpu == "true":
        if not isinstance(metadata_machine_shape, str) or not metadata_machine_shape:
            raise ValueError("kernel-metadata.json machine_shape must be set when enable_gpu=true")
    else:
        if "machine_shape" in metadata and metadata_machine_shape is not None:
            raise ValueError("kernel-metadata.json machine_shape must be omitted/null when enable_gpu!=true")
    if metadata.get("machine_shape") != manifest.get("machine_shape"):
        raise ValueError("kernel-metadata.json machine_shape does not match bundle_manifest.json machine_shape")
    copied_files = manifest.get("copied_files")
    if not isinstance(copied_files, dict):
        raise ValueError("bundle_manifest.json.copied_files must contain an object")
    expected_keys = EMBEDDED_COPIED_FILE_KEYS if data_mode == "embedded" else DATASET_COPIED_FILE_KEYS
    expected_paths = _EMBEDDED_COPIED_FILE_PATHS if data_mode == "embedded" else _DATASET_COPIED_FILE_PATHS
    if set(copied_files) != set(expected_keys):
        raise ValueError(
            "bundle_manifest.json.copied_files must contain exactly: "
            + ", ".join(expected_keys)
        )
    for logical_name, relative_path in copied_files.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"bundle_manifest.json copied_files[{logical_name!r}] must be a non-empty string")
        expected_path = expected_paths[logical_name]
        if relative_path != expected_path:
            raise ValueError(
                f"bundle_manifest.json copied_files[{logical_name!r}] must be {expected_path!r}, got {relative_path!r}"
            )
        resolved = (bundle_dir / relative_path).resolve()
        if not resolved.is_relative_to(bundle_dir):
            raise ValueError(
                f"bundle_manifest.json copied_files[{logical_name!r}] escapes bundle_dir: {relative_path}"
            )
        if not resolved.exists():
            raise ValueError(
                f"bundle_manifest.json copied_files[{logical_name!r}] points to a missing path: {relative_path}"
            )

    if data_mode == "embedded":
        src_llcore = bundle_dir / "src" / "llcore"
        pkg_llcore = bundle_dir / "llcore"
        config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("config.json must contain an object")
        compare_config = config.get("compare_config")
        corpus_sha256 = config.get("corpus_sha256")
        manifest_corpus_sha256 = manifest.get("corpus_sha256")
        manifest_source_sha256 = manifest.get("source_sha256")
        manifest_config_sha256 = manifest.get("config_sha256")
        if not src_llcore.is_dir():
            raise ValueError("bundle is missing required path: src/llcore")
        if not pkg_llcore.is_dir():
            raise ValueError("bundle is missing required path: llcore")
        if not isinstance(compare_config, dict):
            raise ValueError("config.json.compare_config must contain an object")
        if not isinstance(corpus_sha256, str) or len(corpus_sha256) != 64:
            raise ValueError("config.json.corpus_sha256 must be a 64-char sha256 hex string")
        if not isinstance(manifest_corpus_sha256, str) or len(manifest_corpus_sha256) != 64:
            raise ValueError("bundle_manifest.json.corpus_sha256 must be a 64-char sha256 hex string")
        if not isinstance(manifest_source_sha256, str) or len(manifest_source_sha256) != 64:
            raise ValueError("bundle_manifest.json.source_sha256 must be a 64-char sha256 hex string")
        if not isinstance(manifest_config_sha256, str) or len(manifest_config_sha256) != 64:
            raise ValueError("bundle_manifest.json.config_sha256 must be a 64-char sha256 hex string")
        try:
            CompareConfig(**compare_config)
        except Exception as exc:
            raise ValueError(f"config.json.compare_config is invalid: {exc}") from exc
        corpus_path = bundle_dir / "input_corpus.txt"
        actual_corpus_sha256 = _sha256_text(corpus_path)
        actual_config_sha256 = _sha256_text(bundle_dir / "config.json")
        if actual_corpus_sha256 != corpus_sha256:
            raise ValueError("input_corpus.txt sha256 does not match config.json.corpus_sha256")
        if corpus_sha256 != manifest_corpus_sha256:
            raise ValueError("config.json.corpus_sha256 does not match bundle_manifest.json.corpus_sha256")
        actual_source_sha256 = _sha256_tree(src_llcore)
        actual_pkg_source_sha256 = _sha256_tree(pkg_llcore)
        if actual_source_sha256 != manifest_source_sha256:
            raise ValueError("src/llcore sha256 does not match bundle_manifest.json.source_sha256")
        if actual_pkg_source_sha256 != manifest_source_sha256:
            raise ValueError("llcore sha256 does not match bundle_manifest.json.source_sha256")
        if actual_config_sha256 != manifest_config_sha256:
            raise ValueError("config.json sha256 does not match bundle_manifest.json.config_sha256")
        corpus_file_name = config.get("corpus_file_name")
        if not isinstance(corpus_file_name, str) or not corpus_file_name:
            raise ValueError("config.json.corpus_file_name must be a non-empty string")
        config_summary = {
            "block_size": compare_config.get("block_size"),
            "max_iters": compare_config.get("max_iters"),
            "corpus_sha256": corpus_sha256,
            "config_sha256": manifest_config_sha256,
        }
        manifest_summary: dict[str, object] = {
            "kernel_id": manifest_kernel_id,
            "runner": manifest.get("runner"),
            "copied_files": copied_files,
            "source_sha256": manifest_source_sha256,
            "data_mode": data_mode,
        }
    else:
        dataset_sources = metadata.get("dataset_sources")
        dataset_payload_rel = manifest.get("dataset_payload_rel")
        dataset_payload_manifest_sha256 = manifest.get("dataset_payload_manifest_sha256")
        dataset_source = manifest.get("dataset_source")
        dataset_mount_name = manifest.get("dataset_mount_name")
        if not isinstance(dataset_sources, list) or len(dataset_sources) != 1 or not all(
            isinstance(item, str) and item for item in dataset_sources
        ):
            raise ValueError("kernel-metadata.json dataset_sources must contain exactly one non-empty string")
        if dataset_sources[0] != dataset_source:
            raise ValueError("kernel-metadata.json dataset_sources[0] must match bundle_manifest.json.dataset_source")
        if not isinstance(dataset_payload_rel, str) or not dataset_payload_rel:
            raise ValueError("bundle_manifest.json.dataset_payload_rel must be a non-empty string")
        if not isinstance(dataset_payload_manifest_sha256, str) or len(dataset_payload_manifest_sha256) != 64:
            raise ValueError("bundle_manifest.json.dataset_payload_manifest_sha256 must be a 64-char sha256 hex string")
        if not isinstance(dataset_source, str) or not dataset_source:
            raise ValueError("bundle_manifest.json.dataset_source must be a non-empty string")
        if not isinstance(dataset_mount_name, str) or not dataset_mount_name:
            raise ValueError("bundle_manifest.json.dataset_mount_name must be a non-empty string")
        dataset_payload_dir = bundle_dir / dataset_payload_rel
        kaggleignore_path = bundle_dir / KAGGLEIGNORE_NAME
        dataset_manifest_path = dataset_payload_dir / DATASET_PAYLOAD_MANIFEST_NAME
        dataset_metadata_path = dataset_payload_dir / DATASET_METADATA_NAME
        config_path = dataset_payload_dir / "config.json"
        corpus_path = dataset_payload_dir / "input_corpus.txt"
        src_archive_path = dataset_payload_dir / DATASET_SRC_ARCHIVE_NAME
        pkg_archive_path = dataset_payload_dir / DATASET_PKG_ARCHIVE_NAME
        missing_dataset = [
            rel
            for rel in (
                dataset_payload_rel,
                f"{dataset_payload_rel}/{DATASET_PAYLOAD_MANIFEST_NAME}",
                f"{dataset_payload_rel}/{DATASET_METADATA_NAME}",
                f"{dataset_payload_rel}/config.json",
                f"{dataset_payload_rel}/input_corpus.txt",
                f"{dataset_payload_rel}/{DATASET_SRC_ARCHIVE_NAME}",
                f"{dataset_payload_rel}/{DATASET_PKG_ARCHIVE_NAME}",
                f"{dataset_payload_rel}/LICENSE",
                f"{dataset_payload_rel}/NOTICE",
            )
            if not (bundle_dir / rel).exists()
        ]
        if missing_dataset:
            raise ValueError("bundle is missing required dataset payload paths: " + ", ".join(missing_dataset))
        _validate_dataset_bundle_kaggleignore(kaggleignore_path)
        if _sha256_text(dataset_manifest_path) != dataset_payload_manifest_sha256:
            raise ValueError(
                "dataset payload manifest sha256 does not match bundle_manifest.json.dataset_payload_manifest_sha256"
            )
        dataset_manifest = json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        dataset_metadata = json.loads(dataset_metadata_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(dataset_manifest, dict):
            raise ValueError("dataset_payload_manifest.json must contain an object")
        if not isinstance(dataset_metadata, dict):
            raise ValueError("dataset-metadata.json must contain an object")
        if not isinstance(config, dict):
            raise ValueError("dataset payload config.json must contain an object")
        dataset_copied_files = dataset_manifest.get("copied_files")
        if not isinstance(dataset_copied_files, dict):
            raise ValueError("dataset_payload_manifest.json.copied_files must contain an object")
        if set(dataset_copied_files) != set(_DATASET_PAYLOAD_COPIED_FILE_KEYS):
            raise ValueError(
                "dataset_payload_manifest.json.copied_files must contain exactly: "
                + ", ".join(_DATASET_PAYLOAD_COPIED_FILE_KEYS)
            )
        for logical_name, relative_path in dataset_copied_files.items():
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError(
                    f"dataset_payload_manifest.json copied_files[{logical_name!r}] must be a non-empty string"
                )
            expected_path = _DATASET_PAYLOAD_COPIED_FILE_PATHS[logical_name]
            if relative_path != expected_path:
                raise ValueError(
                    "dataset_payload_manifest.json copied_files"
                    f"[{logical_name!r}] must be {expected_path!r}, got {relative_path!r}"
                )
            resolved = (dataset_payload_dir / relative_path).resolve()
            if not resolved.is_relative_to(dataset_payload_dir.resolve()):
                raise ValueError(
                    "dataset_payload_manifest.json copied_files"
                    f"[{logical_name!r}] escapes dataset_payload/: {relative_path}"
                )
            if not resolved.exists():
                raise ValueError(
                    "dataset_payload_manifest.json copied_files"
                    f"[{logical_name!r}] points to a missing path: {relative_path}"
                )
        compare_config = config.get("compare_config")
        corpus_sha256 = config.get("corpus_sha256")
        if not isinstance(compare_config, dict):
            raise ValueError("dataset payload config.json.compare_config must contain an object")
        if not isinstance(corpus_sha256, str) or len(corpus_sha256) != 64:
            raise ValueError("dataset payload config.json.corpus_sha256 must be a 64-char sha256 hex string")
        try:
            CompareConfig(**compare_config)
        except Exception as exc:
            raise ValueError(f"dataset payload config.json.compare_config is invalid: {exc}") from exc
        if dataset_metadata.get("id") != dataset_source:
            raise ValueError("dataset-metadata.json id must match bundle_manifest.json.dataset_source")
        if dataset_metadata.get("title") != dataset_mount_name:
            raise ValueError("dataset-metadata.json title must match bundle_manifest.json.dataset_mount_name")
        licenses = dataset_metadata.get("licenses")
        if not isinstance(licenses, list) or not licenses:
            raise ValueError("dataset-metadata.json licenses must contain at least one entry")
        if not all(isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name") for item in licenses):
            raise ValueError("dataset-metadata.json licenses entries must contain non-empty string name")
        actual_corpus_sha256 = _sha256_text(corpus_path)
        if actual_corpus_sha256 != corpus_sha256:
            raise ValueError("dataset payload input_corpus.txt sha256 does not match config.json.corpus_sha256")
        if dataset_manifest.get("corpus_sha256") != corpus_sha256:
            raise ValueError("dataset payload manifest corpus_sha256 does not match config.json.corpus_sha256")
        actual_source_sha256 = _sha256_zip_tree(src_archive_path, expected_prefix="src/llcore/")
        actual_pkg_source_sha256 = _sha256_zip_tree(pkg_archive_path, expected_prefix="llcore/")
        if dataset_manifest.get("src_tree_sha256") != actual_source_sha256:
            raise ValueError("dataset payload src_llcore.zip contents do not match dataset manifest src_tree_sha256")
        if dataset_manifest.get("pkg_tree_sha256") != actual_pkg_source_sha256:
            raise ValueError("dataset payload pkg_llcore.zip contents do not match dataset manifest pkg_tree_sha256")
        if dataset_manifest.get("src_archive_sha256") != _sha256_text(src_archive_path):
            raise ValueError("dataset payload src_llcore.zip sha256 does not match dataset manifest src_archive_sha256")
        if dataset_manifest.get("pkg_archive_sha256") != _sha256_text(pkg_archive_path):
            raise ValueError("dataset payload pkg_llcore.zip sha256 does not match dataset manifest pkg_archive_sha256")
        if dataset_manifest.get("config_sha256") != _sha256_text(config_path):
            raise ValueError("dataset payload config.json sha256 does not match dataset manifest config_sha256")
        if dataset_manifest.get("license_sha256") != _sha256_text(dataset_payload_dir / "LICENSE"):
            raise ValueError("dataset payload LICENSE sha256 does not match dataset manifest license_sha256")
        if dataset_manifest.get("notice_sha256") != _sha256_text(dataset_payload_dir / "NOTICE"):
            raise ValueError("dataset payload NOTICE sha256 does not match dataset manifest notice_sha256")
        if dataset_manifest.get("dataset_metadata_sha256") != _sha256_text(dataset_metadata_path):
            raise ValueError("dataset-metadata.json sha256 does not match dataset payload manifest")
        publish_safety_summary = _scan_dataset_payload_publish_safety(dataset_payload_dir)
        config_summary = {
            "block_size": compare_config.get("block_size"),
            "max_iters": compare_config.get("max_iters"),
            "corpus_sha256": corpus_sha256,
            "config_sha256": dataset_manifest.get("config_sha256"),
            "dataset_source": dataset_source,
            "dataset_metadata_path": f"{dataset_payload_rel}/{DATASET_METADATA_NAME}",
        }
        manifest_summary = {
            "kernel_id": manifest_kernel_id,
            "runner": manifest.get("runner"),
            "copied_files": copied_files,
            "data_mode": data_mode,
            "dataset_source": dataset_source,
            "dataset_mount_name": dataset_mount_name,
            "dataset_payload_rel": dataset_payload_rel,
            "dataset_publish_dir": dataset_payload_rel,
            "src_tree_sha256": dataset_manifest.get("src_tree_sha256"),
            "pkg_tree_sha256": dataset_manifest.get("pkg_tree_sha256"),
            "publish_safety": publish_safety_summary,
        }

    metadata_summary = {
        "kernel_id": metadata_id,
        "enable_gpu": metadata.get("enable_gpu"),
        "enable_tpu": metadata.get("enable_tpu"),
        "enable_internet": metadata.get("enable_internet"),
        "is_private": metadata.get("is_private"),
        "machine_shape": metadata.get("machine_shape"),
    }
    return {
        "metadata": metadata_summary,
        "config": config_summary,
        "manifest": manifest_summary,
    }


def _run_runner(bundle_dir: Path, *, timeout_s: int) -> dict[str, object]:
    out_json = bundle_dir / "artifacts" / "lm_compare.json"
    out_md = bundle_dir / "artifacts" / "lm_compare.md"
    out_svg = bundle_dir / "artifacts" / "lm_compare.svg"
    out_json.unlink(missing_ok=True)
    out_md.unlink(missing_ok=True)
    out_svg.unlink(missing_ok=True)
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONDONTWRITEBYTECODE": "1"}
    simulated_data_root: Path | None = None
    if manifest.get("data_mode") == "dataset":
        dataset_payload_rel = manifest.get("dataset_payload_rel")
        if isinstance(dataset_payload_rel, str) and dataset_payload_rel:
            simulated_data_root = _build_simulated_dataset_mount((bundle_dir / dataset_payload_rel).resolve())
            env["LLCORE_KAGGLE_DATA_ROOT"] = str(simulated_data_root)
    try:
        proc = subprocess.run(
            [sys.executable, str(bundle_dir / "runner.py")],
            cwd=bundle_dir,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=timeout_s,
        )
    finally:
        if simulated_data_root is not None:
            shutil.rmtree(simulated_data_root, ignore_errors=True)
    payload: dict[str, object] = {
        "returncode": proc.returncode,
        "stdout": _sanitize_report_text(proc.stdout, bundle_dir=bundle_dir),
        "stderr": _sanitize_report_text(proc.stderr, bundle_dir=bundle_dir),
        "output_exists": out_json.is_file(),
    }
    if proc.returncode != 0:
        raise ValueError(f"runner.py failed with rc={proc.returncode}: {proc.stderr.strip()}")
    if not out_json.is_file():
        raise ValueError("runner.py completed without writing artifacts/lm_compare.json")
    if not out_md.is_file():
        raise ValueError("runner.py completed without writing artifacts/lm_compare.md")
    if not out_svg.is_file():
        raise ValueError("runner.py completed without writing artifacts/lm_compare.svg")
    try:
        json.loads(out_json.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"runner.py wrote invalid artifacts/lm_compare.json: {exc}") from exc
    return payload


def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
    bundle_dir = bundle_dir.resolve()
    if not bundle_dir.is_dir():
        raise ValueError(f"bundle directory does not exist: {bundle_dir}")
    result: dict[str, object] = {
        "bundle_dir": _bundle_dir_label(bundle_dir),
        "checks": _validate_bundle_dir(bundle_dir),
        "runner": None,
    }
    if run_runner:
        result["runner"] = _run_runner(bundle_dir, timeout_s=runner_timeout)
    top_level_publish_safety = _top_level_publish_safety_summary(bundle_dir)
    checks = result.get("checks")
    if isinstance(checks, dict):
        manifest = checks.get("manifest")
        if isinstance(manifest, dict):
            manifest["bundle_root_publish_safety"] = top_level_publish_safety
    return result


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Validate a local Kaggle llcore.lm.compare bundle without pushing it. "
            "Can optionally run the bundled runner.py locally."
        )
    )
    ap.add_argument("--bundle-dir", required=True, help="existing Kaggle bundle directory")
    ap.add_argument("--json", help="optional path to write structured preflight report")
    ap.add_argument(
        "--run-runner",
        action="store_true",
        help="run the bundled runner.py locally after structural checks",
    )
    ap.add_argument(
        "--runner-timeout",
        type=int,
        default=300,
        help="timeout in seconds for local runner execution (default: 300)",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.runner_timeout < 1:
        print("error: --runner-timeout must be >= 1", file=sys.stderr)
        return 2
    try:
        report = preflight_bundle(
            Path(args.bundle_dir),
            run_runner=args.run_runner,
            runner_timeout=args.runner_timeout,
        )
    except (ValueError, OSError, zipfile.BadZipFile, subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        _write_json(Path(args.json), report)
    checks = report["checks"]
    checks_map = checks if isinstance(checks, dict) else {}
    metadata = checks_map.get("metadata") if isinstance(checks_map, dict) else {}
    metadata_map = metadata if isinstance(metadata, dict) else {}
    print(
        "[kaggle-preflight]",
        f"dir={report['bundle_dir']}",
        f"gpu={metadata_map.get('enable_gpu')}",
        f"internet={metadata_map.get('enable_internet')}",
        f"runner={'yes' if args.run_runner else 'no'}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
