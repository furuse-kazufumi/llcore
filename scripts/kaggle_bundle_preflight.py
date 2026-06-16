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
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "src")))

from llcore.lm.compare import CompareConfig  # type: ignore[import-untyped]


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
KAGGLEIGNORE_NAME = ".kaggleignore"
DATASET_METADATA_NAME = "dataset-metadata.json"
DATASET_PAYLOAD_MANIFEST_NAME = "dataset_payload_manifest.json"
_BOOL_TEXT_VALUES = {"true", "false"}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_ignored_source_path(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix in {".pyc", ".pyo"}


def _sha256_tree(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not _is_ignored_source_path(p)):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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
    if set(copied_files) != set(expected_keys):
        raise ValueError(
            "bundle_manifest.json.copied_files must contain exactly: "
            + ", ".join(expected_keys)
        )
    for logical_name, relative_path in copied_files.items():
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"bundle_manifest.json copied_files[{logical_name!r}] must be a non-empty string")
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
        src_llcore = dataset_payload_dir / "src" / "llcore"
        pkg_llcore = dataset_payload_dir / "llcore"
        missing_dataset = [
            rel
            for rel in (
                dataset_payload_rel,
                f"{dataset_payload_rel}/{DATASET_PAYLOAD_MANIFEST_NAME}",
                f"{dataset_payload_rel}/{DATASET_METADATA_NAME}",
                f"{dataset_payload_rel}/config.json",
                f"{dataset_payload_rel}/input_corpus.txt",
                f"{dataset_payload_rel}/src/llcore",
                f"{dataset_payload_rel}/llcore",
                f"{dataset_payload_rel}/LICENSE",
                f"{dataset_payload_rel}/NOTICE",
            )
            if not (bundle_dir / rel).exists()
        ]
        if missing_dataset:
            raise ValueError("bundle is missing required dataset payload paths: " + ", ".join(missing_dataset))
        if not kaggleignore_path.is_file():
            raise ValueError("dataset bundle must include .kaggleignore to exclude dataset_payload/ from kernel push")
        kaggleignore_text = kaggleignore_path.read_text(encoding="utf-8")
        if f"{DATASET_PAYLOAD_DIRNAME}/" not in kaggleignore_text.splitlines():
            raise ValueError(".kaggleignore must exclude dataset_payload/")
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
        actual_corpus_sha256 = _sha256_text(corpus_path)
        if actual_corpus_sha256 != corpus_sha256:
            raise ValueError("dataset payload input_corpus.txt sha256 does not match config.json.corpus_sha256")
        if dataset_manifest.get("corpus_sha256") != corpus_sha256:
            raise ValueError("dataset payload manifest corpus_sha256 does not match config.json.corpus_sha256")
        actual_source_sha256 = _sha256_tree(src_llcore)
        actual_pkg_source_sha256 = _sha256_tree(pkg_llcore)
        if dataset_manifest.get("source_sha256") != actual_source_sha256:
            raise ValueError("dataset payload src/llcore sha256 does not match dataset manifest source_sha256")
        if dataset_manifest.get("source_sha256") != actual_pkg_source_sha256:
            raise ValueError("dataset payload llcore sha256 does not match dataset manifest source_sha256")
        if dataset_manifest.get("config_sha256") != _sha256_text(config_path):
            raise ValueError("dataset payload config.json sha256 does not match dataset manifest config_sha256")
        if dataset_manifest.get("license_sha256") != _sha256_text(dataset_payload_dir / "LICENSE"):
            raise ValueError("dataset payload LICENSE sha256 does not match dataset manifest license_sha256")
        if dataset_manifest.get("notice_sha256") != _sha256_text(dataset_payload_dir / "NOTICE"):
            raise ValueError("dataset payload NOTICE sha256 does not match dataset manifest notice_sha256")
        if dataset_manifest.get("dataset_metadata_sha256") != _sha256_text(dataset_metadata_path):
            raise ValueError("dataset-metadata.json sha256 does not match dataset payload manifest")
        config_summary = {
            "block_size": compare_config.get("block_size"),
            "max_iters": compare_config.get("max_iters"),
            "corpus_sha256": corpus_sha256,
            "dataset_source": dataset_source,
        }
        manifest_summary = {
            "kernel_id": manifest_kernel_id,
            "runner": manifest.get("runner"),
            "copied_files": copied_files,
            "data_mode": data_mode,
            "dataset_source": dataset_source,
            "dataset_mount_name": dataset_mount_name,
            "dataset_payload_rel": dataset_payload_rel,
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
    if manifest.get("data_mode") == "dataset":
        dataset_payload_rel = manifest.get("dataset_payload_rel")
        if isinstance(dataset_payload_rel, str) and dataset_payload_rel:
            env["LLCORE_KAGGLE_DATA_ROOT"] = str((bundle_dir / dataset_payload_rel).resolve())
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
    payload: dict[str, object] = {
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
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
        "bundle_dir": str(bundle_dir),
        "checks": _validate_bundle_dir(bundle_dir),
        "runner": None,
    }
    if run_runner:
        result["runner"] = _run_runner(bundle_dir, timeout_s=runner_timeout)
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
    except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
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
