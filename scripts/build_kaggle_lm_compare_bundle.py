# SPDX-License-Identifier: Apache-2.0
"""Build a local Kaggle bundle for llcore.lm.compare without pushing it.

The generated directory is intentionally self-contained:

- `input_corpus.txt` is copied in so the run does not depend on remote fetches.
- `src/llcore/` is snapshotted into the bundle.
- `llcore/` is also copied at the bundle root so Kaggle runtime import-path
  differences do not strand the package behind `src/`.
- `LICENSE` + `NOTICE` are copied in for redistribution review.
- `runner.py` writes compare artifacts under `artifacts/`.

When `--dataset-source` is used, the kernel bundle switches to a dataset-backed
layout:

- runtime inputs move under `dataset_payload/`
- `kernel-metadata.json.dataset_sources` points at the declared dataset slug
- `runner.py` reads from `/kaggle/input/<mount>/...` (or a local override during
  preflight smoke)

Publishing the bundle via `kaggle kernels push -p ...` remains a human-gated step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import zipfile
from dataclasses import asdict
from pathlib import Path
from pathlib import PurePosixPath
sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "src")))

from llcore.lm_compare_config import CompareConfig


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_KERNEL_ID = "furusekazufumi/llcore-lm-compare"
DEFAULT_DATASET_SOURCE = "furusekazufumi/llcore-lm-compare-support"
RUNNER_NAME = "runner.py"
KAGGLEIGNORE_NAME = ".kaggleignore"
DATASET_PAYLOAD_DIRNAME = "dataset_payload"
DATASET_METADATA_NAME = "dataset-metadata.json"
DATASET_PAYLOAD_MANIFEST_NAME = "dataset_payload_manifest.json"
DATASET_SRC_ARCHIVE_NAME = "src_llcore.zip"
DATASET_PKG_ARCHIVE_NAME = "pkg_llcore.zip"
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
_DATASET_TOPLEVEL_COPIED_FILE_PATHS = {
    "metadata": "kernel-metadata.json",
    "runner": RUNNER_NAME,
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
_KERNEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*$")

RUNNER_TEMPLATE_EMBEDDED = """# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for candidate in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(candidate))

from llcore.lm.compare import CompareConfig, compare_on_text


def main() -> int:
    payload = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    corpus_path = ROOT / "input_corpus.txt"
    out_path = ROOT / "artifacts" / "lm_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = corpus_path.read_text(encoding="utf-8")
    cfg = CompareConfig(**payload["compare_config"])
    result = compare_on_text(text, cfg=cfg, out_path=out_path)
    reports = result["reports"]
    print(
        "[compare] wrote",
        out_path,
        f"gpt_ppl={reports['gpt']['model_ppl']:.4f}",
        f"recurrent_ppl={reports['recurrent']['model_ppl']:.4f}",
        f"rwkv_ppl={reports['rwkv']['model_ppl']:.4f}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""

RUNNER_TEMPLATE_DATASET = """# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path
from pathlib import PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parent
env_data_root = os.environ.get("LLCORE_KAGGLE_DATA_ROOT")
if env_data_root:
    DATA_ROOT = Path(env_data_root)
else:
    DATA_ROOT = Path("/kaggle/input") / "{dataset_mount_name}"


EXPECTED_CONFIG_SHA256 = "{config_sha256}"
EXPECTED_CORPUS_SHA256 = "{corpus_sha256}"
EXPECTED_SRC_ARCHIVE_SHA256 = "{src_archive_sha256}"
EXPECTED_PKG_ARCHIVE_SHA256 = "{pkg_archive_sha256}"
SRC_ARCHIVE_NAME = "{src_archive_name}"
PKG_ARCHIVE_NAME = "{pkg_archive_name}"


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    with zipfile.ZipFile(archive_path) as zf:
        infos = zf.infolist()
        if len(infos) > max_entries:
            raise RuntimeError(
                f"archive has too many entries: {{archive_path}} ({{len(infos)}} > {{max_entries}})"
            )
        for info in infos:
            member_name = info.filename.replace("\\\\", "/")
            if not member_name:
                raise RuntimeError(f"archive contains an empty member name: {{archive_path}}")
            if member_name in seen_members:
                raise RuntimeError(f"archive contains a duplicate member name: {{member_name}}")
            seen_members.add(member_name)
            if member_name.startswith("/") or member_name.startswith("../"):
                raise RuntimeError(f"archive member escapes extraction root: {{member_name}}")
            member_path = PurePosixPath(member_name)
            if any(part in {{"", ".", ".."}} for part in member_path.parts):
                raise RuntimeError(f"archive member is not a safe relative path: {{member_name}}")
            if not member_name.startswith(expected_prefix):
                raise RuntimeError(
                    f"archive member does not stay under expected prefix {{expected_prefix!r}}: {{member_name}}"
                )
            mode = info.external_attr >> 16
            if stat.S_IFMT(mode) == stat.S_IFLNK:
                raise RuntimeError(f"symlinks are not allowed in dataset payload archive: {{member_name}}")
            target = (dest_root / member_path).resolve()
            if not target.is_relative_to(dest_root.resolve()):
                raise RuntimeError(f"archive member resolves outside extraction root: {{member_name}}")
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            total_uncompressed += info.file_size
            if total_uncompressed > max_uncompressed_bytes:
                raise RuntimeError(
                    "archive exceeds extracted size budget: "
                    f"{{archive_path}} ({{total_uncompressed}} > {{max_uncompressed_bytes}})"
                )
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted_files += 1
    if extracted_files < 1:
        raise RuntimeError(
            f"archive did not extract any files under expected prefix: {{archive_path}}"
        )


def _prepare_import_tree() -> None:
    extract_root = ROOT / ".dataset_payload_unpack"
    shutil.rmtree(extract_root, ignore_errors=True)
    extract_root.mkdir(parents=True, exist_ok=True)
    src_archive = DATA_ROOT / SRC_ARCHIVE_NAME
    pkg_archive = DATA_ROOT / PKG_ARCHIVE_NAME
    if _sha256_text(src_archive) != EXPECTED_SRC_ARCHIVE_SHA256:
        raise RuntimeError("dataset src archive sha256 mismatch")
    if _sha256_text(pkg_archive) != EXPECTED_PKG_ARCHIVE_SHA256:
        raise RuntimeError("dataset pkg archive sha256 mismatch")
    _safe_extract_zip(src_archive, extract_root, expected_prefix="src/llcore/")
    _safe_extract_zip(pkg_archive, extract_root, expected_prefix="llcore/")
    for candidate in (extract_root, extract_root / "src"):
        sys.path.insert(0, str(candidate))


def main() -> int:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(
            "dataset mount not found: "
            f"{{DATA_ROOT}} (expected Kaggle dataset mount '{{DATA_ROOT.name}}')"
        )
    config_path = DATA_ROOT / "config.json"
    corpus_path = DATA_ROOT / "input_corpus.txt"
    if not config_path.is_file():
        raise FileNotFoundError(f"dataset config missing: {{config_path}}")
    if not corpus_path.is_file():
        raise FileNotFoundError(f"dataset corpus missing: {{corpus_path}}")
    actual_config_sha256 = _sha256_text(config_path)
    if actual_config_sha256 != EXPECTED_CONFIG_SHA256:
        raise RuntimeError(
            "dataset config sha256 mismatch: "
            f"expected {{EXPECTED_CONFIG_SHA256}} got {{actual_config_sha256}}"
        )
    actual_corpus_sha256 = _sha256_text(corpus_path)
    if actual_corpus_sha256 != EXPECTED_CORPUS_SHA256:
        raise RuntimeError(
            "dataset corpus sha256 mismatch: "
            f"expected {{EXPECTED_CORPUS_SHA256}} got {{actual_corpus_sha256}}"
        )
    _prepare_import_tree()
    from llcore.lm.compare import CompareConfig, compare_on_text

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    out_path = ROOT / "artifacts" / "lm_compare.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = corpus_path.read_text(encoding="utf-8")
    cfg = CompareConfig(**payload["compare_config"])
    result = compare_on_text(text, cfg=cfg, out_path=out_path)
    reports = result["reports"]
    print(
        "[compare] wrote",
        out_path,
        f"gpt_ppl={{reports['gpt']['model_ppl']:.4f}}",
        f"recurrent_ppl={{reports['recurrent']['model_ppl']:.4f}}",
        f"rwkv_ppl={{reports['rwkv']['model_ppl']:.4f}}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


README_TEMPLATE = """# llcore lm.compare Kaggle bundle

This folder is a local, deterministic Kaggle kernel bundle for `llcore.lm.compare`.

- Corpus input is pinned via `input_corpus.txt`.
- Compare config is pinned via `config.json`.
- Code is snapshotted under both `src/llcore/` and bundle-root `llcore/`.
- Metadata defaults are safe: private + internet disabled + GPU disabled.

## Local contents

- `kernel-metadata.json`: Kaggle kernel metadata
- `runner.py`: kernel entrypoint
- `config.json`: pinned compare configuration and corpus metadata
- `input_corpus.txt`: copied UTF-8 corpus
- `LICENSE` / `NOTICE`: distribution documents included in the candidate bundle
- `artifacts/`: expected output directory (`lm_compare.json`, `.md`, `.svg`)

## Human-gated publish

Do not push automatically. When ready and approved:

```powershell
kaggle kernels push -p <this_dir>
```

Before pushing with `--enable-gpu`, validate the emitted Kaggle metadata keys
against the live Kaggle schema/CLI. `machine_shape` is only emitted when GPU is
requested, and this builder does not contact Kaggle to verify metadata support.

This bundle also relies on Kaggle's preinstalled `torch`; no torch version is
pinned here.
"""


README_TEMPLATE_DATASET = """# llcore lm.compare Kaggle bundle

This folder is a local, deterministic Kaggle script-kernel bundle for
`llcore.lm.compare`.

- `runner.py` is the only kernel source file Kaggle will execute for
  `kernel_type: "script"`.
- Runtime inputs live under `dataset_payload/` and must be published as a Kaggle
  Dataset referenced by `kernel-metadata.json.dataset_sources`.
- `dataset_payload/` carries `src_llcore.zip` and `pkg_llcore.zip`; `runner.py`
  safely extracts them at runtime before importing `llcore`.
- `.kaggleignore` excludes `dataset_payload/` from `kaggle kernels push` so the
  kernel and dataset payload are not uploaded twice.
- Local smoke uses `LLCORE_KAGGLE_DATA_ROOT` to simulate `/kaggle/input/...`.
- Default ids assume the `furusekazufumi` Kaggle account; override
  `--kernel-id` / `--dataset-source` when publishing from another owner.

## Local contents

- `kernel-metadata.json`: Kaggle kernel metadata
- `runner.py`: kernel entrypoint
- `dataset_payload/`: local dataset candidate (`config.json`, `input_corpus.txt`,
  `src_llcore.zip`, `pkg_llcore.zip`, `LICENSE`, `NOTICE`,
  `dataset-metadata.json`)
- `artifacts/`: expected output directory (`lm_compare.json`, `.md`, `.svg`)

## Human-gated publish

Do not push automatically. Publish order is:

```powershell
kaggle datasets create -p <this_dir>/dataset_payload --dir-mode zip
kaggle datasets version -p <this_dir>/dataset_payload --dir-mode zip -m "update dataset payload"
kaggle kernels push -p <this_dir>
```

Use `create` for the first publish of a dataset slug and `version` for updates.
`--dir-mode zip` remains part of the publish recipe, but the runtime now relies
on uploaded files plus its own safe extraction step rather than Kaggle exposing
expanded package directories directly.
"""


def _sha256_text(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle_notice_text() -> str:
    notice = (REPO_ROOT / "NOTICE").read_text(encoding="utf-8")
    # This sanitize step relies on an exact root-NOTICE sentence match. If the
    # upstream NOTICE wording changes, the post-check below must continue to
    # fail closed until this replacement target is updated.
    sanitized = notice.replace(
        "Commercial licenses are also available; see LICENSE-COMMERCIAL.",
        "Commercial licenses are also available separately from this bundle.",
    )
    if "LICENSE-COMMERCIAL" in sanitized or "Commercial dual-license" in sanitized:
        raise ValueError("NOTICE sanitize failed: commercial-license wording remains after rewrite")
    return sanitized


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


def _write_source_archive(source_root: Path, archive_path: Path, *, prefix: str) -> str:
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in source_root.rglob("*") if p.is_file() and not _is_ignored_source_path(p)):
            rel = path.relative_to(source_root).as_posix()
            info = zipfile.ZipInfo(f"{prefix}/{rel}")
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(info, path.read_bytes())
    return _sha256_text(archive_path)


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _dataset_mount_name(dataset_source: str) -> str:
    return dataset_source.split("/", 1)[1]


def _render_runner(
    dataset_source: str | None,
    *,
    config_sha256: str | None = None,
    corpus_sha256: str | None = None,
    src_archive_sha256: str | None = None,
    pkg_archive_sha256: str | None = None,
) -> str:
    if dataset_source is None:
        return RUNNER_TEMPLATE_EMBEDDED
    if (
        config_sha256 is None
        or corpus_sha256 is None
        or src_archive_sha256 is None
        or pkg_archive_sha256 is None
    ):
        raise ValueError(
            "dataset runner rendering requires config/corpus/archive sha256 values"
        )
    return RUNNER_TEMPLATE_DATASET.format(
        dataset_mount_name=_dataset_mount_name(dataset_source),
        config_sha256=config_sha256,
        corpus_sha256=corpus_sha256,
        src_archive_sha256=src_archive_sha256,
        pkg_archive_sha256=pkg_archive_sha256,
        src_archive_name=DATASET_SRC_ARCHIVE_NAME,
        pkg_archive_name=DATASET_PKG_ARCHIVE_NAME,
    )


def _render_readme(dataset_source: str | None) -> str:
    if dataset_source is None:
        return README_TEMPLATE
    return README_TEMPLATE_DATASET


def _load_json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


def _ignore_source_noise(_src: str, names: list[str]) -> set[str]:
    ignored: set[str] = set()
    for name in names:
        if _is_ignored_source_path(Path(name)):
            ignored.add(name)
    return ignored


def _build_kernel_metadata(
    *,
    kernel_id: str,
    title: str,
    enable_gpu: bool,
    enable_internet: bool,
    is_private: bool,
    machine_shape: str | None,
    dataset_sources: list[str],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "id": kernel_id,
        "title": title,
        "code_file": RUNNER_NAME,
        "language": "python",
        "kernel_type": "script",
        "is_private": "true" if is_private else "false",
        "enable_gpu": "true" if enable_gpu else "false",
        "enable_tpu": "false",
        "enable_internet": "true" if enable_internet else "false",
        "dataset_sources": dataset_sources,
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    if enable_gpu and machine_shape:
        payload["machine_shape"] = machine_shape
    return payload


def _ensure_safe_bundle_dir(bundle_dir: Path) -> Path:
    resolved = bundle_dir.resolve()
    repo_root = REPO_ROOT.resolve()
    src_root = SRC_ROOT.resolve()
    llcore_src = (SRC_ROOT / "llcore").resolve()
    bundle_src_target = (resolved / "src" / "llcore").resolve()
    if resolved == repo_root or repo_root.is_relative_to(resolved):
        raise ValueError(f"bundle_dir overlaps repository root and is unsafe: {resolved}")
    if resolved.is_relative_to(repo_root):
        raise ValueError(
            "bundle_dir inside the git working tree is discouraged; use a repo-external path: "
            f"{resolved}"
        )
    if resolved == src_root or resolved.is_relative_to(src_root):
        raise ValueError(f"bundle_dir inside repo src tree is unsafe: {resolved}")
    if (
        bundle_src_target == llcore_src
        or bundle_src_target.is_relative_to(llcore_src)
        or llcore_src.is_relative_to(bundle_src_target)
    ):
        raise ValueError(f"bundle_dir would overlap repo source snapshot target: {resolved}")
    return resolved


def _is_builder_bundle_dir(bundle_dir: Path) -> bool:
    # This is a cheap recognizer for our own bundle layout, not a full manifest
    # integrity verifier. We intentionally avoid destructive replacement unless
    # the directory already looks like one of our generated bundles.
    manifest_path = bundle_dir / "bundle_manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = _load_json_object(manifest_path)
    runner = manifest.get("runner")
    kernel_id = manifest.get("kernel_id")
    copied_files = manifest.get("copied_files")
    source_sha256 = manifest.get("source_sha256")
    data_mode = manifest.get("data_mode", "embedded")
    expected_keys = EMBEDDED_COPIED_FILE_KEYS if data_mode == "embedded" else DATASET_COPIED_FILE_KEYS
    if not (
        runner == RUNNER_NAME
        and isinstance(kernel_id, str)
        and bool(kernel_id)
        and isinstance(copied_files, dict)
        and set(copied_files) == set(expected_keys)
    ):
        return False
    if data_mode == "embedded":
        if not (isinstance(source_sha256, str) and len(source_sha256) == 64):
            return False
    elif data_mode == "dataset":
        dataset_payload_rel = manifest.get("dataset_payload_rel")
        dataset_payload_manifest_sha256 = manifest.get("dataset_payload_manifest_sha256")
        if not (
            isinstance(dataset_payload_rel, str)
            and dataset_payload_rel
            and isinstance(dataset_payload_manifest_sha256, str)
            and len(dataset_payload_manifest_sha256) == 64
        ):
            return False
        if not isinstance(copied_files, dict):
            return False
        if any(
            copied_files.get(key) != expected
            for key, expected in _DATASET_TOPLEVEL_COPIED_FILE_PATHS.items()
        ):
            return False
    else:
        return False
    required_paths = (
        "kernel-metadata.json",
        "LICENSE",
        "NOTICE",
        "README.md",
        RUNNER_NAME,
    )
    if not all((bundle_dir / rel).exists() for rel in required_paths):
        return False
    if data_mode == "embedded":
        return all((bundle_dir / rel).exists() for rel in ("config.json", "input_corpus.txt", "src/llcore", "llcore"))
    kaggleignore = bundle_dir / KAGGLEIGNORE_NAME
    if not kaggleignore.is_file():
        return False
    if f"{DATASET_PAYLOAD_DIRNAME}/" not in kaggleignore.read_text(encoding="utf-8").splitlines():
        return False
    return all(
        (bundle_dir / rel).exists()
        for rel in (
            f"{DATASET_PAYLOAD_DIRNAME}/{DATASET_METADATA_NAME}",
            f"{DATASET_PAYLOAD_DIRNAME}/{DATASET_PAYLOAD_MANIFEST_NAME}",
            f"{DATASET_PAYLOAD_DIRNAME}/config.json",
            f"{DATASET_PAYLOAD_DIRNAME}/input_corpus.txt",
            f"{DATASET_PAYLOAD_DIRNAME}/{DATASET_SRC_ARCHIVE_NAME}",
            f"{DATASET_PAYLOAD_DIRNAME}/{DATASET_PKG_ARCHIVE_NAME}",
            f"{DATASET_PAYLOAD_DIRNAME}/LICENSE",
            f"{DATASET_PAYLOAD_DIRNAME}/NOTICE",
        )
    )


def _prepare_bundle_target(bundle_dir: Path) -> tuple[Path, Path]:
    if not bundle_dir.exists():
        parent = bundle_dir.parent
    else:
        parent = bundle_dir.parent
        if bundle_dir.is_file():
            raise ValueError(f"bundle_dir points to a file, not a directory: {bundle_dir}")
        entries = list(bundle_dir.iterdir())
        if not entries:
            raise ValueError(
                "bundle_dir already exists and is empty; refusing to delete a non-bundle directory: "
                f"{bundle_dir}"
            )
        if not _is_builder_bundle_dir(bundle_dir):
            raise ValueError(
                "bundle_dir already exists and is not a recognized Kaggle bundle; "
                f"refusing to delete: {bundle_dir}"
            )
    parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(prefix=f"{bundle_dir.name}.tmp-", dir=str(parent))
    )
    staging_dir = temp_root / bundle_dir.name
    staging_dir.mkdir(parents=True, exist_ok=True)
    return temp_root, staging_dir


def build_bundle(
    *,
    bundle_dir: Path,
    corpus_file: Path,
    kernel_id: str,
    title: str,
    machine_shape: str | None,
    enable_gpu: bool,
    enable_internet: bool,
    is_private: bool,
    dataset_source: str | None = None,
    cfg: CompareConfig,
) -> dict[str, object]:
    """Build a local Kaggle bundle and return manifest-level summary data.

    `enable_gpu` and `machine_shape` must agree. Mixed inputs that were
    previously tolerated via direct function calls are now rejected
    fail-closed with `ValueError`.
    """
    if not corpus_file.is_file():
        raise ValueError(f"corpus file does not exist: {corpus_file}")
    if not kernel_id.strip():
        raise ValueError("kernel_id must be a non-empty string")
    if not _KERNEL_ID_RE.fullmatch(kernel_id):
        raise ValueError("kernel_id must match Kaggle owner/slug form (lowercase alnum/hyphen)")
    if not title.strip():
        raise ValueError("title must be a non-empty string")
    if dataset_source is not None and not _KERNEL_ID_RE.fullmatch(dataset_source):
        raise ValueError("dataset_source must match Kaggle owner/slug form (lowercase alnum/hyphen)")
    if enable_gpu and not machine_shape:
        raise ValueError("machine_shape is required when enable_gpu is true")
    if not enable_gpu and machine_shape is not None:
        raise ValueError("machine_shape must be omitted when enable_gpu is false")
    bundle_dir = _ensure_safe_bundle_dir(bundle_dir)
    temp_root, staging_dir = _prepare_bundle_target(bundle_dir)
    backup_dir = temp_root / f"{bundle_dir.name}.backup"
    try:
        dataset_config_sha256: str | None = None
        dataset_corpus_sha256: str | None = None
        (staging_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO_ROOT / "LICENSE", staging_dir / "LICENSE")
        (staging_dir / "NOTICE").write_text(_bundle_notice_text(), encoding="utf-8")
        if dataset_source is None:
            corpus_target = staging_dir / "input_corpus.txt"
            shutil.copyfile(corpus_file, corpus_target)
            src_target = staging_dir / "src" / "llcore"
            shutil.copytree(SRC_ROOT / "llcore", src_target, ignore=_ignore_source_noise)
            pkg_target = staging_dir / "llcore"
            shutil.copytree(SRC_ROOT / "llcore", pkg_target, ignore=_ignore_source_noise)
            config_payload = {
                "compare_config": asdict(cfg),
                "corpus_file_name": corpus_file.name,
                "corpus_sha256": _sha256_text(corpus_file),
            }
            _write_json(staging_dir / "config.json", config_payload)
        else:
            dataset_payload_dir = staging_dir / DATASET_PAYLOAD_DIRNAME
            dataset_payload_dir.mkdir(parents=True, exist_ok=True)
            corpus_target = dataset_payload_dir / "input_corpus.txt"
            shutil.copyfile(corpus_file, corpus_target)
            shutil.copyfile(REPO_ROOT / "LICENSE", dataset_payload_dir / "LICENSE")
            (dataset_payload_dir / "NOTICE").write_text(_bundle_notice_text(), encoding="utf-8")
            source_tree = SRC_ROOT / "llcore"
            src_archive_target = dataset_payload_dir / DATASET_SRC_ARCHIVE_NAME
            pkg_archive_target = dataset_payload_dir / DATASET_PKG_ARCHIVE_NAME
            source_sha256 = _sha256_tree(source_tree)
            src_archive_sha256 = _write_source_archive(source_tree, src_archive_target, prefix="src/llcore")
            pkg_archive_sha256 = _write_source_archive(source_tree, pkg_archive_target, prefix="llcore")
            config_payload = {
                "compare_config": asdict(cfg),
                "corpus_file_name": corpus_file.name,
                "corpus_sha256": _sha256_text(corpus_file),
            }
            _write_json(dataset_payload_dir / "config.json", config_payload)
            dataset_metadata_payload = {
                "title": _dataset_mount_name(dataset_source),
                "id": dataset_source,
                "licenses": [{"name": "other"}],
            }
            _write_json(dataset_payload_dir / DATASET_METADATA_NAME, dataset_metadata_payload)
            dataset_payload_manifest = {
                "dataset_source": dataset_source,
                "dataset_mount_name": _dataset_mount_name(dataset_source),
                "copied_files": dict(_DATASET_PAYLOAD_COPIED_FILE_PATHS),
                "corpus_sha256": config_payload["corpus_sha256"],
                "source_sha256": source_sha256,
                "src_archive_sha256": src_archive_sha256,
                "pkg_archive_sha256": pkg_archive_sha256,
                "config_sha256": _sha256_text(dataset_payload_dir / "config.json"),
                "license_sha256": _sha256_text(dataset_payload_dir / "LICENSE"),
                "notice_sha256": _sha256_text(dataset_payload_dir / "NOTICE"),
                "dataset_metadata_sha256": _sha256_text(dataset_payload_dir / DATASET_METADATA_NAME),
            }
            _write_json(dataset_payload_dir / DATASET_PAYLOAD_MANIFEST_NAME, dataset_payload_manifest)
            (staging_dir / KAGGLEIGNORE_NAME).write_text(f"{DATASET_PAYLOAD_DIRNAME}/\n", encoding="utf-8")
            dataset_config_sha256 = str(dataset_payload_manifest["config_sha256"])
            dataset_corpus_sha256 = str(config_payload["corpus_sha256"])
        (staging_dir / RUNNER_NAME).write_text(
            _render_runner(
                dataset_source,
                config_sha256=dataset_config_sha256,
                corpus_sha256=dataset_corpus_sha256,
                src_archive_sha256=(
                    str(dataset_payload_manifest["src_archive_sha256"]) if dataset_source is not None else None
                ),
                pkg_archive_sha256=(
                    str(dataset_payload_manifest["pkg_archive_sha256"]) if dataset_source is not None else None
                ),
            ),
            encoding="utf-8",
        )
        (staging_dir / "README.md").write_text(_render_readme(dataset_source), encoding="utf-8")
        kernel_metadata = _build_kernel_metadata(
            kernel_id=kernel_id,
            title=title,
            enable_gpu=enable_gpu,
            enable_internet=enable_internet,
            is_private=is_private,
            machine_shape=machine_shape,
            dataset_sources=[] if dataset_source is None else [dataset_source],
        )
        _write_json(staging_dir / "kernel-metadata.json", kernel_metadata)
        runner_sha256 = _sha256_text(staging_dir / RUNNER_NAME)
        manifest_payload: dict[str, object] = {
            "kernel_id": kernel_id,
            "title": title,
            "machine_shape": machine_shape,
            "is_private": kernel_metadata["is_private"],
            "enable_internet": kernel_metadata["enable_internet"],
            "enable_gpu": kernel_metadata["enable_gpu"],
            "enable_tpu": kernel_metadata["enable_tpu"],
            "runner": RUNNER_NAME,
            "runner_sha256": runner_sha256,
            "license_sha256": _sha256_text(staging_dir / "LICENSE"),
            "notice_sha256": _sha256_text(staging_dir / "NOTICE"),
        }
        summary_source_sha256: str | None = None
        summary_config_sha256: str | None = None
        if dataset_source is None:
            source_sha256 = _sha256_tree(src_target)
            config_sha256 = _sha256_text(staging_dir / "config.json")
            summary_source_sha256 = source_sha256
            summary_config_sha256 = config_sha256
            manifest_payload.update(
                {
                    "data_mode": "embedded",
                    "copied_files": {
                        "corpus": "input_corpus.txt",
                        "config": "config.json",
                        "metadata": "kernel-metadata.json",
                        "src_llcore": "src/llcore",
                        "pkg_llcore": "llcore",
                        "license": "LICENSE",
                        "notice": "NOTICE",
                    },
                    "corpus_sha256": config_payload["corpus_sha256"],
                    "source_sha256": source_sha256,
                    "config_sha256": config_sha256,
                }
            )
        else:
            dataset_payload_dir = staging_dir / DATASET_PAYLOAD_DIRNAME
            dataset_payload_manifest_path = dataset_payload_dir / DATASET_PAYLOAD_MANIFEST_NAME
            manifest_payload.update(
                {
                    "data_mode": "dataset",
                    "dataset_source": dataset_source,
                    "dataset_mount_name": _dataset_mount_name(dataset_source),
                    "dataset_payload_rel": DATASET_PAYLOAD_DIRNAME,
                    "dataset_payload_manifest_sha256": _sha256_text(dataset_payload_manifest_path),
                    "copied_files": {
                        "metadata": "kernel-metadata.json",
                        "runner": RUNNER_NAME,
                        "license": "LICENSE",
                        "notice": "NOTICE",
                        "dataset_payload": DATASET_PAYLOAD_DIRNAME,
                    },
                }
            )
        _write_json(staging_dir / "bundle_manifest.json", manifest_payload)

        if bundle_dir.exists():
            os.replace(bundle_dir, backup_dir)
        try:
            os.replace(staging_dir, bundle_dir)
        except Exception:
            if backup_dir.exists() and not bundle_dir.exists():
                os.replace(backup_dir, bundle_dir)
            raise
        shutil.rmtree(backup_dir, ignore_errors=True)
        shutil.rmtree(temp_root, ignore_errors=True)
        return {
            "bundle_dir": str(bundle_dir),
            "kernel_id": kernel_id,
            "title": title,
            "machine_shape": machine_shape,
            "corpus_sha256": config_payload["corpus_sha256"],
            "runner_sha256": runner_sha256,
            "data_mode": "embedded" if dataset_source is None else "dataset",
            "dataset_source": dataset_source,
            "source_sha256": summary_source_sha256,
            "config_sha256": summary_config_sha256,
        }
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build a local Kaggle bundle for llcore.lm.compare. "
            "This does not push or publish anything."
        )
    )
    ap.add_argument("--bundle-dir", required=True, help="output directory for the Kaggle bundle")
    ap.add_argument("--corpus-file", required=True, help="UTF-8 corpus file to embed")
    ap.add_argument("--kernel-id", default=DEFAULT_KERNEL_ID)
    ap.add_argument(
        "--dataset-source",
        default=None,
        help=(
            "optional Kaggle dataset owner/slug for runtime inputs; when set, build "
            "a dataset-backed kernel bundle with local payload under dataset_payload/"
        ),
    )
    ap.add_argument("--title", default="llcore-lm-compare")
    ap.add_argument("--machine-shape", default="NvidiaTeslaT4")
    ap.add_argument("--enable-gpu", action="store_true", help="emit GPU-enabled Kaggle metadata")
    ap.add_argument(
        "--enable-internet",
        action="store_true",
        help="emit internet-enabled Kaggle metadata",
    )
    ap.add_argument("--public", action="store_true", help="emit public Kaggle metadata")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--state-size", type=int, default=64)
    ap.add_argument("--max-iters", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--eval-iters", type=int, default=4)
    ap.add_argument("--throughput-new-tokens", type=int, default=16)
    ap.add_argument("--throughput-repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    return ap


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        cfg = CompareConfig(
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            state_size=args.state_size,
            max_iters=args.max_iters,
            batch_size=args.batch_size,
            eval_iters=args.eval_iters,
            throughput_new_tokens=args.throughput_new_tokens,
            throughput_repeats=args.throughput_repeats,
            seed=args.seed,
        )
        summary = build_bundle(
            bundle_dir=Path(args.bundle_dir),
            corpus_file=Path(args.corpus_file),
            kernel_id=args.kernel_id,
            title=args.title,
            machine_shape=args.machine_shape if args.enable_gpu else None,
            enable_gpu=args.enable_gpu,
            enable_internet=args.enable_internet,
            is_private=not args.public,
            dataset_source=args.dataset_source,
            cfg=cfg,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        "[kaggle-bundle]",
        f"dir={summary['bundle_dir']}",
        f"kernel_id={summary['kernel_id']}",
        f"mode={summary['data_mode']}",
        f"corpus_sha256={str(summary['corpus_sha256'])[:12]}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
