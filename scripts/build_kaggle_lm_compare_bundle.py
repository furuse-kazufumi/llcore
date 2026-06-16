# SPDX-License-Identifier: Apache-2.0
"""Build a local Kaggle bundle for llcore.lm.compare without pushing it.

The generated directory is intentionally self-contained:

- `input_corpus.txt` is copied in so the run does not depend on remote fetches.
- `src/llcore/` is snapshotted into the bundle.
- `LICENSE` + `NOTICE` are copied in for redistribution review.
- `runner.py` reads `config.json` and writes compare artifacts under `artifacts/`.

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
from dataclasses import asdict
from pathlib import Path
from typing import cast

sys.path.insert(0, str((Path(__file__).resolve().parent.parent / "src")))

from llcore.lm.compare import CompareConfig  # type: ignore[import-untyped]


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC_ROOT = REPO_ROOT / "src"
DEFAULT_KERNEL_ID = "furusekazufumi/llcore-lm-compare"
RUNNER_NAME = "runner.py"
REQUIRED_COPIED_FILE_KEYS = ("corpus", "config", "metadata", "src_llcore", "license", "notice")
_KERNEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*/[a-z0-9][a-z0-9-]*$")

RUNNER_TEMPLATE = """# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

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


README_TEMPLATE = """# llcore lm.compare Kaggle bundle

This folder is a local, deterministic Kaggle kernel bundle for `llcore.lm.compare`.

- Corpus input is pinned via `input_corpus.txt`.
- Compare config is pinned via `config.json`.
- Code is snapshotted under `src/llcore/`.
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
    for path in sorted(p for p in root.rglob("*") if p.is_file() and not _is_ignored_source_path(p)):
        rel = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(rel)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


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
        "dataset_sources": [],
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
    if not (
        runner == RUNNER_NAME
        and isinstance(kernel_id, str)
        and bool(kernel_id)
        and isinstance(copied_files, dict)
        and set(copied_files) == set(REQUIRED_COPIED_FILE_KEYS)
        and isinstance(source_sha256, str)
        and len(source_sha256) == 64
    ):
        return False
    required_paths = (
        "kernel-metadata.json",
        "config.json",
        "input_corpus.txt",
        "LICENSE",
        "NOTICE",
        "README.md",
        RUNNER_NAME,
        "src/llcore",
    )
    return (
        all((bundle_dir / rel).exists() for rel in required_paths)
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
    if enable_gpu and not machine_shape:
        raise ValueError("machine_shape is required when enable_gpu is true")
    if not enable_gpu and machine_shape is not None:
        raise ValueError("machine_shape must be omitted when enable_gpu is false")
    bundle_dir = _ensure_safe_bundle_dir(bundle_dir)
    temp_root, staging_dir = _prepare_bundle_target(bundle_dir)
    backup_dir = temp_root / f"{bundle_dir.name}.backup"
    try:
        (staging_dir / "artifacts").mkdir(parents=True, exist_ok=True)
        corpus_target = staging_dir / "input_corpus.txt"
        shutil.copyfile(corpus_file, corpus_target)
        shutil.copyfile(REPO_ROOT / "LICENSE", staging_dir / "LICENSE")
        (staging_dir / "NOTICE").write_text(_bundle_notice_text(), encoding="utf-8")
        src_target = staging_dir / "src" / "llcore"
        shutil.copytree(SRC_ROOT / "llcore", src_target, ignore=_ignore_source_noise)
        (staging_dir / RUNNER_NAME).write_text(RUNNER_TEMPLATE, encoding="utf-8")
        (staging_dir / "README.md").write_text(README_TEMPLATE, encoding="utf-8")
        kernel_metadata = _build_kernel_metadata(
            kernel_id=kernel_id,
            title=title,
            enable_gpu=enable_gpu,
            enable_internet=enable_internet,
            is_private=is_private,
            machine_shape=machine_shape,
        )
        _write_json(staging_dir / "kernel-metadata.json", kernel_metadata)
        source_sha256 = _sha256_tree(src_target)
        config_payload = {
            "compare_config": asdict(cfg),
            "corpus_file_name": corpus_file.name,
            "corpus_sha256": _sha256_text(corpus_file),
        }
        _write_json(staging_dir / "config.json", config_payload)
        runner_sha256 = _sha256_text(staging_dir / RUNNER_NAME)
        config_sha256 = _sha256_text(staging_dir / "config.json")
        manifest_payload = {
            "kernel_id": kernel_id,
            "title": title,
            "machine_shape": machine_shape,
            "is_private": kernel_metadata["is_private"],
            "enable_internet": kernel_metadata["enable_internet"],
            "enable_gpu": kernel_metadata["enable_gpu"],
            "enable_tpu": kernel_metadata["enable_tpu"],
            "runner": RUNNER_NAME,
            "copied_files": {
                "corpus": "input_corpus.txt",
                "config": "config.json",
                "metadata": "kernel-metadata.json",
                "src_llcore": "src/llcore",
                "license": "LICENSE",
                "notice": "NOTICE",
            },
            "corpus_sha256": config_payload["corpus_sha256"],
            "source_sha256": source_sha256,
            "runner_sha256": runner_sha256,
            "config_sha256": config_sha256,
            "license_sha256": _sha256_text(staging_dir / "LICENSE"),
            "notice_sha256": _sha256_text(staging_dir / "NOTICE"),
        }
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
            "source_sha256": source_sha256,
            "runner_sha256": runner_sha256,
            "config_sha256": config_sha256,
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
            cfg=cfg,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary_map = cast(dict[str, str], summary)
    print(
        "[kaggle-bundle]",
        f"dir={summary_map['bundle_dir']}",
        f"kernel_id={summary_map['kernel_id']}",
        f"corpus_sha256={summary_map['corpus_sha256'][:12]}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
