# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _load_builder() -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_kaggle_lm_compare_bundle.py"
    spec = importlib.util.spec_from_file_location("build_kaggle_lm_compare_bundle", script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_script(script_name: str) -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_bundle_writes_metadata_and_copies_inputs(tmp_path: Path) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--kernel-id",
            "furusekazufumi/test-lm-compare",
            "--title",
            "test-lm-compare",
            "--block-size",
            "96",
            "--max-iters",
            "40",
        ]
    )

    assert rc == 0
    metadata = json.loads((bundle_dir / "kernel-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "furusekazufumi/test-lm-compare"
    assert metadata["code_file"] == "runner.py"
    assert metadata["enable_gpu"] == "false"
    assert metadata["enable_tpu"] == "false"
    assert metadata["enable_internet"] == "false"
    assert metadata["is_private"] == "true"
    assert "machine_shape" not in metadata
    config = json.loads((bundle_dir / "config.json").read_text(encoding="utf-8"))
    assert config["compare_config"]["block_size"] == 96
    assert config["compare_config"]["max_iters"] == 40
    assert config["corpus_file_name"] == "corpus.txt"
    assert len(config["corpus_sha256"]) == 64
    manifest = json.loads((bundle_dir / "bundle_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["source_sha256"]) == 64
    assert len(manifest["runner_sha256"]) == 64
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["license_sha256"]) == 64
    assert len(manifest["notice_sha256"]) == 64
    assert manifest["is_private"] == "true"
    assert manifest["enable_internet"] == "false"
    assert manifest["enable_gpu"] == "false"
    assert manifest["enable_tpu"] == "false"
    assert manifest["title"] == "test-lm-compare"
    assert set(manifest["copied_files"]) == {
        "corpus",
        "config",
        "metadata",
        "src_llcore",
        "pkg_llcore",
        "license",
        "notice",
    }
    assert (bundle_dir / "input_corpus.txt").read_text(encoding="utf-8") == "abcabc\n"
    assert (bundle_dir / "LICENSE").is_file()
    assert (bundle_dir / "NOTICE").is_file()
    assert "LICENSE-COMMERCIAL" not in (bundle_dir / "NOTICE").read_text(encoding="utf-8")
    assert (bundle_dir / "runner.py").is_file()
    assert (bundle_dir / "README.md").is_file()
    assert (bundle_dir / "bundle_manifest.json").is_file()
    assert (bundle_dir / "src" / "llcore" / "lm" / "compare.py").is_file()
    assert (bundle_dir / "llcore" / "lm" / "compare.py").is_file()
    assert not (bundle_dir / "src" / "llcore" / "__pycache__").exists()
    assert not (bundle_dir / "llcore" / "__pycache__").exists()


def test_build_bundle_rejects_missing_corpus(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _load_builder()
    missing = tmp_path / "missing.txt"
    bundle_dir = tmp_path / "bundle"

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(missing),
        ]
    )

    assert rc == 2
    assert "corpus file does not exist" in capsys.readouterr().err


def test_build_bundle_rejects_gpu_without_machine_shape(tmp_path: Path) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")

    with pytest.raises(ValueError, match="machine_shape is required"):
        builder.build_bundle(
            bundle_dir=tmp_path / "bundle",
            corpus_file=corpus,
            kernel_id="furusekazufumi/test-lm-compare",
            title="test-lm-compare",
            machine_shape=None,
            enable_gpu=True,
            enable_internet=False,
            is_private=True,
            cfg=builder.CompareConfig(),
        )


def test_build_bundle_rejects_machine_shape_when_gpu_disabled(tmp_path: Path) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")

    with pytest.raises(ValueError, match="machine_shape must be omitted"):
        builder.build_bundle(
            bundle_dir=tmp_path / "bundle",
            corpus_file=corpus,
            kernel_id="furusekazufumi/test-lm-compare",
            title="test-lm-compare",
            machine_shape="NvidiaTeslaT4",
            enable_gpu=False,
            enable_internet=False,
            is_private=True,
            cfg=builder.CompareConfig(),
        )


def test_build_bundle_rejects_unsafe_bundle_dir(capsys: pytest.CaptureFixture[str]) -> None:
    builder = _load_builder()
    repo_root = Path(__file__).resolve().parents[2]
    corpus = repo_root / "tests" / "tmp-kaggle-bundle-corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")

    try:
        rc = builder.main(
            [
                "--bundle-dir",
                str(repo_root),
                "--corpus-file",
                str(corpus),
            ]
        )
    finally:
        corpus.unlink(missing_ok=True)

    assert rc == 2
    assert "unsafe" in capsys.readouterr().err


def test_build_bundle_rejects_repo_internal_bundle_dir(
    capsys: pytest.CaptureFixture[str],
) -> None:
    builder = _load_builder()
    repo_root = Path(__file__).resolve().parents[2]
    corpus = repo_root / "tests" / "tmp-kaggle-bundle-corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = repo_root / "out" / "tmp-kaggle-bundle"
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)

    try:
        rc = builder.main(
            [
                "--bundle-dir",
                str(bundle_dir),
                "--corpus-file",
                str(corpus),
            ]
        )
    finally:
        corpus.unlink(missing_ok=True)
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir, ignore_errors=True)

    assert rc == 2
    assert "repo-external path" in capsys.readouterr().err


def test_build_bundle_rejects_non_bundle_existing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "existing"
    bundle_dir.mkdir()
    (bundle_dir / "notes.txt").write_text("not a bundle\n", encoding="utf-8")

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
        ]
    )

    assert rc == 2
    assert "not a recognized Kaggle bundle" in capsys.readouterr().err


def test_build_bundle_rejects_malformed_bundle_sentinel_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "existing"
    bundle_dir.mkdir()
    (bundle_dir / "bundle_manifest.json").write_text("{\"runner\": \"runner.py\"}\n", encoding="utf-8")

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
        ]
    )

    assert rc == 2
    assert "not a recognized Kaggle bundle" in capsys.readouterr().err


def test_build_bundle_rejects_weak_bundle_sentinel_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "existing"
    bundle_dir.mkdir()
    (bundle_dir / "runner.py").write_text("# fake\n", encoding="utf-8")
    (bundle_dir / "config.json").write_text("{}\n", encoding="utf-8")
    (bundle_dir / "kernel-metadata.json").write_text("{}\n", encoding="utf-8")
    (bundle_dir / "input_corpus.txt").write_text("abc\n", encoding="utf-8")
    (bundle_dir / "README.md").write_text("fake\n", encoding="utf-8")
    (bundle_dir / "src" / "llcore").mkdir(parents=True)
    manifest = {
        "runner": "runner.py",
        "kernel_id": "fake/kernel",
        "copied_files": {
            "corpus": "input_corpus.txt",
            "config": "config.json",
            "metadata": "kernel-metadata.json",
            "src_llcore": "src/llcore",
            "pkg_llcore": "llcore",
            "license": "LICENSE",
            "notice": "NOTICE",
        },
        "source_sha256": "0" * 64,
    }
    (bundle_dir / "bundle_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
        ]
    )

    assert rc == 2
    assert "not a recognized Kaggle bundle" in capsys.readouterr().err


def test_preflight_rejects_tampered_license_or_notice(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _load_builder()
    preflight = _load_script("kaggle_bundle_preflight.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--kernel-id",
            "furusekazufumi/test-lm-compare",
            "--title",
            "test-lm-compare",
        ]
    )
    assert rc == 0

    (bundle_dir / "NOTICE").write_text("tampered notice\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2
    assert "NOTICE sha256 does not match" in capsys.readouterr().err


def test_build_bundle_rejects_empty_existing_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "existing"
    bundle_dir.mkdir()

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
        ]
    )

    assert rc == 2
    assert "empty" in capsys.readouterr().err


def test_build_bundle_removes_stale_files(tmp_path: Path) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    first_rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
        ]
    )
    assert first_rc == 0
    stale = bundle_dir / "stale.txt"
    stale.write_text("old\n", encoding="utf-8")

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
        ]
    )

    assert rc == 0
    assert not stale.exists()


def test_generated_runner_executes_locally_on_tiny_corpus(tmp_path: Path) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("abcde " * 80).strip() + "\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--block-size",
            "16",
            "--n-layer",
            "1",
            "--n-head",
            "1",
            "--n-embd",
            "16",
            "--state-size",
            "8",
            "--max-iters",
            "1",
            "--batch-size",
            "2",
            "--eval-iters",
            "1",
            "--throughput-new-tokens",
            "1",
            "--throughput-repeats",
            "1",
        ]
    )
    assert rc == 0

    proc = subprocess.run(
        [sys.executable, str(bundle_dir / "runner.py")],
        cwd=bundle_dir,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert proc.returncode == 0, proc.stderr
    assert "[compare] wrote" in proc.stdout
    out_json = bundle_dir / "artifacts" / "lm_compare.json"
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(payload["reports"]) == {"gpt", "recurrent", "rwkv"}


def test_bundle_notice_text_rejects_unsanitized_notice(monkeypatch: Any) -> None:
    builder = _load_builder()
    monkeypatch.setattr(
        builder.Path,
        "read_text",
        lambda self, encoding="utf-8": "Commercial licenses are also available; see LICENSE-COMMERCIALX.\n",
    )

    with pytest.raises(ValueError, match="NOTICE sanitize failed"):
        builder._bundle_notice_text()


def test_build_bundle_invalid_compare_config_returns_rc2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--n-head",
            "0",
        ]
    )

    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_build_bundle_rejects_empty_title(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--title",
            "",
        ]
    )

    assert rc == 2
    assert "title must be a non-empty string" in capsys.readouterr().err


def test_build_bundle_rejects_empty_kernel_id(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builder = _load_builder()
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--kernel-id",
            "",
        ]
    )

    assert rc == 2
    assert "kernel_id must be a non-empty string" in capsys.readouterr().err
