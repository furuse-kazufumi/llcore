# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any


def _load_script(script_name: str) -> Any:
    script = Path(__file__).resolve().parents[2] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), script)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load {script}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_prepare_bundle_builds_and_preflights(tmp_path: Path, capsys: Any) -> None:
    script = _load_script("prepare_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("abcde " * 80).strip() + "\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    report_path = tmp_path / "report.json"

    rc = script.main(
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
            "--json",
            str(report_path),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "[kaggle-prepare]" in out
    assert 'kaggle kernels push -p "' in out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["bundle_dir"] == bundle_dir.name
    assert payload["preflight"]["checks"]["metadata"]["enable_gpu"] == "false"
    assert len(payload["preflight"]["checks"]["manifest"]["source_sha256"]) == 64
    assert payload["preflight"]["runner"] is None


def test_prepare_bundle_runner_smoke_passes(tmp_path: Path) -> None:
    script = _load_script("prepare_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("abcde " * 80).strip() + "\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    rc = script.main(
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
            "--run-runner",
        ]
    )

    assert rc == 0
    out_json = bundle_dir / "artifacts" / "lm_compare.json"
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert set(payload["reports"]) == {"gpt", "recurrent", "rwkv"}


def test_prepare_bundle_dataset_mode_reports_dataset_command(tmp_path: Path, capsys: Any) -> None:
    script = _load_script("prepare_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("abcde " * 80).strip() + "\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    report_path = tmp_path / "report.json"

    rc = script.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--dataset-source",
            "furusekazufumi/llcore-lm-compare-support",
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
            "--run-runner",
            "--json",
            str(report_path),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "[dataset-create]" in out
    assert "[dataset-version]" in out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["bundle_dir"] == bundle_dir.name
    assert payload["dataset_source"] == "furusekazufumi/llcore-lm-compare-support"
    assert payload["dataset_visibility"] == "private"
    assert '<bundle_dir>/dataset_payload' in payload["dataset_create_command"]
    assert "--dir-mode zip" in payload["dataset_create_command"]
    assert "--public" not in payload["dataset_create_command"]
    assert "datasets version" in payload["dataset_version_command"]
    assert "--dir-mode zip" in payload["dataset_version_command"]
    assert bundle_dir.name in payload["dataset_version_command"]
    runner = payload["preflight"]["runner"]
    assert isinstance(runner, dict)
    assert "<bundle_dir>" in runner["stdout"]
    assert str(bundle_dir) not in runner["stdout"]
    assert payload["preflight"]["checks"]["manifest"]["dataset_publish_dir"] == "dataset_payload"
    assert (
        payload["preflight"]["checks"]["config"]["dataset_metadata_path"]
        == "dataset_payload/dataset-metadata.json"
    )
    publish_safety = payload["preflight"]["checks"]["manifest"]["publish_safety"]
    assert publish_safety["status"] == "passed"
    assert publish_safety["scanned_top_level_files"] == [
        "LICENSE",
        "NOTICE",
        "config.json",
        "dataset-metadata.json",
        "dataset_payload_manifest.json",
        "input_corpus.txt",
        "pkg_llcore.zip",
        "src_llcore.zip",
    ]
    dataset_payload = bundle_dir / "dataset_payload"
    assert (dataset_payload / "config.json").is_file()
    assert (dataset_payload / "input_corpus.txt").is_file()
    assert (dataset_payload / "src_llcore.zip").is_file()
    assert (dataset_payload / "pkg_llcore.zip").is_file()


def test_prepare_bundle_dataset_mode_reports_public_visibility(tmp_path: Path, capsys: Any) -> None:
    script = _load_script("prepare_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("abcde " * 80).strip() + "\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    report_path = tmp_path / "report.json"

    rc = script.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--dataset-source",
            "furusekazufumi/llcore-lm-compare-support",
            "--dataset-visibility",
            "public",
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
            "--json",
            str(report_path),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "[dataset-visibility] public" in out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["dataset_visibility"] == "public"
    assert "--public" in payload["dataset_create_command"]


def test_prepare_bundle_rejects_badzip_preflight_with_rc2(tmp_path: Path, capsys: Any, monkeypatch: Any) -> None:
    script = _load_script("prepare_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    class _FakeBuild:
        @staticmethod
        def main(argv: list[str]) -> int:
            return 0

    class _FakePreflight:
        zipfile = zipfile
        subprocess = subprocess

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            raise zipfile.BadZipFile("broken archive")

    def _fake_load_script(path: Path, module_name: str) -> Any:
        if module_name == "build_kaggle_lm_compare_bundle":
            return _FakeBuild
        if module_name == "kaggle_bundle_preflight":
            return _FakePreflight
        raise AssertionError(module_name)

    monkeypatch.setattr(script, "_load_script", _fake_load_script)

    rc = script.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
        ]
    )

    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_prepare_bundle_rejects_nonpositive_runner_timeout(
    tmp_path: Path, capsys: Any
) -> None:
    script = _load_script("prepare_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    rc = script.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--runner-timeout",
            "0",
        ]
    )

    assert rc == 2
    assert "--runner-timeout must be >= 1" in capsys.readouterr().err


def test_prepare_bundle_entrypoint_runs_via_subprocess(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("abcde " * 80).strip() + "\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    report_path = tmp_path / "report.json"
    script = Path(__file__).resolve().parents[2] / "scripts" / "prepare_kaggle_lm_compare_bundle.py"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
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
            "--json",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert proc.returncode == 0, proc.stderr
    assert "[kaggle-prepare]" in proc.stdout
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["kernel_id"] == "furusekazufumi/llcore-lm-compare"
