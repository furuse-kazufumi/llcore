# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
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
    assert payload["dataset_source"] == "furusekazufumi/llcore-lm-compare-support"
    assert "dataset_payload" in payload["dataset_create_command"]
    assert "datasets version" in payload["dataset_version_command"]


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
