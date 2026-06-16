# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import shutil
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


def _build_bundle(tmp_path: Path) -> Path:
    builder = _load_script("build_kaggle_lm_compare_bundle.py")
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
    return bundle_dir


def test_preflight_validates_bundle_and_writes_json(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    out_json = tmp_path / "preflight.json"

    rc = preflight.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--json",
            str(out_json),
        ]
    )

    assert rc == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["checks"]["metadata"]["enable_gpu"] == "false"
    assert payload["checks"]["metadata"]["enable_tpu"] == "false"
    assert payload["checks"]["metadata"]["enable_internet"] == "false"
    assert payload["checks"]["metadata"]["is_private"] == "true"
    assert payload["checks"]["metadata"]["machine_shape"] is None
    assert payload["checks"]["manifest"]["runner"] == "runner.py"
    assert len(payload["checks"]["manifest"]["source_sha256"]) == 64
    assert payload["runner"] is None


def test_preflight_runner_smoke_passes(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)

    report = preflight.preflight_bundle(bundle_dir, run_runner=True, runner_timeout=60)

    runner = report["runner"]
    assert isinstance(runner, dict)
    assert runner["returncode"] == 0
    assert runner["output_exists"] is True
    assert (bundle_dir / "artifacts" / "lm_compare.md").is_file()
    assert (bundle_dir / "artifacts" / "lm_compare.svg").is_file()


def test_preflight_runner_smoke_can_be_repeated_without_false_sha_drift(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)

    first = preflight.preflight_bundle(bundle_dir, run_runner=True, runner_timeout=60)
    second = preflight.preflight_bundle(bundle_dir, run_runner=True, runner_timeout=60)

    first_runner = first["runner"]
    second_runner = second["runner"]
    assert isinstance(first_runner, dict)
    assert isinstance(second_runner, dict)
    assert first_runner["returncode"] == 0
    assert second_runner["returncode"] == 0


def test_preflight_rejects_corpus_sha256_drift(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "input_corpus.txt").write_text("tampered\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_source_snapshot_drift(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    compare_path = bundle_dir / "src" / "llcore" / "lm" / "compare.py"
    compare_path.write_text(compare_path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_top_level_package_snapshot_drift(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    compare_path = bundle_dir / "llcore" / "lm" / "compare.py"
    compare_path.write_text(compare_path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_manifest_kernel_id_mismatch(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["kernel_id"] = "furusekazufumi/other-kernel"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_invalid_compare_config(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    config_path = bundle_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    compare_config = config["compare_config"]
    assert isinstance(compare_config, dict)
    compare_config["n_head"] = 0
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_runner_sha256_drift(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    runner_path = bundle_dir / "runner.py"
    runner_path.write_text(runner_path.read_text(encoding="utf-8") + "\n# tampered\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_config_sha256_drift(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    config_path = bundle_dir / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    compare_config = config["compare_config"]
    assert isinstance(compare_config, dict)
    compare_config["max_iters"] = 2
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_runner_smoke_rewrites_stale_artifact(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    out_json = bundle_dir / "artifacts" / "lm_compare.json"
    out_md = bundle_dir / "artifacts" / "lm_compare.md"
    out_svg = bundle_dir / "artifacts" / "lm_compare.svg"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text("{\"stale\": true}\n", encoding="utf-8")
    out_md.write_text("stale\n", encoding="utf-8")
    out_svg.write_text("stale\n", encoding="utf-8")

    report = preflight.preflight_bundle(bundle_dir, run_runner=True, runner_timeout=60)

    runner = report["runner"]
    assert isinstance(runner, dict)
    assert runner["returncode"] == 0
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert "reports" in payload
    assert out_md.read_text(encoding="utf-8") != "stale\n"
    assert out_svg.read_text(encoding="utf-8") != "stale\n"


def test_preflight_rejects_missing_required_file(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    (bundle_dir / "runner.py").unlink()

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_copied_files_escape(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied_files = manifest["copied_files"]
    assert isinstance(copied_files, dict)
    copied_files["corpus"] = "../escape.txt"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_missing_required_copied_file_key(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied_files = manifest["copied_files"]
    assert isinstance(copied_files, dict)
    copied_files.pop("src_llcore")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_missing_top_level_package(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    shutil.rmtree(bundle_dir / "llcore")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_metadata_code_file_mismatch(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    metadata_path = bundle_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["code_file"] = "other.py"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_private_drift(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    metadata_path = bundle_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["is_private"] = "false"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_enable_internet_drift(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    metadata_path = bundle_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["enable_internet"] = "true"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_non_string_bool_text_metadata(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    builder = _load_script("build_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    rc = builder.main(["--bundle-dir", str(bundle_dir), "--corpus-file", str(corpus)])
    assert rc == 0
    metadata_path = bundle_dir / "kernel-metadata.json"
    manifest_path = bundle_dir / "bundle_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata["enable_gpu"] = True
    manifest["enable_gpu"] = True
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_accepts_builder_supported_public_internet_opt_in(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    builder = _load_script("build_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("abcde " * 80).strip() + "\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"

    rc = builder.main(
        [
            "--bundle-dir",
            str(bundle_dir),
            "--corpus-file",
            str(corpus),
            "--public",
            "--enable-internet",
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

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0


def test_preflight_rejects_empty_kernel_id(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    builder = _load_script("build_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    rc = builder.main(["--bundle-dir", str(bundle_dir), "--corpus-file", str(corpus)])
    assert rc == 0
    metadata_path = bundle_dir / "kernel-metadata.json"
    manifest_path = bundle_dir / "bundle_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata["id"] = ""
    manifest["kernel_id"] = ""
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_empty_title(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    builder = _load_script("build_kaggle_lm_compare_bundle.py")
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("abcabc\n", encoding="utf-8")
    bundle_dir = tmp_path / "bundle"
    rc = builder.main(["--bundle-dir", str(bundle_dir), "--corpus-file", str(corpus)])
    assert rc == 0
    metadata_path = bundle_dir / "kernel-metadata.json"
    manifest_path = bundle_dir / "bundle_manifest.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata["title"] = ""
    manifest["title"] = ""
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_gpu_machine_shape_mismatch(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    metadata_path = bundle_dir / "kernel-metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["enable_gpu"] = "true"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_manifest_machine_shape_mismatch(tmp_path: Path) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)
    manifest_path = bundle_dir / "bundle_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["machine_shape"] = "NvidiaTeslaT4"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rc = preflight.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 2


def test_preflight_rejects_timeout_cleanly(tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)

    def _timeout(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(cmd="runner.py", timeout=1)

    monkeypatch.setattr(preflight.subprocess, "run", _timeout)
    rc = preflight.main(["--bundle-dir", str(bundle_dir), "--run-runner", "--runner-timeout", "1"])

    assert rc == 2
    assert "error:" in capsys.readouterr().err


def test_preflight_rejects_nonpositive_runner_timeout(tmp_path: Path, capsys: Any) -> None:
    preflight = _load_script("kaggle_bundle_preflight.py")
    bundle_dir = _build_bundle(tmp_path)

    rc = preflight.main(["--bundle-dir", str(bundle_dir), "--runner-timeout", "0"])

    assert rc == 2
    assert "--runner-timeout must be >= 1" in capsys.readouterr().err


def test_preflight_entrypoint_runs_via_subprocess(tmp_path: Path) -> None:
    bundle_dir = _build_bundle(tmp_path)
    script = Path(__file__).resolve().parents[2] / "scripts" / "kaggle_bundle_preflight.py"
    out_json = tmp_path / "preflight.json"

    proc = subprocess.run(
        [
            sys.executable,
            str(script),
            "--bundle-dir",
            str(bundle_dir),
            "--json",
            str(out_json),
        ],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert proc.returncode == 0, proc.stderr
    assert "[kaggle-preflight]" in proc.stdout
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["checks"]["config"]["block_size"] == 16
