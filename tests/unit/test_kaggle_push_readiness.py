# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import subprocess
from types import SimpleNamespace
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


def _completed(*, rc: int = 0, stdout: str = "", stderr: str = "") -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["kaggle"], returncode=rc, stdout=stdout, stderr=stderr)


def test_check_readiness_runs_preflight_and_kaggle_checks(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    report_path = tmp_path / "ready.json"

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {
                    "metadata": {
                        "kernel_id": "furusekazufumi/test-kernel",
                        "enable_gpu": "false",
                        "enable_internet": "false",
                        "is_private": "true",
                        "machine_shape": None,
                    }
                },
                "runner": None,
            }

    def _fake_load_script(path: Path, module_name: str) -> Any:
        return _FakePreflight

    def _fake_run_kaggle(args: list[str], *, timeout_s: int) -> Any:
        if args[:2] == ["kernels", "list"]:
            return _completed(stdout="ref,title,author\n")
        if args[:2] == ["quota", "-v"]:
            return _completed(stdout="resource,used,total,remaining\nGPU,0.00h,30.00h,30.00h\n")
        raise AssertionError(args)

    monkeypatch.setattr(script, "_load_script", _fake_load_script)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])

    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "[kaggle-ready]" in out
    assert 'kaggle kernels push -p "' in out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["kernel_id"] == "furusekazufumi/test-kernel"
    assert payload["auth"]["authenticated"] is True
    assert payload["auth"]["credential_sources"] == ["kaggle.json"]
    assert payload["quota"]["skipped"] is True
    assert payload["quota"]["reason"] == "cpu bundle does not require accelerator quota"
    assert payload["quota"]["checked_resource"] == "cpu"


def test_check_readiness_rejects_nonpositive_kaggle_timeout(
    tmp_path: Path, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    rc = script.main(["--bundle-dir", str(bundle_dir), "--kaggle-timeout", "0"])

    assert rc == 2
    assert "--kaggle-timeout must be >= 1" in capsys.readouterr().err


def test_check_readiness_fails_cleanly_when_auth_check_fails(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(rc=1, stderr="not logged in\n"),
    )

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_AUTH
    assert "error:" in capsys.readouterr().err


def test_check_readiness_fails_cleanly_when_auth_check_times_out(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])

    def _timeout(args: list[str], *, timeout_s: int) -> Any:
        raise subprocess.TimeoutExpired(cmd=["kaggle", *args], timeout=timeout_s)

    monkeypatch.setattr(script, "_run_kaggle", _timeout)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_AUTH
    assert "error:" in capsys.readouterr().err


def test_check_readiness_fails_cleanly_when_quota_is_exhausted(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {
                    "metadata": {
                        "kernel_id": "furusekazufumi/test-kernel",
                        "enable_gpu": "true",
                        "enable_internet": "false",
                        "is_private": "true",
                        "machine_shape": "NvidiaTeslaT4",
                    }
                },
                "runner": None,
            }

    def _fake_run_kaggle(args: list[str], *, timeout_s: int) -> Any:
        if args[:2] == ["kernels", "list"]:
            return _completed(stdout="ref,title,author\n")
        if args[:2] == ["quota", "-v"]:
            return _completed(stdout="resource,used,total,remaining\nGPU,30.00h,30.00h,0.00h\n")
        raise AssertionError(args)

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_QUOTA
    assert "error:" in capsys.readouterr().err


def test_check_readiness_cpu_bundle_does_not_require_accelerator_quota(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {
                    "metadata": {
                        "kernel_id": "furusekazufumi/test-kernel",
                        "enable_gpu": "false",
                        "enable_internet": "false",
                        "is_private": "true",
                        "machine_shape": None,
                    }
                },
                "runner": None,
            }

    def _fake_run_kaggle(args: list[str], *, timeout_s: int) -> Any:
        if args[:2] == ["kernels", "list"]:
            return _completed(stdout="ref,title,author\n")
        if args[:2] == ["quota", "-v"]:
            raise AssertionError("CPU bundles must skip quota checks")
        raise AssertionError(args)

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "quota_rows=0" in out
    assert "quota_checked=cpu" in out


def test_check_readiness_gpu_bundle_requires_positive_gpu_quota(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {
                    "metadata": {
                        "kernel_id": "furusekazufumi/test-kernel",
                        "enable_gpu": "true",
                        "enable_internet": "false",
                        "is_private": "true",
                        "machine_shape": "NvidiaTeslaT4",
                    }
                },
                "runner": None,
            }

    def _fake_run_kaggle(args: list[str], *, timeout_s: int) -> Any:
        if args[:2] == ["kernels", "list"]:
            return _completed(stdout="ref,title,author\n")
        if args[:2] == ["quota", "-v"]:
            return _completed(
                stdout=(
                    "resource,used,total,remaining\n"
                    "GPU,30.00h,30.00h,0.00h\n"
                    "TPU,0.00h,20.00h,20.00h\n"
                )
            )
        raise AssertionError(args)

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_QUOTA
    assert "GPU quota" in capsys.readouterr().err


def test_check_readiness_rejects_tpu_enabled_bundle(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {
                    "metadata": {
                        "kernel_id": "furusekazufumi/test-kernel",
                        "enable_gpu": "false",
                        "enable_tpu": "true",
                        "enable_internet": "false",
                        "is_private": "true",
                        "machine_shape": None,
                    }
                },
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "TPU-enabled bundles" in capsys.readouterr().err


def test_check_readiness_parses_unit_and_comma_formatted_quota_values(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {
                    "metadata": {
                        "kernel_id": "furusekazufumi/test-kernel",
                        "enable_gpu": "true",
                        "enable_internet": "false",
                        "is_private": "true",
                        "machine_shape": "NvidiaTeslaT4",
                    }
                },
                "runner": None,
            }

    def _fake_run_kaggle(args: list[str], *, timeout_s: int) -> Any:
        if args[:2] == ["kernels", "list"]:
            return _completed(stdout="ref,title,author\n")
        if args[:2] == ["quota", "-v"]:
            return _completed(stdout='resource,used,total\nGPU T4,10.00h,"1,030.00h"\n')
        raise AssertionError(args)

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0
    assert "quota_rows=1" in capsys.readouterr().out


def test_check_readiness_fails_cleanly_when_no_push_credentials_exist(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_credential_sources", lambda: [])

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_AUTH
    assert "no Kaggle push credentials found" in capsys.readouterr().err
