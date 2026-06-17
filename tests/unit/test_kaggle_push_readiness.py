# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib.util
import json
import subprocess
import tempfile
import zipfile
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest


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


def _clear_kaggle_env(monkeypatch: Any) -> None:
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_API_V1_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_KERNEL_RUN_TYPE", raising=False)
    empty_dir = Path(tempfile.gettempdir()) / "llcore-kaggle-test-empty"
    empty_dir.mkdir(exist_ok=True)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(empty_dir))
    monkeypatch.setenv("HOME", str(empty_dir))
    monkeypatch.setenv("USERPROFILE", str(empty_dir))


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
            return _completed(stdout="ref,title,author\nr,t,furusekazufumi\n")
        if args[:2] == ["quota", "-v"]:
            return _completed(stdout="resource,used,total,remaining\nGPU,0.00h,30.00h,30.00h\n")
        raise AssertionError(args)

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", _fake_load_script)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "[kaggle-ready]" in out
    assert 'kaggle kernels push -p "' in out
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["kernel_id"] == "furusekazufumi/test-kernel"
    assert payload["push_payload"]["checked"] is True
    assert "bundle_manifest.json" not in payload["push_payload"]["included_files"]
    assert payload["auth"]["authenticated"] is True
    assert payload["auth"]["credential_sources"] == ["kaggle.json"]
    assert str(payload["auth"]["configured_username"]).lower() == "furusekazufumi"
    assert payload["auth"]["owner_check_status"] == "validated_local_config"
    assert str(payload["auth"]["probe_author"]).lower() == "furusekazufumi"
    assert payload["auth"]["probe_author_status"] == "owner_slug_matches_authenticated_user"
    assert payload["auth"]["probe_row_state"] == "authenticated_account_has_kernels"
    assert payload["auth"]["target_slug_existence"] == "unverified_by_probe"
    assert payload["auth"]["owner_verification_passed"] is False
    assert payload["quota"]["skipped"] is True
    assert payload["quota"]["reason"] == "cpu bundle does not require accelerator quota"
    assert payload["quota"]["checked_resource"] == "cpu"
    assert payload["dataset"]["checked"] is False


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
    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(rc=1, stderr="not logged in\n"),
    )

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_AUTH
    assert "error:" in capsys.readouterr().err


def test_check_readiness_fails_when_kernel_owner_does_not_match_configured_username(
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
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "someone-else")
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(stdout="ref,title,author\nr,t,furusekazufumi\n"),
    )

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "does not match configured Kaggle username" in capsys.readouterr().err


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
    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    def _timeout(args: list[str], *, timeout_s: int) -> Any:
        raise subprocess.TimeoutExpired(cmd=["kaggle", *args], timeout=timeout_s)

    monkeypatch.setattr(script, "_run_kaggle", _timeout)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_AUTH
    assert "error:" in capsys.readouterr().err


def test_check_readiness_reads_username_from_kaggle_json_when_env_missing(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text(
        json.dumps({"username": "furusekazufumi", "key": "dummy"}) + "\n",
        encoding="utf-8",
    )

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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(stdout="ref,title,author\nr,t,furusekazufumi\n"),
    )
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0
    assert "quota_checked=cpu" in capsys.readouterr().out


@pytest.mark.parametrize("payload", ["[]\n", '"x"\n', "42\n"])
def test_configured_kaggle_username_rejects_non_object_json(
    tmp_path: Path, monkeypatch: Any, payload: str
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text(payload, encoding="utf-8")

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    with pytest.raises(script._KaggleConfigValidationError, match="expected top-level object"):
        script._configured_kaggle_username()


@pytest.mark.parametrize(
    ("payload",),
    [
        ('{"username":"furusekazufumi"}\n',),
        ('{"username":"furusekazufumi","key":""}\n',),
        ('{"username":"furusekazufumi","key":42}\n',),
    ],
)
def test_configured_kaggle_username_requires_nonempty_string_key_in_kaggle_json(
    tmp_path: Path, monkeypatch: Any, payload: str
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text(payload, encoding="utf-8")

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    with pytest.raises(script._KaggleConfigValidationError, match="expected non-empty string username and key"):
        script._configured_kaggle_username()


def test_check_readiness_rejects_malformed_kaggle_json_as_validation_error(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text('{"username":"furusekazufumi"}\n', encoding="utf-8")

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
                    }
                },
                "runner": None,
            }

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "kaggle.json present but malformed" in capsys.readouterr().err


def test_check_readiness_allows_api_token_only_auth_without_local_username(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    report_path = tmp_path / "ready.json"
    config_dir = tmp_path / "kaggle-empty"
    config_dir.mkdir()

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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(stdout="ref,title,author\n"),
    )
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-token")

    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["auth"]["configured_username"] is None
    assert payload["auth"]["owner_check_status"] == "advisory_token_auth"
    assert payload["auth"]["probe_author_status"] == "advisory_unverified_empty_probe"
    assert "quota_checked=cpu" in capsys.readouterr().out


def test_check_readiness_allows_api_token_with_malformed_local_config_as_advisory(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    report_path = tmp_path / "ready.json"
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text('{"username":"furusekazufumi"}\n', encoding="utf-8")

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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(stdout="ref,title,author\n"),
    )
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-token")

    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["auth"]["configured_username"] is None
    assert payload["auth"]["owner_check_status"] == "advisory_token_auth_malformed_local_config"
    assert "quota_checked=cpu" in capsys.readouterr().out


def test_check_readiness_rejects_owner_mismatch_even_with_api_token(
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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(stdout="ref,title,author\n"),
    )
    monkeypatch.setattr(script, "_credential_sources", lambda: ["api_token_env", "kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "someone-else")
    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-token")

    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == script.RC_AUTH
    assert "does not match configured Kaggle username" in capsys.readouterr().err


def test_check_readiness_rejects_unreadable_local_config_even_with_api_token(
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
                    }
                },
                "runner": None,
            }

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-token")
    monkeypatch.setattr(
        script,
        "_configured_kaggle_username",
        lambda: (_ for _ in ()).throw(
            script._KaggleConfigValidationError(
                "kaggle.json present but unreadable: denied",
                token_advisory_ok=False,
            )
        ),
    )

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "kaggle.json present but unreadable" in capsys.readouterr().err


def test_check_readiness_ignores_username_only_env_and_falls_back_to_kaggle_json(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text(
        json.dumps({"username": "furusekazufumi", "key": "dummy"}) + "\n",
        encoding="utf-8",
    )

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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(stdout="ref,title,author\nr,t,furusekazufumi\n"),
    )
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("KAGGLE_USERNAME", "someone-else")
    monkeypatch.delenv("KAGGLE_KEY", raising=False)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0
    assert "quota_checked=cpu" in capsys.readouterr().out


def test_configured_kaggle_username_reads_username_from_oauth_credentials(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    creds_path = tmp_path / "credentials.json"
    creds_path.write_text(
        json.dumps({"username": "oauth-user", "refresh_token": "refresh-token"}) + "\n",
        encoding="utf-8",
    )

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_kaggle_oauth_credentials_path", lambda: creds_path)

    assert script._configured_oauth_username() == "oauth-user"


def test_configured_kaggle_username_allows_refresh_token_only_oauth_credentials(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    creds_path = tmp_path / "credentials.json"
    creds_path.write_text(
        json.dumps({"refresh_token": "refresh-token"}) + "\n",
        encoding="utf-8",
    )

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_kaggle_oauth_credentials_path", lambda: creds_path)

    assert script._configured_oauth_username() is None


def test_credential_sources_follow_installed_sdk_token_paths(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    access_token_txt = tmp_path / "access_token.txt"
    access_token_txt.write_text("token\n", encoding="utf-8")
    creds_path = tmp_path / "credentials.json"
    creds_path.write_text(
        json.dumps({"username": "oauth-user", "refresh_token": "refresh-token"}) + "\n",
        encoding="utf-8",
    )

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_kaggle_access_token_path", lambda: tmp_path / "missing-access-token")
    monkeypatch.setattr(script, "_kaggle_access_token_txt_path", lambda: access_token_txt)
    monkeypatch.setattr(script, "_kaggle_oauth_credentials_path", lambda: creds_path)

    assert script._credential_sources() == ["access_token", "oauth_credentials"]


def test_credential_sources_respect_kaggle_config_dir_for_token_and_oauth_files(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "access_token").write_text("token\n", encoding="utf-8")
    (config_dir / "credentials.json").write_text(
        json.dumps({"username": "oauth-user", "refresh_token": "refresh-token"}) + "\n",
        encoding="utf-8",
    )

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    assert script._kaggle_access_token_path() == config_dir / "access_token"
    assert script._kaggle_oauth_credentials_path() == config_dir / "credentials.json"
    assert script._credential_sources() == ["access_token", "oauth_credentials"]


def test_credential_sources_ignore_missing_pathlike_api_token_env(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("KAGGLE_API_TOKEN", str(config_dir / "missing-token.txt"))

    assert script._credential_sources() == []


def test_nonempty_file_returns_false_for_non_utf8_token_file(tmp_path: Path) -> None:
    script = _load_script("kaggle_push_readiness.py")
    token_path = tmp_path / "access_token"
    token_path.write_bytes(b"\xff\xfe\x00")

    assert script._nonempty_file(token_path) is False


def test_configured_kaggle_username_rejects_non_utf8_kaggle_json(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_bytes(b"\xff\xfe\x00")

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))

    with pytest.raises(script._KaggleConfigValidationError, match="unreadable"):
        script._configured_kaggle_username()


def test_check_readiness_ignores_whitespace_only_env_and_falls_back_to_kaggle_json(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    config_dir = tmp_path / "kaggle"
    config_dir.mkdir()
    (config_dir / "kaggle.json").write_text(
        json.dumps({"username": "furusekazufumi", "key": "dummy"}) + "\n",
        encoding="utf-8",
    )

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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(stdout="ref,title,author\nr,t,furusekazufumi\n"),
    )
    monkeypatch.setenv("KAGGLE_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("KAGGLE_USERNAME", "   ")
    monkeypatch.setenv("KAGGLE_KEY", "   ")

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0
    assert "quota_checked=cpu" in capsys.readouterr().out


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
            return _completed(stdout="ref,title,author\nr,t,furusekazufumi\n")
        if args[:2] == ["quota", "-v"]:
            return _completed(stdout="resource,used,total,remaining\nGPU,30.00h,30.00h,0.00h\n")
        raise AssertionError(args)

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_QUOTA
    assert "error:" in capsys.readouterr().err


def test_check_readiness_rejects_missing_kernel_id_as_validation_error(
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
                "checks": {"metadata": {"enable_gpu": "false"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    _clear_kaggle_env(monkeypatch)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "missing kernel_id" in capsys.readouterr().err


def test_check_readiness_rejects_malformed_kernel_id_as_validation_error(
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
                "checks": {"metadata": {"kernel_id": "bad-kernel-id", "enable_gpu": "false"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    _clear_kaggle_env(monkeypatch)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "owner/slug" in capsys.readouterr().err


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
            return _completed(stdout="ref,title,author\nr,t,furusekazufumi\n")
        if args[:2] == ["quota", "-v"]:
            raise AssertionError("CPU bundles must skip quota checks")
        raise AssertionError(args)

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

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
            return _completed(stdout="ref,title,author\nr,t,furusekazufumi\n")
        if args[:2] == ["quota", "-v"]:
            return _completed(
                stdout=(
                    "resource,used,total,remaining\n"
                    "GPU,30.00h,30.00h,0.00h\n"
                    "TPU,0.00h,20.00h,20.00h\n"
                )
            )
        raise AssertionError(args)

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

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
            return _completed(stdout="ref,title,author\nr,t,furusekazufumi\n")
        if args[:2] == ["quota", "-v"]:
            return _completed(stdout='resource,used,total\nGPU T4,10.00h,"1,030.00h"\n')
        raise AssertionError(args)

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["env"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0
    assert "quota_rows=1" in capsys.readouterr().out


def test_check_readiness_allows_first_push_when_probe_has_header_only_csv(
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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(stdout="ref,title,author\n"),
    )
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0
    assert "quota_checked=cpu" in capsys.readouterr().out


def test_check_readiness_reports_header_only_probe_state(
    tmp_path: Path, monkeypatch: Any
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
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"}},
                "runner": None,
            }

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", lambda args, *, timeout_s: _completed(stdout="ref,title,author\n"))
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["auth"]["probe_row_state"] == "header_only_or_first_push"
    assert payload["auth"]["owner_verification_passed"] is False


def test_check_readiness_marks_probe_author_mismatch_as_advisory(
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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(stdout="ref,title,author\nr,t,Furuse Kazufumi\n"),
    )
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["auth"]["probe_author_status"] == "advisory_probe_owner_slug_mismatch"
    captured = capsys.readouterr()
    assert "warning: owner verification is advisory only" in captured.err
    assert "owner=validated_local_config" in captured.out


def test_check_readiness_marks_first_probe_row_mismatch_as_advisory(
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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(
        script,
        "_run_kaggle",
        lambda args, *, timeout_s: _completed(
            stdout="ref,title,author\nr1,t1,Other User\nr2,t2,furusekazufumi\n"
        ),
    )
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["auth"]["probe_author_status"] == "advisory_probe_owner_slug_mismatch"
    captured = capsys.readouterr()
    assert "warning: owner verification is advisory only" in captured.err
    assert "owner=validated_local_config" in captured.out


def test_check_readiness_rejects_bundle_without_license_and_notice(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "src" / "llcore").mkdir(parents=True)
    (bundle_dir / "src" / "llcore" / "__init__.py").write_text(
        "# SPDX-License-Identifier: Apache-2.0\n",
        encoding="utf-8",
    )

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    _clear_kaggle_env(monkeypatch)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "must include LICENSE and NOTICE" in capsys.readouterr().err


def test_check_readiness_rejects_bundle_with_commercial_license_wording(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "src" / "llcore").mkdir(parents=True)
    (bundle_dir / "src" / "llcore" / "backend.py").write_text(
        "# SPDX-License-Identifier: Apache-2.0\n# Apache-2.0 + Commercial dual-license\n",
        encoding="utf-8",
    )
    (bundle_dir / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    (bundle_dir / "NOTICE").write_text("Notice\n", encoding="utf-8")

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    _clear_kaggle_env(monkeypatch)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "commercial-license wording remains" in capsys.readouterr().err


def test_check_readiness_rejects_notice_with_commercial_license_reference(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "src" / "llcore").mkdir(parents=True)
    (bundle_dir / "src" / "llcore" / "__init__.py").write_text(
        "# SPDX-License-Identifier: Apache-2.0\n",
        encoding="utf-8",
    )
    (bundle_dir / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    (bundle_dir / "NOTICE").write_text(
        "Commercial licenses are also available; see LICENSE-COMMERCIAL.\n",
        encoding="utf-8",
    )

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    _clear_kaggle_env(monkeypatch)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "commercial-license wording remains" in capsys.readouterr().err


def test_check_readiness_ignores_commercial_marker_inside_input_corpus(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "src" / "llcore").mkdir(parents=True)
    (bundle_dir / "src" / "llcore" / "__init__.py").write_text(
        "# SPDX-License-Identifier: Apache-2.0\n",
        encoding="utf-8",
    )
    (bundle_dir / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    (bundle_dir / "NOTICE").write_text("Commercial licenses are available separately.\n", encoding="utf-8")
    (bundle_dir / "input_corpus.txt").write_text(
        "This corpus literally mentions LICENSE-COMMERCIAL but is payload text.\n",
        encoding="utf-8",
    )

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"}},
                "runner": None,
            }

    def _fake_run_kaggle(args: list[str], *, timeout_s: int) -> Any:
        if args[:2] == ["kernels", "list"]:
            return _completed(stdout="ref,title,author\nr,t,\n")
        raise AssertionError(args)

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0
    assert "quota_checked=cpu" in capsys.readouterr().out


def test_check_readiness_ignores_commercial_marker_inside_kaggleignored_artifacts(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "src" / "llcore").mkdir(parents=True)
    (bundle_dir / "src" / "llcore" / "__init__.py").write_text("# ok\n", encoding="utf-8")
    (bundle_dir / "artifacts").mkdir()
    (bundle_dir / "artifacts" / "lm_compare.json").write_text(
        '{"note":"Commercial dual-license"}\n', encoding="utf-8"
    )
    (bundle_dir / ".kaggleignore").write_text("artifacts/\n", encoding="utf-8")
    (bundle_dir / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    (bundle_dir / "NOTICE").write_text("Commercial licenses are available separately.\n", encoding="utf-8")

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", lambda args, *, timeout_s: _completed(stdout="ref,title,author\n"))
    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == 0
    assert "quota_checked=cpu" in capsys.readouterr().out


def test_ignored_relative_path_requires_directory_boundary() -> None:
    script = _load_script("kaggle_push_readiness.py")

    prefixes = ("src", "artifacts/")

    assert script._is_ignored_relative_path("src", prefixes) is True
    assert script._is_ignored_relative_path("src/module.py", prefixes) is False
    assert script._is_ignored_relative_path("src_llcore.zip", prefixes) is False
    assert script._is_ignored_relative_path("artifacts", prefixes) is True
    assert script._is_ignored_relative_path("artifacts/lm_compare.json", prefixes) is True


def test_ignored_bundle_prefixes_only_strip_literal_dot_slash(tmp_path: Path) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / ".kaggleignore").write_text("./artifacts/\n.venv/\n..foo\n", encoding="utf-8")

    assert script._ignored_bundle_prefixes(bundle_dir) == ("artifacts/", ".venv/", "..foo")


def test_ignored_bundle_prefixes_rejects_negation_rule(tmp_path: Path) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / ".kaggleignore").write_text("artifacts/\n!runner.py\n", encoding="utf-8")

    with pytest.raises(ValueError, match="negation rules are unsupported"):
        script._ignored_bundle_prefixes(bundle_dir)


def test_ignored_bundle_prefixes_rejects_glob_rule(tmp_path: Path) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / ".kaggleignore").write_text("artifacts/**\n", encoding="utf-8")

    with pytest.raises(ValueError, match="glob patterns are unsupported"):
        script._ignored_bundle_prefixes(bundle_dir)


def test_check_readiness_rejects_archive_member_with_commercial_license_reference(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    dataset_payload = bundle_dir / "dataset_payload"
    dataset_payload.mkdir(parents=True)
    (bundle_dir / "LICENSE").write_text("Apache License 2.0\n", encoding="utf-8")
    (bundle_dir / "NOTICE").write_text("Commercial licenses are available separately.\n", encoding="utf-8")
    with zipfile.ZipFile(dataset_payload / "src_llcore.zip", "w") as zf:
        zf.writestr("src/llcore/__init__.py", "Commercial licenses are also available; see LICENSE-COMMERCIAL.\n")

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    _clear_kaggle_env(monkeypatch)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "commercial-license wording remains" in capsys.readouterr().err


def test_check_readiness_rejects_unreadable_license_text_file(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "src" / "llcore").mkdir(parents=True)
    (bundle_dir / "src" / "llcore" / "__init__.py").write_text("# ok\n", encoding="utf-8")
    (bundle_dir / "LICENSE").write_bytes(b"\xff\xfe\x00")
    (bundle_dir / "NOTICE").write_text("ok\n", encoding="utf-8")

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {"metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"}},
                "runner": None,
            }

    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    _clear_kaggle_env(monkeypatch)

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "non-UTF-8 text file" in capsys.readouterr().err


def test_check_readiness_verifies_dataset_dependency_shas(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    dataset_payload = bundle_dir / "dataset_payload"
    dataset_payload.mkdir(parents=True)
    (dataset_payload / "dataset_payload_manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": "a" * 64,
                "corpus_sha256": "b" * 64,
                "src_tree_sha256": "c" * 64,
                "pkg_tree_sha256": "d" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {
                    "metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"},
                    "manifest": {"data_mode": "dataset", "dataset_source": "furusekazufumi/test-dataset"},
                },
                "runner": None,
            }

    def _fake_run_kaggle(args: list[str], *, timeout_s: int) -> Any:
        if args[:2] == ["kernels", "list"]:
            return _completed(stdout="ref,title,author\nr,t,furusekazufumi\n")
        if args[:2] == ["datasets", "status"]:
            return _completed(stdout="ready\n")
        if args[:2] == ["datasets", "download"]:
            out_dir = Path(args[args.index("-p") + 1])
            (out_dir / "config.json").write_text("config\n", encoding="utf-8")
            (out_dir / "input_corpus.txt").write_text("corpus\n", encoding="utf-8")
            (out_dir / "src_llcore" / "src" / "llcore").mkdir(parents=True)
            (out_dir / "pkg_llcore" / "llcore").mkdir(parents=True)
            (out_dir / "src_llcore" / "src" / "llcore" / "__init__.py").write_text("# src\n", encoding="utf-8")
            (out_dir / "pkg_llcore" / "llcore" / "__init__.py").write_text("# src\n", encoding="utf-8")
            return _completed(stdout="downloaded\n")
        raise AssertionError(args)

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")
    monkeypatch.setattr(
        script,
        "_sha256_text",
        lambda path: {"config.json": "a" * 64, "input_corpus.txt": "b" * 64}.get(path.name, "z" * 64),
    )
    monkeypatch.setattr(
        script,
        "_sha256_tree",
        lambda path: "c" * 64 if "src_llcore" in str(path) else "d" * 64,
    )

    report_path = tmp_path / "ready.json"
    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["dataset"]["checked"] is True
    assert payload["dataset"]["status"] == "ready"
    assert payload["dataset"]["matches"] == {
        "config_sha256": True,
        "corpus_sha256": True,
        "src_tree_sha256": True,
        "pkg_tree_sha256": True,
    }
    assert "quota_checked=cpu" in capsys.readouterr().out


def test_check_readiness_rejects_dataset_manifest_missing_required_sha_keys(
    tmp_path: Path, monkeypatch: Any, capsys: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    dataset_payload = bundle_dir / "dataset_payload"
    dataset_payload.mkdir(parents=True)
    (dataset_payload / "dataset_payload_manifest.json").write_text(
        json.dumps({"config_sha256": "a" * 64, "src_tree_sha256": "c" * 64}) + "\n",
        encoding="utf-8",
    )

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {
                    "metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"},
                    "manifest": {"data_mode": "dataset", "dataset_source": "furusekazufumi/test-dataset"},
                },
                "runner": None,
            }

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    def _fake_run_kaggle(args: list[str], *, timeout_s: int) -> Any:
        if args[:2] == ["kernels", "list"]:
            return _completed(stdout="ref,title,author\nr,t,furusekazufumi\n")
        if args[:2] == ["datasets", "status"]:
            return _completed(stdout="ready\n")
        raise AssertionError(args)

    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_VALIDATION
    assert "dataset payload manifest missing required keys" in capsys.readouterr().err


def test_check_readiness_accepts_warning_prefixed_dataset_ready_status(
    tmp_path: Path, monkeypatch: Any
) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    dataset_payload = bundle_dir / "dataset_payload"
    dataset_payload.mkdir(parents=True)
    (dataset_payload / "dataset_payload_manifest.json").write_text(
        json.dumps(
            {
                "config_sha256": "a" * 64,
                "corpus_sha256": "b" * 64,
                "src_tree_sha256": "c" * 64,
                "pkg_tree_sha256": "c" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "ready.json"

    class _FakePreflight:
        subprocess = SimpleNamespace(TimeoutExpired=subprocess.TimeoutExpired)

        @staticmethod
        def preflight_bundle(bundle_dir: Path, *, run_runner: bool, runner_timeout: int) -> dict[str, object]:
            return {
                "bundle_dir": str(bundle_dir),
                "checks": {
                    "metadata": {"kernel_id": "furusekazufumi/test-kernel", "enable_gpu": "false"},
                    "manifest": {"data_mode": "dataset", "dataset_source": "furusekazufumi/test-dataset"},
                },
                "runner": None,
            }

    def _fake_run_kaggle(args: list[str], *, timeout_s: int) -> Any:
        if args[:2] == ["kernels", "list"]:
            return _completed(stdout="ref,title,author\nr,t,\n")
        if args[:2] == ["datasets", "status"]:
            return _completed(stdout="Warning: outdated client\nready\n")
        if args[:2] == ["datasets", "download"]:
            out_dir = Path(args[args.index("-p") + 1])
            (out_dir / "config.json").write_text("config\n", encoding="utf-8")
            (out_dir / "input_corpus.txt").write_text("corpus\n", encoding="utf-8")
            (out_dir / "src_llcore" / "src" / "llcore").mkdir(parents=True)
            (out_dir / "pkg_llcore" / "llcore").mkdir(parents=True)
            (out_dir / "src_llcore" / "src" / "llcore" / "__init__.py").write_text("# src\n", encoding="utf-8")
            (out_dir / "pkg_llcore" / "llcore" / "__init__.py").write_text("# src\n", encoding="utf-8")
            return _completed(stdout="downloaded\n")
        raise AssertionError(args)

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_run_kaggle", _fake_run_kaggle)
    monkeypatch.setattr(script, "_credential_sources", lambda: ["kaggle.json"])
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: "furusekazufumi")
    monkeypatch.setattr(
        script,
        "_sha256_text",
        lambda path: {"config.json": "a" * 64, "input_corpus.txt": "b" * 64}.get(path.name, "z" * 64),
    )
    monkeypatch.setattr(script, "_sha256_tree", lambda path: "c" * 64)

    rc = script.main(["--bundle-dir", str(bundle_dir), "--json", str(report_path)])

    assert rc == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["dataset"]["status"] == "ready"
    assert payload["dataset"]["status_raw"] == "Warning: outdated client\nready"


def test_iter_kernel_push_files_respects_kaggleignore_and_hashes(tmp_path: Path) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / ".kaggleignore").write_text(
        "dataset_payload/\nartifacts/\npreflight_report.json\nprepare_report.json\n",
        encoding="utf-8",
    )
    (bundle_dir / "kernel-metadata.json").write_text("{}\n", encoding="utf-8")
    (bundle_dir / "runner.py").write_text("# ok\n", encoding="utf-8")
    (bundle_dir / "bundle_manifest.json").write_text("{}\n", encoding="utf-8")
    (bundle_dir / "README.md").write_text("readme\n", encoding="utf-8")
    (bundle_dir / "preflight_report.json").write_text("{}\n", encoding="utf-8")
    (bundle_dir / "artifacts").mkdir()
    (bundle_dir / "artifacts" / "lm_compare.json").write_text("{}\n", encoding="utf-8")
    (bundle_dir / "dataset_payload").mkdir()
    (bundle_dir / "dataset_payload" / "config.json").write_text("{}\n", encoding="utf-8")

    included = script._iter_kernel_push_files(bundle_dir)
    hashes = script._kernel_push_critical_hashes(bundle_dir)

    assert included == [
        ".kaggleignore",
        "README.md",
        "bundle_manifest.json",
        "kernel-metadata.json",
        "runner.py",
    ]
    assert set(hashes) == {
        ".kaggleignore",
        "README.md",
        "bundle_manifest.json",
        "kernel-metadata.json",
        "runner.py",
    }


def test_verify_push_payload_snapshot_rejects_hash_drift(tmp_path: Path) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / ".kaggleignore").write_text("artifacts/\n", encoding="utf-8")
    (bundle_dir / "runner.py").write_text("# ok\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "push_payload": {
                    "included_files": [".kaggleignore", "runner.py"],
                    "critical_hashes": {
                        ".kaggleignore": script._sha256_text(bundle_dir / ".kaggleignore"),
                        "runner.py": "0" * 64,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="push payload hash drifted: runner.py"):
        script._verify_push_payload_snapshot(bundle_dir, snapshot)


def test_verify_push_payload_snapshot_rejects_coverage_gap(tmp_path: Path) -> None:
    script = _load_script("kaggle_push_readiness.py")
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    (bundle_dir / ".kaggleignore").write_text("", encoding="utf-8")
    (bundle_dir / "runner.py").write_text("# ok\n", encoding="utf-8")
    snapshot = tmp_path / "snapshot.json"
    snapshot.write_text(
        json.dumps(
            {
                "push_payload": {
                    "included_files": [".kaggleignore", "runner.py"],
                    "critical_hashes": {
                        "runner.py": script._sha256_text(bundle_dir / "runner.py"),
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="critical_hashes must cover included_files exactly"):
        script._verify_push_payload_snapshot(bundle_dir, snapshot)


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

    _clear_kaggle_env(monkeypatch)
    monkeypatch.setattr(script, "_load_script", lambda path, module_name: _FakePreflight)
    monkeypatch.setattr(script, "_configured_kaggle_username", lambda: None)
    monkeypatch.setattr(script, "_credential_sources", lambda: [])

    rc = script.main(["--bundle-dir", str(bundle_dir)])

    assert rc == script.RC_AUTH
    assert "no Kaggle push credentials found" in capsys.readouterr().err
