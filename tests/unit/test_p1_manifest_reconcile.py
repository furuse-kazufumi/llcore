# SPDX-License-Identifier: Apache-2.0
"""Tests for :mod:`scripts.p1_manifest_reconcile`."""
from __future__ import annotations

from contextlib import redirect_stdout
from importlib import import_module
import io
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch
import zipfile

import pytest
import torch
from llcore.lm import __main__ as lm_main
from llcore.lm.corpus import (
    _verified_manifest_summary,
    build_utf8_corpus_bundle,
    sha256_text,
)

p1_manifest_inspect = import_module("scripts.p1_manifest_inspect")
p1_manifest_reconcile = import_module("scripts.p1_manifest_reconcile")
p1_prepare_aozora = import_module("scripts.p1_prepare_aozora")


def _verified_entry(manifest: Path) -> dict[str, object]:
    return {
        "status": "verified",
        "manifest_path": str(manifest.resolve()),
        "entry_count": 1,
        "generated_by": "scripts/p1_corpus_probe.py",
        "includes_base": True,
        "combined_sha256": "abc123" * 10 + "ab",
        "bundle_sha256": "def456" * 10 + "de",
    }


def _tiny_presets() -> dict[str, dict[str, int | float]]:
    return {
        "smoke": {"n_layer": 1, "n_head": 2, "n_embd": 16, "block_size": 8, "dropout": 0.2},
    }


def _aozora_zip_bytes(text: str = "吾輩《わがはい》は猫である。\n") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sample.txt", text.encode("cp932"))
    return buf.getvalue()


def _write_inspect_json(
    manifest: Path,
    output_json: Path,
    *,
    base_corpus_file: Path | None = None,
) -> tuple[int, str]:
    argv = [str(manifest), "--json", str(output_json)]
    if base_corpus_file is not None:
        argv.extend(["--base-corpus-file", str(base_corpus_file)])
    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_inspect.main(argv)
    return rc, out.getvalue()


def test_main_accepts_matching_verdict_json(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.txt"
    manifest.write_text("extra.txt\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    entry = _verified_entry(manifest)
    inspect_json.write_text(
        json.dumps({"manifest_verification": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert rc == 0
    text = out.getvalue()
    assert "manifest_verification matched entries=1" in text
    assert "generated_by=scripts/p1_corpus_probe.py" in text


def test_main_writes_json_report_for_match(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.txt"
    manifest.write_text("extra.txt\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    report_json = tmp_path / "reconcile.json"
    entry = _verified_entry(manifest)
    inspect_json.write_text(
        json.dumps({"manifest_verification": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json), "--json", str(report_json)]
        )

    assert rc == 0
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["status"] == "matched"
    assert payload["comparison_mode"] == "positional"
    assert payload["inspect_entry_count"] == 1
    assert payload["runtime_entry_count"] == 1
    assert payload["mismatches"] == []


def test_main_accepts_matching_train_state_pt(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.txt"
    manifest.write_text("extra.txt\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    train_state = tmp_path / "train_state.pt"
    entry = _verified_entry(manifest)
    inspect_json.write_text(
        json.dumps({"manifest_verification": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save({"train_meta": {"manifest_verification": [entry]}}, train_state)

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main([str(inspect_json), "--runtime", str(train_state)])

    assert rc == 0
    assert "source=train_state.pt" in out.getvalue()


def test_main_accepts_inspect_wiring_with_runtime_summary_contract(
    tmp_path: Path,
) -> None:
    """Cover inspect->reconcile wiring without claiming cross-producer drift detection.

    This drives the real ``inspect_manifest()`` path on the inspect side, so the
    test exercises how bundle metadata is interpreted (`generated_by`,
    `combined.includes_base`, `effective_entries`, `base_file`). The runtime
    side still uses the shared ``_verified_manifest_summary()`` helper, so this
    is not an independent producer-vs-producer checksum test.
    """
    base = tmp_path / "base.txt"
    extra = tmp_path / "extra.txt"
    manifest = tmp_path / "selected.txt"
    bundle_path = manifest.with_suffix(".txt.bundle.json")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    base.write_text("ab\n", encoding="utf-8")
    extra.write_text("cd\n", encoding="utf-8")
    manifest.write_text("extra.txt\n", encoding="utf-8")
    bundle_path.write_text(
        json.dumps(
            {
                "generated_by": "scripts/p1_corpus_probe.py",
                "manifest_path": str(manifest.resolve()),
                "manifest_sha256": sha256_text(manifest.read_text(encoding="utf-8")),
                "bundle": build_utf8_corpus_bundle([extra], base_file=base),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    inspect_report, inspect_rc = p1_manifest_inspect.inspect_manifest(
        manifest,
        base_corpus_file=base,
    )
    inspect_json.write_text(json.dumps(inspect_report, ensure_ascii=False, indent=2), encoding="utf-8")
    runtime_entry = _verified_manifest_summary(
        manifest,
        generated_by="scripts/p1_corpus_probe.py",
        includes_base=True,
        effective_entries=[extra],
        base_file=base,
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [runtime_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert inspect_rc == 0
    assert rc == 0
    assert "manifest_verification matched entries=1" in out.getvalue()


def test_main_accepts_actual_probe_output_against_runtime_verdict(tmp_path: Path) -> None:
    """Cover disk round-trip + contract fields, not independent hash logic.

    The inspect/runtime summaries still share ``_verified_manifest_summary()``
    and ``build_utf8_corpus_bundle()``. This test therefore focuses on the
    actual producer/runtime handoff: probe selection writes a manifest to disk,
    train resolves that manifest back from disk, and both sides agree on the
    selected entries plus the ``includes_base`` contract.

    This is intentionally a single-manifest path. Positional/order-sensitive
    multi-manifest behavior is covered separately by the synthetic reconcile
    tests below.
    """
    base = tmp_path / "base.txt"
    keep = tmp_path / "extras" / "keep.txt"
    drop = tmp_path / "extras" / "drop.txt"
    keep.parent.mkdir()
    manifest = tmp_path / "selected.txt"
    inspect_json = tmp_path / "inspect.json"
    out_dir = tmp_path / "train"
    base.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    keep.write_text(("abcde" * 20) + "\n", encoding="utf-8")
    drop.write_text(("xyz" * 20) + "\n", encoding="utf-8")

    p1_corpus_probe = import_module("scripts.p1_corpus_probe")
    probe_rc = p1_corpus_probe.main(
        [
            str(base),
            str(keep),
            str(drop),
            "--max-oov-rate",
            "0.2",
            "--max-new-chars",
            "1",
            "--write-manifest",
            str(manifest),
        ]
    )
    manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
    inspect_report, inspect_rc = p1_manifest_inspect.inspect_manifest(
        manifest,
        base_corpus_file=base,
    )
    inspect_json.write_text(json.dumps(inspect_report, ensure_ascii=False, indent=2), encoding="utf-8")

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        train_rc = lm_main.main(
            [
                "train",
                "--corpus-file",
                str(base),
                "--extra-corpus-manifest",
                str(manifest),
                "--config",
                "smoke",
                "--out",
                str(out_dir),
                "--max-iters",
                "2",
                "--batch-size",
                "4",
                "--eval-iters",
                "1",
                "--val-frac",
                "0.2",
                "--seed",
                "11",
            ]
        )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(out_dir / "verdict.json")]
        )

    runtime_verdict = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    runtime_snapshot = torch.load(
        out_dir / lm_main.TRAIN_STATE_NAME, map_location="cpu", weights_only=False
    )
    assert inspect_rc == 0
    assert probe_rc == 0
    assert any("keep.txt" in line for line in manifest_lines)
    assert not any("drop.txt" in line for line in manifest_lines)
    assert inspect_report["manifest"]["effective_entries"] == [str(keep.resolve())]
    assert inspect_report["manifest_verification"][0]["status"] == "verified"
    assert inspect_report["manifest_verification"][0]["includes_base"] is True
    # 0=gate pass, 2=gate fail; both still emit verdict.json for reconciliation.
    assert train_rc in {0, 2}
    assert rc == 0
    assert "manifest_verification matched entries=1" in out.getvalue()
    assert runtime_verdict["manifest_verification"][0]["status"] == "verified"
    assert runtime_verdict["manifest_verification"][0]["includes_base"] is True
    assert (
        runtime_snapshot["train_meta"]["manifest_verification"]
        == runtime_verdict["manifest_verification"]
    )


def test_main_accepts_actual_prepare_output_against_runtime_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover prepare->train->reconcile handoff for an extras-only producer.

    This exercises the real ``p1_prepare_aozora.py`` path that writes corpus
    files, manifest, and sibling bundle to disk. As with the probe round-trip,
    the summary hash logic is still shared; the value here is verifying the
    disk round-trip and the ``includes_base=False`` contract across producer,
    runtime verdict emission, and reconcile.

    This is intentionally a single-manifest path. Positional/order-sensitive
    multi-manifest behavior is covered separately by the synthetic reconcile
    tests below.
    """
    base = tmp_path / "base.txt"
    extras_dir = tmp_path / "prepared"
    manifest = tmp_path / "prepared_extras.txt"
    inspect_json = tmp_path / "inspect.json"
    out_dir = tmp_path / "train"
    base.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")

    class _StaticOpener:
        def open(self, url: str, timeout: float = 30.0) -> object:
            class _Resp:
                def __enter__(self) -> "_Resp":
                    return self

                def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                    return None

                def read(self) -> bytes:
                    return _aozora_zip_bytes()

            return _Resp()

    monkeypatch.setattr(
        p1_prepare_aozora.urllib.request,
        "build_opener",
        lambda *handlers: _StaticOpener(),
    )
    prepare_rc = p1_prepare_aozora.main(
        [
            "--out-dir",
            str(extras_dir),
            "--write-manifest",
            str(manifest),
            "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip",
        ]
    )
    prepared_extra = extras_dir / "aozora_000148_789_ruby_5639.txt"
    manifest_lines = manifest.read_text(encoding="utf-8").splitlines()
    inspect_report, inspect_rc = p1_manifest_inspect.inspect_manifest(
        manifest,
        base_corpus_file=base,
    )
    inspect_json.write_text(
        json.dumps(inspect_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        train_rc = lm_main.main(
            [
                "train",
                "--corpus-file",
                str(base),
                "--extra-corpus-manifest",
                str(manifest),
                "--config",
                "smoke",
                "--out",
                str(out_dir),
                "--max-iters",
                "2",
                "--batch-size",
                "4",
                "--eval-iters",
                "1",
                "--val-frac",
                "0.2",
                "--seed",
                "11",
            ]
        )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(out_dir / "verdict.json")]
        )

    runtime_verdict = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    runtime_snapshot = torch.load(
        out_dir / lm_main.TRAIN_STATE_NAME, map_location="cpu", weights_only=False
    )
    assert prepare_rc == 0
    assert inspect_rc == 0
    assert any("aozora_000148_789_ruby_5639.txt" in line for line in manifest_lines)
    assert inspect_report["manifest"]["effective_entries"] == [str(prepared_extra.resolve())]
    assert inspect_report["manifest_verification"][0]["status"] == "verified"
    assert inspect_report["manifest_verification"][0]["includes_base"] is False
    # 0=gate pass, 2=gate fail; both still emit verdict.json for reconciliation.
    assert train_rc in {0, 2}
    assert rc == 0
    assert "manifest_verification matched entries=1" in out.getvalue()
    assert runtime_verdict["manifest_verification"][0]["status"] == "verified"
    assert runtime_verdict["manifest_verification"][0]["includes_base"] is False
    assert (
        runtime_snapshot["train_meta"]["manifest_verification"]
        == runtime_verdict["manifest_verification"]
    )


def test_main_accepts_actual_multi_manifest_output_against_runtime_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cover ordered probe+prepare manifests through the shipped inspect CLI.

    Two actual producers write separate manifest/bundle pairs to disk, then
    ``p1_manifest_inspect.py --json`` is run once per manifest and
    ``llcore.lm train`` consumes both manifests in the same order. Reconcile
    then concatenates the two inspect JSON files positionally, so this closes
    the actual multi-manifest handoff for the current shipped CLI contract.
    The positional comparison is still content-based: order sensitivity is only
    exercised here because the probe and prepare entries differ in comparable
    fields such as ``generated_by`` and ``includes_base``.

    As elsewhere in this file, ``includes_base`` is checked as provenance
    metadata for each producer entry; base de-dup/corpus composition is still
    enforced by runtime resolve logic rather than by this field alone.
    """
    base = tmp_path / "base.txt"
    keep = tmp_path / "extras" / "keep.txt"
    drop = tmp_path / "extras" / "drop.txt"
    keep.parent.mkdir()
    probe_manifest = tmp_path / "probe_selected.txt"
    prepare_dir = tmp_path / "prepared"
    prepare_manifest = tmp_path / "prepared_extras.txt"
    probe_inspect_json = tmp_path / "probe_inspect.json"
    prepare_inspect_json = tmp_path / "prepare_inspect.json"
    out_dir = tmp_path / "train_multi"
    base.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    keep.write_text(("abcde" * 20) + "\n", encoding="utf-8")
    drop.write_text(("xyz" * 20) + "\n", encoding="utf-8")

    p1_corpus_probe = import_module("scripts.p1_corpus_probe")
    probe_rc = p1_corpus_probe.main(
        [
            str(base),
            str(keep),
            str(drop),
            "--max-oov-rate",
            "0.2",
            "--max-new-chars",
            "1",
            "--write-manifest",
            str(probe_manifest),
        ]
    )

    class _StaticOpener:
        def open(self, url: str, timeout: float = 30.0) -> object:
            class _Resp:
                def __enter__(self) -> "_Resp":
                    return self

                def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                    return None

                def read(self) -> bytes:
                    return _aozora_zip_bytes()

            return _Resp()

    monkeypatch.setattr(
        p1_prepare_aozora.urllib.request,
        "build_opener",
        lambda *handlers: _StaticOpener(),
    )
    prepare_rc = p1_prepare_aozora.main(
        [
            "--out-dir",
            str(prepare_dir),
            "--write-manifest",
            str(prepare_manifest),
            "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip",
        ]
    )

    probe_inspect_rc, _probe_inspect_stdout = _write_inspect_json(
        probe_manifest,
        probe_inspect_json,
        base_corpus_file=base,
    )
    prepare_inspect_rc, _prepare_inspect_stdout = _write_inspect_json(
        prepare_manifest,
        prepare_inspect_json,
        base_corpus_file=base,
    )

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        train_rc = lm_main.main(
            [
                "train",
                "--corpus-file",
                str(base),
                "--extra-corpus-manifest",
                str(probe_manifest),
                "--extra-corpus-manifest",
                str(prepare_manifest),
                "--config",
                "smoke",
                "--out",
                str(out_dir),
                "--max-iters",
                "2",
                "--batch-size",
                "4",
                "--eval-iters",
                "1",
                "--val-frac",
                "0.2",
                "--seed",
                "11",
            ]
        )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [
                str(probe_inspect_json),
                str(prepare_inspect_json),
                "--runtime",
                str(out_dir / "verdict.json"),
            ]
        )

    runtime_verdict = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    runtime_snapshot = torch.load(
        out_dir / lm_main.TRAIN_STATE_NAME, map_location="cpu", weights_only=False
    )
    assert probe_rc == 0
    assert prepare_rc == 0
    assert probe_inspect_rc == 0
    assert prepare_inspect_rc == 0
    # 0=gate pass, 2=gate fail; both still emit verdict.json for reconciliation.
    assert train_rc in {0, 2}
    assert rc == 0
    assert "manifest_verification matched entries=2" in out.getvalue()
    probe_inspect_report = json.loads(probe_inspect_json.read_text(encoding="utf-8"))
    prepare_inspect_report = json.loads(prepare_inspect_json.read_text(encoding="utf-8"))
    assert probe_inspect_report["manifest_verification"][0]["status"] == "verified"
    assert prepare_inspect_report["manifest_verification"][0]["status"] == "verified"
    assert runtime_verdict["manifest_verification"][0]["status"] == "verified"
    assert runtime_verdict["manifest_verification"][0]["generated_by"] == "scripts/p1_corpus_probe.py"
    assert runtime_verdict["manifest_verification"][0]["includes_base"] is True
    assert runtime_verdict["manifest_verification"][1]["status"] == "verified"
    assert runtime_verdict["manifest_verification"][1]["generated_by"] == "scripts/p1_prepare_aozora.py"
    assert runtime_verdict["manifest_verification"][1]["includes_base"] is False
    assert (
        runtime_snapshot["train_meta"]["manifest_verification"]
        == runtime_verdict["manifest_verification"]
    )


def test_main_rejects_actual_multi_manifest_order_mismatch_for_distinct_entries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reject a real probe+prepare bundle when distinct inspect JSON order swaps."""
    base = tmp_path / "base.txt"
    keep = tmp_path / "extras" / "keep.txt"
    drop = tmp_path / "extras" / "drop.txt"
    keep.parent.mkdir()
    probe_manifest = tmp_path / "probe_selected.txt"
    prepare_dir = tmp_path / "prepared"
    prepare_manifest = tmp_path / "prepared_extras.txt"
    probe_inspect_json = tmp_path / "probe_inspect.json"
    prepare_inspect_json = tmp_path / "prepare_inspect.json"
    out_dir = tmp_path / "train_multi"
    base.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    keep.write_text(("abcde" * 20) + "\n", encoding="utf-8")
    drop.write_text(("xyz" * 20) + "\n", encoding="utf-8")

    p1_corpus_probe = import_module("scripts.p1_corpus_probe")
    probe_rc = p1_corpus_probe.main(
        [
            str(base),
            str(keep),
            str(drop),
            "--max-oov-rate",
            "0.2",
            "--max-new-chars",
            "1",
            "--write-manifest",
            str(probe_manifest),
        ]
    )

    class _StaticOpener:
        def open(self, url: str, timeout: float = 30.0) -> object:
            class _Resp:
                def __enter__(self) -> "_Resp":
                    return self

                def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                    return None

                def read(self) -> bytes:
                    return _aozora_zip_bytes()

            return _Resp()

    monkeypatch.setattr(
        p1_prepare_aozora.urllib.request,
        "build_opener",
        lambda *handlers: _StaticOpener(),
    )
    prepare_rc = p1_prepare_aozora.main(
        [
            "--out-dir",
            str(prepare_dir),
            "--write-manifest",
            str(prepare_manifest),
            "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip",
        ]
    )

    probe_inspect_rc, _probe_inspect_stdout = _write_inspect_json(
        probe_manifest,
        probe_inspect_json,
        base_corpus_file=base,
    )
    prepare_inspect_rc, _prepare_inspect_stdout = _write_inspect_json(
        prepare_manifest,
        prepare_inspect_json,
        base_corpus_file=base,
    )

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        train_rc = lm_main.main(
            [
                "train",
                "--corpus-file",
                str(base),
                "--extra-corpus-manifest",
                str(probe_manifest),
                "--extra-corpus-manifest",
                str(prepare_manifest),
                "--config",
                "smoke",
                "--out",
                str(out_dir),
                "--max-iters",
                "2",
                "--batch-size",
                "4",
                "--eval-iters",
                "1",
                "--val-frac",
                "0.2",
                "--seed",
                "11",
            ]
        )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [
                str(prepare_inspect_json),
                str(probe_inspect_json),
                "--runtime",
                str(out_dir / "verdict.json"),
            ]
        )

    assert probe_rc == 0
    assert prepare_rc == 0
    assert probe_inspect_rc == 0
    assert prepare_inspect_rc == 0
    # 0=gate pass, 2=gate fail; both still emit verdict.json for reconciliation.
    assert train_rc in {0, 2}
    assert rc == 1
    text = out.getvalue()
    assert "manifest_verification mismatch" in text
    assert "[diff] entry[0] generated_by:" in text


def test_main_accepts_path_only_difference_between_inspect_and_runtime(tmp_path: Path) -> None:
    inspect_manifest = tmp_path / "inspect" / "selected.txt"
    runtime_manifest = tmp_path / "runtime" / "selected.txt"
    inspect_manifest.parent.mkdir()
    runtime_manifest.parent.mkdir()
    inspect_manifest.write_text("extra.txt\n", encoding="utf-8")
    runtime_manifest.write_text("extra.txt\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.JSON"
    inspect_entry = _verified_entry(inspect_manifest)
    runtime_entry = dict(inspect_entry)
    runtime_entry["manifest_path"] = str(runtime_manifest.resolve())
    inspect_json.write_text(
        json.dumps({"manifest_verification": [inspect_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [runtime_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert rc == 0
    assert "manifest_verification matched entries=1" in out.getvalue()


def test_main_rejects_manifest_verification_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.txt"
    manifest.write_text("extra.txt\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    inspect_entry = _verified_entry(manifest)
    runtime_entry = dict(inspect_entry)
    runtime_entry["bundle_sha256"] = "zzz999" * 10 + "zz"
    inspect_json.write_text(
        json.dumps({"manifest_verification": [inspect_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [runtime_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert rc == 1
    text = out.getvalue()
    assert "manifest_verification mismatch" in text
    assert "[inspect] entries=1" in text
    assert "[runtime] entries=1" in text
    assert "[diff] entry[0] bundle_sha256:" in text


def test_main_accepts_matching_multi_manifest_entries_in_same_order(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a\n", encoding="utf-8")
    second.write_text("b\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    first_entry = _verified_entry(first)
    second_entry = _verified_entry(second)
    second_entry["generated_by"] = "scripts/p1_prepare_aozora.py"
    inspect_json.write_text(
        json.dumps(
            {"manifest_verification": [first_entry, second_entry]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps(
            {"manifest_verification": [first_entry, second_entry]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert rc == 0
    assert "manifest_verification matched entries=2" in out.getvalue()


def test_main_rejects_multi_manifest_order_mismatch_for_distinct_entries(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a\n", encoding="utf-8")
    second.write_text("b\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    first_entry = _verified_entry(first)
    second_entry = _verified_entry(second)
    second_entry["generated_by"] = "scripts/p1_prepare_aozora.py"
    inspect_json.write_text(
        json.dumps(
            {"manifest_verification": [first_entry, second_entry]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps(
            {"manifest_verification": [second_entry, first_entry]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert rc == 1
    text = out.getvalue()
    assert "manifest_verification mismatch" in text
    assert "[diff] entry[0] generated_by:" in text


def test_main_accepts_multi_manifest_order_swap_when_comparable_fields_match(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a\n", encoding="utf-8")
    second.write_text("b\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    first_entry = _verified_entry(first)
    second_entry = dict(first_entry)
    second_entry["manifest_path"] = str(second.resolve())
    inspect_json.write_text(
        json.dumps(
            {"manifest_verification": [first_entry, second_entry]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps(
            {"manifest_verification": [second_entry, first_entry]},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert rc == 0
    assert "manifest_verification matched entries=2" in out.getvalue()


def test_main_writes_json_report_for_mismatch(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.txt"
    manifest.write_text("extra.txt\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    report_json = tmp_path / "reconcile.json"
    inspect_entry = _verified_entry(manifest)
    runtime_entry = dict(inspect_entry)
    runtime_entry["bundle_sha256"] = "zzz999" * 10 + "zz"
    inspect_json.write_text(
        json.dumps({"manifest_verification": [inspect_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [runtime_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json), "--json", str(report_json)]
        )

    assert rc == 1
    payload = json.loads(report_json.read_text(encoding="utf-8"))
    assert payload["status"] == "mismatch"
    assert payload["comparison_mode"] == "positional"
    assert payload["inspect_entry_count"] == 1
    assert payload["runtime_entry_count"] == 1
    assert payload["mismatches"][0]["entry_index"] == 0
    assert any("bundle_sha256" in diff for diff in payload["mismatches"][0]["diffs"])


def test_main_rejects_missing_manifest_verification(tmp_path: Path) -> None:
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    inspect_json.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")
    verdict_json.write_text(json.dumps({}, ensure_ascii=False, indent=2), encoding="utf-8")

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert rc == 1
    assert "missing manifest_verification[]" in out.getvalue()


def test_main_rejects_verified_entries_missing_shared_required_field(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.txt"
    manifest.write_text("extra.txt\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    inspect_entry = _verified_entry(manifest)
    runtime_entry = _verified_entry(manifest)
    inspect_entry.pop("bundle_sha256")
    runtime_entry.pop("bundle_sha256")
    inspect_json.write_text(
        json.dumps({"manifest_verification": [inspect_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [runtime_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert rc == 1
    assert "manifest_verification[0].bundle_sha256 must be a non-empty string" in out.getvalue()


def test_main_rejects_verified_entry_with_type_drift_on_one_side(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.txt"
    manifest.write_text("extra.txt\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    inspect_entry = _verified_entry(manifest)
    runtime_entry = _verified_entry(manifest)
    runtime_entry["entry_count"] = True
    inspect_json.write_text(
        json.dumps({"manifest_verification": [inspect_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [runtime_entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main(
            [str(inspect_json), "--runtime", str(verdict_json)]
        )

    assert rc == 1
    assert "manifest_verification[0].entry_count must be an int" in out.getvalue()


def test_main_rejects_broken_checkpoint_with_formatted_message(tmp_path: Path) -> None:
    inspect_json = tmp_path / "inspect.json"
    train_state = tmp_path / "train_state.PT"
    inspect_json.write_text(
        json.dumps({"manifest_verification": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    train_state.write_bytes(b"not a checkpoint")

    out = io.StringIO()
    with redirect_stdout(out):
        rc = p1_manifest_reconcile.main([str(inspect_json), "--runtime", str(train_state)])

    assert rc == 1
    assert "failed to read checkpoint manifest_verification" in out.getvalue()


def test_main_requires_explicit_runtime_for_json_artifacts(tmp_path: Path) -> None:
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    entry = _verified_entry(tmp_path / "selected.txt")
    inspect_json.write_text(
        json.dumps({"manifest_verification": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as excinfo:
        p1_manifest_reconcile.main([str(inspect_json), str(verdict_json)])

    assert excinfo.value.code == 2


def test_script_runs_via_subprocess_entrypoint(tmp_path: Path) -> None:
    manifest = tmp_path / "selected.txt"
    manifest.write_text("extra.txt\n", encoding="utf-8")
    inspect_json = tmp_path / "inspect.json"
    verdict_json = tmp_path / "verdict.json"
    entry = _verified_entry(manifest)
    inspect_json.write_text(
        json.dumps({"manifest_verification": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    verdict_json.write_text(
        json.dumps({"manifest_verification": [entry]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    repo_root = Path(__file__).resolve().parents[2]

    proc = subprocess.run(
        [
            sys.executable,
            "scripts/p1_manifest_reconcile.py",
            str(inspect_json),
            "--runtime",
            str(verdict_json),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "manifest_verification matched entries=1" in proc.stdout
