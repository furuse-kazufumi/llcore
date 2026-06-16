# SPDX-License-Identifier: Apache-2.0
"""Tests for the char-LM CLI helpers in :mod:`llcore.lm.__main__`."""
from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch
import zipfile

import pytest
import torch

from llcore.lm import __main__ as lm_main
from llcore.lm.corpus import build_utf8_corpus_bundle, sha256_text
from llcore.lm.data import encode_corpus, train_val_split
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import TrainConfig, Trainer
from scripts import p1_corpus_probe
from scripts import p1_prepare_aozora


def _tiny_presets() -> dict[str, dict[str, int | float]]:
    return {
        "smoke": {"n_layer": 1, "n_head": 2, "n_embd": 16, "block_size": 8, "dropout": 0.2},
        "p1": lm_main.MODEL_PRESETS["p1"],
    }


def _train_args(corpus_path: Path, out_dir: Path, *extra: str) -> list[str]:
    return [
        "train",
        "--corpus-file",
        str(corpus_path),
        "--config",
        "smoke",
        "--out",
        str(out_dir),
        "--max-iters",
        "4",
        "--batch-size",
        "4",
        "--eval-iters",
        "1",
        "--val-frac",
        "0.2",
        "--seed",
        "11",
        *extra,
    ]


def _aozora_zip_bytes(text: str = "吾輩《わがはい》は猫である。\n") -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("sample.txt", text.encode("cp932"))
    return buf.getvalue()


def _load_model_state(out_dir: Path) -> dict[str, torch.Tensor]:
    ckpt = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=False)
    model = CharGPT(GPTConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model_state"])
    return deepcopy(model.state_dict())


def _assert_same_state(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> None:
    assert left.keys() == right.keys()
    for name, tensor in left.items():
        torch.testing.assert_close(tensor, right[name], rtol=0.0, atol=0.0)


class _IntentionalStop(RuntimeError):
    pass


def _write_snapshot(
    snapshot_path: Path,
    corpus_path: Path,
    *,
    trainer: Trainer,
    model: CharGPT,
    tok: CharTokenizer,
    text: str,
    requested_extra_corpus_files: list[str] | None = None,
    extra_corpus_manifests: list[str] | None = None,
    extra_corpus_files: list[str] | None = None,
    manifest_verification: list[dict[str, Any]] | None = None,
) -> None:
    lm_main._save_training_snapshot(
        snapshot_path,
        model,
        tok,
        trainer,
        corpus="shakespeare",
        corpus_file=str(corpus_path),
        corpus_sha256=lm_main._corpus_sha256(text),
        config_name="smoke",
        val_frac=0.2,
        requested_extra_corpus_files=requested_extra_corpus_files or [],
        extra_corpus_manifests=extra_corpus_manifests or [],
        extra_corpus_files=extra_corpus_files or [],
        manifest_verification=manifest_verification or [],
    )


def test_cmd_train_writes_training_snapshot(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny.txt"
    out_dir = tmp_path / "run"
    corpus_path.write_text(("abcdefg" * 60) + "\n", encoding="utf-8")

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        rc = lm_main.main(_train_args(corpus_path, out_dir, "--max-iters", "2"))

    assert rc in {0, 2}
    assert (out_dir / lm_main.TRAIN_STATE_NAME).exists()
    assert (out_dir / "verdict.json").exists()
    verdict_json = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    assert verdict_json["corpus_sha256"] == lm_main._corpus_sha256(
        corpus_path.read_text(encoding="utf-8")
    )
    verdict = torch.load(out_dir / "model.pt", map_location="cpu", weights_only=False)
    assert verdict["config"]["dropout"] == 0.2


def test_save_training_snapshot_is_atomic_and_leaves_no_tmp_file(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny.txt"
    corpus_path.write_text(("abcdefg" * 20) + "\n", encoding="utf-8")
    text = corpus_path.read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    model = CharGPT(
        GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16)
    )
    trainer = Trainer(model, TrainConfig(max_iters=2, eval_interval=1, eval_iters=1, seed=11))
    snapshot_path = tmp_path / lm_main.TRAIN_STATE_NAME

    _write_snapshot(snapshot_path, corpus_path, trainer=trainer, model=model, tok=tok, text=text)

    assert snapshot_path.exists()
    assert not list(tmp_path.glob(f"{lm_main.TRAIN_STATE_NAME}.*.tmp"))


def test_save_training_snapshot_failure_preserves_old_snapshot_and_cleans_tmp(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny.txt"
    corpus_path.write_text(("abcdefg" * 20) + "\n", encoding="utf-8")
    text = corpus_path.read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    model = CharGPT(
        GPTConfig(vocab_size=tok.vocab_size, block_size=8, n_layer=1, n_head=2, n_embd=16)
    )
    trainer = Trainer(model, TrainConfig(max_iters=2, eval_interval=1, eval_iters=1, seed=11))
    snapshot_path = tmp_path / lm_main.TRAIN_STATE_NAME
    _write_snapshot(snapshot_path, corpus_path, trainer=trainer, model=model, tok=tok, text=text)
    good_snapshot = snapshot_path.read_bytes()

    with patch("llcore.lm.__main__.os.replace", side_effect=RuntimeError("boom")):
        try:
            _write_snapshot(
                snapshot_path, corpus_path, trainer=trainer, model=model, tok=tok, text=text
            )
        except RuntimeError as exc:
            assert "boom" in str(exc)
        else:
            raise AssertionError("expected snapshot replace failure")

    assert snapshot_path.read_bytes() == good_snapshot
    assert not list(tmp_path.glob(f"{lm_main.TRAIN_STATE_NAME}.*.tmp"))


def test_cli_resume_matches_continuous_training_and_preserves_val_frac(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny.txt"
    corpus_path.write_text(("abcdefg" * 60) + "\n", encoding="utf-8")
    full_out = tmp_path / "full"
    split_out = tmp_path / "split"

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        full_rc = lm_main.main(_train_args(corpus_path, full_out))
        text = corpus_path.read_text(encoding="utf-8")
        torch.manual_seed(11)
        tok = CharTokenizer.from_text(text)
        ids = encode_corpus(text, tok)
        train_ids, val_ids = train_val_split(ids, val_frac=0.2)
        preset = lm_main.MODEL_PRESETS["smoke"]
        model = CharGPT(
            GPTConfig(
                vocab_size=tok.vocab_size,
                block_size=int(preset["block_size"]),
                n_layer=int(preset["n_layer"]),
                n_head=int(preset["n_head"]),
                n_embd=int(preset["n_embd"]),
                dropout=float(preset["dropout"]),
            )
        )
        trainer = Trainer(
            model,
            TrainConfig(
                max_iters=4,
                lr_decay_iters=4,
                warmup_iters=0,
                batch_size=4,
                eval_interval=1,
                eval_iters=1,
                seed=11,
            ),
        )
        split_out.mkdir(parents=True, exist_ok=True)

        def stop_after_two_steps(it: int, tr: float, va: float) -> None:
            _write_snapshot(
                split_out / lm_main.TRAIN_STATE_NAME,
                corpus_path,
                trainer=trainer,
                model=model,
                tok=tok,
                text=text,
            )
            if it >= 1:
                raise _IntentionalStop

        try:
            trainer.train(train_ids, val_ids, on_eval=stop_after_two_steps)
        except _IntentionalStop:
            pass
        resume_rc = lm_main.main(
            [
                "train",
                "--resume-checkpoint",
                str(split_out / lm_main.TRAIN_STATE_NAME),
                "--out",
                str(split_out),
                "--max-iters",
                "4",
                "--val-frac",
                "0.2",
            ]
        )

    assert full_rc in {0, 2}
    assert resume_rc in {0, 2}
    _assert_same_state(_load_model_state(full_out), _load_model_state(split_out))


def test_cli_resume_matches_continuous_training_with_sparse_evals(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny.txt"
    corpus_path.write_text(("abcdefg" * 60) + "\n", encoding="utf-8")
    full_out = tmp_path / "full_sparse"
    split_out = tmp_path / "split_sparse"

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        full_rc = lm_main.main(_train_args(corpus_path, full_out, "--max-iters", "16"))
        text = corpus_path.read_text(encoding="utf-8")
        torch.manual_seed(11)
        tok = CharTokenizer.from_text(text)
        ids = encode_corpus(text, tok)
        train_ids, val_ids = train_val_split(ids, val_frac=0.2)
        preset = lm_main.MODEL_PRESETS["smoke"]
        model = CharGPT(
            GPTConfig(
                vocab_size=tok.vocab_size,
                block_size=int(preset["block_size"]),
                n_layer=int(preset["n_layer"]),
                n_head=int(preset["n_head"]),
                n_embd=int(preset["n_embd"]),
                dropout=float(preset["dropout"]),
            )
        )
        trainer = Trainer(
            model,
            TrainConfig(
                max_iters=16,
                lr_decay_iters=16,
                warmup_iters=1,
                batch_size=4,
                eval_interval=2,
                eval_iters=1,
                seed=11,
            ),
        )
        split_out.mkdir(parents=True, exist_ok=True)

        def stop_after_first_sparse_eval(it: int, tr: float, va: float) -> None:
            _write_snapshot(
                split_out / lm_main.TRAIN_STATE_NAME,
                corpus_path,
                trainer=trainer,
                model=model,
                tok=tok,
                text=text,
            )
            raise _IntentionalStop

        try:
            trainer.train(train_ids, val_ids, on_eval=stop_after_first_sparse_eval)
        except _IntentionalStop:
            pass
        resume_rc = lm_main.main(
            [
                "train",
                "--resume-checkpoint",
                str(split_out / lm_main.TRAIN_STATE_NAME),
                "--out",
                str(split_out),
                "--max-iters",
                "16",
                "--val-frac",
                "0.2",
            ]
        )

    assert full_rc in {0, 2}
    assert resume_rc in {0, 2}
    _assert_same_state(_load_model_state(full_out), _load_model_state(split_out))


def test_cli_resume_rejects_corpus_content_drift(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny.txt"
    out_dir = tmp_path / "run"
    corpus_path.write_text(("abcdefg" * 60) + "\n", encoding="utf-8")

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        lm_main.main(_train_args(corpus_path, out_dir, "--max-iters", "2"))
        corpus_path.write_text(("abcxefg" * 60) + "\n", encoding="utf-8")
        try:
            lm_main.main(
                [
                    "train",
                    "--resume-checkpoint",
                    str(out_dir / lm_main.TRAIN_STATE_NAME),
                    "--out",
                    str(out_dir),
                    "--max-iters",
                    "4",
                    "--val-frac",
                    "0.2",
                ]
            )
        except ValueError as exc:
            assert "corpus contents no longer match" in str(exc)
        else:
            raise AssertionError("expected corpus drift to fail closed")


def test_cli_resume_completed_snapshot_reemits_artifacts(tmp_path: Path) -> None:
    corpus_path = tmp_path / "tiny.txt"
    out_dir = tmp_path / "completed_resume"
    corpus_path.write_text(("abcdefg" * 60) + "\n", encoding="utf-8")

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        first_rc = lm_main.main(_train_args(corpus_path, out_dir, "--max-iters", "2"))
        verdict_path = out_dir / "verdict.json"
        model_path = out_dir / "model.pt"
        tokenizer_path = out_dir / "tokenizer.json"
        viz_path = out_dir / "model_viz.json"
        verdict_path.unlink()
        model_path.unlink()
        tokenizer_path.unlink()
        viz_path.unlink()
        resume_out = io.StringIO()
        with redirect_stdout(resume_out):
            resume_rc = lm_main.main(
                [
                    "train",
                    "--resume-checkpoint",
                    str(out_dir / lm_main.TRAIN_STATE_NAME),
                    "--out",
                    str(out_dir),
                    "--val-frac",
                    "0.2",
                ]
            )

    assert first_rc in {0, 2}
    assert resume_rc in {0, 2}
    assert "re-emitting artifacts from the saved state" in resume_out.getvalue()
    assert verdict_path.exists()
    assert model_path.exists()
    assert tokenizer_path.exists()
    assert viz_path.exists()


def test_cli_train_supports_extra_corpus_files(tmp_path: Path) -> None:
    base_path = tmp_path / "base.txt"
    extra_path = tmp_path / "extra.txt"
    out_dir = tmp_path / "multi"
    base_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    extra_path.write_text(("xyz" * 40) + "\n", encoding="utf-8")

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        rc = lm_main.main(
            [
                "train",
                "--corpus-file",
                str(base_path),
                "--extra-corpus-file",
                str(extra_path),
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

    assert rc in {0, 2}
    snapshot = torch.load(out_dir / lm_main.TRAIN_STATE_NAME, map_location="cpu", weights_only=False)
    assert snapshot["train_meta"]["extra_corpus_files"] == [str(extra_path)]
    tokenizer = snapshot["itos"]
    assert "x" in tokenizer and "z" in tokenizer
    verdict = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["extra_corpus_files"] == [str(extra_path)]


def test_cli_train_supports_extra_corpus_manifest(tmp_path: Path) -> None:
    base_path = tmp_path / "base.txt"
    extra_path = tmp_path / "extras" / "extra.txt"
    extra_path.parent.mkdir()
    manifest_path = tmp_path / "extras.txt"
    out_dir = tmp_path / "multi_manifest"
    base_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    extra_path.write_text(("xyz" * 40) + "\n", encoding="utf-8")
    manifest_path.write_text("# comment\nextras/extra.txt\n", encoding="utf-8")

    out = io.StringIO()
    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        with redirect_stdout(out):
            rc = lm_main.main(
                [
                    "train",
                    "--corpus-file",
                    str(base_path),
                    "--extra-corpus-manifest",
                    str(manifest_path),
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

    assert rc in {0, 2}
    assert "[manifest] unverified (no sibling bundle) path=extras.txt" in out.getvalue()
    snapshot = torch.load(out_dir / lm_main.TRAIN_STATE_NAME, map_location="cpu", weights_only=False)
    assert snapshot["train_meta"]["extra_corpus_files"] == [str(extra_path.resolve())]
    assert snapshot["train_meta"]["manifest_verification"] == [
        {
            "status": "unverified",
            "manifest_path": str(manifest_path.resolve()),
            "reason": "no sibling bundle",
        }
    ]
    verdict = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["extra_corpus_files"] == [str(extra_path.resolve())]
    assert verdict["manifest_verification"] == snapshot["train_meta"]["manifest_verification"]
    assert snapshot["train_meta"]["extra_corpus_manifests"] == [str(manifest_path.resolve())]
    assert snapshot["train_meta"]["requested_extra_corpus_files"] == []


def test_cli_resume_reverifies_manifest_backed_training_without_repassing_manifest(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.txt"
    extra_path = tmp_path / "extras" / "extra.txt"
    extra_path.parent.mkdir()
    manifest_path = tmp_path / "extras.txt"
    out_dir = tmp_path / "resume_manifest"
    base_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    extra_path.write_text(("xyz" * 40) + "\n", encoding="utf-8")
    manifest_path.write_text("extras/extra.txt\n", encoding="utf-8")

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        first_out = io.StringIO()
        with redirect_stdout(first_out):
            first_rc = lm_main.main(
                [
                    "train",
                    "--corpus-file",
                    str(base_path),
                    "--extra-corpus-manifest",
                    str(manifest_path),
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
        resume_out = io.StringIO()
        with redirect_stdout(resume_out):
            resume_rc = lm_main.main(
                [
                    "train",
                    "--resume-checkpoint",
                    str(out_dir / lm_main.TRAIN_STATE_NAME),
                    "--out",
                    str(out_dir),
                    "--max-iters",
                    "4",
                    "--val-frac",
                    "0.2",
                ]
            )

    assert first_rc in {0, 2}
    assert resume_rc in {0, 2}
    assert "[manifest] unverified (no sibling bundle) path=extras.txt" in first_out.getvalue()
    assert "[manifest] unverified (no sibling bundle) path=extras.txt" in resume_out.getvalue()
    snapshot = torch.load(out_dir / lm_main.TRAIN_STATE_NAME, map_location="cpu", weights_only=False)
    verdict = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    expected_manifest_verification = [
        {
            "status": "unverified",
            "manifest_path": str(manifest_path.resolve()),
            "reason": "no sibling bundle",
        }
    ]
    assert snapshot["train_meta"]["manifest_verification"] == expected_manifest_verification
    assert verdict["manifest_verification"] == expected_manifest_verification
    assert snapshot["train_meta"]["extra_corpus_manifests"] == [str(manifest_path.resolve())]
    assert snapshot["train_meta"]["extra_corpus_files"] == [str(extra_path.resolve())]


def test_cli_resume_preserves_legacy_manifest_verification_without_saved_manifest_inputs(
    tmp_path: Path,
) -> None:
    corpus_path = tmp_path / "tiny.txt"
    out_dir = tmp_path / "legacy_resume"
    corpus_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    text = corpus_path.read_text(encoding="utf-8")
    torch.manual_seed(11)
    tok = CharTokenizer.from_text(text)
    preset = _tiny_presets()["smoke"]
    model = CharGPT(
        GPTConfig(
            vocab_size=tok.vocab_size,
            block_size=int(preset["block_size"]),
            n_layer=int(preset["n_layer"]),
            n_head=int(preset["n_head"]),
            n_embd=int(preset["n_embd"]),
            dropout=float(preset["dropout"]),
        )
    )
    trainer = Trainer(
        model,
        TrainConfig(
            max_iters=2,
            lr_decay_iters=2,
            warmup_iters=0,
            batch_size=4,
            eval_interval=1,
            eval_iters=1,
            seed=11,
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    expected_manifest_verification = [
        {
            "status": "unverified",
            "manifest_path": str((tmp_path / "legacy_manifest.txt").resolve()),
            "reason": "no sibling bundle",
        }
    ]
    _write_snapshot(
        out_dir / lm_main.TRAIN_STATE_NAME,
        corpus_path,
        trainer=trainer,
        model=model,
        tok=tok,
        text=text,
        extra_corpus_files=[],
        manifest_verification=expected_manifest_verification,
    )

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        resume_out = io.StringIO()
        with redirect_stdout(resume_out):
            resume_rc = lm_main.main(
                [
                    "train",
                    "--resume-checkpoint",
                    str(out_dir / lm_main.TRAIN_STATE_NAME),
                    "--out",
                    str(out_dir),
                    "--max-iters",
                    "4",
                    "--val-frac",
                    "0.2",
                ]
            )

    assert resume_rc in {0, 2}
    assert "[manifest] unverified (no sibling bundle) path=legacy_manifest.txt" in resume_out.getvalue()
    snapshot = torch.load(out_dir / lm_main.TRAIN_STATE_NAME, map_location="cpu", weights_only=False)
    verdict = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    assert snapshot["train_meta"]["manifest_verification"] == expected_manifest_verification
    assert verdict["manifest_verification"] == expected_manifest_verification


def test_cli_eval_reports_oov_chars_for_extra_corpus(tmp_path: Path) -> None:
    base_path = tmp_path / "base.txt"
    extra_path = tmp_path / "extra.txt"
    out_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval"
    base_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    extra_path.write_text(("xyz" * 20) + "\n", encoding="utf-8")

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        lm_main.main(
            [
                "train",
                "--corpus-file",
                str(base_path),
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
        err = io.StringIO()
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = lm_main.main(
                [
                    "eval",
                    str(out_dir / "model.pt"),
                    "--corpus-file",
                    str(base_path),
                    "--extra-corpus-file",
                    str(extra_path),
                    "--out",
                    str(eval_dir),
                ]
            )

    assert rc in {0, 2}
    assert "out-of-vocabulary chars" in err.getvalue()
    verdict = json.loads((eval_dir / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["oov_chars"] > 0
    assert verdict["oov_rate"] > 0
    assert verdict["extra_corpus_files"] == [str(extra_path)]


def test_cli_eval_supports_extra_corpus_manifest(tmp_path: Path) -> None:
    base_path = tmp_path / "base.txt"
    extra_path = tmp_path / "extras" / "extra.txt"
    extra_path.parent.mkdir()
    manifest_path = tmp_path / "extras.txt"
    out_dir = tmp_path / "train"
    eval_dir = tmp_path / "eval_manifest"
    base_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    extra_path.write_text(("xyz" * 20) + "\n", encoding="utf-8")
    manifest_path.write_text("extras/extra.txt\n", encoding="utf-8")

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        lm_main.main(
            [
                "train",
                "--corpus-file",
                str(base_path),
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
        err = io.StringIO()
        out = io.StringIO()
        with redirect_stdout(out), redirect_stderr(err):
            rc = lm_main.main(
                [
                    "eval",
                    str(out_dir / "model.pt"),
                    "--corpus-file",
                    str(base_path),
                    "--extra-corpus-manifest",
                    str(manifest_path),
                    "--out",
                    str(eval_dir),
                ]
            )

    assert rc in {0, 2}
    assert "[manifest] unverified (no sibling bundle) path=extras.txt" in out.getvalue()
    verdict = json.loads((eval_dir / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["extra_corpus_files"] == [str(extra_path.resolve())]
    assert "out-of-vocabulary chars" in err.getvalue()
    assert verdict["manifest_verification"] == [
        {
            "status": "unverified",
            "manifest_path": str(manifest_path.resolve()),
            "reason": "no sibling bundle",
        }
    ]


def test_probe_written_manifest_round_trips_into_train(tmp_path: Path) -> None:
    base_path = tmp_path / "base.txt"
    keep_path = tmp_path / "extras" / "keep.txt"
    drop_path = tmp_path / "extras" / "drop.txt"
    keep_path.parent.mkdir()
    manifest_path = tmp_path / "selected.txt"
    out_dir = tmp_path / "roundtrip"
    base_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    keep_path.write_text(("abcde" * 20) + "\n", encoding="utf-8")
    drop_path.write_text(("xyz" * 20) + "\n", encoding="utf-8")

    p1_corpus_probe.main(
        [
            str(base_path),
            str(keep_path),
            str(drop_path),
            "--max-oov-rate",
            "0.2",
            "--max-new-chars",
            "1",
            "--write-manifest",
            str(manifest_path),
        ]
    )

    out = io.StringIO()
    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        with redirect_stdout(out):
            rc = lm_main.main(
                [
                    "train",
                    "--corpus-file",
                    str(base_path),
                    "--extra-corpus-manifest",
                    str(manifest_path),
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

    assert rc in {0, 2}
    assert "[manifest] verified selected.txt: entries=1 generated_by=scripts/p1_corpus_probe.py includes_base=True" in out.getvalue()
    snapshot = torch.load(out_dir / lm_main.TRAIN_STATE_NAME, map_location="cpu", weights_only=False)
    verdict = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    assert verdict["extra_corpus_files"] == [str(keep_path.resolve())]
    expected_manifest_verification = [
        {
            "status": "verified",
            "manifest_path": str(manifest_path.resolve()),
            "entry_count": 1,
            "generated_by": "scripts/p1_corpus_probe.py",
            "includes_base": True,
            "combined_sha256": build_utf8_corpus_bundle(
                [keep_path], base_file=base_path
            )["combined"]["sha256"],
            "bundle_sha256": build_utf8_corpus_bundle(
                [keep_path], base_file=base_path
            )["bundle_sha256"],
        }
    ]
    assert verdict["manifest_verification"] == expected_manifest_verification
    assert snapshot["train_meta"]["manifest_verification"] == expected_manifest_verification


def test_cli_train_rejects_drifted_probe_bundle_metadata(tmp_path: Path) -> None:
    base_path = tmp_path / "base.txt"
    extra_path = tmp_path / "extras" / "keep.txt"
    extra_path.parent.mkdir()
    manifest_path = tmp_path / "selected.txt"
    out_dir = tmp_path / "drifted_bundle"
    base_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    extra_path.write_text(("abcde" * 20) + "\n", encoding="utf-8")

    p1_corpus_probe.main(
        [
            str(base_path),
            str(extra_path),
            "--max-oov-rate",
            "0.2",
            "--max-new-chars",
            "1",
            "--write-manifest",
            str(manifest_path),
        ]
    )
    manifest_path.write_text(
        "# Generated by scripts/p1_corpus_probe.py\nextras/keep.txt\n# drift\n",
        encoding="utf-8",
    )

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        try:
            lm_main.main(
                [
                    "train",
                    "--corpus-file",
                    str(base_path),
                    "--extra-corpus-manifest",
                    str(manifest_path),
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
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected drifted manifest bundle to fail closed")

    assert "manifest bundle metadata drift" in message


def test_prepare_written_manifest_round_trips_into_train_and_eval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base_path = tmp_path / "base.txt"
    extras_dir = tmp_path / "prepared"
    manifest_path = tmp_path / "prepared_extras.txt"
    out_dir = tmp_path / "prepared_roundtrip"
    eval_dir = tmp_path / "prepared_eval"
    base_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")

    class _StaticOpener:
        def open(self, url: str, timeout: float = 30.0) -> object:
            class _Resp:
                def __enter__(self) -> "_Resp":
                    return self

                def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
                    return None

                def read(self) -> bytes:
                    return p1_prepare_aozora_test_payload

            return _Resp()

    p1_prepare_aozora_test_payload = _aozora_zip_bytes()
    monkeypatch.setattr(
        p1_prepare_aozora.urllib.request,  # type: ignore[attr-defined]
        "build_opener",
        lambda *handlers: _StaticOpener(),
    )
    p1_prepare_aozora.main(
        [
            "--out-dir",
            str(extras_dir),
            "--write-manifest",
            str(manifest_path),
            "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip",
        ]
    )

    train_out = io.StringIO()
    eval_out = io.StringIO()
    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        with redirect_stdout(train_out):
            train_rc = lm_main.main(
                [
                    "train",
                    "--corpus-file",
                    str(base_path),
                    "--extra-corpus-manifest",
                    str(manifest_path),
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
        with redirect_stdout(eval_out):
            eval_rc = lm_main.main(
                [
                    "eval",
                    str(out_dir / "model.pt"),
                    "--corpus-file",
                    str(base_path),
                    "--extra-corpus-manifest",
                    str(manifest_path),
                    "--out",
                    str(eval_dir),
                ]
            )

    assert train_rc in {0, 2}
    assert eval_rc in {0, 2}
    assert "[manifest] verified prepared_extras.txt: entries=1 generated_by=scripts/p1_prepare_aozora.py includes_base=False" in train_out.getvalue()
    assert "[manifest] verified prepared_extras.txt: entries=1 generated_by=scripts/p1_prepare_aozora.py includes_base=False" in eval_out.getvalue()
    generated_extra = extras_dir / "aozora_000148_789_ruby_5639.txt"
    train_verdict = json.loads((out_dir / "verdict.json").read_text(encoding="utf-8"))
    eval_verdict = json.loads((eval_dir / "verdict.json").read_text(encoding="utf-8"))
    assert train_verdict["extra_corpus_files"] == [str(generated_extra.resolve())]
    assert eval_verdict["extra_corpus_files"] == [str(generated_extra.resolve())]
    expected_manifest_verification = [
        {
            "status": "verified",
            "manifest_path": str(manifest_path.resolve()),
            "entry_count": 1,
            "generated_by": "scripts/p1_prepare_aozora.py",
            "includes_base": False,
            "combined_sha256": build_utf8_corpus_bundle([generated_extra])["combined"]["sha256"],
            "bundle_sha256": build_utf8_corpus_bundle([generated_extra])["bundle_sha256"],
        }
    ]
    assert train_verdict["manifest_verification"] == expected_manifest_verification
    assert eval_verdict["manifest_verification"] == expected_manifest_verification


def test_cli_train_rejects_manifest_with_base_or_duplicate_entries(
    tmp_path: Path,
) -> None:
    base_path = tmp_path / "base.txt"
    extra_path = tmp_path / "extras" / "keep.txt"
    extra_path.parent.mkdir()
    manifest_path = tmp_path / "selected.txt"
    out_dir = tmp_path / "dedup_summary"
    base_path.write_text(("abcdefg" * 40) + "\n", encoding="utf-8")
    extra_path.write_text(("abcde" * 20) + "\n", encoding="utf-8")
    manifest_path.write_text("extras/keep.txt\nbase.txt\nextras/keep.txt\n", encoding="utf-8")
    bundle_payload = {
        "generated_by": "manual-test",
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_text(manifest_path.read_text(encoding="utf-8")),
        "bundle": build_utf8_corpus_bundle([extra_path, base_path, extra_path], base_file=base_path),
    }
    manifest_path.with_suffix(".txt.bundle.json").write_text(
        json.dumps(bundle_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    out = io.StringIO()

    with patch.dict(lm_main.MODEL_PRESETS, _tiny_presets(), clear=True):
        try:
            with redirect_stdout(out):
                lm_main.main(
                    [
                        "train",
                        "--corpus-file",
                        str(base_path),
                        "--extra-corpus-manifest",
                        str(manifest_path),
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
        except ValueError as exc:
            message = str(exc)
        else:
            raise AssertionError("expected manifest base/duplicate entries to fail closed")

    assert "collapse after base/duplicate filtering" in message
