# SPDX-License-Identifier: Apache-2.0
"""CLI for the char-LM: ``py -3.11 -m llcore.lm <command>``.

Commands
--------
train     Fetch/load a corpus, train, evaluate vs unigram, sample, export viz JSON.
generate  Sample text from a saved checkpoint.
export    Write a saved checkpoint to bbycroft/llm-viz model JSON.

CPU-only. ``train`` prints an honest verdict (unigram PPL, model PPL, ratio, a
sample, and the degeneracy check) and writes it to ``<out>/verdict.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import torch

from llcore.lm.corpus import join_corpus_parts, read_utf8_corpus_file, resolve_extra_corpus_files
from llcore.lm.data import (
    encode_corpus,
    fetch_aozora_text,
    fetch_tinyshakespeare,
    train_val_split,
)
from llcore.lm.eval import held_out_report, passes_gate
from llcore.lm.export import save_viz_json
from llcore.lm.generation import generate_text, is_degenerate
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.quant import (
    INT8_CKPT_KIND,
    int8_footprint_bytes,
    load_int8_model,
    save_int8_checkpoint,
)
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import Trainer, TrainConfig

# Model presets (see docs/LM_P0_PLAN.md).
MODEL_PRESETS: dict[str, dict[str, int | float]] = {
    "smoke": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 64, "dropout": 0.0},
    "p1": {"n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 256, "dropout": 0.2},
}
TRAIN_DEFAULTS = {
    "corpus": "shakespeare",
    "config": "smoke",
    "out": "out/lm_run",
    "max_iters": 2000,
    "batch_size": 12,
    "eval_iters": 20,
    "val_frac": 0.1,
    "seed": 1337,
}
TRAIN_STATE_NAME = "train_state.pt"
_TRAIN_STATE_TMP_COUNTER = itertools.count()


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - best-effort console setup
        pass


def _base_corpus_text(corpus: str, corpus_file: str | None) -> str:
    if corpus_file:
        return read_utf8_corpus_file(corpus_file)
    if corpus == "shakespeare":
        return fetch_tinyshakespeare()
    if corpus == "aozora":
        return fetch_aozora_text()
    raise ValueError(f"unknown corpus {corpus!r}")


def _load_corpus(
    corpus: str,
    corpus_file: str | None,
    extra_corpus_files: list[str] | None = None,
) -> str:
    parts = [_base_corpus_text(corpus, corpus_file)]
    for path in extra_corpus_files or []:
        parts.append(read_utf8_corpus_file(path))
    return join_corpus_parts(parts)


def _save_checkpoint(path: Path, model: CharGPT, tok: CharTokenizer) -> None:
    torch.save(
        {
            "config": vars(model.config),
            "model_state": model.state_dict(),
            "itos": tok.itos,
        },
        path,
    )


def _load_checkpoint(path: Path) -> tuple[CharGPT, CharTokenizer]:
    """Load a checkpoint in tensor-only mode."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    config = GPTConfig(**ckpt["config"])
    model = CharGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["itos"])


def _load_any_checkpoint(path: Path) -> tuple[CharGPT, CharTokenizer]:
    """Load an fp32 or int8 checkpoint, transparently. int8 uses mmap streaming-dequant.

    int8 checkpoints (``kind == INT8_CKPT_KIND``, produced by ``quantize``) are loaded
    via :func:`~llcore.lm.quant.load_int8_model` so cold weight pages stay on disk; fp32
    checkpoints fall back to the dense loader. The kind probe is cheap (mmap = lazy).
    """
    head = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if isinstance(head, dict) and head.get("kind") == INT8_CKPT_KIND:
        model, itos = load_int8_model(path, mmap=True)
        return model, CharTokenizer(itos)
    return _load_checkpoint(path)


def _save_training_snapshot(
    path: Path,
    model: CharGPT,
    tok: CharTokenizer,
    trainer: Trainer,
    *,
    corpus: str,
    corpus_file: str | None,
    corpus_sha256: str,
    config_name: str,
    val_frac: float,
    requested_extra_corpus_files: list[str],
    extra_corpus_manifests: list[str],
    extra_corpus_files: list[str],
    manifest_verification: list[dict[str, Any]],
) -> None:
    tmp_path = path.with_name(
        f"{path.name}.{os.getpid()}.{next(_TRAIN_STATE_TMP_COUNTER)}.tmp"
    )
    try:
        # This only targets rename atomicity against interrupted process writes.
        # It intentionally does not fsync, so power-loss durability is out of scope.
        torch.save(
            {
                "kind": "llcore.lm.train_state.v1",
                "config": vars(model.config),
                "model_state": model.state_dict(),
                "itos": tok.itos,
                "trainer_state": trainer.state_dict(),
                "train_config": vars(trainer.cfg),
                "train_meta": {
                    "corpus": corpus,
                    "corpus_file": corpus_file,
                    "requested_extra_corpus_files": list(requested_extra_corpus_files),
                    "extra_corpus_manifests": list(extra_corpus_manifests),
                    "extra_corpus_files": list(extra_corpus_files),
                    "manifest_verification": [dict(item) for item in manifest_verification],
                    "corpus_sha256": corpus_sha256,
                    "config": config_name,
                    "val_frac": val_frac,
                },
            },
            tmp_path,
        )
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _restore_training_snapshot(
    path: Path,
) -> tuple[CharGPT, CharTokenizer, dict[str, object], TrainConfig, dict[str, Any]]:
    """Load a training snapshot in tensor-only mode."""
    snapshot = torch.load(path, map_location="cpu", weights_only=True)
    if snapshot.get("kind") != "llcore.lm.train_state.v1":
        raise ValueError(f"{path} is not a llcore.lm training snapshot")
    model = CharGPT(GPTConfig(**snapshot["config"]))
    model.load_state_dict(snapshot["model_state"])
    tok = CharTokenizer(snapshot["itos"])
    trainer_state = cast(dict[str, object], snapshot["trainer_state"])
    train_cfg = TrainConfig(**snapshot["train_config"])
    train_meta = cast(dict[str, Any], snapshot["train_meta"])
    return model, tok, trainer_state, train_cfg, train_meta


def _resolve_arg(value: Any, default: Any) -> Any:
    return default if value is None else value


def _corpus_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _count_oov_chars(tok: CharTokenizer, text: str) -> int:
    return sum(1 for ch in text if ch not in tok.stoi)


def _resolve_requested_extra_corpus_files(
    extra_files: list[str] | None,
) -> list[str]:
    return [str(Path(path).resolve()) for path in extra_files or []]


def _resolve_extra_corpus_manifests(
    manifest_files: list[str] | None,
) -> list[str]:
    return [str(Path(path).resolve()) for path in manifest_files or []]


def _resume_value(field: str, current: Any, saved: Any) -> Any:
    if current is None:
        return saved
    if current != saved:
        raise ValueError(f"--{field.replace('_', '-')}={current!r} does not match snapshot")
    return current


def _resume_float(field: str, current: float | None, saved: float) -> float:
    if current is None:
        return saved
    if current != saved:
        raise ValueError(f"--{field.replace('_', '-')}={current!r} does not match snapshot")
    return current


def _warn_ignored_resume_arg(name: str, value: Any, saved: Any) -> None:
    if value is not None and value != saved:
        print(
            f"[resume] ignoring --{name.replace('_', '-')}={value!r}; "
            f"snapshot value {saved!r} remains in effect"
        )


def _print_verified_manifest_summaries(summaries: list[dict[str, Any]]) -> None:
    for summary in summaries:
        if summary.get("status") == "unverified":
            print(
                "[manifest] unverified "
                f"(no sibling bundle) path={Path(str(summary['manifest_path'])).name}"
            )
            continue
        combined_sha = summary["combined_sha256"]
        print(
            "[manifest] verified "
            f"{Path(str(summary['manifest_path'])).name}: "
            f"entries={summary['entry_count']} "
            f"generated_by={summary['generated_by']} "
            f"includes_base={summary['includes_base']} "
            f"combined_sha256={str(combined_sha)[:12] if combined_sha else '(empty)'} "
            f"bundle_sha256={str(summary['bundle_sha256'])[:12]}"
        )


def cmd_train(args: argparse.Namespace) -> int:
    if args.resume_checkpoint:
        resume_path = Path(args.resume_checkpoint)
        model, tok, trainer_state, train_cfg, train_meta = _restore_training_snapshot(resume_path)
        corpus = _resume_value("corpus", args.corpus, cast(str | None, train_meta["corpus"]))
        corpus_file = _resume_value(
            "corpus_file", args.corpus_file, cast(str | None, train_meta["corpus_file"])
        )
        requested_extra_corpus_files = _resolve_requested_extra_corpus_files(
            cast(list[str] | None, args.extra_corpus_file)
        )
        extra_corpus_manifests = _resolve_extra_corpus_manifests(
            cast(list[str] | None, args.extra_corpus_manifest)
        )
        saved_requested_extra_corpus_files = cast(
            list[str], train_meta.get("requested_extra_corpus_files", [])
        )
        has_saved_requested_extra_corpus_files = "requested_extra_corpus_files" in train_meta
        saved_extra_corpus_manifests = cast(
            list[str], train_meta.get("extra_corpus_manifests", [])
        )
        has_saved_extra_corpus_manifests = "extra_corpus_manifests" in train_meta
        if has_saved_requested_extra_corpus_files:
            requested_extra_corpus_files = cast(
                list[str],
                _resume_value(
                    "extra_corpus_file",
                    requested_extra_corpus_files or None,
                    saved_requested_extra_corpus_files,
                ),
            )
        if has_saved_extra_corpus_manifests:
            extra_corpus_manifests = cast(
                list[str],
                _resume_value(
                    "extra_corpus_manifest",
                    extra_corpus_manifests or None,
                    saved_extra_corpus_manifests,
                ),
            )
        saved_manifest_verification = cast(
            list[dict[str, Any]], train_meta.get("manifest_verification", [])
        )
        if extra_corpus_manifests:
            verified_manifest_summaries: list[dict[str, Any]] = []
            requested_extra_files = resolve_extra_corpus_files(
                requested_extra_corpus_files,
                extra_corpus_manifests,
                base_file=corpus_file,
                verified_bundle_summaries=verified_manifest_summaries,
            )
        else:
            verified_manifest_summaries = [dict(item) for item in saved_manifest_verification]
            requested_extra_files = cast(
                list[str],
                _resume_value(
                    "extra_corpus_files",
                    requested_extra_corpus_files or None,
                    cast(list[str], train_meta["extra_corpus_files"]),
                ),
            )
        _print_verified_manifest_summaries(verified_manifest_summaries)
        extra_corpus_files = cast(
            list[str],
            _resume_value(
                "extra_corpus_files",
                requested_extra_files or None,
                cast(list[str], train_meta["extra_corpus_files"]),
            ),
        )
        config_name = _resume_value("config", args.config, cast(str | None, train_meta["config"]))
        if corpus is None or config_name is None:
            raise ValueError("training snapshot is missing corpus/config metadata")
        out = Path(_resolve_arg(args.out, str(resume_path.resolve().parent)))
        seed = train_cfg.seed
        val_frac = _resume_float(
            "val_frac", cast(float | None, args.val_frac), float(train_meta["val_frac"])
        )
        snapshot_max_iters = train_cfg.max_iters
        if args.max_iters is not None and args.max_iters != snapshot_max_iters:
            if args.max_iters < snapshot_max_iters:
                raise ValueError(
                    f"--max-iters={args.max_iters} is below snapshot max_iters={snapshot_max_iters}"
                )
            train_cfg.max_iters = args.max_iters
            if train_cfg.lr_decay_iters == snapshot_max_iters:
                train_cfg.lr_decay_iters = args.max_iters
            else:
                print(
                    f"[resume] extended max_iters to {args.max_iters}; "
                    f"lr_decay_iters stays at snapshot value {train_cfg.lr_decay_iters}"
                )
        _warn_ignored_resume_arg("batch_size", args.batch_size, train_cfg.batch_size)
        _warn_ignored_resume_arg("eval_iters", args.eval_iters, train_cfg.eval_iters)
        _warn_ignored_resume_arg("seed", args.seed, seed)
        _warn_ignored_resume_arg("dropout", args.dropout, model.config.dropout)
        trainer = Trainer(model, train_cfg)
        trainer.load_state_dict(trainer_state)
        resume_already_complete = trainer.iter_num >= trainer.cfg.max_iters
        if resume_already_complete:
            print(
                f"[resume] checkpoint already completed at iter {trainer.iter_num}; "
                "re-emitting artifacts from the saved state"
            )
        print(f"[resume] {resume_path} -> iter {trainer.iter_num}/{trainer.cfg.max_iters}")
    else:
        corpus = str(_resolve_arg(args.corpus, TRAIN_DEFAULTS["corpus"]))
        corpus_file = cast(str | None, args.corpus_file)
        requested_extra_corpus_files = _resolve_requested_extra_corpus_files(
            cast(list[str] | None, args.extra_corpus_file)
        )
        extra_corpus_manifests = _resolve_extra_corpus_manifests(
            cast(list[str] | None, args.extra_corpus_manifest)
        )
        verified_manifest_summaries = []
        extra_corpus_files = resolve_extra_corpus_files(
            requested_extra_corpus_files,
            extra_corpus_manifests,
            base_file=corpus_file,
            verified_bundle_summaries=verified_manifest_summaries,
        )
        _print_verified_manifest_summaries(verified_manifest_summaries)
        config_name = str(_resolve_arg(args.config, TRAIN_DEFAULTS["config"]))
        out = Path(_resolve_arg(args.out, TRAIN_DEFAULTS["out"]))
        max_iters = int(_resolve_arg(args.max_iters, TRAIN_DEFAULTS["max_iters"]))
        batch_size = int(_resolve_arg(args.batch_size, TRAIN_DEFAULTS["batch_size"]))
        eval_iters = int(_resolve_arg(args.eval_iters, TRAIN_DEFAULTS["eval_iters"]))
        val_frac = float(_resolve_arg(args.val_frac, TRAIN_DEFAULTS["val_frac"]))
        seed = int(_resolve_arg(args.seed, TRAIN_DEFAULTS["seed"]))
        torch.manual_seed(seed)

    out.mkdir(parents=True, exist_ok=True)
    snapshot_path = out / TRAIN_STATE_NAME

    print(f"[corpus] loading ({corpus_file or corpus}) ...")
    text = _load_corpus(corpus, corpus_file, extra_corpus_files)
    corpus_sha256 = _corpus_sha256(text)
    if args.resume_checkpoint:
        expected_sha256 = cast(str, train_meta["corpus_sha256"])
        if corpus_sha256 != expected_sha256:
            raise ValueError(
                "corpus contents no longer match training snapshot "
                f"(expected sha256={expected_sha256}, got {corpus_sha256})"
            )
    else:
        tok = CharTokenizer.from_text(text)
        preset = MODEL_PRESETS[config_name]
        dropout = args.dropout if args.dropout is not None else float(preset["dropout"])
        model = CharGPT(
            GPTConfig(
                vocab_size=tok.vocab_size,
                block_size=int(preset["block_size"]),
                n_layer=int(preset["n_layer"]),
                n_head=int(preset["n_head"]),
                n_embd=int(preset["n_embd"]),
                dropout=dropout,
            )
        )
        trainer = Trainer(
            model,
            TrainConfig(
                max_iters=max_iters,
                lr_decay_iters=max_iters,
                warmup_iters=min(100, max_iters // 10),
                batch_size=batch_size,
                eval_interval=max(1, max_iters // 8),
                eval_iters=eval_iters,
                seed=seed,
            ),
        )
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=val_frac)
    print(
        f"[corpus] chars={len(text):,}  vocab={tok.vocab_size}  "
        f"train={train_ids.numel():,}  val={val_ids.numel():,}"
    )
    print(
        f"[model] {config_name}: {model.num_params(False):,} params  "
        f"L{model.config.n_layer} H{model.config.n_head} D{model.config.n_embd} "
        f"ctx{model.config.block_size} dropout{model.config.dropout}"
    )

    def on_eval(it: int, tr: float, va: float) -> None:
        print(f"[train] iter {it:>5}  train_loss {tr:.4f}  val_loss {va:.4f}")
        _save_training_snapshot(
            snapshot_path,
            model,
            tok,
            trainer,
            corpus=corpus,
            corpus_file=corpus_file,
            corpus_sha256=corpus_sha256,
            config_name=config_name,
            val_frac=val_frac,
            requested_extra_corpus_files=requested_extra_corpus_files,
            extra_corpus_manifests=extra_corpus_manifests,
            extra_corpus_files=extra_corpus_files,
            manifest_verification=verified_manifest_summaries,
        )

    best_val: float | None
    if args.resume_checkpoint and resume_already_complete:
        best_val = cast(float | None, trainer.best_val)
        assert best_val is None or isinstance(best_val, float)
    else:
        result = trainer.train(train_ids, val_ids, on_eval=on_eval)
        best_val = cast(float, result["best_val_loss"])
        assert isinstance(best_val, float)

    verdict = _evaluate(
        model,
        tok,
        train_ids,
        val_ids,
        corpus=corpus_file or corpus,
        corpus_sha256=corpus_sha256,
        config=config_name,
        max_iters=trainer.cfg.max_iters,
        best_val_loss=best_val,
        seed=seed,
        extra_corpus_files=extra_corpus_files,
        manifest_verification=verified_manifest_summaries,
        oov_chars=0,
    )
    _emit_artifacts(out, model, tok, verdict)
    return 0 if verdict["overall_pass"] else 2


def _evaluate(
    model: CharGPT,
    tok: CharTokenizer,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    *,
    corpus: str,
    corpus_sha256: str,
    config: str,
    max_iters: int,
    best_val_loss: float | None,
    seed: int,
    extra_corpus_files: list[str],
    manifest_verification: list[dict[str, Any]],
    oov_chars: int,
) -> dict[str, object]:
    """Run the held-out-vs-unigram report + a sampled degeneracy check; build verdict."""
    report = held_out_report(model, train_ids, val_ids, tok.vocab_size, model.config.block_size)
    unigram_ppl = report["unigram_ppl"]
    model_ppl = report["model_ppl"]
    seed_char = "\n" if "\n" in tok.stoi else tok.itos[0]
    sample = generate_text(
        model, tok, prompt=seed_char, max_new_tokens=400, temperature=0.8, top_k=40, seed=seed
    )
    min_distinct = 15 if tok.vocab_size >= 20 else max(3, tok.vocab_size // 2)
    degenerate = is_degenerate(sample, min_distinct=min_distinct)
    gate = passes_gate(model_ppl, unigram_ppl)
    oov_rate = oov_chars / max(1, len(train_ids) + len(val_ids))
    return {
        "corpus": corpus,
        "corpus_sha256": corpus_sha256,
        "config": config,
        "extra_corpus_files": list(extra_corpus_files),
        "manifest_verification": [dict(item) for item in manifest_verification],
        "vocab_size": tok.vocab_size,
        "n_params": model.num_params(False),
        "max_iters": max_iters,
        "best_val_loss": best_val_loss if best_val_loss is not None else round(report["model_nll"], 4),
        # unigram is scored on the *exact same* target tokens as the model (airtight).
        "eval_method": "non-overlapping windows; unigram scored on identical target tokens",
        "n_eval_tokens": int(report["n_tokens"]),
        "unigram_ppl": round(unigram_ppl, 4),
        "model_ppl": round(model_ppl, 4),
        "ratio_model_over_unigram": round(model_ppl / unigram_ppl, 4),
        "ppl_gate_pass": gate,
        "degenerate_sample": degenerate,
        "oov_chars": oov_chars,
        "oov_rate": round(oov_rate, 6),
        "overall_pass": bool(gate and not degenerate),
        "sample_head": sample[:200],
    }


def _emit_artifacts(
    out: Path, model: CharGPT, tok: CharTokenizer, verdict: dict[str, object]
) -> None:
    """Write verdict.json + checkpoint + tokenizer + viz JSON, and print the verdict."""
    (out / "verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_checkpoint(out / "model.pt", model, tok)
    tok.save(out / "tokenizer.json")
    save_viz_json(model, out / "model_viz.json", tok, include_vocab=True)
    upll = float(verdict["unigram_ppl"])  # type: ignore[arg-type]
    mpll = float(verdict["model_ppl"])  # type: ignore[arg-type]
    print("\n=== P0 VERDICT ===")
    print(f"  eval tokens : {verdict['n_eval_tokens']} (unigram on identical tokens)")
    print(f"  unigram PPL : {upll:.3f}")
    print(f"  model   PPL : {mpll:.3f}  ({mpll / upll:.3f}x unigram)")
    print(f"  PPL gate    : {'PASS' if verdict['ppl_gate_pass'] else 'FAIL'} (model <= 0.85x)")
    print(f"  degenerate  : {verdict['degenerate_sample']}")
    print(f"  OVERALL     : {'PASS' if verdict['overall_pass'] else 'FAIL'}")
    print(f"  artifacts   : {out}/ (verdict.json, model.pt, model_viz.json)")
    print("\n--- sample (first 200 chars) ---")
    print(verdict["sample_head"])


def cmd_generate(args: argparse.Namespace) -> int:
    # Accept either an fp32 or an int8 (streaming/mmap) checkpoint.
    model, tok = _load_any_checkpoint(Path(args.checkpoint))
    print(
        generate_text(
            model, tok, prompt=args.prompt, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        )
    )
    return 0


def cmd_quantize(args: argparse.Namespace) -> int:
    """Quantize an fp32 checkpoint to an int8 (streaming/mmap) checkpoint."""
    model, tok = _load_checkpoint(Path(args.checkpoint))
    out = Path(args.out) if args.out else Path(args.checkpoint).with_name("model_int8.pt")
    save_int8_checkpoint(model, out, tok.itos)
    fp = int8_footprint_bytes(model)
    ratio = fp["int8_bytes"] / fp["fp32_bytes"]
    print(
        f"wrote {out}  fp32 weights={fp['fp32_bytes'] / 1e6:.1f}MB  "
        f"int8 resident={fp['int8_bytes'] / 1e6:.1f}MB  (ratio {ratio:.3f}, "
        f"{(1 - ratio) * 100:.1f}% smaller)"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Re-evaluate a saved checkpoint against the corpus (no training)."""
    model, tok = _load_checkpoint(Path(args.checkpoint))
    corpus = str(_resolve_arg(args.corpus, TRAIN_DEFAULTS["corpus"]))
    verified_manifest_summaries: list[dict[str, Any]] = []
    extra_corpus_files = resolve_extra_corpus_files(
        cast(list[str] | None, args.extra_corpus_file),
        cast(list[str] | None, args.extra_corpus_manifest),
        base_file=args.corpus_file,
        verified_bundle_summaries=verified_manifest_summaries,
    )
    _print_verified_manifest_summaries(verified_manifest_summaries)
    text = _load_corpus(corpus, args.corpus_file, extra_corpus_files)
    corpus_sha256 = _corpus_sha256(text)
    oov_chars = _count_oov_chars(tok, text)
    if oov_chars > 0:
        print(
            f"[eval] warning: {oov_chars} out-of-vocabulary chars will map to token 0 "
            f"({oov_chars / max(1, len(text)):.6f} of input chars)",
            file=sys.stderr,
        )
    ids = torch.tensor(tok.encode_safe(text), dtype=torch.long)
    train_ids, val_ids = train_val_split(
        ids, val_frac=float(_resolve_arg(args.val_frac, TRAIN_DEFAULTS["val_frac"]))
    )
    out = Path(args.out) if args.out else Path(args.checkpoint).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    verdict = _evaluate(
        model,
        tok,
        train_ids,
        val_ids,
        corpus=args.corpus_file or corpus,
        corpus_sha256=corpus_sha256,
        config="(eval)",
        max_iters=0,
        best_val_loss=None,
        seed=int(_resolve_arg(args.seed, TRAIN_DEFAULTS["seed"])),
        extra_corpus_files=extra_corpus_files,
        manifest_verification=verified_manifest_summaries,
        oov_chars=oov_chars,
    )
    _emit_artifacts(out, model, tok, verdict)
    return 0 if verdict["overall_pass"] else 2


def cmd_export(args: argparse.Namespace) -> int:
    model, tok = _load_checkpoint(Path(args.checkpoint))
    save_viz_json(model, args.out, tok, include_vocab=True)
    print(f"wrote {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="llcore.lm", description="CPU char-LM (P0)")
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("train", help="train + evaluate vs unigram + export")
    t.add_argument("--corpus", choices=["shakespeare", "aozora"], default=None)
    t.add_argument("--corpus-file", default=None, help="local UTF-8 corpus (overrides --corpus)")
    t.add_argument(
        "--extra-corpus-file",
        action="append",
        default=None,
        help="additional UTF-8 corpus to append after the base corpus; may be repeated",
    )
    t.add_argument(
        "--extra-corpus-manifest",
        action="append",
        default=None,
        help="UTF-8 manifest listing extra corpus files (one path per line; # only as full-line comments)",
    )
    t.add_argument("--config", choices=list(MODEL_PRESETS), default=None)
    t.add_argument("--dropout", type=float, default=None, help="override preset dropout")
    t.add_argument("--out", default=None)
    t.add_argument("--max-iters", type=int, default=None)
    t.add_argument("--batch-size", type=int, default=None)
    t.add_argument("--eval-iters", type=int, default=None)
    t.add_argument("--val-frac", type=float, default=None)
    t.add_argument("--seed", type=int, default=None)
    t.add_argument(
        "--resume-checkpoint",
        default=None,
        help="resume from a saved train_state.pt with exact corpus/split/RNG checks",
    )
    t.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="re-evaluate a saved checkpoint vs unigram (no training)")
    ev.add_argument("checkpoint")
    ev.add_argument("--corpus", choices=["shakespeare", "aozora"], default=None)
    ev.add_argument("--corpus-file", default=None)
    ev.add_argument(
        "--extra-corpus-file",
        action="append",
        default=None,
        help="additional UTF-8 corpus to append after the base corpus; may be repeated",
    )
    ev.add_argument(
        "--extra-corpus-manifest",
        action="append",
        default=None,
        help="UTF-8 manifest listing extra corpus files (one path per line; # only as full-line comments)",
    )
    ev.add_argument("--out", default=None, help="defaults to the checkpoint's directory")
    ev.add_argument("--val-frac", type=float, default=None)
    ev.add_argument("--seed", type=int, default=None)
    ev.set_defaults(func=cmd_eval)

    g = sub.add_parser("generate", help="sample from a checkpoint")
    g.add_argument("checkpoint")
    g.add_argument("--prompt", default="\n")
    g.add_argument("--max-new-tokens", type=int, default=500)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--seed", type=int, default=None)
    g.set_defaults(func=cmd_generate)

    e = sub.add_parser("export", help="export a checkpoint to llm-viz JSON")
    e.add_argument("checkpoint")
    e.add_argument("--out", default="model_viz.json")
    e.set_defaults(func=cmd_export)

    q = sub.add_parser("quantize", help="quantize a checkpoint to int8 (streaming/mmap inference)")
    q.add_argument("checkpoint")
    q.add_argument("--out", default=None, help="defaults to <checkpoint dir>/model_int8.pt")
    q.set_defaults(func=cmd_quantize)
    return ap


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
