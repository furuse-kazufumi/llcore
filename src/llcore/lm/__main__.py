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
import json
import sys
from pathlib import Path

import torch

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
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import Trainer, TrainConfig

# Model presets (see docs/LM_P0_PLAN.md).
MODEL_PRESETS: dict[str, dict[str, int | float]] = {
    "smoke": {"n_layer": 4, "n_head": 4, "n_embd": 128, "block_size": 64, "dropout": 0.0},
    "p1": {"n_layer": 6, "n_head": 6, "n_embd": 384, "block_size": 256, "dropout": 0.2},
}


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001 - best-effort console setup
        pass


def _load_corpus(args: argparse.Namespace) -> str:
    if args.corpus_file:
        return Path(args.corpus_file).read_text(encoding="utf-8")
    if args.corpus == "shakespeare":
        return fetch_tinyshakespeare()
    if args.corpus == "aozora":
        return fetch_aozora_text()
    raise ValueError(f"unknown corpus {args.corpus!r}")


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
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = CharGPT(config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, CharTokenizer(ckpt["itos"])


def cmd_train(args: argparse.Namespace) -> int:
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[corpus] loading ({args.corpus_file or args.corpus}) ...")
    text = _load_corpus(args)
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=args.val_frac)
    print(
        f"[corpus] chars={len(text):,}  vocab={tok.vocab_size}  "
        f"train={train_ids.numel():,}  val={val_ids.numel():,}"
    )

    preset = MODEL_PRESETS[args.config]
    dropout = args.dropout if args.dropout is not None else float(preset["dropout"])
    model_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=int(preset["block_size"]),
        n_layer=int(preset["n_layer"]),
        n_head=int(preset["n_head"]),
        n_embd=int(preset["n_embd"]),
        dropout=dropout,
    )
    model = CharGPT(model_cfg)
    print(
        f"[model] {args.config}: {model.num_params(False):,} params  "
        f"L{model_cfg.n_layer} H{model_cfg.n_head} D{model_cfg.n_embd} "
        f"ctx{model_cfg.block_size} dropout{model_cfg.dropout}"
    )

    train_cfg = TrainConfig(
        max_iters=args.max_iters,
        lr_decay_iters=args.max_iters,
        warmup_iters=min(100, args.max_iters // 10),
        batch_size=args.batch_size,
        eval_interval=max(1, args.max_iters // 8),
        eval_iters=args.eval_iters,
        seed=args.seed,
    )

    def on_eval(it: int, tr: float, va: float) -> None:
        print(f"[train] iter {it:>5}  train_loss {tr:.4f}  val_loss {va:.4f}")

    result = Trainer(model, train_cfg).train(train_ids, val_ids, on_eval=on_eval)
    best_val = result["best_val_loss"]
    assert isinstance(best_val, float)

    verdict = _evaluate(
        model, tok, train_ids, val_ids,
        corpus=args.corpus_file or args.corpus, config=args.config,
        max_iters=args.max_iters, best_val_loss=best_val, seed=args.seed,
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
    config: str,
    max_iters: int,
    best_val_loss: float | None,
    seed: int,
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
    return {
        "corpus": corpus,
        "config": config,
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
    model, tok = _load_checkpoint(Path(args.checkpoint))
    print(
        generate_text(
            model, tok, prompt=args.prompt, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        )
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    """Re-evaluate a saved checkpoint against the corpus (no training)."""
    model, tok = _load_checkpoint(Path(args.checkpoint))
    text = _load_corpus(args)
    ids = torch.tensor(tok.encode_safe(text), dtype=torch.long)
    train_ids, val_ids = train_val_split(ids, val_frac=args.val_frac)
    out = Path(args.out) if args.out else Path(args.checkpoint).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    verdict = _evaluate(
        model, tok, train_ids, val_ids,
        corpus=args.corpus_file or args.corpus, config="(eval)",
        max_iters=0, best_val_loss=None, seed=args.seed,
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
    t.add_argument("--corpus", choices=["shakespeare", "aozora"], default="shakespeare")
    t.add_argument("--corpus-file", default=None, help="local UTF-8 corpus (overrides --corpus)")
    t.add_argument("--config", choices=list(MODEL_PRESETS), default="smoke")
    t.add_argument("--dropout", type=float, default=None, help="override preset dropout")
    t.add_argument("--out", default="out/lm_run")
    t.add_argument("--max-iters", type=int, default=2000)
    t.add_argument("--batch-size", type=int, default=12)
    t.add_argument("--eval-iters", type=int, default=20)
    t.add_argument("--val-frac", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=1337)
    t.set_defaults(func=cmd_train)

    ev = sub.add_parser("eval", help="re-evaluate a checkpoint vs unigram (no training)")
    ev.add_argument("checkpoint")
    ev.add_argument("--corpus", choices=["shakespeare", "aozora"], default="shakespeare")
    ev.add_argument("--corpus-file", default=None)
    ev.add_argument("--out", default=None, help="defaults to the checkpoint's directory")
    ev.add_argument("--val-frac", type=float, default=0.1)
    ev.add_argument("--seed", type=int, default=1337)
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
    return ap


def main(argv: list[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
