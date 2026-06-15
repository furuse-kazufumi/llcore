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
from llcore.lm.eval import held_out_perplexity, passes_gate, unigram_perplexity
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
    model_cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        block_size=int(preset["block_size"]),
        n_layer=int(preset["n_layer"]),
        n_head=int(preset["n_head"]),
        n_embd=int(preset["n_embd"]),
        dropout=float(preset["dropout"]),
    )
    model = CharGPT(model_cfg)
    print(f"[model] {args.config}: {model.num_params(False):,} params  cfg={preset}")

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

    unigram_ppl = unigram_perplexity(train_ids, val_ids, tok.vocab_size)
    model_ppl = held_out_perplexity(model, val_ids, model_cfg.block_size, batch_size=32)
    seed_char = "\n" if "\n" in tok.stoi else tok.itos[0]
    sample = generate_text(
        model, tok, prompt=seed_char, max_new_tokens=400, temperature=0.8, top_k=40, seed=args.seed
    )
    min_distinct = 15 if tok.vocab_size >= 20 else max(3, tok.vocab_size // 2)
    degenerate = is_degenerate(sample, min_distinct=min_distinct)
    gate = passes_gate(model_ppl, unigram_ppl)

    verdict = {
        "corpus": args.corpus_file or args.corpus,
        "config": args.config,
        "vocab_size": tok.vocab_size,
        "n_params": model.num_params(False),
        "max_iters": args.max_iters,
        "best_val_loss": result["best_val_loss"],
        "unigram_ppl": round(unigram_ppl, 4),
        "model_ppl": round(model_ppl, 4),
        "ratio_model_over_unigram": round(model_ppl / unigram_ppl, 4),
        "ppl_gate_pass": gate,
        "degenerate_sample": degenerate,
        "overall_pass": bool(gate and not degenerate),
        "sample_head": sample[:200],
    }
    (out / "verdict.json").write_text(
        json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _save_checkpoint(out / "model.pt", model, tok)
    tok.save(out / "tokenizer.json")
    save_viz_json(model, out / "model_viz.json", tok, include_vocab=True)

    print("\n=== P0 VERDICT ===")
    print(f"  unigram PPL : {unigram_ppl:.3f}")
    print(f"  model   PPL : {model_ppl:.3f}  ({model_ppl / unigram_ppl:.3f}x unigram)")
    print(f"  PPL gate    : {'PASS' if gate else 'FAIL'} (model <= 0.85x unigram)")
    print(f"  degenerate  : {degenerate}")
    print(f"  OVERALL     : {'PASS' if verdict['overall_pass'] else 'FAIL'}")
    print(f"  artifacts   : {out}/ (verdict.json, model.pt, model_viz.json)")
    print("\n--- sample (first 200 chars) ---")
    print(sample[:200])
    return 0 if verdict["overall_pass"] else 2


def cmd_generate(args: argparse.Namespace) -> int:
    model, tok = _load_checkpoint(Path(args.checkpoint))
    print(
        generate_text(
            model, tok, prompt=args.prompt, max_new_tokens=args.max_new_tokens,
            temperature=args.temperature, top_k=args.top_k, seed=args.seed,
        )
    )
    return 0


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
    t.add_argument("--out", default="out/lm_run")
    t.add_argument("--max-iters", type=int, default=2000)
    t.add_argument("--batch-size", type=int, default=12)
    t.add_argument("--eval-iters", type=int, default=20)
    t.add_argument("--val-frac", type=float, default=0.1)
    t.add_argument("--seed", type=int, default=1337)
    t.set_defaults(func=cmd_train)

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
