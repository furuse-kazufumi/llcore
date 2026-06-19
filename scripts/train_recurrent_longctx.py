# SPDX-License-Identifier: Apache-2.0
"""Train a constant-state recurrent char-LM and save an arch-tagged checkpoint.

Why this exists
---------------
``compare.py`` trains recurrent / rwkv models *in memory* and discards them, and the CLI
(``llcore.lm.__main__``) only persists :class:`CharGPT`. So there has never been a saved,
reloadable *trained* recurrent model — only random-init ones (used for the memory-curve
proofs). To show that a constant-state model keeps its quality on contexts far longer than
its training ``block_size`` (the long-context ``streaming_nll`` story), we first need a real
trained checkpoint. This trains one via the already-tested :class:`Trainer` and writes
``<out>/model.pt`` (through the tested :mod:`llcore.lm.checkpoint`) plus a ``verdict.json``.

CPU-only. Honest by construction: it reports held-out PPL vs the order-0 unigram baseline
on the *same* held-out tokens, the teacher-forced top-1, and the unigram gate verdict.

Example
-------
    py -3.11 scripts/train_recurrent_longctx.py \
        --corpus-file out/corpus_aozora_multi.txt --arch recurrent \
        --block-size 128 --n-layer 4 --n-embd 128 --state-size 128 \
        --max-iters 3500 --batch-size 32 --out out/lm_recurrent_aozora
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from llcore.lm.checkpoint import save_lm_checkpoint
from llcore.lm.data import encode_corpus, train_val_split
from llcore.lm.eval import held_out_report_any, held_out_top1_report, passes_gate
from llcore.lm.model import CharGPT, GPTConfig
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.rwkv import RWKVConfig, RWKVLM
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import Trainer, TrainConfig

LM = CharGPT | RecurrentLM | RWKVLM


def build_model(arch: str, vocab_size: int, args: argparse.Namespace) -> LM:
    if arch == "recurrent":
        return RecurrentLM(
            RecurrentConfig(
                vocab_size=vocab_size,
                block_size=args.block_size,
                n_layer=args.n_layer,
                n_embd=args.n_embd,
                state_size=args.state_size,
                dropout=args.dropout,
            )
        )
    if arch == "rwkv":
        return RWKVLM(
            RWKVConfig(
                vocab_size=vocab_size,
                block_size=args.block_size,
                n_layer=args.n_layer,
                n_embd=args.n_embd,
                dropout=args.dropout,
            )
        )
    if arch == "gpt":
        return CharGPT(
            GPTConfig(
                vocab_size=vocab_size,
                block_size=args.block_size,
                n_layer=args.n_layer,
                n_head=args.n_head,
                n_embd=args.n_embd,
                dropout=args.dropout,
            )
        )
    raise ValueError(f"unknown arch {arch!r}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="train + save an arch-tagged char-LM checkpoint")
    ap.add_argument("--corpus-file", required=True)
    ap.add_argument("--arch", choices=["recurrent", "rwkv", "gpt"], default="recurrent")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--state-size", type=int, default=128)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max-iters", type=int, default=3500)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-iters", type=int, default=40)
    ap.add_argument("--eval-interval", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threads", type=int, default=0, help="torch.set_num_threads (0 = leave default)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    text = Path(args.corpus_file).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=args.val_frac)
    print(
        f"[corpus] {args.corpus_file}  chars={len(text):,}  vocab={tok.vocab_size}  "
        f"train={train_ids.numel():,}  val={val_ids.numel():,}",
        flush=True,
    )

    model = build_model(args.arch, tok.vocab_size, args)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"[model] arch={args.arch} params={n_params:,} "
        f"L{args.n_layer} D{args.n_embd} ctx{args.block_size} "
        f"(state{args.state_size if args.arch == 'recurrent' else '-'})",
        flush=True,
    )

    train_cfg = TrainConfig(
        learning_rate=args.lr,
        max_iters=args.max_iters,
        warmup_iters=min(100, max(1, args.max_iters // 10)),
        lr_decay_iters=args.max_iters,
        batch_size=args.batch_size,
        eval_interval=args.eval_interval,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )
    trainer = Trainer(model, train_cfg)

    t0 = time.perf_counter()

    def on_eval(it: int, tr: float, va: float) -> None:
        elapsed = time.perf_counter() - t0
        print(f"[train] iter {it:>5}  train {tr:.4f}  val {va:.4f}  ({elapsed/60:.1f} min)", flush=True)
        # periodic checkpoint so a long CPU run is crash-recoverable
        save_lm_checkpoint(out / "model.pt", model, tok.itos)

    result = trainer.train(train_ids, val_ids, on_eval=on_eval)
    best_val = float(result["best_val_loss"])  # type: ignore[arg-type]
    train_minutes = (time.perf_counter() - t0) / 60.0

    save_lm_checkpoint(out / "model.pt", model, tok.itos)

    report = held_out_report_any(model, train_ids, val_ids, tok.vocab_size, args.block_size)
    top1 = held_out_top1_report(model, val_ids, args.block_size)
    gate = passes_gate(report["model_ppl"], report["unigram_ppl"])
    verdict = {
        "arch": args.arch,
        "corpus_file": args.corpus_file,
        "n_params": n_params,
        "block_size": args.block_size,
        "n_layer": args.n_layer,
        "n_embd": args.n_embd,
        "state_size": args.state_size if args.arch == "recurrent" else None,
        "max_iters": args.max_iters,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "vocab_size": tok.vocab_size,
        "train_tokens": int(train_ids.numel()),
        "val_tokens": int(val_ids.numel()),
        "best_val_loss": round(best_val, 4),
        "model_nll": round(report["model_nll"], 4),
        "unigram_nll": round(report["unigram_nll"], 4),
        "model_ppl": round(report["model_ppl"], 4),
        "unigram_ppl": round(report["unigram_ppl"], 4),
        "ratio_model_over_unigram": round(report["model_ppl"] / report["unigram_ppl"], 4),
        "top1_acc": round(top1["top1_acc"], 4),
        "top5_acc": round(top1["top5_acc"], 4),
        "n_eval_tokens": int(report["n_tokens"]),
        "ppl_gate_pass": bool(gate),
        "train_minutes": round(train_minutes, 2),
    }
    (out / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== TRAIN VERDICT ===", flush=True)
    print(f"  arch        : {args.arch}  params={n_params:,}", flush=True)
    print(f"  unigram PPL : {report['unigram_ppl']:.3f}", flush=True)
    print(
        f"  model   PPL : {report['model_ppl']:.3f}  "
        f"({report['model_ppl'] / report['unigram_ppl']:.3f}x unigram)",
        flush=True,
    )
    print(f"  top1 / top5 : {top1['top1_acc']:.3f} / {top1['top5_acc']:.3f}", flush=True)
    print(f"  unigram gate: {'PASS' if gate else 'FAIL'}", flush=True)
    print(f"  trained in  : {train_minutes:.1f} min  -> {out}/model.pt", flush=True)
    return 0 if gate else 2


if __name__ == "__main__":
    raise SystemExit(main())
