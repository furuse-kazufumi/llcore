# SPDX-License-Identifier: Apache-2.0
"""L1 Stage-2: the *correct* plateau experiment — state-carry (TBPTT) training vs state-reset.

``ttt_plateau_experiment.py`` (Stage-1) trains with the stock ``Trainer``, which resets the state
every ``block_size`` (``data.get_batch`` returns RANDOM blocks), so the model never experiences
dependencies longer than ``block_size`` during training. Its ``context_length_curve`` past
``block_size`` therefore measures only the recurrence's *extrapolation inductive bias* — and indeed
the gated-RNN baseline comes out flat (``past_block_gain ≈ 0``). That is the **carry-off** half.

This script adds the **carry-on** half: ``TBPTTTrainer`` streams contiguous ``seg_len`` segments and
carries a *detached* state across ``chunk_size`` gradient windows (``tbptt.py``), so the model is
trained to actually use context out to ``seg_len``. Running both halves over the same arches gives a
2×3 ablation that separates THREE factors for the plateau:

    capacity   : recurrent      vs recurrent-wide      (StateX-style wider state)
    update rule: recurrent      vs gated-deltanet       (data-dependent α/β delta rule, ``ttt.py``)
    training   : --carry off    vs --carry on           (state-reset vs state-carry TBPTT)  ← new axis

If ``past_block_gain`` only becomes positive under ``--carry on``, the plateau's cause was the
*training method*, not capacity or the cell. Fair-compute: ``chunk_size == block_size`` and
``max_updates == max_iters`` match the per-step token budget across halves.

Honest scope: single seed, small CPU char-LM — directional, not publishable. ``past_block_gain`` is a
coarse 2-point read (block_size vs longest c); inspect the full ``ppl_by_context`` curve and, before
any strong claim, re-run with ≥3 seeds (``feedback_benchmark_honest_disclosure``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch

from llcore.lm.data import encode_corpus, train_val_split
from llcore.lm.eval import held_out_report_any
from llcore.lm.longctx_eval import ConstantStateLM, context_length_curve
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import TrainConfig, Trainer
from llcore.lm.tbptt import TBPTTConfig, TBPTTTrainer
from llcore.lm.ttt import TTTLinearConfig, TTTLinearLM


def past_block_gain(nll_by_context: dict[int, float], block_size: int) -> float | None:
    """Relative NLL drop from context==block_size to the largest context (>0 == still improving)."""
    ctxs = sorted(nll_by_context)
    beyond = [c for c in ctxs if c > block_size]
    if block_size not in nll_by_context or not beyond:
        return None
    base = nll_by_context[block_size]
    far = nll_by_context[max(beyond)]
    if base <= 0:
        return None
    return (base - far) / base


def _build(arch: str, vocab: int, args: argparse.Namespace) -> ConstantStateLM:
    if arch == "recurrent":
        return RecurrentLM(RecurrentConfig(
            vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer,
            n_embd=args.n_embd, state_size=args.state_size))
    if arch == "recurrent-wide":
        return RecurrentLM(RecurrentConfig(
            vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer,
            n_embd=args.n_embd, state_size=args.wide_state_size))
    if arch == "gated-deltanet":
        return TTTLinearLM(TTTLinearConfig(
            vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer,
            n_embd=args.n_embd, state_dim=args.ttt_state_dim))
    raise ValueError(f"unknown arch {arch!r}")


def _train(model: ConstantStateLM, carry: str, train_ids: torch.Tensor, val_ids: torch.Tensor,
           args: argparse.Namespace, label: str) -> float:
    t0 = time.perf_counter()

    def on_eval(it: int, tr: float, va: float) -> None:
        print(f"[{label}] upd {it:>5} train {tr:.4f} val {va:.4f} "
              f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

    if carry == "on":
        cfg = TBPTTConfig(seg_len=args.seg_len, chunk_size=args.block_size, batch_size=args.batch_size,
                          max_updates=args.max_iters, learning_rate=args.lr,
                          eval_interval=max(200, args.max_iters // 4), eval_iters=20, seed=args.seed)
        res = TBPTTTrainer(model, cfg).train(train_ids, val_ids, on_eval=on_eval)
    else:
        cfg2 = TrainConfig(learning_rate=args.lr, max_iters=args.max_iters,
                           warmup_iters=min(100, max(1, args.max_iters // 10)),
                           lr_decay_iters=args.max_iters, batch_size=args.batch_size,
                           eval_interval=max(200, args.max_iters // 4), eval_iters=20, seed=args.seed)
        res = Trainer(model, cfg2).train(train_ids, val_ids, on_eval=on_eval)
    return round(float(cast("float", res["best_val_loss"])), 4)


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="state-carry (TBPTT) plateau ablation")
    ap.add_argument("--corpus-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--arches", default="recurrent,recurrent-wide,gated-deltanet")
    ap.add_argument("--carry", choices=["on", "off"], default="on")
    ap.add_argument("--block-size", type=int, default=128)
    ap.add_argument("--seg-len", type=int, default=2048, help="TBPTT segment (state carried this far)")
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=128)
    ap.add_argument("--state-size", type=int, default=128)
    ap.add_argument("--wide-state-size", type=int, default=256)
    ap.add_argument("--ttt-state-dim", type=int, default=96)
    ap.add_argument("--max-iters", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--context-lens", default="16,32,64,128,256,512,1024")
    ap.add_argument("--n-positions", type=int, default=160)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", default="out/tbptt_plateau")
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    context_lens = [int(c) for c in args.context_lens.split(",")]

    text = Path(args.corpus_file).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=args.val_frac)
    print(f"[corpus] chars={len(text):,} vocab={tok.vocab_size} train={train_ids.numel():,} "
          f"val={val_ids.numel():,} carry={args.carry} seg_len={args.seg_len} ctx={context_lens}", flush=True)

    arches = [a.strip() for a in args.arches.split(",") if a.strip()]
    results: dict[str, Any] = {}
    for arch in arches:
        torch.manual_seed(args.seed)
        model = _build(arch, tok.vocab_size, args)
        n_params = sum(p.numel() for p in model.parameters())
        label = f"{arch}/{args.carry}"
        best_val = _train(model, args.carry, train_ids, val_ids, args, label)
        report = held_out_report_any(model, train_ids, val_ids, tok.vocab_size, args.block_size)
        curve = context_length_curve(model, val_ids, context_lens, args.n_positions, args.seed)
        nllbc = {int(k): float(v) for k, v in cast("dict[int, float]", curve["nll_by_context"]).items()}
        gain = past_block_gain(nllbc, args.block_size)
        results[arch] = {
            "n_params": n_params, "carry": args.carry, "best_val_loss": best_val,
            "held_out_ppl": round(report["model_ppl"], 4),
            "ppl_by_context": {k: round(float(torch.tensor(v).exp()), 3) for k, v in nllbc.items()},
            "past_block_gain": (round(gain, 4) if gain is not None else None),
        }
        print(f"[{label}] params={n_params:,} ppl={report['model_ppl']:.2f} "
              f"past_block_gain={results[arch]['past_block_gain']} "
              f"ppl_by_ctx={results[arch]['ppl_by_context']}", flush=True)

    summary = {
        "carry": args.carry, "block_size": args.block_size, "seg_len": args.seg_len,
        "context_lens": context_lens, "n_positions": args.n_positions, "arches": results,
        "note": "Compare against the carry=off run (ttt_plateau / a separate --carry off run). If "
        "past_block_gain only turns positive under carry=on, the plateau was a training-method "
        "artifact, not capacity/cell. Single seed, small CPU model = directional; coarse 2-point "
        "gain — read the full ppl_by_context curve and re-run >=3 seeds before any claim.",
    }
    (out / f"comparison_carry_{args.carry}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== TBPTT PLATEAU (carry={args.carry}) ===", flush=True)
    for arch, r in results.items():
        print(f"  {arch:>15}: ppl={r['held_out_ppl']:.2f} past_block_gain={r['past_block_gain']}", flush=True)
    print(f"[done] wrote {out}/comparison_carry_{args.carry}.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
