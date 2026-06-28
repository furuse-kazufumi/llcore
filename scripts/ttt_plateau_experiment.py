# SPDX-License-Identifier: Apache-2.0
"""L1: does test-time-training (TTT-Linear) move the constant-state effective-context plateau?

llcore's headline null (docs/CONVERSATIONAL_LLCORE_FINDINGS.md / MODEL_LANDSCAPE_2026_06.md §10):
a trained gated-RNN's ``context_length_curve`` drops then PLATEAUS around ``block_size`` — it does
not exploit context past its training window. Two competing hypotheses for *why*, and what fixes it:

  * capacity-limited  -> a wider state should help  (StateX, arXiv:2509.22630): the
        ``recurrent-wide`` arm = same gated-RNN with a larger ``state_size`` (more params, same
        fixed linear update rule).
  * update-rule-limited -> a gradient-based memory should help  (TTT, arXiv:2407.04620): the
        ``ttt-linear`` arm = delta-rule fast-weight state (``llcore.lm.ttt``), which decides what to
        store online instead of via the BPTT horizon.

All arms train with the SAME ``block_size`` and are scored with the confound-free
``context_length_curve`` (fixed target positions, cold-started windows) sweeping context far past
``block_size``. The honest read: which arm keeps lowering NLL beyond ``block_size`` — and is the
gain from capacity (recurrent-wide) or from the update rule (ttt-linear)? Reports ``past_block_gain``
= relative NLL drop from ``c==block_size`` to the longest ``c`` (positive == still using more
context). CPU / float32; designed to run unattended (no push).
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
from llcore.lm.device import resolve_device
from llcore.lm.eval import held_out_report_any
from llcore.lm.longctx_eval import ConstantStateLM, context_length_curve
from llcore.lm.recurrent import RecurrentConfig, RecurrentLM
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import TrainConfig, Trainer
from llcore.lm.ttt import TTTLinearConfig, TTTLinearLM


def past_block_gain(nll_by_context: dict[int, float], block_size: int) -> float | None:
    """Relative NLL drop from context==block_size to the largest context (>0 == still improving).

    Returns ``None`` if ``block_size`` or a larger context is missing from the curve.
    """
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
        return RecurrentLM(
            RecurrentConfig(
                vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer,
                n_embd=args.n_embd, state_size=args.state_size,
            )
        )
    if arch == "recurrent-wide":
        return RecurrentLM(
            RecurrentConfig(
                vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer,
                n_embd=args.n_embd, state_size=args.wide_state_size,
            )
        )
    if arch == "ttt-linear":
        return TTTLinearLM(
            TTTLinearConfig(
                vocab_size=vocab, block_size=args.block_size, n_layer=args.n_layer,
                n_embd=args.n_embd, state_dim=args.ttt_state_dim,
            )
        )
    raise ValueError(f"unknown arch {arch!r}")


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="L1 TTT-Linear plateau experiment vs gated-RNN baselines")
    ap.add_argument("--corpus-file", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--arches", default="recurrent,recurrent-wide,ttt-linear")
    ap.add_argument("--block-size", type=int, default=128)
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
    ap.add_argument("--device", default="auto",
                    help="compute device: auto (cuda if available else cpu) | cpu | cuda | cuda:N")
    ap.add_argument("--out", default="out/ttt_plateau")
    args = ap.parse_args(argv)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    context_lens = [int(c) for c in args.context_lens.split(",")]

    text = Path(args.corpus_file).read_text(encoding="utf-8")
    tok = CharTokenizer.from_text(text)
    ids = encode_corpus(text, tok)
    train_ids, val_ids = train_val_split(ids, val_frac=args.val_frac)
    print(
        f"[corpus] chars={len(text):,} vocab={tok.vocab_size} "
        f"train={train_ids.numel():,} val={val_ids.numel():,} ctx_lens={context_lens}",
        flush=True,
    )

    arches = [a.strip() for a in args.arches.split(",") if a.strip()]
    results: dict[str, Any] = {}
    for arch in arches:
        torch.manual_seed(args.seed)
        model = _build(arch, tok.vocab_size, args)
        n_params = sum(p.numel() for p in model.parameters())
        cfg = TrainConfig(
            learning_rate=args.lr, max_iters=args.max_iters,
            warmup_iters=min(100, max(1, args.max_iters // 10)),
            lr_decay_iters=args.max_iters, batch_size=args.batch_size,
            eval_interval=max(200, args.max_iters // 4), eval_iters=20, seed=args.seed,
        )
        t0 = time.perf_counter()

        def on_eval(it: int, tr: float, va: float, _a: str = arch) -> None:
            print(f"[{_a}] iter {it:>5} train {tr:.4f} val {va:.4f} "
                  f"({(time.perf_counter()-t0)/60:.1f} min)", flush=True)

        tr_res = Trainer(model, cfg).train(train_ids, val_ids, on_eval=on_eval)
        train_min = (time.perf_counter() - t0) / 60.0
        report = held_out_report_any(model, train_ids, val_ids, tok.vocab_size, args.block_size)
        curve = context_length_curve(model, val_ids, context_lens, args.n_positions, args.seed)
        nllbc = {int(k): float(v) for k, v in cast("dict[int, float]", curve["nll_by_context"]).items()}
        gain = past_block_gain(nllbc, args.block_size)
        results[arch] = {
            "n_params": n_params,
            "best_val_loss": round(float(tr_res["best_val_loss"]), 4),  # type: ignore[arg-type]
            "held_out_ppl": round(report["model_ppl"], 4),
            "unigram_ppl": round(report["unigram_ppl"], 4),
            "ratio_over_unigram": round(report["model_ppl"] / report["unigram_ppl"], 4),
            "nll_by_context": nllbc,
            "ppl_by_context": {k: round(float(torch.tensor(v).exp()), 3) for k, v in nllbc.items()},
            "past_block_gain": (round(gain, 4) if gain is not None else None),
            "train_minutes": round(train_min, 2),
        }
        print(f"[{arch}] params={n_params:,} held_out_ppl={report['model_ppl']:.2f} "
              f"past_block_gain={gain if gain is None else round(gain,4)} "
              f"nll_by_ctx={ {k: round(v,4) for k,v in nllbc.items()} }", flush=True)

    summary = {
        "block_size": args.block_size,
        "context_lens": context_lens,
        "n_positions": args.n_positions,
        "arches": results,
        "note": "past_block_gain>0 means the arm keeps lowering NLL past block_size (uses long "
        "context). Compare recurrent-wide (capacity-only, StateX-style) vs ttt-linear (update-rule, "
        "TTT) against the recurrent baseline: a ttt-linear gain that recurrent-wide does NOT match "
        "isolates the benefit to the update rule, not raw state capacity. Single seed, small CPU "
        "model — directional, not publishable; honest-disclosure applies before any strong claim.",
    }
    (out / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== PLATEAU EXPERIMENT SUMMARY ===", flush=True)
    for arch, r in results.items():
        print(f"  {arch:>15}: ppl={r['held_out_ppl']:.2f} past_block_gain={r['past_block_gain']}", flush=True)
    print(f"[done] wrote {out}/comparison.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
