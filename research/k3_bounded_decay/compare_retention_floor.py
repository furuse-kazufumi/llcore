# SPDX-License-Identifier: Apache-2.0
"""PoC K3-1: 保持床 (lower-bounded log-decay) の baseline 比較.

目的は「良くなった」の主張ではなく **2 つの事実を測ること**:

1. **回帰していないか** — 有界化で val loss が悪化しないか (同一 seed / 同一 config)。
2. **チャネル死は実在するか** — 学習後の baseline (``-exp(w)``) で保持率
   ``α = exp(decay)`` が床 ``exp(g_min)`` を下回るチャネルが実際に出るか。
   出なければ「チャネル死は当スケールでは理論上の懸念に留まる」という
   **honest negative** として記録する (それ自体が結論)。

使い方::

    py -3.11 research/k3_bounded_decay/compare_retention_floor.py \
        --corpus-file out/corpus_shakespeare.txt --max-iters 300
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch

from llcore.lm.data import train_val_split
from llcore.lm.rwkv import RWKVConfig, RWKVLM, init_raw_for_log_decay
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import Trainer, TrainConfig


def _retention_stats(model: RWKVLM, log_floor: float) -> dict[str, float | int]:
    """学習後の各 time-mix チャネルの保持率 α を集計する."""
    alphas: list[torch.Tensor] = []
    for block in model.blocks:
        alphas.append(torch.exp(block.time_mixer.log_decay()).detach().flatten())
    alpha = torch.cat(alphas)
    floor = math.exp(log_floor)
    return {
        "n_channels": int(alpha.numel()),
        "alpha_min": float(alpha.min()),
        "alpha_median": float(alpha.median()),
        "alpha_max": float(alpha.max()),
        "n_below_floor": int((alpha < floor).sum()),
        "frac_below_floor": float((alpha < floor).float().mean()),
        "floor": floor,
    }


def _run_one(
    *,
    label: str,
    decay_log_floor: float | None,
    ffn_activation: str,
    ffn_hidden_mult: float,
    train_ids: torch.Tensor,
    val_ids: torch.Tensor,
    vocab_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    cfg = RWKVConfig(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        decay_log_floor=decay_log_floor,
        ffn_activation=ffn_activation,
        ffn_hidden_mult=ffn_hidden_mult,
    )
    model = RWKVLM(cfg)

    # ★統制: パラメータ化を跨いで初期 log-decay を揃える。
    #   zeros 初期化のままだと baseline は -exp(0) = -1 (α=0.368)、有界版は
    #   -5*sigmoid(0) = -2.5 (α=0.082) と **別の記憶レンジから出発**してしまい、
    #   測っているのがパラメータ化の差なのか初期値の差なのか分離できない。
    raw0 = init_raw_for_log_decay(args.init_log_decay, log_floor=decay_log_floor)
    with torch.no_grad():
        for block in model.blocks:
            block.time_mixer.time_decay.fill_(raw0)
    init_alpha = float(torch.exp(model.blocks[0].time_mixer.log_decay()).detach().median())

    trainer = Trainer(
        model,
        TrainConfig(
            max_iters=args.max_iters,
            lr_decay_iters=args.max_iters,
            batch_size=args.batch_size,
            eval_interval=max(1, args.max_iters // 4),
            eval_iters=args.eval_iters,
            seed=args.seed,
        ),
    )
    started = time.time()
    result = trainer.train(train_ids, val_ids)
    elapsed = time.time() - started

    return {
        "label": label,
        "decay_log_floor": decay_log_floor,
        "ffn_activation": ffn_activation,
        "ffn_hidden_mult": ffn_hidden_mult,
        "n_params": sum(p.numel() for p in model.parameters()),
        "best_val_loss": float(result["best_val_loss"]),  # type: ignore[arg-type]
        "init_alpha": init_alpha,
        "elapsed_sec": round(elapsed, 1),
        "retention": _retention_stats(model, args.probe_floor),
        "history": result["history"],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-file", default="out/corpus_shakespeare.txt")
    ap.add_argument("--out", default="out/k3_bounded_decay_compare.json")
    ap.add_argument("--max-iters", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--log-floor", type=float, default=-5.0, help="有界版の g_min")
    ap.add_argument(
        "--init-log-decay",
        type=float,
        default=-1.0,
        help="両 arm で揃える初期 log-decay (既定 -1.0 = baseline の zeros 初期化と同じ α=0.368)",
    )
    ap.add_argument(
        "--probe-floor",
        type=float,
        default=-5.0,
        help="チャネル死の判定に使う床 (baseline 側の集計にも同じ値を使う)",
    )
    ap.add_argument(
        "--arms",
        choices=("decay", "ffn"),
        default="decay",
        help="decay = 減衰パラメータ化を比較 / ffn = squared-ReLU vs SiTU-GLU を比較",
    )
    ap.add_argument(
        "--ffn-activation",
        choices=("sq_relu", "situ_glu"),
        default="sq_relu",
        help="--arms decay のとき両 arm に共通で使う FFN 活性化",
    )
    ap.add_argument(
        "--decay-log-floor-both",
        type=float,
        default=None,
        help="--arms ffn のとき両 arm に共通で使う decay_log_floor (既定 None = 従来 -exp(w))",
    )
    ap.add_argument("--ffn-hidden-mult", type=float, default=4.0, help="channel-mix の hidden 倍率")
    ap.add_argument(
        "--match-ffn-params",
        action="store_true",
        help="--arms ffn のとき SiTU-GLU 側の hidden を 2/3 にして FFN パラメータ数を揃える",
    )
    ap.add_argument("--max-chars", type=int, default=200_000, help="コーパス先頭 N 文字のみ使う")
    args = ap.parse_args()

    text = Path(args.corpus_file).read_text(encoding="utf-8")[: args.max_chars]
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, val_ids = train_val_split(ids, args.val_frac)
    print(f"corpus={args.corpus_file} chars={len(text)} vocab={tok.vocab_size} "
          f"train={train_ids.numel()} val={val_ids.numel()}")

    mult = args.ffn_hidden_mult
    if args.arms == "decay":
        # 減衰のパラメータ化を比較 (FFN は両 arm 共通)
        arm_specs = [
            ("baseline_neg_exp", None, args.ffn_activation, mult),
            ("bounded_sigmoid", args.log_floor, args.ffn_activation, mult),
        ]
    else:
        # FFN 活性化を比較 (減衰は両 arm 共通)。
        # ★統制: SiTU-GLU は up 枝が増えるため同じ hidden 倍率だと FFN パラメータが
        #   1.5 倍になり、「活性化の差」ではなく「容量の差」を測ってしまう。
        #   --match-ffn-params で 2/3 則 (hidden を 8/3 倍) を適用して揃える。
        glu_mult = mult * 2.0 / 3.0 if args.match_ffn_params else mult
        arm_specs = [
            ("baseline_sq_relu", args.decay_log_floor_both, "sq_relu", mult),
            ("situ_glu", args.decay_log_floor_both, "situ_glu", glu_mult),
        ]

    runs = []
    for label, floor, ffn, hmult in arm_specs:
        print(f"--- training {label} (decay_log_floor={floor}, ffn={ffn}, hidden_mult={hmult:.3f}) ---")
        run = _run_one(
            label=label,
            decay_log_floor=floor,
            ffn_activation=ffn,
            ffn_hidden_mult=hmult,
            train_ids=train_ids,
            val_ids=val_ids,
            vocab_size=tok.vocab_size,
            args=args,
        )
        print(f"    best_val_loss={run['best_val_loss']:.4f}  ({run['elapsed_sec']}s, params={run['n_params']:,})")
        print(f"    init_alpha={run['init_alpha']:.4f}  retention={run['retention']}")
        runs.append(run)

    payload = {
        "config": vars(args),
        "runs": runs,
        "delta_val_loss": runs[1]["best_val_loss"] - runs[0]["best_val_loss"],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print(f"delta_val_loss (bounded - baseline) = {payload['delta_val_loss']:+.4f} "
          "(正なら有界版が悪化)")


if __name__ == "__main__":
    main()
