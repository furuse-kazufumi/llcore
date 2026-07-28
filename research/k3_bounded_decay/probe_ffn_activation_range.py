# SPDX-License-Identifier: Apache-2.0
"""学習後の channel-mix hidden 活性の実レンジを測る (softcap が拘束的か判定する).

保持床の実験で「``g_min=-5`` は観測 α_min の 1/39 で非拘束だった」のと同じ罠を
FFN の softcap でも踏まないための検査。**cap β が実活性レンジより大きければ
softcap は一度も発火せず、「コストゼロ」は自明**になる。

使い方::

    py -3.11 research/k3_bounded_decay/probe_ffn_activation_range.py --max-iters 2000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llcore.lm.data import get_batch, train_val_split
from llcore.lm.rwkv import RWKVConfig, RWKVLM, init_raw_for_log_decay
from llcore.lm.tokenizer import CharTokenizer
from llcore.lm.trainer import Trainer, TrainConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus-file", default="out/corpus_shakespeare.txt")
    ap.add_argument("--out", default="out/k3_ffn_range_probe.json")
    ap.add_argument("--max-iters", type=int, default=2000)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--probe-batches", type=int, default=20)
    ap.add_argument("--max-chars", type=int, default=200_000)
    ap.add_argument(
        "--caps",
        default="100,25,10,5,2",
        help="この β で cap したら何 % の活性が影響を受けるかを併せて報告する",
    )
    args = ap.parse_args()

    text = Path(args.corpus_file).read_text(encoding="utf-8")[: args.max_chars]
    tok = CharTokenizer.from_text(text)
    ids = torch.tensor(tok.encode(text), dtype=torch.long)
    train_ids, val_ids = train_val_split(ids, 0.1)

    torch.manual_seed(args.seed)
    cfg = RWKVConfig(
        vocab_size=tok.vocab_size,
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
    )
    model = RWKVLM(cfg)
    raw0 = init_raw_for_log_decay(-1.0, log_floor=None)
    with torch.no_grad():
        for block in model.blocks:
            block.time_mixer.time_decay.fill_(raw0)

    print(f"training {args.max_iters} iters (uncapped sq_relu) ...")
    Trainer(
        model,
        TrainConfig(
            max_iters=args.max_iters,
            lr_decay_iters=args.max_iters,
            batch_size=args.batch_size,
            eval_interval=max(1, args.max_iters),
            eval_iters=5,
            seed=args.seed,
        ),
    ).train(train_ids, val_ids)

    # 学習後モデルで hidden 活性を収集する
    model.eval()
    gen = torch.Generator().manual_seed(args.seed + 99)
    collected: list[torch.Tensor] = []

    hooks = []

    def make_hook() -> object:
        def hook(module, inputs, output):  # type: ignore[no-untyped-def]
            collected.append(output.detach().abs().flatten())
        return hook

    for block in model.blocks:
        # value 射影への入力 = ffn_hidden の出力
        hooks.append(block.channel_mixer.value.register_forward_pre_hook(
            lambda m, inp: collected.append(inp[0].detach().abs().flatten())
        ))

    with torch.no_grad():
        for _ in range(args.probe_batches):
            x, _y = get_batch(val_ids, cfg.block_size, args.batch_size, gen)
            model(x)

    for h in hooks:
        h.remove()

    a = torch.cat(collected)
    caps = [float(c) for c in args.caps.split(",") if c.strip()]
    report = {
        "n_activations": int(a.numel()),
        "abs_max": float(a.max()),
        "abs_mean": float(a.mean()),
        "p50": float(a.quantile(0.50)),
        "p99": float(a.quantile(0.99)),
        "p99_99": float(a.quantile(0.9999)),
        "frac_above_cap": {str(c): float((a > c).float().mean()) for c in caps},
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    for c in caps:
        frac = report["frac_above_cap"][str(c)]
        verdict = "拘束する" if frac > 0 else "**非拘束 (cap が一度も発火しない)**"
        print(f"  β={c:>6}: 影響を受ける活性 {frac:.6%} → {verdict}")


if __name__ == "__main__":
    main()
