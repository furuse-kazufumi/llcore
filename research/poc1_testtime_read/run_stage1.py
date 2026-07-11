# SPDX-License-Identifier: Apache-2.0
"""PoC-1 Stage 1 (ゲート): tiny Gated DeltaNet を MQAR で学習し R0 (単発 read) recall を確立する。

pre-reg §3.1 の write (状態生成) を tiny-CPU で実走。ここで R0 recall が chance を有意に超えねば、
read 変種比較 (Stage 2) は無意味なので **staged-PoC のゲート**として先に確認する
(`feedback_staged_poc_individual_structure`)。honest: これは write が recall を学習できるかの確認で
あって、read 側 novelty の主張ではない。

usage:
  py -3.11 research/poc1_testtime_read/run_stage1.py --steps 400 --state gated_delta
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

# llcore import (RAPTOR_DIR 非依存の直接パス)
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from llcore.lm.ttt import TTTLinearConfig, TTTLinearLM  # noqa: E402

from mqar import MQARConfig, make_batch  # noqa: E402  (同ディレクトリ)


def _chance(cfg: MQARConfig) -> float:
    """recall の chance レベル = 1/(value 種数)。value は [D+1..2D] の D 種から一様。"""
    return 1.0 / cfg.num_keys


@torch.no_grad()
def eval_recall(model: TTTLinearLM, cfg: MQARConfig, gen: torch.Generator,
                *, batches: int = 8, batch_size: int = 128) -> float:
    model.eval()
    correct = 0
    total = 0
    for _ in range(batches):
        inp, tgt, qpos = make_batch(cfg, batch_size, gen)
        logits, _ = model(inp)  # [B,T,V]
        pred = logits.argmax(-1)  # [B,T]
        mask = tgt != -1
        correct += int((pred[mask] == tgt[mask]).sum())
        total += int(mask.sum())
    return correct / max(1, total)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--n-layer", type=int, default=2)
    p.add_argument("--n-embd", type=int, default=128)
    p.add_argument("--state-dim", type=int, default=128)
    p.add_argument("--num-keys", type=int, default=16)
    p.add_argument("--num-pairs", type=int, default=8)
    p.add_argument("--num-queries", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--a-log-init", type=float, default=None,
                   help="faithful セルの A_log を log(値) で再初期化 (機構不変=init のみ)。"
                        "既定 uniform(1,16) は exp(A_log) 巨大→α≈0 で忘却が強く recall に敵対的。"
                        "例: 1.0 で rate=1→α≈0.95 (保持寄り)。")
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)
    eval_gen = torch.Generator().manual_seed(args.seed + 999)

    mcfg = MQARConfig(num_keys=args.num_keys, num_pairs=args.num_pairs,
                      num_queries=args.num_queries, seed=args.seed)
    model = TTTLinearLM(TTTLinearConfig(
        vocab_size=mcfg.vocab_size, block_size=mcfg.seq_len,
        n_layer=args.n_layer, n_embd=args.n_embd, state_dim=args.state_dim, dropout=0.0))
    if args.a_log_init is not None:
        # A_log は学習可能パラメータ (機構ではない)。recall タスク向けに保持寄りへ再初期化する。
        import math
        with torch.no_grad():
            for layer in model.transformer["h"]:
                layer.A_log.fill_(math.log(args.a_log_init))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"[stage1] params={model.num_params():,} vocab={mcfg.vocab_size} seq_len={mcfg.seq_len} "
          f"chance={_chance(mcfg):.3f} state_dim={args.state_dim} n_layer={args.n_layer}")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        model.train()
        inp, tgt, _ = make_batch(mcfg, args.batch_size, gen)
        _, loss = model(inp, tgt)
        assert loss is not None
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % args.eval_every == 0 or step == args.steps:
            rec = eval_recall(model, mcfg, eval_gen)
            print(f"  step {step:4d}  loss {loss.item():.4f}  R0-recall {rec:.3f}  "
                  f"({time.time() - t0:.1f}s)")
    final = eval_recall(model, mcfg, eval_gen, batches=16)
    chance = _chance(mcfg)
    verdict = "GATE-PASS" if final > 3 * chance else "GATE-FAIL"
    print(f"[stage1] final R0-recall={final:.3f} chance={chance:.3f} → {verdict} "
          f"(基準: recall > 3×chance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
