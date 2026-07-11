# SPDX-License-Identifier: Apache-2.0
"""PoC-1 Stage 2b — H3: 状態種 ablation。

pre-reg §3.1(5)/§5: cleanup read の gain が gated-delta / delta-rule で出て、vanilla-additive で
消えることが full GO の第3条件。vanilla でも同 gain なら「どの状態でも効く一般 read トリック」=
novelty 縮小、を切り分ける。

faithful セル (ttt.py) は変更せず、model 構築後に **instance レベルで step を差し替え**て 3 状態種を
実装する (read out = S' q̂ は全種共通 = R0 比較可能):
  - gated_delta      : next = α·S + β·(v − α·S k̂)⊗k̂         (本命 = ttt.py と厳密一致)
  - delta_rule       : next = S + β·(v − S k̂)⊗k̂            (忘却なし・delta 補正あり)
  - vanilla_additive : next = S + β·(v ⊗ k̂)                (忘却なし・delta 補正なし = 純 Hebbian・最弱)

usage: py -3.11 research/poc1_testtime_read/run_h3.py --train-steps 400 --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for pth in (str(_ROOT / "src"), str(_HERE)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from llcore.lm.ttt import TTTLinearConfig, TTTLinearLM  # noqa: E402

import reads as R  # noqa: E402
from mqar import MQARConfig  # noqa: E402
from run_stage2 import collect_states, paired_ci, recall_of, train  # noqa: E402


def _ablation_step(self, h: torch.Tensor, state: torch.Tensor):  # noqa: ANN001
    """state_type に応じた状態更新。read out=S'q̂ と downstream は ttt.py と同一。"""
    k = self.k_proj(h)
    v = self.v_proj(h)
    q = self.q_proj(h)
    khat = F.normalize(k, p=2.0, dim=-1, eps=1e-6)
    qhat = F.normalize(q, p=2.0, dim=-1, eps=1e-6)
    alpha = torch.exp(-self.A_log.exp() * F.softplus(self.a_proj(h) + self.dt_bias))
    beta = torch.sigmoid(self.b_proj(h))
    st = self.state_type
    if st == "gated_delta":
        pred = torch.einsum("bij,bj->bi", state, khat)
        write = torch.einsum("bi,bj->bij", v - alpha * pred, khat)
        next_state = alpha[:, :, None] * state + beta[:, :, None] * write
    elif st == "delta_rule":
        pred = torch.einsum("bij,bj->bi", state, khat)
        write = torch.einsum("bi,bj->bij", v - pred, khat)
        next_state = state + beta[:, :, None] * write
    elif st == "vanilla_additive":
        write = torch.einsum("bi,bj->bij", v, khat)
        next_state = state + beta[:, :, None] * write
    else:
        raise ValueError(st)
    out = torch.einsum("bij,bj->bi", next_state, qhat)
    gated = out * F.silu(self.g_proj(h))
    gated = gated * torch.rsqrt(gated.pow(2).mean(-1, keepdim=True) + 1e-6) * self.o_norm_w
    next_h = self.norm(h + self.out_proj(gated))
    return next_h, next_state


def build(mcfg: MQARConfig, state_type: str, seed: int) -> TTTLinearLM:
    torch.manual_seed(seed)
    model = TTTLinearLM(TTTLinearConfig(vocab_size=mcfg.vocab_size, block_size=mcfg.seq_len,
                                        n_layer=1, n_embd=128, state_dim=128, dropout=0.0))
    layer = model.transformer["h"][0]
    layer.state_type = state_type
    layer.step = types.MethodType(_ablation_step, layer)
    with torch.no_grad():
        layer.A_log.fill_(math.log(1.0))  # 保持寄り init (delta/vanilla では state 更新に無関与だが無害)
    return model


def hopfield_gain(model: TTTLinearLM, mcfg: MQARConfig, seed: int, *, eval_batches: int
                  ) -> dict:
    """R-Hopfield(val で τ 選択)の R0 に対する gain と paired CI を測る。"""
    Sv, Hv, Qv, Yv = collect_states(model, mcfg, torch.Generator().manual_seed(seed + 1),
                                    batches=eval_batches, batch_size=64)
    St, Ht, Qt, Yt = collect_states(model, mcfg, torch.Generator().manual_seed(seed + 2),
                                    batches=eval_batches, batch_size=64)

    def val_rec(tau: float, K: int) -> float:
        return float(recall_of(model, Sv, Hv, Qv, Yv,
                               lambda S, Q: R.r_hopfield(S, Q, K=K, tau=tau)).mean())

    best_tau, best_K = max(((t, K) for t in [0.0, 0.01, 0.05, 0.1, 0.3] for K in [1, 3, 5]),
                           key=lambda tk: val_rec(tk[0], tk[1]))
    r0c = recall_of(model, St, Ht, Qt, Yt, R.r0)
    hopc = recall_of(model, St, Ht, Qt, Yt,
                     lambda S, Q: R.r_hopfield(S, Q, K=best_K, tau=best_tau))
    d, lo, hi = paired_ci(hopc, r0c)
    return {"r0": float(r0c.mean()), "hopfield": float(hopc.mean()),
            "gain": d, "ci": [lo, hi], "sig": bool(lo > 0), "tau": best_tau, "K": best_K}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--num-pairs", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-batches", type=int, default=12)
    args = p.parse_args(argv)

    types_ = ["gated_delta", "delta_rule", "vanilla_additive"]
    results: dict = {st: [] for st in types_}
    for st in types_:
        for seed in args.seeds:
            mcfg = MQARConfig(num_keys=16, num_pairs=args.num_pairs,
                              num_queries=args.num_pairs, seed=seed)
            model = build(mcfg, st, seed)
            train(model, mcfg, steps=args.train_steps, lr=5e-3, batch_size=64, seed=seed)
            g = hopfield_gain(model, mcfg, seed, eval_batches=args.eval_batches)
            results[st].append(g)
            print(f"  {st:<16} seed{seed}: R0={g['r0']:.3f} Hopfield={g['hopfield']:.3f} "
                  f"gain={g['gain']:+.3f} CI[{g['ci'][0]:+.3f},{g['ci'][1]:+.3f}]"
                  f"{'*' if g['sig'] else ''} (τ={g['tau']},K={g['K']})")

    print(f"\n{'state_type':<18} {'mean R0':>8} {'mean gain':>10} {'sig seeds':>10}")
    summary = {}
    for st in types_:
        rs = results[st]
        mr0 = sum(r["r0"] for r in rs) / len(rs)
        mg = sum(r["gain"] for r in rs) / len(rs)
        sig = sum(r["sig"] and r["gain"] > 0 for r in rs)
        summary[st] = {"mean_r0": mr0, "mean_gain": mg, "sig_seeds": sig, "n": len(rs)}
        print(f"{st:<18} {mr0:>8.3f} {mg:>+10.3f} {sig:>7}/{len(rs)}")

    gd = summary["gated_delta"]
    dr = summary["delta_rule"]
    va = summary["vanilla_additive"]
    n = len(args.seeds)
    gain_on_gated = gd["sig_seeds"] >= (n + 1) // 2 or dr["sig_seeds"] >= (n + 1) // 2
    gain_on_vanilla = va["sig_seeds"] >= (n + 1) // 2
    if gain_on_gated and not gain_on_vanilla:
        verdict = ("H3 SUPPORTED: cleanup gain は gated/delta で出て vanilla-additive で消える "
                   "→ novelty regime (状態依存) を裏付け。full GO 条件を満たす方向。")
    elif gain_on_gated and gain_on_vanilla:
        verdict = ("H3 NOT SUPPORTED: vanilla-additive でも gain が出る → 状態非依存の一般 read "
                   "トリック。novelty 縮小 (どの線形状態でも効く cleanup)。")
    else:
        verdict = "H3 INCONCLUSIVE: gated/delta でも gain が頑健でない (Stage2 と不整合 → 要再確認)。"
    print(f"\n[VERDICT/H3] {verdict}")

    out = _HERE / "h3_results.json"
    out.write_text(json.dumps({"config": vars(args), "summary": summary,
                               "verdict": verdict, "per_seed": results},
                              ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[h3] results → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
