# SPDX-License-Identifier: Apache-2.0
"""PoC-1 Stage 2 (本体): 凍結 Gated DeltaNet 状態への post-hoc read 変種比較。

pre-reg §3.2/§3.3/§4/§5。設計 (honest):
- 1 層モデル = 「単一の凍結状態 S」を明確化。各 query 位置で **native forward の next_state** を凍結し
  (R0 = セル native read と厳密一致)、同じ state に read 変種を適用。downstream(gate/norm/proj/lm_head)は
  学習済みを固定 → 変種は `o` だけを差し替える純粋な read 比較。
- ハイパラ (λ/τ/K/η/β) は **val split で選択 → test で評価** (test 上フィッティング禁止)。
- 指標: recall (top-1)。統計: **paired bootstrap CI** で (変種 − R0) と (変種 − R-CCQ)。
- 判定: pre-reg §5 (GO / PARTIAL / NULL)。H3 (状態種 ablation) は Stage 2b に分離。

usage: py -3.11 research/poc1_testtime_read/run_stage2.py --train-steps 400 --num-pairs 6
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from llcore.lm.ttt import TTTLinearConfig, TTTLinearLM  # noqa: E402

import reads as R  # noqa: E402
from mqar import MQARConfig, make_batch  # noqa: E402


def train(model: TTTLinearLM, mcfg: MQARConfig, *, steps: int, lr: float,
          batch_size: int, seed: int) -> None:
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(steps):
        model.train()
        inp, tgt, _ = make_batch(mcfg, batch_size, gen)
        _, loss = model(inp, tgt)
        assert loss is not None
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def collect_states(model: TTTLinearLM, mcfg: MQARConfig, gen: torch.Generator,
                   *, batches: int, batch_size: int
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """query 位置ごとに (S[next_state], h[layer入力], qhat, value_target) を集める。

    1 層専用。native forward を手で回し、各 query 位置の next_state を凍結対象として取り出す。
    R0(S,qhat) はセル native read と厳密一致する (検証: assert)。
    """
    model.eval()
    layer = model.transformer["h"][0]
    wte = model.transformer["wte"]
    S_list, H_list, Q_list, Y_list = [], [], [], []
    for _ in range(batches):
        inp, tgt, qpos = make_batch(mcfg, batch_size, gen)
        B, T = inp.shape
        state = torch.zeros(B, layer.state_dim, layer.state_dim)
        qset = {int(x) for x in qpos.reshape(-1).tolist()}
        # 位置→(batch,query行) の逆引きは使わず、各位置で全 batch を捕捉し query 位置だけ残す
        per_pos_target = torch.full((B, T), -1, dtype=torch.long)
        per_pos_target.scatter_(1, qpos, tgt.gather(1, qpos))
        for pos in range(T):
            h = wte(inp[:, pos])  # eval: dropout off  [B,d]
            qhat = F.normalize(layer.q_proj(h), p=2.0, dim=-1, eps=1e-6)
            _, next_state = layer.step(h, state)
            if pos in qset:
                y = per_pos_target[:, pos]
                m = y != -1
                if m.any():
                    S_list.append(next_state[m])
                    H_list.append(h[m])
                    Q_list.append(qhat[m])
                    Y_list.append(y[m])
            state = next_state
    return (torch.cat(S_list), torch.cat(H_list), torch.cat(Q_list), torch.cat(Y_list))


@torch.no_grad()
def downstream_logits(model: TTTLinearLM, o: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """read 出力 o[B,sd] と layer 入力 h[B,d] を学習済み downstream に通し logits[B,V] を返す。"""
    layer = model.transformer["h"][0]
    gated = o * F.silu(layer.g_proj(h))
    gated = gated * torch.rsqrt(gated.pow(2).mean(-1, keepdim=True) + 1e-6) * layer.o_norm_w
    nh = layer.norm(h + layer.out_proj(gated))
    return model.lm_head(model.transformer["ln_f"](nh))


@torch.no_grad()
def recall_of(model: TTTLinearLM, S: torch.Tensor, H: torch.Tensor, Q: torch.Tensor,
              Y: torch.Tensor, read_fn) -> torch.Tensor:
    """read_fn(S,qhat)->o を評価し、per-instance の correct [N] (0/1) を返す。"""
    o = read_fn(S, Q)
    pred = downstream_logits(model, o, H).argmax(-1)
    return (pred == Y).float()


def paired_ci(a: torch.Tensor, b: torch.Tensor, *, n_boot: int = 2000, seed: int = 0
              ) -> tuple[float, float, float]:
    """paired bootstrap で mean(a − b) の点推定と 95% CI を返す。"""
    d = (a - b).numpy()
    g = torch.Generator().manual_seed(seed)
    n = len(d)
    means = []
    for _ in range(n_boot):
        idx = torch.randint(0, n, (n,), generator=g).numpy()
        means.append(float(d[idx].mean()))
    means.sort()
    return float(d.mean()), means[int(0.025 * n_boot)], means[int(0.975 * n_boot)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--state-dim", type=int, default=128)
    p.add_argument("--num-keys", type=int, default=16)
    p.add_argument("--num-pairs", type=int, default=6)
    p.add_argument("--num-queries", type=int, default=6)
    p.add_argument("--a-log-init", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-batches", type=int, default=12)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    mcfg = MQARConfig(num_keys=args.num_keys, num_pairs=args.num_pairs,
                      num_queries=args.num_queries, seed=args.seed)
    model = TTTLinearLM(TTTLinearConfig(
        vocab_size=mcfg.vocab_size, block_size=mcfg.seq_len,
        n_layer=1, n_embd=128, state_dim=args.state_dim, dropout=0.0))
    with torch.no_grad():
        model.transformer["h"][0].A_log.fill_(math.log(args.a_log_init))
    print(f"[stage2] training 1-layer sd={args.state_dim} pairs={args.num_pairs} "
          f"chance={1/args.num_keys:.3f} ...")
    train(model, mcfg, steps=args.train_steps, lr=args.lr,
          batch_size=args.batch_size, seed=args.seed)

    # val / test で状態を収集 (別 generator = 別系列)
    val = collect_states(model, mcfg, torch.Generator().manual_seed(args.seed + 1),
                         batches=args.eval_batches, batch_size=args.batch_size)
    test = collect_states(model, mcfg, torch.Generator().manual_seed(args.seed + 2),
                          batches=args.eval_batches, batch_size=args.batch_size)
    Sv, Hv, Qv, Yv = val
    St, Ht, Qt, Yt = test
    print(f"[stage2] val N={len(Yv)} test N={len(Yt)}")

    # R0 が native read と一致することを確認 (sanity)
    r0_native = R.r0(Sv, Qv)
    assert torch.allclose(r0_native, torch.einsum("bij,bj->bi", Sv, Qv)), "R0 != native"

    def val_recall(fn) -> float:
        return float(recall_of(model, Sv, Hv, Qv, Yv, fn).mean())

    def test_correct(fn) -> torch.Tensor:
        return recall_of(model, St, Ht, Qt, Yt, fn)

    # ── ハイパラを val で選択 ──
    lam_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    tau_grid = [0.0, 0.01, 0.05, 0.1, 0.3]
    K_grid = [3, 5]
    eta_grid = [0.1, 0.3, 1.0]
    beta_grid = [0.5, 1.0, 2.0, 4.0]

    best_ccq = max(lam_grid, key=lambda L: val_recall(lambda S, Q: R.r_ccq(S, Q, lam=L)))
    best_hop = max(((K, t) for K in K_grid for t in tau_grid),
                   key=lambda kt: val_recall(lambda S, Q: R.r_hopfield(S, Q, K=kt[0], tau=kt[1])))
    best_ista = max(((K, L, e) for K in K_grid for L in lam_grid for e in eta_grid),
                    key=lambda kle: val_recall(
                        lambda S, Q: R.r_ista(S, Q, K=kle[0], lam=kle[1], eta=kle[2])))
    best_beta = max(beta_grid, key=lambda B: val_recall(lambda S, Q: R.r_softmax_hopfield(S, Q, beta=B)))

    variants = {
        "R0(single)": R.r0,
        "R-CCQ": lambda S, Q: R.r_ccq(S, Q, lam=best_ccq),
        "R-Hopfield": lambda S, Q: R.r_hopfield(S, Q, K=best_hop[0], tau=best_hop[1]),
        "R-ISTA": lambda S, Q: R.r_ista(S, Q, K=best_ista[0], lam=best_ista[1], eta=best_ista[2]),
        "softmax-Hopfield": lambda S, Q: R.r_softmax_hopfield(S, Q, beta=best_beta),
        "sparse-FY-Hopfield": R.r_fy_hopfield,
    }
    print(f"[stage2] val-selected: CCQ λ={best_ccq} Hopfield(K,τ)={best_hop} "
          f"ISTA(K,λ,η)={best_ista} softmax β={best_beta}")

    corr = {name: test_correct(fn) for name, fn in variants.items()}
    r0c = corr["R0(single)"]
    ccqc = corr["R-CCQ"]
    print(f"\n{'variant':<20} {'test recall':>11}   {'Δ vs R0 [95% CI]':>26}   {'Δ vs R-CCQ [95% CI]':>26}")
    for name, c in corr.items():
        rec = float(c.mean())
        d0, lo0, hi0 = paired_ci(c, r0c)
        dq, loq, hiq = paired_ci(c, ccqc)
        sig0 = "*" if (lo0 > 0 or hi0 < 0) else " "
        print(f"{name:<20} {rec:>11.3f}   {d0:+.3f} [{lo0:+.3f},{hi0:+.3f}]{sig0}   "
              f"{dq:+.3f} [{loq:+.3f},{hiq:+.3f}]")

    # ── 判定 (pre-reg §5, H1/H2; H3 は Stage 2b) ──
    def beats(name: str, ref: torch.Tensor) -> bool:
        _, lo, hi = paired_ci(corr[name], ref)
        return lo > 0
    iter_names = ["R-ISTA", "R-Hopfield"]
    beats_r0 = any(beats(n, r0c) for n in iter_names)
    beats_ccq = any(beats(n, ccqc) for n in iter_names)
    if beats_r0 and beats_ccq:
        verdict = "GO (H1∧H2 成立: 反復 read が R0 かつ R-CCQ を CI 超で上回る) — H3 で state 依存を確認へ"
    elif beats_r0:
        verdict = "PARTIAL (H1 のみ: R0 は超えるが R-CCQ は超えず → gain は『賢い単発』で説明可)"
    else:
        verdict = "NULL (いずれも CI 超えなし → read 側 test-time は dead / fork C へ pivot)"
    print(f"\n[VERDICT/H1H2] {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
