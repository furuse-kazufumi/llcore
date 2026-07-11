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


def run_once(args: argparse.Namespace, seed: int) -> dict:
    """1 seed 分: 学習 → val でハイパラ選択 → test で per-instance correct + 効果を返す。"""
    torch.manual_seed(seed)
    mcfg = MQARConfig(num_keys=args.num_keys, num_pairs=args.num_pairs,
                      num_queries=args.num_queries, seed=seed)
    model = TTTLinearLM(TTTLinearConfig(
        vocab_size=mcfg.vocab_size, block_size=mcfg.seq_len,
        n_layer=1, n_embd=128, state_dim=args.state_dim, dropout=0.0))
    with torch.no_grad():
        model.transformer["h"][0].A_log.fill_(math.log(args.a_log_init))
    train(model, mcfg, steps=args.train_steps, lr=args.lr, batch_size=args.batch_size, seed=seed)

    val = collect_states(model, mcfg, torch.Generator().manual_seed(seed + 1),
                         batches=args.eval_batches, batch_size=args.batch_size)
    test = collect_states(model, mcfg, torch.Generator().manual_seed(seed + 2),
                          batches=args.eval_batches, batch_size=args.batch_size)
    Sv, Hv, Qv, Yv = val
    St, Ht, Qt, Yt = test
    assert torch.allclose(R.r0(Sv, Qv), torch.einsum("bij,bj->bi", Sv, Qv)), "R0 != native"

    def val_recall(fn) -> float:
        return float(recall_of(model, Sv, Hv, Qv, Yv, fn).mean())

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
    corr = {name: recall_of(model, St, Ht, Qt, Yt, fn) for name, fn in variants.items()}
    r0c, ccqc, fyc = corr["R0(single)"], corr["R-CCQ"], corr["sparse-FY-Hopfield"]
    out: dict = {"seed": seed, "n_test": int(len(Yt)),
                 "hparams": {"ccq_lam": best_ccq, "hop": best_hop, "ista": best_ista, "beta": best_beta},
                 "variants": {}}
    for name, c in corr.items():
        d0, lo0, hi0 = paired_ci(c, r0c)
        dq, loq, hiq = paired_ci(c, ccqc)
        di, loi, hii = paired_ci(c, fyc)  # vs sparse-FY (反復が疎性を超えるか)
        out["variants"][name] = {
            "recall": float(c.mean()),
            "d_r0": d0, "ci_r0": [lo0, hi0], "sig_r0": bool(lo0 > 0 or hi0 < 0),
            "d_ccq": dq, "ci_ccq": [loq, hiq],
            "d_fy": di, "ci_fy": [loi, hii],
            "correct": [int(x) for x in c.tolist()],  # 記事/追試用に per-instance を保存
        }
    return out


def _verdict(seed_results: list[dict]) -> str:
    """複数 seed の集約判定 (H1/H2)。反復 read が R0 かつ R-CCQ を CI 超で上回る seed 数で判定。"""
    iters = ["R-ISTA", "R-Hopfield"]
    go = par = 0
    for r in seed_results:
        v = r["variants"]
        beats_r0 = any(v[n]["ci_r0"][0] > 0 for n in iters)
        beats_ccq = any(v[n]["ci_ccq"][0] > 0 for n in iters)
        if beats_r0 and beats_ccq:
            go += 1
        elif beats_r0:
            par += 1
    n = len(seed_results)
    if go == n:
        return f"GO ({go}/{n} seed で H1∧H2: 反復 read が R0 かつ R-CCQ を CI 超) — 要 H3/大N/長系列で追確認"
    if go + par == n and go >= 1:
        return f"GO-leaning ({go}/{n} GO, {par}/{n} PARTIAL) — 効果は小/疎性寄与に注意"
    if par + go >= 1:
        return f"MIXED ({go} GO / {par} PARTIAL / {n - go - par} NULL) — 頑健でない"
    return f"NULL (0/{n} で CI 超えなし) → read 側 test-time は dead / fork C へ pivot"


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
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--eval-batches", type=int, default=12)
    p.add_argument("--out", type=str, default="research/poc1_testtime_read/stage2_results.json")
    args = p.parse_args(argv)

    results = []
    for s in args.seeds:
        print(f"[stage2] seed {s}: train {args.train_steps} steps, pairs={args.num_pairs} ...")
        r = run_once(args, s)
        results.append(r)
        v = r["variants"]
        print(f"  R0={v['R0(single)']['recall']:.3f} "
              + " ".join(f"{n.split('(')[0].replace('R-','')}={v[n]['recall']:.3f}"
                         f"({v[n]['d_r0']:+.3f}{'*' if v[n]['sig_r0'] else ''})"
                         for n in ["R-ISTA", "R-Hopfield", "R-CCQ", "sparse-FY-Hopfield", "softmax-Hopfield"]))

    # 集約表
    names = ["R0(single)", "R-CCQ", "softmax-Hopfield", "sparse-FY-Hopfield", "R-Hopfield", "R-ISTA"]
    print(f"\n{'variant':<20} {'mean recall':>11} {'mean Δ vs R0':>13} {'sig seeds':>10} "
          f"{'mean Δ vs sparse-FY':>19}")
    for name in names:
        recs = [r["variants"][name]["recall"] for r in results]
        d0s = [r["variants"][name]["d_r0"] for r in results]
        sig = sum(r["variants"][name]["sig_r0"] and r["variants"][name]["d_r0"] > 0 for r in results)
        dfy = [r["variants"][name]["d_fy"] for r in results]
        print(f"{name:<20} {sum(recs)/len(recs):>11.3f} {sum(d0s)/len(d0s):>+13.3f} "
              f"{sig:>7}/{len(results)} {sum(dfy)/len(dfy):>+19.3f}")

    verdict = _verdict(results)
    print(f"\n[VERDICT/H1H2] {verdict}")

    out_path = _ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import json
    payload = {"config": vars(args), "verdict": verdict, "seeds": results}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[stage2] results → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
