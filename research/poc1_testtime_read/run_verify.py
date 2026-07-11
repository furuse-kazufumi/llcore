# SPDX-License-Identifier: Apache-2.0
"""PoC-1 検証追試 (敵対的レビュー由来) — GO 判定の内訳を実測で settle する additive 診断。

pre-reg の判定ルール (§5) は変更しない。本スクリプトは Stage2/H3 の GO を昇格 (GPU スケール) する前に、
敵対的レビューが指摘した confound を CPU で直接検証する **追試** (`feedback_benchmark_honest_disclosure`)。

検証する問い:
  Q1 (機構): R-Hopfield の gain は「非 softmax soft-threshold cleanup」か、それとも val 選択の tau≈0 で
      soft-threshold が恒等化した結果の **S·SᵀS·q̂ (1回のべき乗反復=degree-2 spectral read)** か。
      → 素の r_poly2 = S·SᵀS·q̂ (閾値・チューニング無し) が R-Hopfield-full の gain を再現するか。
      → R-Hopfield-pos (tau を >0 に強制) で gain が残るか (真の疎性 cleanup が効いているか)。
  Q2 (FLOP-matched): r_poly2 と R-CCQ は共に 3 matvec (同一 compute)。SᵀS 項の符号だけが逆
      (poly2 = +S·SᵀS·q̂ / CCQ = S·q̂ − λ·S·SᵀS·q̂)。同 compute で poly2 だけ勝てば、gain は
      「計算を足しただけ」ではなく **degree-2 方向そのもの**が源。
  Q3 (再現性): seed を増やして +0.018 級の効果が CI つきで頑健か。
  Q4 (H3 headroom 交絡, --mode headroom): vanilla の gain 消失は状態構造か、R0 水準 (headroom) の交絡か。
      → 負荷 (num_pairs) を振って R0 水準を揃え、gain が状態種と R0 水準のどちらに従うか。

usage:
  py -3.11 research/poc1_testtime_read/run_verify.py --mode mechanism --seeds 0 1 2 3 4 5 6 7
  py -3.11 research/poc1_testtime_read/run_verify.py --mode headroom  --seeds 0 1 2 3
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _pth in (str(_ROOT / "src"), str(_HERE)):
    if _pth not in sys.path:
        sys.path.insert(0, _pth)

from llcore.lm.ttt import TTTLinearConfig, TTTLinearLM  # noqa: E402

import reads as R  # noqa: E402
from mqar import MQARConfig  # noqa: E402
from run_stage2 import collect_states, paired_ci, recall_of, train  # noqa: E402


# ─── 追加 read 変種 (診断用) ────────────────────────────────────────

def r_poly2(S: torch.Tensor, qhat: torch.Tensor) -> torch.Tensor:
    """degree-2 spectral read o = S·SᵀS·q̂ (閾値・正規化・チューニング一切なし)。

    R-Hopfield(K=1, tau=0) は o = S·normalize(SᵀS·q̂) であり、L1 正規化は downstream の gated-RMSNorm
    (scale 不変) が除去する → **r_poly2 と R-Hopfield(K=1,tau=0) は logits が厳密一致するはず**。
    「gain は疎性 cleanup でなく spectral emphasis」の帰無を体現する最も素朴な baseline。matvec=3。
    """
    return R._S(S, R._St(S, R._S(S, qhat)))  # S( Sᵀ( S q̂ ) )


# matvec (state 行列×ベクトル) の回数 = compute 代理指標。
MATVEC = {
    "R0": 1,
    "R-CCQ": 3,          # S q̂ ; Sᵀ(·) ; S(·)
    "r_poly2(S·SᵀS·q̂)": 3,
}


def _build_1layer(mcfg: MQARConfig, seed: int, *, a_log_init: float = 1.0) -> TTTLinearLM:
    torch.manual_seed(seed)
    model = TTTLinearLM(TTTLinearConfig(
        vocab_size=mcfg.vocab_size, block_size=mcfg.seq_len,
        n_layer=1, n_embd=128, state_dim=128, dropout=0.0))
    with torch.no_grad():
        model.transformer["h"][0].A_log.fill_(math.log(a_log_init))
    return model


def _mechanism_seed(args: argparse.Namespace, seed: int) -> dict:
    """1 seed: Stage2 と同一設定で学習 → val で選択 → test で機構診断。"""
    mcfg = MQARConfig(num_keys=args.num_keys, num_pairs=args.num_pairs,
                      num_queries=args.num_pairs, seed=seed,
                      unique_values=args.unique_values)
    model = _build_1layer(mcfg, seed)
    train(model, mcfg, steps=args.train_steps, lr=args.lr, batch_size=args.batch_size, seed=seed)

    Sv, Hv, Qv, Yv = collect_states(model, mcfg, torch.Generator().manual_seed(seed + 1),
                                    batches=args.eval_batches, batch_size=args.batch_size)
    St, Ht, Qt, Yt = collect_states(model, mcfg, torch.Generator().manual_seed(seed + 2),
                                    batches=args.eval_batches, batch_size=args.batch_size)

    def val_rec(fn) -> float:
        return float(recall_of(model, Sv, Hv, Qv, Yv, fn).mean())

    # ハイパラ選択 (val)。Hopfield-full は tau=0 込み、Hopfield-pos は tau>0 のみ。
    lam_grid = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    K_grid = [1, 3, 5]
    tau_all = [0.0, 0.01, 0.05, 0.1, 0.3]
    tau_pos = [0.01, 0.05, 0.1, 0.3]
    best_ccq = max(lam_grid, key=lambda L: val_rec(lambda S, Q: R.r_ccq(S, Q, lam=L)))
    best_full = max(((K, t) for K in K_grid for t in tau_all),
                    key=lambda kt: val_rec(lambda S, Q: R.r_hopfield(S, Q, K=kt[0], tau=kt[1])))
    best_pos = max(((K, t) for K in K_grid for t in tau_pos),
                   key=lambda kt: val_rec(lambda S, Q: R.r_hopfield(S, Q, K=kt[0], tau=kt[1])))

    variants = {
        "R0": R.r0,
        "R-CCQ": lambda S, Q: R.r_ccq(S, Q, lam=best_ccq),
        "R-Hopfield-full": lambda S, Q: R.r_hopfield(S, Q, K=best_full[0], tau=best_full[1]),
        "R-Hopfield-pos": lambda S, Q: R.r_hopfield(S, Q, K=best_pos[0], tau=best_pos[1]),
        "r_poly2(S·SᵀS·q̂)": r_poly2,
    }
    corr = {n: recall_of(model, St, Ht, Qt, Yt, fn) for n, fn in variants.items()}
    r0c = corr["R0"]
    ccqc = corr["R-CCQ"]
    poly2c = corr["r_poly2(S·SᵀS·q̂)"]

    out: dict = {"seed": seed, "n_test": int(len(Yt)),
                 "sel": {"ccq_lam": best_ccq, "hop_full_K_tau": list(best_full),
                         "hop_pos_K_tau": list(best_pos)},
                 "variants": {}}
    for n, c in corr.items():
        d0, lo0, hi0 = paired_ci(c, r0c)
        dq, loq, hiq = paired_ci(c, ccqc)
        dp, lop, hip = paired_ci(c, poly2c)
        out["variants"][n] = {
            "recall": float(c.mean()),
            "d_r0": d0, "ci_r0": [lo0, hi0], "sig_r0": bool(lo0 > 0),
            "d_ccq": dq, "ci_ccq": [loq, hiq], "sig_ccq": bool(loq > 0),
            "d_poly2": dp, "ci_poly2": [lop, hip],  # Hopfield が poly2 を超えるか (疎性の上乗せ)
        }
    # poly2 と Hopfield-full(tau=0時) の一致検算 (帰無の直接確認)
    out["poly2_matches_hopfield_full"] = bool(
        best_full[1] == 0.0 and torch.allclose(poly2c, corr["R-Hopfield-full"]))
    return out


def _mechanism(args: argparse.Namespace) -> dict:
    seeds = args.seeds
    per_seed = []
    for s in seeds:
        print(f"[verify/mech] seed {s}: train {args.train_steps} steps pairs={args.num_pairs} ...")
        r = _mechanism_seed(args, s)
        per_seed.append(r)
        v = r["variants"]
        print(f"  R0={v['R0']['recall']:.3f} | CCQ={v['R-CCQ']['recall']:.3f}({v['R-CCQ']['d_r0']:+.3f}) "
              f"| Hop-full={v['R-Hopfield-full']['recall']:.3f}({v['R-Hopfield-full']['d_r0']:+.3f}"
              f"{'*' if v['R-Hopfield-full']['sig_r0'] else ''},sel={r['sel']['hop_full_K_tau']}) "
              f"| Hop-pos={v['R-Hopfield-pos']['recall']:.3f}({v['R-Hopfield-pos']['d_r0']:+.3f}"
              f"{'*' if v['R-Hopfield-pos']['sig_r0'] else ''}) "
              f"| poly2={v['r_poly2(S·SᵀS·q̂)']['recall']:.3f}({v['r_poly2(S·SᵀS·q̂)']['d_r0']:+.3f}"
              f"{'*' if v['r_poly2(S·SᵀS·q̂)']['sig_r0'] else ''})")

    names = ["R0", "R-CCQ", "R-Hopfield-full", "R-Hopfield-pos", "r_poly2(S·SᵀS·q̂)"]
    n = len(seeds)
    print(f"\n{'variant':<20}{'matvec':>7}{'mean rec':>10}{'d vs R0':>10}{'sig/n':>8}"
          f"{'d vs poly2':>12}")
    agg = {}
    for name in names:
        recs = [r["variants"][name]["recall"] for r in per_seed]
        d0 = [r["variants"][name]["d_r0"] for r in per_seed]
        sig = sum(r["variants"][name]["sig_r0"] for r in per_seed)
        dp = [r["variants"][name]["d_poly2"] for r in per_seed]
        agg[name] = {"mean_recall": sum(recs) / n, "mean_d_r0": sum(d0) / n,
                     "sig_r0_seeds": sig, "mean_d_poly2": sum(dp) / n, "n": n}
        print(f"{name:<20}{MATVEC.get(name, '?'):>7}{sum(recs)/n:>10.3f}{sum(d0)/n:>+10.3f}"
              f"{sig:>5}/{n}{sum(dp)/n:>+12.3f}")

    # tau=0 選択率 (機構が spectral か)
    tau0_frac = sum(1 for r in per_seed if r["sel"]["hop_full_K_tau"][1] == 0.0) / n
    poly2_match = sum(1 for r in per_seed if r["poly2_matches_hopfield_full"])
    full_gain = agg["R-Hopfield-full"]["mean_d_r0"]
    pos_gain = agg["R-Hopfield-pos"]["mean_d_r0"]
    poly2_gain = agg["r_poly2(S·SᵀS·q̂)"]["mean_d_r0"]
    ccq_gain = agg["R-CCQ"]["mean_d_r0"]

    # honest 判定 (機構ラベル)
    spectral = poly2_gain >= 0.6 * full_gain and full_gain > 0
    thresh_adds = pos_gain >= full_gain + 0.005  # tau>0 が full を明確に超えるか
    flop_specific = poly2_gain > 0 and ccq_gain <= 0.005  # 同 compute で方向依存
    verdict = []
    verdict.append(f"tau=0 selection rate={tau0_frac:.0%} (spectral 機構の傍証)")
    verdict.append(f"poly2==Hopfield-full match seeds={poly2_match}/{n}")
    if spectral and not thresh_adds:
        verdict.append("=> mechanism is SPECTRAL (S·SᵀS·q̂ / degree-2 power-iteration). "
                       "'soft-threshold cleanup' label must be corrected; sparsity adds nothing.")
    elif thresh_adds:
        verdict.append("=> genuine tau>0 soft-threshold beats spectral = sparsity cleanup has substance.")
    else:
        verdict.append("=> mixed/unclear; need more seeds.")
    if flop_specific:
        verdict.append("FLOP-matched: same 3-matvec, poly2(+SᵀS) wins while R-CCQ(-lam·SᵀS) does not "
                       "=> gain is degree-2 direction, not compute (H2 kill-risk passed from another angle).")
    return {"mode": "mechanism", "config": vars(args), "per_seed": per_seed,
            "aggregate": agg, "tau0_frac": tau0_frac, "poly2_match_seeds": poly2_match,
            "verdict_notes": verdict}


# ─── H3 headroom 交絡 (--mode headroom) ─────────────────────────────

def _headroom(args: argparse.Namespace) -> dict:
    """状態種 × 負荷 (num_pairs) を振り、gain が「状態種」か「R0 水準」のどちらに従うかを分離。

    run_h3 の build/hopfield_gain を再利用。vanilla を高負荷 (R0↓) に、gated を低負荷 (R0↑) に置いて、
    (R0, gain) の (状態種, 負荷) 依存を見る。gain が R0 だけで決まるなら headroom 交絡、
    状態種で決まるなら真の状態依存。
    """
    from run_h3 import build, hopfield_gain  # noqa: E402

    types_ = ["gated_delta", "vanilla_additive"]
    loads = args.pairs_grid
    rows = []
    for st in types_:
        for npair in loads:
            for seed in args.seeds:
                mcfg = MQARConfig(num_keys=16, num_pairs=npair, num_queries=npair, seed=seed)
                model = build(mcfg, st, seed)
                train(model, mcfg, steps=args.train_steps, lr=5e-3,
                      batch_size=args.batch_size, seed=seed)
                g = hopfield_gain(model, mcfg, seed, eval_batches=args.eval_batches)
                rows.append({"state": st, "pairs": npair, "seed": seed,
                             "r0": g["r0"], "gain": g["gain"], "sig": g["sig"],
                             "tau": g["tau"], "K": g["K"]})
                print(f"  {st:<16} pairs={npair} seed{seed}: R0={g['r0']:.3f} "
                      f"gain={g['gain']:+.3f}{'*' if g['sig'] else ''} (tau={g['tau']},K={g['K']})")

    summary = {}
    for st in types_:
        for npair in loads:
            sub = [r for r in rows if r["state"] == st and r["pairs"] == npair]
            m = len(sub)
            summary[f"{st}@{npair}"] = {
                "mean_r0": sum(r["r0"] for r in sub) / m,
                "mean_gain": sum(r["gain"] for r in sub) / m,
                "sig_seeds": sum(r["sig"] and r["gain"] > 0 for r in sub), "n": m}

    print(f"\n{'state@pairs':<24}{'mean R0':>9}{'mean gain':>11}{'sig':>6}")
    for k, v in summary.items():
        print(f"{k:<24}{v['mean_r0']:>9.3f}{v['mean_gain']:>+11.3f}{v['sig_seeds']:>4}/{v['n']}")

    notes = ("headroom confound check: if vanilla at higher load (R0 lowered to gated level) still shows "
             "no gain, then H3 SUPPORTED reflects state-type dependence, not R0-level headroom. If gain "
             "appears once vanilla R0 drops, the H3 conclusion is confounded by headroom.")
    return {"mode": "headroom", "config": vars(args), "rows": rows,
            "summary": summary, "notes": notes}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["mechanism", "headroom"], default="mechanism")
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-keys", type=int, default=16)
    p.add_argument("--num-pairs", type=int, default=6)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    p.add_argument("--eval-batches", type=int, default=12)
    p.add_argument("--unique-values", action="store_true",
                   help="MQAR value を非復元 (系列内一意) にし binding-artifact を排除して検証")
    p.add_argument("--pairs-grid", type=int, nargs="+", default=[4, 6, 10],
                   help="headroom モードで振る負荷 (num_pairs)")
    p.add_argument("--out", type=str, default="")
    args = p.parse_args(argv)

    if args.mode == "mechanism":
        result = _mechanism(args)
        default_out = "research/poc1_testtime_read/verify_mechanism.json"
    else:
        result = _headroom(args)
        default_out = "research/poc1_testtime_read/verify_headroom.json"

    print("\n[VERDICT NOTES]")
    for ln in result.get("verdict_notes", [result.get("notes", "")]):
        print(f"  - {ln}")

    out_path = _ROOT / (args.out or default_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[verify] -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
