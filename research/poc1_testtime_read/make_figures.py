# SPDX-License-Identifier: Apache-2.0
"""PoC-1 Stage 2 の記事素材 (図) を生成する。「動き」= 反復 read が段階的に recall を回復する様子。

出力 (research/poc1_testtime_read/figures/):
  - recall_vs_K.png    : 反復回数 K に対する recall (R-ISTA/R-Hopfield の cleanup 動学) + 参照線
  - variants_bar.png   : read 変種ごとの test recall (95% CI エラーバー)
  - cleanup_prob.png    : 1 クエリの「正解値トークンの確率」が反復で鋭くなる様子
  - figures_data.json  : 図の元データ (追試/多言語記事用)

honest: 効果は小 (+0.018) だが 3/3 seed 有意。CPU tiny・H3 未実施・ceiling-relaxation。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for pth in (str(_ROOT / "src"), str(_HERE)):
    if pth not in sys.path:
        sys.path.insert(0, pth)

from llcore.lm.ttt import TTTLinearConfig, TTTLinearLM  # noqa: E402

import reads as R  # noqa: E402
from mqar import MQARConfig  # noqa: E402
from run_stage2 import (collect_states, downstream_logits,  # noqa: E402
                        paired_ci, recall_of, train)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--train-steps", type=int, default=400)
    p.add_argument("--num-pairs", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    torch.manual_seed(args.seed)
    mcfg = MQARConfig(num_keys=16, num_pairs=args.num_pairs, num_queries=args.num_pairs, seed=args.seed)
    model = TTTLinearLM(TTTLinearConfig(vocab_size=mcfg.vocab_size, block_size=mcfg.seq_len,
                                        n_layer=1, n_embd=128, state_dim=128, dropout=0.0))
    with torch.no_grad():
        model.transformer["h"][0].A_log.fill_(math.log(1.0))
    print(f"[fig] training pairs={args.num_pairs} ...")
    train(model, mcfg, steps=args.train_steps, lr=5e-3, batch_size=64, seed=args.seed)

    Sv, Hv, Qv, Yv = collect_states(model, mcfg, torch.Generator().manual_seed(args.seed + 1),
                                    batches=12, batch_size=64)
    St, Ht, Qt, Yt = collect_states(model, mcfg, torch.Generator().manual_seed(args.seed + 2),
                                    batches=12, batch_size=64)

    def val_rec(fn) -> float:
        return float(recall_of(model, Sv, Hv, Qv, Yv, fn).mean())

    def test_corr(fn) -> torch.Tensor:
        return recall_of(model, St, Ht, Qt, Yt, fn)

    # val でハイパラ選択 (K は動学のため後で sweep)
    lam = max([0.001, 0.003, 0.01, 0.03, 0.1],
              key=lambda L: val_rec(lambda S, Q: R.r_ista(S, Q, K=5, lam=L, eta=0.3)))
    eta = max([0.1, 0.3, 1.0], key=lambda e: val_rec(lambda S, Q: R.r_ista(S, Q, K=5, lam=lam, eta=e)))
    tau = max([0.0, 0.01, 0.05, 0.1], key=lambda t: val_rec(lambda S, Q: R.r_hopfield(S, Q, K=5, tau=t)))

    figdir = _HERE / "figures"
    figdir.mkdir(exist_ok=True)
    data: dict = {"config": vars(args), "ista": {"lam": lam, "eta": eta}, "hopfield": {"tau": tau}}

    # ── 図1: recall vs K (cleanup 動学) ──
    Ks = list(range(0, 8))
    ista_rec, hop_rec = [], []
    for K in Ks:
        if K == 0:
            r = float(test_corr(R.r0).mean())  # K=0 = 単発 read
            ista_rec.append(r)
            hop_rec.append(r)
        else:
            ista_rec.append(float(test_corr(lambda S, Q: R.r_ista(S, Q, K=K, lam=lam, eta=eta)).mean()))
            hop_rec.append(float(test_corr(lambda S, Q: R.r_hopfield(S, Q, K=K, tau=tau)).mean()))
    r0 = ista_rec[0]
    ccq_lam = max([0.001, 0.003, 0.01, 0.03, 0.1], key=lambda L: val_rec(lambda S, Q: R.r_ccq(S, Q, lam=L)))
    ref_ccq = float(test_corr(lambda S, Q: R.r_ccq(S, Q, lam=ccq_lam)).mean())
    ref_fy = float(test_corr(R.r_fy_hopfield).mean())
    data["recall_vs_K"] = {"K": Ks, "ista": ista_rec, "hopfield": hop_rec,
                           "r0": r0, "ccq": ref_ccq, "sparse_fy": ref_fy}

    plt.figure(figsize=(6.2, 4.2))
    plt.plot(Ks, ista_rec, "o-", label="R-ISTA (iterative sparse read)", color="#2563eb")
    plt.plot(Ks, hop_rec, "s-", label="R-Hopfield (iterative cleanup)", color="#16a34a")
    plt.axhline(r0, ls="--", color="#6b7280", label=f"R0 single read = {r0:.3f}")
    plt.axhline(ref_ccq, ls=":", color="#f59e0b", label=f"R-CCQ single contraction = {ref_ccq:.3f}")
    plt.axhline(ref_fy, ls=":", color="#a855f7", label=f"sparse-FY 1-step = {ref_fy:.3f}")
    plt.xlabel("read iterations K (test-time, state frozen)")
    plt.ylabel("recall (top-1)")
    plt.title("Post-hoc iterative read recovers recall from a frozen\ngated-delta state (MQAR, CPU tiny)")
    plt.legend(fontsize=8, loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "recall_vs_K.png", dpi=130)
    plt.close()

    # ── 図2: 変種バー (95% CI) ──
    variants = {
        "R0\n(single)": R.r0,
        "R-CCQ": lambda S, Q: R.r_ccq(S, Q, lam=ccq_lam),
        "sparse-FY\n(1-step)": R.r_fy_hopfield,
        "R-Hopfield\n(K=5)": lambda S, Q: R.r_hopfield(S, Q, K=5, tau=tau),
        "R-ISTA\n(K=5)": lambda S, Q: R.r_ista(S, Q, K=5, lam=lam, eta=eta),
    }
    corr = {n: test_corr(fn) for n, fn in variants.items()}
    r0c = corr["R0\n(single)"]
    labels, recs, errs, sigs = [], [], [], []
    for n, c in corr.items():
        rec = float(c.mean())
        d, lo, hi = paired_ci(c, r0c)
        labels.append(n)
        recs.append(rec)
        # CI エラーバーは recall 自身の bootstrap でなく R0 差の CI を recall 上に写像
        errs.append((rec - (r0 + lo), (r0 + hi) - rec) if n != "R0\n(single)" else (0, 0))
        sigs.append(lo > 0)
    data["variants_bar"] = {"labels": [x.replace("\n", " ") for x in labels], "recall": recs}
    plt.figure(figsize=(6.2, 4.2))
    colors = ["#6b7280", "#f59e0b", "#a855f7", "#16a34a", "#2563eb"]
    bars = plt.bar(range(len(labels)), recs, color=colors)
    lo_e = [e[0] for e in errs]
    hi_e = [e[1] for e in errs]
    plt.errorbar(range(len(labels)), recs, yerr=[lo_e, hi_e], fmt="none", ecolor="black", capsize=4)
    for i, (b, s) in enumerate(zip(bars, sigs)):
        if s:
            plt.text(i, recs[i] + 0.004, "*", ha="center", fontsize=14)
    plt.axhline(r0, ls="--", color="#6b7280", alpha=0.6)
    plt.xticks(range(len(labels)), labels, fontsize=8)
    plt.ylabel("recall (top-1)")
    plt.ylim(min(recs) - 0.02, max(recs) + 0.02)
    plt.title("Read-variant recall on the SAME frozen state (95% CI vs R0; * = significant)")
    plt.tight_layout()
    plt.savefig(figdir / "variants_bar.png", dpi=130)
    plt.close()

    # ── 図3: 1 クエリの正解確率が反復で鋭くなる (crosstalk 除去) ──
    idx = 0
    S1 = St[idx:idx + 1]
    H1 = Ht[idx:idx + 1]
    Q1 = Qt[idx:idx + 1]
    y1 = int(Yt[idx])
    probs = []
    for K in Ks:  # 安定な R-Hopfield を使う (ISTA は K で振動するため単一クエリ可視化に不適)
        o = R.r0(S1, Q1) if K == 0 else R.r_hopfield(S1, Q1, K=K, tau=tau)
        pr = F.softmax(downstream_logits(model, o, H1), dim=-1)[0, y1].item()
        probs.append(pr)
    data["cleanup_prob"] = {"K": Ks, "prob_correct": probs, "value_token": y1}
    plt.figure(figsize=(6.2, 4.0))
    plt.plot(Ks, probs, "o-", color="#2563eb")
    plt.xlabel("read iterations K")
    plt.ylabel("P(correct value token)")
    plt.title("Iterative read sharpens the correct-value probability\nfor one query (crosstalk removed)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(figdir / "cleanup_prob.png", dpi=130)
    plt.close()

    (figdir / "figures_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"[fig] recall_vs_K: R0={r0:.3f} → ISTA K=5={ista_rec[5]:.3f} / Hopfield K=5={hop_rec[5]:.3f}")
    print(f"[fig] saved 3 PNG + figures_data.json → {figdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
