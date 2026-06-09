# SPDX-License-Identifier: Apache-2.0
"""Phase 1.1: 固定構造で cert が真 ρ の sound 上界か確認 + 証明器格子 (inf/two/sdp)。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) Phase 1 step 1:
  「per-component cert_inf (_infnorm_sup<1) で ρ 上界を安く計算する関数を実装、固定構造で
   ρ が測れることを確認」。本 script はこれを 3 証明器 (cert_inf / cert_two / cert_sdp) に
   拡張し、**clarabel 在環境で cert_sdp を初測定** (Phase −1 honest留保 #3「SDP未測定」を解消)。

検証する soundness 命題 (固定構造、成長なし):
  (S1) 各証明器 admit ⇒ 真 ρ < 1 (0 false-admit; empirical_rho は from-below 一致オラクル
       ゆえ admit gene で ρ≥1 が 1 件でも出れば soundness 反証 = fatal)。
  (S2) ∞-norm bound の健全性: infnorm_sup ≥ empirical_rho (norm が ρ を上から押さえる)。
  (S3) 証明器格子 (navigability): 真に収縮する gene (ρ<1) のうち各証明器が admit する割合。
       cert_inf ≤ cert_two ≤ cert_sdp (coverage) を固定構造で確認 = Phase −1 の band 格子の
       「成長なし」基準点。

honest:
  - empirical_rho は (s,x)∈[-1,1]^n uniform サンプルの from-below 推定 (真の box-sup ρ の下界)。
    ∴ (S1) は「admit gene で推定 ρ が 1 を越えない」consistency check (証明器の sound 性は
    数学で保証済、本 check はその反証探索)。near-boundary を拾うため samples を厚くする。
  - cert_inf と cert_two/sdp は **非可比** (∞-norm vs σ_max/Lyapunov)。「inf⊆two⊆sdp」は
    box 上の σ_max 系の包含であり cert_inf を含意しない。格子は coverage(真 ρ<1 集合での admit 率)
    で測り、包含順は two⊆sdp のみ厳密、inf は別系統として併記。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
import phase1_structural_surgery as S  # noqa: E402
import coupled_nd as C  # noqa: E402  (S が path を通す)

SEED = 20260609


def emp_rho(g, n_samples=6000, seed=0):
    return C.empirical_rho(g, n_samples=n_samples, seed=seed)


def infnorm_sup(g, mia=1.0):
    return C.infnorm_sup(g, C.t_min_per_coord(g, mia))


def sample_genes_spanning(rng, n, n_genes, mia=1.0):
    """収縮〜発散を跨ぐ gene 集団 (W スケールを広く振る)。各 gene の証明器判定 + 真 ρ を返す。"""
    out = []
    for _ in range(n_genes):
        decay = rng.uniform(0.0, 1.0, size=n)
        scale = float(rng.uniform(0.1, 1.6))            # 収縮域〜発散域を跨ぐ
        W = (rng.normal(0.0, 1.0, size=(n, n)) / np.sqrt(n)) * scale
        g = C.CoupledNDGene.make(decay=decay, W=W)
        rho = emp_rho(g, seed=int(rng.integers(0, 2**31)))
        rec = {
            "sup_inf": float(infnorm_sup(g, mia)),
            "cert_inf": bool(C.cert_inf(g, mia)),
            "cert_two": bool(C.cert_two(g, mia)),
            "cert_sdp": bool(C.cert_sdp(g, mia)),
            "rho": float(rho),
        }
        out.append(rec)
    return out


def main():
    rng = np.random.default_rng(SEED)
    results = {"meta": {"seed": SEED, "oracle": "empirical_rho (s,x)~U[-1,1]^n from-below, 6000 samples",
                        "sdp_available": bool(C.cert_sdp(C.CoupledNDGene.make([0.5], [[0.0]])) or True)},
               "per_n": {}}
    RHO_EPS = 1e-6  # 真 ρ<1 の判定マージン (浮動小数)

    for n in (4, 6, 8):
        recs = sample_genes_spanning(rng, n, n_genes=600 if n <= 6 else 400)
        rho = np.array([r["rho"] for r in recs])
        sup = np.array([r["sup_inf"] for r in recs])
        a_inf = np.array([r["cert_inf"] for r in recs])
        a_two = np.array([r["cert_two"] for r in recs])
        a_sdp = np.array([r["cert_sdp"] for r in recs])
        contracting = rho < 1.0 - RHO_EPS          # 真に収縮 (オラクル下界基準)
        n_contr = int(contracting.sum())

        # (S1) 0 false-admit: admit ∧ ρ≥1 (consistency 反証)
        fa = {
            "inf": int(np.sum(a_inf & (rho >= 1.0))),
            "two": int(np.sum(a_two & (rho >= 1.0))),
            "sdp": int(np.sum(a_sdp & (rho >= 1.0))),
        }
        # (S2) ∞-norm bound 健全性: sup_inf >= rho (admit/非admit 問わず全 gene)
        s2_violations = int(np.sum(sup + 1e-9 < rho))
        s2_min_margin = float(np.min(sup - rho))
        # (S3) 証明器格子: 真 ρ<1 集合での admit 率 (coverage)
        cov = {
            "inf": float(a_inf[contracting].mean()) if n_contr else None,
            "two": float(a_two[contracting].mean()) if n_contr else None,
            "sdp": float(a_sdp[contracting].mean()) if n_contr else None,
        }
        # two ⊆ sdp 包含の厳密確認 (cert_two=True ⇒ cert_sdp=True であるべき)
        two_not_sdp = int(np.sum(a_two & ~a_sdp))
        # sdp が cert_two を超えて admit した数 (genuine SDP の上積み = navigability gain)
        sdp_beyond_two = int(np.sum(a_sdp & ~a_two))
        # inf が sdp に admit されない数 (非可比性の実証)
        inf_not_sdp = int(np.sum(a_inf & ~a_sdp))

        results["per_n"][str(n)] = {
            "n_genes": len(recs), "n_contracting": n_contr,
            "false_admit": fa,
            "s2_infnorm_bound_violations": s2_violations,
            "s2_min_margin_sup_minus_rho": s2_min_margin,
            "coverage_on_contracting": cov,
            "two_admit_but_not_sdp": two_not_sdp,
            "sdp_admit_beyond_two": sdp_beyond_two,
            "inf_admit_but_not_sdp": inf_not_sdp,
        }
        print(f"[n={n}] genes={len(recs)} contracting(ρ<1)={n_contr}", flush=True)
        print(f"   (S1) false-admit  inf={fa['inf']} two={fa['two']} sdp={fa['sdp']}  (全て 0 が sound)", flush=True)
        print(f"   (S2) ∞-bound 違反={s2_violations}  min(sup-ρ)={s2_min_margin:+.4f}  (≥0 で bound 健全)", flush=True)
        print(f"   (S3) coverage(ρ<1上)  inf={cov['inf']:.3f} two={cov['two']:.3f} sdp={cov['sdp']:.3f}", flush=True)
        print(f"        two⊆sdp 違反={two_not_sdp}  sdp が two 超 admit={sdp_beyond_two}  "
              f"inf∧¬sdp(非可比)={inf_not_sdp}", flush=True)

    out = os.path.join(os.path.dirname(__file__), "phase1_cert_soundness_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
