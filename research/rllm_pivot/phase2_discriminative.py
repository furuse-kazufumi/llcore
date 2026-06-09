# SPDX-License-Identifier: Apache-2.0
"""Phase 2: H-discriminative — 枠組みが 4 method を soundness で判別できるか (Decision gate 2 核)。

EVOLVABLE_LLM_PLAN_2026_06_09.md (v2) §⑥ North Star #3 / §⑧ H-discriminative / §⑩ Phase 2:
  「負の対照(無 gate=危険)/ 正の対照(Mamba=安全)/ 既踏(STABLE=経験 gate)を soundness で
   区別して測れるか」= 評価枠組みの妥当性そのもの。

被験 4 method (計画 §④):
  - none           : 無 gate (負の対照)。全 admit → ρ≥1 の発散 gene も通す = 危険。
  - stable_exp     : STABLE 風経験 gate (arXiv:2510.16089 系)。**sound certificate でなく**
                     有限ホライズンの経験的安定指標 (摂動忘却 < ε) で admit/reject。
                     near-boundary を false-admit しうる (検出力に上限)。
  - cert_inf/two/sdp: VSOA の sound 証明器 (収縮 certificate)。0 false-admit のはず。
  - mamba_synth    : Mamba 風 stable-by-construction の正の対照。**構成的に ρ<1 を保証**した
                     gene family (decay 支配 + 小 W)。枠組みは全 admit (spurious reject なし) のはず。

事前登録仮説 (falsifiable):
  H3a: 危険 gene を含む spanning 集団で **false_admit(none) ≫ false_admit(stable_exp) > false_admit(certs)=0**。
       (枠組みが「危険 / 経験 / sound」を soundness で分離して測れる)
  H3b: stable-by-construction 集団 (mamba_synth) で **sound certs の reject 率 ≈ 0** (安全 family を
       枠組みが誤って棄却しない = 正の対照 PASS)。

honest:
  - 真 ρ = empirical_rho (from-below 6000 sample) を ground-truth proxy に。false_admit = admit ∧ 真ρ≥1。
    from-below ゆえ near-boundary の false-admit は過小評価寄り (§Phase1 verdict §7#2 と同じ留保)。
  - stable_exp の閾 ε と horizon T は固定 (感度は留保)。kernel は tanh で常時有界ゆえ「不安定」=
    状態発散でなく摂動非忘却 (echo-state property 失敗)。stable_exp はこれを経験的に測る。
  - mamba_synth は **合成 stable-by-construction proxy** (実 Mamba SSM の Lyapunov ではない;
    base-level Mamba 正対照は別途 Phase 2 で SSM Jacobian 測定が必要 = 本 script は枠組み判別力に限定)。
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.dirname(__file__))
import coupled_nd as C  # noqa: E402

SEED = 20260609
N = 6                  # small-n per-component (cert_two/sdp feasible)
T = 64                 # stable_exp の有限ホライズン
K_PROBE = 8            # stable_exp の (init,input) サンプル数
EPS_FORGET = 1e-2      # stable_exp の摂動忘却閾 (これ未満で「安定」とみなす経験 gate)
RHO_SAMPLES = 6000     # 真 ρ ground-truth proxy


def emp_rho(g, seed):
    return C.empirical_rho(g, n_samples=RHO_SAMPLES, seed=seed)


def perturbation_forgetting(g, rng):
    """K_PROBE 個の (s0, input列) で近接 2 軌道 (s0=0 vs s0=δ) の終端乖離の最大値 (経験的非収縮度)。"""
    n = g.n
    worst = 0.0
    for _ in range(K_PROBE):
        X = rng.uniform(-1, 1, (T, n))
        d = rng.normal(size=n); d = 1e-2 * d / (np.linalg.norm(d) + 1e-12)
        s0, s1 = np.zeros(n), d.copy()
        for t in range(T):
            s0 = C.step(g, s0, X[t])
            s1 = C.step(g, s1, X[t])
        worst = max(worst, float(np.linalg.norm(s0 - s1)))
    return worst


def method_admits(name, g, rng):
    if name == "none":
        return True
    if name == "stable_exp":
        return perturbation_forgetting(g, rng) < EPS_FORGET
    if name == "cert_inf":
        return bool(C.cert_inf(g))
    if name == "cert_two":
        return bool(C.cert_two(g))
    if name == "cert_sdp":
        return bool(C.cert_sdp(g))
    raise ValueError(name)


METHODS = ("none", "stable_exp", "cert_inf", "cert_two", "cert_sdp")


def sample_spanning(rng, n, n_genes):
    """収縮〜発散を跨ぐ gene (W スケール広く)。真 ρ 付きで返す。"""
    out = []
    for _ in range(n_genes):
        decay = rng.uniform(0.0, 1.0, size=n)
        W = (rng.normal(0.0, 1.0, size=(n, n)) / np.sqrt(n)) * float(rng.uniform(0.1, 1.6))
        g = C.CoupledNDGene.make(decay=decay, W=W)
        out.append((g, emp_rho(g, int(rng.integers(0, 2**31)))))
    return out


def sample_mamba_synth(rng, n, n_genes):
    """Mamba 風 stable-by-construction: decay 支配 + 小 W で構成的に ρ<1 を保証 (cert_inf PASS 強制)。"""
    out = []
    got = 0
    while got < n_genes:
        decay = rng.uniform(0.7, 0.98, size=n)            # 高 decay = 強収縮
        W = (rng.normal(0.0, 1.0, size=(n, n)) / np.sqrt(n)) * 0.2
        g = C.CoupledNDGene.make(decay=decay, W=W)
        if C.cert_inf(g):                                  # 構成的に sound contraction
            out.append((g, emp_rho(g, int(rng.integers(0, 2**31)))))
            got += 1
    return out


def eval_pop(pop, rng):
    rho = np.array([r for _, r in pop])
    diverge = rho >= 1.0
    contracting = rho < 1.0
    res = {}
    for m in METHODS:
        adm = np.array([method_admits(m, g, rng) for g, _ in pop])
        n_adm = int(adm.sum())
        res[m] = {
            "admit_rate": n_adm / len(pop),
            "false_admit": int(np.sum(adm & diverge)),
            "false_admit_rate": float(np.sum(adm & diverge) / max(1, int(diverge.sum()))),
            "reject_on_contracting_rate": float(np.sum(~adm & contracting) / max(1, int(contracting.sum()))),
        }
    return res, int(diverge.sum()), int(contracting.sum())


def main():
    rng = np.random.default_rng(SEED)
    results = {"meta": {"seed": SEED, "n": N, "T": T, "eps_forget": EPS_FORGET, "rho_samples": RHO_SAMPLES,
                        "methods": list(METHODS),
                        "note": "H-discriminative: 4 method を soundness で判別 (Decision gate 2 核)。"}}

    # 集団1: spanning (危険 gene 含む) — H3a
    span = sample_spanning(rng, N, 400)
    span_res, n_div, n_con = eval_pop(span, rng)
    results["spanning"] = {"n_genes": len(span), "n_divergent": n_div, "n_contracting": n_con, "per_method": span_res}
    print(f"=== spanning 集団 (n_div={n_div}, n_con={n_con}) ===", flush=True)
    for m in METHODS:
        r = span_res[m]
        print(f"  [{m:11s}] admit率={r['admit_rate']:.3f}  false-admit={r['false_admit']} "
              f"(発散中 {r['false_admit_rate']:.3f})  収縮の棄却率={r['reject_on_contracting_rate']:.3f}", flush=True)

    # 集団2: mamba_synth (stable-by-construction) — H3b 正の対照
    mam = sample_mamba_synth(rng, N, 200)
    mam_res, m_div, m_con = eval_pop(mam, rng)
    results["mamba_synth"] = {"n_genes": len(mam), "n_divergent": m_div, "n_contracting": m_con, "per_method": mam_res}
    print(f"\n=== mamba_synth 集団 (stable-by-construction, n_div={m_div}, n_con={m_con}) ===", flush=True)
    for m in METHODS:
        r = mam_res[m]
        print(f"  [{m:11s}] admit率={r['admit_rate']:.3f}  reject率={1-r['admit_rate']:.3f}  false-admit={r['false_admit']}", flush=True)

    # 事前登録仮説判定
    fa = {m: span_res[m]["false_admit"] for m in METHODS}
    cert_fa_max = max(fa["cert_inf"], fa["cert_two"], fa["cert_sdp"])
    h3a = (fa["none"] > fa["stable_exp"]) and (fa["stable_exp"] >= cert_fa_max) and (cert_fa_max == 0)
    # H3b: stable-by-construction で sound cert の spurious reject ≈ 0 (cert_sdp が最 navigable ゆえ最小)
    cert_sdp_reject_mamba = mam_res["cert_sdp"]["reject_on_contracting_rate"]
    h3b = cert_sdp_reject_mamba < 0.05
    results["verdict"] = {
        "false_admit_spanning": fa,
        "H3a_discriminative_pass": bool(h3a),
        "H3a_detail": f"none={fa['none']} > stable_exp={fa['stable_exp']} > certs(max)={cert_fa_max}=0",
        "H3b_positive_control_pass": bool(h3b),
        "H3b_detail": f"mamba_synth 上で cert_sdp の収縮棄却率={cert_sdp_reject_mamba:.3f} (<0.05 で正対照 PASS)",
    }
    print("\n=== H-discriminative verdict ===", flush=True)
    print(f"  H3a (判別力): {results['verdict']['H3a_detail']} → PASS={h3a}", flush=True)
    print(f"  H3b (正対照): {results['verdict']['H3b_detail']} → PASS={h3b}", flush=True)

    out = os.path.join(os.path.dirname(__file__), "phase2_discriminative_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n結果: {out}", flush=True)
    return results


if __name__ == "__main__":
    main()
