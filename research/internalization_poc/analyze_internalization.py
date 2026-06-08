# SPDX-License-Identifier: Apache-2.0
"""検証シグナル勾配内蔵 本走の confirmatory 分析 — HD1_INTERNALIZATION_PREREG.md §3/§4 の実装.

**本スクリプトは結果取得前に commit する** (分析自由度の事前固定; HD-1 grounding と同規律)。
入力 = result_internalization_s2_n{64,128,256}.json (Kaggle pull) [+ stage1 ファイルがあれば first-check 表示]。

判定規則は全て prereg §3/§4 に従う:
- F 条項: NONE の窓契約死率 (seed 平均) < 5% → その n を除外。全 n 除外 = INVALID。
- H1: ENDO_GRAD 窓契約死率 < NONE (paired Wilcoxon 両側 + median 差 < 0), active n。
  + アーティファクト規律: 連続量 (窓内 max rho / ρ 超過積分) も ENDO_GRAD < NONE 方向一致。
  + A4 留保: 死イベント時 infnorm_sup↔empirical_rho Spearman ≥ 0.8 → 「surrogate sound 近似性の産物」留保。
- H2: ENDO_GRAD λ-sweep の (死回避率, CE) Pareto 曲線 vs ENDO_HARNESS 点。
  HARNESS の死回避率での内挿 CE_GRAD < CE_HARNESS を seed bootstrap で P>0.975 (active n)。
- 代表 p = max over active n; Holm を {p_H1, p_H2} に適用 (min<0.025, 残り<0.05)。

実行::  py -3.11 analyze_internalization.py [result_json ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

_HERE = Path(__file__).resolve().parent
F_THRESHOLD = 0.05            # F 条項 (prereg §3)
PROXY_SOUND_RS = 0.8         # A4 留保閾値
H2_BOOT = 2000               # H2 bootstrap resample 数
H2_PASS_P = 0.975            # H2 成立 P 閾値 (Holm 前)
LAM_SWEEP = [0.03, 0.1, 0.3, 1.0]


def _load(paths):
    recs = {}
    for p in paths:
        d = json.load(open(p, encoding="utf-8"))
        n = d["meta"]["n"]
        recs.setdefault(n, []).extend([r for r in d["records"] if r.get("status", "ok") == "ok"])
    return recs


def _arm_seedmap(records, arm):
    """seed -> record (最初の1件) の dict。"""
    return {r["seed"]: r for r in records if r["arm"] == arm}


def _paired(records, arm_a, arm_b, key):
    """共通 seed で arm_a, arm_b の key を対にして (a_vals, b_vals) を返す。"""
    ma, mb = _arm_seedmap(records, arm_a), _arm_seedmap(records, arm_b)
    seeds = sorted(set(ma) & set(mb))
    return np.array([ma[s][key] for s in seeds]), np.array([mb[s][key] for s in seeds]), seeds


def holm(pvals):
    """Holm-Bonferroni 調整 p (入力 dict name->p, 出力 dict name->adj_p)。"""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items); adj = {}
    prev = 0.0
    for i, (name, p) in enumerate(items):
        a = min(1.0, (m - i) * p)
        a = max(a, prev); prev = a
        adj[name] = a
    return adj


def f_clause(records_by_n):
    """各 n の NONE 窓契約死率 → active 判定。"""
    out = {}
    for n, recs in sorted(records_by_n.items()):
        none = [r["death_rate"] for r in recs if r["arm"] == "NONE"]
        rate = float(np.mean(none)) if none else 0.0
        out[n] = {"none_death_rate": rate, "active": bool(rate >= F_THRESHOLD)}
    return out


def test_H1(records_by_n, active_ns):
    """ENDO_GRAD 窓契約死率 < NONE (paired Wilcoxon) + 連続量規律 + A4。"""
    per_n = {}
    pmax = 0.0
    for n in active_ns:
        recs = records_by_n[n]
        eg, none, seeds = _paired(recs, "ENDO_GRAD", "NONE", "death_rate")
        if len(seeds) < 3:
            per_n[n] = {"error": "insufficient paired seeds"}; pmax = 1.0; continue
        diff = eg - none
        med = float(np.median(diff))
        # 全ゼロ退化 (両 arm 死 0) を honest に扱う
        if np.allclose(diff, 0.0):
            p = 1.0; note = "all-zero (no death signal) -> not detected"
        else:
            try:
                p = float(wilcoxon(eg, none, alternative="two-sided", zero_method="wilcox").pvalue)
            except ValueError:
                p = 1.0
            note = ""
        # アーティファクト規律 (連続量方向一致)
        eg_wmax, none_wmax, _ = _paired(recs, "ENDO_GRAD", "NONE", "window_max_rho")
        eg_rex, none_rex, _ = _paired(recs, "ENDO_GRAD", "NONE", "rho_excess_integral")
        art_wmax = float(np.median(eg_wmax - none_wmax))
        art_rex = float(np.median(eg_rex - none_rex))
        artifact_ok = (art_wmax < 0 and art_rex < 0)
        # A4: NONE 測定点 pool の infnorm_sup↔rho_hat Spearman
        infs, rhos = [], []
        for r in recs:
            if r["arm"] == "NONE":
                for pt in r["trajectory"]:
                    infs.append(pt["infnorm_sup"]); rhos.append(pt["rho_hat"])
        a4_rs = float(spearmanr(infs, rhos).correlation) if len(infs) > 3 else float("nan")
        per_n[n] = {"median_death_diff": med, "p": p, "direction_ok": bool(med < 0),
                    "artifact_wmax_diff": art_wmax, "artifact_rho_excess_diff": art_rex,
                    "artifact_continuous_agree": bool(artifact_ok),
                    "A4_infnorm_rho_spearman": a4_rs,
                    "A4_caveat": bool(a4_rs >= PROXY_SOUND_RS), "note": note}
        pmax = max(pmax, p)
    direction_all = all(per_n[n].get("direction_ok", False) for n in active_ns)
    return {"per_n": per_n, "rep_p": pmax, "all_direction_ok": bool(direction_all)}


def _pareto_interp_ce(lam_points, target_avoid):
    """λ-sweep 点 [(avoid_rate, ce)] を avoid_rate でソートし target_avoid での CE を線形内挿。
    範囲外なら None。"""
    pts = sorted(lam_points, key=lambda t: t[0])
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    if target_avoid < xs[0] or target_avoid > xs[-1]:
        return None
    return float(np.interp(target_avoid, xs, ys))


def test_H2(records_by_n, active_ns):
    """ENDO_GRAD λ-sweep の Pareto (死回避率 vs CE) vs HARNESS 点; bootstrap。"""
    per_n = {}
    pmax = 0.0
    arms_sweep = [f"ENDO_GRAD_L{str(l).replace('.', '')}" for l in LAM_SWEEP]
    for n in active_ns:
        recs = records_by_n[n]
        har = _arm_seedmap(recs, "ENDO_HARNESS")
        sweep_maps = {a: _arm_seedmap(recs, a) for a in arms_sweep}
        # 共通 seed
        common = set(har)
        for a in arms_sweep:
            common &= set(sweep_maps[a])
        common = sorted(common)
        if len(common) < 5 or not all(sweep_maps[a] for a in arms_sweep):
            per_n[n] = {"error": "insufficient H2 arms/seeds"}; pmax = 1.0; continue

        def point(seedset):
            har_av = float(np.mean([1.0 - har[s]["death_rate"] for s in seedset]))
            har_ce = float(np.mean([har[s]["final_ce"] for s in seedset]))
            lam_pts = []
            for a in arms_sweep:
                av = float(np.mean([1.0 - sweep_maps[a][s]["death_rate"] for s in seedset]))
                ce = float(np.mean([sweep_maps[a][s]["final_ce"] for s in seedset]))
                lam_pts.append((av, ce))
            return har_av, har_ce, lam_pts

        har_av0, har_ce0, lam_pts0 = point(common)
        interp0 = _pareto_interp_ce(lam_pts0, har_av0)
        rng = np.random.default_rng(12345 + n)
        wins = oob = 0
        for _ in range(H2_BOOT):
            bs = list(rng.choice(common, size=len(common), replace=True))
            har_av, har_ce, lam_pts = point(bs)
            ce_g = _pareto_interp_ce(lam_pts, har_av)
            if ce_g is None:
                oob += 1; continue
            if ce_g < har_ce:
                wins += 1
        valid = H2_BOOT - oob
        p_win = wins / valid if valid > 0 else 0.0
        p = 1.0 - p_win                                  # 「GRAD が劣らない」片側
        per_n[n] = {"harness_avoid": har_av0, "harness_ce": har_ce0,
                    "interp_grad_ce_at_harness_avoid": interp0,
                    "P(grad_ce<harness_ce)": p_win, "oob_frac": oob / H2_BOOT, "p": p,
                    "pass_pre_holm": bool(p_win > H2_PASS_P)}
        pmax = max(pmax, p)
    return {"per_n": per_n, "rep_p": pmax}


def stage1_summary(paths_s1):
    out = {}
    for p in paths_s1:
        d = json.load(open(p, encoding="utf-8"))
        if "stage1_verdict" in d:
            out[d["meta"]["n"]] = d["stage1_verdict"]
    return out


def main(argv):
    args = [a for a in argv if a.endswith(".json")]
    s2 = [a for a in args if "_s2_" in a] or [str(p) for p in _HERE.glob("result_internalization_s2_n*.json")]
    s1 = [a for a in args if "_s1_" in a] or [str(p) for p in _HERE.glob("result_internalization_s1_n*.json")]

    if s1:
        print("=== stage-1 first-check ===")
        for n, v in sorted(stage1_summary(s1).items()):
            print(f"  n={n}: S1={v['S1_NONE_death_active']['pass']} "
                  f"S2={v['S2_tautology_nonapplicable']['pass']} "
                  f"S3={v['S3_gate_silence']['pass']} go_stage2={v['go_stage2']}")

    if not s2:
        print("\n(no stage-2 result files yet — run internalization_kernel.py RUN_STAGE=2 first)")
        return 0

    recs = _load(s2)
    print(f"\n=== stage-2 confirmatory (n={sorted(recs)}) ===")
    fc = f_clause(recs)
    for n, v in sorted(fc.items()):
        print(f"  F-clause n={n}: NONE death={v['none_death_rate']:.3f} "
              f"({'ACTIVE' if v['active'] else 'INACTIVE -> excluded'})")
    active = [n for n, v in fc.items() if v["active"]]
    if not active:
        print("  *** ALL n EXCLUDED -> INVALID (死 regime が立たない; prereg §5) ***")
        return 0

    h1 = test_H1(recs, active)
    h2 = test_H2(recs, active)
    adj = holm({"H1": h1["rep_p"], "H2": h2["rep_p"]})

    print(f"\n--- H1 (ENDO_GRAD 死 < NONE; active n={active}) ---")
    for n in active:
        d = h1["per_n"][n]
        if "error" in d:
            print(f"  n={n}: {d['error']}"); continue
        print(f"  n={n}: Δdeath_median={d['median_death_diff']:+.3f} p={d['p']:.4f} "
              f"dir_ok={d['direction_ok']} artifact_agree={d['artifact_continuous_agree']} "
              f"A4_rs={d['A4_infnorm_rho_spearman']:.2f}{' [CAVEAT]' if d['A4_caveat'] else ''}")
    print(f"  H1 rep_p={h1['rep_p']:.4f} Holm_adj={adj['H1']:.4f} "
          f"dir_all={h1['all_direction_ok']} -> "
          f"{'PASS' if (adj['H1'] < 0.05 and h1['all_direction_ok']) else 'not supported'}")

    print(f"\n--- H2 (Pareto: ENDO_GRAD λ-sweep vs HARNESS; active n={active}) ---")
    for n in active:
        d = h2["per_n"][n]
        if "error" in d:
            print(f"  n={n}: {d['error']}"); continue
        print(f"  n={n}: harness(avoid={d['harness_avoid']:.3f},ce={d['harness_ce']:.4f}) "
              f"interp_grad_ce={d['interp_grad_ce_at_harness_avoid']} "
              f"P(grad<harness)={d['P(grad_ce<harness_ce)']:.3f} oob={d['oob_frac']:.2f}")
    print(f"  H2 rep_p={h2['rep_p']:.4f} Holm_adj={adj['H2']:.4f} -> "
          f"{'PASS' if adj['H2'] < 0.05 else 'not supported'}")

    verdict = {"f_clause": fc, "active_n": active, "H1": h1, "H2": h2, "holm": adj,
               "H1_PASS": bool(adj["H1"] < 0.05 and h1["all_direction_ok"]),
               "H2_PASS": bool(adj["H2"] < 0.05)}
    json.dump(verdict, open(_HERE / "internalization_analysis_output.json", "w"), indent=1)
    print("\n[saved] internalization_analysis_output.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
