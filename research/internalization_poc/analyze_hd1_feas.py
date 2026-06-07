# SPDX-License-Identifier: Apache-2.0
"""HD-1 接地 feasibility の事前登録最終化用分析 (設計 v3 §5 step 3)。

確認項目:
1. plateau 窓: NONE の ρ(step) 軌跡が plateau に達する step を同定 → measure 窓を固定
2. F 条項: NONE の measure 窓契約死 step 比率 (全窓と plateau 窓の両方)
3. 実害 probe: sep_rate の分布 (全 0 発火の機構 — どれだけ 0 から遠いか)
4. proxy-sound 独立性の素材: 死イベント時 proxy_g の分布
5. E4: REVIVE − REVIVE_ABLATE の CE 差 (記述)
6. OBSERVE β 感度 (記述報告; 本走は 0.5 固定)

confirmatory 検定は行わない (feasibility = 記述のみ)。
実行::  py -3.11 research/internalization_poc/analyze_hd1_feas.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "verified_memory_poc"))
from run_3arm_ab import _ensure_utf8_stdout  # noqa: E402


def main():
    _ensure_utf8_stdout()
    recs = json.loads((_HERE / "results_hd1_grounding_feas.json").read_text(encoding="utf-8"))["records"]
    ns = sorted({r["n"] for r in recs})

    # ---- 1. plateau 窓: NONE の seed 平均 ρ(step) ----------------------------
    print("=== 1. NONE rho_hat(step) (seed mean ± sd) — plateau 同定 ===")
    for n in ns:
        rs = [r for r in recs if r["arm"] == "NONE" and r["n"] == n]
        steps = [p["step"] for p in rs[0]["trajectory"]]
        mat = np.array([[p["rho_hat"] for p in r["trajectory"]] for r in rs])
        mean, sd = mat.mean(axis=0), mat.std(axis=0)
        print(f"-- n={n} (seeds={len(rs)})")
        for i, s in enumerate(steps):
            bar = "#" * int(mean[i] * 20)
            print(f"   step={s:3d} rho={mean[i]:.3f}±{sd[i]:.3f} {bar}")
        # plateau 判定: 隣接測定点の平均 ρ 差が |Δ|<0.02 になって以降を plateau とみなす
        diffs = np.abs(np.diff(mean))
        idx = next((i + 1 for i in range(len(diffs)) if np.all(diffs[i:] < 0.02)), len(steps) - 1)
        frac = steps[idx] / steps[-1]
        print(f"   -> plateau onset: step={steps[idx]} ({frac:.0%} of budget, |Δrho|<0.02 以降)")

    # ---- 2. F 条項: 窓別の NONE 契約死比率 -----------------------------------
    print("\n=== 2. F 条項 (NONE 契約死 step 比率; 窓 = 後半 30/50/70%) ===")
    for n in ns:
        rs = [r for r in recs if r["arm"] == "NONE" and r["n"] == n]
        n_meas = len(rs[0]["trajectory"])
        for tail in (0.3, 0.5, 0.7):
            k = max(int(n_meas * tail), 1)
            frac = float(np.mean([
                np.mean([p["contract_death"] for p in r["trajectory"][-k:]]) for r in rs]))
            print(f"   n={n:3d} tail={tail:.0%}: {frac:6.2%} ({'ACTIVE' if frac >= 0.05 else 'inactive'})")

    # ---- 3. 実害 probe: sep_rate 分布 (0 発火の距離感) ------------------------
    print("\n=== 3. sep_rate 分布 (arm別 max/mean; harm_death=sep_rate>=0) ===")
    for n in ns:
        for arm in ("NONE", "ENDO", "REVIVE", "OBSERVE"):
            rs = [r for r in recs if r["arm"] == arm and r["n"] == n]
            if not rs:
                continue
            vals = np.array([p["sep_rate"] for r in rs for p in r["trajectory"]])
            print(f"   n={n:3d} {arm:8s} mean={vals.mean():+.3f} p95={np.percentile(vals,95):+.3f} "
                  f"max={vals.max():+.3f} (>=0: {np.mean(vals >= 0):.1%})")

    # ---- 4. 死イベント時 proxy_g (OBSERVE 閾値素材) ---------------------------
    print("\n=== 4. proxy_g: 死イベント時 vs 全測定点 (n別) ===")
    for n in ns:
        rs = [r for r in recs if r["n"] == n]
        all_g = np.array([p["proxy_g"] for r in rs for p in r["trajectory"]])
        death_g = np.array([p["proxy_g"] for r in rs for p in r["trajectory"] if p["contract_death"]])
        print(f"   n={n:3d} all: mean={all_g.mean():+.4f}  death({len(death_g)}): "
              + (f"mean={death_g.mean():+.4f} p10={np.percentile(death_g,10):+.4f}"
                 if len(death_g) else "none"))
        # 死イベントの proxy と rho の相関 (proxy-sound 独立性の予備材料)
        pts = [(p["proxy_g"], p["rho_hat"]) for r in rs for p in r["trajectory"]]
        g, rho = np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
        from scipy.stats import spearmanr
        sr = spearmanr(g, rho)
        print(f"          Spearman(proxy_g, rho_hat) = {sr.statistic:+.3f} (p={sr.pvalue:.2e})")

    # ---- 5. E4: REVIVE − ABLATE CE (記述) ------------------------------------
    print("\n=== 5. E4: REVIVE vs REVIVE_ABLATE (CE, 記述のみ) ===")
    for n in ns:
        rv = [r["final_ce"] for r in recs if r["arm"] == "REVIVE" and r["n"] == n]
        ab = [r["final_ce"] for r in recs if r["arm"] == "REVIVE_ABLATE" and r["n"] == n]
        print(f"   n={n:3d} REVIVE={np.mean(rv):.4f} ABLATE={np.mean(ab):.4f} Δ={np.mean(rv)-np.mean(ab):+.4f}")

    # ---- 6. OBSERVE β (記述; 本走 0.5 固定は設計 §3-3 で事前確定済) -----------
    print("\n=== 6. OBSERVE β 感度 (記述; 本走 β=0.5 は設計時固定) ===")
    for n in ns:
        for b in (0.25, 0.5, 0.75):
            rs = [r for r in recs if r["arm"] == "OBSERVE" and r["n"] == n and r["beta"] == b]
            if rs:
                print(f"   n={n:3d} β={b}: ce={np.mean([r['final_ce'] for r in rs]):.4f} "
                      f"d_con={np.mean([r['deaths_contract'] for r in rs]):5.1f} "
                      f"av={np.mean([r['avoids'] for r in rs]):4.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
