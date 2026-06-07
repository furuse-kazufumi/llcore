# SPDX-License-Identifier: Apache-2.0
"""HD-1 接地 本走の confirmatory 分析 — HD1_GROUNDING_PREREG.md §3/§4 の実装。

**本スクリプトは結果取得前に commit する** (分析自由度の事前固定)。
入力 = result_hd1g_n{64,128,256}.json (Kaggle pull)。
判定規則は全て prereg §3 に従う:
- F 条項: NONE の窓契約死率 (seed 平均) < 5% → その n を除外。全 n 除外 = INVALID。
- 実害縮退条項 (A3): NONE の窓実害死率 < 5% → その n の実害 indicator を除外。
- H1: OBSERVE_P2 < NONE (paired Wilcoxon 両側 + median 差 < 0), active n × active indicator。
- H2: REVIVE 最終 CE < ENDO 最終 CE (paired Wilcoxon 両側 + median 差 < 0), active n。
- 代表 p = max over (active n × active indicator); Holm を {p_H1, p_H2} に適用。
- ENDO 反証条項: 全 active n で mean(OBSERVE_P2) − mean(ENDO) ≤ 1/80 → A6 検査
  (NONE 測定点 pool の Spearman(proxy_g, infnorm_sup) ≥ 0.8 → (a) / 未満 → (b))。
- H1 アーティファクト規律: 連続量 (窓内 max rho / ρ 超過積分) の方向一致を要求。

実行::  py -3.11 analyze_hd1_grounding.py [result_json ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, wilcoxon

_HERE = Path(__file__).resolve().parent
F_THRESHOLD = 0.05          # F 条項 / 実害 activity (prereg §3)
EQUIV_DELTA = 1.0 / 80      # ENDO 反証条項の同等マージン (prereg §3)
PROXY_SOUND_RS = 0.8        # A6 閾値
ALPHA = 0.05


def load(paths):
    recs = []
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        recs += [r for r in d["records"] if r.get("status") == "ok"]
    return recs


def wrate(r, key):
    """窓死亡率 = 死フラグ測定点数 / 測定点数 (prereg §2)。"""
    tr = r["trajectory"]
    return sum(1 for p in tr if p[key]) / len(tr)


def paired(recs, n, arm_a, arm_b, fn):
    """seed で対応づけた (a, b) 値のリスト。"""
    A = {r["seed"]: r for r in recs if r["arm"] == arm_a and r["n"] == n}
    B = {r["seed"]: r for r in recs if r["arm"] == arm_b and r["n"] == n}
    seeds = sorted(set(A) & set(B))
    return np.array([fn(A[s]) for s in seeds]), np.array([fn(B[s]) for s in seeds])


def wtest(a, b):
    """paired Wilcoxon (両側) + median 差。全ゼロ差は p=None (退化)。"""
    d = a - b
    if np.all(d == 0):
        return None, 0.0
    return float(wilcoxon(a, b).pvalue), float(np.median(d))


def holm(pvals):
    """Holm 調整 (dict name->p, None は除外)。"""
    items = sorted([(k, v) for k, v in pvals.items() if v is not None], key=lambda kv: kv[1])
    adj, mx = {}, 0.0
    m = len(items)
    for i, (k, v) in enumerate(items):
        mx = max(mx, v * (m - i))
        adj[k] = min(mx, 1.0)
    return adj


def main():
    paths = sys.argv[1:] or sorted(_HERE.glob("result_hd1g_n*.json"))
    if not paths:
        print("no result_hd1g_n*.json found"); return 2
    recs = load(paths)
    ns = sorted({r["n"] for r in recs})
    print(f"=== HD-1 grounding confirmatory analysis (prereg=HD1_GROUNDING_PREREG.md) ===")
    print(f"inputs: {[str(p) for p in paths]} | records={len(recs)} | n levels={ns}\n")

    # ---- gates: F 条項 (n) + 実害 activity (indicator) --------------------------
    active_n, harm_active = [], {}
    for n in ns:
        nones = [r for r in recs if r["arm"] == "NONE" and r["n"] == n]
        fc = float(np.mean([wrate(r, "contract_death") for r in nones]))
        fh = float(np.mean([wrate(r, "harm_death") for r in nones]))
        ok = fc >= F_THRESHOLD
        harm_active[n] = fh >= F_THRESHOLD
        if ok:
            active_n.append(n)
        print(f"[gate] n={n:3d}: NONE contract={fc:6.2%} ({'ACTIVE' if ok else 'EXCLUDED'}) | "
              f"harm={fh:6.2%} ({'active' if harm_active[n] else 'harm indicator excluded'})")
    if not active_n:
        print("\n** ALL n EXCLUDED by F-clause -> experiment INVALID (prereg §3). "
              "Descriptive results only; no confirmatory claims. **")

    # ---- H1: OBSERVE_P2 vs NONE (active n × active indicator) -------------------
    h1_tests, h1_dirs = {}, {}
    for n in active_n:
        inds = ["contract_death"] + (["harm_death"] if harm_active[n] else [])
        for ind in inds:
            a, b = paired(recs, n, "OBSERVE_P2", "NONE", lambda r: wrate(r, ind))
            p, md = wtest(a, b)
            h1_tests[f"n{n}:{ind}"] = p
            h1_dirs[f"n{n}:{ind}"] = md
            print(f"[H1] n={n:3d} {ind:14s}: OBS_P2={a.mean():.4f} NONE={b.mean():.4f} "
                  f"median(diff)={md:+.4f} p={p if p is not None else 'NaN(全ゼロ)'}")
    # ---- H2: REVIVE vs ENDO (CE) -------------------------------------------------
    h2_tests, h2_dirs = {}, {}
    for n in active_n:
        a, b = paired(recs, n, "REVIVE", "ENDO", lambda r: r["final_ce"])
        p, md = wtest(a, b)
        h2_tests[f"n{n}"] = p
        h2_dirs[f"n{n}"] = md
        print(f"[H2] n={n:3d} CE: REVIVE={a.mean():.4f} ENDO={b.mean():.4f} "
              f"median(diff)={md:+.4f} p={p if p is not None else 'NaN'}")

    # ---- family Holm: 代表 p = max over tests (連言) -----------------------------
    rep = {}
    for name, tests, dirs in (("H1", h1_tests, h1_dirs), ("H2", h2_tests, h2_dirs)):
        ps = [v for v in tests.values() if v is not None]
        degenerate = any(v is None for v in tests.values())
        rep[name] = max(ps) if ps and not degenerate else (1.0 if degenerate else None)
        if degenerate:
            print(f"[{name}] 全ゼロ差の退化検定を含む -> 代表 p=1.0 (保守; 同等は反証条項で扱う)")
    adj = holm(rep)
    print()
    verdicts = {}
    for name, tests, dirs in (("H1", h1_tests, h1_dirs), ("H2", h2_tests, h2_dirs)):
        dir_ok = all(v < 0 for v in dirs.values()) and len(dirs) > 0
        p_ok = (name in adj) and all(
            v is not None and v < ALPHA for v in tests.values()) and adj[name] < ALPHA
        verdicts[name] = bool(active_n) and dir_ok and p_ok
        print(f"[{name}] direction-consistent={dir_ok} all-p<0.05={p_ok} "
              f"Holm p={adj.get(name)} -> {'PASS' if verdicts[name] else 'not supported'}")

    # ---- ENDO 反証条項 (decidable 同等規則) --------------------------------------
    if active_n:
        margins = []
        for n in active_n:
            a, b = paired(recs, n, "OBSERVE_P2", "ENDO", lambda r: wrate(r, "contract_death"))
            margins.append(float(a.mean() - b.mean()))
        caught_up = all(m <= EQUIV_DELTA for m in margins)
        print(f"\n[ENDO 反証条項] mean(OBS_P2)-mean(ENDO) per active n = "
              f"{[f'{m:+.4f}' for m in margins]} (δ={EQUIV_DELTA:.4f}) -> "
              f"{'発動' if caught_up else '非発動'}")
        if caught_up:
            for n in active_n:
                pts = [(p["proxy_g"], p["infnorm_sup"]) for r in recs
                       if r["arm"] == "NONE" and r["n"] == n for p in r["trajectory"]]
                rs = spearmanr([x for x, _ in pts], [y for _, y in pts]).statistic
                branch = "(a) proxy 設計の問題 -> H1 無効" if abs(rs) >= PROXY_SOUND_RS \
                    else "(b) sound>>empirical は EA 固有へ格下げ"
                print(f"  [A6] n={n}: Spearman(proxy_g, infnorm_sup)={rs:+.3f} -> {branch}")

    # ---- H1 アーティファクト規律 (連続量の方向一致; binding 解釈規則) -------------
    if verdicts.get("H1"):
        print("\n[アーティファクト規律] H1 PASS のため連続量 2 指標を検査:")
        cont_ok = True
        for n in active_n:
            for label, fn in (("max_rho", lambda r: max(p["rho_hat"] for p in r["trajectory"])),
                              ("rho_excess", lambda r: sum(max(p["rho_hat"] - 1, 0)
                                                           for p in r["trajectory"]))):
                a, b = paired(recs, n, "OBSERVE_P2", "NONE", fn)
                ok = a.mean() < b.mean()
                cont_ok &= ok
                print(f"  n={n:3d} {label:10s}: OBS_P2={a.mean():.4f} NONE={b.mean():.4f} "
                      f"-> {'一致' if ok else '不一致'}")
        print("  => 解釈:", "部分的死回避を提供 (規律通過)" if cont_ok
              else "二値計数アーティファクトの疑い — 主張を「窓計数上の低下」へ弱める (prereg §3)")

    # ---- Exploratory (記述; 各 1 回) ---------------------------------------------
    print("\n=== Exploratory (補正外, 記述) ===")
    for n in ns:                                    # E1
        nones = [r for r in recs if r["arm"] == "NONE" and r["n"] == n]
        if len(nones) >= 4:
            sr = spearmanr([wrate(r, "contract_death") for r in nones],
                           [r["final_ce"] for r in nones])
            print(f"[E1] n={n:3d} NONE 死亡率×CE Spearman={sr.statistic:+.3f} (p={sr.pvalue:.3f})")
    for n in ns:                                    # E2
        pts = [(1.0 if p["contract_death"] else 0.0, p["sep_rate"]) for r in recs
               if r["n"] == n for p in r["trajectory"]]
        flags = [x for x, _ in pts]
        if sum(flags) == 0:
            print(f"[E2] n={n:3d}: 契約死フラグ 0 -> 縮退 (判定不能)")
        else:
            sr = spearmanr(flags, [y for _, y in pts])
            ok = abs(sr.statistic) >= 0.5
            print(f"[E2] n={n:3d} flag×sep_rate Spearman={sr.statistic:+.3f} -> "
                  f"{'契約死を実害 proxy 採用' if ok else '死回避軸は実害 probe で再定義 (分岐発動)'}")
    for n in ns:                                    # E3
        nones = [r for r in recs if r["arm"] == "NONE" and r["n"] == n]
        endos = [r for r in recs if r["arm"] == "ENDO" and r["n"] == n]
        if nones and endos:
            mx = float(np.mean([max(p["rho_hat"] for p in r["trajectory"]) for r in nones]))
            cost = float(np.mean([r["final_ce"] for r in endos])
                         - np.mean([r["final_ce"] for r in nones]))
            hp = float(np.mean([r["final_rho_hp"] for r in nones]))
            print(f"[E3] n={n:3d}: NONE max rho={mx:.3f} (ρ>=1 到達 {'yes' if mx >= 1 else 'no'}) "
                  f"final_rho_hp={hp:.3f} | ENDO-NONE CE cost={cost:+.4f} "
                  f"(HD-1 帯 0.03-0.12; init protocol 差の注意 = prereg §4 E3)")
    for n in active_n:                              # E5
        for tail in (0.3, 0.5, 0.7):
            def wr_tail(r, key="contract_death", tail=tail):
                tr = r["trajectory"]; k = max(int(len(tr) * tail), 1)
                return sum(1 for p in tr[-k:] if p[key]) / k
            a, b = paired(recs, n, "OBSERVE_P2", "NONE", wr_tail)
            print(f"[E5] n={n:3d} tail={tail:.0%}: OBS_P2={a.mean():.4f} NONE={b.mean():.4f}")
    for n in ns:                                    # E8
        a, b = paired(recs, n, "OBSERVE_P2", "OBSERVE_P1", lambda r: wrate(r, "contract_death"))
        if len(a):
            print(f"[E8] n={n:3d} P2={a.mean():.4f} P1={b.mean():.4f} (2-pass 共有の寄与)")
    a, b = paired(recs, 128, "ENDO_K8", "ENDO", lambda r: r["final_ce"])  # E9
    if len(a):
        print(f"[E9] n=128 CE: ENDO_K8={a.mean():.4f} ENDO={b.mean():.4f} (k 感度)")
    # E10: avoid 発動の offline 再構成 — 縮小発動の「直後測定点」で死が消えた割合
    print("[E10] avoid->死点消失タイミング (OBSERVE_P2, kernel 規則の offline 再構成):")
    for n in ns:
        p1 = [r for r in recs if r["arm"] == "OBSERVE_P1" and r["n"] == n]
        pooled = [v for r in p1 for v in r["death_proxy_log"]]
        runs = [r for r in recs if r["arm"] == "OBSERVE_P2" and r["n"] == n]
        fired = died_next = 0
        for r in runs:
            log = list(pooled)
            thr = float(np.percentile(log, 10.0)) if log else None
            tr = r["trajectory"]
            for i, p in enumerate(tr):
                if p["contract_death"]:
                    log.append(p["proxy_ma"]); thr = float(np.percentile(log, 10.0))
                if thr is not None and p["proxy_ma"] >= thr:
                    fired += 1
                    if i + 1 < len(tr) and not tr[i + 1]["contract_death"]:
                        died_next += 1
        if fired:
            print(f"  n={n:3d}: avoid 発動 {fired} 回中、直後測定点が非死 {died_next} "
                  f"({died_next/fired:.0%})")
    # E11: excursion 窓分割 (NONE seed 平均 rho>=1 の測定点)
    print("[E11] excursion 窓分割 (NONE 定義):")
    for n in active_n:
        nones = [r for r in recs if r["arm"] == "NONE" and r["n"] == n]
        steps = [p["step"] for p in nones[0]["trajectory"]]
        mean_rho = np.mean([[p["rho_hat"] for p in r["trajectory"]] for r in nones], axis=0)
        exc = {steps[i] for i in range(len(steps)) if mean_rho[i] >= 1.0}
        if not exc:
            print(f"  n={n:3d}: excursion 窓なし (NONE 平均 rho<1 全点)"); continue
        for label, sel in (("excursion", lambda s: s in exc), ("補集合", lambda s: s not in exc)):
            def wr_sel(r, sel=sel):
                pts = [p for p in r["trajectory"] if sel(p["step"])]
                return (sum(1 for p in pts if p["contract_death"]) / len(pts)) if pts else 0.0
            a, b = paired(recs, n, "OBSERVE_P2", "NONE", wr_sel)
            print(f"  n={n:3d} {label:9s}: OBS_P2={a.mean():.4f} NONE={b.mean():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
