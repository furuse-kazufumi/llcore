# SPDX-License-Identifier: Apache-2.0
"""R-endo viability ④ — regime-aware adaptive meta-controller (sound↔empirical hedging 学習)。

## 動機 (factorial ③ の帰結)

③ で sound 予見 (E) は死回避軸を単独支配した — ただしそれは検証器が環境 κ を**完全観測できる**
(oracle sensing) 前提。certificate は前提が正しい時のみ sound。本 runner は **前提が壊れる regime**
を構成する: 検証器の κ 推定にバイアス (sensed = sense_factor·κ_true)。

- GOOD regime (sense=1.0): 検証器前提が成立 → E は死 0 (③ で確認済)。
- BAD regime (sense=0.6): κ を 40% 過小推定 → L(g, κ_sensed)<1 だが κ_true·a≥1 の **lethal band を
  admit** = certificate が premise 違反で実質 unsound 化 (admit したのに死ぬ = predictive miss)。

問い: **「証明が信頼できる時は証明、前提が壊れたら経験へ退避」を、観測可能な信号だけで学習できるか**
(regime-aware hedging)。llcore 的含意 = fail-closed certificate の次の階層: certificate の前提監視。

## arms (4)

- **NONE**   : gate なし (anchor; 死境界 active 確認用)。
- **E_only** : 常に sound gate (sensed κ)。GOOD で完璧・BAD で miss を被る。
- **O_only** : 常に empirical 観察 gate (death_memory kNN)。κ 推定に依存しない (true 観測に接地)。
- **META**   : 子ごとに確率 t (trust) で E gate、(1−t) で O gate。**t の更新は観測可能信号のみ**:
  各世代、E-screen を通過して評価された子のうち死んだ数 m (= predictive miss) を数え、
  m>0 → t ← max(0.05, t·0.5^m) / m==0 かつ E-screen 子が 1 体以上評価された世代 → t ← min(1, t+0.1)
  (情報がない世代は据え置き)。床 0.05 = 常時微小プローブ (regime 回復の検知コスト)。

R (蘇生) は本実験から除外: `_repair` も検証器モデル (`_admits`) を使うため BAD regime では修復自体が
premise 汚染される — E↔O の対比を crisp に保つため confound を避ける (future work に明記)。

## スケジュール (1 run = 連続 58 世代, 集団/rng/death_memory 継続)

phase1 (G1=10, κ_low, sense 1.0) → 助走 (8, κ_high, GOOD) →
**測定 40 世代 = [GOOD 10, BAD 10, GOOD 10, BAD 10]** (block 交代で regime 追従を試す)。

## 事前登録 (PRE-REGISTRATION — 結果取得前に commit)

n=20 seeds (3000-3019), substrate = linear + highgain, paired sign-flip (10^5, seed=13), α=0.05。
指標 = 測定 40 世代の deaths (block 別) + capability (測定全世代の pop mean fitness の平均)。

- **H_bad**: BAD blocks の deaths: META < E_only (META は miss を検知し empirical へ退避する)。
- **H_good**: GOOD blocks の deaths: META < O_only (META は信頼回復し sound gate を活用する)。
- **H_overall (本丸)**: 測定全体 deaths: META < E_only かつ META < O_only
  (= 固定戦略 2 つを両方上回る hedging の価値)。
- **H_capability**: META capability ≥ max(E_only, O_only) − 0.02。
- **機構透明性 (必須)**: META の mean trust: GOOD blocks > BAD blocks (実際に regime を追従した証拠)。
- 反証: H_overall 不成立 → 「単純 trust 則では hedging 不成立」を honest 報告 (探索的に trust 軌跡を
  精査; EXP3 等は後続)。NONE deaths ≤ 1 (F2) → 死境界 inactive で INVALID。

実行::  py -3.11 research/internalization_poc/run_viability_meta.py
出力::  research/internalization_poc/results_viability_meta.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parents[1] / "src"), str(_HERE.parents[0] / "verified_memory_poc"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_viability_ab as M  # noqa: E402
from llcore.evolution.minimal_ga import (  # noqa: E402
    Individual, Population, crossover_uniform, initialize_random_population,
    tournament_select, uniform_mutate,
)
from run_c_decision import signflip_pvalue  # noqa: E402
from viability_substrates import HighGainSubstrate, LinearSubstrate  # noqa: E402

SUBSTRATES = [LinearSubstrate(), HighGainSubstrate()]
ARMS4 = ["NONE", "E_only", "O_only", "META"]
SENSE_GOOD = 1.0
SENSE_BAD = 0.6              # κ を 40% 過小推定 (lethal band admit = premise 違反)
MEASURE_BLOCKS = [("GOOD", 10), ("BAD", 10), ("GOOD", 10), ("BAD", 10)]
TRUST0, TRUST_FLOOR, TRUST_DECAY, TRUST_RECOVER = 1.0, 0.05, 0.5, 0.1
CAP_TOL = 0.02


def _schedule():
    """[(label, kappa_true, sense_factor)] — phase1 → 助走 → 測定 blocks。"""
    sched = [("P1", M.KAPPA_LOW, SENSE_GOOD)] * M.G1
    sched += [("WU", M.KAPPA_HIGH, SENSE_GOOD)] * M.G2_WARMUP
    for label, n in MEASURE_BLOCKS:
        sense = SENSE_GOOD if label == "GOOD" else SENSE_BAD
        sched += [(label, M.KAPPA_HIGH, sense)] * n
    return sched


def _run_arm_sched(sub, arm, seed, V):
    """連続スケジュール GA (集団/rng/death_memory/trust 継続)。block 別 deaths + trust を返す。"""
    rng = np.random.default_rng(seed)
    dm = []                      # 観察された死 (O / META の empirical 境界)
    trust = TRUST0               # META のみ使用
    sched = _schedule()
    stats = {"deaths": {}, "trust_sum": {}, "trust_n": {}, "cap_sum": 0.0, "cap_n": 0}

    def _eval_count(g, label, kappa_true, e_screened, miss_box):
        fit, surv = M._eval(sub, g, kappa_true, V, rng)
        if surv < 1.0:
            stats["deaths"][label] = stats["deaths"].get(label, 0) + 1
            dm.append(M._normgene(g.clipped()))
            if e_screened:
                miss_box[0] += 1
        return g, fit

    def _gate_ok(child, use_e, kappa_sensed):
        if use_e:
            return M._admits(sub, child, kappa_sensed, V)
        return not M._near_observed_death(child, dm)

    # gen 0: 初期集団を sched[0] で評価 (P1)
    label0, kt0, _ = sched[0]
    miss_box = [0]
    genes = initialize_random_population(M.POP, rng)
    pop = Population(tuple(Individual(*_eval_count(g, label0, kt0, False, miss_box))
                           for g in genes))
    # gen 1..: 子生成 → pre-screen → 評価
    for label, kappa_true, sense in sched[1:]:
        kappa_sensed = sense * kappa_true
        miss_box = [0]
        e_screened_n = 0
        ranked = sorted(pop.individuals, key=lambda i: i.fitness, reverse=True)
        children = [(e.gene, False) for e in ranked[:M.ELITISM]]
        while len(children) < M.POP:
            pa = tournament_select(pop, M.TOURN_K, rng).gene
            if rng.random() < M.CX_RATE:
                pb = tournament_select(pop, M.TOURN_K, rng).gene
                child = crossover_uniform(pa, pb, rng)
            else:
                child = uniform_mutate(pa, M.SIGMA, rng)
            if arm == "NONE":
                children.append((child, False))
                continue
            use_e = (arm == "E_only") or (arm == "META" and rng.random() < trust)
            if arm == "O_only":
                use_e = False
            tries = 0
            while tries < M.RESAMPLE_CAP and not _gate_ok(child, use_e, kappa_sensed):
                src = tournament_select(pop, M.TOURN_K, rng).gene
                child = uniform_mutate(src, M.SIGMA, rng)
                tries += 1
            if use_e:
                e_screened_n += 1
            children.append((child, use_e))
        pop = Population(tuple(Individual(*_eval_count(g, label, kappa_true, scr, miss_box))
                               for g, scr in children))
        # META trust 更新 (観測可能信号のみ: E-screen 通過子の死 = predictive miss)
        if arm == "META":
            m = miss_box[0]
            if m > 0:
                trust = max(TRUST_FLOOR, trust * (TRUST_DECAY ** m))
            elif e_screened_n > 0:
                trust = min(1.0, trust + TRUST_RECOVER)
        if label in ("GOOD", "BAD"):
            stats["trust_sum"][label] = stats["trust_sum"].get(label, 0.0) + trust
            stats["trust_n"][label] = stats["trust_n"].get(label, 0) + 1
            stats["cap_sum"] += float(pop.fitness_array.mean())
            stats["cap_n"] += 1

    return {
        "arm": arm, "seed": seed,
        "deaths_good": int(stats["deaths"].get("GOOD", 0)),
        "deaths_bad": int(stats["deaths"].get("BAD", 0)),
        "deaths_total": int(stats["deaths"].get("GOOD", 0) + stats["deaths"].get("BAD", 0)),
        "capability": float(stats["cap_sum"] / max(stats["cap_n"], 1)),
        "trust_good": float(stats["trust_sum"].get("GOOD", 0.0) / max(stats["trust_n"].get("GOOD", 1), 1)),
        "trust_bad": float(stats["trust_sum"].get("BAD", 0.0) / max(stats["trust_n"].get("BAD", 1), 1)),
    }


def _paired(recs, a, b, key):
    va = np.array([r[key] for r in recs if r["arm"] == a])
    vb = np.array([r[key] for r in recs if r["arm"] == b])
    return va - vb


def _stat(deltas):
    return {
        "mean_delta": float(deltas.mean()), "median_delta": float(np.median(deltas)),
        "n_positive": int((deltas > 0).sum()), "n_negative": int((deltas < 0).sum()),
        "p_signflip_two_sided": signflip_pvalue(deltas, n_resamples=M.PERM_N_RESAMPLES,
                                                seed=M.PERM_RNG_SEED),
    }


def run_substrate(sub):
    V = M.V_BY_SUBSTRATE[sub.name]
    recs = [_run_arm_sched(sub, arm, seed, V) for arm in ARMS4 for seed in M.D_SEEDS]
    means = {arm: {
        "deaths_good": float(np.mean([r["deaths_good"] for r in recs if r["arm"] == arm])),
        "deaths_bad": float(np.mean([r["deaths_bad"] for r in recs if r["arm"] == arm])),
        "deaths_total": float(np.mean([r["deaths_total"] for r in recs if r["arm"] == arm])),
        "capability": float(np.mean([r["capability"] for r in recs if r["arm"] == arm])),
        "trust_good": float(np.mean([r["trust_good"] for r in recs if r["arm"] == arm])),
        "trust_bad": float(np.mean([r["trust_bad"] for r in recs if r["arm"] == arm])),
    } for arm in ARMS4}
    comp = {
        "bad_E_minus_META": _stat(_paired(recs, "E_only", "META", "deaths_bad")),
        "good_O_minus_META": _stat(_paired(recs, "O_only", "META", "deaths_good")),
        "total_E_minus_META": _stat(_paired(recs, "E_only", "META", "deaths_total")),
        "total_O_minus_META": _stat(_paired(recs, "O_only", "META", "deaths_total")),
        "cap_META_minus_E": _stat(_paired(recs, "META", "E_only", "capability")),
        "cap_META_minus_O": _stat(_paired(recs, "META", "O_only", "capability")),
    }
    m = means
    boundary_active = bool(m["NONE"]["deaths_total"] > 1.0)
    h_bad = bool(m["META"]["deaths_bad"] < m["E_only"]["deaths_bad"])
    h_good = bool(m["META"]["deaths_good"] < m["O_only"]["deaths_good"])
    h_overall = bool(m["META"]["deaths_total"] < m["E_only"]["deaths_total"]
                     and m["META"]["deaths_total"] < m["O_only"]["deaths_total"])
    h_cap = bool(m["META"]["capability"] >= max(m["E_only"]["capability"],
                                                m["O_only"]["capability"]) - CAP_TOL)
    h_trust = bool(m["META"]["trust_good"] > m["META"]["trust_bad"])
    return {
        "substrate": sub.name, "V": V, "arm_means": means, "comparisons": comp,
        "boundary_active": boundary_active,
        "H_bad": h_bad, "H_good": h_good, "H_overall": h_overall,
        "H_capability": h_cap, "H_trust_tracks_regime": h_trust,
        "records": recs,
    }


def run_all():
    M._ensure_utf8_stdout()
    t0 = time.time()
    out = {"preregistration": {
        "arms": ARMS4, "seeds": M.D_SEEDS, "substrates": [s.name for s in SUBSTRATES],
        "sense_good": SENSE_GOOD, "sense_bad": SENSE_BAD,
        "measure_blocks": MEASURE_BLOCKS, "G1": M.G1, "G2_WARMUP": M.G2_WARMUP,
        "kappa_low": M.KAPPA_LOW, "kappa_high": M.KAPPA_HIGH,
        "trust_rule": {"t0": TRUST0, "floor": TRUST_FLOOR, "decay_per_miss": TRUST_DECAY,
                       "recover_per_clean_gen": TRUST_RECOVER},
        "cap_tol": CAP_TOL, "alpha": M.ALPHA, "perm_seed": M.PERM_RNG_SEED,
        "hypotheses": ["H_bad", "H_good", "H_overall", "H_capability", "H_trust_tracks_regime"],
        "note": "R(蘇生) は _repair が検証器モデル依存で BAD regime に汚染されるため除外 (future work)",
    }, "substrates": {}}
    for sub in SUBSTRATES:
        res = run_substrate(sub)
        out["substrates"][sub.name] = res
        m = res["arm_means"]
        print(f"  [{sub.name}] total " + " ".join(
            f"{a}={m[a]['deaths_total']:.0f}" for a in ARMS4)
            + " | bad " + " ".join(f"{a}={m[a]['deaths_bad']:.0f}" for a in ("E_only", "O_only", "META"))
            + f" | trust G/B={m['META']['trust_good']:.2f}/{m['META']['trust_bad']:.2f}"
            + f" | H_overall={res['H_overall']} ({time.time()-t0:.0f}s)", flush=True)
    out["wall_seconds"] = round(time.time() - t0, 2)
    return out


def main():
    out = run_all()
    (_HERE / "results_viability_meta.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {_HERE / 'results_viability_meta.json'}")
    print("\n=== ④ regime-aware META (E↔O hedging, n=20, 測定40世代 G/B/G/B) ===")
    for name, res in out["substrates"].items():
        print(f"\n[{name}] active={res['boundary_active']} H_bad={res['H_bad']} H_good={res['H_good']} "
              f"H_overall={res['H_overall']} H_cap={res['H_capability']} "
              f"H_trust={res['H_trust_tracks_regime']}")
        for a in ARMS4:
            m = res["arm_means"][a]
            print(f"   {a:7s} total={m['deaths_total']:6.1f} (G={m['deaths_good']:5.1f} B={m['deaths_bad']:5.1f}) "
                  f"cap={m['capability']:.4f} trust G/B={m['trust_good']:.2f}/{m['trust_bad']:.2f}")
        for k, v in res["comparisons"].items():
            print(f"   {k}: Δ={v['mean_delta']:+.3f} (+{v['n_positive']}/-{v['n_negative']}, "
                  f"p={v['p_signflip_two_sided']:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
