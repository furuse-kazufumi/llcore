# SPDX-License-Identifier: Apache-2.0
"""R-endo viability 反証#2 潰し — 2 軸 (死回避/記憶保存) の頑健性 + 安定↔可塑性トレードオフ。

文献接地 (MEMORY_FORMATION_TAXONOMY.md §反証 #2, 最重要) が中心仮説の生死を決めると指摘:
「sound >> empirical が **linear toy substrate のアーティファクト**の可能性。非線形/高次元/複数 seed で
順序逆転、または ρ<1 が保守的すぎて可塑性 (記憶獲得能力) を殺し能力指標で OBSERVE に負ければ格下げ。」

本 runner は run_viability_ab (助走版: G2_WARMUP + G2_MEASURE) を import し、globals を config ごとに
変えて 5-arm を回し、各 config で以下を報告する。

## 2 軸 (REVIVE 修正後の意味論 — ユーザー批判反映 2026-06-07)

REVIVE は「死を経験してから蘇生」する設計 (死回避=0 はトートロジーのため廃止)。よって死回避軸の
対象は ENDO のみ、REVIVE の真価は記憶保存軸で測る:
- **軸1 死回避** : ENDO (sound 予見) の measure 期 deaths < NONE かつ < OBSERVE (sound < empirical)。
- **軸2 記憶保存**: REVIVE の pop_mean_fitness > NONE (死を経験しても記憶が消えない = 復活の価値)。
- **能力 (可塑性)**: sound arm (ENDO/REVIVE) の best fitness が unsafe arm (NONE/OBSERVE) に
  margin −δ_cap=−0.05 を下回らない (ρ<1 が記憶獲得能力を殺さない)。

## config 軸 (反証の各部分を当てる)

- baseline         : κ_high=2.0, dim=8, delay=8, seeds 3000- (既報の再現アンカー)
- seed_shift       : seeds 4000- (順序が seed 集合に依らないか = 複数 seed 反証)
- kappa_1.5 / 3.0  : catastrophe 規模を変えて順序安定か
- dim_24           : 高次元で順序安定か (低次元 toy 反証)
- hard_mem_delay20 : delay=20 で記憶最適が高 a0 (発散境界近傍) を**要求** → 安定↔可塑性トレードオフを顕在化
                     (ρ<1 が記憶を殺すなら ENDO/REVIVE の capability がここで NONE/OBSERVE に負ける)

非線形性は highgain (tanh) でカバー (linear+highgain を回す)。

## 事前登録 (PRE-REGISTRATION — 結果取得前に commit)

- **H_avoid_robust (軸1)**: 死境界 active な全 config×substrate で ENDO deaths < NONE かつ
  ENDO < OBSERVE。1 つでも逆転 (ENDO ≥ OBSERVE) → sound>>empirical を「環境依存」に格下げ。
- **H_memory_robust (軸2)**: 同上の全 config×substrate で REVIVE pop_mean_fitness > NONE。
  逆転 → 「復活=死んでも記憶」を環境依存に格下げ。
- **H_capability**: sound arm (ENDO/REVIVE) の best fitness ≥ unsafe arm (NONE/OBSERVE) − 0.05。
  特に hard_mem_delay20 で下回れば「安定 vs 可塑性が両立しない」= 中心仮説リスク顕在化。
- **soundness (必須)**: 各 config×substrate で violations==0。>0 → 認証破綻 (F3)。
- **F2 (boundary inactive)**: NONE の measure deaths ≤ 1.0 の config×substrate は死境界 inactive =
  判定から除外し明示報告 (silent drop しない)。
- 各 config × substrate (linear, highgain) で n=12 seeds。deaths/fitness/pop_mean を全 arm で dump。

実行::  py -3.11 research/internalization_poc/run_viability_robustness.py
出力::  research/internalization_poc/results_viability_robustness.json
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
from llcore.fitness import make_fixed_readout  # noqa: E402
from viability_substrates import HighGainSubstrate, LinearSubstrate  # noqa: E402

DELTA_CAP = 0.05   # 能力ギャップ許容 (ENDO/REVIVE が unsafe arm にこれ以上劣ったら可塑性死)
BOUNDARY_MIN_NONE_DEATHS = 1.0   # F2: NONE measure deaths ≤ これ → 死境界 inactive (判定除外を明示報告)
SUBSTRATES = [LinearSubstrate(), HighGainSubstrate()]
N_SEEDS = 12

CONFIGS = [
    {"name": "baseline", "kappa": 2.0, "dim": 8, "delay": 8, "seed0": 3000},
    {"name": "seed_shift", "kappa": 2.0, "dim": 8, "delay": 8, "seed0": 4000},
    {"name": "kappa_1.5", "kappa": 1.5, "dim": 8, "delay": 8, "seed0": 3000},
    {"name": "kappa_3.0", "kappa": 3.0, "dim": 8, "delay": 8, "seed0": 3000},
    {"name": "dim_24", "kappa": 2.0, "dim": 24, "delay": 8, "seed0": 3000},
    {"name": "hard_mem_delay20", "kappa": 2.0, "dim": 8, "delay": 20, "seed0": 3000},
]


def _apply_config(cfg):
    """run_viability_ab の globals を config で上書き (readout は dim 変更時に再構築)。"""
    M.KAPPA_HIGH = cfg["kappa"]
    M.STATE_DIM = cfg["dim"]
    M.DELAY = cfg["delay"]
    M.D_SEEDS = list(range(cfg["seed0"], cfg["seed0"] + N_SEEDS))
    M._READOUT = make_fixed_readout(cfg["dim"], cfg["dim"], seed=1001)


def _run_config_substrate(cfg, sub):
    V = M.V_BY_SUBSTRATE[sub.name]
    recs = [M._run_arm(sub, arm, seed, V) for arm in M.ARMS for seed in M.D_SEEDS]
    means = {arm: {
        "deaths": float(np.mean([r["phase2_deaths"] for r in recs if r["arm"] == arm])),
        "fitness": float(np.mean([r["phase2_final_best_fitness"] for r in recs if r["arm"] == arm])),
        "pop_mean": float(np.mean([r["phase2_pop_mean_fitness"] for r in recs if r["arm"] == arm])),
    } for arm in M.ARMS}
    order = sorted(M.ARMS, key=lambda a: means[a]["deaths"])
    sound_viol, sound_checked = M._soundness_violations(sub, V)
    # F2: 死境界が動いていない config は判定から除外 (明示報告)
    boundary_active = bool(means["NONE"]["deaths"] > BOUNDARY_MIN_NONE_DEATHS)
    # 軸1 死回避: ENDO (sound 予見) < NONE かつ < OBSERVE。REVIVE は死を経験する設計なので対象外。
    avoid_holds = bool(means["ENDO"]["deaths"] < means["NONE"]["deaths"]
                       and means["ENDO"]["deaths"] < means["OBSERVE"]["deaths"])
    # 軸2 記憶保存: REVIVE は死を経験しても集団記憶 (pop_mean) を NONE より保つ。
    memory_holds = bool(means["REVIVE"]["pop_mean"] > means["NONE"]["pop_mean"])
    # 能力 (可塑性): sound arm が unsafe arm に大きく劣らないか。
    unsafe_best_fit = max(means["NONE"]["fitness"], means["OBSERVE"]["fitness"])
    sound_best_fit = max(means["ENDO"]["fitness"], means["REVIVE"]["fitness"])
    capability_ok = bool(sound_best_fit >= unsafe_best_fit - DELTA_CAP)
    return {
        "substrate": sub.name, "arm_means": means, "death_order_low_to_high": order,
        "boundary_active": boundary_active,
        "avoid_holds": avoid_holds, "memory_holds": memory_holds,
        "soundness_violations": sound_viol, "soundness_checked": sound_checked,
        "capability_ok": capability_ok,
        "capability_gap_sound_minus_unsafe": round(sound_best_fit - unsafe_best_fit, 4),
    }


def run_all():
    M._ensure_utf8_stdout()
    t0 = time.time()
    out = {"preregistration": {
        "configs": CONFIGS, "n_seeds": N_SEEDS, "delta_cap": DELTA_CAP,
        "boundary_min_none_deaths": BOUNDARY_MIN_NONE_DEATHS,
        "substrates": [s.name for s in SUBSTRATES],
        "G1": M.G1, "G2_WARMUP": M.G2_WARMUP, "G2_MEASURE": M.G2_MEASURE,
        "revive_semantics": "死を経験してから蘇生 (死回避軸の対象外; 記憶保存軸 pop_mean で測る)",
    }, "results": {}}
    for cfg in CONFIGS:
        _apply_config(cfg)
        out["results"][cfg["name"]] = {}
        for sub in SUBSTRATES:
            res = _run_config_substrate(cfg, sub)
            out["results"][cfg["name"]][sub.name] = res
            m = res["arm_means"]
            print(f"  [{cfg['name']:18s}/{sub.name:8s}] order: {' < '.join(res['death_order_low_to_high'])} | "
                  f"ENDO_d={m['ENDO']['deaths']:.1f} OBS_d={m['OBSERVE']['deaths']:.1f} "
                  f"NONE_d={m['NONE']['deaths']:.1f} | active={res['boundary_active']} "
                  f"avoid={res['avoid_holds']} mem={res['memory_holds']} cap={res['capability_ok']} "
                  f"(gap={res['capability_gap_sound_minus_unsafe']:+.3f}) viol={res['soundness_violations']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # 集計 (死境界 active のみ判定対象; inactive は明示報告)
    all_res = [r for c in out["results"].values() for r in c.values()]
    active = [r for r in all_res if r["boundary_active"]]
    inactive = [(c, s) for c, subs in out["results"].items() for s, r in subs.items()
                if not r["boundary_active"]]
    out["summary"] = {
        "n_config_substrate": len(all_res), "n_active": len(active),
        "inactive_excluded": [f"{c}/{s}" for c, s in inactive],
        "avoid_holds_count": sum(1 for r in active if r["avoid_holds"]),
        "memory_holds_count": sum(1 for r in active if r["memory_holds"]),
        "capability_ok_count": sum(1 for r in active if r["capability_ok"]),
        "soundness_violations_total": sum(r["soundness_violations"] for r in all_res),
        "avoid_holds_all": all(r["avoid_holds"] for r in active),
        "memory_holds_all": all(r["memory_holds"] for r in active),
        "capability_ok_all": all(r["capability_ok"] for r in active),
    }
    out["wall_seconds"] = round(time.time() - t0, 2)
    return out


def main():
    out = run_all()
    (_HERE / "results_viability_robustness.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    s = out["summary"]
    print(f"\nwrote {_HERE / 'results_viability_robustness.json'}")
    print(f"\n=== 反証#2 潰し summary ({s['n_active']}/{s['n_config_substrate']} active config×substrate) ===")
    if s["inactive_excluded"]:
        print(f"boundary inactive (F2, 判定除外): {', '.join(s['inactive_excluded'])}")
    print(f"軸1 死回避 (ENDO < NONE, OBSERVE) holds: {s['avoid_holds_count']}/{s['n_active']} "
          f"(all={s['avoid_holds_all']})")
    print(f"軸2 記憶保存 (REVIVE pop_mean > NONE) holds: {s['memory_holds_count']}/{s['n_active']} "
          f"(all={s['memory_holds_all']})")
    print(f"能力 (sound ≥ unsafe − {DELTA_CAP}) ok: {s['capability_ok_count']}/{s['n_active']} "
          f"(all={s['capability_ok_all']})")
    print(f"soundness violations total: {s['soundness_violations_total']}")
    if (s["avoid_holds_all"] and s["memory_holds_all"] and s["capability_ok_all"]
            and s["soundness_violations_total"] == 0):
        print("\nVERDICT: 2 軸 (sound>>empirical 死回避 + 復活の記憶保存) は全 active config で頑健 + "
              "可塑性も保持 + soundness 健全 = 反証#2 を潰した (linear toy アーティファクトでない)。")
    else:
        print("\nVERDICT: 一部 config で逆転/可塑性死/soundness 破綻 = 該当軸を環境依存に格下げ "
              "(honest disclosure; 該当 config を精査)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
