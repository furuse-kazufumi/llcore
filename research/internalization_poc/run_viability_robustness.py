# SPDX-License-Identifier: Apache-2.0
"""R-endo viability 反証#2 潰し — death-count 順序の頑健性 + 安定↔可塑性トレードオフ。

文献接地 (MEMORY_FORMATION_TAXONOMY.md §反証 #2, 最重要) が中心仮説の生死を決めると指摘:
「sound >> empirical (REVIVE<ENDO<OBSERVE の死回避順序) が **linear toy substrate のアーティファクト**
の可能性。非線形/高次元/複数 seed で順序逆転、または ρ<1 が保守的すぎて可塑性 (記憶獲得能力) を殺し
能力指標で OBSERVE に負ければ格下げ。」

本 runner は run_viability_ab を import し、globals を config ごとに変えて 5-arm を回し、各 config で
(1) **death-count 順序の安定性** (REVIVE/ENDO が NONE/OBSERVE より低死を保つか) と
(2) **能力 (capability) ギャップ** (sound arm ENDO/REVIVE の fitness が unsafe arm NONE/OBSERVE に
劣らないか = ρ<1 が記憶を殺さないか) を報告する。

## config 軸 (反証の各部分を当てる)

- baseline         : κ_high=2.0, dim=8, delay=8, seeds 3000- (既報の再現アンカー)
- seed_shift       : seeds 4000- (順序が seed 集合に依らないか = 複数 seed 反証)
- kappa_1.5 / 3.0  : catastrophe 規模を変えて順序安定か
- dim_24           : 高次元で順序安定か (低次元 toy 反証)
- hard_mem_delay20 : delay=20 で記憶最適が高 a0 (発散境界近傍) を**要求** → 安定↔可塑性トレードオフを顕在化
                     (ρ<1 が記憶を殺すなら ENDO/REVIVE の capability がここで NONE/OBSERVE に負ける)

非線形性は softsat/highgain (tanh) で既にカバー (baseline で linear+highgain を回す)。

## 事前登録 (結果取得前に commit)

- **H_robust (本丸)**: 全 config で death-count が REVIVE ≤ ENDO < {EXO, NONE} を保ち、かつ ENDO < OBSERVE
  (sound < empirical) を保つ。1 config でも逆転 (ENDO ≥ OBSERVE or REVIVE > ENDO 大幅) → sound>>empirical を
  「環境依存」に格下げ。
- **H_capability**: sound arm (ENDO/REVIVE) の phase2 fitness が unsafe arm (NONE/OBSERVE) に対し margin
  −δ_cap=−0.05 を下回らない (ρ<1 が記憶能力を殺さない)。特に hard_mem_delay20 で ENDO/REVIVE fitness ≥
  NONE/OBSERVE − 0.05 を満たすか = 可塑性の生死。下回れば「安定 vs 可塑性が両立しない」= 中心仮説リスク顕在化。
- 各 config × substrate (linear, highgain) で n=12 seeds。death-count + fitness を全 arm で dump。

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
    } for arm in M.ARMS}
    order = sorted(M.ARMS, key=lambda a: means[a]["deaths"])
    sound_viol, sound_checked = M._soundness_violations(sub, V)
    # 反証判定
    revive_le_endo = means["REVIVE"]["deaths"] <= means["ENDO"]["deaths"] + 1e-9
    endo_lt_observe = means["ENDO"]["deaths"] < means["OBSERVE"]["deaths"]
    endo_lt_none = means["ENDO"]["deaths"] < means["NONE"]["deaths"]
    order_holds = bool(revive_le_endo and endo_lt_observe and endo_lt_none)
    # 能力: sound arm が unsafe arm に大きく劣らないか
    unsafe_best_fit = max(means["NONE"]["fitness"], means["OBSERVE"]["fitness"])
    sound_best_fit = max(means["ENDO"]["fitness"], means["REVIVE"]["fitness"])
    capability_ok = bool(sound_best_fit >= unsafe_best_fit - DELTA_CAP)
    return {
        "substrate": sub.name, "arm_means": means, "death_order_low_to_high": order,
        "order_holds": order_holds, "soundness_violations": sound_viol, "soundness_checked": sound_checked,
        "capability_ok": capability_ok,
        "capability_gap_sound_minus_unsafe": round(sound_best_fit - unsafe_best_fit, 4),
    }


def run_all():
    M._ensure_utf8_stdout()
    t0 = time.time()
    out = {"preregistration": {"configs": CONFIGS, "n_seeds": N_SEEDS, "delta_cap": DELTA_CAP,
                               "substrates": [s.name for s in SUBSTRATES]},
           "results": {}}
    for cfg in CONFIGS:
        _apply_config(cfg)
        out["results"][cfg["name"]] = {}
        for sub in SUBSTRATES:
            res = _run_config_substrate(cfg, sub)
            out["results"][cfg["name"]][sub.name] = res
            m = res["arm_means"]
            print(f"  [{cfg['name']:18s}/{sub.name:8s}] order: {' < '.join(res['death_order_low_to_high'])} | "
                  f"ENDO_d={m['ENDO']['deaths']:.1f} OBS_d={m['OBSERVE']['deaths']:.1f} | "
                  f"order_holds={res['order_holds']} cap_ok={res['capability_ok']} "
                  f"(gap={res['capability_gap_sound_minus_unsafe']:+.3f}) viol={res['soundness_violations']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    # 集計
    all_res = [r for c in out["results"].values() for r in c.values()]
    out["summary"] = {
        "n_config_substrate": len(all_res),
        "order_holds_count": sum(1 for r in all_res if r["order_holds"]),
        "capability_ok_count": sum(1 for r in all_res if r["capability_ok"]),
        "soundness_violations_total": sum(r["soundness_violations"] for r in all_res),
        "order_holds_all": all(r["order_holds"] for r in all_res),
        "capability_ok_all": all(r["capability_ok"] for r in all_res),
    }
    out["wall_seconds"] = round(time.time() - t0, 2)
    return out


def main():
    out = run_all()
    (_HERE / "results_viability_robustness.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    s = out["summary"]
    print(f"\nwrote {_HERE / 'results_viability_robustness.json'}")
    print(f"\n=== 反証#2 潰し summary ({s['n_config_substrate']} config×substrate) ===")
    print(f"death-order (REVIVE≤ENDO<OBSERVE,NONE) holds: {s['order_holds_count']}/{s['n_config_substrate']} "
          f"(all={s['order_holds_all']})")
    print(f"capability (sound ≥ unsafe − {DELTA_CAP}) ok: {s['capability_ok_count']}/{s['n_config_substrate']} "
          f"(all={s['capability_ok_all']})")
    print(f"soundness violations total: {s['soundness_violations_total']}")
    if s["order_holds_all"] and s["capability_ok_all"] and s["soundness_violations_total"] == 0:
        print("\nVERDICT: sound>>empirical 順序は全 config で頑健 + 可塑性も保持 + soundness 健全 "
              "= 反証#2 を潰した (linear toy アーティファクトでない)。")
    else:
        print("\nVERDICT: 一部 config で逆転/可塑性死/soundness 破綻 = sound>>empirical を環境依存に格下げ "
              "(honest disclosure; 該当 config を精査)。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
