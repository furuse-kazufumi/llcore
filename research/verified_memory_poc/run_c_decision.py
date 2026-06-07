# SPDX-License-Identifier: Apache-2.0
"""Phase 2a 追補 — 反証条件 (c) の決着 run (n=20 seed, 事前登録付き).

VERDICT.md §6 (c) は n=3 で INCONCLUSIVE (honest red flag) だった:
「trajectory lift が memory horizon に効く」は per-seed delta の符号不一致のため
統計的に主張できず、fitness benefit を謳わなかった。本 runner はこれを n=20 で決着させる。

## 事前登録 (PRE-REGISTRATION — 結果を見る前に本ファイルを commit すること)

- **主仮説 H1 (confirmatory)**: CopyTask delay=8 (最長 memory horizon) において
  paired delta (trajectory_tube − contraction) の test_best_fitness 平均が 0 でない。
  検定 = sign-flip permutation test (両側, n_resamples=100,000, rng seed=7)。α=0.05。
- **delay=0/4 は exploratory** (報告のみ、多重比較補正なしと明記)。
- **判定**:
  - p < 0.05 かつ delta>0 → (c) 棄却ならず: tube gate の fitness 寄与を限定的に主張可
  - p ≥ 0.05 → **(c) 決着: fitness 優位なし** — tube gate の価値は P1 soundness +
    P2 discriminating power のみ、と確定して VERDICT に追記。
    このとき |mean delta| が小さければ「verified gate の fitness tax ≈ 0」
    (安全性は fitness を犠牲にしない) として報告する — negative でも価値ある決着。
- **seed**: 2000..2019 (n=20 新規; pilot 1000-1002 と重ねない)。
- **arm**: contraction / trajectory_tube の 2 本のみ ((c) の比較対象に絞る)。
  none control と P1 cross-check は実施しない (P1/P2 は VERDICT.md で確定済み)。
- **GA/タスク/gate パラメータは run_3arm_ab.py から import** (pilot と完全同一構成)。

実行::

    py -3.11 research/verified_memory_poc/run_c_decision.py

出力::

    research/verified_memory_poc/results_c_decision.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE.parents[1] / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_3arm_ab import (  # noqa: E402  (pilot と同一構成を保証する import)
    GA_KW,
    DELAYS,
    TEST_N_TRIALS,
    TRAIN_N_TRIALS,
    STATE_DIM,
    W_BAR,
    R_MAX,
    _build_tasks,
    _ensure_utf8_stdout,
    _evolve_arm,
    _fitness_func_for,
    _test_fitness,
)

# ---- 事前登録パラメータ (docstring と一致させること) ------------------------
C_SEEDS = list(range(2000, 2020))          # n=20, pilot と独立
C_ARMS = ["contraction", "trajectory_tube"]
PERM_N_RESAMPLES = 100_000
PERM_RNG_SEED = 7
ALPHA = 0.05
CONFIRMATORY_TASK = "copy_d8"              # 主仮説 H1 の対象 (最長 memory horizon)


def signflip_pvalue(deltas: np.ndarray, *, n_resamples: int, seed: int) -> float:
    """paired delta の sign-flip permutation test (両側, +1 補正で保守的)."""
    rng = np.random.default_rng(seed)
    obs = abs(float(deltas.mean()))
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_resamples, deltas.size))
    null = np.abs((signs * deltas[None, :]).mean(axis=1))
    return float((int(np.sum(null >= obs - 1e-15)) + 1) / (n_resamples + 1))


def run_all() -> dict:
    _ensure_utf8_stdout()
    t0 = time.time()
    tasks = _build_tasks()

    cells: dict = {}
    for task_name, (task, baseline, delay) in tasks.items():
        ff = _fitness_func_for(task)
        cells[task_name] = {"baseline_mse": baseline, "delay": delay, "arms": {}}
        for arm in C_ARMS:
            recs = []
            for seed in C_SEEDS:
                res = _evolve_arm(ff, arm, seed)
                best_gene = res.final_best.gene
                rec = {
                    "seed": seed,
                    "best_gene": [best_gene.decay, best_gene.mix, best_gene.gate_str],
                    "train_best_fitness": res.final_best.fitness,
                    "test_best_fitness": _test_fitness(best_gene, task, seed),
                }
                if res.gate_stats is not None:
                    rec.update(
                        n_rejections=res.gate_stats.n_rejections,
                        fallback_count=res.gate_stats.fallback_count,
                    )
                recs.append(rec)
            cells[task_name]["arms"][arm] = recs
            mean_test = float(np.mean([r["test_best_fitness"] for r in recs]))
            print(f"  [{task_name}/{arm}] mean test-fit={mean_test:.4f}  "
                  f"({time.time() - t0:.0f}s)", flush=True)

    # ---- (c) 判定: paired delta + sign-flip permutation -----------------
    stats = {}
    for task_name, cell in cells.items():
        tube = np.array([r["test_best_fitness"] for r in cell["arms"]["trajectory_tube"]])
        cont = np.array([r["test_best_fitness"] for r in cell["arms"]["contraction"]])
        deltas = tube - cont                      # paired (同一 seed)
        p = signflip_pvalue(deltas, n_resamples=PERM_N_RESAMPLES, seed=PERM_RNG_SEED)
        stats[task_name] = {
            "mean_delta_tube_minus_contraction": float(deltas.mean()),
            "std_delta": float(deltas.std(ddof=1)),
            "n_positive": int((deltas > 0).sum()),
            "n_negative": int((deltas < 0).sum()),
            "p_signflip_two_sided": p,
            "deltas_per_seed": {str(s): float(d) for s, d in zip(C_SEEDS, deltas)},
            "role": "confirmatory" if task_name == CONFIRMATORY_TASK else "exploratory",
        }

    h1 = stats[CONFIRMATORY_TASK]
    significant = h1["p_signflip_two_sided"] < ALPHA
    if significant and h1["mean_delta_tube_minus_contraction"] > 0:
        verdict_c = ("H1 supported: tube gate shows a statistically detectable "
                     "fitness advantage on the longest memory horizon (d8)")
    elif significant:
        verdict_c = ("H1 reversed: tube gate is statistically WORSE on d8 — "
                     "report as gate fitness cost")
    else:
        verdict_c = ("(c) settled: no detectable fitness difference (n=20) — "
                     "tube gate value = P1 soundness + P2 discriminating power; "
                     "fitness tax ~ 0 (safety does not cost fitness here)")

    return {
        "preregistration": {
            "confirmatory_task": CONFIRMATORY_TASK,
            "alpha": ALPHA,
            "test": "sign-flip permutation, two-sided",
            "n_resamples": PERM_N_RESAMPLES,
            "perm_rng_seed": PERM_RNG_SEED,
            "seeds": C_SEEDS,
            "arms": C_ARMS,
            "inherits_config_from": "run_3arm_ab.py (GA_KW/DELAYS/W_BAR/R_MAX/readout)",
        },
        "config": {
            "GA_KW": GA_KW, "DELAYS": DELAYS, "TRAIN_N_TRIALS": TRAIN_N_TRIALS,
            "TEST_N_TRIALS": TEST_N_TRIALS, "STATE_DIM": STATE_DIM,
            "W_BAR": W_BAR, "R_MAX": R_MAX,
        },
        "cells": cells,
        "stats": stats,
        "verdict_c": verdict_c,
        "wall_seconds": round(time.time() - t0, 2),
    }


def main() -> int:
    out = run_all()
    out_path = _HERE / "results_c_decision.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    print("\n=== (c) decision ===")
    for task_name, s in out["stats"].items():
        print(f"{task_name}: mean Δ={s['mean_delta_tube_minus_contraction']:+.4f}  "
              f"p={s['p_signflip_two_sided']:.4f}  [{s['role']}]  "
              f"+{s['n_positive']}/-{s['n_negative']}")
    print(f"\nverdict: {out['verdict_c']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
