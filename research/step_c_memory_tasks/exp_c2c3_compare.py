# exp_c2c3_compare.py
# SPDX-License-Identifier: Apache-2.0
"""C2/C3: 多峰タスクで MAP-E が 3 baseline を強化基準で上回るか.

C1 (exp_c1_landscape) で多峰だったタスク = delayed_parity (valley=1.000) /
flip_flop (valley=0.939) を対象にする。delayed_recall は滑らか (valley=0.000)
だったため除外。各 method は同一 n_evals 予算、best gene を進化と独立な fresh
seed で honest 再評価 (elitism artifact 排除)。

出力する診断:
- C2 (hill-climbing が局所最適に詰まる) = baseline の **大域到達率が低い** こと。
  到達率 = honest R² >= τ となる seed の割合を複数閾値 τ∈{0.5,0.7,0.8,0.9} で出す。
  MAP-E が高到達・baseline が低到達なら C2 成立 (任意 cutoff を避け複数閾値併記)。
  cf. 手順4 (STEP4_SELECTION_VERDICT) は honest fitness>0.8 を大域峰 proxy とした。
- C3 (niching が baseline に有意勝利) = MAP-E vs {random,RR,GA} を強化版 honest 基準
  (片側 Wilcoxon p<0.05・|paired_sign_delta|≥0.147・n_seeds≥15) で strict_compare。

再現性のため per-seed の honest R² 配列を exp_c2c3_results.json に保存する
(閾値を後から変えても再実行不要)。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

# Windows cp932 console で R²/τ/日本語 を出力するため UTF-8 へ reconfigure
# (memory: feedback_cli_utf8_stdout_pattern。tee/pipe 経由でも UTF-8 byte を吐く)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))

from memory_tasks import DelayedParityTask, FlipFlopTask
from reservoir import LeakyDelayLineReservoir, gene_bounds, make_behavior, make_eval_once
from strict_compare import strict_compare
from selection_lab import run_methods_over_seeds

# C1 で多峰と確定したタスクのみ。
TASKS = {
    "delayed_parity": DelayedParityTask(seq_len=20, window=5, in_dim=1),
    "flip_flop": FlipFlopTask(seq_len=30, in_dim=2),
}
METHODS = ("map_elites", "random", "rr_hillclimb", "panmictic_ga")
BASELINES = ("random", "rr_hillclimb", "panmictic_ga")
REACH_THRESHOLDS = (0.5, 0.7, 0.8, 0.9)
OUT_JSON = Path(__file__).resolve().parent / "exp_c2c3_results.json"


def _reach_rate(scores: np.ndarray, tau: float) -> float:
    """honest R² >= tau となる seed の割合 (大域峰 proxy 到達率)."""
    return float(np.mean(np.asarray(scores) >= tau))


def main() -> None:
    results: dict[str, dict] = {}
    for name, task in TASKS.items():
        res = LeakyDelayLineReservoir(n_taps=8, in_dim=task.in_dim)
        eval_once = make_eval_once(res, task, n_train=48, n_eval=48)
        behavior = make_behavior(res)
        lo, hi = gene_bounds(res)
        scores = run_methods_over_seeds(
            eval_once, behavior, dim=res.gene_dim, bounds=(lo, hi),
            behavior_bounds=(np.array([0.0, 0.0]), np.array([1.0, 0.5])),
            grid_shape=(12, 12), n_evals=2000, n_seeds=15, honest_n_trials=20,
            sigma=0.15, base_seed=20260530,
        )

        # --- 記録 ---
        results[name] = {
            "per_seed_scores": {m: np.asarray(scores[m]).tolist() for m in METHODS},
            "mean": {m: float(np.mean(scores[m])) for m in METHODS},
            "std": {m: float(np.std(scores[m])) for m in METHODS},
            "reach_rate": {
                m: {f"tau_{tau}": _reach_rate(scores[m], tau) for tau in REACH_THRESHOLDS}
                for m in METHODS
            },
            "strict_compare": {},
        }

        print(f"\n[{name}] mean honest R²: "
              f"MAP-E={np.mean(scores['map_elites']):.4f} "
              f"random={np.mean(scores['random']):.4f} "
              f"RR={np.mean(scores['rr_hillclimb']):.4f} "
              f"GA={np.mean(scores['panmictic_ga']):.4f}")

        # C2: 到達率 (MAP-E 高 vs baseline 低 = baseline が局所最適に詰まる)
        print(f"  C2 到達率 (R²>=τ の seed 割合):")
        for tau in REACH_THRESHOLDS:
            rr = {m: _reach_rate(scores[m], tau) for m in METHODS}
            print(f"    τ={tau}: MAP-E={rr['map_elites']:.2f} "
                  f"random={rr['random']:.2f} RR={rr['rr_hillclimb']:.2f} "
                  f"GA={rr['panmictic_ga']:.2f}")

        # C3: strict gate
        all_pass = True
        print(f"  C3 strict gate (MAP-E vs baseline):")
        for base in BASELINES:
            r = strict_compare(scores["map_elites"], scores[base], "map_elites", base)
            all_pass = all_pass and r.passes
            results[name]["strict_compare"][base] = {
                "diff": r.diff, "wilcoxon_p": r.wilcoxon_p,
                "paired_sign_delta": r.paired_sign_delta, "passes": r.passes,
            }
            print(f"    MAP-E vs {base}: diff={r.diff:+.4f} p={r.wilcoxon_p:.4g} "
                  f"δ={r.paired_sign_delta:+.2f} passes={r.passes}")
        results[name]["c3_all_pass"] = all_pass
        print(f"  => C3 ({name}): {'成立 (③ load-bearing)' if all_pass else '不成立'}")
        # 各タスク完了ごとに保存 (途中クラッシュでも部分結果を残す = 約24分/タスクの再計算回避)
        OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  (中間保存: {OUT_JSON.name})")

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果を保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
