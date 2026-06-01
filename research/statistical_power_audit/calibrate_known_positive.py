# SPDX-License-Identifier: Apache-2.0
"""(A) 校正 — 既知真陽性 corridor に full machinery をかけ、偽陰性境界を地図化.

目的: 「③ が真に勝つのに full strict gate が no-effect と誤判定する (効果量 d, n) 境界」
(= false-negative onset 曲線 d_fn(n)) を地図化し、現行 n (15 / 実験では HONEST_N=30) で
d*=0.16 付近の真の中効果を取り逃がすかを明示する。

材料 (一次情報、再導出しない): exp_knob_sweep.make_corridor_eval(d) / run_methods_crn /
behavior_mean / D=24 を read-only import (改造禁止)。corridor は d*_strict=0.16 が③
load-bearing 開始点。

手順 (DESIGN.calibration_plan):
(1) 効果量軸 = dip depth d を transition 近傍で密に。
(2) 標本数軸 n_seeds ∈ {10,15,30,60}。
(3) 各 (d,n) cell で生数値を残す: per-seed scores 両群, diff, 片側 Wilcoxon p,
    paired_sign_delta, textbook Cliff δ, win_rate, n, gate 各条件の個別 pass, load_bearing。
(4) 教科書 Cliff δ も併記 (min_effect=0.147 がどちらの尺度で効くか可視化)。
(5) 2D pass マップ → false-negative onset 曲線 d_fn(n) を抽出。
予算: 破綻ゲート G1 で <900s。3 base_seed で d_fn(n) の安定性も残す。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _audit_common as AC  # noqa: E402

# transition 近傍 + smooth control (d=0) を含む。d>=0.16 が真陽性域 (③ load-bearing)。
D_LEVELS = [0.00, 0.10, 0.13, 0.14, 0.15, 0.16, 0.18, 0.20]
N_LEVELS = [10, 15, 30, 60]
BASE_SEEDS = [20260530, 777, 31337]
BASELINES = ("rr_hillclimb", "panmictic_ga", "random")


def _cell_record(scores: dict[str, np.ndarray], d: float) -> dict:
    """1 (d,n,seed) cell: MAP-E vs 3 baseline の完全 gate + load_bearing."""
    me = scores["map_elites"]
    gates: dict[str, dict] = {}
    n_beaten = 0
    for b in BASELINES:
        g = AC.eval_gate(me, scores[b], "map_elites", b)
        gates[b] = {
            "diff": g.diff, "wilcoxon_p": g.wilcoxon_p,
            "paired_sign_delta": g.paired_sign_delta,
            "cliff_delta_textbook": g.cliff_delta_textbook,
            "win_rate": g.win_rate, "cohen_dz": g.cohen_dz,
            "cond_diff_pos": g.cond_diff_pos, "cond_p": g.cond_p,
            "cond_n": g.cond_n, "cond_effect": g.cond_effect,
            "passes": g.passes,
        }
        n_beaten += int(g.passes)
    load_bearing = n_beaten == len(BASELINES)
    # best baseline (honest mean 最大) との vs も明示
    best_b = max(BASELINES, key=lambda b: float(scores[b].mean()))
    g_best = AC.eval_gate(me, scores[best_b], "map_elites", best_b)
    return {
        "means": {m: float(scores[m].mean()) for m in scores},
        "best_baseline": best_b,
        "gate_vs_best_baseline": {
            "diff": g_best.diff, "wilcoxon_p": g_best.wilcoxon_p,
            "paired_sign_delta": g_best.paired_sign_delta,
            "cliff_delta_textbook": g_best.cliff_delta_textbook,
            "passes": g_best.passes,
        },
        "gates_vs_each": gates,
        "n_baselines_beaten": n_beaten,
        "load_bearing": load_bearing,
        "per_seed": {m: scores[m].tolist() for m in scores},
    }


def _onset_curve(cells: dict) -> dict:
    """各 n で load_bearing が True に転じる最小 d = d_fn(n) を抽出.

    真の効果は d とともに連続増加するので、d_fn(n) が d*=0.16 より大きければ
    d∈[0.16, d_fn(n)] の真陽性域を取り逃がす (Type II 偏向の直接証拠)。
    """
    out: dict[str, object] = {}
    for n in N_LEVELS:
        # d 昇順で最初に load_bearing=True になる d
        lb_d = None
        for d in D_LEVELS:
            key = f"d={d:.2f}_n={n}"
            if cells.get(key, {}).get("load_bearing"):
                lb_d = d
                break
        # textbook Cliff δ at d*=0.16 (取り逃がしの効果量の証拠)
        cell16 = cells.get(f"d=0.16_n={n}", {})
        cliff_at_dstar = None
        gv = cell16.get("gate_vs_best_baseline")
        if gv:
            cliff_at_dstar = gv.get("cliff_delta_textbook")
        out[str(n)] = {
            "d_fn_load_bearing_onset": lb_d,
            "load_bearing_at_dstar_0.16": cell16.get("load_bearing"),
            "n_baselines_beaten_at_0.16": cell16.get("n_baselines_beaten"),
            "cliff_delta_vs_best_at_0.16": cliff_at_dstar,
            "missed_band": (None if lb_d is None or lb_d <= 0.16
                            else f"[0.16,{lb_d:.2f}) 真陽性域を no-effect 判定"),
        }
    return out


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny grid for import/数値確認")
    args = ap.parse_args()

    guard = AC.RunGuard.start("calibrate_known_positive")

    if args.smoke:
        d_levels = [0.00, 0.16]
        n_levels = [10, 15]
        base_seeds = [20260530]
        n_evals = 800
    else:
        d_levels = D_LEVELS
        n_levels = N_LEVELS
        base_seeds = BASE_SEEDS
        n_evals = 1500

    print(f"[calibrate] d={d_levels} n={n_levels} seeds={base_seeds} n_evals={n_evals}")
    print(f"  corridor D={AC.CORRIDOR_D} gate=(diff>0 ∧ p<0.05 ∧ n>=15 ∧ |psd|>=0.147)")

    # primary seed の full grid + 他 seed は transition 近傍のみ (robustness)
    primary = base_seeds[0]
    cells: dict[str, dict] = {}
    for d in d_levels:
        for n in n_levels:
            sc = AC.corridor_method_scores(d, n_seeds=n, n_evals=n_evals, base_seed=primary)
            rec = _cell_record(sc, d)
            cells[f"d={d:.2f}_n={n}"] = rec
            print(f"  d={d:.2f} n={n}: beaten={rec['n_baselines_beaten']}/3 "
                  f"LB={rec['load_bearing']} "
                  f"vs_best diff={rec['gate_vs_best_baseline']['diff']:+.4f} "
                  f"p={rec['gate_vs_best_baseline']['wilcoxon_p']:.3g} "
                  f"cliff={rec['gate_vs_best_baseline']['cliff_delta_textbook']:+.2f}")

    onset = _onset_curve(cells)

    # robustness: transition 近傍を他 seed で。d_fn(15) と d*=0.16 検出の安定性。
    robustness: dict[str, object] = {}
    near = [d for d in d_levels if 0.10 <= d <= 0.18]
    for bs in base_seeds[1:]:
        rcells: dict[str, dict] = {}
        for d in near:
            sc = AC.corridor_method_scores(d, n_seeds=15, n_evals=n_evals, base_seed=bs)
            rcells[f"d={d:.2f}_n=15"] = _cell_record(sc, d)
        lb_d = next((d for d in near
                     if rcells[f"d={d:.2f}_n=15"]["load_bearing"]), None)
        robustness[str(bs)] = {
            "n": 15, "d_fn_onset": lb_d,
            "load_bearing_at_0.16": rcells.get("d=0.16_n=15", {}).get("load_bearing"),
            "per_d": {k: {"load_bearing": v["load_bearing"],
                          "n_baselines_beaten": v["n_baselines_beaten"]}
                      for k, v in rcells.items()},
        }
        print(f"  robustness seed={bs} (n=15): d_fn_onset={lb_d} "
              f"LB@0.16={robustness[str(bs)]['load_bearing_at_0.16']}")

    meta = guard.finish()
    payload = {
        "_meta": {**meta, "design": "(A) calibration false-negative boundary",
                  "d_levels": d_levels, "n_levels": n_levels,
                  "base_seeds": base_seeds, "n_evals": n_evals,
                  "d_star_strict_known": 0.16,
                  "gate": "diff>0 ∧ one-sided Wilcoxon p<0.05 ∧ n>=15 ∧ |paired_sign_delta|>=0.147",
                  "note": "exp_knob_sweep.make_corridor_eval/run_methods_crn を read-only import。src 無改変。"},
        "cells": cells,
        "false_negative_onset_dfn": onset,
        "robustness_other_seeds": robustness,
    }
    out = AC.dump_json(AC.AUDIT_DIR / "calibrate_known_positive_results.json", payload)
    print(f"[calibrate] wrote {out}  ({meta['wall_clock_s']}s, src_unchanged={meta['src_unchanged']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
