# SPDX-License-Identifier: Apache-2.0
"""FDC-behavior deceptiveness 指標を合成 dip-depth knob で CALIBRATE する.

目的: ``metric_fdc_behavior.deceptiveness`` が、③ load-bearing 閾値 d*=0.16 が確立済みの
合成 corridor (research/step_c_applicability/exp_knob_sweep.make_corridor_eval) 上で
**d とともに単調に増えるか / d*=0.16 でどんな値を取るか / 閾値越え (d>=d*) と整合するか**を測る。

出力: metric curve (d → deceptiveness mean/std/CI), Spearman corr(metric, d), monotonicity,
metric_at_dstar (= d=0.16 での値)。結果は calibrate_fdc_behavior_results.json に保存。

規律 (honest disclosure):
- metric はサンプリング推定 → 各 d で n_repeats=5 の独立 seed で mean ± std / 95% CI を報告。
- 「変にきれいな単調曲線」が出たら内訳を疑う (best_behavior 推定のばらつき・degenerate flag を確認)。
- selection_lab / exp_knob_sweep は read-only import (改変しない)。numpy のみ。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
# 合成 knob (make_corridor_eval, behavior_mean) を read-only import
sys.path.insert(0, str(_HERE.parents[0] / "step_c_applicability"))
sys.path.insert(0, str(_HERE))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

from exp_knob_sweep import D as CORRIDOR_D  # noqa: E402  (= 24)
from exp_knob_sweep import behavior_mean, make_corridor_eval  # noqa: E402
from metric_fdc_behavior import deceptiveness_with_ci  # noqa: E402

# task 指定の calibration sweep (d 値)
D_LEVELS = [0.0, 0.05, 0.10, 0.13, 0.16, 0.20, 0.30, 0.50, 1.0]
D_STAR = 0.16  # 既知の③ load-bearing 閾値 (STEP_C_APPLICABILITY_VERDICT.md)

N_SAMPLES = 4000
HONEST_N = 20
N_REPEATS = 5
BASE_SEED = 20260531


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank 相関 (scipy 非依存、tie は平均ランクで処理)."""
    def _rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        ranks = np.empty(len(v), dtype=np.float64)
        ranks[order] = np.arange(len(v), dtype=np.float64)
        # tie 平均ランク
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        avg = sums / counts
        return avg[inv]
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) < 1e-12 or np.std(ry) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _fdc_true_reference(d: float, true_b: float, rng: np.random.Generator) -> float:
    """control: TRUE global 最適 behavior (b=true_b, 構築上既知) を reference にした FDC.

    操作的 metric ではない (実 task では真の最適を知れない) が、操作的 metric の高 d collapse の
    原因が「best-sample reference の relocation」であることを切り分けるための diagnostic。
    """
    bounds = (np.zeros(CORRIDOR_D), np.ones(CORRIDOR_D))
    eval_once = make_corridor_eval(d)
    lo, hi = bounds
    genes = lo + (hi - lo) * rng.random((N_SAMPLES, CORRIDOR_D))
    fits = np.array([
        float(np.mean([eval_once(g, rng) for _ in range(HONEST_N)])) for g in genes
    ])
    behs = np.array([np.atleast_1d(behavior_mean(g)) for g in genes])
    dist = np.linalg.norm(behs - np.array([true_b])[None, :], axis=1)
    if np.std(fits) < 1e-9 or np.std(dist) < 1e-9:
        return 0.0
    return float(np.corrcoef(fits, -dist)[0, 1])


def main() -> int:
    bounds = (np.zeros(CORRIDOR_D), np.ones(CORRIDOR_D))
    print("FDC-behavior deceptiveness CALIBRATION on synthetic dip-depth knob")
    print(f"D={CORRIDOR_D} behavior=mean / n_samples={N_SAMPLES} honest_n={HONEST_N} "
          f"n_repeats={N_REPEATS} / known d*={D_STAR}")
    print("=" * 90)

    curve: list[dict] = []
    for d in D_LEVELS:
        eval_once = make_corridor_eval(d)
        rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, int(d * 1000)]))
        res = deceptiveness_with_ci(
            eval_once, behavior_mean, bounds, CORRIDOR_D, rng,
            n_samples=N_SAMPLES, honest_n_trials=HONEST_N, n_repeats=N_REPEATS,
        )
        best_beh_mean = float(
            np.mean([r["best_behavior"][0] for r in res["per_repeat"]])
        )
        entry = {
            "d": d,
            "metric": res["deceptiveness_mean"],
            "metric_std": res["deceptiveness_std"],
            "ci95_lo": res["ci95_lo"],
            "ci95_hi": res["ci95_hi"],
            "fdc_mean": res["fdc_mean"],
            "fdc_std": res["fdc_std"],
            "any_degenerate": res["any_degenerate"],
            "best_behavior_mean": best_beh_mean,
        }
        curve.append(entry)
        print(f"  d={d:.2f}: deceptiveness={res['deceptiveness_mean']:.4f} "
              f"±{res['deceptiveness_std']:.4f} "
              f"(CI [{res['ci95_lo']:.4f},{res['ci95_hi']:.4f}]) "
              f"FDC={res['fdc_mean']:+.4f}±{res['fdc_std']:.4f} "
              f"degen={res['any_degenerate']}", flush=True)

    ds = np.array([c["d"] for c in curve], dtype=np.float64)
    ms = np.array([c["metric"] for c in curve], dtype=np.float64)

    spearman = _spearman(ms, ds)

    # --- 診断: 操作的 metric は best-behavior 推定 (= 到達可能 best) に anchor するため、
    #     dip が local 峰高さを超える高 d で reference が global 峰側 (b≈0.69) から
    #     local 峰側 (b≈0.47) へ relocate し、metric が **反転** する (honest disclosure)。
    #     これを (a) 単調な低 d 部分集合の Spearman と (b) TRUE global 最適 (b=0.9, 構築上既知) を
    #     reference にした control で特性化する。control は操作的 metric ではない (真の最適を知れない)
    #     が、collapse の原因が「reference relocation」であることを切り分ける。
    low_mask = ds <= 0.20
    spearman_low = _spearman(ms[low_mask], ds[low_mask])

    best_behs = np.array([c.get("best_behavior_mean", float("nan")) for c in curve])

    # monotone (non-decreasing) チェック — 推定ノイズを考慮して厳密単調と「CI 重複を許す単調」両方
    diffs = np.diff(ms)
    strictly_monotone = bool(np.all(diffs >= 0))
    # ノイズ許容単調: 各隣接ペアで後者の mean が前者の mean - std を下回らない
    stds = np.array([c["metric_std"] for c in curve])
    tol_monotone = bool(np.all(ms[1:] >= ms[:-1] - np.maximum(stds[1:], stds[:-1])))

    # metric_at_dstar
    i_star = D_LEVELS.index(D_STAR)
    metric_at_dstar = float(ms[i_star])

    # reproduces_threshold 判定:
    # metric_at_dstar を「閾値ライン」として、d>=0.16 の点が全てこのライン以上、
    # かつ d<0.16 の点が全てこのライン未満なら、metric crossing が③ load-bearing 開始と一致。
    below = ms[ds < D_STAR]
    at_or_above = ms[ds >= D_STAR]
    sep_below_ok = bool(np.all(below < metric_at_dstar)) if len(below) else True
    sep_above_ok = bool(np.all(at_or_above >= metric_at_dstar)) if len(at_or_above) else True
    reproduces_threshold = bool(sep_below_ok and sep_above_ok)

    # --- TRUE-reference control sweep (diagnostic for the high-d collapse) ---
    control_curve: list[dict] = []
    for d in D_LEVELS:
        rng = np.random.default_rng(np.random.SeedSequence([BASE_SEED, 99, int(d * 1000)]))
        fdc_true = _fdc_true_reference(d, true_b=0.9, rng=rng)
        control_curve.append({"d": d, "fdc_true_ref": fdc_true,
                              "deceptiveness_true_ref": 1.0 - max(0.0, fdc_true)})
    cm = np.array([c["deceptiveness_true_ref"] for c in control_curve])
    spearman_true_ref = _spearman(cm, ds)

    print("=" * 90)
    print("DIAGNOSTIC: operational best-behavior reference vs TRUE global (b=0.9) reference")
    for c, cc in zip(curve, control_curve):
        print(f"  d={c['d']:.2f}: op_best_beh={c['best_behavior_mean']:.3f} "
              f"op_dec={c['metric']:.4f} | TRUE-ref_dec={cc['deceptiveness_true_ref']:.4f}")
    print(f"  Spearman(metric,d) restricted to d<=0.20 = {spearman_low:+.4f}")
    print(f"  Spearman(TRUE-ref metric, d) over full sweep = {spearman_true_ref:+.4f}")
    print("=" * 90)
    print(f"Spearman(metric, d) = {spearman:+.4f}")
    print(f"strictly monotone (mean) = {strictly_monotone} ; "
          f"noise-tolerant monotone = {tol_monotone}")
    print(f"metric_at_dstar (d=0.16) = {metric_at_dstar:.4f}")
    print(f"separation: all(d<0.16) < dstar_val = {sep_below_ok} ; "
          f"all(d>=0.16) >= dstar_val = {sep_above_ok}")
    print(f"reproduces_threshold (metric crossing == ③ becomes load-bearing) = "
          f"{reproduces_threshold}")

    payload = {
        "metric_name": "FDC-behavior (Fitness-Distance Correlation in behavior space)",
        "definition": (
            "Sample genes uniformly in bounds; honest-evaluate each (avg over honest_n_trials); "
            "estimate global-best behavior b* = behavior of the max-fitness sampled gene; "
            "for each sample compute proximity = -||behavior - b*||; FDC = Pearson corr(fitness, proximity); "
            "deceptiveness = 1 - max(0, FDC). Easy/non-deceptive -> FDC~+1 -> deceptiveness~0; "
            "deceptive -> FDC low/negative -> deceptiveness>=1. Applies to any (genotype,behavior,fitness) landscape."
        ),
        "config": {
            "synthetic_corridor_D": CORRIDOR_D, "behavior": "mean(gene)",
            "bounds": "[0,1]^24", "n_samples": N_SAMPLES, "honest_n_trials": HONEST_N,
            "n_repeats": N_REPEATS, "base_seed": BASE_SEED, "d_star": D_STAR,
        },
        "curve": curve,
        "spearman_with_d": spearman,
        "strictly_monotone": strictly_monotone,
        "noise_tolerant_monotone": tol_monotone,
        "metric_at_dstar": metric_at_dstar,
        "reproduces_threshold": reproduces_threshold,
    }
    out_path = _HERE / "calibrate_fdc_behavior_results.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
