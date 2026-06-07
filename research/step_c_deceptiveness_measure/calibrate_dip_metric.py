# SPDX-License-Identifier: Apache-2.0
"""behavior-elite fitness-DIP メトリクスの校正 + 実 task 配置.

(1) 合成 knob (make_corridor_eval(d), behavior=mean, D=24, [0,1]^24) で d を sweep し
    メトリクス曲線 / Spearman(metric, d) / 単調性 / d=0.16 での値 (metric_at_dstar) を測る。
(2) reproduces_threshold: metric_at_dstar をまたぐことが「③が load-bearing (d>=0.16)」と
    一致するか判定。
(3) 実 task (E-A multitask variable_delay_recall / flip_flop) に同じメトリクスを当て、
    合成軸上にどこに落ちるかを measure (numeric 配置は honest に CI 付きで報告)。

研究隔離: exp_knob_sweep.py / reservoir.py / memory_tasks.py を read-only import。
src と selection_lab は非変更。numpy のみ。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parents[0] / "step_c_applicability"))
sys.path.insert(0, str(_HERE.parents[0] / "step_c_memory_tasks"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask"))
sys.path.insert(0, str(_HERE.parents[0] / "ea_multitask" / "candidates"))
sys.path.insert(0, str(_HERE.parents[1] / "src"))

from metric_behavior_elite_dip import deceptiveness_estimate  # noqa: E402


def _utf8() -> None:
    for s in (sys.stdout, sys.stderr):
        rc = getattr(s, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_utf8()

# 校正で使う dip depth levels (タスク指定そのまま)
D_LEVELS = [0.0, 0.05, 0.10, 0.13, 0.16, 0.20, 0.30, 0.50, 1.0]
D_STAR = 0.16

# サンプリング設定: bin あたり期待 = N/bins。dip max の上方バイアス抑制に bin>=~150 確保
N_SAMPLES = 4000
N_BINS = 24
N_SEEDS = 5


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman 順位相関 (scipy 非依存; 同順位は平均順位)."""
    def rank(a: np.ndarray) -> np.ndarray:
        order = np.argsort(a, kind="mergesort")
        r = np.empty(len(a), dtype=np.float64)
        r[order] = np.arange(len(a), dtype=np.float64)
        # 同値の平均順位
        a_sorted = a[order]
        i = 0
        while i < len(a):
            j = i
            while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
                j += 1
            if j > i:
                avg = (r[order[i:j + 1]]).mean()
                r[order[i:j + 1]] = avg
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def calibrate_synthetic() -> dict:
    from exp_knob_sweep import D as SYN_D
    from exp_knob_sweep import behavior_mean, make_corridor_eval

    bounds = (np.zeros(SYN_D), np.ones(SYN_D))
    curve = []
    print("=== (1) 合成 knob 校正 (behavior-elite fitness-DIP) ===")
    print(f"D={SYN_D} behavior=mean  N_samples={N_SAMPLES} bins={N_BINS} seeds={N_SEEDS}")
    for d in D_LEVELS:
        eval_once = make_corridor_eval(d)
        est = deceptiveness_estimate(
            eval_once, behavior_mean, bounds, SYN_D,
            n_seeds=N_SEEDS, n_samples=N_SAMPLES, n_bins=N_BINS, honest_n_trials=1,
        )
        curve.append({"d": d, "metric": est.mean, "std": est.std,
                      "ci95_lo": est.ci95_lo, "ci95_hi": est.ci95_hi,
                      "per_seed": est.per_seed})
        print(f"  d={d:.2f}: dip={est.mean:.4f} ± {est.std:.4f} "
              f"(95%CI [{est.ci95_lo:.4f}, {est.ci95_hi:.4f}])")

    ds = np.array([c["d"] for c in curve])
    ms = np.array([c["metric"] for c in curve])
    rho = _spearman(ds, ms)
    # 単調性: 隣接差が全て >= -tol (CI を踏まえ小ノイズ許容)
    diffs = np.diff(ms)
    tol = 0.5 * float(np.mean([c["std"] for c in curve]))  # 平均ノイズの半分を許容
    monotone = bool(np.all(diffs >= -max(tol, 1e-9)))
    strictly_monotone = bool(np.all(diffs >= 0))
    metric_at_dstar = float(ms[ds == D_STAR][0])

    # reproduces_threshold: d>=d* (=load-bearing) ⟺ metric>=metric_at_dstar が一致するか。
    # 実 verdict の load_bearing は d>=0.16。閾値メトリクスとして metric_at_dstar をまたぐと
    # load-bearing が立つ、という対応が成立するか。
    lb_true = ds >= D_STAR
    metric_cross = ms >= metric_at_dstar - 1e-12
    agree = int(np.sum(lb_true == metric_cross))
    reproduces = bool(agree == len(ds))

    print(f"\n  Spearman(metric, d) = {rho:.4f}")
    print(f"  monotone (tol={tol:.4f}) = {monotone}  / strictly = {strictly_monotone}")
    print(f"  metric_at_dstar (d=0.16) = {metric_at_dstar:.4f}")
    print(f"  reproduces_threshold (metric>=at_dstar ⟺ d>=0.16) = {reproduces} "
          f"({agree}/{len(ds)} levels agree)")

    return {
        "curve": curve, "spearman": rho, "monotone": monotone,
        "strictly_monotone": strictly_monotone,
        "metric_at_dstar": metric_at_dstar, "d_star": D_STAR,
        "reproduces_threshold": reproduces, "agree_levels": f"{agree}/{len(ds)}",
        "monotone_tol": tol,
    }


def measure_real_tasks() -> dict:
    """実 task に同メトリクスを当て、合成軸上の配置を測る (CI 付き)."""
    from reservoir import (LeakyDelayLineReservoir, gene_bounds, make_behavior,
                           make_eval_once)
    from memory_tasks import FlipFlopTask
    from task_mixture import TaskMixture
    from variable_delay_recall import VariableDelayRecallTask

    out: dict = {}
    print("\n=== (3) 実 task 配置 (behavior=2D reservoir descriptor → PCA 1D 射影) ===")

    # --- E-A multitask: exp_ea3 と同 config (train regimes の混合 fitness) ---
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    bounds = gene_bounds(res)
    behavior = make_behavior(res)
    dim = res.gene_dim
    train_regimes = [VariableDelayRecallTask(seq_len=D, distractor_amp=0.2, in_dim=2)
                     for D in (15, 30)]
    ev_ea = make_eval_once(res, TaskMixture(train_regimes), n_train=48, n_eval=48)
    # 確率的 fitness → honest 再評価。bin あたり標本を保つため N は控えめ + honest_n=10。
    est_ea = deceptiveness_estimate(
        ev_ea, behavior, bounds, dim,
        n_seeds=5, n_samples=1200, n_bins=16, honest_n_trials=10,
    )
    out["ea_multitask_variable_delay_recall"] = est_ea.as_dict()
    print(f"  E-A multitask: dip={est_ea.mean:.4f} ± {est_ea.std:.4f} "
          f"(95%CI [{est_ea.ci95_lo:.4f}, {est_ea.ci95_hi:.4f}])")

    # --- flip_flop: 同 reservoir (in_dim=2 一致) + FlipFlop 単一タスク ---
    ev_ff = make_eval_once(res, FlipFlopTask(), n_train=48, n_eval=48)
    est_ff = deceptiveness_estimate(
        ev_ff, behavior, bounds, dim,
        n_seeds=5, n_samples=1200, n_bins=16, honest_n_trials=10,
    )
    out["flip_flop"] = est_ff.as_dict()
    print(f"  flip_flop    : dip={est_ff.mean:.4f} ± {est_ff.std:.4f} "
          f"(95%CI [{est_ff.ci95_lo:.4f}, {est_ff.ci95_hi:.4f}])")

    out["step6_text_proxy"] = {
        "status": "SKIPPED",
        "reason": ("step6_real_proxy/esn_landscape.py exposes ESN/next_char_accuracy with a "
                   "corpus-loading custom signature, not a clean eval_once(gene,rng)->float + "
                   "behavior descriptor. Wiring it faithfully (corpus load + bespoke gene->ESN "
                   "decode + behavior axis) exceeds this session's budget. Per task instructions "
                   "we SKIP rather than fabricate. STEP4_VERDICT §7 already places it qualitatively "
                   "on the smooth (d<d*) side."),
    }
    return out


def main() -> int:
    syn = calibrate_synthetic()
    real = measure_real_tasks()

    # 実 task の配置判定 (合成曲線への内挿は単調性が前提)。
    placement = {}
    if syn["monotone"]:
        # 合成曲線で metric -> d を内挿 (単調なので一意近似)
        ds = np.array([c["d"] for c in syn["curve"]])
        ms = np.array([c["metric"] for c in syn["curve"]])
        order = np.argsort(ms)
        for name in ("ea_multitask_variable_delay_recall", "flip_flop"):
            m = real[name]["mean"]
            d_hat = float(np.interp(m, ms[order], ds[order]))
            below = m < syn["metric_at_dstar"]
            placement[name] = {"metric": m, "implied_d": d_hat,
                               "below_d_star": bool(below)}
    out = {
        "metric_name": "behavior_elite_fitness_dip",
        "synthetic_calibration": syn,
        "real_tasks": real,
        "real_task_placement_on_synthetic_axis": placement,
    }
    path = _HERE / "calibrate_dip_metric_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
