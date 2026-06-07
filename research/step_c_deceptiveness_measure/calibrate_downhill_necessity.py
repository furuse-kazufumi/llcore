# SPDX-License-Identifier: Apache-2.0
"""DOWNHILL-NECESSITY メトリックを合成 dip-depth knob d で calibrate する.

手順:
- d in [0.0,0.05,0.10,0.13,0.16,0.20,0.30,0.50,1.0] で make_corridor_eval(d) を構築
  (behavior=mean, D=24, bounds [0,1]^24; exp_knob_sweep.py を read-only import)。
- 各 d で deceptiveness_with_ci を計算 (mean + SE)。
- metric curve / Spearman(metric, d) / 単調性 / d=0.16 の metric (= metric_at_dstar) を報告。
- reproduces_threshold: metric が metric_at_dstar をまたぐ点が ③ load-bearing 化 (d>=0.16) と
  一致するかを判定。

honest disclosure: sampling noise (SE) を併記し、非単調性があれば隠さず報告する。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "step_c_applicability"))

from exp_knob_sweep import D as CORRIDOR_D  # noqa: E402
from exp_knob_sweep import behavior_mean, make_corridor_eval  # noqa: E402
from metric_downhill_necessity import deceptiveness_with_ci  # noqa: E402


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank corr (numpy only; tie は平均 rank)."""
    def rank(v: np.ndarray) -> np.ndarray:
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        # tie 平均 rank
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        mean_r = sums / counts
        return mean_r[inv]

    rx, ry = rank(x), rank(y)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


def main() -> int:
    d_levels = [0.0, 0.05, 0.10, 0.13, 0.16, 0.20, 0.30, 0.50, 1.0]
    d_star = 0.16
    bounds = (np.zeros(CORRIDOR_D), np.ones(CORRIDOR_D))
    behavior_bounds = (np.zeros(1), np.ones(1))

    # 合成 corridor は確率的 (noise σ=0.008) → fitness_trials=8 で decision noise を抑制。
    # n_bins=24 は exp_knob_sweep の MAP-Elites grid と同一にして公正比較。
    cfg = dict(n_samples=6000, n_bins=24, behavior_bounds=behavior_bounds,
               fitness_trials=8, n_repeats=7)

    print("DOWNHILL-NECESSITY calibration on synthetic dip-depth knob d")
    print(f"D={CORRIDOR_D} behavior=mean / n_samples={cfg['n_samples']} n_bins={cfg['n_bins']} "
          f"fitness_trials={cfg['fitness_trials']} n_repeats={cfg['n_repeats']}")
    print("=" * 78)

    curve: list[dict] = []
    rng = np.random.default_rng(20260531)
    for d in d_levels:
        eval_once = make_corridor_eval(d)
        mean, se, samples = deceptiveness_with_ci(
            eval_once, behavior_mean, bounds, CORRIDOR_D, rng, **cfg)
        curve.append({"d": d, "metric": mean, "se": se,
                      "samples": [round(s, 4) for s in samples]})
        print(f"d={d:.2f}: deceptiveness={mean:.4f} (SE {se:.4f})  "
              f"reach={1-mean:.4f}  samples={[round(s,3) for s in samples]}")

    ds = np.array([c["d"] for c in curve])
    ms = np.array([c["metric"] for c in curve])
    spearman = _spearman(ms, ds)

    # 単調性 (非減少): SE を超える減少が無いか。noise 許容で「SE 内の揺れは単調扱い」。
    strictly_nondec = bool(np.all(np.diff(ms) >= 0))
    ses = np.array([c["se"] for c in curve])
    tol = ses[:-1] + ses[1:]  # 隣接 SE 和を許容幅に
    monotone_within_noise = bool(np.all(np.diff(ms) >= -tol))

    metric_at_dstar = float(ms[ds == d_star][0])

    # reproduces_threshold 判定:
    # ③は d>=d_star で load-bearing。metric が「d_star で metric_at_dstar に達し、d<d_star では
    # それ未満、d>=d_star では以上」を満たせば閾値再現と言える。
    below = ms[ds < d_star]
    atabove = ms[ds >= d_star]
    sep_below = bool(np.all(below < metric_at_dstar)) if len(below) else True
    sep_above = bool(np.all(atabove >= metric_at_dstar)) if len(atabove) else True
    reproduces = bool(sep_below and sep_above and monotone_within_noise)

    print("=" * 78)
    print(f"Spearman(metric, d) = {spearman:+.4f}")
    print(f"monotone non-decreasing (strict)        = {strictly_nondec}")
    print(f"monotone non-decreasing (within SE band) = {monotone_within_noise}")
    print(f"metric_at_dstar (d=0.16) = {metric_at_dstar:.4f}")
    print(f"  below d* all < metric_at_dstar  = {sep_below}")
    print(f"  d>=d* all >= metric_at_dstar    = {sep_above}")
    print(f"reproduces_threshold = {reproduces}")

    payload = {
        "metric_name": "downhill_necessity",
        "config": {k: (v if not isinstance(v, tuple) else "behavior_bounds_[0,1]")
                   for k, v in cfg.items()},
        "d_star": d_star,
        "curve": curve,
        "spearman_with_d": spearman,
        "monotone_strict": strictly_nondec,
        "monotone_within_se": monotone_within_noise,
        "metric_at_dstar": metric_at_dstar,
        "reproduces_threshold": reproduces,
    }
    out = _HERE / "calibration_results.json"
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
