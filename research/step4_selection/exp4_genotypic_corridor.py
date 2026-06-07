# SPDX-License-Identifier: Apache-2.0
"""実験4: ③ が立つ正準状態 — genotypic corridor + fitness dip (deceptive corridor).

実験1-3 の教訓を統合した「diversity 維持が必須」な landscape:
- **behavior = mean(gene)** (1D スカラ ∈ [0,1])。高 behavior = 全 dim が高い genotype 極値
  → random は中心極限で mean≈0.5 に固着し高 behavior に**絶対到達できない** (corridor は genotype 内)。
- **fitness profile along behavior**: 局所最適 (b=0.4, 0.6) → **dip (b≈0.65, ~0)** → 大域最適 (b=0.9, 1.0)。
  - RR-hillclimb: b≈0.5 から局所 0.6 へ climb。dip を越えるには downhill が必要だが (1+1) は downhill 拒否
    → 詰まる。restart も fresh random は必ず b≈0.5 → 同じ罠 (C2)。
  - MAP-Elites: behavior grid を **stepping-stone 保持**で充填 (dip cell も「新規 cell」として保持) →
    b を 0.5→0.9 へ ratchet し大域到達。**downhill を跨ぐのが diversity 維持の本質的効果 (③)**。

③成立条件: MAP-Elites が random / panmictic-GA / RR-hillclimb の 3 baseline 全てに有意勝利
+ MAP-Elites のみ大域到達 (C2/C3/C4 同時)。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection_lab import compare, run_methods_over_seeds  # noqa: E402


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_ensure_utf8_stdout()

D = 24
_NOISE = 0.008


def behavior_mean(gene: np.ndarray) -> np.ndarray:
    """behavior = 全 dim の平均 (1D). 高 behavior = genotype 極値 = random 不到達の corridor."""
    return np.array([gene.mean()])


def corridor_eval(gene: np.ndarray, rng: np.random.Generator) -> float:
    b = float(gene.mean())
    local = 0.60 * np.exp(-((b - 0.40) ** 2) / (2 * 0.08 ** 2))
    glob = 1.00 * np.exp(-((b - 0.90) ** 2) / (2 * 0.06 ** 2))
    return float(max(local, glob) + rng.normal(0, _NOISE))


def main() -> int:
    bounds = (np.zeros(D), np.ones(D))
    results = run_methods_over_seeds(
        corridor_eval, behavior_mean,
        dim=D, bounds=bounds,
        behavior_bounds=(np.zeros(1), np.ones(1)),
        grid_shape=(24,),
        n_evals=6000, n_seeds=20, honest_n_trials=30, sigma=0.10,
    )
    print(f"実験4: D={D} deceptive corridor (behavior=mean, 局所0.60@b=0.4 / dip / 大域1.00@b=0.9)")
    print("=" * 74)
    for k, v in results.items():
        print(f"  {k:14s}: honest best mean={v.mean():.4f} std={v.std():.4f} "
              f"min={v.min():.4f} max={v.max():.4f}")
    print("-" * 74)
    passes = {}
    for base in ("rr_hillclimb", "panmictic_ga", "random"):
        c = compare(results["map_elites"], results[base], "map_elites", base)
        passes[base] = c.passes
        print(f"  MAP-Elites vs {base:13s}: diff={c.diff:+.4f} win={c.win_rate:.2f} "
              f"p={c.wilcoxon_p:.4g} δ={c.cliff_delta:+.2f} → {'③成立側' if c.passes else '有意差なし'}")
    print("=" * 74)
    me = results["map_elites"]
    print(f"  [C2] RR-hillclimb 大域峰(>0.8)不到達率: {float(np.mean(results['rr_hillclimb']<0.8)):.2f}")
    print(f"  [C2] random 大域峰(>0.8)不到達率: {float(np.mean(results['random']<0.8)):.2f}")
    print(f"  [C3] MAP-Elites 大域峰(>0.8)到達率: {float(np.mean(me>0.8)):.2f}")
    all_pass = all(passes.values())
    print(f"\n  実験4 ③成立 (3 baseline 全てに有意勝利): {'YES' if all_pass else 'NO'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
