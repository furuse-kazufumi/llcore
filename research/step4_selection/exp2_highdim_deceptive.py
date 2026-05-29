# SPDX-License-Identifier: Apache-2.0
"""実験2: MAP-Elites の本領 — 高次元 genotype × 低次元 behavior × 欺瞞.

実験1 の教訓: 低次元(2D)+高予算では random が coverage で大域峰に届き ③ が立たない。
MAP-Elites が load-bearing になる正しい構造:
- **高次元 genotype** (D=20): random は behavior corner に対応する genotype 領域を引けない
  (mean が中心極限で 0.5 付近に集中)。
- **低次元 behavior** (2D): illumination が tractable。
- **欺瞞 fitness**: 大域最適 (behavior=(1,1) corner) へ向かう途中に valley があり、局所最適
  (behavior=(0.3,0.3)) が広 basin → hill-climbing は局所で詰まる。

期待: random=corner 不到達 / hill-climbing=欺瞞で局所最適 / panmictic GA=早期収束 /
MAP-Elites=behavior grid の stepping-stone を維持し corner へ ratchet → 大域最適到達 (③成立)。
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

D = 20
_HALF = D // 2
_NOISE = 0.01


def behavior_meanhalves(gene: np.ndarray) -> np.ndarray:
    """behavior = (前半 dim の平均, 後半 dim の平均) ∈ [0,1]^2.

    genotype→behavior は many-to-one。random gene の behavior は中心極限で (0.5,0.5) 付近に集中し、
    corner (1,1) には到達しない = corner は高次元 genotype の極小領域。
    """
    return np.array([gene[:_HALF].mean(), gene[_HALF:].mean()])


def deceptive_eval(gene: np.ndarray, rng: np.random.Generator) -> float:
    """欺瞞 fitness (behavior の関数). 大域最適=corner(1,1) 狭 / 局所最適=(0.3,0.3) 広."""
    b = behavior_meanhalves(gene)
    local = 0.60 * np.exp(-np.sum((b - np.array([0.3, 0.3])) ** 2) / (2 * 0.18 ** 2))
    glob = 1.00 * np.exp(-np.sum((b - np.array([1.0, 1.0])) ** 2) / (2 * 0.07 ** 2))
    return float(max(local, glob) + rng.normal(0, _NOISE))


def main() -> int:
    bounds = (np.zeros(D), np.ones(D))
    results = run_methods_over_seeds(
        deceptive_eval, behavior_meanhalves,
        dim=D, bounds=bounds,
        behavior_bounds=(np.zeros(2), np.ones(2)),
        grid_shape=(12, 12),
        n_evals=4000, n_seeds=20, honest_n_trials=30, sigma=0.12,
    )
    print(f"実験2: 高次元(D={D}) 欺瞞 landscape (局所 0.60 広 / 大域 1.00 corner)")
    print("=" * 70)
    for k, v in results.items():
        print(f"  {k:14s}: honest best mean={v.mean():.4f} std={v.std():.4f} "
              f"min={v.min():.4f} max={v.max():.4f}")
    print("-" * 70)
    for base in ("rr_hillclimb", "panmictic_ga", "random"):
        c = compare(results["map_elites"], results[base], "map_elites", base)
        verdict = "③ 成立側" if c.passes else "有意差なし"
        print(f"  MAP-Elites vs {base:13s}: diff={c.diff:+.4f} win={c.win_rate:.2f} "
              f"p={c.wilcoxon_p:.4g} δ={c.cliff_delta:+.2f} → {verdict}")
    print("=" * 70)
    me = results["map_elites"]
    print(f"  [C2] RR-hillclimb が大域峰(>0.85)不到達率: {float(np.mean(results['rr_hillclimb']<0.85)):.2f}")
    print(f"  [C2] random が大域峰(>0.85)不到達率: {float(np.mean(results['random']<0.85)):.2f}")
    print(f"  [C3] MAP-Elites 大域峰(>0.9)到達率: {float(np.mean(me>0.9)):.2f}")
    # ③成立 = 3 baseline すべてに有意勝利
    all_pass = all(compare(me, results[b], "me", b).passes for b in ("rr_hillclimb", "panmictic_ga", "random"))
    print(f"\n  実験2 ③成立 (3 baseline 全てに有意勝利): {'YES' if all_pass else 'NO'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
