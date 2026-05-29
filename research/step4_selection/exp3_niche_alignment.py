# SPDX-License-Identifier: Apache-2.0
"""実験3: ③ が立つ状態の構築 — 可動 behavior × 高次元 alignment × behavior valley.

実験2 の失敗: behavior=高次元平均だと corner 到達に全 dim を揃える必要があり、valley に勾配が
ないため MAP-Elites の illumination すら届かず全手法が局所最適で停滞。

修正した landscape (MAP-Elites が load-bearing になる構造):
- **behavior = (gene[0], gene[1])** 直接可動な 2 座標。MAP-Elites は corner cell を確実に充填できる。
- **global 最適 = behavior corner (1,1) かつ gene[2:] が target に alignment** (高次元 D-2 dim)。
  → random は alignment を揃えられない / hill-climbing は behavior valley で corner に向かえない。
- **behavior valley**: 局所最適 (behavior=(0.2,0.2), 広 0.6) と global corner の間で fitness が落ちる
  → hill-climbing は局所で詰まる (C2)。
- MAP-Elites のみ: corner niche を **run 全体で維持** し、その niche 内で alignment を蓄積 → global 到達。
  random-restart は corner に restart しても 1 restart 分の予算では alignment 未完 (niche を捨てる)。

③成立条件: MAP-Elites が random / panmictic-GA / RR-hillclimb の **3 baseline 全て**に有意勝利。
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

D = 32
_ALIGN_DIM = D - 2
_TARGET = np.ones(_ALIGN_DIM)  # gene[2:] の alignment target (corner)
_ALIGN_SCALE = 1.6             # alignment 緩急 (大=易)
_NOISE = 0.008


def behavior_xy(gene: np.ndarray) -> np.ndarray:
    """behavior = (gene[0], gene[1]) 直接可動 2 座標."""
    return gene[:2].copy()


def _alignment(gene: np.ndarray) -> float:
    return float(np.exp(-np.sum((gene[2:] - _TARGET) ** 2) / (2 * _ALIGN_SCALE ** 2)))


def deceptive_eval(gene: np.ndarray, rng: np.random.Generator) -> float:
    b = behavior_xy(gene)
    local = 0.60 * np.exp(-np.sum((b - np.array([0.2, 0.2])) ** 2) / (2 * 0.16 ** 2))
    corner_gate = np.exp(-np.sum((b - np.array([1.0, 1.0])) ** 2) / (2 * 0.10 ** 2))
    glob = 1.00 * corner_gate * _alignment(gene)  # corner かつ高次元 alignment で 1.0
    return float(max(local, glob) + rng.normal(0, _NOISE))


def main() -> int:
    bounds = (np.zeros(D), np.ones(D))
    results = run_methods_over_seeds(
        deceptive_eval, behavior_xy,
        dim=D, bounds=bounds,
        behavior_bounds=(np.zeros(2), np.ones(2)),
        grid_shape=(16, 16),
        n_evals=6000, n_seeds=20, honest_n_trials=30, sigma=0.12,
    )
    print(f"実験3: D={D} 可動behavior×alignment(dim={_ALIGN_DIM})×valley (局所0.60/大域1.00 corner)")
    print("=" * 72)
    for k, v in results.items():
        print(f"  {k:14s}: honest best mean={v.mean():.4f} std={v.std():.4f} "
              f"min={v.min():.4f} max={v.max():.4f}")
    print("-" * 72)
    passes = {}
    for base in ("rr_hillclimb", "panmictic_ga", "random"):
        c = compare(results["map_elites"], results[base], "map_elites", base)
        passes[base] = c.passes
        print(f"  MAP-Elites vs {base:13s}: diff={c.diff:+.4f} win={c.win_rate:.2f} "
              f"p={c.wilcoxon_p:.4g} δ={c.cliff_delta:+.2f} → {'③成立側' if c.passes else '有意差なし'}")
    print("=" * 72)
    me = results["map_elites"]
    print(f"  [C2] RR-hillclimb 大域峰(>0.85)不到達率: {float(np.mean(results['rr_hillclimb']<0.85)):.2f}")
    print(f"  [C2] panmictic-GA 大域峰(>0.85)不到達率: {float(np.mean(results['panmictic_ga']<0.85)):.2f}")
    print(f"  [C3] MAP-Elites 大域峰(>0.85)到達率: {float(np.mean(me>0.85)):.2f}")
    all_pass = all(passes.values())
    print(f"\n  実験3 ③成立 (3 baseline 全てに有意勝利): {'YES' if all_pass else 'NO'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
