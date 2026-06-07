# SPDX-License-Identifier: Apache-2.0
"""実験1: 機構検証 — 既知の欺瞞 landscape で MAP-Elites が強 baseline を超えるか.

設計ノート C2-C4 を最小の既知ケースで確認する:
- landscape: 広い凡庸な局所最適 (height 0.7, 広 basin) + 狭い高峰の大域最適 (height 1.0, 極小 basin)。
  random start の大半は広 basin に落ち、hill-climbing は局所最適 0.7 で詰まる (C2)。
- 期待: MAP-Elites は behavior grid で大域峰近傍 cell を維持・精錬し 1.0 に届く (C3)。
  random-restart hill-climbing (強 baseline, 探索量で多峰攻略) に勝てば勝因は探索量でなく
  behavioral 維持 (C4)。

behavior descriptor = gene 座標そのもの (2D)。これは「behavioral niching が効くか」の sanity。
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

# 2D 欺瞞 landscape: 広い局所最適 + 狭い大域最適。
_LOCAL_CENTER = np.array([0.35, 0.35])
_GLOBAL_CENTER = np.array([0.88, 0.88])
_LOCAL_H, _LOCAL_W = 0.70, 0.22   # 広 basin (引き込み域 大)
_GLOBAL_H, _GLOBAL_W = 1.00, 0.045  # 極小 basin (引き込み域 小 = 欺瞞)
_NOISE = 0.01


def deceptive_eval(gene: np.ndarray, rng: np.random.Generator) -> float:
    local = _LOCAL_H * np.exp(-np.sum((gene - _LOCAL_CENTER) ** 2) / (2 * _LOCAL_W ** 2))
    glob = _GLOBAL_H * np.exp(-np.sum((gene - _GLOBAL_CENTER) ** 2) / (2 * _GLOBAL_W ** 2))
    val = max(local, glob)
    return float(val + rng.normal(0, _NOISE))


def behavior_identity(gene: np.ndarray) -> np.ndarray:
    return gene


def main() -> int:
    dim = 2
    bounds = (np.zeros(dim), np.ones(dim))
    results = run_methods_over_seeds(
        deceptive_eval, behavior_identity,
        dim=dim, bounds=bounds,
        behavior_bounds=(np.zeros(dim), np.ones(dim)),
        grid_shape=(12, 12),
        n_evals=2000, n_seeds=20, honest_n_trials=30, sigma=0.10,
    )
    print("実験1: 欺瞞 landscape (広局所最適 0.70 + 狭大域最適 1.00)")
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
    # C2 確認: hill-climbing が局所最適 (~0.70) に詰まっているか
    rr = results["rr_hillclimb"]
    stuck = float(np.mean(rr < 0.85))
    print(f"  [C2] RR-hillclimb が大域峰 (>0.85) に届かない seed 率: {stuck:.2f}")
    me = results["map_elites"]
    reach = float(np.mean(me > 0.9))
    print(f"  [C3] MAP-Elites が大域峰 (>0.9) に届いた seed 率: {reach:.2f}")
    # 合格: MAP-Elites が RR-hillclimb (最強 baseline) に有意勝利
    ok = compare(me, rr, "map_elites", "rr_hillclimb").passes
    print(f"\n  実験1 ③成立 (vs 最強 baseline RR-hillclimb): {'YES' if ok else 'NO'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
