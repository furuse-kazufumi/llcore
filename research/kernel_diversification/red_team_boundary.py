# SPDX-License-Identifier: Apache-2.0
"""Red-team (3) — ③ が RR を排除できる behavior 次元の閾値を探る (境界の定量化).

主張「kernel 選択は低次元すぎる (kernel_id だけ = RR が直接サンプル可能な 1 座標)」を定量化する。

設計: in-basin の theta-corridor 次元 D を 0,1,2,3 とスイープし、behavior niche 軸も同じ D を
含める (kid + theta 上位 D 座標)。各 D で MAP-E vs RR の strict_gate を測り、
**RR を排除できる (MAP-E が RR に strict 勝利) 最小 D** を探す。

予想される構造的ジレンマ (BG9-4 + red_team(2) で観測):
  - D=0 (kid のみ corridor): RR は kid を直接サンプルし target 即達 → MAP-E 優位なし。
  - D 小: RR は kid 直撃後 in-basin theta を hill-climb で登れる → 依然 RR 強い。
  - D 大: theta corridor が体積で締まり RR の in-basin climb は starve する **が、MAP-E も
    同時に starve する** (niche を ratchet するにも corridor が狭すぎ stepping-stone が立たない)。
  → 「RR だけ落ちて MAP-E は通る」窓が存在するかを D スイープで探す。窓が無ければ
     「kernel 空間では③が RR を排除できる behavior 次元は存在しない」= 構造 N/A 定量確証。

honest: corridor 幅は D に依らず一定 (各軸 _W=0.16)。RR は不当に縛らない (kid 直接サンプルそのまま)。
behavior 次元を増やすのは MAP-E の niche 設計の自由 (pre-reg 逸脱 probe と明記)。

read-only import: bg9_driver の run_methods_crn / strict_gate / 定数。src/既存 .py 無改変。git 非実行。

実行: py -3.11 research/kernel_diversification/red_team_boundary.py [--seeds N] [--n-evals M]
出力: research/kernel_diversification/red_team_boundary_results.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import bg9_driver as bg9  # noqa: E402  read-only
from kernel_fitness import kernel_ga_bounds  # noqa: E402
from kernels import GA_DIM, N_KERNELS  # noqa: E402

EvalOnce = Callable[[np.ndarray, np.random.Generator], float]

TARGET_KID = 3.6
LOCAL_KID = 0.5
KID_W = 0.16
THETA_W = 0.16        # 各 theta corridor 軸の幅 (D に依らず一定 = 公平)
THETA_TARGET = 0.9    # 各 theta corridor 軸の目標 frac
NOISE = 0.008
PROXY = 0.8


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    except Exception:  # pragma: no cover
        pass


def _theta_fracs(gene_vec5: np.ndarray) -> np.ndarray:
    lo, hi = kernel_ga_bounds()
    th_lo, th_hi = lo[1:5], hi[1:5]
    th = np.clip(np.asarray(gene_vec5[1:5], dtype=np.float64), th_lo, th_hi)
    return (th - th_lo) / np.maximum(th_hi - th_lo, 1e-12)


def make_eval_dim(theta_dim: int) -> EvalOnce:
    """in-basin theta corridor の次元 D=theta_dim の deceptive kernel-barrier eval.

    target = kid≈3.6 ∧ theta 上位 D 座標が全て THETA_TARGET。D=0 なら theta 不問 (kid corridor のみ)。
    kid 谷も常に彫る (hill-climb の連続登坂を阻む)。corridor 幅は各軸一定。
    """
    if not (0 <= theta_dim <= 3):
        raise ValueError(theta_dim)

    def ev(gene_vec5: np.ndarray, rng: np.random.Generator) -> float:
        kid = float(np.clip(gene_vec5[0], 0.0, N_KERNELS - 1e-9))
        local = 0.55 * np.exp(-((kid - LOCAL_KID) ** 2) / (2 * 0.70 ** 2))
        kid_g = np.exp(-((kid - TARGET_KID) ** 2) / (2 * KID_W ** 2))
        if theta_dim > 0:
            fr = _theta_fracs(gene_vec5)[:theta_dim]
            d2 = np.sum((fr - THETA_TARGET) ** 2)
            theta_g = np.exp(-d2 / (2 * THETA_W ** 2))
        else:
            theta_g = 1.0
        if LOCAL_KID <= kid <= TARGET_KID:
            dip = np.exp(-((kid - 2.0) ** 2) / (2 * 0.80 ** 2))
            barrier = 1.0 - dip
        else:
            barrier = 1.0
        target = 1.0 * kid_g * theta_g * barrier
        return float(max(local, target) + rng.normal(0, NOISE))

    return ev


def make_behavior_dim(theta_dim: int) -> tuple[Callable, tuple[np.ndarray, np.ndarray], tuple[int, ...]]:
    """behavior = (kid, theta 上位 D frac) の (D+1) 次元 niche。grid = (4, 4^D)."""
    def behavior(gene_vec5: np.ndarray) -> np.ndarray:
        kid = float(np.clip(gene_vec5[0], 0.0, N_KERNELS - 1e-9))
        if theta_dim == 0:
            return np.array([kid], dtype=np.float64)
        fr = _theta_fracs(gene_vec5)[:theta_dim]
        return np.concatenate([[kid], fr]).astype(np.float64)

    lo = np.concatenate([[0.0], np.zeros(theta_dim)])
    hi = np.concatenate([[float(N_KERNELS) - 1e-9], np.ones(theta_dim)])
    grid = tuple([N_KERNELS] + [4] * theta_dim)
    return behavior, (lo, hi), grid


def run_dim(theta_dim: int, n_evals: int, n_seeds: int, base_seed: int) -> dict:
    eval_once = make_eval_dim(theta_dim)
    behavior, bb, grid = make_behavior_dim(theta_dim)
    lo, hi = kernel_ga_bounds()
    bounds = (lo, hi)
    res = bg9.run_methods_crn(
        eval_once, behavior, dim=GA_DIM, bounds=bounds, behavior_bounds=bb,
        grid_shape=grid, n_evals=n_evals, n_seeds=n_seeds, honest_n_trials=20,
        sigma=bg9.SIGMA, base_seed=base_seed,
    )
    means = {m: float(res[m].mean()) for m in bg9._BASE_METHODS}
    reach = {m: float(np.mean(res[m] > PROXY)) for m in bg9._BASE_METHODS}
    gates = {}
    beaten = 0
    for b in ("rr_hillclimb", "panmictic_ga", "random"):
        g = bg9.strict_gate(res["map_elites"], res[b], "map_elites", b, min_seeds=3)
        gates[b] = {
            "diff": round(g.diff, 4), "wilcoxon_p": round(g.wilcoxon_p, 4),
            "paired_sign_delta": round(g.paired_sign_delta, 3), "passes": g.passes,
        }
        beaten += int(g.passes)
    return {
        "behavior_dim": theta_dim + 1,   # kid + D theta 軸
        "theta_corridor_dim": theta_dim,
        "means": {k: round(v, 4) for k, v in means.items()},
        "reach_rate": {k: round(v, 3) for k, v in reach.items()},
        "gates": gates,
        "n_baselines_beaten": beaten,
        "beats_rr": gates["rr_hillclimb"]["passes"],
        "map_elites_beats_all_3": beaten == 3,
        "raw_scores": {m: res[m].tolist() for m in bg9._BASE_METHODS},
    }


def run_boundary(n_seeds: int, n_evals: int, base_seed: int = 20260602) -> dict:
    t0 = time.time()
    dims = [0, 1, 2, 3]
    rows = {}
    min_dim_beats_rr = None
    min_dim_all3 = None
    for d in dims:
        r = run_dim(d, n_evals, n_seeds, base_seed)
        rows[f"theta_dim_{d}"] = r
        if r["beats_rr"] and min_dim_beats_rr is None:
            min_dim_beats_rr = r["behavior_dim"]
        if r["map_elites_beats_all_3"] and min_dim_all3 is None:
            min_dim_all3 = r["behavior_dim"]
    wall = time.time() - t0
    return {
        "meta": {
            "task": "red_team(3) ③ が RR を排除できる最小 behavior 次元の境界探索",
            "n_seeds": n_seeds, "n_evals": n_evals, "base_seed": base_seed,
            "theta_corridor_width": THETA_W, "kid_width": KID_W,
            "sigma": bg9.SIGMA, "wall_clock_sec": round(wall, 1),
            "honest_note": (
                "corridor 幅は D に依らず一定 (各軸 0.16)。RR は faithful (kid 直接サンプルそのまま)。"
                "behavior 次元増は MAP-E niche 設計の自由 (pre-reg 逸脱 probe)。"
            ),
        },
        "dims": rows,
        "verdict": {
            "min_behavior_dim_beats_rr": min_dim_beats_rr,
            "min_behavior_dim_beats_all3": min_dim_all3,
            "summary": (
                (f"③ が RR を strict 排除できる最小 behavior 次元 = {min_dim_beats_rr}。"
                 if min_dim_beats_rr is not None else
                 "D=0..3 のどの behavior 次元でも③は RR を strict 排除できなかった。")
                + (f" 3 baseline 全勝の最小次元 = {min_dim_all3}。"
                   if min_dim_all3 is not None else
                   " 3 baseline 全勝の窓も存在しない。corridor を締めると MAP-E も RR と同時に starve するため、"
                   "RR だけ落ちて③が通る behavior 次元は kernel 空間に存在しない (構造 N/A 定量確証)。")
            ),
        },
    }


def _print(res: dict) -> None:
    print("=" * 84)
    print("RED-TEAM (3) — boundary: min behavior dim where ③ beats RR")
    print("=" * 84)
    for key, r in res["dims"].items():
        print(f"\n[{key}] behavior_dim={r['behavior_dim']} (kid + {r['theta_corridor_dim']} theta) "
              f"beaten={r['n_baselines_beaten']}/3 beats_rr={r['beats_rr']} ALL3={r['map_elites_beats_all_3']}")
        print(f"  means: " + " ".join(f"{m}={r['means'][m]:.3f}" for m in bg9._BASE_METHODS))
        print(f"  reach: " + " ".join(f"{m}={r['reach_rate'][m]:.2f}" for m in bg9._BASE_METHODS))
        g = r["gates"]["rr_hillclimb"]
        print(f"    MAP-E vs RR: diff={g['diff']:+.4f} p={g['wilcoxon_p']:.3g} "
              f"δ={g['paired_sign_delta']:+.2f} pass={g['passes']}")
    v = res["verdict"]
    print("\n" + "=" * 84)
    print(f"  min_behavior_dim_beats_rr = {v['min_behavior_dim_beats_rr']}")
    print(f"  min_behavior_dim_beats_all3 = {v['min_behavior_dim_beats_all3']}")
    print(f"  {v['summary']}")
    print(f"  wall = {res['meta']['wall_clock_sec']}s")
    print("=" * 84)


def main() -> int:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=12)
    ap.add_argument("--n-evals", type=int, default=1000)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    res = run_boundary(args.seeds, args.n_evals)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "red_team_boundary_results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    _print(res)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
