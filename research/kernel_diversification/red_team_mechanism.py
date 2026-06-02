# SPDX-License-Identifier: Apache-2.0
"""Red-team (1) — RR-hillclimb の「なぜ勝つか」機構 evidence 化.

BG9-4 の構造的主張:
  「RR-hillclimb は restart ごとに kernel_id∈[0,4) を直接一様サンプルし target kernel basin に
   必ず当たるから、kid-axis 欺瞞 corridor では RR を排除できない。」

本 module は ``selection_lab.random_restart_hillclimb`` を **改変せず**、同一アルゴリズムを
**instrument 版**として red-team 内に複製 (=src/既存無改変規律を守る) し、positive control 上で:
  - restart 回数 / restart ごとの kernel_id サンプル分布
  - target kid-basin に「restart 直撃」した回数 (= 直接サンプルで谷を回避した証拠)
  - best gene の kernel_id が target basin (kid≈3.6) か局所罠 (kid≈0.5) か
  - best を見つけたのが hill-climb 連続由来か restart 由来か
を実測する。**主張を数値で裏付ける or 覆す**。

instrument 版は selection_lab.random_restart_hillclimb の **論理を逐語的に複製**し、計測 hook
だけ足した (アルゴリズム挙動は同一 = faithful)。end で素の selection_lab 版と best_fitness が
seed 一致するか cross-check して複製忠実性を担保する。

read-only import: bg9_driver (eval/定数), selection_lab (cross-check), kernel_fitness, kernels。
src 無改変、既存 .py 無改変。git 非実行。生数値は JSON 保存。

実行: py -3.11 research/kernel_diversification/red_team_mechanism.py [--seeds N] [--n-evals M]
出力: research/kernel_diversification/red_team_mechanism_results.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import bg9_driver as bg9  # noqa: E402  read-only
from kernel_fitness import kernel_ga_bounds  # noqa: E402
from kernels import GA_DIM, N_KERNELS  # noqa: E402
from selection_lab import random_restart_hillclimb  # noqa: E402  cross-check 用


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    except Exception:  # pragma: no cover
        pass


# target / local basin の kid (bg9_driver の positive control 定数)
TARGET_KID = bg9._POS_GLOB_KID      # 3.6
LOCAL_KID = bg9._POS_LOCAL_KID      # 0.5
TARGET_KID_W = bg9._POS_KID_W       # 0.16
# target basin in-basin 判定: |kid - target| <= 2σ (Gaussian 主要部)
TARGET_BAND = 2.0 * TARGET_KID_W


def _clip(gene: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lo, hi = bounds
    return np.clip(gene, lo, hi)


@dataclass
class RRTrace:
    """1 seed の instrumented RR の機構計測."""
    n_evals: int
    n_restarts: int
    restart_kids: list[float] = field(default_factory=list)   # 各 restart で引いた kernel_id
    init_kid: float = 0.0
    n_restart_hits_target: int = 0   # restart が target basin に直撃した回数
    best_kid: float = 0.0            # best gene の kernel_id
    best_fitness: float = 0.0
    best_in_target: bool = False     # best が target basin に居るか
    best_origin: str = ""            # "init" | "restart" | "hillclimb_from_init" | "hillclimb_from_restart"
    reached_target: bool = False     # best_fitness > POS proxy


def instrumented_rr(
    eval_once,
    *,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    n_evals: int,
    sigma: float,
    restart_patience: int,
    rng: np.random.Generator,
) -> RRTrace:
    """selection_lab.random_restart_hillclimb の論理を逐語複製 + 計測 hook.

    アルゴリズム挙動は素の版と完全一致 (同一 rng 列・同一分岐)。best を更新した瞬間の
    「現在の cur が init 起点か restart 起点か」を追跡し best_origin を判定する。
    """
    tr = RRTrace(n_evals=n_evals, n_restarts=0)

    best_g = bounds[0] + (bounds[1] - bounds[0]) * rng.random(dim)
    best_f = eval_once(best_g, rng)
    evals = 1
    tr.init_kid = float(best_g[0])

    cur_g, cur_f = best_g.copy(), best_f
    stall = 0
    # 現在の cur がどの起点 (init / restart) から来たか
    cur_origin = "init"
    best_origin = "init"
    n_moves_since_origin = 0  # origin から hill-climb で何手動いたか

    while evals < n_evals:
        cand = _clip(cur_g + rng.normal(0, sigma, size=dim), bounds)
        f = eval_once(cand, rng)
        evals += 1
        if f >= cur_f:
            cur_g, cur_f = cand, f
            stall = 0
            n_moves_since_origin += 1
        else:
            stall += 1
        if cur_f > best_f:
            best_g, best_f = cur_g.copy(), cur_f
            # best がどこ由来か: origin が restart なら restart 由来、init なら init/hillclimb 由来
            if cur_origin == "init":
                best_origin = "hillclimb_from_init" if n_moves_since_origin > 0 else "init"
            else:
                best_origin = "hillclimb_from_restart" if n_moves_since_origin > 0 else "restart"
        if stall >= restart_patience and evals < n_evals:
            cur_g = bounds[0] + (bounds[1] - bounds[0]) * rng.random(dim)
            cur_f = eval_once(cur_g, rng)
            evals += 1
            stall = 0
            tr.n_restarts += 1
            kid = float(cur_g[0])
            tr.restart_kids.append(kid)
            if abs(kid - TARGET_KID) <= TARGET_BAND:
                tr.n_restart_hits_target += 1
            cur_origin = "restart"
            n_moves_since_origin = 0
            if cur_f > best_f:
                best_g, best_f = cur_g.copy(), cur_f
                best_origin = "restart"

    tr.best_kid = float(best_g[0])
    tr.best_fitness = float(best_f)
    tr.best_in_target = bool(abs(best_g[0] - TARGET_KID) <= TARGET_BAND)
    tr.best_origin = best_origin
    tr.reached_target = bool(best_f > bg9.POS_GLOBAL_PEAK_PROXY)
    return tr


def run_mechanism(n_seeds: int, n_evals: int, base_seed: int = 20260602) -> dict:
    """positive control 上で instrumented RR を n_seeds 走らせ機構を実測 + 素版と cross-check."""
    t0 = time.time()
    eval_once = bg9.make_kernel_barrier_eval(d=1.0)
    lo, hi = kernel_ga_bounds()
    bounds = (lo, hi)
    sigma = bg9.SIGMA
    restart_patience = max(10, n_evals // 20)

    traces: list[RRTrace] = []
    crosscheck_ok = True
    crosscheck_detail = []
    for s in range(n_seeds):
        # bg9 と同一の evo RNG (method_idx=1 = RR)
        rng_inst = np.random.default_rng(np.random.SeedSequence([base_seed, 1, s]))
        tr = instrumented_rr(
            eval_once, dim=GA_DIM, bounds=bounds, n_evals=n_evals, sigma=sigma,
            restart_patience=restart_patience, rng=rng_inst,
        )
        traces.append(tr)
        # cross-check: 素の selection_lab 版を同一 seed で回し best_fitness 一致を確認
        rng_ref = np.random.default_rng(np.random.SeedSequence([base_seed, 1, s]))
        ref = random_restart_hillclimb(
            eval_once, dim=GA_DIM, bounds=bounds, n_evals=n_evals, sigma=sigma,
            restart_patience=restart_patience, rng=rng_ref,
        )
        match = bool(abs(ref.best_fitness - tr.best_fitness) < 1e-12
                     and abs(float(ref.best_gene[0]) - tr.best_kid) < 1e-12)
        crosscheck_detail.append({
            "seed": s, "instrumented_best": tr.best_fitness,
            "reference_best": float(ref.best_fitness), "match": match,
        })
        if not match:
            crosscheck_ok = False

    wall = time.time() - t0

    # 集計
    all_restart_kids = [k for tr in traces for k in tr.restart_kids]
    n_total_restarts = sum(tr.n_restarts for tr in traces)
    n_total_hits = sum(tr.n_restart_hits_target for tr in traces)
    reach = np.mean([tr.reached_target for tr in traces])
    best_in_target = np.mean([tr.best_in_target for tr in traces])
    origins = {}
    for tr in traces:
        origins[tr.best_origin] = origins.get(tr.best_origin, 0) + 1

    # restart kid のヒストグラム (4 kernel basin 別の落ち先)
    kid_hist = [0, 0, 0, 0]
    for k in all_restart_kids:
        kid_hist[int(np.clip(np.floor(k), 0, N_KERNELS - 1))] += 1

    # 「restart 1 回が target band に当たる経験確率」 — 理論値 = band 幅 / 全幅
    band_width = 2 * TARGET_BAND  # ±2σ
    total_kid_range = float(N_KERNELS) - 1e-9  # [0,4)
    p_hit_theory = band_width / total_kid_range
    p_hit_empirical = (n_total_hits / n_total_restarts) if n_total_restarts else 0.0

    # 機構主張の判定
    # 主張裏付け条件: (a) restart kid がほぼ一様 (4 basin に分散) (b) best が target basin に居る
    # 割合が高い (c) best_origin に restart 由来が多い or hillclimb が in-basin で起きている。
    claim_supported = bool(
        best_in_target >= 0.5            # 過半数の seed で best が target kid basin
        and n_total_hits >= 1            # restart が少なくとも 1 回 target band 直撃
        and reach >= 0.5                 # 過半数で target 到達
    )

    return {
        "meta": {
            "task": "red_team(1) RR mechanism evidence on positive control (d=1.0)",
            "n_seeds": n_seeds, "n_evals": n_evals, "base_seed": base_seed,
            "sigma": sigma, "restart_patience": restart_patience,
            "target_kid": TARGET_KID, "local_kid": LOCAL_KID,
            "target_band_pm2sigma": TARGET_BAND,
            "wall_clock_sec": round(wall, 1),
        },
        "crosscheck": {
            "all_match_selection_lab": crosscheck_ok,
            "detail": crosscheck_detail,
            "note": "instrumented RR が selection_lab.random_restart_hillclimb と best_fitness/kid 一致 = faithful 複製",
        },
        "aggregate": {
            "n_total_restarts": n_total_restarts,
            "mean_restarts_per_seed": round(n_total_restarts / max(n_seeds, 1), 2),
            "n_total_restart_hits_target_band": n_total_hits,
            "p_hit_target_band_per_restart_empirical": round(p_hit_empirical, 4),
            "p_hit_target_band_per_restart_theory_uniform": round(p_hit_theory, 4),
            "restart_kid_basin_histogram": {
                "rwkv(0)": kid_hist[0], "mamba(1)": kid_hist[1],
                "hopfield(2)": kid_hist[2], "linear_attn(3)": kid_hist[3],
            },
            "reach_rate_target": round(float(reach), 3),
            "best_in_target_basin_rate": round(float(best_in_target), 3),
            "best_origin_counts": origins,
        },
        "per_seed": [
            {
                "seed": i, "n_restarts": tr.n_restarts,
                "init_kid": round(tr.init_kid, 3),
                "n_restart_hits_target": tr.n_restart_hits_target,
                "best_kid": round(tr.best_kid, 3),
                "best_fitness": round(tr.best_fitness, 4),
                "best_in_target": tr.best_in_target,
                "best_origin": tr.best_origin,
                "reached_target": tr.reached_target,
                "restart_kids_sample": [round(k, 2) for k in tr.restart_kids[:12]],
            }
            for i, tr in enumerate(traces)
        ],
        "verdict": {
            "claim_supported": claim_supported,
            "claim": ("RR は restart で kernel_id を直接一様サンプルし target basin に当たって谷を回避する"),
            "evidence_summary": (
                f"restart {n_total_restarts} 回中 {n_total_hits} 回が target band(±2σ) 直撃 "
                f"(empirical {p_hit_empirical:.3f} vs uniform 理論 {p_hit_theory:.3f})。"
                f"best が target basin に居た seed = {best_in_target:.0%}、target 到達 = {reach:.0%}。"
                f"restart kid は 4 basin に {kid_hist} で分散 (≈一様サンプル)。"
            ),
        },
    }


def _print(res: dict) -> None:
    print("=" * 80)
    print("RED-TEAM (1) — RR mechanism evidence (positive control d=1.0)")
    print("=" * 80)
    a = res["aggregate"]
    print(f"cross-check (instrumented==selection_lab): {res['crosscheck']['all_match_selection_lab']}")
    print(f"restarts total={a['n_total_restarts']} (mean/seed={a['mean_restarts_per_seed']})")
    print(f"restart hits target band: {a['n_total_restart_hits_target_band']} "
          f"(empirical p_hit={a['p_hit_target_band_per_restart_empirical']} "
          f"vs uniform theory={a['p_hit_target_band_per_restart_theory_uniform']})")
    print(f"restart kid basin histogram: {a['restart_kid_basin_histogram']}")
    print(f"reach target rate={a['reach_rate_target']}  best_in_target_rate={a['best_in_target_basin_rate']}")
    print(f"best origin counts: {a['best_origin_counts']}")
    print(f"\nVERDICT claim_supported={res['verdict']['claim_supported']}")
    print(f"  {res['verdict']['evidence_summary']}")
    print(f"wall={res['meta']['wall_clock_sec']}s")
    print("=" * 80)


def main() -> int:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-evals", type=int, default=800)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    res = run_mechanism(args.seeds, args.n_evals)
    out = Path(args.out) if args.out else Path(__file__).resolve().parent / "red_team_mechanism_results.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    _print(res)
    print(f"written: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
