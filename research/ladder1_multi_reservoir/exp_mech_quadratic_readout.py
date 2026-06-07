# SPDX-License-Identifier: Apache-2.0
"""機構 quadratic_readout の quick 測定 — parity の床が外れるか held-out R² で測る.

梯子段1 の問いに対し、本 quick 実験は:

- baseline  : 単層 LeakyDelayLineReservoir n_taps=8 + **線形** ridge readout
              (= Step C と同条件の床。R²≈0.016 を再現するはず)
- mechanism : 単層 reservoir + **明示的2次** ridge readout (quadratic_readout)
              n_taps を 4/8/12/16 で振り、構成別 max R² も取る

を DelayedParityTask(seq_len=20, window=5) で測る。

探索:
- random search: n_random=300, n_seeds=8 (固定 seed)。各 gene を同一 train/eval データで
  評価して max → 構成間 paired 公平 (誤帰属の回避)。
- evolved_search: random でなく **(1+λ) ES** (Gaussian 変異 hill-climb)。mechanism
  (n_taps=12) について、同じ評価 budget で random より良い天井に届くかを見る。

honest 対照 (最重要 — 床が外れても reservoir の功績ではない):
- quadratic_readout は **reservoir のダイナミクスを一切変えず** readout 側に二次特徴を注入する。
  よって floor_lifted=true でも、解いたのは readout の二次多項式であり reservoir ではない。
  attribution は 'readout'。

held-out 厳守: train/eval は別 draw (eval は train rng の続きから引く)。floor_lifted は
held-out max R² > 0.5 を閾値とする (parity chance=0, 完全解=1)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # mech_quadratic_readout
sys.path.insert(0, str(_HERE.parent / "step_c_memory_tasks"))  # reservoir, memory_tasks

from mech_quadratic_readout import (  # noqa: E402
    QuadraticReadoutReservoir,
    gene_bounds as quad_gene_bounds,
    make_eval_once as quad_make_eval_once,
)
from reservoir import (  # noqa: E402
    LeakyDelayLineReservoir,
    gene_bounds as lin_gene_bounds,
    make_eval_once as lin_make_eval_once,
)
from memory_tasks import DelayedParityTask  # noqa: E402

# --- 測定パラメータ ---
N_RANDOM = 300       # random search 1 seed あたりの gene 本数
N_SEEDS = 8          # 固定 seed 数
N_TRAIN = 48         # held-out 評価の train 本数
N_EVAL = 48          # held-out 評価の eval 本数 (train rng の続きから draw)
GENE_BASE = 730_001  # gene draw seed の base
EVAL_BASE = 930_001  # eval データ seed の base (全 gene 同一 → gene 間公平)

# ES (evolved_search) パラメータ — random と同じ評価 budget (N_RANDOM 評価)
ES_LAMBDA = 6        # 1 世代あたり子個体数
ES_SIGMA0 = 0.6      # 初期変異標準偏差 (gene 値域 [-4,4]/[-2,2] に対し中庸)
ES_GENERATIONS = N_RANDOM // ES_LAMBDA  # 評価回数を random と概ね揃える


def _random_search_ceiling(res, make_eval_once, n_random: int, seed_idx: int) -> float:
    """1 seed の random search で到達した max held-out R².

    全 gene を同一 eval データ (default_rng(EVAL_BASE+seed_idx)) で評価 → gene 間公平。
    """
    eval_once = make_eval_once(res, _TASK, n_train=N_TRAIN, n_eval=N_EVAL)
    gene_rng = np.random.default_rng(GENE_BASE + seed_idx)
    best = 0.0
    for _ in range(n_random):
        gene = res.random_gene(gene_rng)
        eval_rng = np.random.default_rng(EVAL_BASE + seed_idx)  # 全 gene 同一 train/eval
        best = max(best, eval_once(gene, eval_rng))
    return best


def _es_ceiling(res, make_eval_once, lo, hi, seed_idx: int) -> float:
    """1 seed の (1+λ) ES で到達した max held-out R² (evolved_search).

    各 gene は同一 eval データ (default_rng(EVAL_BASE+seed_idx)) で評価 → random と公平。
    変異は bounds で clip。sigma は世代ごとに緩く減衰させ収束を助ける。
    """
    eval_once = make_eval_once(res, _TASK, n_train=N_TRAIN, n_eval=N_EVAL)
    gene_rng = np.random.default_rng(GENE_BASE + 5_000 + seed_idx)

    def _fitness(g: np.ndarray) -> float:
        eval_rng = np.random.default_rng(EVAL_BASE + seed_idx)
        return eval_once(g, eval_rng)

    parent = res.random_gene(gene_rng)
    best_fit = _fitness(parent)
    sigma = ES_SIGMA0
    span = hi - lo
    for gen in range(ES_GENERATIONS):
        improved = False
        for _ in range(ES_LAMBDA):
            child = parent + sigma * span * gene_rng.standard_normal(parent.shape)
            child = np.clip(child, lo, hi)
            f = _fitness(child)
            if f > best_fit:
                best_fit, parent, improved = f, child, True
        # 改善が無ければ sigma を絞り、あれば僅かに広げる (探索/活用バランス)
        sigma = max(sigma * (0.95 if not improved else 1.02), 0.05)
    return best_fit


def _measure(label, res, make_eval_once, n_random: int) -> np.ndarray:
    t0 = time.time()
    vals = np.array([
        _random_search_ceiling(res, make_eval_once, n_random, s) for s in range(N_SEEDS)
    ])
    dt = time.time() - t0
    print(f"[{label:22s}] gene_dim={res.gene_dim:3d} "
          f"max R² mean={vals.mean():.4f} std={vals.std():.4f} "
          f"(min={vals.min():.3f} max={vals.max():.3f})  {dt:.1f}s", flush=True)
    return vals


_TASK = DelayedParityTask(seq_len=20, window=5, in_dim=1)


def main() -> None:
    print("=== 機構 quadratic_readout quick 測定 (delayed_parity 5-bit XOR) ===")
    print(f"task=delayed_parity(seq_len=20,window=5)  n_random={N_RANDOM} "
          f"n_seeds={N_SEEDS} n_train={N_TRAIN} n_eval={N_EVAL}\n", flush=True)

    # --- baseline: 単層 n_taps=8 + 線形 ridge (Step C と同条件の床) ---
    base_res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    base_vals = _measure("baseline 1L-8 linear", base_res, lin_make_eval_once, N_RANDOM)
    baseline_max = float(base_vals.max())

    # --- mechanism: 二次 readout, n_taps を振る (構成別 R²) ---
    print("\n--- mechanism quadratic_readout (n_taps 振り) ---", flush=True)
    config_curve: dict[int, np.ndarray] = {}
    for n_taps in (4, 8, 12, 16):
        res = QuadraticReadoutReservoir(n_taps=n_taps, in_dim=1)
        vals = _measure(f"quad n_taps={n_taps}", res, quad_make_eval_once, N_RANDOM)
        config_curve[n_taps] = vals

    # 機構の best 構成 (定義 n_taps=12 を主とするが、全構成の max も報告)
    mech_main = config_curve[12]
    mech_main_max = float(mech_main.max())
    mech_best_max = float(max(v.max() for v in config_curve.values()))

    # --- evolved_search (ES) on mechanism n_taps=12 ---
    print("\n--- evolved_search ((1+λ) ES) on quad n_taps=12 ---", flush=True)
    lo, hi = quad_gene_bounds(QuadraticReadoutReservoir(n_taps=12, in_dim=1))
    res12 = QuadraticReadoutReservoir(n_taps=12, in_dim=1)
    t0 = time.time()
    es_vals = np.array([
        _es_ceiling(res12, quad_make_eval_once, lo, hi, s) for s in range(N_SEEDS)
    ])
    dt = time.time() - t0
    es_max = float(es_vals.max())
    print(f"[quad n_taps=12 ES     ] λ={ES_LAMBDA} gen={ES_GENERATIONS} "
          f"max R² mean={es_vals.mean():.4f} std={es_vals.std():.4f} "
          f"(min={es_vals.min():.3f} max={es_vals.max():.3f})  {dt:.1f}s", flush=True)

    # --- 構成別 max R² 要約 ---
    print("\n=== 構成別 max R² 要約 ===", flush=True)
    print(f"baseline 1L-8 linear      : max={baseline_max:.4f} mean={base_vals.mean():.4f}")
    for n_taps in (4, 8, 12, 16):
        v = config_curve[n_taps]
        print(f"quad n_taps={n_taps:<2d} (random)   : max={float(v.max()):.4f} mean={v.mean():.4f}")
    print(f"quad n_taps=12 (ES)       : max={es_max:.4f} mean={es_vals.mean():.4f}")

    # mechanism の代表値 = random と ES の良い方 (n_taps=12 主, 全構成 best も併記)
    mechanism_max = max(mech_main_max, es_max)

    # --- floor 判定 (held-out max R² > 0.5) ---
    floor_lifted = mechanism_max > 0.5
    print("\n=== floor 判定 ===", flush=True)
    print(f"baseline (1L-8 linear) max R²  = {baseline_max:.4f}  "
          f"{'← Step C の床 R²≈0.016 を再現' if baseline_max < 0.1 else '← 床が baseline でも外れている (要検討)'}")
    print(f"mechanism (quad) max R² (best) = {mechanism_max:.4f}  "
          f"[n_taps=12 random={mech_main_max:.4f}, ES={es_max:.4f}, 全構成 best={mech_best_max:.4f}]")
    print(f"floor_lifted (>0.5) = {floor_lifted}")

    # --- honest 要約 ---
    print("\n=== honest 要約 ===", flush=True)
    print("  attribution = 'readout' (床が外れても reservoir でなく明示的2次 readout の功績)")
    print("  quadratic_readout は reservoir ダイナミクスを変えず readout に二次特徴を注入する対照。")
    print("  baseline と mechanism は同一 reservoir 基質・同一 gene 探索・同一 held-out 評価で、")
    print("  差分は『readout が線形か明示的2次か』の 1 点のみ → 効果を readout に正しく帰属できる。")

    # --- machine-readable summary line ---
    print("\nSUMMARY_JSON_BEGIN", flush=True)
    import json
    summary = {
        "task": "delayed_parity(seq_len=20,window=5)",
        "n_random": N_RANDOM, "n_seeds": N_SEEDS,
        "baseline_1L8_linear_max_r2": baseline_max,
        "baseline_1L8_linear_mean_r2": float(base_vals.mean()),
        "mechanism_quad_n12_random_max_r2": mech_main_max,
        "mechanism_quad_n12_es_max_r2": es_max,
        "mechanism_best_max_r2": mechanism_max,
        "config_curve_max": {str(k): float(v.max()) for k, v in config_curve.items()},
        "config_curve_mean": {str(k): float(v.mean()) for k, v in config_curve.items()},
        "floor_lifted": bool(floor_lifted),
        "attribution": "readout",
    }
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("SUMMARY_JSON_END", flush=True)


if __name__ == "__main__":
    main()
