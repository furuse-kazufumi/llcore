# SPDX-License-Identifier: Apache-2.0
"""梯子段1 機構測定: hybrid_max が delayed_parity の床を外すか.

Step C verdict: 単一 leaky reservoir + 線形 ridge は delayed_parity (5-bit XOR) を解けない
(基質の床, 全 method held-out R²≈0.016)。本実験は最大表現力 anchor 機構 hybrid_max
(並列 2 枝 × 深さ × 乗法ゲート × 2 次 readout 特徴) の **到達天井 (max held-out R²)** を測り、
同条件の単層 n_taps=8 baseline と比較して床が外れるかを判定する。

公平性の肝 (誤帰属・データリークの回避):
- baseline = step_c の LeakyDelayLineReservoir(n_taps=8) を **同じ task / 同じ held-out
  プロトコル / 同じ ridge_lambda** で評価 (Step C の床の再現)。
- 各 seed で全 gene を **同一 train/eval データ** (eval_rng seed 固定) で採点 → gene 間公平。
- held-out: train 系列で readout を fit → 別 draw の eval 系列で R² を採点。train への fit の
  良さは fitness に一切寄与しない (overfit を構造的に排除)。
- baseline / 機構の random search は同一 budget (n_random) / 同一 seed 集合で揃える。

探索:
- baseline と 機構の各「構成 (config)」は **random search** (n_random=300) で天井を測る (公平な
  構成間比較)。
- 機構 best は加えて **ES (evolved_search)** で測る。ES は (μ,λ) Gaussian mutation の自己適応で、
  random と同じ評価予算 (=μ初期 + n_gen×λ ≈ n_random) 内で天井を伸ばせるかを見る。

判定:
- floor_lifted = held-out max R² > 0.5 (parity chance=0, 完全解=1)。0.5 は「parity を実質
  解いている」閾値。
- 構成別 R² (ablation) で、どの機構が天井を押し上げたか (深さ / 乗法 / 2次) を切り分ける。

各 py 実行は CPU 競合で遅い可能性があるが完走させる。結果は stdout + JSON 保存。
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(_HERE))  # mech_hybrid_max
sys.path.insert(0, str(_HERE.parent / "step_c_memory_tasks"))  # memory_tasks, reservoir

from mech_hybrid_max import (
    HybridMaxReservoir,
    eval_on_dataset,
    gene_bounds as mech_bounds,
    make_batched_dataset,
)
from mech_hybrid_max import _sigmoid as _hm_sigmoid
from reservoir import LeakyDelayLineReservoir
from memory_tasks import DelayedParityTask

# src/llcore への相対パス (baseline の batched ridge 評価で fit_ridge_readout を流用)。
sys.path.insert(0, str(_HERE.parent.parent / "src"))
from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402

# ---- 測定設定 (固定 seed で再現可能) ----
N_RANDOM = 300       # random search 1 seed あたりの gene 本数 (到達天井の推定)
N_SEEDS = 8          # 固定 seed 数 (タスク指定)
N_TRAIN = 48         # held-out: train 系列本数
N_EVAL = 48          # held-out: eval 系列本数 (別 draw)
RIDGE_LAMBDA = 1e-2  # baseline / 機構で共通 (公平)
GENE_BASE = 4_200_001
EVAL_BASE = 8_400_001
FLOOR_THRESHOLD = 0.5  # held-out max R² > 0.5 で parity を実質解いたと判定
OUT_JSON = _HERE / "exp_mech_hybrid_max_results.json"

# ES (evolved_search) ハイパラ — random と同等の評価予算に揃える。
ES_MU = 12           # 親個体数
ES_LAMBDA = 24       # 子個体数 / 世代
ES_GENERATIONS = 12  # 世代数 → 評価数 ≈ MU + GEN*LAMBDA = 12 + 288 = 300 (random と一致)
ES_SIGMA0 = 0.6      # 初期変異強度 (gene 空間 leak∈[-4,4]/w∈[-2,2] に対し中庸)
ES_SIGMA_DECAY = 0.92  # 世代ごとの sigma 減衰 (収束のための annealing)


def random_search_ceiling(eval_once, random_gene, n_random: int, seed_idx: int) -> float:
    """1 seed の random search で到達した max held-out R² を返す.

    全 gene を同一 eval データ (default_rng(EVAL_BASE+seed_idx)) で採点 → gene 間公平。
    """
    gene_rng = np.random.default_rng(GENE_BASE + seed_idx)
    best = 0.0
    for _ in range(n_random):
        gene = random_gene(gene_rng)
        eval_rng = np.random.default_rng(EVAL_BASE + seed_idx)  # 全 gene 同一 train/eval
        best = max(best, eval_once(gene, eval_rng))
    return best


def es_search_ceiling(res: HybridMaxReservoir, eval_once, lo, hi, seed_idx: int) -> float:
    """1 seed の (μ,λ) ES で到達した max held-out R² を返す (evolved_search).

    - 初期 μ 個体を bounds 内一様乱数で生成。
    - 各世代: 親をランダム選択 → Gaussian mutation (sigma) → bounds clip → 評価。
      上位 μ を次世代親に。sigma は世代ごとに減衰 (annealing)。
    - 全個体を **同一 eval データ** (EVAL_BASE+seed_idx) で採点 → 個体間公平 + random と
      同一プロトコル。評価数 ≈ random search と一致 (公平比較)。

    fitness landscape は held-out R² (clip[0,1])。ES は random と違い良解近傍を集中探索する。
    """
    gene_rng = np.random.default_rng(GENE_BASE + 100 + seed_idx)
    span = hi - lo

    def evaluate(gene: np.ndarray) -> float:
        eval_rng = np.random.default_rng(EVAL_BASE + seed_idx)  # 全個体同一 train/eval
        return eval_once(np.clip(gene, lo, hi), eval_rng)

    # --- 初期親集団 ---
    parents = [lo + span * gene_rng.random(res.gene_dim) for _ in range(ES_MU)]
    parent_fit = [evaluate(g) for g in parents]
    best = max(parent_fit) if parent_fit else 0.0

    sigma = ES_SIGMA0
    for _ in range(ES_GENERATIONS):
        children: list[np.ndarray] = []
        child_fit: list[float] = []
        for _ in range(ES_LAMBDA):
            # 親をランダムに 1 体選び、span スケールの Gaussian mutation を加える。
            p = parents[int(gene_rng.integers(0, ES_MU))]
            child = p + sigma * span * gene_rng.standard_normal(res.gene_dim)
            child = np.clip(child, lo, hi)
            f = evaluate(child)
            children.append(child)
            child_fit.append(f)
            best = max(best, f)
        # μ+λ 選択: 親 + 子の合併から上位 μ を次世代へ (エリート保存)。
        pool = list(zip(parent_fit + child_fit, parents + children))
        pool.sort(key=lambda kv: kv[0], reverse=True)
        parents = [g for _, g in pool[:ES_MU]]
        parent_fit = [f for f, _ in pool[:ES_MU]]
        sigma *= ES_SIGMA_DECAY
    return best


def summarize(vals: np.ndarray) -> dict:
    """per-seed max R² 配列の要約統計."""
    return {
        "per_seed_max_r2": vals.tolist(),
        "mean": float(vals.mean()),
        "std": float(vals.std()),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }


def main() -> None:
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    seeds = list(range(N_SEEDS))
    results: dict = {
        "task": "delayed_parity(seq_len=20,window=5,in_dim=1)",
        "protocol": {
            "n_random": N_RANDOM, "n_seeds": N_SEEDS,
            "n_train": N_TRAIN, "n_eval": N_EVAL,
            "ridge_lambda": RIDGE_LAMBDA, "floor_threshold": FLOOR_THRESHOLD,
            "es": {"mu": ES_MU, "lambda": ES_LAMBDA, "generations": ES_GENERATIONS,
                   "sigma0": ES_SIGMA0, "sigma_decay": ES_SIGMA_DECAY},
        },
        "configs": {},
    }

    # ===== 1) baseline: 単層 n_taps=8 (random search, Step C 床の再現) =====
    print("=" * 72)
    print("梯子段1 機構測定: hybrid_max — delayed_parity の床は外れるか")
    print("=" * 72)
    print(f"task=delayed_parity(seq_len=20,window=5)  n_random={N_RANDOM} n_seeds={N_SEEDS} "
          f"n_train={N_TRAIN} n_eval={N_EVAL} ridge_lambda={RIDGE_LAMBDA}")
    print(f"floor_lifted 閾値: held-out max R² > {FLOOR_THRESHOLD}\n")

    base_res = LeakyDelayLineReservoir(n_taps=8, in_dim=task.in_dim)
    base_eval = single_eval_once(base_res, task, n_train=N_TRAIN, n_eval=N_EVAL,
                                 ridge_lambda=RIDGE_LAMBDA)
    t0 = time.time()
    base_vals = np.array([
        random_search_ceiling(base_eval, base_res.random_gene, N_RANDOM, s) for s in seeds
    ])
    dt = time.time() - t0
    baseline_max_r2 = float(base_vals.max())
    results["baseline_1L8"] = {**summarize(base_vals), "gene_dim": base_res.gene_dim,
                               "search": "random", "wall_s": round(dt, 1)}
    print(f"[baseline 1L-8 random] gene_dim={base_res.gene_dim:3d} "
          f"max R² mean={base_vals.mean():.4f} (per-seed max best={baseline_max_r2:.4f}) "
          f"{dt:.1f}s")
    print(f"  {'← Step C の床 R²≈0 を再現' if baseline_max_r2 < 0.2 else '← 床が baseline でも外れている (要検討)'}\n")

    # ===== 2) 機構 hybrid_max: 構成別 random search (ablation) =====
    # gate / quadratic / 深さ の寄与を切り分けるための ablation 構成群。
    # (label, layer_taps, use_gate, use_quadratic)
    mech_configs = [
        ("HM-depthonly-2x8",   (8, 8), False, False),  # 並列のみ (乗法/2次なし)
        ("HM-gate-2x8",        (8, 8), True,  False),  # 並列 + 乗法ゲート
        ("HM-quad-2x8",        (8, 8), False, True),   # 並列 + 2次特徴 (ゲートなし)
        ("HM-full-2x8",        (8, 8), True,  True),   # 全部入り (anchor 本体)
        ("HM-full-1x8",        (8,),   True,  True),   # 浅い全部入り (深さ寄与の対照)
    ]
    config_curve_bits: list[str] = []
    mech_random_best = 0.0
    for label, taps, use_gate, use_quad in mech_configs:
        res = HybridMaxReservoir(layer_taps=taps, in_dim=task.in_dim,
                                 use_gate=use_gate, use_quadratic=use_quad)
        eval_once = mech_eval_once(res, task, n_train=N_TRAIN, n_eval=N_EVAL,
                                   ridge_lambda=RIDGE_LAMBDA)
        t0 = time.time()
        vals = np.array([
            random_search_ceiling(eval_once, res.random_gene, N_RANDOM, s) for s in seeds
        ])
        dt = time.time() - t0
        cfg_max = float(vals.max())
        mech_random_best = max(mech_random_best, cfg_max)
        results["configs"][label] = {
            **summarize(vals),
            "layer_taps": list(taps), "use_gate": use_gate, "use_quadratic": use_quad,
            "gene_dim": res.gene_dim, "feature_dim": res.feature_dim,
            "search": "random", "wall_s": round(dt, 1),
        }
        config_curve_bits.append(f"{label}={cfg_max:.3f}")
        print(f"[{label:18s}] feat_dim={res.feature_dim:3d} gene_dim={res.gene_dim:3d} "
              f"max R² mean={vals.mean():.4f} (best={cfg_max:.4f}) {dt:.1f}s")

    # ===== 3) 機構 best (full) を ES (evolved_search) で測る =====
    print(f"\n[evolved_search] full anchor を ES (μ={ES_MU},λ={ES_LAMBDA},gen={ES_GENERATIONS}) で測定…")
    from mech_hybrid_max import gene_bounds as mech_bounds
    full_res = HybridMaxReservoir(layer_taps=(8, 8), in_dim=task.in_dim,
                                  use_gate=True, use_quadratic=True)
    full_eval = mech_eval_once(full_res, task, n_train=N_TRAIN, n_eval=N_EVAL,
                               ridge_lambda=RIDGE_LAMBDA)
    lo, hi = mech_bounds(full_res)
    t0 = time.time()
    es_vals = np.array([
        es_search_ceiling(full_res, full_eval, lo, hi, s) for s in seeds
    ])
    dt = time.time() - t0
    es_best = float(es_vals.max())
    results["configs"]["HM-full-2x8-ES"] = {
        **summarize(es_vals),
        "layer_taps": [8, 8], "use_gate": True, "use_quadratic": True,
        "gene_dim": full_res.gene_dim, "feature_dim": full_res.feature_dim,
        "search": "evolved_search(ES)", "wall_s": round(dt, 1),
    }
    config_curve_bits.append(f"HM-full-ES={es_best:.3f}")
    print(f"[HM-full-2x8-ES    ] feat_dim={full_res.feature_dim:3d} "
          f"max R² mean={es_vals.mean():.4f} (best={es_best:.4f}) {dt:.1f}s")

    # ===== 4) 集計・判定 =====
    mechanism_max_r2 = max(mech_random_best, es_best)
    floor_lifted = bool(mechanism_max_r2 > FLOOR_THRESHOLD)
    baseline_lifted = bool(baseline_max_r2 > FLOOR_THRESHOLD)

    results["summary"] = {
        "baseline_1L8_max_r2": baseline_max_r2,
        "mechanism_max_r2": mechanism_max_r2,
        "mechanism_random_best": mech_random_best,
        "mechanism_es_best": es_best,
        "floor_lifted": floor_lifted,
        "baseline_floor_lifted": baseline_lifted,
        "config_curve": "; ".join(config_curve_bits),
    }

    print("\n" + "=" * 72)
    print("=== 結果サマリ (held-out max R²) ===")
    print(f"  baseline_1L8_max_r2 = {baseline_max_r2:.4f}")
    print(f"  mechanism_max_r2    = {mechanism_max_r2:.4f}  "
          f"(random best={mech_random_best:.4f} / ES best={es_best:.4f})")
    print(f"  構成別 R²: {'; '.join(config_curve_bits)}")
    print(f"  floor_lifted (>{FLOOR_THRESHOLD})  : {floor_lifted}")
    print(f"  baseline も床外れ? : {baseline_lifted}")

    # ===== 5) attribution (機構の性質に照らした正直な帰属) =====
    cfg = results["configs"]
    depth_best = cfg["HM-depthonly-2x8"]["max"]
    gate_best = cfg["HM-gate-2x8"]["max"]
    quad_best = cfg["HM-quad-2x8"]["max"]
    full_best = cfg["HM-full-2x8"]["max"]
    shallow_full_best = cfg["HM-full-1x8"]["max"]

    print("\n=== attribution (構成別 ablation) ===")
    print(f"  深さのみ(並列)     HM-depthonly = {depth_best:.4f}")
    print(f"  +乗法ゲート        HM-gate      = {gate_best:.4f}  (gate寄与 Δ={gate_best-depth_best:+.4f})")
    print(f"  +2次特徴(ゲート無)  HM-quad      = {quad_best:.4f}  (quad寄与 Δ={quad_best-depth_best:+.4f})")
    print(f"  全部入り           HM-full(2x8) = {full_best:.4f}")
    print(f"  浅い全部入り       HM-full(1x8) = {shallow_full_best:.4f}  (深さ寄与 Δ={full_best-shallow_full_best:+.4f})")

    # 主因の機械的推定 (honest): 床外れに最も寄与した機構を Δ から推定。
    if not floor_lifted:
        attribution = "reservoir_expressivity"  # 機構を全部入れても床が外れない = 表現力不足
        attr_note = "最大構成でも held-out で parity を解けず。CPU reservoir+ridge パラダイムの表現力床の証拠。"
    else:
        d_gate = gate_best - depth_best
        d_quad = quad_best - depth_best
        d_depth = full_best - shallow_full_best
        # 2次特徴だけで解けるなら readout、乗法ゲートが主因なら reservoir 側の相互作用、
        # 深さが主因なら reservoir 表現力。最大 Δ を主因とする。
        deltas = {"readout": d_quad, "width": d_gate, "reservoir_expressivity": d_depth}
        attribution = max(deltas, key=deltas.get)
        attr_note = (f"床外れの主因 (最大 Δ): {attribution}. "
                     f"gate Δ={d_gate:+.4f}, quad Δ={d_quad:+.4f}, depth Δ={d_depth:+.4f}.")
    results["summary"]["attribution"] = attribution
    results["summary"]["attribution_note"] = attr_note
    print(f"\n  attribution = {attribution}")
    print(f"  {attr_note}")

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n結果を保存: {OUT_JSON}")


if __name__ == "__main__":
    main()
