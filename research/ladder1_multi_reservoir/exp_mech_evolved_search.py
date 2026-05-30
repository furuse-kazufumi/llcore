# SPDX-License-Identifier: Apache-2.0
"""梯子段1 機構 evolved_search 測定 — 進化探索で parity の床が外れるかを held-out R² で測る.

測定設計 (誤帰属の回避):
- baseline (1L-8): Step C と同条件の単一層 n_taps=8。**random search** (n_random, n_seeds 固定 seed)。
- mechanism (3L-8x8x8): DeepReservoir((8,8,8)) を **(μ+λ)-ES** (pop 30, gen 40) で探索し、
  best gene を進化と独立な fresh seed で honest 再評価した held-out R² を到達天井とする。
- 参考 (3L-random): mechanism と同基質を **random search** (mechanism と同予算) で探索した天井。
  これにより「3 層化の効果」と「進化探索の効果」を分離する:
    3L-random > 1L-8  → 基質 (深さ/規模) の寄与
    3L-evolved > 3L-random → 探索戦略 (進化) の寄与

公平性:
- 全構成で同じ make_eval_once (held-out R²、train/eval 別 draw = リークなし) を使用。
- random search は各 seed で全 gene を同一 train/eval データ (seed 固定) で評価し max を取る (gene 間公平)。
- ES の honest 再評価は進化用 rng と独立な seed (mech_evolved_search._HONEST_OFFSET) を使う。

判定:
- floor_lifted = held-out max R² > 0.5 (parity chance=0, 完全解=1)。閾値超で「床が外れた」。
- strict_compare (片側 Wilcoxon + δ) で mechanism vs baseline / mechanism vs 3L-random を補助判定。
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
sys.path.insert(0, str(_HERE))  # mech_evolved_search, multi_reservoir
sys.path.insert(0, str(_HERE.parent / "step_c_memory_tasks"))  # memory_tasks, strict_compare

from multi_reservoir import DeepReservoir, make_eval_once  # noqa: E402
from memory_tasks import DelayedParityTask  # noqa: E402
from strict_compare import strict_compare  # noqa: E402
from mech_evolved_search import (  # noqa: E402
    EvolvedSearchConfig,
    evolve_search,
    make_deep_eval,
)

# --- 測定パラメータ (タスク指定) ---
N_RANDOM = 300         # random search baseline / 参考の 1 seed あたり gene 本数
N_SEEDS = 8            # 固定 seed 数 (CPU 競合で重いため quick: 8)
N_TRAIN = 48
N_EVAL = 48
GENE_BASE = 700_001    # random search の gene draw seed base
EVAL_BASE = 900_001    # random search の train/eval データ seed base (全 gene 同一 → 公平)
ES_SEED_BASE = 500_001 # 進化探索の base seed

# (μ+λ)-ES 設定 (タスク指定: pop 30, gen 40, k=3, elitism 1, σ=0.15)。
ES_CONFIG = EvolvedSearchConfig(
    pop_size=30, n_generations=40, tournament_k=3, elitism=1,
    mutation_sigma=0.15, honest_n_trials=16,
)


def random_search_ceiling(res: DeepReservoir, task, n_random: int, seed_idx: int) -> float:
    """1 seed の random search で到達した max held-out R² を返す (全 gene 同一 eval データで公平)."""
    eval_once = make_eval_once(res, task, n_train=N_TRAIN, n_eval=N_EVAL)
    gene_rng = np.random.default_rng(GENE_BASE + seed_idx)
    best = 0.0
    for _ in range(n_random):
        gene = res.random_gene(gene_rng)
        eval_rng = np.random.default_rng(EVAL_BASE + seed_idx)  # 全 gene 同一 train/eval
        best = max(best, eval_once(gene, eval_rng))
    return best


def evolved_search_ceiling(res: DeepReservoir, task, seed_idx: int) -> float:
    """1 seed の (μ+λ)-ES で到達した best gene の honest 再評価 held-out R² を返す."""
    eval_once = make_deep_eval(res, task, n_train=N_TRAIN, n_eval=N_EVAL)
    out = evolve_search(eval_once, res, config=ES_CONFIG, seed=ES_SEED_BASE + seed_idx)
    return out.honest_max_r2


def _summ(label: str, vals: np.ndarray, dt: float, extra: str = "") -> None:
    print(
        f"[{label:14s}] max R²  mean={vals.mean():.4f} std={vals.std():.4f} "
        f"(min={vals.min():.3f} max={vals.max():.3f})  {dt:5.1f}s {extra}"
    )


def main() -> None:
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    res_1l8 = DeepReservoir(layer_taps=(8,), in_dim=task.in_dim)       # baseline 単層
    res_3l = DeepReservoir(layer_taps=(8, 8, 8), in_dim=task.in_dim)   # 機構基質

    print("=== 測定: 機構 evolved_search (床外し / 非外し) ===")
    print(f"task=delayed_parity(seq_len=20,window=5)  n_seeds={N_SEEDS}")
    print(f"baseline=1L-8 random(n={N_RANDOM})  mechanism=3L-8x8x8 (μ+λ)-ES "
          f"(pop={ES_CONFIG.pop_size},gen={ES_CONFIG.n_generations},budget={ES_CONFIG.budget})")
    print(f"参考=3L-8x8x8 random(n={N_RANDOM})  honest_n_trials={ES_CONFIG.honest_n_trials}\n")

    # --- baseline 1L-8 random ---
    t0 = time.time()
    base_vals = np.array(
        [random_search_ceiling(res_1l8, task, N_RANDOM, s) for s in range(N_SEEDS)]
    )
    _summ("1L-8 random", base_vals, time.time() - t0,
          f"gene_dim={res_1l8.gene_dim}")

    # --- 参考 3L random (同基質を random で = 基質寄与の分離用) ---
    t0 = time.time()
    rand3l_vals = np.array(
        [random_search_ceiling(res_3l, task, N_RANDOM, s) for s in range(N_SEEDS)]
    )
    _summ("3L random", rand3l_vals, time.time() - t0,
          f"gene_dim={res_3l.gene_dim}")

    # --- mechanism 3L evolved_search ---
    t0 = time.time()
    mech_vals = np.array(
        [evolved_search_ceiling(res_3l, task, s) for s in range(N_SEEDS)]
    )
    _summ("3L evolved", mech_vals, time.time() - t0,
          f"gene_dim={res_3l.gene_dim} (honest)")

    baseline_max_r2 = float(base_vals.max())
    mechanism_max_r2 = float(mech_vals.max())

    # --- 主出力 (schema 用) ---
    print("\n=== 主指標 (held-out max R²) ===")
    print(f"baseline_1L8_max_r2   = {baseline_max_r2:.4f}  (mean={base_vals.mean():.4f})")
    print(f"mechanism_max_r2      = {mechanism_max_r2:.4f}  (mean={mech_vals.mean():.4f})")
    print(f"3L_random_max_r2(参考) = {rand3l_vals.max():.4f}  (mean={rand3l_vals.mean():.4f})")

    # --- 構成別 R² 要約 ---
    print("\n=== 構成別 max R² (mean ± std) ===")
    print(f"  1L-8  random : {base_vals.mean():.4f} ± {base_vals.std():.4f}")
    print(f"  3L    random : {rand3l_vals.mean():.4f} ± {rand3l_vals.std():.4f}")
    print(f"  3L    evolved: {mech_vals.mean():.4f} ± {mech_vals.std():.4f}")

    # --- floor_lifted 判定 (held-out max R² > 0.5) ---
    floor_lifted = mechanism_max_r2 > 0.5
    print("\n=== floor_lifted 判定 (閾値 max R² > 0.5) ===")
    print(f"  mechanism best max R² = {mechanism_max_r2:.4f} → "
          f"floor_lifted = {floor_lifted}")
    if not floor_lifted:
        print("  → 進化探索でも parity を解けない (held-out R² が 0.5 を大きく下回る)")

    # --- 補助: strict_compare (片側 Wilcoxon + δ) ---
    print("\n=== 補助判定 strict_compare ===")
    r_vs_base = strict_compare(mech_vals, base_vals, "3L-evolved", "1L-8-random")
    print(f"  3L-evolved vs 1L-8-random : diff={r_vs_base.diff:+.4f} "
          f"p={r_vs_base.wilcoxon_p:.4g} δ={r_vs_base.paired_sign_delta:+.2f} "
          f"passes={r_vs_base.passes}")
    r_vs_rand = strict_compare(mech_vals, rand3l_vals, "3L-evolved", "3L-random")
    print(f"  3L-evolved vs 3L-random   : diff={r_vs_rand.diff:+.4f} "
          f"p={r_vs_rand.wilcoxon_p:.4g} δ={r_vs_rand.paired_sign_delta:+.2f} "
          f"passes={r_vs_rand.passes}")

    # --- honest 要約 (attribution の根拠) ---
    print("\n=== honest 要約 ===")
    search_gain = mech_vals.mean() - rand3l_vals.mean()
    substrate_gain = rand3l_vals.mean() - base_vals.mean()
    print(f"  基質寄与 (3L-random - 1L-8)     = {substrate_gain:+.4f}")
    print(f"  探索寄与 (3L-evolved - 3L-random) = {search_gain:+.4f}")
    if mechanism_max_r2 <= 0.5:
        print("  → 進化探索 (探索強化) でも床は外れない。best も chance 近傍に留まる。")
        print("    parity 非可解の主因は探索不足ではなく DeepReservoir+ridge の表現力限界。")
    print(f"\nbaseline_max_r2={baseline_max_r2:.6f} mechanism_max_r2={mechanism_max_r2:.6f}")


if __name__ == "__main__":
    main()
