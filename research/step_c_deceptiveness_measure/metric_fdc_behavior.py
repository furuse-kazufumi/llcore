# SPDX-License-Identifier: Apache-2.0
"""候補 deceptiveness 指標 #1: FITNESS-DISTANCE CORRELATION along behavior (FDC-b).

狙い (open question への接続)
-----------------------------
``docs/poc/STEP_C_APPLICABILITY_VERDICT.md`` で③ (MAP-Elites behavioral niching) は
合成 dip-depth knob d 上で **d* = 0.16** を境に load-bearing になることが確立した。
**残る問い = 実 task の欺瞞性は d* 以上か?** これを numeric に測るには、合成 knob と実 task を
**比較可能な 1 本の軸**に乗せる必要がある。本 module はその候補軸を 1 つ実装する。

候補軸 = FDC (Fitness-Distance Correlation), ただし距離を **behavior 空間**で測る変種
-------------------------------------------------------------------------------------
古典的 FDC (Jones & Forrest 1995) は「fitness と (大域最適への genotype 距離) の相関」。
欺瞞的地形では「最適に近づいても fitness が上がらない」ので相関が崩れる。本 metric は
RAPTOR/llcore の③が **behavior descriptor 上**で niche を切ることに合わせ、距離を
**behavior 空間で大域 best behavior までの距離**として測る (FDC-behavior, 略 FDC-b)。

定義 (どんな (genotype, behavior, fitness) landscape にも適用できる operational definition)
-------------------------------------------------------------------------------------------
1. gene を bounds 内で一様に ``n_samples`` 個サンプル。
2. 各 gene を honest 評価 (確率的 fitness は ``honest_n_trials`` 回平均 = noise 平均化)。
3. **大域 best behavior** を推定: サンプル中で fitness 最大の gene の behavior (b*)。
   (別の大きな pool でなく同じサンプルから取る — 追加 budget 不要・自己完結)。
4. 各サンプルの behavior から b* までのユークリッド距離 ``dist_i`` を計算。
5. **proximity_i = -dist_i** (大域 best behavior への近さ)。
6. **FDC = Pearson corr(fitness, proximity)**。
   - 易しい (非欺瞞) 地形: best behavior に近づくほど fitness が上がる → FDC → +1。
   - 欺瞞的地形: 近づいても fitness が上がらない / むしろ下がる → FDC が 0 近傍〜負。
7. **deceptiveness = 1 - max(0, FDC)** ∈ [0, 1+] (FDC が負なら 1 超もありうる)。
   欺瞞性が上がるほど大きくなるよう向きを揃える。

注意: 古典 FDC は「fitness と *距離* の相関」で **負が易しい**。本実装は task 指示の
``1 - max(0, FDC)`` 規約に合わせ、内部で **proximity (= -距離) との相関**を FDC と呼ぶ
(易しい=+1)。符号規約を取り違えないため出力 dict に ``fdc_convention`` を明記する。

サンプリング規律 (honest disclosure)
------------------------------------
- これは **サンプリング推定**であり、サンプル draw に依存する確率変数。``deceptiveness`` は
  単一推定値を返すが、``deceptiveness_with_ci`` は複数の独立 seed で再推定し
  **mean ± std / 95% CI** を返す (推定ノイズを必ず報告する規律, src/honest_eval と同じ精神)。
- 既定 ``n_samples=4000`` (CPU 数秒)。確率的 fitness は ``honest_n_trials=20`` で平均。
- corridor 系のように高 fitness 域が痩せている landscape では、一様サンプルだと best behavior の
  分散が大きい → CI で必ず確かめること。

src 非変更・selection_lab read-only・numpy のみ・CPU 完結。
"""
from __future__ import annotations

from typing import Callable

import numpy as np

EvalOnce = Callable[[np.ndarray, np.random.Generator], float]
BehaviorFn = Callable[[np.ndarray], np.ndarray]
Bounds = tuple[np.ndarray, np.ndarray]


def _sample_genes(
    bounds: Bounds, dim: int, n: int, rng: np.random.Generator
) -> np.ndarray:
    """bounds 内で一様に n 個の gene をサンプル (shape (n, dim))."""
    lo, hi = bounds
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    return lo + (hi - lo) * rng.random((n, dim))


def _honest_fitness(
    eval_once: EvalOnce,
    gene: np.ndarray,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    """gene を n_trials 回平均で評価 (確率的 fitness の noise 平均化).

    n_trials=1 なら 1 回 (決定論的 fitness 想定)。honest_eval.honest_reevaluate と同じ精神だが
    依存を最小化するため本 module 内に小実装を持つ (src 非変更・import 簡素化)。
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    return float(np.mean([eval_once(gene, rng) for _ in range(n_trials)]))


def fdc_behavior_raw(
    eval_once: EvalOnce,
    behavior_fn: BehaviorFn,
    gene_bounds: Bounds,
    dim: int,
    rng: np.random.Generator,
    *,
    n_samples: int = 4000,
    honest_n_trials: int = 20,
) -> dict:
    """1 回のサンプル draw で FDC-behavior と中間統計量を計算して返す.

    Returns
    -------
    dict
        - ``fdc`` : Pearson corr(fitness, proximity=-behavior距離 to best). 易しい=+1, 欺瞞=低/負。
        - ``deceptiveness`` : ``1 - max(0, fdc)``. 欺瞞性が上がるほど大。
        - ``best_behavior`` : 推定大域 best behavior (list).
        - ``best_fitness`` : サンプル中の最大 honest fitness.
        - ``n_samples`` / ``honest_n_trials`` : 設定値.
        - ``fitness_mean`` / ``fitness_std`` : サンプル fitness の分布 (退化検出用).
        - ``dist_std`` : behavior 距離の分散 (0 なら相関が定義不能).
        - ``degenerate`` : fitness か距離の分散がほぼ 0 で FDC が定義できない場合 True.
    """
    genes = _sample_genes(gene_bounds, dim, n_samples, rng)
    fits = np.empty(n_samples, dtype=np.float64)
    behs: list[np.ndarray] = []
    for i in range(n_samples):
        g = genes[i]
        fits[i] = _honest_fitness(eval_once, g, honest_n_trials, rng)
        behs.append(np.atleast_1d(np.asarray(behavior_fn(g), dtype=np.float64)))
    beh_arr = np.array(behs, dtype=np.float64)  # (n, beh_dim)

    # 大域 best behavior = サンプル中 fitness 最大の gene の behavior
    i_best = int(np.argmax(fits))
    best_behavior = beh_arr[i_best]
    best_fitness = float(fits[i_best])

    # behavior 距離 (ユークリッド) と proximity = -距離
    dist = np.linalg.norm(beh_arr - best_behavior[None, :], axis=1)
    proximity = -dist

    fit_std = float(np.std(fits))
    dist_std = float(np.std(dist))
    degenerate = bool(fit_std < 1e-9 or dist_std < 1e-9)

    if degenerate:
        # 分散ゼロ → 相関が定義不能。FDC=0 (情報なし) として deceptiveness=1 を返さず
        # honest に degenerate フラグを立てる。呼び出し側で扱う。
        fdc = 0.0
    else:
        fdc = float(np.corrcoef(fits, proximity)[0, 1])
        if not np.isfinite(fdc):
            fdc = 0.0
            degenerate = True

    deceptiveness = 1.0 - max(0.0, fdc)

    return {
        "fdc": fdc,
        "fdc_convention": "corr(fitness, proximity=-behavior_distance_to_best); easy=+1, deceptive=low/neg",
        "deceptiveness": deceptiveness,
        "best_behavior": best_behavior.tolist(),
        "best_fitness": best_fitness,
        "n_samples": int(n_samples),
        "honest_n_trials": int(honest_n_trials),
        "fitness_mean": float(np.mean(fits)),
        "fitness_std": fit_std,
        "dist_std": dist_std,
        "degenerate": degenerate,
    }


def deceptiveness(
    eval_once: EvalOnce,
    behavior_fn: BehaviorFn,
    gene_bounds: Bounds,
    dim: int,
    rng: np.random.Generator,
    *,
    n_samples: int = 4000,
    honest_n_trials: int = 20,
) -> float:
    """FDC-behavior 由来の deceptiveness スコア (単一推定値).

    どんな ``(genotype, behavior, fitness)`` landscape にも適用できる operational metric:
    gene を bounds 内で一様サンプル → honest 評価 → fitness 最大個体の behavior を大域 best と推定
    → 各サンプルの behavior 距離との Pearson 相関 (FDC) → ``1 - max(0, FDC)``。

    **注**: これはサンプリング推定。推定ノイズが要る場合は :func:`deceptiveness_with_ci` を使う。
    """
    return float(
        fdc_behavior_raw(
            eval_once, behavior_fn, gene_bounds, dim, rng,
            n_samples=n_samples, honest_n_trials=honest_n_trials,
        )["deceptiveness"]
    )


def deceptiveness_with_ci(
    eval_once: EvalOnce,
    behavior_fn: BehaviorFn,
    gene_bounds: Bounds,
    dim: int,
    rng: np.random.Generator,
    *,
    n_samples: int = 4000,
    honest_n_trials: int = 20,
    n_repeats: int = 5,
) -> dict:
    """独立 seed で ``n_repeats`` 回再推定し mean / std / 95% CI を返す (推定ノイズ報告の規律).

    各 repeat は ``rng`` から派生した独立 child seed で別サンプル draw を行う。
    Returns dict: ``deceptiveness_mean`` / ``deceptiveness_std`` / ``ci95_lo`` / ``ci95_hi``
    / ``fdc_mean`` / ``fdc_std`` / ``per_repeat`` (各 repeat の raw dict) / ``n_repeats``。
    """
    if n_repeats < 1:
        raise ValueError(f"n_repeats must be >= 1, got {n_repeats}")
    child_seeds = rng.integers(0, 2**63 - 1, size=n_repeats)
    raws: list[dict] = []
    for cs in child_seeds:
        sub = np.random.default_rng(int(cs))
        raws.append(
            fdc_behavior_raw(
                eval_once, behavior_fn, gene_bounds, dim, sub,
                n_samples=n_samples, honest_n_trials=honest_n_trials,
            )
        )
    dec = np.array([r["deceptiveness"] for r in raws], dtype=np.float64)
    fdc = np.array([r["fdc"] for r in raws], dtype=np.float64)
    mean = float(np.mean(dec))
    std = float(np.std(dec, ddof=1)) if n_repeats > 1 else 0.0
    # 95% CI on the mean (normal approx; n_repeats 小なので近似と明記)
    sem = std / np.sqrt(n_repeats) if n_repeats > 1 else 0.0
    return {
        "deceptiveness_mean": mean,
        "deceptiveness_std": std,
        "ci95_lo": float(mean - 1.96 * sem),
        "ci95_hi": float(mean + 1.96 * sem),
        "fdc_mean": float(np.mean(fdc)),
        "fdc_std": float(np.std(fdc, ddof=1)) if n_repeats > 1 else 0.0,
        "n_repeats": int(n_repeats),
        "any_degenerate": bool(any(r["degenerate"] for r in raws)),
        "per_repeat": raws,
    }


__all__ = [
    "deceptiveness",
    "deceptiveness_with_ci",
    "fdc_behavior_raw",
]
