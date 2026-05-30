# SPDX-License-Identifier: Apache-2.0
"""公正な評価 + 反証(falsification)ハーネス — 進化が「成立」したかを偽りなく測る物差し.

背景 (docs/poc/EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md):
- 報告 best_fitness_curve は elitism の凍結持越し (前世代の noisy fitness を再評価せず保持) により
  +0.29 程度 水増しされる artifact。「best が上がった = 進化が成立」は誤り。
- 進化が本当に成立したかの正しい合格条件 (falsification test):
  「進化に使った乱数と独立な fresh seed で再評価した best が、同じ評価予算の random search を
   ≥15 seed で Wilcoxon p<0.05 かつ Cliff's delta が無視できない大きさで上回る」。

本 module はこの合格条件を再現可能な関数として提供する。task 非依存 (eval_once コールバックを取る)
ので RWKV / 合成 landscape / 将来の実 LLM fitness いずれにも使える。

設計:
- :func:`honest_reevaluate` — gene を進化と独立な fresh rng で n_trials 回平均 (水増し排除)
- :func:`equal_budget` — evolve() の fitness 呼出回数 (= random search の標本数)
- :func:`evolution_vs_random` — 多 seed で 進化 vs 同予算 random を公正比較し統計量を返す
- :class:`FalsificationResult` — 比較結果 (diff / win_rate / wilcoxon_p / cliff_delta / passes)

semver: 新規追加のみ。既存シンボル不変。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from .minimal_ga import (
    evolve,
    initialize_random_population,
    initialize_random_population_g,
)

try:
    from scipy.stats import wilcoxon as _scipy_wilcoxon
    _HAS_SCIPY = True
except ImportError:  # pragma: no cover - scipy は optional
    _HAS_SCIPY = False

# gene を 1 回確率的に評価する関数。決定論的 fitness なら rng は無視してよい。
EvalOnce = Callable[[object, np.random.Generator], float]

# fresh-seed 再評価 / random search が evolve() と乱数を共有しないための seed offset。
_HONEST_OFFSET = 1_000_000
_RANDOM_OFFSET = 2_000_000
_RANDOM_HONEST_OFFSET = 3_000_000


@dataclass(frozen=True)
class FalsificationResult:
    """進化 vs 同予算 random の公正比較結果.

    Attributes
    ----------
    ga_mean / random_mean : float
        各 seed の fresh-seed 再評価 best の平均。
    diff : float
        ``ga_mean - random_mean`` (正で進化優位)。
    win_rate : float
        進化 > random だった seed の割合 (厳密勝率、タイは勝ちに数えない)。
    wilcoxon_p : float
        **片側** paired Wilcoxon signed-rank (H1: 進化 > random) の p 値
        (scipy 不在時は片側符号検定で代替)。
    paired_sign_delta : float
        paired 符号バランス効果量 ``(#正 - #負) / n_seeds`` ([-1,1])。
        **教科書的 Cliff's delta (全 i,j ペア比較) ではない**。paired 設計に合わせた
        符号ベース効果量で pairing 情報を保持する (旧名 ``cliff_delta`` から改名)。
    n_seeds : int
        比較に使った seed 数。
    passes : bool
        進化成立の合格判定 = 監査 §5 の完全な基準:
        ``diff > 0`` かつ 片側 ``wilcoxon_p < alpha`` かつ ``n_seeds >= min_seeds``
        かつ ``abs(paired_sign_delta) >= min_effect`` (効果量が非無視)。
    """

    ga_mean: float
    random_mean: float
    diff: float
    win_rate: float
    wilcoxon_p: float
    paired_sign_delta: float
    n_seeds: int
    passes: bool


def honest_reevaluate(
    eval_once: EvalOnce,
    gene: object,
    *,
    n_trials: int,
    rng: np.random.Generator,
) -> float:
    """gene を進化と独立な fresh rng で ``n_trials`` 回評価し平均を返す (水増し排除).

    elitism による凍結持越し artifact は、進化中に得た noisy fitness をそのまま信じることが原因。
    本関数は進化に未使用の rng で測り直すので、その artifact が構造的に混入しない。
    """
    if n_trials < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")
    return float(np.mean([eval_once(gene, rng) for _ in range(n_trials)]))


def equal_budget(pop_size: int, n_generations: int, elitism: int) -> int:
    """evolve() の fitness 呼出回数 = 初期集団 + 各世代の新個体数.

    random search の標本数をこれに合わせると「同じ評価予算」での公正比較になる。
    入力は ``evolve()`` と同じ契約 (0 <= elitism <= pop_size) を要求する。
    """
    if pop_size < 1:
        raise ValueError(f"pop_size must be >= 1, got {pop_size}")
    if n_generations < 0:
        raise ValueError(f"n_generations must be >= 0, got {n_generations}")
    if not (0 <= elitism <= pop_size):
        raise ValueError(f"elitism must be in [0, pop_size={pop_size}], got {elitism}")
    return pop_size + n_generations * (pop_size - elitism)


def _paired_sign_delta(deltas: np.ndarray) -> float:
    """paired 差分 (ga - rand) の符号バランス効果量 ``(#正 - #負) / n``.

    **教科書的 Cliff's delta ではない** (あれは 2 標本の全 i,j ペア比較)。本統計は
    paired 設計 (同 seed の ga/rand) に合わせ seed ごとの差の符号バランスを測る。
    値域 [-1,1]、+1 で全 seed 進化勝ち、-1 で全 seed 負け、0 で拮抗。
    """
    n = len(deltas)
    if n == 0:
        return 0.0
    pos = int(np.sum(deltas > 0))
    neg = int(np.sum(deltas < 0))
    return (pos - neg) / n


def _sign_test_p_greater(deltas: np.ndarray) -> float:
    """scipy 不在時の代替: **片側** 符号検定 (H1: 正の差が多い = 進化 > random).

    タイ (差=0) は除外し、Binom(n, 0.5) で ``P(X >= wins)`` を返す。
    """
    from math import comb

    wins = int(np.sum(deltas > 0))
    losses = int(np.sum(deltas < 0))
    n = wins + losses
    if n == 0:
        return 1.0
    p = sum(comb(n, i) for i in range(wins, n + 1)) * (0.5 ** n)
    return float(min(1.0, p))


def _paired_p(ga: np.ndarray, rand: np.ndarray) -> float:
    """**片側** paired 比較の p 値 (H1: ga > rand).

    scipy Wilcoxon を ``alternative="greater"`` で使い、不在/全タイ時は片側符号検定で代替する。
    仮説は「進化が random を上回る」の一方向なので両側でなく片側 (両側だと p が 2 倍で過剰に厳しい)。
    """
    deltas = ga - rand
    if np.allclose(deltas, 0.0):
        return 1.0
    if _HAS_SCIPY:
        try:
            # zero_method='wilcox' はタイをドロップ。全要素同一だと例外 → fallback。
            return float(_scipy_wilcoxon(ga, rand, alternative="greater").pvalue)
        except Exception:  # pragma: no cover
            return _sign_test_p_greater(deltas)
    return _sign_test_p_greater(deltas)


def evolution_vs_random(
    eval_once: EvalOnce,
    *,
    codec: Optional[object] = None,
    pop_size: int = 10,
    n_generations: int = 10,
    elitism: int = 1,
    tournament_k: int = 3,
    mutation_sigma: float = 0.15,
    crossover_rate: float = 0.5,
    n_seeds: int = 15,
    honest_n_trials: int = 30,
    base_seed: int = 0,
    alpha: float = 0.05,
    min_seeds: int = 15,
    min_effect: float = 0.147,
) -> FalsificationResult:
    """進化 vs 同予算 random search を ``n_seeds`` で公正比較し合格判定を返す.

    各 seed で:
    1. evolve() を回し (進化用 rng)、final_best.gene を **fresh seed** で honest 再評価。
    2. 同じ評価予算 (= :func:`equal_budget`) の random search を回し、best を **別 fresh seed** で再評価。
    3. paired (ga, rand) を蓄積。

    Parameters
    ----------
    min_seeds : int
        ``passes`` が要求する最小 seed 数 (監査 §5 の「≥15 seed」)。``n_seeds`` がこれ未満なら
        ``passes=False`` (少 seed では「進化成立」と認めない)。関数自体は ``n_seeds`` で実行する
        (少 seed の quick check も可)。
    min_effect : float
        ``passes`` が要求する ``abs(paired_sign_delta)`` の下限 (「効果量が非無視」)。既定 0.147 は
        Cliff's delta の small-effect 境界を実務上の cutoff として流用 (paired_sign_delta に適用)。

    Returns
    -------
    FalsificationResult
        ``passes=True`` なら「進化が同予算 random を有意に上回る = ③ 経由の累積改善が成立」。
        現状の平坦な proxy fitness では ``passes=False`` が想定 (audit 参照)。

    Notes
    -----
    - eval_once は ``(gene, rng) -> float`` の **1 回の確率的評価**。決定論的 fitness なら rng は無視。
    - **確率的 fitness では監査 §5 の基準が ``honest_n_trials >= 30`` を要求** (再評価のノイズ平均化)。
      決定論的 fitness では 1 で可 (平均しても同値)。
    - ``passes`` は監査 §5 の完全基準: ``diff>0 ∧ 片側 p<alpha ∧ n_seeds>=min_seeds
      ∧ |paired_sign_delta|>=min_effect``。
    """
    budget = equal_budget(pop_size, n_generations, elitism)
    ga_scores: list[float] = []
    rand_scores: list[float] = []

    def _fitness(gene: object, rng: np.random.Generator) -> float:
        return eval_once(gene, rng)

    for s in range(n_seeds):
        # --- 進化 ---
        evo_rng = np.random.default_rng(base_seed + s)
        result = evolve(
            _fitness,
            pop_size=pop_size,
            n_generations=n_generations,
            elitism=elitism,
            tournament_k=tournament_k,
            mutation_sigma=mutation_sigma,
            crossover_rate=crossover_rate,
            rng=evo_rng,
            codec=codec,
        )
        honest_rng = np.random.default_rng(base_seed + s + _HONEST_OFFSET)
        ga_scores.append(
            honest_reevaluate(
                eval_once, result.final_best.gene, n_trials=honest_n_trials, rng=honest_rng
            )
        )

        # --- 同予算 random search ---
        rand_rng = np.random.default_rng(base_seed + s + _RANDOM_OFFSET)
        if codec is None:
            cands = initialize_random_population(budget, rand_rng)
        else:
            cands = initialize_random_population_g(budget, codec, rand_rng)
        # 各候補を 1 回 (noisy) 評価し best を選ぶ — evolve と同じ評価予算・同じ noise 条件。
        best_gene = max(cands, key=lambda g: eval_once(g, rand_rng))
        rand_honest_rng = np.random.default_rng(base_seed + s + _RANDOM_HONEST_OFFSET)
        rand_scores.append(
            honest_reevaluate(
                eval_once, best_gene, n_trials=honest_n_trials, rng=rand_honest_rng
            )
        )

    ga = np.array(ga_scores, dtype=np.float64)
    rand = np.array(rand_scores, dtype=np.float64)
    deltas = ga - rand
    diff = float(np.mean(deltas))
    win_rate = float(np.mean(ga >= rand))
    p = _paired_p(ga, rand)
    delta = _cliff_delta(deltas)
    passes = bool(diff > 0.0 and p < alpha)

    return FalsificationResult(
        ga_mean=float(np.mean(ga)),
        random_mean=float(np.mean(rand)),
        diff=diff,
        win_rate=win_rate,
        wilcoxon_p=p,
        cliff_delta=delta,
        n_seeds=n_seeds,
        passes=passes,
    )


__all__ = [
    "EvalOnce",
    "FalsificationResult",
    "honest_reevaluate",
    "equal_budget",
    "evolution_vs_random",
]
