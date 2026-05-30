# SPDX-License-Identifier: Apache-2.0
"""梯子段1 機構 evolved_search — DeepReservoir 天井を進化探索 ((μ+λ)-ES) で測る.

背景 (Step C / ladder1 exp_l1):
- 単一 leaky reservoir + 線形 ridge readout は delayed_parity (5-bit XOR) を解けない
  (Minsky-Papert の床、全 method held-out R²≈0.016)。
- ladder1 の random search (n_random=400) では DeepReservoir((8,8,8)) でも mean max R²≈0.08
  (best seed で 0.24) 止まり。「床が外れない原因が探索不足か表現力限界か」が未切り分け。

本機構の問い:
  random search でなく **進化探索 (簡易 (μ+λ)-ES)** で同じ DeepReservoir((8,8,8)) の gene 空間を
  探れば、random の 0.08 を超えて parity を解く (held-out R² > 0.5) gene が見つかるか？
  - 見つかる → 床が外れない原因は「探索不足」だった (random が貧弱だった)
  - 見つからない → 「表現力限界」(DeepReservoir + ridge では XOR を線形分離する特徴が
    そもそも作れない) であり、探索を強めても無駄 = 機構の天井が床と同義

(μ+λ)-ES 設計 (タスク指定):
- population μ=30, generation 40, tournament 選択 (k=3), elitism (top-1 凍結持越し),
  gaussian mutation σ=0.15, gene_bounds でクリップ。
- crossover は使わず純 mutation ((μ+λ)-ES の素朴形)。λ = pop - elitism の子を毎世代生成。

honest 再評価 (artifact 排除) — src/llcore/evolution/honest_eval.py の発想踏襲:
- elitism は前世代の noisy fitness を再評価せず凍結持越すため、報告 best は構造的に水増しされる
  (audit: +0.29 程度の artifact)。
- よって「進化中に観測した best fitness」は信じず、**進化と独立な fresh seed で best gene を
  honest 再評価** (n_trials 回平均) した値を機構の到達天井とする。
- これにより「elitism 凍結 artifact で床が外れたように見える」誤帰属を排除する。

公平性 (誤帰属の回避):
- 各 gene の評価は make_eval_once の held-out R² (train/eval を別 draw、readout leakage なし)。
- random search baseline と同じ make_eval_once / 同じ n_train・n_eval で評価し、探索戦略のみを
  変えることで「探索を強めた効果」だけを分離する。

research/ 隔離。src は read-only 流用のみ (fit_ridge_readout 経由、非変更)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

_HERE = Path(__file__).resolve().parent
# 同ディレクトリの multi_reservoir (DeepReservoir, gene_bounds, make_eval_once) を流用。
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from multi_reservoir import DeepReservoir, gene_bounds, make_eval_once  # noqa: E402

# gene を 1 回確率的に評価する関数 (gene, rng) -> float。決定論的でなく held-out R² (noisy)。
EvalOnce = Callable[[np.ndarray, np.random.Generator], float]

# 進化用 rng と honest 再評価用 rng の seed を分離する offset。
# honest_eval._HONEST_OFFSET と同じ発想 (進化に使った乱数と再評価の乱数を非共有にする)。
_HONEST_OFFSET = 1_000_000


@dataclass(frozen=True)
class EvolvedSearchResult:
    """1 seed の (μ+λ)-ES 探索結果スナップショット.

    Attributes
    ----------
    best_gene : np.ndarray
        最終世代の (進化中 fitness 基準での) best gene。
    best_fitness_curve : tuple[float, ...]
        各世代の進化中 best fitness (artifact を含みうる — 参考値)。
    honest_max_r2 : float
        best_gene を進化と独立な fresh seed で honest 再評価した held-out R² (機構の到達天井)。
    n_evals : int
        この seed で消費した fitness 評価回数 (= 初期集団 + Σ_gen λ)。探索予算の記録。
    """

    best_gene: np.ndarray
    best_fitness_curve: tuple[float, ...]
    honest_max_r2: float
    n_evals: int


@dataclass(frozen=True)
class EvolvedSearchConfig:
    """(μ+λ)-ES のハイパーパラメータ (タスク指定の既定値).

    Attributes
    ----------
    pop_size : int
        集団サイズ μ。
    n_generations : int
        世代数。
    tournament_k : int
        tournament 選択の標本数 (k=1 で random、k=pop_size で greedy)。
    elitism : int
        各世代で凍結持越しする上位個体数 (再評価しない)。
    mutation_sigma : float
        gaussian mutation の σ。
    honest_n_trials : int
        best_gene の honest 再評価の試行回数 (noisy fitness の平均化)。
    """

    pop_size: int = 30
    n_generations: int = 40
    tournament_k: int = 3
    elitism: int = 1
    mutation_sigma: float = 0.15
    honest_n_trials: int = 16

    def __post_init__(self) -> None:
        if self.pop_size < 1:
            raise ValueError(f"pop_size must be >= 1, got {self.pop_size}")
        if self.n_generations < 0:
            raise ValueError(f"n_generations must be >= 0, got {self.n_generations}")
        if not (0 <= self.elitism <= self.pop_size):
            raise ValueError(
                f"elitism must be in [0, pop_size={self.pop_size}], got {self.elitism}"
            )
        if not (1 <= self.tournament_k <= self.pop_size):
            raise ValueError(
                f"tournament_k must be in [1, pop_size={self.pop_size}], got {self.tournament_k}"
            )
        if self.honest_n_trials < 1:
            raise ValueError(f"honest_n_trials must be >= 1, got {self.honest_n_trials}")

    @property
    def budget(self) -> int:
        """fitness 評価回数 = 初期集団 + 各世代の子個体数 (= random search の公平標本数)."""
        return self.pop_size + self.n_generations * (self.pop_size - self.elitism)


def _tournament_select_idx(
    fitnesses: np.ndarray, k: int, rng: np.random.Generator
) -> int:
    """k 個体を random sampling し最大 fitness の index を返す (tournament 選択)."""
    n = len(fitnesses)
    idx = rng.choice(n, size=k, replace=False)
    return int(idx[np.argmax(fitnesses[idx])])


def evolve_search(
    eval_once: EvalOnce,
    res: DeepReservoir,
    *,
    config: EvolvedSearchConfig = EvolvedSearchConfig(),
    seed: int = 0,
) -> EvolvedSearchResult:
    """簡易 (μ+λ)-ES で DeepReservoir gene 空間を探索し best gene を honest 再評価する.

    探索:
    1. 初期集団 μ を gene_bounds 内 uniform で生成し held-out R² (noisy) で評価。
    2. 各世代:
       - elitism: 上位 ``elitism`` 個体を fitness ごと凍結持越し (再評価しない)。
       - 残り λ = pop - elitism を tournament 選択 → gaussian mutation (σ) → bounds clip
         で生成し、新規 noisy 評価。
       - elite + 子 で次世代を構成。
    3. 最終世代の best gene を **進化と独立な fresh seed** (seed + _HONEST_OFFSET) で
       ``honest_n_trials`` 回 held-out 評価し平均 = honest_max_r2 (artifact 排除)。

    Parameters
    ----------
    eval_once : Callable[[np.ndarray, np.random.Generator], float]
        ``make_eval_once(res, task)`` が返す held-out R² 評価関数 (noisy)。
    res : DeepReservoir
        探索対象の基質 (gene_dim / gene_bounds はここから決まる)。
    config : EvolvedSearchConfig
        (μ+λ)-ES のハイパーパラメータ。
    seed : int
        進化用 rng の base seed。honest 再評価は seed + _HONEST_OFFSET で独立。

    Returns
    -------
    EvolvedSearchResult
        best_gene / best_fitness_curve / honest_max_r2 / n_evals。
        **機構の到達天井は honest_max_r2** (進化中 best_fitness_curve は artifact 込み参考値)。
    """
    lo, hi = gene_bounds(res)
    evo_rng = np.random.default_rng(seed)

    # --- 初期集団 ---
    genes = [res.random_gene(evo_rng) for _ in range(config.pop_size)]
    fitnesses = np.array([eval_once(g, evo_rng) for g in genes], dtype=np.float64)
    n_evals = config.pop_size
    best_curve: list[float] = [float(fitnesses.max())]

    # --- 世代ループ ---
    for _gen in range(config.n_generations):
        # elitism: 上位個体を gene + fitness ごと凍結持越し (再評価しない)。
        # honest_eval.py の audit: これが best 単調性を作るが noisy fitness 水増しの源。
        order = np.argsort(-fitnesses)
        elite_idx = order[: config.elitism]
        new_genes: list[np.ndarray] = [genes[i].copy() for i in elite_idx]
        new_fit: list[float] = [float(fitnesses[i]) for i in elite_idx]

        # λ = pop - elitism 個の子を tournament + gaussian mutation + clip で生成。
        n_children = config.pop_size - config.elitism
        for _ in range(n_children):
            p = _tournament_select_idx(fitnesses, config.tournament_k, evo_rng)
            child = genes[p] + evo_rng.normal(0.0, config.mutation_sigma, size=res.gene_dim)
            child = np.clip(child, lo, hi)  # gene_bounds で範囲外を抑制 (数値安定)
            new_genes.append(child)
            new_fit.append(float(eval_once(child, evo_rng)))  # 子のみ新規評価
            n_evals += 1

        genes = new_genes
        fitnesses = np.array(new_fit, dtype=np.float64)
        best_curve.append(float(fitnesses.max()))

    # --- honest 再評価 (artifact 排除) ---
    best_i = int(np.argmax(fitnesses))
    best_gene = genes[best_i].copy()
    honest_rng = np.random.default_rng(seed + _HONEST_OFFSET)  # 進化と独立な fresh seed
    honest_vals = [eval_once(best_gene, honest_rng) for _ in range(config.honest_n_trials)]
    honest_max_r2 = float(np.mean(honest_vals))

    return EvolvedSearchResult(
        best_gene=best_gene,
        best_fitness_curve=tuple(best_curve),
        honest_max_r2=honest_max_r2,
        n_evals=n_evals,
    )


def make_deep_eval(
    res: DeepReservoir,
    task: object,
    *,
    n_train: int = 48,
    n_eval: int = 48,
    ridge_lambda: float = 1e-2,
) -> EvalOnce:
    """DeepReservoir 用の held-out R² 評価関数を作る (make_eval_once の薄いラッパ).

    random search baseline と本機構で **同一の評価関数** を使うことで、探索戦略のみを
    変えた公平比較を保証する。train/eval を別 draw する held-out 評価なのでデータリークなし。
    """
    return make_eval_once(res, task, n_train=n_train, n_eval=n_eval, ridge_lambda=ridge_lambda)


__all__ = [
    "EvalOnce",
    "EvolvedSearchResult",
    "EvolvedSearchConfig",
    "evolve_search",
    "make_deep_eval",
]
