# SPDX-License-Identifier: Apache-2.0
"""手順 4: ③(差し survival 経由の選択) が hill-climbing を超えて load-bearing になる状態の探索基盤.

設計ノート `docs/poc/STEP4_DESIGN_space_expansion_niching.md` の推奨路線 (iii) MAP-Elites を、
**強 baseline (equal-budget random-restart hill-climbing + panmictic GA)** と並べて公正比較する。

③成立の falsifiable 基準 (設計ノート C1-C4):
- C2: hill-climbing が局所最適に詰まる (deceptive landscape)。
- C3: behavioral niching (MAP-Elites) が同予算 hill-climbing/random を honest 再評価で
  ≥15 seed Wilcoxon p<0.05 + Cliff δ 非無視で上回る。
- C4: 勝因が「behavioral stepping-stone を維持して別 basin を発見」に帰属し、単なる探索量でない
  (= random-restart hill-climbing という強 baseline に勝つことで担保)。

すべて連続 gene ベクトル (np.ndarray) で汎用化。eval_once(gene, rng)->float と behavior(gene)->coords
を差し替えれば任意の landscape / task に適用できる。honest 統計は llcore.evolution.honest_eval を再利用。

research/ 隔離 (src/ 非変更, semver 影響なし)。CPU 完結, numpy のみ。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llcore.evolution.honest_eval import (  # noqa: E402
    _cliff_delta,
    _paired_p,
    honest_reevaluate,
)

# gene = 連続ベクトル。eval_once は (gene, rng)->float の 1 回確率的評価。
EvalOnce = Callable[[np.ndarray, np.random.Generator], float]
Behavior = Callable[[np.ndarray], np.ndarray]


def _clip(gene: np.ndarray, bounds: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    lo, hi = bounds
    return np.clip(gene, lo, hi)


# ---------------------------------------------------------------------------
# MAP-Elites (behavioral niching) — 推奨路線 (iii)
# ---------------------------------------------------------------------------


@dataclass
class MapElitesResult:
    best_fitness: float
    best_gene: np.ndarray
    n_filled_cells: int
    n_evals: int


def map_elites(
    eval_once: EvalOnce,
    behavior: Behavior,
    *,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    behavior_bounds: tuple[np.ndarray, np.ndarray],
    grid_shape: tuple[int, ...],
    n_evals: int,
    init_batch: int,
    sigma: float,
    rng: np.random.Generator,
) -> MapElitesResult:
    """MAP-Elites: behavior descriptor で離散化した archive に各 niche の elite を維持.

    各 iteration: archive から random elite を選び mutate→評価→behavior cell に配置
    (同 cell の既存より良ければ置換)。fitness 多峰性を要求せず behavioral 多様性で動く。
    """
    bd_lo, bd_hi = behavior_bounds
    grid = np.array(grid_shape)

    def cell_of(bd: np.ndarray) -> tuple[int, ...]:
        frac = (bd - bd_lo) / np.maximum(bd_hi - bd_lo, 1e-12)
        idx = np.clip((frac * grid).astype(int), 0, grid - 1)
        return tuple(int(i) for i in idx)

    archive: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    evals = 0

    # 初期 random batch
    for _ in range(min(init_batch, n_evals)):
        g = bounds[0] + (bounds[1] - bounds[0]) * rng.random(dim)
        f = eval_once(g, rng)
        evals += 1
        c = cell_of(behavior(g))
        if c not in archive or f > archive[c][1]:
            archive[c] = (g, f)

    # 進化: elite を mutate して archive 更新
    while evals < n_evals:
        keys = list(archive.keys())
        parent_gene, _ = archive[keys[rng.integers(len(keys))]]
        child = _clip(parent_gene + rng.normal(0, sigma, size=dim), bounds)
        f = eval_once(child, rng)
        evals += 1
        c = cell_of(behavior(child))
        if c not in archive or f > archive[c][1]:
            archive[c] = (child, f)

    best_cell = max(archive, key=lambda k: archive[k][1])
    bg, bf = archive[best_cell]
    return MapElitesResult(best_fitness=bf, best_gene=bg, n_filled_cells=len(archive), n_evals=evals)


# ---------------------------------------------------------------------------
# 強 baseline 1: random-restart hill-climbing (equal budget)
# ---------------------------------------------------------------------------


@dataclass
class SearchResult:
    best_fitness: float
    best_gene: np.ndarray
    n_evals: int


def random_restart_hillclimb(
    eval_once: EvalOnce,
    *,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    n_evals: int,
    sigma: float,
    restart_patience: int,
    rng: np.random.Generator,
) -> SearchResult:
    """(1+1)-ES 風 hill-climb + stall で random restart. ③ なしの強 baseline.

    multimodality を「探索量」で攻略する対照: diversity 維持はしないが restart で広く探す。
    MAP-Elites がこれに勝てば、勝因は探索量でなく behavioral 維持 (C4)。
    """
    best_g = bounds[0] + (bounds[1] - bounds[0]) * rng.random(dim)
    best_f = eval_once(best_g, rng)
    evals = 1
    cur_g, cur_f = best_g.copy(), best_f
    stall = 0
    while evals < n_evals:
        cand = _clip(cur_g + rng.normal(0, sigma, size=dim), bounds)
        f = eval_once(cand, rng)
        evals += 1
        if f >= cur_f:
            cur_g, cur_f = cand, f
            stall = 0
        else:
            stall += 1
        if cur_f > best_f:
            best_g, best_f = cur_g.copy(), cur_f
        if stall >= restart_patience:  # 局所最適から脱出 = 探索量で多峰攻略
            cur_g = bounds[0] + (bounds[1] - bounds[0]) * rng.random(dim)
            cur_f = eval_once(cur_g, rng)
            evals += 1
            stall = 0
            if cur_f > best_f:
                best_g, best_f = cur_g.copy(), cur_f
    return SearchResult(best_fitness=best_f, best_gene=best_g, n_evals=evals)


# ---------------------------------------------------------------------------
# 強 baseline 2: panmictic GA (diversity 維持なし、tournament + mutation)
# ---------------------------------------------------------------------------


def panmictic_ga(
    eval_once: EvalOnce,
    *,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    n_evals: int,
    pop_size: int,
    tournament_k: int,
    sigma: float,
    elitism: int,
    rng: np.random.Generator,
) -> SearchResult:
    """素朴な panmictic GA (niche 維持なし). 早期収束する対照."""
    pop = [bounds[0] + (bounds[1] - bounds[0]) * rng.random(dim) for _ in range(pop_size)]
    fits = [eval_once(g, rng) for g in pop]
    evals = pop_size
    best_i = int(np.argmax(fits))
    best_g, best_f = pop[best_i].copy(), fits[best_i]
    while evals < n_evals:
        order = np.argsort(fits)[::-1]
        new_pop = [pop[order[i]].copy() for i in range(elitism)]
        new_fits = [fits[order[i]] for i in range(elitism)]
        while len(new_pop) < pop_size and evals < n_evals:
            idx = rng.integers(0, pop_size, size=tournament_k)
            parent = pop[idx[np.argmax([fits[i] for i in idx])]]
            child = _clip(parent + rng.normal(0, sigma, size=dim), bounds)
            f = eval_once(child, rng)
            evals += 1
            new_pop.append(child)
            new_fits.append(f)
            if f > best_f:
                best_g, best_f = child.copy(), f
        pop, fits = new_pop, new_fits
    return SearchResult(best_fitness=best_f, best_gene=best_g, n_evals=evals)


# ---------------------------------------------------------------------------
# 公正比較ハーネス (honest 再評価 + 統計)
# ---------------------------------------------------------------------------


@dataclass
class Comparison:
    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    diff: float
    win_rate: float
    wilcoxon_p: float
    cliff_delta: float
    n_seeds: int
    passes: bool  # a が b を有意に上回る (diff>0 & p<alpha)


def compare(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str,
    name_b: str,
    *,
    alpha: float = 0.05,
) -> Comparison:
    deltas = scores_a - scores_b
    diff = float(np.mean(deltas))
    p = _paired_p(scores_a, scores_b)
    return Comparison(
        name_a=name_a,
        name_b=name_b,
        mean_a=float(np.mean(scores_a)),
        mean_b=float(np.mean(scores_b)),
        diff=diff,
        win_rate=float(np.mean(scores_a >= scores_b)),
        wilcoxon_p=p,
        cliff_delta=_cliff_delta(deltas),
        n_seeds=len(scores_a),
        passes=bool(diff > 0.0 and p < alpha),
    )


def run_methods_over_seeds(
    eval_once: EvalOnce,
    behavior: Behavior,
    *,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    behavior_bounds: tuple[np.ndarray, np.ndarray],
    grid_shape: tuple[int, ...],
    n_evals: int,
    n_seeds: int,
    honest_n_trials: int,
    sigma: float = 0.12,
    base_seed: int = 20260530,
) -> dict[str, np.ndarray]:
    """MAP-Elites / RR-hillclimb / panmictic-GA / random を n_seeds で走らせ honest 再評価.

    各 method は同一 n_evals 予算。best gene を **進化と独立な fresh seed** で honest 再評価して
    elitism 由来の noisy fitness 持越し artifact を排除 (honest_eval 同方針)。
    """
    me, rr, ga, rnd = [], [], [], []
    for s in range(n_seeds):
        # MAP-Elites
        r_me = map_elites(
            eval_once, behavior, dim=dim, bounds=bounds, behavior_bounds=behavior_bounds,
            grid_shape=grid_shape, n_evals=n_evals, init_batch=max(20, n_evals // 10),
            sigma=sigma, rng=np.random.default_rng(base_seed + s),
        )
        me.append(honest_reevaluate(eval_once, r_me.best_gene, n_trials=honest_n_trials,
                                    rng=np.random.default_rng(base_seed + s + 10_000_000)))
        # random-restart hill-climb
        r_rr = random_restart_hillclimb(
            eval_once, dim=dim, bounds=bounds, n_evals=n_evals, sigma=sigma,
            restart_patience=max(10, n_evals // 20), rng=np.random.default_rng(base_seed + s + 1),
        )
        rr.append(honest_reevaluate(eval_once, r_rr.best_gene, n_trials=honest_n_trials,
                                    rng=np.random.default_rng(base_seed + s + 20_000_000)))
        # panmictic GA
        r_ga = panmictic_ga(
            eval_once, dim=dim, bounds=bounds, n_evals=n_evals, pop_size=20,
            tournament_k=3, sigma=sigma, elitism=1, rng=np.random.default_rng(base_seed + s + 2),
        )
        ga.append(honest_reevaluate(eval_once, r_ga.best_gene, n_trials=honest_n_trials,
                                    rng=np.random.default_rng(base_seed + s + 30_000_000)))
        # pure random (同予算)
        rrng = np.random.default_rng(base_seed + s + 3)
        cands = [bounds[0] + (bounds[1] - bounds[0]) * rrng.random(dim) for _ in range(n_evals)]
        best = max(cands, key=lambda g: eval_once(g, rrng))
        rnd.append(honest_reevaluate(eval_once, best, n_trials=honest_n_trials,
                                     rng=np.random.default_rng(base_seed + s + 40_000_000)))
    return {
        "map_elites": np.array(me),
        "rr_hillclimb": np.array(rr),
        "panmictic_ga": np.array(ga),
        "random": np.array(rnd),
    }
