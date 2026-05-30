# SPDX-License-Identifier: Apache-2.0
"""E-A 実験ハーネス — 多タスク分布で train 進化 → test(hold-out regime) 汎化を測る.

selection_lab (step4) の MAP-Elites / RR-hillclimb / panmictic-GA / random を流用しつつ、
E-A 固有の 2 点を足す:

1. **③ablation `map_elites_randselect`** (設計 doc の spec「②③殺し①変異のみ」):
   MAP-Elites の生存選択 (fitness ゲート placement = ③) と niche elite 維持 (②) を殺し、
   行動ビニング上の **ランダム親 + 変異 (①) だけ**残す。具体差分は親選択と placement の 2 行のみ。

2. **train/test 分離 runner `run_ea_methods_over_seeds`**: 進化は train regimes の eval_once で
   駆動し、得た best gene を **test (hold-out) regimes の eval_once で fresh-seed honest 再評価**。
   主指標 = test 汎化 R²。train 汎化も併記して汎化ギャップを可視化。

research/ 隔離、src 非変更。selection_lab の関数は read-only import (改変しない)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# step4 selection_lab (MAP-E/baselines) と src/llcore (honest_eval) を import path に
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llcore.evolution.honest_eval import honest_reevaluate  # noqa: E402
from selection_lab import (  # noqa: E402
    MapElitesResult,
    _clip,
    panmictic_ga,
    random_restart_hillclimb,
)

EvalOnce = Callable[[np.ndarray, np.random.Generator], float]
Behavior = Callable[[np.ndarray], np.ndarray]


def map_elites_full(
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
    """完全 MAP-Elites (①②③): archive elite を親に選び、fitness ゲートで cell に配置 (ratchet).

    selection_lab.map_elites と同一仕様。``map_elites_randselect`` と 1 ファイルに並べ、ablation
    差分 (親選択 + placement) が明瞭に対比できるよう本ファイルにも明示実装する。
    """
    return _map_elites_core(
        eval_once, behavior, dim=dim, bounds=bounds, behavior_bounds=behavior_bounds,
        grid_shape=grid_shape, n_evals=n_evals, init_batch=init_batch, sigma=sigma,
        rng=rng, selection_mode="elite",
    )


def map_elites_randselect(
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
    """③ablation (②③殺し①変異のみ): 行動 grid は同じだが選択圧を除去.

    full との差分はちょうど 2 点 (``selection_mode='random'``):
    - **親 = archive elite でなく bounds から uniform random** (②: niche elite を親にしない)。
    - **placement = fitness ゲートなしで cell へ無条件上書き** (③: 生存選択を殺す)。
    変異 (①, gaussian sigma) と行動ビニング機構だけが残る。best は最終 archive の max-fitness 占有者。
    """
    return _map_elites_core(
        eval_once, behavior, dim=dim, bounds=bounds, behavior_bounds=behavior_bounds,
        grid_shape=grid_shape, n_evals=n_evals, init_batch=init_batch, sigma=sigma,
        rng=rng, selection_mode="random",
    )


def _map_elites_core(
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
    selection_mode: str,
) -> MapElitesResult:
    """MAP-Elites 本体. selection_mode='elite' で full、'random' で ②③殺し ablation."""
    if selection_mode not in ("elite", "random"):
        raise ValueError(f"selection_mode must be 'elite'|'random', got {selection_mode!r}")
    bd_lo, bd_hi = behavior_bounds
    grid = np.array(grid_shape)

    def cell_of(bd: np.ndarray) -> tuple[int, ...]:
        frac = (bd - bd_lo) / np.maximum(bd_hi - bd_lo, 1e-12)
        idx = np.clip((frac * grid).astype(int), 0, grid - 1)
        return tuple(int(i) for i in idx)

    archive: dict[tuple[int, ...], tuple[np.ndarray, float]] = {}
    evals = 0
    # global best-of-budget (Codex F2 対応): best_gene を「最終 archive 占有者の max」でなく
    # 「予算内で評価した全個体の max-fitness」にする。randselect は無条件上書きで強個体を忘れる
    # ため、archive-max だと評価予算が同じ random (全 n_evals から best) に対し不当に不利になり
    # C-gen3 の gap を水増ししていた。elite mode では fitness ゲート placement により global best が
    # 自分の cell から退避しないので archive-max ≡ global best (full MAP-E は数値不変)。
    gbest_gene: np.ndarray | None = None
    gbest_f = -np.inf

    def _track(g: np.ndarray, f: float) -> None:
        nonlocal gbest_gene, gbest_f
        if f > gbest_f:
            gbest_f, gbest_gene = f, g

    # 初期 random batch。placement も selection_mode に従う (Codex F-High 対応:
    # 初期ループで fitness ゲートを残すと randselect に ③ が混入し「②③だけの差」が崩れる)。
    for _ in range(min(init_batch, n_evals)):
        g = bounds[0] + (bounds[1] - bounds[0]) * rng.random(dim)
        f = eval_once(g, rng)
        evals += 1
        _track(g, f)
        c = cell_of(behavior(g))
        if selection_mode == "elite":
            if c not in archive or f > archive[c][1]:  # ③ fitness ゲート
                archive[c] = (g, f)
        else:  # 'random': ③ 殺し — 初期から無条件上書き
            archive[c] = (g, f)

    while evals < n_evals:
        if selection_mode == "elite":
            keys = list(archive.keys())
            parent_gene, _ = archive[keys[rng.integers(len(keys))]]
        else:  # 'random': ② 殺し — 親は archive でなく bounds から random
            parent_gene = bounds[0] + (bounds[1] - bounds[0]) * rng.random(dim)
        child = _clip(parent_gene + rng.normal(0, sigma, size=dim), bounds)
        f = eval_once(child, rng)
        evals += 1
        _track(child, f)
        c = cell_of(behavior(child))
        if selection_mode == "elite":
            if c not in archive or f > archive[c][1]:  # ③: fitness ゲート (ratchet)
                archive[c] = (child, f)
        else:  # 'random': ③ 殺し — fitness 無視で無条件上書き
            archive[c] = (child, f)

    assert gbest_gene is not None  # n_evals>=1 で必ず 1 個体は評価される
    return MapElitesResult(best_fitness=gbest_f, best_gene=gbest_gene,
                           n_filled_cells=len(archive), n_evals=evals)


@dataclass
class EAMethodScores:
    """1 method の n_seeds 分の汎化スコア (test 主指標 + train 参考)."""

    test: np.ndarray
    train: np.ndarray


def run_ea_methods_over_seeds(
    eval_train: EvalOnce,
    eval_test: EvalOnce,
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
) -> dict[str, EAMethodScores]:
    """4 method を train で進化 → test(hold-out) で honest 再評価.

    全 method 同一 n_evals 予算。best gene を **進化と独立な fresh seed** で eval_test /
    eval_train の双方で honest 再評価 (elitism noisy 持越し artifact 排除)。主指標は test。
    """
    # equal budget 保証 (Codex F-Low 対応): panmictic_ga は pop_size=20 の初期評価を行うため
    # n_evals < 20 だと開始時点で予算超過し equal-budget が崩れる。明示的に弾く。
    _PANMICTIC_POP = 20
    if n_evals < _PANMICTIC_POP:
        raise ValueError(
            f"n_evals={n_evals} < panmictic pop_size={_PANMICTIC_POP}; "
            "equal-budget が保証できない。n_evals を pop_size 以上にすること。"
        )
    init_batch = max(20, n_evals // 10)
    methods = ("map_elites", "map_elites_randselect", "panmictic_ga", "random")
    test_scores: dict[str, list[float]] = {m: [] for m in methods}
    train_scores: dict[str, list[float]] = {m: [] for m in methods}

    # --- seed 設計 (Codex F3 対応) ---
    # (1) 進化 RNG は (method_idx, s) ごとに SeedSequence で一意・無相関化 (旧 base+s+{0,1,2,3} は
    #     隣接 s でエイリアスし method 間で seed 値が再利用されていた)。method_idx 0-3 を使用。
    # (2) honest 再評価は **同一 replicate s で全 method 共通の seed** (common random numbers)。
    #     こうすると index s の 4 method は同一の fresh タスク draw で採点され、index s による
    #     pairing が真の matched replicate になって paired Wilcoxon の前提を満たす。
    def _evo_rng(method_idx: int, s: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([base_seed, method_idx, s]))

    def _honest_both(gene: np.ndarray, s: int) -> tuple[float, float]:
        te = honest_reevaluate(
            eval_test, gene, n_trials=honest_n_trials,
            rng=np.random.default_rng(np.random.SeedSequence([base_seed, 7, s])))
        tr = honest_reevaluate(
            eval_train, gene, n_trials=honest_n_trials,
            rng=np.random.default_rng(np.random.SeedSequence([base_seed, 8, s])))
        return te, tr

    for s in range(n_seeds):
        # --- MAP-E full (①②③) ---
        r = map_elites_full(
            eval_train, behavior, dim=dim, bounds=bounds, behavior_bounds=behavior_bounds,
            grid_shape=grid_shape, n_evals=n_evals, init_batch=init_batch, sigma=sigma,
            rng=_evo_rng(0, s))
        te, tr = _honest_both(r.best_gene, s)
        test_scores["map_elites"].append(te); train_scores["map_elites"].append(tr)

        # --- MAP-E randselect (②③殺し①のみ) ---
        r = map_elites_randselect(
            eval_train, behavior, dim=dim, bounds=bounds, behavior_bounds=behavior_bounds,
            grid_shape=grid_shape, n_evals=n_evals, init_batch=init_batch, sigma=sigma,
            rng=_evo_rng(1, s))
        te, tr = _honest_both(r.best_gene, s)
        test_scores["map_elites_randselect"].append(te)
        train_scores["map_elites_randselect"].append(tr)

        # --- panmictic GA (①③, ②なし) ---
        r = panmictic_ga(
            eval_train, dim=dim, bounds=bounds, n_evals=n_evals, pop_size=_PANMICTIC_POP,
            tournament_k=3, sigma=sigma, elitism=1, rng=_evo_rng(2, s))
        te, tr = _honest_both(r.best_gene, s)
        test_scores["panmictic_ga"].append(te); train_scores["panmictic_ga"].append(tr)

        # --- pure random (同予算) ---
        rrng = _evo_rng(3, s)
        cands = [bounds[0] + (bounds[1] - bounds[0]) * rrng.random(dim) for _ in range(n_evals)]
        best = max(cands, key=lambda g: eval_train(g, rrng))
        te, tr = _honest_both(best, s)
        test_scores["random"].append(te); train_scores["random"].append(tr)

    return {
        m: EAMethodScores(test=np.array(test_scores[m]), train=np.array(train_scores[m]))
        for m in methods
    }


__all__ = [
    "map_elites_full",
    "map_elites_randselect",
    "EAMethodScores",
    "run_ea_methods_over_seeds",
]
