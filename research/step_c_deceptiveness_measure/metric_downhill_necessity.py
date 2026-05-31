# SPDX-License-Identifier: Apache-2.0
"""DOWNHILL-NECESSITY deceptiveness metric — landscape-agnostic, sampling-based.

候補欺瞞性メトリック 1 種を実装する。直観:

    「behavior 空間で、ランダムな出発点から **単調 (非悪化) な hill-climb だけ**で
     大域最適に到達できる経路の割合 (reach_fraction)。欺瞞的な地形ほど "一度
     下り坂を降りないと高い峰に届かない" ので reach が低い。」

    deceptiveness = 1 - reach_fraction.

これは ③ (MAP-Elites の behavioral niching) が load-bearing になる条件
(STEP_C_APPLICABILITY_VERDICT.md: behavior-elite fitness profile に DIP があり、
高い峰へ行く前にそれを降りねばならない) を **直接** 操作的に測ろうとするもの。
MAP-Elites の archive ratchet は downhill を跨げるが、純粋な hill-climb は跨げない。
よって "hill-climb で届かない割合" が大きい = ③が必要、という対応を狙う。

== 計算手順 (任意の (genotype, behavior, fitness) landscape に適用可能) ==

1. **behavior-elite fitness profile を sampling で構築**:
   - gene_bounds 内から n_samples 個の gene を一様に draw。
   - 各 gene を eval_once で評価し、behavior_fn(gene) で behavior 座標を計算。
   - behavior 空間を grid (各軸 n_bins) に離散化し、各 cell の **elite fitness**
     (= その cell に落ちた gene の最大 fitness) を記録する。これは MAP-Elites の
     archive を近似する "到達可能な最良 fitness の behavior プロファイル"。
   - 確率的 fitness は cell 内で複数サンプルの max を取る (noisy 上振れを許容するが、
     elite profile はもともと max 演算なので保守側にはならない。CI は別途報告)。

2. **大域最適 cell を同定**: elite profile の argmax cell。

3. **monotone hill-climb の reach 推定**: 占有された各 cell を出発点とし、
   グリッド近傍 (Moore neighborhood) のうち **fitness が現在以上** の cell へ
   貪欲に (最急上昇) 移動する。下り (悪化) には決して進めない。これを動けなく
   なるまで繰り返し、停止 cell が大域最適 cell なら "reach 成功"。
   - 出発点は「ランダムな behavior から」を満たすため、占有 cell を一様重みで扱う
     (= ランダムな初期 behavior に最も近い占有 niche から登る近似)。
   - reach_fraction = (大域最適へ到達した出発 cell 数) / (全出発 cell 数)。

4. **deceptiveness = 1 - reach_fraction**。1 に近いほど欺瞞的 (downhill 必須)。

== sampling noise / CI ==

reach_fraction は有限サンプルの占有 cell 集合と確率的 fitness の draw に依存する。
本 module は **n_repeats 回 (異なる seed)** 全手順を反復し、deceptiveness の
平均と標準誤差 (SE = std / sqrt(n_repeats)) を報告する。`deceptiveness(...)` は
平均を返し、`deceptiveness_with_ci(...)` は (mean, se, samples) を返す。

注意 (honest disclosure):
- これは **behavior プロファイルの欺瞞性** の sampling 推定であって、genotype 空間
  そのものの hill-climb 可能性ではない (MAP-Elites も behavior 空間で動くので整合)。
- grid 解像度 (n_bins) と sample 数に感度がある。calibration では固定値を使い、
  d 横断で同一設定にして相対比較の公正性を保つ。
- 占有が疎な高次元 behavior では cell 連結性が壊れて reach が人工的に下がりうる。
  E-A (2D behavior) では特に注意 (caveat に明記)。

research/ 隔離。numpy のみ。src/ と selection_lab.py は触らない (本 module は import すらしない)。
"""
from __future__ import annotations

import itertools
import sys
from typing import Callable

import numpy as np


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_ensure_utf8_stdout()

EvalOnce = Callable[[np.ndarray, np.random.Generator], float]
BehaviorFn = Callable[[np.ndarray], np.ndarray]


def _build_elite_profile(
    eval_once: EvalOnce,
    behavior_fn: BehaviorFn,
    gene_bounds: tuple[np.ndarray, np.ndarray],
    dim: int,
    *,
    n_samples: int,
    n_bins: int,
    behavior_bounds: tuple[np.ndarray, np.ndarray],
    fitness_trials: int,
    rng: np.random.Generator,
) -> dict[tuple[int, ...], float]:
    """behavior grid 上の elite fitness profile を sampling で構築する.

    各 cell key -> その cell に落ちた gene の最大 (mean over fitness_trials) fitness。
    確率的 fitness は cell ごとに fitness_trials 回平均してから max を取る (noisy 上振れ抑制)。
    """
    lo, hi = gene_bounds
    bd_lo, bd_hi = behavior_bounds
    bdim = len(bd_lo)
    grid = np.full(bdim, n_bins, dtype=int)

    def cell_of(bd: np.ndarray) -> tuple[int, ...]:
        frac = (bd - bd_lo) / np.maximum(bd_hi - bd_lo, 1e-12)
        idx = np.clip((frac * grid).astype(int), 0, grid - 1)
        return tuple(int(i) for i in idx)

    profile: dict[tuple[int, ...], float] = {}
    for _ in range(n_samples):
        g = lo + (hi - lo) * rng.random(dim)
        if fitness_trials <= 1:
            f = float(eval_once(g, rng))
        else:
            f = float(np.mean([eval_once(g, rng) for _ in range(fitness_trials)]))
        c = cell_of(np.asarray(behavior_fn(g), dtype=np.float64))
        if c not in profile or f > profile[c]:
            profile[c] = f
    return profile


def _neighbors(cell: tuple[int, ...], n_bins: int) -> list[tuple[int, ...]]:
    """Moore neighborhood (各軸 ±1, 0; 自分を除く) の grid 内 cell を列挙する."""
    bdim = len(cell)
    out: list[tuple[int, ...]] = []
    for delta in itertools.product((-1, 0, 1), repeat=bdim):
        if all(d == 0 for d in delta):
            continue
        nb = tuple(cell[i] + delta[i] for i in range(bdim))
        if all(0 <= nb[i] < n_bins for i in range(bdim)):
            out.append(nb)
    return out


def _reach_fraction(profile: dict[tuple[int, ...], float], n_bins: int) -> float:
    """占有 cell から monotone (非悪化) 最急上昇 hill-climb で大域最適 cell へ届く割合.

    各占有 cell を出発点に、現 fitness 以上の近傍へ貪欲に最急上昇。動けなくなった停止 cell が
    大域最適 cell なら成功。reach_fraction = 成功出発 cell 数 / 全占有 cell 数。
    """
    if not profile:
        return 1.0  # 占有ゼロは病的 — 欺瞞性ゼロ扱い (sample 不足。caveat で扱う)
    cells = list(profile.keys())
    global_cell = max(cells, key=lambda c: profile[c])
    reached = 0
    for start in cells:
        cur = start
        # 最急上昇で停留点まで移動 (非悪化のみ許容; 厳密上昇でなく >= で plateau も渡る)。
        # plateau 上の循環を防ぐため visited を保持。
        visited = {cur}
        while True:
            cur_f = profile[cur]
            best_nb = None
            best_f = cur_f
            for nb in _neighbors(cur, n_bins):
                if nb in profile and profile[nb] >= best_f and nb not in visited:
                    # 厳密上昇を優先するが、同値 plateau も許す (downhill だけを禁止)。
                    if profile[nb] > best_f or (best_nb is None and profile[nb] == best_f):
                        best_f = profile[nb]
                        best_nb = nb
            if best_nb is None:
                break
            cur = best_nb
            visited.add(cur)
        if cur == global_cell:
            reached += 1
    return reached / len(cells)


def deceptiveness_with_ci(
    eval_once: EvalOnce,
    behavior_fn: BehaviorFn,
    gene_bounds: tuple[np.ndarray, np.ndarray],
    dim: int,
    rng: np.random.Generator,
    *,
    n_samples: int = 4000,
    n_bins: int = 24,
    behavior_bounds: tuple[np.ndarray, np.ndarray] | None = None,
    fitness_trials: int = 1,
    n_repeats: int = 5,
) -> tuple[float, float, list[float]]:
    """DOWNHILL-NECESSITY 欺瞞性を n_repeats 回推定し (mean, SE, per-repeat samples) を返す.

    Parameters
    ----------
    n_samples : int
        elite profile 構築に使う gene draw 数 (1 repeat あたり)。
    n_bins : int
        behavior 各軸の grid 解像度 (calibration の合成 1D corridor は 24 = exp_knob_sweep grid と同一)。
    behavior_bounds : (lo, hi) | None
        behavior 座標の範囲。None なら各軸 [0,1] (合成 corridor 用)。実 task では実測レンジを渡す。
    fitness_trials : int
        確率的 fitness の cell 内平均化回数 (decision noise を抑える)。決定論なら 1。
    n_repeats : int
        全手順の独立反復数 (sampling noise の SE 推定用)。
    """
    if behavior_bounds is None:
        # behavior 次元を 1 個 draw で推定 (合成 corridor は 1D / E-A は 2D)。
        lo, hi = gene_bounds
        probe_g = lo + (hi - lo) * rng.random(dim)
        bdim = len(np.atleast_1d(np.asarray(behavior_fn(probe_g))))
        behavior_bounds = (np.zeros(bdim), np.ones(bdim))

    samples: list[float] = []
    for _ in range(n_repeats):
        sub = np.random.default_rng(rng.integers(0, 2**63 - 1))
        profile = _build_elite_profile(
            eval_once, behavior_fn, gene_bounds, dim,
            n_samples=n_samples, n_bins=n_bins, behavior_bounds=behavior_bounds,
            fitness_trials=fitness_trials, rng=sub,
        )
        reach = _reach_fraction(profile, n_bins)
        samples.append(1.0 - reach)
    mean = float(np.mean(samples))
    se = float(np.std(samples, ddof=1) / np.sqrt(len(samples))) if len(samples) > 1 else 0.0
    return mean, se, samples


def deceptiveness(
    eval_once: EvalOnce,
    behavior_fn: BehaviorFn,
    gene_bounds: tuple[np.ndarray, np.ndarray],
    dim: int,
    rng: np.random.Generator,
    **kw,
) -> float:
    """DOWNHILL-NECESSITY 欺瞞性スコア (mean over n_repeats)。1 に近いほど欺瞞的.

    署名: deceptiveness(eval_once, behavior_fn, gene_bounds, dim, rng, **kw) -> float。
    kw は deceptiveness_with_ci に転送 (n_samples / n_bins / behavior_bounds / fitness_trials / n_repeats)。
    """
    mean, _se, _samples = deceptiveness_with_ci(
        eval_once, behavior_fn, gene_bounds, dim, rng, **kw
    )
    return mean


__all__ = ["deceptiveness", "deceptiveness_with_ci"]
