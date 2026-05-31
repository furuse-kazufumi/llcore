# SPDX-License-Identifier: Apache-2.0
"""Behavior-elite fitness-DIP — ③ (MAP-Elites behavioral niching) が load-bearing に
なる「欺瞞性 (deceptiveness)」を 1 つの数値で測る候補メトリクス.

== 動機 (一次情報, 再導出しない) ==
docs/poc/STEP_C_APPLICABILITY_VERDICT.md が確立した機構層の事実:
- 合成 knob = ramp に彫った dip 深さ d∈[0,1]。③が厳格 load-bearing (MAP-E が 3 baseline
  全勝) になる閾値 **d*=0.16** (3 base_seed で一致)。d<d* で③不要、d>=d* で③ load-bearing。
- 開いた問い: **実 task の欺瞞性は d* 以上か?** 現状は定性的「指紋」論証のみ。実 task を
  同一軸上に numeric に置く物差しが要る。

== このメトリクスの操作的定義 (任意の (genotype, behavior, fitness) landscape で計算可能) ==
1. gene_bounds から N 個の gene を一様サンプリング。
2. 各 gene の behavior 記述子 (任意次元) を計算し **1D に射影** (多次元なら第 1 主成分;
   合成 knob は元から 1D = mean)。射影軸上を K bin に等分。
3. 各 bin で fitness の **最大値** (= behavior→best-fitness envelope; behavior-elite profile)。
   noisy fitness は honest 再評価 (n_trials 平均) で測る。
4. envelope の **大域ピーク bin** (global optimum behavior) を特定。
5. 大域ピークへ「最も登りにくい側」から到達する path 上で、**最深の正規化 downhill step (dip)**
   を測る:
       envelope を bin 順に走査し、各 local attractor (近傍より高い bin) から global peak へ
       向かう向きに、その途中で踏まねばならない谷の深さ
       = (出発点までの running-max envelope) − (谷底 envelope) を **大域ピーク高で正規化**。
   全 local attractor 起点のうち最大の正規化 dip を返す。
6. = 「より高い峰へ行くのに降りねばならない谷の、大域ピーク比での深さ」。合成 knob では
   ≈ d を復元するはず (dip 中央 b=0.65 で ramp が 0.8→0.8(1-d) に沈むので
   正規化 dip ≈ 0.8·d / 1.0 = 0.8d。スケール差は線形で Spearman/単調性は保たれる)。

== サンプリング由来のノイズ (honest disclosure) ==
- envelope の各 bin max は有限標本の max なので **上方バイアス** (extreme value)。bin あたりの
  サンプル数が少ないと envelope が痩せ、谷が浅く見える (dip 過小評価)。bin あたり期待
  >= ~30 サンプルを確保する N を選ぶ。
- dip の値は finite-sample max のばらつきを持つ。複数 seed で測り **mean ± std (および t 系 95%CI)**
  を必ず報告する。1 seed の点推定は信用しない。
- 高次元 behavior の 1D 射影 (PCA) は分散最大方向であって「③が効く方向」とは限らない。
  実 task で射影が landscape の欺瞞構造を捉えそこねうる点は §caveats / verdict で明記する。

研究隔離: research/step_c_deceptiveness_measure/。src/ と selection_lab.py は read-only import のみ
(本ファイルは両者を一切変更しない)。numpy のみ。CPU 完結。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# src/llcore (honest_eval) を read-only import path に。
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llcore.evolution.honest_eval import honest_reevaluate  # noqa: E402

EvalOnce = Callable[[np.ndarray, np.random.Generator], float]
Behavior = Callable[[np.ndarray], np.ndarray]


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_ensure_utf8_stdout()


# ---------------------------------------------------------------------------
# core: behavior-elite envelope から最深の正規化 dip を抽出
# ---------------------------------------------------------------------------


def _deepest_normalized_dip(envelope: np.ndarray) -> float:
    """1D behavior-elite fitness envelope から最深の正規化 dip を返す.

    各 local attractor (近傍より高い bin、または端点) を起点に、global-peak bin へ向かう
    方向に envelope を走査し、その途中で踏む谷の深さ = (起点までの running-max) − (谷底) を
    global-peak 高で正規化。全 local attractor 起点のうち最大値を返す。

    Parameters
    ----------
    envelope : np.ndarray
        shape (K,) の各 bin の best-fitness。NaN (空 bin) は欠損として無視
        (前方の有効値で橋渡しせず、有効 bin のみで走査する)。

    Returns
    -------
    float
        最深の正規化 dip (>=0)。global peak へ降下なしで届けるなら 0。
    """
    valid = ~np.isnan(envelope)
    if valid.sum() < 2:
        return 0.0
    idx = np.flatnonzero(valid)
    vals = envelope[idx]  # 有効 bin の値 (bin 順)
    n = len(vals)

    g_peak = float(np.max(vals))
    if g_peak <= 0:
        return 0.0
    g_pos = int(np.argmax(vals))

    # local attractor (= strict に近傍より高い、または端点) を起点候補とする。
    deepest = 0.0
    for i in range(n):
        # 起点 i から global peak (g_pos) へ向かう向き
        if i == g_pos:
            continue
        step = 1 if g_pos > i else -1
        # i は「越えるべき谷の手前の到達高」: i から g_pos まで running-max で
        # 「これまで登った最高到達点」を保ち、最低点との差が dip。
        run_max = vals[i]
        path_min_dip = 0.0
        j = i
        while j != g_pos:
            j += step
            # j までに到達した最高点 (= run_max) から見た現在地の沈み込み
            dip_here = run_max - vals[j]
            if dip_here > path_min_dip:
                path_min_dip = dip_here
            if vals[j] > run_max:
                run_max = vals[j]
        # この path で「最も深く沈む点」を正規化
        norm_dip = path_min_dip / g_peak
        if norm_dip > deepest:
            deepest = norm_dip
    return float(deepest)


def _project_behavior_1d(beh: np.ndarray) -> np.ndarray:
    """behavior 行列 (N, B) を 1D に射影する.

    B==1 ならそのまま。B>1 なら第 1 主成分 (分散最大方向) へ射影。
    """
    beh = np.asarray(beh, dtype=np.float64)
    if beh.ndim == 1:
        return beh
    if beh.shape[1] == 1:
        return beh[:, 0]
    centered = beh - beh.mean(axis=0, keepdims=True)
    # SVD で第 1 主成分。標本数 < 次元でも安定。
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    pc1 = vt[0]
    return centered @ pc1


# ---------------------------------------------------------------------------
# public metric (single estimate)
# ---------------------------------------------------------------------------


def deceptiveness(
    eval_once: EvalOnce,
    behavior_fn: Behavior,
    gene_bounds: tuple[np.ndarray, np.ndarray],
    dim: int,
    rng: np.random.Generator,
    *,
    n_samples: int = 4000,
    n_bins: int = 24,
    honest_n_trials: int = 1,
    min_per_bin: int = 8,
) -> float:
    """behavior-elite fitness-DIP 欺瞞性メトリクス (単一推定).

    Parameters
    ----------
    eval_once : (gene, rng) -> float
        1 回の (確率的) fitness 評価。
    behavior_fn : (gene) -> np.ndarray
        behavior 記述子 (1D or 多次元)。
    gene_bounds : (lo, hi)
        各 shape (dim,) の探索範囲。一様サンプリングに使う。
    dim : int
        genotype 次元。
    rng : np.random.Generator
        サンプリング + honest 再評価用。
    n_samples : int
        サンプル gene 数。bin あたり期待 = n_samples/n_bins が >= ~30 になるよう選ぶ
        (envelope max の上方バイアスを抑えるため)。
    n_bins : int
        behavior 射影軸の bin 数。
    honest_n_trials : int
        各 gene の fitness を honest 再評価する試行数。決定論的 fitness は 1 で可。
        確率的 fitness (reservoir R² 等) は >=10 推奨。
    min_per_bin : int
        この未満のサンプルしか入らない bin は envelope から欠損扱い (痩せた max を排除)。

    Returns
    -------
    float
        最深の正規化 dip (>=0)。合成 knob では ≈ 0.8·d を復元する想定。
    """
    lo, hi = np.asarray(gene_bounds[0], dtype=np.float64), np.asarray(gene_bounds[1], dtype=np.float64)
    genes = lo + (hi - lo) * rng.random((n_samples, dim))

    fits = np.empty(n_samples, dtype=np.float64)
    behs: list[np.ndarray] = []
    for k in range(n_samples):
        g = genes[k]
        if honest_n_trials <= 1:
            fits[k] = eval_once(g, rng)
        else:
            fits[k] = honest_reevaluate(eval_once, g, n_trials=honest_n_trials, rng=rng)
        behs.append(np.atleast_1d(np.asarray(behavior_fn(g), dtype=np.float64)))
    beh_mat = np.array(behs, dtype=np.float64)  # (N, B)

    b1d = _project_behavior_1d(beh_mat)  # (N,)
    bmin, bmax = float(b1d.min()), float(b1d.max())
    if bmax - bmin < 1e-12:
        return 0.0  # behavior が一点に縮退 → niche 構造なし → dip 定義不能 = 0
    edges = np.linspace(bmin, bmax, n_bins + 1)
    bin_idx = np.clip(np.digitize(b1d, edges[1:-1]), 0, n_bins - 1)

    envelope = np.full(n_bins, np.nan, dtype=np.float64)
    for b in range(n_bins):
        mask = bin_idx == b
        cnt = int(mask.sum())
        if cnt >= min_per_bin:
            envelope[b] = float(np.max(fits[mask]))
    return _deepest_normalized_dip(envelope)


# ---------------------------------------------------------------------------
# sampling-noise aware estimate (multi-seed mean +/- CI)
# ---------------------------------------------------------------------------


@dataclass
class DipEstimate:
    """複数 seed の dip 推定の集約 (サンプリングノイズを明示)."""

    mean: float
    std: float
    ci95_lo: float
    ci95_hi: float
    per_seed: list[float]
    n_seeds: int
    n_samples: int
    n_bins: int
    honest_n_trials: int

    def as_dict(self) -> dict:
        return {
            "mean": self.mean, "std": self.std,
            "ci95_lo": self.ci95_lo, "ci95_hi": self.ci95_hi,
            "per_seed": self.per_seed, "n_seeds": self.n_seeds,
            "n_samples": self.n_samples, "n_bins": self.n_bins,
            "honest_n_trials": self.honest_n_trials,
        }


def deceptiveness_estimate(
    eval_once: EvalOnce,
    behavior_fn: Behavior,
    gene_bounds: tuple[np.ndarray, np.ndarray],
    dim: int,
    *,
    n_seeds: int = 5,
    base_seed: int = 20260531,
    n_samples: int = 4000,
    n_bins: int = 24,
    honest_n_trials: int = 1,
    min_per_bin: int = 8,
) -> DipEstimate:
    """``deceptiveness`` を n_seeds 回測り mean ± std と t 系 95%CI を返す.

    finite-sample max の上方バイアス由来のばらつきを CI で見せる (1 seed 点推定の禁止)。
    """
    vals: list[float] = []
    for s in range(n_seeds):
        rng = np.random.default_rng(np.random.SeedSequence([base_seed, s]))
        vals.append(deceptiveness(
            eval_once, behavior_fn, gene_bounds, dim, rng,
            n_samples=n_samples, n_bins=n_bins,
            honest_n_trials=honest_n_trials, min_per_bin=min_per_bin,
        ))
    arr = np.array(vals, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    # t 近似 95%CI (n が小さいので t)。scipy 不在でも近似 t 値表で済ます。
    se = std / np.sqrt(len(arr)) if len(arr) > 1 else 0.0
    _T95 = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
            7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
    t = _T95.get(len(arr), 1.96)
    return DipEstimate(
        mean=mean, std=std,
        ci95_lo=mean - t * se, ci95_hi=mean + t * se,
        per_seed=vals, n_seeds=len(arr),
        n_samples=n_samples, n_bins=n_bins, honest_n_trials=honest_n_trials,
    )


__all__ = [
    "deceptiveness",
    "deceptiveness_estimate",
    "DipEstimate",
]
