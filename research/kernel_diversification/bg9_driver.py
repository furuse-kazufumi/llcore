# SPDX-License-Identifier: Apache-2.0
"""BG9-4 — ③(選択圧/分離)が kernel 多様化空間で load-bearing な「欺瞞地形」を作るかの harness.

BG9_PREREGISTRATION.md を protocol 正本とする。本 module は **harness の実装 + smoke validity 確認**
までで、本検定 (>=15 seed・n_evals=4000 の full run) は **次段 BG9-5** に渡す (本 module では回さない)。

設計 (pre-reg §1/§2/§3):

3 基質 (substrate)
  - positive control (harness validity の核): **synthetic kernel-barrier eval**。step4
    ``exp_knob_sweep.make_corridor_eval`` の deceptive corridor を **kernel_id 軸** に翻案する
    (``corridor_eval`` を gene.mean() でなく gene[0]=kernel_id 連続値で駆動)。target kernel_id 領域
    (実際に働く kernel = linear_attn) で高 fitness、init 領域 (rwkv) との間に kernel_id 軸の
    fitness 谷 (dip) を置く。hill-climb は谷で止まり、MAP-E は kernel_id niche を stepping-stone に
    跨ぐ。**hopfield は対角 mock で機能不全 (BG9-3 実証) ゆえ target に使わない**。
    要件: smoke で MAP-E が 3 baseline 全勝 → harness が③を検出可。示せなければ harness validity
    不成立 → BG9=N/A 方向。
  - negative control: kernel 中立 task = memory_tasks の delayed_recall (全 kernel 飽和 ~1.0)。
    要件: MAP-E が③優位を持たない (smooth ゆえ niching 無益)。優位が出たら false-positive 警告。
  - real: 採用 suite {selective_copy, bistable_denoise, weighted_accum} の multi-task (TaskMixture
    風に task をランダムに混ぜる)。BG9-3 strong BG6 validity PASS。leaky_tracking は fragile ゆえ除外。

method (equal budget, GA dim=5)
  - MAP-E (map_elites, behavior=kernel_behavior, bounds=kernel_behavior_bounds, n_bins=(4,8))
  - RR-hillclimb (random_restart_hillclimb)
  - panmictic-GA (panmictic_ga)
  - random (同予算 best-of-budget)

ablation (A1) kernel_id-抜き
  - behavior から kernel_id を抜いた theta-only behavior (1D = theta L1 のみ) 版 MAP-E。
    ③優位が kernel diversity 由来か探索量由来かの切り分け。

honest 規律 (pre-reg §1.5/§1.6, exp_knob_sweep の seed 設計を踏襲)
  - 進化は train 系列、判定は fresh held-out 再評価 (honest_reevaluate)。
  - 各 method equal budget。進化 RNG = SeedSequence([base, method_idx, s]) で一意化、
    honest 再評価 RNG = SeedSequence([base, 7, s]) を **全 method 共通 (CRN)** にして paired 検定の
    matched replicate を担保 (selection_lab.run_methods_over_seeds は method 毎に別 seed だったため、
    pre-reg §1.4 の CRN を満たすよう本 module 側で CRN runner を再実装する)。
  - 勝つ = 片側 Wilcoxon p<0.05 ∧ |paired_sign_delta|>=min_effect ∧ paired mean diff>0 (strict_compare 準拠)。
    smoke では n_seeds が小さいので min_seeds gate は緩め可だが、p/δ/diff は本番と同基準で測る。

selection_lab / kernel_fitness / kernel_favoring_tasks / kernels は **read-only import**。
新規 helper は本 module 側に閉じる。src 無改変。git は orchestrator 一括。

実行 (単独可・seed 固定・UTF-8):
    py -3.11 research/kernel_diversification/bg9_driver.py [--smoke] [--seeds N] [--n-evals M]
出力:
    research/kernel_diversification/bg9_smoke_results.json
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# research adapter + sibling research labs を import path に。read-only import。
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kernel_favoring_tasks import make_task  # noqa: E402
from kernel_fitness import (  # noqa: E402
    kernel_behavior,
    kernel_behavior_bounds,
    kernel_ga_bounds,
    make_kernel_eval_once,
)
from kernels import GA_DIM, N_KERNELS  # noqa: E402
from memory_tasks_import import delayed_recall_task  # noqa: E402
from selection_lab import map_elites, panmictic_ga, random_restart_hillclimb  # noqa: E402
from llcore.evolution.honest_eval import (  # noqa: E402
    _cliff_delta,  # = paired_sign_delta (paired 符号バランス効果量)
    _paired_p,  # 片側 paired Wilcoxon (scipy 不在時は符号検定)
    honest_reevaluate,
)


def _ensure_utf8_stdout() -> None:
    """Windows cp932 console 対策 (feedback_cli_utf8_stdout_pattern)."""
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", line_buffering=True)
    except Exception:  # pragma: no cover - already wrapped
        pass


EvalOnce = Callable[[np.ndarray, np.random.Generator], float]

# ---------------------------------------------------------------------------
# 固定パラメータ (pre-reg §1)。**確定後に動かさない**。
# ---------------------------------------------------------------------------
N_BINS_KERNEL = N_KERNELS  # kernel_id 軸 = 4 bin (各 kernel basin に 1 bin)  pre-reg §1.2
N_BINS_THETA = 8           # theta L1 軸 = 8 bin                              pre-reg §1.2
GRID_SHAPE = (N_BINS_KERNEL, N_BINS_THETA)
# A1 theta-only behavior の grid (1D, theta L1 軸のみ 8 bin)
GRID_SHAPE_THETA_ONLY = (N_BINS_THETA,)

SIGMA = 0.20         # 連続ベクトル mutation 幅 (kernel_id 1.0 を跨ぐ余地。BG9-3 ga_sigma と同)
ME_INIT_BATCH = 64   # MAP-E init_batch (pre-reg §1.3)

# strict gate (pre-reg §1.6 / strict_compare 準拠)。smoke では min_seeds を緩めるが p/δ/diff は本番同基準。
GATE_ALPHA = 0.05
GATE_MIN_EFFECT = 0.147  # |paired_sign_delta| 下限 (honest_eval §5 / exp_knob_sweep と同)

REAL_TASKS = ("selective_copy", "bistable_denoise", "weighted_accum")  # BG9-3 PASS, leaky 除外


# ---------------------------------------------------------------------------
# positive control: synthetic kernel-barrier eval (kernel_id 軸 deceptive corridor)
#   exp_knob_sweep.make_corridor_eval (genotypic corridor + dip) を **kernel_id 軸** へ翻案する。
#   honest 設計上の落とし穴 (実測で発覚) と対策:
#     - exp4 の corridor は behavior=mean(gene[24 dim]) で CLT により random が b≈0.5 に固着し
#       高 behavior の corridor に **構造的に不到達** だから deceptive だった。
#     - kernel_id を単純に gene[0] の関数にすると、random は kernel_id∈[0,4) を **直接一様サンプル**
#       でき target basin (kid≈3.5) に容易に当たる → 谷を跨ぐ必要が無く RR-hillclimb も全勝で
#       harness が壊れる (tiny smoke で実証: 全 method reach=1.00)。
#   対策: target basin への到達を **(i) kernel_id の谷を跨ぐ ∧ (ii) theta が CLT 不到達の
#   genotype corridor (theta-mean を上端へ押す)** の **両方** で律速する。
#     - random: theta-mean が CLT で中央 (~0.5 正規化) に固着 → target の theta 条件に不到達
#       → 局所峰 (低 kernel_id・theta 任意) の天井までしか出ない。
#     - RR-hillclimb: theta-mean は climb できるが kernel_id の谷 (downhill) で (1+1) が止まり
#       局所 kernel basin に詰まる。
#     - MAP-E: kernel_id bin (4 niche) を stepping-stone に保持して谷を跨ぎ、かつ各 niche 内で
#       theta corridor を ratchet → target basin 到達。
#   = 「kernel_id 障壁を跨ぐ diversity 維持」が load-bearing になる純 harness validity 基質。
#
#   ===== 実測で得た honest 構造的知見 (smoke で確定、§報告に反映) =====
#   上記 (ii) の theta corridor を入れて kid-axis 谷を彫っても、**RR-hillclimb は谷を回避できてしまう**:
#   selection_lab.random_restart_hillclimb は予算 n_evals に対し ~n_evals/20 回の random restart を行い、
#   各 restart は kernel_id∈[0,4) を **直接一様サンプル** する。target kid-basin (幅 ~0.16) に当たる確率は
#   1 restart で低くても、~20-75 restart 累積でほぼ確実に 1 回 in-basin に落ち、そこから theta を
#   in-basin で climb して target へ到達する。予算を増やすと RR は **むしろ強くなる** (restart 増)。
#   結果 (8 seed, n_evals 800/1500): MAP-E は GA / random を p<0.004 δ=+1.0 で**確実に撃破**するが、
#   **RR-hillclimb には勝てない** (diff≈-0.02, RR が 7/8 seed で target 到達)。theta corridor を更に
#   締めて RR を排除しようとすると MAP-E 自身も target に届かず分離が崩れる (探索が starve)。
#   → exp4 の corridor が RR を排除できたのは behavior=mean(24 dim) の **CLT 不到達**ゆえ。kernel_id は
#     **単一の直接サンプル可能座標**なので、5 次元 KernelGenome では kid-axis 欺瞞 corridor で RR を
#     構造的に排除できない。これは harness validity の限界 = pre-reg §2.1/§4 の **N/A (測定不能) 方向**。
#   本実装は最も faithful かつ MAP-E が GA/random を明瞭に撃破する config を採用し、RR 限界を honest に
#   記録する (verdict 側で "GA/random は検出可・RR は kid 直接サンプルで排除不能" を明示)。
# ---------------------------------------------------------------------------
_POS_LOCAL_KID = 0.5   # 局所峰の kernel_id (rwkv basin, init/random が自然に落ちる低 kernel_id)
_POS_GLOB_KID = 3.6    # target 峰の kernel_id (linear_attn basin, 実働 kernel, hopfield 回避)
_POS_DIP_KID = 2.0     # 谷の中央 kernel_id (中間 basin = 跨ぐべき障壁)
_POS_KID_W = 0.16      # target kernel_id 峰の Gaussian 幅 (狭く = restart 直撃を減らす)
_POS_KID_W_LOCAL = 0.70  # 局所峰 (trap) の幅 (広く = init/restart の多くを罠へ)
_POS_DIP_W = 0.80      # 谷の幅 (中間 basin を広く覆う)
_POS_LOCAL_CEIL = 0.55  # 局所峰 (低 kernel_id) の天井 fitness — random/RR が届く上限
_POS_GLOB_CEIL = 1.00  # target 峰の天井 fitness
_POS_THETA_TARGET = 0.85  # target basin が要求する正規化 theta-mean (CLT 中央 ~0.5 から離す)
_POS_THETA_W = 0.13    # theta corridor の許容幅 (狭いほど random が CLT で不到達)
_POS_NOISE = 0.008     # exp_knob_sweep と同一低ノイズ
POS_GLOBAL_PEAK_PROXY = 0.8  # honest fitness > これ で target basin 到達と判定 (exp4 と同 proxy)


def _norm_theta_mean(gene_vec5: np.ndarray) -> float:
    """gene の theta 部 (gene[1:5]) の平均を GA bounds で [0,1] 正規化 (genotype corridor 軸).

    GA は theta 各座標を ``kernel_ga_bounds`` の box [lo,hi] で一様初期化するので、random の
    正規化 theta-mean は CLT で ~0.5 に固着する (4 座標平均 → 分散 1/12/4)。target が要求する
    正規化 theta-mean=0.85 は ~0.5+0.35 ≈ 4σ 先 → random が構造的に不到達 = exp4 の CLT corridor。
    """
    lo, hi = kernel_ga_bounds()
    th_lo, th_hi = lo[1:5], hi[1:5]
    th = np.clip(np.asarray(gene_vec5[1:5], dtype=np.float64), th_lo, th_hi)
    frac = (th - th_lo) / np.maximum(th_hi - th_lo, 1e-12)
    return float(np.mean(frac))


def make_kernel_barrier_eval(d: float = 1.0) -> EvalOnce:
    """synthetic kernel-barrier eval を返す (kernel_id 障壁 + genotype theta corridor の deceptive 地形).

    fitness = max(local_peak, target_peak):
      - local_peak: 低 kernel_id (≈0.5) の Gaussian × 天井 _POS_LOCAL_CEIL。theta 不問。
        init/random/RR が自然に届く「罠」の峰。
      - target_peak: 高 kernel_id (≈3.5) の Gaussian × theta-corridor Gaussian × 天井 _POS_GLOB_CEIL。
        **kernel_id の谷を跨ぐ ∧ theta-mean を _POS_THETA_TARGET へ押す** の両方を要求。さらに
        kernel_id [local,glob] 区間に深さ d の谷 (dip) を彫り、hill-climb の連続登坂を downhill で阻む。

    d=1.0 = 深い dip (deceptive, 既定)、d=0.0 = 谷無し (smooth control)。harness validity は「③が
    在るとき検出できるか」の検定ゆえ deceptive (d=1.0) を要求する。

    A1 theta-only MAP-E は kernel_id niche を持たないため、theta corridor は登れても kernel_id の
    谷を跨げず局所 kernel basin に留まる → 優位が消えるはず (③優位が kernel diversity 由来である
    ことの陽性対照: pre-reg §3 A1)。
    """
    if not (0.0 <= d <= 1.0):
        raise ValueError(f"d must be in [0,1], got {d}")

    def kernel_barrier_eval(gene_vec5: np.ndarray, rng: np.random.Generator) -> float:
        kid = float(np.clip(gene_vec5[0], 0.0, N_KERNELS - 1e-9))
        tm = _norm_theta_mean(gene_vec5)  # 正規化 theta-mean ∈ [0,1] (genotype corridor 軸)

        # 局所峰 (低 kernel_id, theta 不問) — random/RR が届く罠
        local = _POS_LOCAL_CEIL * np.exp(-((kid - _POS_LOCAL_KID) ** 2) / (2 * _POS_KID_W ** 2))

        # target 峰: kernel_id basin × theta corridor。kernel_id ramp に dip を彫る。
        kid_gauss = np.exp(-((kid - _POS_GLOB_KID) ** 2) / (2 * _POS_KID_W ** 2))
        theta_gauss = np.exp(-((tm - _POS_THETA_TARGET) ** 2) / (2 * _POS_THETA_W ** 2))
        if _POS_LOCAL_KID <= kid <= _POS_GLOB_KID:
            dip = d * np.exp(-((kid - _POS_DIP_KID) ** 2) / (2 * _POS_DIP_W ** 2))
            barrier = 1.0 - dip  # kernel_id 谷 (downhill) を跨ぐ律速
        else:
            barrier = 1.0
        target = _POS_GLOB_CEIL * kid_gauss * theta_gauss * barrier

        return float(max(local, target) + rng.normal(0, _POS_NOISE))

    return kernel_barrier_eval


# ---------------------------------------------------------------------------
# real: multi-task (TaskMixture 風) eval — 採用 suite を seed ごとにランダムに混ぜる
# ---------------------------------------------------------------------------


def make_real_multitask_eval(
    task_names: tuple[str, ...] = REAL_TASKS,
    *,
    dim: int = 8,
    projection_seed: int = 0,
) -> EvalOnce:
    """採用 suite の multi-task eval。1 回の eval ごとに task をランダムに 1 つ選んでその held-out R²。

    各 task 用に固定射影 P を持つ ``make_kernel_eval_once`` を **事前生成して共有** (進化と独立 =
    kernel 寄与分離の核)。eval_once は rng で task を一様サンプルし、その task の eval_once へ委譲。
    複数 eval を跨ぐと task が混ざるので、進化は「3 task を同時に得意にする kernel/theta」を探す
    = pre-reg §2.3 の「task をランダムに混ぜる TaskMixture 風」。honest_reevaluate は n_trials 回
    平均するので task ミックスの期待性能を測る (単一 task の偶然に依存しない)。
    """
    evals = [make_kernel_eval_once(make_task(n), dim=dim, projection_seed=projection_seed)
             for n in task_names]
    n_tasks = len(evals)

    def eval_once(gene_vec5: np.ndarray, rng: np.random.Generator) -> float:
        idx = int(rng.integers(0, n_tasks))
        return evals[idx](gene_vec5, rng)

    return eval_once


# ---------------------------------------------------------------------------
# negative control: kernel 中立 task (delayed_recall, 全 kernel 飽和)
# ---------------------------------------------------------------------------


def make_negative_eval(*, dim: int = 8, projection_seed: int = 0) -> EvalOnce:
    """kernel 中立 eval = delayed_recall (BG6 で全 kernel R²≈1.0 飽和)。smooth/中立対照。"""
    return make_kernel_eval_once(delayed_recall_task(), dim=dim, projection_seed=projection_seed)


# ---------------------------------------------------------------------------
# behavior descriptor (real/negative は kernel_behavior。A1 は theta-only)。
# ---------------------------------------------------------------------------


def theta_only_behavior(gene_vec5: np.ndarray) -> np.ndarray:
    """A1 ablation 用 theta-only behavior (1D = theta L1 のみ, kernel_id を抜く).

    kernel_behavior の第2成分 (theta L1) のみを返す。kernel_id niche を持たない MAP-E に対応し、
    ③優位が kernel diversity 維持由来か探索量/theta niching 由来かを切り分ける (pre-reg §3 A1)。
    """
    return kernel_behavior(gene_vec5)[1:2]  # shape (1,)


def theta_only_behavior_bounds() -> tuple[np.ndarray, np.ndarray]:
    """theta-only behavior の box bounds (kernel_behavior_bounds の theta L1 軸のみ)."""
    lo, hi = kernel_behavior_bounds()
    return lo[1:2], hi[1:2]


# ---------------------------------------------------------------------------
# strict gate (pre-reg §1.6 / strict_compare 準拠) — selection_lab.compare の上位互換
# ---------------------------------------------------------------------------


@dataclass
class GateResult:
    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    diff: float
    win_rate: float
    wilcoxon_p: float
    paired_sign_delta: float
    n_seeds: int
    passes: bool  # diff>0 ∧ p<alpha ∧ n>=min_seeds ∧ |δ|>=min_effect


def strict_gate(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str,
    name_b: str,
    *,
    alpha: float = GATE_ALPHA,
    min_seeds: int,
    min_effect: float = GATE_MIN_EFFECT,
) -> GateResult:
    """a が b を上回るかの完全 strict gate (pre-reg §1.6).

    diff>0 ∧ 片側 Wilcoxon p<alpha ∧ n_seeds>=min_seeds ∧ |paired_sign_delta|>=min_effect。
    smoke では min_seeds を緩める (n=3-5) が p/δ/diff は本番同基準。本番 (BG9-5) は min_seeds=15。
    """
    deltas = scores_a - scores_b
    diff = float(np.mean(deltas))
    p = _paired_p(scores_a, scores_b)
    delta = _cliff_delta(deltas)
    passes = bool(
        diff > 0.0
        and p < alpha
        and len(scores_a) >= min_seeds
        and abs(delta) >= min_effect
    )
    return GateResult(
        name_a=name_a, name_b=name_b,
        mean_a=float(np.mean(scores_a)), mean_b=float(np.mean(scores_b)),
        diff=diff, win_rate=float(np.mean(scores_a >= scores_b)),
        wilcoxon_p=p, paired_sign_delta=delta, n_seeds=len(scores_a), passes=passes,
    )


# ---------------------------------------------------------------------------
# CRN method runner (exp_knob_sweep.run_methods_crn を踏襲、本 module に再実装)
#   selection_lab の map_elites/random_restart_hillclimb/panmictic_ga は read-only import。
# ---------------------------------------------------------------------------

_BASE_METHODS = ("map_elites", "rr_hillclimb", "panmictic_ga", "random")


def run_methods_crn(
    eval_once: EvalOnce,
    behavior: Callable[[np.ndarray], np.ndarray],
    *,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    behavior_bounds: tuple[np.ndarray, np.ndarray],
    grid_shape: tuple[int, ...],
    n_evals: int,
    n_seeds: int,
    honest_n_trials: int,
    sigma: float,
    base_seed: int,
    init_batch: int = ME_INIT_BATCH,
) -> dict[str, np.ndarray]:
    """4 method (MAP-E/RR-hillclimb/panmictic-GA/random) を CRN paired で n_seeds 走らせ honest 再評価.

    - 進化 RNG = SeedSequence([base, method_idx, s]) で method×seed を一意・無相関化。
    - honest 再評価 RNG = SeedSequence([base, 7, s]) を **全 method 共通 (CRN)** に
      → index s の 4 method が同一 fresh タスク draw で採点 = matched replicate (paired Wilcoxon 前提充足)。
    全 method equal budget (同一 n_evals)。fresh-seed honest 再評価で elitism 持越し artifact 排除。
    """
    out: dict[str, list[float]] = {m: [] for m in _BASE_METHODS}

    def _evo_rng(method_idx: int, s: int) -> np.random.Generator:
        return np.random.default_rng(np.random.SeedSequence([base_seed, method_idx, s]))

    def _honest(gene: np.ndarray, s: int) -> float:
        return honest_reevaluate(
            eval_once, gene, n_trials=honest_n_trials,
            rng=np.random.default_rng(np.random.SeedSequence([base_seed, 7, s])),
        )

    for s in range(n_seeds):
        r_me = map_elites(
            eval_once, behavior, dim=dim, bounds=bounds, behavior_bounds=behavior_bounds,
            grid_shape=grid_shape, n_evals=n_evals, init_batch=min(init_batch, n_evals),
            sigma=sigma, rng=_evo_rng(0, s),
        )
        out["map_elites"].append(_honest(r_me.best_gene, s))

        r_rr = random_restart_hillclimb(
            eval_once, dim=dim, bounds=bounds, n_evals=n_evals, sigma=sigma,
            restart_patience=max(10, n_evals // 20), rng=_evo_rng(1, s),
        )
        out["rr_hillclimb"].append(_honest(r_rr.best_gene, s))

        r_ga = panmictic_ga(
            eval_once, dim=dim, bounds=bounds, n_evals=n_evals, pop_size=20,
            tournament_k=3, sigma=sigma, elitism=1, rng=_evo_rng(2, s),
        )
        out["panmictic_ga"].append(_honest(r_ga.best_gene, s))

        rrng = _evo_rng(3, s)
        cands = [bounds[0] + (bounds[1] - bounds[0]) * rrng.random(dim) for _ in range(n_evals)]
        best = max(cands, key=lambda g: eval_once(g, rrng))
        out["random"].append(_honest(best, s))

    return {m: np.array(out[m]) for m in _BASE_METHODS}


def run_theta_only_mapelites(
    eval_once: EvalOnce,
    *,
    dim: int,
    bounds: tuple[np.ndarray, np.ndarray],
    n_evals: int,
    n_seeds: int,
    honest_n_trials: int,
    sigma: float,
    base_seed: int,
    init_batch: int = ME_INIT_BATCH,
) -> np.ndarray:
    """A1 ablation: theta-only behavior (kernel_id 抜き) の MAP-E を CRN で n_seeds 走らせ honest 再評価.

    base_seed / honest 再評価 seed は run_methods_crn と完全に合わせる (同一 CRN replicate で paired 比較可)。
    進化 RNG は別 method_idx (=4) を使い run_methods_crn の 0-3 と衝突させない。
    """
    out: list[float] = []
    for s in range(n_seeds):
        r = map_elites(
            eval_once, theta_only_behavior, dim=dim, bounds=bounds,
            behavior_bounds=theta_only_behavior_bounds(), grid_shape=GRID_SHAPE_THETA_ONLY,
            n_evals=n_evals, init_batch=min(init_batch, n_evals), sigma=sigma,
            rng=np.random.default_rng(np.random.SeedSequence([base_seed, 4, s])),
        )
        out.append(honest_reevaluate(
            eval_once, r.best_gene, n_trials=honest_n_trials,
            rng=np.random.default_rng(np.random.SeedSequence([base_seed, 7, s])),  # 同一 CRN
        ))
    return np.array(out)


# ---------------------------------------------------------------------------
# 1 substrate を回す (4 method + A1 ablation + strict gate)
# ---------------------------------------------------------------------------


@dataclass
class SubstrateResult:
    substrate: str
    means: dict[str, float]
    stds: dict[str, float]
    reach_rate: dict[str, float] | None  # positive control のみ (target basin 到達率)
    gates: dict[str, dict]               # MAP-E vs 各 baseline の strict gate
    n_baselines_beaten: int
    load_bearing: bool                   # 3 baseline 全勝 (strict gate)
    a1_theta_only_mean: float            # A1 theta-only MAP-E honest mean
    a1_vs_baselines: dict[str, dict]     # A1 theta-only MAP-E vs 各 baseline の strict gate
    a1_n_baselines_beaten: int           # A1 が勝った baseline 数
    a1_attribution: str                  # ③優位の帰属 (kernel diversity / 探索量) の判定文
    raw_scores: dict[str, list[float]]   # 生数値 (JSON 保存)


def run_substrate(
    name: str,
    eval_once: EvalOnce,
    behavior: Callable[[np.ndarray], np.ndarray],
    behavior_bounds: tuple[np.ndarray, np.ndarray],
    *,
    n_evals: int,
    n_seeds: int,
    honest_n_trials: int,
    base_seed: int,
    min_seeds_gate: int,
    is_positive_control: bool = False,
) -> SubstrateResult:
    """1 基質に対し 4 method を CRN で回し、MAP-E が 3 baseline 全勝か + A1 ablation を測る.

    kernel-channel 数 (dim=8) は呼び出し側で eval_once 構築時に焼き込み済 (固定射影 P)。本関数の
    GA は GA_DIM=5 の gene ベクトルを進化させるため kernel-channel dim を引数に取らない。
    """
    lo, hi = kernel_ga_bounds()
    bounds = (lo, hi)
    baselines = ("rr_hillclimb", "panmictic_ga", "random")

    # NOTE: GA は dim=5 の gene ベクトル ([kernel_id, theta0..3]) を進化させる。kernel-channel 数
    # (dim 引数, =8) は eval_once の内部 (固定射影 P) に既に焼き込まれており GA dim ではない。
    res = run_methods_crn(
        eval_once, behavior, dim=GA_DIM, bounds=bounds, behavior_bounds=behavior_bounds,
        grid_shape=GRID_SHAPE, n_evals=n_evals, n_seeds=n_seeds,
        honest_n_trials=honest_n_trials, sigma=SIGMA, base_seed=base_seed,
    )
    means = {m: float(res[m].mean()) for m in _BASE_METHODS}
    stds = {m: float(res[m].std()) for m in _BASE_METHODS}

    reach = None
    if is_positive_control:
        reach = {m: float(np.mean(res[m] > POS_GLOBAL_PEAK_PROXY)) for m in _BASE_METHODS}

    gates: dict[str, dict] = {}
    n_beaten = 0
    for b in baselines:
        g = strict_gate(res["map_elites"], res[b], "map_elites", b, min_seeds=min_seeds_gate)
        gates[b] = asdict(g)
        n_beaten += int(g.passes)
    load_bearing = n_beaten == len(baselines)

    # --- A1 ablation: theta-only MAP-E (kernel_id 抜き) ---
    a1 = run_theta_only_mapelites(
        eval_once, dim=GA_DIM, bounds=bounds, n_evals=n_evals, n_seeds=n_seeds,
        honest_n_trials=honest_n_trials, sigma=SIGMA, base_seed=base_seed,
    )
    a1_gates: dict[str, dict] = {}
    a1_beaten = 0
    for b in baselines:
        g = strict_gate(a1, res[b], "map_elites_theta_only", b, min_seeds=min_seeds_gate)
        a1_gates[b] = asdict(g)
        a1_beaten += int(g.passes)

    # ③優位の帰属判定 (pre-reg §3 A1)
    if not load_bearing:
        attribution = "N/A — full MAP-E が 3 baseline 全勝でないため帰属判定の前提なし。"
    elif a1_beaten < n_beaten:
        attribution = (
            f"kernel diversity 維持由来 (H7 支持): kernel_id を抜くと優位が減衰 "
            f"(full beaten={n_beaten}/3 → theta-only beaten={a1_beaten}/3)。"
        )
    else:
        attribution = (
            f"探索量/theta niching 由来 (H7 弱い): kernel_id を抜いても優位が残る "
            f"(full beaten={n_beaten}/3, theta-only beaten={a1_beaten}/3)。kernel-union 特有でない。"
        )

    return SubstrateResult(
        substrate=name, means=means, stds=stds, reach_rate=reach,
        gates=gates, n_baselines_beaten=n_beaten, load_bearing=load_bearing,
        a1_theta_only_mean=float(a1.mean()), a1_vs_baselines=a1_gates,
        a1_n_baselines_beaten=a1_beaten, a1_attribution=attribution,
        raw_scores={m: res[m].tolist() for m in _BASE_METHODS} | {"map_elites_theta_only": a1.tolist()},
    )


# ---------------------------------------------------------------------------
# smoke driver (pre-reg §6 step BG9-5 の前段: harness validity 確認)
# ---------------------------------------------------------------------------

# smoke 既定 (小 budget・小 seed)。本番 (BG9-5) は n_seeds=15, n_evals=4000。
SMOKE_DEFAULTS = dict(
    n_evals=800,          # smoke (pre-reg 例 600-1000)。本番 4000。
    n_seeds=5,            # smoke (pre-reg 例 3-5)。本番 15。
    honest_n_trials=20,   # fresh held-out 再評価 trial。本番 30。
    dim=8,                # kernel channel 数。
    base_seed=20260602,
    min_seeds_gate=3,     # smoke では n>=3 で gate 可 (本番 15)。p/δ/diff は本番同基準。
)


def run_smoke(cfg: dict) -> dict:
    """positive → negative → real の順で 3 基質を回し harness validity を確認する (pre-reg §6)."""
    t0 = time.time()
    dim = cfg["dim"]  # kernel-channel 数 = eval 構築時にのみ使う (GA dim ではない)
    common = dict(
        n_evals=cfg["n_evals"], n_seeds=cfg["n_seeds"],
        honest_n_trials=cfg["honest_n_trials"], base_seed=cfg["base_seed"],
        min_seeds_gate=cfg["min_seeds_gate"],
    )
    bh = kernel_behavior
    bb = kernel_behavior_bounds()

    substrates: dict[str, SubstrateResult] = {}

    # 1) positive control を **先に** (harness validity の核)
    substrates["positive"] = run_substrate(
        "positive", make_kernel_barrier_eval(d=1.0), bh, bb,
        is_positive_control=True, **common,
    )
    # 2) negative control (kernel 中立, false-positive チェック)
    substrates["negative"] = run_substrate(
        "negative", make_negative_eval(dim=dim, projection_seed=0), bh, bb, **common,
    )
    # 3) real (一次像のみ, 断定しない)
    substrates["real"] = run_substrate(
        "real", make_real_multitask_eval(REAL_TASKS, dim=dim, projection_seed=0), bh, bb, **common,
    )
    wall = time.time() - t0

    pos = substrates["positive"]
    neg = substrates["negative"]
    real = substrates["real"]

    # === harness validity 判定 (pre-reg §2.1/§4) ===
    if pos.load_bearing:
        harness_verdict = "VALID — positive control で MAP-E が 3 baseline 全勝 = harness が③を検出可。"
        harness_code = "VALID"
    else:
        harness_verdict = (
            f"INVALID — positive control で MAP-E が 3 baseline 全勝せず "
            f"(beaten={pos.n_baselines_beaten}/3) = harness が③を検出できない → BG9 = N/A 方向。"
        )
        harness_code = "INVALID"

    # negative control: 優位なし (null) を確認できたか
    neg_false_positive = neg.load_bearing
    neg_note = (
        "WARNING — negative control で MAP-E が 3 baseline 全勝 = false-positive リスク (N/A 警告)。"
        if neg_false_positive else
        "OK — negative control で MAP-E が③優位を持たない (null 確認)。"
    )

    # real: smoke 一次像 (断定しない)
    real_note = (
        f"real smoke 一次像 (断定禁止): MAP-E が baseline を beaten={real.n_baselines_beaten}/3。"
        + (" 勝ち気配あり (要本番 >=15 seed 確認)。"
           if real.n_baselines_beaten >= 1 else " 勝ち気配なし/僅差。")
    )

    return {
        "meta": {
            "stage": "BG9-4/BG9-5(smoke) — harness 実装 + smoke validity 確認",
            "preregistration": "BG9_PREREGISTRATION.md",
            "config": cfg,
            "fixed_params": {
                "n_bins": list(GRID_SHAPE), "sigma": SIGMA, "me_init_batch": ME_INIT_BATCH,
                "gate_alpha": GATE_ALPHA, "gate_min_effect": GATE_MIN_EFFECT,
                "real_tasks": list(REAL_TASKS),
            },
            "wall_clock_sec": round(wall, 1),
            "honest_note": (
                "smoke 水準 (小 budget・小 seed)。本検定 (>=15 seed, n_evals=4000) は次段 BG9-5。"
                "fitness は固定射影 P + per-gene held-out ridge readout で kernel 寄与を分離。"
                "kernel dynamics は対角 mock (full kernel 非主張)。positive control は kernel_id 軸の"
                "合成 deceptive corridor (実 fitness でない harness validity 専用)。"
                "整いすぎた③成立は内訳を疑う (feedback_benchmark_honest_disclosure)。"
            ),
        },
        "substrates": {k: asdict(v) for k, v in substrates.items()},
        "verdicts": {
            "harness_validity_code": harness_code,
            "harness_validity": harness_verdict,
            "positive_load_bearing": pos.load_bearing,
            "positive_n_baselines_beaten": pos.n_baselines_beaten,
            "positive_a1_attribution": pos.a1_attribution,
            "negative_false_positive": neg_false_positive,
            "negative_note": neg_note,
            "real_smoke_primary_image": real_note,
            "real_n_baselines_beaten": real.n_baselines_beaten,
        },
    }


def _print_summary(res: dict) -> None:
    print("=" * 88)
    print("BG9 harness smoke — positive(validity) / negative(null) / real(一次像)")
    print("=" * 88)
    for name in ("positive", "negative", "real"):
        s = res["substrates"][name]
        print(f"\n[{name}] n_baselines_beaten={s['n_baselines_beaten']}/3 "
              f"load_bearing={'YES' if s['load_bearing'] else 'no'}")
        for m in _BASE_METHODS:
            extra = ""
            if s["reach_rate"] is not None:
                extra = f" reach={s['reach_rate'][m]:.2f}"
            print(f"    {m:18s}: mean={s['means'][m]:.4f} std={s['stds'][m]:.4f}{extra}")
        for b in ("rr_hillclimb", "panmictic_ga", "random"):
            g = s["gates"][b]
            print(f"      MAP-E vs {b:13s}: diff={g['diff']:+.4f} p={g['wilcoxon_p']:.3g} "
                  f"δ={g['paired_sign_delta']:+.2f} pass={g['passes']}")
        print(f"    A1 theta-only MAP-E: mean={s['a1_theta_only_mean']:.4f} "
              f"beaten={s['a1_n_baselines_beaten']}/3")
        print(f"    A1 attribution: {s['a1_attribution']}")
    v = res["verdicts"]
    print("\n" + "=" * 88)
    print(f"  harness validity [{v['harness_validity_code']}]: {v['harness_validity']}")
    print(f"  negative: {v['negative_note']}")
    print(f"  real: {v['real_smoke_primary_image']}")
    print(f"  wall-clock = {res['meta']['wall_clock_sec']}s")
    print("=" * 88)


def main() -> int:
    _ensure_utf8_stdout()
    ap = argparse.ArgumentParser(description="BG9 harness smoke driver (harness validity 確認)")
    ap.add_argument("--smoke", action="store_true", help="smoke 既定で実行 (既定動作)")
    ap.add_argument("--seeds", type=int, default=None, help="n_seeds 上書き (smoke 既定 5)")
    ap.add_argument("--n-evals", type=int, default=None, help="n_evals 上書き (smoke 既定 800)")
    ap.add_argument("--out", type=str, default=None, help="結果 JSON の出力先 (既定 bg9_smoke_results.json)")
    args = ap.parse_args()

    cfg = dict(SMOKE_DEFAULTS)
    if args.seeds is not None:
        cfg["n_seeds"] = int(args.seeds)
    if args.n_evals is not None:
        cfg["n_evals"] = int(args.n_evals)

    res = run_smoke(cfg)
    out_path = Path(args.out) if args.out else Path(__file__).resolve().parent / "bg9_smoke_results.json"
    out_path.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    _print_summary(res)
    print(f"written: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
