# SPDX-License-Identifier: Apache-2.0
"""Step B — KernelGenome を eval/進化に接続する research 隔離 adapter.

STAGE_3B_DESIGN.md §6 Step B (項番 4-5) の実装。dim=5 連続 GA ベクトル
``[kernel_id, theta0..theta3]`` を :class:`~kernels.KernelGenome` に往復させ、
memory_tasks の held-out R² を fitness とする ``eval_once(gene_vec5, rng) -> float``
コールバックを作る。既存 ``ea_lab`` / ``selection_lab`` の連続ベクトル operator
(``_clip`` / ``rng.normal(size=5)``) を **無改変流用**できる形 (固定長 union) で返す。

honest 設計 (kernel の寄与を分離する仕組み):

- **入力射影 P (固定 seed)**: memory_tasks の入力 ``(L, in_dim)`` を kernel が回す
  ``(L, dim)`` チャネルへ ``x @ P`` で写す。kernel は **対角スカラ写像** (各チャネル独立・
  同一 theta) なので、射影なしでは全 ``dim`` チャネルが同一軌跡に潰れ readout に信号が無い。
  P は per-task 固定 seed (進化と独立) で、進化対象の theta/kernel_id とは別に固定する
  = reservoir.py の固定 reservoir 構造 (``w_in``) と同じ役割。**P は進化させない**。
- **ridge readout (per-gene, held-out)**: train 系列で gene の final state → target を ridge
  で fit、held-out 系列で R²。reservoir.make_eval_once と同手順 (train/eval を rng の続きで
  別 draw → leakage なし)。readout は gene ごとに fit するが train/test 分離で暗記排除。
- **honest な分離**: 射影 P は固定 seed・readout は held-out で、変動するのは kernel_id + theta
  のみ。よって held-out R² の gene 間差分は **kernel dynamics の寄与に帰属**する
  (射影・readout の自由度ではない)。kernel dynamics は対角 mock (kernels.py のスコープ宣言どおり、
  full kernel 性能は主張しない)。

reservoir.py との対応:
    reservoir : 固定 reservoir 構造 (n_taps, leak/w_in は gene) + per-gene ridge readout
    本 adapter: 固定 入力射影 P (kernel theta/kernel_id が gene) + per-gene ridge readout
両者とも「固定構造 + 進化パラメータ + held-out ridge」で gene の表現力を測る同一思想。

research/ 隔離、src は read-only 流用のみ (StateUpdateGene/run_sequence/fit_ridge_readout)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# research skeleton (kernels.py) を同ディレクトリから import
sys.path.insert(0, str(Path(__file__).resolve().parent))
# src/llcore (read-only 流用: fit_ridge_readout のみ。run_sequence は kernels 経由)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kernels import (  # noqa: E402
    GA_DIM,
    KERNEL_DIMS,
    KERNEL_NAMES,
    KERNEL_THETA_LOWER,
    KERNEL_THETA_UPPER,
    MAX_DIM,
    N_KERNELS,
    KernelGenome,
    run_sequence_kernel,
)
from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402

EvalOnce = Callable[[np.ndarray, np.random.Generator], float]

# 進化と独立な固定 seed の名前空間。入力射影 P の生成にのみ使う (kernel の寄与分離の核)。
PROJECTION_SEED_NS = 90_210_001


# ---------------------------------------------------------------------------
# dim=5 GA ベクトル <-> KernelGenome 往復 + GA bounds (固定長 union)
# ---------------------------------------------------------------------------


def gene_vec_to_genome(gene_vec5: np.ndarray) -> KernelGenome:
    """dim=5 連続 GA ベクトル ``[kernel_id, theta0..theta3]`` → clip 済 KernelGenome.

    ``KernelGenome.from_array`` → ``clipped()`` の薄いラッパ。GA operator が出した生ベクトル
    (clip 前) を安全に genome 化する単一窓口。
    """
    arr = np.asarray(gene_vec5, dtype=np.float64)
    if arr.shape != (GA_DIM,):
        raise ValueError(f"gene_vec5 must be shape ({GA_DIM},), got {arr.shape}")
    return KernelGenome.from_array(arr).clipped()


def kernel_ga_bounds() -> tuple[np.ndarray, np.ndarray]:
    """dim=5 GA 探索の box bounds ``(lo, hi)`` (shape (5,) 各).

    - kernel_id: ``[0, N_KERNELS)`` (floor で kernel index に落ちる連続値)。上端は
      ``N_KERNELS - 1e-9`` (半開区間)。
    - theta[0:3]: **全 kernel の theta 範囲の union** (各座標の min lower / max upper)。
      union を採るのは GA が任意 kernel_id へ動けるよう全 kernel の有効域を覆うため。
      ``clipped()`` が genome 化時に当該 kernel の真の範囲へ再 clip するので緩い box で安全。
    - theta[3] (junk): ``[0, 1]`` (kernels.py の junk clip 規約)。

    GA operator (``_clip(gene, bounds)``) にそのまま渡せる。
    """
    # theta 先頭 3 次元の union box
    lows = np.stack([KERNEL_THETA_LOWER[n] for n in KERNEL_NAMES])  # (n_kernel, 3)
    highs = np.stack([KERNEL_THETA_UPPER[n] for n in KERNEL_NAMES])
    theta_lo = lows.min(axis=0)  # (3,)
    theta_hi = highs.max(axis=0)
    lo = np.concatenate([[0.0], theta_lo, [0.0]]).astype(np.float64)  # (5,)
    hi = np.concatenate([[float(N_KERNELS) - 1e-9], theta_hi, [1.0]]).astype(np.float64)
    return lo, hi


# ---------------------------------------------------------------------------
# 固定 入力射影 P (進化と独立、kernel 寄与分離の核)
# ---------------------------------------------------------------------------


def make_projection(in_dim: int, dim: int, *, seed: int) -> np.ndarray:
    """入力 ``(L, in_dim)`` → kernel チャネル ``(L, dim)`` の固定線形射影 P (shape (in_dim, dim)).

    進化と独立な固定 seed で生成し、**進化中は不変**。各 kernel チャネルに異なる入力刺激を
    与え、対角スカラ kernel の ``dim`` チャネルが互いに区別される軌跡を持つようにする
    (= readout がデコードできる信号を生む)。reservoir の固定 ``w_in`` 構造に対応。

    列を単位ノルム正規化して、kernel に入る刺激スケールを in_dim に依らず一定に保つ。
    """
    rng = np.random.default_rng(np.random.SeedSequence([PROJECTION_SEED_NS, in_dim, dim, seed]))
    P = rng.standard_normal((in_dim, dim))
    # 各 kernel チャネル (列) を単位ノルムに正規化 (入力スケール安定化)
    norms = np.linalg.norm(P, axis=0, keepdims=True)
    P = P / np.maximum(norms, 1e-12)
    return P.astype(np.float64)


# ---------------------------------------------------------------------------
# fitness ブリッジ: KernelGenome → held-out R²
# ---------------------------------------------------------------------------


def _collect_states_targets(
    genome: KernelGenome,
    task: object,
    projection: np.ndarray,
    n: int,
    dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """n 本の task 系列を生成 → 固定 P で射影 → kernel で回し (final_state, target) を収集."""
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(n):
        inputs, target = task.generate(rng)  # inputs: (L, in_dim)
        projected = np.asarray(inputs, dtype=np.float64) @ projection  # (L, dim)
        traj = run_sequence_kernel(genome, projected)  # (L+1, dim)
        states.append(traj[-1])  # 最終時刻の state のみ (reservoir.make_eval_once と同じ)
        targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
    return np.array(states, dtype=np.float64), np.array(targets, dtype=np.float64)


def kernel_fitness(
    gene_vec5: np.ndarray,
    task: object,
    *,
    projection: np.ndarray,
    dim: int,
    n_train: int = 48,
    n_eval: int = 48,
    ridge_lambda: float = 1e-2,
    rng: np.random.Generator,
    clip: bool = True,
) -> float:
    """KernelGenome (dim=5 ベクトル) の held-out R² fitness.

    手順 (reservoir.make_eval_once と同型):
    1. dim=5 ベクトル → ``KernelGenome.from_array().clipped()``。
    2. train 系列: 固定 P で射影 → kernel で回す → final state → target を ridge で fit。
    3. held-out 系列 (rng の続きから draw, train と独立) で予測し R² = 1 - MSE/分散。
    4. ``clip=True`` (既定, GA 選択圧用) なら [0,1] に clip。

    honest: 射影 P は固定・readout は held-out なので、fitness の gene 間差分は kernel
    dynamics (kernel_id + theta) に帰属する。kernel dynamics は対角 mock (full kernel 非主張)。

    Parameters
    ----------
    projection : np.ndarray
        :func:`make_projection` が返す固定射影 (shape (in_dim, dim))。進化中不変。
    """
    if n_train < 1 or n_eval < 1:
        raise ValueError(f"n_train/n_eval must be >= 1, got {n_train}/{n_eval}")
    genome = gene_vec_to_genome(gene_vec5)
    s_tr, y_tr = _collect_states_targets(genome, task, projection, n_train, dim, rng)
    readout = fit_ridge_readout(s_tr, y_tr, ridge_lambda=ridge_lambda)

    s_ev, y_ev = _collect_states_targets(genome, task, projection, n_eval, dim, rng)
    pred = np.atleast_2d(readout(s_ev))  # (n_eval, out_dim)
    if pred.shape[0] != y_ev.shape[0]:  # single-sample readout の保険 (n_eval>=2 では不要)
        pred = pred.reshape(y_ev.shape)
    mse = float(np.mean((pred - y_ev) ** 2))
    var = float(np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
    r2 = 1.0 - mse / max(var, 1e-12)
    return float(np.clip(r2, 0.0, 1.0)) if clip else float(r2)


def make_kernel_eval_once(
    task: object,
    *,
    dim: int = 8,
    n_train: int = 48,
    n_eval: int = 48,
    ridge_lambda: float = 1e-2,
    projection_seed: int = 0,
    clip: bool = True,
) -> EvalOnce:
    """1 task に対する ``eval_once(gene_vec5, rng) -> float`` を作る (ea_lab/selection_lab 互換).

    入力射影 P は ``(task.in_dim, dim)`` を ``projection_seed`` で **固定生成**し、返した
    クロージャ内で不変に共有する (進化と独立 = kernel 寄与分離の核)。返り値は
    ``selection_lab.run_methods_over_seeds`` / ``ea_lab.run_ea_methods_over_seeds`` の
    ``eval_once`` にそのまま渡せる (dim=5 の連続ベクトルを引数に取る)。

    Notes
    -----
    異なる ``projection_seed`` は異なる固定 P = 異なる「固定構造」に対応する。BG6 では
    seed ごとに P を変えて (進化 RNG とは別に) 射影自由度に対する best-kernel 写像の頑健性も
    見る (各 seed の best-kernel が安定か)。
    """
    in_dim = int(getattr(task, "in_dim"))
    projection = make_projection(in_dim, dim, seed=projection_seed)

    def eval_once(gene_vec5: np.ndarray, rng: np.random.Generator) -> float:
        return kernel_fitness(
            gene_vec5, task, projection=projection, dim=dim,
            n_train=n_train, n_eval=n_eval, ridge_lambda=ridge_lambda, rng=rng, clip=clip,
        )

    return eval_once


# ---------------------------------------------------------------------------
# 2D behavior descriptor = (kernel_id 連続値, theta L1 norm) — STAGE_3B_DESIGN §6 項5
# ---------------------------------------------------------------------------


def kernel_behavior(gene_vec5: np.ndarray) -> np.ndarray:
    """MAP-Elites behavior descriptor ``(kernel_id 連続値, theta L1 norm)`` (shape (2,)).

    - 軸1 = kernel_id 連続値 (clip 後)。離散障壁を niche 軸に入れる (DESIGN §5.4)。
    - 軸2 = 当該 kernel が実際に使う theta 先頭 dim の L1 norm (junk 除外)。
    behavior は **連続値 kernel_id を使う** (探索は滑らか、floor は decode 時のみ)。
    """
    g = gene_vec_to_genome(gene_vec5)
    name = g.kernel_name()
    used = g.theta[: KERNEL_DIMS[name]]
    return np.array([float(g.kernel_id), float(np.sum(np.abs(used)))], dtype=np.float64)


def kernel_behavior_bounds() -> tuple[np.ndarray, np.ndarray]:
    """behavior descriptor の box bounds ``(lo, hi)`` (shape (2,) 各, MAP-Elites grid 用).

    - kernel_id: ``[0, N_KERNELS)``。
    - theta L1: ``[0, max_dim_l1]``。max は全 kernel の theta 範囲端点の |.| 和の最大。
    """
    max_l1 = 0.0
    for n in KERNEL_NAMES:
        dim = KERNEL_DIMS[n]
        lo, hi = KERNEL_THETA_LOWER[n][:dim], KERNEL_THETA_UPPER[n][:dim]
        max_l1 = max(max_l1, float(np.sum(np.maximum(np.abs(lo), np.abs(hi)))))
    lo = np.array([0.0, 0.0], dtype=np.float64)
    hi = np.array([float(N_KERNELS) - 1e-9, max_l1], dtype=np.float64)
    return lo, hi


@dataclass(frozen=True)
class BridgeSanity:
    """adapter 健全性 smoke の結果 (検証項目 a/b/c)."""

    all_kernels_finite_fitness: bool  # (a) 4 kernel 全てで finite な fitness
    rwkv_runseq_consistent: bool      # (b) kernel_id=0 経路 == 既存 run_sequence (BG5 spirit)
    same_seed_reproducible: bool      # (c) 同 seed 再現で fitness bit 一致
    per_kernel_fitness: dict          # kernel 名 -> sample fitness (可視化)


def bridge_sanity_check(*, dim: int = 8, seed: int = 20260602) -> BridgeSanity:
    """adapter 健全性 smoke (STAGE_3B 検証 a/b/c).

    (a) ``from_array(rng.normal(size=5)).clipped()`` が 4 kernel 全てで finite fitness。
    (b) kernel_id=0 (rwkv) 経路が既存 run_sequence と整合 (BG5 spirit; 射影後の入力で
        run_sequence_kernel と直接 StateUpdateGene+run_sequence が bit 一致)。
    (c) 同 seed 再現で fitness bit 一致。
    """
    from llcore.state_update import StateUpdateGene, run_sequence  # noqa: E402

    from memory_tasks_import import flipflop_task  # 同ディレクトリ helper (下で定義)

    task = flipflop_task()
    eval_once = make_kernel_eval_once(task, dim=dim, projection_seed=0)

    # (a) 各 kernel id で finite fitness
    per_kernel: dict[str, float] = {}
    all_finite = True
    rng = np.random.default_rng(seed)
    for kidx, name in enumerate(KERNEL_NAMES):
        vec = rng.normal(size=GA_DIM)
        vec[0] = kidx + 0.5  # 当該 kernel に floor で落ちる kernel_id
        f = eval_once(vec, np.random.default_rng(seed + 100 + kidx))
        per_kernel[name] = float(f)
        if not np.isfinite(f):
            all_finite = False

    # (b) BG5 spirit: 射影後入力で rwkv 経路 == StateUpdateGene+run_sequence bit 一致
    proj = make_projection(task.in_dim, dim, seed=0)
    rwkv_ok = True
    rng_b = np.random.default_rng(seed + 7)
    for _ in range(8):
        th = np.array([rng_b.random(), rng_b.uniform(-1, 1), rng_b.uniform(-2, 2), rng_b.random()])
        g = KernelGenome(kernel_id=0.0, theta=th).clipped()
        inputs, _ = task.generate(rng_b)
        projected = np.asarray(inputs, dtype=np.float64) @ proj
        got = run_sequence_kernel(g, projected)
        gene = StateUpdateGene(decay=float(g.theta[0]), mix=float(g.theta[1]),
                               gate_str=float(g.theta[2]))
        ref = run_sequence(projected, gene)
        if not np.array_equal(got, ref):
            rwkv_ok = False
            break

    # (c) 同 seed 再現で fitness bit 一致
    vec = np.random.default_rng(seed + 11).normal(size=GA_DIM)
    f1 = eval_once(vec, np.random.default_rng(seed + 12))
    f2 = eval_once(vec, np.random.default_rng(seed + 12))
    reproducible = bool(f1 == f2)

    return BridgeSanity(
        all_kernels_finite_fitness=bool(all_finite),
        rwkv_runseq_consistent=bool(rwkv_ok),
        same_seed_reproducible=reproducible,
        per_kernel_fitness=per_kernel,
    )


__all__ = [
    "EvalOnce",
    "gene_vec_to_genome",
    "kernel_ga_bounds",
    "make_projection",
    "kernel_fitness",
    "make_kernel_eval_once",
    "kernel_behavior",
    "kernel_behavior_bounds",
    "BridgeSanity",
    "bridge_sanity_check",
]
