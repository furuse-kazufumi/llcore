# SPDX-License-Identifier: Apache-2.0
"""Per-gene ridge readout — landscape の平坦化を解消する公正な fitness (CPU 手順 2).

背景 (docs/poc/EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md §7b):
- :class:`~llcore.fitness.tasks.FixedReadout` は **seed 固定の random 線形写像**で、
  gene が保持する信号に適応しない。結果として「どの gene でも fitness がほぼ同じ」
  = landscape が平坦化し、進化が探索すべき構造が消える (本番 CopyTask 上位 20 遺伝子の
  真の spread=0.0007 / 天井 ~0.2 プラトー)。診断は **固定 random readout が平坦化の
  最大要因**と結論づけた。
- 切り分け診断では、readout を最適化 (共進化) すると copy d=8 で landscape に構造が出現し
  GA が同予算 random を有意に上回った (p=0.0005, Cliff δ=+0.62)。

本 module はその「readout 最適化」を **per-gene ridge 回帰 (closed-form, held-out 評価)**
として本番 fitness に配線する。各 gene について:

1. **train** sequence で gene を回し、final state → target の線形 readout を ridge で fit。
2. **held-out** sequence で予測し、R² (= 1 - MSE/分散) を fitness とする。

これは reservoir computing / echo state network の標準評価 (fixed dynamics + 学習 readout)
であり、gene の表現力 (state が task 信号を線形デコード可能な形で保持するか) を測る。
train/eval を分離するため readout の暗記 (leakage) は構造的に排除される。

honest 留保:
- これは依然 "**probe-based fitness**" であり gene の純粋 fitness ではない。ただし readout が
  各 gene に適応する点で fixed readout より「gene の真の表現力」に近い。
- AdditionTask は ``||sum x||`` の非線形量で、この評価設定では線形 readout の held-out
  R² が負 (mean 予測以下) に留まる (診断: 最適化しても null)。state は tanh 非線形の出力なので
  これは「数学的にデコード不能」の証明ではなく **この設定・このサンプルでの観測**。本 fitness が
  addition で平坦なのは task 選別 (CPU 手順 5) の判断材料になる。
- R² は held-out で負になりうる (mean 予測より悪い) ため ``clip=True`` (既定) で [0, 1] に clip。
  clip 下限は「mean 予測と同等以下 = 選択に使えない」を 0 に潰す honest な床。**ただし clip 後の
  0.0 は raw R²<0 を潰した値で「raw=0 の信号皆無」とは識別不能** (Codex pair-review High finding
  2026-05-30)。信号の有無を診断するときは ``clip=False`` で raw R² の符号・spread を見ること。

semver: 新規 module 追加のみ。既存シンボル不変 ([[feedback_implementation_status_record]])。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from llcore.state_update import StateUpdateGene, run_sequence

# ridge fitness を falsification harness (honest_eval.evolution_vs_random) に渡すための
# eval_once コールバック型: (gene, rng) -> float の 1 回の確率的評価。
RidgeEvalOnce = Callable[[object, np.random.Generator], float]


@dataclass(frozen=True)
class RidgeReadout:
    """ridge 回帰で fit された線形 readout (bias 込み).

    Attributes
    ----------
    weight_aug : np.ndarray
        shape (state_dim + 1, out_dim)。最終行が bias。
        予測は ``[state, 1] @ weight_aug``。
    ridge_lambda : float
        L2 正則化係数 (bias は非正則化)。
    state_dim : int
        入力 state 次元。
    out_dim : int
        出力 (target) 次元。
    """

    weight_aug: np.ndarray
    ridge_lambda: float
    state_dim: int
    out_dim: int

    def __call__(self, state: np.ndarray) -> np.ndarray:
        """state を readout する.

        ``state`` が (state_dim,) なら (out_dim,) を、(N, state_dim) なら (N, out_dim) を返す。
        """
        arr = np.asarray(state, dtype=np.float64)
        single = arr.ndim == 1
        s = np.atleast_2d(arr)  # (N, state_dim)
        if s.shape[1] != self.state_dim:
            raise ValueError(
                f"state last dim {s.shape[1]} != readout state_dim {self.state_dim}"
            )
        aug = np.concatenate([s, np.ones((s.shape[0], 1))], axis=1)  # (N, D+1)
        pred = aug @ self.weight_aug  # (N, out_dim)
        return pred[0] if single else pred


def fit_ridge_readout(
    states: np.ndarray,
    targets: np.ndarray,
    *,
    ridge_lambda: float = 1e-2,
) -> RidgeReadout:
    """final state → target の線形 readout を ridge 回帰 (closed-form) で fit する.

    解は正規方程式 ``(A^T A + λ P) W = A^T Y`` (A=bias 拡張 state, P=bias を除く単位行列)。
    bias 列は正則化しない (標準的な ridge 実装)。

    Parameters
    ----------
    states : np.ndarray
        shape (N, state_dim) — 各 sample の final state。
    targets : np.ndarray
        shape (N,) or (N, out_dim) — 対応する target。
    ridge_lambda : float
        L2 正則化係数 (>0 で病的共線性に頑健)。

    Returns
    -------
    RidgeReadout
    """
    S = np.asarray(states, dtype=np.float64)
    Y = np.asarray(targets, dtype=np.float64)
    if S.ndim != 2:
        raise ValueError(f"states must be 2D (N, state_dim), got {S.shape}")
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.shape[0] != S.shape[0]:
        raise ValueError(f"states N={S.shape[0]} != targets N={Y.shape[0]}")
    if ridge_lambda < 0:
        raise ValueError(f"ridge_lambda must be >= 0, got {ridge_lambda}")

    n, d = S.shape
    aug = np.concatenate([S, np.ones((n, 1))], axis=1)  # (N, D+1)
    penalty = np.eye(d + 1)
    penalty[-1, -1] = 0.0  # bias は正則化しない
    gram = aug.T @ aug + ridge_lambda * penalty  # (D+1, D+1)
    rhs = aug.T @ Y  # (D+1, out_dim)
    # solve は singular でも lstsq fallback で安定化 (ridge_lambda=0 の病的ケース)。
    try:
        weight_aug = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:  # pragma: no cover - λ>0 では到達しにくい
        weight_aug = np.linalg.lstsq(gram, rhs, rcond=None)[0]
    return RidgeReadout(
        weight_aug=weight_aug,
        ridge_lambda=float(ridge_lambda),
        state_dim=d,
        out_dim=Y.shape[1],
    )


def _collect_states_targets(
    gene: StateUpdateGene,
    task: object,
    n: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """gene を n 本の sequence で回し (final_state, target) を収集する."""
    states: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for _ in range(n):
        inputs, target = task.generate(rng)
        trajectory = run_sequence(inputs, gene)
        states.append(trajectory[-1])
        targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
    return np.array(states, dtype=np.float64), np.array(targets, dtype=np.float64)


def ridge_fitness(
    gene: StateUpdateGene,
    task: object,
    *,
    n_train: int = 64,
    n_eval: int = 64,
    ridge_lambda: float = 1e-2,
    rng: np.random.Generator,
    clip: bool = True,
) -> float:
    """gene の per-gene ridge readout fitness (held-out R²).

    手順:
    1. ``n_train`` 本で readout を fit (train set)。
    2. ``n_eval`` 本の **held-out** で予測し R² = 1 - MSE / target分散 を計算。
    3. ``clip=True`` (既定、進化の選択圧として使う場合) なら [0, 1] に clip。

    train/eval を別 sequence で引くため readout の暗記 (leakage) は起こらない。
    決定論性は呼出側で ``rng`` の seed を固定して管理する。

    Parameters
    ----------
    clip : bool
        ``True`` (既定) → fitness として [0, 1] に clip (GA の選択圧用)。
        ``False`` → **raw R²** をそのまま返す (負値あり)。

        honest 注 (Codex pair-review 2026-05-30 High finding): clip 後の 0.0 は
        「mean 予測以下 (raw R²<0)」を潰した値であり、「raw R²=0 = 信号皆無」とは
        識別できない。「線形 readout が有用信号を出すか」を診断する用途では
        ``clip=False`` で raw R² を見て、負値か・spread があるかを確認すること。

    Returns
    -------
    float
        ``clip=True`` なら [0, 1]、``clip=False`` なら raw R² (負値含む)。
        gene の state が target を線形デコード可能なほど高い。
    """
    if n_train < 1 or n_eval < 1:
        raise ValueError(f"n_train/n_eval must be >= 1, got {n_train}/{n_eval}")
    s_tr, y_tr = _collect_states_targets(gene, task, n_train, rng)
    readout = fit_ridge_readout(s_tr, y_tr, ridge_lambda=ridge_lambda)

    s_ev, y_ev = _collect_states_targets(gene, task, n_eval, rng)
    pred = readout(s_ev)  # (n_eval, out_dim)
    if pred.ndim == 1:
        pred = pred[:, None]
    mse = float(np.mean((pred - y_ev) ** 2))
    # held-out target の分散 (= mean 予測の MSE) を正規化分母に使う = R²。
    var = float(np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
    r2 = 1.0 - mse / max(var, 1e-12)
    return float(np.clip(r2, 0.0, 1.0)) if clip else float(r2)


def make_ridge_eval_once(
    task: object,
    *,
    n_train: int = 64,
    n_eval: int = 64,
    ridge_lambda: float = 1e-2,
) -> RidgeEvalOnce:
    """falsification harness 用の ``eval_once(gene, rng) -> float`` を作る.

    返り値は :func:`llcore.evolution.honest_eval.evolution_vs_random` の ``eval_once``
    にそのまま渡せる。各呼出で ``rng`` から fresh な train/eval sequence を引くため、
    honest 再評価 (進化 rng と独立な fresh seed) の前提を満たす。

    Examples
    --------
    >>> from llcore.fitness import CopyTask, make_ridge_eval_once
    >>> from llcore.evolution.honest_eval import evolution_vs_random
    >>> eval_once = make_ridge_eval_once(CopyTask(state_dim=8, out_dim=8))
    >>> result = evolution_vs_random(eval_once, n_seeds=15)  # doctest: +SKIP
    """

    def eval_once(gene: object, rng: np.random.Generator) -> float:
        return ridge_fitness(
            gene, task,
            n_train=n_train, n_eval=n_eval, ridge_lambda=ridge_lambda, rng=rng,
        )

    return eval_once


__all__ = [
    "RidgeReadout",
    "RidgeEvalOnce",
    "fit_ridge_readout",
    "ridge_fitness",
    "make_ridge_eval_once",
]
