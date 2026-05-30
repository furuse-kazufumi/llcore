# SPDX-License-Identifier: Apache-2.0
"""機構 parallel_gated — K 個の独立 reservoir + 要素積ゲーティング (多細胞の非線形分業).

梯子段1 の問い: 単一 leaky reservoir + 線形 ridge readout は delayed_parity (5-bit XOR)
を解けない (Minsky-Papert の床、held-out R²≈0.016)。本機構はその床を **reservoir 間の
要素ごとの積 (multiplicative gating)** で外せるかを検証する。

機構の定義:
    K=4 個の独立 :class:`~reservoir.LeakyDelayLineReservoir` (各 n_taps=6) を並列に走らせ、
    各 reservoir の最終状態 h^k (k=1..K) を得る。readout が見る特徴は

        feature = concat([h^1, ..., h^K])                  # 線形項 (= 単に大きい reservoir)
                  ⊕ concat_{i<j}( h^i ⊙ h^j )              # ペア間 要素積 (乗法的相互作用)

    つまり「各 reservoir の状態の連結」に加えて、reservoir ペア間の **要素積** を ridge
    readout の追加特徴にする。

原理 (なぜ XOR の床が外れうるか):
    parity / XOR は a XOR b = (a≠b) であり、線形分離不能だが、入力に **乗法的相互作用項
    a*b** を加えると線形分離可能になる (古典的な XOR の特徴拡張)。各 reservoir が過去ビット
    の異なる線形写像 (時定数違い) を状態に保持するなら、reservoir 間の要素積 h^i ⊙ h^j は
    「異なる過去ビット表現の積」= 乗法的相互作用項を近似し、単一線形 readout の床を外せる
    可能性がある。これを「多細胞の非線形分業」(各 reservoir = 細胞、積 = 細胞間の非線形
    結合) として検証する。

honest 注 (誤帰属の回避):
    床が外れたとしても、それは **reservoir の表現力 (dynamics)** の貢献ではなく、
    **readout 側の二次特徴 (h^i ⊙ h^j)** が XOR を線形分離可能にしただけかもしれない。
    すなわち本質は quadratic readout であり、attribution は ``readout`` になりうる。
    実験スクリプト側で「同条件の単層 n_taps=8 baseline」と比較し、さらに積項の寄与か
    単なる規模 (concat の次元増) かを切り分ける必要がある (exp_mech_parallel_gated.py)。

gene = 各 reservoir の (leak, w_in) を連結 (各 reservoir の gene_dim = n_taps + n_taps*in_dim)。
全 reservoir は同一 (n_taps, in_dim)。reservoir 間で gene を共有せず独立に進化させる。

research/ 隔離。src は read-only 流用のみ (fit_ridge_readout, 非変更)。step_c の
LeakyDelayLineReservoir / gene_bounds を import 流用 (改変禁止)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

# src/llcore への相対パス (mech_parallel_gated.py → ladder1_multi_reservoir → research → llcore/src)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# step_c_memory_tasks (LeakyDelayLineReservoir, gene_bounds) を流用するため path 追加。
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step_c_memory_tasks"))

from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402
from reservoir import (  # noqa: E402
    LeakyDelayLineReservoir,
    gene_bounds as _single_gene_bounds,
)


@dataclass(frozen=True)
class ParallelGatedReservoir:
    """K 個の独立 leaky reservoir + ペア間要素積ゲーティング (機構 parallel_gated).

    各 reservoir は同一 ``(n_taps, in_dim)`` を持ち、gene は reservoir ごとに独立。
    最終状態の特徴空間は「全 reservoir 状態の連結」+「reservoir ペア間の要素積」。

    Attributes
    ----------
    n_reservoirs : int
        並列 reservoir 数 K。機構定義は K=4。
    n_taps : int
        各 reservoir の隠れユニット数。機構定義は 6。
    in_dim : int
        外部入力 x[t] の次元 (= task.in_dim)。
    """

    n_reservoirs: int = 4
    n_taps: int = 6
    in_dim: int = 1

    # 単一 reservoir 雛形 (全 reservoir で gene レイアウト共通)。__post_init__ で確定。
    _proto: LeakyDelayLineReservoir = field(default=None, init=False, compare=False)

    def __post_init__(self) -> None:
        if self.n_reservoirs < 1:
            raise ValueError(f"n_reservoirs must be >= 1, got {self.n_reservoirs}")
        if self.n_taps < 1:
            raise ValueError(f"n_taps must be >= 1, got {self.n_taps}")
        if self.in_dim < 1:
            raise ValueError(f"in_dim must be >= 1, got {self.in_dim}")
        proto = LeakyDelayLineReservoir(n_taps=self.n_taps, in_dim=self.in_dim)
        object.__setattr__(self, "_proto", proto)

    @property
    def per_reservoir_gene_dim(self) -> int:
        """1 reservoir あたりの gene 次元 = n_taps + n_taps*in_dim."""
        return self._proto.gene_dim

    @property
    def gene_dim(self) -> int:
        """全体 gene 次元 = K × per_reservoir_gene_dim (reservoir 独立)."""
        return self.n_reservoirs * self.per_reservoir_gene_dim

    @property
    def n_pairs(self) -> int:
        """要素積を取る reservoir ペア数 = C(K, 2)."""
        return self.n_reservoirs * (self.n_reservoirs - 1) // 2

    @property
    def feature_dim(self) -> int:
        """readout が見る特徴次元.

        - 線形項: K × n_taps (全 reservoir 状態の連結)
        - 積項:   C(K,2) × n_taps (ペアごとに要素積、同次元なので n_taps 本)
        """
        return self.n_reservoirs * self.n_taps + self.n_pairs * self.n_taps

    def _split_gene(self, gene: np.ndarray) -> list[np.ndarray]:
        """全体 gene を K 個の reservoir gene に等分割する."""
        gene = np.asarray(gene, dtype=np.float64)
        if gene.shape[0] != self.gene_dim:
            raise ValueError(
                f"gene dim {gene.shape[0]} != expected {self.gene_dim}"
            )
        d = self.per_reservoir_gene_dim
        return [gene[k * d : (k + 1) * d] for k in range(self.n_reservoirs)]

    def features(self, gene: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """系列を K reservoir に流し、最終状態から特徴ベクトルを構成する.

        Parameters
        ----------
        gene : np.ndarray
            shape (gene_dim,)。
        inputs : np.ndarray
            shape (T, in_dim)。

        Returns
        -------
        feature : np.ndarray
            shape (feature_dim,)。concat([h^1..h^K]) ⊕ concat_{i<j}(h^i ⊙ h^j)。
            各 h は tanh 飽和のため有限、積も有限 → 全体 finite 保証。
        """
        sub_genes = self._split_gene(gene)
        finals: list[np.ndarray] = []
        for g in sub_genes:
            # 各 reservoir を独立に走らせ最終時刻の状態を取得 (n_taps,)。
            finals.append(self._proto.run(g, inputs)[-1])

        # 線形項: 全 reservoir 状態の連結。
        linear = np.concatenate(finals)  # (K*n_taps,)

        # 積項: 全ペア (i<j) の要素積。乗法的相互作用 = XOR を線形分離可能にする狙い。
        prods: list[np.ndarray] = []
        for i, j in combinations(range(self.n_reservoirs), 2):
            prods.append(finals[i] * finals[j])  # (n_taps,)

        if prods:
            return np.concatenate([linear] + prods)
        return linear  # K=1 (ペアなし) では線形項のみ

    def random_gene(self, rng: np.random.Generator) -> np.ndarray:
        """gene_bounds の範囲で一様乱数 gene を生成する."""
        lo, hi = gene_bounds(self)
        return lo + (hi - lo) * rng.random(self.gene_dim)


def gene_bounds(res: ParallelGatedReservoir) -> tuple[np.ndarray, np.ndarray]:
    """gene 探索範囲 = 単一 reservoir の bounds を K 回連結する.

    各 reservoir は step_c の :func:`reservoir.gene_bounds` と同じ値域
    (leak_raw∈[-4,4] → sigmoid 後 0.018〜0.982、w_in∈[-2,2]) を持つ。

    Returns
    -------
    lo, hi : np.ndarray
        それぞれ shape (gene_dim,)。
    """
    lo1, hi1 = _single_gene_bounds(res._proto)
    lo = np.concatenate([lo1] * res.n_reservoirs)
    hi = np.concatenate([hi1] * res.n_reservoirs)
    return lo, hi


def make_eval_once(
    res: ParallelGatedReservoir,
    task: object,
    *,
    n_train: int = 64,
    n_eval: int = 64,
    ridge_lambda: float = 1e-2,
):
    """ridge readout による held-out R² 評価コールバックを作る (既存資産と同契約).

    train/eval を **別 draw** で引くため readout の暗記 (leakage) は構造的に起こらない。
    特徴は :meth:`ParallelGatedReservoir.features` (線形項 + 積項)。
    fitness = clip([0, 1]) の held-out R²。

    Returns
    -------
    eval_once : Callable[[np.ndarray, np.random.Generator], float]
        ``(gene, rng) -> float`` の評価関数。
    """
    def _collect(gene: np.ndarray, n: int, rng: np.random.Generator):
        feats: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for _ in range(n):
            inputs, target = task.generate(rng)
            feats.append(res.features(gene, inputs))  # 最終状態由来の特徴
            targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
        return np.array(feats, dtype=np.float64), np.array(targets, dtype=np.float64)

    def eval_once(gene: np.ndarray, rng: np.random.Generator) -> float:
        # train で readout を fit。
        s_tr, y_tr = _collect(gene, n_train, rng)
        readout = fit_ridge_readout(s_tr, y_tr, ridge_lambda=ridge_lambda)
        # eval は train の続きから draw → train 列とは独立 (leakage なし)。
        s_ev, y_ev = _collect(gene, n_eval, rng)
        pred = np.atleast_2d(readout(s_ev))
        mse = float(np.mean((pred - y_ev) ** 2))
        var = float(np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
        r2 = 1.0 - mse / max(var, 1e-12)
        return float(np.clip(r2, 0.0, 1.0))

    return eval_once


__all__ = [
    "ParallelGatedReservoir",
    "gene_bounds",
    "make_eval_once",
]
