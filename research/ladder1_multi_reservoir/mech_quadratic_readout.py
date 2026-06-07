# SPDX-License-Identifier: Apache-2.0
"""機構 quadratic_readout — 単層 reservoir の最終状態に明示的2次特徴を足した readout.

梯子段1 の問い「複数 reservoir 結合で parity(5-bit XOR) の床が外れるか」に対し、本機構は
**reservoir を増やさず readout を非線形化する対照**を提供する。単層 ``LeakyDelayLineReservoir``
(n_taps=12) の最終状態 ``h`` に対し、readout 入力を::

    phi(h) = [ h_1, ..., h_N,  h_i * h_j  (i <= j の pairwise 積) ]

に拡張して ridge fit する。pairwise 積 (= 二次項) は XOR を線形分離可能にする古典的特徴
(Minsky & Papert 1969 の床は「線形 readout」前提であり、明示的2次特徴を与えると消える)。

honest 対照 (最重要):
- これは **reservoir の表現力 (ダイナミクス) を改善する機構ではない**。reservoir はそのまま、
  readout 側に手で二次特徴を注入しているだけである。よって床が外れて parity が解けても
  「reservoir が parity を解いた」ことには **ならない**。解いたのは readout の二次多項式である。
- したがって floor_lifted=true でも attribution は 'readout' とする (機構の性質に正直に)。
- 比較の公平性のため、reservoir 本体・gene レイアウト・bounds・データ生成は step_c の
  ``LeakyDelayLineReservoir`` と完全に共有する。差分は **最終状態の特徴展開 1 点のみ**。

research/ 隔離。src は read-only 流用のみ (fit_ridge_readout, 非変更)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# src/llcore への相対パス (mech_quadratic_readout.py → ladder1_multi_reservoir → research → llcore/src)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# step_c の単層 reservoir 基質を流用 (改変禁止, import のみ)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step_c_memory_tasks"))

from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402
from reservoir import LeakyDelayLineReservoir  # noqa: E402
from reservoir import gene_bounds as _single_gene_bounds  # noqa: E402


def quadratic_features(h: np.ndarray) -> np.ndarray:
    """状態ベクトル h を [h, pairwise 積 (i<=j)] に展開する.

    1 本 (shape (N,)) でも複数本 (shape (M, N)) でも受ける。

    展開後の次元は ``N + N*(N+1)/2`` (線形項 N + 上三角を含む対称二次項)。
    対角 (i==j) を含めるのは h_i^2 が「単一ユニットの非線形」も供給し、XOR に有効なため。

    Parameters
    ----------
    h : np.ndarray
        shape (N,) または (M, N)。reservoir の最終状態。

    Returns
    -------
    phi : np.ndarray
        入力が (N,) なら (D,)、(M, N) なら (M, D)。D = N + N*(N+1)/2。
        h が finite なら phi も finite (積のみ、発散演算なし)。
    """
    arr = np.asarray(h, dtype=np.float64)
    single = arr.ndim == 1
    s = np.atleast_2d(arr)  # (M, N)
    M, N = s.shape
    # 上三角 (i <= j) のインデックス。対角を含む。
    iu, ju = np.triu_indices(N)
    quad = s[:, iu] * s[:, ju]  # (M, N*(N+1)/2)
    phi = np.concatenate([s, quad], axis=1)  # (M, N + N(N+1)/2)
    return phi[0] if single else phi


@dataclass(frozen=True)
class QuadraticReadoutReservoir:
    """単層 leaky reservoir + 明示的2次 readout 特徴.

    reservoir 本体 (ダイナミクス, gene, bounds) は step_c の :class:`LeakyDelayLineReservoir`
    と完全に同一。本クラスは ``run`` の出力 (= reservoir の状態列) を二次展開する readout
    用 helper ``feature(h)`` を足すだけの薄いラッパである。

    Attributes
    ----------
    n_taps : int
        reservoir の隠れユニット数。機構定義に従い既定 12。
    in_dim : int
        外部入力次元。
    """

    n_taps: int = 12
    in_dim: int = 1

    @property
    def _base(self) -> LeakyDelayLineReservoir:
        """流用する単層 reservoir 基質 (ダイナミクス本体)."""
        return LeakyDelayLineReservoir(n_taps=self.n_taps, in_dim=self.in_dim)

    @property
    def gene_dim(self) -> int:
        """gene 次元は基底 reservoir と同一 (readout 拡張は gene を増やさない)."""
        return self._base.gene_dim

    @property
    def feature_dim(self) -> int:
        """二次展開後の readout 入力次元 = n_taps + n_taps*(n_taps+1)/2."""
        N = self.n_taps
        return N + N * (N + 1) // 2

    def run(self, gene: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """基底 reservoir のダイナミクスで系列を流し、全時刻状態を返す (生の線形状態).

        二次展開は readout 入力作成時 (`feature`) に行う。run は基底と同一なので、
        reservoir の表現力は一切変えていないことが構造的に保証される。
        """
        return self._base.run(gene, inputs)

    def feature(self, h: np.ndarray) -> np.ndarray:
        """最終状態 h を二次展開した readout 入力特徴 phi(h) を返す."""
        return quadratic_features(h)

    def random_gene(self, rng: np.random.Generator) -> np.ndarray:
        """基底 reservoir の bounds で一様乱数 gene を生成する."""
        return self._base.random_gene(rng)


def gene_bounds(res: QuadraticReadoutReservoir) -> tuple[np.ndarray, np.ndarray]:
    """gene 探索範囲 = 基底単層 reservoir と同一 (readout は gene を持たない)."""
    return _single_gene_bounds(res._base)


def make_eval_once(
    res: QuadraticReadoutReservoir,
    task: object,
    *,
    n_train: int = 64,
    n_eval: int = 64,
    ridge_lambda: float = 1e-2,
):
    """二次 readout の held-out R² 評価コールバックを作る (step_c と同契約).

    手順は step_c の make_eval_once と同一だが、ridge fit / 予測の入力を ``res.feature``
    で二次展開する点のみが異なる。train/eval を別 draw するため leakage は構造的になし。

    Returns
    -------
    eval_once : Callable[[np.ndarray, np.random.Generator], float]
        ``(gene, rng) -> float`` の held-out R² (clip [0, 1])。
    """
    def _collect(gene: np.ndarray, n: int, rng: np.random.Generator):
        """n 本の sequence を生成し (二次展開 final feature, target) を収集する."""
        feats: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for _ in range(n):
            inputs, target = task.generate(rng)
            final = res.run(gene, inputs)[-1]  # 最終時刻の生状態 (N,)
            feats.append(res.feature(final))  # 二次展開 (D,)
            targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
        return np.array(feats, dtype=np.float64), np.array(targets, dtype=np.float64)

    def eval_once(gene: np.ndarray, rng: np.random.Generator) -> float:
        # train で二次特徴上の ridge readout を fit
        x_tr, y_tr = _collect(gene, n_train, rng)
        readout = fit_ridge_readout(x_tr, y_tr, ridge_lambda=ridge_lambda)
        # eval は rng の続きから draw → train 列と独立 (held-out, leakage なし)
        x_ev, y_ev = _collect(gene, n_eval, rng)
        pred = np.atleast_2d(readout(x_ev))  # (n_eval, out_dim)
        mse = float(np.mean((pred - y_ev) ** 2))
        var = float(np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
        r2 = 1.0 - mse / max(var, 1e-12)
        return float(np.clip(r2, 0.0, 1.0))

    return eval_once


__all__ = [
    "QuadraticReadoutReservoir",
    "quadratic_features",
    "gene_bounds",
    "make_eval_once",
]
