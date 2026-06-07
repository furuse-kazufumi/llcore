# SPDX-License-Identifier: Apache-2.0
"""複数 reservoir 結合基質 (DeepESN 風スタック) — 梯子段1 の「多細胞」基質.

Step C verdict (2026-05-30) は単一 leaky-delay-line reservoir + ridge readout を
「単細胞レベル基質」と位置づけ、delayed_parity (5-bit XOR) を **基質の床** と判定した
(単一 reservoir は Minsky-Papert の意味で XOR を線形分離可能な特徴を作れず全 method R²≈0)。

本モジュールは梯子段1 = 「複数 reservoir 結合 (多細胞)」を実装する。仮説は
**線形記憶 × 非線形分業** = 浅い層が異なる時定数で過去を保持し、深い層が前層の状態を
tanh で非線形合成することで、ridge (線形) readout からでも XOR を分離可能な特徴空間が
立ち上がるか、というもの (DeepESN: Gallicchio & Micheli 2017)。

構造:
    layer 0:   u_0[t] = x[t]               (in_dim_0 = task.in_dim)
    layer k>0: u_k[t] = h_{k-1}[t]         (in_dim_k = n_taps_{k-1})
    h_k[t] = (1 - a_k) ⊙ h_k[t-1] + a_k ⊙ tanh(W_in_k @ u_k[t] + h_k[t-1])
    readout 状態 = concat([h_0[T-1], ..., h_{K-1}[T-1]])   (全層の最終状態を結合)

単一層 (n_taps=(N,)) は既存 reservoir.LeakyDelayLineReservoir と数式的に等価。K>1 で
層間非線形合成が加わる。gene は層ごとに leak (時定数) と W_in (層入力重み) を持つ。

research/ 隔離。src は read-only 流用のみ (fit_ridge_readout, 非変更)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# src/llcore への相対パス (multi_reservoir.py → ladder1_multi_reservoir → research → llcore/src)
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数値安定な sigmoid。exp overflow を防ぐため引数を [-500, 500] にクリップ."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


@dataclass(frozen=True)
class DeepReservoir:
    """K 層スタック leaky-integrator reservoir (多細胞基質).

    Attributes
    ----------
    layer_taps : tuple[int, ...]
        各層の隠れユニット数。len が層数 K。例 (8, 8, 8) = 3 層 × 8 taps。
    in_dim : int
        外部入力 x[t] の次元 (= task.in_dim)。
    """

    layer_taps: tuple[int, ...] = (8, 8)
    in_dim: int = 1

    # 各層の入力次元 in_dim_k を __post_init__ で確定 (層0=in_dim, 層k=前層 taps)。
    _layer_in_dims: tuple[int, ...] = field(default=(), init=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.layer_taps) == 0:
            raise ValueError("layer_taps must have at least one layer")
        if any(n <= 0 for n in self.layer_taps):
            raise ValueError(f"all layer_taps must be positive: {self.layer_taps}")
        in_dims = [self.in_dim]
        for n_prev in self.layer_taps[:-1]:
            in_dims.append(n_prev)
        # frozen dataclass なので object.__setattr__ で確定する。
        object.__setattr__(self, "_layer_in_dims", tuple(in_dims))

    @property
    def n_layers(self) -> int:
        return len(self.layer_taps)

    @property
    def total_taps(self) -> int:
        """readout が見る状態次元 = 全層 taps の総和."""
        return int(sum(self.layer_taps))

    @property
    def gene_dim(self) -> int:
        """gene 次元 = Σ_k (leak[n_k] + W_in[n_k × in_dim_k])."""
        total = 0
        for n_k, d_k in zip(self.layer_taps, self._layer_in_dims):
            total += n_k + n_k * d_k
        return total

    def _layer_slices(self):
        """gene 内の各層 (leak_raw, w_in) を切り出すスライス境界を yield する.

        Yields
        ------
        (n_k, d_k, leak_slice, w_slice) : tuple
            n_k=層 taps, d_k=層入力次元, leak_slice/w_slice=gene 内インデックス範囲。
        """
        off = 0
        for n_k, d_k in zip(self.layer_taps, self._layer_in_dims):
            leak_slice = slice(off, off + n_k)
            off += n_k
            w_slice = slice(off, off + n_k * d_k)
            off += n_k * d_k
            yield n_k, d_k, leak_slice, w_slice

    def run(self, gene: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """gene が定める K 層ダイナミクスで系列を流し、全層の各時刻状態を返す.

        Parameters
        ----------
        gene : np.ndarray
            shape (gene_dim,)。
        inputs : np.ndarray
            shape (T, in_dim)。

        Returns
        -------
        states : np.ndarray
            shape (T, total_taps)。全層の状態を層順に concat したもの。finite 保証
            (各層 tanh 飽和)。最終時刻 states[-1] を readout 特徴に使う想定。
        """
        gene = np.asarray(gene, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)
        T = inputs.shape[0]

        # 各層の (leak, W_in) を展開し、層状態 h_k を 0 初期化。
        leaks: list[np.ndarray] = []
        w_ins: list[np.ndarray] = []
        hs: list[np.ndarray] = []
        for n_k, d_k, leak_slice, w_slice in self._layer_slices():
            leaks.append(_sigmoid(gene[leak_slice]))
            w_ins.append(gene[w_slice].reshape(n_k, d_k))
            hs.append(np.zeros(n_k, dtype=np.float64))

        states = np.empty((T, self.total_taps), dtype=np.float64)
        for t in range(T):
            u = inputs[t]  # 層0 への入力 = 外部入力
            for k in range(self.n_layers):
                drive = w_ins[k] @ u  # (n_k,)
                hs[k] = (1.0 - leaks[k]) * hs[k] + leaks[k] * np.tanh(drive + hs[k])
                u = hs[k]  # 次層への入力 = 当層状態 (層間非線形合成)
            states[t] = np.concatenate(hs)
        return states

    def random_gene(self, rng: np.random.Generator) -> np.ndarray:
        """gene_bounds の範囲で一様乱数 gene を生成する."""
        lo, hi = gene_bounds(self)
        return lo + (hi - lo) * rng.random(self.gene_dim)


def gene_bounds(res: DeepReservoir) -> tuple[np.ndarray, np.ndarray]:
    """gene 探索範囲。leak_raw∈[-4,4] (sigmoid 後 0.018〜0.982)、W_in∈[-2,2].

    単一 reservoir (reservoir.gene_bounds) と同じ値域を層ごとに連結する。
    """
    los: list[np.ndarray] = []
    his: list[np.ndarray] = []
    for n_k, d_k, _, _ in res._layer_slices():
        los.append(np.full(n_k, -4.0))
        his.append(np.full(n_k, 4.0))
        los.append(np.full(n_k * d_k, -2.0))
        his.append(np.full(n_k * d_k, 2.0))
    return np.concatenate(los), np.concatenate(his)


def make_eval_once(
    res: DeepReservoir,
    task: object,
    *,
    n_train: int = 64,
    n_eval: int = 64,
    ridge_lambda: float = 1e-2,
):
    """ridge readout による held-out R² 評価コールバックを作る (単一 reservoir 版と同契約).

    train/eval を分離して引くため readout の暗記 (leakage) は構造的に起こらない。
    fitness = clip([0, 1]) の held-out R²。

    Returns
    -------
    eval_once : Callable[[np.ndarray, np.random.Generator], float]
    """
    def _collect(gene: np.ndarray, n: int, rng: np.random.Generator):
        states: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for _ in range(n):
            inputs, target = task.generate(rng)
            states.append(res.run(gene, inputs)[-1])  # 最終時刻の全層状態
            targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
        return np.array(states, dtype=np.float64), np.array(targets, dtype=np.float64)

    def eval_once(gene: np.ndarray, rng: np.random.Generator) -> float:
        s_tr, y_tr = _collect(gene, n_train, rng)
        readout = fit_ridge_readout(s_tr, y_tr, ridge_lambda=ridge_lambda)
        s_ev, y_ev = _collect(gene, n_eval, rng)  # train の続きから draw → 独立
        pred = np.atleast_2d(readout(s_ev))
        mse = float(np.mean((pred - y_ev) ** 2))
        var = float(np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
        r2 = 1.0 - mse / max(var, 1e-12)
        return float(np.clip(r2, 0.0, 1.0))

    return eval_once


def make_behavior(res: DeepReservoir):
    """behavior descriptor = (全層平均実効記憶長, 全層 leak 標準偏差).

    層を跨いだ leak (時定数) の 1次・2次モーメントを niche 軸にする。単一 reservoir 版
    make_behavior の多層拡張。Step C と同様 w_in を捨てる近似で、niche 軸の改良は
    ③テスト統合フェーズで検討する (現スコープは時定数分布のみ)。

    Returns
    -------
    behavior : Callable[[np.ndarray], np.ndarray]  # (gene,) -> shape (2,)
    """
    def behavior(gene: np.ndarray) -> np.ndarray:
        leaks: list[np.ndarray] = []
        for _, _, leak_slice, _ in res._layer_slices():
            leaks.append(_sigmoid(gene[leak_slice]))
        leak = np.concatenate(leaks)
        eff_mem = np.mean(1.0 / np.maximum(leak, 1e-3))
        eff_mem_norm = float(np.tanh(eff_mem / 50.0))
        return np.array([eff_mem_norm, float(np.std(leak))], dtype=np.float64)

    return behavior


__all__ = [
    "DeepReservoir",
    "gene_bounds",
    "make_eval_once",
    "make_behavior",
]
