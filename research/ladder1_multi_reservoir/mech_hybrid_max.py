# SPDX-License-Identifier: Apache-2.0
"""機構 hybrid_max — 梯子段1 の「最大表現力」anchor 基質.

Step C verdict (2026-05-30): 単一 leaky-delay-line reservoir + 線形 ridge readout は
delayed_parity (5-bit XOR) を解けない (基質の床, 全 method held-out R²≈0.016)。これは
Minsky-Papert の意味で「単一 reservoir は XOR を線形分離可能な特徴を作れない」床である。

本 module は **3 機構を全部入れした最大表現力構成 (anchor)** を実装する:

1. **DeepReservoir の深さ** — 各「枝 (branch)」を K 層スタックにし、層間 tanh 非線形合成で
   高次の時間特徴を立ち上げる (DeepESN: Gallicchio & Micheli 2017)。
2. **parallel_gated の乗法結合** — 独立な 2 本の reservoir 枝 A / B を並列に走らせ、最終時刻の
   状態を **要素積 h_A ⊙ h_B** で乗法ゲートする。乗法は線形読み出しに「2 因子の AND/XOR 様
   相互作用」を持ち込む (gating: LSTM/GRU、bilinear pooling の系譜)。
3. **quadratic_readout の 2 次特徴** — readout に渡す特徴ベクトルを `[h, h⊙h(対角), 乗法ゲート,
   選抜ペア積]` に拡張する。線形 ridge のまま「2 次までの相互作用」を表現でき、parity の
   非線形分離面を線形 readout で張れる可能性を最大化する。

設計意図 (honest):
- これは「**解けるかどうかの上限 (天井)**」を測るための anchor である。最大構成が held-out で
  parity を解けなければ「CPU reservoir + ridge パラダイムでは parity は解けない」強い証拠になる。
  解ければ、どの機構が効いたか (深さ / 乗法 / 2次特徴) を切り分ける材料になる。
- 最大構成は overfit / readout 依存リスクが最も高い。よって評価は必ず held-out (train/eval を
  別 draw) R² で行い、attribution は機構の性質に照らして正直に付ける。例えば 2 次特徴だけで
  解けるなら、それは reservoir のダイナミクスではなく **readout の表現力** が床を外したのであり
  attribution='readout' とすべきである。

構造 (1 枝 = 1 DeepReservoir 相当):
    branch g ∈ {A, B}:
        layer 0:   u_0[t] = x[t]
        layer k>0: u_k[t] = h^g_{k-1}[t]
        h^g_k[t] = (1-a^g_k) ⊙ h^g_k[t-1] + a^g_k ⊙ tanh(W^g_k u_k[t] + h^g_k[t-1])
    final base 状態 = concat([全枝・全層の h[T-1]])           (= 深さ + 並列)
    gate 状態      = h_A_last_layer[T-1] ⊙ h_B_last_layer[T-1]  (= 乗法結合)
    readout 特徴   = quadratic_expand(concat([base, gate]))     (= 2 次特徴)

研究 (research/) 隔離。src は read-only 流用のみ (fit_ridge_readout, 非変更)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# src/llcore への相対パス
#   mech_hybrid_max.py → ladder1_multi_reservoir → research → llcore/src
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """数値安定な sigmoid。exp overflow を防ぐため引数を [-500, 500] にクリップ."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0)))


# quadratic_readout でペア積を全 O(D^2) 取ると state_dim が爆発し ridge が不安定/遅くなる。
# 「全部入れ」の趣旨を保ちつつ計算可能性を担保するため、ペア積は乗法ゲート由来の高情報
# ブロック (gate 状態) と base の **先頭 PAIR_CAP 次元** に限定する。対角 (自乗) は全次元取る。
PAIR_CAP = 12


def _quadratic_expand(feat: np.ndarray, *, pair_cap: int = PAIR_CAP) -> np.ndarray:
    """1D 特徴ベクトルを 2 次特徴に拡張する (quadratic_readout の核).

    生成する特徴:
    - 元の 1 次特徴 feat               … D 次元
    - 自乗 (対角項) feat**2            … D 次元
    - 先頭 min(D, pair_cap) 次元の上三角ペア積 feat_i*feat_j (i<j) … C(m,2) 次元

    全 O(D^2) ペアは state_dim 爆発を招くため先頭 pair_cap 次元に限定 (module docstring 参照)。
    線形 ridge のまま「2 次までの相互作用」を表現可能にし、parity の非線形分離面を線形
    readout で張れるようにする。

    Parameters
    ----------
    feat : np.ndarray
        shape (D,) の 1 次特徴。
    pair_cap : int
        ペア積を取る先頭次元数の上限。

    Returns
    -------
    np.ndarray
        2 次拡張した特徴ベクトル (1D)。finite (入力が finite なら積も finite)。
    """
    feat = np.asarray(feat, dtype=np.float64).ravel()
    parts = [feat, feat * feat]
    m = min(feat.shape[0], pair_cap)
    if m >= 2:
        iu, ju = np.triu_indices(m, k=1)
        parts.append(feat[iu] * feat[ju])
    return np.concatenate(parts)


def _quadratic_dim(base_dim: int, *, pair_cap: int = PAIR_CAP) -> int:
    """quadratic_expand 後の特徴次元を返す (readout の state_dim 計算用)."""
    m = min(base_dim, pair_cap)
    n_pairs = m * (m - 1) // 2
    return base_dim + base_dim + n_pairs


@dataclass(frozen=True)
class HybridMaxReservoir:
    """hybrid_max 機構 — 並列 2 枝 × K 層スタック × 乗法ゲート × 2 次 readout 特徴.

    最大表現力 anchor。各「枝」は DeepReservoir 相当の K 層 leaky-integrator スタック。
    2 枝を並列に走らせ、最終層の状態を要素積でゲートし、全状態 + ゲートを 2 次特徴へ拡張する。

    Attributes
    ----------
    layer_taps : tuple[int, ...]
        1 枝あたりの各層の隠れユニット数。len が層数 K。両枝で共通構造。
        例 (8, 8) = 各枝 2 層 × 8 taps。
    in_dim : int
        外部入力 x[t] の次元 (= task.in_dim)。
    use_gate : bool
        True で乗法ゲート (h_A_last ⊙ h_B_last) を特徴に加える (parallel_gated の核)。
    use_quadratic : bool
        True で readout 特徴を 2 次拡張する (quadratic_readout の核)。
    pair_cap : int
        2 次拡張のペア積上限 (state_dim 爆発抑制)。
    """

    layer_taps: tuple[int, ...] = (8, 8)
    in_dim: int = 1
    use_gate: bool = True
    use_quadratic: bool = True
    pair_cap: int = PAIR_CAP

    # 各層の入力次元 (層0=in_dim, 層k=前層 taps)。枝 A/B で同一構造なので 1 本ぶんを保持。
    _layer_in_dims: tuple[int, ...] = field(default=(), init=False, compare=False)

    def __post_init__(self) -> None:
        if len(self.layer_taps) == 0:
            raise ValueError("layer_taps must have at least one layer")
        if any(n <= 0 for n in self.layer_taps):
            raise ValueError(f"all layer_taps must be positive: {self.layer_taps}")
        in_dims = [self.in_dim]
        for n_prev in self.layer_taps[:-1]:
            in_dims.append(n_prev)
        object.__setattr__(self, "_layer_in_dims", tuple(in_dims))

    # ----- 構造プロパティ -----

    @property
    def n_layers(self) -> int:
        """1 枝あたりの層数 K."""
        return len(self.layer_taps)

    @property
    def n_branches(self) -> int:
        """並列枝数 (hybrid_max は常に 2 枝)."""
        return 2

    @property
    def taps_per_branch(self) -> int:
        """1 枝の全層 taps の総和."""
        return int(sum(self.layer_taps))

    @property
    def _branch_gene_dim(self) -> int:
        """1 枝の gene 次元 = Σ_k (leak[n_k] + W_in[n_k × in_dim_k])."""
        total = 0
        for n_k, d_k in zip(self.layer_taps, self._layer_in_dims):
            total += n_k + n_k * d_k
        return total

    @property
    def gene_dim(self) -> int:
        """全 gene 次元 = 2 枝ぶん (枝 A の gene + 枝 B の gene)."""
        return self.n_branches * self._branch_gene_dim

    @property
    def _base_feature_dim(self) -> int:
        """2 次拡張前の base 特徴次元 = 全枝・全層状態 (+ gate があればその次元).

        gate = 最終層状態の要素積なので次元は layer_taps[-1]。
        """
        dim = self.n_branches * self.taps_per_branch
        if self.use_gate:
            dim += self.layer_taps[-1]
        return dim

    @property
    def feature_dim(self) -> int:
        """readout が見る最終特徴次元 (2 次拡張後 or base)."""
        base = self._base_feature_dim
        return _quadratic_dim(base, pair_cap=self.pair_cap) if self.use_quadratic else base

    # ----- gene 展開 -----

    def _branch_slices(self, branch: int):
        """gene 内の指定枝の各層 (n_k, d_k, leak_slice, w_slice) を yield する.

        Parameters
        ----------
        branch : int
            0=枝 A, 1=枝 B。枝 B は枝 A のぶんだけオフセットして格納される。
        """
        off = branch * self._branch_gene_dim
        for n_k, d_k in zip(self.layer_taps, self._layer_in_dims):
            leak_slice = slice(off, off + n_k)
            off += n_k
            w_slice = slice(off, off + n_k * d_k)
            off += n_k * d_k
            yield n_k, d_k, leak_slice, w_slice

    def _run_branch(self, gene: np.ndarray, branch: int, inputs: np.ndarray):
        """1 枝の K 層ダイナミクスを流し (全層状態, 最終層の最終状態) を返す.

        Returns
        -------
        all_last : np.ndarray
            shape (taps_per_branch,) — 全層の最終時刻状態を層順に concat。
        top_last : np.ndarray
            shape (layer_taps[-1],) — 最上層の最終時刻状態 (ゲート用)。
        """
        leaks: list[np.ndarray] = []
        w_ins: list[np.ndarray] = []
        hs: list[np.ndarray] = []
        for n_k, d_k, leak_slice, w_slice in self._branch_slices(branch):
            leaks.append(_sigmoid(gene[leak_slice]))
            w_ins.append(gene[w_slice].reshape(n_k, d_k))
            hs.append(np.zeros(n_k, dtype=np.float64))

        T = inputs.shape[0]
        for t in range(T):
            u = inputs[t]
            for k in range(self.n_layers):
                drive = w_ins[k] @ u
                hs[k] = (1.0 - leaks[k]) * hs[k] + leaks[k] * np.tanh(drive + hs[k])
                u = hs[k]
        all_last = np.concatenate(hs)
        top_last = hs[-1]
        return all_last, top_last

    def features(self, gene: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """gene が定める hybrid_max 機構で系列を流し、readout 特徴ベクトルを返す.

        Parameters
        ----------
        gene : np.ndarray
            shape (gene_dim,)。
        inputs : np.ndarray
            shape (T, in_dim)。

        Returns
        -------
        feat : np.ndarray
            shape (feature_dim,) の 1D 特徴。finite 保証 (tanh 飽和 + 有限積)。
        """
        gene = np.asarray(gene, dtype=np.float64)
        inputs = np.asarray(inputs, dtype=np.float64)

        a_all, a_top = self._run_branch(gene, 0, inputs)
        b_all, b_top = self._run_branch(gene, 1, inputs)

        parts = [a_all, b_all]
        if self.use_gate:
            # 乗法ゲート: 最終層状態の要素積 (2 因子相互作用)。次元は layer_taps[-1]。
            parts.append(a_top * b_top)
        base = np.concatenate(parts)

        if self.use_quadratic:
            return _quadratic_expand(base, pair_cap=self.pair_cap)
        return base

    def random_gene(self, rng: np.random.Generator) -> np.ndarray:
        """gene_bounds の範囲で一様乱数 gene を生成する."""
        lo, hi = gene_bounds(self)
        return lo + (hi - lo) * rng.random(self.gene_dim)


def gene_bounds(res: HybridMaxReservoir) -> tuple[np.ndarray, np.ndarray]:
    """gene 探索範囲。leak_raw∈[-4,4] (sigmoid 後 0.018〜0.982)、W_in∈[-2,2].

    単一 reservoir (reservoir.gene_bounds) / DeepReservoir と同じ値域を 2 枝ぶん連結する。
    """
    los: list[np.ndarray] = []
    his: list[np.ndarray] = []
    for branch in range(res.n_branches):
        for n_k, d_k, _, _ in res._branch_slices(branch):
            los.append(np.full(n_k, -4.0))
            his.append(np.full(n_k, 4.0))
            los.append(np.full(n_k * d_k, -2.0))
            his.append(np.full(n_k * d_k, 2.0))
    return np.concatenate(los), np.concatenate(his)


def make_eval_once(
    res: HybridMaxReservoir,
    task: object,
    *,
    n_train: int = 64,
    n_eval: int = 64,
    ridge_lambda: float = 1e-2,
):
    """ridge readout による held-out R² 評価コールバックを作る (既存基質と同契約).

    train/eval を **別 draw** で引くため readout の暗記 (leakage) は構造的に起こらない。
    fitness = clip([0, 1]) の held-out R²。返り値 ``eval_once(gene, rng) -> float``。

    honest 注: hybrid_max は特徴次元が大きく overfit しやすい。held-out 分離を厳守し、
    train で fit した readout を **未見の eval 系列**で採点することでこのリスクを構造的に
    排除する (train データ自体に対する fit の良さは fitness に一切寄与しない)。
    """
    def _collect(gene: np.ndarray, n: int, rng: np.random.Generator):
        feats: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        for _ in range(n):
            inputs, target = task.generate(rng)
            feats.append(res.features(gene, inputs))
            targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
        return np.array(feats, dtype=np.float64), np.array(targets, dtype=np.float64)

    def eval_once(gene: np.ndarray, rng: np.random.Generator) -> float:
        s_tr, y_tr = _collect(gene, n_train, rng)
        readout = fit_ridge_readout(s_tr, y_tr, ridge_lambda=ridge_lambda)
        s_ev, y_ev = _collect(gene, n_eval, rng)  # train の続きから draw → 独立 held-out
        pred = np.atleast_2d(readout(s_ev))
        mse = float(np.mean((pred - y_ev) ** 2))
        var = float(np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
        r2 = 1.0 - mse / max(var, 1e-12)
        return float(np.clip(r2, 0.0, 1.0))

    return eval_once


__all__ = [
    "HybridMaxReservoir",
    "gene_bounds",
    "make_eval_once",
]
