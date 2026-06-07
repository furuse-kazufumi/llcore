# SPDX-License-Identifier: Apache-2.0
"""Neural ODE gene representation + forward Euler discretization.

llcore approach (RWKV-style state update gene) を **連続時間 vector field** に
移植する PoC。3 パラメータ最小の gene を持ち、forward Euler で discretization
する設計。CPU 完結 (numpy のみ)。

Falsifiable 命題に関わる部分:
    連続時間 vector field を低次元 gene (A, W, b) ∈ R^3 で表現し、
    forward Euler (dt=0.01, T=2.0, N=200 step) で 4 次元状態空間を進化させる。

設計判断:
- gene = (A, W, b) スカラー 3 値。各次元に同じ係数を適用 (最小 gene)。
- dim=4: SNN/GNN 拡張時にも揃えるための共通 state 次元。
- vector field: dx/dt = A * x + W * tanh(b * x)
  * 線形項 A*x (stable 領域に bias、A<=0 で漸近安定)
  * 非線形項 W * tanh(b*x) (saturating, 有界)
- 物理範囲 clip (Hurwitz / Lipschitz invariant 検証可能領域):
  * A ∈ [-2, 0]   (negative real part → 線形項単独で stable)
  * W ∈ [-1, 1]   (非線形項の振幅)
  * b ∈ [-2, 2]   (非線形項の傾き、sech^2 <= 1 で抑える)

honest 留保:
- スカラー係数 (3 パラメータ) は llcore RWKV-style と同じ「最小 gene」 (低次元
  探索可能性) を達成するための単純化。多次元 (A ∈ R^{dim×dim}) への拡張は
  Stage 1+ で議論 (Codex Q2 review で eigenvalue 真の条件への乖離を honest 報告)。
- forward Euler は最も単純な ODE 離散化で、dt が荒いと真の vector field の
  Lipschitz を保存しない (Codex Q3 review)。本 PoC では sound 上界として
  ``|A| + |W|*|b|`` を Z3 で検査するため、discretization artifact は別途 G8 で
  測る (analytic Lipschitz と forward Euler effective Lipschitz の乖離率)。

依存: numpy のみ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


# clip 範囲定数 (Z3 verifier と一致させる)
A_LOW, A_HIGH = -2.0, 0.0
W_LOW, W_HIGH = -1.0, 1.0
B_LOW, B_HIGH = -2.0, 2.0

# discretization 既定
DEFAULT_DT = 0.01
DEFAULT_T = 2.0
DEFAULT_N_STEP = 200  # = T / DT
DEFAULT_DIM = 4


@dataclass(frozen=True)
class NeuralODEGene:
    """Neural ODE vector field の 3 パラメータ gene.

    vector field: ``dx/dt = A * x + W * tanh(b * x)``

    Attributes
    ----------
    A : float
        線形項係数 (clip 範囲 [-2, 0]; stable 領域に bias).
    W : float
        非線形項振幅 (clip 範囲 [-1, 1]).
    b : float
        tanh 引数の傾き (clip 範囲 [-2, 2]).
    """

    A: float
    W: float
    b: float

    def clipped(self) -> "NeuralODEGene":
        """clip 範囲に収めた新 gene を返す."""
        return NeuralODEGene(
            A=float(np.clip(self.A, A_LOW, A_HIGH)),
            W=float(np.clip(self.W, W_LOW, W_HIGH)),
            b=float(np.clip(self.b, B_LOW, B_HIGH)),
        )

    def as_array(self) -> np.ndarray:
        """(3,) numpy array (mutation / crossover 用)."""
        return np.array([self.A, self.W, self.b], dtype=np.float64)

    @classmethod
    def from_array(cls, arr: Iterable[float]) -> "NeuralODEGene":
        """3 要素 array から復元."""
        a = np.asarray(list(arr), dtype=np.float64)
        if a.shape != (3,):
            raise ValueError(f"expected shape (3,), got {a.shape}")
        return cls(A=float(a[0]), W=float(a[1]), b=float(a[2]))

    def analytic_lipschitz_upper(self) -> float:
        """vector field の analytic Lipschitz 上界 |A| + |W|*|b| を返す.

        ``f(x) = A*x + W*tanh(b*x)`` の Jacobian:
            J(x) = A + W*b*sech^2(b*x)
        |sech^2| <= 1 より:
            ||J(x)|| <= |A| + |W| * |b|

        sound (= 真の Lipschitz 以上). この上界を Z3 が独立計算した値と一致させる
        ことで sanity check が成立 (G1 / G8).
        """
        g = self.clipped()
        return abs(g.A) + abs(g.W) * abs(g.b)

    def hurwitz_test(self) -> float:
        """平衡点 x=0 近傍の Jacobian 線形化値 A + W*b を返す.

        J(0) = A + W*b*sech^2(0) = A + W*b (sech^2(0)=1).
        Hurwitz stability (eigenvalue 実部 < 0) のスカラー 1D 簡約:
            A + W*b < 0 → 局所安定.

        honest 留保 (Codex Q2): 真の Hurwitz は多次元 J(0) の固有値実部条件で、
        本 PoC スカラー版は dim=4 でも同じ係数を全次元に適用するため A+W*b の
        sign が dim=4 行列 (A+Wb)*I_4 の eigenvalue (重根) と一致。
        多次元化 (A ∈ R^{4×4}) では結論が変わる (Stage 1+ で議論).
        """
        g = self.clipped()
        return g.A + g.W * g.b


def vector_field(gene: NeuralODEGene, x: np.ndarray) -> np.ndarray:
    """vector field f(x) = A*x + W*tanh(b*x).

    gene の係数を全次元に同じく適用 (最小 gene 仕様).

    Parameters
    ----------
    gene : NeuralODEGene
        clip 推奨 (関数内で clipped() 適用).
    x : np.ndarray
        shape (dim,) の状態ベクトル.

    Returns
    -------
    np.ndarray
        shape (dim,) の dx/dt.
    """
    g = gene.clipped()
    return g.A * x + g.W * np.tanh(g.b * x)


def forward_euler(
    gene: NeuralODEGene,
    x0: np.ndarray,
    *,
    T: float = DEFAULT_T,
    N: int = DEFAULT_N_STEP,
) -> np.ndarray:
    """forward Euler で N step 軌道 trajectory を生成.

    x_{n+1} = x_n + dt * f(x_n), dt = T / N.

    Parameters
    ----------
    gene : NeuralODEGene
        vector field 係数.
    x0 : np.ndarray
        shape (dim,) の初期状態.
    T : float
        積分時間 (default 2.0).
    N : int
        step 数 (default 200). dt = T/N = 0.01.

    Returns
    -------
    np.ndarray
        shape (N+1, dim) の軌道. trajectory[0] = x0, trajectory[N] = x(T).

    honest 留保:
    - forward Euler は陽解法で stability region 限定。dt > 2/|A| 等で発散 risk。
      本 PoC は A ∈ [-2, 0], dt=0.01 → 2/|A|>=1 で安全。
    - Codex Q3 review で discretization artifact 言及 (G8 で測る).
    """
    if N < 1:
        raise ValueError(f"N must be >= 1, got {N}")
    if T <= 0:
        raise ValueError(f"T must be > 0, got {T}")
    dt = T / N
    x = np.asarray(x0, dtype=np.float64).copy()
    if x.ndim != 1:
        raise ValueError(f"x0 must be 1D, got shape {x.shape}")
    traj = np.zeros((N + 1, x.shape[0]), dtype=np.float64)
    traj[0] = x
    for n in range(N):
        x = x + dt * vector_field(gene, x)
        traj[n + 1] = x
    return traj


def empirical_lipschitz(
    gene: NeuralODEGene,
    *,
    dim: int = DEFAULT_DIM,
    n_samples: int = 64,
    range_: float = 1.0,
    rng: np.random.Generator | None = None,
) -> float:
    """vector field の **経験的 Lipschitz** を測定.

    sample 点 x_i (uniform in [-range_, range_]^dim) で
    ||f(x_i) - f(x_j)|| / ||x_i - x_j|| の最大値を取る (有限差分版 Lipschitz).

    analytic Lipschitz (|A| + |W|*|b|) と乖離が無いか G8 で確認.
    """
    if rng is None:
        rng = np.random.default_rng(12345)
    xs = rng.uniform(-range_, range_, size=(n_samples, dim))
    fs = np.array([vector_field(gene, x) for x in xs])
    max_ratio = 0.0
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            dx = xs[i] - xs[j]
            df = fs[i] - fs[j]
            norm_dx = float(np.linalg.norm(dx))
            if norm_dx < 1e-12:
                continue
            ratio = float(np.linalg.norm(df) / norm_dx)
            if ratio > max_ratio:
                max_ratio = ratio
    return max_ratio


__all__ = [
    "A_HIGH",
    "A_LOW",
    "B_HIGH",
    "B_LOW",
    "DEFAULT_DIM",
    "DEFAULT_DT",
    "DEFAULT_N_STEP",
    "DEFAULT_T",
    "NeuralODEGene",
    "W_HIGH",
    "W_LOW",
    "empirical_lipschitz",
    "forward_euler",
    "vector_field",
]
