# SPDX-License-Identifier: Apache-2.0
"""機構 wide_single — 単層 超ワイド reservoir で parity の床が「幅だけ」で外れるか.

梯子段1 の対照機構の一つ。multi_reservoir.py (層間非線形合成=深さ) とは逆に、
**深さを増やさず幅 (n_taps) だけ**を 8 → 24 → 48 → 64 と振り、random search の
到達天井 (held-out max R²) がどこまで上がるかを測る。

動機 (reservoir computing 理論):
- 十分広い random reservoir は universal approximator に近づく (Maass 2002 LSM,
  Jaeger 2001 ESN)。固定 random 射影でも、幅が大きければ非線形特徴 (tanh で歪んだ
  state) の中に XOR を線形分離できる方向が偶然含まれる確率が上がる、というのが
  「幅スケーリングで床が外れる」仮説。
- ただし readout は依然 **線形 ridge**。Minsky-Papert の床 (単一線形分離器は XOR 不可)
  は readout の線形性に由来するので、幅を増やしても readout が線形なら本質的限界は
  残る。幅が効くとすれば「reservoir の非線形ダイナミクスが XOR を線形分離可能な
  state に折り畳む」経路のみ。これが random reservoir で十分起きるかは経験的問題。

honest 帰属 (attribution):
- 床が外れた場合、要因は **reservoir のダイナミクス表現力 (幅由来のランダム射影の豊かさ)**
  であって readout ではない (readout は線形 ridge のまま固定)。よって attribution は
  ``reservoir_expressivity`` または ``width`` 系。
- 一方、幅をいくら増やしても床が外れない場合は、線形 readout の限界 (Minsky-Papert) が
  幅では超えられないことの経験的確認になる。これは「機構が無効」ではなく「幅単独では
  不足」という陰性知見。

実装方針:
- 基質は **既存 reservoir.LeakyDelayLineReservoir をそのまま流用 (改変禁止)**。
  本 module は (a) 幅構成の列挙、(b) random search 天井測定、(c) baseline (n_taps=8) との
  公平比較、を frozen dataclass でまとめる薄いオーケストレーション層。
- gene / 評価のセマンティクスは step_c の単層版と完全に同一 (held-out ridge R²)。
  深さ機構 (DeepReservoir) と公平に比較するため eval パラメータ (n_train/n_eval) も
  揃えられるよう引数化する。

research/ 隔離。src は read-only 流用のみ (fit_ridge_readout 経由、非変更)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# 既存単層基質 (step_c_memory_tasks/reservoir.py) を import 流用する。
# このパス追加は step_c の既存ファイル作法 (parents[N]/'src') と同列の sys.path 操作。
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "step_c_memory_tasks"))  # reservoir, memory_tasks

from reservoir import (  # noqa: E402  (改変禁止・流用のみ)
    LeakyDelayLineReservoir,
    gene_bounds,
    make_eval_once,
)

# 幅スイープの既定構成 (n_taps)。8=床基準、24/48/64=超ワイド。
DEFAULT_WIDTHS: tuple[int, ...] = (8, 24, 48, 64)


@dataclass(frozen=True)
class WideSingleConfig:
    """機構 wide_single の 1 構成 (単層・幅 n_taps の reservoir).

    Attributes
    ----------
    n_taps : int
        単層 reservoir の幅 (隠れユニット数)。これだけを振る。
    in_dim : int
        入力次元 (DelayedParityTask は 1)。
    """

    n_taps: int = 8
    in_dim: int = 1

    # 流用基質を内部に保持 (frozen なので __post_init__ で object.__setattr__)。
    _reservoir: LeakyDelayLineReservoir = field(default=None, init=False, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_taps <= 0:
            raise ValueError(f"n_taps must be positive: {self.n_taps}")
        if self.in_dim <= 0:
            raise ValueError(f"in_dim must be positive: {self.in_dim}")
        res = LeakyDelayLineReservoir(n_taps=self.n_taps, in_dim=self.in_dim)
        object.__setattr__(self, "_reservoir", res)

    @property
    def reservoir(self) -> LeakyDelayLineReservoir:
        """流用している単層 reservoir 本体 (改変禁止)."""
        return self._reservoir

    @property
    def label(self) -> str:
        """構成ラベル (例 '1L-24wide')."""
        return f"1L-{self.n_taps}wide"

    @property
    def gene_dim(self) -> int:
        """gene 次元 = 基質 reservoir の gene_dim (= n_taps + n_taps*in_dim)."""
        return self._reservoir.gene_dim

    @property
    def total_taps(self) -> int:
        """readout が見る状態次元 = n_taps (単層なので幅そのもの)."""
        return self.n_taps

    def random_gene(self, rng: np.random.Generator) -> np.ndarray:
        """bounds 内の一様乱数 gene を生成する (基質に委譲)."""
        return self._reservoir.random_gene(rng)

    def run(self, gene: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """基質ダイナミクスで系列を流し全時刻 state を返す (基質に委譲)."""
        return self._reservoir.run(gene, inputs)

    def make_eval_once(self, task: object, *, n_train: int = 48, n_eval: int = 48,
                       ridge_lambda: float = 1e-2):
        """held-out ridge R² 評価コールバックを作る (基質 make_eval_once に委譲).

        train/eval を別 draw するため readout の暗記 (leakage) は構造的に起こらない。
        深さ機構と公平比較するため n_train/n_eval を引数化 (既定は exp の値に合わせる)。
        """
        return make_eval_once(
            self._reservoir, task,
            n_train=n_train, n_eval=n_eval, ridge_lambda=ridge_lambda,
        )

    def gene_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """gene 探索範囲 (基質 gene_bounds に委譲)."""
        return gene_bounds(self._reservoir)


def random_search_ceiling(
    cfg: WideSingleConfig,
    task: object,
    *,
    n_random: int,
    seed_idx: int,
    n_train: int = 48,
    n_eval: int = 48,
    gene_base: int = 700_001,
    eval_base: int = 900_001,
) -> float:
    """1 seed の random search で到達した max held-out R² を返す.

    公平性の肝 (exp_l1_expressivity と同一作法):
    - gene は ``default_rng(gene_base + seed_idx)`` から連続 draw (seed 固定で再現)。
    - **全 gene を同一 eval データ** (``default_rng(eval_base + seed_idx)`` を毎回張り直し)
      で評価する → gene 間で train/eval が共通になり比較が公平 (誤帰属の回避)。

    Parameters
    ----------
    cfg : WideSingleConfig
        測定する幅構成。
    task : object
        ``generate(rng) -> (inputs, target)`` を持つタスク。
    n_random : int
        random search の gene 本数 (到達天井の推定精度)。
    seed_idx : int
        seed のインデックス (per-seed の独立試行)。

    Returns
    -------
    float
        この seed で random search が到達した held-out max R² (∈ [0, 1])。
    """
    eval_once = cfg.make_eval_once(task, n_train=n_train, n_eval=n_eval)
    gene_rng = np.random.default_rng(gene_base + seed_idx)
    best = 0.0
    for _ in range(n_random):
        gene = cfg.random_gene(gene_rng)
        # 全 gene 同一 train/eval データ列にするため eval rng を毎回張り直す。
        eval_rng = np.random.default_rng(eval_base + seed_idx)
        best = max(best, eval_once(gene, eval_rng))
    return best


def width_ceiling_curve(
    task: object,
    *,
    widths: tuple[int, ...] = DEFAULT_WIDTHS,
    n_random: int = 300,
    n_seeds: int = 8,
    in_dim: int = 1,
    n_train: int = 48,
    n_eval: int = 48,
    gene_base: int = 700_001,
    eval_base: int = 900_001,
) -> dict[int, np.ndarray]:
    """各幅構成について per-seed の random search 天井 (max R²) を測り辞書で返す.

    Returns
    -------
    dict[int, np.ndarray]
        ``{n_taps: per_seed_max_r2 (shape (n_seeds,))}``。
    """
    curve: dict[int, np.ndarray] = {}
    for w in widths:
        cfg = WideSingleConfig(n_taps=w, in_dim=in_dim)
        vals = np.array([
            random_search_ceiling(
                cfg, task,
                n_random=n_random, seed_idx=s,
                n_train=n_train, n_eval=n_eval,
                gene_base=gene_base, eval_base=eval_base,
            )
            for s in range(n_seeds)
        ], dtype=np.float64)
        curve[w] = vals
    return curve


__all__ = [
    "WideSingleConfig",
    "DEFAULT_WIDTHS",
    "random_search_ceiling",
    "width_ceiling_curve",
]
