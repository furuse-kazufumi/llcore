# SPDX-License-Identifier: Apache-2.0
"""variable_delay_recall — 可変遅延の cue 保持タスク族 (T-maze 風, regime = 遅延長 D).

E-A フェーズ (E_A_DESIGN_multitask_generalization.md) の③検定用候補タスク。

狙い (honest):
- 素の :class:`DelayedRecallTask` (step_c) は遅延区間が完全無情報 (全 0) なので、
  leak≈0 の固定点でほぼ完璧に保持でき held-out R²≈0.999 と **易しすぎる** (Step C 既知)。
- 本タスクは cue 提示後の遅延区間に **微小ノイズ (distractor)** を毎時刻注入する。
  ノイズは ``w_in`` 経由で leaky-integrator の保持状態へ漏れ込み、cue の符号を徐々に
  汚染する。よって「完璧な固定点保持」は成立せず、leak を
  「cue 保持 (leak 小)」と「ノイズ棄却 (leak 大)」の間で釣り合わせる必要がある。
- regime 軸 = **有効遅延長 D** (= seq_len)。D が長いほど蓄積ノイズが増え、最適な
  時定数 (leak) が変わる。よって D ごとに好む leak が異なる = behavior descriptor
  (eff_mem / leak) と整合した **niche 構造**が生じる。
- hold-out = 未学習 D への extrapolation (D の外挿は時定数の外挿を要求する)。

タスク契約 (step_c / TaskMixture と同一):
    generate(rng) -> (inputs: (seq_len, in_dim), target: (out_dim,))
- inputs[0, 0] = cue (±1)。残り時刻の ch0 は N(0, distractor_amp) のノイズ。
- ch1 (in_dim=2 のとき) は cue 区間/遅延区間を通じ独立な N(0, distractor_amp) ノイズ。
  cue は ch0 の t=0 のみに乗るため、reservoir は「t=0 の ch0 符号」を最終状態まで
  保持しなければ解けない (記憶必須。地形は手で作らない)。
- target = cue (±1)。最終状態のみで解く設計 (make_eval_once は run(...)[-1] を使う)。

全 regime で in_dim / out_dim を同一に保つ (TaskMixture が 1 reservoir に入れるため)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VariableDelayRecallTask:
    """cue(±1) を t=0 に与え、distractor ノイズ付きの遅延 D の後、最終時刻に再生する.

    Attributes
    ----------
    seq_len : int
        系列長 = 有効遅延長 D。regime 軸。長いほど蓄積ノイズが増え難しい。
    distractor_amp : float
        遅延区間に注入する N(0, amp) ノイズの標準偏差。cue=±1 に対する SNR を決める。
        0.0 なら素の DelayedRecall (R²≈0.999 と易しい)。
    in_dim : int
        入力次元。1 = cue 単一チャネル。2 = cue チャネル + 独立ノイズチャネル
        (保持を一段難しくする distractor チャネル)。
    out_dim : int
        出力次元 (常に 1 = cue の符号)。
    cue_amp : float
        cue の振幅 (±cue_amp)。target も同符号で ±cue_amp ではなく ±1 に正規化する
        (R² 計算は分散正規化されるため振幅自体は影響しないが、ノイズ比は cue_amp で決まる)。
    """

    seq_len: int = 30
    distractor_amp: float = 0.5
    in_dim: int = 2
    out_dim: int = 1
    cue_amp: float = 1.0

    def __post_init__(self) -> None:
        if self.seq_len < 1:
            raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
        if self.distractor_amp < 0:
            raise ValueError(f"distractor_amp must be >= 0, got {self.distractor_amp}")
        if self.in_dim not in (1, 2):
            raise ValueError(f"in_dim must be 1 or 2, got {self.in_dim}")
        if self.out_dim != 1:
            raise ValueError(f"out_dim must be 1, got {self.out_dim}")

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """1 本の系列を生成する.

        ch0: t=0 に cue (±cue_amp)、t>=1 に N(0, distractor_amp) のノイズ。
        ch1 (in_dim=2 のみ): 全時刻 N(0, distractor_amp) の独立ノイズ (distractor)。
        target: cue の符号 (±1)。
        """
        cue_sign = float(rng.choice([-1.0, 1.0]))
        inputs = np.zeros((self.seq_len, self.in_dim), dtype=np.float64)

        # ch0: 遅延区間 (t>=1) に distractor ノイズ。t=0 は cue で上書き。
        if self.distractor_amp > 0 and self.seq_len > 1:
            inputs[1:, 0] = rng.normal(0.0, self.distractor_amp, size=self.seq_len - 1)
        inputs[0, 0] = cue_sign * self.cue_amp

        # ch1: 全時刻独立ノイズ (in_dim=2 のとき)。cue は一切載せない =
        #   reservoir は ch0 の t=0 を保持するしかなく、記憶必須性が保たれる。
        if self.in_dim == 2 and self.distractor_amp > 0:
            inputs[:, 1] = rng.normal(0.0, self.distractor_amp, size=self.seq_len)

        target = np.array([cue_sign], dtype=np.float64)
        return inputs, target


def make_regimes(
    delays: tuple[int, ...] = (15, 30, 45, 60),
    *,
    distractor_amp: float = 0.5,
    in_dim: int = 2,
    cue_amp: float = 1.0,
) -> list[VariableDelayRecallTask]:
    """遅延長 D を振った regime 群を作る (全て同一 in_dim/out_dim).

    Parameters
    ----------
    delays : tuple[int, ...]
        regime ごとの有効遅延長 D (= seq_len)。
    distractor_amp, in_dim, cue_amp :
        全 regime 共通のタスクパラメータ。
    """
    return [
        VariableDelayRecallTask(
            seq_len=int(d),
            distractor_amp=distractor_amp,
            in_dim=in_dim,
            cue_amp=cue_amp,
        )
        for d in delays
    ]


__all__ = ["VariableDelayRecallTask", "make_regimes"]
