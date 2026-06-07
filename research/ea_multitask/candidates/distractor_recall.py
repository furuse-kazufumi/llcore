# SPDX-License-Identifier: Apache-2.0
"""distractor_recall — 干渉下の選択的記憶タスク族 (E-A ③検定 候補).

T-maze / DelayedRecall (memory_tasks.DelayedRecallTask) の干渉版。

契約 (task_mixture.Task / make_eval_once と同一):
    generate(rng) -> (inputs: (seq_len, in_dim), target: (out_dim,))

ダイナミクス (記憶必須 — 過去を保持しないと最終 state から解けない):
- t=0 に cue (±1) を呈示する。
- 中間区間 (cue 直後 〜 最終時刻の手前) に、各時刻 ``distractor_prob`` の確率で
  振幅 ``a`` の **ランダム ± パルス distractor** を散布して保持を妨害する。
  distractor は cue と同じ 1 チャネルに乗る (cue/distractor を区別する追加信号は無い)
  ため、reservoir は「最初の値だけを選択的に保持し、後続の大振幅ノイズを無視」する
  時定数配分を進化させねばならない。
- target = 最初に呈示した cue (±1)。

regime 軸 = distractor 振幅 ``a``。
  a が大きいほど干渉が強く保持が難しい → 干渉下の選択的記憶。a が cue 振幅 (=1) を
  超える regime (a=1.1) は distractor が cue より大きく、単純な「最大絶対値を覚える」
  ヒューリスティックを破綻させる狙い。

honest 設計メモ:
- in_dim は **全 regime 同一 (=1)** — TaskMixture / 単一 reservoir 要件。
- seq_len も全 regime 同一に固定 (a だけを動かして「振幅」効果を分離。seq_len を一緒に
  動かすと交絡する)。
- 地形は手で作らない: distractor の位置・符号・cue の符号は全て rng 由来。
- 最終時刻には distractor を置かない (最終 state を distractor で汚さないため。cue 再生も
  しない — 再生すると記憶不要になり too-easy になる)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistractorRecallTask:
    """t=0 の cue を、中間の振幅 a の ± distractor 散布をかいくぐって最終時刻に思い出す.

    Attributes
    ----------
    seq_len : int
        系列長。全 regime 共通 (a だけを regime 軸にする)。
    distractor_amp : float
        distractor パルスの振幅 a (regime 軸)。cue 振幅は 1.0 固定。
    distractor_prob : float
        中間各時刻に distractor を置く確率。
    in_dim : int
        入力次元 = 1 (cue と distractor は同一チャネル)。全 regime 同一。
    out_dim : int
        出力次元 = 1 (cue ±1)。
    """

    seq_len: int = 30
    distractor_amp: float = 0.5
    distractor_prob: float = 0.4
    in_dim: int = 1
    out_dim: int = 1

    def __post_init__(self) -> None:
        if self.seq_len < 3:
            raise ValueError(f"seq_len={self.seq_len} must be >= 3 (cue + gap + read)")
        if self.distractor_amp < 0:
            raise ValueError(f"distractor_amp={self.distractor_amp} must be >= 0")
        if not (0.0 <= self.distractor_prob <= 1.0):
            raise ValueError(
                f"distractor_prob={self.distractor_prob} must be in [0,1]"
            )

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.zeros((self.seq_len, 1), dtype=np.float64)
        cue = float(rng.choice([-1.0, 1.0]))
        inputs[0, 0] = cue
        # 中間区間 [1, seq_len-1) に振幅 a の ± distractor を散布。
        # 最終時刻 (seq_len-1) は汚さない (最終 state を distractor で潰さないため)。
        for t in range(1, self.seq_len - 1):
            if rng.random() < self.distractor_prob:
                sign = float(rng.choice([-1.0, 1.0]))
                inputs[t, 0] = sign * self.distractor_amp
        return inputs, np.array([cue], dtype=np.float64)


__all__ = ["DistractorRecallTask"]
