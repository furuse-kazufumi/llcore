# SPDX-License-Identifier: Apache-2.0
"""BG9-3 — 各 kernel の力学的特徴から第一原理で導いた kernel-favoring task suite.

BG6 で memory_tasks は大半 **kernel 中立** (delayed_recall 全 kernel 飽和 R²≈1.0、
delayed_parity 全 kernel 失敗 R²≈0、flipflop のみ弁別) と判明した
(``bg6_specialist_results.json``)。BG9 の real substrate を inert にしないため、各 kernel が
**構造的に得意なはずの** task を設計する。

honesty 規律 (BG9_PREREGISTRATION §0 / feedback_benchmark_honest_disclosure):
- **③に勝たせるための逆設計をしない**。各 task は kernel の step 式 (``kernels.py``) から
  第一原理で導く。「kernel X が勝つように捻る」のは禁止。
- 対角スカラ mock ゆえ kernel 間の差は「種類」でなく「程度」かもしれない。**実際に弁別するか
  は (2) 強 BG6 = ``bg6_strong.py`` で測る**。各 docstring の「得意」は **仮説** であり、
  捏造して非定数を作らない。弱ければ弱いと正直に報告する。

task 契約 (memory_tasks と同一): ``generate(rng) -> (inputs (L, in_dim), target (out_dim,))``。
入力は有界 |x|<=1 (kernels.py の有界化前提を尊重)。fitness bridge (``kernel_fitness.py``) は
``inputs @ P`` で dim チャネルへ射影 → kernel を対角に回す → **最終 state のみ** を ridge readout。
よって各 task は「最終時刻の state が答えを保持できる」形でなければならない (memory_tasks と同思想)。

なぜ「対角スカラ mock でも kernel 差が出うる」か (第一原理):
fitness bridge は射影後の各チャネル ``c`` に対し ``s_{t+1}^c = step(s_t^c, x_t^c, theta)`` を
回す。step 式は kernel ごとに異なる **非線形性 / ゲート / 累積則** を持つ:
  - rwkv  : s' = decay·s + (1-decay)·tanh(mix·x + gate·s)   … 有界平滑 + 自己回帰 tanh
  - mamba : a=σ(α·x+β); s' = a·s + (1-a)·(gain·x)            … **入力依存ゲート** a(x)
  - hopf  : s' = (1-η)·s + η·tanh(β·(ξ·tanh(s)+x))            … **tanh 二重飽和アトラクタ**
  - linA  : φ=softplus(w·x); s' = lam·s + φ·(v_gain·x)        … **大きさ重み付き累積** φ(x)·x
最終 state が target の十分統計量を保持するには、その task の「保持/忘却/累積/飽和」要求に
step 式の関数形が適合する必要がある。適合度が kernel 間で変われば held-out R² が変わる
(= 弁別)。ただし射影 P が各チャネルに異なる x を与え、ridge が全チャネル線形結合できるため、
**弁別が薄まる可能性は十分にある** (それを (2) で検定)。

各 task の入力は全て |x|<=1。out_dim は readout 安定化のため 1 (memory_tasks と同じ)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# (1) mamba 向き — gated/selective copy (入力依存ゲート a=σ(α·x+β) が要)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SelectiveCopyTask:
    """**mamba 仮説**: gate 信号が立った step の値だけを保持し distractor を無視する.

    力学的根拠 (1 段): mamba の step は ``a=σ(α·x+β)`` の **入力依存ゲート** を持ち、
    gate チャネルが高いとき ``a≈1`` で過去 state を保持、gate が低い (distractor) とき
    ``a≈0`` で書込みを無視 ──「いつ書く/書かないか」を **入力で選択** できる唯一の kernel。
    decay 一定の rwkv / linA や、入力非依存に飽和する hopfield は本来この選択保持が苦手なはず。

    系列構成 (in_dim=2): ch0=value (±0.8 のスカラ)、ch1=gate (1.0 のときのみ "保持せよ")。
    値は複数 step で変わるが、**最後に gate=1 が立った step の value** が target。それ以外の
    value は distractor。gate が立つのは 1〜数回 (ランダム位置)。最終 state が「最後の
    gated value」を選択保持できるかを問う。
    """

    seq_len: int = 24
    in_dim: int = 2
    out_dim: int = 1
    gate_prob: float = 0.18
    value_amp: float = 0.8

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.zeros((self.seq_len, 2), dtype=np.float64)
        # value チャネルは毎 step ランダム (distractor 含む)、|value|<=value_amp
        values = rng.uniform(-self.value_amp, self.value_amp, size=self.seq_len)
        inputs[:, 0] = values
        # gate を最低 1 回は立てる (target 定義の well-posedness 担保)
        gate_steps = [t for t in range(self.seq_len) if rng.random() < self.gate_prob]
        if not gate_steps:
            gate_steps = [int(rng.integers(0, self.seq_len))]
        for t in gate_steps:
            inputs[t, 1] = 1.0
        last_gated = max(gate_steps)
        target = values[last_gated]
        return inputs, np.array([target], dtype=np.float64)


# ---------------------------------------------------------------------------
# (2) hopfield 向き — bistable hold / denoising (tanh アトラクタが ±へ引く)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BistableDenoiseTask:
    """**hopfield 仮説**: ノイズ下で 2 値状態 (±1) を頑健に保持・脱ノイズする.

    力学的根拠 (1 段): hopfield の step は ``tanh(β·(ξ·tanh(s)+x))`` の **二重飽和** を持ち、
    β/ξ>0 のとき state を ±アトラクタへ引き込む ── 入力に乗った小ノイズを **アトラクタが
    吸収** して符号を安定化できる唯一の kernel。線形累積の linA や減衰平滑の rwkv は
    ノイズを平均化はできてもアトラクタ的な「符号への吸着」が無いため、弱い符号証拠を
    増幅しにくいはず。

    系列構成 (in_dim=1): 真の符号 sign∈{-1,+1} を持ち、各 step は
    ``sign·signal_amp + noise`` (noise は ±noise_amp の一様)。**個々の step は符号が
    曖昧なほどノイズが大きい** (signal_amp < noise_amp)。最終 state が真の符号 (target=±1)
    に脱ノイズ収束できるかを問う。アトラクタ無しだと最終 state は平均≒sign·signal_amp の
    弱信号で符号が不安定。
    """

    seq_len: int = 24
    in_dim: int = 1
    out_dim: int = 1
    signal_amp: float = 0.25
    noise_amp: float = 0.75

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        sign = float(rng.choice([-1.0, 1.0]))
        noise = rng.uniform(-self.noise_amp, self.noise_amp, size=self.seq_len)
        stream = sign * self.signal_amp + noise
        # |x|<=1 を保証 (signal_amp+noise_amp=1.0 ちょうどなので clip は安全余地)
        stream = np.clip(stream, -1.0, 1.0)
        inputs = stream.reshape(self.seq_len, 1)
        return inputs, np.array([sign], dtype=np.float64)


# ---------------------------------------------------------------------------
# (3) linear_attn 向き — weighted running-sum (φ=softplus(w·x) の大きさ重み付き累積)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WeightedAccumulationTask:
    """**linear_attn 仮説**: 入力の大きさで重みづけた累積 (Σ φ(x)·x) を最終時刻に答える.

    力学的根拠 (1 段): linA の step は ``φ=softplus(w·x)`` の **大きさ依存ゲイン** で
    ``φ(x)·x`` を ``lam`` 減衰累積する ── 大きい |x| ほど強く足し込む **重み付き running-sum**
    を素直に表せる唯一の kernel (linear attention の Σφ(k)v に対応)。sigmoid 飽和ゲートの
    mamba や tanh 飽和の hopfield/rwkv は大振幅入力で飽和し、線形な重み付き累積を歪めるはず。

    系列構成 (in_dim=1): 各 step は |x|<=1 の一様乱数。target = Σ_t w(x_t)·x_t を
    ``[-1,1]`` に正規化 (w(x)=|x| の大きさ重み)。すなわち **「大きい入力ほど効く合計」** で、
    線形累積 kernel が有利なはず。
    """

    seq_len: int = 20
    in_dim: int = 1
    out_dim: int = 1
    amp: float = 1.0

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        xs = rng.uniform(-self.amp, self.amp, size=self.seq_len)
        inputs = xs.reshape(self.seq_len, 1)
        # 大きさ重み付き累積 (linear-attention 的 Σ |x|·x)。決定論的 target。
        weighted = float(np.sum(np.abs(xs) * xs))
        # 期待 |target| 上限 ~ seq_len で正規化して target を [-1,1] 帯に収める
        # (E[|x|·x]=0, Var 有限。max |Σ| <= seq_len·amp^2 だが実効は √seq_len オーダ)
        norm = self.seq_len * (self.amp ** 2)
        target = float(np.clip(weighted / max(norm, 1e-12), -1.0, 1.0))
        return inputs, np.array([target], dtype=np.float64)


# ---------------------------------------------------------------------------
# (4) rwkv 向き — gated leaky integration / 低域追従 (decay+tanh の有界平滑)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LeakyTrackingTask:
    """**rwkv 仮説**: ゆっくり変わる目標を gated leaky smoothing で低域追従する.

    力学的根拠 (1 段): rwkv の step は ``decay·s + (1-decay)·tanh(...)`` の **指数移動平均
    (leaky integrator)** 構造を持ち、高周波ノイズを減衰しつつ低周波目標へ追従できる
    ── 「過去を一定率で漏らしながら平滑追従」する唯一の純 EMA 型 kernel。入力依存に書込み
    ゲートを開閉する mamba や、アトラクタで離散化する hopfield、減衰累積で和を取る linA は
    連続な低域追従に最適化されていないはず。

    系列構成 (in_dim=1): 低周波の真信号 (ランダム位相の sin、周期 ~ seq_len) に高周波
    ノイズを重畳した stream。**target = 最終時刻の真信号値** (ノイズ除去後の低域成分)。
    最終 state が leaky smoothing で低域を追えるかを問う。|x|<=1 を保証。
    """

    seq_len: int = 28
    in_dim: int = 1
    out_dim: int = 1
    signal_amp: float = 0.55
    noise_amp: float = 0.4

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        # 低周波の真信号: 周期 = seq_len 前後、ランダム位相
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        period = float(rng.uniform(self.seq_len * 0.8, self.seq_len * 1.6))
        t = np.arange(self.seq_len, dtype=np.float64)
        true_signal = self.signal_amp * np.sin(2.0 * np.pi * t / period + phase)
        noise = rng.uniform(-self.noise_amp, self.noise_amp, size=self.seq_len)
        stream = np.clip(true_signal + noise, -1.0, 1.0)
        inputs = stream.reshape(self.seq_len, 1)
        # target = 最終時刻の真信号 (低域追従の到達点)
        target = float(np.clip(true_signal[-1], -1.0, 1.0))
        return inputs, np.array([target], dtype=np.float64)


# ---------------------------------------------------------------------------
# suite レジストリ (名前 -> ファクトリ)。順序固定 (再現性)。
# 各 task の「仮説 kernel」を併記 (検定対象であり保証ではない)。
# ---------------------------------------------------------------------------

FAVORING_TASKS = {
    "selective_copy": SelectiveCopyTask,        # hypothesis: mamba_selective
    "bistable_denoise": BistableDenoiseTask,    # hypothesis: hopfield_dense
    "weighted_accum": WeightedAccumulationTask,  # hypothesis: linear_attn
    "leaky_tracking": LeakyTrackingTask,        # hypothesis: rwkv
}

# 仮説写像 (BG6_strong で実測写像と突き合わせて「逆設計でなく第一原理が当たったか」を見る)。
HYPOTHESIS_BEST_KERNEL = {
    "selective_copy": "mamba_selective",
    "bistable_denoise": "hopfield_dense",
    "weighted_accum": "linear_attn",
    "leaky_tracking": "rwkv",
}


def make_task(name: str):
    """task 名からインスタンスを返す薄い factory。"""
    if name not in FAVORING_TASKS:
        raise KeyError(f"unknown favoring task: {name!r} (known: {list(FAVORING_TASKS)})")
    return FAVORING_TASKS[name]()


__all__ = [
    "SelectiveCopyTask",
    "BistableDenoiseTask",
    "WeightedAccumulationTask",
    "LeakyTrackingTask",
    "FAVORING_TASKS",
    "HYPOTHESIS_BEST_KERNEL",
    "make_task",
]
