# SPDX-License-Identifier: Apache-2.0
"""Izhikevich 2D neuron gene + forward Euler simulator.

falsifiable 命題 (本ファイル該当部):
    Izhikevich 神経モデル (連続時間 ODE 2 変数)::

        dv/dt = 0.04 v^2 + 5 v + 140 - u + I
        du/dt = a (b v - u)
        if v >= 30 mV: v <- c, u <- u + d   (spike + reset)

    を 4 パラメータ gene ``(a, b, c, d) ∈ R^4`` で表現でき、forward Euler
    (dt = 0.25 ms) で finite ・spike list は構造的に dt 不可分性を尊重し、
    Izhikevich 2003 原典の clip 範囲下で **RS / IB / CH / FS** の 4 firing-type
    が単一 family として進化空間に含まれる。

物理範囲 clip (Izhikevich 2003 原典 + 進化空間妥当性):

- a ∈ [0.01, 0.10]  - u recovery 速度
  (RS: 0.02, FS: 0.1, IB: 0.02, CH: 0.02)
- b ∈ [0.20, 0.30]  - subthreshold sensitivity
  (RS/FS/IB: 0.2, CH: 0.2 (原典) - 進化幅で 0.25-0.3 含む)
- c ∈ [-65, -50] mV - reset value
  (RS: -65, FS: -65, IB: -55, CH: -50)
- d ∈ [2, 8]        - recovery 増分
  (RS: 8, FS: 2, IB: 4, CH: 2)

これにより RS / IB / CH / FS の 4 firing-pattern type を進化空間内で表現可能.

固定パラメータ:
- V_PEAK = 30 mV (spike threshold)
- V_INIT = -70 mV (初期膜電位)
- U_INIT = -14 mV (初期 u, 静止時 ≈ b * V_INIT)
- dt = 0.25 ms (forward Euler 刻み幅)

honest 留保:
- forward Euler は 1 次精度。Izhikevich 原論文の 2 段 half-step trick は
  本 PoC では使わない (素朴 Euler で簡潔性優先, Stage 3+ で比較研究).
- v² 非線形項により 1-step overshoot は LIF より大きい. v=30 付近で
  dt=0.25*(0.04*900+150+140) ≈ 81.5 mV overshoot. Z3 invariant では
  「v ∈ [-80, 30] clip 内なら v_next は assumed-input contract 下で
  v_next_max <= V_PEAK + 1-step worst overshoot 上界」として扱う.
- refractory なし: Izhikevich は明示的不応期を持たず、c (reset) + d (u jump)
  でリカバリ. firing rate 上界は dt 不可分性 (1-step / spike) からのみ導出.
- ``c < V_PEAK`` を clip 後にも保証する必要 → :meth:`clipped` で post-clip 補正.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 固定 Izhikevich パラメータ
V_PEAK: float = 30.0    # mV, spike threshold (Izhikevich 2003)
V_INIT: float = -70.0   # mV, 初期膜電位 (静止)
U_INIT: float = -14.0   # 初期 u (静止状態 ≈ b * V_INIT for b=0.2)
DT: float = 0.25        # ms, forward Euler 刻み幅


# 物理範囲 clip (Izhikevich 2003 原典 + RS/FS/IB/CH を覆う)
A_MIN, A_MAX = 0.01, 0.10
B_MIN, B_MAX = 0.20, 0.30
C_MIN, C_MAX = -65.0, -50.0
D_MIN, D_MAX = 2.0, 8.0

# Z3 invariant で使う状態変数の範囲
# v は pre-spike clip: [V_RESET_LOW_BOUND, V_PEAK]
V_PRE_MIN: float = -80.0     # mV, c の最低値より少し下 (reset 後初期)
V_PRE_MAX: float = V_PEAK    # = 30 mV, spike 直前

# u は b*v の積分なので range を概算:
#   b ∈ [0.2, 0.3], v ∈ [-80, 30] ⇒ b*v ∈ [-24, 9]
#   d (spike 後 +d) ∈ [2, 8] で u が累積するが、定常状態では b*v-u ≈ 0 で平衡
#   進化空間で u が暴走しないことを前提 (G1 invariant で検査)
U_MIN: float = -25.0
U_MAX: float = 25.0   # spike + d 累積を許容するが Z3 でこの range 内に収まるか検査

# 入力範囲想定 (Z3 invariant でも使う)
# Izhikevich の I はモデル単位で典型 5〜20 で発火を誘起 (LIF より単位スケール大)
I_MAX_ABS: float = 10.0   # 既定 assumed-input contract


@dataclass(frozen=True)
class IzhikevichGene:
    """Izhikevich neuron 4 パラメータ gene.

    Attributes
    ----------
    a : float
        u recovery 速度. Clip 範囲 [0.01, 0.10].
    b : float
        subthreshold sensitivity. Clip 範囲 [0.20, 0.30].
    c : float
        reset value (mV). Clip 範囲 [-65, -50], かつ ``c < V_PEAK - 5`` を post-clip で保証.
    d : float
        recovery 増分. Clip 範囲 [2, 8].
    """

    a: float
    b: float
    c: float
    d: float

    def clipped(self) -> "IzhikevichGene":
        """物理範囲に clip した新 gene を返す.

        post-clip で ``c < V_PEAK - 5`` を保証 (clip 範囲 [-65,-50] vs V_PEAK=30 で
        構造的に成立するが念のため)。
        """
        a_c = float(np.clip(self.a, A_MIN, A_MAX))
        b_c = float(np.clip(self.b, B_MIN, B_MAX))
        c_c = float(np.clip(self.c, C_MIN, C_MAX))
        d_c = float(np.clip(self.d, D_MIN, D_MAX))
        # c < V_PEAK - 5 を保証 (clip 範囲ですでに成立だが念のため)
        if c_c >= V_PEAK - 5.0:
            c_c = V_PEAK - 5.0
            c_c = float(np.clip(c_c, C_MIN, C_MAX))
        return IzhikevichGene(a=a_c, b=b_c, c=c_c, d=d_c)

    def as_array(self) -> np.ndarray:
        return np.array([self.a, self.b, self.c, self.d], dtype=np.float64)

    def firing_type_guess(self) -> str:
        """gene のパラメータから best-match firing type を guess (Izhikevich 2003 Table).

        判定 priority:
        - FS (Fast Spiking)        : a >= 0.08 かつ d <= 3
        - CH (Chattering)          : c >= -52 かつ d <= 3 (= 高 c + 弱 d)
        - IB (Intrinsically Bursting): -60 <= c <= -52 かつ d >= 3 かつ d <= 6
        - RS (Regular Spiking)     : default (a 小, c 低, d 大)
        """
        g = self.clipped()
        # FS: 速い recovery + 低 d (Izhikevich 2003: a=0.1, b=0.2, c=-65, d=2)
        if g.a >= 0.08 and g.d <= 3.0:
            return "FS"
        # CH: 高 c (Izhikevich 2003: a=0.02, b=0.2, c=-50, d=2)
        if g.c >= -52.0 and g.d <= 3.0:
            return "CH"
        # IB: 中間 c + 中間 d (Izhikevich 2003: a=0.02, b=0.2, c=-55, d=4)
        if -60.0 <= g.c <= -52.0 and 3.0 < g.d <= 6.0:
            return "IB"
        # RS: default (Izhikevich 2003: a=0.02, b=0.2, c=-65, d=8)
        return "RS"


def simulate_izh(
    gene: IzhikevichGene,
    I_input: np.ndarray,
    T: float = 200.0,
    dt: float = DT,
    v_init: float | None = None,
    u_init: float | None = None,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """forward Euler Izhikevich simulation.

    Parameters
    ----------
    gene : IzhikevichGene
        Izhikevich パラメータ (内部で ``clipped()`` を適用).
    I_input : np.ndarray
        入力電流時系列 shape=(n_steps,). ``n_steps == int(T/dt)`` 期待.
        足りないと最後の値で hold, 多いと頭から truncate.
    T : float
        シミュレーション時間 (ms).
    dt : float
        刻み幅 (ms).
    v_init, u_init : float | None
        初期状態. None なら ``V_INIT``, ``b * V_INIT``.

    Returns
    -------
    V_trace : np.ndarray
        shape=(n_steps+1,) の膜電位履歴 (mV).
    u_trace : np.ndarray
        shape=(n_steps+1,) の recovery variable 履歴.
    spike_times : list[float]
        spike が発生した時刻 (ms) のリスト.
    """
    g = gene.clipped()
    n_steps = int(round(T / dt))

    # 入力長合わせ
    if len(I_input) < n_steps:
        last_val = I_input[-1] if len(I_input) > 0 else 0.0
        I = np.concatenate([I_input, np.full(n_steps - len(I_input), last_val)])
    else:
        I = I_input[:n_steps]

    V_trace = np.zeros(n_steps + 1, dtype=np.float64)
    u_trace = np.zeros(n_steps + 1, dtype=np.float64)
    V_trace[0] = V_INIT if v_init is None else v_init
    u_trace[0] = (g.b * V_trace[0]) if u_init is None else u_init

    spike_times: list[float] = []

    for k in range(n_steps):
        v = V_trace[k]
        u = u_trace[k]

        # Izhikevich forward Euler
        dv = 0.04 * v * v + 5.0 * v + 140.0 - u + I[k]
        du = g.a * (g.b * v - u)
        v_next = v + dt * dv
        u_next = u + dt * du

        # spike detection (v_next >= V_PEAK)
        if v_next >= V_PEAK:
            spike_times.append((k + 1) * dt)
            v_next = g.c
            u_next = u_next + g.d

        # NaN/Inf 検出 (発散時の safety guard)
        if not np.isfinite(v_next) or not np.isfinite(u_next):
            # 発散: 残り steps を 0 で埋めて打ち切り
            V_trace[k + 1:] = 0.0
            u_trace[k + 1:] = 0.0
            break

        V_trace[k + 1] = v_next
        u_trace[k + 1] = u_next

    return V_trace, u_trace, spike_times


def make_constant_input(
    T: float,
    dt: float = DT,
    I_value: float = 10.0,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """定常入力 + noise (Izhikevich 原論文は constant DC injection で各 firing pattern を誘起).

    I(t) = I_value + N(0, noise_std)
    """
    n_steps = int(round(T / dt))
    rng = rng if rng is not None else np.random.default_rng(0)
    if noise_std > 0:
        noise = rng.normal(0.0, noise_std, n_steps)
    else:
        noise = np.zeros(n_steps)
    return np.full(n_steps, I_value) + noise


def firing_rate_hz(spike_times: list[float], T: float) -> float:
    """spike list から firing rate (Hz) を計算."""
    if T <= 0.0:
        return 0.0
    return len(spike_times) / (T / 1000.0)


__all__ = [
    "IzhikevichGene",
    "simulate_izh",
    "make_constant_input",
    "firing_rate_hz",
    "V_PEAK",
    "V_INIT",
    "U_INIT",
    "DT",
    "A_MIN",
    "A_MAX",
    "B_MIN",
    "B_MAX",
    "C_MIN",
    "C_MAX",
    "D_MIN",
    "D_MAX",
    "V_PRE_MIN",
    "V_PRE_MAX",
    "U_MIN",
    "U_MAX",
    "I_MAX_ABS",
]
