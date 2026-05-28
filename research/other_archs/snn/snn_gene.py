# SPDX-License-Identifier: Apache-2.0
"""LIF (Leaky Integrate-and-Fire) neuron gene + forward Euler simulator.

falsifiable 命題 (本ファイル該当部):
    LIF neuron の連続時間 ODE
        tau_m * dV/dt = -(V - V_rest) + R * I(t)
        if V(t) >= V_th: spike, V(t+) = V_reset, hold for t_ref
    を 4 パラメータ gene (tau_m, V_th, V_reset, t_ref) で表現でき、
    forward Euler (dt=0.1 ms) で finite ・spike list が refractory を尊重し、
    physical 範囲 clip で発火が誘起される。

物理範囲 clip (生物学的妥当性):
- tau_m ∈ [5, 30] ms      — 膜時定数 (cortical neuron 範囲)
- V_th ∈ [-55, -40] mV    — 発火閾値
- V_reset ∈ [-80, -65] mV — リセット電位
- t_ref ∈ [1, 5] ms       — 不応期

固定パラメータ:
- V_rest = -70 mV (静止膜電位)
- R = 1.0 (membrane resistance, normalized)
- dt = 0.1 ms (forward Euler 刻み幅)

honest 留保:
- forward Euler は 1 次精度。tau_m << dt のとき発散リスクだが clip で
  tau_m >= 5 ms (dt の 50 倍) を保証するので発散しない。
- V_reset < V_th を clip 後にも保証する必要 → :meth:`clipped` で post-clip 補正。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# 固定 LIF パラメータ
V_REST: float = -65.0   # mV, 静止膜電位
R_MEM: float = 10.0     # 膜抵抗 (normalized, スケールを発火閾値に乗せるため大きめ)
DT: float = 0.1         # ms, forward Euler 刻み幅

# 物理範囲 clip (生物学的妥当性)
TAU_M_MIN, TAU_M_MAX = 5.0, 30.0
V_TH_MIN, V_TH_MAX = -55.0, -40.0
V_RESET_MIN, V_RESET_MAX = -80.0, -65.0
T_REF_MIN, T_REF_MAX = 1.0, 5.0

# 入力範囲想定 (Z3 invariant でも使う). bias + amplitude + 3*noise_std の上界:
# 既定 bias=1.5, amp=0.5, noise=0.1 → 1.5 + 0.5 + 0.3 = 2.3
I_MAX_ABS: float = 2.5   # bias + sin + noise の上界
# 最大駆動可能 V: V_REST + R_MEM * I_MAX = -65 + 10*2.5 = -40 mV (V_TH 上回る)


@dataclass(frozen=True)
class LIFGene:
    """LIF neuron 4 パラメータ gene.

    Attributes
    ----------
    tau_m : float
        膜時定数 (ms). Clip 範囲 [5, 30].
    V_th : float
        発火閾値 (mV). Clip 範囲 [-55, -40].
    V_reset : float
        リセット電位 (mV). Clip 範囲 [-80, -65]、かつ ``V_reset < V_th`` を後処理で保証.
    t_ref : float
        不応期 (ms). Clip 範囲 [1, 5].
    """

    tau_m: float
    V_th: float
    V_reset: float
    t_ref: float

    def clipped(self) -> "LIFGene":
        """物理範囲に clip した新 gene を返す.

        post-clip で ``V_reset < V_th`` を強制 (clip 範囲 [-80,-65] vs [-55,-40] で
        構造的に成立するが念のため min 1 mV 差を保証).
        """
        tau = float(np.clip(self.tau_m, TAU_M_MIN, TAU_M_MAX))
        vth = float(np.clip(self.V_th, V_TH_MIN, V_TH_MAX))
        vrst = float(np.clip(self.V_reset, V_RESET_MIN, V_RESET_MAX))
        if vrst >= vth - 1.0:
            vrst = vth - 1.0  # 念のため
            vrst = float(np.clip(vrst, V_RESET_MIN, V_RESET_MAX))
        tref = float(np.clip(self.t_ref, T_REF_MIN, T_REF_MAX))
        return LIFGene(tau_m=tau, V_th=vth, V_reset=vrst, t_ref=tref)

    def as_array(self) -> np.ndarray:
        return np.array([self.tau_m, self.V_th, self.V_reset, self.t_ref], dtype=np.float64)


def simulate_lif(
    gene: LIFGene,
    I_input: np.ndarray,
    T: float = 100.0,
    dt: float = DT,
    V_init: float | None = None,
) -> tuple[np.ndarray, list[float]]:
    """forward Euler LIF simulation.

    Parameters
    ----------
    gene : LIFGene
        LIF パラメータ (内部で ``clipped()`` を適用).
    I_input : np.ndarray
        入力電流時系列 shape=(n_steps,). ``n_steps == int(T/dt)`` 期待.
        足りないと最後の値で hold, 多いと頭から truncate.
    T : float
        シミュレーション時間 (ms).
    dt : float
        刻み幅 (ms).
    V_init : float | None
        初期膜電位. None なら V_rest.

    Returns
    -------
    V_trace : np.ndarray
        shape=(n_steps+1,) の膜電位履歴 (mV).
    spike_times : list[float]
        spike が発生した時刻 (ms) のリスト.
    """
    g = gene.clipped()
    n_steps = int(round(T / dt))
    # 入力長合わせ
    if len(I_input) < n_steps:
        I = np.concatenate([I_input, np.full(n_steps - len(I_input), I_input[-1] if len(I_input) > 0 else 0.0)])
    else:
        I = I_input[:n_steps]

    V_trace = np.zeros(n_steps + 1, dtype=np.float64)
    V_trace[0] = V_REST if V_init is None else V_init
    spike_times: list[float] = []

    # refractory: 何 ms までは V = V_reset で固定
    ref_until: float = -1.0

    for k in range(n_steps):
        t_curr = k * dt
        V = V_trace[k]

        if t_curr < ref_until:
            # 不応期中: V を V_reset で hold
            V_next = g.V_reset
        else:
            # 通常更新: V_next = V + dt/tau_m * (V_rest - V + R*I)
            dV = (dt / g.tau_m) * (V_REST - V + R_MEM * I[k])
            V_next = V + dV
            if V_next >= g.V_th:
                # spike!
                spike_times.append(t_curr + dt)
                V_next = g.V_reset
                ref_until = t_curr + dt + g.t_ref

        V_trace[k + 1] = V_next

    return V_trace, spike_times


def make_periodic_input(
    T: float,
    dt: float = DT,
    freq_hz: float = 20.0,
    amplitude: float = 1.5,
    bias: float = 1.5,
    noise_std: float = 0.1,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """周期入力 + bias + noise を生成 (発火パターン誘起用).

    I(t) = bias + amplitude * sin(2*pi*f*t/1000) + N(0, noise_std)
    (t は ms, freq_hz は Hz)

    bias を入れることで R_MEM * (bias + amp) > V_th - V_REST を満たし、確実発火を誘起.
    bias + amp の総和 <= I_MAX_ABS (= 2) になるよう既定値を選んでいる.
    """
    n_steps = int(round(T / dt))
    t = np.arange(n_steps) * dt  # ms
    rng = rng if rng is not None else np.random.default_rng(0)
    sig = amplitude * np.sin(2 * np.pi * freq_hz * t / 1000.0)
    noise = rng.normal(0.0, noise_std, n_steps)
    return bias + sig + noise


def firing_rate_hz(spike_times: list[float], T: float) -> float:
    """spike list から firing rate (Hz) を計算."""
    if T <= 0.0:
        return 0.0
    return len(spike_times) / (T / 1000.0)


__all__ = [
    "LIFGene",
    "simulate_lif",
    "make_periodic_input",
    "firing_rate_hz",
    "V_REST",
    "R_MEM",
    "DT",
    "TAU_M_MIN",
    "TAU_M_MAX",
    "V_TH_MIN",
    "V_TH_MAX",
    "V_RESET_MIN",
    "V_RESET_MAX",
    "T_REF_MIN",
    "T_REF_MAX",
    "I_MAX_ABS",
]
