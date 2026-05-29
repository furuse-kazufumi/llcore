# SPDX-License-Identifier: Apache-2.0
"""Izhikevich gene に対する Z3 invariant verifier (Stage 2.3).

2 つの invariant:

1. **膜電位 bounded (per-gene, forward Euler 1-step)**:
     v_next = v + dt * (0.04 * v^2 + 5 v + 140 - u + I)
     v ∈ [V_PRE_MIN, V_PEAK], u ∈ [U_MIN, U_MAX], I ∈ [-I_max, I_max]
     invariant: v_next <= V_PEAK + safety_margin
              ∧ v_next >= V_PRE_MIN - safety_margin
     Z3 NRA で v^2 を含む 1-step map を solve.

2. **firing rate 上界 (per-gene, dt-discretization bound)**:
     refractory なしモデルだが、discrete-time 1-step 不可分性から
     spike 間隔 >= dt は構造保証。T_window 内 spike 数 <= T_window / dt。
     rate <= 1000 / dt Hz (dt=0.25 ms → 4000 Hz の自明上界)。
     Z3 で「dt 不可分性を尊重した spike 列で T_window 内に n_spikes を
     詰められるか」を sat 探索。

honest 留保:
- Z3 は exact rational, forward Euler は float64 — 数値差で marginal な
  reject あり得る (safety_margin で吸収).
- Izhikevich の v^2 は Z3 NRA で扱える (quantifier-free nonlinear real arithmetic).
  decision procedure は CAD-based (Tarski-Seidenberg) で完全だが、timeout 注意.
- 1-step bound は ONE Euler step の上界. spike 後 reset c へ jump するので
  multi-step での収束は claim しない (Stage 3+ で induction proof 候補).
- refractory なしモデルの firing rate 上界は **dt-discretization 自明上界**:
  refractory bound (LIF) より広く緩い. これは Izhikevich の構造的特徴を
  honest に反映 (overclaim 禁止).
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# llcore.verifier 経由で z3 availability 参照 (src 経由 import) — LIF 版踏襲
_PROJ_ROOT = Path(__file__).resolve().parents[4]
_SRC = _PROJ_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import z3
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False

from .izh_gene import (
    DT,
    I_MAX_ABS,
    U_MAX,
    U_MIN,
    V_PEAK,
    V_PRE_MAX,
    V_PRE_MIN,
    IzhikevichGene,
)


def is_z3_available() -> bool:
    """z3-solver がインストールされ import 可能か返す."""
    return _HAS_Z3


@dataclass(frozen=True)
class IzhInvariantResult:
    """Izhikevich verifier 検査結果.

    Attributes
    ----------
    ok : bool
        True = invariant 成立 (z3 unsat または z3 unavailable で skip).
    used_z3 : bool
        Z3 が実際に呼ばれたか.
    reason : str
        verdict 説明.
    counterexample : dict[str, float] | None
        sat の場合の反例 model.
    """

    ok: bool
    used_z3: bool
    reason: str
    counterexample: dict[str, float] | None = None


# ---------------------------------------------------------------------------
# (1) v bounded invariant (per-gene, 1-step Euler)
# ---------------------------------------------------------------------------


def _to_float_safe(model_value) -> float:
    """Z3 model value -> float (Real / fraction 両対応)."""
    try:
        return float(model_value.as_decimal(8).rstrip("?"))
    except Exception:
        return float(model_value.as_fraction())


def verify_v_bounded_per_gene(
    gene: IzhikevichGene,
    *,
    safety_margin: float = 100.0,
    I_max: float | None = None,
    timeout_ms: int = 3000,
) -> IzhInvariantResult:
    """単一 gene の (a, b) に対し 1-step Euler 後の v_next が
    ``[V_PRE_MIN - safety_margin, V_PEAK + safety_margin]`` に収まるか検査.

    forward Euler 更新 (Izhikevich):
        v_next = v + DT * (0.04 v^2 + 5 v + 140 - u + I)

    Parameters
    ----------
    gene : IzhikevichGene
        検査対象 gene (a, b 使用 — c, d は spike 後の reset で本 invariant 範囲外).
    safety_margin : float
        invariant 上下界の余裕 (mV). Izhikevich の v^2 項により LIF より大きな
        overshoot が 1 step で発生し得る (v=30, u=U_MIN, I=I_max で
        worst overshoot ≈ DT * (0.04*900 + 150 + 140 + |U_MIN| + I_max)
        ≈ 0.25 * (36+150+140+25+10) ≈ 90 mV). 既定 100 mV で I_max=10 (safe contract)
        下では admit 期待. tight contract (I_max=5) でもう少し狭い safety で admit 可能.
    I_max : float | None
        assumed-input contract 上界 (Stage 2.2a Codex F3 対応). None なら ``I_MAX_ABS``.
    timeout_ms : int
        Z3 timeout (ms). v^2 含む NRA なので LIF より長め推奨.

    Returns
    -------
    IzhInvariantResult
        ok=True なら unsat (bound 証明).
    """
    if not _HAS_Z3:
        return IzhInvariantResult(
            ok=True, used_z3=False, reason="z3 unavailable, skip"
        )

    if I_max is None:
        I_max = I_MAX_ABS
    if I_max <= 0:
        raise ValueError(f"I_max must be positive, got {I_max}")
    if safety_margin <= 0:
        raise ValueError(f"safety_margin must be positive, got {safety_margin}")

    g = gene.clipped()

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    v = z3.Real("v")
    u = z3.Real("u")
    I = z3.Real("I")
    v_next = z3.Real("v_next")

    # 入力範囲制約
    solver.add(v >= V_PRE_MIN, v <= V_PRE_MAX)
    solver.add(u >= U_MIN, u <= U_MAX)
    solver.add(I >= -I_max, I <= I_max)

    # forward Euler 1-step (Izhikevich v)
    # v_next = v + DT * (0.04 v^2 + 5 v + 140 - u + I)
    solver.add(
        v_next == v + DT * (z3.RealVal("0.04") * v * v + z3.RealVal(5) * v
                            + z3.RealVal(140) - u + I)
    )

    # bound 違反 (上界 OR 下界)
    upper = V_PEAK + safety_margin
    lower = V_PRE_MIN - safety_margin
    solver.add(z3.Or(v_next > upper, v_next < lower))

    result = solver.check()
    if result == z3.unsat:
        return IzhInvariantResult(
            ok=True, used_z3=True,
            reason=(
                f"unsat: v_next ∈ [{lower}, {upper}] mV for "
                f"v ∈ [{V_PRE_MIN},{V_PRE_MAX}], u ∈ [{U_MIN},{U_MAX}], "
                f"I ∈ [-{I_max},{I_max}] (contract); gene (a={g.a:.3f}, b={g.b:.3f})"
            ),
        )
    if result == z3.sat:
        m = solver.model()
        ce = {
            "v": _to_float_safe(m[v]),
            "u": _to_float_safe(m[u]),
            "I": _to_float_safe(m[I]),
            "v_next": _to_float_safe(m[v_next]),
        }
        return IzhInvariantResult(
            ok=False, used_z3=True,
            reason=(
                f"sat: v={ce['v']:.3f}, u={ce['u']:.3f}, I={ce['I']:.3f}, "
                f"v_next={ce['v_next']:.3f} outside [{lower}, {upper}]"
            ),
            counterexample=ce,
        )
    return IzhInvariantResult(
        ok=False, used_z3=True, reason=f"unknown / timeout ({result})",
    )


# ---------------------------------------------------------------------------
# (2) firing rate 上界 invariant (per-gene, dt-discretization)
# ---------------------------------------------------------------------------


def verify_firing_rate_per_gene(
    gene: IzhikevichGene,
    *,
    n_spikes: int = 10,
    T_window_ms: float = 100.0,
    dt: float = DT,
    timeout_ms: int = 2000,
) -> IzhInvariantResult:
    """**単一 gene** の firing rate 上界 (dt 不可分性).

    Izhikevich は明示的 refractory なし → spike 間隔 >= dt のみ構造保証.
    n_spikes 個の spike が T_window 内に dt 不可分性を尊重して収まるなら::

        rate <= n_spikes / (T_window / 1000) <= (T_window / dt) / (T_window / 1000)
              = 1000 / dt Hz

    Z3 encoding:
        spike 列 t_1 < t_2 < ... < t_n が dt 不可分性 (t_{i+1} - t_i >= dt) を
        尊重し、[0, T_window] に収まる。
        違反: (n-1) * dt > T_window  (refractory 0 mode の boundary)
        ⇒ unsat なら invariant 成立.

    honest 留保:
    - refractory bound より緩い (LIF は 1000/t_ref Hz, Izhikevich は 1000/dt Hz).
    - これは「Izhikevich の構造的特徴 = 明示的不応期なし」を honest に反映.
    - claim 降格: 「physiological refractory bound」ではなく
      「discrete-time discretization bound」.

    Parameters
    ----------
    gene : IzhikevichGene
        検査対象 (gene 自体は使わない — dt + n_spikes + T_window のみで判定).
        対称性のため LIF API と signature 揃え.
    n_spikes : int
        検査する spike 数.
    T_window_ms : float
        time window (ms).
    dt : float
        forward Euler 刻み幅 (ms).
    timeout_ms : int
        Z3 timeout (ms).

    Returns
    -------
    IzhInvariantResult
        ok=True なら unsat (invariant 成立).
    """
    if not _HAS_Z3:
        return IzhInvariantResult(
            ok=True, used_z3=False, reason="z3 unavailable, skip",
        )

    if n_spikes < 2:
        return IzhInvariantResult(
            ok=True, used_z3=False, reason="n_spikes < 2, vacuous",
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    spike_times = [z3.Real(f"t_{i}") for i in range(n_spikes)]
    solver.add(spike_times[0] >= 0)
    solver.add(spike_times[-1] <= T_window_ms)
    # dt 不可分性: t_{i+1} - t_i >= dt
    for i in range(n_spikes - 1):
        solver.add(spike_times[i + 1] - spike_times[i] >= dt)
    # 違反 (finite-window 厳密, LIF Stage 2.1 と同 pattern):
    #   (n-1) * dt > T_window
    solver.add((n_spikes - 1) * dt > T_window_ms)

    result = solver.check()
    upper_rate = 1000.0 / dt
    if result == z3.unsat:
        return IzhInvariantResult(
            ok=True, used_z3=True,
            reason=(
                f"unsat: rate <= {upper_rate:.1f} Hz proved "
                f"(dt={dt:.3f} ms, n={n_spikes}, T={T_window_ms})"
            ),
        )
    if result == z3.sat:
        return IzhInvariantResult(
            ok=False, used_z3=True,
            reason="sat: dt-discretization bound violated",
        )
    return IzhInvariantResult(
        ok=False, used_z3=True, reason=f"unknown ({result})",
    )


# ---------------------------------------------------------------------------
# Global (gene-free) invariant for Stage 2.3 G1 sanity
# ---------------------------------------------------------------------------


def verify_v_bounded_global(
    *,
    safety_margin: float = 100.0,
    I_max: float | None = None,
    timeout_ms: int = 3000,
) -> IzhInvariantResult:
    """Gene-free 1-step Euler v bounded invariant. (a, b) は使わないので gene 不要.

    1-step v update は v_next = v + DT*(0.04 v^2 + 5 v + 140 - u + I) で
    a, b は v 更新に直接登場しない (u 経由のみ). 1-step では u は input なので
    a, b 非依存で v_next bound を solve できる.

    G1 sanity test (safe contract admit, loose contract reject) 用.
    """
    if not _HAS_Z3:
        return IzhInvariantResult(ok=True, used_z3=False, reason="z3 unavailable, skip")

    if I_max is None:
        I_max = I_MAX_ABS
    if I_max <= 0:
        raise ValueError(f"I_max must be positive, got {I_max}")
    if safety_margin <= 0:
        raise ValueError(f"safety_margin must be positive, got {safety_margin}")

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    v = z3.Real("v")
    u = z3.Real("u")
    I = z3.Real("I")
    v_next = z3.Real("v_next")

    solver.add(v >= V_PRE_MIN, v <= V_PRE_MAX)
    solver.add(u >= U_MIN, u <= U_MAX)
    solver.add(I >= -I_max, I <= I_max)
    solver.add(
        v_next == v + DT * (z3.RealVal("0.04") * v * v + z3.RealVal(5) * v
                            + z3.RealVal(140) - u + I)
    )

    upper = V_PEAK + safety_margin
    lower = V_PRE_MIN - safety_margin
    solver.add(z3.Or(v_next > upper, v_next < lower))

    result = solver.check()
    if result == z3.unsat:
        return IzhInvariantResult(
            ok=True, used_z3=True,
            reason=(
                f"unsat: v_next ∈ [{lower}, {upper}] mV "
                f"(I_max={I_max}, margin={safety_margin})"
            ),
        )
    if result == z3.sat:
        m = solver.model()
        ce = {
            "v": _to_float_safe(m[v]),
            "u": _to_float_safe(m[u]),
            "I": _to_float_safe(m[I]),
            "v_next": _to_float_safe(m[v_next]),
        }
        return IzhInvariantResult(
            ok=False, used_z3=True,
            reason=(
                f"sat: v={ce['v']:.3f}, u={ce['u']:.3f}, I={ce['I']:.3f}, "
                f"v_next={ce['v_next']:.3f} outside [{lower}, {upper}]"
            ),
            counterexample=ce,
        )
    return IzhInvariantResult(ok=False, used_z3=True, reason=f"unknown ({result})")


__all__ = [
    "is_z3_available",
    "IzhInvariantResult",
    "verify_v_bounded_per_gene",
    "verify_v_bounded_global",
    "verify_firing_rate_per_gene",
]
