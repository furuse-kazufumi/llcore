# SPDX-License-Identifier: Apache-2.0
"""SNN (LIF) gene に対する Z3 invariant verifier.

3 つの invariant:
1. **Firing rate 上界** (refractory bound, 構造的):
     rate <= 1000 / t_ref  Hz
   - t_ref ∈ [1, 5] ms より rate <= 1000 / 1 = 1000 Hz, t_ref=5 で 200 Hz.
   - 構造的に成立する境界条件だが、「進化対象として t_ref を選ぶことで
     symbolic に上界を保証できる」ことが verifier 統合の意義 (verdict Q2 で議論).

2. **膜電位 bounded**:
     forward Euler 1 step 後 V_next が (V_rest, V_th + safety_margin) に収まる
     上界解析: V_next = V + dt/tau_m * (V_rest - V + R*|I|_max)
     - sat (反例あり) なら overshoot 検出 (gene reject).
     - unsat なら全 (V, I) 範囲で overshoot なし.

3. **Shielded RL hint (sketch)**: SNN 出力 firing rate を policy action shield として
     rate_action <= R_safe を gene が常に満たすか symbolic 検査.
     - これは Codex Q5 推奨の ProSh / Adaptive GR(1) shielding を verifier 統合する
       **最小 sketch**. 本 PoC では mock 入力のみ (実 RL policy との接続は将来研究).

honest 留保:
- Z3 は exact rational, forward Euler は float — 数値差で marginal な reject あり得る.
- safety_margin = 1.0 mV (Euler 誤差吸収).
- shielded RL は **sketch only**, ProSh / Adaptive GR(1) 本格統合は overclaim 禁止.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

# llcore.verifier の is_z3_available を再利用 (src 経由 import)
_PROJ_ROOT = Path(__file__).resolve().parents[3]
_SRC = _PROJ_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import z3
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False

from .snn_gene import (
    DT,
    I_MAX_ABS,
    R_MEM,
    T_REF_MAX,
    T_REF_MIN,
    TAU_M_MAX,
    TAU_M_MIN,
    V_RESET_MAX,
    V_RESET_MIN,
    V_REST,
    V_TH_MAX,
    V_TH_MIN,
    LIFGene,
)


def is_z3_available() -> bool:
    """z3-solver がインストールされ import 可能か返す."""
    return _HAS_Z3


@dataclass(frozen=True)
class SNNInvariantResult:
    """SNN verifier 検査結果.

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
# (1) Firing rate 上界 invariant
# ---------------------------------------------------------------------------


def verify_firing_rate_bound(
    *,
    n_spikes: int = 10,
    T_window_ms: float = 100.0,
    timeout_ms: int = 2000,
) -> SNNInvariantResult:
    """clip 範囲下で「n_spikes 個の spike が T_window 内に refractory を尊重して
    収まるなら、firing rate <= 1000/t_ref Hz が成立する」を symbolic に証明.

    数学的構造:
        spike 列 t_1 < t_2 < ... < t_n が refractory を尊重するなら
        t_{i+1} - t_i >= t_ref  (forall i)
        spike 全部が [0, T_window] に入るので t_n - t_1 >= (n-1) * t_ref
        spike 数 n に対し T_window >= (n-1) * t_ref ⇒ n <= 1 + T_window/t_ref
        rate = n / (T_window/1000) [Hz]
        (n-1) * t_ref <= T_window より n <= T_window/t_ref + 1
        large T 漸近で rate <= 1000/t_ref Hz が出る.

    本 verifier は **「refractory 制約を満たす spike 列で rate が 1000/t_ref を
    超える反例があるか」** を sat 探索. unsat なら invariant 成立.

    Z3 encoding (n_spikes 個の spike 列):
        t_ref ∈ [T_REF_MIN, T_REF_MAX]
        0 <= t_1 < t_2 < ... < t_n <= T_window
        t_{i+1} - t_i >= t_ref  (refractory)
        違反: n_spikes / (T_window/1000) > 1000 / t_ref
              <=> n_spikes * t_ref > T_window
        ⇒ refractory 制約と矛盾するので unsat になるはず.

    Returns
    -------
    SNNInvariantResult
        ok=True なら unsat (refractory 構造保証成立).
    """
    if not _HAS_Z3:
        return SNNInvariantResult(
            ok=True, used_z3=False,
            reason="z3-solver unavailable, skip (vacuous True)",
        )

    if n_spikes < 2:
        return SNNInvariantResult(
            ok=True, used_z3=False,
            reason="n_spikes < 2, vacuously satisfied",
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    t_ref = z3.Real("t_ref")
    solver.add(t_ref >= T_REF_MIN, t_ref <= T_REF_MAX)

    spike_times = [z3.Real(f"t_{i}") for i in range(n_spikes)]
    # 0 <= t_1, t_n <= T_window
    solver.add(spike_times[0] >= 0)
    solver.add(spike_times[-1] <= T_window_ms)
    # refractory: t_{i+1} - t_i >= t_ref
    for i in range(n_spikes - 1):
        solver.add(spike_times[i + 1] - spike_times[i] >= t_ref)

    # invariant 違反: rate = n / (T/1000) > 1000 / t_ref
    # <=> n_spikes * t_ref > T_window
    solver.add(n_spikes * t_ref > T_window_ms)

    result = solver.check()
    if result == z3.unsat:
        return SNNInvariantResult(
            ok=True, used_z3=True,
            reason=(
                f"unsat: {n_spikes} spikes in {T_window_ms} ms with refractory "
                f"⇒ rate <= 1000/t_ref proved (no violating sequence exists)"
            ),
        )
    if result == z3.sat:
        model = solver.model()

        def _to_f(v):
            try:
                return float(model[v].as_decimal(6).rstrip("?"))
            except Exception:
                return float(model[v].as_fraction())

        ce = {"t_ref": _to_f(t_ref)}
        for i, st in enumerate(spike_times):
            ce[f"t_{i}"] = _to_f(st)
        return SNNInvariantResult(
            ok=False, used_z3=True,
            reason=f"sat: counterexample t_ref={ce['t_ref']:.3f}, spikes={[ce[f't_{i}'] for i in range(n_spikes)]}",
            counterexample=ce,
        )
    return SNNInvariantResult(
        ok=False, used_z3=True, reason=f"unknown / timeout ({result})",
    )


def verify_firing_rate_per_gene(
    gene: LIFGene,
    *,
    n_spikes: int = 10,
    T_window_ms: float = 100.0,
    timeout_ms: int = 1000,
) -> SNNInvariantResult:
    """**単一 gene** の t_ref に対し「n_spikes 個の refractory-respecting spike 列が
    T_window 内で rate <= 1000/t_ref を満たす」を Z3 で証明.

    refractory period 構造から数学的に成立するが、verifier 経由で per-gene
    latency 計測 + online gate として使う想定 (PoC online gate latency 測定用).
    """
    if not _HAS_Z3:
        return SNNInvariantResult(
            ok=True, used_z3=False, reason="z3 unavailable, skip",
        )

    if n_spikes < 2:
        return SNNInvariantResult(
            ok=True, used_z3=False, reason="n_spikes < 2, vacuous",
        )

    g = gene.clipped()
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    spike_times = [z3.Real(f"t_{i}") for i in range(n_spikes)]
    solver.add(spike_times[0] >= 0)
    solver.add(spike_times[-1] <= T_window_ms)
    for i in range(n_spikes - 1):
        solver.add(spike_times[i + 1] - spike_times[i] >= g.t_ref)
    # 違反: n_spikes / (T/1000) > 1000 / t_ref
    # <=> n_spikes * t_ref > T  (固定値で比較)
    solver.add(n_spikes * g.t_ref > T_window_ms)

    result = solver.check()
    upper_rate = 1000.0 / g.t_ref
    if result == z3.unsat:
        return SNNInvariantResult(
            ok=True, used_z3=True,
            reason=f"unsat: rate <= {upper_rate:.2f} Hz proved (t_ref={g.t_ref:.2f} ms, n={n_spikes}, T={T_window_ms})",
        )
    if result == z3.sat:
        return SNNInvariantResult(
            ok=False, used_z3=True,
            reason=f"sat: refractory bound violated (n_spikes * t_ref > T)",
        )
    return SNNInvariantResult(
        ok=False, used_z3=True, reason=f"unknown ({result})",
    )


# ---------------------------------------------------------------------------
# (2) 膜電位 bounded invariant
# ---------------------------------------------------------------------------


def verify_membrane_bounded(
    *,
    safety_margin: float = 1.0,
    timeout_ms: int = 2000,
) -> SNNInvariantResult:
    """forward Euler 1 step 後の V_next が安全範囲に収まるか.

    forward Euler 更新:
        V_next = V + (dt / tau_m) * (V_rest - V + R * I)

    解析対象は「不応期外、まだ spike していない時点」の V_next.
    V ∈ [V_reset_min, V_th_max] = [-80, -40] mV (clip 範囲広め),
    I ∈ [-I_MAX, +I_MAX] = [-2, 2],
    tau_m ∈ [TAU_M_MIN, TAU_M_MAX] = [5, 30].

    invariant:
        V_next <= V_th + safety_margin  (overshoot 上界)
        V_next >= V_rest - safety_margin (under-resting bound)

    Z3 で 「V_next > V_th_max + safety_margin OR V_next < V_rest - safety_margin」
    を sat 探索. unsat なら invariant 成立.

    Returns
    -------
    SNNInvariantResult
        ok=True なら unsat (bound 証明).
    """
    if not _HAS_Z3:
        return SNNInvariantResult(
            ok=True, used_z3=False, reason="z3 unavailable, skip",
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    tau_m = z3.Real("tau_m")
    V = z3.Real("V")
    I = z3.Real("I")
    V_next = z3.Real("V_next")

    solver.add(tau_m >= TAU_M_MIN, tau_m <= TAU_M_MAX)
    # V の許容範囲: V_RESET_MIN から V_TH_MAX まで (spike 直前まで)
    # 下界は V_RESET_MIN を許す (不応期明けの開始点)
    solver.add(V >= V_RESET_MIN, V <= V_TH_MAX)
    solver.add(I >= -I_MAX_ABS, I <= I_MAX_ABS)

    # forward Euler 更新 (R_MEM=1.0 固定)
    # V_next * tau_m = V * tau_m + DT * (V_REST - V + R_MEM * I)
    solver.add(V_next * tau_m == V * tau_m + DT * (V_REST - V + R_MEM * I))

    # bound: [V_RESET_MIN - margin, V_TH_MAX + margin]
    # (V_REST 基準でなく V_RESET_MIN 基準 ← 不応期明け V_reset スタート許容)
    upper = V_TH_MAX + safety_margin
    lower = V_RESET_MIN - safety_margin
    solver.add(z3.Or(V_next > upper, V_next < lower))

    result = solver.check()
    if result == z3.unsat:
        return SNNInvariantResult(
            ok=True, used_z3=True,
            reason=(
                f"unsat: V_next ∈ [{lower}, {upper}] mV for "
                f"V ∈ [{V_RESET_MIN},{V_TH_MAX}], I ∈ [-{I_MAX_ABS},{I_MAX_ABS}], "
                f"tau_m ∈ [{TAU_M_MIN},{TAU_M_MAX}]"
            ),
        )
    if result == z3.sat:
        m = solver.model()

        def _to_f(name):
            try:
                return float(m[name].as_decimal(6).rstrip("?"))
            except Exception:
                # Real (fraction) を float へ
                return float(m[name].as_fraction())

        ce = {
            "tau_m": _to_f(tau_m),
            "V": _to_f(V),
            "I": _to_f(I),
            "V_next": _to_f(V_next),
        }
        return SNNInvariantResult(
            ok=False, used_z3=True,
            reason=(
                f"sat: counterexample tau_m={ce['tau_m']:.3f}, V={ce['V']:.3f}, "
                f"I={ce['I']:.3f}, V_next={ce['V_next']:.3f}"
            ),
            counterexample=ce,
        )
    return SNNInvariantResult(
        ok=False, used_z3=True, reason=f"unknown / timeout ({result})",
    )


def verify_membrane_bounded_per_gene(
    gene: LIFGene,
    *,
    safety_margin: float = 1.0,
    timeout_ms: int = 1000,
) -> SNNInvariantResult:
    """単一 gene の tau_m に対し膜電位 bound が成立するか (online gate)."""
    if not _HAS_Z3:
        return SNNInvariantResult(ok=True, used_z3=False, reason="z3 unavailable, skip")

    g = gene.clipped()
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    V = z3.Real("V")
    I = z3.Real("I")
    V_next = z3.Real("V_next")

    # 単一 gene なので tau_m, V_th 固定値
    tau_m_const = g.tau_m
    v_th_const = g.V_th
    v_reset_const = g.V_reset

    solver.add(V >= v_reset_const, V <= v_th_const)  # spike 手前
    solver.add(I >= -I_MAX_ABS, I <= I_MAX_ABS)
    # tau_m_const > 0 確定なので両辺 tau_m_const 掛けなくて OK
    solver.add(V_next == V + (DT / tau_m_const) * (V_REST - V + R_MEM * I))

    upper = v_th_const + safety_margin
    lower = v_reset_const - safety_margin
    solver.add(z3.Or(V_next > upper, V_next < lower))

    result = solver.check()
    if result == z3.unsat:
        return SNNInvariantResult(
            ok=True, used_z3=True,
            reason=(
                f"unsat (per-gene): V_next ∈ [{lower}, {upper}] mV "
                f"(tau_m={tau_m_const:.2f}, V_th={v_th_const:.2f})"
            ),
        )
    if result == z3.sat:
        m = solver.model()

        def _to_f(name):
            try:
                return float(m[name].as_decimal(6).rstrip("?"))
            except Exception:
                return float(m[name].as_fraction())

        ce = {"V": _to_f(V), "I": _to_f(I), "V_next": _to_f(V_next)}
        return SNNInvariantResult(
            ok=False, used_z3=True,
            reason=(
                f"sat (per-gene): V={ce['V']:.3f}, I={ce['I']:.3f}, "
                f"V_next={ce['V_next']:.3f} > {upper}"
            ),
            counterexample=ce,
        )
    return SNNInvariantResult(ok=False, used_z3=True, reason=f"unknown ({result})")


# ---------------------------------------------------------------------------
# (3) Shielded RL hint sketch
# ---------------------------------------------------------------------------


def verify_shielded_rl_hint(
    gene: LIFGene,
    *,
    R_safe_hz: float = 200.0,
    timeout_ms: int = 1000,
) -> SNNInvariantResult:
    """**Sketch only**: SNN 出力 firing rate を policy action shield として
    rate_action <= R_safe_hz を gene が常に満たすか symbolic 検査.

    Codex Q5 推奨の **ProSh** (Probabilistic Shielding, Achiam et al. 2017〜)
    および **Adaptive GR(1) shielding** (Bloem et al., reactive synthesis) を
    verifier 統合する最小 sketch.

    本 PoC の sketch:
        - gene.t_ref から構造的に rate_max = 1000/t_ref Hz
        - shield 制約: rate_max <= R_safe_hz
        - Z3 で「rate_max > R_safe_hz」を sat 探索
        - sat なら gene は shield 違反 (例: t_ref=1 ms で rate_max=1000 Hz, R_safe=200 で違反)

    honest 留保:
    - 実 RL policy との接続なし (rate→action mapping は本 PoC スコープ外).
    - ProSh / Adaptive GR(1) との対応関係:
        * ProSh = 確率的に shield (本 sketch は決定論的上界 shield)
        * Adaptive GR(1) = LTL spec → reactive controller (本 sketch は単純不等式)
    - したがって「Shielded RL を実装した」ではなく
      「Shielded RL hint の Z3 sketch を提示した」が accurate な主張.

    Parameters
    ----------
    gene : LIFGene
        検査対象 gene.
    R_safe_hz : float
        shield 上界 (Hz). 既定 200 Hz (motor command 上限想定).
    timeout_ms : int
        Z3 timeout.

    Returns
    -------
    SNNInvariantResult
        ok=True なら gene は shield 制約を満たす (rate_max <= R_safe).
        ok=False なら gene は shield 違反.
    """
    if not _HAS_Z3:
        return SNNInvariantResult(
            ok=True, used_z3=False, reason="z3 unavailable, skip (vacuous True)",
        )

    g = gene.clipped()
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    rate_max = z3.Real("rate_max")
    # gene の構造的 rate_max = 1000 / t_ref
    upper = 1000.0 / g.t_ref
    solver.add(rate_max == upper)
    # shield 違反 = rate_max > R_safe
    solver.add(rate_max > R_safe_hz)

    result = solver.check()
    if result == z3.unsat:
        return SNNInvariantResult(
            ok=True, used_z3=True,
            reason=(
                f"unsat: rate_max={upper:.2f} Hz <= R_safe={R_safe_hz} Hz "
                f"(t_ref={g.t_ref:.2f} ms) — shield 制約満たす"
            ),
        )
    if result == z3.sat:
        return SNNInvariantResult(
            ok=False, used_z3=True,
            reason=(
                f"sat: rate_max={upper:.2f} Hz > R_safe={R_safe_hz} Hz "
                f"(t_ref={g.t_ref:.2f} ms) — shield 違反"
            ),
            counterexample={"rate_max": float(upper), "R_safe": float(R_safe_hz)},
        )
    return SNNInvariantResult(
        ok=False, used_z3=True, reason=f"unknown ({result})",
    )


__all__ = [
    "is_z3_available",
    "SNNInvariantResult",
    "verify_firing_rate_bound",
    "verify_firing_rate_per_gene",
    "verify_membrane_bounded",
    "verify_membrane_bounded_per_gene",
    "verify_shielded_rl_hint",
]
