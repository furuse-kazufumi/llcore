# SPDX-License-Identifier: Apache-2.0
"""State update invariants (Z3 で symbolic 検証).

検証する主不変量 (Stage 1a):
    **「StateUpdateGene が clip 範囲 (decay∈[0,1], mix∈[-1,1], gate_str∈[-2,2])
    に収まり、入力が |x| <= 1 のとき、|state| <= 1 が time step を跨いで保たれる」**

数学的根拠 (RWKV-style convex combination):
    s' = decay * s + (1-decay) * tanh(mix*x + gate_str*s)
    |s'| <= decay * |s| + (1-decay) * |tanh(.)|
        <= decay * |s| + (1-decay) * 1     (|tanh| <= 1)
    if |s| <= 1:
        <= decay + (1-decay) = 1
    → invariant preserved.

Z3 encoding:
- Real variables: decay, mix, gate_str, s, x, s_next
- Constraints: clip 範囲 + 入力範囲 + |s| <= 1 (仮定)
- tanh は直接表現不能 → ``|tanh(z)| <= min(|z|, 1)`` の上界で近似 (sound)
- 検査: 「|s_next| > 1 となる counterexample が存在するか」
- unsat → invariant 成立 (clip 範囲下では違反不能)
- sat → counterexample あり = invariant 違反、gene 設計に問題

honest 留保:
- tanh 近似は保守的 (上界 OK なら実値 OK だが、実値が OK でも上界が NG の偽 reject あり)
- 数値計算では float64 rounding で marginally 違反する可能性 (Z3 は exact rational)
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import z3
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False

from llcore.state_update import StateUpdateGene


def is_z3_available() -> bool:
    """z3-solver がインストールされ import 可能か返す."""
    return _HAS_Z3


@dataclass(frozen=True)
class InvariantResult:
    """Z3 検査結果.

    Attributes
    ----------
    ok : bool
        True = 不変量成立 (z3 unsat または z3 unavailable で skip)
    used_z3 : bool
        Z3 が実際に呼ばれたか (False なら fallback)
    reason : str
        verdict 説明 (counterexample 含む)
    counterexample : dict[str, float] | None
        sat の場合の反例 model (decay, mix, gate_str, s, x の値)
    """

    ok: bool
    used_z3: bool
    reason: str
    counterexample: dict[str, float] | None = None
    solver_status: str = "unknown"  # "sat" | "unsat" | "unknown" (z3 verdict; "unknown" also = no-z3/timeout)


def verify_state_norm_invariant(
    *,
    max_input_abs: float = 1.0,
    state_bound: float = 1.0,
    timeout_ms: int = 1000,
) -> InvariantResult:
    """**clip 範囲下の全 gene について** state_norm 有界 invariant を Z3 で検証.

    検査命題: ``∀(decay∈[0,1], mix∈[-1,1], gate_str∈[-2,2], |s|<=state_bound,
    |x|<=max_input_abs). |s_next| <= state_bound``

    実装: 反例探索 (∃ ... |s_next| > state_bound) を Z3 で。
    unsat = 反例なし = invariant 成立。

    Parameters
    ----------
    max_input_abs : float
        入力の上界 |x| <= max_input_abs.
    state_bound : float
        state 不変量上界 |s| <= state_bound.
    timeout_ms : int
        Z3 timeout (ms).

    Returns
    -------
    InvariantResult
        ok=True (unsat) なら全 clip gene で invariant 保たれる証明。
        ok=False (sat) なら counterexample あり。
    """
    if not _HAS_Z3:
        return InvariantResult(
            ok=True,
            used_z3=False,
            reason="z3 not installed, invariant assumed by mathematical argument (sound: RWKV-style convex combination)",
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    # Gene parameters (clip 範囲)
    decay = z3.Real("decay")
    mix = z3.Real("mix")
    gate_str = z3.Real("gate_str")
    solver.add(decay >= 0, decay <= 1)
    solver.add(mix >= -1, mix <= 1)
    solver.add(gate_str >= -2, gate_str <= 2)

    # state / input (前ステップ仮定: |s| <= state_bound, |x| <= max_input_abs)
    s = z3.Real("s")
    x = z3.Real("x")
    solver.add(s >= -state_bound, s <= state_bound)
    solver.add(x >= -max_input_abs, x <= max_input_abs)

    # tanh argument
    pre = mix * x + gate_str * s
    # tanh upper/lower (sound bound: |tanh(z)| <= min(|z|, 1))
    # → upper = if |pre| <= 1: pre else sign(pre)
    # 簡単のため tanh ∈ [-1, 1] で表現 (もっと厳しい近似)
    tanh_val = z3.Real("tanh_val")
    solver.add(tanh_val >= -1, tanh_val <= 1)
    # tanh は単調かつ符号一致なので: sign(pre) == sign(tanh_val) のヒント
    # ただし sound 上界のためここでは制約しない (free in [-1, 1])

    # next state
    s_next = decay * s + (1 - decay) * tanh_val

    # 反例: |s_next| > state_bound が成立するか
    solver.add(z3.Or(s_next > state_bound, s_next < -state_bound))

    result = solver.check()
    if result == z3.unsat:
        # 反例なし = invariant 成立
        return InvariantResult(
            ok=True,
            used_z3=True,
            reason=f"unsat: invariant |state|<={state_bound} holds for all clipped gene with |x|<={max_input_abs}",
        )
    elif result == z3.sat:
        m = solver.model()

        def _eval_float(var: z3.ArithRef) -> float:
            v = m.eval(var, model_completion=True)
            try:
                return float(v.as_decimal(10).rstrip("?"))
            except Exception:
                return float(v.numerator_as_long()) / float(v.denominator_as_long())

        ce = {
            "decay": _eval_float(decay),
            "mix": _eval_float(mix),
            "gate_str": _eval_float(gate_str),
            "s": _eval_float(s),
            "x": _eval_float(x),
            "tanh_val": _eval_float(tanh_val),
        }
        return InvariantResult(
            ok=False,
            used_z3=True,
            reason=f"sat: counterexample exists where |s_next|>{state_bound}",
            counterexample=ce,
        )
    else:
        return InvariantResult(
            ok=False,
            used_z3=True,
            reason=f"z3 returned {result} (likely timeout or unknown)",
        )


def verify_gene_safe(
    gene: StateUpdateGene,
    *,
    max_input_abs: float = 1.0,
    state_bound: float = 1.0,
    timeout_ms: int = 500,
) -> InvariantResult:
    """**単一 gene** に対し state_norm 有界 invariant を Z3 で検証.

    clip された具体 gene 値で symbolic 検査 (∃ counter |s|, |x|).
    online gate として進化ループから呼ばれる想定。

    Returns
    -------
    InvariantResult
        ok=True なら gene は invariant を破らない (= 進化集団に admit)。
        ok=False なら reject (進化選択圧で淘汰)。
    """
    if not _HAS_Z3:
        return InvariantResult(
            ok=True,
            used_z3=False,
            reason="z3 not installed, gene admitted by default",
        )

    # v2 fix (Codex 2026-05-29 指摘): mix / gate_str を式に正しく使う.
    # v1 では tanh_val を free in [-1, 1] にしていたが、これでは gene-specific
    # reasoning にならない (decay のみで判定)。v2 では tanh の引数
    # ``pre = mix*x + gate_str*s`` を計算し、|tanh(pre)| <= min(|pre|, 1) の
    # **tighter sound bound** を Z3 で表現する。
    g = gene.clipped()
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    s = z3.Real("s")
    x = z3.Real("x")
    solver.add(s >= -state_bound, s <= state_bound)
    solver.add(x >= -max_input_abs, x <= max_input_abs)

    decay_q = z3.RealVal(g.decay)
    mix_q = z3.RealVal(g.mix)
    gate_q = z3.RealVal(g.gate_str)

    # tanh の引数 = pre, tanh の値は |tanh(pre)| <= min(|pre|, 1) (sound tighter)
    pre = mix_q * x + gate_q * s
    tanh_val = z3.Real("tanh_val")
    # |tanh(pre)| <= |pre| (tanh は 1-Lipschitz, tanh(0)=0)
    solver.add(tanh_val * tanh_val <= pre * pre)
    # |tanh(pre)| <= 1
    solver.add(tanh_val >= -1, tanh_val <= 1)
    # tanh と pre は符号一致 (tanh は奇関数)
    solver.add(tanh_val * pre >= 0)

    s_next = decay_q * s + (1 - decay_q) * tanh_val
    solver.add(z3.Or(s_next > state_bound, s_next < -state_bound))

    result = solver.check()
    if result == z3.unsat:
        return InvariantResult(
            ok=True,
            used_z3=True,
            reason=f"gene admit: |state|<={state_bound} preserved for d={g.decay:.3f}, m={g.mix:.3f}, g={g.gate_str:.3f}",
            solver_status="unsat",
        )
    elif result == z3.sat:
        return InvariantResult(
            ok=False,
            used_z3=True,
            reason=f"gene reject: invariant violation found (d={g.decay:.3f}, m={g.mix:.3f}, g={g.gate_str:.3f})",
            solver_status="sat",
        )
    else:
        return InvariantResult(
            ok=False,
            used_z3=True,
            reason=f"z3 returned {result}",
            solver_status="unknown",
        )
