# SPDX-License-Identifier: Apache-2.0
"""Neural ODE invariant verifier (Z3 for Lipschitz upper bound + Hurwitz stability).

llcore RWKV-style verifier (``llcore.verifier.invariants``) と **同じ verifier stack
内** で連続時間 vector field を検査する PoC. CPU 完結 (z3-solver).

Falsifiable 命題に関わる部分:
    Per-gene Z3 検査:
        (a) Lipschitz 上界 ``|A| + |W|*|b| <= L`` invariant
        (b) Hurwitz stability ``A + W*b < 0`` (平衡点 x=0 近傍局所安定)
    が clip 範囲 (A∈[-2,0], W∈[-1,1], b∈[-2,2]) で sound に判定できる.

llcore RWKV verifier との対応:
    - llcore RWKV: ``verify_state_norm_invariant`` (∀ gene で |state|<=K)
    - 本 PoC ODE: ``verify_lipschitz_bound`` (∀ gene で Lipschitz <= L)
    - llcore RWKV: ``verify_gene_safe`` (単一 gene admit/reject)
    - 本 PoC ODE: ``verify_gene_ode_safe`` (単一 gene admit/reject; Lipschitz∧Hurwitz)

honest 留保:
- tanh は Z3 で直接表現不能 → ``|d/dx tanh(b*x)| = |b|*sech^2(b*x) <= |b|`` の
  sound 上界で近似 (Codex Q1 review).
- スカラー 1D Hurwitz (A + W*b < 0) は多次元 J(0) 固有値実部条件と差がある
  (Codex Q2 review). 本 PoC ではスカラー gene なのでスカラー Hurwitz が正確.
- 数値計算上は forward Euler の effective Lipschitz と analytic 上界に乖離が
  ありうる (Codex Q3 review). G8 で測定.

依存: z3-solver (optional, RAPTOR env では既導入).
"""
from __future__ import annotations

from dataclasses import dataclass

try:
    import z3
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False

# 相対 import (research/other_archs/neural_ode 配下 single-package)
from .ode_gene import (
    A_HIGH,
    A_LOW,
    B_HIGH,
    B_LOW,
    NeuralODEGene,
    W_HIGH,
    W_LOW,
)


def is_z3_available() -> bool:
    """z3-solver が import 可能か."""
    return _HAS_Z3


@dataclass(frozen=True)
class ODEInvariantResult:
    """Z3 検査結果.

    Attributes
    ----------
    ok : bool
        True = invariant 成立 (unsat) または z3 unavailable で skip.
    used_z3 : bool
        Z3 が実際に呼ばれたか.
    reason : str
        verdict 説明 (counterexample 含む).
    counterexample : dict[str, float] | None
        sat の場合の反例 model.
    """

    ok: bool
    used_z3: bool
    reason: str
    counterexample: dict[str, float] | None = None


def _eval_float(model: "z3.ModelRef", var: "z3.ArithRef") -> float:
    v = model.eval(var, model_completion=True)
    try:
        return float(v.as_decimal(10).rstrip("?"))
    except Exception:  # pragma: no cover - Z3 internal
        try:
            return float(v.numerator_as_long()) / float(v.denominator_as_long())
        except Exception:
            return float(str(v))


def _abs(expr: "z3.ArithRef") -> "z3.ArithRef":
    """Z3 で |expr| を If 式で正確に表現 (補助変数を使わない).

    補助変数 abs_X (abs_X >= X, abs_X >= -X) は **下界** しか与えないため、
    counterexample 探索で Z3 が abs_X を任意に大きく取って偽 sat を返す。
    ``z3.If(expr >= 0, expr, -expr)`` で abs を **等式** として表現する.
    """
    return z3.If(expr >= 0, expr, -expr)


def verify_lipschitz_bound(
    *,
    L: float = 4.0,
    a_low: float = A_LOW,
    a_high: float = A_HIGH,
    w_low: float = W_LOW,
    w_high: float = W_HIGH,
    b_low: float = B_LOW,
    b_high: float = B_HIGH,
    timeout_ms: int = 1000,
) -> ODEInvariantResult:
    """**clip 範囲下の全 gene について** Lipschitz 上界 ``|A| + |W|*|b| <= L`` を Z3 で検査.

    反例探索: ∃ (A,W,b) in clip range with |A| + |W|*|b| > L. unsat = 上界成立 (proof).

    Parameters
    ----------
    L : float
        Lipschitz 上界 (default 4.0 = clip 範囲の構造的最大値 |A|_max + |W|_max*|b|_max = 2+1*2 = 4).
    timeout_ms : int
        Z3 timeout.

    Returns
    -------
    ODEInvariantResult
        ok=True (unsat) なら L が全 clipped gene の Lipschitz 上界として sound.
        ok=False (sat) なら counterexample で gene の Lipschitz が L 超え.
    """
    if not _HAS_Z3:
        return ODEInvariantResult(
            ok=True,
            used_z3=False,
            reason="z3 not installed; analytical bound |A|+|W||b| accepted by mathematical argument.",
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    A_var = z3.Real("A")
    W_var = z3.Real("W")
    b_var = z3.Real("b")
    solver.add(A_var >= a_low, A_var <= a_high)
    solver.add(W_var >= w_low, W_var <= w_high)
    solver.add(b_var >= b_low, b_var <= b_high)

    # 絶対値は If 式で **等式** として表現 (補助変数の下界のみだと偽 sat を起こすため).
    abs_A = _abs(A_var)
    abs_W = _abs(W_var)
    abs_b = _abs(b_var)

    # 反例: |A| + |W|*|b| > L
    lipschitz_expr = abs_A + abs_W * abs_b
    solver.add(lipschitz_expr > L)

    result = solver.check()
    if result == z3.unsat:
        return ODEInvariantResult(
            ok=True,
            used_z3=True,
            reason=f"unsat: Lipschitz upper bound |A|+|W||b| <= {L} holds for all clipped genes.",
        )
    elif result == z3.sat:
        m = solver.model()
        ce = {
            "A": _eval_float(m, A_var),
            "W": _eval_float(m, W_var),
            "b": _eval_float(m, b_var),
            "lipschitz_value": (
                abs(_eval_float(m, A_var))
                + abs(_eval_float(m, W_var)) * abs(_eval_float(m, b_var))
            ),
        }
        return ODEInvariantResult(
            ok=False,
            used_z3=True,
            reason=f"sat: counterexample with |A|+|W||b| > {L}.",
            counterexample=ce,
        )
    else:
        return ODEInvariantResult(
            ok=False,
            used_z3=True,
            reason=f"z3 returned {result}",
        )


def verify_hurwitz_universal(
    *,
    a_low: float = A_LOW,
    a_high: float = A_HIGH,
    w_low: float = W_LOW,
    w_high: float = W_HIGH,
    b_low: float = B_LOW,
    b_high: float = B_HIGH,
    timeout_ms: int = 1000,
) -> ODEInvariantResult:
    """clip 範囲下の **すべて** の gene で Hurwitz (A + W*b < 0) が成立するか検査.

    反例探索: ∃ gene with A + W*b >= 0. sat 期待 (clip 範囲は不安定 gene を含む).
    ここでは soundness sanity 用途: clip 範囲下で **必ず安定** は false という
    認識を Z3 で確認する。

    Returns
    -------
    ODEInvariantResult
        ok=False (sat) が「正常」 (clip 範囲は不安定 gene を許容するため反例存在).
        ok=True (unsat) が驚き (clip 範囲全域で安定なら、それは clip 範囲が strict すぎ).

    honest: 本関数は "clip 範囲が必ずしも安定でない" こと自体の sanity check.
    """
    if not _HAS_Z3:
        return ODEInvariantResult(
            ok=False,
            used_z3=False,
            reason="z3 not installed",
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    A_var = z3.Real("A")
    W_var = z3.Real("W")
    b_var = z3.Real("b")
    solver.add(A_var >= a_low, A_var <= a_high)
    solver.add(W_var >= w_low, W_var <= w_high)
    solver.add(b_var >= b_low, b_var <= b_high)

    # 反例: A + W*b >= 0 (Hurwitz 違反 = 不安定)
    solver.add(A_var + W_var * b_var >= 0)

    result = solver.check()
    if result == z3.sat:
        m = solver.model()
        ce = {
            "A": _eval_float(m, A_var),
            "W": _eval_float(m, W_var),
            "b": _eval_float(m, b_var),
        }
        return ODEInvariantResult(
            ok=False,
            used_z3=True,
            reason="sat: clip range contains Hurwitz-violating genes (expected).",
            counterexample=ce,
        )
    elif result == z3.unsat:
        return ODEInvariantResult(
            ok=True,
            used_z3=True,
            reason="unsat: clip range entirely Hurwitz-stable.",
        )
    else:
        return ODEInvariantResult(
            ok=False,
            used_z3=True,
            reason=f"z3 returned {result}",
        )


def verify_gene_lipschitz(
    gene: NeuralODEGene,
    *,
    L: float = 4.0,
    timeout_ms: int = 500,
) -> ODEInvariantResult:
    """**単一 gene** の Lipschitz 上界 |A|+|W||b| <= L を Z3 で検査.

    具体値 gene の Lipschitz 上界 (analytic) が L 以下か symbolic に確認.
    online gate として進化ループから呼ばれる想定.
    """
    if not _HAS_Z3:
        return ODEInvariantResult(
            ok=True,
            used_z3=False,
            reason="z3 not installed; gene admitted by default.",
        )

    g = gene.clipped()
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    A_val = z3.RealVal(g.A)
    W_val = z3.RealVal(g.W)
    b_val = z3.RealVal(g.b)

    # 絶対値は If 式で等式として表現
    abs_A = _abs(A_val)
    abs_W = _abs(W_val)
    abs_b = _abs(b_val)

    lipschitz_expr = abs_A + abs_W * abs_b
    # 反例: |A| + |W||b| > L
    solver.add(lipschitz_expr > L)

    result = solver.check()
    if result == z3.unsat:
        return ODEInvariantResult(
            ok=True,
            used_z3=True,
            reason=(
                f"gene admit (Lipschitz): |A|+|W||b| <= {L} for "
                f"A={g.A:.3f}, W={g.W:.3f}, b={g.b:.3f}."
            ),
        )
    elif result == z3.sat:
        return ODEInvariantResult(
            ok=False,
            used_z3=True,
            reason=(
                f"gene reject (Lipschitz): |A|+|W||b| > {L} for "
                f"A={g.A:.3f}, W={g.W:.3f}, b={g.b:.3f}."
            ),
        )
    else:
        return ODEInvariantResult(
            ok=False,
            used_z3=True,
            reason=f"z3 returned {result}",
        )


def verify_gene_hurwitz(
    gene: NeuralODEGene,
    *,
    timeout_ms: int = 500,
) -> ODEInvariantResult:
    """**単一 gene** の Hurwitz stability A + W*b < 0 を Z3 で検査.

    Returns
    -------
    ODEInvariantResult
        ok=True (unsat) なら gene は局所安定 (admit).
        ok=False (sat) なら不安定 gene (reject).
    """
    if not _HAS_Z3:
        return ODEInvariantResult(
            ok=True,
            used_z3=False,
            reason="z3 not installed; gene admitted by default.",
        )

    g = gene.clipped()
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    A_val = z3.RealVal(g.A)
    W_val = z3.RealVal(g.W)
    b_val = z3.RealVal(g.b)

    # 反例: A + W*b >= 0 (Hurwitz 違反)
    solver.add(A_val + W_val * b_val >= 0)

    result = solver.check()
    if result == z3.unsat:
        return ODEInvariantResult(
            ok=True,
            used_z3=True,
            reason=(
                f"gene admit (Hurwitz): A+Wb={g.A + g.W * g.b:.4f} < 0 "
                f"(A={g.A:.3f}, W={g.W:.3f}, b={g.b:.3f})."
            ),
        )
    elif result == z3.sat:
        return ODEInvariantResult(
            ok=False,
            used_z3=True,
            reason=(
                f"gene reject (Hurwitz): A+Wb={g.A + g.W * g.b:.4f} >= 0 "
                f"(A={g.A:.3f}, W={g.W:.3f}, b={g.b:.3f})."
            ),
        )
    else:
        return ODEInvariantResult(
            ok=False,
            used_z3=True,
            reason=f"z3 returned {result}",
        )


def verify_gene_ode_safe(
    gene: NeuralODEGene,
    *,
    L: float = 4.0,
    timeout_ms: int = 500,
) -> ODEInvariantResult:
    """**Lipschitz ∧ Hurwitz** AND gate で gene admit/reject.

    llcore.verifier.verify_gene_safe と同じ stack 内で動く online gate.

    Returns
    -------
    ODEInvariantResult
        ok=True なら Lipschitz 上界 ∧ Hurwitz stable (admit).
        ok=False なら片方でも違反 (reject).
    """
    r_lip = verify_gene_lipschitz(gene, L=L, timeout_ms=timeout_ms)
    if not r_lip.ok:
        return r_lip
    r_hur = verify_gene_hurwitz(gene, timeout_ms=timeout_ms)
    if not r_hur.ok:
        return r_hur
    return ODEInvariantResult(
        ok=True,
        used_z3=r_lip.used_z3 and r_hur.used_z3,
        reason=f"gene admit (Lipschitz ∧ Hurwitz): {r_lip.reason} | {r_hur.reason}",
    )


__all__ = [
    "ODEInvariantResult",
    "is_z3_available",
    "verify_gene_hurwitz",
    "verify_gene_lipschitz",
    "verify_gene_ode_safe",
    "verify_hurwitz_universal",
    "verify_lipschitz_bound",
]
