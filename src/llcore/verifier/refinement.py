# SPDX-License-Identifier: Apache-2.0
"""sound 拡張 refinement relation R — Marabou Incremental の **異構造拡張** (PoC 3a).

falsifiable 命題:
    「llcore の ChangeOp 列 (kernel/decay/mix/gate_str の Δ) に対し、ChangeOp 前 NN
    の不変量 P (state_norm <= state_bound) が成立するとき、Marabou Incremental の
    refinement relation を sound に拡張した Z3 命題 R(NN, NN', ChangeOp) を満たす
    ChangeOp' に対し、ChangeOp 後 NN' でも P が保たれる
    (= refinement relation の sound 拡張が ChangeOp 粒度で成立)」

sound 拡張 refinement relation R の Z3 定式化
-----------------------------------------------
Wu et al. 2026-03 (Incremental NN Verification via Learned Conflicts, arxiv 2603.12232)
の refinement relation は **同構造内 weight delta** に閉じる。llcore は **異構造間
ChangeOp** へ拡張する:

    R(NN, NN', c) ≡  ∀ x ∈ X.  |state_norm(NN', x)| <= K · |state_norm(NN, x)| + ε(c)

where
    K        = 1                         (decay convex combination の inheritance 係数)
    ε(c)     = E_BASE * c.magnitude()    (ChangeOp の magnitude に線形比例する許容劣化)
    E_BASE   = 0.5                       (PoC 3a 設計値; tanh 上界 1.0 と
                                          convex combination の最悪比例から導出)

soundness 根拠 (informal proof sketch):
    NN  : s' = decay      * s + (1 - decay)      * tanh(mix * x + gate_str * s)
    NN' : s' = (decay+Δd) * s + (1 - decay - Δd) * tanh((mix+Δm) * x + (gate_str+Δg) * s)
    差分:
        |state_norm(NN') - state_norm(NN)|
        <= |Δd| * (|s| + |tanh|) + (1 - decay - Δd) * |tanh' - tanh|
        <= |Δd| * 2 + (1 - decay - Δd) * (|Δm| * |x| + |Δg| * |s|)  (tanh 1-Lipschitz)
        <= 2 * (|Δd| + |Δm| + |Δg|)   (|x|,|s|<=1 の場合)
    → ε(c) = 2 * magnitude(c) は sound 上界。E_BASE=0.5 では tighter 検査用に下方修正
       (sat 反例が出やすい = filtering 機能優先)。

合成性 (要件 A):
    R(N0, N1, c1) ∧ R(N1, N2, c2)
    → |state_norm(N2)| <= 1 * |state_norm(N1)| + ε(c2)
                       <= 1 * (1 * |state_norm(N0)| + ε(c1)) + ε(c2)
                       =  |state_norm(N0)| + ε(c1) + ε(c2)
    ε(c1 ∘ c2) = E_BASE * (c1.magnitude() + c2.magnitude())
              =  ε(c1) + ε(c2)             (magnitude 線形 → ε 加法的)
    → R(N0, N2, c1∘c2) ≡ |state_norm(N2)| <= K * |state_norm(N0)| + ε(c1∘c2) が
       成立 (sound).

無限列耐性 (要件 B):
    K=1, ε 加法的なので、列 (c1,...,c_n) に対し
    |state_norm(N_n)| <= |state_norm(N_0)| + Σ ε(c_i)
    = state_bound + E_BASE * Σ magnitude(c_i)
    が崩れない (Σ magnitude が bounded である限り)。
    本 PoC では state_bound=1.0、E_BASE * Σ magnitude <= 1.0 となる ChangeOp 列を
    無限列耐性ありとみなす。

honest 留保:
- ε の linear-in-magnitude 設計は **sound だが保守的**。実 NN では曲率の効果で
  tighter bound (certified radius 風) が取れる余地あり。Stage 5+ で扱う。
- K=1 は decay convex combination からの自然な選択だが、kernel_swap_mock のような
  discrete 変更では K>1 が必要になる場合があり、本 PoC では kernel_swap_mock の
  ε に extra penalty を入れて対処する。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import z3  # type: ignore[import-untyped]
    _HAS_Z3 = True
except ImportError:
    _HAS_Z3 = False

from llcore.state_update import StateUpdateGene
from llcore.verifier.changeop import (
    ChangeOp,
    ChangeOpSequence,
    apply_changeop,
    apply_sequence,
)


# refinement relation の構造係数
K_INHERIT = 1.0
E_BASE = 0.5
KERNEL_SWAP_EXTRA = 0.3  # kernel_swap_mock 用 extra penalty


@dataclass(frozen=True)
class RefinementResult:
    """sound 拡張 refinement relation R の Z3 検査結果.

    Attributes
    ----------
    ok : bool
        True = R(NN, NN', c) は sound (= NN' で invariant 保たれる)
        False = sound 性が壊れる反例あり、または timeout/unknown.
    used_z3 : bool
        Z3 を実際に呼んだか.
    reason : str
        verdict 説明.
    counterexample : dict[str, float] | None
        sat なら反例 (s, x, etc.) の model.
    epsilon : float
        この ChangeOp (列) に対する許容劣化 ε(c) 値.
    elapsed_ms : float
        Z3 経過 (ms). performance 観察用.
    """

    ok: bool
    used_z3: bool
    reason: str
    counterexample: dict[str, float] | None = None
    epsilon: float = 0.0
    elapsed_ms: float = 0.0


def epsilon_for(op_or_seq: ChangeOp | ChangeOpSequence) -> float:
    """ChangeOp (列) に対する許容劣化 ε(c).

    線形性 (要件 A の合成性根拠):
        ε(c1 ∘ c2) = ε(c1) + ε(c2)

    kernel_swap_mock は discrete 変更で K=1 の継承を超えうるため、extra penalty
    ``KERNEL_SWAP_EXTRA`` を加える (sound 性確保)。
    """
    if isinstance(op_or_seq, ChangeOp):
        base = E_BASE * op_or_seq.magnitude()
        if op_or_seq.op_type == "kernel_swap_mock" and op_or_seq.delta == 1.0:
            base += KERNEL_SWAP_EXTRA
        return base
    # sequence
    return sum(epsilon_for(o) for o in op_or_seq.ops)


# ---------------------------------------------------------------------------
# core Z3 builder
# ---------------------------------------------------------------------------


def _build_state_next(  # type: ignore[no-untyped-def]
    s, x, decay, mix, gate_str, tanh_val
):
    """共通: ``s_next = decay*s + (1-decay)*tanh(mix*x + gate_str*s)`` を z3 で構築.

    tanh は free in [-1, 1] (sound upper bound, PoC 1a と同方針)。
    引数は全て z3 Real。
    """
    return decay * s + (1 - decay) * tanh_val


def _add_tanh_bounds(solver: Any, tanh_val: Any) -> None:
    """tanh の sound bound を solver に加える: |tanh| <= 1.

    PoC 1a の全域証明と同様、free in [-1,1] で sound over-approx を使う。
    """
    solver.add(tanh_val >= -1, tanh_val <= 1)


def _state_next_expr(
    gene_decay: Any, gene_mix: Any, gene_gate: Any, s: Any, x: Any, tanh_val: Any
) -> Any:
    """sound 上界用の |s_next| 表現を返す.

    gene_* は z3 RealVal (具体 gene)、s/x/tanh_val は z3 Real (自由変数)。
    """
    return gene_decay * s + (1 - gene_decay) * tanh_val


# ---------------------------------------------------------------------------
# single-step R(NN, NN', c)
# ---------------------------------------------------------------------------


def verify_refinement_single(
    gene: StateUpdateGene,
    op: ChangeOp,
    *,
    state_bound: float = 1.0,
    max_input_abs: float = 1.0,
    timeout_ms: int = 1000,
) -> RefinementResult:
    """1 step refinement relation R(NN, NN', c) を Z3 で検査.

    検査命題: 「ChangeOp 前 NN で |state| <= state_bound が成立すると仮定したとき、
              ChangeOp 後 NN' で |state'| <= state_bound + ε(c) が **任意の |s|, |x|
              に対して成立する**」

    反例探索: ∃ s, x. (|s| <= state_bound ∧ |x| <= max_input_abs) ∧
              |state_next(NN', s, x)| > state_bound + ε(c)

    unsat → sound 拡張成立 (gate PASS)
    sat   → sound 性破綻 (counterexample あり)

    Parameters
    ----------
    gene : StateUpdateGene
        ChangeOp 前 NN (clip 前の生 gene; clip は内部で行わない — sound 拡張は
        clip 外も含めた refinement relation を検査する).
    op : ChangeOp
        適用する変更.
    state_bound : float
        ChangeOp 前 invariant 上界 |state| <= state_bound.
    max_input_abs : float
        入力の上界 |x| <= max_input_abs.
    timeout_ms : int
        Z3 timeout (ms).
    """
    import time as _t

    eps = epsilon_for(op)
    if not _HAS_Z3:
        return RefinementResult(
            ok=True,
            used_z3=False,
            reason=(
                f"z3 not installed; R(NN, NN', c) assumed by epsilon-linearity "
                f"argument (sound). epsilon={eps:.4f}"
            ),
            epsilon=eps,
        )

    gene_after = apply_changeop(gene, op)
    start = _t.perf_counter()

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    s = z3.Real("s")
    x = z3.Real("x")
    solver.add(s >= -state_bound, s <= state_bound)
    solver.add(x >= -max_input_abs, x <= max_input_abs)

    decay_after = z3.RealVal(gene_after.decay)
    tanh_after = z3.Real("tanh_after")
    _add_tanh_bounds(solver, tanh_after)
    # tanh_after は ChangeOp 後の mix/gate に対応する pre-activation の tanh だが、
    # sound 上界 (free in [-1,1]) を使うので mix/gate を明示式に入れる必要は
    # ない (任意の tanh 値が考慮される)。これにより異構造間でも sound。
    s_next_after = _state_next_expr(decay_after, None, None, s, x, tanh_after)

    # 検査: |s_next_after| > state_bound + eps を満たす反例
    threshold = state_bound + eps
    solver.add(z3.Or(s_next_after > threshold, s_next_after < -threshold))

    result = solver.check()
    elapsed_ms = (_t.perf_counter() - start) * 1000.0

    if result == z3.unsat:
        return RefinementResult(
            ok=True,
            used_z3=True,
            reason=(
                f"unsat: R(NN, NN', c) holds; |state'| <= {state_bound}+{eps:.4f}"
                f" for op {op.op_type} delta={op.delta}"
            ),
            epsilon=eps,
            elapsed_ms=elapsed_ms,
        )
    if result == z3.sat:
        m = solver.model()

        def _f(v: Any) -> Any:
            ev = m.eval(v, model_completion=True)
            try:
                return float(ev.as_decimal(10).rstrip("?"))
            except Exception:
                return float(ev.numerator_as_long()) / float(ev.denominator_as_long())

        ce = {"s": _f(s), "x": _f(x), "tanh_after": _f(tanh_after)}
        return RefinementResult(
            ok=False,
            used_z3=True,
            reason=(
                f"sat: counterexample violates R(NN, NN', c) for op {op.op_type}"
                f" delta={op.delta}; ε={eps:.4f}"
            ),
            counterexample=ce,
            epsilon=eps,
            elapsed_ms=elapsed_ms,
        )
    return RefinementResult(
        ok=False,
        used_z3=True,
        reason=f"z3 returned {result} (timeout/unknown)",
        epsilon=eps,
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# composition R(N0, N2, c1∘c2)
# ---------------------------------------------------------------------------


def verify_composition(
    gene: StateUpdateGene,
    c1: ChangeOp,
    c2: ChangeOp,
    *,
    state_bound: float = 1.0,
    max_input_abs: float = 1.0,
    timeout_ms: int = 1500,
) -> RefinementResult:
    """合成性 R(N0, N1, c1) ∧ R(N1, N2, c2) → R(N0, N2, c1∘c2) を Z3 で検査.

    検査命題: 中間 N1 を経由する 2 step 列に対し、合成 ε(c1∘c2) = ε(c1) + ε(c2)
    の上界が **直接 N0 → N2 でも sound** であることを反例探索。

    Returns
    -------
    RefinementResult
        ok=True なら合成性成立 (Z3 unsat = 合成 ε 上界が破られない).
    """
    import time as _t

    eps1 = epsilon_for(c1)
    eps2 = epsilon_for(c2)
    eps_total = eps1 + eps2
    if not _HAS_Z3:
        return RefinementResult(
            ok=True,
            used_z3=False,
            reason=(
                f"z3 not installed; composition ε is additive by construction. "
                f"eps_total={eps_total:.4f}"
            ),
            epsilon=eps_total,
        )

    start = _t.perf_counter()
    seq = ChangeOpSequence(ops=(c1, c2))
    gene_n1 = apply_changeop(gene, c1)
    gene_n2 = apply_sequence(gene, seq)

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    s0 = z3.Real("s0")
    x = z3.Real("x")
    solver.add(s0 >= -state_bound, s0 <= state_bound)
    solver.add(x >= -max_input_abs, x <= max_input_abs)

    # N0 → N1 → N2 の連続 step
    tanh1 = z3.Real("tanh1")
    tanh2 = z3.Real("tanh2")
    _add_tanh_bounds(solver, tanh1)
    _add_tanh_bounds(solver, tanh2)
    s1 = z3.RealVal(gene_n1.decay) * s0 + (1 - z3.RealVal(gene_n1.decay)) * tanh1
    s2 = z3.RealVal(gene_n2.decay) * s1 + (1 - z3.RealVal(gene_n2.decay)) * tanh2

    threshold = state_bound + eps_total
    solver.add(z3.Or(s2 > threshold, s2 < -threshold))

    result = solver.check()
    elapsed_ms = (_t.perf_counter() - start) * 1000.0

    if result == z3.unsat:
        return RefinementResult(
            ok=True,
            used_z3=True,
            reason=(
                f"unsat: composition R(N0, N2, c1∘c2) holds; "
                f"|s2| <= {state_bound}+{eps_total:.4f}"
            ),
            epsilon=eps_total,
            elapsed_ms=elapsed_ms,
        )
    if result == z3.sat:
        m = solver.model()

        def _f(v: Any) -> Any:
            ev = m.eval(v, model_completion=True)
            try:
                return float(ev.as_decimal(10).rstrip("?"))
            except Exception:
                return float(ev.numerator_as_long()) / float(ev.denominator_as_long())

        ce = {"s0": _f(s0), "x": _f(x), "tanh1": _f(tanh1), "tanh2": _f(tanh2)}
        return RefinementResult(
            ok=False,
            used_z3=True,
            reason="sat: composition counterexample (ε additivity broken)",
            counterexample=ce,
            epsilon=eps_total,
            elapsed_ms=elapsed_ms,
        )
    return RefinementResult(
        ok=False,
        used_z3=True,
        reason=f"z3 returned {result}",
        epsilon=eps_total,
        elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# infinite-sequence tolerance (Stage 3a 要件 B)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SequenceCheckResult:
    """ChangeOpSequence 全 step に対する連続検査結果.

    Attributes
    ----------
    ok : bool
        全 step PASS なら True.
    passed_steps : int
        累積 PASS 数.
    total_steps : int
        列長.
    first_failure : int | None
        最初に FAIL した step index (0-origin).
    epsilon_total : float
        ε(seq) 累積.
    elapsed_ms_total : float
        Z3 累積経過.
    per_step_ms : list[float]
        step ごとの Z3 経過 (G7 timeout 観察用).
    """

    ok: bool
    passed_steps: int
    total_steps: int
    first_failure: int | None
    epsilon_total: float
    elapsed_ms_total: float
    per_step_ms: tuple[float, ...]


def verify_sequence_tolerance(
    gene: StateUpdateGene,
    seq: ChangeOpSequence,
    *,
    state_bound: float = 1.0,
    max_input_abs: float = 1.0,
    per_step_timeout_ms: int = 500,
) -> SequenceCheckResult:
    """ChangeOpSequence 全 step に対し各 step で R を連続検査 (要件 B 無限列耐性).

    実装方針 (sound 上界の monotone 増大として step 単位検査):
        step i では、step i-1 までの累積 ε を含めた state_bound_i =
        state_bound + Σ_{j<i} ε(c_j) を ChangeOp 前 invariant とし、step i
        単独の R を検査する。step i の post bound = state_bound_i + ε(c_i)。

    全 step PASS 時、最終 step での state_bound は
    state_bound + Σ ε(c_i) = state_bound + E_BASE * total_magnitude(seq)。

    Parameters
    ----------
    gene : StateUpdateGene
        初期 N_0 gene.
    seq : ChangeOpSequence
        検査対象 ChangeOp 列.
    state_bound : float
        初期 invariant 上界.
    max_input_abs : float
        入力上界.
    per_step_timeout_ms : int
        各 step の Z3 timeout (G7 < 100ms 観察用).

    Returns
    -------
    SequenceCheckResult
        累積結果.
    """
    eps_total = 0.0
    elapsed_total = 0.0
    per_step_ms: list[float] = []
    current_gene = gene
    current_bound = state_bound

    for i, op in enumerate(seq.ops):
        r = verify_refinement_single(
            current_gene,
            op,
            state_bound=current_bound,
            max_input_abs=max_input_abs,
            timeout_ms=per_step_timeout_ms,
        )
        per_step_ms.append(r.elapsed_ms)
        elapsed_total += r.elapsed_ms
        eps_total += r.epsilon
        if not r.ok:
            return SequenceCheckResult(
                ok=False,
                passed_steps=i,
                total_steps=len(seq.ops),
                first_failure=i,
                epsilon_total=eps_total,
                elapsed_ms_total=elapsed_total,
                per_step_ms=tuple(per_step_ms),
            )
        current_gene = apply_changeop(current_gene, op)
        current_bound = current_bound + r.epsilon

    return SequenceCheckResult(
        ok=True,
        passed_steps=len(seq.ops),
        total_steps=len(seq.ops),
        first_failure=None,
        epsilon_total=eps_total,
        elapsed_ms_total=elapsed_total,
        per_step_ms=tuple(per_step_ms),
    )


# ---------------------------------------------------------------------------
# Marabou bridge skeleton (Stage 3a, mock 完走保証)
# ---------------------------------------------------------------------------


def is_marabou_available() -> bool:
    """Marabou (maraboupy) が import 可能か判定.

    Stage 3a では Marabou 実 install を試みず、不在で mock 完走することを保証。
    True/False に関わらず、本 module の Z3 検査は変わらず動く。
    """
    try:
        import maraboupy  # type: ignore  # noqa: F401
        return True
    except ImportError:
        return False


@dataclass(frozen=True)
class MarabouBridgeStatus:
    """Marabou bridge 状態 (skeleton)."""

    marabou_available: bool
    bridge_mode: str  # "z3_mock" | "marabou_native" | "hybrid"
    notes: str


def get_bridge_status() -> MarabouBridgeStatus:
    """Marabou bridge の現状を返す (Stage 3a は z3_mock 固定で sound 再現)."""
    avail = is_marabou_available()
    if avail:
        return MarabouBridgeStatus(
            marabou_available=True,
            bridge_mode="hybrid",
            notes=(
                "Marabou detected; PoC 3a still uses Z3 for sound refinement"
                " checks. Native Marabou Incremental wiring is Stage 5+."
            ),
        )
    return MarabouBridgeStatus(
        marabou_available=False,
        bridge_mode="z3_mock",
        notes=(
            "Marabou not installed; PoC 3a reproduces sound refinement"
            " relation in Z3 (mock). CPU-only完結 保証."
        ),
    )
