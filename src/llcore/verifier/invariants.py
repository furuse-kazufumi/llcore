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

import numpy as np

try:
    import z3  # type: ignore[import-untyped]
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

    # tanh argument は mix*x + gate_str*s。厳密には |tanh(z)| <= min(|z|, 1) の sound 上界が
    # 使えるが、ここでは簡単のため緩い側の tanh ∈ [-1, 1] で表現する(より保守的=安全側の近似)。
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


# ===========================================================================
# Stage 1b — 状態方向 Lipschitz contraction invariant (純 ADDITIVE)
# ===========================================================================
#
# 既存 verify_state_norm_invariant / verify_gene_safe は |s|<=1 の有界性のみを
# 弁別 (clip 範囲全 gene で無条件 admit する受動的不変量) を扱う。本節は質的に
# 強い不変量 = **状態方向 contraction (Lipschitz 定数 L<1)** の Z3 証明を、
# 既存関数を 1 文字も変えず新規関数として追加する (semver 互換)。
#
# 命題 (PoC 1b, falsifiable):
#     RWKV 更新写像
#         s' = decay·s + (1−decay)·tanh(mix·x + gate_str·s)
#     は各座標独立 (対角写像) で、座標ヤコビは
#         ∂s'/∂s = decay + (1−decay)·gate_str·t,   t = sech²(pre) = 1−tanh²(pre) ∈ (0,1]
#     状態方向 Lipschitz 定数を L = sup_{|s|≤1,|x|≤1} |∂s'/∂s| と定義する。
#     clip 済み gene g について、Z3 が
#         「∃ t∈[0,1]. |decay+(1−decay)·gate_str·t| ≥ 1」
#     を unsat と判定したとき、その gene は L<1 (state-direction contraction certified)。
#     これは sup-norm での global contraction を含意し、Banach により一意固定点と
#     |s|≤1 有界 (state_norm 整合) を保証する。
#
# Z3 encoding (sound free-variable abstraction):
#     変数: t = z3.Real("t") のみ (per-gene online gate)。
#     定数化: d = z3.RealVal(decay), g = z3.RealVal(gate_str)。
#     制約: (1) 0≤t≤1 — sech²(pre) の真域 (0,1] の over-approx (端点込みで真に包含)。
#           (2) J = d + (1−d)·g·t   (t について一次=純線形なので高速)。
#     反例探索: solver.add(z3.Or(J ≥ 1, J ≤ −1))。
#     判定: unsat → sup|J|<1 = L<1 certified。sat → L≥1 になり得る点が存在 = 保守的 reject。
#           unknown/timeout → fail-closed で reject。
#
# soundness 論拠:
#     真の t = sech²(pre) は (0,1] が値域で、|s|≤1,|x|≤1 の achievable 集合は (0,1] の
#     さらに狭い閉部分集合 (正の下限あり)。自由 t∈[0,1] はこれを真に包含する over-approx。
#     J は t について一次 (monotone) なので、t∈[0,1] 上の |J| の上限は端点 t∈{0,1} で
#     達成され、真の achievable 集合上の sup を必ず上から押さえる。よって自由 t 領域で
#     |J|≥1 を見つけられない (unsat) ならば、より狭い achievable 領域でも当然見つからない
#     = 真に L<1。これが unsat⟹certified の健全性。逆に sat は achievable でない t を
#     使った反例かもしれず保守的 (false reject 可) で良い (fail-closed 規律と整合)。
#     ※ x と mix は J に t を介してのみ入るため、t を自由化した時点で式から消え、
#       ソルバは t 1 変数の線形可行性問題に縮約される。


def verify_retention_floor(
    *,
    log_floor: float,
    steps: int = 1,
    parameterization: str = "bounded_sigmoid",
    timeout_ms: int = 2000,
) -> InvariantResult:
    """**保持床 (retention floor)** — 忘却がある速度より速くならないことを Z3 で検証.

    検査命題 (steps=T, 床 = ``log_floor``):
        ``∀ raw params. Σ_{i=1..T} g_i > T * log_floor  かつ  ∀i. g_i < 0``

    保持率は ``α_i = exp(g_i)`` で累積保持は ``exp(Σ g_i)`` なので、この命題は
    「T ステップ経っても保持率が ``exp(T * log_floor)`` を下回らない」と同値
    (``exp`` は狭義単調増加。**この単調性の一段だけは Z3 の外の数学的議論**で、
    以降の線形部分を Z3 が厳密に証明する — 下記 honest 留保を参照)。

    Parameters
    ----------
    log_floor : float
        log 空間の床 (負値)。K3 は ``g_min = -5``。
    steps : int
        累積を見るステップ数 T (>= 1)。
    parameterization : str
        ``"bounded_sigmoid"`` = ``g = log_floor * sigmoid(w)`` (K3 式)。
        ``"neg_exp"`` = ``g = -exp(w)`` (llcore の従来式、下に非有界)。
    timeout_ms : int
        Z3 timeout (ms)。

    Returns
    -------
    InvariantResult
        ok=True (unsat) なら床が破れない証明。ok=False (sat) なら反例あり
        = その parameterization では床を保証できない。

    Raises
    ------
    ValueError
        ``log_floor >= 0`` / ``steps < 1`` / 未知の ``parameterization``。

    Notes
    -----
    Z3 encoding (sound abstraction):
    - ``sigmoid(w)`` は Z3 で直接表現できないため、**値域そのもの** ``σ ∈ (0, 1)``
      を持つ自由変数に置き換える。sigmoid が実際に取り得る値の集合は (0,1) と一致
      するので、この抽象は緩めも狭めもしない (exact range abstraction)。
    - ``exp(w)`` も同様に ``e > 0`` の自由変数に置き換える。こちらも値域と一致。
    - よって Z3 が解くのは ``g_i`` に関する**線形実数論**の問題であり厳密。

    honest 留保:
    - ``α = exp(g)`` への変換は Z3 の外側 (exp の単調性) に依存する。log 空間で
      床が証明できても、実装が ``exp`` を別の近似で計算していれば話は別。
    - **実数上の証明であり float の飽和を扱っていない**。Z3 は ``σ ∈ (0,1)`` の
      開区間を仮定するが、float32 実装では ``sigmoid(1e4) == 1.0`` と飽和して
      ``g == log_floor`` ちょうどに到達する (``tests/unit/test_poc_k3_bounded_decay.py``
      の G1c で固定)。**床 ``α >= exp(log_floor)`` 自体は等号成立を含めて保たれる**が、
      「開区間である」という Z3 の結論を float 実装の記述として引用してはならない。
    - ``neg_exp`` が sat になるのは「床を保証できない」ことの証明であって、
      「学習後の実パラメータが実際に床を割る」ことの証明ではない。
    """
    if log_floor >= 0.0:
        raise ValueError(f"log_floor must be < 0, got {log_floor}")
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    if parameterization not in ("bounded_sigmoid", "neg_exp"):
        raise ValueError(
            f"unknown parameterization {parameterization!r} "
            "(expected 'bounded_sigmoid' or 'neg_exp')"
        )

    if not _HAS_Z3:
        return InvariantResult(
            ok=parameterization == "bounded_sigmoid",
            used_z3=False,
            reason=(
                "z3 not installed; bounded_sigmoid の床は sigmoid の値域 (0,1) から "
                "解析的に従う (g = log_floor*σ ∈ (log_floor, 0))"
            ),
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    log_decays = []
    for i in range(steps):
        if parameterization == "bounded_sigmoid":
            # sigmoid の値域を厳密に写す自由変数 (over/under-approx なし)
            sigma = z3.Real(f"sigmoid_{i}")
            solver.add(sigma > 0, sigma < 1)
            log_decays.append(log_floor * sigma)
        else:
            # exp の値域 (0, inf) を厳密に写す自由変数
            e = z3.Real(f"exp_{i}")
            solver.add(e > 0)
            log_decays.append(-e)

    cumulative = z3.Sum(log_decays) if steps > 1 else log_decays[0]

    # 反例探索: 「床を割る」または「減衰していない (g >= 0)」
    violations = [cumulative <= steps * log_floor]
    violations.extend(g >= 0 for g in log_decays)
    solver.add(z3.Or(*violations))

    status = solver.check()
    if status == z3.unsat:
        return InvariantResult(
            ok=True,
            used_z3=True,
            reason=(
                f"unsat: {parameterization} は {steps} step 累積で "
                f"log-retention > {steps * log_floor} を破れない "
                f"(= 保持率 > exp({steps * log_floor}))"
            ),
            solver_status="unsat",
        )
    if status == z3.sat:
        model = solver.model()
        counterexample = {
            str(d): float(model[d].as_fraction()) for d in model.decls()
        }
        return InvariantResult(
            ok=False,
            used_z3=True,
            reason=(
                f"sat: {parameterization} では {steps} step 累積の保持床 "
                f"{steps * log_floor} を割る割り当てが存在する (床を保証できない)"
            ),
            counterexample=counterexample,
            solver_status="sat",
        )
    return InvariantResult(
        ok=False,
        used_z3=True,
        reason=f"z3 returned unknown (timeout {timeout_ms}ms) — fail-closed で reject",
        solver_status="unknown",
    )


def verify_activation_bound(
    *,
    beta_gate: float,
    beta_up: float,
    claimed_bound: float | None = None,
    timeout_ms: int = 2000,
) -> InvariantResult:
    """**SiTU-GLU の出力有界性** を Z3 で検証.

    検査命題::

        ∀ gate_pre, up_pre.
            |[β_g·tanh(g/β_g)·Sigmoid(g)] · [β_u·tanh(u/β_u)]| <= claimed_bound

    ``claimed_bound`` 既定は ``β_gate * β_up`` (解析上界)。

    Parameters
    ----------
    beta_gate, beta_up : float
        gate 枝 / up 枝の softcap 係数 (正)。
    claimed_bound : float | None
        主張する上界。None なら ``beta_gate * beta_up``。積より小さい値を渡すと
        Z3 が反例を返す (soundness のクロスチェックに使える)。
    timeout_ms : int
        Z3 timeout (ms)。

    Returns
    -------
    InvariantResult
        ok=True (unsat) なら上界が破れない証明。

    Raises
    ------
    ValueError
        ``beta_gate`` / ``beta_up`` が正でない、または ``claimed_bound <= 0``。

    Notes
    -----
    Z3 encoding (sound abstraction):
    - ``tanh(·)`` を値域と一致する自由変数 ``t ∈ (-1, 1)`` に、``Sigmoid(·)`` を
      ``s ∈ (0, 1)`` に置き換える。どちらも値域と厳密に一致するので緩めていない。
    - ただし ``tanh(g/β_g)`` と ``Sigmoid(g)`` は**同じ** ``g`` の関数なので独立ではない。
      独立変数として扱うのは **over-approximation** (実際に到達できない組合せも許す)
      であり、unsat ⟹ 真に有界 という含意は保たれる (逆に sat は保守的で偽 reject
      になりうる = fail-closed 側)。
    - 結果として Z3 が解くのは 3 変数の非線形実数論。

    honest 留保:
    - 実数上の証明であり float の丸め・飽和は扱っていない (:func:`verify_retention_floor`
      と同じ留保)。
    """
    if beta_gate <= 0.0:
        raise ValueError(f"beta_gate must be > 0, got {beta_gate}")
    if beta_up <= 0.0:
        raise ValueError(f"beta_up must be > 0, got {beta_up}")
    bound = beta_gate * beta_up if claimed_bound is None else claimed_bound
    if bound <= 0.0:
        raise ValueError(f"claimed_bound must be > 0, got {bound}")

    if not _HAS_Z3:
        return InvariantResult(
            ok=bound >= beta_gate * beta_up,
            used_z3=False,
            reason=(
                "z3 not installed; |β_g·tanh|<=β_g, Sigmoid<=1, |β_u·tanh|<=β_u より "
                f"解析上界 {beta_gate * beta_up} が従う"
            ),
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    tanh_gate = z3.Real("tanh_gate")
    sigmoid_gate = z3.Real("sigmoid_gate")
    tanh_up = z3.Real("tanh_up")
    solver.add(tanh_gate > -1, tanh_gate < 1)
    solver.add(sigmoid_gate > 0, sigmoid_gate < 1)
    solver.add(tanh_up > -1, tanh_up < 1)

    out = (beta_gate * tanh_gate * sigmoid_gate) * (beta_up * tanh_up)
    solver.add(z3.Or(out > bound, out < -bound))  # 反例探索

    status = solver.check()
    if status == z3.unsat:
        return InvariantResult(
            ok=True,
            used_z3=True,
            reason=f"unsat: |SiTU-GLU| <= {bound} を破る入力は存在しない",
            solver_status="unsat",
        )
    if status == z3.sat:
        model = solver.model()
        return InvariantResult(
            ok=False,
            used_z3=True,
            reason=(
                f"sat: 主張上界 {bound} を超える割り当てが存在する "
                f"(解析上界は {beta_gate * beta_up})"
            ),
            counterexample={str(d): float(model[d].as_fraction()) for d in model.decls()},
            solver_status="sat",
        )
    return InvariantResult(
        ok=False,
        used_z3=True,
        reason=f"z3 returned unknown (timeout {timeout_ms}ms) — fail-closed で reject",
        solver_status="unknown",
    )


@dataclass(frozen=True)
class LipschitzResult:
    """状態方向 Lipschitz contraction 検査結果 (Stage 1b).

    Attributes
    ----------
    contraction : bool | None
        True  = L<1 が certified (Z3 unsat / state-direction contraction)。
        False = reject (Z3 sat または timeout/unknown = fail-closed)。
        None  = z3 不在で未判定 (assumed; used_z3=False)。
        ※ fail-closed 規律: used_z3=False のとき呼び出し側は None を「未検証」
          として扱い、ゲートとしては reject 側に倒すこと。
    L_upper_bound : float | None
        端点 t∈{0,1} で達成される sup_{t∈[0,1]}|J| の解析上界
        max(|decay|, |decay + (1−decay)·gate_str|)。Z3 を呼ばずとも閉形式で
        計算でき、contraction 判定と必ず一致する (健全性のクロスチェックに使う)。
        z3 不在時も計算可能。
    used_z3 : bool
        Z3 が実際に呼ばれたか (False なら fallback)。
    solver_status : str
        "unsat" (certified) | "sat" (reject) | "unknown" (timeout/no-z3)。
    reason : str
        verdict 説明。
    """

    contraction: bool | None
    L_upper_bound: float | None
    used_z3: bool
    solver_status: str
    reason: str


def _lipschitz_upper_bound(decay: float, gate_str: float) -> float:
    """状態方向ヤコビ J(t) = decay + (1−decay)·gate_str·t の |J| 上界 (閉形式).

    J は t∈[0,1] について一次なので sup|J| は端点 t∈{0,1} で達成:
        t=0 → J=decay,  t=1 → J=decay+(1−decay)·gate_str
    よって sup_{t∈[0,1]}|J| = max(|decay|, |decay + (1−decay)·gate_str|).

    Returns
    -------
    float
        L の解析上界 (free-t over-approx 下の真の sup)。
    """
    j0 = decay
    j1 = decay + (1.0 - decay) * gate_str
    return max(abs(j0), abs(j1))


def verify_lipschitz_contraction(
    gene: StateUpdateGene,
    *,
    max_input_abs: float = 1.0,
    timeout_ms: int = 1000,
) -> LipschitzResult:
    """**単一 gene** の状態方向 Lipschitz contraction (L<1) を Z3 で証明.

    DESIGN (PoC 1b) の sound free-variable abstraction を実装する。座標ヤコビ
    ``J(t) = decay + (1−decay)·gate_str·t`` (t = sech²(pre) ∈ (0,1] を free t∈[0,1]
    に over-approx) について、反例 ``|J| ≥ 1`` を Z3 が見つけられない (unsat) なら
    L<1 を certified する。

    既存 :func:`verify_gene_safe` (state_norm online gate) とは別関数であり、
    既存挙動は一切変更しない (純 ADDITIVE / semver 互換)。

    Parameters
    ----------
    gene : StateUpdateGene
        検査対象 gene (内部で ``clipped()`` を通す = 実行モデルと一致)。
    max_input_abs : float
        入力上界 |x| <= max_input_abs。free-t 抽象では J に直接効かないが、
        将来の achievable-t 精緻化 (t_min = sech²(|mix|+|gate_str|)) 用に保持。
    timeout_ms : int
        Z3 timeout (ms)。timeout/unknown は fail-closed で reject。

    Returns
    -------
    LipschitzResult
        contraction=True (unsat) なら L<1 certified。
        contraction=False (sat / timeout) なら reject。
        contraction=None (z3 不在) なら assumed (used_z3=False)。

    Notes
    -----
    L<1 ⟹ (Banach) 一意固定点 + |s|≤1 有界 (state_norm) を構造的に含意する。
    よって contraction certified gene は必ず state_norm も満たす (BG5 で実証)。
    """
    g = gene.clipped()
    l_bound = _lipschitz_upper_bound(g.decay, g.gate_str)

    if not _HAS_Z3:
        # fail-closed: z3 不在では contraction を None(未決)で返す。数式上界 l_bound は参考値として
        # L_upper_bound に載せるが判定には使わない。呼び出し側は used_z3=False を未検証(reject 側)に扱うこと。
        return LipschitzResult(
            contraction=None,
            L_upper_bound=l_bound,
            used_z3=False,
            solver_status="unknown",
            reason=(
                "z3 not installed: contraction undecided (assumed by closed-form "
                f"upper bound L<={l_bound:.4f}; treat used_z3=False as unverified/fail-closed)"
            ),
        )

    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    # 唯一の自由変数: t = sech²(pre) ∈ (0,1] を free t∈[0,1] に over-approx。
    t = z3.Real("t")
    solver.add(t >= 0, t <= 1)

    d = z3.RealVal(g.decay)
    gate = z3.RealVal(g.gate_str)
    # 座標ヤコビ (t について一次 = 純線形)。
    j = d + (1 - d) * gate * t

    # 反例探索: |J| >= 1 となる t が存在するか。
    solver.add(z3.Or(j >= 1, j <= -1))

    result = solver.check()
    if result == z3.unsat:
        return LipschitzResult(
            contraction=True,
            L_upper_bound=l_bound,
            used_z3=True,
            solver_status="unsat",
            reason=(
                f"unsat: sup|J|<1 certified (L<={l_bound:.4f}<1, state-direction "
                f"contraction) for d={g.decay:.3f}, m={g.mix:.3f}, g={g.gate_str:.3f}"
            ),
        )
    if result == z3.sat:
        return LipschitzResult(
            contraction=False,
            L_upper_bound=l_bound,
            used_z3=True,
            solver_status="sat",
            reason=(
                f"sat: |J|>=1 reachable (L>={l_bound:.4f}, no contraction certified; "
                f"conservative reject) for d={g.decay:.3f}, m={g.mix:.3f}, g={g.gate_str:.3f}"
            ),
        )
    # unknown / timeout → fail-closed (reject)。
    return LipschitzResult(
        contraction=False,
        L_upper_bound=l_bound,
        used_z3=True,
        solver_status="unknown",
        reason=f"z3 returned {result} (timeout/unknown) — fail-closed reject",
    )


def empirical_lipschitz(
    gene: StateUpdateGene,
    *,
    n_samples: int = 2000,
    max_input_abs: float = 1.0,
    seed: int = 0,
) -> float:
    """状態方向 Lipschitz 定数 L を有限差分で経験推定する (敵対検証用).

    Z3 の解析上界 :func:`verify_lipschitz_contraction` (L_upper_bound) の健全性を
    実数値でクロスチェックするためのヘルパ。各サンプル点 (s, x) で
    ``∂s'/∂s ≈ (eval_step(s+h) − eval_step(s−h)) / (2h)`` を中央差分で測り、
    その絶対値の最大を返す。

    健全性チェック (BG3/(iv) で使用):
        empirical_L <= Z3 の L_upper_bound  (経験値は解析上界以下でなければならない)

    Parameters
    ----------
    gene : StateUpdateGene
        検査対象 gene (内部で clip)。
    n_samples : int
        ランダムサンプル点数 (s, x) ∈ [−1,1]×[−max_input_abs,max_input_abs]。
    max_input_abs : float
        入力上界 |x| <= max_input_abs。
    seed : int
        乱数シード (再現性)。

    Returns
    -------
    float
        max |∂s'/∂s| の経験推定 (状態方向 Lipschitz 定数 L の下界寄り経験値)。
    """
    g = gene.clipped()
    rng = np.random.default_rng(seed)
    h = 1e-6

    # スカラ座標で評価 (写像は各座標独立 = 対角)。
    s = rng.uniform(-1.0, 1.0, size=n_samples)
    x = rng.uniform(-max_input_abs, max_input_abs, size=n_samples)

    def _step(state: np.ndarray, inp: np.ndarray) -> np.ndarray:
        pre = g.mix * inp + g.gate_str * state
        return g.decay * state + (1.0 - g.decay) * np.tanh(pre)

    s_plus = _step(s + h, x)
    s_minus = _step(s - h, x)
    deriv = (s_plus - s_minus) / (2.0 * h)
    return float(np.max(np.abs(deriv)))
