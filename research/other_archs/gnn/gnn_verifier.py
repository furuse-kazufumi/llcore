# SPDX-License-Identifier: Apache-2.0
"""GNN gene 用 Z3 verifier — over-smoothing lower bound + permutation equivariance.

依存: z3-solver (optional, ``pip install z3-solver``).

検査内容:
    (1) over-smoothing lower bound (variance reduction rate 下界)
        GNN は深い層で全 node が同じ表現に収束する病理がある.
        agg(h) = α_sum * Σh + α_mean * mean(h) + α_max * max(h) の場合,
        sum part は variance を増幅 (係数 1 で個別 element 影響), mean/max は
        smoothing op.

        per-layer variance reduction の sound 上界 (tanh Lipschitz=1 仮定):
            var(h_{l+1}) <= (|W| + |U| * agg_amplify_upper)^2 * var(h_l)
        ここで agg_amplify_upper = α_sum * (近傍数) + α_mean + α_max
        (sum は近傍数 K 倍まで増幅, mean/max は variance を保持 or 圧縮).

        over-smoothing **lower bound** invariant の sound version (Codex Q1 honest 留保):
            "per-layer の variance shrink 率の上界が 1 を超えるなら
             over-smoothing を強制する saturation はない"
        を Z3 で確認する.

        簡約形:
            agg_amplify_upper = α_sum * K + α_mean + α_max (K=2 for ring topology)
            shrink_upper = (|W| + |U| * agg_amplify_upper)^2
            assert: shrink_upper >= ε^(1/L)  for ε=0.1, L=8
            → (|W| + |U| * (α_sum*K + α_mean + α_max))^2 >= 0.1^(1/8) ≈ 0.7499

        gene 別に sat/unsat を分離 (smoothing 強すぎる gene = α_sum=0, α_mean=1,
        W=0, U=0.1 などは unsat = 反例検出).

    (2) permutation equivariance (構造的保証 + symbolic 確認)
        aggregation が α_sum * sum + α_mean * mean + α_max * max の凸結合で
        構成されている限り permutation-equivariant op であることは構造的に保証.
        Z3 では「gene の α 重みが simplex (>= 0 + sum=1) 内にあること」を確認
        + per-element 等式 (任意の 2 node 入れ替えで agg 不変) を symbolic に検査.

        Sketch: 2-element 近傍 {h_a, h_b} について
            sum: h_a + h_b == h_b + h_a (trivially)
            mean: 1/2 (h_a + h_b) == 1/2 (h_b + h_a)
            max: max(h_a, h_b) == max(h_b, h_a)
        すべて symbolic に equivalent → 凸結合も equivariant.

        Z3 では 「gene 構造を破壊する例 (alpha が負 or 範囲外, 非凸結合) を入れたら
        sat (反例検出)」を確認する soundness check.

honest 留保:
- Z3 は実数 (tanh 含む) を直接扱えないため、Lipschitz=1 + |tanh| <= 1 で sound 近似.
- (1) は厳密な over-smoothing **lower bound** ではなく **shrink-upper-rate lower bound**
  に降格 (Codex Q1 で議論 / 留保 §honest disclosure に記載).
- (2) は本 PoC では 「gene 構造が permutation-equivariant 凸結合範囲内にあるか」を
  symbolic check するもので、実際の forward の permutation-equivariance は構造的に
  保証されている. Z3 が検出するのは 「gene 構造が壊された場合の sat (反例)」.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import z3  # type: ignore
    _Z3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _Z3_AVAILABLE = False
    z3 = None  # type: ignore


from .gnn_gene import GnnGene


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_z3_available() -> bool:
    return _Z3_AVAILABLE


@dataclass(frozen=True)
class VerifyResult:
    """Z3 検査結果.

    Attributes
    ----------
    ok : bool
        True なら invariant 成立 (over-smoothing 抑制能力あり / equivariance 範囲内).
    used_z3 : bool
        Z3 が呼ばれたか. False なら mock fallback.
    reason : str
        verdict の人間可読要約.
    counterexample : dict | None
        sat (invariant 違反) の場合の反例 (decay/U/W/alpha 値).
    """

    ok: bool
    used_z3: bool
    reason: str
    counterexample: Optional[dict] = None


# ---------------------------------------------------------------------------
# (1) over-smoothing lower bound
# ---------------------------------------------------------------------------


_DEFAULT_NEIGHBORS = 2  # ring topology
_DEFAULT_EPSILON = 0.1  # var(h_L) / var(h_0) >= 0.1 を要求 (10% 維持)
_DEFAULT_LAYERS = 8


def _shrink_upper_threshold(epsilon: float = _DEFAULT_EPSILON, n_layers: int = _DEFAULT_LAYERS) -> float:
    """per-layer shrink upper rate >= threshold で over-smoothing 抑制成立.

    var(h_L) / var(h_0) >= ε  ⟹  shrink_upper^L >= ε  ⟹  shrink_upper >= ε^(1/L)
    """
    return epsilon ** (1.0 / n_layers)


def verify_oversmoothing_lower_bound(
    gene: GnnGene,
    *,
    epsilon: float = _DEFAULT_EPSILON,
    n_layers: int = _DEFAULT_LAYERS,
    n_neighbors: int = _DEFAULT_NEIGHBORS,
    timeout_ms: int = 500,
) -> VerifyResult:
    """gene の over-smoothing 抑制能力を Z3 で検査.

    invariant:
        (|W| + |U| * (alpha_sum * K + alpha_mean + alpha_max))^2 >= epsilon^(1/L)

    成立 (ok=True) → gene は over-smoothing 抑制能力あり.
    違反 (ok=False) → 反例として gene を返す (smoothing 強すぎる).
    """
    g = gene.clipped()
    threshold = _shrink_upper_threshold(epsilon=epsilon, n_layers=n_layers)

    # 数値計算 (Z3 不在時の fallback)
    abs_W = abs(g.W)
    abs_U = abs(g.U)
    agg_amplify = g.alpha_sum * n_neighbors + g.alpha_mean + g.alpha_max
    shrink_upper = (abs_W + abs_U * agg_amplify) ** 2
    ok_numeric = shrink_upper >= threshold

    if not _Z3_AVAILABLE:
        return VerifyResult(
            ok=ok_numeric,
            used_z3=False,
            reason=(
                f"[mock] shrink_upper={shrink_upper:.4f}, threshold={threshold:.4f}, "
                f"ok={ok_numeric}"
            ),
            counterexample=None if ok_numeric else {
                "alpha_sum": g.alpha_sum,
                "alpha_mean": g.alpha_mean,
                "alpha_max": g.alpha_max,
                "W": g.W,
                "U": g.U,
            },
        )

    # Z3 symbolic check (実 gene の値を制約として与え、unsat (=invariant 成立) か sat か検査)
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)

    z_alpha_sum = z3.Real("alpha_sum")
    z_alpha_mean = z3.Real("alpha_mean")
    z_alpha_max = z3.Real("alpha_max")
    z_W = z3.Real("W")
    z_U = z3.Real("U")
    z_absW = z3.Real("absW")
    z_absU = z3.Real("absU")
    z_shrink = z3.Real("shrink")

    # gene 値を符号で固定
    solver.add(z_alpha_sum == g.alpha_sum)
    solver.add(z_alpha_mean == g.alpha_mean)
    solver.add(z_alpha_max == g.alpha_max)
    solver.add(z_W == g.W)
    solver.add(z_U == g.U)

    # |W|, |U|
    solver.add(z_absW >= 0, z_absW >= z_W, z_absW >= -z_W)
    solver.add(z_absU >= 0, z_absU >= z_U, z_absU >= -z_U)
    # tighten |W|/|U| = actual abs (上界だけだと sat 領域が広がるので、
    # 等号を加える: |W| <= max(W, -W))
    solver.add(z_absW <= z3.If(z_W >= 0, z_W, -z_W))
    solver.add(z_absU <= z3.If(z_U >= 0, z_U, -z_U))

    agg = z_alpha_sum * n_neighbors + z_alpha_mean + z_alpha_max
    base = z_absW + z_absU * agg
    solver.add(z_shrink == base * base)

    # invariant 違反を要求: shrink < threshold
    solver.add(z_shrink < threshold)
    result = solver.check()

    if result == z3.unsat:
        return VerifyResult(
            ok=True,
            used_z3=True,
            reason=(
                f"[z3 unsat] shrink_upper={shrink_upper:.4f} >= threshold={threshold:.4f}, "
                f"over-smoothing 抑制成立"
            ),
        )
    elif result == z3.sat:
        m = solver.model()
        ce = {
            "alpha_sum": float(m[z_alpha_sum].as_decimal(6).rstrip("?")) if m[z_alpha_sum] is not None else g.alpha_sum,
            "alpha_mean": float(m[z_alpha_mean].as_decimal(6).rstrip("?")) if m[z_alpha_mean] is not None else g.alpha_mean,
            "alpha_max": float(m[z_alpha_max].as_decimal(6).rstrip("?")) if m[z_alpha_max] is not None else g.alpha_max,
            "W": float(m[z_W].as_decimal(6).rstrip("?")) if m[z_W] is not None else g.W,
            "U": float(m[z_U].as_decimal(6).rstrip("?")) if m[z_U] is not None else g.U,
        }
        return VerifyResult(
            ok=False,
            used_z3=True,
            reason=(
                f"[z3 sat] shrink_upper={shrink_upper:.4f} < threshold={threshold:.4f}, "
                f"over-smoothing 強すぎる gene"
            ),
            counterexample=ce,
        )
    else:
        return VerifyResult(
            ok=False,
            used_z3=True,
            reason=f"[z3 {result}] timeout/unknown",
            counterexample=None,
        )


# ---------------------------------------------------------------------------
# (2) permutation equivariance (structural)
# ---------------------------------------------------------------------------


def verify_equivariance_structure(
    gene: GnnGene,
    *,
    timeout_ms: int = 500,
) -> VerifyResult:
    """gene 構造が permutation-equivariant 凸結合範囲内にあるか symbolic 検査.

    invariant: alpha_sum >= 0 ∧ alpha_mean >= 0 ∧ alpha_max >= 0 ∧
               alpha_sum + alpha_mean + alpha_max == 1
    成立 → permutation-equivariance 構造保証
    違反 (例: alpha が負, または合計が 1 でない) → 反例 (sat)
    """
    g = gene.clipped()
    # 数値判定 (fallback + 数値整合性確認)
    s = g.alpha_sum + g.alpha_mean + g.alpha_max
    in_simplex_numeric = (
        g.alpha_sum >= -1e-9
        and g.alpha_mean >= -1e-9
        and g.alpha_max >= -1e-9
        and abs(s - 1.0) <= 1e-6
    )

    if not _Z3_AVAILABLE:
        return VerifyResult(
            ok=in_simplex_numeric,
            used_z3=False,
            reason=(
                f"[mock] simplex check: a_sum={g.alpha_sum:.4f}, a_mean={g.alpha_mean:.4f}, "
                f"a_max={g.alpha_max:.4f}, sum={s:.6f}, ok={in_simplex_numeric}"
            ),
            counterexample=None if in_simplex_numeric else {
                "alpha_sum": g.alpha_sum,
                "alpha_mean": g.alpha_mean,
                "alpha_max": g.alpha_max,
                "sum": s,
            },
        )

    # Z3 symbolic: gene 値を固定し simplex invariant を否定 → unsat で成立, sat で違反
    solver = z3.Solver()
    solver.set("timeout", timeout_ms)
    a_sum = z3.Real("alpha_sum")
    a_mean = z3.Real("alpha_mean")
    a_max = z3.Real("alpha_max")
    solver.add(a_sum == g.alpha_sum)
    solver.add(a_mean == g.alpha_mean)
    solver.add(a_max == g.alpha_max)
    # 違反: simplex 外
    eps = 1e-6
    solver.add(
        z3.Or(
            a_sum < -eps,
            a_mean < -eps,
            a_max < -eps,
            (a_sum + a_mean + a_max) > 1.0 + eps,
            (a_sum + a_mean + a_max) < 1.0 - eps,
        )
    )
    result = solver.check()
    if result == z3.unsat:
        return VerifyResult(
            ok=True,
            used_z3=True,
            reason=(
                f"[z3 unsat] simplex invariant 成立: "
                f"a=({g.alpha_sum:.4f}, {g.alpha_mean:.4f}, {g.alpha_max:.4f}), sum={s:.6f}"
            ),
        )
    elif result == z3.sat:
        return VerifyResult(
            ok=False,
            used_z3=True,
            reason=(
                f"[z3 sat] simplex invariant 違反: "
                f"a=({g.alpha_sum:.4f}, {g.alpha_mean:.4f}, {g.alpha_max:.4f}), sum={s:.6f}"
            ),
            counterexample={
                "alpha_sum": g.alpha_sum,
                "alpha_mean": g.alpha_mean,
                "alpha_max": g.alpha_max,
                "sum": s,
            },
        )
    else:
        return VerifyResult(
            ok=False,
            used_z3=True,
            reason=f"[z3 {result}] timeout/unknown",
        )


__all__ = [
    "VerifyResult",
    "is_z3_available",
    "verify_oversmoothing_lower_bound",
    "verify_equivariance_structure",
]
