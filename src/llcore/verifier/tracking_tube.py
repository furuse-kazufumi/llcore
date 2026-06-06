# SPDX-License-Identifier: Apache-2.0
"""Tracking-tube reporter (T1 Phase 1 (b)) — additive, read-only.

設計 doc ``docs/design/target_trajectory_verifier_2026_06_06.md`` の **案 B「外乱 tube
ゲート」** を src 品質で移植する純 ADDITIVE レポータ。既存 contraction ゲート
(``cert_inf`` / :class:`InfNormBackend`) の出力

    L = sup‖J_s‖_∞ < 1          (状態方向 Lipschitz; 既存 box 端点量)

を *そのまま再利用* し、入力ゲイン

    G = sup‖J_x‖_∞ = max_i (1−decay_i)·Σ_j|V_ij|

と外乱上界 ``w̄`` から **閉形式** の tracking tube 半径

    r = G·w̄ / (1 − L)            (L<1 のときのみ有限; それ以外 +∞)

を導出して報告する。**新規 Z3 証明は不要** (contraction が証明済みなら tracking tube は
系として導出される — Banach + Lipschitz 合成の標準帰結)。

設計 doc の初版スコープに従い、tube は **cert_inf (∞-norm contraction) PASS の gene
に限定** する (``cert_two`` / ``cert_sdp`` のみ通る回転的 contraction は P-norm 版が要り、
初版では扱わない)。本モジュールは ``certifies()`` 等の既存 public API を 1 文字も変えない
(read-only / semver 互換)。

数式・係数は PoC ``research/target_trajectory_poc/poc_target_trajectory.py`` の
``state_lipschitz_inf`` / ``input_gain_inf`` をそのまま移植し、その結果 JSON
(case A/B/C/D) を golden 値としてテストで一致確認する。

honest 留保 (設計 doc §6):
- 保証するのは「*与えられた* 参照軌道への追従誤差が tube に閉じる」ことのみ。参照軌道
  自体の妥当性 (それが「良い」軌道か) と feasibility は検査外 (タスク fitness の責任)。
- L, G の Lipschitz 上界は achievable-t box ``[t_min,1]^n`` 上の sup を端点列挙 (sound;
  Z3 / vertex 列挙と同じ box)。
- ``cert_inf`` 非 PASS (L≥1) の gene は tube=+∞ (=保証なし) を返す。状態自体は tanh で
  有界に留まるが、追従は保証されない (PoC case D が negative control)。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .backends import (
    _coupled_arrays,
    _infnorm_sup,
    _t_min,
)


@dataclass(frozen=True)
class TrackingTubeResult:
    """外乱 tube レポータの結果 (read-only; 案 B).

    Attributes
    ----------
    contraction_ok : bool
        既存 ∞-norm contraction ゲート (:class:`InfNormBackend` / ``cert_inf``) の verdict。
        True なら L<1 が certified され tube が有限になる。
    L_state : float
        状態方向 Lipschitz 上界 ``L = sup‖J_s‖_∞`` (achievable-t box 端点列挙の閉形式;
        既存 contraction ゲートが ρ<1 の証拠に使う量そのもの)。
    G_input : float
        入力方向ゲイン ``G = sup‖J_x‖_∞ = max_i (1−decay_i)·Σ_j|V_ij|`` (t_i=1 で sup)。
    w_bar : float
        外乱上界 ``‖d‖_∞ ≤ w̄`` (設計者が与える。tube はこれに線形にスケール)。
    tube_radius : float
        定常 tube 半径 ``r = G·w̄/(1−L)``。``contraction_ok`` が False (L≥1) のとき
        ``float("inf")`` = 保証なし。
    admits : bool
        ゲートとして使う場合の判定 = ``contraction_ok ∧ tube_radius ≤ r_max`` (fail-closed)。
        ``r_max`` が None のときは ``contraction_ok`` のみ (tube 有限なら admit)。
    r_max : float | None
        tube 半径の許容上限 (ゲート用)。None なら上限なし。
    """

    contraction_ok: bool
    L_state: float
    G_input: float
    w_bar: float
    tube_radius: float
    admits: bool
    r_max: float | None = None


def input_gain_inf(gene: Any, *, max_input_abs: float = 1.0) -> float:
    """入力方向ゲイン ``G = sup‖J_x‖_∞`` を閉形式で計算する (read-only).

    入力方向ヤコビ ``∂s'/∂x = diag((1−decay)⊙t) @ V`` の各行 i の abs-sum は
    ``(1−decay_i)·t_i·Σ_j|V_ij|``。各 ``t_i ∈ (0,1]`` に単調増加なので sup は ``t_i=1``
    で達成され、``G = max_i (1−decay_i)·Σ_j|V_ij|``。PoC ``input_gain_inf`` と一致。

    Parameters
    ----------
    gene : Any
        duck-typed gene。scalar ``StateUpdateGene`` (decay/mix/gate_str float) なら
        n=1 coupled に持ち上げ、coupled gene (decay shape (n,), W (n,n), 任意 V) なら
        そのまま読む (:func:`llcore.verifier.backends._coupled_arrays` 経由で clip)。
    max_input_abs : float
        入力上界 |x| ≤ max_input_abs。G の閉形式には直接効かないが (sup は t=1 端点)、
        API 互換と将来の achievable-t 精緻化用に保持。

    Returns
    -------
    float
        入力ゲイン上界 G (≥ 0)。
    """
    decay, _W, V = _coupled_arrays_any(gene)
    # 行 i の abs-sum = (1-decay_i) * Σ_j|V_ij| が t_i=1 で sup。
    rows = (1.0 - decay) * np.abs(V).sum(axis=1)
    return float(np.max(rows))


def state_lipschitz_inf(gene: Any, *, max_input_abs: float = 1.0) -> float:
    """状態方向 Lipschitz 上界 ``L = sup‖J_s‖_∞`` を閉形式で計算する (read-only).

    既存 :class:`InfNormBackend` が contraction (ρ<1) の証拠に使う sup-box ∞-norm
    (``_infnorm_sup`` over achievable-t box) そのものを再利用する。PoC
    ``state_lipschitz_inf`` と一致。

    Parameters
    ----------
    gene : Any
        duck-typed gene (scalar or coupled; :func:`input_gain_inf` と同じ規約)。
    max_input_abs : float
        入力上界 |x| ≤ max_input_abs (achievable-t box の t_min を決める)。

    Returns
    -------
    float
        状態 Lipschitz 上界 L (≥ 0)。L<1 なら state-direction contraction。
    """
    decay, W, V = _coupled_arrays_any(gene)
    return _infnorm_sup(decay, W, _t_min(decay, W, V, max_input_abs))


def tracking_tube(
    gene: Any,
    *,
    w_bar: float,
    max_input_abs: float = 1.0,
    r_max: float | None = None,
) -> TrackingTubeResult:
    """contraction-certified gene の追従 tube 半径を閉形式で報告する (read-only / 案 B).

    既存 ∞-norm contraction 基準 (``cert_inf`` / :class:`InfNormBackend` と同じ
    ``L = sup‖J_s‖_∞ < 1``) で gene を検査し、PASS なら L と ``G = sup‖J_x‖_∞`` から
    tube 半径 ``r = G·w̄/(1−L)`` を返す。REJECT (L≥1) なら ``tube_radius = +∞``
    (= 追従保証なし)。

    soundness: tube 不等式 ``limsup_t ‖s_act−s_ref‖_∞ ≤ G·w̄/(1−L)`` は **定理**
    (Banach + Lipschitz 合成, 設計 doc §2)。ただし参照軌道 feasibility と「軌道の妥当性」
    は検査外 (タスク側の責任; honest 留保)。

    Parameters
    ----------
    gene : Any
        検査対象 gene (scalar ``StateUpdateGene`` or coupled gene; 内部で clip)。
    w_bar : float
        外乱上界 ``‖d‖_∞ ≤ w̄`` (≥ 0)。tube はこれに線形。
    max_input_abs : float
        入力上界 |x| ≤ max_input_abs (achievable-t box を決める)。
    r_max : float | None
        ゲートとして使う場合の tube 半径許容上限。None なら上限なし
        (``admits = contraction_ok``)。

    Returns
    -------
    TrackingTubeResult
        L / G / tube_radius / contraction_ok / admits を含む read-only レポート。

    Raises
    ------
    ValueError
        ``w_bar < 0`` (外乱上界は非負; fail-loud)。
    """
    if w_bar < 0.0:
        raise ValueError(f"w_bar must be non-negative, got {w_bar!r}")

    L = state_lipschitz_inf(gene, max_input_abs=max_input_abs)
    G = input_gain_inf(gene, max_input_abs=max_input_abs)

    # contraction 判定 = ∞-norm contraction (cert_inf 相当): L = sup‖J_s‖_∞ < 1。
    # これは :class:`InfNormBackend` / coupled ``cert_inf`` の判定基準そのもの。L を
    # 正規化済みの (scalar も coupled も統一した) box 端点量から計算しているため、
    # scalar / coupled 双方で整合する (InfNormBackend に raw gene を渡すと scalar の
    # W 不在で fail-closed False になるため、ここでは L<1 を直接基準にする)。
    contraction_ok = bool(L < 1.0)

    # tube 半径: L<1 でのみ有限 (幾何級数収束)。それ以外は +∞ (= 追従保証なし)。
    if contraction_ok:
        tube_radius = float(G * w_bar / (1.0 - L))
    else:
        tube_radius = float("inf")

    tube_finite = bool(np.isfinite(tube_radius))
    if r_max is None:
        admits = bool(contraction_ok and tube_finite)
    else:
        admits = bool(contraction_ok and tube_finite and tube_radius <= r_max)

    return TrackingTubeResult(
        contraction_ok=contraction_ok,
        L_state=L,
        G_input=G,
        w_bar=float(w_bar),
        tube_radius=tube_radius,
        admits=admits,
        r_max=r_max,
    )


# --------------------------------------------------------------------------- #
# gene 正規化 (scalar StateUpdateGene → n=1 coupled 持ち上げ)。
# --------------------------------------------------------------------------- #


def _coupled_arrays_any(gene: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """scalar / coupled どちらの gene でも (decay (n,), W (n,n), V (n,n)) を返す.

    - coupled gene (``W`` 属性を持つ) → :func:`backends._coupled_arrays` をそのまま使用。
    - scalar ``StateUpdateGene`` (``W`` 無し, ``gate_str`` 有り) → n=1 coupled に持ち上げ:
      ``decay=[decay]``, ``W=[[gate_str]]``, ``V=[[mix]]``。これは scalar 更新式
      ``s' = decay·s + (1−decay)·tanh(gate_str·s + mix·x)`` の coupled 形 (V≠I) と一致し、
      ``L = |decay + (1−decay)·t·gate_str|`` (Stage 1b の J(t)) ・
      ``G = (1−decay)·|mix|`` を正しく与える。

    Notes
    -----
    scalar gene は入力ゲインが ``mix`` (V=mix) なので、coupled ヘルパに V=I を渡すと
    G が誤る。そのため scalar は明示的に V=[[mix]] を組み立てる。
    """
    if hasattr(gene, "W"):
        # coupled gene (CoupledNDGene 等)。backends の clip 規約に委譲。
        return _coupled_arrays(gene)
    # scalar StateUpdateGene: n=1 coupled に持ち上げる (clip 範囲は genes.clipped() と同じ)。
    g = gene.clipped() if hasattr(gene, "clipped") else gene
    decay = np.clip(np.asarray([float(g.decay)], dtype=np.float64), 0.0, 1.0)
    W = np.clip(np.asarray([[float(g.gate_str)]], dtype=np.float64), -2.0, 2.0)
    V = np.asarray([[float(g.mix)]], dtype=np.float64)
    return decay, W, V
