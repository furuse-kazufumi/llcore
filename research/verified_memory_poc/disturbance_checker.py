# SPDX-License-Identifier: Apache-2.0
"""Phase 2a ステップ2 — trajectory-tube **定理の cross-check** (research, port).

設計 doc ``docs/research/phase2a_verified_memory_evolution_design_2026_06_06.md``
§4.5-2 に対応。``research/target_trajectory_poc/poc_target_trajectory.py:109``
``rollout_with_disturbance`` (CoupledNDGene + ``step``) を **scalar 基板**
(``StateUpdateGene`` + ``run_sequence``) に **再ポイント** (copy でなく port) する。

何をするか:
  contraction-certified な scalar gene について、bounded disturbance ‖d_t‖∞ ≤ w̄ を
  入力に注入した実軌道 ``s_act`` と、外乱なしの参照軌道 ``s_ref`` の追従誤差
  ``e_t = s_act,t − s_ref,t`` の **steady-state 後半の実測 limsup‖e_t‖∞** を 64 seed で測る。
  これを certified tube 半径 ``r = G·w̄/(1−L)`` と比較する (P1 の cross-check)。

honest 留保 (設計 doc §4.3 訂正3 / §5.2):
  - これは **tube 定理の cross-check** であって gate soundness の *一部ではない*。
    gate の soundness は閉形式 (Banach 系) tube 不等式が担保し、本 driver はその数値裏付け。
  - 参照軌道は「同じ gene + 外乱なし入力の系自身の解」なので feasibility 残差 ρ_feas=0
    (eq(5) が完全に sound になる唯一のケース)。参照が「良い」memory かは検査外。
  - L は ``state_lipschitz_inf`` (tracking_tube と同一の閉形式 numpy 比較; 同一 L 定義)
    を使う。``empirical_lipschitz`` (有限差分) は G も e_t も測らないため tube 半径の
    cross-check には使わない (設計 doc §4.3 訂正3)。

port の要点 (CoupledNDGene → scalar):
  - 基板写像 ``step`` → ``eval_step`` (RWKV-style leak integrator)。
  - 軌道生成 ``synthesize_reference`` 相当 = 外乱なし ``run_sequence`` (zero 初期 state)。
  - 外乱注入 ``rollout_with_disturbance`` 相当 = 入力 + uniform d∈[−w̄,w̄] の ``run_sequence``。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.state_update import StateUpdateGene, run_sequence  # noqa: E402
from llcore.verifier.tracking_tube import (  # noqa: E402
    input_gain_inf,
    state_lipschitz_inf,
    tracking_tube,
)


def rollout_with_disturbance(
    gene: StateUpdateGene,
    x_ref: np.ndarray,
    *,
    w_bar: float,
    seed: int,
) -> np.ndarray:
    """外乱 d_t (‖d‖∞ ≤ w̄) を入力に注入した実軌道 s_act を回す (scalar 基板に port).

    s_act[0]=0 (= s_ref[0]; 初期一致), s_act[t+1]=eval_step(s_act[t], x_ref[t]+d[t])。
    d は一様 [−w̄, w̄]^dim (sup-norm worst をなぞる)。元 PoC は CoupledNDGene + step だが、
    ここでは scalar StateUpdateGene + run_sequence に再ポイント。

    Parameters
    ----------
    gene : StateUpdateGene
        scalar 更新カーネル。
    x_ref : np.ndarray
        shape (L, dim) — 参照入力列。
    w_bar : float
        外乱上界 ‖d‖∞ ≤ w̄ (≥0)。
    seed : int
        外乱 RNG seed (再現性)。

    Returns
    -------
    np.ndarray
        shape (L+1, dim) — 外乱入力下の state 軌跡 (initial を含む)。
    """
    if w_bar < 0.0:
        raise ValueError(f"w_bar must be non-negative, got {w_bar!r}")
    rng = np.random.default_rng(seed)
    L, dim = x_ref.shape
    d = rng.uniform(-w_bar, w_bar, size=(L, dim))
    return run_sequence(x_ref + d, gene)


@dataclass(frozen=True)
class TubeCrossCheck:
    """trajectory-tube 定理の cross-check 結果 (read-only).

    Attributes
    ----------
    contraction_ok : bool
        L<1 (achievable-t box 上の閉形式判定; tracking_tube と同一 L 定義)。
    L_state : float
        状態方向 Lipschitz 上界 (= state_lipschitz_inf)。
    G_input : float
        入力方向ゲイン (= input_gain_inf)。
    certified_tube : float
        証明 tube 半径 r = G·w̄/(1−L) (L≥1 では +∞)。
    empirical_max_err_steady : float
        steady-state 後半の実測 limsup‖e_t‖∞ (全 seed の max)。
    empirical_max_err_all : float
        全 step の実測 max‖e_t‖∞ (過渡含む; 参考)。
    tube_holds : bool
        実測 ≤ certified_tube + tol (P1 の数値裏付け)。L≥1 では tube=∞ なので
        vacuously True にならないよう contraction_ok=False のとき False を返す。
    n_seeds : int
        外乱 driver の seed 数。
    """

    contraction_ok: bool
    L_state: float
    G_input: float
    certified_tube: float
    empirical_max_err_steady: float
    empirical_max_err_all: float
    tube_holds: bool
    n_seeds: int


def tube_cross_check(
    gene: StateUpdateGene,
    *,
    w_bar: float,
    seq_len: int = 256,
    dim: int = 8,
    n_seeds: int = 64,
    max_input_abs: float = 1.0,
    settle_frac: float = 0.5,
    ref_seed: int = 12345,
    tol: float = 1e-9,
) -> TubeCrossCheck:
    """1 gene の certified tube と実測追従誤差を比較する (P1 cross-check).

    手順:
      1. 参照入力 x_ref を一様 [−max_input_abs, max_input_abs]^(L,dim) で生成。
      2. 外乱なしの参照軌道 s_ref = run_sequence(x_ref) (= 系自身の解, ρ_feas=0)。
      3. n_seeds 回、x_ref + uniform d∈[−w̄,w̄] で s_act を回し、
         e_t = s_act − s_ref の steady-state 後半 sup-norm 実測 limsup を取る。
      4. certified tube r = G·w̄/(1−L) と比較。

    Parameters
    ----------
    gene : StateUpdateGene
        検査対象 scalar gene。
    w_bar : float
        外乱上界 (≥0)。
    seq_len : int
        入力列長 L (steady-state を見るため十分長く)。
    dim : int
        state / input 次元。
    n_seeds : int
        外乱 driver の seed 数 (設計 doc は N≥64)。
    max_input_abs : float
        |x_ref| ≤ max_input_abs (achievable-t box / 入力 bound)。
    settle_frac : float
        過渡を捨てる割合 (後半 (1−settle_frac) を steady-state とみなす)。
    ref_seed : int
        参照入力 RNG seed。
    tol : float
        tube_holds 判定の数値許容。

    Returns
    -------
    TubeCrossCheck
        L/G/certified_tube/実測誤差/tube_holds を含む read-only レポート。
    """
    g = gene.clipped()
    # tracking_tube と同一の L 定義 (閉形式 numpy 比較; 設計 doc §4.3 訂正1/3)。
    tt = tracking_tube(g, w_bar=w_bar, max_input_abs=max_input_abs)
    L = state_lipschitz_inf(g, max_input_abs=max_input_abs)
    G = input_gain_inf(g, max_input_abs=max_input_abs)
    certified_tube = tt.tube_radius

    rng = np.random.default_rng(ref_seed)
    x_ref = rng.uniform(-max_input_abs, max_input_abs, size=(seq_len, dim))
    s_ref = run_sequence(x_ref, g)

    settle = int(seq_len * settle_frac)
    max_err_steady = 0.0
    max_err_all = 0.0
    for sd in range(n_seeds):
        s_act = rollout_with_disturbance(g, x_ref, w_bar=w_bar, seed=1000 + sd)
        err = np.max(np.abs(s_act - s_ref), axis=1)  # 各 t の sup-norm 誤差
        max_err_all = max(max_err_all, float(np.max(err)))
        max_err_steady = max(max_err_steady, float(np.max(err[settle:])))

    # contraction でないと tube=∞ で vacuous に holds になるため、contraction_ok を要件にする。
    tube_holds = bool(tt.contraction_ok and max_err_steady <= certified_tube + tol)

    return TubeCrossCheck(
        contraction_ok=tt.contraction_ok,
        L_state=L,
        G_input=G,
        certified_tube=certified_tube,
        empirical_max_err_steady=max_err_steady,
        empirical_max_err_all=max_err_all,
        tube_holds=tube_holds,
        n_seeds=n_seeds,
    )
