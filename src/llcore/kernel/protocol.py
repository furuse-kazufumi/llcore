# SPDX-License-Identifier: Apache-2.0
"""Kernel plugin Protocol 群 (0.2.0a0 設計、S1).

設計 doc: ``docs/design/kernel_plugin_0_2_0a0.md`` (Codex pair-review 5 Findings 反映済).

research/ で実証した複数アーキ (SNN-LIF / Neural ODE / GNN / Izhikevich) を本流に
**構造破綻なく additive 取り込み**するための plugin 境界。3 抽象 + 戻り値型:

- :class:`GeneCodec`      — gene <-> numpy 往復 + 物理範囲 bounds (GA operator 一般化用)
- :class:`Trajectory`     — kernel 横断の simulate 戻り値 (意味論差を field で明示, M1)
- :class:`Kernel`         — 1 アーキの forward dynamics + ChangeOp 適用
- :class:`VerifierBackend` — per-gene Z3 invariant online gate

honest 留保 (設計 doc §2.2):
- Kernel Protocol は「**same design pattern**」を契約化するもので「same verifier stack」ではない。
  各 kernel の :attr:`Trajectory.kind` で意味論差を型レベルに明示し、上流が分岐する。
- per-gene verifier の真正性 (gene を Z3 symbolic に投入しているか vs box 流用) は backend
  docstring で明示する (Izhikevich F1 再発防止)。

semver: 本 module は **新規追加のみ**。0.1.0a0 の既存シンボル (StateUpdateGene / evolve /
verify_gene_safe 等) は一切変更しない (設計 doc §5 (D))。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

import numpy as np

from llcore.verifier import ChangeOp, InvariantResult

# gene 型は kernel ごとに異なる (RWKV=StateUpdateGene, SNN=LIFGene, ...)。
GeneT = TypeVar("GeneT")
# VerifierBackend は gene を入力位置でのみ使う (戻り値に出さない) ため contravariant 版を使う。
GeneT_contra = TypeVar("GeneT_contra", contravariant=True)


@dataclass(frozen=True, eq=False)
class Trajectory:
    """kernel 横断の simulate 戻り値.

    意味論差を field で明示する (設計 doc M1: RWKV ``run_sequence`` と SNN
    ``simulate_lif`` は戻り値が別物。``np.ndarray`` 1 本では吸収できないため正規化).

    ``eq=False`` (Codex review S1 Low 反映): 自動生成 ``__eq__`` は ``primary``
    (np.ndarray) を ``==`` 比較し ambiguous truth value で落ちるため、identity 比較に
    する。Trajectory 同士の中身比較が必要なら ``np.array_equal(a.primary, b.primary)``
    を明示的に使う。

    Attributes
    ----------
    primary : np.ndarray
        主軌跡。RWKV=state 列 (L+1, dim)、SNN=膜電位 V 列 (T,)。
    events : tuple[float, ...]
        離散イベント時刻列。SNN=spike 時刻。RWKV は空 tuple。
    kind : str
        意味論ラベル。``"state"`` (連続 state 列) / ``"spike_voltage"`` (膜電位+spike)。
        上流 (fitness / verifier) が意味論分岐に使う。
    """

    primary: np.ndarray
    events: tuple[float, ...] = ()
    kind: str = "state"


@runtime_checkable
class GeneCodec(Protocol[GeneT]):
    """gene <-> numpy 往復 + 物理範囲 bounds を宣言する契約 (GA operator 一般化用).

    ``StateUpdateGene`` / ``LIFGene`` / ``IzhikevichGene`` は全て ``as_array`` /
    ``clipped`` を持つので、codec は「dim と bounds を宣言する薄い層」で済む。

    ``clip`` を codec が持つのが重要: LIF は ``V_reset < V_th``、Izhikevich は
    ``c < V_PEAK`` という **box clip では表せない依存制約**があるため、単純 bounds clip
    では不足。各 gene の ``clipped()`` メソッドへ委譲して吸収する。
    """

    @property
    def dim(self) -> int:
        """gene parameter 数 (RWKV=3, LIF=4, Izhikevich=4)."""
        ...

    @property
    def lower(self) -> np.ndarray:
        """各 param の下界, shape (dim,)."""
        ...

    @property
    def upper(self) -> np.ndarray:
        """各 param の上界, shape (dim,)."""
        ...

    def to_array(self, gene: GeneT) -> np.ndarray:
        """gene を numpy へ, shape (dim,)."""
        ...

    def from_array(self, arr: np.ndarray) -> GeneT:
        """numpy から gene を復元 (shape (dim,) 前提)."""
        ...

    def clip(self, gene: GeneT) -> GeneT:
        """物理範囲 + 依存制約を反映した clipped gene を返す."""
        ...


@runtime_checkable
class Kernel(Protocol[GeneT]):
    """1 アーキの forward dynamics + 構造変更 op を束ねる plugin.

    Attributes
    ----------
    name : str
        kernel 識別子 (``"rwkv"`` / ``"snn_lif"`` / ``"izhikevich"``)。
    codec : GeneCodec[GeneT]
        この kernel の gene codec。
    change_op_types : tuple[str, ...]
        この kernel が受け付ける ChangeOp の op_type 集合 (C4 一般化)。
        RWKV: decay/mix/gate_shift, SNN: tau/vth/tref_shift 等。
    """

    name: str
    codec: GeneCodec[GeneT]
    change_op_types: tuple[str, ...]

    def simulate(
        self, inputs: np.ndarray, gene: GeneT, initial_state: np.ndarray | None = None
    ) -> Trajectory:
        """L step trajectory を :class:`Trajectory` で返す.

        RWKV は ``run_sequence`` を、SNN は ``simulate_lif`` を adapter で包んで
        正規化する (設計 doc M1)。
        """
        ...

    def apply_change_op(self, gene: GeneT, op: ChangeOp) -> GeneT:
        """ChangeOp を適用した新 gene を返す (純関数).

        設計 doc M2: 既存 ``ChangeOp`` 型をそのまま受ける ((op_type, delta) には
        潰さない)。``ChangeOpSequence`` / ``refinement.py`` 接続の再利用性を保つため。
        kernel ごとに op_type の意味が違う。

        **S3 延期事項 (Codex review S1 Medium)**: 現 ``ChangeOp.__post_init__`` は RWKV 4 種
        (``decay/mix/gate/kernel_swap_mock``) 以外の op_type を拒否するため、非 RWKV kernel
        (SNN の ``tau_shift`` 等) はまだ ``ChangeOp`` を構築できない。op_type 検証を kernel 別
        (``change_op_types`` 参照) に拡張するのは SNN-LIF 移植 (S3, 設計 doc §4.1) で行う。
        S1 時点では RWKV 準拠例として動作する。
        """
        ...


@runtime_checkable
class VerifierBackend(Protocol[GeneT_contra]):
    """per-gene online gate. 進化ループから 1 gene ずつ呼ばれる Z3 invariant 検査.

    戻り値は本流 :class:`InvariantResult` に正規化する (設計 doc 補足 Finding:
    research verifier は独自 result 型を返すため adapter で統一)。
    """

    name: str

    def verify_gene_safe(self, gene: GeneT_contra) -> InvariantResult:
        """gene が安全 invariant を破らないか検査. ``ok=True`` で進化集団に admit."""
        ...

    def is_available(self) -> bool:
        """z3 が import 可能か (False なら ``ok=True`` default で skip)."""
        ...


# 構造的タイピングの便宜上、frozen dataclass の field 名 dim/lower/upper を
# property で宣言した。実装側 (rwkv.py の RWKVCodec) も property で揃える。
__all__ = ["GeneT", "Trajectory", "GeneCodec", "Kernel", "VerifierBackend"]
