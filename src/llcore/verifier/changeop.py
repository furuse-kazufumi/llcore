# SPDX-License-Identifier: Apache-2.0
"""ChangeOp — atomic 構造変更 operation (Stage 3a 独自軸 #5 核).

llcore の進化ループ内で **NN の "異なる構造" 遷移を 1 step 単位** に表現するための
最小データ型。Marabou Incremental NN Verification (Wu et al. 2026-03) は
「**同構造内** での weight 微変更」に対する refinement relation を扱うが、
llcore は **異なる構造間** (kernel/decay/mix/gate 構成の swap や shift) を
ChangeOp 単位の small-step transition に分解し、sound 拡張 refinement relation を
ChangeOp 粒度で定義する。

ChangeOp = (op_type, delta) の atomic タプル:
- ``op_type`` ∈ {decay_shift, mix_shift, gate_shift, kernel_swap_mock}
- ``delta``   = float (shift 量) または kernel 識別子 (kernel_swap_mock 時)

合成性 (Composability):
    R(N0, N1, c1) ∧ R(N1, N2, c2) → R(N0, N2, c1∘c2)
の数式的根拠は ``compose`` の docstring 参照。本 module は Z3 検査 module
(``refinement.py``) と独立に純粋 dataclass + 列計算のみを担う。

honest 留保:
- kernel_swap_mock は実 NN kernel 交換ではなく gate 構造の mock スイッチ。
  実 RWKV/SSM kernel への置き換えは Stage 5+ で扱う。
- ChangeOp の magnitude (`delta` の絶対値) が ``epsilon`` を線形に決める。
  非線形 epsilon は将来 (Wong-Carlini-Mądry 的 certified radius) で拡張。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence


# ChangeOp op_type 値域。enum でなく文字列 literal で扱う (Z3 構築側で
# 文字列分岐するため; mypy literal type は型注釈で表現)。
OP_TYPES = ("decay_shift", "mix_shift", "gate_shift", "kernel_swap_mock")


@dataclass(frozen=True)
class ChangeOp:
    """Atomic 構造変更 operation.

    Attributes
    ----------
    op_type : str
        変更種別 (``decay_shift`` / ``mix_shift`` / ``gate_shift`` /
        ``kernel_swap_mock``).
    delta : float
        - ``*_shift``: shift 量 (符号あり)。例: decay_shift, delta=-0.1 →
          decay を 0.1 減じる。
        - ``kernel_swap_mock``: kernel 切替を mock する識別 float (0 or 1)。
          0 = identity kernel、1 = recurrent-strengthened kernel mock。

    Notes
    -----
    sound 拡張 refinement relation の ε (許容劣化) は ``magnitude()`` に
    線形比例する設計 (PoC 3a)。
    """

    op_type: str
    delta: float

    def __post_init__(self) -> None:
        if self.op_type not in OP_TYPES:
            raise ValueError(
                f"unknown op_type {self.op_type!r}; expected one of {OP_TYPES}"
            )
        if self.op_type == "kernel_swap_mock" and self.delta not in (0.0, 1.0):
            raise ValueError(
                f"kernel_swap_mock requires delta in (0.0, 1.0), got {self.delta}"
            )

    def magnitude(self) -> float:
        """変更の "大きさ" (|delta|).

        sound 拡張 refinement relation の epsilon 上限を決める基礎量。
        kernel_swap_mock は 0/1 discrete だが、mock では magnitude を delta そのもの
        と同一視 (1=切替あり=大変更, 0=無変更)。
        """
        return float(abs(self.delta))

    @classmethod
    def identity(cls) -> ChangeOp:
        """無変更 ChangeOp (delta=0)。"""
        return cls(op_type="decay_shift", delta=0.0)


@dataclass(frozen=True)
class ChangeOpSequence:
    """ChangeOp 列 (合成性検査の最小単位).

    composition ``c1 ∘ c2`` は ``ChangeOpSequence(ops=(c1, c2))`` で表現。
    sound 拡張 R(N0, N_k, c1∘…∘ck) を Z3 で検査する際に sequential に適用される。

    無限列耐性 (要件 B): ``ops`` は任意長の Sequence で、100 step を超えても
    state_norm bound が崩れないことを ``refinement.py`` が検査する。

    Attributes
    ----------
    ops : tuple[ChangeOp, ...]
        順序付き ChangeOp 列。``ops[0]`` が最初に適用される。
    """

    ops: tuple[ChangeOp, ...] = field(default_factory=tuple)

    def __len__(self) -> int:
        return len(self.ops)

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self.ops)

    def append(self, op: ChangeOp) -> ChangeOpSequence:
        """ChangeOp を 1 個追加した新列を返す (frozen, 元列不変)."""
        return ChangeOpSequence(ops=self.ops + (op,))

    def total_magnitude(self) -> float:
        """全 ChangeOp の magnitude 和.

        sound 拡張 R の epsilon 累積は magnitude 和に比例する (epsilon 線形性、
        ``refinement.py`` 参照)。無限列耐性の数学的限界は **magnitude 和が
        bounded** な ChangeOp 列のみ (それ以外は trivially state_norm が発散)。
        """
        return float(sum(op.magnitude() for op in self.ops))

    def compose(self, other: ChangeOpSequence) -> ChangeOpSequence:
        """``self`` の後に ``other`` を適用する合成列を返す.

        合成性 (要件 A):
            R(N0, N1, self) ∧ R(N1, N2, other)
            → R(N0, N2, self.compose(other))

        Z3 で R(N0, N2, c1∘c2) を直接検査するときに使う列構築。
        """
        return ChangeOpSequence(ops=self.ops + other.ops)


# ---------------------------------------------------------------------------
# Convenience constructors (curriculum で頻用)
# ---------------------------------------------------------------------------


def decay_shift(delta: float) -> ChangeOp:
    """decay を delta だけシフトする ChangeOp."""
    return ChangeOp(op_type="decay_shift", delta=float(delta))


def mix_shift(delta: float) -> ChangeOp:
    """mix を delta だけシフトする ChangeOp."""
    return ChangeOp(op_type="mix_shift", delta=float(delta))


def gate_shift(delta: float) -> ChangeOp:
    """gate_str を delta だけシフトする ChangeOp."""
    return ChangeOp(op_type="gate_shift", delta=float(delta))


def kernel_swap_mock(swap: bool) -> ChangeOp:
    """kernel 切替 mock ChangeOp (Stage 3a, 実 NN kernel 交換ではない)."""
    return ChangeOp(op_type="kernel_swap_mock", delta=1.0 if swap else 0.0)


def apply_changeop(
    gene: "StateUpdateGeneLike", op: ChangeOp
) -> "StateUpdateGeneLike":
    """ChangeOp を gene に適用した新 gene を返す (純関数, side effect なし).

    Parameters
    ----------
    gene : StateUpdateGeneLike
        ``decay`` / ``mix`` / ``gate_str`` を持つ任意の object (Protocol 風).
        ``StateUpdateGene`` (state_update.genes) または同等 dataclass.
    op : ChangeOp
        適用する変更.

    Returns
    -------
    StateUpdateGeneLike
        変更後 gene (clip はここでは行わない — refinement 側で範囲外も検査).
    """
    # 動的 import を避けるため、import を関数内に置かず、シグネチャに依存しない
    # 構造的タイピングで処理。StateUpdateGene は frozen dataclass。
    from llcore.state_update import StateUpdateGene

    decay, mix, gate_str = gene.decay, gene.mix, gene.gate_str
    if op.op_type == "decay_shift":
        decay = decay + op.delta
    elif op.op_type == "mix_shift":
        mix = mix + op.delta
    elif op.op_type == "gate_shift":
        gate_str = gate_str + op.delta
    elif op.op_type == "kernel_swap_mock":
        # mock: swap=1 で gate_str を反転 (recurrent 構造の mock 切替)
        if op.delta == 1.0:
            gate_str = -gate_str
    return StateUpdateGene(decay=decay, mix=mix, gate_str=gate_str)


def apply_sequence(
    gene: "StateUpdateGeneLike", seq: ChangeOpSequence
) -> "StateUpdateGeneLike":
    """ChangeOp 列を順次適用した最終 gene を返す."""
    g = gene
    for op in seq.ops:
        g = apply_changeop(g, op)
    return g


# Type alias を表す Protocol は同 module 内では不要 (StateUpdateGene 自体を使う)。
# 静的解析向けに forward reference を残す。
class StateUpdateGeneLike:  # pragma: no cover - typing-only proto
    decay: float
    mix: float
    gate_str: float


# Convenience: arbitrary length sequence generator (curriculum 等で使う)
def sequence_from_iter(ops: Sequence[ChangeOp]) -> ChangeOpSequence:
    """任意 iterable から ChangeOpSequence を作る."""
    return ChangeOpSequence(ops=tuple(ops))
