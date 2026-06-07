# SPDX-License-Identifier: Apache-2.0
"""Stage 3b kernel 多様化 — 最小 skeleton (KernelGenome + 4 kernel forward dynamics).

DESIGN_kernel_diversification_3b.md の `gene_representation` / `kernel_maps` を最小実装する
**research 隔離** skeleton。src/ は **読むだけで再利用**し一切改変しない。

要点 (DESIGN 準拠):
- :class:`KernelGenome` — タグ付き union genome ``(kernel_id, theta[MAX_DIM])``。
  ``kernel_id`` は連続実数 ``k∈[0,n)`` を floor 離散化 (連続ベクトル GA operator を無改変流用)。
  ``theta`` は固定長 MAX_DIM=4。各 kernel は先頭 ``dim(kernel_id)`` 次元のみ自分の codec で
  解釈し、残余を junk DNA として無視。GA dim = 1 + MAX_DIM = 5。
- **後方互換 (BG5)**: ``kernel_id=0`` (rwkv) のとき ``theta[:3]`` を既存
  ``StateUpdateGene(decay, mix, gate_str)`` に decode し、**既存 ``run_sequence`` を
  そのまま呼ぶ**。3-param 既存実験は kernel_id=0 部分空間に完全埋め込みされる。
- 他 3 kernel (mamba/hopfield/linear_attn) は **対角スカラ mock** (full 実装ではない,
  honest: DESIGN §2 のスコープ宣言 — OTHERARCH 規律で toy analogue に降格)。

honest 留保:
- 本 module は forward dynamics + decode のみ。Z3 gate (BG1/2/3) は `smoke_kernel_gates.py`、
  state_norm/finite の動的 smoke は `smoke_kernels.py` が担当 (関心分離)。
- mamba/hopfield/linear_attn は 1-state 対角簡約で、multi-head 二次相互作用 / 高次元
  retrieval を落とす。claim は「kernel 別 state-update を同一 genome 表現に載せられる」
  という mechanism feasibility に限定し、full kernel の性能主張はしない。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# src を import path に (DESIGN の「読むだけ再利用」: 既存 StateUpdateGene/run_sequence のみ)
_SRC = str(Path(__file__).resolve().parents[2] / "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from llcore.state_update import StateUpdateGene, run_sequence  # noqa: E402

# ---------------------------------------------------------------------------
# union genome の固定パラメータ
# ---------------------------------------------------------------------------
MAX_DIM = 4  # theta 固定長 (rwkv/mamba/hopfield/linear_attn は全て dim=3 + junk 1)
N_KERNELS = 4  # 0=rwkv, 1=mamba_selective, 2=hopfield_dense, 3=linear_attn
GA_DIM = 1 + MAX_DIM  # = 5。連続ベクトル GA operator が触る次元数。

KERNEL_NAMES: tuple[str, ...] = (
    "rwkv",
    "mamba_selective",
    "hopfield_dense",
    "linear_attn",
)

# 各 kernel が実際に使う theta 次元数 (残余は junk DNA)
KERNEL_DIMS: dict[str, int] = {
    "rwkv": 3,
    "mamba_selective": 3,
    "hopfield_dense": 3,
    "linear_attn": 3,
}

# clip 範囲 (theta 先頭 dim のみ。junk 次元は [0,1] でクリップ = GA 探索の暴走防止)
# rwkv:        (decay∈[0,1], mix∈[-1,1], gate_str∈[-2,2])  — 既存 StateUpdateGene 範囲と一致
# mamba:       (alpha∈[-2,2], beta∈[-2,2], gain∈[-1,1])
# hopfield:    (eta∈[0,1], beta∈[0,3], xi∈[-1,1])
# linear_attn: (w∈[-2,2], lam∈[0,1], v_gain∈[-1,1])  — lam<1 帯で有界化
KERNEL_THETA_LOWER: dict[str, np.ndarray] = {
    "rwkv": np.array([0.0, -1.0, -2.0]),
    "mamba_selective": np.array([-2.0, -2.0, -1.0]),
    "hopfield_dense": np.array([0.0, 0.0, -1.0]),
    "linear_attn": np.array([-2.0, 0.0, -1.0]),
}
KERNEL_THETA_UPPER: dict[str, np.ndarray] = {
    "rwkv": np.array([1.0, 1.0, 2.0]),
    "mamba_selective": np.array([2.0, 2.0, 1.0]),
    "hopfield_dense": np.array([1.0, 3.0, 1.0]),
    "linear_attn": np.array([2.0, 1.0, 1.0]),
}


@dataclass(frozen=True)
class KernelGenome:
    """タグ付き union genome ``(kernel_id, theta[MAX_DIM])``.

    Attributes
    ----------
    kernel_id : float
        連続実数 ``k∈[0, N_KERNELS)``。:meth:`kernel_index` で floor 離散化。
        連続値で持つことで既存連続ベクトル GA operator (rng.normal, clip) を無改変流用できる。
    theta : np.ndarray
        固定長 MAX_DIM=4 のパラメータベクトル。先頭 ``dim`` 次元のみ当該 kernel が解釈。
    """

    kernel_id: float
    theta: np.ndarray

    def __post_init__(self) -> None:
        arr = np.asarray(self.theta, dtype=np.float64)
        if arr.shape != (MAX_DIM,):
            raise ValueError(f"theta must be shape ({MAX_DIM},), got {arr.shape}")
        object.__setattr__(self, "theta", arr)

    # --- GA 連続ベクトル <-> genome 往復 (固定 dim=GA_DIM=5) ---
    def as_array(self) -> np.ndarray:
        """連続 GA ベクトル ``[kernel_id, theta0..theta3]`` (shape (5,)) へ."""
        return np.concatenate([[self.kernel_id], self.theta]).astype(np.float64)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> "KernelGenome":
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (GA_DIM,):
            raise ValueError(f"expected shape ({GA_DIM},), got {arr.shape}")
        return cls(kernel_id=float(arr[0]), theta=arr[1:].copy())

    def kernel_index(self) -> int:
        """floor 離散化 + 範囲 clamp で kernel index (0..N_KERNELS-1) を返す."""
        k = int(np.floor(self.kernel_id))
        return int(np.clip(k, 0, N_KERNELS - 1))

    def kernel_name(self) -> str:
        return KERNEL_NAMES[self.kernel_index()]

    def clipped(self) -> "KernelGenome":
        """kernel_id を [0, N_KERNELS) に、theta 先頭 dim を kernel 範囲に、junk を [0,1] に clip."""
        name = self.kernel_name()
        dim = KERNEL_DIMS[name]
        lo, hi = KERNEL_THETA_LOWER[name], KERNEL_THETA_UPPER[name]
        new_theta = self.theta.copy()
        new_theta[:dim] = np.clip(new_theta[:dim], lo, hi)
        # junk 次元 (dim..MAX_DIM) は探索暴走防止に [0,1] へ。decode 時無視されるので意味論非干渉。
        if dim < MAX_DIM:
            new_theta[dim:] = np.clip(new_theta[dim:], 0.0, 1.0)
        # kernel_id は [0, N_KERNELS) 半開区間へ (上端は floor で N_KERNELS-1 に落ちるが念のため)
        kid = float(np.clip(self.kernel_id, 0.0, N_KERNELS - 1e-9))
        return KernelGenome(kernel_id=kid, theta=new_theta)


# ===========================================================================
# 各 kernel の 1-step 対角写像 (DESIGN §2 kernel_maps) — research 隔離 mock。
# 契約: step(s, x, theta3) -> s'  (s,x はスカラまたは np.ndarray, 各座標独立=対角)
#   theta3 = 当該 kernel が解釈する先頭 dim 次元 (numpy array, shape (dim,))
# ===========================================================================


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _softplus(z: np.ndarray) -> np.ndarray:
    # 数値安定 softplus = max(z,0) + log1p(exp(-|z|))
    return np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))


def rwkv_step(s, x, th):
    """既存 RWKV と同式: s' = decay*s + (1-decay)*tanh(mix*x + gate*s)."""
    decay, mix, gate = th
    return decay * s + (1.0 - decay) * np.tanh(mix * x + gate * s)


def mamba_step(s, x, th):
    """対角 1-state selective SSM mock: a=sigmoid(alpha*x+beta); s'=a*s+(1-a)*(gain*x)."""
    alpha, beta, gain = th
    a = _sigmoid(alpha * x + beta)
    return a * s + (1.0 - a) * (gain * x)


def hopfield_step(s, x, th):
    """1-pattern 連想想起 対角 mock: z=beta*(xi*tanh(s)+x); s'=(1-eta)*s+eta*tanh(z)."""
    eta, beta, xi = th
    z = beta * (xi * np.tanh(s) + x)
    return (1.0 - eta) * s + eta * np.tanh(z)


def linattn_step(s, x, th):
    """bounded linear attention 対角 mock: phi=softplus(w*x); s'=lam*s + phi*(v_gain*x)."""
    w, lam, v_gain = th
    phi = _softplus(w * x)
    return lam * s + phi * (v_gain * x)


_KERNEL_STEPS = {
    "rwkv": rwkv_step,
    "mamba_selective": mamba_step,
    "hopfield_dense": hopfield_step,
    "linear_attn": linattn_step,
}


def decode_theta(genome: KernelGenome) -> np.ndarray:
    """genome から当該 kernel が解釈する theta 先頭 dim 次元を抽出 (junk 無視)."""
    name = genome.kernel_name()
    dim = KERNEL_DIMS[name]
    return genome.theta[:dim].copy()


def run_sequence_kernel(
    genome: KernelGenome,
    inputs: np.ndarray,
    initial_state: np.ndarray | None = None,
) -> np.ndarray:
    """KernelGenome の選択 kernel で L step trajectory を回す.

    **後方互換 (BG5)**: kernel_id=0 (rwkv) のときは既存 ``StateUpdateGene`` + 既存
    ``run_sequence`` をそのまま呼ぶ (src 再利用、bit 一致担保)。他 kernel は research mock。

    Parameters
    ----------
    genome : KernelGenome
        clip 済みでなくても内部で clip される。
    inputs : np.ndarray
        shape (L, dim) — L step の入力列。
    initial_state : np.ndarray | None
        shape (dim,) — None なら zero 初期化。

    Returns
    -------
    states : np.ndarray
        shape (L+1, dim) — initial を含む全 step の state。
    """
    if inputs.ndim != 2:
        raise ValueError(f"inputs must be 2D (L, dim), got {inputs.shape}")
    g = genome.clipped()
    name = g.kernel_name()
    th = decode_theta(g)

    if name == "rwkv":
        # 既存 src 経路 (StateUpdateGene + run_sequence) を再利用 = 後方互換の核 (BG5)。
        gene = StateUpdateGene(decay=float(th[0]), mix=float(th[1]), gate_str=float(th[2]))
        return run_sequence(inputs, gene, initial_state=initial_state)

    # 他 kernel: research mock を座標独立 (対角) に L step 適用。
    step = _KERNEL_STEPS[name]
    L, dim = inputs.shape
    state = (
        np.zeros(dim, dtype=np.float64)
        if initial_state is None
        else initial_state.astype(np.float64).copy()
    )
    states = np.empty((L + 1, dim), dtype=np.float64)
    states[0] = state
    for t in range(L):
        state = step(state, inputs[t], th)
        states[t + 1] = state
    return states


__all__ = [
    "MAX_DIM",
    "N_KERNELS",
    "GA_DIM",
    "KERNEL_NAMES",
    "KERNEL_DIMS",
    "KERNEL_THETA_LOWER",
    "KERNEL_THETA_UPPER",
    "KernelGenome",
    "rwkv_step",
    "mamba_step",
    "hopfield_step",
    "linattn_step",
    "decode_theta",
    "run_sequence_kernel",
]
