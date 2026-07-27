# SPDX-License-Identifier: Apache-2.0
"""有界活性化 (bounded activations).

低ビット化・形式検証のどちらにとっても「出力が設計で有界」であることは効く。
実行時に外れ値が出ないことを祈るのではなく、**値域を関数の形で閉じる**。

収録:
- :func:`softcap` — ``β·tanh(x/β)``。原点近傍はほぼ恒等、大入力で ``±β`` に飽和。
- :func:`situ_glu` — Kimi K3 の Sigmoid Tanh Unit GLU。SwiGLU の両枝に softcap を
  掛けて出力を ``|f| <= β_gate · β_up`` に有界化する。

由来: Kimi K3 技術レポート §2.3.2 (Eq. 12)。K3 は 2.8T スケールで routed expert の
活性爆発を抑えるために導入し、``β1 = 4`` (gate 枝) / ``β2 = 25`` (up 枝) を採用した。

honest 留保:
- K3 の動機は 2.8T MoE の極端な疎性下での活性爆発であり、**小規模で同じ利得がある
  保証はない**。採用可否は必ず自前のベースライン比較で判断すること。
- 有界化は表現力の上限を課す。上界 ``β_gate·β_up`` が実際の活性レンジより小さいと
  性能を削る側に働く。
"""
from __future__ import annotations

import torch

__all__ = ["situ_glu", "softcap"]

#: Kimi K3 が採用した softcap 係数 (gate 枝, up 枝)。上界は積 = 100。
K3_BETA_GATE = 4.0
K3_BETA_UP = 25.0


def softcap(x: torch.Tensor, *, beta: float) -> torch.Tensor:
    """滑らかな上下限 ``β·tanh(x/β)``.

    ``tanh`` は ``(-1, 1)`` なので出力は ``(-β, β)`` に収まる。原点近傍では
    ``tanh(t) ≈ t`` より ``β·tanh(x/β) ≈ x`` となり、局所的な応答を壊さない。
    hard clamp と違い全域で微分可能。

    Parameters
    ----------
    x : torch.Tensor
        入力。値域の制約なし。
    beta : float
        飽和値 (正)。

    Returns
    -------
    torch.Tensor
        ``[-β, β]`` に収まる値 (float 飽和により端点に到達しうる)。

    Raises
    ------
    ValueError
        ``beta <= 0`` の場合。
    """
    if beta <= 0.0:
        raise ValueError(f"beta must be > 0, got {beta}")
    return beta * torch.tanh(x / beta)


def situ_glu(
    gate_pre: torch.Tensor,
    up_pre: torch.Tensor,
    *,
    beta_gate: float = K3_BETA_GATE,
    beta_up: float = K3_BETA_UP,
) -> torch.Tensor:
    """Sigmoid Tanh Unit GLU (Kimi K3 §2.3.2)。

    ``SiTU-GLU = [softcap(g, β_gate) ⊙ Sigmoid(g)] ⊙ softcap(u, β_up)``

    SwiGLU ``[g·Sigmoid(g)] ⊙ u`` は両枝とも非有界で、大きな座標が同時に立つと
    外れ値を生む。SiTU-GLU は gate 枝の線形因子と up 枝の双方に softcap を掛けるため

        ``|SiTU-GLU| <= β_gate · 1 · β_up``

    が**入力に依らず**成り立つ (``Sigmoid <= 1``)。原点近傍では ``softcap ≈ 恒等``
    なので SwiGLU の局所応答をほぼ保つ。

    Parameters
    ----------
    gate_pre : torch.Tensor
        gate 枝の線形出力 ``W_g x``。
    up_pre : torch.Tensor
        up 枝の線形出力 ``W_u x``。``gate_pre`` と broadcast 可能な形。
    beta_gate : float
        gate 枝の softcap 係数 (既定 = K3 の 4.0)。
    beta_up : float
        up 枝の softcap 係数 (既定 = K3 の 25.0)。

    Returns
    -------
    torch.Tensor
        ``|出力| <= beta_gate * beta_up`` を満たす値。

    Raises
    ------
    ValueError
        いずれかの ``beta`` が正でない場合。
    """
    gate = softcap(gate_pre, beta=beta_gate) * torch.sigmoid(gate_pre)
    return gate * softcap(up_pre, beta=beta_up)
