# SPDX-License-Identifier: Apache-2.0
"""cert_inf の微分可能 surrogate — 真の内的化 (R-endo 本来の問い) の踏み石。

HD-1 接地 (HD1_GROUNDING_DESIGN.md §7) が指す将来実験: cert_inf = infnorm_sup < 1 の
sound 上界 infnorm_sup を **torch で微分可能な surrogate** として実装し、補助損失で
モデル自身の勾配に組み込む ("モデルが自分を ρ<1 域へ押す")。

infnorm_sup は abs / max / 和 / 積 の合成なので subgradient 可能。本ファイルは
(1) numpy 版 (hd1_highdim_evo.infnorm_sup) との数値一致を確認し
(2) raw_W へ勾配が流れることを確認する = 設計に「微分可能性を実コードで検証済み」と書く根拠。

実行::  py -3.11 research/internalization_poc/cert_surrogate.py  (検証 self-test)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

try:
    sys.stdout.reconfigure(encoding="utf-8")   # cp932 console 対策 (既知パターン)
except Exception:
    pass

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parents[0] / "highdim_evolution"))
import hd1_highdim_evo as H  # noqa: E402  (numpy 正本)


def infnorm_sup_torch(decay: torch.Tensor, W: torch.Tensor, max_input_abs: float = 1.0):
    """infnorm_sup の torch 版 (subgradient 可能)。numpy 版と同一の box-端点 sup を計算。

    decay, W は raw でなく core 値 (decay=sigmoid(raw_decay), W=2·tanh(raw_W))。
    勾配は decay/W → raw_decay/raw_W へ流れる。
    """
    decay = decay.clamp(0.0, 1.0)
    W = W.clamp(-2.0, 2.0)
    absW = W.abs()
    diag_W = torch.diagonal(W)
    M = absW.sum(dim=1) + max_input_abs                 # t_min_per_coord の入力
    t_lo = 1.0 - torch.tanh(M) ** 2                     # 活性化導関数の下界 (per coord)
    off = absW.sum(dim=1) - diag_W.abs()                # 行の非対角 abs 和
    rows = []
    for ti in (t_lo, torch.ones_like(t_lo)):            # box 端点 {t_lo, 1.0} の sup
        diag = (decay + (1.0 - decay) * ti * diag_W).abs()
        rows.append(diag + (1.0 - decay) * ti * off)
    return torch.stack(rows).max()                      # max over (2 端点 × n 行)


def cert_surrogate_loss(decay, W, threshold: float = 1.0):
    """片側 hinge: admit set 内 (infnorm_sup < threshold) では勾配ゼロ、外で押し返す。"""
    return torch.relu(infnorm_sup_torch(decay, W) - threshold)


def _self_test():
    rng = np.random.default_rng(0)
    max_abs_err = 0.0
    for n in (8, 16, 32):
        for _ in range(5):
            decay = rng.uniform(0.1, 0.95, n)
            W = rng.normal(0, 0.4, (n, n))
            # numpy 正本
            np_val = H.infnorm_sup(decay, W, H.t_min_per_coord(decay, W))
            # torch 版
            dt = torch.tensor(decay, dtype=torch.float64)
            Wt = torch.tensor(W, dtype=torch.float64)
            to_val = float(infnorm_sup_torch(dt, Wt))
            max_abs_err = max(max_abs_err, abs(np_val - to_val))
    print(f"[1] numpy↔torch max abs err over 15 cases = {max_abs_err:.2e} "
          f"({'OK (<1e-9)' if max_abs_err < 1e-9 else 'MISMATCH'})")

    # 勾配が raw_W へ流れるか (実際の再パラメータ化 W=2·tanh(raw_W) 経由)
    torch.manual_seed(0)
    raw_W = torch.randn(16, 16, requires_grad=True)
    raw_decay = torch.randn(16, requires_grad=True)
    decay = torch.sigmoid(raw_decay)
    W = 2.0 * torch.tanh(raw_W)
    loss = cert_surrogate_loss(decay, W, threshold=0.95)
    loss.backward()
    gW = raw_W.grad.abs().sum().item()
    gd = raw_decay.grad.abs().sum().item()
    print(f"[2] surrogate loss={float(loss):.4f}  |grad raw_W|={gW:.4f}  |grad raw_decay|={gd:.4f} "
          f"({'OK (flows)' if gW > 0 else 'NO GRAD'})")

    # admit set 内では勾配ゼロ (片側 hinge) — 縮小した core で確認
    raw_W2 = (raw_W.detach() * 0.05).clone().requires_grad_(True)
    W2 = 2.0 * torch.tanh(raw_W2)
    d2 = torch.sigmoid(raw_decay.detach())
    l2 = cert_surrogate_loss(d2, W2, threshold=0.95)
    inside = H.cert_inf(d2.numpy(), W2.detach().numpy())
    print(f"[3] shrunk core: cert_inf={inside}  surrogate loss={float(l2):.4f} "
          f"({'OK (0 inside admit)' if (inside and float(l2) == 0.0) else 'check'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
