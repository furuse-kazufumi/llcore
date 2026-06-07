# SPDX-License-Identifier: Apache-2.0
"""llcore.kernel — Kernel plugin アーキテクチャ (0.2.0a0, S1).

設計 doc: ``docs/design/kernel_plugin_0_2_0a0.md``.

複数アーキを本流に additive 取り込みするための plugin 境界:
- :mod:`llcore.kernel.protocol` — GeneCodec / Trajectory / Kernel / VerifierBackend
- :mod:`llcore.kernel.rwkv`     — RWKV 準拠例 (本流委譲 wrapper)

次段 (S3): ``llcore.kernel.snn_lif`` で SNN-LIF を 2 例目として追加予定。
"""
from .protocol import GeneCodec, Kernel, Trajectory, VerifierBackend
from .rwkv import RWKVCodec, RWKVKernel, RWKVStateNormBackend

__all__ = [
    # protocol
    "GeneCodec",
    "Kernel",
    "Trajectory",
    "VerifierBackend",
    # rwkv 準拠例
    "RWKVCodec",
    "RWKVKernel",
    "RWKVStateNormBackend",
]
