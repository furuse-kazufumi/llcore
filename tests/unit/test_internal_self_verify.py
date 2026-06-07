# SPDX-License-Identifier: Apache-2.0
"""R-endo: StateUpdateGene.is_verified_trajectory_tube の回帰テスト (additive).

内的検証器 PoC で追加した read-only 自己検証メソッドが
(1) 外部 gate (tracking_tube) と verdict 一致 (H3 correctness 不変量)
(2) 環境 (w_bar) に結合して verdict が変わる (entity が現環境を sense する)
(3) read-only (gene を破壊しない)
を恒久検証する。method 追加が既存挙動を壊さない additive 規律の自動検出器。
"""
from __future__ import annotations

import numpy as np
import pytest

from llcore.state_update import StateUpdateGene
from llcore.verifier.tracking_tube import tracking_tube

W_BAR = 0.1
R_MAX = 0.05


def test_self_verdict_matches_external_gate():
    """H3 correctness: self-verdict == 外部 gate verdict (random 5000 gene, 0 乖離)."""
    rng = np.random.default_rng(0)
    disagreements = 0
    for _ in range(5000):
        g = StateUpdateGene(
            decay=float(rng.uniform(0, 1)),
            mix=float(rng.uniform(-1, 1)),
            gate_str=float(rng.uniform(-2, 2)),
        )
        self_verdict = g.is_verified_trajectory_tube(W_BAR, R_MAX)
        external = bool(tracking_tube(g, w_bar=W_BAR, r_max=R_MAX).admits)
        if self_verdict != external:
            disagreements += 1
    assert disagreements == 0


def test_environment_coupling():
    """entity の self-verdict は現在の外乱 w_bar に追従する (環境結合)."""
    # decay=0.5,gate_str=0 → G=0.5,L=0.5,tube=G·w̄/(1−L)=w̄。w̄=0.05→tube0.05≤r_max admit、
    # w̄=0.20→tube0.20>r_max reject。
    g = StateUpdateGene(decay=0.5, mix=1.0, gate_str=0.0)
    assert g.is_verified_trajectory_tube(0.05, R_MAX) is True
    assert g.is_verified_trajectory_tube(0.20, R_MAX) is False


def test_readonly_does_not_mutate_gene():
    """method 呼び出しが gene を破壊しない (frozen dataclass の read-only 保証)."""
    g = StateUpdateGene(decay=0.7, mix=-0.3, gate_str=0.4)
    snapshot = (g.decay, g.mix, g.gate_str)
    _ = g.is_verified_trajectory_tube(W_BAR, R_MAX)
    assert (g.decay, g.mix, g.gate_str) == snapshot


@pytest.mark.parametrize("r_max", [None, 0.05, 0.5])
def test_matches_external_under_varied_rmax(r_max):
    """r_max を変えても外部 gate と一致 (None=contraction のみ含む)."""
    rng = np.random.default_rng(7)
    for _ in range(500):
        g = StateUpdateGene(
            decay=float(rng.uniform(0, 1)),
            mix=float(rng.uniform(-1, 1)),
            gate_str=float(rng.uniform(-2, 2)),
        )
        assert g.is_verified_trajectory_tube(W_BAR, r_max) == bool(
            tracking_tube(g, w_bar=W_BAR, r_max=r_max).admits
        )
