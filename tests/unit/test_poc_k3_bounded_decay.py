# SPDX-License-Identifier: Apache-2.0
"""PoC K3-1: lower-bounded log-decay (保持床) の unit tests.

背景 (Kimi K3 技術レポート §2.1.1 の転用、ただし動機は差し替え):
    K3 は chunkwise linear attention の ``1/Γ`` 逆数が BF16 レンジを超える問題を
    ``g = g_min * Sigmoid(e^A z)`` で解いた。llcore の RWKV-4 は running-max
    (log-sum-exp) で既に数値安定なので **その数値的動機は転用できない**。

    転用するのは別の性質: 現行の ``decay = -exp(w)`` は保持率
    ``α = exp(decay) ∈ (0, 1)`` に **下限が無く**、チャネルが α→0 で「死ぬ」
    (1 ステップで全部忘れる) ことを設計上排除できない。有界化すると
    ``α ∈ (exp(g_min), 1)`` となり **「1 ステップでこれ以上速くは忘れられない」
    という保持床が証明可能**になる。llcore の北極星 (メモリ効率 = 保持能力) と
    Verified-Plasticity に直結するのはこちら。
"""
from __future__ import annotations

import math

import pytest
import torch

from llcore.lm.rwkv import (
    RWKVConfig,
    RWKVTimeMix,
    bounded_log_decay,
    init_raw_for_log_decay,
)
from llcore.verifier import InvariantResult, is_z3_available, verify_retention_floor


# ── G1: 純関数としての有界性 ───────────────────────────────────────────
def test_g1_bounded_log_decay_never_crosses_the_floor() -> None:
    """[G1] 極端な raw 値でも log-decay が [g_min, 0] を出ない (床を割らない)."""
    raw = torch.tensor([-1e4, -50.0, -1.0, 0.0, 1.0, 50.0, 1e4])
    g = bounded_log_decay(raw, log_floor=-5.0)

    assert torch.all(g >= -5.0), f"log_floor を下回った: {g}"
    assert torch.all(g <= 0.0), f"正の decay が出た: {g}"


def test_g1c_float_saturation_attains_the_endpoints() -> None:
    """[G1c] 実数では開区間だが float32 は端点に到達する (Z3 証明との差を明示).

    ``sigmoid`` は数学的には (0, 1) だが float32 では ``sigmoid(1e4) == 1.0`` と
    飽和するため ``g == log_floor`` ちょうどが出る。床 ``α >= exp(log_floor)`` は
    保たれる (等号成立は違反ではない) が、**Z3 が証明する開区間は float 実装の
    厳密な記述ではない**。この差を回帰として固定しておく。
    """
    g = bounded_log_decay(torch.tensor([1e4, -1e4]), log_floor=-5.0)

    assert float(g[0]) == pytest.approx(-5.0, abs=0.0), "飽和で床ちょうどに達する"
    assert float(g[1]) == pytest.approx(0.0, abs=0.0), "飽和で減衰ゼロに達する"


def test_g1b_bounded_log_decay_rejects_nonnegative_floor() -> None:
    """[G1b] log_floor >= 0 は減衰にならないので拒否する."""
    with pytest.raises(ValueError, match="log_floor"):
        bounded_log_decay(torch.zeros(3), log_floor=0.0)


# ── G2: additive であること (既定は現行挙動のまま) ─────────────────────
def test_g2_default_config_keeps_unbounded_neg_exp_decay() -> None:
    """[G2] decay_log_floor 未指定なら従来の -exp(w) のまま (回帰ゼロ)."""
    cfg = RWKVConfig(vocab_size=16, block_size=8, n_embd=4, n_layer=1)
    assert cfg.decay_log_floor is None

    mixer = RWKVTimeMix(cfg)
    with torch.no_grad():
        mixer.time_decay.copy_(torch.tensor([0.0, 1.0, 2.0, 3.0]))

    expected = -torch.exp(mixer.time_decay)
    assert torch.allclose(mixer.log_decay(), expected)


def test_g2b_config_rejects_nonnegative_decay_log_floor() -> None:
    """[G2b] decay_log_floor >= 0 は config 段階で弾く."""
    with pytest.raises(ValueError, match="decay_log_floor"):
        RWKVConfig(vocab_size=16, block_size=8, decay_log_floor=0.0)


# ── G3: モデル経路での保持床 (経験的) ──────────────────────────────────
def test_g3_bounded_config_enforces_retention_floor_empirically() -> None:
    """[G3] 有界版は極端な time_decay でも α >= exp(g_min) を割らない."""
    floor = -5.0
    cfg = RWKVConfig(vocab_size=16, block_size=8, n_embd=4, n_layer=1, decay_log_floor=floor)
    mixer = RWKVTimeMix(cfg)
    with torch.no_grad():
        mixer.time_decay.copy_(torch.tensor([-1e4, -10.0, 10.0, 1e4]))

    alpha = torch.exp(mixer.log_decay())
    assert torch.all(alpha >= math.exp(floor)), f"保持床を割った: {alpha}"
    assert torch.all(alpha <= 1.0), f"保持率が 1 を超えた: {alpha}"


def test_g3b_unbounded_config_can_kill_a_channel() -> None:
    """[G3b] 現行 -exp(w) は α→0 (チャネル死) を許してしまう = 床が無い証拠."""
    cfg = RWKVConfig(vocab_size=16, block_size=8, n_embd=2, n_layer=1)
    mixer = RWKVTimeMix(cfg)
    with torch.no_grad():
        mixer.time_decay.copy_(torch.tensor([5.0, 10.0]))

    alpha = torch.exp(mixer.log_decay())
    assert torch.all(alpha < math.exp(-5.0)), f"床を割れなかった: {alpha}"


# ── G4-G6: Z3 による証明 ───────────────────────────────────────────────
@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
def test_g4_z3_proves_single_step_retention_floor() -> None:
    """[G4] 有界版は 1 ステップ保持床を Z3 が証明する (unsat = 反例なし)."""
    r = verify_retention_floor(log_floor=-5.0, steps=1, parameterization="bounded_sigmoid")

    assert isinstance(r, InvariantResult)
    assert r.ok
    assert r.used_z3
    assert r.solver_status == "unsat"


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
def test_g5_z3_finds_counterexample_for_unbounded_decay() -> None:
    """[G5] soundness: 現行 -exp(w) では床が成立せず Z3 が反例を出す."""
    r = verify_retention_floor(log_floor=-5.0, steps=1, parameterization="neg_exp")

    assert not r.ok
    assert r.used_z3
    assert r.solver_status == "sat"
    assert r.counterexample is not None


@pytest.mark.skipif(not is_z3_available(), reason="z3-solver not installed")
def test_g6_z3_proves_cumulative_floor_over_multiple_steps() -> None:
    """[G6] T ステップ累積でも床 (>= steps*g_min) が保たれることを証明."""
    r = verify_retention_floor(log_floor=-5.0, steps=8, parameterization="bounded_sigmoid")

    assert r.ok
    assert r.used_z3
    assert r.solver_status == "unsat"


# ── G8: パラメータ化を跨いで初期保持率を揃える (統制実験の前提) ────────
def test_g8_init_raw_reproduces_target_for_both_parameterizations() -> None:
    """[G8] 目標 log-decay を与えると、どちらのパラメータ化でもそれを再現する raw が出る.

    これが無いと baseline と有界版で **初期保持率が違ったまま比較**してしまう
    (zeros 初期化だと -exp(0)=-1 vs -5*sigmoid(0)=-2.5 でズレる)。
    """
    target = -1.0

    raw_unbounded = init_raw_for_log_decay(target, log_floor=None)
    got_unbounded = -torch.exp(torch.tensor([raw_unbounded]))
    assert float(got_unbounded[0]) == pytest.approx(target, rel=1e-6)

    raw_bounded = init_raw_for_log_decay(target, log_floor=-5.0)
    got_bounded = bounded_log_decay(torch.tensor([raw_bounded]), log_floor=-5.0)
    assert float(got_bounded[0]) == pytest.approx(target, rel=1e-6)


def test_g8b_init_raw_rejects_target_outside_the_floor() -> None:
    """[G8b] 床より下の目標は有界版で表現できないので拒否する."""
    with pytest.raises(ValueError, match="log_floor"):
        init_raw_for_log_decay(-9.0, log_floor=-5.0)


def test_g8c_init_raw_rejects_nonnegative_target() -> None:
    """[G8c] target >= 0 は減衰にならないので拒否する."""
    with pytest.raises(ValueError, match="target_log_decay"):
        init_raw_for_log_decay(0.0, log_floor=None)


def test_g7_verify_retention_floor_rejects_bad_steps() -> None:
    """[G7] steps < 1 は無意味なので拒否する."""
    with pytest.raises(ValueError, match="steps"):
        verify_retention_floor(log_floor=-5.0, steps=0)
