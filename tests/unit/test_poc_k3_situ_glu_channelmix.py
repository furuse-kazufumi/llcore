# SPDX-License-Identifier: Apache-2.0
"""PoC K3-2b: SiTU-GLU を RWKV channel-mix に additive option として配線するテスト.

RWKV-4 の channel-mix FFN は ``value(relu(key(x))**2)`` = **squared-ReLU** で、
非有界かつ二次に伸びる。低ビット化・形式検証の観点では最も扱いにくい形なので、
有界な SiTU-GLU を選べるようにする。**既定は従来のまま (回帰ゼロ)**。
"""
from __future__ import annotations

import pytest
import torch

from llcore.lm.activations import K3_BETA_GATE, K3_BETA_UP
from llcore.lm.rwkv import RWKVChannelMix, RWKVConfig, RWKVLM


def _cfg(**kw: object) -> RWKVConfig:
    base: dict[str, object] = {"vocab_size": 16, "block_size": 8, "n_embd": 8, "n_layer": 1}
    base.update(kw)
    return RWKVConfig(**base)  # type: ignore[arg-type]


# ── G1: 既定は従来挙動 ────────────────────────────────────────────────
def test_g1_default_ffn_activation_is_squared_relu() -> None:
    """[G1] 既定は squared-ReLU のまま (additive であること)."""
    cfg = _cfg()
    assert cfg.ffn_activation == "sq_relu"

    mixer = RWKVChannelMix(cfg)
    assert not hasattr(mixer, "up"), "既定モードで GLU 用の up 射影を作ってはいけない"


def test_g1b_default_path_matches_hand_computed_squared_relu() -> None:
    """[G1b] 既定経路の数値が従来式 ``value(relu(key(xk))**2)`` と一致する."""
    torch.manual_seed(0)
    cfg = _cfg()
    mixer = RWKVChannelMix(cfg)
    x = torch.randn(2, cfg.n_embd)
    prev = torch.randn(2, cfg.n_embd)

    got = mixer.step(x, prev)

    xk = x * mixer.mix_k + prev * (1.0 - mixer.mix_k)
    xr = x * mixer.mix_r + prev * (1.0 - mixer.mix_r)
    k = torch.relu(mixer.key(xk))
    expected = torch.sigmoid(mixer.receptance(xr)) * mixer.value(k * k)

    assert torch.allclose(got, expected, atol=1e-6)


# ── G2: situ_glu モード ───────────────────────────────────────────────
def test_g2_situ_glu_mode_builds_an_up_projection() -> None:
    """[G2] situ_glu を選ぶと GLU に必要な up 射影が生える."""
    mixer = RWKVChannelMix(_cfg(ffn_activation="situ_glu"))

    assert hasattr(mixer, "up"), "GLU なのに up 射影が無い"
    assert mixer.up.out_features == mixer.key.out_features


def test_g2b_situ_glu_hidden_activation_is_bounded() -> None:
    """[G2c] situ_glu モードでは value へ渡る中間活性が |a| <= β_gate*β_up に有界.

    squared-ReLU は入力を大きくすると二次で発散するので、この上界が対照になる。
    """
    cfg = _cfg(ffn_activation="situ_glu")
    mixer = RWKVChannelMix(cfg)
    x = torch.randn(4, cfg.n_embd) * 1e3
    prev = torch.randn(4, cfg.n_embd) * 1e3

    hidden = mixer.ffn_hidden(x, prev).detach()

    assert torch.all(torch.isfinite(hidden))
    assert torch.all(hidden.abs() <= K3_BETA_GATE * K3_BETA_UP)


def test_g2c_squared_relu_hidden_is_not_bounded() -> None:
    """[G2c] 対照: 既定の squared-ReLU は同じ入力で上界を大きく超える."""
    cfg = _cfg()
    mixer = RWKVChannelMix(cfg)
    x = torch.randn(4, cfg.n_embd) * 1e3
    prev = torch.randn(4, cfg.n_embd) * 1e3

    hidden = mixer.ffn_hidden(x, prev).detach()

    assert float(hidden.abs().max()) > K3_BETA_GATE * K3_BETA_UP, "対照不成立"


def test_g2d_config_rejects_unknown_ffn_activation() -> None:
    """[G2d] 未知の活性化名は config 段階で弾く (fail-closed)."""
    with pytest.raises(ValueError, match="ffn_activation"):
        _cfg(ffn_activation="geglu")  # 未実装の名前


# ── G3: 実経路 (LM 全体) で動く ───────────────────────────────────────
def test_g3_full_lm_forward_works_in_situ_glu_mode() -> None:
    """[G3] LM 全体の forward/loss が situ_glu モードで通る (実経路 e2e)."""
    torch.manual_seed(0)
    cfg = _cfg(ffn_activation="situ_glu", n_layer=2)
    model = RWKVLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 4))

    logits, loss = model(idx, targets=idx)

    assert logits.shape == (2, 4, cfg.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "勾配が流れていない"
    assert all(torch.isfinite(g).all() for g in grads), "非有限の勾配が出た"


# ── G4: パラメータ数の統制 (公平比較の前提) ───────────────────────────
def _ffn_params(mixer: RWKVChannelMix) -> int:
    return sum(
        p.numel()
        for n, p in mixer.named_parameters()
        if n.split(".")[0] in ("key", "up", "value")
    )


def test_g4_default_hidden_mult_is_four() -> None:
    """[G4] 既定の hidden 倍率は 4 (RWKV-4 本来の 4x FFN)."""
    cfg = _cfg()
    assert cfg.ffn_hidden_mult == 4.0
    assert RWKVChannelMix(cfg).key.out_features == 4 * cfg.n_embd


def test_g4b_situ_glu_has_more_params_at_the_same_hidden_mult() -> None:
    """[G4b] 同じ hidden 倍率だと SiTU-GLU は 1.5 倍のパラメータを持つ (交絡の実証).

    sq_relu = d*4d + 4d*d = 8d^2 に対し situ_glu は up 射影が増えて 12d^2。
    倍率を揃えたまま比較すると「活性化の差」ではなく「容量の差」を測ってしまう。
    """
    sq = _ffn_params(RWKVChannelMix(_cfg(n_embd=64)))
    glu = _ffn_params(RWKVChannelMix(_cfg(n_embd=64, ffn_activation="situ_glu")))

    assert glu > sq
    # 厳密には bias 項が乗るので 1.5 ちょうどにはならない
    # (d=64, bias=True で 49728/33088 = 1.5029)。
    assert glu / sq == pytest.approx(1.5, rel=1e-2)


def test_g4c_two_thirds_rule_matches_param_count() -> None:
    """[G4c] hidden を 8/3 倍にすると SiTU-GLU の FFN パラメータ数が sq_relu と一致する.

    SwiGLU 論文と同じ 2/3 則: 3 枝 × d × h = 2 枝 × d × 4d → h = 8d/3。
    """
    sq = _ffn_params(RWKVChannelMix(_cfg(n_embd=96)))
    glu = _ffn_params(
        RWKVChannelMix(
            _cfg(n_embd=96, ffn_activation="situ_glu", ffn_hidden_mult=8.0 / 3.0)
        )
    )

    assert glu == pytest.approx(sq, rel=0.01), f"sq={sq} glu={glu}"


def test_g4d_config_rejects_nonpositive_hidden_mult() -> None:
    """[G4d] hidden 倍率 <= 0 は拒否する."""
    with pytest.raises(ValueError, match="ffn_hidden_mult"):
        _cfg(ffn_hidden_mult=0.0)


# ── G5: swiglu (有界性だけを切り分けるための対照) ─────────────────────
def test_g5_swiglu_mode_is_available() -> None:
    """[G5] swiglu を選べる (SiTU-GLU との差が「有界化のみ」になる対照)."""
    cfg = _cfg(ffn_activation="swiglu")
    mixer = RWKVChannelMix(cfg)

    assert hasattr(mixer, "up"), "GLU なのに up 射影が無い"


def test_g5b_swiglu_has_same_param_count_as_situ_glu() -> None:
    """[G5b] swiglu と situ_glu はパラメータ数が同一 (差は活性化だけ)."""
    a = _ffn_params(RWKVChannelMix(_cfg(n_embd=64, ffn_activation="swiglu")))
    b = _ffn_params(RWKVChannelMix(_cfg(n_embd=64, ffn_activation="situ_glu")))

    assert a == b


def test_g5c_swiglu_hidden_is_unbounded_but_situ_glu_is_not() -> None:
    """[G5c] 同じ GLU 構造でも swiglu は非有界、situ_glu は有界 (対照の成立)."""
    torch.manual_seed(0)
    x = torch.randn(4, 8) * 1e3
    prev = torch.randn(4, 8) * 1e3

    sw = RWKVChannelMix(_cfg(ffn_activation="swiglu")).ffn_hidden(x, prev).detach()
    si = RWKVChannelMix(_cfg(ffn_activation="situ_glu")).ffn_hidden(x, prev).detach()

    assert float(sw.abs().max()) > K3_BETA_GATE * K3_BETA_UP, "swiglu が有界に見える"
    assert torch.all(si.abs() <= K3_BETA_GATE * K3_BETA_UP)
