# SPDX-License-Identifier: Apache-2.0
"""Tests for NativeQwenBackend — llcore 自前 forward を ChatBackend 化したもの。

実モデルロードは重いので、ここでは (1) lazy 性 (構築ではロードしない) (2) ChatSession への
注入互換 (3) fail-closed health (4) nucleus サンプリングの純ロジック を検証する。
HF transformers との出力一致は scripts/prove_native_matches_hf.py が実モデルで実演する (別経路)。
"""
from __future__ import annotations

import torch

from llcore.chat.native_backend import NativeQwenBackend
from llcore.chat.session import ChatSession, GenerationSettings


def test_init_does_not_load() -> None:
    """構築だけではモデルをロードしない (lazy)。"""
    b = NativeQwenBackend("D:/models/does-not-exist")
    assert b._model is None
    assert b.load_seconds is None


def test_injectable_into_chatsession_without_load() -> None:
    """ChatBackend Protocol 互換: ChatSession に注入でき、ask しなければロードしない。"""
    b = NativeQwenBackend("D:/models/whatever")
    session = ChatSession(b, system_prompt="sys")
    assert session.turn_count == 0
    assert b._model is None


def test_health_false_on_missing_model_dir() -> None:
    """存在しないモデルディレクトリ → fail-closed で False (例外を投げない)。"""
    b = NativeQwenBackend("D:/models/definitely-not-here-xyz-123")
    assert b.health() is False


def test_sample_next_top_p_one_returns_valid_token() -> None:
    """top_p=1.0 のサンプリングが語彙内の妥当なトークンを返す。"""
    b = NativeQwenBackend("x")
    b._torch = torch  # _sample_next が使う torch を注入 (ロード回避)
    logits = torch.tensor([0.1, 2.0, 0.5, -1.0, 3.0])
    s = GenerationSettings(max_new_tokens=1, do_sample=True, temperature=1.0, top_p=1.0)
    torch.manual_seed(0)
    tok = b._sample_next(logits, s)
    assert 0 <= tok < int(logits.numel())


def test_sample_next_tiny_top_p_collapses_to_argmax() -> None:
    """top_p を極小にすると最頻トークン (argmax) のみ残り、決定論的になる。"""
    b = NativeQwenBackend("x")
    b._torch = torch
    logits = torch.tensor([0.1, 2.0, 0.5, -1.0, 3.0])  # argmax = index 4
    s = GenerationSettings(max_new_tokens=1, do_sample=True, temperature=1.0, top_p=1e-6)
    for seed in range(5):
        torch.manual_seed(seed)
        assert b._sample_next(logits, s) == 4
