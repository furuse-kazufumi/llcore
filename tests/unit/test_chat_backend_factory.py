# SPDX-License-Identifier: Apache-2.0
"""Tests for build_backend — chat バックエンド factory。

``--native``/``--model`` の組から会話バックエンドを選ぶ純関数。HF transformers
ラッパー (:class:`TransformersBackend`) と llcore 自前 forward
(:class:`NativeQwenBackend`) を一箇所で選択し、``llcore.chat`` CLI と
``scripts/chat_endurance_probe.py`` の両方が同じ規約 (native=ローカルディレクトリ) を
共有できるようにする。構築は lazy なのでここでは torch/モデルロードを伴わない。
"""
from __future__ import annotations

from llcore.chat.__main__ import build_backend
from llcore.chat.backend import TransformersBackend
from llcore.chat.native_backend import NativeQwenBackend


def test_native_true_builds_native_qwen_backend() -> None:
    """native=True で NativeQwenBackend を返し、model はローカルディレクトリとして渡る。"""
    backend, label = build_backend(
        model="D:/models/Qwen2.5-0.5B-Instruct", native=True, seed=7, int8=False
    )
    assert isinstance(backend, NativeQwenBackend)
    assert backend.model_dir == "D:/models/Qwen2.5-0.5B-Instruct"
    assert label == "D:/models/Qwen2.5-0.5B-Instruct"


def test_native_int8_flag_propagates() -> None:
    """--int8 が NativeQwenBackend の int8 ロード指定に伝播する。"""
    backend, _ = build_backend(model="D:/m", native=True, seed=None, int8=True)
    assert isinstance(backend, NativeQwenBackend)
    assert backend.int8 is True


def test_native_false_builds_transformers_backend() -> None:
    """native=False で HF transformers ラッパー (TransformersBackend) を返す。"""
    backend, label = build_backend(model=None, native=False, seed=3, int8=False)
    assert isinstance(backend, TransformersBackend)
    # native でない場合 label は resolve_model_id された HF モデル ID
    assert label == backend.model_id


def test_native_false_ignores_int8() -> None:
    """int8 は native 専用 — HF 経路では TransformersBackend が選ばれるだけ。"""
    backend, _ = build_backend(model="HF/x", native=False, seed=None, int8=True)
    assert isinstance(backend, TransformersBackend)
