# SPDX-License-Identifier: Apache-2.0
"""SentenceEncoderBackend のユニットテスト — モデル解決 / fail-closed / lazy load / 正規化を検証。"""
from __future__ import annotations

import sys
import time
import types
from typing import Any, Sequence

import numpy as np
import pytest

from llcore.clip.text_encoders import (
    DEFAULT_TEXT_ENCODER_MODEL,
    TEXT_ENCODER_MODEL_ENV_VAR,
    SentenceEncoderBackend,
    TextEncoderDependencyError,
    resolve_text_encoder_model_id,
)


# -- モデル ID 解決 (明示 > env > default) ---------------------------------------


def test_resolve_model_id_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(TEXT_ENCODER_MODEL_ENV_VAR, raising=False)
    assert resolve_text_encoder_model_id() == DEFAULT_TEXT_ENCODER_MODEL
    assert resolve_text_encoder_model_id(None) == DEFAULT_TEXT_ENCODER_MODEL
    assert resolve_text_encoder_model_id("  ") == DEFAULT_TEXT_ENCODER_MODEL  # 空白のみは default


def test_resolve_model_id_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TEXT_ENCODER_MODEL_ENV_VAR, "org/env-model")
    assert resolve_text_encoder_model_id() == "org/env-model"


def test_resolve_model_id_explicit_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(TEXT_ENCODER_MODEL_ENV_VAR, "org/env-model")
    assert resolve_text_encoder_model_id("org/explicit-model") == "org/explicit-model"
    assert resolve_text_encoder_model_id(" org/explicit-model ") == "org/explicit-model"


# -- fail-closed (sentence-transformers 不在) ------------------------------------


def test_dependency_error_when_sentence_transformers_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """sentence-transformers 不在時は TextEncoderDependencyError で fail-closed。"""
    # sys.modules に None を入れると import sentence_transformers が ImportError になる
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    backend = SentenceEncoderBackend()
    with pytest.raises(TextEncoderDependencyError, match="pip install sentence-transformers"):
        backend.encode_texts(["hello"])


# -- 入力検証 --------------------------------------------------------------------


def test_empty_texts_fail_closed() -> None:
    backend = SentenceEncoderBackend()
    with pytest.raises(ValueError, match="empty"):
        backend.encode_texts([])


# -- fake SentenceTransformer (lazy load / shape / dtype / 正規化) ----------------


class _FakeSentenceTransformer:
    """SentenceTransformer の fake — 故意に非正規化ベクトルを返す (強制正規化の検証用)。"""

    instances: list["_FakeSentenceTransformer"] = []

    def __init__(self, model_id: str) -> None:
        self.model_id = model_id
        _FakeSentenceTransformer.instances.append(self)

    def encode(
        self,
        texts: Sequence[str],
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
    ) -> Any:
        # normalize_embeddings=True が渡されても非正規化を返す
        # (数値ドリフトの想定 — backend 側の自前正規化が吸収すること)
        rng = np.random.default_rng(len(texts))
        return rng.normal(size=(len(texts), 6)).astype(np.float64) * 7.5


@pytest.fixture()
def fake_st_module(monkeypatch: pytest.MonkeyPatch) -> type[_FakeSentenceTransformer]:
    _FakeSentenceTransformer.instances = []
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = _FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    return _FakeSentenceTransformer


def test_lazy_load_until_first_encode(fake_st_module: type[_FakeSentenceTransformer]) -> None:
    """init 時点ではモデル未ロード (lazy)。初回 encode で 1 度だけロード。"""
    backend = SentenceEncoderBackend("org/fake-model")
    assert fake_st_module.instances == []  # init ではロードしない
    assert backend.load_seconds is None
    backend.encode_texts(["a"])
    assert len(fake_st_module.instances) == 1
    assert fake_st_module.instances[0].model_id == "org/fake-model"
    assert backend.load_seconds is not None
    backend.encode_texts(["b", "c"])
    assert len(fake_st_module.instances) == 1  # 再ロードしない


def test_encode_texts_shape_dtype_and_l2_norm(
    fake_st_module: type[_FakeSentenceTransformer],
) -> None:
    """戻りは (n, d) float32、fake が非正規化を返しても自前で L2 正規化を強制。"""
    backend = SentenceEncoderBackend("org/fake-model")
    out = backend.encode_texts(["alpha", "beta", "gamma"])
    assert out.shape == (3, 6)
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)


# -- 実モデル smoke (ローカルキャッシュ前提, ネットワーク遮断) --------------------


def test_real_minilm_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    """all-MiniLM-L6-v2 実ロード smoke — HF へのネットワークアクセスは遮断 (offline)。"""
    # env は import より先に設定 (huggingface_hub は import 時に定数へ読み込む)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    pytest.importorskip("sentence_transformers")
    # 既に import 済みのプロセスでも offline を強制 (定数を直接上書き)
    hf_constants = pytest.importorskip("huggingface_hub.constants")
    monkeypatch.setattr(hf_constants, "HF_HUB_OFFLINE", True, raising=False)
    monkeypatch.delenv(TEXT_ENCODER_MODEL_ENV_VAR, raising=False)
    backend = SentenceEncoderBackend()
    assert backend.model_id == DEFAULT_TEXT_ENCODER_MODEL
    t0 = time.time()
    out = backend.encode_texts(["my name is kazufumi", "i live in japan"])
    encode_seconds = time.time() - t0
    assert out.shape == (2, 384)  # all-MiniLM-L6-v2 の次元 = 384
    assert out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    assert backend.load_seconds is not None and backend.load_seconds > 0
    print(
        f"\n[smoke] load={backend.load_seconds:.2f}s "
        f"encode(2 texts, ロード込み計測外)={encode_seconds - backend.load_seconds:.3f}s"
    )
