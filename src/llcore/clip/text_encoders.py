# SPDX-License-Identifier: Apache-2.0
"""llcore.clip.text_encoders — sentence-transformers ベースの text-only エンコーダ。

ROADMAP M1「encoder 差し替えオプション」: AnnotationStore は
``encode_texts(Sequence[str]) -> (n, d) numpy`` を持つ任意のエンコーダを受けるため、
text-only 検索向けに MiniLM backend を default の :class:`~llcore.clip.backend.ClipBackend`
(SigLIP) の代替として提供する。

honest 開示 (head-to-head 実測, out/retrieval_head_to_head.json):
- 短句 text-only retrieval は MiniLM が優位 (hard MRR 0.2956 vs SigLIP 0.1893、
  encode 0.1s vs 14.6s)。
- 本 backend は **text-only 専用**。cross-modal (画像との連結 = text↔image 共有埋め込み)
  には使えない — その用途は :class:`~llcore.clip.backend.ClipBackend` を使うこと。

設計 (backend.py と同パターン):
- 重い依存 (sentence-transformers) は optional extra ``text`` に隔離。
  不在時は :class:`TextEncoderDependencyError` で fail-closed (黙って劣化しない)。
- ロードは初回 encode まで遅延 (lazy load)。ロード時間は ``load_seconds`` に記録。
- 埋め込みは L2 正規化済 float32 — 類似度 = 内積 = cosine。
"""
from __future__ import annotations

import os
import time
from typing import Any, Sequence

# default = all-MiniLM-L6-v2 (Apache-2.0 — FullSense の dual-license と整合)。
# sentence-transformers ライブラリ自体も Apache-2.0。
DEFAULT_TEXT_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT_ENCODER_MODEL_ENV_VAR = "LLCORE_TEXT_ENCODER_MODEL"


class TextEncoderDependencyError(RuntimeError):
    """optional extra ``text`` (sentence-transformers) が未インストール。"""


def _import_sentence_transformer() -> Any:
    """sentence_transformers.SentenceTransformer を lazy import。不在なら fail-closed。"""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise TextEncoderDependencyError(
            "llcore.clip.text_encoders には sentence-transformers が必要です: "
            "pip install sentence-transformers "
            '(または pip install "llmesh-llcore[text]")'
        ) from exc
    return SentenceTransformer


def resolve_text_encoder_model_id(explicit: str | None = None) -> str:
    """モデル ID を解決する。優先順: 明示引数 > env LLCORE_TEXT_ENCODER_MODEL > default。"""
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get(TEXT_ENCODER_MODEL_ENV_VAR, "").strip()
    if env:
        return env
    return DEFAULT_TEXT_ENCODER_MODEL


class SentenceEncoderBackend:
    """sentence-transformers ベースの text-only 埋め込みバックエンド (CPU, lazy load)。

    AnnotationStore の encoder protocol (``encode_texts``) を満たす。
    **text-only 専用** — 画像メソッド (``encode_images``) は持たない。
    cross-modal (text↔image) には :class:`~llcore.clip.backend.ClipBackend` を使うこと。

    Args:
        model_id: HF モデル ID。None なら :func:`resolve_text_encoder_model_id` で解決。

    ロードは初回 encode まで遅延する。ロード時間は ``load_seconds`` に記録。
    """

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = resolve_text_encoder_model_id(model_id)
        self._model: Any = None
        self.load_seconds: float | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        sentence_transformer_cls = _import_sentence_transformer()
        t0 = time.time()
        model = sentence_transformer_cls(self.model_id)
        self.load_seconds = time.time() - t0
        self._model = model

    # -- 埋め込み -------------------------------------------------------------

    def encode_texts(self, texts: Sequence[str]) -> Any:
        """テキスト列を L2 正規化済み float32 埋め込み (n, d) numpy 配列にする。

        ``normalize_embeddings=True`` で正規化済みを受けつつ、ClipBackend と同様に
        自前でも L2 正規化を強制する (モデル/版差の数値ドリフト吸収)。
        """
        if not texts:
            raise ValueError("texts must not be empty")
        self._ensure_loaded()
        vecs = self._model.encode(
            list(texts), normalize_embeddings=True, convert_to_numpy=True
        )
        return _l2_normalize_f32(vecs)


def _l2_normalize_f32(arr: Any) -> Any:
    """(n, d) 配列を行ごとに L2 正規化し float32 で返す (零ベクトルは fail-soft)。"""
    import numpy as np

    a = np.asarray(arr, dtype=np.float64)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    return (a / np.maximum(norms, 1e-12)).astype(np.float32)
