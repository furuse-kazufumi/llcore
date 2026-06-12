# SPDX-License-Identifier: Apache-2.0
"""llcore.clip — CLIP 機能 (text ↔ image 共有埋め込み空間 = 連結性の基盤)。

ユーザー戦略 (2026-06-11)「会話 → アノテーション → 物事の連結性 → 世界モデル」の
連結性ステップを担う。テキストと画像を同一埋め込み空間に写し、類似度 (= 連結の強さ)
を測る。chat レイヤと同じ思想: 既存のオープン重みは製品ではなく**進化型のベース**。

設計 (llcore.chat と同パターン):
- 重い依存 (torch / transformers / pillow) は optional extra ``clip`` に隔離。
  不在時は :class:`ClipDependencyError` で fail-closed。
- default モデル = google/siglip-base-patch16-224 (**Apache-2.0**, CPU 完結)。
  代替 = laion/CLIP-ViT-B-32-laion2B-s34B-b79K (MIT)。openai/clip はライセンスタグ
  不明のため不採用。Qwen 系回避の商用制約に従う。
- 埋め込みは L2 正規化済 — 類似度 = 内積 = cosine。

text-only 検索向けには :class:`SentenceEncoderBackend` (sentence-transformers /
all-MiniLM-L6-v2, Apache-2.0) を差し替えオプションとして提供する。head-to-head 実測
(out/retrieval_head_to_head.json) で短句 retrieval は MiniLM が hard MRR 0.2956 vs
SigLIP 0.1893、encode 0.1s vs 14.6s と優位。cross-modal (text↔image) は引き続き
:class:`ClipBackend` を使う — SentenceEncoderBackend は画像メソッドを持たない。

使い方::

    py -3.11 -m llcore.clip --image photo.jpg --labels "a cat,a dog,a car"
    py -3.11 -m llcore.clip --query "a sleeping cat" --texts "a cat,a dog,an airplane"

    from llcore.clip import ClipBackend
    clip = ClipBackend()
    T = clip.encode_texts(["a cat", "a dog"])      # (2, d) L2 正規化済
    I = clip.encode_images(["photo.jpg"])           # (1, d)
    sim = I @ T.T                                   # cosine 類似度行列
"""
from llcore.clip.annotations import (
    AnnotationStore,
    annotation_id,
    id_cosine,
    id_to_unit_vector,
    is_fact,
    is_question,
    is_request,
    split_annotations,
)
from llcore.clip.backend import (
    DEFAULT_CLIP_MODEL,
    CLIP_MODEL_ENV_VAR,
    ClipBackend,
    ClipDependencyError,
    resolve_clip_model_id,
    zero_shot,
)
from llcore.clip.text_encoders import (
    DEFAULT_TEXT_ENCODER_MODEL,
    TEXT_ENCODER_MODEL_ENV_VAR,
    SentenceEncoderBackend,
    TextEncoderDependencyError,
    resolve_text_encoder_model_id,
)

__all__ = [
    "CLIP_MODEL_ENV_VAR",
    "DEFAULT_CLIP_MODEL",
    "DEFAULT_TEXT_ENCODER_MODEL",
    "TEXT_ENCODER_MODEL_ENV_VAR",
    "AnnotationStore",
    "ClipBackend",
    "ClipDependencyError",
    "SentenceEncoderBackend",
    "TextEncoderDependencyError",
    "annotation_id",
    "id_cosine",
    "id_to_unit_vector",
    "is_fact",
    "is_question",
    "is_request",
    "resolve_clip_model_id",
    "resolve_text_encoder_model_id",
    "split_annotations",
    "zero_shot",
]
