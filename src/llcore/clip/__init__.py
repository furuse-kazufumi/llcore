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

使い方::

    py -3.11 -m llcore.clip --image photo.jpg --labels "a cat,a dog,a car"
    py -3.11 -m llcore.clip --query "a sleeping cat" --texts "a cat,a dog,an airplane"

    from llcore.clip import ClipBackend
    clip = ClipBackend()
    T = clip.encode_texts(["a cat", "a dog"])      # (2, d) L2 正規化済
    I = clip.encode_images(["photo.jpg"])           # (1, d)
    sim = I @ T.T                                   # cosine 類似度行列
"""
from llcore.clip.annotations import AnnotationStore, annotation_id, split_annotations
from llcore.clip.backend import (
    DEFAULT_CLIP_MODEL,
    CLIP_MODEL_ENV_VAR,
    ClipBackend,
    ClipDependencyError,
    resolve_clip_model_id,
    zero_shot,
)

__all__ = [
    "CLIP_MODEL_ENV_VAR",
    "DEFAULT_CLIP_MODEL",
    "AnnotationStore",
    "ClipBackend",
    "ClipDependencyError",
    "annotation_id",
    "resolve_clip_model_id",
    "split_annotations",
    "zero_shot",
]
