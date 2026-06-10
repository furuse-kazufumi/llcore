# SPDX-License-Identifier: Apache-2.0
"""llcore.clip.backend — transformers ベースの CLIP/SigLIP 埋め込みバックエンド (CPU 完結)。

CLIPModel / SiglipModel が共通に持つ ``get_text_features`` / ``get_image_features`` に
対して実装するため、モデル系列に依存しない。埋め込みは L2 正規化して返す
(類似度 = 内積 = cosine; モデル固有の logit scale / bias には依存しない — 順位は
単調変換で不変。較正確率が必要な場合はモデル固有の式を使うこと、と開示する)。
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence, Union

if TYPE_CHECKING:  # pragma: no cover - 型チェック時のみ
    from types import ModuleType

    from PIL.Image import Image as PILImage

    ImageLike = Union[str, Path, PILImage]
else:
    ImageLike = Any

# default = SigLIP (Apache-2.0, FullSense の dual-license と整合)。
# 代替 = laion/CLIP-ViT-B-32-laion2B-s34B-b79K (MIT)。openai/clip はライセンスタグ無しのため不採用。
DEFAULT_CLIP_MODEL = "google/siglip-base-patch16-224"
CLIP_MODEL_ENV_VAR = "LLCORE_CLIP_MODEL"


class ClipDependencyError(RuntimeError):
    """optional extra ``clip`` (torch / transformers / pillow) が未インストール。"""


def _import_deps() -> tuple["ModuleType", Any, Any, Any]:
    """torch / transformers / PIL を lazy import する。不在なら fail-closed。"""
    try:
        import torch
        from PIL import Image
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:
        raise ClipDependencyError(
            "llcore.clip には optional extra 'clip' が必要です: "
            'pip install "llmesh-llcore[clip]" '
            "(torch / transformers / pillow がインストールされます)"
        ) from exc
    return torch, AutoProcessor, AutoModel, Image


def resolve_clip_model_id(explicit: str | None = None) -> str:
    """モデル ID を解決する。優先順: 明示引数 > env LLCORE_CLIP_MODEL > default。"""
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get(CLIP_MODEL_ENV_VAR, "").strip()
    if env:
        return env
    return DEFAULT_CLIP_MODEL


class ClipBackend:
    """CLIP/SigLIP 共有埋め込みバックエンド (CPU, frozen, lazy load)。

    Args:
        model_id: HF モデル ID。None なら :func:`resolve_clip_model_id` で解決。

    ロードは初回 encode まで遅延する。ロード時間は ``load_seconds`` に記録。
    """

    def __init__(self, model_id: str | None = None) -> None:
        self.model_id = resolve_clip_model_id(model_id)
        self._torch: ModuleType | None = None
        self._pil_image: Any = None
        self._processor: Any = None
        self._model: Any = None
        self.load_seconds: float | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch, auto_processor, auto_model, pil_image = _import_deps()
        t0 = time.time()
        processor = auto_processor.from_pretrained(self.model_id)
        model = auto_model.from_pretrained(self.model_id)
        model.eval()
        for name in ("get_text_features", "get_image_features"):
            if not hasattr(model, name):
                raise ClipDependencyError(
                    f"{self.model_id} は {name} を持たず CLIP 機能に使えません "
                    "(CLIP/SigLIP 系のモデル ID を指定してください)"
                )
        self.load_seconds = time.time() - t0
        self._torch = torch
        self._pil_image = pil_image
        self._processor = processor
        self._model = model

    # -- 埋め込み -------------------------------------------------------------

    def encode_texts(self, texts: Sequence[str]) -> Any:
        """テキスト列を L2 正規化済み埋め込み (n, d) numpy 配列にする。"""
        if not texts:
            raise ValueError("texts must not be empty")
        self._ensure_loaded()
        assert self._torch is not None
        # SigLIP は max_length padding で学習されているため固定 (CLIP でも有効)
        inputs = self._processor(
            text=list(texts), padding="max_length", truncation=True, return_tensors="pt"
        )
        with self._torch.no_grad():
            feats = self._model.get_text_features(**inputs)
        return _l2_normalize(self._feature_tensor(feats).numpy())

    def encode_images(self, images: Sequence[ImageLike]) -> Any:
        """画像列 (パス or PIL Image) を L2 正規化済み埋め込み (n, d) にする。"""
        if not images:
            raise ValueError("images must not be empty")
        self._ensure_loaded()
        assert self._torch is not None
        pil_images = [self._to_pil(im) for im in images]
        inputs = self._processor(images=pil_images, return_tensors="pt")
        with self._torch.no_grad():
            feats = self._model.get_image_features(**inputs)
        return _l2_normalize(self._feature_tensor(feats).numpy())

    def _feature_tensor(self, out: Any) -> Any:
        """get_*_features の戻りを tensor に正規化する。

        transformers のバージョン/モデル系列により tensor 直 (CLIP 4.x) と
        出力オブジェクト (SigLIP 5.x = BaseModelOutputWithPooling) の両方がある。
        """
        assert self._torch is not None
        if self._torch.is_tensor(out):
            return out
        for attr in ("text_embeds", "image_embeds", "pooler_output"):
            v = getattr(out, attr, None)
            if v is not None and self._torch.is_tensor(v):
                return v
        raise ClipDependencyError(
            f"{self.model_id} の get_*_features 戻り値から特徴 tensor を取り出せません "
            f"(型: {type(out).__name__})"
        )

    def _to_pil(self, im: ImageLike) -> Any:
        if isinstance(im, (str, Path)):
            path = Path(im)
            if not path.exists():
                raise FileNotFoundError(f"image not found: {path}")
            return self._pil_image.open(path).convert("RGB")
        return im.convert("RGB") if hasattr(im, "convert") else im


def _l2_normalize(arr: Any) -> Any:
    import numpy as np

    a = np.asarray(arr, dtype=np.float64)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    return a / np.maximum(norms, 1e-12)


def zero_shot(
    backend: ClipBackend,
    image: ImageLike,
    labels: Sequence[str],
    template: str = "a photo of {}",
) -> list[tuple[str, float]]:
    """zero-shot 分類: 画像とラベル文の cosine 類似度で降順ランキング。

    返り値の score は cosine 類似度 (較正確率ではない — 順位付けヒューリスティック)。
    """
    if not labels:
        raise ValueError("labels must not be empty")
    texts = [template.format(lab) for lab in labels]
    T = backend.encode_texts(texts)
    I_ = backend.encode_images([image])
    sims = (I_ @ T.T)[0]
    order = sims.argsort()[::-1]
    return [(labels[int(i)], float(sims[int(i)])) for i in order]
