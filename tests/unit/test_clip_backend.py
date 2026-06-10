# SPDX-License-Identifier: Apache-2.0
"""llcore.clip のユニットテスト (実モデル不要 — fake backend でランキング機構を検証)。

実モデル (SigLIP) を使うスモークは scripts/clip_smoke.py で別途実行する。
"""
from __future__ import annotations

import sys
from typing import Any, Sequence

import numpy as np
import pytest

from llcore.clip.backend import (
    CLIP_MODEL_ENV_VAR,
    DEFAULT_CLIP_MODEL,
    ClipBackend,
    ClipDependencyError,
    _l2_normalize,
    resolve_clip_model_id,
    zero_shot,
)


class FakeClipBackend:
    """encode_* を決め打ちベクトルで返すテスト用バックエンド (zero_shot は duck-typing)。"""

    def __init__(self, text_vecs: dict[str, list[float]], image_vec: list[float]) -> None:
        self._text_vecs = text_vecs
        self._image_vec = image_vec
        self.seen_texts: list[str] = []

    def encode_texts(self, texts: Sequence[str]) -> Any:
        self.seen_texts.extend(texts)
        return _l2_normalize(np.array([self._text_vecs[t] for t in texts]))

    def encode_images(self, images: Sequence[Any]) -> Any:
        return _l2_normalize(np.array([self._image_vec for _ in images]))


# -- モデル ID 解決 / 制約 -----------------------------------------------------


def test_resolve_clip_model_id_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CLIP_MODEL_ENV_VAR, raising=False)
    assert resolve_clip_model_id(None) == DEFAULT_CLIP_MODEL
    monkeypatch.setenv(CLIP_MODEL_ENV_VAR, "org/env-clip")
    assert resolve_clip_model_id(None) == "org/env-clip"
    assert resolve_clip_model_id("  org/explicit  ") == "org/explicit"
    assert resolve_clip_model_id("   ") == "org/env-clip"


def test_default_clip_model_is_license_safe() -> None:
    """ライセンス制約: Apache-2.0 の SigLIP を default に (openai/clip はタグ無し→不採用)。"""
    assert "qwen" not in DEFAULT_CLIP_MODEL.lower()
    assert not DEFAULT_CLIP_MODEL.startswith("openai/")
    assert "siglip" in DEFAULT_CLIP_MODEL.lower()


def test_missing_torch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    backend = ClipBackend(model_id="org/whatever")
    with pytest.raises(ClipDependencyError, match=r"llmesh-llcore\[clip\]"):
        backend._ensure_loaded()


# -- 埋め込み / ランキング機構 -------------------------------------------------


def test_l2_normalize_unit_norms() -> None:
    arr = np.array([[3.0, 4.0], [0.0, 0.0], [1.0, 0.0]])
    out = _l2_normalize(arr)
    assert np.allclose(np.linalg.norm(out[0]), 1.0)
    assert np.allclose(np.linalg.norm(out[2]), 1.0)
    assert np.all(np.isfinite(out))  # ゼロベクトルでも発散しない


def test_zero_shot_ranking_and_template() -> None:
    fake = FakeClipBackend(
        text_vecs={
            "a photo of cat": [1.0, 0.0],
            "a photo of dog": [0.7, 0.7],
            "a photo of car": [0.0, 1.0],
        },
        image_vec=[1.0, 0.1],  # cat にほぼ平行
    )
    ranking = zero_shot(fake, object(), ["cat", "dog", "car"], template="a photo of {}")
    assert [lab for lab, _ in ranking] == ["cat", "dog", "car"]
    assert ranking[0][1] > ranking[1][1] > ranking[2][1]
    assert "a photo of cat" in fake.seen_texts  # template が適用されている


def test_zero_shot_empty_labels_rejected() -> None:
    fake = FakeClipBackend(text_vecs={}, image_vec=[1.0, 0.0])
    with pytest.raises(ValueError, match="labels"):
        zero_shot(fake, object(), [])


def test_encode_empty_inputs_rejected() -> None:
    backend = ClipBackend(model_id="org/whatever")
    with pytest.raises(ValueError, match="texts"):
        backend.encode_texts([])
    with pytest.raises(ValueError, match="images"):
        backend.encode_images([])


# -- CLI ----------------------------------------------------------------------


def test_cli_split_csv_and_parser() -> None:
    from llcore.clip.__main__ import _split_csv, build_parser

    assert _split_csv("a cat, a dog ,,a car") == ["a cat", "a dog", "a car"]
    args = build_parser().parse_args(["--image", "x.jpg", "--labels", "a,b"])
    assert args.image == "x.jpg"
    assert args.labels == "a,b"
