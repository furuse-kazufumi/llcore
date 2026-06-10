# SPDX-License-Identifier: Apache-2.0
"""llcore.chat.backend — transformers ベースの実モデル会話バックエンド (CPU 完結)。

research/rllm_pivot/phase0_framework_harness.py で実証済みの SmolLM2 CPU ロード
パターンを会話用に製品化したもの。重い依存 (torch / transformers) はこのモジュール
内で lazy import し、不在時は :class:`ChatDependencyError` で fail-closed に拒否する
(黙って mock に劣化しない)。

ベースモデル選定の制約 (memory feedback_qwen_commercial_barrier / 計画 §Phase0):
- Apache-2.0 / MIT 系のみ (FullSense の Apache-2.0 + Commercial dual-license と整合)。
- Qwen 系は商用障壁のため避ける。
- default = SmolLM2-360M-Instruct (Apache-2.0, CPU で数秒ロード, 段階的会話 5/5)。
  env ``LLCORE_CHAT_MODEL`` または引数で SmolLM2-135M-Instruct (軽量) 等へ差し替え可。
"""
from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, Any, Sequence

from llcore.chat.session import GenerationSettings, Message

if TYPE_CHECKING:  # pragma: no cover - 型チェック時のみ (実行時は lazy import)
    from types import ModuleType

# default は 360M: 段階的会話スモークで 5/5 (135M は名前想起=文脈引継ぎで 4/5)。
# 軽さ優先なら --model / env で SmolLM2-135M-Instruct を指定。
DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
MODEL_ENV_VAR = "LLCORE_CHAT_MODEL"

# 生成予算を context 窓に収める際の安全マージン (template 特殊トークン等の揺らぎ吸収)
_CONTEXT_SAFETY_MARGIN = 16
# モデル config から context 長が取れない場合の保守的 fallback
_FALLBACK_CONTEXT_TOKENS = 2048


class ChatDependencyError(RuntimeError):
    """optional extra ``chat`` (torch / transformers) が未インストール。"""


def _import_transformers() -> tuple["ModuleType", Any, Any]:
    """torch / transformers を lazy import する。不在なら fail-closed。"""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ChatDependencyError(
            "llcore.chat の実モデル会話には optional extra 'chat' が必要です: "
            'pip install "llmesh-llcore[chat]" '
            "(torch / transformers がインストールされます)"
        ) from exc
    return torch, AutoTokenizer, AutoModelForCausalLM


def resolve_model_id(explicit: str | None = None) -> str:
    """モデル ID を解決する。優先順: 明示引数 > env LLCORE_CHAT_MODEL > default。

    明示引数・env とも空白を strip し、空白のみは「未指定」として次順位へ落とす。
    """
    if explicit and explicit.strip():
        return explicit.strip()
    env = os.environ.get(MODEL_ENV_VAR, "").strip()
    if env:
        return env
    return DEFAULT_MODEL


class TransformersBackend:
    """transformers AutoModelForCausalLM を用いる会話バックエンド (CPU, frozen)。

    Args:
        model_id: HF モデル ID。None なら :func:`resolve_model_id` で解決。
        seed: 再現性用。指定時は generate 毎に torch.manual_seed を打つ。
        max_context_tokens: prompt + 生成の合計トークン予算。None なら
            モデル config の max_position_embeddings (取れなければ 2048)。

    ロードは初回 generate まで遅延する (lazy)。ロード時間は ``load_seconds`` に記録。
    """

    def __init__(
        self,
        model_id: str | None = None,
        seed: int | None = None,
        max_context_tokens: int | None = None,
    ) -> None:
        self.model_id = resolve_model_id(model_id)
        self._seed = seed
        self._max_context_tokens = max_context_tokens
        self._torch: ModuleType | None = None
        self._tok: Any = None
        self._model: Any = None
        self.load_seconds: float | None = None

    # -- ロード -------------------------------------------------------------

    def ensure_loaded(self) -> None:
        """モデル/トークナイザを明示的にロードする (research ハーネス向け公開 API)。"""
        self._ensure_loaded()

    @property
    def tokenizer(self) -> Any:
        """ロード済みトークナイザ (未ロードならロードを誘発)。"""
        self._ensure_loaded()
        return self._tok

    @property
    def model(self) -> Any:
        """ロード済みモデル (未ロードならロードを誘発)。research 用途 (hidden 抽出等)。"""
        self._ensure_loaded()
        return self._model

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        torch, auto_tokenizer, auto_model = _import_transformers()
        t0 = time.time()
        tok = auto_tokenizer.from_pretrained(self.model_id)
        try:
            model = auto_model.from_pretrained(self.model_id, dtype=torch.float32)
        except TypeError:
            # transformers<5 は dtype= 未対応 (torch_dtype=)。後方互換 fallback。
            model = auto_model.from_pretrained(self.model_id, torch_dtype=torch.float32)
        model.eval()
        self.load_seconds = time.time() - t0
        self._torch = torch
        self._tok = tok
        self._model = model
        if self._max_context_tokens is None:
            cfg_max = getattr(model.config, "max_position_embeddings", None)
            self._max_context_tokens = (
                int(cfg_max) if cfg_max else _FALLBACK_CONTEXT_TOKENS
            )

    # -- 履歴の予算内トリミング ----------------------------------------------

    def _templated_token_count(self, msgs: list[dict[str, str]]) -> int:
        text = self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        return len(self._tok(text, add_special_tokens=False)["input_ids"])

    def _trim_to_budget(
        self, msgs: list[dict[str, str]], max_new_tokens: int
    ) -> list[dict[str, str]]:
        """prompt が context 予算を超える場合、古いターンから落とす。

        system (先頭) と末尾の user は必ず残す。user/assistant の対単位で
        落とし、交互順を崩さない。それでも収まらない場合は ValueError
        (fail-closed — 黙って末尾を切り詰めない)。
        """
        assert self._max_context_tokens is not None
        budget = self._max_context_tokens - max_new_tokens - _CONTEXT_SAFETY_MARGIN
        if budget <= 0:
            raise ValueError(
                f"max_new_tokens={max_new_tokens} が context 予算 "
                f"{self._max_context_tokens} に対して大きすぎます"
            )
        out = list(msgs)
        head = 1 if out and out[0]["role"] == "system" else 0
        while self._templated_token_count(out) > budget:
            # head の直後 = 最古の非 system ターン。末尾 user だけは残す。
            if len(out) - head <= 1:
                # これ以上トリムできない = system prompt + 最新 user の固定部分だけで
                # 予算超過 (どちらが主因かはケースによる — 帰責を断定しない)
                raise ValueError(
                    "履歴をこれ以上トリムできませんが context 予算を超えています "
                    f"(budget={budget} tokens; system prompt または最新 user メッセージが"
                    "大きすぎる、あるいは max_new_tokens が大きすぎます)"
                )
            if (
                len(out) - head >= 3
                and out[head]["role"] == "user"
                and out[head + 1]["role"] == "assistant"
            ):
                del out[head : head + 2]  # 対で落とす (交互順維持)
            else:
                del out[head]
        return out

    # -- 生成 ---------------------------------------------------------------

    def generate(self, messages: Sequence[Message], settings: GenerationSettings) -> str:
        """履歴に対する assistant 応答を生成して返す (ChatBackend Protocol 実装)。"""
        self._ensure_loaded()
        assert self._torch is not None
        if self._seed is not None:
            self._torch.manual_seed(self._seed)

        msgs = self._trim_to_budget(
            [m.as_dict() for m in messages], settings.max_new_tokens
        )
        prompt_text = self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
        # template が特殊トークンを含むため add_special_tokens=False (BOS 二重付与防止)
        inputs = self._tok(prompt_text, return_tensors="pt", add_special_tokens=False)

        pad_id = self._tok.pad_token_id
        if pad_id is None:
            pad_id = self._tok.eos_token_id
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": settings.max_new_tokens,
            "pad_token_id": pad_id,
        }
        if settings.do_sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=settings.temperature,
                top_p=settings.top_p,
            )
        else:
            gen_kwargs["do_sample"] = False
        if settings.repetition_penalty != 1.0:
            gen_kwargs["repetition_penalty"] = settings.repetition_penalty

        with self._torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0, inputs["input_ids"].shape[1] :]
        return str(self._tok.decode(new_tokens, skip_special_tokens=True))
