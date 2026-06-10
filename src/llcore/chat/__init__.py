# SPDX-License-Identifier: Apache-2.0
"""llcore.chat — 実在小型 LLM (SmolLM2-Instruct) ベースの基本会話レイヤ。

EVOLVABLE_LLM_PLAN_2026_06_09 Phase 0「実在の小型オープン LLM を base に据え
baseline 機能 (coherent text / 基本 Q&A) を継承」を製品コードとして固定するモジュール。
研究ハーネス (research/rllm_pivot/phase0_framework_harness.py) で実証済みの
SmolLM2 CPU ロードパターンを、複数ターン会話セッションとして再利用可能にする。

設計:
- 重い依存 (torch / transformers) は optional extra ``chat`` に隔離。
  不在時は :class:`ChatDependencyError` で fail-closed (黙って劣化しない)。
- バックエンドは :class:`ChatBackend` Protocol で注入可能 — テストは FakeBackend、
  実会話は :class:`TransformersBackend`。
- ベースモデルは Apache-2.0 の SmolLM2-Instruct (Qwen 回避の商用ライセンス制約に従う)。

使い方::

    py -3.11 -m llcore.chat                 # 対話 REPL
    py -3.11 -m llcore.chat --prompt "Hi!"  # ワンショット

    from llcore.chat import ChatSession, TransformersBackend
    session = ChatSession(TransformersBackend())
    print(session.ask("What is the capital of France?"))
"""
from llcore.chat.backend import (
    DEFAULT_MODEL,
    MODEL_ENV_VAR,
    ChatDependencyError,
    TransformersBackend,
    resolve_model_id,
)
from llcore.chat.session import (
    DEFAULT_SYSTEM_PROMPT,
    ChatBackend,
    ChatSession,
    GenerationSettings,
    Message,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_SYSTEM_PROMPT",
    "MODEL_ENV_VAR",
    "ChatBackend",
    "ChatDependencyError",
    "ChatSession",
    "GenerationSettings",
    "Message",
    "TransformersBackend",
    "resolve_model_id",
]
