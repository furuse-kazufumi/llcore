# SPDX-License-Identifier: Apache-2.0
"""llcore.chat.session — 複数ターン会話セッション (履歴管理 + 段階的話題遷移の土台)。

このモジュールは重い依存 (torch / transformers) を一切 import しない。
バックエンドは :class:`ChatBackend` Protocol として注入する — 実モデルは
:mod:`llcore.chat.backend` の ``TransformersBackend``、テストは FakeBackend。

履歴の不変条件:
- 先頭は system message (system_prompt が空文字なら無し)。
- 以降は user / assistant が交互。``ask()`` が失敗した場合、当該 user turn は
  履歴に残さない (ロールバック) — 壊れた履歴で次ターンを汚さない fail-closed 設計。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, concise assistant. "
    "Answer the user's questions directly and briefly."
)


@dataclass(frozen=True)
class Message:
    """会話の 1 メッセージ。role は "system" | "user" | "assistant"。"""

    role: str
    content: str

    def as_dict(self) -> dict[str, str]:
        """transformers の chat template が期待する dict 形式に変換する。"""
        return {"role": self.role, "content": self.content}


@dataclass(frozen=True)
class GenerationSettings:
    """生成パラメータ。小型 instruct モデル向けの保守的 default。

    do_sample=False (greedy) は決定論的だが小型モデルでは反復ループしやすい。
    default は低温サンプリング + 軽い repetition_penalty。

    構築時に fail-closed 検証する (temperature=0 等は transformers の generate 実行時
    まで検出が遅延するため、ここで早期に拒否する)。
    """

    max_new_tokens: int = 256
    temperature: float = 0.3
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError(f"max_new_tokens must be > 0, got {self.max_new_tokens}")
        if self.do_sample and self.temperature <= 0.0:
            raise ValueError(
                f"temperature must be > 0 when do_sample=True, got {self.temperature}; "
                "決定論的デコードには do_sample=False (--greedy) を使う"
            )
        if self.do_sample and not (0.0 < self.top_p <= 1.0):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p}")


class ChatBackend(Protocol):
    """応答生成バックエンドの Protocol (構造的型付け — 継承不要)。"""

    def generate(self, messages: Sequence[Message], settings: GenerationSettings) -> str:
        """履歴全体 (system 含む、末尾は user) に対する assistant 応答テキストを返す。"""
        ...


class ChatSession:
    """複数ターン会話セッション。

    会話履歴を保持し、文脈引継ぎ (前ターンへの言及) と段階的な話題遷移を
    バックエンド非依存に支える。

    Args:
        backend: 応答生成バックエンド (:class:`ChatBackend` 互換)。
        system_prompt: system message。空文字なら system 無しで開始。
        settings: 生成パラメータ。省略時は :class:`GenerationSettings` の default。
    """

    def __init__(
        self,
        backend: ChatBackend,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        settings: GenerationSettings | None = None,
    ) -> None:
        self._backend = backend
        self._system_prompt = system_prompt
        self.settings = settings if settings is not None else GenerationSettings()
        self._history: list[Message] = []
        if system_prompt:
            self._history.append(Message("system", system_prompt))

    @property
    def history(self) -> tuple[Message, ...]:
        """現在の会話履歴 (読み取り専用ビュー)。"""
        return tuple(self._history)

    @property
    def turn_count(self) -> int:
        """完了した user→assistant 交換の数。"""
        return sum(1 for m in self._history if m.role == "assistant")

    def ask(self, user_text: str) -> str:
        """user メッセージを 1 ターン送り、assistant 応答を返す。

        Raises:
            ValueError: user_text が空白のみの場合 (fail-closed)。
            Exception: バックエンド失敗はそのまま伝播する。その場合、
                当該 user turn は履歴からロールバックされる。
        """
        text = user_text.strip()
        if not text:
            raise ValueError("user message must not be empty")
        self._history.append(Message("user", text))
        try:
            reply = self._backend.generate(tuple(self._history), self.settings)
            if not isinstance(reply, str):
                raise TypeError(
                    f"ChatBackend.generate must return str, got {type(reply).__name__}"
                )
            reply = reply.strip()
            if not reply:
                # 空応答も fail-closed (空 user の拒否と対称)。履歴を汚さない。
                raise ValueError("backend returned an empty reply")
        except BaseException:
            self._history.pop()
            raise
        self._history.append(Message("assistant", reply))
        return reply

    def reset(self) -> None:
        """履歴を初期状態 (system message のみ、無ければ空) に戻す。"""
        self._history.clear()
        if self._system_prompt:
            self._history.append(Message("system", self._system_prompt))
