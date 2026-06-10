# SPDX-License-Identifier: Apache-2.0
"""llcore.chat のユニットテスト (実モデル不要 — FakeBackend で履歴/CLI 機構を検証)。

実モデル (SmolLM2-Instruct) を使う段階的会話スモークは scripts/chat_staged_smoke.py
で別途実行する。ここでは torch/transformers 無しで通る機構テストのみ。
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import pytest

from llcore.chat.backend import (
    DEFAULT_MODEL,
    MODEL_ENV_VAR,
    ChatDependencyError,
    TransformersBackend,
    resolve_model_id,
)
from llcore.chat.session import (
    DEFAULT_SYSTEM_PROMPT,
    ChatSession,
    GenerationSettings,
    Message,
)


class FakeBackend:
    """generate 呼び出しを記録し、canned 応答を返すテスト用バックエンド。"""

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self.calls: list[list[dict[str, str]]] = []
        self._replies = list(replies or [])

    def generate(
        self, messages: Sequence[Message], settings: GenerationSettings
    ) -> str:
        self.calls.append([m.as_dict() for m in messages])
        if self._replies:
            return self._replies.pop(0)
        return f"reply-{len(self.calls)}"


class FailingBackend:
    def generate(
        self, messages: Sequence[Message], settings: GenerationSettings
    ) -> str:
        raise RuntimeError("backend failure")


# -- ChatSession: 履歴機構 ----------------------------------------------------


def test_ask_appends_user_and_assistant() -> None:
    backend = FakeBackend(replies=["Hi there!"])
    session = ChatSession(backend)
    reply = session.ask("Hello!")
    assert reply == "Hi there!"
    roles = [m.role for m in session.history]
    assert roles == ["system", "user", "assistant"]
    assert session.history[0].content == DEFAULT_SYSTEM_PROMPT
    assert session.history[1].content == "Hello!"
    assert session.history[2].content == "Hi there!"


def test_multi_turn_passes_full_history_to_backend() -> None:
    """文脈引継ぎ: 2 ターン目のバックエンド入力に 1 ターン目の交換が含まれる。"""
    backend = FakeBackend(replies=["Nice to meet you, Kazufumi.", "Your name is Kazufumi."])
    session = ChatSession(backend)
    session.ask("My name is Kazufumi.")
    session.ask("What is my name?")
    second_call = backend.calls[1]
    contents = [m["content"] for m in second_call]
    assert "My name is Kazufumi." in contents
    assert "Nice to meet you, Kazufumi." in contents
    assert second_call[-1] == {"role": "user", "content": "What is my name?"}


def test_staged_topic_shift_keeps_all_stages_in_order() -> None:
    """段階的に内容を変えた会話 (挨拶→Q&A→文脈→話題転換) が履歴に順序通り並ぶ。"""
    stages = [
        "Hello! Who are you?",
        "What is the capital of France?",
        "My name is Kazufumi. Please remember it.",
        "Let's change the topic to cooking. Suggest one pasta dish.",
    ]
    session = ChatSession(FakeBackend())
    for prompt in stages:
        session.ask(prompt)
    user_turns = [m.content for m in session.history if m.role == "user"]
    assert user_turns == stages
    assert session.turn_count == 4
    # user / assistant の交互順 (system の後)
    roles = [m.role for m in session.history][1:]
    assert roles == ["user", "assistant"] * 4


def test_empty_user_message_rejected_and_history_unchanged() -> None:
    session = ChatSession(FakeBackend())
    before = session.history
    with pytest.raises(ValueError):
        session.ask("   ")
    assert session.history == before


def test_backend_failure_rolls_back_user_turn() -> None:
    session = ChatSession(FailingBackend())
    with pytest.raises(RuntimeError, match="backend failure"):
        session.ask("Hello!")
    assert [m.role for m in session.history] == ["system"]


def test_no_system_prompt() -> None:
    session = ChatSession(FakeBackend(), system_prompt="")
    session.ask("Hi")
    assert [m.role for m in session.history] == ["user", "assistant"]


def test_reset_restores_initial_state() -> None:
    session = ChatSession(FakeBackend())
    session.ask("Hello!")
    session.reset()
    assert [m.role for m in session.history] == ["system"]
    assert session.turn_count == 0


# -- backend: モデル ID 解決 / fail-closed -----------------------------------


def test_resolve_model_id_priority(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert resolve_model_id(None) == DEFAULT_MODEL
    monkeypatch.setenv(MODEL_ENV_VAR, "org/env-model")
    assert resolve_model_id(None) == "org/env-model"
    assert resolve_model_id("org/explicit") == "org/explicit"


def test_default_model_is_apache_smollm2_not_qwen() -> None:
    """ベース選定制約: Qwen 回避 + SmolLM2-Instruct (Apache-2.0) を default に。"""
    assert "qwen" not in DEFAULT_MODEL.lower()
    assert "SmolLM2" in DEFAULT_MODEL
    assert "Instruct" in DEFAULT_MODEL


def test_missing_torch_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """torch 不在時は ChatDependencyError (黙って劣化しない)。"""
    monkeypatch.setitem(sys.modules, "torch", None)  # import torch を強制失敗させる
    backend = TransformersBackend(model_id="org/whatever")
    with pytest.raises(ChatDependencyError, match=r"llmesh-llcore\[chat\]"):
        backend._ensure_loaded()


# -- CLI --------------------------------------------------------------------


def test_cli_parser_and_settings() -> None:
    from llcore.chat.__main__ import build_parser, settings_from_args

    args = build_parser().parse_args(
        ["--prompt", "Q1", "--prompt", "Q2", "--greedy", "--max-new-tokens", "64"]
    )
    assert args.prompt == ["Q1", "Q2"]
    settings = settings_from_args(args)
    assert settings.do_sample is False
    assert settings.max_new_tokens == 64


def test_cli_run_prompts_with_fake_backend(capsys: pytest.CaptureFixture[str]) -> None:
    from llcore.chat.__main__ import run_prompts

    session = ChatSession(FakeBackend(replies=["A1", "A2"]))
    exchanges = run_prompts(session, ["Q1", "Q2"])
    assert exchanges == [("Q1", "A1"), ("Q2", "A2")]
    out = capsys.readouterr().out
    assert "you> Q1" in out
    assert "llcore> A2" in out


def test_cli_transcript_roundtrip(tmp_path: Path) -> None:
    import json

    from llcore.chat.__main__ import write_transcript

    session = ChatSession(FakeBackend(replies=["A1"]))
    session.ask("Q1")
    path = tmp_path / "transcript.json"
    write_transcript(session, "org/model", path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["model"] == "org/model"
    assert data["messages"][-1] == {"role": "assistant", "content": "A1"}
