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


# -- 敵対的レビュー反映の回帰テスト (2026-06-10 workflow confirmed findings) ----


def test_generation_settings_fail_closed_validation() -> None:
    """temperature=0 等は transformers の generate 実行時まで検出が遅延する —
    構築時に fail-closed で拒否する (confirmed finding #0)。"""
    with pytest.raises(ValueError, match="temperature"):
        GenerationSettings(do_sample=True, temperature=0.0)
    with pytest.raises(ValueError, match="top_p"):
        GenerationSettings(do_sample=True, top_p=0.0)
    with pytest.raises(ValueError, match="max_new_tokens"):
        GenerationSettings(max_new_tokens=0)
    # greedy 時は temperature 不使用なので 0 を許容
    settings = GenerationSettings(do_sample=False, temperature=0.0)
    assert settings.do_sample is False


def test_empty_assistant_reply_fails_closed_with_rollback() -> None:
    """空応答は空 user の拒否と対称に fail-closed (confirmed finding #8)。"""
    session = ChatSession(FakeBackend(replies=["   "]))
    with pytest.raises(ValueError, match="empty reply"):
        session.ask("Hello!")
    assert [m.role for m in session.history] == ["system"]


def test_non_str_reply_fails_closed_with_rollback() -> None:
    """バックエンドの契約違反 (str 以外) も rollback 窓内で拒否 (confirmed finding #3)。"""

    class BrokenBackend:
        def generate(
            self, messages: Sequence[Message], settings: GenerationSettings
        ) -> str:
            return None  # type: ignore[return-value]

    session = ChatSession(BrokenBackend())
    with pytest.raises(TypeError, match="must return str"):
        session.ask("Hello!")
    assert [m.role for m in session.history] == ["system"]


def test_resolve_model_id_strips_whitespace(monkeypatch: pytest.MonkeyPatch) -> None:
    """明示引数の空白 strip + 空白のみは未指定扱い (confirmed finding #5)。"""
    monkeypatch.delenv(MODEL_ENV_VAR, raising=False)
    assert resolve_model_id("  org/x  ") == "org/x"
    assert resolve_model_id("   ") == DEFAULT_MODEL


def test_repl_dispatch_command() -> None:
    """REPL コマンド dispatch は純粋ロジックとして検証 (confirmed finding #16)。"""
    from llcore.chat.__main__ import dispatch_command

    session = ChatSession(FakeBackend(replies=["A1"]))
    session.ask("Q1")

    handled, should_exit, _ = dispatch_command(session, "/exit")
    assert (handled, should_exit) == (True, True)
    handled, should_exit, _ = dispatch_command(session, "/quit")
    assert (handled, should_exit) == (True, True)

    handled, should_exit, out = dispatch_command(session, "/history")
    assert (handled, should_exit) == (True, False)
    assert "[user] Q1" in out and "[assistant] A1" in out

    handled, should_exit, out = dispatch_command(session, "/reset")
    assert (handled, should_exit) == (True, False)
    assert session.turn_count == 0

    handled, _, _ = dispatch_command(session, "hello")
    assert handled is False


# -- scripts/chat_staged_smoke.py の判定ロジック -------------------------------


def _load_smoke_module():  # type: ignore[no-untyped-def]
    import importlib.util

    root = Path(__file__).resolve().parents[2]
    spec = importlib.util.spec_from_file_location(
        "chat_staged_smoke", root / "scripts" / "chat_staged_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_smoke_keyword_hit_word_boundary() -> None:
    """auto-check の偽陽性防止: ASCII は単語境界照合 (confirmed finding #13)。"""
    smoke = _load_smoke_module()
    assert smoke._keyword_hit("2 + 2 = 4.", "4")
    assert not smoke._keyword_hit("in the year 2024", "4")
    assert smoke._keyword_hit("the capital is paris.", "paris")
    assert not smoke._keyword_hit("a comparison of methods", "paris")
    assert not smoke._keyword_hit("fourteen items", "four")
    # 日本語は包含判定
    assert smoke._keyword_hit("首都は東京です", "東京")


def test_smoke_exit_code_critical_stages_only() -> None:
    """stage4 の偶然ヒットで stage2/3 全滅を隠さない (confirmed finding #2/#11)。"""
    smoke = _load_smoke_module()

    def turn(stage: str, verdict: str, expect: list[str] | None) -> dict[str, object]:
        return {"stage": stage, "auto_check": verdict, "expected_keywords": expect}

    # stage2/3 全滅 + stage4 ヒット → 1 (基本会話不成立)
    results = [
        turn("stage2_simple_qa", "unexpected", ["paris"]),
        turn("stage3_context_carryover", "unexpected", ["kazufumi"]),
        turn("stage4_topic_shift", "expected", ["mars"]),
    ]
    assert smoke.exit_code(results) == 1
    # stage3 が 1 つ通れば 0
    results[1] = turn("stage3_context_carryover", "expected", ["kazufumi"])
    assert smoke.exit_code(results) == 0
