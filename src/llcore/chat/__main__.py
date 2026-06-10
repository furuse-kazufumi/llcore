# SPDX-License-Identifier: Apache-2.0
"""llcore.chat CLI — 対話 REPL / ワンショット実行。

使い方::

    py -3.11 -m llcore.chat                                 # 対話 REPL
    py -3.11 -m llcore.chat --prompt "Hello!"               # ワンショット
    py -3.11 -m llcore.chat --prompt "Q1" --prompt "Q2"     # 複数ターンを順に実行
    py -3.11 -m llcore.chat --model HuggingFaceTB/SmolLM2-360M-Instruct

REPL コマンド: /exit (終了) /reset (履歴クリア) /history (履歴表示)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Sequence

from llcore.chat.backend import (
    ChatDependencyError,
    TransformersBackend,
    resolve_model_id,
)
from llcore.chat.session import DEFAULT_SYSTEM_PROMPT, ChatSession, GenerationSettings


def _ensure_utf8_stdout() -> None:
    """Windows cp932 console で日本語/絵文字が UnicodeEncodeError にならないようにする。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llcore.chat",
        description="llcore basic chat — SmolLM2-Instruct (CPU, on-prem) との基本会話",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HF モデル ID (default: env LLCORE_CHAT_MODEL または SmolLM2-135M-Instruct)",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=None,
        metavar="TEXT",
        help="非対話モード: 指定した user メッセージを順に送って終了 (複数指定可)",
    )
    parser.add_argument(
        "--system",
        default=DEFAULT_SYSTEM_PROMPT,
        help=(
            "system prompt (空文字で履歴から system を外す。注意: SmolLM2 の "
            "chat template は system 不在時に独自 default system を注入する)"
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="サンプリング無効 (決定論的だが小型モデルは反復しやすい)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="再現性用 seed (generate 毎に適用)"
    )
    parser.add_argument(
        "--transcript",
        type=Path,
        default=None,
        metavar="PATH",
        help="終了時に会話履歴を JSON で書き出すパス",
    )
    parser.add_argument(
        "--show-timing", action="store_true", help="ロード/各ターンの所要秒を表示"
    )
    return parser


def settings_from_args(args: argparse.Namespace) -> GenerationSettings:
    return GenerationSettings(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.greedy,
    )


def run_prompts(
    session: ChatSession, prompts: Sequence[str], show_timing: bool = False
) -> list[tuple[str, str]]:
    """非対話モード: prompts を順に送り (user, reply) のリストを返す。"""
    exchanges: list[tuple[str, str]] = []
    for prompt in prompts:
        print(f"you> {prompt}", flush=True)
        t0 = time.time()
        reply = session.ask(prompt)
        elapsed = time.time() - t0
        print(f"llcore> {reply}", flush=True)
        if show_timing:
            print(f"  [{elapsed:.1f}s]", flush=True)
        exchanges.append((prompt, reply))
    return exchanges


def dispatch_command(session: ChatSession, line: str) -> tuple[bool, bool, str]:
    """REPL コマンド処理。(handled, should_exit, output) を返す (純粋ロジック=テスト可能)。"""
    if line in ("/exit", "/quit"):
        return True, True, ""
    if line == "/reset":
        session.reset()
        return True, False, "(履歴をクリアしました)"
    if line == "/history":
        lines = [f"  [{m.role}] {m.content}" for m in session.history]
        return True, False, "\n".join(lines) if lines else "(履歴なし)"
    return False, False, ""


def repl(session: ChatSession, show_timing: bool = False) -> None:
    """対話 REPL。/exit /reset /history をサポート。"""
    print("llcore chat — /exit で終了, /reset で履歴クリア, /history で履歴表示", flush=True)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue
        handled, should_exit, output = dispatch_command(session, line)
        if handled:
            if output:
                print(output, flush=True)
            if should_exit:
                return
            continue
        t0 = time.time()
        try:
            reply = session.ask(line)
        except ValueError as exc:
            # 空入力は上で弾いているため、ここに来る ValueError は生成系
            # (生成パラメータ / context 予算 / 空応答) のみ
            print(f"(生成エラー: {exc})", flush=True)
            continue
        print(f"llcore> {reply}", flush=True)
        if show_timing:
            print(f"  [{time.time() - t0:.1f}s]", flush=True)


def write_transcript(session: ChatSession, model_id: str, path: Path) -> None:
    payload = {
        "model": model_id,
        "settings": {
            "max_new_tokens": session.settings.max_new_tokens,
            "temperature": session.settings.temperature,
            "top_p": session.settings.top_p,
            "do_sample": session.settings.do_sample,
            "repetition_penalty": session.settings.repetition_penalty,
        },
        "messages": [m.as_dict() for m in session.history],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"(transcript: {path})", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        settings = settings_from_args(args)
    except ValueError as exc:
        # 生成パラメータの fail-closed 検証 (temperature=0 等) はロード前に拒否
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    model_id = resolve_model_id(args.model)
    backend = TransformersBackend(model_id=model_id, seed=args.seed)
    session = ChatSession(backend, system_prompt=args.system, settings=settings)

    print(f"model: {model_id} (CPU)", flush=True)
    exit_code = 0
    try:
        if args.prompt:
            run_prompts(session, args.prompt, show_timing=args.show_timing)
        else:
            repl(session, show_timing=args.show_timing)
    except ChatDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        exit_code = 2
    except KeyboardInterrupt:
        print("\n(中断しました)", file=sys.stderr, flush=True)
        exit_code = 130
    except (ValueError, RuntimeError, OSError) as exc:
        # モデル取得失敗 (HF 由来は OSError 系) / 生成エラー — traceback でなく要点のみ
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        # Ctrl+C やエラーで中断しても、そこまでの会話を失わない
        if args.show_timing and backend.load_seconds is not None:
            print(f"(model load: {backend.load_seconds:.1f}s)", flush=True)
        if args.transcript is not None and session.turn_count > 0:
            try:
                write_transcript(session, model_id, args.transcript)
            except OSError as exc:
                print(f"error: transcript 書込失敗: {exc}", file=sys.stderr, flush=True)
                exit_code = exit_code or 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
