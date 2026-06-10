# SPDX-License-Identifier: Apache-2.0
"""llcore.clip CLI — zero-shot 分類 / テキスト検索の最小インターフェース。

使い方::

    py -3.11 -m llcore.clip --image photo.jpg --labels "a cat,a dog,a car"
    py -3.11 -m llcore.clip --query "a sleeping cat" --texts "a cat,a dog,an airplane"
    py -3.11 -m llcore.clip --image a.jpg --query "a red square"
"""
from __future__ import annotations

import argparse
import sys
from typing import Sequence

from llcore.clip.backend import ClipBackend, ClipDependencyError, zero_shot


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def _split_csv(s: str) -> list[str]:
    items = [x.strip() for x in s.split(",")]
    return [x for x in items if x]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llcore.clip",
        description="llcore CLIP 機能 — text↔image 共有埋め込み (CPU, on-prem)",
    )
    parser.add_argument("--model", default=None, help="HF モデル ID (default: SigLIP base)")
    parser.add_argument("--image", default=None, help="画像パス")
    parser.add_argument("--labels", default=None, help="zero-shot ラベル (カンマ区切り)")
    parser.add_argument("--texts", default=None, help="検索対象テキスト群 (カンマ区切り)")
    parser.add_argument("--query", default=None, help="検索クエリ (テキスト)")
    parser.add_argument(
        "--template", default="a photo of {}", help="zero-shot ラベルのテンプレート"
    )
    return parser


def run(args: argparse.Namespace) -> int:
    backend = ClipBackend(model_id=args.model)
    print(f"model: {backend.model_id} (CPU)", flush=True)

    did_something = False
    if args.image and args.labels:
        ranking = zero_shot(backend, args.image, _split_csv(args.labels), args.template)
        print(f"zero-shot ({args.image}):", flush=True)
        for label, score in ranking:
            print(f"  {score:+.4f}  {label}", flush=True)
        did_something = True
    if args.query and args.texts:
        texts = _split_csv(args.texts)
        T = backend.encode_texts(texts)
        q = backend.encode_texts([args.query])
        sims = (q @ T.T)[0]
        order = sims.argsort()[::-1]
        print(f'text retrieval (query="{args.query}"):', flush=True)
        for i in order:
            print(f"  {float(sims[int(i)]):+.4f}  {texts[int(i)]}", flush=True)
        did_something = True
    if args.image and args.query and not args.labels:
        I_ = backend.encode_images([args.image])
        q = backend.encode_texts([args.query])
        sim = float((I_ @ q.T)[0, 0])
        print(f'similarity("{args.image}", "{args.query}") = {sim:+.4f}', flush=True)
        did_something = True

    if not did_something:
        print(
            "usage: --image+--labels (zero-shot) / --query+--texts (検索) / "
            "--image+--query (1 対 1 類似度)",
            file=sys.stderr,
            flush=True,
        )
        return 2
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    _ensure_utf8_stdout()
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except ClipDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    except (RuntimeError, OSError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
