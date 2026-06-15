# SPDX-License-Identifier: Apache-2.0
"""Character-level tokenizer.

The vocabulary is the sorted set of characters in a corpus; token ids are
0-indexed contiguous integers, which is exactly what the model's embedding table
and the llm-viz exporter expect. Deterministic by construction
(``sorted(set(text))``), so a given corpus always yields the same id mapping.
"""
from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from pathlib import Path


class CharTokenizer:
    """Map characters to/from contiguous 0-indexed integer ids.

    Parameters
    ----------
    chars : Iterable[str]
        Distinct single-character strings, in the desired id order. Duplicates or
        non-length-1 entries raise :class:`ValueError`.
    """

    def __init__(self, chars: Iterable[str]) -> None:
        itos = list(chars)
        for c in itos:
            if len(c) != 1:
                raise ValueError(f"vocab entries must be single chars, got {c!r}")
        self.itos: list[str] = itos
        self.stoi: dict[str, int] = {c: i for i, c in enumerate(itos)}
        if len(self.stoi) != len(self.itos):
            raise ValueError("duplicate characters in vocab")

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        """Build a tokenizer whose vocab is ``sorted(set(text))``."""
        if not text:
            raise ValueError("cannot build a tokenizer from empty text")
        return cls(sorted(set(text)))

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        """Encode ``text`` to ids. Raises :class:`KeyError` on an out-of-vocab char."""
        return [self.stoi[c] for c in text]

    def encode_safe(self, text: str, default: int = 0) -> list[int]:
        """Encode ``text``, mapping out-of-vocab chars to ``default`` (fail-soft)."""
        return [self.stoi.get(c, default) for c in text]

    def decode(self, ids: Sequence[int]) -> str:
        """Decode ids back to a string. Raises :class:`IndexError` on an invalid id."""
        return "".join(self.itos[i] for i in ids)

    def to_dict(self) -> dict[str, object]:
        return {"itos": self.itos}

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> CharTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["itos"])

    def __len__(self) -> int:
        return self.vocab_size

    def __eq__(self, other: object) -> bool:
        return isinstance(other, CharTokenizer) and other.itos == self.itos

    def __repr__(self) -> str:
        return f"CharTokenizer(vocab_size={self.vocab_size})"
