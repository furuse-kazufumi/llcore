# SPDX-License-Identifier: Apache-2.0
"""Corpus loading + batching for the char-LM.

Two corpora are supported out of the box (see ``docs/LM_P0_PLAN.md``):

- **English tiny-shakespeare** — the canonical Karpathy char-rnn ``input.txt``
  (known-baseline smoke test).
- **Japanese Aozora Bunko** — public-domain works (e.g. 夏目漱石「吾輩は猫である」),
  distributed as Shift-JIS zips with ruby / editorial markup that must be stripped.

The network fetchers use only the stdlib. The Aozora *cleaning* logic is a pure
function (:func:`clean_aozora`) so it can be unit-tested offline.
"""
from __future__ import annotations

import io
import re
import urllib.request
import zipfile

import torch

from llcore.lm.tokenizer import CharTokenizer

TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)
# 夏目漱石「吾輩は猫である」 (Aozora card 000148) — public domain.
AOZORA_WAGAHAI_ZIP_URL = "https://www.aozora.gr.jp/cards/000148/files/789_ruby_5639.zip"

_RUBY_READING = re.compile(r"《[^》]*》")  # 漢字《かんじ》 reading annotation
_EDITOR_NOTE = re.compile(r"［＃[^］]*］")  # ［＃...］ editorial note
_DASHED_LINE = re.compile(r"^-{20,}\s*$", re.MULTILINE)
_TEISHO_FOOTER = re.compile(r"^底本：", re.MULTILINE)


def clean_aozora(raw: bytes | str, encoding: str = "cp932") -> str:
    """Strip Aozora Bunko markup, returning plain UTF-8 body text.

    Removes: the two dashed header blocks (title/author + symbol legend), ruby
    readings ``《…》`` and the ``｜`` ruby anchor, editorial notes ``［＃…］``, and the
    trailing ``底本：`` colophon. ``cp932`` (a Shift-JIS superset) is used by default.

    Parameters
    ----------
    raw : bytes | str
        Raw file contents (Shift-JIS bytes) or an already-decoded string.
    encoding : str
        Codec used when ``raw`` is bytes. Defaults to ``cp932``.
    """
    text = raw.decode(encoding) if isinstance(raw, bytes) else raw
    # The body sits after the second dashed line; fall back to the whole text.
    parts = _DASHED_LINE.split(text)
    body = parts[2] if len(parts) >= 3 else text
    # Drop the trailing colophon (底本：...).
    body = _TEISHO_FOOTER.split(body)[0]
    body = _RUBY_READING.sub("", body)
    body = body.replace("｜", "")
    body = _EDITOR_NOTE.sub("", body)
    body = body.replace("　", "")  # full-width indentation space
    body = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    return body + "\n"


def extract_aozora_text_from_zip_bytes(zip_bytes: bytes) -> str:
    """Extract and clean the first ``.txt`` payload from an Aozora zip blob."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = [name for name in zf.namelist() if name.lower().endswith(".txt")]
        if not names:
            raise ValueError("no .txt payload in Aozora zip")
        raw = zf.read(names[0])
    return clean_aozora(raw)


def fetch_aozora_text(url: str = AOZORA_WAGAHAI_ZIP_URL, *, timeout: float = 30.0) -> str:
    """Download an Aozora zip, extract its single ``.txt``, and clean it."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - Aozora
        zip_bytes = resp.read()
    try:
        return extract_aozora_text_from_zip_bytes(zip_bytes)
    except ValueError as exc:
        raise ValueError(f"no .txt in Aozora zip {url}") from exc


def fetch_tinyshakespeare(url: str = TINY_SHAKESPEARE_URL, *, timeout: float = 30.0) -> str:
    """Download the tiny-shakespeare corpus as UTF-8 text."""
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - GitHub raw
        raw: bytes = resp.read()
    return raw.decode("utf-8")


def encode_corpus(text: str, tokenizer: CharTokenizer) -> torch.Tensor:
    """Encode a corpus string to a 1-D ``torch.long`` tensor of ids."""
    return torch.tensor(tokenizer.encode(text), dtype=torch.long)


def train_val_split(
    ids: torch.Tensor, val_frac: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split ids into (train, val) by a *contiguous* trailing held-out fraction.

    Contiguous (not shuffled) so the validation slice is genuinely unseen
    continuation — correct for an autoregressive LM.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    n = ids.size(0)
    split = int(n * (1.0 - val_frac))
    if split <= 0 or split >= n:
        raise ValueError("corpus too small for the requested split")
    return ids[:split], ids[split:]


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` random contiguous (x, y) blocks; y is x shifted by one."""
    n = data.size(0)
    if n < block_size + 1:
        raise ValueError(f"data length {n} < block_size+1 ({block_size + 1})")
    ix = torch.randint(n - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])
    return x, y
