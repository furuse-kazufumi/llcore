# SPDX-License-Identifier: Apache-2.0
"""RAD コーパス path-based lookup (stdlib only, llive 非依存)。

API は 3 つの薄い関数:

- :func:`list_domains` — RAD 根ディレクトリ下の ``<domain>_corpus_v2`` 列挙
- :func:`search` — 指定分野群を正規表現で grep し :class:`RADHit` のリストを返す
- :func:`read_doc` — :class:`RADHit` または path を受け取り本文を読む

環境変数:

- ``LLCORE_RAD_DIR`` — RAD 根ディレクトリ (default ``~/.llcore/rad``; 不在なら graceful degrade)

honest 留保:
- 大規模 corpus を grep するため I/O bound。検索は分野指定推奨。
- corpus2skill 階層 (cluster_*/SKILL.md) は読まない素朴 grep。質より速さ優先。
- 結果の relevance score は無し (PoC 3+ で TF-IDF / embedding 拡張)。
"""
from __future__ import annotations

import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RAD_DIR = Path(os.environ.get("LLCORE_RAD_DIR", str(Path.home() / ".llcore" / "rad")))
_DOMAIN_SUFFIX = "_corpus_v2"
_MAX_SNIPPET_CHARS = 240


@dataclass(frozen=True)
class RADHit:
    """単一の grep ヒット.

    Attributes
    ----------
    path : Path
        ヒットしたファイルの絶対パス.
    domain : str
        RAD 分野 (``<domain>_corpus_v2`` の domain 部分).
    line_no : int
        マッチした行番号 (1-indexed).
    snippet : str
        マッチ行 + 文脈の短い抜粋 (最大 ``_MAX_SNIPPET_CHARS`` 文字).
    """

    path: Path
    domain: str
    line_no: int
    snippet: str


def list_domains(rad_dir: Path | None = None) -> list[str]:
    """RAD 根ディレクトリ下の ``<domain>_corpus_v2`` 一覧.

    Returns
    -------
    list[str]
        domain 名のリスト (例: ``["neural_network", "deep_learning", ...]``).
        RAD ディレクトリが存在しなければ空リスト.
    """
    root = rad_dir or DEFAULT_RAD_DIR
    if not root.is_dir():
        return []
    domains: list[str] = []
    for entry in root.iterdir():
        if entry.is_dir() and entry.name.endswith(_DOMAIN_SUFFIX):
            domains.append(entry.name[: -len(_DOMAIN_SUFFIX)])
    return sorted(domains)


def _iter_markdown_files(domain_dir: Path) -> Iterator[Path]:
    """domain ディレクトリ配下の .md ファイルを yield."""
    for p in domain_dir.rglob("*.md"):
        if p.is_file():
            yield p


def search(
    pattern: str,
    domains: Iterable[str] | None = None,
    *,
    rad_dir: Path | None = None,
    max_hits: int = 50,
    flags: int = re.IGNORECASE,
) -> list[RADHit]:
    """指定分野群を正規表現 ``pattern`` で grep.

    Parameters
    ----------
    pattern : str
        Python regex (``re.compile`` 可能であること).
    domains : Iterable[str] | None
        検索対象分野名のリスト. ``None`` なら :func:`list_domains` の全分野.
        非存在分野は skip (warning 出さず honest fail-quiet, 大規模 corpus への
        誤検索を抑止).
    rad_dir : Path | None
        RAD 根ディレクトリ. ``None`` なら ``LLCORE_RAD_DIR`` env / default.
    max_hits : int
        早期打ち切り (大規模 corpus 暴走防止).
    flags : int
        ``re`` flags (default ``IGNORECASE``).

    Returns
    -------
    list[RADHit]
        最大 ``max_hits`` 件のヒット.
    """
    root = rad_dir or DEFAULT_RAD_DIR
    if not root.is_dir():
        return []
    regex = re.compile(pattern, flags)
    target_domains = list(domains) if domains is not None else list_domains(root)
    hits: list[RADHit] = []
    for domain in target_domains:
        domain_dir = root / f"{domain}{_DOMAIN_SUFFIX}"
        if not domain_dir.is_dir():
            continue
        for md_path in _iter_markdown_files(domain_dir):
            try:
                with md_path.open("r", encoding="utf-8", errors="replace") as fh:
                    for line_no, line in enumerate(fh, start=1):
                        if regex.search(line):
                            snippet = line.strip()
                            if len(snippet) > _MAX_SNIPPET_CHARS:
                                snippet = snippet[:_MAX_SNIPPET_CHARS] + "…"
                            hits.append(
                                RADHit(
                                    path=md_path,
                                    domain=domain,
                                    line_no=line_no,
                                    snippet=snippet,
                                )
                            )
                            if len(hits) >= max_hits:
                                return hits
                            break  # 1 ファイル 1 hit (path level uniqueness)
            except OSError:
                continue
    return hits


def read_doc(target: RADHit | Path, *, max_chars: int = 8000) -> str:
    """RADHit またはパスからファイル本文を読み出す.

    Parameters
    ----------
    target : RADHit | Path
        :class:`RADHit` または :class:`pathlib.Path`.
    max_chars : int
        切り詰める文字数上限. 長い paper の暴走防止.

    Returns
    -------
    str
        ファイル本文 (UTF-8 decode, replace errors). 長すぎる場合は末尾切詰.

    Raises
    ------
    FileNotFoundError
        ファイルが見つからない場合.
    """
    path = target.path if isinstance(target, RADHit) else target
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n…(truncated)"
    return text
