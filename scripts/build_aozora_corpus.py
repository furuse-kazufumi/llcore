# SPDX-License-Identifier: Apache-2.0
"""青空文庫マルチ作品コーパス・ビルダー — public-domain な複数作品を 1 つに束ねる。

P1「データ追加」レバー検証用。単一作品(吾輩は猫である 289K字)では大容量モデルが
過学習する/予算に支配される、という所見を受け、**同一著者の新字新仮名 public-domain
作品を複数連結**して数MB級の日本語 char-LM コーパスを作る。

データ源は青空文庫公式の作品マスター CSV
(``list_person_all_extended_utf8.zip``)。これを著者(人物ID)・文字遣い・著作権で
フィルタし、各作品の zip を取得 → :func:`llcore.lm.data.clean_aozora` で整形 → 連結。

honest 留保:
  - **新字新仮名のみ**に限定(旧字旧仮名を混ぜると vocab が膨張し char 分布が二極化)。
  - 著作権フラグが「なし」(public domain)の作品だけを取り込む(fail-closed)。
  - held-out は連結後の末尾 10%(既存 :func:`train_val_split` 準拠)= 末尾作品中心の
    cross-work held-out。汎化寄りで保守的だが、1 作品の文体に偏りうる点は開示する。

使い方::

    py -3.11 scripts/build_aozora_corpus.py                     # 漱石 全新字新仮名 PD
    py -3.11 scripts/build_aozora_corpus.py --author-id 000148 --max-works 20
    py -3.11 scripts/build_aozora_corpus.py --out out/corpus_aozora_multi.txt
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

from llcore.lm.data import clean_aozora

MASTER_ZIP_URL = "https://www.aozora.gr.jp/index_pages/list_person_all_extended_utf8.zip"
SOSEKI_PERSON_ID = "000148"  # 夏目漱石


def _download(url: str, timeout: float = 60.0) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 - aozora
        data: bytes = resp.read()
    return data


def _load_master_rows(cache: Path) -> list[dict[str, str]]:
    """master CSV を取得(キャッシュ)し dict 行のリストで返す。"""
    if cache.exists() and cache.stat().st_size > 10_000:
        raw = cache.read_bytes()
    else:
        print(f"[master] downloading {MASTER_ZIP_URL} ...")
        zip_bytes = _download(MASTER_ZIP_URL)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            name = next(n for n in zf.namelist() if n.lower().endswith(".csv"))
            raw = zf.read(name)
        cache.write_bytes(raw)
        print(f"[master] cached {len(raw):,} bytes -> {cache}")
    text = raw.decode("utf-8-sig")
    return list(csv.DictReader(io.StringIO(text)))


def _select(
    rows: list[dict[str, str]], author_id: str, orthography: str
) -> list[dict[str, str]]:
    """人物ID・文字遣い・public-domain・zip URL ありで作品を絞り、作品名で dedup。"""
    seen: set[str] = set()
    out: list[dict[str, str]] = []
    for r in rows:
        if r.get("人物ID") != author_id:
            continue
        if r.get("文字遣い種別") != orthography:
            continue
        # public domain (fail-closed): 著作権フラグが「なし」のみ
        if r.get("作品著作権フラグ", "あり") != "なし":
            continue
        url = (r.get("テキストファイルURL") or "").strip()
        if not url.lower().endswith(".zip"):
            continue
        title = (r.get("作品名") or "").strip()
        if title in seen:
            continue
        seen.add(title)
        out.append(r)
    return out


def _fetch_work(row: dict[str, str]) -> str | None:
    """1 作品の zip を取得し本文を整形して返す(失敗時 None, fail-safe)。"""
    url = (row.get("テキストファイルURL") or "").strip()
    enc = (row.get("テキストファイル符号化方式") or "ShiftJIS").strip()
    codec = "cp932" if "shift" in enc.lower() else "utf-8"
    try:
        zip_bytes = _download(url, timeout=30.0)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
            names = [n for n in zf.namelist() if n.lower().endswith(".txt")]
            if not names:
                return None
            raw = zf.read(names[0])
        return clean_aozora(raw, encoding=codec)
    except Exception as exc:  # noqa: BLE001 - per-work fail-safe (skip on error)
        print(f"[skip] {row.get('作品名')} ({url}): {exc}")
        return None


def main(argv: list[str] | None = None) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:  # noqa: BLE001
        pass
    ap = argparse.ArgumentParser(description="Build a multi-work Aozora corpus")
    ap.add_argument("--author-id", default=SOSEKI_PERSON_ID, help="aozora 人物ID (default 漱石)")
    ap.add_argument("--orthography", default="新字新仮名", help="文字遣い種別 filter")
    ap.add_argument("--max-works", type=int, default=None, help="cap number of works")
    ap.add_argument("--out", default="out/corpus_aozora_multi.txt")
    ap.add_argument("--master-cache", default="out/aozora_master.csv")
    args = ap.parse_args(argv)

    rows = _load_master_rows(Path(args.master_cache))
    selected = _select(rows, args.author_id, args.orthography)
    selected.sort(key=lambda r: (r.get("作品名") or ""))
    if args.max_works is not None:
        selected = selected[: args.max_works]
    print(f"[select] {len(selected)} works (author={args.author_id}, {args.orthography}, PD)")

    parts: list[str] = []
    manifest: list[dict[str, object]] = []
    for i, row in enumerate(selected, 1):
        body = _fetch_work(row)
        if not body or len(body) < 200:
            continue
        parts.append(body)
        manifest.append({"title": (row.get("作品名") or "").strip(), "chars": len(body)})
        print(f"[{i:>3}/{len(selected)}] {row.get('作品名'):<24} {len(body):>8,} chars")

    if not parts:
        print("no works fetched")
        return 1

    corpus = "\n\n".join(parts) + "\n"
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(corpus, encoding="utf-8")
    vocab = len(set(corpus))
    man_path = out_path.with_name(out_path.stem + "_manifest.json")
    man_path.write_text(
        json.dumps(
            {
                "author_id": args.author_id,
                "orthography": args.orthography,
                "n_works": len(manifest),
                "total_chars": len(corpus),
                "vocab_size": vocab,
                "works": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"\n[corpus] {len(manifest)} works  {len(corpus):,} chars  vocab~{vocab}\n"
        f"         -> {out_path}\n         -> {man_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
