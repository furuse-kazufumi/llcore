# SPDX-License-Identifier: Apache-2.0
"""Fetch Aozora zip URLs into cleaned UTF-8 corpora plus a train-ready manifest.

This bridges the current gap between the cheap candidate probe and actual data
preparation: given one or more Aozora Bunko zip URLs, download them, strip the
standard ruby/editorial markup, write UTF-8 corpus files, and optionally emit a
manifest that can flow directly into ``scripts/p1_corpus_probe.py`` or
``llcore.lm train --extra-corpus-manifest``.
"""
from __future__ import annotations

import argparse
from email.message import Message
import json
import os
import re
import sys
import urllib.request
from urllib.error import HTTPError
from pathlib import Path
from typing import Any, NoReturn, cast
from urllib.parse import urlparse

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llcore.lm.corpus import build_utf8_corpus_bundle, sha256_text
from llcore.lm.data import extract_aozora_text_from_zip_bytes

_CARD_RE = re.compile(r"/cards/(\d+)/files/")
_ALLOWED_SCHEME = "https"
_ALLOWED_HOSTS = {"www.aozora.gr.jp", "aozora.gr.jp"}

def _metadata_path_for(output_path: Path) -> Path:
    return output_path.with_suffix(output_path.suffix + ".source.json")


def _read_url_manifest(path: Path) -> list[str]:
    """Read one URL-per-line manifest.

    Blank lines and full-line ``#`` comments are ignored. Inline comments are
    intentionally unsupported so malformed lines fail closed as URLs.
    """
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        urls.append(stripped)
    return urls


def _resolve_url_inputs(urls: list[str], manifest_paths: list[Path]) -> list[str]:
    ordered_unique = dict.fromkeys(urls)
    for manifest in manifest_paths:
        ordered_unique.update(dict.fromkeys(_read_url_manifest(manifest)))
    return list(ordered_unique)


def _default_output_name(url: str) -> str:
    parsed = urlparse(url)
    stem = Path(parsed.path).stem
    if not stem:
        raise ValueError(f"cannot derive output name from URL {url!r}")
    card_match = _CARD_RE.search(parsed.path)
    prefix = f"aozora_{card_match.group(1)}_" if card_match else "aozora_"
    safe_stem = re.sub(r"[^0-9A-Za-z._-]+", "_", stem)
    return f"{prefix}{safe_stem}.txt"


def _validate_aozora_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != _ALLOWED_SCHEME:
        raise ValueError(
            f"unsupported URL scheme for Aozora fetch: {url!r} "
            f"(expected {_ALLOWED_SCHEME!r})"
        )
    host = parsed.hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"unsupported Aozora host for fetch: {url!r} "
            f"(expected one of {sorted(_ALLOWED_HOSTS)!r})"
        )


def _download_aozora_text(url: str, *, timeout: float) -> str:
    _validate_aozora_url(url)
    try:
        with _open_aozora_url(url, timeout=timeout) as resp:
            zip_bytes = resp.read()
        return extract_aozora_text_from_zip_bytes(zip_bytes)
    except Exception as exc:  # noqa: BLE001 - include failing URL for batch diagnosis
        raise ValueError(f"failed to fetch or decode Aozora zip {url!r}: {exc}") from exc


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: object,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> NoReturn:
        raise HTTPError(
            newurl,
            code,
            f"redirects are not allowed for Aozora fetch ({msg})",
            cast(Message, headers),
            cast(Any, fp),
        )


def _open_aozora_url(url: str, *, timeout: float) -> Any:
    opener = urllib.request.build_opener(_NoRedirectHandler())
    return opener.open(url, timeout=timeout)


def _load_existing_metadata(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid provenance metadata in {path!r}: expected JSON object")
    return cast(dict[str, Any], payload)


def _write_output_metadata(output_path: Path, record: dict[str, Any]) -> None:
    metadata_path = _metadata_path_for(output_path)
    metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def _tmp_path_for(path: Path, *, counter: int) -> Path:
    return path.with_name(f"{path.name}.{os.getpid()}.{counter}.tmp")


def _write_prepared_outputs(pending_writes: list[tuple[Path, str, dict[str, Any]]]) -> None:
    staged: list[tuple[Path, Path, Path, Path]] = []
    committed_pairs: list[tuple[Path, Path]] = []
    counter = 0
    try:
        for output_path, text, record in pending_writes:
            metadata_path = _metadata_path_for(output_path)
            output_tmp = _tmp_path_for(output_path, counter=counter)
            counter += 1
            metadata_tmp = _tmp_path_for(metadata_path, counter=counter)
            counter += 1
            output_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            output_tmp.write_text(text, encoding="utf-8")
            metadata_tmp.write_text(
                json.dumps(record, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            staged.append((output_path, metadata_path, output_tmp, metadata_tmp))

        for output_path, metadata_path, output_tmp, metadata_tmp in staged:
            committed_pairs.append((output_path, metadata_path))
            os.replace(output_tmp, output_path)
            os.replace(metadata_tmp, metadata_path)
    except Exception:
        for output_path, metadata_path in reversed(committed_pairs):
            for committed in (metadata_path, output_path):
                if committed.exists():
                    committed.unlink()
        raise
    finally:
        for _output_path, _metadata_path, output_tmp, metadata_tmp in staged:
            for tmp_path in (output_tmp, metadata_tmp):
                if tmp_path.exists():
                    tmp_path.unlink()


def _plan_output_paths(urls: list[str], out_dir: Path) -> list[tuple[str, Path, bool]]:
    planned: list[tuple[str, Path, bool]] = []
    seen_outputs: dict[str, str] = {}
    for url in urls:
        _validate_aozora_url(url)
        output_name = _default_output_name(url)
        prior = seen_outputs.get(output_name)
        if prior is not None and prior != url:
            raise ValueError(
                f"output filename collision for {output_name!r}: {prior!r} vs {url!r}"
            )
        output_path = out_dir / output_name
        if output_path.exists():
            metadata_path = _metadata_path_for(output_path)
            if not metadata_path.exists():
                raise ValueError(
                    f"refusing to reuse existing corpus file {str(output_path)!r} "
                    f"for URL {url!r} without provenance metadata {str(metadata_path)!r}"
                )
            metadata = _load_existing_metadata(metadata_path)
            if metadata.get("url") != url:
                raise ValueError(
                    f"refusing to overwrite existing corpus file {str(output_path)!r}: "
                    f"metadata URL {metadata.get('url')!r} does not match {url!r}"
                )
            planned.append((url, output_path, True))
            seen_outputs[output_name] = url
            continue
        seen_outputs[output_name] = url
        planned.append((url, output_path, False))
    return planned


def _write_manifest(path: Path, corpus_paths: list[Path]) -> None:
    lines = ["# Generated by scripts/p1_prepare_aozora.py"]
    for corpus_path in corpus_paths:
        try:
            rel = os.path.relpath(corpus_path, start=path.parent)
        except ValueError:
            rel = str(corpus_path)
        lines.append(Path(rel).as_posix())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_manifest_bundle_metadata(path: Path, corpus_paths: list[Path]) -> Path:
    metadata_path = path.with_suffix(path.suffix + ".bundle.json")
    payload = {
        "generated_by": "scripts/p1_prepare_aozora.py",
        "manifest_path": str(path.resolve()),
        "manifest_sha256": sha256_text(path.read_text(encoding="utf-8")),
        "bundle": build_utf8_corpus_bundle(corpus_paths),
    }
    metadata_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata_path


def prepare_aozora_corpora(
    urls: list[str],
    *,
    out_dir: Path,
    timeout: float = 30.0,
) -> list[dict[str, Any]]:
    """Download and clean one or more Aozora zip URLs into ``out_dir``.

    Pre-flight validation is fail-closed: output name collisions are detected
    before any download/write starts, and non-``cp932`` payloads abort the whole
    batch rather than being skipped.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    plans = _plan_output_paths(urls, out_dir)
    records: list[dict[str, Any]] = []
    pending_writes: list[tuple[Path, str, dict[str, Any]]] = []
    for url, output_path, reused in plans:
        if reused:
            text = output_path.read_text(encoding="utf-8")
            sha256 = sha256_text(text)
            metadata = _load_existing_metadata(_metadata_path_for(output_path))
            if metadata.get("sha256") != sha256:
                raise ValueError(
                    f"refusing to reuse existing corpus file {str(output_path)!r}: "
                    f"metadata sha256 {metadata.get('sha256')!r} does not match {sha256!r}"
                )
            record = {
                "url": url,
                "path": str(output_path.resolve()),
                "chars": len(text),
                "vocab_size": len(set(text)),
                "sha256": sha256,
                "reused_existing": True,
                "metadata_path": str(_metadata_path_for(output_path).resolve()),
            }
            records.append(record)
            continue
        text = _download_aozora_text(url, timeout=timeout)
        sha256 = sha256_text(text)
        record = {
            "url": url,
            "path": str(output_path.resolve()),
            "chars": len(text),
            "vocab_size": len(set(text)),
            "sha256": sha256,
            "reused_existing": False,
            "metadata_path": str(_metadata_path_for(output_path).resolve()),
        }
        records.append(record)
        pending_writes.append((output_path, text, record))
    _write_prepared_outputs(pending_writes)
    return records


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Prepare cleaned UTF-8 corpora from Aozora zip URLs")
    ap.add_argument("urls", nargs="*", help="Aozora zip URLs to fetch")
    ap.add_argument("--out-dir", required=True, help="directory for cleaned UTF-8 corpus files")
    ap.add_argument(
        "--url-manifest",
        action="append",
        default=None,
        help="UTF-8 manifest listing Aozora zip URLs (one per line; # only as full-line comments)",
    )
    ap.add_argument(
        "--write-manifest",
        default=None,
        help="optional corpus manifest to write for probe/train/eval consumption",
    )
    ap.add_argument("--json", default=None, help="optional path to dump a JSON preparation report")
    ap.add_argument("--timeout", type=float, default=30.0, help="per-download timeout in seconds")
    args = ap.parse_args(argv)

    urls = _resolve_url_inputs(list(args.urls), [Path(path) for path in args.url_manifest or []])
    if not urls:
        ap.error("expected at least one Aozora zip URL")
    records = prepare_aozora_corpora(
        urls,
        out_dir=Path(args.out_dir),
        timeout=args.timeout,
    )
    print(f"[prepared] {len(records)} corpora into {args.out_dir}")
    for record in records:
        print(
            f"  - {Path(str(record['path'])).name}: chars={record['chars']} "
            f"vocab={record['vocab_size']} sha256={str(record['sha256'])[:12]} "
            f"reused={record['reused_existing']} url={record['url']}"
        )
    if args.write_manifest:
        manifest_path = Path(args.write_manifest)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        corpus_paths = [Path(str(record["path"])) for record in records]
        _write_manifest(manifest_path, corpus_paths)
        bundle_path = _write_manifest_bundle_metadata(manifest_path, corpus_paths)
        print(f"wrote {manifest_path}")
        print(f"wrote {bundle_path}")
    if args.json:
        payload = {"out_dir": str(Path(args.out_dir).resolve()), "records": records}
        json_path = Path(args.json)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
