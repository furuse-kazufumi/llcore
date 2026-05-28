# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``llcore.rad`` — stdlib-only RAD lookup.

実 RAD コーパス (D:/docs/) に依存しない。tmp_path に minimal な
``<domain>_corpus_v2/`` 構造を作り、API の振る舞いを単独検証する。
"""
from __future__ import annotations

from pathlib import Path

import pytest

from llcore.rad import RADHit, list_domains, read_doc, search


@pytest.fixture
def fake_rad(tmp_path: Path) -> Path:
    """tmp_path に minimal な RAD 構造を作る."""
    nn_dir = tmp_path / "neural_network_corpus_v2" / "cluster_01"
    nn_dir.mkdir(parents=True)
    (nn_dir / "doc_001_mamba.md").write_text(
        "# Mamba paper\n\nMamba is a selective state space model.\n",
        encoding="utf-8",
    )
    (nn_dir / "doc_002_rwkv.md").write_text(
        "# RWKV-7\n\nRWKV-7 is a recurrent linear attention model.\n",
        encoding="utf-8",
    )
    dl_dir = tmp_path / "deep_learning_corpus_v2"
    dl_dir.mkdir()
    (dl_dir / "doc_001_ff.md").write_text(
        "# Forward-Forward\n\nForward-forward is an alternative to backprop.\n",
        encoding="utf-8",
    )
    # 非 corpus ディレクトリ (filter されるべき)
    (tmp_path / "not_a_corpus").mkdir()
    return tmp_path


def test_list_domains_filters_v2_suffix(fake_rad: Path) -> None:
    """``_corpus_v2`` suffix のディレクトリのみ列挙."""
    domains = list_domains(fake_rad)
    assert set(domains) == {"neural_network", "deep_learning"}


def test_list_domains_missing_dir_returns_empty(tmp_path: Path) -> None:
    """非存在 RAD は空リスト (fail-quiet)."""
    assert list_domains(tmp_path / "nonexistent") == []


def test_search_single_domain(fake_rad: Path) -> None:
    """1 分野で grep し正しくヒット."""
    hits = search("RWKV", domains=["neural_network"], rad_dir=fake_rad)
    assert len(hits) == 1
    assert hits[0].domain == "neural_network"
    assert "RWKV-7" in hits[0].snippet


def test_search_all_domains_default(fake_rad: Path) -> None:
    """domains=None で全分野横断."""
    hits = search("forward", rad_dir=fake_rad)
    assert any(h.domain == "deep_learning" for h in hits)


def test_search_case_insensitive(fake_rad: Path) -> None:
    """大文字小文字を区別しない (default flag)."""
    hits_lower = search("mamba", rad_dir=fake_rad)
    hits_upper = search("MAMBA", rad_dir=fake_rad)
    assert len(hits_lower) == len(hits_upper) == 1


def test_search_max_hits_cap(fake_rad: Path) -> None:
    """``max_hits`` で早期打ち切り."""
    hits = search("the", rad_dir=fake_rad, max_hits=1)
    assert len(hits) <= 1


def test_search_nonexistent_domain_skipped(fake_rad: Path) -> None:
    """存在しない分野指定は honest fail-quiet."""
    hits = search("Mamba", domains=["nonexistent_domain"], rad_dir=fake_rad)
    assert hits == []


def test_read_doc_from_hit(fake_rad: Path) -> None:
    """RADHit から本文取得."""
    hits = search("Mamba", rad_dir=fake_rad)
    assert hits
    text = read_doc(hits[0])
    assert "selective state space model" in text


def test_read_doc_truncates(fake_rad: Path, tmp_path: Path) -> None:
    """長文は max_chars で切詰."""
    big = fake_rad / "neural_network_corpus_v2" / "big.md"
    big.write_text("x" * 20000, encoding="utf-8")
    text = read_doc(big, max_chars=100)
    assert len(text) <= 200  # truncate footer 込み
    assert text.endswith("(truncated)")


def test_radhit_dataclass_frozen() -> None:
    """RADHit は frozen (不変)."""
    h = RADHit(path=Path("/x"), domain="d", line_no=1, snippet="s")
    with pytest.raises(Exception):  # FrozenInstanceError
        h.snippet = "modified"  # type: ignore[misc]
