# SPDX-License-Identifier: Apache-2.0
"""Tests for the eval-cache persistence layer that makes long NAS runs resumable.

A full proxy-v2 ``nas_pareto`` run spends hours building ``measure()``'s caches — the scalar
``(pct, Δnll)`` per genome and (under ``--proxy-v2``) the per-window Δnll vector behind it. The
script only writes ``nas_pareto.json`` once, at the very end, so a kill/restart (e.g. ccr re-login)
loses every forward pass. ``eval_cache_io`` snapshots those two caches to disk atomically and reloads
them on restart **only when the run parameters match** (model/context/base-nll), so a re-run hits the
cache instead of recomputing. Pure file I/O — no model needed.

Test style mirrors ``test_eval_proxy.py`` (lazy imports inside test fns, numpy for the vectors).
"""
from __future__ import annotations

import numpy as np

CatGenome = tuple[int, ...]


def _meta() -> dict[str, object]:
    return {"model_dir": "D:/models/Qwen2.5-0.5B-Instruct", "inner_context": 1024,
            "base_nll": 4.4155, "n_layer": 24, "proxy_v2": True}


def test_roundtrip_preserves_scalar_and_vector(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache

    scalar: dict[tuple[CatGenome, bool], tuple[float, float]] = {
        ((0, 1, 2), False): (34.5, -0.0071),
        ((1, 1, 0), True): (50.0, 0.1234),
    }
    vector: dict[tuple[CatGenome, bool], np.ndarray] = {
        ((0, 1, 2), False): np.array([0.10, -0.05, 0.20, -0.02]),
    }
    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, scalar, vector, _meta())

    out = load_eval_cache(p, _meta())
    assert out is not None
    s2, v2 = out
    # scalar keys are restored as (tuple[int,...], bool) and values as a (float, float) tuple
    assert s2 == scalar
    assert all(isinstance(k[0], tuple) and isinstance(k[1], bool) for k in s2)
    assert set(v2.keys()) == set(vector.keys())
    assert np.allclose(v2[((0, 1, 2), False)], vector[((0, 1, 2), False)])


def test_load_returns_none_on_meta_mismatch(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache

    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, {((0,), False): (1.0, 2.0)}, {}, _meta())

    stale = {**_meta(), "base_nll": 9.9999}  # a different run — must NOT resume
    assert load_eval_cache(p, stale) is None


def test_load_returns_none_on_missing_file(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache

    assert load_eval_cache(tmp_path / "nope.json", _meta()) is None


def test_load_returns_none_on_corrupt_file(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache

    p = tmp_path / "eval_cache.json"
    p.write_text("{ this is not valid json", encoding="utf-8")
    assert load_eval_cache(p, _meta()) is None


def test_save_is_atomic_no_tmp_left(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import save_eval_cache

    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, {((0, 1), False): (10.0, -0.01)}, {}, _meta())
    assert p.exists()
    # the temp file used for the atomic replace must not linger
    assert list(tmp_path.glob("*.tmp")) == []


def test_roundtrip_with_empty_vector_v1(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache

    scalar = {((0, 0, 0), False): (0.0, 0.0), ((2, 2, 2), False): (88.0, 0.5)}
    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, scalar, {}, _meta())  # v1 path: no per-window vectors

    out = load_eval_cache(p, _meta())
    assert out is not None
    s2, v2 = out
    assert s2 == scalar
    assert v2 == {}


# --- cross-machine resume robustness (base_nll float drift + relocated model/corpus paths) -------
# A snapshot taken on one machine must still resume on another (e.g. local Windows -> Linux CI /
# Kaggle / a moved candidate dir) where (a) BLAS rounding shifts base_nll in the ~6th decimal and
# (b) the absolute model/corpus paths differ but name the SAME artifact. The identity guard must
# tolerate those two while still rejecting a genuinely different run.


def test_resume_tolerates_base_nll_float_drift(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache

    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, {((0,), False): (1.0, 2.0)}, {}, _meta())

    drifted = {**_meta(), "base_nll": 4.415499}  # ~1e-6 BLAS drift — same logical run
    assert load_eval_cache(p, drifted) is not None


def test_resume_rejects_base_nll_beyond_tolerance(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache

    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, {((0,), False): (1.0, 2.0)}, {}, _meta())

    # 0.0045 >> float drift => a different corpus/model => must NOT resume
    assert load_eval_cache(p, {**_meta(), "base_nll": 4.42}) is None


def test_resume_tolerates_relocated_model_and_corpus_paths(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache

    saved = {**_meta(), "text_file": "out/corpus_aozora_multi.txt"}
    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, {((0,), False): (1.0, 2.0)}, {}, saved)

    relocated = {
        **saved,
        "model_dir": "/kaggle/input/qwen25/Qwen2.5-0.5B-Instruct",  # same basename
        "text_file": "/tmp/data/corpus_aozora_multi.txt",           # same basename
    }
    assert load_eval_cache(p, relocated) is not None


def test_resume_rejects_different_model_basename(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache

    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, {((0,), False): (1.0, 2.0)}, {}, _meta())

    bigger = {**_meta(), "model_dir": "D:/models/Qwen2.5-1.5B-Instruct"}  # different artifact
    assert load_eval_cache(p, bigger) is None


def test_resume_rejects_extra_or_missing_meta_key(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache

    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, {((0,), False): (1.0, 2.0)}, {}, _meta())

    assert load_eval_cache(p, {**_meta(), "fast_windows": 8}) is None  # extra key
    missing = {k: v for k, v in _meta().items() if k != "n_layer"}
    assert load_eval_cache(p, missing) is None  # missing key


def test_resume_rejects_non_numeric_base_nll_mismatch(tmp_path) -> None:
    from llcore.runtime.eval_cache_io import load_eval_cache, save_eval_cache

    p = tmp_path / "eval_cache.json"
    save_eval_cache(p, {((0,), False): (1.0, 2.0)}, {}, _meta())

    # a corrupt/non-numeric base_nll must not slip through the tolerance branch
    assert load_eval_cache(p, {**_meta(), "base_nll": "4.4155"}) is None
