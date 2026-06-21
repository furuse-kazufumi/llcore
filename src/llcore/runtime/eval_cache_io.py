# SPDX-License-Identifier: Apache-2.0
"""Resumable on-disk persistence for the NAS eval caches (``scripts/nas_pareto.py``).

A full proxy-v2 run spends hours filling ``measure()``'s two caches — the scalar ``(pct, Δnll)``
per ``(genome, use_distill)`` and, under ``--proxy-v2``, the per-window Δnll **vector** behind that
scalar. The script writes its report only once at the end, so a kill/restart (ccr re-login, OOM,
power) discards every forward pass. This module snapshots both caches atomically and reloads them on
restart — but only when the run parameters match (``meta``), so a *different* run never resumes a
stale cache. The match is strict on every identity field except two, so a snapshot survives being
moved between machines while still rejecting a genuinely different run: filesystem paths
(``model_dir`` / ``text_file``) are compared by basename (relocation-tolerant) and ``base_nll`` within
a small tolerance (cross-platform BLAS float drift) — see :func:`_meta_matches`. Pure ``json`` +
``numpy``; no model dependency.

Format (one JSON object)::

    {"meta": {...run identity...},
     "entries": {"0,1,2|0": {"scalar": [pct, dn], "vector": [d0, d1, ...]}, ...}}

The key ``"0,1,2|0"`` encodes ``((0, 1, 2), False)``. ``vector`` is omitted for genomes evaluated
without proxy-v2 (the v1 path). Writes go to ``<path>.tmp`` then ``os.replace`` onto ``path`` so a
crash mid-write never corrupts an existing snapshot.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CatGenome = tuple[int, ...]
Key = tuple[CatGenome, bool]
Scalar = dict[Key, tuple[float, float]]
Vector = dict[Key, np.ndarray]


# Fields that name a filesystem artifact: compared by basename so a snapshot survives the artifact
# being relocated (local Windows -> Linux CI / Kaggle / a moved candidate dir) as long as it is the
# SAME file. ``base_nll`` (a float) acts as the content check that backs this up: a different model
# behind the same basename shifts base_nll far past the tolerance and is rejected anyway.
_PATH_META_KEYS = frozenset({"model_dir", "text_file"})
# Cross-platform BLAS rounding shifts a single forward's mean CE in the ~6th decimal; this tolerance
# absorbs that drift while staying orders of magnitude below any genuine corpus/model difference.
_BASE_NLL_TOL = 1e-3


def _meta_matches(saved: object, expected: dict[str, object]) -> bool:
    """True iff ``saved`` identifies the same run as ``expected`` (resume is then safe).

    Strict equality on every field EXCEPT: path fields compared by basename (relocation-tolerant)
    and ``base_nll`` compared within :data:`_BASE_NLL_TOL` (float-drift-tolerant). The key SET must
    match exactly, so an added/removed identity field never silently resumes a stale cache.
    """
    if not isinstance(saved, dict) or saved.keys() != expected.keys():
        return False
    for key, exp_val in expected.items():
        saved_val = saved[key]
        if key in _PATH_META_KEYS:
            if os.path.basename(str(saved_val)) != os.path.basename(str(exp_val)):
                return False
        elif key == "base_nll":
            if not (isinstance(saved_val, (int, float)) and isinstance(exp_val, (int, float))):
                # a non-numeric base_nll is malformed; fall back to strict equality (fail-closed)
                if saved_val != exp_val:
                    return False
            elif abs(float(saved_val) - float(exp_val)) > _BASE_NLL_TOL:
                return False
        elif saved_val != exp_val:
            return False
    return True


def _key_to_str(genome: CatGenome, use_distill: bool) -> str:
    return ",".join(str(g) for g in genome) + "|" + ("1" if use_distill else "0")


def _key_from_str(s: str) -> Key:
    g_str, d_str = s.rsplit("|", 1)
    genome = tuple(int(x) for x in g_str.split(","))
    return (genome, d_str == "1")


def save_eval_cache(
    path: str | Path, scalar: Scalar, vector: Vector, meta: dict[str, object]
) -> None:
    """Atomically write both caches plus ``meta`` to ``path`` (via a ``.tmp`` + ``os.replace``)."""
    entries: dict[str, dict[str, list[float]]] = {}
    for key, (pct, dn) in scalar.items():
        entries[_key_to_str(*key)] = {"scalar": [float(pct), float(dn)]}
    for key, vec in vector.items():
        entries.setdefault(_key_to_str(*key), {})["vector"] = [
            float(x) for x in np.asarray(vec).ravel()
        ]
    payload = {"meta": meta, "entries": entries}
    path = Path(path)
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def load_eval_cache(
    path: str | Path, meta: dict[str, object]
) -> tuple[Scalar, Vector] | None:
    """Reload caches saved by :func:`save_eval_cache`, or ``None`` if absent/corrupt/``meta`` mismatch.

    Returning ``None`` (never raising) lets the caller treat "no usable resume point" uniformly: a
    missing file, a truncated/corrupt JSON, or a snapshot from a *different* run all mean "start fresh".
    """
    p = Path(path)
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    if not isinstance(payload, dict) or not _meta_matches(payload.get("meta"), meta):
        return None
    scalar: Scalar = {}
    vector: Vector = {}
    entries = payload.get("entries", {})
    if not isinstance(entries, dict):
        return None
    for s, e in entries.items():
        key = _key_from_str(s)
        if "scalar" in e:
            pct, dn = e["scalar"]
            scalar[key] = (float(pct), float(dn))
        if "vector" in e:
            vector[key] = np.array(e["vector"], dtype=float)
    return scalar, vector
