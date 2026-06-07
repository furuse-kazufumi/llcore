# SPDX-License-Identifier: Apache-2.0
"""TDD for the verified-evolution byte-LM substrate (R-LLM-0).

Run: py -3.11 -m pytest test_lm.py -q
Covers: embedding soundness contract, Lemma-1 (|s|<1), determinism, L1 (admit => empirical
contraction), L2 (ungated pool has expansive genes), L0 (a contracting gene beats unigram).
"""
from __future__ import annotations

import numpy as np
import pytest

from lm_substrate import (
    ByteEmbedding,
    CoupledNDGene,
    LMTask,
    cert_inf,
    cert_two,
    cert_sdp,
    empirical_contraction,
    hidden_stability,
    load_corpus,
    reservoir_states,
    to_ids,
)

N = 6  # tiny for fast tests (2^6=64 vertices => cert_two/sdp tractable)


@pytest.fixture(scope="module")
def task():
    data = load_corpus(max_bytes=4096)
    return LMTask(emb=ByteEmbedding.make(n=N, seed=0), ids=to_ids(data),
                  readout_steps=80, lr=0.5)


def _rand_gene(rng, *, decay_lo=0.0, w_scale=1.0):
    decay = rng.uniform(decay_lo, 1.0, size=N)
    W = rng.standard_normal((N, N)) * w_scale
    return CoupledNDGene.make(decay=decay, W=np.clip(W, -2, 2))


# --- soundness contract --------------------------------------------------- #


def test_embedding_strictly_bounded():
    emb = ByteEmbedding.make(n=N, seed=3, scale=5.0)  # even with large scale, tanh bounds it
    assert np.all(np.abs(emb.table) < 1.0)


def test_lemma1_state_bounded_even_for_expansive(task):
    """Lemma 1: |s|<1 for ANY gene (contracting or not) given bounded embedding."""
    rng = np.random.default_rng(1)
    for _ in range(20):
        g = _rand_gene(rng, decay_lo=0.0, w_scale=2.0)  # includes expansive genes
        mx, nan = hidden_stability(g, task._emb_seq)
        assert not nan
        assert mx < 1.0 + 1e-9


# --- determinism ---------------------------------------------------------- #


def test_fitness_deterministic(task):
    g = CoupledNDGene.make(decay=np.full(N, 0.6), W=np.zeros((N, N)))
    assert task.held_out_ce(g) == task.held_out_ce(g)


def test_embedding_deterministic():
    a = ByteEmbedding.make(n=N, seed=0).table
    b = ByteEmbedding.make(n=N, seed=0).table
    assert np.array_equal(a, b)


# --- L1: admit => empirically contracting on the REAL substrate ----------- #


def test_admit_implies_empirical_contraction(task):
    rng = np.random.default_rng(2)
    checked = 0
    for _ in range(300):
        g = _rand_gene(rng, decay_lo=0.2, w_scale=0.5)
        if cert_inf(g) or cert_two(g) or cert_sdp(g):
            rho = empirical_contraction(g, task._emb_seq, stride=5)
            assert rho < 1.0, f"admitted gene empirically expansive: rho={rho}"
            checked += 1
        if checked >= 25:
            break
    assert checked >= 5, "too few admitted genes to validate L1"


# --- L2: the ungated pool genuinely contains expansive genes (non-vacuous) - #


def test_ungated_pool_has_expansive(task):
    rng = np.random.default_rng(4)
    expansive = 0
    total = 0
    for _ in range(120):
        g = _rand_gene(rng, decay_lo=0.0, w_scale=2.0)
        total += 1
        if empirical_contraction(g, task._emb_seq, stride=9) >= 1.0:
            expansive += 1
    assert expansive / total > 0.05, f"only {expansive}/{total} expansive (oracle would be vacuous)"


def test_obviously_expansive_gene_rejected(task):
    g = CoupledNDGene.make(decay=np.zeros(N), W=2.0 * np.eye(N))  # J=diag(2t), rho up to 2
    assert not cert_inf(g)
    assert not cert_two(g)
    assert not cert_sdp(g)
    assert empirical_contraction(g, task._emb_seq, stride=9) >= 1.0


# --- L0: a contracting gene functions as an LM (beats unigram) ------------- #


def test_L0_contracting_gene_beats_unigram(task):
    rng = np.random.default_rng(7)
    best = float("inf")
    for _ in range(2000):
        g = _rand_gene(rng, decay_lo=0.3, w_scale=0.6 / np.sqrt(N))
        if cert_two(g):
            ce = task.held_out_ce(g)
            best = min(best, ce)
        if best < task.unigram_ce - 0.05:
            break
    assert best < task.unigram_ce, f"best LM CE {best} did not beat unigram {task.unigram_ce}"
