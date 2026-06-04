# SPDX-License-Identifier: Apache-2.0
"""CPU-real tiny byte-level language model with an EVOLVABLE + VERIFIED recurrent core.

This realizes llcore's "Transformer core" claim on CPU as an actual (tiny) language model:
the recurrent state-mixing core is the arc's ``CoupledNDGene`` (decay, W) — the *evolvable +
contraction-verified* part — wired into a real next-byte prediction pipeline with a real
held-out cross-entropy / perplexity fitness.

Pipeline (see PREREGISTRATION.md §1; reservoir / ESN style, deterministic, numpy-only)::

    token x_t ∈ {0..255}
    e(x_t) = tanh(E[x_t]) ∈ (-1,1)^n             # FIXED seeded byte-embedding ("language sense organ")
    s_t = decay⊙s_{t-1} + (1-decay)⊙tanh(W s_{t-1} + e(x_t))   # arc CoupledNDGene recurrence (V=I)
    logits_t = R s_t + c                          # readout, ridge-fit per gene (closed form, deterministic)
    loss = mean CE(softmax(logits_t), x_{t+1})    # real next-byte LM loss
    fitness = exp(-held_out_CE) ∈ (0,1]           # per-byte likelihood (headroom; no exp(-MSE) ceiling)

SOUNDNESS contract (PREREGISTRATION.md §2): the embedding is tanh-bounded (|e|<1) and the
recurrence keeps |s|<1 (Lemma 1), so ``max_input_abs=1.0`` is a sound input bound and the arc
certifiers (``coupled_nd.cert_inf/two/sdp``) prove contraction of the LM recurrence for ALL byte
inputs. Do NOT remove the tanh on the embedding — it is load-bearing for soundness.

Additive research only: reuses ``../verified_evolution_sdp_gate/coupled_nd.py``; src/ untouched.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

# --- wire in the arc's n-dim verified substrate (soundness-critical reuse) --- #
_HERE = os.path.dirname(os.path.abspath(__file__))
_SDP_GATE = os.path.normpath(os.path.join(_HERE, "..", "verified_evolution_sdp_gate"))
if _SDP_GATE not in sys.path:
    sys.path.insert(0, _SDP_GATE)

from coupled_nd import (  # noqa: E402
    CoupledNDGene,
    cert_inf,
    cert_two,
    cert_sdp,
    classify_region,
    jacobian,
    step,
)

VOCAB = 256  # byte-level
MAX_INPUT_ABS = 1.0  # SOUNDNESS contract: |e(x)|_inf < 1 by tanh ⇒ this is a sound input bound. LOCKED.


# --------------------------------------------------------------------------- #
# Corpus (deterministic, offline, self-contained).
# --------------------------------------------------------------------------- #


def load_corpus(max_bytes: int = 16384, *, root: str | None = None) -> bytes:
    """Deterministic byte corpus = sorted llcore research/*.md + src/**/*.py, concatenated.

    Self-contained (no download). Sorted for reproducibility; truncated to ``max_bytes``.
    """
    root = root or os.path.normpath(os.path.join(_HERE, "..", ".."))  # llcore/
    paths: list[str] = []
    research = os.path.join(root, "research")
    src = os.path.join(root, "src")
    for base, _, files in os.walk(research):
        for f in files:
            if f.endswith(".md"):
                paths.append(os.path.join(base, f))
    for base, _, files in os.walk(src):
        for f in files:
            if f.endswith(".py"):
                paths.append(os.path.join(base, f))
    paths.sort()
    buf = bytearray()
    for p in paths:
        try:
            with open(p, "rb") as fh:
                buf.extend(fh.read())
                buf.extend(b"\n\n")
        except OSError:
            continue
        if len(buf) >= max_bytes:
            break
    return bytes(buf[:max_bytes])


def to_ids(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.uint8).astype(np.int64)


# --------------------------------------------------------------------------- #
# Byte-embedding (a "sense organ"; FIXED, seeded, tanh-bounded for soundness).
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ByteEmbedding:
    """token ∈ {0..255} -> tanh(E[token]) ∈ (-1,1)^n.  FIXED (seeded), shared across all genes.

    Multimodal-extensible (PREREGISTRATION.md §7): any ModalityEncoder that returns |.|_inf<1
    plugs into the same verified core. Vision (patch embedding) is a future drop-in sense organ.
    """

    n: int
    seed: int = 0
    scale: float = 1.0
    table: np.ndarray = field(default=None, repr=False)  # (256, n), pre-tanh'd

    @classmethod
    def make(cls, n: int, seed: int = 0, scale: float = 1.0) -> "ByteEmbedding":
        rng = np.random.default_rng(seed)
        raw = scale * rng.standard_normal((VOCAB, n))
        return cls(n=n, seed=seed, scale=scale, table=np.tanh(raw))

    def encode_ids(self, ids: np.ndarray) -> np.ndarray:
        """(len,) byte ids -> (len, n) embeddings, each |.|_inf < 1."""
        return self.table[ids]


# --------------------------------------------------------------------------- #
# Reservoir rollout (the verified recurrent core in action).
# --------------------------------------------------------------------------- #


def reservoir_states(gene: CoupledNDGene, emb_seq: np.ndarray) -> np.ndarray:
    """Run s_t = step(gene, s_{t-1}, e(x_t)) over an embedded sequence (V=I ⇒ input=embedding).

    Parameters
    ----------
    gene : CoupledNDGene  (decay, W); V=I assumed (input is the embedding directly).
    emb_seq : (T, n) embeddings.

    Returns
    -------
    (T, n) hidden states; states[t] is produced AFTER consuming token t (predicts token t+1).
    """
    gc = gene.clipped()
    n = gc.n
    s = np.zeros(n, dtype=np.float64)
    T = emb_seq.shape[0]
    out = np.empty((T, n), dtype=np.float64)
    decay = gc.decay
    one_minus = 1.0 - decay
    W = gc.W
    for t in range(T):
        s = decay * s + one_minus * np.tanh(W @ s + emb_seq[t])
        out[t] = s
    return out


def hidden_stability(gene: CoupledNDGene, emb_seq: np.ndarray) -> tuple[float, bool]:
    """Independent from-below stability oracle (L1/L2): max|s| over a real sequence, NaN flag.

    A SOUND contraction gate must keep this bounded (|s|<1 by Lemma 1 for any contracting OR
    non-contracting gene actually — tanh bounds it; the real failure is divergence of the
    *certificate-relevant* Jacobian, measured via empirical contraction below). We additionally
    report whether the linearized map is empirically expansive (sup rho(J) over the visited
    states), which is the genuine soundness signal an admitted gene must satisfy (<1)."""
    S = reservoir_states(gene, emb_seq)
    if not np.all(np.isfinite(S)):
        return float("inf"), True
    return float(np.max(np.abs(S))), False


def empirical_contraction(gene: CoupledNDGene, emb_seq: np.ndarray, *, stride: int = 7) -> float:
    """From-below sup of rho(J) over states actually VISITED while reading the corpus.

    Soundness consistency check on the REAL substrate: a gene admitted by a sound contraction
    gate must have this < 1 (the certificate guarantees it over the t-box ⊇ visited states).
    Non-vacuous here: ungated genes (e.g. decay~0, large W) ARE expansive, so this can falsify."""
    gc = gene.clipped()
    S = reservoir_states(gc, emb_seq)
    mx = 0.0
    for t in range(0, S.shape[0], stride):
        # Jacobian at the visited state; input bounded by the embedding (|e|<1). Use the
        # actual visited (s, e) pair via the t = sech^2(pre) at that point.
        x = emb_seq[t]
        J = jacobian(gc, S[t - 1] if t > 0 else np.zeros(gc.n), x)
        r = float(np.max(np.abs(np.linalg.eigvals(J))))
        if r > mx:
            mx = r
    return mx


# --------------------------------------------------------------------------- #
# Ridge readout (closed-form, deterministic) + real cross-entropy.
# --------------------------------------------------------------------------- #


def _augment(S: np.ndarray) -> np.ndarray:
    """Append a bias column of ones: (T,n) -> (T,n+1)."""
    return np.concatenate([S, np.ones((S.shape[0], 1))], axis=1)


def fit_ridge_readout(S_train: np.ndarray, y_next_train: np.ndarray, *, alpha: float = 1e-2) -> np.ndarray:
    """Closed-form ridge of next-token one-hot on (augmented) hidden states.

    Returns R_aug : (n+1, VOCAB).  logits = _augment(S) @ R_aug.
    """
    F = _augment(S_train)                       # (T, n+1)
    d = F.shape[1]
    Y = np.zeros((F.shape[0], VOCAB), dtype=np.float64)
    Y[np.arange(F.shape[0]), y_next_train] = 1.0
    A = F.T @ F + alpha * np.eye(d)
    B = F.T @ Y
    return np.linalg.solve(A, B)                # (n+1, VOCAB)


def cross_entropy(logits: np.ndarray, y_true: np.ndarray) -> float:
    """Mean per-token softmax cross-entropy (nats). logits (T,VOCAB), y_true (T,)."""
    m = logits.max(axis=1, keepdims=True)
    logZ = m[:, 0] + np.log(np.sum(np.exp(logits - m), axis=1))
    true_logit = logits[np.arange(logits.shape[0]), y_true]
    return float(np.mean(logZ - true_logit))


def unigram_ce(train_next: np.ndarray, heldout_next: np.ndarray, *, alpha: float = 1.0) -> float:
    """Baseline: byte-frequency unigram CE on the held-out next-tokens (nats)."""
    counts = np.bincount(train_next, minlength=VOCAB).astype(np.float64) + alpha
    p = counts / counts.sum()
    logp = np.log(p)
    return float(-np.mean(logp[heldout_next]))


# --------------------------------------------------------------------------- #
# LM task (deterministic Objective over CoupledNDGene).
# --------------------------------------------------------------------------- #


@dataclass
class LMTask:
    """Deterministic next-byte LM fitness over a CoupledNDGene (Objective for evolvable_core).

    fitness(gene) = exp(-held_out_CE) using a per-gene ridge readout. Higher = better. The
    embedding is FIXED and shared (fair comparison across genes / gates). Held-out positions are
    strictly later in time than train (no leakage)."""

    emb: ByteEmbedding
    ids: np.ndarray                 # full corpus byte ids
    train_frac: float = 0.8
    alpha: float = 1e-2
    name: str = "byte_lm"
    # cached split
    _emb_seq: np.ndarray = field(default=None, repr=False)
    _split: int = field(default=0, repr=False)
    _train_next: np.ndarray = field(default=None, repr=False)
    _heldout_next: np.ndarray = field(default=None, repr=False)

    def __post_init__(self):
        # inputs are tokens 0..T-2, targets are tokens 1..T-1
        self._emb_seq = self.emb.encode_ids(self.ids[:-1])      # (T-1, n) inputs
        targets = self.ids[1:]                                  # (T-1,)
        T = self._emb_seq.shape[0]
        self._split = int(T * self.train_frac)
        self._train_next = targets[: self._split]
        self._heldout_next = targets[self._split :]

    @property
    def unigram_ce(self) -> float:
        return unigram_ce(self._train_next, self._heldout_next)

    @property
    def unigram_fitness(self) -> float:
        return float(np.exp(-self.unigram_ce))

    def held_out_ce(self, gene: CoupledNDGene) -> float:
        S = reservoir_states(gene, self._emb_seq)
        if not np.all(np.isfinite(S)):
            return float("inf")
        S_tr, S_ho = S[: self._split], S[self._split :]
        R = fit_ridge_readout(S_tr, self._train_next, alpha=self.alpha)
        logits_ho = _augment(S_ho) @ R
        return cross_entropy(logits_ho, self._heldout_next)

    def fitness(self, gene: CoupledNDGene) -> float:
        ce = self.held_out_ce(gene)
        if not np.isfinite(ce):
            return 0.0
        return float(np.exp(-ce))


__all__ = [
    "VOCAB",
    "MAX_INPUT_ABS",
    "ByteEmbedding",
    "CoupledNDGene",
    "LMTask",
    "cert_inf",
    "cert_two",
    "cert_sdp",
    "classify_region",
    "cross_entropy",
    "empirical_contraction",
    "fit_ridge_readout",
    "hidden_stability",
    "load_corpus",
    "reservoir_states",
    "to_ids",
    "unigram_ce",
]
