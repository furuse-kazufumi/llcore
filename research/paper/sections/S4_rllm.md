## 4. R-LLM: the verified core as a real tiny byte-LM (L0/L1/L2)

### 4.1 Motivation and scope

The preceding sections establish a contraction-verified, evolvable recurrent
core on synthetic dynamical tasks. A fair objection is that llcore claims a
"Transformer core" yet, in `src/`, runs only low-dimensional dynamics with a
synthetic or ESN-proxy fitness — it is not exercised as an actual language
model (PREREGISTRATION.md). The R-LLM stage closes that gap with the smallest
falsifiable construction: it wires the *same* verified recurrent core into a
real next-byte prediction pipeline whose fitness is held-out cross-entropy /
perplexity (lm_substrate.py, PREREGISTRATION.md §1).

We state the limitation up front, before any result, because it bounds every
claim that follows. This is a **reservoir / echo-state (ESN) byte-language
model** — a fixed, seeded byte-embedding feeds a recurrent core, and a per-gene
closed-form readout produces logits. It is **not a gradient-trained
Transformer**: there is no attention, no learned embedding, no end-to-end
backprop through the core. A genuine softmax-attention Transformer under
end-to-end training is explicitly deferred to a later GPU stage
(PREREGISTRATION.md §1, VERDICT.md §5). The questions we ask are therefore
about *relative* behaviour (does the verified core function as a minimal LM,
and is the contraction gate load-bearing), not about absolute perplexity, which
at this tiny scale is weak by design (PREREGISTRATION.md §1 "honest 限定").

### 4.2 The reservoir byte-LM pipeline

For a byte token `x_t ∈ {0..255}` the pipeline is (lm_substrate.py module
docstring; PREREGISTRATION.md §1):

```
e(x_t)   = tanh(E[x_t]) ∈ (-1,1)^n          # fixed seeded byte-embedding (the "sense organ")
s_t      = decay ⊙ s_{t-1} + (1-decay) ⊙ tanh(W s_{t-1} + e(x_t))   # arc CoupledNDGene recurrence (V=I)
logits_t = R s_t + c                         # per-gene logistic readout
loss     = mean CE(softmax(logits_t), x_{t+1})   # real next-byte LM loss
fitness  = exp(-held_out_CE) ∈ (0,1]         # per-byte likelihood
```

Three design choices are load-bearing:

- **Fixed `tanh` embedding as a "sense organ."** `E` is a fixed, seeded random
  table passed through `tanh`, so `|e(x)|_∞ < 1` for every byte
  (lm_substrate.py `ByteEmbedding.make`, which stores `np.tanh(raw)`). It is
  shared across all genes so that fitness comparisons are fair. The `tanh` is
  not cosmetic: it is the input bound that makes the soundness argument hold,
  and the source explicitly marks it as not-to-be-removed (lm_substrate.py
  module docstring, `MAX_INPUT_ABS = 1.0  # ... LOCKED`).

- **The arc `CoupledNDGene` recurrence as the evolvable, verified core.** The
  state update is exactly the arc's coupled recurrence with `V = I` (input is
  the embedding directly); the gene `(decay, W)` is the evolvable and
  contraction-verified object, reused unchanged from
  `../verified_evolution_sdp_gate/coupled_nd.py` (lm_substrate.py
  `reservoir_states`; src/ untouched, additive-only).

- **Closed-form, deterministic logistic readout.** A naive ridge-to-one-hot
  readout collapses to ~uniform under softmax (its outputs are ~`1/VOCAB`
  scale), so it is kept only as a fast pre-screen and is *not* used for the LM
  fitness (lm_substrate.py `fit_ridge_readout` docstring). The CE-proper
  readout is a multinomial-logistic head fit by deterministic momentum gradient
  descent from a **zero initialization** (lm_substrate.py
  `fit_logistic_readout`). The zero-init is important for the baseline: with the
  bias column, the zero-feature limit *exactly* recovers the byte-unigram, so
  held-out CE is a fair "does reservoir memory help over no-context" test
  (lm_substrate.py `fit_logistic_readout` docstring; VERDICT.md §1). The readout
  is byte-for-byte identical (same steps, lr, l2, embedding, split) across every
  gene, so CE ordering reflects core dynamics, not readout capacity
  (VERDICT.md §2 "Readout controlled").

The corpus is the llcore `research/*.md` + `src/**/*.py` byte-concatenated,
sorted for reproducibility, and split train/held-out 80/20 in time order with no
leakage (lm_substrate.py `load_corpus`, `LMTask.__post_init__`;
PREREGISTRATION.md §1). The whole pipeline is CPU-only, numpy-only, and
deterministic with no per-evaluation RNG.

A dimensional limit is inherent and disclosed: `cert_two` / `cert_sdp` enumerate
the `2^n` vertices of the Jacobian t-box, so the reported runs use **n = 8**
(`2^8 = 256` vertices, all certifiers tractable). Scaling to `n = 32+`
needs a vertex-free sound certifier and is deferred to a later R-LLM-1 stage
(PREREGISTRATION.md §6, VERDICT.md §5).

### 4.3 Soundness on the real substrate (theorem first, oracle second)

Soundness is a *theorem*, not a measurement. The certifiers `cert_inf`,
`cert_two`, `cert_sdp` are reused unchanged from the arc, and three lemmas carry
their guarantee onto the LM recurrence: the state stays in `(-1,1)^n`
(state-boundedness), the `tanh` embedding gives `|e(x)|_∞ < 1` so
`max_input_abs = 1.0` is a sound input bound, and the reachable Jacobians are
covered by the t-box the certifiers reason over (PREREGISTRATION.md §2,
Lemmas 1–3). Consequently any gene a sound certifier admits has `ρ(J) < 1` over
the t-box and the LM recurrence contracts for all byte inputs (echo-state
property). The empirical figures below are a *from-below consistency check* —
they can falsify, but they corroborate rather than constitute soundness
(VERDICT.md §2; this terminology was corrected in Codex pair-review, "outcome-blind
/ non-leaky" rather than "independent").

### 4.4 L0 — the tiny reservoir LM actually functions (PASS)

On the landscape corpus (12288 B, `unigram_CE = 5.2399`), the best *contracting*
held-out CE in every certifier region beats the Laplace-smoothed byte-unigram
baseline by **0.40–0.53 nats** (VERDICT.md §1):

| region | n | best contracting CE | beats unigram by | % empirically expansive |
|---|---|---|---|---|
| inf | 346 | 4.8377 | 0.4022 | 0.0 |
| two_norm_only | 100 | 4.7954 | 0.4445 | 0.0 |
| sdp_only | 189 | 4.7525 | 0.4874 | 0.0 |
| non_certified | 265 | 4.7052 (raw 4.6684) | 0.5347 (raw 0.5715) | 78.9 |

(All figures: VERDICT.md §1.) The baseline is not a strawman: the logistic
readout's zero-feature limit *exactly* recovers the unigram, so any gain is
genuine sequential signal, and the held-out positions are strictly later than
train (no temporal leakage). The TDD suite independently asserts that a
contracting gene found by random search beats the unigram CE
(test_lm.py `test_L0_contracting_gene_beats_unigram`). **L0 holds.**

Figure: best contracting held-out CE per certifier region vs. the unigram
baseline (data: VERDICT.md §1 table).

### 4.5 L1 — admitted genes are stable on the real substrate (PASS)

All three *certified* regions (inf, two_norm, sdp) are **0.0% empirically
expansive** on the real byte-LM — no admitted gene was observed expansive
(VERDICT.md §2). This is the consistency check, not the proof: it confirms the
certifier admitted nothing observably expansive (VERDICT.md §2). The TDD suite
encodes the same property as a regression test — every gene admitted by any
certifier must have empirical contraction `ρ < 1` on the real corpus
(test_lm.py `test_admit_implies_empirical_contraction`), and Lemma 1
(state-boundedness `|s| < 1`) is verified even for deliberately expansive genes
(test_lm.py `test_lemma1_state_bounded_even_for_expansive`). The label
`classify_region` is outcome-blind — a pure function of the Jacobian box, never
reading reservoir states, readout, corpus, or CE — and `held_out_ce` never calls
any certifier, so there is no leakage or circularity; the label is *not*
statistically independent of CE, because both are downstream of the same sampled
`(decay, W)`, which is exactly why the region carries fitness signal
(VERDICT.md §2). **L1 holds.**

### 4.6 L2 — the contraction gate is load-bearing (PASS)

For the gate to do real work the oracle must be non-vacuous: the ungated
population must actually contain expansive genes. It does — the `non_certified`
region is **78.9% empirically expansive** (VERDICT.md §2). So the gate excludes
a large genuinely-expansive population, and certification is load-bearing rather
than vacuously satisfied (VERDICT.md §2). The TDD suite pins this from both
sides: the ungated pool is required to contain a non-trivial fraction of
expansive genes (test_lm.py `test_ungated_pool_has_expansive`, asserts
`expansive/total > 0.05`), and an obviously expansive gene
(`decay = 0`, `W = 2·I`) is rejected by all three certifiers
(test_lm.py `test_obviously_expansive_gene_rejected`). **L2 holds.**

Figure: empirically-expansive fraction, certified regions (0.0%) vs.
non_certified (78.9%) (data: VERDICT.md §2).

### 4.7 What R-LLM establishes, and what it does not

L0/L1/L2 establish that the verified-evolution core *genuinely runs as a real
tiny n=8 byte-LM*: it beats the no-context unigram baseline, its sound
admissions are stable, and the contraction gate excludes a real expansive
population (VERDICT.md §0, §4 "Bottom line"). We deliberately do **not** report
the L3 "verifier-perplexity frontier" as a language-learning result here. The
red-teamed verdict narrows it sharply: under evolution, relaxing the
over-conservative inf gate to a sound relaxation does let evolution reach lower
held-out CE (robust, 10/10 paired seeds, p = 0.000977), but the honest mechanism
is **evolvability / navigability, not language learning** — the inf region
*contains* genes better than unigram that the inf-gated search cannot reach, the
same-corpus region ceilings are roughly equal, and the gate-gap persists on a
shuffled (structureless) corpus at ~107% of its real-run size on the CE scale,
so it is an essentially structure-independent optimization effect (VERDICT.md
§0, §3c). We carry that conclusion at its red-teamed strength and do not inflate
it: L3 is evolvability, not language acquisition.

This positions R-LLM as a faithful but narrow substrate result. It removes the
proxy gap — the verified core is shown to function as an actual LM — without
over-claiming that a stronger verifier unlocks real language learning.

---

**Source files (primary, this section only):**
lm_substrate.py · PREREGISTRATION.md · VERDICT.md · test_lm.py
(all under `research/verified_lm_evolution/`).
