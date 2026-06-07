## 6. Navigability is EA-specific: gradient training escapes the trap (BG10)

A recurring claim from the earlier verified-evolution arc was the L3 "payoff": a *looser but
still sound* verifier appears to unlock lower perplexity than a tighter one. Section 6 tests
the mechanism behind that claim with a gradient-trained, tiny byte/char language model, and
finds that the apparent payoff is an artifact of *how the search moves through the feasible
set* — specifically, of random-mutation evolution — and not a property of the verifier itself.
Under gradient descent the effect vanishes.

### 6.1 Setup

We build a small gated linear-recurrent char LM (`bg10_gpu_lm.py`). Its per-layer state-mixing
core `(decay, W)` is the contraction-*verified*, *evolvable* part; everything else (embedding,
in/out projections `U`/`P`, MLP, LayerNorm, readout) is an ordinary gradient-trained wrapper
(`PREREGISTRATION_BG10.md` §1). The recurrence is
`s_t = decay⊙s_{t-1} + (1-decay)⊙tanh(W·s_{t-1} + x_core)` with `x_core = tanh(U·emb)`, so the
input bound `|x_core|<1` (`max_input_abs=1`) holds by construction (`bg10_gpu_lm.py` lines
202–248). State dimension is kept small (n=8 in the runs reported here) so the contraction
certifiers stay tractable.

Four admission gates are compared, in increasing conservativeness of the *sound* contraction
test they apply to a candidate core: `none` (no gate), `sdp` (SDP / common-Lyapunov),
`two` (induced 2-norm at box vertices), and `inf` (∞-norm sup). The certifiers `cert_inf`,
`cert_two`, `cert_sdp` are inlined verbatim from the arc's `coupled_nd.py` and were self-tested
to 200/200 parity with the originals, plus a runtime soundness self-test (contracting core
admitted, expansive core rejected) (`NAVIGABILITY_GPU_VERDICT.md` header; `bg10_gpu_lm.py`
lines 80–184).

Each gate is run under two optimizer regimes (`PREREGISTRATION_BG10.md` §2):

- **GRAD** — the core is *gradient-trained*; after (every `cert_every`) optimizer steps an
  infeasible core move is rejected and rolled back to the last feasible point (projection by
  reject-and-revert) (`bg10_gpu_lm.py` lines 287–317).
- **EVO** — the core is *evolved* by gated random mutation on top of a gradient-warm-trained,
  then frozen, wrapper (`bg10_gpu_lm.py` lines 319–370).

A **null control** shuffles the corpus to destroy sequential structure
(`bg10_gpu_lm.py` lines 254–263), so a real effect must disappear there.

The result reported here is **CONFIRMED on CPU over 6 seeds × 4 gates, with a matching null
run** (`result_confirm.json`, `result_confirm_null.json`; smoke in `bg10_results_smoke.json`).
GPU was not required: the model is tiny and the bottleneck is the CPU certifier, not the LM
forward pass (`NAVIGABILITY_GPU_VERDICT.md` status header). The 8-seed GPU full run is an
optional cross-check, not a prerequisite for the conclusion (`NAVIGABILITY_GPU_VERDICT.md`
"Honest limits").

### 6.2 Results

The headline numbers (6 seeds, real corpus, `unigram_CE = 3.2512`)
(`NAVIGABILITY_GPU_VERDICT.md` §Evidence):

| gate | GRAD mean CE | EVO mean CE | EVO admit rate | GRAD reject |
|---|---|---|---|---|
| none | 2.4858 | 2.6198 | 1.00 | 0.00 |
| inf  | 2.4886 | 2.6138 | ~0.01 (trapped) | ~0.15 |
| two  | 2.4865 | 2.6167 | ~0.25 | ~0.07 |
| sdp  | 2.4842 | 2.6179 | ~0.58 | ~0.02 |

*Figure: GRAD vs EVO held-out CE and EVO child-admit rate across the four gates (data:
`result_confirm.json` / `NAVIGABILITY_GPU_VERDICT.md`).*

**(G2) Q-NAV — the navigability mechanism is confirmed for evolution.** The EVO child-admit
rate is strongly *monotone in gate strictness*: `inf` admits roughly 1% of mutations (so random
search is almost always blocked and evolution is effectively trapped), `two` ~25%, `sdp` ~58%,
`none` 100% (`NAVIGABILITY_GPU_VERDICT.md` §Evidence). This monotonicity is robust across the 6
seeds and — crucially — is *also present in the null* (`NAVIGABILITY_GPU_VERDICT.md` §Evidence),
which means it is a property of the *geometry of the feasible set* (how rarely a random
mutation lands inside it), not of the task being learned.

**(G3) Q-GRAD — gradient escapes the trap.** GRAD reaches mean CE ≈ 2.485 for *every* gate
(`inf ≈ none ≈ two ≈ sdp` within noise: 2.4886 / 2.4858 / 2.4865 / 2.4842), all well below the
EVO band of ≈ 2.616 (`NAVIGABILITY_GPU_VERDICT.md` §Evidence). Gradient finds feasible descent
directions even inside the very thin `inf`-feasible set: its core reject rate at `inf` is ~0.15
(it does bump against the gate) yet it still converges to the same CE as the ungated case. So
the gate imposes *no perplexity penalty under gradient* (`NAVIGABILITY_GPU_VERDICT.md`
§Evidence, Q-GRAD).

**(G4) Q-PAYOFF + null — learning is real and structure-dependent, not gate-created.** On the
real corpus GRAD's 2.485 is far below the unigram 3.251, i.e. genuine learning, and it is
gate-insensitive. On the null (shuffled) corpus all gates *tie at roughly the unigram level*
(GRAD ≈ 3.309 vs unigram 3.277; EVO ≈ 3.316) (`NAVIGABILITY_GPU_VERDICT.md` §Evidence). With no
sequential structure to learn, there is nothing for any gate to "unlock" — confirming that the
learning seen on the real corpus is real-structure-dependent and gradient-driven, and that the
gate does not create it.

*Figure: real vs null held-out CE per gate, both regimes, showing the null collapse to unigram
(data: `result_confirm.json`, `result_confirm_null.json`).*

**(G1) Soundness.** The runtime cert self-test passed (contracting admitted, expansive
rejected). All gate-admitted cores were empirically contracting (GRAD spectral-radius proxy
ρ in 0.82–0.95, i.e. < 1), and the test is non-vacuous: an ungated (`none`) core on the null
exhibits ρ > 1 (`NAVIGABILITY_GPU_VERDICT.md` §Evidence, Soundness). Pre-registered gates
recorded: G1 PASS, G2 Q-NAV PASS, G3 Q-GRAD PASS, G4 null-ties PASS (`NAVIGABILITY_GPU_VERDICT.md`
§Pre-reg gates).

### 6.3 Interpretation (G1–G4)

The verified-evolution arc had concluded that "SDP is the right verifier" in part because it
admits more contracting genes (coverage) and is *more navigable for evolution*. BG10 adds the
dual to that statement: **the navigability advantage is specific to random-mutation search.**
A gradient-trained model is indifferent to the gate, so one may use the soundest / most
conservative verifier (`inf`) at essentially zero perplexity cost
(`NAVIGABILITY_GPU_VERDICT.md` §Interpretation).

The practical upshot for an llcore realized as a *real* gradient-trained LM (the realistic path
for a Transformer/LM): **pick the verifier purely for soundness and coverage, not for
navigability — gradient handles the rest.** The L3 relaxed-verifier "payoff" does *not* transfer
to gradient-trained LMs; it is an evolution / random-search artifact that gradient sidesteps
(`NAVIGABILITY_GPU_VERDICT.md` §Interpretation, "Practical upshot").

We keep the conclusion's scope where the red-teamed verdict left it: this is about *navigability
of the contraction gate under evolution vs gradient*, i.e. an evolvability/optimization
property of the feasible set — not a statement about language learning per se beyond the
held-out CE measured here.

### 6.4 Honest limits

Within the bounds of what was actually measured (`NAVIGABILITY_GPU_VERDICT.md` §Honest limits):

- The confirmed run is small: n=8 state, 1 layer, d=64 char-LM, on CPU, with the certifier
  checked every `cert_every`=4 steps (8 for smoke) rather than every step.
- In this setup EVO evolves the core *on top of a gradient-warm-trained, then frozen, wrapper*.
  As a result the navigability trap shows up cleanly in the *admit rate* (inf ~1%) but is
  **masked in the final EVO CE**: because the wrapper already carries most of the loss, EVO's CE
  is roughly equal across gates (≈ 2.616) even though `inf` is admit-rate-trapped. A pure-EVO
  setup with no gradient wrapper would be expected to show CE separation (cf. the arc's L3
  `lm_substrate` run). The *gradient-escapes* finding (G3) is unaffected by this and is clean.
- The 8-seed GPU full run (larger config) is an optional confirmation expected to reproduce;
  it is not required for the conclusion.
- We do not generalize beyond the measured range: this evidence speaks to a tiny verified-core
  recurrent LM under contraction gates, and we make no claim about larger models, other
  verifier families, or other optimizers than the two regimes tested
  (`PREREGISTRATION_BG10.md` §0, §3 G5 — honest-negative outcomes are all valid recorded
  results).
