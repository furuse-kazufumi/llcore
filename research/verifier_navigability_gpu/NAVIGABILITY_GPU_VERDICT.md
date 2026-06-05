# BG10 / navigability VERDICT — gradient escapes the trap that traps evolution

> **Status (2026-06-05): CONFIRMED on CPU, 6 seeds × 4 gates, real + null** (`result_confirm.json`,
> `result_confirm_null.json`; smoke `bg10_results_smoke.json`). GPU NOT needed — the model is tiny and the
> bottleneck is the CPU certifier, not the LM forward. **Kaggle full (8-seed GPU, Tesla T4) CONFIRMED
> 2026-06-06 — reproduces every gate; see "GPU full cross-check" below (`result_full.json` / `result_full_null.json`).**
> Certs inlined verbatim and self-tested (200/200 parity with `coupled_nd.py`; runtime self-test PASS).

## One-line result
**The navigability penalty of an over-conservative-but-sound verifier is REAL for evolution (random
mutation) but VANISHES IN FINAL CE under gradient. "Which verifier" matters for evolutionary search;
for a gradient-trained LM the contraction gate is essentially free *in held-out CE*.** The L3 "a looser
sound verifier unlocks lower perplexity" effect is an evolution/random-search artifact (navigability),
which gradient sidesteps.

> **Scope / honesty (Codex pair-review 2026-06-06, adopted).** Two precisions on the above: (1) the EVO
> trap shown here is in the **child-admit rate** (inf ~1% vs none 100%), **not** in the final EVO CE —
> the gated EVO mean CE is nearly tied and non-monotone across gates (`inf 2.6138 ≈ two 2.6167 ≈ sdp
> 2.6179 ≈ none 2.6198`; `inf` is even fractionally best), because the gradient-warm-trained frozen
> wrapper carries most of the loss (see Honest limits). The clean CE separation lives in the pure-EVO L3
> `lm_substrate` run, not here. (2) "free under gradient" means **free in final CE**, not zero
> interaction: GRAD's per-step reject rate IS gate-dependent (`none 0.000 / sdp 0.0125 / two 0.0667 /
> inf 0.1514`) — gradient bumps the gate ~15% of steps at `inf` yet still converges to the same CE. (3)
> This is one tiny n=8, 1-layer, 6-seed CPU setup; "pick the verifier on soundness/coverage, not
> navigability" is the practical reading **for gradient-trained LMs at this scale on final CE**, not a
> proven law for all gradient-trained LMs.

## Evidence (6 seeds, real corpus, unigram_CE=3.2512)
| gate | GRAD mean CE | EVO mean CE | **EVO admit rate** | GRAD reject |
|---|---|---|---|---|
| none | 2.4858 | 2.6198 | 1.00 | 0.00 |
| inf  | 2.4886 | 2.6138 | **~0.01 (trapped)** | ~0.15 |
| two  | 2.4865 | 2.6167 | ~0.25 | ~0.07 |
| sdp  | 2.4842 | 2.6179 | ~0.58 | ~0.02 |

- **Q-NAV (navigability mechanism) — CONFIRMED.** EVO child-admit rate is strongly **monotone in gate
  strictness**: inf ~1% (mutations almost never pass → evolution trapped) → two ~25% → sdp ~58% → none
  100%. Robust across 6 seeds AND present in the null (structure-independent → it is the *geometry* of the
  feasible set, not the task).
- **Q-GRAD (the killer question) — gradient ESCAPES.** GRAD reaches ~2.485 for **every** gate (inf = none
  = two = sdp, within noise), all ≪ EVO ~2.616. Gradient finds feasible descent directions inside even the
  tiny inf-feasible set (reject ~15% but still converges). So the verifier gate imposes **no perplexity
  penalty under gradient**.
- **Q-PAYOFF + null — CONFIRMED.** Real: GRAD 2.485 ≪ unigram 3.251 (genuine learning), gate-insensitive.
  **Null (shuffled): all gates tie at ~unigram** (GRAD ~3.309 ≈ unigram 3.277; EVO ~3.316) = no structure
  to learn. So the learning is real-structure-dependent and gradient-driven; the gate does not create it.
- **Soundness (G1).** Runtime cert self-test PASS (contracting admitted, expansive rejected). All
  gate-admitted cores empirically contracting (GRAD rho 0.82–0.95 < 1). Non-vacuous (ungated `none` on null
  shows rho > 1).

## Interpretation (connects the arc)
- The verified-evolution arc concluded "SDP is the right verifier" partly because it admits more
  contracting genes (coverage) and is more *navigable* for **evolution**. This BG10 result adds the dual:
  **that navigability advantage is specific to random-mutation search. A gradient-trained model is
  indifferent to the gate** — you may use the soundest/most-conservative verifier (inf) at zero cost.
- Practical upshot for llcore-as-real-LM: if the core is **gradient-trained** (the realistic path for a
  real Transformer/LM), pick the verifier purely for soundness/coverage, not for navigability — gradient
  handles the rest. The L3 "payoff" does **not** transfer to gradient-trained LMs.

## Honest limits
- n=8, 1 layer, tiny d=64 char-LM; CPU; cert_every=4 (confirm) / 8 (smoke), not every-step.
- EVO here evolves the core on top of a **gradient-warm-trained wrapper**, so the navigability trap shows
  cleanly in *admit-rate* but is **masked in final CE** (the wrapper carries most of the loss; EVO CE is
  ~equal across gates despite inf being trapped). A pure-EVO (no gradient wrapper) setup would show CE
  separation (cf. the L3 lm_substrate run). The *gradient-escapes* finding is unaffected and clean.
- Optional confirmation: Kaggle full (8 seeds, GPU forward, larger config) — expected to reproduce; not
  required for the conclusion.

## GPU full cross-check (Kaggle Tesla T4, 8 seeds × 4 gates, 2026-06-06) — CONFIRMED

Independent reproduction on GPU with a stronger config (grad_steps 400, evo_gens 150, d=64, T=64,
cert_every=3; `result_full.json` / `result_full_null.json`, pulled via `kaggle kernels output`). Absolute
CE differs from the CPU run (better config ⇒ lower CE; unigram 3.354 vs CPU 3.251, GRAD ~2.20 vs ~2.485),
but **every gate-level conclusion reproduces**:

| | gate | GRAD CE | GRAD reject | EVO CE | **EVO admit** | gated sound-viol |
|---|---|---|---|---|---|---|
| **REAL** (unigram 3.3542) | none | 2.204 | 0.000 | 2.287 | 1.000 | 0 |
| | inf | 2.237 | 0.270 | 2.304 | **0.007 (trapped)** | 0 |
| | two | 2.222 | 0.185 | 2.293 | 0.087 | 0 |
| | sdp | 2.205 | 0.028 | 2.286 | 0.736 | 0 |
| **NULL** (unigram 3.3021) | none | 3.316 | 0.000 | 3.314 | 1.000 | 0 |
| | inf | 3.317 | 0.172 | 3.317 | **0.002** | 0 |
| | two | 3.317 | 0.165 | 3.316 | 0.085 | 0 |
| | sdp | 3.317 | 0.157 | 3.315 | 0.343 | 0 |

- **G2 Q-NAV reproduced**: EVO admit-rate monotone in gate strictness (inf ~0.7% → none 100%), present
  in BOTH real and null (structure-independent geometry).
- **G3 Q-GRAD reproduced**: GRAD reaches the same final CE across all gates (2.20–2.24, all ≪ unigram
  3.354), i.e. gate-indifferent in final loss — though GRAD reject is gate-dependent (inf 0.27) and it
  still converges (the Codex-noted nuance, confirmed at scale).
- **EVO CE masked reproduced**: EVO CE is gate-insensitive (~2.29), so the trap shows in admit-rate, not
  final CE (gradient-warm wrapper), exactly as the CPU run / honest-caveat predicted.
- **G4 null ties reproduced**: on the shuffled corpus all gates' GRAD & EVO CE collapse to ~unigram.
- **Soundness reproduced**: 0 violations on gated cores (real + null); ungated `none` is non-vacuous.

**Net: the 8-seed GPU full run independently confirms the 6-seed CPU BG10 conclusion on every pre-reg
gate.** (Honest: absolute CE values are config-specific, not identical; the *qualitative* conclusion and
the gate ordering are.)

## Pre-reg gates
G1 soundness PASS · G2 Q-NAV PASS (admit monotone) · **G3 Q-GRAD PASS (gradient escapes)** · G4 null ties
PASS. **GPU full cross-check (Kaggle T4, 8 seeds): all four reproduced (2026-06-06).**
