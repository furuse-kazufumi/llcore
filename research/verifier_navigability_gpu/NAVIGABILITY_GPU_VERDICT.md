# BG10 / navigability VERDICT — gradient escapes the trap that traps evolution

> **Status (2026-06-05): CONFIRMED on CPU, 6 seeds × 4 gates, real + null** (`result_confirm.json`,
> `result_confirm_null.json`; smoke `bg10_results_smoke.json`). GPU NOT needed — the model is tiny and the
> bottleneck is the CPU certifier, not the LM forward. Kaggle full (8-seed GPU) is an optional cross-check.
> Certs inlined verbatim and self-tested (200/200 parity with `coupled_nd.py`; runtime self-test PASS).

## One-line result
**The navigability penalty of an over-conservative-but-sound verifier is REAL for evolution (random
mutation) but VANISHES under gradient. "Which verifier" matters for evolutionary search; for a
gradient-trained LM the contraction gate is essentially FREE.** The L3 "a looser sound verifier unlocks
lower perplexity" effect is an evolution/random-search artifact (navigability), which gradient sidesteps.

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

## Pre-reg gates
G1 soundness PASS · G2 Q-NAV PASS (admit monotone) · **G3 Q-GRAD PASS (gradient escapes)** · G4 null ties PASS.
