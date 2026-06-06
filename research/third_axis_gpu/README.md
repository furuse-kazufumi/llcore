# M3 (③本丸) — is QD/behavioral niching load-bearing on a REAL LLM loss landscape? (GPU)

> Status (2026-06-06): **pre-registered + 3-lens red-teamed (2 blockers fixed & CPU-validated) +
> smoke 14/14 OK; pushing to Kaggle T4 (feasibility → full).**
> Pre-reg: `PREREGISTRATION_M3.md` (incl. pre-run amendments §7). Kernel: `m3_kernel.py`
> (Stage-B scaffold lineage). The last open path of the ③ arc after BG9 closed every CPU route.

## Question

MAP-Elites (behavioral-archive niching = ③) vs direct-sampling baselines (random / warm-restart
RR-hillclimb / panmictic GA) at equal eval budget, on the **real held-out-CE landscape of the
Stage-B hybrid Transformer's verified-core subspace** (n=64 ⇒ 4,160 dims, trunk frozen after
gradient warm-up). Adam-on-core reported as the gradient strong baseline (not a ③ judge).
BG9's pre-registered hypothesis: ③ needs difficulty in high-dim behavior space unreachable by
direct sampling — the full-LLM core landscape is the first substrate in the arc that could host it.

## Validity machinery (judged before the real verdict)

- **P+** = Step4 exp2 `deceptive_eval` faithfully transplanted onto a fixed 20-entry W probe
  (unit window, Gaussian local/global, observation noise 0.01, honest noiseless rescore).
  **CPU-validated 4/4 pre-GPU** (MAP-E reaches the corner 0.92–1.00; M1/M2/M3 stall ≤0.60) at
  E_synth=20,000 — after TWO failed corridor designs (full-mean CLT-frozen; V-valley breaks the
  archive ratchet) caught by the red-team + validation loop. The harness can detect ③.
- **N0** = smooth concave control (false-positive guard).
- **L0 landscape suite** (user-requested GPU extension): 1,000 genomes/seed (random + σ-sweep
  warm-perturbations) → real CE + behavior coords. Mechanism evidence: direct-sampling difficulty +
  local smoothness of the real landscape.

## Red-team summary (pre-push)

2 blockers (P+ untraversable at n=64 — guaranteed N/A; B1 archive collapsed to 2–6/64 bins),
3 majors (cold-restart RR not the BG9 baseline → warm-restart; descriptor text/code drift; 1-of-2
descriptor multiplicity → B1 primary, B2 secondary), minors (GA budget parity, B2 empty-octile
imputation + std-adaptive scaling, M5 model restore, jitter floor). All fixed and re-validated
before any GPU run; full trail in `PREREGISTRATION_M3.md §7`.

## VERDICT (full run, Kaggle T4, 60/60 ok, 8192 s, 2026-06-06): ③ NEGATIVE (decisive)

n=64, E=6000, 4 seeds; `result_m3_full.json`. unigram CE 3.3987.

| gate | result |
|---|---|
| **P+ (deceptive corridor, B1p)** | **VALID 4/4** — MAP-E reaches the corner (0.982–0.999); M1=0.336, M2=0.598, M3=0.600 stall at/below the local optimum ⇒ the harness CAN detect ③ |
| **N0 (smooth concave)** | **false-positive 0/4** ⇒ the harness does not hallucinate a niching win |
| **REAL (B1, primary)** | **M1=M2=M3=M4 bit-identical** (CE 1.8439, all four stuck at the warm core), M4 wins **0/4** |
| **REAL (B2, secondary)** | identical — M4/B2 wins 0/4 |
| **M5 (gradient)** | the only mover: CE 1.844 → **1.562** |
| **L0 landscape** | random **0/600** beat the warm core; every perturbation direction is uphill (σ 0.03→1.0 ⇒ CE 1.81→2.46) — a smooth unimodal **bowl** with zero deception |

**Conclusion: behavioral niching (③ / MAP-Elites) is NOT load-bearing on the real small-LLM core
loss landscape.** With a harness proven able to detect ③ (P+ valid) and proven not to false-positive
(N0 null), every selection method — random, warm-restart RR, panmictic GA, MAP-Elites under two
pre-registered descriptors — is pinned at the warm core; only gradient descends. The L0 map explains
why: the landscape is a smooth unimodal bowl (no local optima to escape, no stepping-stones to chain),
exactly the structure under which BG9 predicted ③ cannot help (difficulty must live in a high-dim
behavior space unreachable by direct sampling — absent here). This closes the ③ arc's last path
(after Step4 synthetic-positive / Step C-D real-CPU / BG9 kernel-union) with the same 3-stage
evidence shape: valid positive control + null negative control + decisive real negative.

**Implication for the terrain-design thread (E/F/G):** since the *raw* next-token-CE landscape hosts
no deception, making ③ stand requires *engineering* a multi-modal terrain on top of the real LM — the
verifier-shell / reasoning-chain / riddle / shiritori ideas. M3 NEGATIVE is the empirical motivation
for that thread: the bowl must be deformed, not discovered.

**Honest scope:** core subspace (4,160 dims), not full weight space; B1+B2 descriptors; E=6000; n=64;
4 seeds (sign-consistency, no p-theater). M1=M2=M3=M4 being *bit-identical* means σ=0.12 mutation never
once improved the warm core — consistent with the L0 bowl, but a different σ / a non-warm start could
move (untested; the warm-start + this σ is the pre-registered setup).

## How to run

Same Kaggle procedure as `../highdim_evolution/README.md` (4 gotchas incl.
`machine_shape: NvidiaTeslaT4`). Feasibility (n=64, E=1504, 2 seeds, B1) → full (E=6000, 4 seeds,
B1+B2 + L0 landscape ×4). Output: `result_m3.json` (checkpointed, resumable).
