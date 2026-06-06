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

## How to run

Same Kaggle procedure as `../highdim_evolution/README.md` (4 gotchas incl.
`machine_shape: NvidiaTeslaT4`). Feasibility (n=64, E=1504, 2 seeds, B1) → full (E=6000, 4 seeds,
B1+B2 + L0 landscape ×4). Output: `result_m3.json` (checkpointed, resumable).
