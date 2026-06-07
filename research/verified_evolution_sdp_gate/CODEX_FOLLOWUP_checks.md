# Codex follow-up checks — measured evidence for the fixes (2026-06-03)

Numbers behind the VERDICT.md corrections for Codex findings #1, #4, #8.

## #1 — proper G1 on ADMITTED CHILDREN @20 000 samples (rotation, 150 unique children/gate)

| gate | unique children checked | divergent @20k |
|---|---|---|
| inf_norm | 150 | **0** |
| two_norm | 150 | **0** |
| sdp | 150 | **0** |

Consistent with the certifier theorems (the soundness guarantee). The default-run G1 used
the final population @2500; this is the preregistered admitted-child test at the higher rate.

## #4 — fallback / rejection pressure (mean per run, 15 seeds)

| task | inf_norm [rej, fb] | two_norm | sdp |
|---|---|---|---|
| rotation | [373, **0.5**] | [422, 0.5] | [318, 0.1] |
| benign | [289, 0.5] | [252, 0.6] | [167, 0.1] |
| nonnormal | [257, 0.6] | [212, 0.6] | [143, 0.1] |

inf-gated rotation rejects ~373 children/run but **falls back only ~0.5×/run** ⇒ NOT
fallback-flooded; the low inf rotation fitness is genuine inf-region search hitting the
0.38 ceiling, not a fallback artifact.

## #8 — contracting-filtered per-region max (rotation, 3000 random genes)

| region | n | n_contracting | max_raw | max_contracting |
|---|---|---|---|---|
| inf | 406 | 406 | 0.276 | 0.276 |
| two_norm_only | 164 | 164 | 0.493 | 0.493 |
| sdp_only | 201 | 201 | 0.590 | 0.590 |
| non_certified | 2229 | 396 | 0.595 | **0.571** |

(Absolute values are lower than the 10175-gene exp1 because this pool is random-only, no
GA-visited genes.) The point: `non_certified` raw max (0.595) drops to 0.571 when filtered
to empirically-contracting genes — some high-fitness `non_certified` genes are divergent,
confirming Codex #8. The certified regions (inf/two/sdp) are contracting by construction.

Reproduce: the inline script under the VERDICT §3/§4/§5 corrections, seed=0/11, py 3.11.
