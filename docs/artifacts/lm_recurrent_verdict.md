# Recurrent LM Verdict

This note is the current head-to-head verdict packet for `feat/lm-recurrent`.
It is based only on tracked artifacts in this directory.

## Evidence Base

- `64/160` seeds:
  - [pilot160](./lm_recurrent_pilot160.json)
  - [pilot160_seed2026](./lm_recurrent_pilot160_seed2026.json)
  - [pilot160_seed7](./lm_recurrent_pilot160_seed7.json)
- `64/240` seeds:
  - [pilot240](./lm_recurrent_pilot240.json)
  - [pilot240_seed2026](./lm_recurrent_pilot240_seed2026.json)
  - [pilot240_seed7](./lm_recurrent_pilot240_seed7.json)
- low-fidelity long-window proxy:
  - [pilot256_40](./lm_recurrent_pilot256_40.json)
- memory curves:
  - [64/160 memory@T](./lm_recurrent_pilot160.svg)
  - [64/240 memory@T](./lm_recurrent_pilot240.svg)
  - [256/40 memory@T](./lm_recurrent_pilot256_40.svg)

## Capability Table

| Budget | Seeds | RWKV raw-PPL best | RWKV unigram-floor pass | GPT/Recurrent ordering stable? |
| --- | ---: | :---: | :---: | :---: |
| `64/160` | 3 | yes (`3/3`) | yes (`3/3`) | no |
| `64/240` | 3 | yes (`3/3`) | yes (`3/3`) | no |
| `256/40` | 1 | no | no | n/a |

Representative tracked rows:

| Run | GPT PPL | Recurrent PPL | RWKV PPL | Raw order | Unigram floor |
| --- | ---: | ---: | ---: | --- | --- |
| `pilot160` | 184.804 | 180.560 | 158.116 | `RWKV < Recurrent < GPT` | `recurrent/rwkv pass` |
| `pilot160_seed2026` | 180.897 | 207.580 | 161.650 | `RWKV < GPT < Recurrent` | `gpt/rwkv pass` |
| `pilot240` | 142.577 | 129.430 | 110.355 | `RWKV < Recurrent < GPT` | `gpt/recurrent/rwkv pass` |
| `pilot240_seed2026` | 137.109 | 144.936 | 109.534 | `RWKV < GPT < Recurrent` | `gpt/recurrent/rwkv pass` |

## Memory Verdict

- `RecurrentLM` and `RWKVLM` use constant-size recurrent state.
- GPT memory grows linearly with prompt length via analytic KV projection.
- The tracked memory curves already show the decisive structural split:
  - recurrent/RWKV: flat state bytes
  - GPT: linear bytes/token slope
- `64/160` and `64/240` share the same memory curve because the config is identical.

## Current Verdict

- The strongest supported capability claim is: **RWKV is the most reproducible current candidate**.
- Basis:
  - raw PPL best in `6/6` tracked seeds across the two matched budgets (`64/160`, `64/240`)
  - unigram-floor pass in `6/6` tracked seeds across the same runs
  - GPT and Recurrent continue to swap second place, so the non-RWKV ranking is not stable
- The strongest supported systems claim is: **recurrent models buy the intended memory property**.
  - constant-size recurrent state for `RecurrentLM` and `RWKVLM`
  - GPT remains structurally bounded by `block_size` for exact execution and linear KV projection beyond that

## Non-Claims

- This is **not** a final winner declaration across all budgets or seeds.
- The `0.85 x unigram` threshold is only a loose unigram floor, not the stronger "genuinely learned char-LM" bar from `held_out_report_any`.
- Throughput is secondary evidence only.
  - tracked pilot artifacts were captured with `throughput_repeats=1`
  - long-prompt marginal decode estimates can therefore be noisy
- `pilot256_40` is a low-fidelity proxy only and should not be mixed into the matched-budget winner claim.
