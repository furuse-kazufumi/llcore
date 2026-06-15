# Recurrent LM Interim Summary

This note consolidates the currently tracked head-to-head pilot artifacts for
`feat/lm-recurrent`. It is an audit aid, not a final verdict.

## Tracked artifacts

- [lm_recurrent_pilot120.json](./lm_recurrent_pilot120.json)
- [lm_recurrent_pilot120.md](./lm_recurrent_pilot120.md)
- [lm_recurrent_pilot120.svg](./lm_recurrent_pilot120.svg)
- [lm_recurrent_pilot160.json](./lm_recurrent_pilot160.json)
- [lm_recurrent_pilot160.md](./lm_recurrent_pilot160.md)
- [lm_recurrent_pilot160.svg](./lm_recurrent_pilot160.svg)
- [lm_recurrent_pilot160_seed2026.json](./lm_recurrent_pilot160_seed2026.json)
- [lm_recurrent_pilot160_seed2026.md](./lm_recurrent_pilot160_seed2026.md)
- [lm_recurrent_pilot160_seed2026.svg](./lm_recurrent_pilot160_seed2026.svg)
- [lm_recurrent_pilot256_40.json](./lm_recurrent_pilot256_40.json)
- [lm_recurrent_pilot256_40.md](./lm_recurrent_pilot256_40.md)
- [lm_recurrent_pilot256_40.svg](./lm_recurrent_pilot256_40.svg)

## Capability snapshot

| Run | block_size | max_iters | batch_size | GPT PPL | Recurrent PPL | RWKV PPL | Unigram PPL | Raw order | Strict gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| pilot120 | 64 | 120 | 8 | 209.511 | 217.469 | 197.369 | 215.459 | RWKV < GPT < Recurrent | all fail |
| pilot160 | 64 | 160 | 8 | 184.804 | 180.560 | 158.116 | 215.459 | RWKV < Recurrent < GPT | recurrent/rwkv pass |
| pilot160_seed2026 | 64 | 160 | 8 | 180.897 | 207.580 | 161.650 | 215.459 | RWKV < GPT < Recurrent | gpt/rwkv pass |
| pilot256_40 | 256 | 40 | 4 | 554.407 | 666.469 | 891.822 | 215.058 | GPT < Recurrent < RWKV | all fail |

## Readout

- `pilot120` is the strongest currently tracked sub-`160`-iteration run, but even there all three models fail the strict unigram gate. The raw PPL ordering is therefore provisional.
- `pilot160` and `pilot160_seed2026` improve enough to cross the strict unigram gate for a subset of models, but the ranking is still seed-sensitive: RWKV stays best on raw PPL in both, while GPT and Recurrent swap order and gate status.
- That gate is only the deliberately loose `0.85 x unigram` floor used by the comparison harness. These pilots do not yet clear the stronger "genuinely learned char-LM" bar noted in `held_out_report_any`, so this remains an interim capability signal rather than a settled learning verdict.
- `pilot256_40` uses a different budget and is explicitly a low-fidelity proxy. It is useful only to show that the comparison harness still runs at `block_size=256`.
- The raw ordering changes across runs and seeds, so ranking fidelity is not yet stable enough to claim a full winner.
- The strongest supported claim remains structural rather than capability-based: `RecurrentLM` and `RWKVLM` run with constant-size recurrent state, while GPT memory grows linearly with prompt length.

## Reproduction

- Formal gate command:
  `py -3.11 -m pytest tests/unit -k lm -q && py -3.11 -m mypy src/llcore/lm/ && py -3.11 -m ruff check src/llcore/lm/`
- `pilot120`:
```powershell
@'
from pathlib import Path
from llcore.lm.compare import CompareConfig, compare_on_text
from llcore.lm.data import fetch_aozora_text

compare_on_text(
    fetch_aozora_text(),
    cfg=CompareConfig(
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        state_size=64,
        max_iters=120,
        batch_size=8,
        eval_iters=4,
        throughput_new_tokens=4,
        throughput_repeats=1,
        seed=1337,
    ),
    out_path=Path("docs/artifacts/lm_recurrent_pilot120.json"),
)
'@ | py -3.11 -
```
- `pilot256_40`:
```powershell
@'
from pathlib import Path
from llcore.lm.compare import CompareConfig, compare_on_text
from llcore.lm.data import fetch_aozora_text

compare_on_text(
    fetch_aozora_text(),
    cfg=CompareConfig(
        block_size=256,
        n_layer=2,
        n_head=4,
        n_embd=64,
        state_size=64,
        max_iters=40,
        batch_size=4,
        eval_iters=2,
        throughput_new_tokens=4,
        throughput_repeats=1,
        seed=1337,
    ),
    out_path=Path("docs/artifacts/lm_recurrent_pilot256_40.json"),
)
'@ | py -3.11 -
```
- `pilot160`:
```powershell
@'
from pathlib import Path
from llcore.lm.compare import CompareConfig, compare_on_text
from llcore.lm.data import fetch_aozora_text

compare_on_text(
    fetch_aozora_text(),
    cfg=CompareConfig(
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        state_size=64,
        max_iters=160,
        batch_size=8,
        eval_iters=4,
        throughput_new_tokens=4,
        throughput_repeats=1,
        seed=1337,
    ),
    out_path=Path("docs/artifacts/lm_recurrent_pilot160.json"),
)
'@ | py -3.11 -
```
- `pilot160_seed2026`:
```powershell
@'
from pathlib import Path
from llcore.lm.compare import CompareConfig, compare_on_text
from llcore.lm.data import fetch_aozora_text

compare_on_text(
    fetch_aozora_text(),
    cfg=CompareConfig(
        block_size=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        state_size=64,
        max_iters=160,
        batch_size=8,
        eval_iters=4,
        throughput_new_tokens=4,
        throughput_repeats=1,
        seed=2026,
    ),
    out_path=Path("docs/artifacts/lm_recurrent_pilot160_seed2026.json"),
)
'@ | py -3.11 -
```
