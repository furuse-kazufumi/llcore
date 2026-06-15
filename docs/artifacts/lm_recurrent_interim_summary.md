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
- [lm_recurrent_pilot160_seed7.json](./lm_recurrent_pilot160_seed7.json)
- [lm_recurrent_pilot160_seed7.md](./lm_recurrent_pilot160_seed7.md)
- [lm_recurrent_pilot160_seed7.svg](./lm_recurrent_pilot160_seed7.svg)
- [lm_recurrent_pilot240.json](./lm_recurrent_pilot240.json)
- [lm_recurrent_pilot240.md](./lm_recurrent_pilot240.md)
- [lm_recurrent_pilot240.svg](./lm_recurrent_pilot240.svg)
- [lm_recurrent_pilot240_seed2026.json](./lm_recurrent_pilot240_seed2026.json)
- [lm_recurrent_pilot240_seed2026.md](./lm_recurrent_pilot240_seed2026.md)
- [lm_recurrent_pilot240_seed2026.svg](./lm_recurrent_pilot240_seed2026.svg)
- [lm_recurrent_pilot240_seed7.json](./lm_recurrent_pilot240_seed7.json)
- [lm_recurrent_pilot240_seed7.md](./lm_recurrent_pilot240_seed7.md)
- [lm_recurrent_pilot240_seed7.svg](./lm_recurrent_pilot240.svg)
- [lm_recurrent_pilot256_40.json](./lm_recurrent_pilot256_40.json)
- [lm_recurrent_pilot256_40.md](./lm_recurrent_pilot256_40.md)
- [lm_recurrent_pilot256_40.svg](./lm_recurrent_pilot256_40.svg)

Note: the `64/160` and `64/240` memory SVGs are config-dependent rather than seed-dependent, so byte-identical copies can occur across tracked runs. The existing tracked copies are kept for audit continuity, but future additions should prefer a shared SVG reference whenever the generated memory plot would be identical. `pilot240_seed7.svg` already reuses the shared `pilot240.svg` reference in this index for that reason.

## Capability snapshot

| Run | block_size | max_iters | batch_size | GPT PPL | Recurrent PPL | RWKV PPL | Unigram PPL | Raw order | Unigram floor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| pilot120 | 64 | 120 | 8 | 209.511 | 217.469 | 197.369 | 215.459 | RWKV < GPT < Recurrent | all fail |
| pilot160 | 64 | 160 | 8 | 184.804 | 180.560 | 158.116 | 215.459 | RWKV < Recurrent < GPT | recurrent/rwkv pass |
| pilot160_seed2026 | 64 | 160 | 8 | 180.897 | 207.580 | 161.650 | 215.459 | RWKV < GPT < Recurrent | gpt/rwkv pass |
| pilot160_seed7 | 64 | 160 | 8 | 171.710 | 187.037 | 152.733 | 215.459 | RWKV < GPT < Recurrent | gpt/rwkv pass |
| pilot240 | 64 | 240 | 8 | 142.577 | 129.430 | 110.355 | 215.459 | RWKV < Recurrent < GPT | gpt/recurrent/rwkv pass |
| pilot240_seed2026 | 64 | 240 | 8 | 137.109 | 144.936 | 109.534 | 215.459 | RWKV < GPT < Recurrent | gpt/recurrent/rwkv pass |
| pilot240_seed7 | 64 | 240 | 8 | 130.051 | 138.675 | 107.578 | 215.459 | RWKV < GPT < Recurrent | gpt/recurrent/rwkv pass |
| pilot256_40 | 256 | 40 | 4 | 554.407 | 666.469 | 891.822 | 215.058 | GPT < Recurrent < RWKV | all fail |

## Readout

- `pilot120` is the strongest currently tracked sub-`160`-iteration run, but even there all three models fail the unigram floor. The raw PPL ordering is therefore provisional.
- The three `pilot160*` runs improve enough to clear the unigram floor for a subset of models, but the ranking is still seed-sensitive in the GPT-vs-Recurrent slot: RWKV stays best on raw PPL in all three, while GPT and Recurrent swap order and floor status.
- Across those three `64/160` seeds, RWKV is the only model that stays on the same side of the unigram floor every time: RWKV passes `3/3`, GPT passes `2/3`, and Recurrent passes `1/3`.
- The three `pilot240*` runs show the higher-budget direction more clearly: all three models clear the loose `0.85 x unigram` floor in all three seeds, and RWKV stays best on raw PPL in all three. GPT and Recurrent still swap order, so the "best non-RWKV" slot remains seed-sensitive even after the longer schedule.
- Across those three `64/240` seeds, RWKV is stable on both axes that matter for the interim claim: raw PPL best `3/3`, unigram-floor pass `3/3`. GPT and Recurrent both pass the floor in `3/3`, but their relative ordering is not stable.
- That floor is only the deliberately loose `0.85 x unigram` threshold used by the comparison harness. These pilots do not yet clear the stronger "genuinely learned char-LM" bar noted in `held_out_report_any`, so this remains an interim capability signal rather than a settled learning verdict.
- Throughput is secondary evidence only in these tracked pilots. The historical `pilot*` artifacts were captured with `throughput_repeats=1`, so long-prompt marginal decode estimates can be noisy. In one untracked repeat=3 rerun, the previously pathological `pilot240_seed2026` recurrent `prompt_len=256` point moved to `decode_tok_per_s≈8.75` and `total_tok_per_s≈7.93`, but that rerun is not tracked as an artifact and should be treated only as a debugging sanity check, not a committed benchmark datum.
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
- `pilot160_seed7`:
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
        seed=7,
    ),
    out_path=Path("docs/artifacts/lm_recurrent_pilot160_seed7.json"),
)
'@ | py -3.11 -
```
- `pilot240`:
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
        max_iters=240,
        batch_size=8,
        eval_iters=4,
        throughput_new_tokens=4,
        throughput_repeats=1,
        seed=1337,
    ),
    out_path=Path("docs/artifacts/lm_recurrent_pilot240.json"),
)
'@ | py -3.11 -
```
- `pilot240_seed2026`:
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
        max_iters=240,
        batch_size=8,
        eval_iters=4,
        throughput_new_tokens=4,
        throughput_repeats=1,
        seed=2026,
    ),
    out_path=Path("docs/artifacts/lm_recurrent_pilot240_seed2026.json"),
)
'@ | py -3.11 -
```
- `pilot240_seed7`:
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
        max_iters=240,
        batch_size=8,
        eval_iters=4,
        throughput_new_tokens=4,
        throughput_repeats=1,
        seed=7,
    ),
    out_path=Path("docs/artifacts/lm_recurrent_pilot240_seed7.json"),
)
'@ | py -3.11 -
```
