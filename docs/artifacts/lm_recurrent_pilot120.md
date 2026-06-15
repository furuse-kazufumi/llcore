# Recurrent LM Head-to-Head

| Model | PPL | Unigram PPL | Ratio vs GPT | Passes unigram floor |
| --- | ---: | ---: | ---: | :---: |
| gpt | 209.511 | 215.459 | 1.000 | no |
| recurrent | 217.469 | 215.459 | 1.038 | no |
| rwkv | 197.369 | 215.459 | 0.942 | no |

## Caveats

- All compared models fail the unigram gate; the capability ranking is not yet a publishable head-to-head verdict.
- GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements.
