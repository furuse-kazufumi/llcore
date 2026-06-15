# Recurrent LM Head-to-Head

| Model | PPL | Unigram PPL | Ratio vs GPT | Passes unigram floor |
| --- | ---: | ---: | ---: | :---: |
| gpt | 171.710 | 215.459 | 1.000 | yes |
| recurrent | 187.037 | 215.459 | 1.089 | no |
| rwkv | 152.733 | 215.459 | 0.889 | yes |

## Caveats

- At least one model fails the unigram gate; treat this run as undertrained unless a longer schedule confirms the ranking.
- GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements.
