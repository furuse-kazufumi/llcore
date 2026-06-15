# Recurrent LM Head-to-Head

| Model | PPL | Unigram PPL | Ratio vs GPT | Passes unigram floor |
| --- | ---: | ---: | ---: | :---: |
| gpt | 184.804 | 215.459 | 1.000 | no |
| recurrent | 180.560 | 215.459 | 0.977 | yes |
| rwkv | 158.116 | 215.459 | 0.856 | yes |

## Caveats

- At least one model fails the unigram gate; treat this run as undertrained unless a longer schedule confirms the ranking.
- GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements.
