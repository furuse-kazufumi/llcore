# Recurrent LM Head-to-Head

| Model | PPL | Unigram PPL | Ratio vs GPT | Passes unigram floor |
| --- | ---: | ---: | ---: | :---: |
| gpt | 180.897 | 215.459 | 1.000 | yes |
| recurrent | 207.580 | 215.459 | 1.148 | no |
| rwkv | 161.650 | 215.459 | 0.894 | yes |

## Caveats

- At least one model fails the unigram gate; treat this run as undertrained unless a longer schedule confirms the ranking.
- GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements.
