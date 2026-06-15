# Recurrent LM Head-to-Head

| Model | PPL | Unigram PPL | Ratio vs GPT | Passes unigram floor |
| --- | ---: | ---: | ---: | :---: |
| gpt | 137.109 | 215.459 | 1.000 | yes |
| recurrent | 144.936 | 215.459 | 1.057 | yes |
| rwkv | 109.534 | 215.459 | 0.799 | yes |

## Caveats

- GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements.
