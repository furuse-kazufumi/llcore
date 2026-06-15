# Recurrent LM Head-to-Head

| Model | PPL | Unigram PPL | Ratio vs GPT | Passes unigram floor |
| --- | ---: | ---: | ---: | :---: |
| gpt | 130.051 | 215.459 | 1.000 | yes |
| recurrent | 138.675 | 215.459 | 1.066 | yes |
| rwkv | 107.578 | 215.459 | 0.827 | yes |

## Caveats

- GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements.
