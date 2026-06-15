# Recurrent LM Head-to-Head

| Model | PPL | Unigram PPL | Ratio vs GPT | Passes gate |
| --- | ---: | ---: | ---: | :---: |
| gpt | 142.577 | 215.459 | 1.000 | yes |
| recurrent | 129.430 | 215.459 | 0.908 | yes |
| rwkv | 110.355 | 215.459 | 0.774 | yes |

## Caveats

- GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements.
