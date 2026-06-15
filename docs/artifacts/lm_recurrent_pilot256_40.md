# Recurrent LM Head-to-Head

| Model | PPL | Unigram PPL | Ratio vs GPT | Passes gate |
| --- | ---: | ---: | ---: | :---: |
| gpt | 554.407 | 215.058 | 1.000 | no |
| recurrent | 666.469 | 215.058 | 1.202 | no |
| rwkv | 891.822 | 215.058 | 1.609 | no |

## Caveats

- All compared models fail the unigram gate; the capability ranking is not yet a publishable head-to-head verdict.
- GPT KV bytes beyond block_size are analytic projection only; recurrent state bytes are executable constant-state measurements.
