# VNN-COMP `online-arch-evo` benchmark generator (PoC 7a)

Seed-based benchmark instance generator for the proposed `online-arch-evo`
category, following the VNN-COMP benchmark-proposal convention.

## Usage

```bash
py -3.11 scripts/vnncomp_benchmark/generate_properties.py <seed> [--out <dir>] [--n <count>] [--timeout <s>]
# example:
py -3.11 scripts/vnncomp_benchmark/generate_properties.py 42 --out out/vnncomp_sample --n 5
```

Requires the optional `onnx` package (`pip install onnx`) for real `.onnx` writing.

## Output layout

```
<out>/
  onnx/model_<i>.onnx          # RwkvTimeMix-chain parent network (custom op, domain llcore.rwkv)
  vnnlib/invariant_<i>.vnnlib  # |state| <= state_bound given |input| <= max_input_abs (scalar subset)
  changeop/changeop_<i>.jsonl  # per-step ChangeOp stream (category extension)
  instances.csv                # onnx, vnnlib, timeout, changeop  (one row per instance)
```

The `.onnx`/`.vnnlib` files round-trip through the reference implementation's
`parse_model_onnx` / `parse_invariant_vnnlib` and feed `verify_net` directly.

## Honest scope

- `online-arch-evo` extends the standard VNN-COMP I/O contract with a `.changeop`
  stream, so `instances.csv` carries an extra column beyond the standard
  `onnx,vnnlib,timeout`.
- Single `RwkvTimeMix` kernel + `reparam_inplace` family fully supported (matches
  the reference impl's capability matrix).
- This is a **sample generator for the category proposal**, not a finalised
  competition benchmark. See `docs/papers/vnn_comp_online_arch_evolution_proposal.md`.
