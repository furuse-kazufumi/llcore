# SPDX-License-Identifier: Apache-2.0
"""VNN-COMP `online-arch-evo` benchmark instance generator (seed-based).

Conforms to the VNN-COMP benchmark-proposal convention (rules.md "Benchmark
Proposals"): a single `generate_properties.py` that takes a random seed as its
only positional argument and emits, deterministically for that seed:

* ``onnx/model_<i>.onnx``        — the parent RwkvTimeMix-chain network
* ``vnnlib/invariant_<i>.vnnlib``— the |state| <= state_bound invariant
* ``changeop/changeop_<i>.jsonl``— the per-step ChangeOp stream (category extension)
* ``instances.csv``              — one row per instance: onnx, vnnlib, timeout[, changeop]

The first three reuse the reference implementation's writers (`write_model_onnx`,
`write_invariant_vnnlib`), so generated files round-trip through
`parse_model_onnx` / `parse_invariant_vnnlib`.

NOTE (honest scope): `online-arch-evo` extends the standard VNN-COMP I/O contract
with a `.changeop` stream; the `instances.csv` therefore carries an extra column.
The `.onnx`/`.vnnlib` halves are standard. Real `.onnx` writing requires the
`onnx` package (optional dep). This is a sample generator for the category
proposal (PoC 7a), not a finalised competition benchmark.

Usage::

    py -3.11 scripts/vnncomp_benchmark/generate_properties.py 42 --out out/vnncomp_sample --n 5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Reuse the reference implementation's writers + clip box.
_HERE = Path(__file__).resolve()
_SCRIPTS = _HERE.parents[1]
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import poc_7a_vnn_comp_reference_impl as ref  # noqa: E402
from llcore.state_update import StateUpdateGene  # noqa: E402

_CLIP = ref.KERNEL_CLIP["RwkvTimeMix"]
_PARAMS = ("decay", "mix", "gate_str")


def _rand_in_clip(rng: np.random.Generator, key: str) -> float:
    lo, hi = _CLIP[key]
    return float(rng.uniform(lo, hi))


def _rand_block(rng: np.random.Generator) -> StateUpdateGene:
    return StateUpdateGene(**{k: _rand_in_clip(rng, k) for k in _PARAMS})


def _rand_changeop(rng: np.random.Generator, step: int) -> dict:
    """A single in-clip reparam_inplace ChangeOp (the fully-supported family)."""
    key = _PARAMS[int(rng.integers(len(_PARAMS)))]
    return {
        "step": step,
        "op": "reparam_inplace",
        "family_id": f"f_{key}_only",
        "target": {"layer": "L0"},
        "args": {key: _rand_in_clip(rng, key)},
    }


def generate(seed: int, out_dir: Path, n_instances: int = 5, timeout_s: float = 60.0) -> Path:
    """Generate ``n_instances`` benchmark instances for ``seed`` into ``out_dir``."""
    rng = np.random.default_rng(seed)
    onnx_dir = out_dir / "onnx"
    vnnlib_dir = out_dir / "vnnlib"
    changeop_dir = out_dir / "changeop"
    for d in (onnx_dir, vnnlib_dir, changeop_dir):
        d.mkdir(parents=True, exist_ok=True)

    rows: list[str] = []
    for i in range(n_instances):
        n_blocks = int(rng.integers(1, 4))  # 1..3 blocks
        net = ref.NetworkState(blocks=[_rand_block(rng) for _ in range(n_blocks)])
        # invariant bounds: state_bound in [0.8, 1.5], max_input_abs in [0.5, 1.0]
        state_bound = float(rng.uniform(0.8, 1.5))
        max_input_abs = float(rng.uniform(0.5, 1.0))

        onnx_path = onnx_dir / f"model_{i}.onnx"
        vnnlib_path = vnnlib_dir / f"invariant_{i}.vnnlib"
        changeop_path = changeop_dir / f"changeop_{i}.jsonl"

        ref.write_model_onnx(net, onnx_path)
        ref.write_invariant_vnnlib(state_bound, max_input_abs, vnnlib_path)

        n_steps = int(rng.integers(3, 8))  # 3..7 ChangeOps
        with changeop_path.open("w", encoding="utf-8") as fh:
            for step in range(1, n_steps + 1):
                fh.write(json.dumps(_rand_changeop(rng, step)) + "\n")

        rows.append(
            f"onnx/{onnx_path.name},vnnlib/{vnnlib_path.name},{timeout_s},changeop/{changeop_path.name}"
        )

    csv_path = out_dir / "instances.csv"
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return csv_path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("seed", type=int, help="random seed (only positional arg, per VNN-COMP convention)")
    p.add_argument("--out", default="out/vnncomp_sample", help="output directory")
    p.add_argument("--n", type=int, default=5, help="number of instances")
    p.add_argument("--timeout", type=float, default=60.0, help="per-instance timeout (seconds)")
    args = p.parse_args(argv)
    out_dir = Path(args.out)
    csv_path = generate(args.seed, out_dir, n_instances=args.n, timeout_s=args.timeout)
    print(f"generated {args.n} instance(s) for seed={args.seed} -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
