# SPDX-License-Identifier: Apache-2.0
"""Build + preflight a local Kaggle llcore.lm.compare bundle without pushing it.

This wrapper keeps all work local:

1. Build a deterministic bundle from a local corpus file.
2. Run local preflight checks (optionally with runner smoke).
3. Print the exact human-gated `kaggle kernels push -p ...` command to use later.

It never contacts Kaggle by itself.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
BUILD_SCRIPT = HERE / "build_kaggle_lm_compare_bundle.py"
PREFLIGHT_SCRIPT = HERE / "kaggle_bundle_preflight.py"
DEFAULT_RUNNER_TIMEOUT = 300


def _load_script(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Build and locally preflight a Kaggle llcore.lm.compare bundle. "
            "Does not push or publish anything."
        )
    )
    ap.add_argument("--bundle-dir", required=True, help="output directory for the Kaggle bundle")
    ap.add_argument("--corpus-file", required=True, help="UTF-8 corpus file to embed")
    ap.add_argument("--kernel-id", default="furusekazufumi/llcore-lm-compare")
    ap.add_argument("--title", default="llcore-lm-compare")
    ap.add_argument("--machine-shape", default="NvidiaTeslaT4")
    ap.add_argument("--enable-gpu", action="store_true", help="emit GPU-enabled Kaggle metadata")
    ap.add_argument("--enable-internet", action="store_true", help="emit internet-enabled Kaggle metadata")
    ap.add_argument("--public", action="store_true", help="emit public Kaggle metadata")
    ap.add_argument("--block-size", type=int, default=64)
    ap.add_argument("--n-layer", type=int, default=2)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--n-embd", type=int, default=64)
    ap.add_argument("--state-size", type=int, default=64)
    ap.add_argument("--max-iters", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--eval-iters", type=int, default=4)
    ap.add_argument("--throughput-new-tokens", type=int, default=16)
    ap.add_argument("--throughput-repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--run-runner", action="store_true", help="run bundled runner locally after build")
    ap.add_argument(
        "--runner-timeout",
        type=int,
        default=DEFAULT_RUNNER_TIMEOUT,
        help="runner timeout seconds (default: 300)",
    )
    ap.add_argument("--json", help="optional combined report path")
    return ap


def prepare_bundle(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.runner_timeout < 1:
        print("error: --runner-timeout must be >= 1", file=sys.stderr)
        return 2
    build_mod = _load_script(BUILD_SCRIPT, "build_kaggle_lm_compare_bundle")
    preflight_mod = _load_script(PREFLIGHT_SCRIPT, "kaggle_bundle_preflight")

    build_argv = [
        "--bundle-dir",
        args.bundle_dir,
        "--corpus-file",
        args.corpus_file,
        "--kernel-id",
        args.kernel_id,
        "--title",
        args.title,
        "--machine-shape",
        args.machine_shape,
        "--block-size",
        str(args.block_size),
        "--n-layer",
        str(args.n_layer),
        "--n-head",
        str(args.n_head),
        "--n-embd",
        str(args.n_embd),
        "--state-size",
        str(args.state_size),
        "--max-iters",
        str(args.max_iters),
        "--batch-size",
        str(args.batch_size),
        "--eval-iters",
        str(args.eval_iters),
        "--throughput-new-tokens",
        str(args.throughput_new_tokens),
        "--throughput-repeats",
        str(args.throughput_repeats),
        "--seed",
        str(args.seed),
    ]
    if args.enable_gpu:
        build_argv.append("--enable-gpu")
    if args.enable_internet:
        build_argv.append("--enable-internet")
    if args.public:
        build_argv.append("--public")

    build_rc = int(build_mod.main(build_argv))
    if build_rc != 0:
        return build_rc

    try:
        preflight_report = preflight_mod.preflight_bundle(
            Path(args.bundle_dir),
            run_runner=args.run_runner,
            runner_timeout=args.runner_timeout,
        )
    except (ValueError, OSError, preflight_mod.subprocess.TimeoutExpired) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    combined_report = {
        "bundle_dir": str(Path(args.bundle_dir).resolve()),
        "kernel_id": args.kernel_id,
        "push_command": f'kaggle kernels push -p "{Path(args.bundle_dir).resolve()}"',
        "preflight": preflight_report,
    }
    if args.json:
        Path(args.json).write_text(
            json.dumps(combined_report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    print(
        "[kaggle-prepare]",
        f"dir={Path(args.bundle_dir).resolve()}",
        f"runner={'yes' if args.run_runner else 'no'}",
    )
    if args.run_runner:
        print(
            "[note]",
            "runner smoke uses local CPU training and may still require a larger "
            "--runner-timeout for heavier configs.",
        )
    print("[next]", combined_report["push_command"])
    return 0


def main(argv: list[str] | None = None) -> int:
    return prepare_bundle(argv)


if __name__ == "__main__":
    raise SystemExit(main())
