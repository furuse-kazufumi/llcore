# SPDX-License-Identifier: Apache-2.0
"""**静的ランタイム床**(static RSS floor)スイープ — a9「最適化する項を間違えていた」の
load-bearing な数値(Python baseline / `import torch` 税 / モデルロード後 RSS)を、
**再走可能なコミット済みハーネス**として固める。

``recurrent_runtime_rss.py``(動的=文脈長依存メモリ)と ``recurrent_latency_sweep.py``
(compute 軸)に続く 3 本目で、こちらは **文脈長に依存しない静的な土台**を測る。

測るもの(各ステージを別プロセスで隔離して RSS を読む)
------------------------------------------------------
- **python**: torch も何も重い import をしていない素の Python プロセスの RSS。
- **torch**: ``import torch`` した直後の RSS。差分 = **言語ランタイム税**。
- **model**: torch + 豆 CharGPT を 1 個構築した後の RSS。差分 = モデル本体の常駐増。

さらに model ステージでは ``int8_footprint_bytes`` で **量子化後の重み実体(MB)** を出し、
プロセス RSS が重み本体の何倍か(=「足場」の比)を出す。

honest 留保
-----------
- RSS は OS の WorkingSetSize 実測(Windows は ``GetProcessMemoryInfo``、非 Windows は
  ``/proc/self/statm`` フォールバック、取得失敗時 0)。プロセス生涯の最大ではなく **測定時点**の値。
- 各ステージは **別プロセス**で測る(同一プロセスでは torch を un-import できないため)。
- int8=○MB は **その config の例示**。a9 本文の 1.51MB は別の特定 config の過去値で、ここで再現は
  狙わない。再現・検証の対象は **config 非依存の torch 税(~184MB)と baseline(~13MB)** の方。

使い方::

    py -3.11 scripts/runtime_floor_rss.py
    py -3.11 scripts/runtime_floor_rss.py --n-embd 176 --n-layer 4 --repeats 3
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any, cast

from llcore.runtime.rss import working_set_bytes as _working_set_bytes

RESULT_PREFIX = "RESULT_JSON="
STAGES = ("python", "torch", "model")


def run_stage(stage: str, n_embd: int, n_layer: int, n_head: int, vocab: int) -> dict[str, Any]:
    """1 ステージ分の RSS(と model なら int8 実体)を測る(隔離 subprocess 内)。"""
    result: dict[str, Any] = {"stage": stage}
    if stage == "python":
        pass  # 重い import をしない素の状態
    elif stage == "torch":
        import torch  # noqa: F401
    else:  # model
        import torch  # noqa: F401

        from llcore.lm.model import CharGPT, GPTConfig
        from llcore.lm.quant import int8_footprint_bytes

        torch.manual_seed(1337)
        model = CharGPT(GPTConfig(vocab_size=vocab, block_size=256, n_layer=n_layer,
                                  n_head=n_head, n_embd=n_embd))
        model.eval()
        fb = int8_footprint_bytes(cast(Any, model))
        result["int8_mb"] = round(fb["int8_bytes"] / 1e6, 3)
        result["fp32_mb"] = round(fb["fp32_bytes"] / 1e6, 3)
        result["params_m"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 3)
    result["rss_mb"] = round(_working_set_bytes() / 1e6, 1)
    return result


def _spawn_stage(stage: str, args: argparse.Namespace) -> dict[str, Any]:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--worker", stage,
           "--n-embd", str(args.n_embd), "--n-layer", str(args.n_layer),
           "--n-head", str(args.n_head), "--vocab", str(args.vocab)]
    import subprocess

    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"stage {stage} failed (rc={proc.returncode}): {proc.stderr.strip()[:300]}")
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith(RESULT_PREFIX):
            return cast("dict[str, Any]", json.loads(line[len(RESULT_PREFIX):]))
    raise RuntimeError(f"stage {stage} produced no result line")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="static runtime RSS floor: python / +torch / +model")
    ap.add_argument("--n-embd", type=int, default=176)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--n-head", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--json", default="out/runtime_floor_rss.json")
    ap.add_argument("--worker", choices=list(STAGES), default=None)
    args = ap.parse_args(argv)

    # --- worker role ---
    if args.worker is not None:
        result = run_stage(args.worker, args.n_embd, args.n_layer, args.n_head, args.vocab)
        print(RESULT_PREFIX + json.dumps(result, ensure_ascii=False))
        return 0

    # --- parent role ---
    if args.repeats < 1:
        print("error: --repeats must be >= 1", file=sys.stderr)
        return 2

    print(f"config: n_embd={args.n_embd} L={args.n_layer} H={args.n_head} vocab={args.vocab} "
          f"repeats={args.repeats}")
    # 各ステージを repeats 回隔離測定し中央値を採る(RSS のラン間ノイズを抑制)。
    per_stage: dict[str, list[dict[str, Any]]] = {s: [] for s in STAGES}
    for _ in range(args.repeats):
        for stage in STAGES:
            per_stage[stage].append(_spawn_stage(stage, args))

    summary: dict[str, dict[str, Any]] = {}
    for stage in STAGES:
        runs = per_stage[stage]
        rss_median = round(statistics.median(r["rss_mb"] for r in runs), 1)
        entry: dict[str, Any] = {"rss_mb_median": rss_median, "rss_mb_runs": [r["rss_mb"] for r in runs]}
        if stage == "model":
            entry["int8_mb"] = runs[-1]["int8_mb"]
            entry["fp32_mb"] = runs[-1]["fp32_mb"]
            entry["params_m"] = runs[-1]["params_m"]
        summary[stage] = entry

    py_rss = summary["python"]["rss_mb_median"]
    torch_rss = summary["torch"]["rss_mb_median"]
    model_rss = summary["model"]["rss_mb_median"]
    int8_mb = summary["model"]["int8_mb"]
    torch_tax = round(torch_rss - py_rss, 1)
    scaffold_ratio = round(model_rss / int8_mb, 1) if int8_mb > 0 else float("nan")

    print("\n| stage | RSS (MB, median) |")
    print("|---|---|")
    print(f"| python (素) | {py_rss} |")
    print(f"| + import torch | {torch_rss} |")
    print(f"| + 豆モデル | {model_rss} |")
    print(f"\n[headline] 言語ランタイム税(torch - python)= {torch_tax} MB。 "
          f"豆モデルの int8 重み実体 = {int8_mb} MB。 "
          f"プロセス RSS {model_rss}MB は重み本体の {scaffold_ratio}x(=足場が支配)。")
    print("[honest] RSS は測定時点の WorkingSetSize 実測。int8 MB は当 config の例示("
          "a9 本文の 1.51MB は別 config の過去値)。検証対象は config 非依存の torch 税と baseline。")

    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "config": {"n_embd": args.n_embd, "n_layer": args.n_layer, "n_head": args.n_head,
                   "vocab": args.vocab, "repeats": args.repeats},
        "stages": summary,
        "torch_tax_mb": torch_tax,
        "scaffold_ratio": scaffold_ratio,
    }
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
