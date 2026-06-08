# SPDX-License-Identifier: Apache-2.0
"""internalization 本走 kernel を Kaggle 無料 T4 へ push (windowless 単一プロセス).

committed `internalization_kernel.py` の RUN_N/RUN_STAGE default だけを書換えた派生ファイルを push
(実験ロジックは不変; 環境変数を Kaggle script kernel で渡せないため default 置換で n/stage を固定)。

実行::  py -3.11 push_internalization.py <stage> <n> [<n> ...]
  例:   py -3.11 push_internalization.py 1 64 128 256     # stage-1 (first-check) 3 本
        py -3.11 push_internalization.py 2 64             # stage-2 (confirmatory) n=64
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent
USER = "furusekazufumi"


def push(stage: int, n: int):
    api = KaggleApi(); api.authenticate()
    src = (HERE / "internalization_kernel.py").read_text(encoding="utf-8")
    # default 置換のみ (構造不変; future-import を壊さない)
    src = src.replace('RUN_N = int(os.environ.get("RINT_N", "64"))',
                      f'RUN_N = int(os.environ.get("RINT_N", "{n}"))')
    src = src.replace('RUN_STAGE = int(os.environ.get("RINT_STAGE", "1"))',
                      f'RUN_STAGE = int(os.environ.get("RINT_STAGE", "{stage}"))')
    assert f'"{n}"))' in src and f'"{stage}"))' in src, "default 置換に失敗"
    d = Path(tempfile.mkdtemp(prefix=f"rint_s{stage}_n{n}_"))
    (d / "internalization_kernel.py").write_text(src, encoding="utf-8")
    slug = f"{USER}/rint-s{stage}-n{n}"
    meta = {"id": slug, "title": f"rint-s{stage}-n{n}", "code_file": "internalization_kernel.py",
            "language": "python", "kernel_type": "script", "is_private": "false",
            "enable_gpu": "true", "machine_shape": "NvidiaTeslaT4", "enable_tpu": "false",
            "enable_internet": "true", "dataset_sources": [], "competition_sources": [],
            "kernel_sources": [], "model_sources": []}
    json.dump(meta, open(d / "kernel-metadata.json", "w"))
    api.kernels_push(str(d))
    print(f"pushed {slug}", flush=True)


def main():
    if len(sys.argv) < 3:
        print("usage: push_internalization.py <stage> <n> [<n> ...]"); return 2
    stage = int(sys.argv[1]); ns = [int(x) for x in sys.argv[2:]]
    for n in ns:
        push(stage, n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
