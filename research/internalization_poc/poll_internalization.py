# SPDX-License-Identifier: Apache-2.0
"""Kaggle internalization kernel 完了 poller (windowless 単一プロセス).

kernels_output で result_internalization_s{stage}_n{n}.json を取得し meta.status を probe。
全 slug が done/error になったら結果 JSON をローカル保存 + stage1_verdict/タイミングを表示して exit。

実行::  py -3.11 poll_internalization.py <user/kernel> [<user/kernel> ...]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

HERE = Path(__file__).resolve().parent
POLL_S = 150


def main():
    slugs = sys.argv[1:]
    if not slugs:
        print("usage: poll_internalization.py <user/kernel> [...]"); return 2
    api = KaggleApi(); api.authenticate()
    t0 = time.time()
    finished = {}
    while len(finished) < len(slugs):
        for slug in slugs:
            if slug in finished:
                continue
            d = tempfile.mkdtemp(prefix="rint_poll_")
            try:
                api.kernels_output(slug, path=d)
                files = os.listdir(d)
                results = [f for f in files if f.startswith("result_internalization") and f.endswith(".json")]
                if not files:
                    print(f"[{time.time()-t0:7.0f}s] {slug}: running (no output yet)", flush=True)
                elif not results:
                    print(f"[{time.time()-t0:7.0f}s] {slug}: FINISHED WITHOUT RESULT (early crash?) {files}", flush=True)
                    for f in files:
                        if f.endswith(".log"):
                            tail = open(os.path.join(d, f), encoding="utf-8", errors="replace").read()[-2500:]
                            print(f"--- {f} tail ---\n{tail}", flush=True)
                    finished[slug] = "no_result"
                else:
                    for f in results:
                        obj = json.load(open(os.path.join(d, f)))
                        meta = obj.get("meta", {}); st = meta.get("status")
                        nrec = len(obj.get("records", []))
                        print(f"[{time.time()-t0:7.0f}s] {slug}: status={st} records={nrec}", flush=True)
                        if st in ("done", "error"):
                            shutil.copy(os.path.join(d, f), HERE / f)   # ローカル保存
                            if "stage1_verdict" in obj:
                                print(f"  stage1_verdict: {json.dumps(obj['stage1_verdict'], ensure_ascii=False)}", flush=True)
                            # per-job 平均時間 (タイミング見積)
                            wall = None
                            try:
                                import datetime as _dt
                                s = _dt.datetime.fromisoformat(meta["start_time"]); e = _dt.datetime.fromisoformat(meta["end_time"])
                                wall = (e - s).total_seconds()
                                print(f"  wall={wall:.0f}s for {nrec} records = {wall/max(1,nrec):.1f}s/record", flush=True)
                            except Exception:
                                pass
                            finished[slug] = st
            except Exception as e:
                print(f"[{time.time()-t0:7.0f}s] {slug}: not ready ({type(e).__name__})", flush=True)
            finally:
                shutil.rmtree(d, ignore_errors=True)
        if len(finished) < len(slugs):
            time.sleep(POLL_S)
    print(f"ALL FINISHED: {finished}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
