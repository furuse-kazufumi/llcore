# SPDX-License-Identifier: Apache-2.0
"""Kaggle kernel 完了 poller (HD-1 grounding 本走用, windowless 単一プロセス).

kernels status API は 500 を返すため (運用教訓)、`kernels_output` の成功 +
result_hd1g_n*.json の meta.status を completion probe に使う。
最初に完了 (done/error) した kernel を報告して exit 0。

実行::  py -3.11 poll_hd1g.py <slug> [<slug> ...]
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time

from kaggle.api.kaggle_api_extended import KaggleApi


def main():
    slugs = sys.argv[1:]
    if not slugs:
        print("usage: poll_hd1g.py <user/kernel> [...]")
        return 2
    api = KaggleApi()
    api.authenticate()
    t0 = time.time()
    while True:
        for slug in slugs:
            d = tempfile.mkdtemp(prefix="hd1g_poll_")
            try:
                api.kernels_output(slug, path=d)
                for f in os.listdir(d):
                    if f.startswith("result_hd1g") and f.endswith(".json"):
                        meta = json.load(open(os.path.join(d, f))).get("meta", {})
                        st = meta.get("status")
                        nrec = "?"
                        try:
                            nrec = len(json.load(open(os.path.join(d, f))).get("records", []))
                        except Exception:
                            pass
                        print(f"[{time.time()-t0:7.0f}s] {slug}: status={st} records={nrec}", flush=True)
                        if st in ("done", "error"):
                            print(f"FINISHED {slug} status={st}")
                            return 0
            except Exception as e:
                print(f"[{time.time()-t0:7.0f}s] {slug}: not ready ({type(e).__name__})", flush=True)
            finally:
                shutil.rmtree(d, ignore_errors=True)
        time.sleep(300)


if __name__ == "__main__":
    raise SystemExit(main())
