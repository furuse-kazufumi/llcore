# SPDX-License-Identifier: Apache-2.0
"""Phase 2a 追補 — d8 効果の 2 独立 n=20 run の pooled (n=40) 解析 (再現用).

§10 ((c) 決着, seeds 2000-2019) と §11 (w̄ 軸 NEGATIVE の w̄_task=0.00 セル =
delay=8/seq_len=32/外乱なし の fresh-seed 独立再現, seeds 3000-3019) は、
**同一構成の d8 タスクを独立 seed で 2 回**測ったことになる。本 script は両者の
paired delta を pool し、d8 効果の頑健性を sign-flip permutation で再評価する。

honest 動機 (feedback_benchmark_honest_disclosure): §10 の single-run p=0.0056 は
seed 有利端であり、fresh n=20 単独では p=0.093 (borderline)。pooled n=40 で robust
に有意か (p, trimmed, drop-max|Δ|) を確認し、論文 §9 の証拠を single-run から pooled
へ格上げしつつ single-run の seed 感受性を併記するための数値を出す。

実行::

    py -3.11 research/verified_memory_poc/run_d8_pooled.py

入力 (既存) ::

    results_c_decision.json   (§10, seeds 2000-2019, copy_d8)
    results_wbar_decision.json(§11, seeds 3000-3019, w0.00)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from run_3arm_ab import _ensure_utf8_stdout  # noqa: E402
from run_c_decision import signflip_pvalue  # noqa: E402

PERM_N_RESAMPLES = 100_000
PERM_RNG_SEED = 7


def _load_deltas(path: Path, stats_key: str) -> np.ndarray:
    d = json.loads(path.read_text(encoding="utf-8"))
    per_seed = d["stats"][stats_key]["deltas_per_seed"]
    return np.array([per_seed[k] for k in sorted(per_seed, key=int)], dtype=np.float64)


def _summary(name: str, v: np.ndarray) -> dict:
    p = signflip_pvalue(v, n_resamples=PERM_N_RESAMPLES, seed=PERM_RNG_SEED)
    return {
        "name": name, "n": int(v.size),
        "mean_delta": float(v.mean()), "median_delta": float(np.median(v)),
        "n_positive": int((v > 0).sum()), "n_negative": int((v < 0).sum()),
        "p_signflip_two_sided": p,
    }


def main() -> int:
    _ensure_utf8_stdout()
    d8 = _load_deltas(_HERE / "results_c_decision.json", "copy_d8")
    w0 = _load_deltas(_HERE / "results_wbar_decision.json", "w0.00")
    pool = np.concatenate([d8, w0])

    rows = [
        _summary("phase2a_d8_s2000-19", d8),
        _summary("replication_w0_s3000-19", w0),
        _summary("pooled_n40", pool),
    ]
    # pooled robustness
    k = max(1, int(0.1 * pool.size))
    trimmed = float(np.sort(pool)[k:pool.size - k].mean())
    idx = int(np.argmax(np.abs(pool)))
    p_drop = signflip_pvalue(np.delete(pool, idx), n_resamples=PERM_N_RESAMPLES, seed=PERM_RNG_SEED)

    out = {
        "rows": rows,
        "pooled_robustness": {
            "trimmed_10pct": trimmed,
            "p_after_dropping_max_abs_seed": p_drop,
        },
        "interpretation": (
            "d8 効果の方向は 2/2 独立 run で再現 (pooled +29/-11)。pooled n=40 で robust に有意 "
            "(p=%.4f, trimmed=%+.4f, drop-max p=%.4f)。ただし single-run p は seed 感受性あり "
            "(s2000-19: 0.0056 / s3000-19: 0.093)。論文は pooled を主証拠とし single-run 感受性を併記する。"
        ) % (rows[2]["p_signflip_two_sided"], trimmed, p_drop),
    }
    (_HERE / "results_d8_pooled.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=== d8 effect: independent runs + pooled ===")
    for r in rows:
        print(f"{r['name']:24s} n={r['n']:2d} mean={r['mean_delta']:+.4f} "
              f"median={r['median_delta']:+.4f} +{r['n_positive']}/-{r['n_negative']} "
              f"p={r['p_signflip_two_sided']:.4f}")
    print(f"pooled trimmed10%={trimmed:+.4f}  drop-max|Δ| p={p_drop:.4f}")
    print(f"\nwrote {_HERE / 'results_d8_pooled.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
