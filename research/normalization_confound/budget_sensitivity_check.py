# SPDX-License-Identifier: Apache-2.0
"""G1 予算感度 + 谷の noise 起源チェック (honest disclosure).

smoke で baseline(clip=hard,D60) が既に valley_fraction=1.0=deceptive と出た。これは
(a) vdr 地形が本当に多峰 か (b) 少サンプル R² noise が midpoint を見かけ谷にしている
かを切り分ける。C1 の谷判定 fm < min(fi,fj) - 0.05*(|min|+eps) に対し、
midpoint の R² 評価ノイズ std が 0.05*min を超えるなら「谷」は noise 由来でありうる。

手順:
1. 同一 gene の eval_once を K 回 (異 seed) 評価し、R² の評価ノイズ std を n_eval 別に測る。
   → ノイズ std >> 0.05*fitness なら midpoint 谷は信頼できない (予算を上げる必要)。
2. baseline C1 を n_train/n_eval ∈ {24,40,64} × n_restarts ∈ {4,8} で測り
   valley_fraction の予算依存を見る (G1: 予算縮小が valley_fraction を 0.05 超動かすか)。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for p in (_ROOT / "research" / "step_c_memory_tasks",
          _ROOT / "research" / "ea_multitask" / "candidates",
          str(_HERE)):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from landscape_map import multimodality_report  # noqa: E402
from reservoir import LeakyDelayLineReservoir, gene_bounds  # noqa: E402
from variable_delay_recall import make_regimes  # noqa: E402
from c1_clip_eval import make_eval_once_clipswitch  # noqa: E402

for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass


def eval_noise(res, task, *, n_train, n_eval, clip="hard", K=20, base_seed=20260530):
    """ランダム gene 1 個を K 回 (異 seed) 評価し R² 評価ノイズ std を測る."""
    lo, hi = gene_bounds(res)
    e = make_eval_once_clipswitch(res, task, n_train=n_train, n_eval=n_eval, clip=clip)
    rng0 = np.random.default_rng(base_seed)
    g = lo + (hi - lo) * rng0.random(res.gene_dim)
    vals = [e(g, np.random.default_rng(base_seed + 1000 + k)) for k in range(K)]
    vals = np.array(vals)
    return float(vals.mean()), float(vals.std())


def main():
    vdr = {t.seq_len: t for t in make_regimes(delays=(15, 30, 45, 60),
                                              distractor_amp=0.2, in_dim=2)}
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    task = vdr[60]

    print("=== (1) eval noise std vs n_eval (D60, clip=hard) ===")
    print("    谷閾値 ~ 0.05*fitness。noise std がこれを超えると midpoint 谷は noise 起源の疑い")
    for n in (24, 40, 64, 96):
        m, sd = eval_noise(res, task, n_train=n, n_eval=n, K=20)
        thr = 0.05 * abs(m)
        flag = "NOISE-DOMINATED" if sd > thr else "ok"
        print(f"  n_train=n_eval={n:3d}: R2 mean={m:.3f} eval_noise_std={sd:.4f} "
              f"0.05*fit={thr:.4f} -> {flag}")

    print("\n=== (2) baseline valley_fraction vs budget (D60, clip=hard, sigma=0.15) ===")
    for (nr, ne, ntr) in ((4, 120, 24), (8, 150, 40), (8, 300, 48)):
        t0 = time.time()
        e = make_eval_once_clipswitch(res, task, n_train=ntr, n_eval=ntr, clip="hard")
        lo, hi = gene_bounds(res)
        rep = multimodality_report(e, dim=res.gene_dim, bounds=(lo, hi),
                                   n_restarts=nr, n_evals=ne, sigma=0.15,
                                   base_seed=20260530)
        dt = time.time() - t0
        print(f"  n_restarts={nr:2d} n_evals={ne:3d} n_train={ntr:2d}: "
              f"valley_fraction={rep['valley_fraction']:.3f} "
              f"n_optima={rep['n_optima']} is_mm={rep['is_multimodal']} ({dt:.1f}s)")


if __name__ == "__main__":
    main()
