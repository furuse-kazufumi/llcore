# SPDX-License-Identifier: Apache-2.0
"""実験5: ③ が立つ境界の特定 (honest control) — 欺瞞があるときだけ MAP-Elites が勝つ.

実験4 で MAP-Elites が 3 baseline 全てに圧勝したが、「MAP-Elites が万能だから」ではなく
「landscape が欺瞞的 (fitness dip) だから」であることを示す対照実験。同じ corridor (behavior=mean,
genotypic corridor) で **fitness dip の有無**だけを変える:

- (A) deceptive: 局所最適 → dip → 大域最適 (実験4 と同じ)。MAP-Elites のみ大域到達のはず。
- (B) smooth: dip なし・behavior に対して大域最適へ単調増加。hill-climbing も大域へ climb できるはず
  → MAP-Elites の優位が**消える** (有意差なし or baseline 同等)。

これにより「③ が load-bearing なのは欺瞞 (diversity 維持が downhill 跨ぎに必須) な regime に限る」を
falsifiable に切り出す。= ③ の将来性は「実 task が欺瞞的 corridor 構造を持つか」に帰着する。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from selection_lab import compare, run_methods_over_seeds  # noqa: E402


def _ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        rc = getattr(stream, "reconfigure", None)
        if rc:
            try:
                rc(encoding="utf-8")
            except (ValueError, OSError):
                pass


_ensure_utf8_stdout()

D = 24
_NOISE = 0.008


def behavior_mean(gene: np.ndarray) -> np.ndarray:
    return np.array([gene.mean()])


def deceptive_eval(gene: np.ndarray, rng: np.random.Generator) -> float:
    """(A) 欺瞞: 局所0.60@b=0.4 → dip → 大域1.00@b=0.9."""
    b = float(gene.mean())
    local = 0.60 * np.exp(-((b - 0.40) ** 2) / (2 * 0.08 ** 2))
    glob = 1.00 * np.exp(-((b - 0.90) ** 2) / (2 * 0.06 ** 2))
    return float(max(local, glob) + rng.normal(0, _NOISE))


def smooth_eval(gene: np.ndarray, rng: np.random.Generator) -> float:
    """(B) 非欺瞞: b に対して大域最適 (b=0.9) へ単調増加 (dip なし)。"""
    b = float(gene.mean())
    # b<=0.9 で単調増加、b=0.9 で 1.0、以降緩やかに低下。dip / 局所罠なし。
    val = 1.00 * np.exp(-((b - 0.90) ** 2) / (2 * 0.30 ** 2))
    return float(val + rng.normal(0, _NOISE))


def _run(name: str, eval_fn) -> bool:
    bounds = (np.zeros(D), np.ones(D))
    res = run_methods_over_seeds(
        eval_fn, behavior_mean,
        dim=D, bounds=bounds, behavior_bounds=(np.zeros(1), np.ones(1)),
        grid_shape=(24,), n_evals=6000, n_seeds=20, honest_n_trials=30, sigma=0.10,
    )
    print(f"\n--- 条件 {name} ---")
    for k, v in res.items():
        print(f"  {k:14s}: mean={v.mean():.4f} (reach>0.8: {float(np.mean(v>0.8)):.2f})")
    cme_rr = compare(res["map_elites"], res["rr_hillclimb"], "me", "rr")
    print(f"  MAP-Elites vs RR-hillclimb: diff={cme_rr.diff:+.4f} p={cme_rr.wilcoxon_p:.4g} "
          f"δ={cme_rr.cliff_delta:+.2f} → {'③成立側' if cme_rr.passes else '有意差なし'}")
    return cme_rr.passes


def main() -> int:
    print("実験5: ③ が立つ境界 (欺瞞の有無で MAP-Elites 優位が出るか消えるか)")
    print("=" * 74)
    a = _run("(A) deceptive (dip あり)", deceptive_eval)
    b = _run("(B) smooth (dip なし・単調)", smooth_eval)
    print("\n" + "=" * 74)
    print(f"  (A) 欺瞞: MAP-Elites > RR-hillclimb = {a}  (期待 True)")
    print(f"  (B) 非欺瞞: MAP-Elites > RR-hillclimb = {b}  (期待 False = 優位消失)")
    ok = a and not b
    print(f"\n  実験5 境界確認 (③優位は欺瞞 regime 限定): {'YES' if ok else 'NO'}")
    print("  → ③ の将来性は『実 task / 実 LLM fitness が欺瞞的 corridor 構造を持つか』に帰着する。")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
