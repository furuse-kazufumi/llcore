# SPDX-License-Identifier: Apache-2.0
"""PoC (CPU 手順 2): per-gene ridge readout で landscape を un-flatten する.

EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md §7b の CPU 手順 2 を単独実行可能な形で実証する。

falsifiable 命題:
  (P1) per-gene ridge readout (held-out) は fixed random readout より gene 間の
       fitness spread を広げ、最良 gene を高 R² に押し上げる (= landscape un-flatten)。
  (P2) しかし copy delay=0 は un-flatten 後『容易な単峰』になり、同予算 random search が
       GA と互角 → ③(選択)が立つことの証明にはならない。
  (P3) delay≥4 / addition は線形 readout で原理的にデコード困難で全 gene ~0
       → 3-param leak integrator には『構造的だが難しい』中間 regime が無い。

→ 結論 (honest): readout 修正だけでは ③ は立たない。CPU 手順 4 (空間拡張 + 分離機構)
   が真の unlock である、という診断 §7b の核を経験的に裏づける negative result。

実行: ``py -3.11 scripts/poc_ridge_readout_unflatten.py``
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

import numpy as np


def _ensure_utf8_stdout() -> None:
    """Windows cp932 console で em-dash/日本語を出すため stdout を UTF-8 化."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (ValueError, OSError):  # pragma: no cover
                pass


_ensure_utf8_stdout()

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llcore.evolution.honest_eval import evolution_vs_random  # noqa: E402
from llcore.fitness import (  # noqa: E402
    AdditionTask,
    CopyTask,
    calibrate_baseline,
    evaluate_gene,
    make_fixed_readout,
    make_ridge_eval_once,
    ridge_fitness,
)
from llcore.state_update import StateUpdateGene  # noqa: E402


def _random_genes(n: int, seed: int) -> list[StateUpdateGene]:
    rng = np.random.default_rng(seed)
    return [
        StateUpdateGene(
            float(rng.uniform(0, 1)), float(rng.uniform(-1, 1)), float(rng.uniform(-2, 2))
        )
        for _ in range(n)
    ]


def p1_unflatten() -> bool:
    """P1: ridge は fixed より spread が広く高 R² に届く."""
    print("\n=== P1: un-flatten (fixed vs ridge, copy d=8 delay=0) ===")
    copy = CopyTask(state_dim=8, out_dim=8, seq_len=32, delay=0)
    genes = _random_genes(40, seed=0)

    fr = make_fixed_readout(8, 8, seed=123)
    task_fixed = dataclasses.replace(copy, baseline_mse=calibrate_baseline(copy, fr))
    fixed = np.array(
        [evaluate_gene(g, task_fixed, fr, np.random.default_rng(7), n_trials=5) for g in genes]
    )
    ridge = np.array(
        [ridge_fitness(g, copy, n_train=64, n_eval=64, rng=np.random.default_rng(7)) for g in genes]
    )
    print(f"  fixed : mean={fixed.mean():.4f} std={fixed.std():.4f} max={fixed.max():.4f}")
    print(f"  ridge : mean={ridge.mean():.4f} std={ridge.std():.4f} max={ridge.max():.4f}")
    print(f"  spread ratio (ridge/fixed std) = {ridge.std()/max(fixed.std(),1e-9):.2f}")
    ok = ridge.std() > fixed.std() and ridge.max() > 0.9 > fixed.max()
    print(f"  [P1] un-flatten holds: {ok}")
    return ok


def p2_easy_no_selection() -> bool:
    """P2: copy delay=0 は un-flatten 後容易になり GA≈random (③ 未証明).

    eval-noise を n_train で掃引 (Codex Low finding: n_train=6 数値の再現性確保)。
    """
    print("\n=== P2: copy delay=0 は容易 → GA ≈ random (③ 未証明) ===")
    copy = CopyTask(state_dim=8, out_dim=8, seq_len=32, delay=0)
    results = []
    for n_train in (6, 12, 32):
        ev = make_ridge_eval_once(copy, n_train=n_train, n_eval=n_train)
        r = evolution_vs_random(
            ev, pop_size=10, n_generations=10, n_seeds=12, honest_n_trials=15, base_seed=20260530
        )
        print(
            f"  n_train={n_train:2d}: GA={r.ga_mean:.4f} RAND={r.random_mean:.4f} diff={r.diff:+.4f} "
            f"win={r.win_rate:.2f} p={r.wilcoxon_p:.4g} delta={r.cliff_delta:+.2f} passes={r.passes}"
        )
        results.append(r)
    # 全ノイズ水準で「進化成立」の合格条件 (passes) は立たない = ③ 未証明
    ok = all(not r.passes for r in results)
    print(f"  [P2] no-selection-advantage across noise levels holds: {ok}")
    return ok


def p3_no_useful_signal_regime() -> bool:
    """P3: delay≥4 / addition は clip 後 fitness 平坦 (選択信号なし)。

    honest (Codex High finding): raw R² (clip=False) は **負** で、clip が 0 化している。
    「原理的に不能」ではなく『この評価設定で線形 readout が有用信号を出さない』。
    """
    print("\n=== P3: delay≥4 / addition は clip 後 fitness 平坦 (raw R² は負) ===")
    genes = _random_genes(20, seed=1)
    for name, task in [
        ("copy delay=4", CopyTask(state_dim=8, out_dim=8, seq_len=32, delay=4)),
        ("addition    ", AdditionTask(state_dim=8, out_dim=1, seq_len=32)),
    ]:
        clipped = np.array(
            [ridge_fitness(g, task, n_train=64, n_eval=64, rng=np.random.default_rng(7)) for g in genes]
        )
        raw = np.array(
            [ridge_fitness(g, task, n_train=64, n_eval=64, rng=np.random.default_rng(7), clip=False)
             for g in genes]
        )
        print(
            f"  {name}: clipped max={clipped.max():.4f} | raw R² mean={raw.mean():+.4f} "
            f"std={raw.std():.4f} min={raw.min():+.4f} max={raw.max():+.4f}"
        )
    # clip 後の選択信号不在 (GA が使えるのは clip 済 fitness) を判定。
    copy4 = np.array(
        [ridge_fitness(g, CopyTask(state_dim=8, out_dim=8, seq_len=32, delay=4),
                       n_train=64, n_eval=64, rng=np.random.default_rng(7)) for g in genes]
    )
    addv = np.array(
        [ridge_fitness(g, AdditionTask(state_dim=8, out_dim=1, seq_len=32),
                       n_train=64, n_eval=64, rng=np.random.default_rng(7)) for g in genes]
    )
    ok = copy4.max() < 0.1 and addv.max() < 0.15
    print(f"  [P3] no usable selection signal (post-clip) holds: {ok}")
    return ok


def main() -> int:
    print("PoC CPU 手順 2 — per-gene ridge readout で landscape un-flatten")
    print("=" * 64)
    results = {"P1": p1_unflatten(), "P2": p2_easy_no_selection(), "P3": p3_no_useful_signal_regime()}
    print("\n" + "=" * 64)
    for k, v in results.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}")
    print(
        "\n結論 (honest): ridge readout は fitness の scale を un-flatten する (real capability)\n"
        "が、3-param leak integrator では copy delay=0=容易 (random も天井) / delay≥4・addition=\n"
        "clip 後 fitness 平坦 (raw R² は負・小 spread) で、③(選択) が立つ『構造的かつ難しい』\n"
        "中間 regime をこの評価設定・サンプルでは作れない。真の unlock は CPU 手順 4 (空間拡張 +\n"
        "分離機構)。これは負だが情報量のある結果で、診断 §7b の核を経験的に裏づける。"
    )
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
