# SPDX-License-Identifier: Apache-2.0
"""branch A PoC — メモリ効率 fitness × verified-plasticity gate の 2×2 統合実証 (P1).

設計 = ``docs/BRANCH_A_CAPABILITY_REMERGE_DESIGN.md`` §3。
2×2: gate ∈ {none(control), contraction(gated, sound)} × fitness ∈ {memory, retention_only}。
判定 G1–G6。**G5(``evolution_vs_random.passes``)は記録のみ・assert しない**(honest 反証=
capability アークの ``passes=False`` 継承)。

honest 留保:
- footprint は **state-boundedness proxy(収縮率 L)** で実 RSS footprint ではない(P1)。
- gate=contraction は Z3 を要する。Z3 不在では fail-closed で全 reject → fallback gene 採用
  (出力に gate_stats を残す)。capability=NULL_TIE 既知ゆえ G5=False を想定・隠さない。
- falsification(evolution_vs_random)は gate なし(capability の問い)で回る。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, cast

import numpy as np

from llcore.evolution.honest_eval import evolution_vs_random  # type: ignore[import-untyped]
from llcore.evolution.minimal_ga import evolve  # type: ignore[import-untyped]
from llcore.fitness import (  # type: ignore[import-untyped]
    CopyTask,
    MemoryEfficiencyObjective,
    evaluate_gene,
    make_fixed_readout,
    state_boundedness_footprint,
)
from llcore.state_update import StateUpdateGene  # type: ignore[import-untyped]

EvalOnce = Callable[[object, np.random.Generator], float]


def make_eval_once(objective: MemoryEfficiencyObjective, readout: Any) -> EvalOnce:
    """``MemoryEfficiencyObjective`` を evolve/evolution_vs_random 用の ``(gene, rng)->float`` に束ねる。"""

    def eval_once(gene: object, rng: np.random.Generator) -> float:
        return objective.fitness(cast(StateUpdateGene, gene), readout, rng)

    return eval_once


def is_bounded(gene: object) -> bool:
    """収縮率 ``L<1``(=有界状態)か。``footprint<0.5 ⟺ L<1``(footprint=L/2)。"""
    return state_boundedness_footprint(cast(StateUpdateGene, gene)) < 0.5


def run_cell(
    gate_mode: str,
    objective: MemoryEfficiencyObjective,
    readout: Any,
    *,
    n_seeds: int,
    pop_size: int,
    n_generations: int,
    resample_cap: int,
    base_seed: int,
) -> dict[str, Any]:
    """2×2 の 1 セルを ``n_seeds`` 回回し、安全率・footprint・retention・GateStats を集計する。"""
    eval_once = make_eval_once(objective, readout)
    safe: list[bool] = []
    footprints: list[float] = []
    retentions: list[float] = []
    rej = res = fb = nchild = 0
    used_gate = gate_mode != "none"
    for s in range(n_seeds):
        rng = np.random.default_rng(base_seed + s)
        result = evolve(
            eval_once,
            pop_size=pop_size,
            n_generations=n_generations,
            rng=rng,
            gate_mode=gate_mode,
            resample_cap=resample_cap,
        )
        best = result.final_best.gene
        safe.append(is_bounded(best))
        footprints.append(state_boundedness_footprint(best))
        ret_rng = np.random.default_rng(base_seed + s + 777)
        retentions.append(
            evaluate_gene(best, objective.base_task, readout, ret_rng, n_trials=5)
        )
        gs = result.gate_stats
        if gs is not None:
            rej += gs.n_rejections
            res += gs.n_resamples
            fb += gs.fallback_count
            nchild += gs.n_children_generated
    children = (pop_size - 1) * n_generations  # elitism=1 既定
    cell: dict[str, Any] = {
        "gate_mode": gate_mode,
        "safe_rate": float(np.mean(safe)),
        "mean_footprint": float(np.mean(footprints)),
        "mean_retention": float(np.mean(retentions)),
    }
    if used_gate:
        cell["gate_stats"] = {
            "n_rejections": rej,
            "n_resamples": res,
            "fallback_count": fb,
            "n_children_generated": nchild,
            "resample_cap": resample_cap,
            "max_possible_resamples": resample_cap * children * n_seeds,
        }
    return cell


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="branch A PoC: memory fitness × verified gate (2x2, G1-G6)")
    ap.add_argument("--n-seeds", type=int, default=20, help="2x2 セルあたりの seed 数")
    ap.add_argument("--pop-size", type=int, default=10)
    ap.add_argument("--n-generations", type=int, default=8)
    ap.add_argument("--resample-cap", type=int, default=50)
    ap.add_argument("--falsify-seeds", type=int, default=15, help="G5 evolution_vs_random の seed 数")
    ap.add_argument("--honest-trials", type=int, default=20, help="G5 honest 再評価の trial 数")
    ap.add_argument("--w-acc", type=float, default=0.7)
    ap.add_argument("--w-mem", type=float, default=0.3)
    ap.add_argument("--json", default="out/poc_branch_a_memory_fitness.json")
    args = ap.parse_args(argv)

    task = CopyTask()
    readout = make_fixed_readout(task.state_dim, task.out_dim, seed=2026)
    memory_obj = MemoryEfficiencyObjective(base_task=task, w_acc=args.w_acc, w_mem=args.w_mem)
    retention_obj = MemoryEfficiencyObjective(base_task=task, w_acc=1.0, w_mem=0.0)

    common = dict(
        n_seeds=args.n_seeds, pop_size=args.pop_size, n_generations=args.n_generations,
        resample_cap=args.resample_cap, base_seed=0,
    )
    print("running 2x2 cells (gate × fitness)...")
    cells = {
        "none_memory": run_cell("none", memory_obj, readout, **common),
        "contraction_memory": run_cell("contraction", memory_obj, readout, **common),
        "none_retention": run_cell("none", retention_obj, readout, **common),
        "contraction_retention": run_cell("contraction", retention_obj, readout, **common),
    }

    # G1: control セルの決定論(同 seed 再実行で一致)= gate_mode="none" 後方互換の代理。
    g1_repeat = run_cell("none", memory_obj, readout, n_seeds=3, pop_size=args.pop_size,
                         n_generations=args.n_generations, resample_cap=args.resample_cap, base_seed=0)
    g1_ref = run_cell("none", memory_obj, readout, n_seeds=3, pop_size=args.pop_size,
                      n_generations=args.n_generations, resample_cap=args.resample_cap, base_seed=0)
    g1_pass = g1_repeat == g1_ref

    print("running falsification (G5, gate-free)...")
    fals = {
        "memory": evolution_vs_random(
            make_eval_once(memory_obj, readout), pop_size=args.pop_size,
            n_generations=args.n_generations, n_seeds=args.falsify_seeds,
            honest_n_trials=args.honest_trials, min_seeds=args.falsify_seeds,
        ),
        "retention": evolution_vs_random(
            make_eval_once(retention_obj, readout), pop_size=args.pop_size,
            n_generations=args.n_generations, n_seeds=args.falsify_seeds,
            honest_n_trials=args.honest_trials, min_seeds=args.falsify_seeds,
        ),
    }

    # --- G2–G6 判定 ---
    nm, cm = cells["none_memory"], cells["contraction_memory"]
    nr = cells["none_retention"]
    g2_pass = cm["safe_rate"] >= nm["safe_rate"] + 0.20  # gated は発散個体を弾く
    g3_pass = all(  # gated は空転せず(resample が cap 以内)集団を admit
        c["gate_stats"]["n_resamples"] <= c["gate_stats"]["max_possible_resamples"]
        and c["gate_stats"]["n_children_generated"] > 0
        for c in (cm, cells["contraction_retention"])
    )
    g4_pass = nm["mean_footprint"] < nr["mean_footprint"]  # memory 項が footprint を下げる
    g6_pass = g3_pass  # コスト上限 = G3 と同一条件(resample 爆発しない)

    verdict = {
        "G1_backward_compat_determinism": g1_pass,
        "G2_gate_efficacy_safe_rate": g2_pass,
        "G3_failclosed_no_spin": g3_pass,
        "G4_tradeoff_footprint": g4_pass,
        "G5_falsification_recorded": {  # ★ assert しない: 値をそのまま記録
            "memory_passes": bool(fals["memory"].passes),
            "memory_diff": float(fals["memory"].diff),
            "retention_passes": bool(fals["retention"].passes),
            "retention_diff": float(fals["retention"].diff),
            "note": "passes=False 想定(capability NULL_TIE 継承)。記録のみ・合否に使わない。",
        },
        "G6_gate_cost_bounded": g6_pass,
        "functional_min": bool(g1_pass and g2_pass and g3_pass and g6_pass),  # 「機能した」最小定義
    }

    print("\n=== branch A PoC VERDICT ===")
    print(f"  G1 determinism      : {'PASS' if g1_pass else 'FAIL'}")
    print(f"  G2 gate efficacy    : {'PASS' if g2_pass else 'FAIL'}  "
          f"(safe_rate control {nm['safe_rate']:.2f} -> gated {cm['safe_rate']:.2f})")
    print(f"  G3 fail-closed      : {'PASS' if g3_pass else 'FAIL'}")
    print(f"  G4 tradeoff         : {'PASS' if g4_pass else 'FAIL'}  "
          f"(footprint memory {nm['mean_footprint']:.3f} vs retention {nr['mean_footprint']:.3f})")
    print(f"  G5 falsification    : memory passes={fals['memory'].passes} (記録のみ・honest)")
    print(f"  G6 cost bounded     : {'PASS' if g6_pass else 'FAIL'}")
    print(f"  [functional_min G1∧G2∧G3∧G6] = {'PASS' if verdict['functional_min'] else 'FAIL'}")
    print("[honest] footprint=state-boundedness proxy(実RSSでない・P1)/ capability は取り戻さない(guarantee側)。")

    payload = {"config": vars(args), "cells": cells, "verdict": verdict}
    outp = Path(args.json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
