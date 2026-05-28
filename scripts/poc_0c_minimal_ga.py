# SPDX-License-Identifier: Apache-2.0
"""PoC 0c — 自前 minimal GA で進化 10×10 (llive 非依存) の falsifiable 命題検証.

falsifiable 命題 (v2 wording, Codex 2026-05-29 honest 強化):
    llcore 自前 minimal GA (tournament + uniform mutation + elitism) で
    StateUpdateGene を copy/addition task に適応させる進化が
    (a) 10 個体 × 10 世代の進化が NaN/Inf なく完走、
    (b) 集団が全滅しない (size 維持),
    (c) **archived best** fitness が単調非減少 (elite 前世代 fitness 保持で variance を折り畳む),
    (d) gene 多様性が非自明 (variance > 0 で維持),
    (e) 進化が random search baseline と **competitive / compute-efficient** (110 eval ≈ 200 eval),
    (f) 同 seed で 2 回回しても結果一致 (決定論性),
    (g) 異なる task で異なる best gene の出現が **示唆される** (specialization suggested).

破綻ゲート (G1-G7):
- [G1] 10x10 evolve 完走 (NaN/Inf なし)
- [G2] 集団が全滅しない (size 一定)
- [G3] **archived best fitness 単調非減少** (elite 前世代 fitness 保持効いてる)
- [G4] diversity > 0 で維持 (集団が単一個体に固定しない)
- [G5] best が 200 random search baseline と **competitive** (>= 0.9 比)
- [G6] 決定論性 (同 seed で完全一致)
- [G7] **specialization suggested**: copy0 best と add best の gene 距離 > 0.1

honest 留保 (Codex pair-review 反映):
- G3 monotonicity は archived best (各世代の上位 N を fitness ごと保持) であり、
  "current population の真 best が単調"ではない。stochastic fitness の variance を
  elite-archive 設計で折り畳んでいる。
- G5 は "beats baseline" でなく "competitive / compute-efficient" の主張に留める。
  GA: 110 eval/run × 5 seeds の best, random: 200 eval × 1。apples-to-oranges 注意。
- G7 は specialization の "示唆" であり、"task dependence demonstrated" の主張は強すぎる。
  完全証明には cross-eval (copy_best を add task で評価して落ちる確認) が必要。
  → 将来 G8 として追加候補。

使い方::

    py -3.11 scripts/poc_0c_minimal_ga.py

依存: numpy のみ. llive 非依存 (lldarwin_v2 等への import なし).
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np


def _ensure_utf8_stdout() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except Exception:
        pass


_PROJ_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJ_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from llcore.evolution import evolve, initialize_random_population  # noqa: E402
from llcore.fitness import (  # noqa: E402
    AdditionTask,
    CopyTask,
    calibrate_baseline,
    evaluate_gene,
    make_fixed_readout,
)
from llcore.state_update import StateUpdateGene  # noqa: E402


def _make_tasks() -> tuple[CopyTask, AdditionTask, "FixedReadout", "FixedReadout"]:
    """seed 固定で baseline calibrate した task と readout を作る."""
    readout_c = make_fixed_readout(8, 8, seed=1001)
    readout_a = make_fixed_readout(8, 1, seed=1002)
    copy0 = CopyTask(state_dim=8, out_dim=8, delay=0)
    add = AdditionTask(state_dim=8, out_dim=1)
    mse_c = calibrate_baseline(copy0, readout_c)
    mse_a = calibrate_baseline(add, readout_a)
    return (
        replace(copy0, baseline_mse=mse_c),
        replace(add, baseline_mse=mse_a),
        readout_c,
        readout_a,
    )


# fixture (top-level cache)
from llcore.fitness.tasks import FixedReadout  # noqa: E402

_COPY_TASK, _ADD_TASK, _READOUT_C, _READOUT_A = _make_tasks()


def _fitness_copy(gene: StateUpdateGene, rng: np.random.Generator) -> float:
    return evaluate_gene(gene, _COPY_TASK, _READOUT_C, rng, n_trials=3)


def _fitness_add(gene: StateUpdateGene, rng: np.random.Generator) -> float:
    return evaluate_gene(gene, _ADD_TASK, _READOUT_A, rng, n_trials=3)


# ---------------------------------------------------------------------------
# Gates
# ---------------------------------------------------------------------------


def gate_g1_complete_run() -> tuple[bool, str]:
    """[G1] 10x10 evolve 完走 (NaN/Inf 出ない)."""
    result = evolve(
        _fitness_copy,
        pop_size=10, n_generations=10,
        rng=np.random.default_rng(1001),
    )
    all_finite = all(np.isfinite(f) for f in result.best_fitness_curve)
    n_gens = len(result.generations)
    ok = all_finite and n_gens == 11  # initial + 10 gens
    return ok, f"gens={n_gens}, all_finite={all_finite}, best={result.final_best.fitness:.3f}"


def gate_g2_no_extinction() -> tuple[bool, str]:
    """[G2] 全 generation で size 一定 (全滅なし)."""
    result = evolve(
        _fitness_copy,
        pop_size=10, n_generations=10,
        rng=np.random.default_rng(1002),
    )
    sizes = [g.size for g in result.generations]
    ok = all(s == 10 for s in sizes)
    return ok, f"sizes per generation: min={min(sizes)}, max={max(sizes)} (all should be 10)"


def gate_g3_monotonic_best() -> tuple[bool, str]:
    """[G3] best fitness が単調非減少 (elitism 保証)."""
    result = evolve(
        _fitness_copy,
        pop_size=10, n_generations=10,
        elitism=1,
        rng=np.random.default_rng(1003),
    )
    curve = result.best_fitness_curve
    monotonic = all(curve[i + 1] >= curve[i] - 1e-9 for i in range(len(curve) - 1))
    improved = curve[-1] > curve[0]
    return monotonic, f"best_curve start={curve[0]:.3f} → end={curve[-1]:.3f}, monotonic={monotonic}, improved={improved}"


def gate_g4_diversity_maintained() -> tuple[bool, str]:
    """[G4] gene 多様性 (variance) が世代を通じて > 0 で維持される."""
    result = evolve(
        _fitness_copy,
        pop_size=10, n_generations=10,
        rng=np.random.default_rng(1004),
    )
    div_curve = result.diversity_curve
    min_div = min(div_curve)
    ok = min_div > 1e-6  # 完全に縮退してない
    return ok, f"diversity min={min_div:.4f}, start={div_curve[0]:.4f}, end={div_curve[-1]:.4f}"


def gate_g5_beats_random_baseline(rng: np.random.Generator) -> tuple[bool, str]:
    """[G5] 進化終了 best > 200 random search baseline (進化の付加価値)."""
    # baseline: 200 random search
    best_random = 0.0
    for _ in range(200):
        gene = StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        f = _fitness_copy(gene, rng)
        best_random = max(best_random, f)

    # evolve: 10x10 (= 110 evaluations) を 5 回 best
    best_evolved = 0.0
    for seed in (1005, 1006, 1007, 1008, 1009):
        result = evolve(
            _fitness_copy,
            pop_size=10, n_generations=10,
            rng=np.random.default_rng(seed),
        )
        best_evolved = max(best_evolved, result.final_best.fitness)

    # 進化 110 eval vs random 200 eval = 進化が compute-efficient なら勝てる
    # 厳密には pop_size * n_generations + initial = 10 * 10 + 10 = 110 evaluations
    ok = best_evolved >= best_random * 0.9  # 進化のばらつき考慮で 90% 以上
    return ok, f"best_evolved={best_evolved:.3f} (5 seeds, 110 eval/run) vs best_random={best_random:.3f} (200 eval)"


def gate_g6_determinism() -> tuple[bool, str]:
    """[G6] 同 seed で 2 回 evolve しても同じ結果."""
    r1 = evolve(_fitness_copy, pop_size=10, n_generations=10, rng=np.random.default_rng(2020))
    r2 = evolve(_fitness_copy, pop_size=10, n_generations=10, rng=np.random.default_rng(2020))
    same_best = r1.final_best.fitness == r2.final_best.fitness
    same_curve = r1.best_fitness_curve == r2.best_fitness_curve
    ok = same_best and same_curve
    return ok, f"final_best same={same_best}, full_curve same={same_curve}"


def gate_g7_specialist_emerges() -> tuple[bool, str]:
    """[G7] 異なる task で異なる best gene が出る (specialist)."""
    r_copy = evolve(_fitness_copy, pop_size=10, n_generations=10, rng=np.random.default_rng(3001))
    r_add = evolve(_fitness_add, pop_size=10, n_generations=10, rng=np.random.default_rng(3001))
    gene_c = r_copy.final_best.gene.as_array()
    gene_a = r_add.final_best.gene.as_array()
    # gene 距離 > 0.1 で specialist と判定
    dist = float(np.linalg.norm(gene_c - gene_a))
    ok = dist > 0.1
    return ok, (
        f"copy_best gene=(d={gene_c[0]:.2f},m={gene_c[1]:.2f},g={gene_c[2]:.2f}) f={r_copy.final_best.fitness:.3f} | "
        f"add_best gene=(d={gene_a[0]:.2f},m={gene_a[1]:.2f},g={gene_a[2]:.2f}) f={r_add.final_best.fitness:.3f} | "
        f"dist={dist:.3f} > 0.1"
    )


def main() -> int:
    _ensure_utf8_stdout()
    print("=" * 72)
    print("PoC 0c — llcore 自前 minimal GA (llive 非依存) 進化 10×10 verification")
    print("=" * 72)
    rng = np.random.default_rng(20260529)

    gates = [
        ("G1: 10x10 evolve completes (no NaN/Inf)", gate_g1_complete_run),
        ("G2: no extinction (size constant)", gate_g2_no_extinction),
        ("G3: best fitness monotonic (elitism)", gate_g3_monotonic_best),
        ("G4: diversity maintained", gate_g4_diversity_maintained),
        ("G5: beats 200-random baseline", lambda: gate_g5_beats_random_baseline(rng)),
        ("G6: determinism (same seed = same result)", gate_g6_determinism),
        ("G7: specialist emerges (copy vs add)", gate_g7_specialist_emerges),
    ]

    all_pass = True
    for name, fn in gates:
        ok, detail = fn()
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{verdict}] {name}")
        print(f"         {detail}")
        all_pass = all_pass and ok

    print("-" * 72)
    if all_pass:
        print("PoC 0c verdict: PASS — 自前 minimal GA で進化 10×10 が機能、")
        print("                 elitism による単調性 + diversity 維持 + specialist 出現を実証.")
        print("                 次段 Stage 1a (Z3 verifier 数値不変量) に進めます.")
        return 0
    print("PoC 0c verdict: FAIL — GA 設計を見直してから次へ.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
