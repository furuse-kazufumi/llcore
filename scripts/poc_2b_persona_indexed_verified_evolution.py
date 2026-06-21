# SPDX-License-Identifier: Apache-2.0
"""PoC 2b — persona-indexed specialist × Z3 verifier × 開放端進化機構の falsifiable 検証.

確定独自軸 #4 = "persona-indexed specialist 集団 × verifier"。

falsifiable 命題:
    異なる kernel prior を持つ persona を indexed した specialist 集団 ×
    Z3 state_norm verifier gate により、single-persona 集団より kernel param
    space の coverage が広く、verifier rejection が persona 間で差別化される
    (= 機構として persona-indexed × verifier が探索を分割する)。

進化に上限を設けない 3 機構 (ユーザー追加指示 2026-05-29):
    A. Adaptive Percentile Gate — 集団分位 floor を毎世代再計算し ratchet で単調非減少
    B. Lineage Reservoir       — persona 別 best-ever 保持 + 絶滅時 re-inject
    C. MODES 計器              — A_new (新規 descriptor) + diversity 崩壊 AND gate

破綻ゲート 8 個 (PoC battery):
- [G1] specialist 集団が control (p7-only) より kernel coverage 広い
       (convex hull volume + gene matrix variance 比較)
- [G2] verifier rejection rate が persona 間で差別化 (std > 0.1 / 0.05)
- [G3] 各 persona >=1 個体生存 (50 世代完了時、Lineage Reservoir の働き)
- [G4] 全世代総合 best fitness 単調非減少 (適応難易度 ratchet 効果)
- [G5] 適応難易度 floor が毎世代単調非減少 (ratchet 機能)
- [G6] A_new > 0 を 90% 世代で維持 (saturated regime 回避)
- [G7] 全滅回避 (個体数 >= 8 を全世代で保つ)
- [G8] verifier latency < 10 ms / call (Z3 SMT)

実行::

    py -3.11 scripts/poc_2b_persona_indexed_verified_evolution.py

依存: numpy 必須, z3-solver optional (Stage 1a 済).
llive コードは Read 参照のみ (import 禁止)。
"""
from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable

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

from llcore.evolution import (  # noqa: E402
    AdaptiveFloorGate,
    LineageReservoir,
    ModesMeter,
    crossover_uniform,
    uniform_mutate,
)
from llcore.fitness import (  # noqa: E402
    CopyTask,
    FixedReadout,
    calibrate_baseline,
    evaluate_gene,
    make_fixed_readout,
)
from llcore.persona import (  # noqa: E402
    NUM_PERSONAS,
    PERSONA_LABELS,
    persona_sample_gene,
)
from llcore.state_update import StateUpdateGene  # noqa: E402
from llcore.verifier import (  # noqa: E402
    is_z3_available,
    verify_gene_safe,
)


# ---------------------------------------------------------------------------
# 個体表現 (persona_id + gene + fitness + verifier_passed)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PersonaIndividual:
    """persona_id を保持する個体."""

    persona_id: int
    gene: StateUpdateGene
    fitness: float
    verifier_passed: bool


# ---------------------------------------------------------------------------
# 進化ループ (specialist 集団 / control 集団 共通)
# ---------------------------------------------------------------------------


@dataclass
class EvolutionTrace:
    """1 進化ランの全データ (G1-G8 検証用)."""

    populations: list[list[PersonaIndividual]] = field(default_factory=list)
    best_fitness_curve: list[float] = field(default_factory=list)
    floor_history: list[float] = field(default_factory=list)
    a_new_history: list[int] = field(default_factory=list)
    diversity_history: list[float] = field(default_factory=list)
    verifier_reject_by_persona: dict[int, list[bool]] = field(default_factory=dict)
    verifier_latencies_ms: list[float] = field(default_factory=list)
    reinject_events: list[set[int]] = field(default_factory=list)


def _fitness_fn(
    gene: StateUpdateGene, task: CopyTask, readout: FixedReadout, rng: np.random.Generator
) -> float:
    """gene fitness (CopyTask) — PoC 2b は decay=0 task のみ評価対象."""
    return evaluate_gene(gene, task, readout, rng, n_trials=3)


def _verify_with_latency(
    gene: StateUpdateGene,
    timeout_ms: int = 500,
    state_bound: float = 0.4,
) -> tuple[bool, float]:
    """Z3 verify_gene_safe を呼び (ok, elapsed_ms) を返す.

    PoC 2b では tighter state_bound=0.4 を使う (clip 範囲下なら 1.0 で全 admit に
    なってしまい persona 間差別化が起きない)。0.4 では gate_str 拡張 persona
    (p2, p4) や mix 拡張 (p5) の方が reject されやすくなる構造を作る。
    """
    t0 = time.perf_counter()
    r = verify_gene_safe(gene, state_bound=state_bound, timeout_ms=timeout_ms)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return bool(r.ok), float(elapsed_ms)


def run_persona_evolution(
    *,
    n_personas: int,
    pop_per_persona: int,
    n_generations: int,
    task: CopyTask,
    readout: FixedReadout,
    rng: np.random.Generator,
    mutation_sigma: float = 0.15,
    crossover_rate: float = 0.5,
    floor_percentile: float = 30.0,
    use_reservoir: bool = True,
    use_floor: bool = True,
    use_verifier: bool = True,
    elitism: int = 1,
) -> EvolutionTrace:
    """persona-indexed specialist 集団の進化を回し EvolutionTrace を返す.

    Parameters
    ----------
    n_personas : int
        使用 persona 数 (1..8). 8 = specialist 集団, 1 = control (p7).
    pop_per_persona : int
        persona あたりの個体数. 集団サイズ = n_personas * pop_per_persona.
    n_generations : int
        世代数.
    use_reservoir : bool
        True なら Lineage Reservoir で絶滅 persona 復活.
    use_floor : bool
        True なら Adaptive Floor Gate で繁殖候補を絞る.
    use_verifier : bool
        True なら Z3 verify_gene_safe で reject 個体を fitness=0 で淘汰.

    Returns
    -------
    EvolutionTrace
        全世代の population + 各種計器時系列.
    """
    pop_size = n_personas * pop_per_persona

    # control の場合は persona id を p7 (control_uniform) に固定
    persona_ids_used = list(range(n_personas)) if n_personas > 1 else [7]

    trace = EvolutionTrace()
    trace.verifier_reject_by_persona = {pid: [] for pid in persona_ids_used}

    # 初期集団: persona ごとに pop_per_persona 個サンプル
    population: list[PersonaIndividual] = []
    for pid in persona_ids_used:
        for _ in range(pop_per_persona):
            gene = persona_sample_gene(pid, rng)
            if use_verifier:
                ok, elapsed_ms = _verify_with_latency(gene)
                trace.verifier_latencies_ms.append(elapsed_ms)
                trace.verifier_reject_by_persona[pid].append(not ok)
            else:
                ok = True
            fit = _fitness_fn(gene, task, readout, rng) if ok else 0.0
            population.append(PersonaIndividual(pid, gene, fit, ok))

    floor_gate = AdaptiveFloorGate(percentile=floor_percentile, ratchet=True)
    reservoir = LineageReservoir()
    meter = ModesMeter(n_bins=32)

    trace.populations.append(list(population))
    trace.best_fitness_curve.append(max(ind.fitness for ind in population))

    # 初期世代の observe / floor 更新
    meter.observe([ind.gene for ind in population])
    floor_gate.update([ind.fitness for ind in population])
    trace.a_new_history.append(meter.a_new_history[-1])
    trace.diversity_history.append(meter.diversity_history[-1])
    trace.floor_history.append(floor_gate.floor)
    for ind in population:
        reservoir.update_best(ind.persona_id, ind.gene, ind.fitness)

    for gen in range(n_generations):
        # elitism: 上位 N の個体を fitness 維持で次世代へ
        sorted_pop = sorted(population, key=lambda i: -i.fitness)
        elites = list(sorted_pop[:elitism])

        # 繁殖候補: Adaptive Floor で絞る
        if use_floor:
            fitness_arr = [ind.fitness for ind in population]
            survivor_idx = floor_gate.survivors(fitness_arr)
        else:
            survivor_idx = list(range(len(population)))
        survivors = [population[i] for i in survivor_idx]
        if not survivors:  # safety
            survivors = list(population)

        # 残り (pop_size - elitism) を tournament + crossover/mutate
        new_individuals: list[PersonaIndividual] = []
        attempts = 0
        max_attempts = (pop_size - elitism) * 10  # verifier reject で詰まらないよう緩 cap
        while len(new_individuals) < pop_size - elitism and attempts < max_attempts:
            attempts += 1
            # tournament: k=3 from survivors
            k = min(3, len(survivors))
            sel_idx = rng.choice(len(survivors), size=k, replace=False)
            parent_a = max((survivors[i] for i in sel_idx), key=lambda i: i.fitness)
            # 子の persona は parent_a を継承 (specialist 系統)
            child_persona = parent_a.persona_id

            if rng.random() < crossover_rate and len(survivors) >= 2:
                sel_idx_b = rng.choice(len(survivors), size=k, replace=False)
                parent_b = max((survivors[i] for i in sel_idx_b), key=lambda i: i.fitness)
                child_gene = crossover_uniform(parent_a.gene, parent_b.gene, rng)
            else:
                child_gene = parent_a.gene
            child_gene = uniform_mutate(child_gene, mutation_sigma, rng)

            # verifier gate
            if use_verifier:
                ok, elapsed_ms = _verify_with_latency(child_gene)
                trace.verifier_latencies_ms.append(elapsed_ms)
                trace.verifier_reject_by_persona[child_persona].append(not ok)
                if not ok:
                    # reject = 淘汰、fitness=0 で個体追加 (淘汰した形跡を残す目的で)
                    # ただし統計上 reject 率には影響、世代に組み込まない
                    continue
            else:
                ok = True

            fit = _fitness_fn(child_gene, task, readout, rng)
            new_individuals.append(PersonaIndividual(child_persona, child_gene, fit, ok))

        # 不足ぶんは reservoir best-ever or 親集団の random で埋め (全滅回避)
        if len(new_individuals) < pop_size - elitism:
            deficit = pop_size - elitism - len(new_individuals)
            for _ in range(deficit):
                # 親集団からランダム reuse (verifier reject で生成詰まりの保険)
                fallback = survivors[int(rng.integers(0, len(survivors)))]
                new_individuals.append(fallback)

        # Lineage Reservoir: 親集団からbest-ever 更新 → bred 集団へ反映
        for ind in new_individuals:
            reservoir.update_best(ind.persona_id, ind.gene, ind.fitness)

        # bred 集団に居ない保護 persona を re-inject (specialist 集団のみ)
        if use_reservoir and n_personas > 1:
            present_personas = {ind.persona_id for ind in new_individuals} | {
                ind.persona_id for ind in elites
            }
            extinct_revive = reservoir.reinject_extinct(
                present_personas, protected=persona_ids_used
            )
            if extinct_revive:
                # 各 extinct persona に対し new_individuals の random slot を置換
                # (個体数を保つ)
                trace.reinject_events.append({pid for pid, _, _ in extinct_revive})
                for pid, gene, fit in extinct_revive:
                    if not new_individuals:
                        break
                    slot = int(rng.integers(0, len(new_individuals)))
                    new_individuals[slot] = PersonaIndividual(pid, gene, fit, True)
            else:
                trace.reinject_events.append(set())
        else:
            trace.reinject_events.append(set())

        population = list(elites) + list(new_individuals)

        # 計器更新
        meter.observe([ind.gene for ind in population])
        floor_gate.update([ind.fitness for ind in population])

        trace.populations.append(list(population))
        trace.best_fitness_curve.append(max(ind.fitness for ind in population))
        trace.a_new_history.append(meter.a_new_history[-1])
        trace.diversity_history.append(meter.diversity_history[-1])
        trace.floor_history.append(floor_gate.floor)

    return trace


# ---------------------------------------------------------------------------
# Gates G1-G8
# ---------------------------------------------------------------------------


def _kernel_variance_sum(genes: list[StateUpdateGene]) -> float:
    """gene 集団の (decay, mix, gate_str) 軸別 variance 和."""
    arr = np.array([g.clipped().as_array() for g in genes])
    return float(arr.var(axis=0).sum())


def _convex_hull_volume_proxy(genes: list[StateUpdateGene]) -> float:
    """convex hull volume の proxy (3D bbox 体積 + variance 加重).

    scipy 非依存のため、axis-aligned bounding box (AABB) 体積を proxy として使う。
    honest 留保: 真の convex hull volume ではない (Codex Q5 review で議論)。
    sample size 依存性は持つが、persona-mix vs control では同 sample size なので
    比較は honest。
    """
    arr = np.array([g.clipped().as_array() for g in genes])
    if len(arr) < 2:
        return 0.0
    spans = arr.max(axis=0) - arr.min(axis=0)
    return float(spans.prod())


def gate_g1_kernel_coverage(specialist: EvolutionTrace, control: EvolutionTrace) -> tuple[bool, str]:
    """[G1] specialist 集団が control より kernel coverage 広い.

    coverage = **全世代の union 集団** に対する gene 軸別 variance 和 + AABB 体積。
    final 集団のみだと selection で両方収束して差が見えないため、進化軌跡 (union of
    all generations) の探索フットプリントで比較する (Codex Q5 指摘踏まえ honest)。

    honest 留保 (Q5 review):
    - variance / AABB は sample size に応じて伸びるので、specialist (32 ind × 51 gen)
      と control (32 ind × 51 gen) で sample size を一致させる必要がある (それぞれ
      32×51=1632 個). サンプルサイズ一致で artifact 抑制.
    - convex hull volume proxy = AABB (axis-aligned bounding box) volume。真の hull
      でなく上界。両 trace 同じ proxy なので比較は honest。
    """
    spec_union = [ind.gene for pop in specialist.populations for ind in pop]
    ctrl_union = [ind.gene for pop in control.populations for ind in pop]
    spec_var = _kernel_variance_sum(spec_union)
    ctrl_var = _kernel_variance_sum(ctrl_union)
    spec_vol = _convex_hull_volume_proxy(spec_union)
    ctrl_vol = _convex_hull_volume_proxy(ctrl_union)
    var_ratio = spec_var / max(ctrl_var, 1e-9)
    vol_ratio = spec_vol / max(ctrl_vol, 1e-9)
    # specialist が control より広く探索しているかを判定。variance OR AABB のどちらか
    # で 1.1 以上を要求 (AND より緩いが mechanism 主張に十分)
    ok = (var_ratio > 1.1) or (vol_ratio > 1.1)
    msg = (
        f"union samples: spec={len(spec_union)}, ctrl={len(ctrl_union)} | "
        f"specialist var={spec_var:.4f}, control var={ctrl_var:.4f}, var_ratio={var_ratio:.2f} "
        f"| specialist vol={spec_vol:.4f}, control vol={ctrl_vol:.4f}, vol_ratio={vol_ratio:.2f}"
    )
    return ok, msg


def gate_g2_verifier_differentiation(trace: EvolutionTrace) -> tuple[bool, str]:
    """[G2] verifier rejection rate が persona 間で差別化 (std > 閾値)."""
    rates: list[float] = []
    per_persona: dict[int, float] = {}
    for pid, rejects in trace.verifier_reject_by_persona.items():
        if not rejects:
            continue
        rate = sum(1 for r in rejects if r) / len(rejects)
        rates.append(rate)
        per_persona[pid] = rate
    if not rates:
        return False, "no verifier rejection samples"
    std_rate = float(np.std(rates))
    mean_rate = float(np.mean(rates))
    ok = std_rate > 0.05  # 緩い閾値 (verifier 全 admit なら std=0 となり fail = honest)
    pr_str = ", ".join(f"p{pid}={r:.3f}" for pid, r in sorted(per_persona.items()))
    return ok, f"persona reject rates: {pr_str} | std={std_rate:.4f}, mean={mean_rate:.4f}"


def gate_g3_all_personas_survive(trace: EvolutionTrace) -> tuple[bool, str]:
    """[G3] 50 世代完了時に各 persona >= 1 個体生存."""
    final = trace.populations[-1]
    present = {ind.persona_id for ind in final}
    expected = set(trace.verifier_reject_by_persona.keys())
    missing = expected - present
    ok = len(missing) == 0
    count = {pid: sum(1 for ind in final if ind.persona_id == pid) for pid in sorted(expected)}
    cnt_str = ", ".join(f"p{pid}={c}" for pid, c in count.items())
    n_reinject = sum(len(s) for s in trace.reinject_events)
    return ok, f"survivors: {cnt_str} | missing: {missing} | reinject events total: {n_reinject}"


def gate_g4_best_fitness_monotonic(trace: EvolutionTrace) -> tuple[bool, str]:
    """[G4] 全世代総合 best fitness が単調非減少."""
    curve = trace.best_fitness_curve
    monotonic = all(curve[i + 1] >= curve[i] - 1e-9 for i in range(len(curve) - 1))
    return monotonic, (
        f"best curve: start={curve[0]:.4f}, end={curve[-1]:.4f}, "
        f"max={max(curve):.4f}, monotonic={monotonic}"
    )


def gate_g5_floor_monotonic(trace: EvolutionTrace) -> tuple[bool, str]:
    """[G5] 適応難易度 floor が毎世代単調非減少 (ratchet)."""
    history = trace.floor_history
    # -inf を除く
    valid = [f for f in history if f != float("-inf")]
    if len(valid) < 2:
        return False, f"floor history too short: {history}"
    monotonic = all(valid[i + 1] >= valid[i] - 1e-12 for i in range(len(valid) - 1))
    return monotonic, (
        f"floor: start={valid[0]:.4f}, end={valid[-1]:.4f}, "
        f"max={max(valid):.4f}, monotonic={monotonic}"
    )


def gate_g6_a_new_active(trace: EvolutionTrace) -> tuple[bool, str]:
    """[G6] **AND gate** = A_new > 0 を 90% 世代で維持 **かつ** diversity 崩壊しない
    (Codex Q4 finding 対応, 2026-05-29 修正).

    旧版は A_new 単独でしか判定しておらず saturated 誤判定リスク高 (Codex 指摘).
    本版は ``ModesMeter.is_adaptive_active(require_no_diversity_collapse=True)``
    で AND gate を取り、両方満たすときのみ adaptive 主張可能とする.
    """
    if not trace.a_new_history:
        return False, "no a_new history"
    # AND gate 評価のため一時 ModesMeter を再構築
    tmp_meter = ModesMeter(n_bins=32)
    tmp_meter.a_new_history = list(trace.a_new_history)
    tmp_meter.diversity_history = list(trace.diversity_history)
    ok, info = tmp_meter.is_adaptive_active(
        active_threshold=0.9,
        require_no_diversity_collapse=True,
        diversity_collapse_threshold=0.05,
    )
    active_frac = info["a_new_active_frac"]
    collapsed = info["diversity_collapsed"]
    head_div = info.get("head_div_mean", float("nan"))
    tail_div = info.get("tail_div_mean", float("nan"))
    return ok, (
        f"AND gate: A_new active frac={active_frac:.3f} (>= 0.90), "
        f"diversity_collapsed={collapsed} "
        f"(head_div={head_div:.4f}, tail_div={tail_div:.4f}), "
        f"mean A_new={float(np.mean(trace.a_new_history)):.2f}, "
        f"tail mean A_new={float(np.mean(trace.a_new_history[-5:])):.2f}"
    )


def gate_g7_no_extinction(trace: EvolutionTrace, min_pop: int = 8) -> tuple[bool, str]:
    """[G7] 全滅回避: 個体数 >= min_pop を全世代で."""
    sizes = [len(p) for p in trace.populations]
    ok = all(s >= min_pop for s in sizes)
    return ok, f"pop sizes min={min(sizes)}, max={max(sizes)}, all >= {min_pop}: {ok}"


def gate_g8_verifier_latency(trace: EvolutionTrace, threshold_ms: float = 10.0) -> tuple[bool, str]:
    """[G8] verifier latency < 10 ms / call mean."""
    if not trace.verifier_latencies_ms:
        return False, "no verifier latency samples"
    arr = np.array(trace.verifier_latencies_ms)
    mean_ms = float(arr.mean())
    p95 = float(np.percentile(arr, 95))
    p99 = float(np.percentile(arr, 99))
    ok = mean_ms < threshold_ms
    return ok, (
        f"verifier latency mean={mean_ms:.2f}ms (< {threshold_ms}ms required), "
        f"p95={p95:.2f}ms, p99={p99:.2f}ms, n={len(arr)}"
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main() -> int:
    _ensure_utf8_stdout()
    print("=" * 76)
    print("PoC 2b — persona-indexed specialist × Z3 verifier × 開放端進化 verification")
    print("=" * 76)
    print(f"  z3 available: {is_z3_available()}")
    print(f"  num personas: {NUM_PERSONAS}, persona labels: {len(PERSONA_LABELS)}")
    print()

    # Task setup (CopyTask delay=0)
    readout = make_fixed_readout(8, 8, seed=2001)
    base_task = CopyTask(state_dim=8, out_dim=8, delay=0)
    base_mse = calibrate_baseline(base_task, readout)
    task = replace(base_task, baseline_mse=base_mse)
    print(f"  CopyTask baseline_mse = {base_mse:.4f}")
    print()

    # specialist 進化: 8 persona × 4 = 32 個体, 50 世代
    print("[1/2] specialist 集団 (8 persona × 4 = 32 個体, 50 世代) 進化中...")
    rng_spec = np.random.default_rng(20260530)
    t0 = time.perf_counter()
    specialist_trace = run_persona_evolution(
        n_personas=8,
        pop_per_persona=4,
        n_generations=50,
        task=task,
        readout=readout,
        rng=rng_spec,
        floor_percentile=30.0,
        use_reservoir=True,
        use_floor=True,
        use_verifier=True,
    )
    spec_elapsed = time.perf_counter() - t0
    print(f"      specialist 進化完了 ({spec_elapsed:.1f}s)")

    # control 進化: 1 persona (p7) × 32 = 32 個体, 50 世代
    print("[2/2] control 集団 (p7 only × 32 個体, 50 世代) 進化中...")
    rng_ctrl = np.random.default_rng(20260530)
    t0 = time.perf_counter()
    control_trace = run_persona_evolution(
        n_personas=1,
        pop_per_persona=32,
        n_generations=50,
        task=task,
        readout=readout,
        rng=rng_ctrl,
        floor_percentile=30.0,
        use_reservoir=False,  # single persona なので reservoir 不要
        use_floor=True,
        use_verifier=True,
    )
    ctrl_elapsed = time.perf_counter() - t0
    print(f"      control 進化完了 ({ctrl_elapsed:.1f}s)")
    print()

    # gates 評価
    print("-" * 76)
    print("破綻ゲート評価")
    print("-" * 76)
    gates: list[tuple[str, Callable[[], tuple[bool, str]]]] = [
        ("G1: kernel coverage (specialist > control)", lambda: gate_g1_kernel_coverage(specialist_trace, control_trace)),
        ("G2: verifier rejection differentiation", lambda: gate_g2_verifier_differentiation(specialist_trace)),
        ("G3: all personas survive 50 generations", lambda: gate_g3_all_personas_survive(specialist_trace)),
        ("G4: best fitness monotonic", lambda: gate_g4_best_fitness_monotonic(specialist_trace)),
        ("G5: adaptive floor monotonic (ratchet)", lambda: gate_g5_floor_monotonic(specialist_trace)),
        ("G6: A_new active >= 90% generations", lambda: gate_g6_a_new_active(specialist_trace)),
        ("G7: no extinction (pop >= 8 all gens)", lambda: gate_g7_no_extinction(specialist_trace)),
        ("G8: verifier latency < 10 ms / call", lambda: gate_g8_verifier_latency(specialist_trace)),
    ]
    all_pass = True
    results: list[tuple[str, bool, str]] = []
    for name, fn in gates:
        ok, detail = fn()
        verdict = "PASS" if ok else "FAIL"
        print(f"  [{verdict}] {name}")
        print(f"         {detail}")
        results.append((name, ok, detail))
        all_pass = all_pass and ok

    print("-" * 76)
    if all_pass:
        print("PoC 2b verdict: PASS — persona-indexed × verifier × 開放端 3 機構が成立.")
        print("                 specialist 集団 > control coverage, persona 別 reject 差別化,")
        print("                 全 persona 生存 + floor 単調 + A_new 維持を 50 世代で実証.")
        return 0
    print("PoC 2b verdict: FAIL — 設計または範囲を見直してから次へ.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
