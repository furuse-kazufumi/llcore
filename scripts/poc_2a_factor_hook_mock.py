# SPDX-License-Identifier: Apache-2.0
"""PoC 2a — factor_hook × state update kernel (mock) の falsifiable 検証.

honest 留保 (Codex 2026-05-29 指摘): 実装上の接続先は RWKV-7 weight ではなく
StateUpdateGene の RWKV-**inspired** state update kernel。論文/外向け wording は
"RWKV mock" ではなく "RWKV-inspired state update kernel mock" に絞る。

falsifiable 命題:
    llcore 自前 factor_hook protocol が
    (a) FactorSnapshot で 10 因子保持 + clamp [0, 1]、
    (b) NoopFactorHook で Δ=1.0 (factor_hook OFF 等価)、
    (c) HeuristicFactorHook で uncertainty 高 → Δ<1 (慎重)、
    (d) apply_hook_to_gene で gene.decay が Δ により動的調整、
    (e) 異なる snapshot で異なる effective gene が出る (factor 区別性)、
    (f) 決定論性、
    (g) 進化ループに hook 注入しても完走する (mock smoke 統合)。

破綻ゲート (G1-G7):
- [G1] FactorSnapshot 10 因子保持 + 未指定 0.5 default + clamp [0,1]
- [G2] NoopFactorHook 常に 1.0
- [G3] HeuristicFactorHook uncertainty 高で Δ<1, integrate 高で Δ>1
- [G4] apply_hook_to_gene で decay が動的調整 (NoopHook と HeuristicHook で異なる)
- [G5] 異なる snapshot (uncertainty 高 vs 低) で effective gene が変わる
- [G6] 決定論性
- [G7] 進化ループ smoke (hook 経由 fitness 評価で 10×10 evolve 完走)
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

from llcore.evolution import evolve  # noqa: E402
from llcore.factor_hook import (  # noqa: E402
    FACTOR_NAMES,
    FactorSnapshot,
    HeuristicFactorHook,
    NoopFactorHook,
    apply_hook_to_gene,
)
from llcore.fitness import (  # noqa: E402
    CopyTask,
    calibrate_baseline,
    evaluate_gene,
    make_fixed_readout,
)
from llcore.state_update import StateUpdateGene  # noqa: E402


def gate_g1_factor_snapshot() -> tuple[bool, str]:
    """[G1] FactorSnapshot 10 因子保持 + default 0.5 + clamp [0,1]."""
    snap = FactorSnapshot(values={"uncertainty": 0.9, "integrate": 0.2})
    # 未指定因子は default 0.5
    assert snap.get("structurize") == 0.5
    # clamp
    snap_oor = FactorSnapshot(values={"uncertainty": 1.5, "exploration": -0.3})
    cu = snap_oor.get("uncertainty")
    ce = snap_oor.get("exploration")
    # vector が canonical 順 + 10 個
    vec = snap.vector()
    ok = (
        len(vec) == 10
        and len(FACTOR_NAMES) == 10
        and cu == 1.0 and ce == 0.0
    )
    return ok, f"len=10={len(vec)==10}, clamp uncertainty 1.5→{cu}, exploration -0.3→{ce}"


def gate_g2_noop_returns_one() -> tuple[bool, str]:
    """[G2] NoopFactorHook 常に 1.0."""
    hook = NoopFactorHook()
    snap_a = FactorSnapshot(values={"uncertainty": 0.9})
    snap_b = FactorSnapshot(values={"uncertainty": 0.1, "integrate": 0.9})
    d1 = hook.delta_for(snap_a)
    d2 = hook.delta_for(snap_b)
    ok = d1 == 1.0 and d2 == 1.0
    return ok, f"noop returns 1.0: snap_a={d1}, snap_b={d2}"


def gate_g3_heuristic_directionality() -> tuple[bool, str]:
    """[G3] HeuristicFactorHook の方向性: uncertainty 高 → Δ<1, integrate 高 → Δ>1.

    honest 留保: 式 (llive 設計踏襲) は neutral (全因子 0.5) で Δ≈1.28 にバイアスする
    (signal = (0.5+0.5+0.25-0.75)/2 = 0.25 → exp(0.25) = 1.28)。これは "integrate +
    structurize" の正寄与が "uncertainty" 1.5x 重みより小幅に勝つため。Δ=1 厳密追求は
    将来課題 (v0.2 で normalization 改修候補)。本 G3 では directionality (方向性) のみ要求。
    """
    hook = HeuristicFactorHook(sensitivity=1.0)
    snap_unc = FactorSnapshot(values={"uncertainty": 1.0, "integrate": 0.0, "structurize": 0.0})
    snap_int = FactorSnapshot(values={"uncertainty": 0.0, "integrate": 1.0, "structurize": 1.0})
    snap_neutral = FactorSnapshot(values={})  # baseline 0.5 (注: 中立固定点でなく all-0.5)
    d_unc = hook.delta_for(snap_unc)
    d_int = hook.delta_for(snap_int)
    d_neu = hook.delta_for(snap_neutral)
    # directionality 主判定: d_unc < d_neu < d_int (monotone) かつ d_unc<1<d_int
    monotone = d_unc < d_neu < d_int
    bracket = d_unc < 1.0 < d_int
    ok = monotone and bracket
    return ok, f"uncertainty Δ={d_unc:.3f} < neutral Δ={d_neu:.3f} < integrate Δ={d_int:.3f} (monotone={monotone}, bracket={bracket})"


def gate_g4_apply_hook_modifies_decay() -> tuple[bool, str]:
    """[G4] apply_hook_to_gene で decay が動的調整される (Noop と Heuristic で異なる)."""
    gene = StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5)
    noop = NoopFactorHook()
    heuristic = HeuristicFactorHook(sensitivity=1.5)
    snap = FactorSnapshot(values={"uncertainty": 0.9})  # Δ<1 期待
    g_noop = apply_hook_to_gene(gene, noop, snap)
    g_heu = apply_hook_to_gene(gene, heuristic, snap)
    # noop: decay 不変, heuristic: decay 変化
    ok = abs(g_noop.decay - gene.decay) < 1e-9 and abs(g_heu.decay - gene.decay) > 0.01
    return ok, f"noop decay={g_noop.decay:.3f} (=orig 0.5), heuristic decay={g_heu.decay:.3f}"


def gate_g5_snapshot_distinguishability() -> tuple[bool, str]:
    """[G5] 異なる snapshot で異なる effective gene."""
    gene = StateUpdateGene(decay=0.5, mix=0.5, gate_str=0.5)
    hook = HeuristicFactorHook(sensitivity=1.5)
    snap_a = FactorSnapshot(values={"uncertainty": 0.9})  # 慎重
    snap_b = FactorSnapshot(values={"integrate": 0.9, "structurize": 0.9})  # 大胆
    g_a = apply_hook_to_gene(gene, hook, snap_a)
    g_b = apply_hook_to_gene(gene, hook, snap_b)
    dist = abs(g_a.decay - g_b.decay)
    ok = dist > 0.05
    return ok, f"snap_a→decay={g_a.decay:.3f}, snap_b→decay={g_b.decay:.3f}, dist={dist:.3f}"


def gate_g6_determinism() -> tuple[bool, str]:
    """[G6] 同 hook + snapshot で 2 回呼び同結果."""
    hook = HeuristicFactorHook(sensitivity=1.0)
    snap = FactorSnapshot(values={"uncertainty": 0.7, "integrate": 0.3})
    d1 = hook.delta_for(snap)
    d2 = hook.delta_for(snap)
    gene = StateUpdateGene(decay=0.6, mix=0.4, gate_str=0.7)
    g1 = apply_hook_to_gene(gene, hook, snap)
    g2 = apply_hook_to_gene(gene, hook, snap)
    ok = d1 == d2 and g1 == g2
    return ok, f"d1=d2={d1==d2}, gene1=gene2={g1==g2}"


def gate_g7_evolution_smoke_with_hook() -> tuple[bool, str]:
    """[G7] 進化ループに factor_hook 注入で 10×10 evolve 完走 (mock smoke)."""
    readout = make_fixed_readout(8, 8, seed=1001)
    base_task = CopyTask(state_dim=8, out_dim=8, delay=0)
    task = replace(base_task, baseline_mse=calibrate_baseline(base_task, readout))

    hook = HeuristicFactorHook(sensitivity=0.8)
    # 仮の snapshot (進化過程中は cognitive_mesh 由来だが mock では固定)
    snap = FactorSnapshot(values={"uncertainty": 0.6, "integrate": 0.4, "structurize": 0.4})

    def hooked_fitness(gene: StateUpdateGene, rng: np.random.Generator) -> float:
        # hook で gene の decay を動的調整してから evaluate
        eff_gene = apply_hook_to_gene(gene, hook, snap)
        return evaluate_gene(eff_gene, task, readout, rng, n_trials=3)

    result = evolve(hooked_fitness, pop_size=10, n_generations=10, rng=np.random.default_rng(2024))
    finite = all(np.isfinite(f) for f in result.best_fitness_curve)
    monotonic = all(
        result.best_fitness_curve[i + 1] >= result.best_fitness_curve[i] - 1e-9
        for i in range(len(result.best_fitness_curve) - 1)
    )
    ok = finite and monotonic and result.final_best.fitness > 0.0
    return ok, f"evolved 10x10 with hook, best={result.final_best.fitness:.3f}, monotonic={monotonic}"


def main() -> int:
    _ensure_utf8_stdout()
    print("=" * 72)
    print("PoC 2a — factor_hook × RWKV mock 接続 falsifiable verification")
    print("=" * 72)

    gates = [
        ("G1: FactorSnapshot 10 factors + clamp", gate_g1_factor_snapshot),
        ("G2: NoopFactorHook always 1.0", gate_g2_noop_returns_one),
        ("G3: HeuristicFactorHook directionality", gate_g3_heuristic_directionality),
        ("G4: apply_hook_to_gene modifies decay", gate_g4_apply_hook_modifies_decay),
        ("G5: snapshot distinguishability", gate_g5_snapshot_distinguishability),
        ("G6: determinism", gate_g6_determinism),
        ("G7: evolution smoke with hook", gate_g7_evolution_smoke_with_hook),
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
        print("PoC 2a verdict: PASS — factor_hook × state update kernel (mock) 接続成立.")
        print("                 認知状態が gene を動的調整、進化ループに統合可能.")
        print("                 → llcore Stage 0-2 全 PoC 完走 = CPU PoC battery 完成.")
        return 0
    print("PoC 2a verdict: FAIL — hook 設計を見直してから次へ.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
