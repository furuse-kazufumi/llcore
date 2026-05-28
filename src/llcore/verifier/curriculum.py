# SPDX-License-Identifier: Apache-2.0
"""MCC 風 ChangeOp curriculum — verifier-pass 率ベース淘汰 (Stage 3a 要件 C).

llive `poc_minimal_criterion_coevolution.py` (Read のみ) の MCC アイデアを ChangeOp
集団に適用:
- ChangeOp の minimal criterion = "Z3 refinement 検査が pass しつつ ε(c) が現 frontier
  以上 (= 自明すぎない)"
- 集団は世代を経るごとに ChangeOp magnitude (=難度) が漸増する frontier を持つ
  (auto-curriculum)
- 固定 ChangeOp 集合の上限を回避 (= 進化に上限を設けない、要件 C)

minimal criterion 設計 (LLM 苦手軸ベースでない、純 verifier-pass 率ベース):
- a ChangeOp survives iff:
    (i) verify_refinement_single() returns ok=True (sound 拡張保持) — pass 必要
    (ii) ε(c) >= frontier_quantile (現集団の上位 percentile に居る難度のみ生存)
- mutation: magnitude を gaussian noise で perturb (positive bias で frontier 押し上げ)
- 新規流入: 既存 frontier 周辺で random sample (探索維持)

POET-lite 風: 解 (= gene の安全性証明) と問 (= ChangeOp) を共進化させる構造。
ChangeOp が verifier 落とすほど難しすぎず、ε が小さすぎず frontier 維持される。

要件 B 無限列耐性との接続:
- curriculum で生成した ChangeOp 列を `verify_sequence_tolerance` に流し、
  100 step を超えても state_norm bound が崩れないことを連続検査可能。
- magnitude 漸増 (frontier 上昇) は monotone なので **Σ magnitude 不発散** が
  curriculum design 側 (e.g. magnitude_cap) の保証。

honest 留保:
- "pass 率高い ChangeOp ばかり残り単調になるリスク" (Q4) に対しては、frontier
  quantile (epsilon_floor) を底上げする pressure で対抗する。詳細は ``evolve_one_generation``
  の docstring の "anti-monotone" 段落参照。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from llcore.state_update import StateUpdateGene
from llcore.verifier.changeop import (
    ChangeOp,
    ChangeOpSequence,
    decay_shift,
    gate_shift,
    kernel_swap_mock,
    mix_shift,
)
from llcore.verifier.refinement import (
    epsilon_for,
    verify_refinement_single,
)


# ---------------------------------------------------------------------------
# Curriculum state
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CurriculumGeneration:
    """1 世代分の curriculum スナップショット.

    Attributes
    ----------
    generation : int
        世代番号 (0-origin).
    population : tuple[ChangeOp, ...]
        生存 ChangeOp 集団.
    pass_rate : float
        本世代の verifier-pass 率 (admit / total).
    epsilon_frontier : float
        本世代の frontier ε (上位 percentile cutoff). saturation 検出に使う.
    median_epsilon : float
        本世代の median ε (curriculum 難度の中央値).
    """

    generation: int
    population: tuple[ChangeOp, ...]
    pass_rate: float
    epsilon_frontier: float
    median_epsilon: float


@dataclass
class CurriculumState:
    """累積 curriculum 状態 (mutable; generation 履歴を保持).

    Attributes
    ----------
    generations : list[CurriculumGeneration]
        全世代スナップショット (進化軌跡).
    rng : random.Random
        deterministic 乱数源.
    magnitude_cap : float
        ChangeOp magnitude 上限 (Σ magnitude bounded 保証).
    """

    generations: list[CurriculumGeneration] = field(default_factory=list)
    rng: random.Random = field(default_factory=lambda: random.Random(20260529))
    magnitude_cap: float = 1.0

    @property
    def last_frontier(self) -> float:
        if not self.generations:
            return 0.0
        return self.generations[-1].epsilon_frontier

    @property
    def frontier_slope(self) -> float:
        """直近 frontier の勾配 (G8 auto-curriculum 検査用)."""
        if len(self.generations) < 2:
            return 0.0
        recent = self.generations[-min(5, len(self.generations)):]
        x = list(range(len(recent)))
        y = [g.epsilon_frontier for g in recent]
        if len(x) < 2:
            return 0.0
        # 単純線形回帰の傾き
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den = sum((xi - mean_x) ** 2 for xi in x)
        if den == 0:
            return 0.0
        return num / den


# ---------------------------------------------------------------------------
# population init / random sample
# ---------------------------------------------------------------------------


def _random_changeop(rng: random.Random, max_mag: float) -> ChangeOp:
    """frontier 構築用 random ChangeOp.

    op_type は uniform、delta は |·| <= max_mag の uniform。
    kernel_swap_mock は Bernoulli(0.2) で混入 (discrete 変更の探索維持)。
    """
    if rng.random() < 0.2:
        return kernel_swap_mock(swap=(rng.random() < 0.5))
    op_type = rng.choice(("decay_shift", "mix_shift", "gate_shift"))
    delta = rng.uniform(-max_mag, max_mag)
    ctor = {
        "decay_shift": decay_shift,
        "mix_shift": mix_shift,
        "gate_shift": gate_shift,
    }[op_type]
    return ctor(delta)


def initial_population(
    n: int = 32, *, max_mag: float = 0.5, seed: int = 20260529
) -> tuple[ChangeOpSequence, CurriculumState]:
    """初期集団 + 空 curriculum state を作る."""
    state = CurriculumState(rng=random.Random(seed))
    pop = tuple(_random_changeop(state.rng, max_mag) for _ in range(n))
    return ChangeOpSequence(ops=pop), state


# ---------------------------------------------------------------------------
# evolve one generation (MCC core)
# ---------------------------------------------------------------------------


def evolve_one_generation(
    gene: StateUpdateGene,
    population: tuple[ChangeOp, ...],
    state: CurriculumState,
    *,
    epsilon_floor_quantile: float = 0.5,
    mutation_sigma: float = 0.05,
    state_bound: float = 1.0,
    max_input_abs: float = 1.0,
    timeout_ms: int = 200,
    refill_to: int | None = None,
) -> tuple[ChangeOp, ...]:
    """MCC 1 世代の淘汰 + 変異 + refill を実行.

    手順:
    1. 各 ChangeOp に対し verify_refinement_single() を呼び ok/ε を取る
    2. ok=True ∧ ε >= frontier_floor を満たすものを生存
    3. 生存集団に gaussian mutation (magnitude を sigma で perturb)
    4. 不足分を random sample で補充 (frontier 周辺 +α で探索維持)
    5. CurriculumGeneration をスナップショット → state.generations に append

    "anti-monotone" pressure (Q4 対応):
        epsilon_floor_quantile を 0.5 (median) 以上に設定すると、frontier の下位
        半数が淘汰される → 上位だけ残り pass 率高い ChangeOp が固定化しがちに
        なる。これを避けるため:
        - 生存集団に "noise mutation" で magnitude を ±sigma perturb (退行も許容)
        - refill で frontier 上限の **少し上** からも random sample (frontier 押上)
        この 2 つで monotone-only への退化を防ぐ。

    Returns
    -------
    new_population : tuple[ChangeOp, ...]
        次世代集団 (生存 + 変異 + refill).
    """
    refill_to = refill_to or len(population)

    # 1. 評価
    results = []
    for op in population:
        r = verify_refinement_single(
            gene,
            op,
            state_bound=state_bound,
            max_input_abs=max_input_abs,
            timeout_ms=timeout_ms,
        )
        results.append((op, r.ok, r.epsilon))

    n = len(results)
    n_admit = sum(1 for _, ok, _ in results if ok)
    pass_rate = n_admit / n if n else 0.0

    # 2. frontier quantile cutoff (生存集団から計算)
    admitted_eps = sorted([eps for _, ok, eps in results if ok])
    if admitted_eps:
        cut_idx = int(len(admitted_eps) * epsilon_floor_quantile)
        cut_idx = max(0, min(len(admitted_eps) - 1, cut_idx))
        epsilon_floor = admitted_eps[cut_idx]
        median_eps = admitted_eps[len(admitted_eps) // 2]
    else:
        epsilon_floor = 0.0
        median_eps = 0.0

    survivors = [op for op, ok, eps in results if ok and eps >= epsilon_floor]

    # 3. mutation
    rng = state.rng
    mutated: list[ChangeOp] = []
    for op in survivors:
        if op.op_type == "kernel_swap_mock":
            # discrete: 確率 sigma で swap 反転
            if rng.random() < mutation_sigma:
                mutated.append(kernel_swap_mock(swap=op.delta == 0.0))
            else:
                mutated.append(op)
        else:
            new_delta = op.delta + rng.gauss(0.0, mutation_sigma)
            new_delta = max(-state.magnitude_cap, min(state.magnitude_cap, new_delta))
            ctor = {
                "decay_shift": decay_shift,
                "mix_shift": mix_shift,
                "gate_shift": gate_shift,
            }[op.op_type]
            mutated.append(ctor(new_delta))

    # 4. refill: frontier の少し上から sample (anti-monotone pressure)
    while len(mutated) < refill_to:
        # frontier の +α 上を狙う
        target_mag = min(state.magnitude_cap, max(epsilon_floor + 0.05, 0.1))
        mutated.append(_random_changeop(rng, max_mag=target_mag))

    new_pop = tuple(mutated[:refill_to])

    # 5. snapshot
    gen_idx = len(state.generations)
    snapshot = CurriculumGeneration(
        generation=gen_idx,
        population=new_pop,
        pass_rate=pass_rate,
        epsilon_frontier=epsilon_floor,
        median_epsilon=median_eps,
    )
    state.generations.append(snapshot)
    return new_pop


def run_curriculum(
    gene: StateUpdateGene,
    *,
    n_generations: int = 8,
    pop_size: int = 24,
    initial_max_mag: float = 0.3,
    seed: int = 20260529,
    state_bound: float = 1.0,
    max_input_abs: float = 1.0,
    per_changeop_timeout_ms: int = 200,
    epsilon_floor_quantile: float = 0.5,
    mutation_sigma: float = 0.05,
    magnitude_cap: float = 1.0,
) -> CurriculumState:
    """N 世代 curriculum を回す convenience wrapper.

    "進化に上限を設けない" 要件:
        magnitude_cap (= 個別 ChangeOp の上限) は state_bound と E_BASE から
        sequence_tolerance が崩れない範囲で設定する。世代数 N は任意であり、
        frontier_slope が plateau しない限り無限に走らせて意味がある (POET-lite
        spirit).

    Returns
    -------
    CurriculumState
        全世代スナップショット付き final state.
    """
    state = CurriculumState(
        rng=random.Random(seed),
        magnitude_cap=magnitude_cap,
    )
    pop = tuple(
        _random_changeop(state.rng, max_mag=initial_max_mag) for _ in range(pop_size)
    )
    for _ in range(n_generations):
        pop = evolve_one_generation(
            gene,
            pop,
            state,
            epsilon_floor_quantile=epsilon_floor_quantile,
            mutation_sigma=mutation_sigma,
            state_bound=state_bound,
            max_input_abs=max_input_abs,
            timeout_ms=per_changeop_timeout_ms,
            refill_to=pop_size,
        )
    return state


def is_saturated(state: CurriculumState, *, plateau_eps: float = 1e-4) -> bool:
    """frontier_slope が plateau_eps 未満なら saturation 発生とみなす.

    G6.b "上限なし" の falsify 用: 真に上限なしなら is_saturated=False を期待。
    """
    if len(state.generations) < 3:
        return False
    return abs(state.frontier_slope) < plateau_eps and math.isfinite(state.frontier_slope)
