# SPDX-License-Identifier: Apache-2.0
"""llcore 自前 minimal GA — tournament + uniform mutation の素朴 GA.

設計判断 (ユーザー指示 2026-05-29):
- llive lldarwin_v2 / Genome3D / persona_evolution への依存なし
- state_update gene 3 次元に特化、素朴 tournament で十分
- 拡張は v0.2+ で (ε-lexicase / novelty 等は自前実装で追従、llive 参考のみ)

最小 API:
- :class:`Individual` — gene + fitness の tuple
- :class:`Population` — frozen container
- :func:`tournament_select` — k 個 random sample → max fitness
- :func:`uniform_mutate` — 各 gene parameter に gaussian noise
- :func:`crossover_uniform` — 2 親の gene を独立に親 0/1 から選ぶ
- :func:`evolve` — 進化 main loop (CPU 完結, deterministic with seed)

破綻防止:
- 全滅回避: tournament k <= pop_size で構造的保証
- 数値安定: gene.clipped() で範囲外を抑える
- 単調非減少 best: elitism (top-1 必ず次世代に残す)
- 決定論性: rng を毎呼出で渡す
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import numpy as np

from llcore.state_update import StateUpdateGene

if TYPE_CHECKING:
    # 型注釈のみ (実行時 import なし = import cycle 完全回避)。
    # GeneCodec は llcore.kernel.protocol で定義 (S1)。
    from llcore.kernel.protocol import GeneCodec

# S2: gene 型を一般化する TypeVar。RWKV では StateUpdateGene に束縛される。
GeneT = TypeVar("GeneT")


@dataclass(frozen=True)
class Individual(Generic[GeneT]):
    """gene + fitness の immutable tuple.

    S2 (0.2.0a0): ``gene`` を ``Generic[GeneT]`` 化。``Individual(gene=StateUpdateGene(...),
    fitness=...)`` の既存呼び出しは型・実行とも不変 (後方互換)。
    """

    gene: GeneT
    fitness: float


@dataclass(frozen=True)
class Population(Generic[GeneT]):
    """immutable individual list (順序保持).

    S2: ``Generic[GeneT]`` 化。:attr:`gene_matrix` は ``gene.as_array()`` の duck typing
    に依存する (RWKV / LIF は両方 ``as_array`` を持つ)。``as_array`` を持たない gene 型では
    :func:`evolve` 側が codec 経由で diversity を計算するため、gene_matrix は呼ばれない。
    """

    individuals: tuple[Individual[GeneT], ...]

    @property
    def size(self) -> int:
        return len(self.individuals)

    @property
    def best(self) -> Individual:
        return max(self.individuals, key=lambda ind: ind.fitness)

    @property
    def fitness_array(self) -> np.ndarray:
        return np.array([ind.fitness for ind in self.individuals], dtype=np.float64)

    @property
    def gene_matrix(self) -> np.ndarray:
        """shape (N, 3): (decay, mix, gate_str) per row."""
        return np.array([ind.gene.as_array() for ind in self.individuals])


# ---------------------------------------------------------------------------
# selection
# ---------------------------------------------------------------------------


def tournament_select(
    pop: Population, k: int, rng: np.random.Generator
) -> Individual:
    """k 個体を random sampling し最大 fitness を返す.

    k=1 で random 選択、k=size で best (greedy)。標準は k=3。
    """
    if k < 1 or k > pop.size:
        raise ValueError(f"k={k} must be in [1, {pop.size}]")
    indices = rng.choice(pop.size, size=k, replace=False)
    candidates = [pop.individuals[i] for i in indices]
    return max(candidates, key=lambda ind: ind.fitness)


# ---------------------------------------------------------------------------
# mutation / crossover
# ---------------------------------------------------------------------------


def uniform_mutate(
    gene: StateUpdateGene, sigma: float, rng: np.random.Generator
) -> StateUpdateGene:
    """各 gene parameter に独立 gaussian noise (σ=sigma).

    clip 範囲 (decay [0,1], mix [-1,1], gate_str [-2,2]) を gene.clipped() で守る。
    """
    noise = rng.normal(0.0, sigma, size=3)
    arr = gene.as_array() + noise
    return StateUpdateGene.from_array(arr).clipped()


def crossover_uniform(
    parent_a: StateUpdateGene,
    parent_b: StateUpdateGene,
    rng: np.random.Generator,
) -> StateUpdateGene:
    """各 parameter を 50/50 で a/b から独立に選ぶ (uniform crossover)."""
    arr_a = parent_a.as_array()
    arr_b = parent_b.as_array()
    mask = rng.integers(0, 2, size=3).astype(bool)
    child_arr = np.where(mask, arr_a, arr_b)
    return StateUpdateGene.from_array(child_arr).clipped()


# ---------------------------------------------------------------------------
# S2: gene 型非依存の汎用 operator (GeneCodec 経由)
#
# RWKV codec に対しては上記 RWKV 専用版と **RNG ストリームまで byte-identical**:
# - mutate:  rng.normal(0, sigma, size=dim) は dim=3 で同一 draw
# - crossover: rng.integers(0, 2, size=dim) は dim=3 で同一 draw
# - init:    各 param を per-element scalar rng.uniform(lo, hi) で順に draw (旧版と同順)
# RWKV in-range gene では codec.clip が identity なので fitness も byte-identical。
# (test_kernel_ga_generalization で evolve(codec=RWKVCodec()) == evolve(codec=None) を証明)
# ---------------------------------------------------------------------------


def uniform_mutate_g(
    gene: GeneT, codec: "GeneCodec[GeneT]", sigma: float, rng: np.random.Generator
) -> GeneT:
    """gene 型非依存の uniform mutation (codec 経由).

    各 param に独立 gaussian noise (σ=sigma) → codec.clip で範囲制約を反映。
    """
    noise = rng.normal(0.0, sigma, size=codec.dim)
    arr = codec.to_array(gene) + noise
    return codec.clip(codec.from_array(arr))


def crossover_uniform_g(
    parent_a: GeneT,
    parent_b: GeneT,
    codec: "GeneCodec[GeneT]",
    rng: np.random.Generator,
) -> GeneT:
    """gene 型非依存の uniform crossover (codec 経由).

    各 param を 50/50 で a/b から独立に選ぶ。
    """
    arr_a = codec.to_array(parent_a)
    arr_b = codec.to_array(parent_b)
    mask = rng.integers(0, 2, size=codec.dim).astype(bool)
    child_arr = np.where(mask, arr_a, arr_b)
    return codec.clip(codec.from_array(child_arr))


def initialize_random_population_g(
    pop_size: int, codec: "GeneCodec[GeneT]", rng: np.random.Generator
) -> list[GeneT]:
    """gene 型非依存の random 集団生成 (codec.lower/upper box 内 uniform).

    各 param を per-element scalar ``rng.uniform(lo, hi)`` で **順に** draw する
    (RWKV 専用 :func:`initialize_random_population` の draw 順と一致させ byte-identity を保つ)。
    codec.clip で依存制約 (LIF の V_reset<V_th 等) を反映。
    """
    lower = codec.lower
    upper = codec.upper
    genes: list[GeneT] = []
    for _ in range(pop_size):
        arr = np.array(
            [float(rng.uniform(lo, hi)) for lo, hi in zip(lower, upper)],
            dtype=np.float64,
        )
        genes.append(codec.clip(codec.from_array(arr)))
    return genes


# ---------------------------------------------------------------------------
# evolution loop
# ---------------------------------------------------------------------------


FitnessFunc = Callable[[StateUpdateGene, np.random.Generator], float]
# S2 honest 注記 (Codex/semver review): 設計 doc §3.2 M3 は FitnessFunc 自体の TypeVar 化を
# 提案したが、既存 import の型エラー回避 (後方互換厳守) のため FitnessFunc は **具象のまま据え置き**、
# 一般 gene 型用に FitnessFuncG を additive 追加する。実行時は両者とも duck typing で gene 型非依存。
FitnessFuncG = Callable[[GeneT, np.random.Generator], float]


# ---------------------------------------------------------------------------
# T1 Phase 1 (a): 証明ゲートの本配線 (additive / 後方互換)
#
# research/verified_evolution/gated_evolve.py の検問規律 (fail-closed resample +
# known-safe fallback) を src 品質で evolve() に移植する。``gate_mode="none"`` (既定)
# では一切ゲートを呼ばず追加 RNG draw も消費しない = 旧挙動 byte-identical。gated mode
# では子個体生成のたびに証明し、不合格は resample、cap 到達で known-safe fallback。
#
# 設計の互換性:
# - evolve() の新引数 ``gate_mode`` / ``resample_cap`` はデフォルトで現行挙動を完全保存。
# - EvolutionResult に optional ``gate_stats`` フィールド (default None) を additive 追加。
#   既存の keyword 構築 (gated_evolve.py / minimal_ga.py) は無改変で動作する。
# - gate は scalar StateUpdateGene を対象に verify_gene_safe / verify_lipschitz_contraction
#   を呼ぶ。これらは内部で gene.clipped() を通すため codec パスとも整合する。
# ---------------------------------------------------------------------------


# 証明ゲートのモード文字列 ("none" は無ゲート control = 旧挙動 byte-identical)。
GateMode = str  # "none" | "state_norm" | "contraction" | "trajectory_tube"

# Known-safe fallback gene: decay=0.5, mix=0.0, gate_str=0.0.
# 閉形式 Lipschitz 上界 = max(|0.5|, |0.5 + 0.5*0.0|) = 0.5 < 1 (state-direction
# contraction)、かつ convex combination で |s|<=1 (state_norm)。gated_evolve.py の
# FALLBACK_GENE と同一値 (両実装の挙動一致を保つため)。
_FALLBACK_GENE = StateUpdateGene(decay=0.5, mix=0.0, gate_str=0.0)


def _gate_admits(
    gene: StateUpdateGene,
    mode: GateMode,
    *,
    w_bar: float | None = None,
    r_max: float | None = None,
) -> bool:
    """``gene`` が ``mode`` の証明ゲートに admit されるか (fail-closed).

    research/verified_evolution/gated_evolve.py::_gate_admits と挙動一致
    ("none"/"state_norm"/"contraction" の 3 mode)。Phase 2a で
    ``"trajectory_tube"`` を additive 追加 (1 分岐)。

    Parameters
    ----------
    gene : StateUpdateGene
        検査対象 gene (検証器が内部で clipped() を通す)。
    mode : GateMode
        - ``"none"``        — 無条件 admit (ゲート無効; 旧挙動)。
        - ``"state_norm"``  — ``verify_gene_safe(gene).ok`` (Z3 |s|<=1 gate)。
        - ``"contraction"`` — ``verify_lipschitz_contraction(gene).contraction is True``
          (Z3 L<1 gate; fail-closed: hard True のみ admit)。
        - ``"trajectory_tube"`` — Phase 2a。``tracking_tube(gene, w_bar=w_bar,
          r_max=r_max).admits`` (L<1 ∧ tube 有限 ∧ r = G·w̄/(1−L) ≤ r_max)。
          **honest 注記 (設計 doc §4.3 訂正 1)**: この gate の L は achievable-t box
          上の閉形式 numpy 比較であり、``"contraction"`` の free-t Z3 unsat 証明とは
          **別の L 定義** (non-comparable)。「Z3-exact contraction + tube」ではなく
          「閉形式 (Banach 系) で導かれた tube 境界」である。
    w_bar : float | None
        ``"trajectory_tube"`` 用の外乱上界 (≥0)。他 mode では未使用。None かつ
        trajectory_tube → fail-loud (黙って通さない)。
    r_max : float | None
        ``"trajectory_tube"`` 用の tube 半径許容上限。None なら上限なし
        (tube 有限であれば admit)。

    Returns
    -------
    bool
        admit するなら True。

    Raises
    ------
    ValueError
        未知の ``mode`` (fail-loud)。trajectory_tube で w_bar 未指定 (fail-loud)。
    """
    if mode == "none":
        return True
    # 遅延 import (import cycle 回避 + gate 無効時のオーバーヘッドゼロ)。
    from llcore.verifier.invariants import (
        verify_gene_safe,
        verify_lipschitz_contraction,
    )

    if mode == "state_norm":
        return bool(verify_gene_safe(gene).ok)
    if mode == "contraction":
        # fail-closed: Z3 unsat (certified) の hard True のみ admit。
        # None (z3 不在) / False (sat / timeout) はすべて reject 側。
        return verify_lipschitz_contraction(gene).contraction is True
    if mode == "trajectory_tube":
        # Phase 2a: trajectory-tube 含包ゲート (additive 1 分岐)。
        # tracking_tube は read-only で contraction_ok ∧ tube 有限 ∧ tube≤r_max を
        # 内部で AND する (fail-closed)。ここでは例外も握り潰して reject 扱いにする
        # (設計 doc §2.1 / §4.3: admit=False / 例外 / tube=∞ / r>r_max は全て reject)。
        if w_bar is None:
            raise ValueError(
                "gate_mode='trajectory_tube' requires w_bar (disturbance bound); "
                "got None (fail-loud, do not admit silently)"
            )
        from llcore.verifier.tracking_tube import tracking_tube

        try:
            return bool(tracking_tube(gene, w_bar=w_bar, r_max=r_max).admits)
        except Exception:
            # tube 計算が例外 (想定外の gene 等) → fail-closed reject。
            return False
    raise ValueError(f"unknown gate_mode {mode!r}")


@dataclass(frozen=True)
class GateStats:
    """証明ゲートの集計 (gated mode 時のみ ``EvolutionResult.gate_stats`` に格納).

    Attributes
    ----------
    gate_mode : GateMode
        使用したゲートモード。
    n_rejections : int
        ゲートに reject された子の総数 (全 resample を跨いで計上)。
    n_resamples : int
        reject 後に再生成した resample 試行の総数。
    fallback_count : int
        resample cap に達して known-safe fallback gene を採用した回数。
    n_children_generated : int
        集団に admit された (非 elite) 子の総数。
    """

    gate_mode: GateMode
    n_rejections: int
    n_resamples: int
    fallback_count: int
    n_children_generated: int


def evaluate_population(
    genes: list[StateUpdateGene],
    fitness_func: FitnessFunc,
    rng: np.random.Generator,
) -> Population:
    """全個体の fitness 計算 → Population を返す."""
    individuals: list[Individual] = []
    for gene in genes:
        f = fitness_func(gene, rng)
        individuals.append(Individual(gene=gene, fitness=float(f)))
    return Population(individuals=tuple(individuals))


def initialize_random_population(
    pop_size: int, rng: np.random.Generator
) -> list[StateUpdateGene]:
    """random gene 集団を生成 (clip 範囲内 uniform)."""
    return [
        StateUpdateGene(
            decay=float(rng.uniform(0.0, 1.0)),
            mix=float(rng.uniform(-1.0, 1.0)),
            gate_str=float(rng.uniform(-2.0, 2.0)),
        )
        for _ in range(pop_size)
    ]


@dataclass(frozen=True)
class EvolutionResult:
    """進化結果のスナップショット.

    Attributes
    ----------
    generations : tuple[Population, ...]
        世代ごとの Population (世代 0 から最終世代まで).
    best_fitness_curve : tuple[float, ...]
        各世代の best fitness (進化進捗の主指標).
    diversity_curve : tuple[float, ...]
        各世代の gene_matrix 分散 (gene 多様性指標).
    gate_stats : GateStats | None
        T1 Phase 1 (a): 証明ゲート集計。``gate_mode="none"`` (既定) では ``None``
        (additive; 既存呼び出しと後方互換)。gated mode 時のみ :class:`GateStats` が入る。
    """

    generations: tuple[Population, ...]
    best_fitness_curve: tuple[float, ...]
    diversity_curve: tuple[float, ...]
    gate_stats: "GateStats | None" = field(default=None)

    @property
    def final_best(self) -> Individual:
        return self.generations[-1].best


def evolve(
    fitness_func: FitnessFunc,
    *,
    pop_size: int = 10,
    n_generations: int = 10,
    tournament_k: int = 3,
    mutation_sigma: float = 0.15,
    crossover_rate: float = 0.5,
    elitism: int = 1,
    rng: np.random.Generator | None = None,
    initial_pop: list[StateUpdateGene] | None = None,
    codec: "GeneCodec | None" = None,
    gate_mode: GateMode = "none",
    resample_cap: int = 50,
    w_bar: float | None = None,
    r_max: float | None = None,
) -> EvolutionResult:
    """進化 main loop.

    Parameters
    ----------
    fitness_func : Callable[[gene, rng], float]
        個体評価関数。rng は再現性のため受け渡し。
    pop_size : int
        集団サイズ。
    n_generations : int
        世代数。
    tournament_k : int
        tournament 選択 size (1=random, pop_size=greedy)。
    mutation_sigma : float
        gaussian noise sigma。
    crossover_rate : float
        crossover を実行する確率 (それ以外は mutation のみ)。
    elitism : int
        各世代で上位 N を必ず次世代に残す。
    rng : np.random.Generator
        seed 固定の再現性のため必須 (None なら default_rng())。
    initial_pop : list[StateUpdateGene] | None
        初期集団。None なら random 生成。
    codec : GeneCodec | None
        S2 (0.2.0a0): gene 型非依存化のための codec。

        - ``None`` (既定) → **RWKV 専用旧パス** (``initialize_random_population`` /
          ``uniform_mutate`` / ``crossover_uniform`` をそのまま使用、挙動 byte-identical)。
        - codec 指定 → 汎用 ``*_g`` operator を codec 経由で使用 (任意 gene 型に対応)。
          diversity も codec.to_array で計算 (gene が ``as_array`` を持たなくてよい)。

        RWKV では ``evolve(codec=RWKVCodec())`` と ``evolve(codec=None)`` は byte-identical
        (operator が RNG ストリームまで一致する設計)。
    gate_mode : GateMode
        T1 Phase 1 (a): 子個体に課す証明ゲート (additive / 後方互換)。

        - ``"none"`` (既定) → **無ゲート control**。ゲートを 1 度も呼ばず追加 RNG draw も
          消費しないため、旧挙動と **byte-identical** (既存テスト全 green を保証)。
        - ``"state_norm"`` → ``verify_gene_safe(child).ok`` を満たす子のみ admit
          (Z3 |s|<=1 gate)。
        - ``"contraction"`` → ``verify_lipschitz_contraction(child).contraction is True``
          を満たす子のみ admit (Z3 L<1 gate; fail-closed)。
        - ``"trajectory_tube"`` (Phase 2a) → ``tracking_tube(child, w_bar=w_bar,
          r_max=r_max).admits`` を満たす子のみ admit (L<1 ∧ tube 有限 ∧
          r=G·w̄/(1−L)≤r_max; fail-closed)。``w_bar`` 必須。

          honest 注記 (設計 doc §4.3 訂正 1): trajectory_tube の L は achievable-t
          box 上の **閉形式 numpy 比較**であり、``"contraction"`` の free-t Z3 unsat
          証明とは別の L 定義 (non-comparable)。

        gated mode では reject された子を ``resample_cap`` 回まで再生成 (fail-closed)、
        cap 到達で known-safe fallback gene を採用する。集計は :class:`GateStats` として
        ``EvolutionResult.gate_stats`` に格納される。
        **scalar StateUpdateGene 専用** (``codec`` 併用時は ``gate_mode="none"`` 必須;
        現行ゲート検証器が scalar gene を前提とするため)。
    resample_cap : int
        gated mode で 1 子あたり許す resample 回数の上限。cap 到達で fallback。
    w_bar : float | None
        ``gate_mode="trajectory_tube"`` の外乱上界 ‖d‖∞≤w̄ (≥0)。他 mode では未使用。
        trajectory_tube で None → fail-loud。
    r_max : float | None
        ``gate_mode="trajectory_tube"`` の tube 半径許容上限。None なら上限なし。

    Returns
    -------
    EvolutionResult
        全世代のスナップショット + best_fitness_curve + diversity_curve。
        gated mode では ``gate_stats`` に :class:`GateStats` が入る (それ以外は None)。

    Raises
    ------
    ValueError
        ``codec`` と gated ``gate_mode`` の併用 (未対応の組合せ; fail-loud)。
    """
    if rng is None:
        rng = np.random.default_rng()
    if elitism > pop_size:
        raise ValueError(f"elitism={elitism} must be <= pop_size={pop_size}")
    if tournament_k > pop_size:
        raise ValueError(f"tournament_k={tournament_k} must be <= pop_size={pop_size}")
    if gate_mode != "none" and codec is not None:
        # 現行の証明ゲートは scalar StateUpdateGene 前提 (coupled gene 用ゲートは
        # tracking_tube / backends 側に別配線)。誤用を fail-loud で弾く。
        raise ValueError(
            "gated gate_mode is only supported for the scalar StateUpdateGene path "
            "(codec=None); coupled-gene gating uses verifier backends instead"
        )

    # S2: operator 選択 (codec=None は旧 RWKV パスを完全保存 = byte-identical)。
    if codec is None:
        def _init(n: int, r: np.random.Generator):
            return initialize_random_population(n, r)

        def _mut(g, r: np.random.Generator):
            return uniform_mutate(g, mutation_sigma, r)

        def _cx(a, b, r: np.random.Generator):
            return crossover_uniform(a, b, r)

        def _div(p: Population) -> float:
            return float(p.gene_matrix.var())
    else:
        def _init(n: int, r: np.random.Generator):
            return initialize_random_population_g(n, codec, r)

        def _mut(g, r: np.random.Generator):
            return uniform_mutate_g(g, codec, mutation_sigma, r)

        def _cx(a, b, r: np.random.Generator):
            return crossover_uniform_g(a, b, codec, r)

        def _div(p: Population) -> float:
            mat = np.array([codec.to_array(ind.gene) for ind in p.individuals])
            return float(mat.var())

    # 初期集団
    if initial_pop is None:
        initial_pop = _init(pop_size, rng)
    elif len(initial_pop) != pop_size:
        raise ValueError(f"initial_pop size {len(initial_pop)} != pop_size {pop_size}")

    def _make_child(p: Population):
        """tournament(+crossover)+mutate で子 gene を 1 個生成する.

        RNG draw 順は ``"none"`` mode で旧 ``evolve()`` と完全一致:
        tournament_select → rng.random() → [tournament_select + crossover] → mutate。
        """
        parent_a = tournament_select(p, tournament_k, rng)
        if rng.random() < crossover_rate:
            parent_b = tournament_select(p, tournament_k, rng)
            child = _cx(parent_a.gene, parent_b.gene, rng)
        else:
            child = parent_a.gene
        return _mut(child, rng)

    pop = evaluate_population(initial_pop, fitness_func, rng)
    generations: list[Population] = [pop]
    best_curve: list[float] = [pop.best.fitness]
    diversity_curve: list[float] = [_div(pop)]

    # T1 Phase 1 (a): ゲート集計カウンタ (gate_mode="none" では全 0 のまま使用しない)。
    n_rejections = 0
    n_resamples = 0
    fallback_count = 0
    n_children = 0

    for _gen in range(n_generations):
        # elitism: 上位 N を子に「fitness ごと」持ち越し (再評価しない)
        #
        # fitness は確率的 (fitness_func 内 task.generate が rng 使用) なので、
        # elite を再評価すると fitness が変動して best 単調性が崩れる。
        # → elite は前世代の Individual (gene + fitness) をそのまま次世代に。
        sorted_inds = sorted(pop.individuals, key=lambda ind: -ind.fitness)
        elites: list[Individual] = list(sorted_inds[:elitism])

        # 残り (pop_size - elitism) を tournament + mutation/crossover で生成し新評価
        new_genes: list = []
        while len(new_genes) < pop_size - elitism:
            child = _make_child(pop)
            if gate_mode == "none":
                # control: ゲートを呼ばず追加 draw も消費しない = 旧挙動 byte-identical。
                new_genes.append(child)
                continue
            # gated: 子を証明 → 不合格は fail-closed resample → cap 到達で fallback。
            if _gate_admits(child, gate_mode):
                new_genes.append(child)
                n_children += 1
                continue
            n_rejections += 1
            admitted = False
            for _ in range(resample_cap):
                n_resamples += 1
                child = _make_child(pop)
                if _gate_admits(child, gate_mode):
                    new_genes.append(child)
                    n_children += 1
                    admitted = True
                    break
                n_rejections += 1
            if not admitted:
                # cap 到達 → known-safe fallback (RNG draw を消費しない)。
                new_genes.append(_FALLBACK_GENE)
                n_children += 1
                fallback_count += 1

        new_pop = evaluate_population(new_genes, fitness_func, rng)
        # elite (旧 fitness 維持) + 新個体 (新評価) を結合
        pop = Population(individuals=tuple(elites) + new_pop.individuals)
        generations.append(pop)
        best_curve.append(pop.best.fitness)
        diversity_curve.append(_div(pop))

    gate_stats = (
        None
        if gate_mode == "none"
        else GateStats(
            gate_mode=gate_mode,
            n_rejections=n_rejections,
            n_resamples=n_resamples,
            fallback_count=fallback_count,
            n_children_generated=n_children,
        )
    )
    return EvolutionResult(
        generations=tuple(generations),
        best_fitness_curve=tuple(best_curve),
        diversity_curve=tuple(diversity_curve),
        gate_stats=gate_stats,
    )
