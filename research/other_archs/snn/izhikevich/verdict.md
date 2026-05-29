# PoC SNN Izhikevich — LIF への llcore approach 移植 **一般化** verdict (2026-05-29)

## 命題 (falsifiable)

> Izhikevich 神経モデル (2D ODE) の 4 パラメータ ``(a, b, c, d) ∈ R^4`` を gene 化し、
> Z3 で per-gene
> **(a) firing rate bound (refractory なしだが dt-discretization から導出) +
> (b) v bounded invariant (forward Euler 1-step + assumed-input contract)**
> を検査することで llcore approach が **LIF より広い firing pattern 表現**
> (**RS / IB / CH / FS** の 4 firing-type を単一 gene family で含む) の進化空間でも
> 成立する mechanism を実証 (CPU 完結, 32 個体 × 30 世代).
>
> **Stage 2.2a (LIF) の同 design pattern** を踏襲し、AdaptiveFloorGate は直接 reuse、
> LineageReservoir / ModesMeter は IzhikevichGene 専用 minimal 再実装 (= Neural ODE / LIF
> と同じ「partial stack reuse」pattern).

## 設計概要

### Izhikevich gene (4 パラメータ)

```
dv/dt = 0.04 v^2 + 5 v + 140 - u + I
du/dt = a (b v - u)
if v >= 30 mV: v <- c, u <- u + d  (spike + reset)
gene = (a, b, c, d) ∈ R^4
```

物理範囲 clip (Izhikevich 2003 原典 Table 1 + RS/FS/IB/CH を覆う):

| param | range | 役割 | 4 type canonical |
|---|---|---|---|
| a | [0.01, 0.10] | u recovery 速度 | RS/IB/CH: 0.02, FS: 0.1 |
| b | [0.20, 0.30] | subthreshold sensitivity | RS/IB/CH/FS: 0.2 (進化幅で 0.25-0.3 含む) |
| c | [-65, -50] mV | reset value | RS/FS: -65, IB: -55, CH: -50 |
| d | [2, 8] | recovery 増分 | RS: 8, IB: 4, CH/FS: 2 |

固定: ``V_PEAK = 30 mV``, ``V_INIT = -70 mV``, ``DT = 0.25 ms``, ``I_MAX_ABS = 10.0`` (assumed-input contract).

### Z3 invariant (2 種)

#### (1) v bounded (per-gene, 1-step forward Euler)

```
v_next = v + DT * (0.04 v^2 + 5 v + 140 - u + I)
```

エンコーディング:

- ``v ∈ [V_PRE_MIN=-80, V_PEAK=30] mV``
- ``u ∈ [U_MIN=-25, U_MAX=25]``
- ``I ∈ [-I_max, +I_max]`` (assumed-input contract, Codex F3 対応で Stage 2.2a と同じ)
- 違反: ``v_next > V_PEAK + safety_margin OR v_next < V_PRE_MIN - safety_margin``
- unsat なら invariant 成立

honest 留保:

- ``v^2`` 非線形項により 1-step worst-case overshoot は LIF より大きい. v=30, u=U_MIN, I=I_max で
  ``DT * (0.04*900 + 150 + 140 + 25 + 10) ≈ 0.25 * 361 ≈ 90 mV``. 既定 safety_margin=100 mV.
- Z3 NRA (quantifier-free nonlinear real arithmetic, CAD-based) で v^2 を含む式を solve.
- 「continuous-time dynamics 保存」claim せず、「**discrete-time forward Euler 1-step map** の
  invariant」と明示 (Codex F2 教訓).

#### (2) firing rate 上界 (per-gene, dt-discretization)

```
spike 列 t_1 < ... < t_n ⊆ [0, T_window], t_{i+1} - t_i >= dt
⇒ n <= 1 + T_window / dt
⇒ rate <= 1000 / dt Hz   (dt=0.25 ms → 4000 Hz 自明上界)
```

エンコーディング: 違反 ``(n-1) * dt > T_window`` を Z3 で sat 探索.

honest 留保 (重要):

- LIF の refractory bound (``1000 / t_ref`` Hz) より緩い. これは「Izhikevich は
  **明示的不応期なし** (c + d でリカバリ)」という構造的特徴を honest に反映.
- claim 降格: 「physiological refractory bound」ではなく「**discrete-time discretization
  bound**」. Stage 2.2a LIF と同じ overclaim 回避 pattern.
- per-gene API は LIF と signature 揃え (gene 自体は使わず dt のみで判定).

### 進化器 (自前 minimal GA, LIF 版踏襲)

- 集団 32 = 4 firing-type × 8 個体
- 世代 30 (LIF の 50 世代より短縮: CPU 効率 + Izhikevich Z3 latency 大きめ)
- selection: AdaptiveFloorGate (llcore.evolution.adaptive_floor) で下位 30% 切り
- mutation: gaussian σ = 0.08 × range
- crossover: uniform (各軸独立 50%)
- elitism: 1
- fitness: ``1 / (1 + |measured_rate - target_rate| / max(target, 1))``
- 入力: 定常 DC ``I = I_value + N(0, 1.0)`` (Izhikevich 2003 fig.1 方式)
- MCC curriculum: ``I_value: 5 → 12``, ``target_rate: 30 → 80 Hz``

### Open-ended 4 機構 (llive 由来 → llcore 再実装 → SNN/Izhikevich 移植)

- (a) **AdaptiveFloorGate** (llcore.evolution import 直接 reuse) — percentile=30, ratchet=True
- (b) **IzhLineageReservoir** (IzhikevichGene 専用 minimal, llcore LineageReservoir 構造踏襲) — 4 firing-type の best-ever 保持
- (c) **IzhModesMeter** (4D quantize bin=16) — A_new + diversity 時系列
- (d) **MCC curriculum** — I_value 5→12, target_rate 30→80 Hz の漸増

## 実行結果 (PASS)

実行コマンド: ``py -3.11 D:/projects/llcore/research/other_archs/snn/izhikevich/poc.py``

実行時間: **8.7 秒** (32 個体 × 30 世代)
verifier reject: **0 件** (clip 範囲下 + margin=100 mV では構造的に admit が期待される)

### G1-G8 PASS サマリ

| Gate | 結果 | 数値 |
|---|---|---|
| G1: Z3 v bounded contract | PASS | safe(I_max=5, margin=100) admit; loose(I_max=50, margin=5) reject with CE |
| G2: Z3 per-gene v bounded | PASS | RS canonical margin=100 admit; tight margin=1 reject (overshoot CE) |
| G3: best fitness monotonic | PASS | start=1.0000 → end=1.0000, big_drops(>10%)=0 |
| G4: 4 firing-type lineage survive | PASS | RS=6, IB=24, CH=1, FS=1, missing=∅, reinject events=44 |
| G5: A_new active >= 90% + no collapse | PASS | active frac=1.000, div head=0.545, tail=0.512, collapsed=False |
| G6: rate error improved | PASS | init mean fit=0.668 → final mean=0.860 (best stays at 1.0) |
| G7: verifier latency < 15 ms | PASS | mean=4.00 ms, p95=6.04, p99=6.33 (n=962) |
| G8: firing-type distribution | PASS | lineage types={RS,IB,CH,FS}, gene-guess types={FS,IB,RS} (>= 3) |

### pytest battery

実行コマンド: ``py -3.11 -m pytest research/other_archs/snn/izhikevich/test_izhikevich.py -v``

結果: **22 passed, 1 skipped** (skipped = z3 unavailable fallback path, 環境上 skip 期待)

実行時間: 2.21 秒

## Honest 留保 (主要)

1. **G3 monotonic は trivial PASS**: 初期集団 IB type で偶然 target rate に当てた個体が
   fitness=1.0 を出してしまうため、best curve は start=1.0 → end=1.0 で平坦. ratchet 効果は
   ratchet が必要なケースでは確認できていない (mean fitness 0.668 → 0.860 改善は確認).
   Stage 3+ で target rate をより難しい曲線にする (例: 200 Hz など RS 単独では届かない target)
   検証推奨.

2. **G4 lineage 分布偏り**: 最終世代で RS=6, IB=24, CH=1, FS=1 と IB に集中. Lineage Reservoir
   は **絶滅防止** のみ保証 (全 4 type を均等に保つわけではない). G8 で gene-guess が 3 種類
   観測されている (FS/IB/RS, CH guess は最終世代では消失 — c 値が CH 範囲から離れた) ことから、
   **進化圧で gene が IB 領域に drift** している honest 観察.

3. **G7 latency** は LIF (3.55 ms) と比べて 4.00 ms と若干遅め. v^2 を含む NRA solve が
   LIF の 1 次線形より重いが、15 ms 閾値内で十分実用的.

4. **forward Euler dt=0.25 ms** は Izhikevich 2003 原典の 2 段 half-step trick より粗い.
   v^2 項により Euler 誤差は LIF より大きく、特に spike 直前で overshoot 顕著. Stage 3+ で
   Heun / RK4 比較 + Z3 invariant の精度依存性検証推奨.

5. **refractory なしモデルの firing rate bound** は ``1000 / dt = 4000 Hz`` という自明上界に
   降格. LIF の ``1000 / t_ref`` と比べて生物学的に意味の薄い bound. Stage 3+ で ISI
   (inter-spike interval) lower bound を u dynamics から導出する形に拡張余地.

6. **u 範囲 [-25, 25] は推測**: spike 後 ``u += d (d <= 8)`` の累積で u が上昇しうるが、
   定常状態では ``b*v - u → 0`` で平衡. Stage 3+ で u global bounded invariant を multi-step
   induction で証明する形が必要 (現状は 1-step only).

7. **G2 「tight margin で reject」は構造的に自明**: margin=1 で reject は overshoot CE が
   1-step Euler の worst case (≈90 mV) を margin=1 で吸収できないため. これは「進化対象として
   の制約検出能力」を示すが、「病的 gene 検出」とは言えない (gene 自体は健全, contract が tight すぎる).
   Stage 3+ で 病的 a 値 (a=0 等, clip 前) を gene-level で reject する verifier 拡張余地.

## LIF Stage 2.2a との対応関係 (partial stack reuse pattern)

| 機構 | LIF (Stage 2.2a) | Izhikevich (Stage 2.3) | reuse 方法 |
|---|---|---|---|
| AdaptiveFloorGate | llcore.evolution import 直接 | 同 import 直接 | **完全 reuse** |
| LineageReservoir | LIF 専用 minimal | Izhikevich 専用 minimal | **構造踏襲, 型適応** |
| ModesMeter | 4D quantize bin=16 (tau,Vth,Vrst,tref) | 4D quantize bin=16 (a,b,c,d) | **構造踏襲, 軸入替** |
| MCC curriculum | input freq 5→50 Hz, target 30→80 Hz | I_value 5→12, target 30→80 Hz | **構造踏襲, 入力 type 変更** |
| Z3 verifier API | LIF: SNNInvariantResult | Izh: IzhInvariantResult | **同型, dataclass 再定義** |
| Z3 invariant (1) | firing rate bound (refractory) | firing rate bound (dt-discretization) | **構造類似, 物理意義降格** |
| Z3 invariant (2) | 膜電位 1-step Euler bound (linear) | v 1-step Euler bound (v^2 nonlinear) | **NRA 拡張, 同 pattern** |

= **Codex Q5 で言及された「partial stack reuse」pattern** が SNN/Izhikevich 一般化で成立確認.

## Codex review prompt (gpt-5.4 へ)

```
You are gpt-5.4 reviewing llcore research/other_archs/snn/izhikevich PoC.

# Files to review (Read actual code)
- D:/projects/llcore/research/other_archs/snn/izhikevich/izh_gene.py
- D:/projects/llcore/research/other_archs/snn/izhikevich/izh_verifier.py
- D:/projects/llcore/research/other_archs/snn/izhikevich/poc.py
- D:/projects/llcore/research/other_archs/snn/izhikevich/test_izhikevich.py
- D:/projects/llcore/research/other_archs/snn/izhikevich/verdict.md

# Q1-Q6
Q1: Izhikevich の v^2 非線形を Z3 (quantifier-free real arithmetic) で扱う際の
    sound 性は保証されているか? float64 simulator との乖離は?
Q2: 4 firing-type (RS/IB/CH/FS) の clip 範囲は文献 (Izhikevich 2003) と整合か?
    進化空間として狭すぎ / 広すぎリスクは?
Q3: refractory なしモデルで firing rate bound を主張する根拠は? dt 分割のみ?
    LIF (1000/t_ref) との bound 降格は honest に開示されているか?
Q4: LIF PoC との「同 design pattern + partial stack reuse」claim は本 PoC でも
    成立するか? (AdaptiveFloorGate 直接 reuse, ModesMeter / Reservoir 自前再実装)
Q5: G6 fitness 改善は selection 圧 (target rate fitness) の trivial 結果でないか?
    LIF Stage 2 で議論された Goodhart は本 PoC でも honest に開示されているか?
    特に G3 best curve が start=1.0 から平坦であることは Goodhart リスクか?
Q6: G8 「4 firing-type 分布」は selection 圧 + lineage reservoir 維持の trivial
    結果か、機構として 4 type が安定 attractor になっている証拠か?
    lineage 分布偏り (IB=24, CH=FS=1) は何を意味するか?

Reply in Japanese, technical terms in original.
```

## Codex review record (2026-05-29, gpt-5.4) — **claim 範囲 honest 降格**

Codex pair-review [[feedback_codex_pair_review_for_llcore]] で 5 Findings + Q1-Q6 詳細 verdict。
[[feedback_benchmark_honest_disclosure]] に従い実装維持 + claim 降格 (LIF/Neural ODE/GNN/Marabou と同 pattern)。

### Findings (5 件)

| # | severity | 内容 | 対応 |
|---|---|---|---|
| 1 | **高** | `verify_v_bounded_per_gene` の **「per-gene」表現は overclaim**: `gene.clipped()` するだけで制約式に a,b,c,d が**実質入っていない**。assumed state/input contract 下の **1-step map proof** に過ぎない | **claim 降格**: 「per-gene verifier」→「**assumed state (`u ∈ [-25,25]`) / input (`I ∈ [-I_max, I_max]`) contract 下の 1-step Euler map proof** (gene clipped で box 流用)」。Q1/Q4 の "gene 差まで証明" 読みを撤回 |
| 2 | **高** | `firing rate bound` は neuron dynamics でなく **dt packing bound**: `verify_firing_rate_per_gene` は gene 自体を使わない、`t_{i+1}-t_i >= dt` と `(n-1)*dt > T_window` のみ。`per-gene firing rate invariant` は誤解 | **claim 降格**: 「per-gene firing rate invariant」を撤回、「**dt discretization packing bound (gene-independent)**」に正名化。PoC 本文の命題側も同じく降格 |
| 3 | 中 | G8「4 firing-type distribution」は弱い mechanism 証拠: lineage 4 種維持は `sample_initial_gene` type bias + `LineageReservoir.reinject_extinct()` で **構造的に保証**。`RS=6, IB=24, CH=1, FS=1` は **"4 attractors" でなく "1 dominant basin (IB) + 3 preserved labels via reservoir"** | claim 降格: 「4 firing-type が安定 attractor」を撤回、「**1 dominant basin (IB) + 3 preserved labels (RS/CH/FS) via reservoir**」に honest 正名化 |
| 4 | 中 | G6/G3 Goodhart triviality: G6 は target-rate error inverse fitness + 同じ fitness を gate にする自明性、G3 best monotonic は `elitism=1` + ratchet で構造的に維持 | 既存留保 §honest §G3/G6 と一致するが PoC gate 名は強い → wording 修正 (「fitness 改善」を「fitness shaping の有効性」「fitness の selection-induced monotonicity」に降格) |
| 5 | 中 | Q1 soundness の **限定**: Z3 QF_NRA としては sound、simulator 全体に対して sound でない (`exact rational vs float64` + **reset omitted** + **u dynamics abstracted**) | Q1 claim 降格: 「Z3 QF_NRA sound」→「**sound but for an abstraction (real-valued 1-step Euler map with boxed u/I, reset omitted, u dynamics abstracted)**」、not end-to-end |

### Q1-Q6 要点

- **Q1**: Z3 QF_NRA としては sound、但し証明対象は「real-valued 1-step Euler map with boxed `u,I`」で、`float64 simulator + reset dynamics` 全体ではない (Finding #5)
- **Q2 ✓**: clip range は RS/IB/CH/FS canonical を全部カバー、文献整合。但し `b` を 0.2 近傍以外に広げ、`c,d` を canonical 4 type 寄りに切るので 「Izhikevich family 全般」には狭い (進化空間として PoC 目的内では妥当)
- **Q3**: firing rate bound の根拠は **dt 分割のみ** (Finding #2)、verdict.md と verifier docstring は honest 降格済
- **Q4 ✓**: `same design pattern + partial stack reuse` は成立 (`AdaptiveFloorGate` 直接 reuse、`Reservoir`/`ModesMeter` shape-preserving local 再実装)。但し verifier 側の **"per-gene" reuse story は弱い** (Finding #1)
- **Q5**: G6 fitness 改善は trivial (Finding #4)。`G3 best curve start=1.0` 平坦は **「初期に already solved 個体がいた」可能性**で Goodhart/metric saturation の匂い強い
- **Q6**: G8 は 4 type が stable attractor である **証拠にならない**。`IB=24, CH=FS=1` は「selection pressure 下で IB-like region がこの task で有利、CH/FS は自律 attractor として安定でない」(Finding #3)

**残余リスク** (Codex 指摘): tests はこの framing を補強するもので、**反証的 tests はほぼない**。特に verifier が gene-independent である点、G8 が reservoir-trivial である点を突く test は未整備 (Stage 2.4 候補)。

### 残る正当 claim (post-降格)

- Izhikevich v² 非線形を **Z3 QF_NRA で 1-step Euler map proof** として扱える (latency 4ms = LIF より僅か遅め、threshold 15ms 内)
- **same design pattern + partial stack reuse**: AdaptiveFloorGate 直接 reuse、Reservoir/ModesMeter は shape-preserving local 再実装 (LIF と同 pattern)
- 進化空間 clip 範囲は RS/IB/CH/FS canonical を文献整合でカバー (Q2 confirm)
- firing rate bound は **dt packing bound (gene-independent)** として honest 動作

### 残る "気付き"

- **「per-gene」表現の罠**: `gene.clipped()` で box 制約に流用するだけでは gene a,b,c,d を Z3 制約式に入れたことにならない。**真の per-gene verifier には gene の各 parameter を Z3 symbolic variable として制約に入れる**必要がある (Stage 2.4 候補)
- **lineage reservoir + sample initial type bias の組合せで G8 が trivial 化**: 4 type 維持 claim は構造的保証であり mechanism evidence でない (Codex Finding #3, GNN G6/G8 と同パターン)
- **dt packing bound vs neuron dynamics bound の境界明示**: LIF `1000/t_ref` (refractory-based) → Izhikevich `1000/dt` (dt-based) は **アーキ間で性質が異なる** ことを honest 明示
- **反証的 test の不在**: 4 PoC battery 全てで mechanism claim を補強する test は揃うが、反証的 test (verifier が gene-independent を突く / G8 reservoir-trivial を突く) が未整備 = **Codex pair-review が反証的 test 役を担う構造**になっている。Stage 2.4 で反証 test 内製化

## 次段提案 (Stage 2.4+ / Stage 3 想定)

1. **AdEx (Adaptive Exponential I&F) gene 一般化** (Brette & Gerstner 2005) — 5-param で
   adaptation 含む family。Izhikevich の v^2 を exp(v) に置換した hybrid model.
2. **u global bounded invariant**: 1-step bound を multi-step induction に拡張 (Stage 3 candidate).
3. **多 neuron network** (Izhikevich 2003 reservoir 1000 neuron). Z3 lateral coupling invariant.
4. **fitness target を非自明化**: 初期集団で偶然 1.0 到達しない hard target (例: 200 Hz, ISI CV<0.1).
5. **Heun / RK4 比較**: Euler vs higher-order, Z3 invariant の精度依存性.
6. **Loihi 2 / SpiNNaker 2 deploy bridge**: Izhikevich は SpiNNaker 2 で実装済、進化済 gene を
   neurocore config に export.

## 出力ファイル一覧

- `D:/projects/llcore/research/other_archs/snn/izhikevich/__init__.py` — 公開 API
- `D:/projects/llcore/research/other_archs/snn/izhikevich/izh_gene.py` — IzhikevichGene + simulator + 入力生成
- `D:/projects/llcore/research/other_archs/snn/izhikevich/izh_verifier.py` — Z3 invariant 2 種 + global/per-gene
- `D:/projects/llcore/research/other_archs/snn/izhikevich/poc.py` — main entry (G1-G8 runner)
- `D:/projects/llcore/research/other_archs/snn/izhikevich/test_izhikevich.py` — pytest battery (22 passed, 1 skipped)
- `D:/projects/llcore/research/other_archs/snn/izhikevich/verdict.md` — 本 doc
