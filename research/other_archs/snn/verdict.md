# PoC SNN — LIF への llcore approach 移植 verdict (2026-05-29)

## 命題 (falsifiable)

> LIF (Leaky Integrate-and-Fire) neuron model の係数 (tau_m, V_th, V_reset, t_ref) を
> 低次元 gene 化し、Z3 で **firing rate stability invariant (rate <= 1/t_ref) +
> 膜電位 bounded invariant** を per-gene 検査することで、llcore approach が
> **discrete spike + 時間積分混在** のアーキでも成立する mechanism を実証
> (CPU 完結, 32 neuron × 50 世代).
>
> **Shielded RL hint**: SNN 出力 firing rate を shield 制約 (e.g., max rate <= R_safe)
> として Z3 で検査することで Codex Q5 推奨の **ProSh / Adaptive GR(1) shielding** の
> verifier 統合 sketch を提示.

## 設計概要

### LIF gene (4 パラメータ)

```
tau_m * dV/dt = -(V - V_rest) + R * I(t)
if V(t) >= V_th: spike, V(t+) = V_reset, hold for t_ref
gene = (tau_m, V_th, V_reset, t_ref) ∈ R^4
```

物理範囲 clip:
- `tau_m ∈ [5, 30]` ms
- `V_th ∈ [-55, -40]` mV
- `V_reset ∈ [-80, -65]` mV
- `t_ref ∈ [1, 5]` ms

固定: `V_REST = -65 mV`, `R_MEM = 10` (normalized), `dt = 0.1 ms`, `I_MAX_ABS = 2.5`.
- max 駆動可能 V = -65 + 10 * 2.5 = -40 mV (V_TH_MAX 越え)
- 入力 `I(t) = bias + amplitude * sin(2*pi*f*t/1000) + N(0, noise_std)` (既定: bias=1.5, amp=0.5)

### Z3 invariant (3 種)

#### (1) Firing rate 上界 (refractory bound)

「n_spikes 個の refractory-respecting spike 列が T_window 内で
rate <= 1000/t_ref Hz を満たす」を Z3 で symbolic に証明.

エンコーディング:
- spike 列 `t_1 < t_2 < ... < t_n`
- 制約: `0 <= t_1, t_n <= T_window`, `t_{i+1} - t_i >= t_ref`
- 違反: `n_spikes * t_ref > T_window` (= rate > 1000/t_ref)
- unsat なら invariant 成立

これは構造的に成立する境界 (Codex Q2 議論対象だが、進化対象として
**t_ref を symbolic 制約に組み込み Shield と交差検査できる** ことに verifier 統合の意義がある).

#### (2) 膜電位 bounded

forward Euler 1 step 後 V_next の範囲を symbolic に証明.

エンコーディング:
- `tau_m ∈ [TAU_M_MIN, TAU_M_MAX]`, `V ∈ [V_RESET_MIN, V_TH_MAX]`, `I ∈ [-I_MAX, +I_MAX]`
- `V_next * tau_m = V * tau_m + DT * (V_REST - V + R_MEM * I)`
- 違反: `V_next > V_TH_MAX + margin OR V_next < V_RESET_MIN - margin`
- safety_margin=5 mV で unsat 証明 (Euler overshoot 余裕)
- safety_margin=0 では sat (overshoot CE 検出)

#### (3) Shielded RL hint (sketch のみ)

gene の構造的 rate_max = 1000/t_ref を shield 上界 R_safe と比較.

- t_ref=5 ms → rate_max=200 Hz, R_safe=200 → 境界 unsat (admit)
- t_ref=1 ms → rate_max=1000 Hz, R_safe=200 → sat (reject = shield 違反)
- t_ref=3 ms → rate_max≈333 Hz, R_safe=400 → unsat (admit)

**honest 留保**: 本 PoC は **sketch のみ**. ProSh (Probabilistic Shielding) /
Adaptive GR(1) shielding (LTL spec → reactive controller) との対応は doc 末尾に明記.

### 進化器 (自前 minimal GA)

- 集団 32 = 4 firing-type × 8 個体
- 世代 50
- selection: AdaptiveFloorGate (llcore.evolution.adaptive_floor) で下位 30% 切り
- mutation: gaussian σ = 0.08 × range
- crossover: uniform (各軸独立 50%)
- elitism: 1
- fitness: `1 / (1 + |measured_rate - target_rate| / target_rate)`

### Open-ended 4 機構 (llive 由来 → llcore 再実装 → SNN 移植)

- (a) AdaptiveFloorGate (llcore.evolution import OK) — percentile=30, ratchet=True
- (b) LIFLineageReservoir (LIFGene 専用 minimal, llcore LineageReservoir 構造踏襲) — 4 firing-type の best-ever 保持
- (c) LIFModesMeter (4D quantize bin=16) — A_new + diversity 時系列
- (d) MCC curriculum — 入力 freq 5→50 Hz, target rate 30→80 Hz の漸増

## 実行結果 (PASS)

実行コマンド: `py -3.11 D:/projects/llcore/research/other_archs/snn/poc.py`
実行時間: **12.1 秒** (32 個体 × 50 世代)
verifier reject: **0 件** (clip 範囲下では構造的に admit が期待される)

### G1-G8 PASS サマリ

| Gate | 結果 | 数値 |
|---|---|---|
| G1: Z3 firing rate bound | PASS | global unsat=True, per-gene 3/3 ok, latency mean=3.37 ms |
| G2: Z3 membrane bounded | PASS | safe margin=5 mV unsat; margin=0 で overshoot CE 検出 |
| G3: best fitness monotonic | PASS | start=0.8182 → end=1.0000, big_drops(>10%)=0 |
| G4: 4 firing-type lineage survive | PASS | fast=1, regular=17, burst=1, low_threshold=13, missing=0, reinject=74 |
| G5: A_new active >= 90% + no collapse | PASS | active frac=1.000, div head=0.4982, tail=0.4613 |
| G6: rate error improved | PASS | init best fit=0.8182 → final best fit=1.0000 |
| G7: verifier latency < 10 ms | PASS | mean=3.55 ms, p95=5.86, p99=8.10 (n=1582) |
| G8: Shielded RL hint admit/reject | PASS | 3 ケース全 expected と一致 |

### pytest battery

実行コマンド: `py -3.11 -m pytest research/other_archs/snn/test_snn.py -v`
結果: **17 passed, 1 skipped** (skipped は z3 未インストール時の fallback path テスト, 環境上 skip 期待)
実行時間: 1.18 秒

## ProSh / Adaptive GR(1) shielding 先行 + neuromorphic context 位置づけ

### Shielded RL の先行研究との対応 (honest)

| 先行 | 本 PoC の関係 |
|---|---|
| **ProSh** (Probabilistic Shielding, Alshiekh+ AAAI 2018; Achiam 等 safety layer) | 確率的安全制約 (P[unsafe] <= ε). 本 PoC は決定論的上界 (rate_max <= R_safe). ProSh の **確率分布上界化** sketch は将来 (Stage 5+). |
| **Adaptive GR(1) shielding** (Bloem+ CAV 2015 reactive synthesis, Wongpiromsarn+) | LTL spec → reactive controller. Shield は環境観測ごとに safe action へ補正. 本 PoC は spec 「rate <= R_safe」を Z3 で 1 ショット検査. **GR(1) reactive synthesis ループは含まない**. |
| **Safe RL via shielding** (Alshiekh+ 2018, Carr+ 2023) | RL policy の output を shield filter で post-process. 本 PoC は policy gene を **先に** Z3 で検査して unsafe gene を進化から排除 (preventive shield). post-process shield ではない. |

**主張可能ライン**:
「LIF gene の構造 invariant (t_ref ⇒ rate_max) を Z3 で symbolic に保証することで、
shield 制約 (max rate <= R_safe) を進化過程の admission gate として組み込める
**最小 mechanism sketch** を提示」.

**overclaim 禁止 (Q6 対応)**:
- 「ProSh / Adaptive GR(1) を実装した」 → No, sketch のみ
- 「real RL 統合した」 → No, mock 3 ケース
- 「shield と policy の closed loop」 → No, 1 ショット検査

### Neuromorphic context (Loihi / TrueNorth / SpiNNaker)

LIF は neuromorphic chip の最頻 neuron model:
- **Intel Loihi 2** (2021) — programmable LIF + spike-based learning. tau, V_th 等は per-neuron 設定可.
- **IBM TrueNorth** (2014) — 256 neuron/core, simple LIF (8-bit integer state).
- **SpiNNaker 2** (2020 Manchester) — ARM core + LIF / Izhikevich 混在.

**本 PoC の位置づけ**:
- LIF 4 パラメータ gene 化は **Loihi / SpiNNaker の per-neuron config 空間と整合**.
  進化で見つけた gene を Loihi の neurocore config として deploy する形 (将来研究).
- TrueNorth は integer LIF なので本 PoC float64 + Z3 Real は **直接マップ不可**.
  Q5 で議論される Izhikevich / AdEx 一般化方向に拡張余地.
- 32 neuron は Loihi 1 chip (130K neuron) の 0.024%、SpiNNaker board (約 1M) の 0.003%.
  CPU 完結 PoC として **scaling 検証は本 PoC 範囲外** (Codex Q5 で議論).

## Honest 留保 (主要)

1. **forward Euler は 1 次精度**. tau_m << dt のとき発散リスクだが clip で tau_m >= 5 ms = 50 * dt を保証. それでも Stage 5+ で Heun / RK4 比較推奨.
2. **firing rate bound (G1) は構造的に成立**. Z3 検査の意義は (a) symbolic 表現の sanity (b) shield 制約との交差検査 (c) per-gene latency 計測 — **gene 探索の制約として使う** ことは G8 Shielded RL hint で初めて意味を持つ.
3. **膜電位 bounded (G2) の I_MAX_ABS=2.5 仮定は固定**. 実 RL では action distribution に応じて I の range が変動するため、**spec robustness は将来課題** (Codex Q3 議論対象).
4. **Shielded RL hint (G8) は mock 3 ケースのみ**. ProSh / Adaptive GR(1) との対応関係は本 doc で明示しているが、**実 RL policy との接続は本 PoC スコープ外**.
5. **LIF parameter clip は cortical neuron 想定で狭い**. Izhikevich (4-param 派生で resonator / bursting) や AdEx (adaptation 含む 5-param) への一般化余地あり (Codex Q5).
6. **進化に上限を設けない 4 機構の SNN への直訳** は構造保証 (Adaptive Floor / Lineage / MODES / MCC は アーキ非依存) だが、**4 firing-type の選択** は arbitrary (cortical taxonomy は数十種). Stage 2+ で taxonomy 拡張余地.

## Codex review prompt (gpt-5.4 へ)

```
You are gpt-5.4 reviewing llcore research/other_archs/snn PoC (SNN + Shielded RL hint への llcore approach 移植).

# Files to review (Read actual code)
- D:/projects/llcore/research/other_archs/snn/snn_gene.py
- D:/projects/llcore/research/other_archs/snn/snn_verifier.py
- D:/projects/llcore/research/other_archs/snn/poc.py
- D:/projects/llcore/research/other_archs/snn/test_snn.py
- D:/projects/llcore/research/other_archs/snn/verdict.md

# Q1-Q6
Q1: LIF model の forward Euler discretization は連続時間 SNN dynamics を保存するか? Z3 invariant が discrete artifact になっていないか?
Q2: Firing rate <= 1000/t_ref bound は constructive (構造的) で常に成立するが、Z3 で symbolic 検査する意義は何か? 「進化対象としての制約」の意味は?
Q3: 膜電位 bounded invariant の R*I_max upper bound は I_max=2.5 仮定に sensitive。robust な spec への一般化は?
Q4: Shielded RL hint は本 PoC で sketch のみ。Codex Q5 推奨 ProSh / Adaptive GR(1) との位置づけ doc は accurate か? overclaim はないか?
Q5: LIF parameter clip (tau_m 5-30 ms など) は生物学的に妥当だが、進化空間として狭すぎないか? Izhikevich / AdEx などへの一般化余地は?
Q6: G8 (Shielded RL hint 動作確認) は mock であり、real RL 統合との gap は honest に開示されているか?

Reply in Japanese, technical terms in original.
```

## Codex review record (2026-05-29, gpt-5.4) — **重要 finding: off-by-one bug 検出 + claim 降格**

Codex pair-review [[feedback_codex_pair_review_for_llcore]] で 5 Findings + Q1-Q6 詳細。
**Finding #1 は実装 bug** (PoC 0a v1 zero attractor 級の発見) → 実装修正済。他は claim 降格。

### Findings (5 件)

| # | severity | 内容 | 対応 |
|---|---|---|---|
| 1 | **高 (実装 bug)** | **firing rate bound off-by-one**: 旧実装 `n_spikes * t_ref > T_window` は fence-post error で over-strict (false positive)。正しい finite-window 厳密 violation 条件は `(n-1)*t_ref > T_window` (refractory から `t_n - t_1 >= (n-1)*t_ref` より `n <= 1 + T/t_ref`)。例: t_ref=5ms, T=100ms で 21 spike (`0,5,...,100`) は boundary case で構造的に許容、旧実装は不当 reject していた。「rate ≤ 1000/t_ref を finite horizon の exact safety property として扱うのは不正確」 | **実装修正済** (snn_verifier.py の `n*t_ref > T` を `(n-1)*t_ref > T` に変更, 2 箇所)。17/17 pytest 全 PASS 維持 (旧実装 false positive 改善 = sound 強化)。Codex finding を実コード検証 [[feedback_external_ai_verify]] で受容 |
| 2 | 中 | 膜電位 invariant は continuous-time LIF の invariant でなく **forward Euler 1-step map の invariant**。reset 前の補助量の bound | claim 降格: 「continuous-time dynamics を保存」表現を撤回、「dt=0.1ms / tau >= 5ms 範囲で discrete-time Euler 1-step bound」に明示 |
| 3 | 中 | `verify_membrane_bounded` の `I_MAX_ABS=2.5` 依存強い: scenario-specific assumption、policy/RL drive range 変化で即無効化 | claim 降格: 「robust invariant」→「assumed-input contract (`I ∈ [-I_max, I_max]`) 下の boundedness」。Stage 2+ で `I_max` parameter 化 + input contract (`|I|`, `|ΔI|`) 拡張 |
| 4 | 中 | Shielded RL hint の「ProSh / Adaptive GR(1) verifier 統合 sketch」は前のめり: 実装は `rate_max=1000/t_ref` vs `R_safe` の単純比較のみ、state-dependent shielding / policy shielding / reactive synthesis / probabilistic safety どれも無し | claim 降格: 「verifier 統合 sketch」→「**toy analogue / inspired by** ProSh and GR(1) shielding」。G8 「policy gene の Z3 shield 検査」→「**gene-level rate cap check**」に正名化 |
| 5 | 中 | test battery が off-by-one + discrete/continuous gap を捕まえない: Q1/Q2 核心の regression test なし | Stage 2 候補: finite-window n=1+T/t_ref boundary case の test 追加 + discrete-time artifact 明示 test |

### Q1-Q6 要点

- **Q1 ✗**: forward Euler は continuous-time dynamics を保存しない、Z3 invariant は明確に discrete-time artifact。membrane bound は reset 前 Euler step を見ている、hybrid system 全体の invariant でない
- **Q2**: firing-rate bound Z3 意義 = 「構造制約を verifier API に載せる」「latency 測定」「他 constraint と合成可能」。**但し finite-window 厳密でない bound (Finding #1 修正前) は意味不明確、修正後の `(n-1)*t_ref > T` は finite-window 厳密**。進化制約として明快にするには `t_ref >= 1000/R_safe` の直接 contract が推奨
- **Q3**: robust 化必要、`I_max` を spec parameter 化 + input contract (`|I|`, `|ΔI|`) 拡張 + `V* = V_rest + R*I` equilibrium-relative bound が次段
- **Q4**: 後半 disclosure は honest、前半「ProSh/GR(1) sketch」は **「toy shield predicate」**に降格
- **Q5**: clip は PoC OK だが進化空間狭い (tonic spiking 周辺のみ表現可能)、adaptation/rebound/resonator/class-2 excitability は **Izhikevich / AdEx 一般化**で自然
- **Q6**: G8 mock 明示は十分 honest、「policy gene Z3 shield 検査」→ 「**gene-level rate cap check**」が正確

**総評** (Codex): 「Open questions はありません。主修正点は firing rate の finite-window 定式化と、continuous-time ではなく discrete-time Euler PoC であることの明示」

### 残る正当 claim (post-修正 + post-降格)

- LIF gene の **discrete-time forward Euler 1-step map** (dt=0.1ms, tau >= 5ms) に対し Z3 で firing rate bound (finite-window 厳密、修正後) + membrane bounded (assumed-input contract 下) の検査が動作
- llcore approach の **gene 化 + verifier admit/reject pattern** が SNN/discrete spike 系に **部分移植**できた
- AdaptiveFloorGate は直接 reuse、LineageReservoir/ModesMeter は LIF 専用 minimal 再実装 (Neural ODE と同じ「partial stack reuse」pattern)
- **G8 gene-level rate cap check** は toy analogue として動作 (ProSh / Adaptive GR(1) の **inspired-by** sketch)

### 残る "気付き" (本 PoC で得られた知見)

- **Codex pair-review で off-by-one bug 発見** (Finding #1): Stage 0a v1 zero attractor、Stage 2b Q4 AND gate と並ぶ重要 finding。pair-review 規律 [[feedback_codex_pair_review_for_llcore]] が SNN でも機構として機能 (Claude 単独実装 + 17 tests PASS でも見落としていた fence-post error を Codex が検出)
- **Z3 spike 列の symbolic 構成**: refractory 制約 `t_{i+1} - t_i >= t_ref` で n 個の spike 列を Z3 上に構築 → 「rate ≤ 1000/t_ref」を symbolic 矛盾で証明する pattern が成立 = discrete-time hybrid system の verification pattern として llcore へ流用候補
- **「same verifier stack」claim の限界** (Neural ODE Q5 と同じ): 直接 reuse は AdaptiveFloorGate のみ、LineageReservoir/ModesMeter は型適応 wrapper、SNN-specific Z3 verifier API は別物 → llcore 本流取り込み時は「kernel plugin + verifier backend plugin」pattern が必要

## 次段提案 (Stage 2+ 想定)

1. **Izhikevich / AdEx gene 一般化** (Codex Q5)
2. **多 neuron network** (32 → 256, lateral connection 含む) + Z3 lateral coupling invariant
3. **Shielded RL real-loop**: gym CartPole / safety-gym で SNN policy + Z3 shield filter
4. **Heun / RK4 比較** (Euler vs higher-order, Z3 invariant の精度依存性)
5. **Loihi/SpiNNaker deploy bridge**: 進化済 gene を neurocore config に export

## 出力ファイル一覧

- `D:/projects/llcore/research/other_archs/snn/__init__.py` — 公開 API
- `D:/projects/llcore/research/other_archs/snn/snn_gene.py` — LIFGene + simulator + 入力生成
- `D:/projects/llcore/research/other_archs/snn/snn_verifier.py` — Z3 invariant 3 種
- `D:/projects/llcore/research/other_archs/snn/poc.py` — main entry (G1-G8 runner)
- `D:/projects/llcore/research/other_archs/snn/test_snn.py` — pytest battery (17 passed)
- `D:/projects/llcore/research/other_archs/snn/verdict.md` — 本 doc
