# SCOUT — Kernel Diversification 文献調査 (Stage 3a/3b 着手前)

**作成日**: 2026-06-01
**目的**: rwkv / mamba / hopfield / linear-attn を「**進化可能な state-update gene**」として
扱うために、(1) 各 kernel の state-update 数式形と収縮性/安定性条件、(2)「異 task に異 kernel が
選ばれる specialist 出現」の NAS/進化先行例 を一次寄りに収集する。
**構造規律**: src/ 不変。本 doc は research/kernel_diversification/ 隔離。git は orchestrator 一括。
honest disclosure — 「機構が定式化できる (feasibility)」と「llcore で実証した (実装)」を区別する。

---

## 0. 既存 llcore 資産の再利用設計 (読むだけ・改変なし)

実コードを読んで確認した「破綻ゲートに使える既存 verifier pattern」:

- **`src/llcore/state_update/genes.py`** — RWKV gene は **各座標独立 (対角写像)** の leak integrator:
  `s' = decay·s + (1−decay)·tanh(mix·x + gate_str·s)`。convex combination + |tanh|≤1 で
  |s|≤1 有界が構造的に成立。
- **`src/llcore/verifier/invariants.py`** — 2 段の Z3 gate が**既に**存在し、これがそのまま
  「各 kernel が Stage1 gate pass」の破綻ゲート定義に流用できる:
  1. `verify_gene_safe` — state_norm 有界 (|s_next|≤state_bound)。`|tanh(pre)| ≤ min(|pre|,1)` の
     sound 上界で Z3 反例探索。
  2. `verify_lipschitz_contraction` — **状態方向 Lipschitz L<1** を座標ヤコビ
     `J(t) = decay + (1−decay)·gate_str·t`, `t=sech²(pre)∈(0,1]` を free `t∈[0,1]` に over-approx し、
     端点 `t∈{0,1}` の閉形式上界 `L = max(|decay|, |decay+(1−decay)·gate_str|)` を Z3 で証明。
     L<1 ⟹ Banach 一意固定点 + 有界。`empirical_lipschitz` で中央差分クロスチェック。
- **`src/llcore/kernel/protocol.py`** — Kernel / GeneCodec / VerifierBackend の 3 Protocol が
  既にあり、各 kernel を **additive plugin** として載せる契約が確定済 (semver D)。
  honest 留保が doc 化済: 「same design pattern であって same verifier stack ではない」
  (`research/other_archs/OTHERARCH_VERDICT.md` §5.1 で 3 PoC 全て撤回した overclaim)。

**設計含意**: 4 kernel を gene 化する破綻ゲート (= Stage1 gate pass) は、各 kernel の
**state-direction Jacobian の sup ノルム < 1** を Z3 で証明できるか、に正規化できる。
RWKV は対角ヤコビが閉形式で端点評価 → Z3 線形可行性に縮約 (高速・既存実装)。問題は
mamba/hopfield/linear-attn が同じ「Jacobian sup-norm < 1 の sound over-approx」に乗るか。
以下、各 kernel ごとにこれを評価する。

---

## 1. Per-kernel state-update 数式形と収縮性/安定性条件

### 1.1 RWKV (= 既存 baseline、対角 leak integrator)

- **state update**: `s_{t+1} = decay·s_t + (1−decay)·tanh(mix·x_t + gate_str·s_t)` (各座標独立)。
- **安定性条件 (既証明)**: convex combination + |tanh|≤1 ⟹ |s|≤1 有界。
  状態方向収縮は `L = max(|decay|, |decay+(1−decay)·gate_str|) < 1` (Z3 unsat で certified)。
- **Z3 適合度**: ◎ 既存実装で線形可行性に縮約。3-param gene。

### 1.2 Mamba / 選択的 SSM (selective state space)

- **state update (連続→離散)**: 連続 SSM `h'(t)=A·h(t)+B·x(t)`, `y=C·h` を **ZOH 離散化** で
  `h_t = Ā·h_{t−1} + B̄·x_t`, `Ā = exp(Δ·A)`, `B̄ = (Δ·A)^{-1}(exp(Δ·A)−I)·Δ·B`。
  Mamba の「選択性」は B,C,Δ を **入力依存関数**にする (LTV 化) こと
  ([Gu & Dao 2023, arXiv:2312.00752](https://arxiv.org/pdf/2312.00752))。
- **安定性条件**: 対角 A の **実部が負** (Re(λ_i)<0) なら `|exp(Δ·λ_i)| = exp(Δ·Re(λ_i)) < 1`
  (Δ>0)。S4/S4D は HiPPO 由来で A を負実部に構造化し、数値安定のため **log(A) を保持**
  ([Towards Data Science S4 解説](https://towardsdatascience.com/structured-state-space-models-visually-explained-86cfe2757386/))。
  ただし **Mamba2 の A は常に安定とは限らず**、A に明示的安定性制約を課すと loss/perplexity が
  改善するとの報告 ([Sparse Mamba, arXiv:2409.00563](https://arxiv.org/pdf/2409.00563) /
  [Mamba-CL, arXiv:2411.15469](https://arxiv.org/html/2411.15469))。
- **Z3 適合度**: ○〜△。**対角 (diagonal) SSM に限定**すれば各座標 `|Ā_i| = exp(Δ·a_i)<1` は
  `Δ·a_i < 0` の **線形制約** に落ち、RWKV と同じ「対角ヤコビ sup<1」枠に乗る (gene = 各座標の
  `a_i<0` と `Δ>0`、または対数空間 param)。`exp` は単調なので符号制約で sound に扱える
  (`exp(z)<1 ⟺ z<0`)。選択性 (入力依存 Δ,B,C) を入れると LTV で Jacobian が入力依存になり、
  RWKV の free-`t` over-approx と同型の「入力依存ゲインを worst-case 区間で over-approx」が必要 (要 PoC)。
- **gene 案**: `(a_diag<0, Δ_base>0, input_gain)` の対角 SSM が最小 falsifiable 単位。

### 1.3 Modern Hopfield (連続状態, attention 等価)

- **state update**: `ξ_{new} = X·softmax(β·X^T·ξ)` (X=格納パターン行列)。これは **エネルギー**
  `E(ξ) = −lse(β, X^Tξ) + ½ξ^Tξ + const` の **CCCP 更新**で、Ramsauer 2020 が
  「**1 step で停留点へ大域収束**」を証明 ([Hopfield Networks is All You Need, arXiv:2008.02217](https://arxiv.org/abs/2008.02217) /
  [NeurIPS 2020 PDF](https://proceedings.neurips.cc/paper/2020/file/da4902cb0bc38210839714ebdcf0efc3-Paper.pdf))。
  log-sum-exp Lagrangian のとき更新則は **transformer の attention に一致**
  ([Linear/Modern Hopfield 解説](https://www.emergentmind.com/topics/modern-hopfield-networks-mhns))。
- **安定性条件**: エネルギーが各 step **単調非増加** (収束は energy descent で保証、Lipschitz とは別系統)。
  patterns が well-separated なら **1 step で contractive** に retrieval、誤差は指数的に抑制。
  **逆温度 β** が支配: 大 β → 単一パターン近傍の鋭い極小 (specialist 的)、小 β → 複数パターン平均の
  大域 fixed point。**臨界 β_c で相転移** (大域 attractor → pattern-specific minima)
  ([Temperature Phase Transition, arXiv:2311.18434](https://arxiv.org/abs/2311.18434))。
- **Z3 適合度**: △。収束証明は **Lyapunov/energy descent** であって state-norm Lipschitz とは
  **質的に別**。llcore の既存 gate (Jacobian sup-norm) は直接流用不可。代替破綻ゲートとして
  「**ΔE ≤ 0 (energy 単調非増加) を Z3 で証明**」か「softmax Jacobian の作用素ノルム上界
  `≤ β·λ_max(cov)` を Z3 制約」のいずれか (要 PoC、softmax は Z3 非線形で over-approx 設計が要)。
  これは `OTHERARCH_VERDICT.md` の Neural ODE「Lyapunov/Hurwitz 別系統」教訓と同型。
- **gene 案**: `(β>0, n_patterns, pattern_scale)`。破綻ゲート = energy 単調性 or 縮約 β 領域。

### 1.4 Linear Attention (累積状態 RNN 形, Katharopoulos 2020)

- **state update**: kernel 特徴写像 φ で attention を線形化し、**行列値状態** `S_t ∈ R^{d×d}` の
  累積で RNN 化: `S_t = S_{t−1} + φ(k_t)·v_t^T`, `z_t = z_{t−1} + φ(k_t)`,
  出力 `o_t = (φ(q_t)^T·S_t) / (φ(q_t)^T·z_t)`
  ([Transformers are RNNs, arXiv:2006.16236](https://arxiv.org/abs/2006.16236) /
  [Schoelkopf blog](https://haileyschoelkopf.github.io/blog/2024/linear-attn/))。
- **安定性条件**: 純累積 `S_t=S_{t−1}+kv^T` は **decay を持たない積分器**なので状態が**単調増大**し得る
  (有界でない)。正規化項 `z` は **数値不安定の原因**で実装ではしばしば省略され、省略すると
  ノルムが発散しうる。**Lipschitz 安定化には decay/gating が必須**: gated linear attention
  (GLA) / RetNet の `S_t = γ·S_{t−1} + k_t v_t^T` (γ<1) なら RWKV と同型の指数減衰で有界化。
- **Z3 適合度**: ○ (gated 版に限る)。`S_t = γ·S_{t−1} + outer(k,v)`, γ<1 は **RWKV の対角 decay と
  同じ Banach 縮約**。破綻ゲート = `γ<1` + 入力ノルム有界 ⟹ 幾何級数で `‖S‖ ≤ ‖kv‖/(1−γ)`。
  **無 decay の素の linear attention は破綻ゲートを通らない (= honest negative の候補)**。
- **gene 案**: `(decay γ∈[0,1), feature_scale, normalizer_on/off)`。最小単位は対角 γ の gated 版。

---

## 2. 「異 task に異 kernel が選ばれる specialist 出現」の NAS/進化先行例

### 2.1 直接先行 (multi-task NAS / 進化)

- **Multi-Task NAS + Transfer Rank (KTNAS)** — 進化的 cross-task NAS。task 間で構造知識を転移し、
  source↔target の **ranking disorder** (順位逆転) が downstream 性能を劣化させる問題を扱う
  ([arXiv:2504.00772](https://arxiv.org/html/2504.00772))。
  → llcore 含意: 「異 task に異 kernel」は **task 間の fitness ランキング非一貫性**が前提条件。
  ranking disorder が無い (= どの task でも同 kernel が最強) なら specialist は出現しない =
  Stage 3b の **honest negative ゲート**として使える指標。
- **ENCAS (Evolutionary Neural Cascade Search)** — 複数 supernetwork を跨いで **異種アーキの
  cascade** を進化し精度/FLOPs の Pareto 前線を構成 ([arXiv:2203.04011](https://arxiv.org/pdf/2203.04011))。
  → 「集団が単一アーキに固定せず異種を保持する」既存実証。
- **Lamarckian Evolution NAS** ([arXiv:1804.09081](https://arxiv.org/pdf/1804.09081)) /
  **多目的進化 CNN 設計** ([arXiv:1912.01369](https://arxiv.org/pdf/1912.01369)) — Pareto 集団で
  多様アーキを維持する一般枠。

### 2.2 specialist 出現の機構 (QD / niching / island)

- **Quality Diversity (MAP-Elites / Novelty Search)** — behavior 空間を bin に分け各 bin の最良個体を
  保持。**「各 achievable behavior の最良例で空間を埋める」= specialist 集合を陽に作る**
  ([Pugh+ 2016 Frontiers QD](https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full) /
  [Monte Carlo Elites, arXiv:2104.08781](https://arxiv.org/pdf/2104.08781))。
  → Stage 3b で **behavior descriptor を kernel_id or task-id にした MAP-Elites** にすれば
  「異 task に異 kernel」を **陽に強制せず emergent に観測**できる設計に直結。
- **Island Model** — 準独立な subpopulation が各々進化し elite を migration。**niche を空間的に隔離**して
  早期収束を回避、局所適応を促す ([Population/Island model 解説](https://www.emergentmind.com/topics/population-and-island-model))。
  → kernel ごとに island を分けると「単一 kernel 固定」を構造的に防げるが、それは**強制であって
   emergent ではない** (honest 留保: 強制 island は Stage 3b の「固定しない」を自明に満たすので
   破綻ゲートとして弱い)。**混合集団で emergent に分離するか**が本質的問い。
- **Adaptive Species Discovery (speciation in EA)** — 集団から種を適応的に発見し多様性を維持
  ([Speciation in EA](https://www.researchgate.net/publication/220741366))。NEAT 系の speciation も同系統。

### 2.3 llcore 内の直接前提研究 (再利用可)

- **`research/ea_multitask/`** (task_mixture.py / ea_lab.py / exp_ea1..3) — **既に**
  TaskMixture (regime 構造タスク族) + MAP-Elites + honest train/test 分離が実装済。
  ③(選択圧/分離)を **hold-out 汎化**で測る土俵。`map_elites_full` / `honest_reevaluate` /
  `FlipFlopTask` が揃う。
  → **Stage 3b はこの土俵に kernel_id 軸を足すだけで成立**。ゼロから書く必要なし。
- **`research/step_c_deceptiveness_measure/`** — deceptiveness (FDC / elite-dip) を実測し
  「3-param 空間は単峰 broad-basin で③不要」に到達した一連の検証 (circular reasoning /
  descriptor dependence / sampling threshold を個別 VERIFY 済)。
  → **kernel 多様化 = 探索空間拡張で richer 地形が③ load-bearing になるか**を問い直す道。
  Stage 3b の前に「kernel 混合空間が deceptive か」を同じ deceptiveness 計測で測れる。

---

## 3. 統合所見 (戦略含意)

1. **破綻ゲート (Stage 3a) は kernel ごとに verifier 系統が分かれる**:
   - RWKV / gated-linear-attn / 対角 Mamba → **state-direction Jacobian sup-norm < 1** (既存 Z3 gate
     を sound に一般化、線形/単調制約に縮約可能)。
   - Modern Hopfield → **energy descent (ΔE≤0)** = 別系統。既存 gate は流用不可、新 Lyapunov gate が要。
   - 素の (無 decay) linear attention → **破綻ゲートを通らない honest negative 候補** (有界化に
     decay/gating が必要)。
   → `OTHERARCH_VERDICT.md` の「same design pattern, NOT same verifier stack」教訓が**そのまま再適用**。
   各 kernel の `Trajectory.kind` と VerifierBackend を分けて意味論差を型に出す既存設計が効く。

2. **specialist 出現 (Stage 3b) はゼロから作らない**: `research/ea_multitask/` の TaskMixture +
   MAP-Elites + honest 再評価に **kernel_id を behavior descriptor or gene 軸**として足すのが最短。
   「集団が単一 kernel に固定しない」の破綻ゲートは **強制 island では自明に満たす (弱い)**ので、
   **混合集団で emergent に分離すること**を要件にすべき。判定指標は §2.1 の **ranking disorder /
   task 間 fitness 順位逆転**: 逆転が無ければ specialist は出ない (honest negative)。

3. **③研究との接続**: Step4 結論「真の unlock は探索空間を 3-param から拡張」を、kernel 多様化が
   **richer architecture 空間が deceptive (③ load-bearing) 地形を持つか**として検定可能にする。
   既存 deceptiveness 計測 (FDC/elite-dip, `step_c_deceptiveness_measure/`) を kernel 混合空間に
   適用するのが論理的次段。**「拡張すれば③が効く」は仮説であり未実証** (honest)。

---

## 4. Sources (一次/二次)

- Gu & Dao, *Mamba: Linear-Time Sequence Modeling with Selective State Spaces*, arXiv:2312.00752 — https://arxiv.org/pdf/2312.00752
- *Sparse Mamba: Controllability, Observability, Stability*, arXiv:2409.00563 — https://arxiv.org/pdf/2409.00563
- *Mamba-CL: Optimizing Selective SSM in Null Space*, arXiv:2411.15469 — https://arxiv.org/html/2411.15469
- S4 / structured SSM 解説 (ZOH, HiPPO, 負実部安定化) — https://towardsdatascience.com/structured-state-space-models-visually-explained-86cfe2757386/
- Ramsauer et al., *Hopfield Networks is All You Need*, arXiv:2008.02217 — https://arxiv.org/abs/2008.02217 ; NeurIPS 2020 PDF — https://proceedings.neurips.cc/paper/2020/file/da4902cb0bc38210839714ebdcf0efc3-Paper.pdf
- *Temperature-Dependent Phase Transition in Modern Hopfield Networks*, arXiv:2311.18434 — https://arxiv.org/abs/2311.18434
- Modern Hopfield Networks 解説 (β, energy minima 3 種) — https://www.emergentmind.com/topics/modern-hopfield-networks-mhns
- Katharopoulos et al., *Transformers are RNNs: Linear Attention*, arXiv:2006.16236 — https://arxiv.org/abs/2006.16236 ; 累積状態解説 — https://haileyschoelkopf.github.io/blog/2024/linear-attn/
- *Multi-Task NAS using Architecture Embedding and Transfer Rank (KTNAS)*, arXiv:2504.00772 — https://arxiv.org/html/2504.00772
- *Evolutionary Neural Cascade Search (ENCAS)*, arXiv:2203.04011 — https://arxiv.org/pdf/2203.04011
- *Efficient Multi-objective NAS via Lamarckian Evolution*, arXiv:1804.09081 — https://arxiv.org/pdf/1804.09081
- Pugh et al., *Quality Diversity: A New Frontier*, Frontiers 2016 — https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2016.00040/full
- *Monte Carlo Elites (QD as multi-armed bandit)*, arXiv:2104.08781 — https://arxiv.org/pdf/2104.08781
- Population & Island Model 解説 — https://www.emergentmind.com/topics/population-and-island-model
- *Speciation in Evolutionary Algorithms: Adaptive Species Discovery* — https://www.researchgate.net/publication/220741366

### llcore 内部参照 (一次・実コード)
- `src/llcore/state_update/genes.py` — RWKV 対角 gene
- `src/llcore/verifier/invariants.py` — `verify_gene_safe` / `verify_lipschitz_contraction` / `empirical_lipschitz`
- `src/llcore/kernel/protocol.py` — Kernel / GeneCodec / VerifierBackend Protocol (additive plugin 契約)
- `research/other_archs/OTHERARCH_VERDICT.md` — 「same design pattern, NOT same verifier stack」教訓
- `research/ea_multitask/` — TaskMixture + MAP-Elites + honest train/test (Stage 3b 土俵の既存実装)
- `research/step_c_deceptiveness_measure/` — deceptiveness 計測 (3-param=単峰 broad-basin, ③不要 到達)
