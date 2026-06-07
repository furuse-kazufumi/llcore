# 理想目標軌道への収束 (tracking) を llcore contraction ゲートの検査指標に取り入れる設計

**日付**: 2026-06-06
**対象**: llcore 検証器 (Z3 / vertex-LMI contraction ゲート, `ρ(J)<1`, fail-closed)
**種別**: 設計 + フィジビリティ (additive 提案。`src/` 改変なし、`research/` 追加のみ)
**PoC**: `research/target_trajectory_poc/poc_target_trajectory.py` (n=2,3 実行済み)

**実装着地 (2026-06-06, additive)**: 案 B を `src/llcore/verifier/tracking_tube.py` に純 ADDITIVE 移植 (read-only レポータ `tracking_tube()` + `state_lipschitz_inf` / `input_gain_inf`)。PoC 結果 JSON の case A/B/C/D を golden 値としてテスト一致確認 (`tests/unit/test_tracking_tube.py`)。`certifies()` 等の既存 API は 1 文字も不変。あわせて T1 Phase 1 (a) として証明ゲートを出荷側 `src/llcore/evolution/minimal_ga.py::evolve()` に本配線 (`gate_mode` / `resample_cap` を additive 追加、既定 `"none"` は旧挙動 byte-identical、`research/verified_evolution/gated_evolve.py` と全モード挙動一致をテストで実証)。

---

## 0. 要旨 (3 行)

- **推奨案 = 案 B「外乱 tube ゲート」**: 既存 contraction ゲートの出力 `L = sup‖J_s‖_∞ < 1` を *そのまま再利用* し、入力ゲイン `G = sup‖J_x‖_∞` と外乱上界 `w̄` から閉形式の tube 半径 `r = G·w̄/(1−L)` を追加レポートする。**新規 Z3 証明は不要** (contraction が証明済みなら tracking tube は系として導出される)。
- **健全に証明できるのは**:「contraction (ρ<1, 既存ゲート) **+** 参照軌道が系の解 (feasibility) **+** 外乱有界 (`‖d‖≤w̄`)」⇒「追従誤差 `‖s_act−s_ref‖_∞ ≤ r` (tube に閉じ込め)」。**証明できないのは**: 参照軌道 *自体の妥当性* (それが「良い」軌道か) と、feasibility (参照軌道が系の解であること) — これらは数値同定でありタスク側の責任。
- **PoC 数値裏付け**: 契約ゲート PASS の 3 gene (A/B/C) は追従誤差/外乱比 0.50/0.78/1.04 で理論 tube `G/(1−L)`=1.76/2.29/2.00 の内側、非契約 gene D は 9.3 倍に増幅 (tube=∞=保証なし)。ゲートが tracking 保証に load-bearing であることを示す。

---

## 1. 背景と動機

### 1.1 現状の llcore ゲートが検査しているもの

`src/llcore` の検証器と `research/verified_evolution_sdp_gate/coupled_nd.py` の n-dim 契約器が証明するのは、進化する記憶コアの状態力学

```
s_{t+1} = decay ⊙ s_t + (1 − decay) ⊙ tanh(W s_t + V x_t),   V = I        … (1)
```

(gene = `(decay ∈ [0,1]^n, W ∈ [−2,2]^{n×n})`) について、**ヤコビ `J(t) = diag(decay) + diag((1−decay)⊙t) W` (t_i = sech²(pre_i) ∈ (0,1]) が achievable-t box `[t_min, 1]^n` 上で `ρ(J) < 1` (contraction)** であること、ただ 1 点。証明手段は 3 つの sound 契約器:

| 契約器 | 証明する十分条件 | 手段 |
|---|---|---|
| `cert_inf` | `sup‖J‖_∞ < 1` (各行 abs-sum の端点 sup) | 閉形式 (Z3 と等価, decorative 注記あり) |
| `cert_two` | `2^n` box 頂点で `σ_max(J) < 1` | 頂点 SVD 列挙 |
| `cert_sdp` | 共通二次 Lyapunov `P≻0`, `P − J_v^T P J_v ≻ 0` を全頂点で | cvxpy / CLARABEL (fail-closed) |

3 つとも box 上 `ρ(J)<1` を含意し、**Banach 不動点定理**で「一意固定点 + `|s|<1` 有界」を保証する。これは「**系が contracting なら任意の 2 軌道が指数収束する** (incremental / δ-stability)」という contraction theory の標準帰結 (Lohmiller & Slotine 1998) の離散版である。

### 1.2 ユーザー要望と gap

> 「理想の目標軌道があれば、それを検査器の検査指標に取り入れたい」

現状ゲートは「**どこかに** 収束する」(ρ<1) は言うが、「**望ましい軌道 `s_ref` に** 収束する」は言わない。後者は control 文脈での **trajectory tracking** であり、検証派の先行研究 (Sun, Jha, Fan; arXiv 2011.12569, CoRL 2020 "Learning Certified Control using Contraction Metric"; Manchester & Slotine 2017 "Control Contraction Metrics") の中核テーマである。

**本設計の鍵となる観察**: contraction theory では tracking は contraction から *ほぼ無料で導出* される。「2 軌道が指数収束する」(=ρ<1) が証明済みなら、片方を参照軌道 `s_ref`、もう片方を外乱付き実軌道 `s_act` と置くだけで、追従誤差は外乱の大きさに比例した **tube** に閉じ込められる。つまり **新しい重い証明を足すのではなく、既存 contraction 証明の出力を tracking 指標に *翻訳* する** のが正しい設計方針となる。

---

## 2. 理論: contraction ⇒ tracking tube の標準導出

### 2.1 連続時間 CCM の形 (参考)

Control Contraction Metric (CCM) では、リーマン計量 `M(x) ≻ 0` と微分動力学 `δ̇x = A(x) δx` (`A = ∂f/∂x`) に対し、契約条件

```
Ṁ + A^T M + M A + 2λ M ⪯ 0          … (2)
```

が成り立てば、増分 Lyapunov 関数 `V = δx^T M δx` が `V̇ ≤ −2λ V` で減衰し、任意の 2 軌道が rate `λ` で指数収束する。有界外乱 `w` (`‖w‖≤w̄`) 下では steady-state tube

```
√(V) ≤ (w̄ / λ) · sup‖M^{1/2} B‖,   ‖x − x_ref‖ ≤ √(cond(M)) · √(V)/√(λ_min(M))   … (3)
```

が立ち、追従誤差は計量の条件数 `cond(M)` でスケールされた tube に閉じ込められる (Singh+ 2023, Zhao+ 2021 "Tube-Certified Trajectory Tracking", arXiv 2109.04453)。

### 2.2 llcore への離散・sup-norm 翻訳 (本設計が実際に使う形)

llcore の写像 (1) は **離散時間 + 対角支配** なので、上記をリーマン計量 `M=I`・sup-norm に落とした **より単純で sound な形** で扱える。

写像を `F(s, x)` と書き、2 軌道を:
- 参照: `s_ref[t+1] = F(s_ref[t], x_ref[t])` (外乱なし)
- 実: `s_act[t+1] = F(s_act[t], x_ref[t] + d[t])` (入力外乱 `‖d[t]‖_∞ ≤ w̄`)

`F` は box 上で **状態方向 L-Lipschitz** (`‖∂F/∂s‖_∞ ≤ L`)、**入力方向 G-Lipschitz** (`‖∂F/∂x‖_∞ ≤ G`)。誤差 `e[t] = s_act[t] − s_ref[t]` について三角不等式 + 平均値定理:

```
‖e[t+1]‖_∞ = ‖F(s_act[t], x_ref[t]+d[t]) − F(s_ref[t], x_ref[t])‖_∞
           ≤ L ‖e[t]‖_∞ + G ‖d[t]‖_∞
           ≤ L ‖e[t]‖_∞ + G w̄                                            … (4)
```

`L < 1` (= contraction ゲート出力) なら幾何級数が収束し、初期一致 `e[0]=0` から:

```
‖e[t]‖_∞ ≤ G w̄ · (1 − L^t)/(1 − L)   ↗   limsup_t ‖e[t]‖_∞ ≤ r := G w̄/(1 − L)   … (5)
```

**これが本設計の tube 半径。** `L` は既存ゲートが contraction の証拠に使う `sup‖J_s‖_∞` そのもの、`G` は入力ヤコビ `diag((1−decay)⊙t)V` の sup (各 `t_i≤1` で sup → `G = max_i (1−decay_i)Σ_j|V_ij|`)、`w̄` は設計者が与える外乱上界。**3 つすべてが既存の box 列挙 (Z3/vertex と同じ achievable-t box) から閉形式で出る。**

### 2.3 L (状態 Lipschitz) と ρ (spectral radius) の差に関する honest 注記

ゲートは厳密には `ρ(J)<1` を証明するが、tube 不等式 (4) は **induced norm** `L=‖J_s‖_∞<1` を要求する。一般に `ρ ≤ ‖J‖` なので **`‖J‖_∞<1 ⇒ ρ<1` だが逆は不成立**。すなわち:

- `cert_inf` (sup-norm 契約) を通った gene は `L=sup‖J_s‖_∞<1` が直接成立 → **tube (5) がそのまま sound**。
- `cert_two` / `cert_sdp` のみ通った gene (非正規/回転的 contraction) は `ρ<1` だが `‖J‖_∞≥1` のことがあり、(5) の `L` を `‖J‖_∞` で取ると tube が発散しうる。この場合は **適切な計量 `P` (cert_sdp が返す Lyapunov 行列) の下での `P`-重み付きノルム `‖e‖_P` で同じ導出を行う** (§3 案 B-2)。`σ_max` や Lyapunov 計量を使えば `cert_two/sdp` でも sound な tube が立つが、`cond(P)` 倍のルーズさが入る (式 (3) の `cond(M)` に対応)。**初版は `cert_inf` PASS gene に tube を限定する**のが最も単純で sound (PoC もこれ)。

---

## 3. 設計: 「理想目標軌道」の複数定義案と Z3 検査指標化

「理想目標軌道」を llcore 文脈で何にするかで 3 案。各案について (a) 何を sound に証明できるか、(b) 検査コスト、(c) 健全性 (証明 vs 数値)、(d) 既存ゲートへの増設、を述べる。

### 案 A — 記憶保持/忘却カーブ目標軌道 (memory decay reference)

**定義**: 「理想の記憶保持曲線」を参照軌道とする。例: 入力 impulse 後に状態が目標時定数 `τ*` で減衰する `s_ref[t] = s_0 · exp(−t/τ*)` (Ebbinghaus 忘却曲線 / 望ましい consolidation カーブ)。

- **(a) sound に証明できること**: contraction 済み gene について、impulse 応答が `s_ref` の tube `r=G w̄/(1−L)` 内にあること (外乱 = 入力ノイズ or gene 摂動)。**ただし `s_ref` が exp 減衰そのものは式 (1) の解ではない** (tanh 非線形のため) → feasibility 残差 `ρ_feas = max_t ‖s_ref[t+1]−F(s_ref[t],x_ref[t])‖` が 0 でない。よって「目標カーブへの追従」は **`ρ_feas` 込みの tube** `‖e‖ ≤ (G w̄ + ρ_feas)/(1−L)` (modeling error を外乱に吸収) でしか言えない。
- **(b) コスト**: 追加 Z3 不要 (`L,G` は既存閉形式)。`ρ_feas` は `T` step の forward 評価 = O(T·n²)。
- **(c) 健全性**: tube 不等式は **証明**。`ρ_feas` は **数値**。「`τ*` が良い時定数か」は **検査不能** (設計判断)。
- **(d) 増設**: `decay` から実効時定数 `τ_eff ≈ −1/ln(decay)` を解析計算し、`|τ_eff − τ*|` をペナルティ/レポートに足すだけ。ゲートとしては「contraction PASS かつ `τ_eff ∈ [τ*−Δ, τ*+Δ]`」の AND。

### 案 B — 参照 consolidation 軌道 / 教師軌道 (feasible reference, **推奨**)

**定義**: 参照入力列 `x_ref` に対する **系自身の解** を参照軌道とする (`s_ref[t+1] = F(s_ref[t], x_ref[t])`)。実運用では「望ましい状態列 `s_ref`」が先に与えられ、それを近似実現する `x_ref` または gene を同定する逆問題になるが、検査指標としては「**与えられた参照軌道が (i) 系の解で (ii) その近傍の実軌道が外乱下でも tube に留まる**」を見る。

- **(a) sound に証明できること**: 「contraction (既存ゲート, `cert_inf` ⇒ `L<1`) + feasibility (`ρ_feas≈0`) + 外乱有界 (`w̄`)」⇒ 「追従誤差 `≤ G w̄/(1−L)`」。**式 (5) が完全に sound に立つ唯一の案** (`s_ref` が真に系の解なので modeling error 項なし)。
- **(b) コスト**: **追加 Z3 ゼロ**。`L,G` = 既存 box 端点列挙の閉形式 (`cert_inf` と同じ box, O(n²))。`ρ_feas`/数値 tube 確認は forward rollout のみ。`cert_two/sdp` 由来 gene を含めるなら `P`-norm 版で頂点 LMI を 1 回 (既存 `cert_sdp` が返す `P` を流用)。
- **(c) 健全性**: tube 不等式 **証明** (Banach + Lipschitz 合成)。feasibility は **数値同定** (`ρ_feas` 測定; 厳密 0 は構成的に作る場合のみ)。「`s_ref` が *良い* 記憶/タスク軌道か」は **検査不能** (タスク fitness 側の責任)。
- **(d) 増設**: 既存 `VerifierBackend.certifies(gene)→bool` を一切変えず、新規 **read-only レポータ** `tracking_tube(gene, x_ref, w_bar) → {L, G, tube_radius, feasibility_residual}` を追加。進化ゲートに組み込むなら「contraction PASS かつ `tube_radius ≤ r_max`」の AND ゲート (fail-closed: tube=∞ or `ρ_feas>ε` なら reject)。

### 案 C — タスク由来の教師軌道 (task-derived target, R-LLM 文脈)

**定義**: R-LLM (byte-LM, `research/verified_lm_evolution`) で、参照軌道を「held-out コーパスを読んだときの理想 reservoir 状態列」(例: 低 CE を出す既知良 gene の状態軌道、または教師 reservoir) とする。

- **(a) sound に証明できること**: 「2 つの gene/入力が近い ⇒ reservoir 状態軌道が近い」(incremental stability) を tube で。だが **教師軌道は別 gene の解** なので、同一 gene 内の feasibility が成立せず、案 A 同様 modeling error 項が入る。さらに LM では入力 `x_t = tanh(E[byte])` が離散・データ依存で「外乱 `w̄`」の解釈が弱い。
- **(b) コスト**: `n=8`, `2^8=256` 頂点まで現行契約器で可。`L,G` 閉形式。
- **(c) 健全性**: tube は sound だが「教師軌道が理想か」が **二重に検査不能** (教師選択 + feasibility)。R-LLM VERDICT が既に示すように、ここでの payoff は「学習」でなく「evolvability」寄りなので、**tracking 指標も `過剰主張に注意`**。
- **(d) 増設**: 案 B のレポータを `lm_substrate.reservoir_states` の出力に適用。

### 3.x 案比較表

| 軸 | 案 A 忘却カーブ | **案 B 参照解 (推奨)** | 案 C タスク教師 |
|---|---|---|---|
| `s_ref` が系の解か (feasibility) | × (modeling error) | **○ (構成的に解)** | × (別 gene の解) |
| 式 (5) tube が完全 sound か | △ (誤差吸収版) | **○ 完全** | △ |
| 追加 Z3 コスト | ゼロ | **ゼロ** | ゼロ (n≤8) |
| 「軌道の妥当性」検査 | 不能 | 不能 | 不能 (二重) |
| 既存ゲート増設の容易さ | 易 (τ ペナルティ) | **易 (read-only レポータ)** | 中 (LM 配線) |
| 過剰主張リスク | 中 | **低** | 高 (evolvability 混同) |

**推奨 = 案 B**。理由 3 行:
1. **式 (5) の tube が *完全に* sound に立つ唯一の案** (参照軌道が真に系の解なので modeling error 項がゼロ; 他案は誤差を外乱に吸収する近似が要る)。
2. **追加証明コストがゼロ** — `L`(=既存 `sup‖J_s‖_∞`) と `G` は contraction ゲートが既に計算している box 端点量の再利用で、新規 Z3 クエリを一切増やさない (fail-closed 規律もそのまま継承)。
3. **既存ゲートを 1 文字も変えない additive レポータ**として増設でき、`src/` 不可侵・semver 互換の llcore 開発規律 (PoC 1b の純 ADDITIVE 方針) に最も整合する。

---

## 4. CPU PoC とフィジビリティ評価

### 4.1 PoC 構成

`research/target_trajectory_poc/poc_target_trajectory.py` (約 230 行, numpy + 既存 `coupled_nd`)。`py -3.11` で実行確認済 (z3 4.16.0 / numpy 2.4.4 環境)。3 命題を小次元で検証:

- **P1 feasibility**: `synthesize_reference` で参照軌道を系の解として構成 → 残差 `ρ_feas`。
- **P2 contraction**: 既存 `cert_inf` / `cert_two` でゲート、`L=sup‖J_s‖_∞`, `G=sup‖J_x‖_∞` を閉形式算出。
- **P3 tracking tube**: 入力に一様外乱 `d∈[−w̄,w̄]^n` を 64 シード注入し、定常区間 (後半 T/2) の追従誤差 sup を理論 tube `r=G w̄/(1−L)` と比較。

### 4.2 実行結果 (実測値)

```
case                         cert_inf    L       G      tube    emp_err  err/inp  holds  feas_res
A_strong_contraction_n2      True     0.9150  0.1500  0.0882   0.02505   0.501   True   0.00e+00
B_coupled_n3                 True     0.8250  0.4000  0.1143   0.03893   0.779   True   0.00e+00
C_near_boundary_n2           True     0.7000  0.6000  0.1000   0.05225   1.045   True   0.00e+00
D_noncontract_control_n2     False    1.6800  0.8500     inf   1.86695   9.335   True*  0.00e+00
```

- **A/B/C (contraction PASS)**: 追従誤差はすべて理論 tube `r` の内側 (`holds=True`)。誤差/外乱比 0.50/0.78/1.04 は理論ゲイン `G/(1−L)`=1.76/2.29/2.00 の下に収まり、**式 (5) の tube が実測を sound に上から押さえる**ことを確認。feasibility 残差は構成的に厳密 0。
- **D (non-contraction control, ゲート REJECT)**: `L=1.68≥1` で理論 tube は `∞` (=保証なし)。実測でも追従誤差が外乱の **9.3 倍**に増幅 (A/B/C の 0.5〜1.0 倍と桁違い)。状態自体は tanh で有界に留まるが **tracking は保証されない** — ゲートが tracking 保証に load-bearing であることの negative control。(`holds=True*` は「tube=∞ なので形式上 `err≤∞`」の自明値であり、保証の不在を意味する。)

### 4.3 フィジビリティ評価

| 項目 | 評価 |
|---|---|
| 実装コスト | **低**。read-only レポータ ~60 行 + テスト。既存契約器流用。 |
| 計算コスト | **ゼロ増**。`L,G` は `cert_inf` と同じ box 端点列挙 (O(n²))。Z3 クエリ数不変。 |
| 次元スケール | `cert_inf` 系は **次元非依存** (閉形式)。`cert_two/sdp` 由来 gene の `P`-norm tube は `2^n` 頂点律速 = 既存契約器と同じ天井 (n≤~12)。 |
| 健全性 | tube 不等式は **定理** (Banach + Lipschitz 合成)。feasibility と「軌道妥当性」は **証明外**。 |
| 既存ゲート互換 | **完全 additive**。`certifies()` 不変、semver 互換、fail-closed 継承。 |

**結論**: 案 B は **新規証明負荷ゼロ・既存 box 量の翻訳のみ** でユーザー要望 (「望ましい軌道に収束」の検査) を sound に満たせる。フィジビリティは高い。

---

## 5. 推奨実装 (additive, src 不可侵)

`research/target_trajectory_poc/` を起点に、次の read-only レポータを `research/` に置く (将来 `src/llcore/verifier/` への昇格は別途レビュー):

```python
@dataclass(frozen=True)
class TrackingTubeResult:
    contraction_ok: bool      # 既存ゲート (cert_inf) の結果
    L_state: float            # sup‖J_s‖_∞  (既存 box 量)
    G_input: float            # sup‖J_x‖_∞
    feasibility_residual: float   # 数値同定 (証明ではない)
    tube_radius: float        # G·w̄/(1−L), L>=1 なら inf
    admits: bool              # contraction_ok ∧ feasible ∧ tube<=r_max (fail-closed)

def tracking_tube(gene, x_ref, *, w_bar, r_max=None, eps_feas=1e-6) -> TrackingTubeResult: ...
```

進化ゲートに組むなら `_gate_admits` に `mode="tracking"` を追加し `tracking_tube(...).admits` を AND する (既存 `gated_evolve.py` の `"contraction"` モードと同型, fail-closed)。

---

## 6. 限界の honest 開示

1. **「理想軌道の妥当性」は証明できない。** 本枠組が保証するのは「*与えられた* 参照軌道への追従誤差が tube に閉じる」ことだけ。その参照軌道が「良い記憶曲線/良いタスク軌道」かは **タスク fitness 側の責任**であり、検証器の射程外 (control 理論でも reference は所与)。
2. **feasibility は数値同定であって証明ではない (案 B でも)。** PoC では参照軌道を構成的に系の解として作った (`ρ_feas=0`)。実運用で「望ましい `s_ref` が与えられる」逆問題では `ρ_feas>0` となり、tube は modeling error を外乱に吸収した `(G w̄+ρ_feas)/(1−L)` に緩む。`ρ_feas` の上界 sound 化 (例: Lipschitz による被覆) は将来課題。
3. **`cert_two/sdp` 由来 gene の tube は `P`-norm でしか sound に立たず `cond(P)` 倍ルーズ。** 初版は `cert_inf` PASS に tube を限定するのが最も clean。回転的 contraction (R-LLM で 39/50 が non_certified) への tube 拡張は SDP 計量を使う追加実装が要る。
4. **入力外乱 `w̄` の解釈はタスク依存。** 合成タスクでは入力ノイズで自然だが、R-LLM の離散 byte 入力では `w̄` の物理的意味が弱い (案 C の過剰主張リスク; R-LLM VERDICT の「payoff = evolvability, not learning」と同じ留保)。
5. **本 tube は離散・sup-norm の保守版。** 連続 CCM (式 (2)(3)) のリーマン計量 `M(x)` を学習する Sun+ 2011.12569 の路線は表現力は高いが SDP 学習負荷が重く、llcore の「CPU・閉形式・fail-closed」規律とは別物。本設計はあえて **既存契約量の再利用で済む最小 sound 版**を採る。

---

## 付録: 参照文献

- Sun, Jha, Fan, "Learning Certified Control using Contraction Metric," CoRL 2020 (arXiv:2011.12569) — 学習 contraction metric で目標軌道追従を保証。
- Manchester, Slotine, "Control Contraction Metrics: Convex and Intrinsic Criteria for Nonlinear Feedback Design," IEEE TAC 2017 — CCM の凸基準。
- Zhao, Manchester et al., "Tube-Certified Trajectory Tracking for Nonlinear Systems With Robust Control Contraction Metrics," arXiv:2109.04453 — tube 半径と計量条件数の関係 (式 (3) の出典)。
- Lohmiller, Slotine, "On Contraction Analysis for Non-linear Systems," Automatica 1998 — incremental stability の原典。
- llcore 内部: `research/verified_evolution_sdp_gate/coupled_nd.py` (n-dim 契約器), `research/coupled_z3_contraction/z3_infnorm_certifier.py` (Z3 inf-norm), `research/verified_lm_evolution/VERDICT.md` (R-LLM の evolvability 留保), `src/llcore/verifier/invariants.py` (Stage 1b Lipschitz contraction)。
