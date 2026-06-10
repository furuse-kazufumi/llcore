# The Verified-Plasticity Evaluation Framework

**Measuring *guarantee*, not *capability*, in the online structural adaptation of small LLM adapters**

---

**Status:** Framework deliverable / 論文化ドラフト (capstone). Not submitted.
**Date:** 2026-06-10.
**Discipline:** honest-disclosure — capability(性能)と guarantee(証明付き安定)を決して混同しない。
**正本データ:** `research/rllm_pivot/PHASE_{M1,1,2}_VERDICT.md` + 各 `phase*.py` / `*_results.json`。
**設計経緯:** `docs/EVOLVABLE_LLM_PLAN_2026_06_09.md` + `docs/SYSTEMATIZATION_2026_06_09.md`。
**実装:** `research/verified_evolution_sdp_gate/coupled_nd.py`(GeneCodec / Objective / VerifierBackend)+ `src/llcore/evolution/minimal_ga.py`(`evolve`)。
**再現性:** 全コード/データ [github.com/furuse-kazufumi/llcore](https://github.com/furuse-kazufumi/llcore)。モデル = SmolLM2-135M / Mamba-130M(いずれも Apache-2.0)。GPU 不要・全 `$0`/CPU。

> 本 doc は、それ自体を単独で引用可能な research artifact とすることを目的とする。連載解説記事(#38 防衛的公開 / #39 small-n 壁 / #40 capability-negative)は読者向けの噛み砕きであり、本 doc はその上に立つ学術 capstone である(数値・結論は同一の一次データに依拠する)。

---

## Abstract

凍結した小型 LLM に後付けした小次元の recurrent adapter が、その**構造そのもの**を online に適応させながら、(i) 発散せず (ii) 破滅的忘却を起こさないか、さらにその適応を**進化(evolution)**で駆動したとき**勾配降下(gradient descent)**に対し性能で競合しうるか、を問う。我々は脆い単一機構ではなく**評価枠組み自体を deliverable** とする立場をとり、**Verified-Plasticity Evaluation Framework** を提示する。これは「online 構造適応は証明可能に収縮的(ρ<1)に保てるか」を第一級・falsifiable な指標とし、候補 method を 6 装置の統計的厳密性ハーネス(事前登録 / Holm 連言 / artifact 規律 / 反証条項 / 自己検出力監査 / 反 over-claim critic)の下で比較する。

`$0`/CPU の実験弧から、確定した知見は次の通りである。

1. **GUARANTEE は立つ。** 実 Net2Net 構造手術(width_grow)の下で、sound certificate(cert_two / cert_sdp / cert_inf)は **0 観測 false-admit** を保つ。とりわけ cert_sdp は誤許可 0% かつ過剰棄却 4.6% で **sound かつ最も navigable** である。
2. **経験ベースの gate は危険である。** STABLE 風経験 gate は真に発散する gene の **84.2%(80/95)を「安全」と誤許可**する(EPS_FORGET=1e-2/T=64/K_PROBE=8 固定の設定依存値。方向=STABLE は危険は頑健だが 84% を設定非依存数値として扱わない)。基質は tanh で常時有界なため、有限ホライズン観測では「摂動を忘れたように見える」が真 ρ≥1 であり、sound certificate でなければ見抜けない。
3. **CAPABILITY は NEGATIVE である。** 実 SmolLM2-135M 由来の交差エントロピー地形で、MAP-Elites は finite-difference 勾配を 20/20 で上回る(一見 EXISTS)が、**強い解析(autograd)勾配が同予算で 19/20 でこれを逆転**する。進化の見かけの勝ちは**弱いベースラインの artifact** であり、synthetic 地形でも(平均で)一貫する。

結論として、本枠組みの価値は **capability ではなく guarantee** にある。そして本研究で最も意味のある方法論的成果は **negative-by-design かつ自己誘発的**である — 枠組みに組み込んだ strong-gradient meta-gate が、publish 前に capability の魅力的な false-positive を実際に 1 件排除した。

---

## 1. Motivation — なぜ「評価枠組み」なのか

### 1.1 stability-plasticity の TRIZ 矛盾

LLM の**構造**(重みだけでなく次元・トポロジーそのもの)を online に適応させる問題は、古典的な **stability-plasticity の矛盾**の上にある。構造の自由度を上げる(plasticity を増す)ほど、発散と破滅的忘却のリスクが上がる(stability を損なう)。逆に by-construction で構造を縛れば安定するが、表現力(plasticity)を犠牲にする。

TRIZ の観点では、これは「**改善したい特性 = 構造適応の自由度** vs **悪化する特性 = 系の安定性/制御可能性**」という典型的な technical contradiction である。本研究の解は、両者を時間軸で分離する **prove-then-reject** にある。**探索(変異の提案)は自由**にし、**採用(次世代への組込み)だけを sound certificate で gate** する。すなわち「適応は許すが、発散・忘却は許さない」**homeostatic constraint** として small-n per-component(n≤4-6)域で矛盾を解消する(高次元の navigable-sound certifier は §5 の通り未解決)。これは「by-construction で最初から縛る」設計(後述 §6)とは双対の設計軸である。

### 1.2 なぜ method でなく枠組みを deliverable にするか

「我々の method が強い」と自分で主張しても意味がない。新規 method を売る前に、まず**測る物差し**を作るべきである。理由は二つ。

第一に、**capability と guarantee は routinely 混同される**。「賢くなった(perplexity 改善)」と「安定している(発散しない)」は別物であり、後者を測らずに前者を主張する研究が多い。本枠組みはこの二つを構造的に分離し、それぞれを独立に falsifiable に測る。

第二に、**進化系の主張は弱いベースラインで容易に false-positive を生む**(§4.2 で実証)。「物差し」の側に、勝った瞬間に強い対戦相手を呼ぶ反証機構(meta-gate)を組み込んでおかなければ、誇張は構造的に止まらない。したがって deliverable は**脆い単一機構ではなく、falsifiable に測り method 間で比較する評価枠組みそのもの**とする。

---

## 2. Framework definition — 被験 method・6 装置・3 plug-point

### 2.1 何を測るか(第一級指標 = North Star)

1. **成長操作下 soundness** — width_grow / branch_add の構造手術後も certificate が false-admit を起こさない。
2. **coupling-aware 合成 soundness** — per-block AND は結合下で不 sound(禁止)、full-system certificate が必須。
3. **枠組み判別力(H-discriminative)** — 無 gate(危険)/ STABLE 風経験 gate(危険)/ sound cert(安全)/ Mamba(自明安全)を soundness で分離して測れる。
4. **framework 性** — 3 plug-point の 1 オブジェクト差替拡張性。
5. **capability verdict** — terrain 上で EXISTS / NULL / ARTIFACT を確定する。

### 2.2 被験 method(plug-in 比較対象)

| method | 役割 | 性質 |
|---|---|---|
| **VSOA**(cert-gated topology evolution) | 主被験 | sound certificate gate 付きの構造進化 |
| **無 gate** | 負の対照 | gate を外した同一進化(危険側の床) |
| **STABLE 風経験 gate** | 既踏比較 | 有限ホライズン観測ベースの経験的 gate |
| **Mamba-130M** | 正の対照 | stable-by-construction(構造的安定の SSM) |

### 2.3 6 装置(methodology backbone)

事前登録(結果を見る前に判定基準を固定)/ Holm 連言(複数条件 AND を多重比較補正)/ artifact 規律(見かけの勝ちは強ベースラインで再検証)/ 反証条項(各仮説に falsification 条件を明記)/ 自己検出力監査(NULL は検出力不足でないか確認)/ 反 over-claim critic(主張を内部 AI が攻撃)。

この methodology 層が **honest-disclosure を構造に持ち込む**。これ無しに「進化が本物」と主張する権利は成立しない、というのが本枠組みの立場である。

### 2.4 3 plug-point(framework-ness)

```python
codec     = CoupledNDGeneCodec(n)         # 基質 (GeneCodec): 任意次元 n の coupled recurrent adapter
objective = RotationNDObjective(n=n)       # 方向 (Objective): 差替可能な fitness
verifier  = make_nd_verifier("sdp")       # gate  (VerifierBackend): none / inf_norm / two_norm / sdp
# verifier.certifies(gene) が ρ<1 を sound に判定。各 plug-point は 1 オブジェクト差替で evolve に載る。
```

**gate 規律(最重要):** 安定 certificate を破らないことのみで gate し、**fitness(capability)では gate しない**。これが §1.1 の homeostatic constraint の実装である。

### 2.5 基質と「安定性指標の正体」

基質は coupled n 次元の recurrent adapter

```
s' = decay ⊙ s + (1 − decay) ⊙ tanh(W s + x),   decay ∈ [0,1]^n,  W ∈ [−2, 2]^{n×n}
```

を、凍結 LLM の hidden state の低次元射影に適用する。

ここで**安定性指標の正体**を厳密に述べる。kernel は tanh で**状態が常時有界**であるため、不安定は**状態ノルムの発散としては現れない**。不安定は **echo-state property の失敗**(線形化感度における摂動の持続・増幅)として現れる。収縮 ρ<1 とは「摂動を忘れる」性質である。この区別こそが、ノルム監視や有限ホライズン「忘却テスト」を無効化する核心である(§4.1)。

---

## 3. Methods — certificate と gate の構成

### 3.1 sound certificate の梯子(cert ladder)

到達可能な Jacobian box(2^n 頂点)上で ρ(J)<1 を sound に判定する 3 段。すべて ρ(J)<1 を含意する(=合格なら本当に収縮)。

| certifier | 判定法 | コスト | 性質 |
|---|---|---|---|
| `cert_inf` | 閉形式 ‖J‖_∞ < 1(各行の絶対値和) | `O(n²)`, **solver-free** | 安いが保守的(navigability 低) |
| `cert_two` | 全 2^n 頂点で σ_max < 1(SVD) | `2^n` | 中位 |
| `cert_sdp` | 共通二次 Lyapunov LMI(cvxpy / CLARABEL, fail-closed) | `2^n` 頂点 + SDP | 最 navigable |

soundness の監査には、真のスペクトル半径を**下から**近似する独立オラクル `empirical_rho` を用いる。「0 観測 false-admit」はこの from-below 監査の結果であり、強い consistency 証拠だが**機械証明ではない**(§5)。

### 3.2 STABLE 風経験 gate(既踏比較)

有限ホライズン(EPS_FORGET=1e-2 / T=64 / K_PROBE=8 固定)で摂動の忘却を経験的に観測し、忘れたら「安全」とする gate。これは多くの実用 continual-learning 系が依拠する「観測ベース安全判定」の代表である。本研究はこれを**負の既踏ベースライン**として測る(§4.1 で 84% 見逃しを確定)。

### 3.3 Mamba SSM Lyapunov 正対照

正対照は**学習でなく parameterization の自明性を検定**する。Mamba-130M は全 24 層の SSM が連続対角 `A = −exp(A_log) < 0`(589,824 channel,state エントリの 100%)であるため、任意の Δ>0 に対し `λ_max = max(Δ·A) ≤ 0` が**構造的に自明に成立**する。すなわち任意の valid Mamba が stable-by-construction である。**ただしこの PASS は SSM 状態再帰の安定性のみであり、conv1d / SiLU / MLP を含む full Lipschitz ではない。代表 Δ で marginal な channel は非厳密(≤0 を over-claim しない)。**

対して SmolLM2-135M(Llama 系)は SSM 状態再帰を持たない(self_attn + mlp のみ)。したがって安定性は**後付けの gate で初めて課される**。枠組みは「安全な土台(gate 不要)」と「gate が要る土台」を **base-level で分離**できる(base-level 判別 PASS)。

### 3.4 per-row 不変条件 gate / coupling certificate

per-component の局所判定では cert_inf は small-n で脆い(§4.1 注)。ブロック間結合(coupling)については、**per-block AND(各ブロック単体の合格を AND)は結合下で不 sound** であり禁止する。結合系では**系全体を一括で証明する full-system certificate** を必須とする(Phase 1 確認)。これは「各部品が安全でも合成すると暴走しうる」盲点への対処である。

---

## 4. Results — exact 数値

すべて `$0`/CPU。各 Phase は独立に敵対的検証(3 独立 skeptic + 実機 3 seed 再走)済で、機構的結論を覆す **MAJOR 0 / 全 MINOR**、数値 mismatch ゼロ。

### 4.1 H-discriminative 枠組み判別力(GUARANTEE 主役)

n=6、95 発散 / 305 収縮 gene の集団に対する、発散 gene の false-admit と収縮 gene の過剰棄却。

| method | 発散 gene の false-admit | 収縮の過剰棄却 | 判定 |
|---|---|---|---|
| 無 gate(負対照) | **95/95 = 100%** | — | 危険(発散を全部通す) |
| STABLE 風経験 gate | **80/95 = 84.2%** | — | 危険(発散の 84% を「安全」と誤許可。EPS_FORGET=1e-2/T=64/K_PROBE=8 固定の設定依存値。方向=STABLE は危険は頑健だが 84% を設定非依存数値として扱わない) |
| `cert_inf`(sound) | **0%** | 70.5% | 安全だが保守的 |
| `cert_two`(sound) | **0%** | 52.8% | 安全・中位 |
| `cert_sdp`(sound) | **0%** | **4.6%** | **sound かつ最 navigable** |
| Mamba 風正対照(0 発散集団) | 全 method 0 false-admit | — | 安全 family を誤棄却しない |

**STABLE 風 gate が 84% 見逃す理由(機構):** kernel は tanh で常時有界なため、有限ホライズン観測では「摂動忘却したように見える」が真 ρ≥1(echo-state property の失敗)。sound certificate でなければ見抜けない。

**さらに強い反例:** ρ≈2.9 の発散 gene でも、**単一軌道の摂動感度すら発散しない**(実測 1 → 2e-14。tanh 飽和 + 方向ミスアラインのため)。状態ノルム監視も、有限忘却テストも、単一軌道感度も、ρ≥1 を見逃す。box-sup の sound certificate(σ_max = 4.87 > 1 で reject)のみが見抜く。これが「経験は騙され、sound cert だけが見抜く」の最も鋭い実例である。

### 4.2 capability:NULL_TIE → ARTIFACT+NEGATIVE(honest-disclosure の真価)

**(A) synthetic 多峰地形(K=6 basin)= NULL_TIE。**
MAP-Elites ≈ gradient ≈ random。ME vs gradient で mean_diff +0.028 / Wilcoxon p=0.39 / sign_delta=0(n=20)。全方向 4 条件 AND 不成立 = **純粋な引き分け**。**+0.028 を検出する power は実質ゼロ(wins=losses=10, sign_delta=0)。NULL_TIE は decisive negative ではない。** capability 優位の未実証であり、「進化の敗北の decisive proof」でも「powered な等価性 proof」でもない(§5)。

**(B) 実 SmolLM2-CE 地形(§7.1 of PHASE_2_VERDICT)= ARTIFACT+NEGATIVE。** この地形は full-vocab softmax でなく hidden-クラスタ CE proxy である。
SmolLM2-135M の中間層 hidden を n=6 に射影し、「次に来る hidden クラスタ」を当てる CE 地形(合成ガウスでなくモデル自身の内部ダイナミクス由来)を構築。同予算(forward CE 評価回数)で held-out 文の予測を 20 seed 比較。**ただし「同予算」は forward CE 評価回数の同一であって、有効更新ステップ数は torch 2000 Adam vs finite-diff/evolution ~95 と非対称(これが ARTIFACT の機構)。** held-out 平均 fitness(= −CE、高いほど良い):

| method | held-out 平均 | 備考 |
|---|---|---|
| **strong analytic gradient(torch Adam)** | **−1.446** | **全 method 最良** |
| MAP-Elites | −1.454 | 2 位 |
| random | −1.473 | |
| finite-diff gradient | −1.483 | 最下位 |

- ME vs finite-diff:diff **+0.029**、**20/20**、p=9.5e-7 → 4 条件 AND **成立**(一見 EXISTS)。
- ME vs strong analytic gradient:diff **+0.008(勾配側)**、**19/20** で勾配が逆転、p=3.5e-4 → 4 条件 AND **不成立**。

**∴ ME の勝ちは finite-diff の弱さ(cold-start / dim+1 評価/step / 予算内 ~95 step)の artifact。** 強い勾配では gradient > evolution = **実地形でも capability NEGATIVE**。synthetic 地形に強い解析勾配を足しても勾配が最高平均(分散が大きく paired 検定は引き分け止まり)であり、**実地形は paired 19/20 で decisive、synthetic は torch vs ME の paired 4 条件 AND は TIE(mean のみ gradient 最良)— 両地形で「gradient が最高平均」が一貫**。

**★ honest-disclosure の真価:** strong-gradient meta-gate が無ければ「進化が実地形で 20/20 capability 勝利」という false-positive を誤結論していた。「勝った気になる前に内訳を疑う」が、data の上で実際に false-positive を 1 件排除した実例である。これは負けの報告ではなく、**枠組みが機能した報告**である。

### 4.3 framework 性(F8)

**(b) 3 plug-point swap = PASS。** GeneCodec / Objective / VerifierBackend を 1 オブジェクト差替で交換(src 無改変 = git diff 空、pytest 17 green、per-gene two⇒sdp / inf⇒sdp が 3000 gene で 0 違反)。

**(a) 構造多様性 → 汎化 load-bearing = NULL(第一級 NULL)。** 「構造多様性が汎化を助ける」仮説は held-out diff +0.011 / p=0.55 で**立たず**。これも正直に開示する。

### 4.4 Mamba SSM Lyapunov 正対照(§7.3)

Mamba-130M 全 24 層で `A = −exp(A_log) < 0`(589,824 ch)→ λ_max ≤ 0 自明 PASS。SmolLM2 は SSM 不在(Llama, self_attn + mlp のみ)= gate 必須 → **base-level 判別 PASS**。正対照は parameterization の自明性(任意の valid Mamba で構造的成立 = 学習でなくパラメタライズを検定)。**この PASS は SSM 状態再帰の安定性のみで conv1d / SiLU / MLP の full Lipschitz ではない。代表 Δ で marginal な channel は非厳密(≤0 を over-claim しない)。**

### 4.5 統合判定

| Decision gate | 内容 | 判定 |
|---|---|---|
| **Decision gate 1** | small-n per-component 域の soundness / feasibility | **PASS** |
| **Decision gate 2** | 枠組み妥当性 / capability / 価値所在 | 枠組み妥当性 **PASS** / capability **NEGATIVE** / 価値 **GUARANTEE 確定** |

**∴ 「進化可能な LLM」= 進化が性能で勝つ枠組みではなく、「online で構造適応しても発散・破滅的忘却しないことを sound に保証・測定する枠組み」として確立。**

---

## 5. Honest limitations — decisive negative と未実証の厳密な区別

over-claim 禁止。以下を **decisive(確定した否定)**と **unestablished(未実証)**に分けて明記する。

**decisive negatives(確定した否定):**
- **capability は売れない。** 強い解析勾配が進化を上回る(実地形 19/20 有意、両地形で平均一貫)。進化の価値は perplexity 改善ではない。
- **verified 構造進化は small-n per-component(n≤4-6)限定。** 高次元で「navigable かつ scalable な sound certifier」は**現存しない**(第一級 negative)。2^n 頂点コストは cert_sdp でも不変(SDP は navigability の天井を上げただけで壁を破っていない)。
- **per-block AND は禁止。** coupling 下で不 sound。

**unestablished(未実証 — NULL を敗北と断定しない):**
- **capability NULL_TIE は「非有意の引き分け」**であり、「進化が勾配に劣る decisive proof」でも「powered な等価性 proof」でもない(power 未分析)。
- **多峰性の頑健範囲は「>1(多峰)」まで。** 40 basin は高次元 hillclimb 非収束 artifact の可能性。
- **gate 中立性は held-out 限定・capability flat regime の観測。** train 側は 0.25 差で archive 探索制約があり、gate を掛けると実地形で −0.028 落ちる(可塑性を測定可能に削る。ただし進化に capability 優位が無いので結論には不影響)。
- **STABLE 84% は設定依存。** EPS_FORGET=1e-2 / T=64 / K_PROBE=8 固定で感度未測定。方向(STABLE は危険)は頑健だが、「84%」を設定非依存数値として扱わない。
- **構造多様性 → 汎化 = NULL(立たず)。**

**measurement caveats(測定上の留保):**
- **soundness は 0 観測 false-admit**(empirical_rho from-below の強 consistency であり、絶対証明でも機械証明でもない。near-boundary を取りこぼしうる)。
- **実 CE は hidden-クラスタ CE proxy**(full-vocab softmax でない。small n で full-vocab は退化)。
- 「strong gradient is best」は backprop が exact 勾配を無料で与える前提(これはまさに実 LLM 学習が行うことなので現実的比較)。

**scope caveats(範囲の留保):**
- **adapter scope:** 「実 LLM」修飾は後付け adapter に限定(base 凍結)。**tiny → SmolLM2 の load-bearing transfer は未検証**。
- 普及/市場価値:guarantee-only 枠組みは地味で、consumer story と需要側証拠は未確立(ユーザー判断待ち)。

---

## 6. Related work — 機構は既踏、貢献は certificate gate × 安さ

### 6.1 アルゴリズム的先行(機構は既踏)

- **NEAT 系の topology evolution** — 構造を進化させる機構は既踏。本研究の貢献は「進化機構の新規性」ではなく、**構造変異の採用を sound contraction certificate で gate する点**にある。
- **DARTS / PostNAS** — architecture search は既踏。それらは accuracy / latency / FLOPs で競う。本研究は**guarantee(ρ<1 の sound 保証)を第一級指標にする**点で軸が異なる。
- **by-construction Lipschitz 制約**(Enforced-Lipschitz / R2DN 系) — 構造で収縮性を強制する双対設計。本研究の prove-then-reject は、**構造制約なしに任意更新を検査**できる(表現力を犠牲にしない)代わりに証明コストを払う、という設計トレードオフを取る(§1.1)。
- **進化ループ × 健全検査 gate**(Katz & Peled 系の GP × model-checking) — gate を進化に置く**パターン自体は既踏**。したがって我々は gate パターンの新規性を主張しない。**記憶コアの収縮性への適用 × small-n での安さ(0.013h ≪ 30h)× certificate ladder の完全特性評価**が貢献である。

要するに**貢献 = certificate gate(soundness)× 安さ(small-n feasibility)× 評価枠組み化**であり、個々の機構の発明ではない。

### 6.2 未検証の自己改善主張との対比(第三者未検証 = 事実のみ)

近年、自己改善・継続学習を謳う著名 OSS が多数ある(2026-06-10 競合スキャン):hermes-agent(NousResearch, 189k★)は「20+ スキルで 40% 高速」、ECC(211.8k★)は Continuous Learning、headroom は learn を掲げる。**これらの性能/安定性の優位主張は、いずれも第三者未検証の自社ベンチである**(star 数は人気の証であって性能優位の証ではない)。本記述は競合を貶めるものではなく、「**未検証である**」という事実のみを述べる。

本枠組みの位置づけは明確である。verified-plasticity は、この種の「賢くなった/安定した」という主張が**本物か幻かを sound certificate で falsifiable に判別する道具**である。実際、§4.2 で我々自身の「進化が 20/20 で勝った」という主張を、自分の枠組みが幻(ARTIFACT)と判定した。

### 6.3 世界モデルの「保証ではない」との対比

世界モデル研究者 藤吉弘亘(2026 講演 p.51)は、世界モデルを「**安全設計に寄与するが保証ではない**」と本人が honest に限定している。verified-plasticity は、対照的に sound certificate によって **GUARANTEE(証明付き安定)を出す**点で異なる(ただし small-n per-component 域に限る、という honest 限定付きで)。

加えて、同講演の核「**人が与える/機械が自ら獲得する境界が、歴史的に広がり続けた**」というテーゼは、本研究の進化テーゼ(構造を人が固定せず機械が online に獲得する)と**同型**である。本枠組みは、その「機械が自ら獲得する」境界拡張に対し、**発散しないことの保証**を後付けで課す装置と位置づけられる。

### 6.4 比喩的位置づけ(Langton's ant)

ラングトンの蟻は、単純決定論ルールが**見かけの秩序/複雑さ**を生む系である。経験的観測は騙され、sound certificate がその幻を見抜く。本研究では「見かけの安定」(STABLE gate が 84% 見逃す, §4.1)も「見かけの進化」(ME が弱い勾配に 20/20 で勝つ, §4.2)も、いずれも本質では deterministic-simple に collapse する。経験は見かけに騙され、sound certificate だけが本質を見る — これが 3 段(#38→#39→#40)の honest disclosure が収束する一点である。

---

## 7. Conclusion — 価値は GUARANTEE

Verified-Plasticity Evaluation Framework は、小型 LLM adapter の online 構造適応が**証明可能に収縮的に保たれるか**を、method-agnostic かつ falsifiable に測り、sound certifier と「発散の 84% を見逃す経験 gate」を判別する枠組みとして確立した。

その honest な verdict は、**価値は guarantee であって capability ではない**ということである。進化は、実地形でも synthetic 地形でも、強い勾配に性能で勝たない。最も意味のある方法論的成果は negative-by-design かつ自己誘発的である — 枠組みの strong-gradient meta-gate が、魅力的な capability false-positive(弱いベースライン相手の 20/20)を、主張される前に止めた。

我々はこう主張する。「provably evolvable」な系が最も必要とするのは、いずれかの単一機構ではなく、**この自己懐疑の規律**である。「online で構造を変えても発散・破滅的忘却しないことを、sound に保証・測定する」— それが地味であっても、賢さを盛らず安全性で勝負すると決めた以上、これが正直な姿である。

**Open(未解決):** (1) n≤6 を超えるスケール(navigable-scalable certifier の発見)。(2) tiny → 実 LLM transfer の load-bearing 検証。(3) consumer story + 需要側証拠(ユーザー判断待ち)。(4) 投稿先選定。

---

**関連 doc:** `PHASE_M1_VERDICT.md` / `PHASE_1_VERDICT.md` / `PHASE_2_VERDICT.md` / `SYSTEMATIZATION_2026_06_09.md` / `EVOLVABLE_LLM_PLAN_2026_06_09.md` / `PAPER_DRAFT_verified_plasticity.md`(extended abstract)。
