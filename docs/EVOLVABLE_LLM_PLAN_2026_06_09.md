# 「進化可能な LLM」FW — 確定再設計計画

**作成**: 2026-06-09
**前提文書**: `docs/SYSTEMATIZATION_2026_06_09.md`(確定結論=GUARANTEE は立つ / CAPABILITY は decisive NEGATIVE / 4 条件 / 賭け 1-4)、`research/rllm_pivot/topology_evolution_prior_art.md`(prior-art マップ)
**統合方法**: 5 設計案 + 3 敵対 judge スコアを統合。勝者 framing を核に runner-up の長所を接ぐ(単純な 1 案選択ではない)。
**規律**: 全編 honest-disclosure。capability(進化が perplexity/CE で勾配を上回る)と guarantee(進化した個体が証明付きで安定)を**決して混同しない**。NEGATIVE を前景化する。load-bearing な実装事実は `src/llcore/` で一次照合済。

> **一行宣言**: 本計画は「**guarantee 主軸 + capability terrain-design を明示ラベル付き falsifiable 副線**」の構成を採る。judge 2 案が同点(各 total 19/25)だが、honest_alignment 満点の guarantee-first(案 A=VSOA)を**主軸**に、terrain-capability-bet(案 B)を **meta-gate 付き副線**に、verified-topology-NAS(案 3, prior-art 象限 A=最も未踏)の **三者分業 framing** を接ぐ。

---

## ① エグゼクティブ・サマリ

**FW を一文で**: 実小型 LLM(SmolLM2-135M)に後付けした **n≤16 の verified recurrent adapter block** に対し、その**離散トポロジー(width / branch / op)を進化が探索**し、各構造変更を**訓練ループ内に常駐する per-component cert_inf fail-closed gate** が `ρ<1` で admit/reject し、**重みは gradient に全面譲渡**する——「証明付きで安定な構造変更だけが世代を越える」三者分業フレームワーク。

**価値命題(guarantee 主軸)**: 売りは guarantee 一本。「PostNAS と同じ後付け構造探索だが、**採用する各 architecture が収縮 certificate(ρ<1, echo-state 安定)を持つ**——provably-stable architecture を産む NAS を、Kaggle T4 級の安さで訓練ループ内に常駐させ、無 gate baseline 比で online drift(出力ノルム発散 ρ→1 越境)を有意に抑える」。**capability(perplexity 改善・進化が勾配に勝つ)は一切売りにしない**(M3 戒め)。prior-art 象限 A(verified gate × 構造進化 × LLM 認知核=corpus 全実装研究で空白)に正確に着地する。

**最大の賭け**: 案の唯一の価値「0 *観測* false-admit」が、**未実装の `width_grow`(Net2Net function-preserving 拡張)× cert_inf soundness の整合**に全面依存する。しかも `width_grow` が n を増やすと `_t_min` の `M=Σ|W|` が単調増加 → `sech²(M)` が縮む → achievable-t box が広がる → 拡張後に `sup‖J‖∞` が 1 を越えやすくなる**構造的圧力**がある。これは「リスク」ではなく**存立条件**であり、Phase 1 で潰せねば案全体が崩壊する(§⑩ 賭け 1 / 撤退条件参照)。

---

## ② 5 設計の比較と勝者選定

### 2.1 比較表

| 案 | angle | judge total | 核心 | fatal flaw / 最大弱点 |
|---|---|---|---|---|
| **A** | A-guarantee-first(VSOA: Verified-Stable Online Adaptation) | **19/25** | 構造変更を per-component cert_inf fail-closed gate で admit/reject。capability 封印・guarantee 主軸。次元の壁を「block を小 n に切る」設計制約に転化 | `width_grow×Net2Net×cert_inf` soundness 未証明=**存立条件**(リスクでなく)。width_grow が box を広げ soundness が壊れやすい構造的圧力。novelty は STABLE との 2 点差に圧縮 |
| **B** | B-terrain-capability-bet(Multimodal-Landscape Verified Evolution) | **19/25** | 多峰/欺瞞地形を意図的に構成し ③(MAP-Elites)が gradient を held-out CE で上回るか BG10 meta-gate 付きで falsify。capability の**存否**を決着 | 最尤シナリオ(NULL→guarantee 退避)の着地点が案 A=「高価な多峰検証を経て案 A が既に居る場所に戻る」公算。novelty は方法論資産に限定 |
| **3** | C-role-split-verified-NAS(Contraction-Gated Topology Mutation) | (truncated, 推定 guarantee 系) | gradient=重み / 進化=離散 topology / verified gate=provably-stable architecture のみ採用、の三者分業。prior-art 象限 A「最も未踏」に着地 | 案 A と同系の guarantee-niche。NAS 機構自体は Jet-Nemotron/PostNAS で枯れ、正味 novelty は「certificate × NAS × 安さ」の交差のみ |
| 4 | (入力 JSON で truncated) | — | — | — |
| 5 | (入力 JSON で truncated) | — | — | — |

> ⚠ honest 留保: 入力 JSON で案 3 は途中まで、案 4/5 は本文未到達。確認できた範囲(案 1=A / 案 2=B / 案 3 冒頭)で統合判断する。案 3 は案 A と同じ guarantee-first 系で prior-art 象限 A に着地し、framing(三者分業 NAS)が最も明快——よってその **framing を主軸の説明枠**として採用する。

### 2.2 勝者選定理由

1. **honest 制約整合が最優先**: judge は 2 案とも honest_alignment=5 / guarantee_capability_honesty=5(B は 4)を付け、SYSTEMATIZATION の confirmatory/NEGATIVE 線引きと矛盾しない唯一の framing が **guarantee-first** であることを確認。CAPABILITY decisive NEGATIVE(M3)を主張で踏まない案だけが 3 敵対レビューを構造的に通過する。
2. **B 単独は最尤で A に合流する**: judge B verdict「最尤シナリオは NULL→guarantee 退避、その着地点は prior-art 既定本命(案 A)と一致」。よって B を主軸にすると高価な遠回りになる。**B は capability の存否を falsify する副線**として価値を保つ(EXISTS なら最大差別化、NULL なら decisive negative が guarantee-niche を正当化)。
3. **3 案の framing が最も運べる**: 「gradient=重み / 進化=離散 topology / verified gate=安定 architecture」の三者分業は、§7.2-3(b)「進化を勾配の解けない離散自由度へ移す」を実装に直結させ、M3(素の loss 地形で進化は無価値)を**構造的に回避**する。
4. **接ぎ木の方針**: **A の核(small-n verified block + cert_inf fail-closed gate + width_grow soundness を存立条件として最優先検証)** + **B の副線(terrain-design + capability-vs-artifact meta-gate)** + **3 の framing(三者分業 NAS、prior-art 象限 A)**。

---

## ③ 確定 framing(統合版)

### 3.1 FW の定義

> **「進化可能な LLM」FW = 実小型 LLM の重みを gradient に全面譲渡し、進化は gradient が微分できない離散トポロジー空間(block の width / branch / op 種別)だけを探索し、各構造変更候補を訓練ループ内に常駐する per-component sound contraction certifier(cert_inf, `ρ<1`)が fail-closed に admit/reject し、admit された provably-stable な architecture のみを gradient が訓練する——三者分業フレームワーク。**

これは Jet-Nemotron/PostNAS を「証明付き」にした NAS であり、llcore Stage-B「検証 core を訓練ループ内 long-range path に」を実 base へ拡張したものである。

### 3.2 価値命題(guarantee 主軸・capability 封印)

| 項目 | 内容 |
|---|---|
| **第一義の売り** | 採用 architecture **全件が訓練ループ内 cheap gate で形式安定 certificate(ρ<1)を持つ**。無 gate baseline 比で online drift(出力ノルム ρ→1 越境)を有意に抑える |
| **capability の扱い** | perplexity は gradient/PostNAS と**同等を目指す(勝つと言わない=M3 honest)**。可塑性を殺さない(無 gate と perplexity 有意差なし)ことのみ示す |
| **正味 novelty(狭く明示)** | 「certificate × NAS mutation × 実 LLM × T4 級の安さ」の統合縮約。機構(node/branch を動かす / MAP-Elites / certificate)はどれも単独先行あり=盛れない。**capability 目新しさはゼロ** |
| **prior-art 着地** | 象限 A(形式安定 gate × 逐次自己改変 × LLM 認知核=corpus 全実装研究で空白)。STABLE(2510.16089)が gate 発想を既踏、差は「パラメータ編集→構造変更」「経験 budget→sound cert_inf」の 2 点のみ(honest) |
| **FullSense 接続** | on-prem / Approval Bus / 責任ある AI 哲学と整合する「迂回できない安定 gate」。fail-closed = FullSense デフォルト |

### 3.3 North Star(測れる成功)

guarantee 主軸の北極星は **2 点同時成立**(perplexity 改善幅は北極星に**含めない**):

1. **certified-stable rate = 100%**: 進化が探索した N 個の architecture 候補のうち admit された**全件**が `ρ<1` certificate を持ち、**0 *観測* false-admit を実 LLM block で維持**。
2. **capability 非劣化 + drift 抑制**: admit 集団から gradient 訓練した best architecture が、(i) 無 gate baseline と perplexity で**有意差なし**(可塑性を殺していない)、(ii) online drift(長系列出力ノルム発散 ρ→1 越境)が無 gate / STABLE 風経験 gate より**有意に小さい**(片側検定 p<0.05, 事前登録)。

**副線(terrain-design)の北極星**: `capability-or-artifact verdict ∈ {EXISTS, NULL, ARTIFACT}` を proper power で 1 つ確定すること(perplexity 改善幅ではない)。

---

## ④ base 選定

| 役割 | モデル | license | 理由 |
|---|---|---|---|
| **主 base** | **SmolLM2-135M** (HuggingFace) | Apache-2.0 | (a) certificate 計算成立帯(Lipschitz-Transformer arXiv:2507.13338 が 2M〜145M で成立)と param 帯が重なる**唯一の安全帯**、(b) 135M なら Kaggle T4 で full fine-tune + 多世代探索が現実的、(c) Qwen 回避で商用障壁なし(`feedback_qwen_commercial_barrier`) |
| **副 base(正の対照)** | **Mamba-130M** (state-spaces) | Apache-2.0 | stable-by-construction(arXiv:2406.00209, 非正の最大 Lyapunov 指数)= **cert が自明 PASS する正の対照**。gate の判別力(SmolLM2 で reject 発生 / Mamba で自明 PASS)を示す。GPU カーネル依存ゆえ**副経路のみ**(主経路は T4 純正に保つ) |
| **再現性予備** | Pythia-160M (EleutherAI) | Apache-2.0 | 学習過程 checkpoint 公開 = ablation/再現性に強い |
| **除外** | Gemma 系 | Gemma license(Apache/MIT 要件外) | license 要件外で除外 |

> ⚠ honest 留保: param/license は公開情報からの記載。**採用前に各モデルカードで再確認**(prior-art §⑧)。base 凍結 + 探索対象 block のみ可変(PostNAS 流の MLP/attention 凍結戦略)で T4 budget を守る。1B 級 scratch CPU 学習は**不可能と明言**——base 継承を死守。

---

## ⑤ 最初の 1 構造部品 + gate 設計

### 5.1 最初に進化させる構造単位

**SmolLM2-135M に後付けする 1 個の recurrent adapter block(verified core)** = `CoupledNDGene` 型 `(decay∈[0,1]^n, W∈[-2,2]^{n×n})`、**n を小さく固定(n=8〜16)**。

なぜこの低次元 certificate を許すか(**次元の壁を設計制約に転化**):

1. **certificate は block 全体でなく*この小 n の Jacobian* だけに掛かる**——実 LLM の数千次元 weight space を直接 certify しない(`2^n` 壁を回避する設計上の分離)。
2. `cert_inf` は `_infnorm_sup(decay, W, t_lo) < 1`(`backends.py:111-119`)で **per-row O(n²) 閉形式・頂点列挙ゼロ**=ladder で**唯一スケールする段**(SYSTEMATIZATION §1.2 表「唯一スケール」)。一次照合済: `_infnorm_sup` は `for i in range(n)` の per-row ループで 2^n 列挙を含まない。
3. この block を Stage-B 流に「実 Transformer 内の唯一の long-range/adaptive path」に配線すれば、core が load-bearing(Stage-B B-G1 4/4)かつ certify 対象は n 次元に閉じる。

地形多峰性は「block の topology(branch 有無 / op 種別 / memory horizon)という離散自由度」に注入し、連続 `(decay, W)` は gradient に任せる分業(§6 の三者分業を実装に焼く)。

### 5.2 gate 設計(どの certificate を実 LLM 次元で成立させるか)

**per-component 低次元 cert_inf を主力**にする。SYSTEMATIZATION §5.2 が示す通り、実 LLM 次元では `cert_two`/`cert_sdp` は `2^n` 壁(n=32 で 1099GB)で測定不能、vertex-free B2 も n=16 で cert_inf に収束(scale 解にならない)。よって:

| 設計判断 | 内容 |
|---|---|
| **block を小 n に切る** | 各 block の Jacobian に cert_inf(O(n²) 閉形式 `sup‖J‖∞<1`, 頂点フリー)を per-component に掛ける=**次元の壁の*下*で動く**設計 |
| **soundness 範囲** | achievable-t box(`_t_min = 1 - tanh²(Σ|W| + max_input·Σ|V|)`, `backends.py:96-98`)上で保証。0 *観測* false-admit を継承(機械証明でなく数値検査+独立 eigen 再検査と明示) |
| **multi-block 合成** | 各 block 独立 cert_inf を **AND** して保守的に合成安定を保証(per-component 合成は低次元のまま)。**⚠ 後述の coupling 盲点リスクを承知の上で**——§⑪ 留保参照 |
| **gate 常駐位置** | **訓練ループ内に常駐**(Stage-B B-G4: post-hoc certify は 17-19x コスト + 学習破壊)。admit = `ρ<1` sound 証明済の topology 変異のみ世代越え |
| **fail-closed admission** | `minimal_ga` の fail-closed admission(reject → resample_cap=50 → known-safe fallback `(decay=0.5, mix=0, gate_str=0)`, 閉形式 L=0.5<1)を流用 |
| **SDP rung の扱い** | 副 base Mamba(stable-by-construction)や n≤8 の対照でのみ「より緩い feasible set が navigable」を示す**補助**に使い、主張の主軸には載せない |
| **gate 選択基準** | BG10 教訓: **gate を soundness/coverage のみで選び navigability で選ばない**(gradient が gate-indifferent ゆえ、§6.2 条件 3) |
| **Z3/SMT** | orthogonal optional として封印(decorative。SYSTEMATIZATION §1.2: 証明の主役は収縮 certifier) |

---

## ⑥ 進化機構(ChangeOp の実構造手術化)

### 6.1 何を進化させるか

上記 recurrent adapter block の **離散トポロジー自由度**(branch 追加/削除、op 入替、memory horizon ±1、width ±1)。連続 `(decay, W)` は **gradient が担当**、進化は **gradient の解けない離散層**を担当する分業。

### 6.2 ChangeOp の実構造手術化(pivot 必須作業)

**⚠ 現 ChangeOp は実構造手術でない**(一次照合済): `changeop.py` の `apply_changeop`(line 160-193)は scalar `StateUpdateGene` の 3 float(`decay`/`mix`/`gate_str`)のみを操作し、`kernel_swap_mock`(line 21, 189-192)は「実 NN kernel 交換ではなく gate 構造の mock スイッチ」と明記。`CoupledNDGene` の `(decay, W)` n 次元構造には一切触れない。VSOA の全機構は新規実装を要する:

| move | 実装 | gate |
|---|---|---|
| **param shift**(既存) | 現 ChangeOp の decay/mix/gate_shift をそのまま | per-component cert_inf 再計算 |
| **`width_grow`** | W を `(n)→(n+1)` に **Net2Net** 流 function-preserving 拡張で初期化(拡張直後は元と同関数, arXiv:1511.05641) | 拡張後 Jacobian が box 上で `sup‖J‖∞<1` を破らないことを **stress 検証**(§⑩ 賭け 1) |
| **`branch_add`** | 並列 recurrent path 追加 | 追加後 per-block cert_inf を AND 合成 |
| **`op_swap`** | kernel 種別の*実*置換(mock でない) | 置換後 cert_inf 再計算 |

各 move は世代横断で certifier に問い合わせ、**変更後の per-block cert_inf が `ρ<1` を sound に証明できなければ fail-closed reject**。gate するのは「move が安定 certificate を破らないこと」**のみ**で、fitness(capability)で gate しない——適応は許すが発散・忘却は許さない **homeostatic constraint**。Net2Net 初期化で「構造を安全に大きくする」を function-preserving に担保し、cert_inf がその上で online 力学の収縮を保証する**二段構え**。

> ⚠ Net2Net × cert_inf 整合の構造的圧力(honest): `width_grow` で n が増えると `_t_min = 1 - tanh²(M)` の `M = Σ|W|` が単調増加 → `sech²(M)` が縮む → `t_min` が下がり achievable-t box が広がる → 拡張後に `sup‖J‖∞` が 1 を越えやすくなる。**この soundness が VSOA の存立条件**(§⑩ 賭け 1)。

---

## ⑦ 事前登録 existence-bet

### 7.1 主軸(guarantee, falsifiable, capability 封印, negative 可)

> **H-feasibility**: SmolLM2-135M に後付けした n≤16 verified adapter block に対し、per-component cert_inf fail-closed gate 付きの構造変更(width_grow/branch_add/op_swap を含む)online 適応 N 世代閉ループは、T4/30h budget の **X% 以内**で完走する。
>
> **H-stability(certified-stable rate)**: admit された**全件**が `ρ<1` certificate を持ち、無 gate baseline が `ρ_eff>1` へ越境する適応系列で gate 付きは `ρ_eff<1` を **0 *観測* false-admit** で維持する(HD-1 の ρ→1.95 越境の再現を実 LLM core で示す)。
>
> **H-drift / H-forgetting**: 同一適応予算で、gate 付きは online drift(出力ノルム発散)/ 保持タスク held-out CE 劣化が無 gate / STABLE 風経験 gate より片側 p<0.05 で小さい。

**封印**: capability(絶対 perplexity 改善・進化 > 勾配)は仮説に**入れない**(M3 戒め)。可塑性非劣化(無 gate と perplexity 有意差なし)のみ確認。

### 7.2 副線(terrain-design, capability-vs-artifact meta-gate)

> **H-EXISTS**: SmolLM2-135M の long-range memory block に、(a) behavioral-reach 欺瞞 corridor または (b) 構造化推論/riddle/shiritori 由来の離散多峰、を持つ事前登録 fitness family を構成したとき、`ρ<1` fail-closed gate 付き MAP-Elites(③)が同予算 gradient と random を fresh-seed held-out CE で honest_eval 4 条件 AND(`diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n_seeds≥15 ∧ |paired_sign_delta|≥0.147`)で上回る constellation が ≥1 個**存在する**。
>
> **capability-vs-artifact meta-gate(BG10)**: ③ が勝った constellation に **gradient-on-same-terrain で利得が消えないこと**を要求。消えれば `ARTIFACT`(navigability 最適化現象)判定 → guarantee 副線へ退避。消えなければ `EXISTS=True` を proper power(n80)で確定試行。

**verdict ∈ {EXISTS, NULL, ARTIFACT}**: capability を主張するのでなく**存否を決着**させる。EXISTS なら FullSense 唯一の capability 差別化、NULL/ARTIFACT なら「実 small-LLM 損失地形は単峰/navigability artifact」の **decisive negative が guarantee-niche を正当化**(どちらも publishable な honest 結論)。

### 7.3 negative 全面許容

- 全 family で ③ が gradient に並ぶ/負ける(M3 の full-LLM 再現)→ `EXISTS=False` 確定、guarantee 副線へ退避と事前宣言。
- `ρ<1` 強制で適応タスク性能が baseline より有意劣化(可塑性が殺される=M3/H2 が示唆する後者リスク)→「contraction gate が能力を殺さない」が反証され、guarantee の実用価値が縮小と honest に宣言。
- gate 付き・無 gate で忘却差が NULL(H2 系の系統的反証圧が再発)→ guarantee が忘却に効かないと開示。

---

## ⑧ Kaggle feasibility-first PoC

**最初の 1 本で「安さ × 形式保証」を潰す。** T4(16GB×2, 30h, resumable)に収めるか外挿で判定。

| 項目 | 内容 |
|---|---|
| **最大の計算リスク** | 「構造変更 1 回ごとの cert_inf 検証 × N 世代 × M block」が budget を食うか。**特に width_grow が n を成長させると cert コストが n² で増大**(judge B fatal flaw: 固定 n 前提と本案の構造成長が緊張) |
| **cert_inf コスト見込み** | per-block O(n²)・頂点列挙ゼロで n≤16 なら μ秒〜ms 級=forward/backward に対し無視可の**見込みだが要実測** |
| **PoC 合否判定** | Phase 0: SmolLM2-135M load + 数百 step fine-tune が T4 で回ることだけ確認 → Phase 1: 「変異 1 回 + cert_inf 1 回 + held-out CE 1 回」の wall-time/MB を実測し 30h に N 世代分が収まるか外挿。**width_grow 後の成長 n でも再計測**(固定 n の 1 回計測で過小評価しない) |
| **超過時の縮小** | base は SmolLM2-135M 固定、block n を 8 へ縮小 or family 数を削る。重い base 学習(1B scratch CPU)は明示放棄 |
| **resumable** | MAP-Elites archive / checkpoint で 30h 跨ぎに対応 |
| **GPU カーネル依存** | Mamba は副 base のみ(主経路を T4 純正に保つ) |

---

## ⑨ Phase 0/1/2 ロードマップ + decision gate

### Phase 0 — base 継承(リスクほぼ無)
- SmolLM2-135M を Kaggle Notebook で load → 数百 step fine-tune が T4 で回ることを確認。副 base Mamba-130M を control に。
- 「動かせる最小構造単位」= 1 個の verified adapter block(n≤16 の `(decay,W)`)を 1 個だけ特定。**トポロジー全探索はしない。**
- **instrument 校正**(副線準備): 構成した fitness family が本当に多峰/欺瞞かを決定論化(eval_noise を機械 eps へ)して valley_fraction で検証(deceptiveness-measure N/A の instrument 問題を最初に潰す。positive control=合成多峰、negative control=ESN 単峰)。
- **Decision gate 0**: load + 数百 step が T4 で回る → GO。回らねば縮小。

### Phase 1 — 最初の 1 部品の検証付き online 適応(feasibility 主役 + **存立条件検証**)
1. その block の Jacobian に対し per-component cert_inf(`_infnorm_sup<1`)で `ρ` 上界を安く計算する関数を実装し、固定構造で `ρ` が測れることを確認。
2. **ChangeOp を実構造手術へ拡張**(`width_grow`=Net2Net function-preserving / `branch_add` / `op_swap`)を **`width_grow` 1 種だけ**実装。
3. **存立条件 stress 検証(最優先)**: `width_grow` 1 回で拡張後 Jacobian が box 上で `sup‖J‖∞<1` を破らないことを stress 検証(0 false-admit を**成長操作下で**再確認)。**ここが通らなければ即 案 3/評価枠組みへ退避**(§⑩ 賭け 1)。
4. mutation × gate の 1 ループ:構造変異 → 変異後 cert_inf 再計算 → `ρ<1` で admit / さもなくば fail-closed reject の閉ループを 1 世代回す。
5. **feasibility 測定(合否判定)**: 変異 1 回 + cert 検証 1 回の秒/MB を実測し 30h に N 世代収まるか外挿(成長 n でも再計測)。
- **Decision gate 1**: (3) 存立条件 PASS **かつ** (5) feasibility PASS → Phase 2 へ。(3) FAIL → 即退避。

### Phase 2 — homeostasis 主張(guarantee 主軸)+ 差別化深掘り
- admit された変異のみ世代横断で積み、無 gate / STABLE 風経験 gate に対し (a) `ρ_eff<1` 維持 0 観測 false-admit、(b) 保持タスク忘却/drift の有意減を**事前登録検定**で示す。capability は二の次、guarantee メトリクスを主軸に報告。
- Mamba 副 base で gate 判別力(SmolLM2 で reject 発生 / Mamba で自明 PASS)を対照。
- **副線(任意・余力時)**: terrain-design family 上で MAP-Elites vs gradient vs random を honest_eval → capability-vs-artifact meta-gate で EXISTS/NULL/ARTIFACT を 1 つ確定。
- **Decision gate 2**: H-stability/H-drift PASS → guarantee 差別化として結実。副線 EXISTS → capability 差別化を追加主張。

---

## ⑩ make-or-break 賭け 1-4 への回答 + 撤退条件

| 賭け | SYSTEMATIZATION の定義 | 本計画の回答 | negative 時の退避 |
|---|---|---|---|
| **賭け 1(存立条件: width_grow×cert_inf soundness)** | 案の唯一の価値「0 false-admit」が未実装の width_grow×Net2Net×cert_inf 整合に依存。width_grow が box を広げ soundness が壊れやすい構造的圧力 | **Phase 1 step 3 を最優先**: width_grow 1 種を実装 → 拡張後 Jacobian が box 上で `sup‖J‖∞<1` を破らないことを stress 検証(0 false-admit を成長操作下で再確認) | **FAIL → 即 案 3(固定アーキの guarantee 副線=topology を成長させず param-shift + branch のみ)または評価枠組みへ退避を事前登録に明記** |
| **賭け 2(スケール: 2^n 壁)** | n=32 で 1099GB。vertex-free B2 も n=16 で cert_inf に収束 | **block を小 n(n≤16)に切る設計制約で 2^n 壁の*下*で動く**。cert_inf(O(n²)頂点フリー)を主力に。SDP rung は補助のみ | cert_inf 単独に縮退でも guarantee は成立(最弱だが sound)。賭け 4 と連動 |
| **賭け 3(transfer: tiny→実 LLM)** | 全実証が proxy/mock/小スケール(~0.5M)。実 LLM transfer 未検証 | SmolLM2-135M を主 base に選定(cert 成立帯 2M〜145M と重なる)。**「切り出した低次元 core が実 LLM で load-bearing か」は Stage-B tiny からの外挿=未検証と明示**(§⑪) | transfer 未達なら guarantee の実 LLM 妥当性が縮小と honest 開示。Stage-B 範囲に主張を限定 |
| **賭け 4(guarantee の scale 連結崩壊)** | 実 LLM 次元で cert_inf 単独に縮退すると「強い verifier が fitness 解放」L3 物語が検証不能=ladder 階梯価値が消える | **L3 物語を主軸に載せない**(SDP rung は補助)。主軸は「cert_inf 1 段で 0 false-admit を保つこと」のみ。階梯価値を主張しないことで賭け 4 を回避 | ladder 階梯が崩れても「安いが最弱な cert_inf 1 段」の guarantee は残る。忘却に効かねば案 3/評価枠組みへ |

**総合撤退条件**: Phase 1 step 3(存立条件)FAIL なら案全体崩壊 → **即 案 3(成長させない固定 topology の guarantee)または評価枠組み(Verified-Plasticity Eval)へ退避**。capability 副線が NULL/ARTIFACT でも guarantee 主軸は独立に PASS 可能なので、副線の失敗は案全体を崩さない。

---

## ⑪ honest 留保

1. **正味 novelty の狭さ**: STABLE(2510.16089)が「LLM 編集を stability budget で clip-or-reject」で gate 発想を既踏。差は「パラメータ編集→構造変更」「経験 budget→sound cert_inf」の **2 点のみ**=査読で「delta が小さい・既存の自明な組合せ」と評される最大リスク。CT-BaB(2411.18235, 固定アーキ)/ Net2Net(1511.05641, 関数保存止まりで online 安定 certificate でない)/ COMP(2508.08144, LLM 非対象・approximate)/ Jet-Nemotron(2508.15884, 無 certificate・200B token)も近接。**capability 目新しさはゼロと明記**。corpus の「空白象限」主張は内部判断=外部再現不能と honest 開示。

2. **transfer 未検証**: 「切り出した低次元 core が実 LLM で本当に load-bearing か」は Stage-B tiny-scale(~0.5M, 1 corpus, char-level, T4)からの外挿で、SmolLM2-135M への transfer は SYSTEMATIZATION が明示的に「未検証」とする領域(賭け 3)。Phase 1 feasibility が通っても guarantee の実 LLM 妥当性は別途残る。

3. **coupling 盲点リスク**: cert_inf の per-component AND 合成は、実 Transformer 内で block 間に residual/attention 経由の coupling がある場合に soundness が崩れうる。SYSTEMATIZATION §3.3 は「対角 scalar heuristic が 1267/3270 誤 admit、coupling-awareness が load-bearing」と明記。**block を小 n に切って独立扱いする設計はこの coupling 盲点を再導入するリスク**——per-block で閉じても block 間 coupling は未 certify。Phase 2 で block 間 coupling を含む安定性を別途検証する必要を明記。

4. **H2 系の系統的反証圧**: 「内的化/autonomy/safety 機構が優位を生む」仮説は SYSTEMATIZATION で 3 件とも NULL/not-supported(HD-1 H2 Holm p=0.058 / R-endo H2 Δ=−0.04 p=0.67 / viability autonomy NULL)。**忘却抑制(H-forgetting)が NULL に終わる確率は構造的に高い**(memory 軸 Δ≈+0.0134 極小の前例)。条件 4(memory horizon)が 4 本柱中最弱という確定事実と整合させ、事前登録で NULL を許容し、立たねば honest に開示する。

5. **soundness は機械証明でない**: 0 *観測* false-admit は「float `eigvalsh>0` の数値検査 + JSR oracle 片側有限長下界 + 独立 eigen 再検査」レベルであり、機械検証された定理ではない。確立は小スケール基質に限られる(SYSTEMATIZATION §0.1)。

6. **普及ファネルとして地味**: capability を捨てた分、派手な数値で人を惹く力がなく、SNS 拡散性(`project_f25_demo_polish` の採用ファネル先頭)としては弱い。guarantee-niche の純度 + feasibility 実証で勝負する。
