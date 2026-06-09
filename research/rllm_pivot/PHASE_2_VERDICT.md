# Phase 2 VERDICT — H-discriminative(枠組み妥当性)+ capability terrain-bet(EXISTS/NULL/ARTIFACT)

**作成**: 2026-06-09 / **$0/CPU** / seed=20260609 / 前提: `PHASE_1_VERDICT.md`(Decision gate 1 PASS)+ `EVOLVABLE_LLM_PLAN_2026_06_09.md`(主軸=Verified-Plasticity Evaluation Framework)
**規律**: honest-disclosure。capability(進化が勾配に勝つ)と guarantee(証明付き安定)を混同しない。NULL は失敗でなく確証的 negative=研究成果。
**実装**: `phase2_discriminative.py` / `phase2_capability_terrain.py` → 各 `*_results.json`。

---

## 0. 一行 verdict

**Decision gate 2 = 枠組み妥当性 PASS / capability = NULL_TIE(進化の優位は未実証)。** 評価枠組みは 4 method を soundness で**明確に判別**(無 gate=危険・STABLE 風経験 gate=危険・sound cert=安全・Mamba 風=安全)。一方 capability 副線は、多峰かつ識別力ある地形で、**進化(MAP-Elites, gated/ungated)と gradient/random が held-out で統計的に区別不能**(全方向 4 条件 AND 不成立、ME vs gradient mean_diff=+0.028/p=0.39/sign_delta=0、逆向きも非有意=純粋な引き分け、n=20)。→ **「進化が capability で勾配に勝つ」という EXISTS は支持されず**(M3 の負と整合)。ただしこれは **非有意の引き分け = capability 優位の未実証**であって「進化が勾配に劣る」proof でも powered な等価性 proof でもない(absence of evidence ≠ evidence of absence)。**戦略含意は同じ: capability は売りにできず、枠組みの価値は GUARANTEE 側に確定**(capability封印が data で正当化)。bonus: ρ<1 gate は held-out では可塑性を有意に殺さない(ただし train 側では archive 探索を制約、§3/§4)。

---

## 1. H-discriminative — 枠組み判別力(`phase2_discriminative_results.json`, North Star #3)

収縮〜発散を跨ぐ gene 集団(n=6, 95 発散 / 305 収縮)で 4 method の admit を真 ρ=empirical_rho と突合。

| method | admit率 | false-admit(発散を通した数) | 発散中の false-admit 率 | 収縮の棄却率 |
|---|---|---|---|---|
| **none**(無 gate=負の対照) | 1.000 | **95** | 1.000 | 0.000 |
| **stable_exp**(STABLE 風経験 gate) | 0.963 | **80** | **0.842** | 0.000 |
| **cert_inf**(sound) | 0.225 | **0** | 0.000 | 0.705 |
| **cert_two**(sound) | 0.360 | **0** | 0.000 | 0.528 |
| **cert_sdp**(sound) | 0.728 | **0** | 0.000 | **0.046** |

mamba_synth(stable-by-construction, 0 発散)集団: 全 method admit 1.000 / 0 false-admit(正の対照=安全 family を誤棄却しない)。

**確定知見**:
1. **判別力 PASS(H3a)**: false-admit 順序 = **none 95 > stable_exp 80 > sound certs 0**。枠組みは「危険 / 経験的だが危険 / sound」を soundness で明確に分離して測れる。
2. **★STABLE 風経験 gate は発散 gene の 84% を false-admit** — kernel は tanh で常時有界ゆえ有限ホライズン観測では「摂動忘却したように見える」が真 ρ≥1(echo-state property 失敗)。**sound certificate でないと見抜けない**=「ラングトンの蟻」の幻([[reference_article_idea_inventory]] §2)を経験 gate は見抜けない。**verified-plasticity の存在意義そのもの**。
3. **正の対照 PASS(H3b)**: stable-by-construction 集団で cert_sdp の収縮棄却率 0.000 = 安全 family を誤棄却しない。
4. **cert_sdp が sound かつ最 navigable**(0 false-admit・収縮棄却 4.6% のみ)= small-n per-component gate の第一候補(Phase 1 知見と一致)。

---

## 2. capability terrain-bet — EXISTS/NULL/ARTIFACT(`phase2_capability_terrain_results.json`, F12/BG10)

synthetic 多峰地形(K=6 basin max-of-Gaussian、behavior 空間)で、ρ<1-gate 付き/無し MAP-Elites・gradient(有限差分)・gradient_strong(restart64=meta-gate)・random を **同予算 B=2000 train 評価**で走らせ、**held-out fitness**(別入力)を n=20 seed で paired 比較。honest_eval 4 条件 AND(diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n_seeds≥15 ∧ |paired_sign_delta|≥0.147)。

**前提検証(H-multimodal)**: theta 空間 (20 次元) で F9 find_basins → cluster 数 = **40**。ただし n_starts=40 で 40 cluster = 高次元で hillclimb が収束しきらず全 start が別 cluster に落ちた可能性が高く、**「40 個の真の局所最適」の証拠ではない**(F9 instrument は 8 次元・merge_radius=0.4 で校正済、20 次元・radius=0.5 へは未校正)。**頑健に言えるのは「多峰(>1)」まで**。地形識別力: random held-out 平均 = **0.489**(0.05-0.95=天井/床でない、discriminating)= capability 前提(多峰 + 識別力)は成立。

**held-out 平均**: random 0.489 / gradient 0.507 / gradient_strong 0.549 / **mapelites 0.535** / mapelites_gate 0.513(全て ~0.49-0.55 に密集)。

| 比較 | mean_diff | Wilcoxon p | paired_sign_delta | 4 条件 AND |
|---|---|---|---|---|
| ME vs gradient | +0.028 | 0.392 | +0.000 | **False** |
| ME vs gradient_strong(meta-gate) | −0.014 | 0.594 | — | **False** |
| gradient vs ME | −0.028 | 0.622 | — | **False** |
| gate vs ungate(ρ<1 が可塑性を殺すか) | −0.022 | 0.608 | — | NS |

**VERDICT = NULL_TIE**: **進化(MAP-Elites)は、多峰で識別力ある地形でも、同予算の gradient/random を held-out で上回れない**(どの向きも 4 条件 AND 不成立=純粋な引き分け)。
- **「進化が capability で勾配に勝つ」(H-EXISTS)は支持されず** = M3 の負と整合。「最尤 NULL」(計画 §2.1)の予測通り。
- **ただし NULL_TIE は「非有意の引き分け」**: ME vs gradient は wins=losses=10 / sign_delta=0 / p=0.39 で、+0.028 の小差を検出する power は実質ゼロ。**これは「進化が勾配に劣る decisive NEGATIVE」でも、powered な等価性 proof でもない**(power 分析/MDE 未実施)。honest には「capability 優位の**未実証**」。コードの verdict ラベルも NULL_TIE(差なし)で NULL(勾配≥進化)とは別物。
- 4 条件 AND は保守的(sign_delta=0 で条件4が機構的に失敗)+ budget=2000 の収束十分性は未検証 → NULL が under-optimization 由来でない保証は弱い(honest)。
- meta-gate(BG10)は moot: ME が gradient を上回らないため ARTIFACT 判定の余地なし。なお gradient_strong(restart64)は予算配分上「深さでなく幅」(各 restart ~1 step)に退化しており「より強い勾配」ではない=meta-gate 前提は本予算では部分的に崩れている(結論不変だが honest 留保)。
- **bonus(North Star #6 副次)**: ρ<1 gate は **held-out では**可塑性を有意に殺さない(gate vs ungate diff=−0.022, p=0.61, NS)。**ただし train 側は mapelites=0.971 vs mapelites_gate=0.720 と 0.25 の差**= gate は archive 探索を顕著に制約しており、held-out が flat(全 method ~0.5)な regime ゆえ train 差が held-out に出ていないだけ。「可塑性を殺さない」は held-out 限定・capability flat 前提つきの弱い主張。

---

## 3. Decision gate 2 統合判定

| 項目 | 結果 | 判定 |
|---|---|---|
| **H-discriminative**(枠組み妥当性, North Star #3) | none 95 > stable_exp 80 > certs 0、正対照 0 棄却 | **PASS** |
| **capability 副線**(F12, EXISTS/NULL/ARTIFACT) | 多峰・識別力地形で進化 ≈ 勾配 ≈ random(全方向 AND 不成立) | **NULL_TIE**(capability 優位 未実証・非有意) |
| **gate 中立性**(North Star #6 副次) | held-out では gate は可塑性を有意に殺さない(train 側は制約あり) | gate-neutral(held-out 限定・弱支持) |

**→ 枠組みは妥当(method を soundness で判別できる)、capability 優位は data で未実証(NULL_TIE) = 価値は GUARANTEE に確定。** これは脆い単一機構に賭けない (b) 主軸選択の正しさそのもの: **機構(進化)が capability を生まなくても、「枠組みの妥当性 + 測定された capability 優位の不在(非有意)+ STABLE 風経験 gate の高 false-admit 危険性」が第一級 deliverable として残る。**

---

## 4. honest 留保

1. **synthetic 地形であり実 SmolLM2-CE 損失地形ではない**。「多峰が保証された地形ですら進化が勝てない」clean probe。実 LLM CE terrain(heavier follow-up)で覆る確率は低いが未検証(実地形は単峰の可能性すらあり、その場合 capability はさらに立たない)。
2. **高分散**: per-seed held-out は 0.00-0.97 と大きく振れ、運(初期化・seed)が支配的。これ自体が #25 monoculture(遺伝的浮動)/ ラングトンの蟻(見かけの構造=ノイズ)の再現。NULL は「差が無い」であって「全 method が優秀」ではない(全 method が random 同等)。
3. **gate 中立性は capability flat な regime での観測**: どの method も random を大きく超えないため「可塑性」自体が強く行使されていない。gate が可塑性を殺さないのは「殺すべき可塑性が元々乏しい」可能性を排除できない(honest)。
4. **gradient は有限差分**(解析勾配でない)。実 LLM の解析勾配ならより強い可能性=NULL を過大評価する方向だが、多峰での cold-start 勾配の弱さは現実的。
5. **paired_sign_delta = net-win-fraction**(教科書 Cliff's delta でない、計画 §⑬整合)。

---

## 5. 次セッション候補

- **実 SmolLM2-CE 地形での capability 再測定**(F12 の本番、実 LLM adapter CE)。NULL を実地形で確証 or 反証。
- **framework 性(F8)**: 3 plug-point(GeneCodec/Objective/VerifierBackend)拡張性のテスト化 + topology 多様化の汎化 load-bearing(B-G1)。
- **Mamba 固有安定性 正対照**(SSM Jacobian Lyapunov)で base-level 判別。
- **consumer story + 動きで魅せるデモ(F11, 確認必要=ユーザー明示判断)**: 無 gate ρ→1.95 発散 vs gate ρ<1 リアルタイム可視化。
- **普及メタ記事**: 「verified-plasticity = ラングトンの蟻の幻(見かけの安定/進化)を sound cert で見抜く」= honest disclosure の集大成(STABLE 84% 危険 + capability NULL を題材)。

---

## 6. 敵対的検証(2 実験並列、2026-06-09)

verdict の主張を 2 実験の `results.json` + `.py` に独立 agent で突合(workflow `phase2-verdict-adversarial-verify`)。**結果 = MAJOR 0 / 全 MINOR**。数値の mismatch はゼロ。検出された MINOR を本文に反映済 + 残りを以下に記録:
- **(反映済) framing 是正**: 「capability decisive NEGATIVE / NULL / proper power で確証」→「**NULL_TIE = 非有意の引き分け = capability 優位の未実証**」へ(§0/§2/§3)。data は ME vs gradient 完全引き分け(sign_delta=0)で「勾配が進化に勝つ」証拠ではない。
- **(反映済) 40 basin の過信**→「高次元 hillclimb 非収束アーティファクトの可能性、頑健には多峰(>1)まで」(§2)。
- **(反映済) gate 中立性の片落ち**→ train 側 0.25 差(archive 探索制約)を明記(§2/§3)。
- **(反映済) gradient_strong の退化**(restart64=幅優先で「強い勾配」でない)+ 4 条件 AND の保守性 + budget 収束未検証(§2)。
- **(残留 low) gated MAP-Elites の予算非対称**: cert_inf 受理チェック(resample 含む)は budget 非計上。「予算=fitness 評価回数」定義としては defensible だが、総計算量・有効探索試行は非対称。
- **(残留 low) STABLE 84% false-admit は設定依存**(EPS_FORGET=1e-2/T=64/K_PROBE=8 固定、感度未測定)。方向(STABLE 危険)は頑健だが「84%」を設定非依存の数値のように扱わない。
- **(残留 low) empirical_rho from-below** ゆえ STABLE の false-admit はむしろ過小評価寄り(=判別力 PASS を弱めない方向)。per-seed 分散(§4 留保2)は JSON 未保存(stdout のみ)で再現性ギャップ。

→ **検証後も Decision gate 2 の結論不変**: H-discriminative PASS / capability 優位 未実証(NULL_TIE)/ 価値 = GUARANTEE。指摘は全て framing と留保の精度向上で、機構的結論を覆すものは無かった(MAJOR 0)。

正本データ = `phase2_discriminative_results.json` / `phase2_capability_terrain_results.json` / 実装 = 各 `.py` / 検証 = workflow transcript。

---

## 7. Phase 2 残り完遂 (2026-06-09 続き) — 実 SmolLM2-CE capability / F8 framework性 / Mamba Lyapunov / demo

§5 の「次セッション候補」を本セッションで完遂。3 実験 + デモ。各々 honest-disclosure 規律 + 敵対的検証(§7.5)。

### 7.1 ★実 SmolLM2-CE 地形での capability 再測定 = **ARTIFACT+NEGATIVE**(`phase2_capability_realce.py` + `_results.json`)

§4 honest 留保 #1(「synthetic 地形であり実 SmolLM2-CE 損失地形でない」)の heavier follow-up。synthetic Gaussian を
**実 SmolLM2-135M hidden 由来の次クラスタ予測 CE 地形**(layer-15 hidden → train PCA top-32 部分空間 → per-seed
ランダム 32→n=6 射影 → centroid 最近傍 CE、K=6、cluster は train 文のみ fit=リークなし)へ置換。BUDGET=2000、n_seeds=20、
held-out 文(最適化中 未観測)で汎化 CE 比較。地形は discriminating(norm_score 0.178、floor でも ceiling でもない)・
多峰(F9 basin=40、ただし synthetic 同様 高次元 hillclimb 非収束 artifact の可能性で頑健には >1)。

| 比較 | mean_diff | Wilcoxon p | sign_delta | 4条件AND |
|---|---|---|---|---|
| **ME vs gradient (finite-diff)** | +0.0289 | 9.5e-7 | **+1.00 (20/0)** | **True** |
| ME vs gradient_strong (restart64 finite-diff) | +0.0272 | 9.5e-7 | +1.00 | True |
| **ME vs gradient_torch (★解析 exact 勾配 meta-gate)** | −0.0080 | 1.00 | −0.90 | **False** |
| **gradient_torch vs ME (逆向き)** | +0.0080 | 3.5e-4 | +0.90 (19/1) | **True** |
| ME vs random | +0.0191 | 9.5e-7 | +1.00 | True |
| gate vs ungate (ρ<1 が可塑性を殺すか) | −0.0285 | 1.00 | −1.00 (0/20) | — |

held-out 平均: **gradient_torch −1.446 (全 method 最良)** > mapelites −1.454 > random −1.473 > gradient_strong −1.481 >
gradient(fd) −1.483 ≈ mapelites_gate −1.483。

**★verdict = ARTIFACT+NEGATIVE**: MAP-Elites は finite-diff gradient を **20/20** で上回る(diff +0.029、一見 EXISTS)が、
**解析(torch Adam, exact gradient via autograd, 同予算=forward CE 評価回数)** が ME を **19/20 で逆に上回る**(diff +0.008, p=3.5e-4)。
→ **ME の勝ちは finite-diff gradient の弱さ(cold-start・dim+1 評価/step・予算内 ~95 step)の ARTIFACT**。強い勾配を与えると
**gradient > evolution = 実 LLM 地形でも capability NEGATIVE**(M3 + synthetic NULL_TIE と整合)。**→ capability 封印・guarantee 主軸が
実地形 data で正当化**。
- ★**honest-disclosure の真価**([[feedback_benchmark_honest_disclosure]]): strong-gradient meta-gate が無ければ「進化が実地形で
  capability に勝つ(20/20!)」と **false-positive を誤結論**していた。「勝った気になる前に内訳を疑う」が機能。
- ★**synthetic NULL_TIE への含意**: §2 の synthetic 実験も **同じ弱い finite-diff gradient** を使用 → そこでの「引き分け(NULL_TIE)」は
  実は **強い勾配なら gradient 優位**で、capability negative は **synthetic でも過小評価**だった可能性が高い(実地形 torch meta-gate が露呈)。
- **gate コスト(honest)**: 実地形では ρ<1 gate が held-out fitness を実測 −0.028 制約(ME+gate −1.483 < ME −1.454, 0/20)=
  synthetic の flat regime と違い、ここでは gate が可塑性を測定可能に削る。ただし進化に capability 優位が無い(gradient 勝利)ため
  capability 結論には影響せず。
- honest 留保: 実 vocab の full-softmax CE でなく hidden-クラスタ CE proxy(n≤6 readout で full-vocab は degenerate ゆえ) /
  torch backward を free 計上(=実 LLM の backprop 勾配と同扱い、budget=forward CE 評価回数)/ centroid β=クラスタ間分離スケール
  (事前原理的・結果非依存) / 多峰性 instrument は高次元 hillclimb artifact 留保継承。

### 7.2 framework 性 (F8) = (b) 3 plug-point swap **PASS** / (a) 汎化 load-bearing **NULL**(`phase2_framework_f8.py` + `_results.json` + `test_framework_f8.py`)

North Star #4。src `llcore.evolution.minimal_ga.evolve()` を **無改変**で使用(`git diff src/` 空、固定 protocol adapter のみ additive)。
- **(b) 3 plug-point 拡張性 = PASS**: GeneCodec(`CoupledNDGeneCodec` n=2/3/4)・Objective(`RotationNDObjective` period/radius 違い)・
  VerifierBackend(`make_nd_verifier` none/inf_norm/two_norm/sdp)を各々 **1 オブジェクト差替**で同一 evolve ループに載せ替え動作。
  final 集団 admit 数(決定論 seed)= none 24 / inf_norm 9 / two_norm 9 / **sdp 17**。**保証される関係は per-gene subset
  (`cert_two(g)⟹cert_sdp(g)` の fast-path / `cert_inf(g)⟹cert_sdp(g)`、3000 gene で 0 違反=pytest が assert する核)**であり、
  集団ごとの count-level ladder は per-sample 偶発(別 gate で進化した別集団の count 比較ゆえ)。**pytest 17 passed**。
  honest: verifier gate は fitness-wrapper proxy(src の scalar gate path は codec 併用で fail-loud ゆえ、plan §⑦ 方針=verifier backend が担う)・明記済。
- **(a) 汎化 load-bearing = NULL**: 構造多様 archive(MAP-Elites over topology descriptor, cells≈56)vs param-shift-only baseline を
  task family(RotationNDObjective 8 task、train 5/held-out 3)・同予算 720・20 seed で比較。held-out diff=+0.011, p=0.55, sign_delta=0
  → 4条件AND=False = **多様性が汎化に load-bearing という証拠は立たず(第一級 NULL)**。二次観察(AND 不算入): paramshift が gap 大
  (0.200 vs 0.162)で過学習寄りだが p=0.215 NS。honest: synthetic task family(実 SmolLM2-CE でない)、proxy。
- **含意**: framework の価値は (b) 拡張性 + 判別力(§1, §7.3)に在り、(a) 多様性→汎化 は立たず。over-claim しない。

### 7.3 Mamba 固有安定性 正対照 (SSM Lyapunov) = base-level 判別 **PASS**(`phase2_mamba_lyapunov.py` + `_results.json`)

Phase 1 で defer された正対照を実機完遂。Mamba-130M を load し全 24 層の SSM `A = -exp(A_log)` を抽出。
- **全 24 層・589,824 (channel,state) entry で A 実部 < 0**(frac=1.0)。離散 `Ā=exp(Δ·A)` の `|対角|≤1`、Lyapunov
  `λ_max=max(Δ·A) ≤ 0`。**global λ_max(代表 Δ=softplus(dt_bias))= −1.80e-9 ≤ 0**、Δ スイープ [1e-4..1e2] 全域で全層 **厳密 λ<0**
  (A<0 ゆえ Δ>0 なら符号 Δ 非依存)。= **stable-by-construction の自明 PASS**(arXiv:2406.00209 と一致)。
- **SmolLM2-135M 対比**: model_type=llama、SSM 再帰キー **0 件**(self_attn + mlp のみ)= 固有安定 certificate の概念が base に
  構造的に不在 → 安定性は後付け adapter+gate に依存(Phase 0 harness の load-bearing と整合)。
- **→ base-level 判別 PASS**: 枠組みは「stable-by-construction な base(Mamba 自明 PASS)」と「gate を要する base(SmolLM2)」を
  base レベルで明確分離 = North Star #3 を base レベルで補強。
- honest: Δ 入力依存(代表値+スイープで近似)/ 対角 A 前提(この hf checkpoint で成立)/ ZOH 離散化 / **SSM 状態再帰の安定性のみ**
  (conv1d/SiLU/MLP の full Lipschitz でない)/ 代表 Δ で marginal な channel は Δ≈0 由来(A≈0 でない)=非厳密(≤0)を over-claim せず。

### 7.4 動きで魅せるデモ (F11 技術成果物) (`phase2_demo_gate_discrimination.py` + `.svg` + `.json`)

★**honest 再設計**: 計画当初の「無 gate ρ→1.95 出力ノルム発散」アニメは **この tanh 基質では物理的に起きない** —
状態は tanh で常時有界・しかも **単一軌道の摂動感度すら ρ≥1 で発散しない**(実測: ρ≈2.9 の発散 gene でも実軌道は tanh 飽和+
方向ミスアラインで感度減衰)。= 状態ノルム監視も有限忘却テストも単一軌道感度も **ρ≥1 を見逃す** = §1 の STABLE 84% false-admit の
根本。**不安定を見抜けるのは box-sup の sound certificate のみ**。よってデモは「経験 vs certificate の判別力差」が唯一 honest な形。
- **headline バーチャート SVG**(集団 data-backed, §1 の phase2_discriminative 再可視化): 発散の false-admit 率 = 無 gate 100% /
  STABLE 風 84% / sound cert(inf/two/sdp)0%。収縮の過剰棄却率 = cert_inf 70% / cert_two 53% / **cert_sdp 4.6%**(=sound かつ最 navigable)。
- **単一 gene evidence**: ρ≈2.9 発散 gene の実軌道感度 1→2e-14(経験は『安全』と誤認)だが certificate box-sup σ_max=4.87>1 で reject。
- matplotlib 非依存(FullSense 宣言的 SVG 方針)、SMIL アニメ、静止フレーム完成形。consumer story/市場判断はユーザー明示判断に deferred。

### 7.5 敵対的検証 (workflow `phase2-rest-adversarial-verify`, 3 独立 skeptic 並列)

3 実験を独立 skeptic agent が code+data で再導出・反証(数値再計算 + 実機再走)。**結果 = all_verdicts_hold=True / MAJOR 0 / 全 MINOR**。
- **A 独立確認(load-bearing)**: 全 paired stats を json と 10 桁一致で再計算 + **実 SmolLM2 load して 3 seed 独立再走 → 全 seed で torch_beats_ME=True かつ ME_beats_finite_grad=True を決定論的に再現** = ARTIFACT+NEGATIVE 確証。beta はクラスタ幾何から最適化前に導出=p-hack でないと確認。
- **MINOR(反映済 / 結論不変)**:
  - (A) torch は token-pooled CE、numpy terrain は sentence-averaged CE で ~1e-4 差(全 optimizer は同一 sentence-avg `heldout()` で採点ゆえ apples-to-apples、1e-4 ≪ 0.008 の torch-ME gap、むしろ torch を僅かに handicap=ARTIFACT 結論は conservative)。
  - (A) `self.scale=std(allX)` は全 20 文で計算(mu/PCA/centers は train-only)=軽微リーク。**単一 global scalar を全 optimizer・train/held-out に同一適用ゆえ符号不変**(verifier 確認)。honest 改善余地として記録(本 run の results.json は本実装、結論非影響)。
  - (A) budget-parity は forward-eval 数の同一であって有効更新ステップ数は非対称(torch 2000 Adam 更新 vs finite-diff ~95 更新)= verdict が ARTIFACT 源として明示済。
  - (B) verifier-axis seed が hash() で非再現だった → **決定論 seed に修正済 + 再走**(admit none=24/inf=9/two=9/sdp=17、pytest 17 passed、(b) PASS / (a) NULL 不変)。**保証される関係は per-gene subset(two⟹sdp / inf⟹sdp、3000 gene で 0 違反)** であり count-level ladder は per-sample 偶発(§7.2 訂正反映)。
  - (B) verifier gate は fitness-wrapper proxy(disclosed)・synthetic task family(disclosed)。
  - (C) 正対照は **parameterization の自明性**(A=-exp(A_log)<0 は任意 Mamba checkpoint で成立=学習でなくパラメタライズを検定)= verdict が "TRIVIALLY satisfied/stable-by-construction" と明示済。"marginal" は strict 閾値 count 0 と prose "λ~0" で scale が異なるだけ(矛盾でない)。SSM 状態再帰のみ(full Lipschitz でない)=disclosed。
- **→ 検証後も §7.6 の結論不変(MAJOR 0)**: 指摘は全て reproducibility/framing/留保精度で、機構的結論を覆すものは無し。

### 7.6 Phase 2 完遂 統合判定

| 軸 | 結果 | 判定 |
|---|---|---|
| H-discriminative (§1) | none 95 > stable_exp 80 > certs 0 | **PASS** |
| **capability (synthetic §2 + 実 SmolLM2-CE §7.1)** | synthetic NULL_TIE / 実地形 ARTIFACT+NEGATIVE(強い勾配 > 進化) | **capability NEGATIVE**(実地形で確定・honest-disclosure で false-positive 排除) |
| framework 性 F8 (§7.2) | (b) 3 plug-point swap PASS / (a) 汎化 load-bearing NULL | (b) PASS / (a) NULL |
| base-level 判別 Mamba (§7.3) | 全層固有安定 自明 PASS / SmolLM2 gate 必須 | **PASS** |
| gate 中立性 (§7.1) | 実地形では gate が held-out −0.028 制約(synthetic は flat) | gate-cost あり(capability 結論に非影響) |

**→ Phase 2 完遂。枠組み妥当(判別力 PASS × base-level PASS × 3 plug-point PASS)、capability は実地形でも NEGATIVE(強い勾配が進化に勝つ、
honest-disclosure が finite-diff の false-positive を排除)、価値 = GUARANTEE で確定。** 「脆い単一機構に賭けない (b) 主軸」の正しさが
3 実験で多面的に裏づけられた。

正本データ(§7) = `phase2_capability_realce_results.json` / `phase2_framework_f8_results.json` / `phase2_mamba_lyapunov_results.json` /
`phase2_demo_gate_discrimination.json` / 実装 = 各 `.py` / 検証 = workflow `phase2-rest-adversarial-verify` transcript。
