# Step C Verdict — 記憶タスクで欺瞞 corridor が自然発生し ③(適者生存・選択) が load-bearing になるか

> **状態**: 確定 (two-way 決着)。C1 = 実測済 / C2-C3 = `exp_c2c3_compare` 実測済 / C4 = MAP-E 非勝利のため moot。
> **結論**: **③ は実記憶タスク (この基質) で load-bearing でない** = CPU 撤退の精密化 (両方向決着・honest)。
> base_seed=20260530, n_seeds=15, strict gate = diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n≥15 ∧ |paired_sign_delta|≥0.147。

---

## 凡例 — 進化4要素 (Darwin/Mayr)

法律の甲乙丙のように番号で呼ぶ規約。平易な用語集 → [`YOUGO_平易版.md`](./YOUGO_平易版.md)。

- **①変異** (variation) — 設計を少しランダムに変える。
- **②遺伝** (heredity) — 親の設計が子に引き継がれる。
- **③適者生存・選択** (selection = 適応度の差による差し survival) — **本 verdict の主役。「③」はこの 3 番目。**
- **④過剰繁殖** (over-reproduction) — たくさん子を作る。

本書の「①」〜「④」はこの番号を指す (特に「③」= 適者生存・選択)。研究ログでは番号で略記し、用語集では言葉で書く ([`YOUGO_平易版.md`](./YOUGO_平易版.md) line 6-7)。

### かみ砕いた説明 (専門外向け)

設計探しは「霧の中で一番高い山を探す登山」です (高さ=設計の良さ)。普通の山登り (今より少し高い方へ進むだけ) は **ニセ頂上で止まる「だまし地形」** に弱く、そこでは「いろんなタイプの設計を捨てずに残し、谷を飛び石で渡る」やり方 (多様性探索 = 進化の③適者生存) が効くはずでした。今回それを **粘菌の迷路** にたとえると分かりやすい — 粘菌は迷路でも餌までの最短路を見つけますが、それは迷路に「行けば近づく」滑らかな手がかりがあるから。今回測った 2 つの記憶問題は、片方が **難しすぎて誰も登れない崖 (湖の小島)**、もう片方が **簡単すぎて誰でも登れる一つ山 (粘菌の迷路型)** で、**「③だけが渡れる、だましだが渡れる回廊」がこの基質では自然に現れませんでした**。だから「③は無力」ではなく「この問題・この基質では③の出番が無かった」というのが結論です。

---

## 2. 背景 — ③ がここまで「空転」してきた経緯と本 Step C の位置

本 verdict が決着させるのは **「実タスク本来の性質から欺瞞 corridor が自然発生し、そこで ③ が load-bearing になるか」** の一点。上流の確定事実を逐語参照しつつ、本 Step C はその CPU 手順の延長として実記憶タスクで両方向決着させる。

| 段階 | 確定した事実 |
|---|---|
| **監査** (`EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md`) | ①変異=成立 / ②遺伝=成立 / **③④=機構はあるが空転** (③④空転)。定数 fitness で改善=0、honest 再評価で GA は同予算 random を有意に上回らない。報告 best +0.29 は elitism 凍結 artifact。③失敗の主因は **landscape の平坦さ** (本番 CopyTask は SNR を上げても GA=random、勝つ時も tournament_k=1 で勝つ=勝因は③でなく elitism+変異の hill-climbing)。 |
| **手順4** (`STEP4_SELECTION_VERDICT.md` §0/§2) | **合成** deceptive corridor で MAP-Elites が 3 baseline 全勝 (大域到達 95% vs 0%, p=1.9e-6, δ=+1.00, 60 seed) → **③ が load-bearing になる状態の存在証明**。勝因は archive の stepping-stone ratchet (C4 確証)。ただし優位は **欺瞞 regime 限定** — dip を外した smooth corridor では MAP-E は RR-hillclimb / panmictic-GA に優位を失い、**smooth では③不要**。 |
| **手順6** (`STEP4_SELECTION_VERDICT.md` §7) | 実テキスト ESN+ridge readout proxy (next-char) の landscape は **滑らか/単峰**。MAP-E は 3 baseline に有意差なし or pure random にのみ勝つ → **③不要 (hill-climbing/coverage で十分)** の暫定結論。 |
| **本 Step C** | 合成では③が立ち (手順4 境界)、最近接の実テキスト proxy では立たなかった (手順6)。残る問い = **人工注入なし・標準的な記憶タスク本来の難しさだけで欺瞞 corridor が現れ、そこで③が load-bearing になるか**。両方向 (Yes=③実在 / No=CPU 撤退) のどちらでも決着とする (STEP_C_DESIGN §2/§6)。 |

---

## 3. C1-C4 結果

`exp_c1_landscape` (C1) / `exp_c2c3_compare` (C2/C3, `exp_c2c3_results.json`) / `exp_c4_ablation` (C4)。base_seed=20260530, n_seeds=15。

### 3.1 C1 — landscape 多峰性 (valley_fraction, is_multimodal = ≥0.2)

| 記憶タスク | valley_fraction | is_multimodal | 所見 |
|---|---|---|---|
| **delayed_parity** | **1.000** | **True (多峰)** | 全ペアの中点が谷。遺伝子空間で強く多峰。 |
| **flip_flop** | **0.939** | **True (多峰)** | ほぼ全ペアで谷。遺伝子空間で多峰。 |
| delayed_recall | 0.000 | False (滑らか) | 谷ペアなし=単峰/滑らか。C1 不成立 → **exp_c2c3 から除外**。 |

**=> `any_multimodal = True`** (parity / flip_flop)。C1 は parity / flip_flop で成立したが、**これは遺伝子空間の多峰性に過ぎず、③ が立つ十分条件ではない** (§4 でミスリードと判定)。

### 3.2 C2 / C3 — niching が baseline 全てに strict gate で勝つか (核心)

C1 で多峰が出た delayed_parity / flip_flop に対してのみ測定 (`exp_c2c3_results.json`)。

**delayed_parity** (mean R² 全 method ≈ 0.003-0.004 / reach-rate 全τで 0.0):

| MAP-E vs baseline | diff | 片側 Wilcoxon p | paired_sign_delta | passes |
|---|---|---|---|---|
| random | -0.0008 | 0.85 | -0.20 | **False** |
| random-restart hill-climb | -0.0007 | 0.83 | -0.20 | **False** |
| panmictic-GA | -0.0002 | 0.51 | +0.07 | **False** |

→ 全 method が **ほぼゼロ** (mean R² ≈ 0.003-0.004、reach-rate 全τで 0.0)。MAP-E は 3/3 で passes=False。C2 = baseline が大域に届かない (reach-rate 0.0) が、**MAP-E も届かない** = 全員が基質の床に張り付いた状態。

**flip_flop** (mean R² 全 method ≈ 0.945-0.953 / reach-rate τ=0.8 で全 1.0):

| MAP-E vs baseline | diff | 片側 Wilcoxon p | paired_sign_delta | passes |
|---|---|---|---|---|
| random | +0.0041 | 0.15 | +0.33 | **False** |
| random-restart hill-climb | +0.0036 | 0.26 | +0.07 | **False** |
| panmictic-GA | -0.0041 | 0.97 | -0.20 | **False** |

→ 全 method が **≈0.95 で頂上到達** (reach-rate τ=0.8 で全 1.0)。MAP-E は random に対して符号は正だが p=0.15 で有意でなく、panmictic-GA には負け、3/3 で passes=False。C2 = baseline が大域に届く (詰まらない) = 欺瞞トラップが機能していない。

**c3_all_pass: 両タスク False** (`exp_c2c3_results.json`)。

### 3.3 C4 — 勝因は探索量でなく diversity 維持か

`exp_c4_ablation` は flip_flop 完了時点で未完。ただし **C3 が両タスク✗ = MAP-E が勝っていない**ため、C4 (= 勝因帰属の ablation) は **moot / N/A**。勝因を問う前提 (MAP-E が勝っている) が成立しないので、C4 は結論に寄与しない。

---

## 4. 結論 — 両方向決着 (honest)

**③ は実記憶タスク (この reservoir+ridge 基質) で load-bearing でない。** = 監査 (③④空転) と手順6 (実テキスト proxy 滑らか) の系列を、**実記憶タスクで精密化した CPU 撤退**である。

両タスク C3 ✗ だが **棄却の理由は正反対**:

- **delayed_parity = 難しすぎて全員 ≈ 0** (基質の床 / **湖の小島**)。mean R² ≈ 0.003-0.004、reach-rate 全τ 0.0。誰も登れていないので MAP-E と baseline の差が分離不能 (diff ≈ ±0.0008, p ≥ 0.51)。これは **③ 以前に単細胞 (単一 reservoir) 基質が XOR (parity) を解けない**ことによる **confounded** な失敗 — 「③が効かない」のではなく「基質が床に張り付いて③以前で詰んでいる」(navigation frame §6b confound #3 基質ボトルネック)。
- **flip_flop = 簡単すぎて全員 ≈ 0.95** (滑らかな一つ山 / **粘菌の迷路型**)。reach-rate τ=0.8 で全 method 1.0。**でたらめ (random) でも解けてしまう**ので、谷を渡る ③ の追加価値が存在しない (C2 不成立=欺瞞トラップが機能しない)。手順4 の smooth corridor / 手順6 の実テキスト proxy と同型の「③不要」regime。

**=> ③ が効く『欺瞞的だが航行可能な回廊』(deceptive but navigable corridor) が、自然な記憶タスクにこの基質では現れなかった。** 手順4 で確認した「③は欺瞞 regime 限定で立つ」という境界、および手順6 の「実タスク proxy は滑らかで③不要」が、**実記憶タスクでも再確認された**。

**C1 多峰性 (valley_fraction) はミスリード。** parity=1.000 / flip_flop=0.939 という遺伝子空間の強い多峰性は、③ が立つことを意味しなかった。**遺伝子空間で多峰でも、川/湖判別器である C3 (= MAP-E vs random/RR/GA) が両タスクとも棄却**した (navigation frame §2: C1 は川/湖を区別しない、parity=1.000 はむしろ湖の小島寄りの兆候)。多峰 ≠ 航行可能。

---

## 5. Decision tree (STEP_C_DESIGN §2/§6 を逐語適用)

spec §2/§6 を尊重し C1-C4 全ゲートで判定する。

```
C1: any_multimodal?
 └─ Yes (parity / flip_flop で多峰 — 実測 True)
        │
        C3: 多峰タスクのいずれかで MAP-E が 3 baseline 全てに strict gate 勝利?
        ├─ Yes → 【③ 実在・load-bearing】 … 該当せず
        │
        └─ No  ← 本件 (両タスク c3_all_pass=False / 全 baseline passes=False)
              → 【③ 路線 CPU 撤退】
                 多峰でも③は hill-climbing から分離できず load-bearing でない。
                 negative result を「失敗でなく決着」として verdict 化
                 (STEP_C_DESIGN §2「撤退の条件: ... or C3 不成立」)。
```

- **C3 ✗ (両タスク)** → ③ 実在の条件 (C1-C4 全成立) を満たさない = **撤退**。特に delayed_parity / flip_flop とも RR-hillclimb に strict gate で勝てない (passes=False) = ③ を hill-climbing から分離できていない (手順4 §7b の教訓に準拠)。
- **C4 は MAP-E 非勝利のため moot** — 勝因帰属は MAP-E が勝った場合にのみ意味を持つ。本件は C3 で既に撤退が確定するため C4 は結論を左右しない。
- **GPU 投資は本 verdict では判断しない** (STEP_C_DESIGN §2)。撤退の根拠は「GPU = 実 LLM 損失地形の欺瞞性を測る賭け」に限定される前段となる。

---

## 6. Honest 留保 (`feedback_benchmark_honest_disclosure` 準拠)

positive bias (③実在に寄せたい誘惑) を排し、C3 不成立を素直に撤退と書く規律を最優先とする。

1. **(a) delayed_parity の C3✗ は基質ボトルネックで confounded — clean な③テストではない**: mean R² ≈ 0.003-0.004 / reach-rate 0.0 は「③が効かない」証拠ではなく、**③以前に単一 reservoir 基質が XOR (parity) を解けず床に張り付いた**結果 (navigation frame §6b #3)。この failure は「③無力」とは読めない。clean な③テストには parity を解ける基質 (多細胞=結合 reservoir 等) が必要 (§7 次実験)。

2. **(b) flip_flop は too-easy で C2 (欺瞞) 不成立**: 全 method ≈ 0.95 / reach-rate 1.0 = 欺瞞トラップが機能していない。C2 (hill-climbing が詰まる) が成立しないので、そもそも ③ の出番が無い regime。C3✗ は「③が負けた」でなく「③が要らなかった」。

3. **(c) reservoir+ridge proxy ≠ full LLM / CPU→GPU 分解 (機構 vs 前提)**: 基質は固定 reservoir 力学の gene 化 + per-gene ridge readout で、backprop full LLM とは別物 (navigation frame §6e)。本 verdict は **機構 (a)** = 「欺瞞構造+航行可能な流れを持つ landscape で③が立つか」を CPU で検証したものであり、**前提 (b)** = 「実 GPU/LLM 訓練 landscape が実際に欺瞞構造を持つか」は CPU では立証できない (GPU でしか試せない経験的問)。撤退は機構層の結果であって前提を否定するものではない。

4. **(d) 5 confound (navigation frame §6b)**: ① descriptor 衝突 (粗 2D BD が異質解を同セルに潰す) / ② lake-island (最適解が behavior でも孤立=航行不能地形) / ③ 基質ボトルネック (本件 parity に直撃) / ④ BD アライメント (現 BD が基質パラメタ寄りで課題難度勾配に未アライン疑い) / ⑤ wrong-tool (QD 自体が欺瞞領域で次善 = Boldi/Ding/Spector 2023 lexicase「Objectives Are All You Need」arXiv:2311.02283)。C3✗ をこれらと分離するまで「③が答え/無力」を断定不可。

5. **(e) ③ ablation は MAP-E vs random/RR/GA であり、より clean な MAP-E_randselect は未実施**: 本 verdict の③分離は MAP-E vs {random / RR-hillclimb / panmictic-GA} だが、navigation frame §7e が指す **MAP-E vs MAP-E_randselect** (archive-elite 選択を fresh random gene 引きに置換=②③を殺し①変異のみ残す) の方が③の最も clean な分離。後続実験の標準 ablation に採用予定 (本件では未走)。

6. **(f) 結論は『③は無力』でなく『この基質・タスク帯で出番無し』**: delayed_parity は基質が床 (confounded)、flip_flop は too-easy (③不要)、delayed_recall は滑らか。**③が効く欺瞞かつ航行可能な regime がこの基質の標準記憶タスク帯に現れなかった**だけであり、③ そのものの一般的無力を主張しない (手順4 で③の存在証明は済んでいる)。strict gate の min_effect=0.147 は Cliff's delta small-effect 境界を実務 cutoff として **流用** したもので、本統計 (paired_sign_delta) は教科書的 Cliff's delta ではない (意味づけを曖昧にしない)。

---

## 7. 次実験 (CPU 安価順)

撤退は「③ 路線の打ち止め」ではなく「この基質・タスク帯では出番無し」の決着なので、confound を潰す次の安価な CPU 実験が続く。

1. **梯子・段1 (多細胞 = 結合 reservoir)** — 留保 (a) を潰す最優先。単一 reservoir が parity を解けない (床) のが confound なので、**結合 reservoir で delayed_parity が解けるか** (基質の床を上げられるか) を測る。解けて初めて parity 上で③の clean なテストが可能になる。CPU 安価。
2. **E-A (多タスク分布)** — navigation frame §7e。単一固定タスクを window/seq_len/型混合の記憶タスク **分布** に拡張し、hold-out window への **汎化** で③寄与が出るか。flip_flop の too-easy (固定タスクは一発勝負で滑らかになりやすい) を分布化で崩す。前提3点 (難度の分布形状一致 / pseudo-multimodality 常時併走 / hold-out は外挿) を事前登録。CPU 安価。
3. **GPU (前段検定後)** — 上記 CPU 機構実験で③の load-bearing 条件を厚くした上で、**実 LLM fitness 地形が欺瞞構造を実際に持つか** (前提 b) を試す。これは CPU では立証不能な経験的問であり、GPU 投資は本 verdict ではなく機構検証の進捗に条件づける。

---

## 参照 doc (逐語引用元)

- `STEP_C_DESIGN_memory_task_deception.md` — 判定枠組み一次仕様 (§2 合格/撤退条件、§5 C1-C4、§6 判定、§7 無注入約束)。
- `STEP_C_VERDICT_SCAFFOLD.md` — 本 verdict の骨子 (凡例 / decision tree / honest 留保の先書き)。
- `STEP_C_NAVIGATION_FRAME.md` — 解釈枠 (C3=川/湖判別器、5 confound §6b、基質ボトルネック、CPU→GPU 分解 §6e、海パラダイム §8、§7e 環境テーゼ・MAP-E_randselect ablation)。
- `EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md` — 物差しと動機 (③④空転、+0.29 elitism artifact、landscape 平坦が主因)。
- `STEP4_SELECTION_VERDICT.md` — 核心一次証拠 (§0 合成 corridor で③存在証明、§3 smooth で消失、§7 実テキスト proxy 滑らかで③不要)。
- `YOUGO_平易版.md` — ①-④凡例 (line 6-7)、核心テーゼ (line 42-43)、これまでの結論 (line 124-128)。
- `exp_c1_landscape` / `exp_c2c3_compare` (`exp_c2c3_results.json`) / `exp_c4_ablation` — C1-C4 実測ソース (base_seed=20260530, n_seeds=15)。
```
