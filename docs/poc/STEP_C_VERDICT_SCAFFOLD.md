# Step C Verdict (SCAFFOLD) — 記憶タスクで欺瞞 corridor が自然発生し ③(適者生存・選択) が load-bearing になるか

> **状態**: SCAFFOLD (未完)。C1 のみ実測済。C2/C3/C4 は実行中。**現時点で結論を断定しない**。
> 数値が揃った時点で `<<PENDING>>` を埋め、decision tree に従い結論を一意に確定する。

---

## 凡例 — 進化4要素 (Darwin/Mayr)

法律の甲乙丙のように番号で呼ぶ規約。平易な用語集 → [`YOUGO_平易版.md`](./YOUGO_平易版.md)。

- **①変異** (variation) — 設計を少しランダムに変える。
- **②遺伝** (heredity) — 親の設計が子に引き継がれる。
- **③適者生存・選択** (selection = 適応度の差による差し survival) — **本 verdict の主役。「③」はこの 3 番目。**
- **④過剰繁殖** (over-reproduction) — たくさん子を作る。

本書の「①」〜「④」はこの番号を指す (特に「③」= 適者生存・選択)。
研究ログでは番号で略記し、用語集では言葉で書く ([`YOUGO_平易版.md`](./YOUGO_平易版.md) line 6-7)。

---

## 1. 背景 — ③ がここまで「空転」してきた経緯

本 verdict が決着させるのは **「実タスク本来の性質から欺瞞 corridor が自然発生し、そこで ③ が load-bearing になるか」** の一点。それ以外 (①②成立 / ③④空転 / landscape 平坦が主因 / readout 修正だけでは不足 / 真の unlock は空間拡張+niching) は上流で確定済みの事実列として逐語参照する。

| 段階 | doc | 確定した事実 |
|---|---|---|
| 監査 | `EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md` §1 | ①変異=成立 / ②遺伝=成立 / **③生存・繁殖の差=「機構はあるが空転」** / ④過剰繁殖=「③依存で空転」。定数 fitness で改善=0.0000 (選択は配線済) だが honest 再評価で GA は同予算 random を有意に上回らない (GA−RAND=−0.011, 5/10 勝, Wilcoxon p=0.77)。 |
| 監査 §2-1 | 同上 | 報告 best +0.29 は **elitism 凍結 artifact** (best 0.473 → fresh-seed 再評価 0.183 に崩落)。「best 単調上昇=進化成立」は不成立 → 主指標は集団 mean / fresh-seed 再評価。 |
| 監査 §7b | 同上 | ③失敗の主因は **landscape の平坦さ**。clean 構造 landscape では GA 圧勝 (単峰 GA−RAND=+0.122, p<1e-4, Cliff δ=+0.97) だが、本番 CopyTask は SNR を 0.38→3.51 に上げても GA=random (律速は上位20遺伝子の真の spread=0.0007 の平坦プラトー)。**かつ GA が勝つ時も tournament_k=1 (選択差なし) で勝つ → 勝因は ③ でなく elitism+変異の hill-climbing。③ そのものは未分離。** |
| 手順4 | `STEP4_SELECTION_VERDICT.md` §0/§2 | **合成**欺瞞 corridor で MAP-Elites (behavioral niching) が 3 baseline {random / panmictic-GA / random-restart hill-climbing} 全勝 (大域到達 95% vs 0%, p=1.9e-6, δ=+1.00, 60 seed) → **③ が load-bearing になる状態が実在する存在証明**。勝因は archive の stepping-stone ratchet (init_batch を 600→30 に削っても 100% 到達 = C4 確証)。 |
| 手順4 §3/§0-2 | 同上 | ただし優位は **欺瞞 regime 限定**。dip を外した smooth corridor では MAP-Elites は RR-hillclimb (p=0.29)・panmictic-GA に優位を失い pure random にのみ勝つ。 |
| 手順6 | `STEP4_SELECTION_VERDICT.md` §7 | 実テキスト ESN+ridge readout proxy (next-char) の landscape は **滑らか/単峰** で欺瞞 corridor が自然出現せず。exp7 (A) 3-param ESN は 3 baseline 全てに有意差なし、(B) 40-dim per-neuron leak は RR/panmictic に負け pure random にのみ勝つ → **③不要 (hill-climbing/coverage で十分)** の暫定結論。 |
| 結論 | 上流総括 | 「機構健全 → GPU で③が立つ」は論理が一段飛んでいる。GPU 投資は CPU 手順で「効くと分かる」まで保留 (条件付き)。**本 Step C はこの CPU 手順の延長 — 実タスク(記憶タスク)由来の欺瞞性を両方向決着する位置づけ。** |

**したがって本 verdict の問い (1 点に絞り込み済)**: 合成地形では③が立った (手順4)。最近接の実テキスト proxy では立たなかった (手順6)。では **人工注入なし・標準的な記憶タスク本来の難しさだけで欺瞞 corridor が現れ、そこで③が load-bearing になるか**。Yes なら「③ は実在 (人工注入なし)」、全タスク No なら「③ 路線は CPU で打ち止め」。

---

## 2. C1-C4 診断基準 (STEP_C_DESIGN §5、手順4 falsifiable 基準を流用)

| 基準 | 内容 | 測り方 (実コードで確認済) |
|---|---|---|
| **C1** | landscape が多峰 (分離した良解 peak が複数) | `landscape_map.multimodality_report`: random-restart hill-climb を `n_restarts` 回し収束点を集め、各ペアの中点を再評価。中点 fitness < min(両端) − 5%マージン なら「谷」。`valley_fraction = 谷ペア数 / 全ペア数`。**`is_multimodal = valley_fraction >= 0.2`**。 |
| **C2** | hill-climbing が局所最適に詰まる | RR-hillclimb / panmictic-GA / random の大域到達率が低い。(定性 — 定量閾値は doc に未明記、判定根拠を明示する) |
| **C3** | niching が baseline に有意勝利 | MAP-Elites が **3 baseline 全て**に強化版 strict gate (`strict_compare` = honest_eval `passes`) で勝利。 |
| **C4** | 勝因が探索量でなく diversity 維持 | init_batch ablation で coverage でなく archive ratchet が勝因と確証。(定性 — 定量閾値は doc に未明記、方向性のみ) |

**strict gate (passes) 完全基準** (`src/llcore/evolution/honest_eval.py` + `research/.../strict_compare.py` で実コード一致確認済):

```
passes = diff > 0  ∧  片側 Wilcoxon p < alpha(=0.05)  ∧  n_seeds >= min_seeds(=15)  ∧  |paired_sign_delta| >= min_effect(=0.147)
```

- 片側 Wilcoxon は `alternative='greater'` (H1: 進化 > random)。両側でなく片側 (仮説が一方向)。
- `paired_sign_delta = (#正 − #負) / n_seeds`、範囲 [-1,1]。**教科書的 Cliff's delta ではない** (paired 設計の符号バランス効果量。旧名 `cliff_delta` から改名)。閾値 0.147 は Cliff's delta small-effect 境界を実務 cutoff として**流用**したものであり、この意味づけを verdict で曖昧にしない。

---

## 3. C1 結果 (実測済) — 多峰性は parity / flip-flop で自然発生

`exp_c1_landscape.py` (base_seed=20260530, n_restarts=12, n_evals=400, sigma=0.15):

| 記憶タスク | valley_fraction | is_multimodal (≥0.2) | 所見 |
|---|---|---|---|
| **delayed_parity** | **1.000** | **True** | 全ペアの中点が谷 = 強く多峰。標準タスク定義のみ (人工注入なし) で欺瞞地形が自然発生。 |
| **flip_flop** | **0.939** | **True** | ほぼ全ペアで谷 = 多峰。set/reset 記憶スイッチが局所最適トラップを自然形成。 |
| **delayed_recall** | **0.000** | **False (滑らか)** | 谷ペアなし = 単峰/滑らか。このタスクでは③不要 (hill-climbing で十分) の候補。 |

**=> `any_multimodal = True`。** delayed_parity / flip_flop は **人工注入なし**で C1 が成立した。これは手順6 の実テキスト proxy (滑らか/単峰) とは異なり、**記憶タスクという問題本来の性質から欺瞞 corridor が自然発生した**ことを意味する (STEP_C_DESIGN §7 の無注入約束: 難易度を上げただけで多峰が現れたことをもって「自然発生」を主張、peak は手で配置していない)。

**C1 段階の honest 留保**: C1 ✓ は「欺瞞地形が存在しうる」ことを示すに過ぎない。③ が実際に **load-bearing** (= 多様性維持を持つ MAP-Elites だけが谷を渡れて baseline は詰まる) かは C2/C3/C4 が決める。多峰でも baseline が偶然届くなら③は不要。**C1 単独では結論しない。**

---

## 4. C2 / C3 / C4 結果

> **<<PENDING: `exp_c2c3_compare.py` / `exp_c4_ablation.py` 実行中 (別プロセスで CPU 実験走行中)。`exp_c2c3c4_run.log` に進捗。>>**
> C1 で多峰が出た **delayed_parity / flip_flop** に対してのみ C2/C3/C4 を測定する (delayed_recall は C1 不成立のため③不要側で打ち止め)。

### 4.1 C2 — hill-climbing は局所最適に詰まるか

**<<PENDING>>** RR-hillclimb / panmictic-GA / random の大域到達率を記録する。

- **詰まった場合 (大域到達率が低い)**: C2 ✓。欺瞞地形が baseline を実際にトラップしている = ③ が立つ前提が整う。手順4 exp4 の正準状態 (baseline 0% 到達) と同型。
- **詰まらなかった場合 (baseline が大域に届く)**: C2 ✗。C1 で多峰でも baseline が谷を偶然渡れるなら③は不要。手順4 exp1 (低次元+高予算で random が coverage 勝ち) と同型のリスク。**定量閾値が doc 未明記のため、判定根拠 (到達率の具体値と「低い」の線引き) を明示する。**

### 4.2 C3 — niching が baseline 全てに strict gate で勝つか (核心)

**<<PENDING>>** MAP-Elites vs {random / panmictic-GA / random-restart hill-climbing} の `strict_compare` 結果 (各 baseline ごとに diff / 片側 Wilcoxon p / paired_sign_delta / n_seeds / passes)。

- **3 baseline 全て passes=True の場合**: **C3 ✓**。これが③ load-bearing の決定打。strict gate (diff>0 ∧ 片側 p<0.05 ∧ n≥15 ∧ |delta|≥0.147) を **3/3 で**満たして初めて C3 成立とする (1 つでも passes=False なら C3 ✗)。
- **いずれかの baseline で passes=False の場合**: **C3 ✗**。MAP-Elites が一部 baseline に勝てない = niching の追加価値が分離できない。手順4 §7b の教訓「GA が勝っても tournament_k=1 で勝つなら勝因は③でなく hill-climbing」に倣い、**特に RR-hillclimb に strict gate で勝てなければ ③ は hill-climbing から分離できていない**と判定する。
- **honest 規律**: 集団 mean / fresh-seed 再評価が主指標 (監査 §2-1 の +0.29 elitism artifact 崩落事例より)。報告 best の単調上昇は信用しない。

### 4.3 C4 — 勝因は探索量 (coverage) でなく diversity 維持 (ratchet) か

**<<PENDING>>** init_batch ablation (例 600→30) で MAP-Elites の到達率が保たれるか。

- **ratchet が勝因の場合 (init_batch 削減でも到達率維持)**: C4 ✓ 方向。手順4 §1 と同型 (init_batch 30 でも 100% 到達, pure random は 6000 点でも 0%) = 勝因は archive stepping-stone ratchet。
- **coverage が勝因の場合 (init_batch 削減で到達率が崩れる)**: C4 ✗ 方向。勝因が初期サンプル量 = ③ でなく数撃ちゃ当たる。手順4 exp1 と同型。
- **honest 規律**: C4 は定量閾値が doc 未明記 + init_batch と ratchet 段数の confound あり。**方向性 (どちらが勝因か) の解釈のみ**を述べ、厳密な分離は主張しない (§6 留保参照)。

---

## 5. 結論

> **<<PENDING: C2/C3/C4 の数値未取得。下記 decision tree により数値が入れば結論が一意に確定する。現時点で断定しない。>>**

### Decision tree (STEP_C_DESIGN §6 / §2 を逐語適用)

```
C1: any_multimodal?
 ├─ No  (全タスクで C1 不成立 = 全部滑らか)
 │     → 【③ 路線 CPU 撤退】negative result を verdict 化。
 │        (本件は C1 = True のためこの枝には入らない)
 │
 └─ Yes (parity / flip_flop で多峰 — 確定済)
        │
        C3: 多峰タスクのいずれかで MAP-Elites が 3 baseline 全てに strict gate 勝利?
        ├─ Yes (3/3 passes=True, かつ C2 でその baseline が局所最適に詰まっている)
        │     → 【③ 実在・load-bearing】
        │        実タスク本来の性質から欺瞞 corridor が現れ ③ が立つことを
        │        人工注入なしで実証 (= STEP_C_DESIGN §2「③ 実在と認める条件」C1-C4 全成立)。
        │        C4 は勝因が ratchet (diversity 維持) 方向であることを補強材料として併記。
        │
        └─ No (どこかの baseline で passes=False / 特に RR-hillclimb に勝てない)
              → 【③ 路線 CPU 撤退】
                 多峰でも③は hill-climbing から分離できず load-bearing でない。
                 negative result を「失敗でなく決着」として verdict 化
                 (STEP_C_DESIGN §2「撤退の条件: ... or C3 不成立」)。
```

**C4 の位置づけ**: C4 は結論を反転させる主基準ではなく、C3 ✓ 時に「勝因が ratchet (diversity 維持) か coverage (探索量) か」の**方向性**を補強するのみ。confound (init_batch と ratchet 段数) があるため断定はしない。

**GPU 投資判断**: 本 verdict では GPU を判断しない (STEP_C_DESIGN §2)。撤退の場合「GPU = 実 LLM 損失地形の欺瞞性を測る賭け」に限定される根拠の前段とする。③ 実在の場合でも、実 LLM fitness が欺瞞 corridor 構造を持つかは別途証明を要する (監査 §7b)。

---

## 6. Honest 留保 (先書き — verdict 完成時も保持)

`feedback_benchmark_honest_disclosure` に従い、positive bias (③実在に寄せたい誘惑) を排し、C3 不成立を素直に撤退と書く規律を最優先とする。

1. **proxy であって full LLM ではない**: 基質は reservoir のダイナミクスを gene 化 + per-gene ridge readout (固定力学 + 学習読み出し)。backprop で学習する full LLM とは別。記憶タスクで C1-C4 が全成立しても、**実 LLM 損失地形が欺瞞的である保証ではない** (proxy の限界、監査 §6 / 手順4 §7 と同じ外挿留保)。全実験は CPU・toy・低ノイズ。

2. **難易度スイープ範囲の限定**: 「自然発生」の主張は **タスク本来の難しさ (delay 長・窓幅) の範囲のスイープのみ**で多峰が現れたことに基づく (STEP_C_DESIGN §7)。地形を恣意的にいじって欺瞞を捏造していないが、難易度を上げれば何でも多峰化しうるので、「標準形タスクの妥当な難易度範囲」を逸脱していないかを明示する。

3. **seed 依存 / short-run drift**: 監査 §6 で「短 run では genetic drift が control を上振れさせる」「lens 1 の決定的分離は報告より弱い」と確認済。C3 判定は複数 base_seed で robustness を確認し、short-run の drift 上振れに注意する (手順4 は 3 seed × 20 = 60 seed で seed 非依存を確認済 = 同水準を要求)。

4. **C4 の confound (方向性のみ)**: C4 は init_batch ablation で測るが、**init_batch (初期サンプル量) と ratchet 段数 (archive が刻む stepping-stone の数) が交絡**する。doc に定量閾値の明記もない。よって C4 は「勝因が ratchet 方向か coverage 方向か」の**方向性解釈に限定**し、「diversity 維持が単独原因」とは断定しない。

5. **C2 / C4 の定性基準**: C2「大域到達率が低い」・C4「archive ratchet が勝因」には C1 (valley_fraction≥0.2) / C3 (strict gate) ほどの formalize された定量閾値が doc にない。verdict では**これら定性判定の根拠 (具体的到達率・ablation 前後の到達率差) を必ず明示**し、線引きを読者が追えるようにする。

6. **③ と niching/hill-climbing の分離限界**: 手順4 §4 が留保する通り、MAP-Elites の勝因は厳密には「behavioral diversity 維持が fitness valley 跨ぎを可能にする」であって「差し survival 率の差」を単独分離したものではない (niching と selection の複合)。本 verdict も「③ (広義の選択+多様性)」として扱い、純粋な tournament 選択の単独効果を主張しない。手順4 §7b の「GA が勝っても tournament_k=1 で勝つ」事例を引き、③ を hill-climbing から分離する負担 (= RR-hillclimb に strict gate で勝つこと) を C3 が担う設計であることを明記する。

7. **strict gate の効果量の意味づけ**: min_effect=0.147 は honest_eval docstring 上「Cliff's delta small-effect 境界を実務 cutoff として**流用** (paired_sign_delta に適用)」であり、本統計 (`_paired_sign_delta`) は教科書的 Cliff's delta ではない。verdict で効果量を引く際はこの流用を曖昧にしない。

8. **C1 は spec が予言した通り出たが結論ではない**: 本 doc 群 (STEP_C_DESIGN) は spec/設計であり、C1 多峰性が記憶タスクで実際に現れるかは経験的問題だった。C1 = True (parity/flip_flop) は予言が当たったが、③ load-bearing の verdict は C2/C3/C4 のデータで裏取りするまで確定しない (鵜呑み禁止)。

---

## 参照 doc (逐語引用元)

- `STEP_C_DESIGN_memory_task_deception.md` — 本 verdict の判定枠組み一次仕様 (§2 合格/撤退条件、§5 C1-C4、§6 判定、§7 無注入約束)。
- `EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md` — 物差しと動機 (§1 ③④空転、§2-1 +0.29 elitism artifact、§7b landscape 平坦が主因 / ③ 未分離)。
- `STEP4_SELECTION_VERDICT.md` — 核心一次証拠 (§0 合成 corridor で③存在証明、§3 smooth では消失、§7 実テキスト proxy 滑らかで③不要)。
- `YOUGO_平易版.md` — ①-④凡例の正規ソース (line 6-7)、核心テーゼ (line 42-43)、これまでの結論 (line 124-128)。
- `src/llcore/evolution/honest_eval.py` / `research/step_c_memory_tasks/strict_compare.py` — strict gate 実コード (passes 基準一致確認済)。
- `research/step_c_memory_tasks/landscape_map.py` — C1 多峰性診断 (is_multimodal = valley_fraction ≥ 0.2)。
