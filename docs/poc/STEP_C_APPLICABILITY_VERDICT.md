# Step C 適用条件 Verdict — ③ (behavioral niching) が立つ欺瞞性の閾値を特性化した

> **凡例 — 進化4要素**: ①変異 / ②遺伝 / **③適者生存・選択** (本書の主役) / ④過剰繁殖。
> 2026-05-31。research/step_c_applicability/ 隔離・src 非変更・push 未。
> 設計 = `STEP_C_APPLICABILITY_DESIGN.md`。一次情報 = `STEP4_SELECTION_VERDICT.md` / `E_A_VERDICT.md` /
> `STEP_C_VERDICT.md` (再導出せず継承)。
> base_seed=20260530 (robustness: 777 / 31337 でも一致)。n_seeds=20, n_evals=6000, honest_n=30。
> strict gate = `diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n≥15 ∧ |paired_sign_delta|≥0.147`。

---

## 0. 結論 (3 行)

1. **欺瞞性 (dip 深さ d) を smooth(d=0)→deceptive(d=1) に sweep し、③の strict-gate 優位を d の関数として
   特性化した。** これで過去の binary negative が**特性化曲線**になった。
2. **③が厳格 load-bearing (MAP-E が 3 baseline 全勝) になる閾値 d* = 0.16** (3 base_seed で一致)。
   transition は **比較的 sharp** — d=0.10 で非 LB、d=0.13 で部分 LB (RR のみ撃破)、d=0.16 で全 baseline
   撃破。**幅 ~0.03-0.06 の急な遷移**で、真の不連続ではないが gradual でもない。
3. **実 task (E-A multitask / Step C memory) は定性的に d < d* (smooth 側) に落ちる** — baseline が大域に
   到達し MAP-E が baseline に勝てない観測パターンが本 sweep の d<d* と同型。**ただし numeric な d 値は
   付けられない** (写像が無い、§4 で明記)。過去の negative は「③無力」でなく「実 task の欺瞞性が
   閾値未満だった」と整合的に説明できる。

---

## 1. 特性化曲線 (canonical sweep, base_seed=20260530)

各 d で MAP-Elites を honest 再評価し、3 baseline (RR-hillclimb / panmictic-GA / random) に対し完全 strict
gate を適用。advantage = MAP-E mean − best baseline mean。reach = honest fitness>0.8 (大域峰 proxy) の割合。

| d (dip深さ) | MAP-E | RR-hc (reach) | panmictic-GA | random | advantage | baseline撃破 | 状態 |
|---|---|---|---|---|---|---|---|
| 0.00 | 0.996 | **0.998 (1.00)** | **0.998** | 0.853 | −0.002 | 1/3 | not-LB |
| 0.05 | 0.996 | **0.998 (1.00)** | **0.998** | 0.829 | −0.002 | 1/3 | not-LB |
| 0.10 | 0.998 | **0.998 (1.00)** | **0.998** | 0.803 | −0.001 | 1/3 | not-LB |
| **0.13** | 0.994 | 0.886 (0.65) | **0.998** | 0.787 | −0.004 | **2/3** | **partial-LB** |
| **0.16** | 0.993 | 0.756 (0.25) | 0.710 | 0.772 | **+0.221** | **3/3** | **STRICT-LB** |
| 0.20 | 0.996 | 0.720 (0.15) | 0.687 | 0.749 | +0.247 | 3/3 | STRICT-LB |
| 0.30 | 0.995 | 0.692 (0.10) | 0.659 | 0.705 | +0.290 | 3/3 | STRICT-LB |
| 0.40 | 0.997 | 0.682 (0.10) | 0.652 | 0.676 | +0.315 | 3/3 | STRICT-LB |
| 0.50 | 0.993 | 0.646 (0.00) | 0.646 | 0.658 | +0.335 | 3/3 | STRICT-LB |
| 0.60 | 0.995 | 0.642 (0.00) | 0.642 | 0.648 | +0.347 | 3/3 | STRICT-LB |
| 0.70 | 0.996 | 0.638 (0.00) | 0.638 | 0.641 | +0.355 | 3/3 | STRICT-LB |
| 0.85 | 0.997 | 0.652 (0.05) | 0.634 | 0.635 | +0.345 | 3/3 | STRICT-LB |
| 1.00 | 0.997 | 0.648 (0.05) | 0.631 | 0.631 | +0.349 | 3/3 | STRICT-LB |

(太字 = その帯で大域に到達している/閾値をまたぐ行。完全 gate 値は `exp_knob_sweep_results.json`。)

### 読み方

- **MAP-E は全 d で大域到達 (≈0.99, reach=1.00)** — dip の深さに関わらず archive ratchet で谷を渡る。
  これは exp4 の C4 (勝因 = stepping-stone ratchet) が dip 深さに頑健であることの確認。
- **baseline は d とともに崩れる**: d≤0.10 では RR/GA とも大域到達 (reach 1.0, mean≈0.998)。d=0.13 で
  RR が部分的に罠落ち (reach 0.65)、d≥0.16 で RR/GA とも大域到達率が急落 (reach≤0.25)。
- **advantage の跳躍**: d=0.10 の −0.001 → d=0.16 の +0.221。**adv はほぼ 0 を維持し、閾値で階段状に
  正へ跳ぶ** (gradual な ramp ではない)。

## 2. 閾値と transition の性質

- **厳格閾値 d* = 0.16** (③ が 3 baseline 全勝 = exp4 の③成立定義)。**3 base_seed (20260530/777/31337)
  全てで d*=0.16** — seed 非依存。
- **部分閾値 = 0.13** (climbing baseline RR-hillclimb に初勝利するが panmictic-GA はまだ越えられる)。
- **transition は sharp (急)** だが **真の不連続ではない**: 幅 ~0.03-0.06 (d=0.10〜0.16) に
  「not-LB → partial-LB → strict-LB」の 2 段遷移がある。**baseline ごとに dip 耐性が異なる**ことが
  この幅の正体 — panmictic-GA は RR-hillclimb より少し深い dip まで (population の確率的越境で) 渡れる
  ので、RR が先に (d≈0.13) 落ち、GA が後 (d≈0.16) に落ちる。**「閾値帯」(threshold band) d∈[0.13,0.16]**
  と呼ぶのが最も honest。
- **random の扱い (Codex pair-review で訂正)**: 当初「random は d 非依存で自明に beaten」と書いたが
  **これは誤り**。実測では random の reach は d とともに下がり (d=0: 1.00 → d=0.10: 0.50 → d=0.16: 0.25)、
  **d≥0.16 では climbing baseline (RR/GA) が dip で崩れて random を下回るため、random が最強 baseline になる**
  (`best_baseline_name=random`)。`load_bearing` 判定は **3 baseline 全勝**を要求する = MAP-E は各 d で
  最強 baseline (高 d では random) を strict gate で上回る必要があり、**むしろ保守的**。よって random の
  扱いを誤っても閾値判定は健全 (3 全勝を課すので自明勝利による水増しは構造的に起きない)。ただし
  「random は意味ある baseline でない」という当初の narrative は撤回する。d≤0.10 で beaten=1/3 (random のみ)
  なのは「smooth 側では MAP-E が climbing baseline に勝てない」ことの裏返しで、③不在の正しい指標。

## 3. honest-disclosure チェック (`feedback_benchmark_honest_disclosure`)

「変にきれいな結果は内訳を疑う」に従い、自分の結果を疑った内訳:

1. **d=0 で本当に smooth か (knob 設計の検証)**: 初期版は「谷を平床 (flat floor) で埋める」設計で、
   d=0 でも MAP-E が baseline に勝ってしまった (adv=+0.13)。原因は **平床=勾配 0 で hill-climb の登坂
   信号が消える** = 浅い谷でも弱い罠になっていた。これを **正勾配 ramp 基線**に修正 (設計 §3-4) し、
   d=0 で RR/GA が reach=1.00 (大域到達)・MAP-E が僅かに負け (adv=−0.002) を確認 = exp5 の「smooth で
   優位消失」を再現。**この修正をしなければ閾値が偽って低く出ていた** (捏造を 1 段で回避した内訳)。
2. **MAP-E が全 d で ≈0.99 なのは出来すぎでは**: いいえ。MAP-E の archive ratchet は dip を「新規 cell」
   として保持し downhill を跨ぐので、深い dip でも大域に届く (exp4 C4 と同型機序)。MAP-E 側が一定なのは
   **baseline 側だけが d で崩れる**という非対称が本質で、それが特性化を成立させている (片側だけが動く)。
3. **random の自明勝利の隔離**: §2 の通り random は全 d で beaten。これを load-bearing の根拠に使うと
   d=0 ですら「1/3 撃破」で③有りに見える誤読を生む。**strict 判定を 3 baseline 全勝に固定**し、
   閾値報告も climbing baseline ベースにして自明勝利を除外した。
4. **gate の効果量統計の留保**: strict gate の `paired_sign_delta` は honest_eval の通り **教科書的
   Cliff's delta ではない** (paired 符号バランス効果量)。min_effect=0.147 は Cliff small-effect 境界を
   実務 cutoff として流用したもの。意味づけを曖昧にしない (一次情報の留保を継承)。
5. **baseline の強さ**: RR-hillclimb は固定 sigma・非悪化受理のみの素朴 (1+1)-ES、panmictic-GA も
   tuned ではない (STEP4 verdict §4 の留保を継承)。**lexicase 等の objective-based 強 baseline は未検定**
   (STEP_C_VERDICT §6d の wrong-tool 交絡)。よって本 d* は「③ (QD) が *この baseline 群* を上回り始める
   dip 深さ」であって「QD が最適手法になる dip 深さ」ではない。tuned optimizer なら d* は右にずれうる
   (= ③が立つには本測定よりさらに深い dip が要る可能性)。**本 d* は楽観側 (③に有利側) の推定**と明記。
6. **合成 landscape の限界**: 全て behavior=mean の 1D 合成 corridor・低ノイズ (σ=0.008)・CPU toy。
   実 LLM fitness の次元/ノイズ/欺瞞構造とは別物。本特性化は **機構層** (「dip 深さがこの閾値を超えると
   ③が立つ」) の証明であって、**前提層** (「実 task の欺瞞性が d* を超えるか」) は CPU では立証不能
   (STEP_C_VERDICT §6c の機構/前提分解を継承)。
7. **knob は exp4/exp5 の "厳密内挿" ではない (Codex pair-review で訂正)**: 本 landscape は全 d で
   `max(local, glob, ramp*(1-dip))` を使う。exp4 endpoint は `max(local, glob)` のみ、exp5 endpoint は
   別の広い Gaussian (`smooth_eval`) なので、**d=0/d=1 は exp5/exp4 の eval 関数そのものではない**。
   正確には「exp4/exp5 に**着想を得た** ramp-with-dip の新 toy family」で、d=0 = 単調 smooth control
   (RR/GA が大域到達・③優位なし を実測再現)、d=1 = 深い dip の deceptive corridor。よって d*=0.16 は
   **この新 family 内の閾値**であって「exp4↔exp5 間の純粋な 1-parameter 閾値」ではない。主張をこの水準に限定する。
8. **random narrative の訂正 + robustness 証跡**: §2 の通り random は d 非依存でなく、当初 narrative を撤回
   (Codex Med)。また「777/31337 で d* 一致」は当初本文主張のみだったが、`exp_knob_sweep_results.json` の
   `robustness_other_base_seeds` に閾値近傍 reduced sweep の実測 (両 seed とも d*_strict=0.16) を**成果物として保存**した。
   なお `dip_center_corridor` (旧 `valley_floor` 0.6×(1-d) は計算誤り→中央 ramp 高 0.8×(1-d) に訂正) も results に記録。

## 4. 実 task の honest 配置 — この軸のどこに落ちるか

**最重要かつ最も慎重に書くべき部分。** 過去の negative (E-A multitask / Step C memory) はこの欺瞞性軸の
どこに落ちるのか。

### 厳密な numeric 配置は **できない** (明示)

本 knob `d` は behavior=mean の合成 corridor 上の dip 深さという**合成専用の座標**。実 task は
behavior 記述子も fitness も別物 (E-A = ESN reservoir パラメータ × 多タスク汎化 R² / Step C = 記憶タスク
R²) で、それらを同じ `d` スケールに射影する厳密な写像は **存在しない**。したがって「E-A は d=0.07」の
ような numeric 配置は**捏造になるので行わない**。

### 定性的配置は可能 — 実 task は d < d* (smooth 側) と同型

numeric は無理だが、**観測パターンの同型性**で定性配置はできる。本 sweep の各 regime の「指紋」:

| regime | baseline reach | MAP-E vs baseline | 観測指紋 |
|---|---|---|---|
| d < d* (smooth) | climbing baseline が大域到達 (reach≈1) | MAP-E は baseline に勝てない (adv≈0 or 負) | **baseline が詰まらず MAP-E 優位なし** |
| d ≥ d* (deceptive) | climbing baseline が罠落ち (reach↓) | MAP-E が全 baseline 撃破 (adv≫0, δ→+1) | baseline が詰まり MAP-E のみ大域 |

過去の実 task の観測 (一次情報から):
- **E-A multitask** (`E_A_VERDICT.md`): 全 method ≈0.62-0.70 で天井近傍、**panmictic-GA が MAP-E を
  僅かに上回り** (0.702>0.682)、MAP-E は random にも有意差なし。= **baseline が詰まらず MAP-E 優位なし**
  → 本 sweep の **d < d* (smooth) の指紋と一致**。
- **Step C memory** (`STEP_C_VERDICT.md`): delayed_parity=床 (全員≈0)、flip_flop=天井 (全員≈0.95)。
  flip_flop は baseline が大域に到達 (reach τ=0.8 で全 1.0) し MAP-E 優位なし → **d < d* 側の指紋**。
  delayed_parity は**床 (誰も登れない)** で本 sweep の corridor とは別機序 (基質ボトルネック)、この軸に
  乗らない (配置不能と明記)。
- **手順6 実テキスト proxy** (`STEP4_SELECTION_VERDICT.md §7`): 滑らか/単峰、MAP-E は baseline に
  有意差なし → **d < d* 側**。

→ **定性結論 (状況証拠・断定でない): ③をクリーンに検定できた実 task (E-A multitask / flip_flop /
手順6 proxy) の観測指紋は、いずれも本特性化軸の d < d* (smooth 側) と整合的**。これは過去の honest
negative を説明する **作業仮説**を与える: **「実 task が『③無力』だった」のではなく「実 task の (自然な)
欺瞞性が閾値に届かなかった」可能性が、観測と矛盾しない** (numeric 配置は §4.3 の通り未実証なので断定はしない)。
delayed_parity は床効果で本軸に乗らない (別の confound) ので配置から除外する。

### 配置の信頼度 (overclaim 回避)

この定性配置は **観測指紋の同型性**に基づく**状況証拠**であり、実 task の欺瞞性を直接測定したものでは
ない。「実 task の欺瞞性 < d*」を**断定**するには実 task の landscape を本 knob と同じ枠組み (behavior
記述子上の dip 深さ) で実測する必要があり、それは未実施。**現時点の主張は「実 task の観測は d<d* と
矛盾しない (consistent with)」であって「実 task の d < d* である (proven)」ではない。**

## 5. verdict — 特性化は成功した (positive な characterization)

設計 §5 の「成功」定義を満たした:
- advantage(d) は d≈0 近傍でほぼ 0 (or 僅か負) → **d*=0.16 を境に階段状に正へ跳び**、d≥0.16 で
  strict-gate load-bearing が d=1.0 まで安定継続。**単調な閾値構造**が出た。
- transition は **sharp だが幅 ~0.03-0.06 の閾値帯** (RR が d≈0.13、GA が d≈0.16 で落ちる 2 段遷移)。
  真の不連続でなく「baseline ごとに dip 耐性が異なる帯」として特性化できた = **部分成功以上の clean な
  特性化**。
- §2 反証条件 (a)(b)(c) いずれも該当せず (全 d で③無し でも 全 d で③有り でもなく、閾値が明確に定義可能)。

**これは「③ は人工的に狭い regime でしか効かない」型の弱い結果ではない** — d∈[0.16, 1.0] という
**広い deceptive 帯で③は安定 load-bearing** (advantage が d とともにむしろ増大)。狭いのは「smooth 側の
③不要帯」(d∈[0, 0.13]) の方。**③ が立つには「ある程度以上深い dip が要る」が、深い側は頑健**、という
のが特性化の核。

ただし §3-4 の留保 (lexicase 未検定で d* は楽観側 / 合成 toy / 実 task は状況証拠でしか配置できない) に
より、**「実 AI 設計探索で③がペイする」とは依然言えない** — 本特性化が確定したのは **機構層の閾値**で
あって、**実 task の欺瞞性が d* を超えるかは未証明** (GPU でしか測れない前提層, E_A_VERDICT §次(b) と整合)。

## 6. 次アクション候補 (ユーザー判断)

1. **二次 knob の追加特性化** (CPU 安価): misalignment (behavior↔fitness 勾配のずれ) / corridor
   narrowness を第 2 軸にして 2D 欺瞞性マップを描く → ③の load-bearing 領域を面で特性化。
2. **lexicase baseline の追加** (STEP_C_VERDICT §6d / E_A_VERDICT 残課題): wrong-tool 交絡を排除し
   d* が右にずれるか測る (③が QD として次善でないかの検定)。
3. **実 task の欺瞞性を本枠組みで実測**: E-A/記憶タスクの landscape を behavior 記述子上の dip 深さとして
   定量化し、§4 の定性配置を numeric に格上げ (状況証拠→実測)。これが「実 task < d*」を断定する唯一の道。
4. **GPU 前提検定** (E_A_VERDICT §次(b)): 実 LLM 損失地形の欺瞞性測定。本 verdict の機構層 d*=0.16 を
   「実 task がこれを超えるか」の判定基準として持ち込む。

## 7. 成果物

- 設計: `docs/poc/STEP_C_APPLICABILITY_DESIGN.md`
- 実験: `research/step_c_applicability/exp_knob_sweep.py` (selection_lab read-only 再利用 / 完全 strict
  gate 再実装 / ea_lab.py の SeedSequence+CRN seed 設計踏襲)
- 結果: `research/step_c_applicability/exp_knob_sweep_results.json` (13 levels + 2 閾値 + 完全 gate 値)
- 本 verdict: `docs/poc/STEP_C_APPLICABILITY_VERDICT.md`
- robustness: base_seed 20260530 / 777 / 31337 で d*=0.16 一致 (再現確認済)

## 参照

- `STEP4_SELECTION_VERDICT.md` (exp4 deceptive / exp5 smooth = 本 knob の端点, ③存在証明と境界)
- `E_A_VERDICT.md` (多タスク汎化 honest negative = §4 で d<d* 配置)
- `STEP_C_VERDICT.md` (記憶タスク N/A・床/天井, §6 honest 留保群を継承)
- `research/step4_selection/selection_lab.py` / `research/ea_multitask/ea_lab.py` (read-only 再利用)
- `src/llcore/evolution/honest_eval.py` (strict gate / honest_reevaluate)
- `[[feedback_benchmark_honest_disclosure]]` / `[[feedback_codex_pair_review_for_llcore]]` /
  `[[project_llcore_init_2026_05_29]]`
