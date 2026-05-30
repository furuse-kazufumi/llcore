# E-A 設計 — 多タスク分布での③(選択圧/分離)寄与を hold-out 汎化で検定

> 2026-05-30。research/ea_multitask/ 隔離予定、src 非変更。
> 上流 = Step C verdict「次(CPU安価2本柱)(2) E-A=多タスク分布」/ 梯子段1 VERDICT「次の優先経路」。

## 問いと背景

梯子段1 (LADDER1_VERDICT) は **parity 経路が degree-5 床で詰む**ことを確定した
(degree-2 readout は degree-2 単項式のみ可解、positive control で原理確定)。よって③
(分離が選択を可能にする = niching/selection 圧) を **parity の床に縛られない別の土俵**で検定する。

Step C verdict §6(g) の無限後退打ち切り方針: 「parity に固執しない」。
本フェーズは **多タスク分布 (regime 構造を持つタスク族) での hold-out 汎化** を土俵に、
③ が「汎化する解」を生むかを ablation で測る。

**核心の問い**: タスク分布上で進化させたとき、**behavioral niching (③ の分離) は、未知 regime への
汎化 (hold-out generalization) を、選択圧を殺した対照より有意に高めるか。**

step4 verdict の教訓を内蔵: ③ が load-bearing だったのは **欺瞞的 corridor のみ**で、
**滑らか/単峰では③不要** (exp5/exp7)。よって E-A も **まず多タスク landscape が欺瞞的/多峰か**を
確認し (C-gen2)、滑らかなら「③不要 (honest negative)」と正直に結論づける両方向決着の設計とする。

## 基質選定 (parity 床を回避)

Step C 実測より、**単一 leaky reservoir + ridge で解ける**タスクを採用 (床問題を回避):
- **FlipFlop** (set/reset 保持): Step C で全 method ≈0.95 飽和 = 可解。
- **DelayedRecall** (cue 再生): Step C で R²≈0.999 = 可解。
- **DelayedParity は除外** (degree-5 床、梯子段1 で確定)。

基質 = `LeakyDelayLineReservoir` (step_c/reservoir.py) または `DeepReservoir`
(ladder1/multi_reservoir.py)。gene/bounds/held-out eval 契約は既存流用。

## タスク分布 (regime 構造 = 「風/天候」メタファー)

単一タスクでなく **regime パラメータで連続変化するタスク族**を分布とする。候補軸:
- **FlipFlop の `pulse_prob` ∈ {0.1, 0.2, 0.3, 0.4}** (パルス密度 = 「天候の荒れ具合」)。
- **FlipFlop の `seq_len` ∈ {20, 30, 40}** (記憶保持要求長 = 「遅延の長さ」)。
- 必要なら **DelayedRecall の delay** と混合。

**TaskMixture** (新規, generate 側で `rng.choice(regimes, p=weights)`): 1 評価内で
複数 regime の系列を混ぜて出題。汎化 = 学習 regime と **disjoint な hold-out regime** で評価。

- **train regimes** = {pulse_prob ∈ {0.1, 0.3}} など (例)
- **test (hold-out) regimes** = {pulse_prob ∈ {0.2, 0.4}} (学習に未使用)
- データリーク厳禁: train/test regime を draw レベルで完全分離。

## ③ ablation (selection だけを殺す)

調査済の差し替え点 = `selection_lab.py:100` の親選択。

ユーザー spec (Step C verdict): 「③ablation = MAP-E vs MAP-E_randselect = **②③殺し①変異のみ**」。
= fitness による生存選択 (③) と niche elite 維持 (②) を殺し、行動ビニング上の **変異 (①) だけ**残す。

| 群 | behavior grid | parent | placement (生存) | 残すもの |
|---|---|---|---|---|
| **MAP-E (full)** | あり | archive elite から選択 | **fitness ゲート** (f>cell 既存で置換=ratchet) | ①②③ |
| **MAP-E_randselect (②③殺し)** | あり (grid 維持) | **bounds から random** | **無条件配置** (fitness 無視、最後の child を保持) | ① 変異のみ |
| **panmictic-GA** | なし (単一集団) | tournament 選択 | elitism + tournament | ①③ (③選択あり、②niching なし) |
| **random search** | なし | なし | best のみ保持 | 床/規模対照 |

判定の切り分け (因子分解):
- **MAP-E > MAP-E_randselect** → **②③ (niching + 選択) が ① 変異だけより寄与**。
- **MAP-E > panmictic-GA** → **② (niching) が ③ 選択だけ (panmictic) より上乗せ**。
- **MAP-E > random** → 全体が無探索に勝つ。
③ の load-bearing は MAP-E が **MAP-E_randselect と panmictic-GA の両方**に strict gate で勝つことで主張。

## fitness と主指標

- **fitness (進化中)** = train regimes 上の held-out ridge R² (既存 make_eval_once、train/eval 別 draw)。
- **主指標 = hold-out 汎化 R²** = 進化後の best gene を **test regimes (未学習)** で fresh-seed
  honest 再評価 (honest_reevaluate, n_trials≥16)。
  - MAP-E は archive 全体の汎化も測る (best-cell / archive-ensemble の両方を報告)。

## Falsifiable ゲート (両方向決着)

| ゲート | 内容 | 通過基準 |
|---|---|---|
| **C-gen1 (基質床)** | 基質が in-distribution の個別 regime を解けるか | train regime で best R² > 0.5 (床でないこと) |
| **C-gen2 (欺瞞性/汎化ギャップ)** | 多タスク landscape が多峰 or 汎化ギャップを持つか | train→test の汎化ギャップ有意 or landscape 多峰 (step4 C1 流用)。無ければ「滑らか→③不要」を honest 結論 |
| **C-gen3 (③ test)** | MAP-E の hold-out 汎化 > MAP-E_randselect | strict_compare passes (n_seeds≥15, 片側 p<0.05, \|δ\|≥0.147) |
| **C-gen4 (分離の独自性)** | MAP-E > panmictic-GA かつ > random | strict_compare passes (③ が hill-climbing/規模を超える) |

**③ load-bearing 確定** = C-gen1 ∧ C-gen3 ∧ C-gen4 通過 (C-gen2 で欺瞞性/ギャップ前提を確認)。
**③ 非 load-bearing (honest negative)** = C-gen2 で landscape が滑らか or C-gen3 不通過。
→ いずれも Step C verdict §6(g) の無限後退を止める決着点。

## honest 規律 (benchmark_honest_disclosure / external_ai_verify)

- hold-out regime を draw レベルで完全分離 (汎化の水増し排除)。
- 全群 **equal budget** (n_evals 厳守、MAP-E と baseline の評価回数を揃える)。
- 進化中 best (noisy) でなく **fresh-seed honest 再評価**を主指標 (elitism artifact 排除)。
- random search baseline を必ず併置。
- 各 exp で **Codex pair-review** (read-only) → findings を実コード検証後に反映。
- negative-but-informative も削除せず verdict に残す。
- src 非変更、research/ea_multitask/ 隔離、pytest 必須、push 未 (review 後)。

## 実装計画 (TDD, 小単位)

1. `task_mixture.py` — TaskMixture (regime 集合 + weights + train/test split)、+ tests。
2. `ea_lab.py` — MAP-E / MAP-E_randselect / panmictic-GA / random を共通 budget で回す runner
   (selection_lab を流用、randselect は親選択差し替え)、hold-out 汎化評価、+ tests。
3. `exp_ea1_substrate_floor.py` — C-gen1 (基質が個別 regime を解くか)。
4. `exp_ea2_landscape.py` — C-gen2 (汎化ギャップ/多峰性)。
5. `exp_ea3_ablation.py` — C-gen3/C-gen4 (③ ablation strict_compare)。
6. `E_A_VERDICT.md` — ③ load-bearing or honest negative の決着。

各 step commit 前に Codex pair-review。py 実行は集約 (Windows flash 削減、CREATE_NO_WINDOW 済)。

## 関連

- [[project_llcore_init_2026_05_29]] — llcore 全体 (Step C verdict / 梯子段1 含む)
- [[feedback_codex_pair_review_for_llcore]] / [[feedback_external_ai_verify]] — review 規律
- [[feedback_benchmark_honest_disclosure]] — 水増し排除・両方向決着
- Step C verdict §6(g) / 梯子段1 VERDICT「次の優先経路」
