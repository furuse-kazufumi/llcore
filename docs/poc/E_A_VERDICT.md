# E-A VERDICT — 多タスク分布の hold-out 汎化で③(選択圧/分離)は load-bearing か

> 2026-05-31。research/ea_multitask/ 隔離、src 非変更、push 未。
> 設計=`E_A_DESIGN_multitask_generalization.md`。Step C verdict「次(CPU安価2本柱)(2)」の決着。

## 問い

梯子段1 で parity 経路が degree-5 床で詰んだため、③(分離が選択を可能にする=niching+選択)を
**parity に縛られない土俵=多タスク分布の hold-out 汎化**で検定する。土俵は④候補を workflow で
並列探索し、**variable_delay_recall**(cue + distractor ノイズ、遅延 D 可変)を選定
(medium 難易度・正の汎化ギャップ・niche 構造あり・敵対検証 trustworthy)。FlipFlop は
too-easy(全 regime ≈0.95 飽和・汎化ギャップ負=非診断)のため不採用 (exp_ea1)。

## 方法 (exp_ea3, n_seeds=15, n_evals=400)

基質=単層 leaky reservoir(n_taps=8, in_dim=2)+ ridge readout。タスク=VariableDelayRecall
(distractor_amp=0.2)。train regimes=遅延 D{15,30}、test(hold-out)=D{45,60}(より長い遅延への
extrapolation)。4 method を equal budget で進化→best gene を **test regimes で fresh-seed honest
再評価**(主指標=test 汎化 R²)。strict gate = 片側 Wilcoxon p<0.05 ∧ |paired_sign_delta|≥0.147 ∧ n≥15。

③ablation(設計 spec「②③殺し①変異のみ」):
- **MAP-E (full ①②③)**: behavior grid + archive elite を親 + fitness ゲート placement。
- **MAP-E_randselect (②③殺し)**: grid 同じ・親=bounds から random・placement 無条件(選択圧除去)。
- **panmictic-GA (①③, ②なし)**: tournament 選択、niching なし。
- **random (同予算)**: 対照。

## 結果 (Codex pair-review 後の修正版、2026-05-31)

下表は **方法論修正後の再実行値** (HONEST_N=30 / CRN pairing / global-best-of-budget)。
修正前の旧値と結論は同じ (③ は load-bearing でない) で、むしろより明快になった
(panmictic が MAP-E をわずかに上回る)。旧値・修正内容は次節「Codex pair-review」を参照。

| method | test 汎化 R² (mean±std) | train | gap |
|---|---|---|---|
| **MAP-E (full)** | **0.682 ± 0.115** | 0.898 | +0.216 |
| MAP-E_randselect (②③殺し) | 0.557 ± 0.108 | 0.872 | +0.315 |
| panmictic-GA (②なし) | 0.702 ± 0.083 | 0.915 | +0.213 |
| random | 0.620 ± 0.105 | 0.877 | +0.258 |

| ゲート | 比較 | diff | p (片側) | δ | passes |
|---|---|---|---|---|---|
| **C-gen3** | MAP-E > randselect | +0.126 | 0.0151 | +0.60 | **True** |
| C-gen4a | MAP-E > panmictic | **−0.019** | 0.598 | −0.07 | False |
| C-gen4b | MAP-E > random | +0.062 | 0.126 | +0.20 | False |

## Codex pair-review (gpt-5.4, read-only, 2026-05-31) — verdict 信頼性監査

verdict 先行 commit で deferred にしていた pair-review を実施 ([[feedback_codex_pair_review_for_llcore]])。
Codex は当初版を **「現状の結論は信頼できない、再実行要」** と判定。7 findings を実コードで一件ずつ
検証 ([[feedback_external_ai_verify]]) し、rerun blocker 3 件を修正した:

- **F3 (High) seed 設計**: 進化 seed が method 間でエイリアス (`base+s+{0,1,2,3}`)、かつ honest 再評価
  seed が method 毎に異なり **index s が真の matched replicate でない** → paired Wilcoxon の前提崩壊。
  → 進化 seed を `SeedSequence([base,method_idx,s])` で一意化 + honest 再評価を index s で **全 method
  共通 (common random numbers)** に。同一 replicate で 4 method が同じタスク draw で採点される。
- **F2 (Med) randselect archive 忘却**: `best` を最終 archive (6×6=36 occupant) の max から選んでいた。
  randselect は無条件上書きで強個体を忘れ、全 400 から best を採る random より不利 → C-gen3 の gap を
  水増し (③有利方向のバイアス)。→ 全 method で **global best-of-budget** を読み出す (full MAP-E は
  fitness ゲートで global best が退避しないため数値不変、randselect のみ公平化)。
- **F7 (Med) honest_n<30**: `HONEST_N=16` は honest_eval §5 の確率的 fitness 基準 (≥30) 未満。→ 30 に。
- F1/F4/F5 (Low) = equal budget・train/test リーク無し・strict gate ロジックは健全 (バグ無し確認)。
- F6 (Med) 結論の射程: 単一 budget(400)/grid(6×6)/descriptor(w_in 無視) に対し結論が広い → 下記で限定。

**修正の効き**: randselect を global-best 化で強くした (0.536→0.557) のに C-gen3 はまだ PASS
(③ vs 無選択ドリフトの差は本物)。一方 panmictic が MAP-E を逆転 (0.702>0.682) し C-gen4a は
**負の diff** に。**結論は修正前後で不変** = honest-negative は方法論的に健全な土台で確認された。
回帰テスト 22 pass (`tests/test_ea_lab.py`、③除去 witness を archive 占有者に移して維持)。

## 結論 (honest)

**③ は本分布で load-bearing でない (honest negative)。**

- **C-gen3 PASS / C-gen4 FAIL**。MAP-E は「②③を殺し①変異だけ残した randselect」には strict gate で
  有意に勝つ(p=0.015, δ=+0.60)。だが **panmictic-GA(選択あり・niching なし)には逆に僅かに負け
  (−0.019)、random にも有意差なし**。
- 解釈: **MAP-E > randselect が示すのは「何らかの選択 > 無選択ドリフト」**であって、③(behavioral
  niching=分離)固有の寄与ではない。② niching を外した panmictic が MAP-E と同等以上(全 method ≈0.62-0.70 で
  天井近傍)=**この多タスク汎化 landscape は十分に滑らかで、分離(③)が無くても単純な選択 or random で
  同じ汎化に到達する**。
- = Step C / step4 (exp5/exp7 smooth) / 梯子段1 と**一貫**: ③ は **欺瞞的 corridor 限定で load-bearing**
  (step4 exp4 で存在証明)、滑らか/実問題近 landscape では不要。

## ③ 研究 (Step C → 梯子段1 → E-A) の総括

- ③(分離/niching が選択を可能にする)は **機構として本物**(step4 の合成欺瞞 corridor で decisively
  load-bearing)。だが **parity(基質の床/degree-5)・記憶タスク・多タスク汎化のいずれの "実問題寄り"
  土俵でも、landscape が滑らかで③は不要**だった。
- 「③に将来性なし」は誇張だが、**「実 AI 設計探索で③(QD/niching)がペイする証拠は CPU 範囲では乏しい」**
  が honest な現状。③が効くには **欺瞞的 landscape の存在が前提**で、それが実問題で自然に現れるかは未確認
  (full LLM 損失地形の欺瞞性測定=GPU 投資が唯一の残る検証路、本 proxy 証拠では投資根拠は弱い)。

## 次 (ユーザー判断)

- (a) ③ 路線を保留し llcore 別軸(検証ゲート/kernel plugin S3/論文化)へ転換。
- (b) full LLM(GPU)で損失 landscape の欺瞞性を本測定(proxy の限界を超える唯一の道だが投資)。
- (c) 実 task で欺瞞 corridor を意図的に作れるか追試(③の適用条件の特定)。

## 規律・成果物

- 全 method equal budget・hold-out 厳守(train/test regime 分離・リークなし)・fresh-seed honest 再評価。
- research/ea_multitask/(task_mixture / ea_lab / exp_ea1 / exp_ea3 / candidates)。exp_ea3_results.json。
- ④土俵候補は workflow で並列設計+敵対検証(variable_delay_recall=trustworthy 採用)。
- **Codex pair-review 未実施(次セッション)**: 結果が strict gate で明快(C-gen4 FAIL)なため verdict 先行。
- 関連: [[project_llcore_init_2026_05_29]] / [[feedback_codex_pair_review_for_llcore]] /
  [[feedback_benchmark_honest_disclosure]] / Step C verdict / 梯子段1 VERDICT。push 未。
