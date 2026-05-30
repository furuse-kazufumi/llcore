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

## 結果

| method | test 汎化 R² (mean±std) | train | gap |
|---|---|---|---|
| **MAP-E (full)** | **0.696 ± 0.068** | 0.908 | +0.212 |
| MAP-E_randselect (②③殺し) | 0.536 ± 0.206 | 0.777 | +0.241 |
| panmictic-GA (②なし) | 0.665 ± 0.098 | 0.904 | +0.238 |
| random | 0.635 ± 0.108 | 0.892 | +0.257 |

| ゲート | 比較 | diff | p (片側) | δ | passes |
|---|---|---|---|---|---|
| **C-gen3** | MAP-E > randselect | +0.161 | 0.0062 | +0.73 | **True** |
| C-gen4a | MAP-E > panmictic | +0.031 | 0.126 | +0.07 | False |
| C-gen4b | MAP-E > random | +0.062 | 0.076 | +0.47 | False |

## 結論 (honest)

**③ は本分布で load-bearing でない (honest negative)。**

- **C-gen3 PASS / C-gen4 FAIL**。MAP-E は「②③を殺し①変異だけ残した randselect」には strict gate で
  有意に勝つ(p=0.006, δ=+0.73)。だが **panmictic-GA(選択あり・niching なし)にも random にも有意差なし**。
- 解釈: **MAP-E > randselect が示すのは「何らかの選択 > 無選択ドリフト」**であって、③(behavioral
  niching=分離)固有の寄与ではない。② niching を外した panmictic が MAP-E と同等(全 method ≈0.63-0.70 で
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
