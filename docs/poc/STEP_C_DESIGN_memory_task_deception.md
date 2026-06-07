# CPU 手順 (c) 設計 — 記憶タスクで欺瞞 corridor が自然に現れ ③ が立つか

> **凡例 — 進化4要素 (Darwin/Mayr)**: ①変異 (variation) / ②遺伝 (heredity) / ③適者生存・選択 (selection = 適応度の差による差し survival) / ④過剰繁殖 (over-reproduction)。本書の「①」〜「④」はこの番号を指す (特に「③」= 適者生存)。平易な用語集 → [`YOUGO_平易版.md`](./YOUGO_平易版.md)。

**位置づけ**: `EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md` next_plan の選択肢 (c)。手順 4 (`STEP4_SELECTION_VERDICT.md`: 合成欺瞞 corridor で ③ 成立を実証) と手順 6 (`STEP6` 系: 実テキスト proxy は滑らかで ③ 不要) を受け、**「実タスク本来の性質から欺瞞 corridor が自然に現れるか」**を CPU で厳格に決着させる。
**ステータス**: 設計承認済 (ユーザー 2026-05-30)。spec → 実装計画へ。
**構造規律**: research/ 隔離・src 非変更・CPU 完結・各 exp で Codex pair-review ([[feedback_codex_pair_review_for_llcore]])・honest 留保を verdict に明記 ([[feedback_benchmark_honest_disclosure]])。

---

## 1. 目的と問い

ユーザーの根本的懐疑: **「本当に進化するのか？」**

現状 (監査 + 手順 1-6): 進化4要素のうち①②は成立、③④は機構があっても空転。累積改善 (= 進化) は **人工的に仕込んだ合成欺瞞地形でしか示せていない**。「自分で意地悪に作った地形でしか勝てない」なら本物の進化とは言えない。

(c) はこの懐疑に CPU で決着をつける: **人工注入なしで、問題そのものの性質から欺瞞 corridor が自然に生まれ、そこで ③ (選択による累積改善) が効くか**を測る。

## 2. 合格基準 (ユーザー確定 2026-05-30 = 厳格・両方向決着)

- **③ 実在と認める条件**: 実タスク本来の性質から欺瞞 corridor が現れ、そこで ③ が load-bearing になる (下記 C1-C4 全成立)。**人工注入は不可**。
- **③ 路線 CPU 撤退の条件**: 全タスクで landscape が滑らか / ③ が不要。**negative result も明確な決着**として verdict 化する。
- GPU 投資は本実験では判断しない。撤退時は「GPU = 実 LLM 損失地形の欺瞞性を測る賭け」に限定される根拠とする。

## 3. アプローチ — 記憶タスク群 (人工注入でない標準的難タスク)

「覚えていないと解けない」= 長期依存×非線形のタスクは、**中間の部分解では点が伸びず、正しい記憶ダイナミクスに到達した瞬間に急に解ける**性質を持ち、本来的に欺瞞的 (局所最適トラップ) になりうる。これは地形を手で作るのでなく、タスク本来の難しさ。

**根拠 (古典・コーパス外の定説。RAD コーパスは 2024-26 LLM 中心で進化計算古典は手薄と honest 確認済)**:
- Lehman & Stanley 2011 (novelty search): deceptive maze で目的関数ベース探索が局所最適に嵌まり、多様性探索が抜ける。
- Bengio et al. 1994: 長期依存タスクは局所/勾配探索で難しい (局所最適トラップ)。
- Sussillo & Barak 2013 (flip-flop) / NEAT の T-maze・速度なし倒立振子: 記憶を要するタスクが neuroevolution の標準難問。

**採用する記憶タスク (3 種)**:
1. **delayed parity** — 遅延窓内の記号の XOR/パリティを答える。
2. **flip-flop** — set/reset で 1-bit を保持する記憶スイッチ (Sussillo & Barak)。
3. **delayed recall** — 系列冒頭の合図を遅延後に出力 (T-maze 風 非マルコフ記憶)。

## 4. ハーネスと基質

- **比較ハーネス**: 手順 4 の `research/step4_selection/selection_lab.py` を流用。MAP-Elites (behavioral niching) vs 3 baseline {random / panmictic-GA / random-restart hill-climbing}。`eval_once` を記憶タスク fitness に差し替えるだけ。
- **公正判定**: 強化版 `src/llcore/evolution/honest_eval.py` の `evolution_vs_random` / `FalsificationResult.passes` (fresh-seed 再評価・同予算・n_seeds≥15・片側 Wilcoxon p<0.05・|paired_sign_delta|≥0.147)。
- **基質 / gene**: 固定 ESN (Echo State Network) では記憶構造を進化できない → **reservoir のダイナミクスを gene 化**する。手順 4 の leaky delay-line 空間拡張を継承 (3-param では表現力不足のため)。readout は手順 2 の per-gene ridge (held-out, `src/llcore/fitness/ridge_readout.py`) を流用。

## 5. 診断基準 C1-C4 (手順 4 の falsifiable 基準を流用)

| 基準 | 内容 | 測り方 |
|---|---|---|
| C1 | landscape が多峰 (分離した良解 peak が複数) | grid/random sample で peak 検出。random-restart hill-climb 収束点の中点が谷になるペアが存在 |
| C2 | hill-climbing が局所最適に詰まる | RR-hillclimb / panmictic-GA / random の大域到達率が低い |
| C3 | niching が baseline に有意勝利 | MAP-Elites が 3 baseline 全てに honest_eval (強化版 passes) で勝利 |
| C4 | 勝因が探索量でなく diversity 維持 | init_batch ablation で coverage でなく archive ratchet が勝因と確証 |

## 6. 合格 / 撤退の判定

- **いずれかの記憶タスクで C1-C4 全成立** → **③ 実在** (実タスク本来の性質から、人工注入なし)。
- **全タスクで C1 が出ない (滑らか) or C3 不成立** → **③ 路線 CPU 撤退**。negative result を verdict 化。

## 7. ズルをしない約束 (人工注入でないことの担保)

- タスク定義は標準形 (parity / flip-flop / recall)。地形を恣意的にいじらない。
- 難易度パラメータ (delay 長・窓幅) は「タスク本来の難しさ」の範囲のみスイープ。地形に手を加えて欺瞞を捏造することは禁止。
- 「自然発生」の主張は、難易度を上げただけで C1 多峰性が現れたことをもって行う (peak を手で配置しない)。

## 8. 実装段階 (各段階で Codex pair-review)

1. **記憶タスク 3 種の `eval_once` 実装** + reservoir gene 基質 + landscape map (C1 多峰性チェック)。
2. **多峰が出たタスクで MAP-E vs 3 baseline** (C2/C3) を honest_eval で測定。
3. **勝因 ablation** (C4: init_batch sweep)。
4. **verdict** (③ 実在 or CPU 撤退) を `STEP_C_VERDICT.md` に。honest 留保を明記。

## 9. かみ砕き版 (山登りのたとえ)

「進化」が本物なら、ただの山登りや当てずっぽうより良い設計に届くはず。だが今までは**人工的に作った「だまし地形」でしか進化が勝てなかった**。(c) は、地形を仕込むのでなく**問題そのもの**から「だまし地形」が自然にできるかを見る。

使うのは「**覚えていないと解けない問題**」(flip-flop = 記憶スイッチ / delayed parity / delayed recall)。これらは**中途半端な覚え方では点が伸びず、正しい覚え方に届いた瞬間に急に解ける**ので、「ニセ頂上 → 谷 → 本物の頂上」のだまし地形が自然に生まれる候補。

```
 点↑                          本物の頂上
   |                            /\
   |   ニセ頂上                /  \
   |    /\        谷          /    \
   |___/  \________________/        \___
        ↑停滞          ↑普通の探索は谷を下れず本物に届かない
```

確認は4ステップ: ①本物のだまし地形か (頂上が複数+谷) ②普通の山登りは詰まるか ③進化 (飛び石) だけが本物の頂上に届くか ④勝因は賢い選択か数撃ちゃ当たるか。

結果は2通り、どちらでも決着: **どれかの問題で4つ全部○なら「進化は本物」** / **全部滑らかなら「③路線は CPU では打ち止め」**(GPU は実物 AI で測る賭けに限定)。
