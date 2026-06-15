# llcore 再計画 — 「LLM としての機能確保」優先 (2026-06-13)

> Goal (ユーザー設定, Stop-hook): **llcore の計画を再度立て直し、LLM としての機能確保 (capability) を最優先にする。**
> 本書はその再計画の正本。承認後 spec → writing-plans → Phase 0 実装へ。

## 1. なぜ再計画するか (honest disclosure)

- **Phase 2 verdict = capability NEGATIVE / NULL_TIE**: 検証可塑性 (verified plasticity) の枠組みは健全に機能したが、肝心の「進化は勾配より良い LM を作れるか」は **実地形でも引き分け〜負け** (evolution ≈ gradient、解析勾配 meta-gate で 19/20 逆転)。出ている価値は「**健全性の GUARANTEE**」であって「強い LM」ではない、と自分のベンチで確定済み。
- **llcore はまだ LLM ではない**: char 単位 tiny-shakespeare 級で、なぞなぞも解けない。「**最低限 LLM にならねば成果ゼロ**」(memory: `feedback_llcore_must_become_llm_relevant`)。
- **ユーザー判断**: 「検証機を作るばっかりより、LLM を作りたい」+ 本 Goal。→ capability を主役へ。
- **進化アルゴリズムは捨てない**: 「重み最適化レベルで勝てない」だけ。勾配が踏めない **離散/構造探索 (NAS)** に再配置する。

## 2. 新しい優先順位

1. **LLM capability (機能確保) = 最優先**（北極星）
2. **進化 = NAS**（アーキ/ハイパーの構造探索。勾配=重み / 進化=構造）
3. **verified-plasticity / 3D viz = 二次**（capability に貢献する *機能/説明物* へ降格。捨てず維持。今日の地形アニメ SVG はその「正直な検証結果の説明図」として完成済み）

## 3. 確定要件 (2026-06-13 ブレストで決定)

| 軸 | 決定 |
|---|---|
| 計算資源 | **CPU 完結 nano/char 級**（このPCは GPU 非搭載・`torch 2.12.0+cpu`）。capability 拡大は後段で**無料クラウド GPU** (Kaggle/Colab) |
| 手法分担 | 重み = 勾配 (AdamW) / 構造 = 進化 (NAS) |
| **Phase 1 合格線** | **日本語 char-LM が自然な続きを生成 + held-out perplexity が unigram を明確に下回る** |
| topology | **GPT-2 互換**（学習済みを llm-viz / clean-room 3D で *そのまま歩ける* よう。model-data 形式は確認済: `{shape,dtype:torch.float32,data:base64}` の flat dict + config） |

## 4. ロードマップ (capability-first)

- **P0 — de-risk (数日)**: CPU char-Transformer トレーナを `src/llcore/lm/` に**自前実装**（GPT-2 topology = llm-viz 互換。nanoGPT(MIT, Karpathy) を *正しさの参照* に使うが、コードは llcore 自前）。小さな日本語 char コーパスで学習 → **held-out PPL < unigram** ＆ 生成が崩れないことを確認。英 tiny-shakespeare で smoke も（既知ベースラインで学習器を実証）。
- **P1 — 合格線到達**: 日本語で **unigram を明確に下回る PPL + 自然な続き**。学習済みを gpt-nano JSON schema に export → **自分のモデルを clean-room 3D で歩く**（「最低限 LLM、しかも見える」）。
- **P2 — 進化 = NAS**: char-LM のアーキ/ハイパーを進化探索（候補は **短時間 proxy 学習**で評価 → 勝者を本学習）。勝者を 3D で歩く。任意で**検証可塑性/安定性を NAS の制約**に（llcore identity の戻し込み・二次）。
- **P3 — capability 拡大**: 無料クラウド GPU へ。**BPE + 大きめ日本語コーパス** → 「簡単な質問応答」級へ。

## 5. 成果物の置き場 / 命名

- 学習器: `src/llcore/lm/`（新規）
- 3D viz: clean-room（Bycroft コード非依存・自前 Apache-2.0。`D:/projects/llcore-viz` の bbycroft/llm-viz は **無ライセンス=ローカル研究のみ・公開不可**。memory: `project_llcore_3d_viz_llmviz_fork`）

## 6. honest 留保 (over-claim 防止)

- **P1 (CPU char) は「最低限 LLM = それっぽい生成」まで**。なぞなぞ/質問応答級は **P3 (クラウド GPU)** で初めて届く。CPU-nano では届かない、を明示する。
- **NAS は CPU で候補ごとに学習 = 高コスト**。proxy 短学習 + 小探索でしか回せない → 予算管理を設計に含める。
- 「進化で LLM」のロマンは残すが、Phase 2 の負け実績を消さない。NAS は「進化が勝てる土俵 (離散構造)」に限定する honest な配置。

## 7. 次アクション

1. 本再計画をユーザーが確認・承認
2. Phase 0 の spec → `writing-plans` で実装計画
3. P0 実装着手（`src/llcore/lm/` トレーナ + 日本語 char コーパス）

---
関連 memory: `feedback_llcore_must_become_llm_relevant` / `project_llcore_evolvable_llm_replan_2026_06_09` / `project_llcore_3d_viz_llmviz_fork` / `feedback_benchmark_honest_disclosure` / `feedback_no_solo_ai_judgment`
