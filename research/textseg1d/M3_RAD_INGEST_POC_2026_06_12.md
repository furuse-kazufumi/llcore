# M3 RAD コーパス取込 PoC — 世界知識の注入 (2026-06-12)

> ROADMAP M3「世界知識の注入」の最初の PoC。M1 で確定した MiniLM backend
> (`SentenceEncoderBackend`, cosine MRR 0.947 @ 22 会話 probe) の AnnotationStore に、
> 既製の世界知識 md docs (RAD corpus) を取り込み、(1) 会話に無い世界知識質問への
> grounding 効果と (2) store 大規模化による会話 retrieval への干渉を実測した。
> 正本 = `out/rad_ingest_poc.json`、スクリプト = `scripts/rad_ingest_poc.py`。

## 実装概要

- **取込対象**: `D:\docs\loop_engineering_corpus_src\` — 39 md docs / 4 スコープ
  (autonomous-agent-loops / control-feedback-loops / learning-loops / operational-ci-loops)、
  frontmatter (title/scope/category) 付き、**日英混在** (本文の大半は日本語 + 英語術語埋込)。
- **取込方法**: frontmatter / コードフェンス / 見出し記号 / リンク / URL を rule-based で
  素朴に除去し、doc 単位で `AnnotationStore.add_text(role="corpus", group=doc毎採番)`。
  group は 10 刻みで採番し、`adjacency_window=1` の共起エッジが doc を跨がないようにした。
- **世界知識 probe**: corpus 実テキストを先に読み、答えが verbatim 実在する英語 18 問を
  **測定前に事前登録** (4 スコープから 4/5/5/4)。測定後の変更なし — 失敗 probe も全て残す。
- **検索経路**: `store.query(q, k=10, exclude_questions=True)` の cosine 固定。
  R@1/R@3/MRR は connectivity_bench と同一計算 (`rank_of`/`mrr` を import)。
- **encoder**: `SentenceEncoderBackend` (all-MiniLM-L6-v2) 固定。

## store 規模・取込時間

| store | annotations | 共起エッジ | 取込時間 |
|---|---|---|---|
| 会話のみ (connectivity_bench.SOURCES, 35 turns) | 97 | 1,116 | 15.1s (encoder lazy load 込) |
| 会話 + RAD (39 docs) | **1,071** (corpus 分 +974) | 49,360 | corpus 分 **17.0s** |

store は M1 の 97 → 1,071 annotations へ **約 11 倍**。encode_saved_ratio は 0.0395
(corpus は dedup がほぼ効かない一回性テキストなので低くて当然 — 会話ログとは性質が違う)。

## 3 条件の結果 (確定値)

| 条件 | probe | R@1 | R@3 | MRR |
|---|---|---|---|---|
| (a) 会話のみ store | 世界知識 18 問 | **0.000** | 0.000 | **0.0000** |
| (b) 会話+RAD store | 世界知識 18 問 | **0.611** | 0.611 | **0.6389** |
| (c) 会話+RAD store | 既存 22 会話 probe | 0.909 | 1.000 | **0.9470** |
| (c0, 参考) 会話のみ store | 既存 22 会話 probe | 0.909 | 1.000 | 0.9470 |

- **(a) floor 確認**: 期待通り全滅 (18/18 が圏外)。会話 store は世界知識を一切持たない。
- **(b) 注入の効果**: MRR 0 → 0.639。18 問中 11 問が rank 1、2 問が rank 4、5 問が圏外。
- **(c) 干渉**: **ゼロ**。store 11 倍化後も 22 会話 probe は per-probe 単位で c0 と完全一致
  (R@1 0.909 / R@3 1.000 / MRR 0.947 — M1 正本 `out/connectivity_bench_minilm.json` とも一致)。
  974 件の corpus annotation は会話質問の top-10 に割り込んでこなかった。

## per-probe 内訳 (世界知識 18 問 — 改善も全滅も全て)

| rank | probe (要約) | gold |
|---|---|---|
| 1 | canary deployment (traffic %) | canary |
| 1 | watchdog / stuck process detection | heartbeat, watchdog |
| 1 | MAPE-K stages | mape-k |
| 1 | OODA (Boyd) stages | observe-orient-decide-act, ooda |
| 1 | circuit breaker recovery state | half-open |
| 1 | AlphaZero self-play | self-play, mcts |
| 1 | EWC / continual learning | catastrophic forgetting |
| 1 | active learning selection | uncertainty sampling |
| 1 | value iteration fixed point | bellman |
| 1 | Reflexion memory | episodic memory |
| 1 | CoVe verification questions | verification questions, hallucination |
| 4 | GitOps convergence target | desired state |
| 4 | chaos engineering hypothesis | steady state |
| 0 (圏外) | PID error signal | setpoint |
| 0 (圏外) | MPC receding horizon | receding horizon |
| 0 (圏外) | RLHF reward model | reward model, bradley-terry |
| 0 (圏外) | Voyager skill library | skill library, voyager |
| 0 (圏外) | ToT backtracking | backtracking |

### 失敗 5 問の post-hoc 分析 (probe 変更はしない — 解釈のみ)

top-5 の実ヒットを目視確認したところ、失敗の主因は「retrieval の失敗」と
「gold 判定基準の取りこぼし」が混在している:

- **正しい doc は取れているが gold 文字列が別断片/日本語側にある** (4/5):
  - *receding horizon*: top-5 に MPC の annotation が 2 件入っているが、その断片は
    「…再観測の**後退ホライズン**」 — 答えは日本語表記で取れており、英語 verbatim の
    「receding horizon」を含む断片 (llive 適用の 1 文のみ) が top-10 に入らなかった。
  - *backtracking*: rank 1-2 が Tree of Thoughts の annotation (正解 doc)。gold を含む
    断片「自己評価 + backtracking で誤経路を能動的に剪定」だけが top-10 圏外。
  - *setpoint*: rank 1 が「adaptive pid（neural network を…」 (正解 doc の別断片)。
  - *rlhf reward model*: rank 1 が RLVR の annotation (「rlhf の枠組みで**報酬モデル**を…」)
    — 内容は正答だが gold が日本語表記「報酬モデル」のため不一致。
- **本当に retrieval が外している** (1/5): *voyager skill library* — top-5 に Voyager doc が
  入らず、plan-and-execute / LATS 等の別 agent loop が並んだ。

つまり厳密 verbatim gold での MRR 0.639 は**下限値**であり、doc レベルの正答率は
目視ではこれより高い。ただしこれは post-hoc の定性観察で、事前登録した測定値は
0.639 のまま確定とする (数値の差し替えはしない)。

## 解釈 (honest)

1. **M3 仮説の最初の支持**: 既製 md docs を AnnotationStore にそのまま流し込むだけで、
   会話に一切出ていない世界知識質問の MRR が 0 → 0.639 (R@1 0.611)。取込は 39 docs /
   974 annotations で 17 秒・追加実装は markdown 除去のみ — 取込コストは極めて低い。
2. **干渉ゼロは予想より良い**: M1 の懸念「store 桁違い化で 0.947 が保つか」に対し、
   11 倍化で劣化ゼロ (per-probe 完全一致)。会話 probe と corpus annotation の埋め込みが
   MiniLM 空間で十分分離している。ただし 11 倍は「桁違い」の入口にすぎない —
   100 倍 (10 万 annotations 級) での再測が必要。
3. **失敗モードの主犯は日英混在**: 失敗 5 問中 4 問は正解 doc 自体は上位に取れており、
   断片化 (答えが同 doc 内の別 annotation に割れる) と日本語表記 (後退ホライズン/報酬モデル)
   が verbatim gold と噛み合わなかった。retrieval の質の問題というより、
   **短句アノテーション分割と多言語 gold 判定の問題**。M3 続行時の改善候補:
   (i) doc レベルの gold 判定の併記、(ii) 多言語 encoder (例 multilingual MiniLM) の
   head-to-head、(iii) 隣接 annotation への 1-hop 展開 (共起エッジは既に張ってある)。
4. R@1 = R@3 = 0.611 (rank 2-3 への着地がゼロ) — 当たる probe は rank 1 で当たり、
   外す probe は top-10 圏外という二極化。corpus の術語密度が高く、当たるときは
   術語 annotation が強く引かれるため。

## 限界

- **probe は自作 18 問・小規模**。事前登録で cherry-pick は排したが、probe 設計者 =
  corpus 読者なので「答えやすい質問を無意識に選ぶ」バイアスは残る。
- **corpus は 1 分野 (loop engineering) のみ・39 docs**。分野横断・大規模での一般化は未検証。
- **gold = キーワード包含** (verbatim)。上記の通り日英混在 corpus では undercount し、
  逆に頻出術語 (例 "ooda") では緩すぎる方向の誤差もありうる。
- **日英混在は MiniLM に不利な条件**: all-MiniLM-L6-v2 は英語中心。0.639 は
  この混在条件込みの値であり、英語専用 corpus / 多言語 encoder での値ではない。
- markdown 除去は素朴な rule-based — 除去漏れの記号断片がアノテーションに混入しうる。
- 干渉測定は 22 会話 probe のみ。会話とトピックが重なる corpus (例: 料理 docs) を
  入れた場合の干渉は未測定 (本 corpus は会話トピックと重ならない分野なので有利な条件)。

## 再現

```
py -3.11 scripts/rad_ingest_poc.py        # -> out/rad_ingest_poc.json
py -3.11 -m pytest tests/unit -q          # 390 passed (src 変更なし)
py -3.11 -m ruff check scripts/rad_ingest_poc.py
```
