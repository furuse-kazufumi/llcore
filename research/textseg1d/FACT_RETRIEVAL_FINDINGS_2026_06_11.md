# 事実検索の honest 知見 (優先1, 2026-06-11)

## 問題 (差別化レビューが指摘した生成前段の弱点)
新規セッションで過去会話の事実に答えるため AnnotationStore を retrieval 前段に使うと、
質問クエリ「What is my name?」が**質問文**「what is my name」(cosine 1.0) ばかり拾い、
事実文が上位に来ず augmented 出力が baseline 同等だった。

## 対策1: 事実抽出 (質問除外 + ロール付与) — 部分的に有効
- `is_question()` (英 疑問詞/助動詞始まり, 日本語 末尾か) で質問アノテーションを判定。
- `query(exclude_questions=True)` で平叙文 (事実候補) のみ検索。`role` フィルタも追加。
- 結果: 質問文は除外できた。だが**より深い問題が露呈** ↓。

## ★決定的 honest 発見: SigLIP の短句検索品質が根本ボトルネック
「my name is kazufumi」(答えの実体を含む真の事実) は、質問「what is my name」に対し
**第10位** (cosine 0.795) — 無関係な「thank you」(0.824) / 「hello」(0.819) /
「name one famous novel」(0.813) / 「yes」(0.798) より**下**。

→ SigLIP text encoder は短句 retrieval で「thank you」を答えより上に置く = 実用品質に達していない。
これは差別化調査が引いた Jina CLIP (arXiv:2405.20204)「vanilla CLIP は MTEB retrieval で
SBERT の ~1/3」を**実会話データで再現**した形。

## 含意 (差別化への直撃)
- 「CLIP/SigLIP をテキスト記憶/検索に使える」という naive value prop は**本データでも反証**。
- 差別化は (a) surface-bigram 補償 + (b) verified-evolution 結合 に賭けるしかなく、
  CLIP retrieval 品質そのものは売りにできない (over-claim 禁止)。
- 優先2 (head-to-head vs SBERT/E5/BGE) で定量化 → 専用テキスト埋め込みが大幅に勝つなら、
  **AnnotationStore のエンコーダを CLIP から専用テキスト埋め込みへ差し替え、CLIP は
  cross-modal が要る時だけ**、という設計判断が筋 (連結性基盤としては専用埋め込み + surface 層)。

## 残課題 (生成前段)
- augmented でも LLM (360M) が「Emily Wilson」と hallucinate — retrieval 品質に加え、
  小型 LLM の grounding 力も弱い。事実が top に来ても注入文脈を無視しうる。
- 事実抽出の次段 = **属性束縛つき事実** (subject-predicate-object) 抽出だが、これは
  rule-based では限界 (LLM 必要 → 計算削減目的と相反)。trade-off を要設計判断。

正本データ: `out/annotation_generation_poc_results.json` (事実抽出版) +
`out/retrieval_head_to_head.json` (優先2)。
