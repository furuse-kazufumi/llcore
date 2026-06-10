# 連結性検索 PoC (優先3 第一段, 2026-06-11)

## 仮説と動機
head-to-head (優先2) で**全 dense encoder が hard R@1=0** — 似た短句 101 個から質問→答えを
cosine 単独で繋げない。ユーザーの「連結性」直感 = **会話の隣接構造をグラフのエッジ**として持てば
橋渡しできるか。

## 実装 (AnnotationStore)
- `add_text(..., group=turn)`: 同一/隣接 turn (`adjacency_window`) のアノテーションに**共起エッジ**を張る。
- `query_connected()`: cosine 事実検索を base に、cosine 上位 seed (過去の質問でも可) の**共起近傍 (=その
  答え) を加点** (max マージ=単調改善, cosine を下回らない)。
- `is_request()` / `is_fact()` 追加: 「suggest…」「let's…」「name one…」等の**依頼文**も事実から除外
  (質問だけでなく依頼も答えでない — 答えを押し出す pollution を排除)。

## 結果 (実会話 35 turn / 97 アノテーション / 524 共起エッジ)

| query | cosine rank | connected rank |
|---|---|---|
| what is my name → kazufumi | **>10 (圏外)** | **2** |
| where do i live → japan/tokyo | **>10 (圏外)** | **5** |
| what pasta dish → carbonara | 6 | 6 (後退なし) |

**機構は実証された**: cosine で発見不能 (>10) だった答えが、共起ホップで **top-5 に到達** (2/3 probe)。
依頼除外で pasta の後退も解消 (単調性回復)。

## honest 留保 (R@1 は依然 0)
- 答えは top-5 に来るが **rank1 ではない**。rank1 を占めるのは「new topic」等の **hub ノード**
  (会話の全 turn に共起する高頻度フィラー)。
- 共起は「会話で隣接した」だけの**弱信号** (因果/正解保証でない)。
- encoder は CLIP のまま (head-to-head では MiniLM がやや上だが R@1=0 は同じ — 本 PoC の主眼は
  encoder でなくグラフ)。

## 次段 (優先3 続き)
1. **hub 抑制 (IDF)**: 共起が広すぎるノード (new topic 等) を down-weight → rank1 を答えに明け渡す。
2. **entity coref エッジ**: 「my name is kazufumi」と「your name is kazufumi」「what is my name」を
   実体 (kazufumi) で結ぶ強エッジ。turn 隣接より強い連結信号。
3. **encoder 差し替え**: AnnotationStore に MiniLM backend オプション (CLIP は cross-modal 専用)。
4. **cert gate × 連結性教師**: 連結グラフを verified adapter の進化信号にし、sound gate が経験 gate を
   上回るか実 LLM hidden で測る (差別化の本命、現状はグラフ単体まで)。

正本: `out/connectivity_retrieval_poc.json` + 実装 `src/llcore/clip/annotations.py`。
