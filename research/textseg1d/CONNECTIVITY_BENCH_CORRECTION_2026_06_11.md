# ★連結性グラフの honest 訂正 (22-probe ベンチ, M1.2, 2026-06-11)

## 何が起きたか
優先3 で「連結性グラフが cosine を上回る (MRR 0.056→0.389)」と報告したが、これは **3 probe への
過剰適合 + 弱い cosine baseline のアーティファクト**だった。M1.2 で評価を 22 probe に拡張したところ、
結論が**覆った**。honest-disclosure 規律 ([[feedback_benchmark_honest_disclosure]]) とベンチ拡張が
この誤りを捕まえた = 設計通りの機能。

## 22-probe ベンチ確定結果 (`out/connectivity_bench.json`)
| method | R@1 | R@3 | MRR |
|---|---|---|---|
| cosine (事実のみ) | 0.591 | 0.818 | **0.727** |
| connected (IDF なし) | 0.045 | 0.500 | 0.321 |
| connected (IDF) | 0.591 | 0.818 | **0.727** |

**per-probe: IDF 連結は 22/22 で cosine と完全一致 (0 差)。**

## 何が真実だったか
1. **実の勝因 = 事実抽出 (質問・依頼の除外)**。cosine を事実のみに絞ると MRR 0.727 と強い。
   優先1 の `is_question`/`is_request`/`is_fact` フィルタが本当の load-bearing 改善だった。
2. **共起連結グラフは頑健な差別化ではない**:
   - IDF あり → boost が完全に中和され **cosine と同一** (効果ゼロ・害ゼロ)。
   - IDF なし → 名前等 2-3 の難ケースは助ける (name 6→2) が **大半 (~14 probe) を害する** (MRR 0.321)。
3. **3 probe の「>10→2」はアーティファクト**: あの cosine baseline は質問を除外しておらず弱かった。
   事実抽出した cosine では name は既に rank 6 (>10 でない)。連結の上乗せは IDF が打ち消す。

## 訂正後の honest な立ち位置
- **retrieval は「事実抽出 + cosine (できれば MiniLM)」で大半解決**。連結グラフは差別化の核ではない。
- 共起 (turn 隣接) は弱信号。名前のような**語彙不一致の質問→答え**だけが難しく、そこは
  **entity-coref エッジ** (共有実体で強連結) が将来の候補だが、broad では少数派ゆえ差別化の主軸にできない。
- **差別化は retrieval グラフでなく、(a) 検証付き可塑性 (verified plasticity) と (b) 世界知識統合
  (RAD→AnnotationStore, ROADMAP M3)** に置くのが正直。connectivity を over-claim しない。

## ROADMAP への反映
- M1 の「連結グラフ堅牢化」は**降格** — entity-coref は難ケース用の小改善に留める。
- 主軸を M3 (世界知識注入) と M2 (cert × 教師) に移す。
- 評価は今後 20+ probe ベンチ (`scripts/connectivity_bench.py`) を標準とし、3 probe の cherry-pick を禁止。

正本: `out/connectivity_bench.json` + `scripts/connectivity_bench.py`。
旧楽観報告: CONNECTIVITY_PoC_2026_06_11.md (本訂正で上書き)。
