# M1 残作業の決着 — entity-coref エッジ + MiniLM encoder (22-probe, 2026-06-12)

> CONNECTIVITY_BENCH_CORRECTION_2026_06_11.md の後続。ROADMAP M1 の残り 2 項目
> (entity coref エッジ / encoder 差し替えオプション) を実装し 22-probe で実測した。
> 受入基準 (事前登録) = entity 変種が cosine MRR を下回らないこと (broad 非破壊)。

## 実装

1. **entity-coref エッジ** (`src/llcore/clip/annotations.py`):
   - `informative_tokens()` — stopword (機能語+疑問語) 除外の内容語抽出。
     ★`_REQUEST_WORDS` は含めない ("name"/"list"/"give" は依頼動詞と実体語を兼ねるため
     位置非依存の entity 抽出では殺してはならない — テストで発見した実バグ)。
   - token 転置 index (行追加で無効化) + `entity_df()`。
   - `query_connected(entity_hop=True)` — クエリ/seed と希少トークン (df ≤ cap =
     max(8, 5% rows)) を共有する事実行へ **cosine base + 加算マージン**
     `entity_boost(=0.1 固定) × IDF(token) × 源重み`。
     乗算 boost でなく加算マージンにしたのは noIDF 失敗 (broad 崩壊) の教訓 —
     cosine の大域順位を保ち局所でのみ並べ替える。
2. **MiniLM encoder オプション** (`src/llcore/clip/text_encoders.py`, 別 agent 実装):
   - `SentenceEncoderBackend` (all-MiniLM-L6-v2, Apache-2.0, text-only 専用)。
     lazy load / fail-closed / L2 正規化強制。optional extra `text`。
   - CLIP/SigLIP は cross-modal (text↔image) 専用に残す。
3. `scripts/connectivity_bench.py` に `--encoder {clip,minilm}` + `entity` メソッド配線。

## 22-probe 確定結果

| encoder | method | R@1 | R@3 | MRR |
|---|---|---|---|---|
| CLIP (SigLIP) | cosine (事実のみ) | 0.591 | 0.818 | 0.727 |
| CLIP | connected_IDF | 0.591 | 0.818 | 0.727 |
| CLIP | **entity** | **0.682** | **0.909** | **0.797** |
| MiniLM | **cosine (事実のみ)** | **0.909** | **1.000** | **0.947** |
| MiniLM | connected_IDF | 0.818 | 1.000 | 0.902 |
| MiniLM | entity | 0.909 | 1.000 | 0.947 |

正本 = `out/connectivity_bench_clip.json` / `out/connectivity_bench_minilm.json`。

### per-probe 内訳

- **CLIP + entity: 5 probe 改善 / 0 probe 悪化** (受入基準 PASS, 改善まで確認):
  `where do i live` 圏外→5 / `what is 10 plus 15` 3→1 / `name an italian dish` 2→1 /
  `name a classical composer` 4→3 / `what is carbonara made with` 4→3。
- **MiniLM + entity: 全 22 probe で cosine と同一** (非破壊・効果ゼロ)。
- **MiniLM + connected_IDF: 2 probe 悪化** (`where do i live` 1→2, `name a famous novel` 1→2)。

## 解釈 (honest)

1. **encoder を強くするのが最も効く**。MiniLM cosine 単独 MRR 0.947 は CLIP の全変種を
   大差で上回る。この規模 (97 annotations) の会話 retrieval は「事実抽出 + MiniLM cosine」で
   ほぼ解決。
2. **entity-coref は弱い encoder の補償としてのみ価値がある** (CLIP 0.727→0.797)。
   強い encoder では仕事が残っていない (0.947→0.947、悪化もしない)。
   → 採用判断: entity_hop は **既定 off のオプションとして残す** (CLIP 経路=cross-modal
   文脈で text retrieval が必要な場合の補助輪)。差別化の主軸にはしない (訂正 doc の通り)。
3. **共起連結 (cooccur hop) は強い encoder では微害** (MiniLM 0.947→0.902)。
   「単調改善設計 (cosine を下回らない)」は行スコア単位では真だが、誤った行への加点が
   gold を押し下げる順位反転は防げない — 設計上の主張を down-claim する。
4. M1 降格判断 (差別化の主軸は M3 世界知識 + M2 cert×教師) を追認。M1 はこれでクローズ。

## 限界

- 22 probe / 97 annotations は小規模。MiniLM の 0.947 が大規模 store でも保つかは
  M3 (RAD 取込で store が桁違いに増える) で再測する。
- entity 抽出は英語トークン正規表現ベース — 日本語の言及検出は未対応 (将来課題)。
- entity_boost=0.1 は固定値 (probe での調整はしない — 過剰適合防止)。
