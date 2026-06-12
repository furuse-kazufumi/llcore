# M3 分野単位スコープ (per-row domain タグ) — corpus 間食い合いの復元実証 (2026-06-12)

> ROADMAP M3 の残設計課題「分野/corpus 単位のスコープ」の実装 + 実証。
> 正本データ = `out/rad_domain_filter_check.json`、スクリプト =
> `scripts/rad_domain_filter_check.py`、実装 = `src/llcore/clip/annotations.py`
> (commit: per-row domain タグ)。

## 背景 — role フィルタの限界

role スコープ絞り込み (`M3_TOPIC_OVERLAP_2026_06_12.md` 追記) は会話 retrieval を
完全防衛したが、loop / astro corpus はどちらも `role="corpus"` のため
**corpus 間の食い合い**は防げない。(iii) +800 astro store で loop 18 probe は
`role="corpus"` 絞り込みでも MRR 0.4895 — M3.0 単独取込時の 0.639 から
押し下げられたままだった。

## 設計判断 — (b) per-row 分野タグを採用

NEXT_SESSION (2026-06-12) の候補 2 案から **(b) 行ごとの分野タグ** を採用
((a) group 帯域割当は group の意味 (会話 turn / doc 区切り) と分野の二重役割化で
複雑になるため見送り):

- `add_text(domain="astro")` — 初出行に分野タグを記録 (roles と同パターン)
- `query(domain=..., exclude_domains=...)` — positive / negative 絞り込み
  (矛盾指定は fail-closed、既定 None は全件 = 後方互換)
- role (誰が言ったか) と domain (どの知識分野か) は**直交軸** — 併用で AND
- save/load 後方互換: `domains` キー無し旧 JSON は全行 None で読める
- unit +6 = 399 PASS

RAD 接地: per-corpus の namespace partitioning は先行研究で主流の設計
(KG / Semantic-ID ベースの scoped retrieval — llm_corpus_v2 doc_1000 SkillGraph,
mlops_corpus_v2 doc_0193 GeoGR / doc_0495 Snapchat Semantic IDs)。

## 測定 — (iii) +800 store を domain タグ付きで再構築

構成: 会話 (domain=None) + loop 39 docs (domain="loop") + astro 800 docs
(domain="astro") = 23,169 annotations。encoder = MiniLM (続投確定)。
probe = 事前登録済みの loop 18 (WORLD_PROBES) + 会話 22 (PROBES)。

| 条件 | 対象 probe | R@1 | R@3 | MRR | lat |
|---|---|---|---|---|---|
| world nofilter (床) | loop 18 | 0.444 | 0.444 | 0.4895 | 33.7ms |
| world `role="corpus"` (role の限界) | loop 18 | 0.444 | 0.444 | 0.4895 | 32.3ms |
| **world `domain="loop"` (本命)** | loop 18 | **0.611** | **0.611** | **0.6389** | 29.8ms |
| conv `exclude_roles={"corpus"}` (回帰確認) | 会話 22 | 0.909 | 1.000 | 0.9470 | 35.5ms |

- **復元 = 完全**: `domain="loop"` で loop 18 probe は M3.0 単独取込時
  (astro 混入前) の R@1 0.611 / R@3 0.611 / MRR 0.6389 に**全 metric 一致で復元**。
  astro 22k annotations の押し下げを分野スコープが完全に遮断した。
- **role の限界の in-run 再現**: `role="corpus"` は nofilter と完全同値
  (0.4895) — corpus 行しか rank 上位に来ない状況では role は何も絞らない。
- **会話側回帰なし**: exclude_roles 経路は 0.947 を維持 (domain 実装が
  既存フィルタを壊していない)。

## 解釈 (honest)

1. **M3 のスコープ設計は二層で完結**: 会話 vs 世界知識 = role (exclude_roles)、
   世界知識内の分野間 = domain。両者の直交で「混載 store 1 つ + クエリ時スコープ」
   という運用が成立する — RAD ~48 分野の全量取込に進む前提条件が揃った。
2. **0.6389 への「完全一致」は出来すぎではない**: domain="loop" に絞った瞬間、
   検索対象は M3.0 の loop annotations と同一集合になる (会話行除外の差はあるが、
   M3.0 で会話行が loop probe の rank を奪った事例はゼロ)。つまり復元は
   トートロジーに近い構造的必然であり、驚くべき結果ではない。本測定の価値は
   「実装がその構造どおりに動く」ことの確認 + 会話側の回帰なし証明にある。
3. **限界 (未解決のまま開示)**:
   - probe が知識分野を自己申告する前提 — 実運用ではクエリ→分野の推定
     (ルータ) が必要になる。これは M3 残課題でなく将来の設計課題。
   - 初出時の値が勝つ per-row 単一値 — 同一アノテーションが複数分野に出る場合、
     後の分野は記録されない。多分野共有語 (例 "the model converges") は
     最初に取り込んだ corpus の domain を持つ。分野横断検索 (domain=None 指定なし)
     では問題にならないが、絞り込み時の取りこぼし要因になりうる。
   - 18 probe / 2 corpus の小規模実証 — ~48 分野全量での再測が次。

## 次の一手

1. ANN 化 (faiss optional extra) — 10 万 annotations 級で総当たり cosine が限界
2. RAD 全量取込 (~48 分野、各 corpus に domain タグ) → M2 (cert × 連結性教師)
