# ARTICLE_SEEDS — 記事化のための気付きログ (append-only)

> 運用 (ユーザー指示 2026-06-12): 作業中の新しい気付きを、後から記事にまとめる材料として
> ここに残す。各エントリ = 気付き / 数値・根拠 (正本へのポインタ) / 効く記事側面
> (feedback_daily_articles_policy の 13 側面)。**自動上書きされない append-only**。
> SESSION_SUMMARY.md は毎ターン上書きされるためここに書く。

---

## 2026-06-12 (セッション: M3 ANN 化 → RAD 全量取込 → M2 設計)

### 1. 「ANN 化 = 速い」は規模の前提を隠している
- **気付き**: faiss HNSW を 23k annotations に入れても速くならない (exact 19.6ms →
  ann 16.9ms)。query latency の支配項が **MiniLM の query encode (~15ms)** で、
  23k の総当たり cosine 自体は数 ms しかないため。ANN の本領は総当たり matmul が
  encode を上回る ~10 万行超から。
- **根拠**: out/rad_ann_check.json / research/textseg1d/M3_ANN_HNSW_2026_06_12.md。
  recall@10 0.9825 (min 0.8)、フィルタ併用 MRR は exact と完全一致。
- **側面**: ベンチ / honest disclosure / 教訓。「最適化はボトルネックの所在を測ってから」
  の実例として鮮明 (Amdahl の法則の retrieval 版)。

### 2. 過去の null 結果が将来の設計判断を救う (honest disclosure の実利)
- **気付き**: 全量取込で doc 内全ペア共起エッジは ~2,000 万エッジ (dict 数 GB +
  save 膨張) になると判明。ここで **M1 の null 結果「cooccur hop は強 encoder
  (MiniLM) で効果ゼロ〜微害」が効いて、躊躇なく group=None (共起なし) を選べた**。
  もし M1 で「連結性が効く」という over-claim を放置していたら、2,000 万エッジを
  抱え込む誤設計をしていた。失敗・null を消さず記録する文化の直接的な配当。
- **根拠**: scripts/rad_full_ingest.py 設計判断 / M1 の honest 訂正
  (CONNECTIVITY_BENCH_CORRECTION_2026_06_11.md) / ROADMAP M1 ⚠️。
- **側面**: honest disclosure / 哲学 / 教訓 / TRIZ (リソースを使わない解決)。

### 3. トップレベル ls の罠 — 規模見積もりが 2.5 倍ずれる
- **気付き**: RAD corpus の規模をトップレベル `ls | wc -l` で見積もると 17.8k docs、
  再帰 glob (`**/*.md`) だと **44,836 docs (2.5 倍)**。corpus がサブディレクトリ
  (cluster_XX/c_YY/docs/) 構造だったため。見積もりを encode 時間 (~90 分 → ~4 時間)
  に反映してから走らせたので事故にならなかった。
- **根拠**: rad_full_ingest.py の corpus_domains() dry チェック実測。
- **側面**: 教訓 / 実装報告。「数える方法そのものを検証する」— 計測の計測。

### 4. RAD 接地 30 分で研究の新規性マップが引ける (AI 駆動研究ワークフロー)
- **気付き**: M2 (cert gate × 会話連結性教師) の設計前に、RAD 49 分野を Explore
  agent に grep させたら ~30 分で差別化軸が確定した: (a) verified continual
  learning = 重複中 (OGPSA/Recovery Guarantees が接近) / (b) graph as supervision
  = 重複高 (GraphWalk/GAAMA) / (c) **会話連結性を教師信号に = 重複低 (キーワード
  "coreference supervision" "discourse structure learning" はコーパス内ヒットゼロ)**。
  → 3 軸の交点が空白と確認してから設計に入った。車輪の再発明防止が「事前 30 分」で
  買える。
- **根拠**: docs/M2_CERT_CONNECTIVITY_DESIGN_2026_06_12.md §1 (調査結果の正本)。
- **側面**: 戦略 / エコシステム / AI 駆動研究の方法論 / 実装報告。

### 5. 研究の circularity 回避 — 教師信号は「自分の実装」でなく「外部事実」に接地する
- **気付き**: M2 で連結性グラフ (自前実装) を教師にすると「グラフ実装が正しい」前提の
  循環論証になる。gold を**会話 JSON の turn 構造という構造的事実**に接地することで
  回避 (turn 境界は実装と独立に存在する)。M1 で連結性グラフ自体が over-claim だったと
  訂正した経緯があるからこそ、この罠に気付けた。
- **根拠**: M2_CERT_CONNECTIVITY_DESIGN_2026_06_12.md §1 注意 / §2.2。
- **側面**: 哲学 / 認知科学 / 教訓。

### 6. dedup が corpus 取込の実質規模を 2 割減らす
- **気付き**: AnnotationStore の store 全体 dedup で、aerospace 71k instances →
  53.7k 新規行 (75%)、agents 45.4k → 36k (79%)。arXiv abstract 定型句
  (「we propose...」等) が corpus 間で共有されるため。ユニーク行数ベースの
  メモリ/時間見積もりは instance 数の 75-80% で引くのが実務的。
- **根拠**: rad_full_ingest.py 実走ログ (out/rad_full_store_progress.json)。
- **側面**: 実装報告 / ベンチ。

### 7. 15.7 GB 物理メモリで 100 万行 store を扱う見積もり術
- **気付き**: RSS 実測 2 点 (53.7k 行 = 764 MB, 89.7k 行 = 910 MB) から増分
  ~4 KB/行を抽出し、100 万行 ≈ 4.5 GB 平常 + save 瞬間ピーク (JSON dumps 一時
  文字列 + 行列倍化) +2-3 GB と外挿 → GO 判断。保険として 200k 行ごと checkpoint
  save + --resume (OOM 死しても再開可能) を実装してから走らせた。
  「外挿 + 保険」のセットで不確実な長時間ジョブを安全に回す。
- **根拠**: rad_full_ingest.py (checkpoint/load_progress) / 実走ログ。
- **側面**: 実装報告 / 教訓 / ユーザー体験 (家庭用 PC スペックでの研究)。

### 8. 「会話 35 turns = 122 annotations」の小ささ自体が設計を駆動する
- **気付き**: M2 の会話教師はわずか 122 annotations (境界率 0.281)。この小ささゆえ
  (1) 統計は seed 族 (地形族) で稼ぐ realce 方式を踏襲、(2) 経験 gate の観測
  ホライズン T=64 と系列長が同オーダー = **経験 gate に有利寄りの設定**であることを
  事前開示に回した (不利設定で sound cert が勝てば主張が強くなる)。データの小ささを
  「弱点の隠蔽」でなく「設計の制約条件 + 主張の強化材」に変換する。
- **根拠**: M2_CERT_CONNECTIVITY_DESIGN_2026_06_12.md §2.5。
- **側面**: honest disclosure / 哲学 / 技術設計。
