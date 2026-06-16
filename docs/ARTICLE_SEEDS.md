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

### 16. 採用する「頂点」自体が発散境界上にいる (best_rho 1.000)
- **気付き**: M2.1 seed 0 無 gate で、fitness 最良 gene (= 実運用なら採用する個体)
  の empirical_rho がちょうど **1.000**。archive の 94% が発散というだけでなく、
  **教師が選ぶ頂点そのものが安定境界上**にある。直感的解釈: 表現力 (CE を下げる
  ダイナミクスの豊かさ) は収縮性の境界near傍で最大化される — 教師に従うほど危険に
  近づく構造。cert_inf の採用 gene も rho_max 0.993-0.998 と境界に張り付く
  (ただし証明付きで 1 未満) のが対になる観測で、「**進化は gate があってもなくても
  境界を目指す。違いは『境界のどちら側か』だけ**」という記事の核になる一文が取れた。
- **根拠**: out/m2_gate_cmp.log seed 0 / M2_SMOKE_2026_06_12.md の rho_max 行。
- **側面**: 哲学 / honest disclosure / ベンチ / 認知科学 (探索と安全の幾何学)。

### 15. fail-closed gate は「最初の admit」を設計しないと空転する
- **気付き**: cert_inf (µs 判定) に切替えても MAP-Elites が空転した。実測で
  **random init の cert_inf admit = 0/200** — admit 領域 (W の行絶対値和 < 1 ≒
  W≈0 近傍) は一様 init からはほぼ確率ゼロ。gate を速くしても「最初の合格者」が
  いなければ探索は永遠に始まらない。解 = **W→0 反復縮小 fallback** (W=0 で
  J=diag(decay) は必ず admit、実測 200/200)。これは evolve() に既にあった
  known-safe fallback 規律の MAP-Elites 移植漏れ — **安全機構は『規律』であって
  個別実装の『最適化』ではない**から、新しい探索器に載せ替えるたびに規律ごと
  移植する必要がある。#13 (実効コスト) と合わせ、「gated 探索の 2 大落とし穴 =
  コスト爆発と空転」として記事 1 本になる。
- **根拠**: m2_connectivity_poc.py の fallback commit / 検証 0/200 → 200/200。
- **側面**: 技術設計 / 教訓 / TRIZ (事前対策原理)。

### 13. sound gate のコストは「判定 1 回の速さ」でなく「reject 率 × resample 構造」で決まる
- **気付き**: cert_sdp (1 判定 ~数百 ms) を MAP-Elites の gate にしたら smoke が
  1 時間級に停滞 (13 分で seed 0 すら未完)。原因は判定単体でなく、**探索分布の
  ~94% が発散側 → gate がほぼ全 reject → resample で判定回数が数千回に膨張**という
  掛け算。µs 判定の cert_inf への切替で解決 (sound 性は ladder 共通)。本測定では
  包含関係 (inf admit → sdp も admit) を使う inf-first cascade を設計する。
  一般形: **gate の実効コスト = 判定コスト × 判定回数で、判定回数は探索分布と
  通過率の関数** — gate 単体のマイクロベンチでは見えない。
- **根拠**: m2_connectivity_poc.py v2→v3 (commit diff) / 実測 CPU 13.9 分停滞。
- **側面**: 技術設計 / 実装報告 / 教訓。

### 14. 「学習できる × 安全を選ばない」の同時観測 (M2 v2 seed 0)
- **気付き**: readout v2 で T1 (turn 境界予測) は train CE 0.5186 < floor 0.6269、
  held-out 0.4386 < floor 0.5495 と**初めて床を破り学習が成立**。同時に無 gate
  archive は 65 gene 中 61 (94%) が ρ≥1 のまま。つまり**会話教師は「学習可能な
  信号」だが「安全な gene を選ぶ圧」を全く持たない** — capability と guarantee の
  直交性が 1 つの smoke で同時に観測された。M2 本測定の主張の縮図。
- **根拠**: out/m2_poc.log (v2 seed 0) / m2_connectivity_poc.json (v3 完走後に確定)。
- **側面**: honest disclosure / ベンチ / 哲学。

### 12. run_in_background はセッションと運命を共にする — 長時間ジョブは detached へ
- **気付き**: エージェント環境の「バックグラウンド実行」はセッション終了 = プロセス死。
  3.5 時間の encode ジョブが**セッション切替で 2 回連続死亡** (計 ~25 分の encode 損失
  + checkpoint 不発)。OS レベルの detached プロセス (PowerShell `Start-Process
  -WindowStyle Hidden` + ログ/エラーのファイルリダイレクト) に切り替えて解決。
  シェルの `&` やエージェントの background とは生存スコープが違う — 「ジョブの寿命 >
  セッションの寿命」なら detached 一択。監視はログファイル経由で行う。
- **根拠**: 2026-06-12 実損 2 回 (b73itrccl / btve2wjrh の死亡ログ)。
- **側面**: 教訓 / 実装報告 / ユーザー体験 (AI 駆動開発の実務的罠)。

### 11. 「floor を仮説族に包含させる」— 識別力設計の一般原理
- **気付き**: M2.0 readout v1 (X 空間 centroid + 分離スケール β) は「定数予測 (クラス
  事前)」を表現できない族で、**最適化しても train CE 0.9022 > floor 0.6269** という
  病理を起こした。v2 で log-prior bias を加え「クラス重心が分離しない → 事前予測 =
  floor」を族に包含させると、自明 gene (decay=1, s≡0) で **CE = floor 厳密一致
  (Δ −0.0000)** を検証できた。床が踏めない仮説族は改善量を測れない —
  「ベースラインを族に包含させてから最適化する」は識別力設計の一般原理。
  realce の centroid readout 修正と対になる 2 例目で、パターンとして記事化できる。
- **根拠**: m2_connectivity_poc.py v1→v2 (commit diff) / floor 包含の検証ログ。
- **側面**: 技術設計 / 教訓 / 認知科学 (測定の妥当性)。

### 10. 無 gate archive 69/69 全部 ρ≥1 — 会話教師の危険性の初観測 (要 v2 確定)
- **気付き**: M2.0 smoke v1 seed 0 で、無 gate MAP-Elites archive の **69 gene 全部が
  empirical_rho ≥ 1 (max 3.088)**。random init (W~U[-2,2]) の大半が発散側なのは
  想定内だが、fitness 選択 (会話教師) がそれを**全く淘汰しない**ことの初観測。
  「会話で賢くなる方向」と「力学的に安全な方向」が無相関 (または逆相関) なら、
  M2 の guarantee 主張 (sound cert が要る) の核になる。v2 readout での再測で確定させる。
- **根拠**: smoke v1 seed 0 ログ (rho_max 3.088, rho>=1: 69/69)。
- **側面**: honest disclosure / ベンチ / 哲学 (capability と safety の直交性)。

### 9. checkpoint は「量」と「時間」の二軸で切る (実損から)
- **気付き**: 全量取込の checkpoint を「200k 行ごと」のみにした結果、セッション死亡で
  2 corpus 分 (89.7k 行、~15 分の encode) を全損した。行数閾値は save コストの
  amortize を最適化するが、**失うものは時間に比例する**。「100k 行 OR 15 分の早い方」
  の二軸に修正。長時間ジョブの checkpoint 設計は「コスト最適」でなく「損失上限」で
  決める。
- **根拠**: rad_full_ingest.py の CHECKPOINT_EVERY_SEC 追加 (2026-06-12)。
  実損 = aerospace 513.8s + agents 380.6s の再 encode。
- **側面**: 教訓 / 実装報告。

### 8. 「会話 35 turns = 122 annotations」の小ささ自体が設計を駆動する
- **気付き**: M2 の会話教師はわずか 122 annotations (境界率 0.281)。この小ささゆえ
  (1) 統計は seed 族 (地形族) で稼ぐ realce 方式を踏襲、(2) 経験 gate の観測
  ホライズン T=64 と系列長が同オーダー = **経験 gate に有利寄りの設定**であることを
  事前開示に回した (不利設定で sound cert が勝てば主張が強くなる)。データの小ささを
  「弱点の隠蔽」でなく「設計の制約条件 + 主張の強化材」に変換する。
- **根拠**: M2_CERT_CONNECTIVITY_DESIGN_2026_06_12.md §2.5。
- **側面**: honest disclosure / 哲学 / 技術設計。

## 2026-06-13 (セッション: RAD コーパス拡張ローテーション継続)

### 17. API が全部死んでも corpus は「構造化 fallback」で前に進める
- **気付き**: `corpus2skill` の要約器は ANTHROPIC org disabled で全滅、さらに今回は
  `OPENAI_API_KEY` も未設定だった。それでも `SKILL.md` の frontmatter・子クラスタ・
  docs title・頻出語だけで `Overview / Key Knowledge / When Useful / Navigation`
  を**決定論的に再生成**すれば、少なくとも「辿れる corpus」は完成する。LLM 要約が
  なくても探索導線は壊さずに済む、という運用上の保険が取れた。
- **根拠**: `self_evolving_agents_corpus_v2.staging` の `_STAGING_META/DECISIONS.md`
  (環境依存の別ドライブ staging 出力)、
  `metadata.json` (`summaries_generated: 0`)。
- **側面**: 実装報告 / 教訓 / AI 駆動研究ワークフロー。
- **2026-06-16 追記**: その後の指揮者セッション実測では `ANTHROPIC_API_KEY` は present かつ valid と確認済み。ここでの `ANTHROPIC org disabled` は 2026-06-13 時点の観測であり、現在状態の正本は `docs/next_plan.md` を参照する。fallback を保険として持つ教訓自体は維持。

### 18. 広い topical query は recall を稼ぐが、leaf cluster にノイズとして返ってくる
- **気付き**: `self-evolving` / `reflection` / `test-time training` 系で 16 クエリを広く張ると、
  807 docs まで一気に集まる一方、少数の葉クラスタに材料・天文・VQA など周辺分野が混ざる。
  つまり topical corpus 作成では「まず recall で広く拾い、人間レビューで precision を
  戻す」二段運用が現実的。完全自動で precision まで仕上げようとすると再実行コストが高い。
- **根拠**: `self_evolving_agents_corpus_v2.staging` の `_STAGING_META/queries.txt` と
  `_STAGING_META/DECISIONS.md` (いずれも環境依存の別ドライブ staging 出力)。
- **側面**: 教訓 / 実装報告 / honest disclosure。

## 2026-06-16 (セッション: self_evolving_agents rerun query 精密化)

### 19. precision 改善 query は「ノイズ除去」だけでなく flagship 回収の監視が必要
- **気付き**: `Reflexion` や `AI Scientist` のような有名軸は、`ti:` tightening で
  cross-domain ノイズを減らせる一方、**派生研究の量が増えるほど本命論文が検索順位に埋もれる**。
  実際に `ti:"AI Scientist"` 系は domain-specific 派生が先に大量ヒットし、flagship
  本体を確実に残すには `ti:"The AI Scientist"` / `ti:"The AI Scientist-v2"` の
  専用回収行と lightweight probe が必要だった。precision チューニングは
  「病気を見つけた」だけで治療完了と見なしてはいけない、という honest-disclosure の
  実例。
- **根拠**: `docs/next_plan.md` の rerun 前ゲート記録、
  `D:\docs\self_evolving_agents_corpus_v2.staging\_STAGING_META\queries_refined_candidate.txt`、
  同 `DECISIONS.md` の lightweight probe メモ。
- **側面**: honest disclosure / 教訓 / 戦略 / 実装報告。

### 20. query 1 行の正しさより「query file 全体のふるまい」を疑うべき
- **気付き**: `ti:Reflexion` 完全クエリや `The AI Scientist-v2` 単独行が各 flagship を
  1 件回収できても、それだけでは **query file 全体としての最終 precision / recall 改善**
  は証明できない。単独 probe は局所的な壊れ方検知には効くが、複数 query の重複・順位埋没・
  周辺分野混入は本 rerun 後の before/after 比較でしか確定しない。検索戦略の評価単位は
  「1 行」ではなく「query set + downstream cluster quality」だと明示できた。
- **根拠**: `docs/next_plan.md` の `queries_refined_candidate.txt` ゲート注記、
  `self_evolving_agents_corpus_v2_stopwordcheck` の off-topic 混入観測、
  staging 側 `_STAGING_META/DECISIONS.md`。
- **側面**: honest disclosure / ベンチ / 教訓 / 戦略。

## 2026-06-16 (セッション: verified_safe_learning v1→v2 migration)

> 注: #21〜#28 は同一 migration 論点の別アングル。記事ドラフト化フェーズでは 1〜2 本へ統合前提で扱う。

### 21. 知識資産の v1→v2 移行は「中身」より入口契約を壊さないことが本体
- **気付き**: `verified_safe_learning` の v2 staging は 818 docs / 64 clusters の
  階層 corpus として豊かだが、既存 live v1 は flat `SKILL.md` 起点で、
  publish 対象 `D:\docs\<topic>_corpus_v2` を直接使う `raptor-rad-ingest` は
  **`D:/docs/<topic>_corpus_v2/SKILL.md`** を入口契約として前提化していた。つまり
  v2 化の難所は「情報量を増やすこと」より、
  **人間とツールがどの entrypoint を読みに来るか** を壊さず移行すること。内容差分だけ見て
  rename すると、知識そのものは改善しても運用上は退行しうる。
- **根拠**: `docs/next_plan.md` の verified_safe_learning publish 判断メモ、
  `D:\tools\raptor\libexec\raptor-rad-ingest` の `D:/docs/<topic>_corpus_v2/SKILL.md`
  前提、`D:\docs\verified_safe_learning_corpus_v2` と
  `D:\docs\verified_safe_learning_corpus_v2.staging` の構造比較。
- **側面**: 実装報告 / 教訓 / エコシステム / ユーザー体験。

### 22. 互換 shim は「目次の複製」ではなく「説明 1 段落 + 新入口リンク」で足りる
- **気付き**: `raptor-rad-ingest` が top-level `SKILL.md` から実際に使うのは
  top-level `description:` と `collected:` で、frontmatter 後の最初の非見出し段落は
  `description:` 欠落時のフォールバックだった。つまり hierarchical v2 へ移るときの
  互換 shim は、旧 v1 の 97 ノート一覧を再現する必要はなく、**列 0 の
  `description:` と frontmatter 内の `collected:` を満たしたうえで**「v2 へ移行した」という
  1 段落の説明と `INDEX.md` への明示リンクを置けば、人間向け導線は十分になる。
- **根拠**: `D:\tools\raptor\libexec\raptor-rad-ingest` の `_read_description()` /
  `_read_collected()` 実装、`docs/next_plan.md` の verified_safe_learning shim メモ、
  既存 live `SKILL.md` と staging `INDEX.md` の先頭比較。
- **側面**: 実装報告 / 教訓 / 戦略 / ユーザー体験。

### 23. 後方互換 migration は「何を残すか」より「何を観測して退行判定するか」を固定すべき
- **気付き**: `verified_safe_learning` の中間案で本当に守りたいのは、旧 v1 の
  97 ノート一覧を top-level に複製することではなく、**publish 後も `rad-ingest`
  が `RAD_INDEX.md` を `(no SKILL.md)` へ退行させないこと** と、人間が `SKILL.md`
  から `INDEX.md` へ 1 hop で辿れることだった。互換 migration は「残す内容」を
  議論するより先に、**退行判定の観測点を 2-3 個に固定する**方が人間ゲートを通しやすい。
- **注記**: #21 / #22 と同テーマの別アングル。記事ドラフト化する段階では統合前提。
- **根拠**: `docs/next_plan.md` の中間案チェックリスト、`D:\tools\raptor\libexec\raptor-rad-ingest`
  の reindex 契約、`D:\docs\verified_safe_learning_corpus_v2` と staging `INDEX.md`
  の構造比較。
- **側面**: 教訓 / 実装報告 / 戦略 / ユーザー体験。

### 24. fail-closed な migration では「副作用を隔離して落とせる検査」を先に設計する
- **気付き**: `verified_safe_learning` の中間案では、`--reindex` を打つ前に
  frontmatter フェンス数、frontmatter 内の `description:` / `collected:`、`INDEX.md`
  1 hop 導線、`INDEX.md` 実在、本文側の想定外 `collected:` 出現などを事前フィルタで
  見たうえで、isolated dry-run を 1 回流せば、かなりの割合の壊れ方を共有 index 無変更の
  まま弾ける。fail-closed な移行は、本番反映後の観測より先に**副作用を temp 配下へ隔離して
  落とせる検査列**を作る方が実務的。特に
  本文側 `collected:` 検査は brittle な split/固定行数ではなく、frontmatter フェンスを
  先に確定して本文区間だけを見る方が壊れにくい。
- **根拠**: `docs/next_plan.md` の publish 前隔離チェック例、
  `D:\tools\raptor\libexec\raptor-rad-ingest` の `RAD_INDEX.md` 再生成副作用、
  live `SKILL.md` と staging `INDEX.md` の先頭比較。
- **側面**: 教訓 / 実装報告 / 戦略 / ユーザー体験。

### 25. migration の人間ゲートは「どの案が正しいか」より「どの破壊半径を受け入れるか」で切る
- **気付き**: `verified_safe_learning` の v1→v2 移行 3 択は、機能差の比較というより
  破壊半径の比較として書いた方が人間判断が速い。最小リスク案は live 名を守る代わりに
  二重系を残し、中間案は live 名を v2 に進めつつ shim + static gate で entrypoint 契約を
  fail-closed に保ち、最大変更案は `rad-ingest` 側改修まで含めて構造統一を優先する。
  つまり migration ゲートは「どの世界観が美しいか」ではなく、**どの半径の変更まで今この場で
  承認できるか** に翻訳した方が詰まりにくい。
- **根拠**: `docs/next_plan.md` の verified_safe_learning 3 択比較、shim 草案、
  static gate と隔離チェック例、RAD / hacker corpus の fail-closed / compatibility
  既存教訓。
- **側面**: 教訓 / 戦略 / 哲学 / ユーザー体験。

### 26. fail-closed な gate は文字列一致で閉じず、実消費者の isolated dry-run で閉じる
- **気付き**: `verified_safe_learning` の shim 検査で、frontmatter の
  `description:` / `collected:` や `INDEX.md` 導線を grep するだけでは、
  YAML 構文破損や実消費者との読取ズレを完全には拾えない。最終的な go/no-go は
  **isolated copy を別 `docs-root` に置き、`rad-ingest --reindex --docs-root <temp>` を
  1 回だけ流して `RAD_INDEX.md` の `verified_safe_learning_corpus_v2` 行が退行しないこと**
  で閉じる方が、文字列 gate より監査可能で fail-closed になる。
- **根拠**: `D:\tools\raptor\libexec\raptor-rad-ingest` の `--docs-root` / `write_index()`
  実装、`docs/next_plan.md` の static gate 改訂、verified-safe-learning 側の
  entrypoint migration 論点。
- **側面**: 教訓 / 実装報告 / honest disclosure / 戦略。

### 27. dry-run は「別パスに置く」だけでは足りず、「消費者が期待する親ディレクトリ構造」を再現する必要がある
- **気付き**: `rad-ingest --reindex --docs-root ...` の isolated dry-run は、
  staging を別場所へコピーするだけでは不十分で、**`--docs-root` が走査するのは
  `*_corpus_v2` ディレクトリの親** だと意識して temp root を組まないと、検査自体が
  空振りする。dry-run の信頼性は「隔離されているか」だけでなく、**実消費者が期待する
  ディレクトリ形をそのまま再現しているか** に依存する。
- **根拠**: `D:\tools\raptor\libexec\raptor-rad-ingest` の `--docs-root` /
  `write_index()` 実装、`docs/next_plan.md` の isolated dry-run 手順。
- **側面**: 教訓 / 実装報告 / 戦略 / ユーザー体験。

### 28. rollback を書くなら、rollback 素材をいつ作るかまで手順化しないと fail-closed は閉じない
- **気付き**: `verified_safe_learning` の publish 手順で「退行したら `RAD_INDEX.md` を戻す」とだけ
  書いても、**その退避コピーをいつ作るか** がチェックリストに無いと運用時に抜けやすい。
  しかも本件では index だけでなく live corpus 本体の置換も起きるため、rollback 素材は
  `RAD_INDEX.md` 単体では足りず、**旧 live corpus ディレクトリも同じタイミングで退避**
  しておく必要がある。fail-closed な rollback は、復旧方針そのものより先に、
  **本番 `rename/reindex` の直前に index と corpus 本体の両方の退避を作る**
  ところまで手順化して初めて閉じる。
- **根拠**: `docs/next_plan.md` の verified_safe_learning publish 前後チェックリスト、
  `RAD_INDEX.md` rollback 行、RAD / hacker corpus の auditability・fail-closed 既存教訓。
- **側面**: 教訓 / 実装報告 / honest disclosure / ユーザー体験。

## 2026-06-16 (セッション: article seed collector 実装確認)

### 29. 収集規約は「想定」で運用せず、collector 実装を1回読むだけで曖昧さが消える
- **気付き**: `ARTICLE_SEEDS.md` の運用で曖昧だった「同日複数セクションは衝突するか」「何が集約対象として通るか」は、`collect_research_seeds.py` を実読すると即座に解消した。実装上の **parser 最小条件** は、`## YYYY-MM-DD` を date key とし、`###` 見出し配下で `**気付き**` または `**側面**` のどちらかに**同一行で非空の値**があることだった。したがって同日複数セッションは date 単位で併存集約される。一方で、これはあくまで collector が受理する最小条件であり、**producer 契約としての seed 規約 (`気付き` + `根拠` + `側面`) を置き換えるものではない**。特に `根拠` は parser 非強制でも、記事ドラフト化と監査のために契約上は維持すべき必須項目だと分かった。運用規約の曖昧さは、推測を積み重ねるより collector 本体を一度読む方が安い。
- **根拠**: 2026-06-16 に観測した `D:\projects\fullsense\tools\collect_research_seeds.py` の `DATE_RE` / `ENTRY_RE` / `flush()` 実装（repo 外・dirty 作業木なので再利用前に要再取得）、`docs/next_plan.md` の article seed 規約メモ更新。
- **側面**: 教訓 / 実装報告 / 戦略 / エコシステム。

### 30. append-only の知見ログでは、後日 supersede した観測も「削る」のではなく新しい注記として残すべき
- **気付き**: `ARTICLE_SEEDS.md` を append-only として運用するなら、後日状態が変わった観測を「いまは古いから」と削るのは筋が悪い。正しい扱いは、元の観測を残したまま、**新しい事実を別の追記として supersede させる**ことだった。たとえば #17 の `ANTHROPIC org disabled` は 2026-06-13 時点では事実で、その後 2026-06-16 に `ANTHROPIC_API_KEY` が present かつ valid と再確認された。この種の変化は過去 entry の削除ではなく、append-only の新しい注記として残した方が監査しやすい。
- **根拠**: `docs/next_plan.md` の `ANTHROPIC_API_KEY` valid 記録、同ファイルの article seed append-only 方針、`docs/SESSION_SUMMARY.md` の再開ポインタ。
- **側面**: 教訓 / honest disclosure / 戦略 / エコシステム。
- **注記**: #17 の 2026-06-16 追記は旧形式の supersede 注記として残しつつ、以後の supersede は同様のインライン改変を増やさず、numbered seed の append で行う。

### 31. artifact 名が同じでも producer と consumer の契約が 1 bit ずれると round-trip は壊れる
- **気付き**: `prepare` と `probe` の両方が `<manifest>.bundle.json` を書いていても、それだけでは互換性は保証されない。実際に今回、`prepare` は extras-only の bundle (`includes_base=false`)、`probe` は base+extras の bundle (`includes_base=true`) を生成していたため、**同じ manifest/bundle という見た目でも consumer 側 verify は必ず fail** した。重要だったのは artifact 名の統一ではなく、**契約フィールド `includes_base` の意味を producer/consumer が共有し、verify が bundle の自己申告に追従すること**だった。値を一律に揃えるより、「この bundle は base を含む/含まない」を双方が同じ意味で解釈できるかが round-trip を決める。
- **根拠**: `docs/next_plan.md` の 2026-06-16 追記（prepare bundle mismatch の修正ログ）、`src/llcore/lm/corpus.py` の `build_utf8_corpus_bundle()` / `verify_corpus_manifest_bundle()`、`tests/unit/test_lm_cli.py` の prepare manifest round-trip 回帰 **(既存回帰として存在を参照、今回は未実行)**。
- **側面**: 教訓 / honest disclosure / 実装報告 / 戦略。

### 32. fail-closed provenance は「書ける」だけでなく「読む側も再検証して止まれる」まで閉じて初めて成立する
- **気付き**: manifest 横に provenance JSON を出すだけでは、監査可能なようでいて実際には drift を見逃す。今回の bundle hardeningで効いたのは、producer 側の sidecar / bundle 生成よりも、**consumer 側 (`resolve_extra_corpus_files`) が sibling bundle を再計算して manifest 本体・base 側 drift を consume 時点で fail-closed に止める**ようにしたことだった。さらに比較から絶対 path (`files[].path`) を外し、**内容由来フィールド (各 file の `sha256` / `chars` / `vocab_size`、`combined`、`bundle_sha256`) だけで判定する**ようにしたことで、配置移動のような非本質差分は許しつつ、内容 drift だけを止める監査線になった。
- **根拠**: `docs/PROGRESS.md` / `docs/next_plan.md` の bundle verify 追記、`src/llcore/lm/corpus.py` の `verify_corpus_manifest_bundle()`、`tests/unit/test_lm_corpus.py` の manifest/base drift reject と stale absolute path 無視、`tests/unit/test_lm_cli.py` の probe-manifest drift reject **(既存回帰として存在を参照、今回は未実行)**。
- **側面**: 教訓 / 実装報告 / honest disclosure / エコシステム。
