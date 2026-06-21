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
- **2026-06-21 一次再確認**: nas_pareto の needle/2048 resume 走を harness の `run_in_background=true` で起動したら、**ポーリング用シェルのターン終了で kill**(log/err が完全に空=Python 例外でなく外部 kill が決め手)。同一コマンドを `Start-Process -WindowStyle Hidden -RedirectStandardOutput/Error -PassThru` で**完全 detached** 再起動 → 生存継続を確認。この seed の教訓が別ジョブ・別月で再現=「ジョブ寿命>セッション寿命なら detached 一択」の汎用性が強化された。**検出シグナル=background ジョブが消えたとき log/err が空なら外部 kill を疑う(traceback があればクラッシュ)**。
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

---

## 2026-06-17 (セッション: メモリ効率 pivot 第二歩 (a)mmap + (b)int8 実測)

> 正本ドキュメント = `docs/MEMORY_EFFICIENCY_FINDINGS.md`(3 本柱の全数値・表・honest 留保)。
> 以下は記事化のための気付き要約。出力 JSON = `out/mmap_weights_poc.json` /
> `out/int8_quant_footprint*.json`(shakespeare/multi_smoke/realp1)。

### 33. mmap の load 時メモリは「ほぼ固定コスト」— だから大きいモデルほど効く
- **気付き**: `torch.load(mmap=True)` + `load_state_dict(assign=True)` で重みを file-backed の
  まま割り当てると、load 直後の ΔRSS は **モデルサイズによらず ~1.4–1.5 MB のほぼ一定**(mmap セット
  アップ + メタデータ unpickling の固定コスト)。よって param 7.73 MB の smoke では ×0.218 だが、
  53.91 MB の realp1 では **×0.028(load 時 RSS を ~2.8% に遅延)**。「mmap で省メモリ」を漠然と
  言うより、**固定コスト構造 → 大モデルほど相対効果大**という規模則が本質。eager は load 時に
  ΔRSS ≈ モデルサイズを即全載するのと対照的。全バイト touch すると mmap も ~51.5 MB へ伸び
  =「使った working set の分だけ fault-in」。forward logits は eager と完全一致(max|Δ|=0.0)。
- **根拠**: `scripts/mmap_weights_poc.py`(別プロセス隔離計測)/ `out/mmap_weights_poc.json` /
  `docs/MEMORY_EFFICIENCY_FINDINGS.md` (a)。realp1 で 2 回再測 ×0.028/0.027 で安定。
- **側面**: 技術設計 / honest disclosure / 教訓 / TRIZ(制約=少 RAM を「遅延と固定コスト」で反転)/
  ユーザー体験(「自宅 PC で動く」の裏付け)。
- **honest 留保**: 恩恵は部分 working set / ページキャッシュ共有 / コールド起動遅延。全載ワーク
  ロードでは最終 RSS は eager に近づく(touch 行が示す)。RAM 超モデルの検証は別途要実測。

### 34. int8 weight-only は char-LM でも「約 4× 圧縮・品質ほぼ無劣化」が再現する
- **気付き**: 2-D 重みを対称 int8 量子化すると、重み常駐を **約 3.9×(74–75% 削減)** 圧縮しつつ、
  held-out PPL 劣化は **0.1% 未満**(多くは 0.02% 未満)。英語(vocab 65)・日本語単一本・日本語
  マルチ作品の 3 モデルで一貫。per_channel(行ごと scale)は per_tensor より重み誤差が小さい
  (rel-RMSE ~0.007 vs ~0.013–0.016)。llama.cpp/GGUF の常識を自前 char-LM で実測再確認。
- **根拠**: `scripts/int8_quant_footprint.py` / `out/int8_quant_footprint*.json` /
  `docs/MEMORY_EFFICIENCY_FINDINGS.md` (b)。
- **側面**: ベンチ / 業界比較 / 実装報告 / honest disclosure。
- **honest 留保**: weights-only(activation は fp32)/ dequant fp32 forward の simulated quant=
  **速度は未測定** / footprint は scale と非量子化 1-D params を含む実合計(理想下限 0.25 に対し
  実値 0.25–0.26)。

### 35. 「構造プロット」を「実機計測」へ昇格させる、が pivot 第一歩の型
- **気付き**: recurrent verdict は memory@T 曲線を「config 由来の構造プロット(実測でない)」と
  honest に明記していた。北極星転換後の最初の作業は、新説を足すことではなく **既存の主張を実機
  数値へ格上げ**すること(state_bytes は実測テンソル、GPT KV/attn は解析値+RSS 裏取り)。負け筋
  (capability)を捨て、勝ち筋(memory)を「実証」へ昇格する、という pivot の作法そのもの。
- **根拠**: `scripts/memory_footprint_harness.py` / `out/mem_footprint.json` /
  memory:`project_llcore_memory_efficiency_pivot`。
- **側面**: 哲学 / 戦略 / honest disclosure / 教訓。

### 36. 「メモリで線形か指数か」— ニューラルの賢さは指数的には伸びない(多エージェント検証)
- **気付き**: ユーザー論点「アルゴリズムはメモリ容量に対し線形/指数で性能が伸び、レジーム別の使い分けが
  勝敗を決める」を 8 アルゴリズム族 × 各 2 レンズ敵対検証(26 エージェント)で整理。結論: **ニューラル
  capability がメモリに対し指数的に伸びる族は存在しない**。指数が本物なのは古典計算の 2 か所だけ —
  (a)メモ化/DP(重複部分問題があるとき**のみ**指数時間→多項式=指数的スピードアップ)、(b)modern
  Hopfield(容量が次元 d に指数、ただし footprint→容量は線形・分離条件付き・容量≠知能)。圧倒的多数は
  **べき乗則/対数(劣線形)** — scaling law / MoE / RAG(log(corpus))/ SSM 品質。「emergence で指数的に
  賢くなる」は Schaeffer 2023 の通り**不連続メトリクスの測定アーティファクト**で、「指数の支出×線形の利得」
  の誤読。**完全性批評が overclaim を 1 件検出**: 「大 RAM→大モデル常駐」は無条件には誤り — Beyond-Chinchilla
  ではサービング量が多いほど compute-optimal は“より小さい N×長い学習 D”へ動く(推論回数で割り戻す)。
- **根拠**: `docs/MEMORY_SCALING_STRATEGY.md`(訂正反映済み正本・族別表・regime→primitive 決定則)。
- **側面**: 業界比較 / honest disclosure / 教訓 / 哲学 / 戦略 / 認知科学(emergence の誤読)。

### 37. 量子化 cliff はモデルが大きいほど低ビットに頑健・PPL-only gate は壊れた 2bit を PASS する
- **気付き**: 批評推奨の反証可能実験(int8 ビット幅スイープ {8..2}bit)を 2 モデルで実測。**cliff_then_flat
  を確認**しつつ、(1)**cliff 位置はモデルサイズ依存** — 小 1.36M は 3bit で劣化(+11.6%)・2bit 破綻だが、
  大 11.9M は **3bit でも実用**(+4.8% / top1 -0.7pp)、cliff は 2bit = **大モデルほど低ビットに頑健**
  (冗長性、literature 整合)。(2)**PPL だけの合否は危険** — 11.9M の 2bit は top1 が -13.5pp(28.7%→15.2%
  =半減近く壊れている)のに unigram PPL gate は PASS。→ 合否に hard-capability proxy が要る。(3)**honest
  反証**: 「top1 は PPL より先に劣化する」という事前予想は本データでは不成立、両者 lockstep。誇張せず記録。
- **根拠**: `scripts/quant_bitwidth_sweep.py` / `out/quant_bitwidth_sweep*.json` /
  `docs/MEMORY_EFFICIENCY_FINDINGS.md` (b')。
- **側面**: ベンチ / honest disclosure / 教訓 / 実装報告 / 業界比較。

### 38. 「使える RAM < モデル」でも動く — working-set 上限で mmap の RAM 超を実機実証
- **気付き**: 「仮想メモリでモデルを回す」は漠然と言われるが、実際にどう成立するかを実機で示した。
  130M params(**522MB**)のモデルを、Windows の **working-set hard max を 358MB(= モデルの 68%)に
  設定したプロセスで forward 完走**(`SetProcessWorkingSetSizeEx`・強制成功・peak WS 357.7MB ≤ 上限)、
  しかも **出力(logits)は無制限実行と完全一致**。機構: read-only な mmap ページは clean なので、上限
  超過時に OS が**破棄**(pagefile 書込み不要)し、必要になったら**再 fault で disk から読み直す**
  (llama.cpp 流)。つまり「**使える物理 RAM がモデルより小さくても、正しく動く**」。さらに int8 量子化で
  ディスクは 0.25×(4x 縮小)= 置くページ自体が減る。これが pivot memo の「✅ 本筋 = working set を
  小さく・予測可能に」の実証。
- **honest 留保**: avail RAM が限られる本マシンでは「RAM 総量超の巨大モデル」ではなく「working-set 上限 <
  モデルサイズ」で同性質を実証。hard-max 強制可否は環境依存なので `cap_set_ok` と実測 peak をそのまま記録。
  int8 は disk/load まで(per-layer streaming dequant forward は将来)。
- **根拠**: `scripts/mmap_ram_exceed_poc.py` / `out/mmap_ram_exceed_poc.json` /
  `docs/MEMORY_EFFICIENCY_FINDINGS.md` (a')。
- **側面**: 技術設計 / 実装報告 / honest disclosure / ユーザー体験(自宅 PC で大モデル)/ TRIZ(少 RAM を
  仮想メモリで反転)/ 未来予測(GPU/大 RAM で跳ねる土台)。

### 39. int8 streaming 推論で「常駐は減るが peak は圧力下でしか減らない」— allocator の壁を honest に
- **気付き**: int8 を resident に保ち forward 内で層ごとに dequant→即解放する `Int8Linear` で、130M params の
  **常駐モデルを 72% 削減**(dense fp32 538.6MB → stream int8 148.9MB)。だが **圧力(working-set 上限)が
  無いと peak working set はほぼ減らない**(963→882MB)= torch caching allocator が解放した fp32 を OS に
  返さず、transient 活性も乗るため。素朴な「層ごとに捨てれば peak も減る」直感は CPU では崩れる。**削減が
  顕在化したのは working-set 上限 368MB(< dense 常駐 539MB)で stream が完走した時**(出力は dense と完全
  一致)。教訓: メモリ最適化の効果は「常駐の下限」と「圧力下で動くか」に出るのであって、無風時の peak には
  出ない。計測する指標を間違えると「効果なし」と誤読する。
- **根拠**: `scripts/int8_streaming_infer.py` / `out/int8_streaming_infer.json` /
  `docs/MEMORY_EFFICIENCY_FINDINGS.md` (c)。
- **側面**: 教訓 / honest disclosure / 実装報告 / 技術設計 / 認知科学(直感の落とし穴)。

### 40. 「文脈でメモリが膨らむ Transformer / 平坦な recurrent」を実機 peak RSS で裏取り
- **気付き**: harness は state_bytes(実測)+ KV/attn(解析値)までだった。実生成ワークロードを文脈長 T で
  振り、別プロセスで peak working set を実測すると、**T 256→2048(×8)で GPT は peak WS ×2.65(固定 baseline
  ~205MB を引くと文脈コストは ~25→403MB と超線形=attn O(T²) が大 T で支配)/ Recurrent・RWKV は ×1.00
  (定数状態で平坦)**。構造的に決着済みの「recurrent=定数状態 / Transformer=文脈線形(+attn 二次)」を、
  解析値でなく**実機 peak RSS**で示せた。これがメモリ効率北極星で recurrent が勝ち筋である土台の実証。
- **根拠**: `scripts/recurrent_runtime_rss.py` / `out/recurrent_runtime_rss.json` /
  `docs/MEMORY_EFFICIENCY_FINDINGS.md` (0')。
- **側面**: 技術設計 / ベンチ / honest disclosure / 実装報告 / 業界比較(SSM vs Transformer の長文脈)。

### 41. per-group 量子化は低ビットの床を押し下げるが「2bit を安全」には RTN 超が要る
- **気付き**: per-channel(群=行全体)を per-group(行を群ごとに区切り scale)へ拡張すると、低ビット品質は
  **単調改善**(scale 増で footprint は微増)。realp1 2bit で top1 劣化が **full -13.5pp → group32 -5.3pp
  (≈60% 減)**、multi_smoke 2bit は group≤64 が **ppl-gate を救出**(full は FAIL)。**だが strict
  capability-gate(top1 fp32 比 97% 保持)は 2bit では RTN per-group でも届かない**(realp1 group32 で 81.5%)。
  = **3bit が実用床**(realp1 は per-channel で既に cap-gate PASS)、**2bit を安全にするには RTN 超(GPTQ/AWQ の
  誤差補償 or QAT)が必要** — これは GPU/将来課題。「群を細かくすれば 2bit も救える」という期待は CPU・RTN の
  範囲では半分だけ正しい(品質は上がるが gate は越えない)、と honest に確定。
- **根拠**: `scripts/quant_group_compare.py` / `out/quant_group_compare*.json` /
  `docs/MEMORY_EFFICIENCY_FINDINGS.md` (b'')。
- **側面**: ベンチ / honest disclosure / 教訓 / 実装報告 / 業界比較(GPTQ/AWQ/QAT との位置づけ)。

### 42. GPTQ は「重みをわざと不正確にして出力を正確にする」— 誤差補償量子化の逆説
- **気付き**: GPTQ(Frantar et al. 2022)を自前実装して検証すると、2bit で **weight 誤差は RTN より大きい
  (0.61→0.68)のに output 誤差は小さい(78.6→71.2)**。理由: GPTQ が最小化するのは ‖W−Ŵ‖²(重み誤差)
  ではなく **‖(W−Ŵ)X‖²(出力誤差)** で、入力 Hessian H=Σxᵀx を使って「ある列の量子化誤差を、まだ量子化
  していない列へ伝播」させ、出力空間で打ち消す。直感「量子化は重みを正確に近似するほど良い」は誤りで、
  **重みの忠実度を捨てて出力の忠実度を買う**のが正解。これは「何を誤差の指標にするか(weight space vs
  function space)で最適解が変わる」という量子化(ひいては圧縮一般)の核心。校正データ(活性化統計)が
  必要なのも、出力誤差は入力分布に依存するから。
- **根拠**: `scripts/gptq_compare.py`(probe で weight↑/output↓ を確認)。realp1/multi_smoke の cap-gate
  経験値は `out/gptq_compare*.json` / `docs/MEMORY_EFFICIENCY_FINDINGS.md` 参照。
- **側面**: 教訓 / 認知科学(指標の取り違え)/ honest disclosure / 技術設計 / 業界比較。

### 43. 「最新の GPTQ が常に最強」ではない — 極低ビットでは粒度が誤差補償に勝つことがある
- **気付き**: RTN per-channel / per-group / GPTQ の 3 手法を同条件比較。realp1 2bit で GPTQ は RTN を大きく
  改善(top1 劣化 -13.35→-6.38pp ≈ 52% 減)するが、**per-group32 RTN(-5.31pp)が GPTQ-per-channel(-6.38pp)を
  上回った**。= 極低ビット・小モデルでは「**粒度(scale を細かく)> 誤差補償(Hessian)**」になり得る(両者は
  直交・相補的で、GPTQ+per-group が真の SOTA)。さらに 3 手法すべてで **strict cap-gate(top1 fp32 比 97%)を
  2bit では越えられず**、3bit は全手法で楽に PASS = **3bit が PTQ の実用床、2bit は QAT(学習時量子化)領域**、
  という床の位置は手法を変えても動かない(手法は各ビットの damage を減らすだけ)。「新しい手法を足せば床が
  下がる」という期待は外れ、床を動かすには質的に別のアプローチ(QAT)が要る、が自前実測の結論。
- **根拠**: `scripts/gptq_compare.py` / `scripts/quant_group_compare.py` / `out/gptq_compare*.json` /
  `out/quant_group_compare*.json` / `docs/MEMORY_EFFICIENCY_FINDINGS.md` (b'')(b''')。
- **側面**: ベンチ / honest disclosure / 教訓 / 業界比較(PTQ vs QAT)/ 技術設計。

### 44. QAT は PTQ を 2bit で約3倍引き離すが、tiny モデルでは「最後の壁」を越えられない(量子化アーク締め)
- **気付き**: 「2bit は QAT 領域」を自前実証。fake-quant + STE で学習(量子化を見越して重みが適応)すると、
  multi_smoke 2bit で **top1 30.10%(fp32 36.28% の 82.9% 保持)** に達し、**PTQ GPTQ 12.07% / RTN 7.98% を
  +18〜22pp 圧倒(約3倍の保持)**。= 量子化アークの全手法(RTN→per-group→GPTQ→QAT)で 2bit top1 は
  8→12→30% と単調に改善し、**QAT が質的にジャンプ**。**だが strict 97% cap-gate は QAT でも越えられず**
  (82.9%)。教訓2つ: ①「学習時量子化(QAT)は後処理(PTQ)と次元が違う」を数値で確認(誤差補償 GPTQ より
  さらに上)②**それでも tiny char-LM の 2bit には最後の壁が残る**=床を動かす質的アプローチ(QAT)は効くが、
  本当に 2bit を安全化するには「モデル規模 / 学習予算 / 学習可能 scale(LSQ)」が要る。**「新手法を足せば床が
  下がる」期待は、QAT で大きく前進したが完全制覇には至らず**、を honest に確定(アークの締め)。
- **根拠**: `scripts/qat_train.py` / `out/qat_train_2bit.json` / `docs/MEMORY_EFFICIENCY_FINDINGS.md` (d)。
- **側面**: ベンチ / honest disclosure / 教訓 / 業界比較(PTQ vs QAT)/ 認知科学(段階的改善 vs 質的跳躍)。

### 45. PoC を「実 CLI 推論パス」へ昇格 — int8 を量子化して mmap streaming で日本語生成
- **気付き**: メモリ効率の PoC 群(scripts/)を `src/llcore/lm/quant.py`(Int8Linear streaming-dequant /
  save_int8_checkpoint / load_int8_model[mmap])へ昇格し、`llcore.lm` CLI に `quantize` サブコマンド + int8
  対応 generate を配線。実機: `llcore.lm quantize model.pt`(5.5MB→**1.5MB resident, 72.6% 減**)→
  `llcore.lm generate model_int8.pt` が **mmap streaming-dequant で読み込み、コヒーレントな日本語を生成**
  (「僕は大事なんぞです」等の青空調)。= 研究 PoC が「再利用可能 module + 実用 CLI」になった一歩。教訓:
  実験スクリプトと製品コードの境界を越える時、(a)src は import-untyped ignore 不要(mypy strict 通る)
  (b)tied 重み(lm_head/wte)は state_dict 経由で扱う(named_parameters は dedup して欠ける)(c)int8/fp32
  checkpoint は `kind` フィールドで透過判別、の 3 点が配線の勘所。
- **根拠**: `src/llcore/lm/quant.py` / `src/llcore/lm/__main__.py`(quantize subcommand)/
  `tests/unit/test_lm_quant.py`(round-trip 検証)。
- **側面**: 実装報告 / 技術設計 / 教訓 / ユーザー体験(自宅 PC で量子化推論)。

---

## 2026-06-18 (セッション: techno-edge 生成AIウィークリー#147 を一次情報検証 → llcore/portfolio マッピング。ユーザーが Telegram で共有した記事が出発点。5技術すべて is_real=confirmed・honest 監査 verdict=合格)

> ★共通 honest 注記(全エントリに効く・記事化時 必須要件): **各社のベンチ数値はすべて開発元 self-report で第三者再現は未確認**。一次情報(arXiv論文/公式repo/公式blog/HF model card)で実在は確認済みだが、優位主張は各社の評価条件下のもの。記事で横並びにする時は必ずこの一文を本文に入れる(feedback_benchmark_honest_disclosure)。

### 46. 2026年6月、業界が一斉に「小型・低メモリ・ローカル」へ収斂 — llcore pivot が主流の正面に来た(追い風と脅威の両面)
- **気付き**: 同じ週(2026-06)に **Gemma 4 12B(16GB級・Apache 2.0)/ PaddleOCR-VL 0.9B / NVIDIA Cosmos の小バリアント Edge 2B・Nano 16B / Hermes Agent(local self-host)** が出揃い、「小型・低メモリ・ローカル実行」が業界の主流潮流に。llcore の北極星(2026-06-16 転換=メモリ効率・自宅 PC で動く)と FullSense「ローカルこそ AI の居場所」が**トレンドの正面**に立つ=pivot のタイミングは正しかった、の傍証。**だが脅威も同時に確定**: 大手(Google/Baidu/NVIDIA)が高性能小モデルを **Apache/open で無料配布**するため、llcore が「自前で小モデルを育てる」価値は**モデル本体では勝てない**。→ llcore の価値は**モデルではなく「メモリ効率の手法・計測・gate(int8 3.9×/mmap 358MB完走/定数状態×1.00/capability-gate≥97%/cliff のモデルサイズ依存実測)」=再利用可能なインフラ層**に置くべき(pivot 後の実装は既にこの方向)。
- **根拠**: 一次情報= Gemma4 blog.google + HF `google/gemma-4-12B` / PaddleOCR arXiv:2606.03264 + HF / Cosmos3 NVIDIA newsroom + HF。llcore 側= `docs/MEMORY_EFFICIENCY_FINDINGS.md`(3本柱+量子化アーク実測)。**注記: Cosmos は本体 64B(学習20兆トークン)で本質は大規模、Edge2B/Nano16B は小バリアントを持つだけ**(「Cosmos も小型志向」と数えるのは限定付き)。
- **側面**: 業界比較 / 戦略 / 未来予測 / honest disclosure / エコシステム。

### 47. 「OSS が Gemini を超えた」の内訳を疑う — Cosmos3 と PaddleOCR が示す業界のベンチ cherry-pick
- **気付き**: 派手な「proprietary 超え」主張は**負け軸の省略 / 専用ベンチ**で成立していた、を一次情報で実演。**(a) Cosmos 3**: 記事の「自動運転/ロボ制御で Gemini 3.1 Pro 超え」は**半分誤り** — 技報 Table 10 で Driving は 79.3 vs 47.2(Cosmos 勝ち, ただし 3 ベンチのみ母数小)だが **Robotics 57.8 vs 58.2 / General 73.7 vs 77.5 は Gemini が勝つ**。しかも **NVIDIA 公式 press/blog は Gemini 比較を一切せず「open models 内 1 位」と慎重に限定**、Gemini 数値は技報の詳細表にだけ存在。二次情報が拡大解釈した典型。**(b) PaddleOCR-VL-1.6**: 「0.9B が 235B Qwen3-VL・Gemini 3 Pro 超え(OmniDocBench v1.6 で 96.33)」は数値上 true だが **文書解析専用ベンチ上の話**で「汎用知能で超えた」ではない。汎用 VLM は文書解析に最適化されていないため**専用 0.9B が勝つのは構造的に有利**、かつ比較スコアは全て **Baidu 自前測定**。= FullSense ベンチ規律「異常に良い結果は勝った気になる前に内訳を疑う」の生きた教材。
- **根拠**: Cosmos3 技報 Table 10(Driving/Robotics/General/SmartInfra の category-average)/ PaddleOCR arXiv:2606.03264 Table 2(Baidu 側測定)。honest 監査(workflow critic)が「T2V サブメトリクス勝ち(AV/Physics)を使うなら『T2V 全体は open 内 1 位・closed の Veo/Seedance に負け』とペア提示」も要件化。
- **側面**: honest disclosure / 業界比較 / 教訓 / ベンチ。**FullSense の構造的差別化軸**。

### 48. 記憶でキャラ/状態を「定数的に」保つ — MangaFlow の story section memory と llcore 定数状態 recurrent の同じ発想
- **気付き**: 東大+香港科技大(HKUST広州)の **MangaFlow**(arXiv:2605.28173)は、マンガ生成のコマ間キャラ一貫性を **story section memory**(各セクションにキャラ/シーン/オブジェクト参照を紐づけパネル間で再利用する外部記憶)で担保し、ablation で **CIDS 0.619→0.582 / CSD 0.668→0.547** と寄与を実証。= llcore の「記憶で状態を定数的に保持(RWKV/Mamba 定数状態)」+ llive 4 層メモリ訴求の**他ドメイン傍証**になる(記憶機構は LLM だけでなくマルチモーダル生成の一貫性でも効く)。manga-md-poc(宣言的コマ→SVG)とも設計思想が近い先行例。**さらに MangaFlow の自認の限界=「stylized 顔での話者帰属・吹き出し配置が困難」は、bazue index(159 hard case=話者≠中央被写体)が突こうとしている まさにその弱点**=bazue ベンチの存在意義の外部裏づけ。
- **根拠**: arXiv:2605.28173 Table1/Table2 + ablation。**記事化時 必須注記**:(1)「作画」は MangaFlow 自体でなく**外部クラウド拡散モデル(Gemini 2.5 Flash Image / FLUX.2 9B)**が担う制御層、(2)Layout IoU 100%/Coverage 99.98% は**幾何座標で明示配置する設計上ほぼ自明な自作メトリクス**でピクセル生成ベースラインと土俵が違う、(3)**商用サイト mangaflow.studio は本論文と無関係の別製品(混同回避を明記)**、(4)査読前 v1・引用ゼロ・GitHub 公開記載なし。
- **側面**: エコシステム / 認知科学(記憶と一貫性)/ 技術設計 / マルチモーダル。manga-md-poc・bazue・llive に接続。 **本執筆済み (2026-06-21)**: 技術版 `c3-memory-for-consistency.md` + 非エンジニア版 `c3-memory-for-consistency-general.md`。C系。記憶で一貫性=ドメイン横断の普遍骨格(a7定数状態/llive 4層メモリの傍証)、ablation(CIDS 0.619→0.582/CSD 0.668→0.547)、必須注記4点(作画は外部拡散/IoU100%は自作メトリクス/商用サイト無関係/査読前self-report)、stylized顔の話者帰属弱点=bazueベンチ存在意義の裏づけ、を反映。性能SOTA主張でなく方向性傍証。

### 49. 「成長するエージェント」vs「責任を持って成長するエージェント」— Hermes Agent と llive
- **気付き**: Nous Research の **Hermes Agent**(OSS, MIT)は built-in learning loop(①タスク後 skill 自動生成 ②使用中 skill 自己改善 ③記憶永続化 ④FTS5 横断検索 ⑤Honcho user modeling)で「使うほど育つ」を謳い、**llive の『自己進化・4 層メモリ・派生集団進化』と真正面に競合**。だが honest に見ると差別化の余地が明確: Hermes は **learning loop の有効性を示す独立ベンチ・査読論文がゼロ**(arXiv 検索 0 件、効果は公式自称のみ)、かつ二次情報(Pebblous)が**「汚染ループ」リスク(誤 skill を蓄積・再利用)**を指摘。= llive の **Approval Bus + HITL + honest disclosure** は「成長=常に改善ではない」へのまさに責任ある回答。**llive は『成長する』より『責任を持って成長する(誤 skill 汚染を止める)』を前面に**出すべき、という positioning の確定。
- **根拠**: GitHub `NousResearch/hermes-agent`(MIT, 2025-07 作成)+ release v2026.6.5「Surface Release」(Desktop=既存 core の GUI ガワ、新モデル/新FWではない)。**注記: stars 196,554 は GitHub API 実測だが star の質(bot/campaign 由来比率)は未検証** — 「マインドシェア先行」の脅威確度は規模が大きいぶん留保付き。二次記事の「180,000/24,600/4か月」は数値も時期もバラバラ=API 実測を正とする。
- **側面**: 自己進化 / 哲学(責任所在を architecture に)/ 業界比較 / エコシステム。**llive 差別化の核**。 **本執筆済み (2026-06-21)**: 技術版 `c2-responsible-growth.md` + 非エンジニア版 `c2-responsible-growth-general.md`(自己流フォームが固まるスポーツ選手のたとえ)。C系。Hermes 5点learning loop/汚染ループrisk/独立ベンチ0件/star 196554は質未検証/llive Approval Bus+HITLは「責任ある成長」だが llive自身も未証明=設計の賭け、と両者に同じ刃(s2自己監査の作法)を反映。 **2026-06-21 web再照合で訂正**: Hermes は「作成2025-07/star 196,554」と記載していたが誤り → 正しくは「2026-02リリース/star 数万(2026/4 約47k-57k)」。196,554 は他ソース未裏付けで撤回。c2 draft 修正済み(=外部数値を自分でも検証する s2 作法の実地適用)。MIT・learning loop・独立ベンチ欠如は確認済み。

### 50. open weights ライセンスの実務地図 — Gemma4 Apache 2.0 化・Cosmos OpenMDW・Qwen 障壁
- **気付き**: 同週のリリースで **open モデルのライセンスが 3 種に割れた**:(a)**Gemma 4 = Apache 2.0**(Gemma 1/2/3 の独自 Gemma Terms of Use から**変更**=本物の前進、商用障壁ゼロ)(b)**Cosmos 3 = OpenMDW 1.1**(Linux Foundation の model-centric ライセンス=**OSI 認定の古典的 OSS ではない**「open model」ライセンス、商用条件・派生物条項は要条文実読)(c)**PaddleOCR = Apache 2.0**(ただし ERNIE-4.5 ベースで派生ライセンス要確認)。FullSense は Apache-2.0 + Commercial dual-license で、かつ `feedback_qwen_commercial_barrier`(Qwen 依存は商用障壁)を回避したい立場。→ **「Apache 2.0 のローカル実行可能モデルが増えた」は llmesh on-prem hub のローカルバックエンド選択肢を増やす追い風**であると同時に、llcore がモデルで競合しない判断を補強する。
- **根拠**: Gemma4 HF model card(Apache 2.0)/ Cosmos3 HF blog(OpenMDW 1.1)/ PaddleOCR HF(Apache-2.0)。FullSense 側= `feedback_qwen_commercial_barrier` / `project_fullsense_brand`(dual-license)。
- **側面**: 戦略 / ライセンス / エコシステム / 業界比較。llmesh のローカルバックエンド戦略に直結。 **本執筆済み (2026-06-21)**: 技術版 `c1-open-weights-license-map.md` + 非エンジニア版 `c1-open-weights-license-map-general.md`(レンタル品の条件のたとえ)。連載 C系(エコシステム/戦略)を新設。3種分類(Gemma4 Apache2.0化/Cosmos3 OpenMDW=非OSI/PaddleOCR Apache+ERNIE継承注意)/openを3問い(種類・商用条件・派生継承)に分解/llmesh on-prem追い風+llcore非競合の補強/「束縛力は条文・法的助言でない」留保を反映。

### 51. ★連載素材: 強豪5社の手法 × 我々との共通点(専用 doc に詳細)
- **気付き**: ユーザー指示(2026-06-18「強豪のやり方を解析して記事に・特に我々との共通点・記事ネタ集めに情報収集も」)を受け、Gemma4/Cosmos3/PaddleOCR/MangaFlow/Hermes の**手法を一次情報で deep-read → FullSense/llcore との共通点を抽出 → honest 監査**した濃い記事素材を別 doc に用意。記事の核 = 3 グループ(引き算/再利用/責任を持って育つ)。最強の切り口 = 「自宅 CPU の小実験が frontier の何を**再導出**でき、どこが本当に**違う**(差別化)のかを honest に切り分ける」= career-grade で信頼を生む([[feedback_articles_career_advancement]])。
- **正本**: `docs/ARTICLE_MATERIAL_2026_06_competitor_methods.md`(フック3案 / 5社解説 / 共通点マップ[関係タイプ+honest] / 比喩表 / 13側面タグ / 構成案。honest 監査の必須修正=競合数値にラベル・規模差 caveat・我田引水2件格下げ[Cosmos↔llive=loose_analogy / bazue=未検証 shared_problem_framing「裏取り」禁止]・arXiv ID 執筆前確認 を適用済)。立ち位置の土台 = `docs/POSITIONING_VS_LLAMACPP.md`。
- **側面**: 業界比較 / honest disclosure / 技術設計 / 哲学 / 教訓 / 戦略(キャリア)。技術者向け(QIITA_SUMMARY 2-3万字)+非エンジニア向け(QIITA_GENERAL)の2系統で展開可。

### 52. ★記事マスターシードバンク(26本+連載構成案・専用 doc)
- **気付き**: ユーザー指示(2026-06-18「llterm が頑張れるよう記事ネタをいっぱい提供」)を受け、56本生成→編集長が **26本(S級7/A級11/B級8)+連載構成案2案** に統合した master seed bank を専用 doc に用意。honest 監査 verdict=**出版可(条件付き)**。編集長の推し = **S2 cherry-pick 5型(看板)/ A7 制御理論=SSM(クロスドメイン白眉)/ S1 PPL trap(入口)/ B1 負けを見せる(着地)**。
- **正本**: `docs/ARTICLE_IDEA_BANK_2026_06.md`(冒頭に連載レベル恒久 caveat + honest 監査の必須 fix 7件[連載 caveat バナー・S1 過剰一般化修正・thin 4本救済・A10 線引き・アナロジー密度・B6 外挿明記]を記載済。llterm はこれを適用して記事化)。
- **側面**: 全13側面を網羅(特に honest disclosure / 業界比較 / 技術設計 / 哲学 / 教訓 / クロスドメイン / 戦略)。**llterm の選べるメニュー**。

### 53. LSQ(学習可能 scale)で「2bit 制覇」に再挑戦 → +1.1pp しか報われなかった(負けの実データ)
- **気付き**: 量子化アークの締めとして **LSQ(Learned Step Size Quantization, Esser et al. ICLR2020)を自前実装**し、固定 scale QAT(82.9%)を学習可能 scale が越えられるか実測。結果 = **multi_smoke 2bit で top1 30.48% / retention 84.0% = 固定 scale QAT を +1.1pp 上回るが、strict 97% cap-gate には遠く届かず FAIL**。手法系譜 RTN 22%→GPTQ 33%→QAT 82.9%→LSQ 84.0% は単調改善だが、**「手法を上げれば床が下がる」期待は LSQ でもほぼ報われない**。prior-art の予言(LSQ 自身が小モデル SqueezeNext で 2bit -14pt、k-bit scaling law/QiD が「小モデルは冗長性が無く 2bit を吸収できない」)が自前実測で裏取りされた。**床を動かすのは手法でなく規模/学習予算/VQ codebook**(2bit 90%+ は 7B+ でのみ成立)。3bit が PTQ 実用床のまま。
- **根拠**: `out/qat_lsq_2bit.json`(top1 30.48%/retention 84.0%/cap-gate FAIL)/ `scripts/qat_train.py --method lsq`(lsq_quant/LSQLinear/convert_to_lsq, 純粋追加・既存不変)/ `tests/unit/test_qat_train.py`(LSQ 7件, 全12 passed/ruff/mypy green)/ `docs/MEMORY_EFFICIENCY_FINDINGS.md` (d')。prior-art = arXiv:1902.08153 / 2212.09720 / 2411.17691 / EfficientQAT 2407.11062。
- **側面**: 教訓 / honest disclosure / ベンチ / 技術設計 / 業界比較(PTQ→QAT→LSQ の系譜と床の所在)。**B1「負けを見せる」の最新実データ**(=記事 b1-show-your-losses.md に反映済)。

### 54. 負け筋(capability)の機構資産を勝ち筋(メモリ効率)に再配線できる — branch A 統合の honest 設計
- **気付き**: capability の進化探索は `evolution_vs_random().passes=False`(進化≒random)で**負け筋**確定。だが、そこで作った機構資産(minimal_ga + verified-plasticity gate[Z3/SDP の sound 収縮証明]+ falsification harness)は、勝ち筋「メモリ効率」の**「探索を sound にする」用途へ転用できる**。= 機構は捨てず北極星だけ載せ替える(pivot 作法の延長)。**核心 honest**: 「メモリ指標を適応度に」「accuracy×memory スカラ化」「fail-closed 制約付き進化」「検証器を gate に」は**すべて HW-NAS(MnasNet)/多目的 NAS(NSGA-II)/Deb 2000/CEGIS の再導出(既知)**(prior-art confidence high)。誠実な独自は**「進化×sound 収縮 gate×メモリ北極星×recurrent 力学」の狭い四点結合 + 経験 gate(84% false-admit)vs sound cert(0% false-admit)の判別力計測 + 自宅 CPU 再現性**に限る。**branch A は capability を取り戻さない=guarantee 側に価値を置く**(capability と guarantee の直交性, [[feedback_benchmark_honest_disclosure]])。
- **根拠**: 設計正本 `docs/BRANCH_A_CAPABILITY_REMERGE_DESIGN.md`(統合アーキ/再導出 vs 独自/PoC G1–G6/honest 見積もり)。実装着手 = `src/llcore/fitness/memory_objective.py`(`MemoryEfficiencyObjective` スカラ化 + `state_boundedness_footprint`=収縮率 L proxy、verifier 閉形式と同値性 test、純粋追加)+ `tests/unit/test_branch_a_memory_fitness.py`(7 passed/ruff/mypy green)。**P1 留保=footprint は state-boundedness proxy で実 RSS footprint でない**(実 footprint を fitness にする P2 は将来)。prior-art = MnasNet/HAQ/NSGA-II/Deb2000/CEGIS/Dohare2024。
- **側面**: 戦略 / 哲学 / 教訓 / 業界比較 / honest disclosure / 技術設計。非エンジニア向けフック=「負けた研究の道具を勝ち筋に使い回す」物語性、技術者向け信頼=「メモリ指標を適応度にするのは NAS の常識・独自は狭い」。 **本執筆済み (2026-06-21)**: 技術版 `b7-rewire-losing-tools.md` + 非エンジニア版 `b7-rewire-losing-tools-general.md`(釣り道具の使い回しのたとえ)。capability負け筋→機構(minimal_ga/Z3 sound gate/falsification)をメモリ効率へ転用/再導出(MnasNet/NSGA-II/Deb/CEGIS)と狭い独自4点結合/経験gate 84% false-admit vs sound cert 0%/branch Aは賢さでなくguaranteeに価値(P1留保)を反映。

### 55. 安全 gate を作ったら、自分の目的関数がそれを冗長にした — branch A PoC の示唆的な FAIL
- **気付き**: 進化探索に sound な安全 gate(収縮 L<1 を Z3 証明、不合格は fail-closed 棄却)を噛ませる PoC を回したら、**設計の主張(G2: 「gate なし ~6% 安全 vs gate あり ~100% 安全」)が実測で反証**された。実際は **gate なしでも既に safe_rate 95–100%**。理由 = メモリ効率目的(収縮率を下げる報酬)も retention(遅延再現)も**どちらも有界 gene を選好する**ため、目的関数が既に安全側を報酬しており、gate を足しても価値が出ない(retention で 0.95→1.00 と僅かに上がるのみ)。**「目的が安全側を報酬するなら、安全 gate は冗長」**=gate の価値は『発散を好む目的』でしか顕在化しない、という非自明な境界条件を実測で確定。さらに **accuracy×memory トレードオフが想定外の場所に出た**: メモリ目的は footprint を半減(0.375→0.149)させたが、その代償は retention の微減だけでなく **「random を超える capability edge の喪失」**(retention passes=True diff+0.012 → memory passes=False diff+0.004)。安全 gate を作って「効かない」と判明する方が、効くより面白い honest disclosure 教材(B1「負けを見せる」系)。
- **根拠**: `docs/BRANCH_A_POC_VERDICT.md`(2×2 全数値・G1–G6)/ `out/poc_branch_a_memory_fitness.json` / `scripts/poc_branch_a_memory_fitness.py` / `tests/unit/test_branch_a_memory_fitness.py`(11 passed/ruff/mypy green、既存 837 非破壊)。Z3 稼働(rej 71/196・fallback 0)。P1 留保=footprint は state-boundedness proxy で実 RSS でない。
- **側面**: honest disclosure / 教訓 / 技術設計 / 認知科学(期待の反証)/ ベンチ。**「期待した結果が出ない方が示唆的」**の実例として B1 連載に組込可。 **本執筆済み (2026-06-21)**: 技術版 `b4-safety-gate-redundant.md` + 非エンジニア版 `b4-safety-gate-redundant-general.md`(ガードレールの出番なしのたとえ)。G2反証(gateなしでも95-100%安全)/目的関数が安全側を報酬する境界条件/メモリ攻めすぎでcapability edge喪失/Z3稼働(71/196棄却)/footprintはproxyのP1留保を反映。

### 56. 散らばった「勝ち筋の道具」を 1 つの受付窓口に畳む — `llcore.memory` ツールキット化（packaging の誠実さ）
- **気付き**: メモリ効率の検証済みプリミティブ(int8 量子化/mmap streaming ロード/定数状態 recurrent/cap-gate)は `scripts/` と `lm/` に**散らばっていて、外から「どれを呼べば何が分かるのか」が見えなかった**。これを `llcore.memory` という 1 つの facade(受付窓口)に畳み、**fp32 モデルを 1 個渡すと「どれだけ小さくなり、その代償に何を失うか」を 1 回で返す** `measure_memory()` を足した。実機 honest 数値(実トレーニング済 ckpt; いずれも `int8_footprint_bytes` の **resident-weight-byte 会計**=params+buffers[量子化不可な causal-mask fp32 を含む保守側]で、on-disk file size とは別物): **realp1 fp32 49.23MB→int8 13.97MB(71.6%減)/ multi_smoke 5.50→1.51MB(72.6%減)**、青空マルチ全文(330,368 token)で **top-1 retention 100.0%**(fp32 0.3629→int8 0.3631)・cap-gate PASS。**重要な honest 注記=int8 が fp32 を 0.0002 だけ上回るのは「改善」ではなく同点 argmax の測定ノイズ**(この規模では int8 量子化の capability コストが実質ゼロ、という FINDINGS (b)「per-channel int8 の PPL 劣化 <0.1%」の追認)。**記事の核 = 「研究の価値は派手な新規アルゴリズムより、既存の勝ち筋を “誰でも 1 行で検収できる形” に packaging すること」**=フロンティアの再導出を自宅 CPU で誠実に道具化する物語。
- **根拠**: `src/llcore/memory.py`(facade 再 export[同一オブジェクト] + `MemoryReport`/`measure_memory` + CLI `py -3.11 -m llcore.memory report`)/ `tests/unit/test_memory_facade.py`(22 件: facade 同一性・配線一致・**呼出側 fp32 モデル非破壊**・CLI footprint/json/retention + **昇格ゲート** + **KV 成長軸**)→ **全 unit 870 passed**、ruff/mypy strict green。**MemoryReport は 3 軸を 1 レポートに**: ①量子化 footprint(静的)②capability retention/cap-gate ③`--context-lens` で **GPT KV キャッシュの文脈長線形成長 vs 定数状態 recurrent の平坦**(実走 realp1: T=256 4.72MB→T=2048 37.75MB ×8.0 線形、`constant_state_bytes` は T 非依存)。正本 doc = `docs/MEMORY_TOOLKIT.md`。**honest 位置づけ=新規アルゴリズムでなく packaging**: プリミティブは llama.cpp/GGUF の**再導出**(`docs/POSITIONING_VS_LLAMACPP.md`, confidence high)で、誠実な独自は**「footprint 勝ちを capability gate で fail-closed に検収する運用」のみ**(PPL だけの gate は top-1 が半減した壊れた低ビットを PASS させる=FINDINGS (b') の実証への是正)。**この gate を昇格ゲートとして実体化**= CLI `--save-int8` は **cap-gate PASS のときだけ int8 ckpt を emit / FAIL・コーパス未指定(capability 未計測)は fail-closed で書込拒否**(`--force` で運用者上書き)。実機 smoke 確認: コーパス無→rc=2 拒否(ファイル無)、--force→3.8MB 書込。`measure_memory` は eval データ無しなら capability を `None` で返し「無いを捏造しない」。
- **側面**: 技術設計 / 教訓 / honest disclosure / 業界比較(再導出 vs 独自の線引き)/ エコシステム / 戦略(「派手さより道具化」のキャリア訴求)。**B 系「負けを見せる」と対になる「勝ち筋を道具にする」連載の着地**。 **本執筆済み (2026-06-21)**: 技術版 `b6-packaging-wins.md` + 非エンジニア版 `b6-packaging-wins-general.md`(散らばった台所道具→受付窓口のたとえ)。3軸MemoryReport/realp1 71.6%減・multi 72.6%減/retention100%(0.0002はノイズと明記)/独自=cap-gate fail-closed運用のみ(再導出はllama.cpp)/良いHWほど効くは未計測仮説、を反映。「良い HW ほど効く」設計指針(int8→GPU の真 int8 GEMM / mmap→大 RAM の共有ページキャッシュ / 定数状態→長文脈)も同記事で展開可(**ただしいずれも未計測=設計仮説。速度は FINDINGS #34 で未測定と明記**)。

### 57. AI レビュアーの「捏造だ」を一次証拠で覆したら、犯人は causal-mask buffer だった — メモリ会計の罠
- **気付き**: ツールキットの敵対レビューで AI 批評家(gem-critic)が **「realp1 の footprint 49.23MB→13.97MB は捏造、正は 47.66→12.10MB だ」と CRITICAL 判定**。だが鵜呑みにせず一次証拠で検証(`feedback_no_solo_ai_judgment`)したら **false-positive** だった。差 1.57MB の正体 = **6 層 × 256×256 × 4B の causal-mask buffer**。私のツールキットの `int8_footprint_bytes` は params **+ buffers** を数える「resident-weight-byte 会計」、批評家が「正」とした旧 script の数字は **params-only 会計**。**同じモデルでも会計の取り方で footprint が変わる**。しかも量子化不可な mask を fp32 で含める方が int8 比を**保守的(71.6% と控えめ=underclaim)**に見せる=honest 方向。**教訓2つ**: (1)メモリ数値は必ず「何を数えた会計か(params/buffers/RSS/on-disk)」を明示せよ、混ぜると別人の数字になる (2)AI のもっともらしい指摘ほど一次証拠で一件ずつ検証せよ — 今回は検証が誤指摘を止め、検証しなければ正しい数字を「捏造」と信じて書き換えていた。
- **根拠**: `int8_footprint_bytes`(`src/llcore/lm/quant.py`、parameters()+buffers() を走査)/ realp1 実測 fp32 49,231,872B(=49.23MB, mask 込み) vs 旧 `out/int8_quant_footprint_realp1.json` 47,659,008B(=11,914,752 params×4, mask 抜き)。差 1,572,864B = 6×256×256×4。検証セッションログ(2026-06-19)。
- **側面**: honest disclosure / 教訓 / 技術設計 / 認知科学(AI 所見の検証規律)。**B 系「負けを見せる/疑う」連載の白眉候補**=「AI レビューが間違っていた回」。**本執筆済み (2026-06-21)**: `docs/articles/drafts/b3-ai-reviewer-was-wrong.md`。1.57MB を素因数分解し causal-mask buffer(6×256×256×4=1,572,864B)と特定、params-only vs params+buffers の会計差と判定、検証規律2教訓を記事化。非エンジニア版 `b3-ai-reviewer-was-wrong-general.md` も完了 (スーツケースの中身 vs 本体ごとのたとえ)。非エンジニア向けフック=「AI 同士の相互レビューでも、最後は一次証拠で人/別 AI が確かめないと誤りを採用してしまう」。
- **挿絵候補(ユーザー提案 2026-06-19)**: スナックバス江 江の名コマ「**極刑でよくない？／よくないのよ**」(`fullsense/docs/articles/assets/bazue_all/162.jpg`)。テーマ完全一致 = 「捏造は極刑でよくない？」→ だが #57 の捏造は **AI の冤罪**だったので「(証拠も見ず断罪するのは)よくないのよ」の二段オチに使える。★**公開前ブロッカー(honest)**: このコマは **Alu カタログ(727コマ)に未収録**(「極刑」キャプション該当なし)、かつ **catalog 行番号 ≠ ローカル連番**(catalog #162 は別コマ『独り占めはズルじゃない?』crop=`ao2WfwQLCjcD4CXc1CnJ`、bazue_all は 206 ファイルのみ)。**ユーザー指示(2026-06-19)で記事には `bazue_all` の生crop を直接使用してよい**(公開ライセンス形態の最終判断はユーザー。`reference_alu_manga_crops` の「Alu permalink のみ」ルールは本件では緩和)。**メタ皮肉**: 挿絵を「番号162」で引いたら番号がズレて別コマを指した = 本記事の教訓(番号でなく内容を一次照合)を挿絵探しで自ら再演した(=記事内エピソードに昇格できる)。

### 58. ランダムモデルでは retention テストが揺れる → 判定を純関数に切り出して決定化(TDD の設計シグナル)
- **気付き**: 昇格ゲート(cap-gate PASS のときだけ int8 を emit)を TDD で書こうとしたら、**未学習ランダムモデルの top-1 は near-uniform で int8 量子化ノイズに脆く、retention が境界を跨いで flaky**になることに気付いた。フル CLI 経由で「gate PASS のとき書く」を直接テストすると非決定的。そこで **昇格判定ロジックを純関数 `_should_promote(report, force) -> (bool, reason)` に切り出し**、ハンドメイドの `MemoryReport`(gate True/False/None)で決定的にテスト → CLI 統合テストは決定的経路(`--min-retention 0.0` で必ず PASS / コーパス無で必ず拒否)だけに限定。**これは TDD の格言「テストしにくい=設計が不明瞭/結合が強すぎる」のままの体験**: テストの難しさが「判定と I/O が混ざっている」設計の臭いを教え、純関数への分離という正しい設計に導いた。
- **根拠**: `src/llcore/memory.py` `_should_promote`(純関数)/ `tests/unit/test_memory_facade.py`(`_should_promote` 単体 4 件で True/False/None×force を決定的に固定 + CLI 統合は決定的経路のみ)。`superpowers:test-driven-development` の「When Stuck: Test hard = design unclear」。
- **側面**: 技術設計 / 教訓 / 実装報告。**A 系「テストが設計を教える」小記事**。フック=「テストが flaky なのはテストのせいでなく設計が混ざっているサイン」。 **本執筆済み (2026-06-21)**: 技術版 `a11-test-hard-design-smell.md` + 非エンジニア版 `a11-test-hard-design-smell-general.md`(味見=味付けと盛付けを分ける/料理のたとえ)。flakyの原因=ランダムモデルでretentionが境界跨ぎ+判定とI/O癒着→`_should_promote`純関数に分離して決定化、を反映。

### 59. int8 で 4MB 削った隣で、言語ランタイムが 184MB 食っていた — 「Rust にすれば?」への実測回答
- **気付き**: 「メモリ効率を徹底するなら Rust 実装で更に効率化しないか?」を実測で検証(`feedback_rust_usage_matters`=「Rust化=効率↑」は自動でない、を一次データで)。**ctypes で RSS 実測: Python interpreter baseline 13.4MB → `import torch` で +183.9MB(197.3MB)→ multi_smoke モデル load で 213.6MB。int8 重みは 1.51MB =「プロセス RSS はモデルの 142×」**。= **量子化(int8 4×=数MB 節約)/mmap/定数状態の勝ちは表現・OS・アルゴリズム由来で言語非依存**だが、この豆モデル規模での**支配項は『言語ランタイムの baseline(torch だけで +184MB)』**だった。**ここが Rust/native の真の効きどころ**: lean な native バイナリ(C++ の llama.cpp / Rust の candle・mistral.rs)は baseline が桁違いに小さく(典型 数MB〜十数MB)、~180MB 規模の削減=int8/mmap の数MB よりはるかに大きく「working set を小さく予測可能に」の北極星に直結。さらに Rust は allocator を握れるので FINDINGS (c)「解放した fp32 を torch caching allocator が OS に返さず平時 peak が減らない」を `madvise(MADV_DONTNEED)`/明示解放で正攻法で直せる。
- **根拠(実測 2026-06-19)**: ctypes `GetProcessMemoryInfo` WorkingSetSize = baseline 13.4MB / +torch 197.3MB(tax +183.9MB)/ +model 213.6MB、`int8_footprint_bytes`=1.51MB → 142×。FINDINGS (c)。**honest な反対側**: Rust は int8 を小さくも mmap を良くも定数状態を定数にもしない(全部言語非依存)。素の Rust matmul は torch の BLAS(MKL/OpenBLAS)より**遅い**=速度は candle/BLAS バインドが要る(「Rust=速い」も自動でない)。**規模依存が肝**: 184MB tax が支配的なのは豆モデル規模ゆえ。GB 級の重みなら baseline は相対的に無視できる → **Rust の baseline 優位は llcore が実際に住む領域(小モデル・小 RAM)で最大・重みが支配する大規模で最小**。これは llama.cpp が既に実証した動き=positioning 上「再導出」で、誠実な独自(cap-gate)は言語非依存。**未確証**: Rust/candle 版の baseline RSS は未計測(主張の最終確証は要 candle 実測)。移植するなら `project_llove_rust_migration` の規律(Python 成熟後に hot path のみ Rust)に沿い、狙いは『推論パスを native 化して interpreter tax を消す』(速度狙いでない)。
- **側面**: 技術設計 / honest disclosure / 業界比較 / 教訓 / 戦略。**フック=「最適化する項を間違えていた — int8 で 4MB 削った隣で言語ランタイムが 184MB 食っていた」**。memory-efficiency 連載の技術者向け中核 +「Rust 化の前に baseline を測れ」の実践教訓。 **本執筆済み (2026-06-21)**: 技術版 `a9-measure-before-rust.md` + 非エンジニア版 `a9-measure-before-rust-general.md`(荷物4kg vs 台車184kg のたとえ)。支配項(静的=torch baseline 184MB)/Rust の真の効きどころ(baseline)/言語非依存の勝ち/規模依存/candle 未計測の honest gap を反映。

### 60. llcore に話しかけてみた — 漱石風の断片は出るが「会話」はできない(pivot 理由の実演)
- **気付き**: 「llcore は環境次第で一般的な LLM くらいの会話ができるか?」を最良モデル(realp1=青空文庫 p1, 11.9M params, CPU)で実生成して確認。prompt「こんにちは。」→『そうなには、何の長いのだ」「あの？おや、そんなさ、何だよ。先生はちゃいいい」…』、prompt「日本で一番高い山は」→『長いであっている。それから君子を見たら一番ですから…寒月君は…芋の伯父さん…』。= **青空文庫(漱石『吾輩は猫である』含む)の文体・登場人物名を真似た“それっぽい日本語の断片”は出るが、意味は通らず、質問に答えられない(富士山は出ない)**。char 単位 tiny LM(top-1 次文字 ~36%)の正直な天井。**「環境次第で会話レベルに」= ならない**: 制約は環境でなく本質(char 単位・極小 params・極小データ)。良い HW = より大きいモデルをより多くのデータで訓練できる、だけで、それは llcore を別物(P3 クラウド規模)にする話。**だからこそ北極星を capability から memory 効率へ pivot した**(capability は CPU char-LM では構造的頭打ち=負け筋、と honest disclosure 済 [[project_llcore_memory_efficiency_pivot]])。会話 LLM が要るなら FullSense 設計では llmesh 経由で既存 LLM(Gemma4 等)を on-prem で回す = llcore はチャットボットでなく研究ビークル(進化・検証・メモリ効率)。
- **根拠(実生成 2026-06-19)**: `py -3.11 -m llcore.lm generate out/lm_aozora_realp1/model.pt`(上記2サンプル, seed 1/2, temp 0.8 top-k 40)。top-1 次文字 0.3629(aozora multi)。pivot 正本=memory `project_llcore_memory_efficiency_pivot`。
- **側面**: honest disclosure / 業界比較 / 哲学 / 教訓 / ユーザー体験。**フック=「自作 LLM に話しかけたら漱石の幽霊が出た — でも会話はできなかった」**。「見せて語る」honest 記事(なぜ賢さを諦めメモリに賭けたかを実出力で示す)。B 系・pivot 物語と接続。 **本執筆済み (2026-06-21)**: 技術版 `b5-talked-to-llcore.md` + 非エンジニア版 `b5-talked-to-llcore-general.md`。実生成2サンプル(漱石の幽霊)/top-1 0.3629/「環境でなく本質の制約」/北極星をcapability→memory効率へpivotした理由/llcore=研究ビークル(会話はllmesh経由の既存LLM)を反映。
- **挿絵候補(ユーザー提案 2026-06-19)**: `bazue_all/030.jpg` = タツ兄の顔を故意に8bitドット化したメタ演出コマが「**ハーブか何かやっておられる？**」(=ヤク中か?と訝るリアクション)。llcore に話しかけて漱石風の支離滅裂が返る #60 のツッコミとして完璧(かつドット顔=「低ビット量子化で潰れても“それと分かる”」の視覚比喩として量子化アーク #34/37/41–44/53 にも転用可)。**ユーザー指示(2026-06-19)で `bazue_all` 生crop を直接使用してよい**(集英社『週刊ヤングジャンプ』公式の無料 SNS 共有素材=合法。非商用・出典明記)。★**重要訂正(2026-06-19)**: 当初私はコマの © 『バーチャル式部』を信じ「バス江と別シリーズ」と帰属したが**誤り**=**030 は《スナックバス江》**(作風がパロディ・メタフィクションで一部コマのロゴが偽作品名『パーチャル/パープル式部』等のパロディ。GT `bazue_all/index.md` L15 明記=別作品の混入ではない)。AI critic + GT index で訂正。**教訓**: 「番号で当てず内容で照合」(#57)に加え、**“一次証拠”の publisher © すら偽ロゴという罠**があり複数ソース照合が要る。AI critic は #57 では false-positive(捏造指摘が冤罪)・本件では true-positive(誤帰属を正しく検出)=**どちらにも転ぶので必ず実物+GTで照合**。

### 61. 文脈を4倍にしたら Transformer のメモリは5倍に膨れ、recurrent は1mmも動かなかった — 長文脈ほど構造で勝つ
- **気付き**: 「むりに bit を下げる」より**支配項を攻める**方針(#59)で、メモリの**動的支配項=文脈長で伸びる KV/attention**を実機 peak RSS で長文脈まで測定。**T 1024→4096(×4 文脈)で GPT peak WS ×5.04(331.9→607.8→1673.0MB=超線形=attention O(T²)が支配)/ Recurrent ×1.00(205MB 平坦)/ RWKV ×1.00(216MB 平坦)**。T=4096 で **GPT 1673MB は recurrent 205MB の 8.1×**、外挿で T=8192 は GPT ~6.5GB(本機 3.6GB RAM で OOM=物理的な壁)に対し recurrent は 205MB 据置。= **「長文脈ほど定数状態 recurrent が GPT を構造的に引き離す」を解析値でなく実機 peak RSS で確定**(前回 256→2048 の ×2.65 より、長文脈ほど超線形が鋭くなる=固定費の割合が下がるため)。**北極星「working set を小さく予測可能に」の本命が量子化(静的)でなくアーキ(動的・長文脈)であること**の実証。bit を下げない判断(#59)と表裏一体。
- **根拠(実機 2026-06-20)**: `scripts/recurrent_runtime_rss.py --lengths 1024,2048,4096`(別プロセス隔離・WinAPI peak WS)→ `out/recurrent_runtime_rss_long.json`。**src 強化**: `RecurrentLM/RWKVLM.streaming_nll`(純粋追加)= **block_size 制限なし・O(chunk)メモリで任意長文脈をスコア**(GPT は attention O(T²)+block_size 制限で不可能な領域を recurrent は平坦に処理できる、を実機能化)。検証=streaming_nll が短文脈で forward の loss と一致 + block16 の 12.5倍長(T=200)を処理。`tests/unit/test_recurrent_streaming.py` 6件 + 既存 recurrent/rwkv/compare 回帰 green、ruff/mypy strict、既存非破壊。**honest 留保**: peak WS は torch ランタイム baseline(~205MB=#59 の支配項)+ 固定重み + T 依存バッファの合算で、クリーンな信号は**増分トレンド**(GPT は膨張・recurrent は平坦)。recurrent の「平坦 205MB」の中身はほぼ torch baseline(定数状態自体は KB)=「伸びる項(attention/KV)が無い」の意。random 未学習モデル(メモリ挙動はアーキ依存=学習非依存ゆえメモリ計測には妥当)。GPT.generate は実運用では block_size crop で有界(本測は厳密長文脈の必要量)。
- **側面**: 技術設計 / ベンチ / honest disclosure / 業界比較 / 未来予測。**フック=「文脈を4倍にしたら、Transformer のメモリは5倍に膨れ、recurrent は1mmも動かなかった」**。draft a7(SSM=制御工学)の実機メモリ支柱・memory-efficiency 連載の「動的支配項」中核。「bit でなくアーキ」を数字で示す。
- **本執筆済み (2026-06-21)**: `docs/articles/drafts/a8-context-memory-blowup.md`。a7 の実機メモリ対編。実測 `out/recurrent_runtime_rss_long.json` を全公開 (GPT 331.9→607.8→1673.0MB ×5.04 / recurrent 205 平坦 / RWKV 216 平坦、T=4096 で GPT=recurrent の 8.1×、外挿 T=8192 GPT~6.5GB は 3.6GB 機で OOM)。b2 のメタ皮肉 (自宅機が長文脈 attention で OOM) の構造的理由を O(L²) として説明。honest 留保 (極小ランダムモデル/トレンドを読む/peak WS は torch baseline 込み) と prior-art (a7/SUPRA/Mamba) 明記。非エンジニア版 `a8-context-memory-blowup-general.md` も完了 (会議の総当たり vs ノート回しのたとえ)。

### 62. 「proxy は noisy」で終わらせない — ノイズを測って“勝った”の主張自体を抑制する NAS の honest-disclosure 層
- **気付き**: メモリ vs 品質の Pareto NAS(`scripts/nas_pareto.py` proxy-v2)で、memetic(NSGA-II)が greedy ベースラインに勝ったかを判定する際、**training-free / proxy 指標は actual performance としばしば乖離する**(先行研究 MTF-PDNS arXiv:2407.20656 等が「proxy-noise trade-off」として明言済み=確立領域)。そこで本作業の新規性は**新しい探索演算子ではなく、proxy の不確実性を定量化して verdict 自体を律する disclosure 層**に置いた。具体的に 5 段で主張を抑制: (1) proxy に paired multi-window bootstrap CI、(2) GA 選抜窓と**disjoint な fresh holdout** で winner's-curse を除去し `optimism_gap = selection − holdout` を開示、(3) proxy-vs-judge の Kendall τ<0.7 で positive verdict を 'suggestive' へ降格、(4) HV gain の勝ちは **CI_lo>0 のときだけ**発火(点推定では発火させない)、(5) memetic≈greedy は **separable landscape の honest negative** として隠さず明示。= **先行研究が「proxy は noisy」と認める所を、本作業は「noisy さを測って主張を抑制する」段まで進めた**点が差分。scope は `next_token_nll_proxy` 固定で、**会話品質クレームは一切混ぜない**(別の disclosed generation eval に分離)。
- **根拠**: `src/llcore/runtime/eval_proxy.py`(852行: bootstrap CI / exact sign test / Wilcoxon / Kendall τ-b / winner's-curse holdout / 共有参照 HV)+ `pareto_metrics.py` を精読し methodology 健全と監査(是正不要)。レポート生成器 `scripts/nas_pareto_report.py`(read-only, torch 非依存, holdout 主導 + positioning 常時開示)+ 回帰テスト `tests/unit/test_nas_pareto_report.py` 13 passed / ruff / mypy strict green(commit `d791f78`)。**実 verdict は overnight 走(`out/nas_pareto_v2full`)の `nas_pareto.json` 着地後に追記**(本シード時点では memetic_vs_greedy / regime 依存 / optimism_gap / hv_gain_ci の数値は未着=honest に未確定と明記)。
- **実数値追記(2026-06-21, overnight 走完了 386 real evals / 23,849s)**: base all-softmax nll 4.4155(ppl≈82.72)。**ここが記事の核=二段構造の honest disclosure**: (A) zero-shot(selection 窓スカラ)では `memetic frontier dominates greedy: HV +15.3%`(greedy 58.47 → evolved 67.44)で memetic 勝ち。(B) ところが rigorous tier の HEADLINE(holdout)では **verdict = suppressed**(`confidence: suppressed — max optimism_gap 0.0652 > CI half-width floor 0.0204`)。つまり **「selection で見えた勝ちは、winner's-curse 補正後のノイズ床を超える楽観バイアスを含むので主張を抑制した」**。一方 **HV gain(holdout)は +16.8%(95% CI 16.2..17.7%, p_memetic_wins 1.000)**で CI_lo>0 を満たすため HV 次元での memetic 優位だけは発火 = 「frontier 個別点の verdict は黙らせるが、集約 HV の勝ちは CI が支持する限り残す」という粒度別の誠実さ。Kendall τ=1.00(proxy-vs-judge 整合、降格なし)。regime 依存(最アグレッシブ 83.9% genome の context sweep)は **L=256: Δnll 0.761 → 512: 1.012 → 1024: 1.182** と長文ほど劣化増大(constant-state failure mode の兆候, cf. SUPRA)。**honest gap=計画では 2048 tok まで回す想定だったが実出力は 256/512/1024 のみ**、かつ **needle/passkey は `--needle` off で UNTESTED**(長距離 copy 失敗は未検証ギャップとして開示)。attention-KL(診断専用・fitness 非配線)は mean 3.68 / max 7.67(layer 9)。**最強のフック=「自作 NAS は zero-shot で『+15.3% で勝った』と言ったが、ノイズ床補正後に frontier verdict を自ら suppress した。ただし HV の勝ちだけは CI が支持したので残した」=主張の粒度ごとに自信を数えて取捨する誠実さ**。レポート正本 `out/nas_pareto_v2full/nas_pareto_report.md`。
- **側面**: honest disclosure / ベンチ / 技術設計 / 業界比較(proxy-NAS の確立領域との線引き)/ 哲学(「勝った気になる前に内訳を疑う」の機構化)/ 教訓。**フック=「自作 NAS が“勝った”と言う前に、その自信を数えて黙らせる仕組みを入れた」**。B 系「負けを見せる/疑う」連載に直結し、#53(2bit 負け)・#55(gate が冗長化した FAIL)と同じ「主張を抑制する誠実さ」の系譜。
- **本執筆済み (2026-06-21)**: `docs/articles/drafts/b2-suppress-your-win.md`(B部 b2、150 行)。タイトル「『+15.3% で勝った』を、自分のノイズ床で黙らせた — NAS の勝利宣言を撤回する仕組みを作る」。zero-shot +15.3% 勝利を holdout optimism_gap (max 0.0652 > CI 床 0.0204) で suppress、HV gain +16.8% (CI 16.2..17.7%, CI_lo>0) のみ残す二段構造を実数値で全公開。needle UNTESTED / 2048 未測 / attention-KL fitness 非配線 の honest gap 明示。commit `2b6fad9`。非エンジニア向け (QIITA_GENERAL 系) は未着手。
- **2026-06-21 訂正**: 当初「2048 は inner-context=1024 設計のため構造的に未出力」と書いたが**誤り**。`context_sweep` (src/llcore/runtime/eval_proxy.py:461) は `make_windows` で inner-loop 長と独立にコーパスから任意長窓を切る実装、コーパスも 230 万トークンと十分長いので 2048 窓は作成可能。実際は**その走の `--context-sweep` 設定が 256/512/1024 までだった=単なる未測ギャップ**。b2 §5 に訂正済み(訂正自体を「未検証を構造のせいにしない」実演として記事化)。2048 sweep + needle は次走で埋める課題。

  - **2026-06-21 続報(ハードウェア律速 + メタ皮肉)**: needle/2048 の honest gap を埋めようと resume 走(GA は eval_cache から 386 evals 復元でスキップ、rigorous tier + 2048 sweep + needle のみ)を 2 度起動。だが **2048tok の full-attention forward が working set 3.9GB に膨れ、物理 RAM 3.6GB を超えてスワップ thrashing**(needle-lengths を 4096→2048 に落としても再発)。2 度とも完走せず kill。→ **「2048+ は測れない壁ではなく、このハードウェアでは測れない壁」**。記事級のメタ皮肉=「定数状態が長文脈でメモリを溢れさせる失敗モードを測ろうとして、測る側の自宅 CPU が長文脈でメモリを溢れさせた」。b2 §5 / b2-general を「RAM 律速で未実測、GPU オフロードが次の正手」に更新。元 nas_pareto.json(22:37 完走版、context_sweep=256/512/1024)は未上書きで無傷。

### 63. cross-machine な resume は「厳密一致」をやめて「basename + tolerance」で開く — でも安全網は残す
- **気付き**: 長時間ジョブの再開キャッシュ(eval_cache)が **別マシンで resume できない**問題に当たった。原因は identity 判定の `meta` 厳密一致(`==`)で、(1)`model_dir`/`text_file` の**絶対パスがマシンごとに違う**、(2)`base_nll`(1 forward の平均 CE)が **Windows↔Linux の BLAS 差で 6 桁目がずれる**。これを `_meta_matches` に置換: **path 系は basename 比較・base_nll は 1e-3 tolerance・他は厳密一致 + キー集合一致**。緩めすぎない安全網=別モデルを同名 basename で渡しても base_nll が tolerance を超えて reject(content チェック)。=「resume identity は厳密一致が安全」という素朴な実装が、移植性を殺す。**識別に効く軸(model 種別・corpus・config)だけ厳密にし、環境依存の軸(絶対パス・float 丸め)は寛容に**するのが正しい設計。
- **根拠(2026-06-21)**: `src/llcore/runtime/eval_cache_io.py` `_meta_matches`(commit `b11a235`)、回帰テスト `tests/unit/test_eval_cache_io.py` 12 passed(cross-machine 6 件追加)、ruff/mypy strict green、全 unit 991 passed で回帰なし。ローカル事前検証=prefix fixture の base_nll が cache meta と diff 4.98e-08 で一致 + CI 相当 relocated meta で resume OK(scalar 386/vector 386 復元)。
- **側面**: 技術設計 / 教訓 / honest disclosure(安全網の設計)/ 計算オフロード。**フック=「resume が別マシンで落ちる本当の理由は、コードでなく『絶対パスと float の 6 桁目』だった」**。計算オフロード(GH Actions/Kaggle)を現実にする地味な前提条件。

### 64. RAM 律速のジョブをオフロードするとき、コーパス全文は要らない — 「先頭プレフィックス + resume snapshot + fail-fast」
- **気付き**: 自宅 RAM 3.6GB では 2048tok の full-attention forward が working set 3.9GB に膨れて thrash([a8]/#40 の構造が測る側に牙を剥いた)。GPU でなく **RAM が支配項**なので GH Actions 標準ランナー(7GB)で解ける。だが素朴にやると (a)GA を CI で fresh 実行=2 コアで ~26h>6h 上限、(b)9.8MB コーパス全文を repo にコミット、の二重苦。解は 3 点セット: **(1)GA 結果(eval_cache snapshot)を fixture 化して resume → GA をスキップ、(2)コーパスは base_nll を厳密再現し全 holdout/sweep/needle 窓(最大~32768tok)を満たす先頭 20 万字プレフィックス(580KB)だけコミット(全文 9.8MB 不要)、(3)resume 失敗(meta mismatch)時は 26h GA 再走の前に fail-fast(`grep [resume] || exit 1`)**。= 重いオフロードは「全部送る」でなく「再現に必要な最小集合 + 安全な早期失敗」で設計する。
- **根拠(2026-06-21)**: `.github/workflows/nas-needle-offload.yml` + `ci/fixtures/`(corpus prefix 580KB / eval_cache 109KB)(commit `1853d0b`)。プレフィックスは 50k 文字 skip 後 171,918 tok を生成(必要 32,768 を大幅超過)。push 手前まで構築・ローカル resume 実証済み。**残: push=human gate(外部公開)**。#63 の cross-machine resume 堅牢化が前提。
- **側面**: 技術設計 / 計算オフロード / 教訓 / honest disclosure(必要最小集合の見極め)。**フック=「問題は GPU 不足でなく RAM 不足。コーパスは全文でなく先頭 20 万字で足りた」**。[a8](動的支配項)/[a9](静的支配項)の実務的続編=「支配項を見極めて、それだけを攻める」をオフロード設計に適用。
