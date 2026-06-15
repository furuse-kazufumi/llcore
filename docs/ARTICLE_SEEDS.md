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
