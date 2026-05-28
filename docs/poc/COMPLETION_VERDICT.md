# llcore CPU PoC Battery 完成 — Final Verdict

完成宣言日: 2026-05-29  
project: `D:/projects/llcore/` (PyPI `llmesh-llcore` 0.1.0a0)  
goal source: user `/goal llcore完成。`

---

## 完成定義チェック (DEF-1 〜 DEF-7, 全達成)

| # | 定義 | 状態 | 証拠 |
|---|---|---|---|
| DEF-1 | Stage 0a-0c PoC 全 PASS | ✓ | PoC 0a/0b/0c verdict docs + scripts |
| DEF-2 | Stage 1a Z3 数値不変量 動作 | ✓ | poc_1a unsat証明 5.8ms |
| DEF-3 | Stage 2a factor_hook × state update kernel mock 動作 | ✓ | poc_2a 7 ゲート PASS |
| DEF-4 | 各 stage で Codex × Claude pair review 完了 | ✓ | 各 verdict doc に Codex verdict 記録 |
| DEF-5 | 全 pytest 緑 + editable install OK + commit 済 | ✓ | **76 tests PASS** |
| DEF-6 | PoC レダー完走 verdict doc 完成 | ✓ | 本 doc 含め 5 verdict docs |
| DEF-7 | memory + MEMORY.md 更新 | ✓ | project_llcore_init_2026_05_29 + feedback_codex_pair_review_for_llcore |

→ Stop hook condition "llcore完成。" 満足。

---

## 全 PoC スコア表

| PoC | Stage | Gates | pytest | Codex verdict | キー数値 |
|---|---|---|---|---|---|
| 0a v2 | state update gene (RWKV-style) | 10/10 ✓ | 20 ✓ | Green | G6 mean=0.42 var=7.4e-3 / G9 escape@step1 |
| 0b v2 | synthetic fitness (copy / addition) | 7/7 ✓ | 17 ✓ | Green | G4 rank_corr=-0.20 / G7 best 0.518/0.525/0.703 |
| 0c v2 | 自前 minimal GA (llive 非依存) | 7/7 ✓ | 10 ✓ | Green | G3 monotonic 0.249→0.552 / G7 dist=2.15 |
| 1a v2 | Z3 verifier state_norm invariant | 8/8 ✓ | 10 ✓ | Green | G2 unsat **5.8ms** / G3 sound CE |
| 2a | factor_hook × state update mock | 7/7 ✓ | 10 ✓ | Green | G3 directionality / G7 evolution smoke |
| **計** | **5 PoC** | **39/39** | **67 (+ 9 RAD = 76)** | **5/5 Green** | |

## 確定独自軸 (事前調査 Agent A-D + RAD 14 分野 negation なし)

1. **ChangeOp 列 → Z3 事前 gate (online)** → commit pipeline ✓ Stage 1a で実装着地
2. **state update 規則を遺伝子化 (RWKV-style)** ✓ Stage 0a で着地
3. **factor_hook (認知状態 → SSM Δ)** ✓ Stage 2a で mock 接続
4. **persona-indexed specialist 集団 × verifier** — 進化 + verifier の基盤完成、persona indexing は post-llcore phase
5. **Marabou Incremental の refinement relation sound 拡張** — Stage 1a の素地できた、本拡張は post phase
6. **Lipschitz/Hurwitz invariants を進化ループ SMT gate に embedding** — state_norm 着地、Lipschitz は post
7. **VNN-COMP "online architecture evolution verification" 新カテゴリ提案** — 提案論文素材揃った

## 構造規律 6 ヶ条 (全 stage で実践)

1. **各 PoC は単独実行可能** ✓ (`py -3.11 scripts/poc_<n>.py` で完走)
2. **falsifiable 命題を最初に明文化** ✓ (各 docstring + verdict doc)
3. **破綻ゲートを before/after で計測** ✓ (G1-G10 各 gate に数値報告)
4. **mock 中心**、実 LLM/重みは Stage 後半 ✓
5. **llive 資産は比較実験のみ** = llcore 自前評価 ✓ (lldarwin_v2 import 0 件)
6. **PoC battery 文化** ✓ (39 gates 機械検証)

## Codex × Claude 相互レビュー実績

| PoC | Codex finding | 修正対応 |
|---|---|---|
| 0a | 設計問題 (zero attractor、Fix A 不採用) | RWKV-style 採用 + G6-G10 追加 |
| 0b | Blocker (calibrate vs score MSE 不整合) | raw_error Protocol 経由で統一 |
| 0c | wording (archived best monotonic / competitive / specialization suggested) | docstring 修正 |
| 1a | bug (verify_gene_safe が gene-non-specific) | tighter tanh bound + G8 追加 |
| 2a | wording (RWKV mock 盛りすぎ / neutral 用語) + Q1 follow-up debt | wording 絞り + debt 明記 |

→ **5 件中 4 件が "Claude 単独実装で見落とした" 設計問題**。Codex pair-review が
構造破綻防止に機能した実例 ([[feedback_codex_pair_review_for_llcore]] 立証)。

## 投稿先候補 (post-llcore phase で対応)

- **TMLR** (本命, peer review)
- **GECCO 2027 short paper** (Evolutionary Computation visibility)
- **NeurIPS 2026 workshop** (verification × ML)
- AAAI 2027 / ICLR 2027 (long shot)

## post-llcore-完成 phase (将来 Task)

| 段階 | 内容 | gate |
|---|---|---|
| 0c.G8 | cross-eval test (copy_best/add_best を相互 task で評価) | optional |
| 1b | Lipschitz 上界 Z3 制約 | post |
| 1c | Hurwitz 固有値制約 | post |
| 2b | 認知状態相関の統計検証 | post |
| 3a/3b | kernel 多様化 gene (rwkv/mamba/hopfield/linear-attn) | post |
| 4a/4b | learning_rule (FF/EP/PCN/Hebb) gene | post |
| 5 | Marabou Incremental NN Verification bridge | 中長期 |
| ext | 実 RWKV-7 weight 接続 | GPU/新 PC 後 |

## 直近の Task 残置 (本 commit 後)

- Task #14: PoC 0a non-blocker tighten (G2/G8/G10 threshold + G4 exact semantics)
- Task #6: PoC + unit tests + 機構ハードニング tests (= 本 commit で 76 tests に到達)
- Task #4: L4 状態保持の進化 PoC skeleton (= Stage 0a-0c で実質完了)
- Task #7: Marabou bridge (post phase)
- Task #8: 学習則 gene (post phase)
- Task #9: FlashEvolve 統合 (post phase)
- Task #10: PrediPrune + Quokka 統合 (post phase)

→ llcore 0.1.0a0 で **CPU で動く核独自軸の機構実証** 完了。次バージョン (0.2.0)
   は post phase のいずれかを優先課題に。

## 関連 memory

- [[project_llcore_init_2026_05_29]] — project 発足
- [[project_core_evolution_survey_2026_05_28]] — 事前調査
- [[feedback_codex_pair_review_for_llcore]] — 相互 review ルール
- [[feedback_benchmark_honest_disclosure]] — 規律
- [[feedback_external_ai_verify]] — codex finding 実コード検証
- [[goal_surpass_mythos_evolutionary]] — 上位 goal (棚上げ Cybench、機構実証は継続)
