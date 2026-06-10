# llcore — Verified Neural Architecture Evolution on CPU

**Status**: 研究本体は論文 draft 段階 (2026-05-29 発足 → 2026-06-07 時点で §1-§10 + 318 tests green)  
**License**: Apache-2.0 + Commercial (dual)  
**Position**: 独立研究 project。llive (FullSense family) と思想を共有するが、**llcore 単独で完結する設計**。llive 仕様が進化の妨げになる局面では迷わず独立路線を取る (設計判断 2026-05-29)。

## 研究成果 (date of record)

本リポジトリの主成果物 (いずれも事前登録 → 結果の順で記録し、negative も全強度で開示):

- **論文 draft**: `research/paper/PAPER_DRAFT.md` — 進化する再帰コアを健全な収縮証明 (sound contraction
  certifier ladder) で fail-closed にゲートする研究の全結果 (§1-§10 + honest-disclosure box + 再現表)
- **verified memory evolution (trajectory-tube gate)**: `research/verified_memory_poc/` — 一段収縮証明を
  閉形式の軌道チューブ保証 (`limsup‖e‖ ≤ G·w̄/(1−L)`) に拡張し、事前登録 n=40 で記憶 horizon への
  用量反応を確認 (論文 §9)
- **記憶形成 3 機構 × viability 基質**: `research/internalization_poc/` — 「検証器を誰が持つか」を
  死ねる環境で測定 (自己予見 / 復活修復 / 社会的観察; 論文 §9.6 + `VIABILITY_VERDICT.md`)
- **GPU スケール実験 (HD-1 / Stage-B)**: `research/highdim_evolution/` + `research/rllm_stage_b/` —
  無拘束 gradient は収縮域を出るのが default (entropic drift)、実 Transformer 内で検証コアは
  load-bearing (論文 §7)
- **知見スライド (ja/en, CC BY 4.0)**: `slides/` — 主要知見を 10 枚に要約 (pptx + pdf)。
  出典明示で商用利用可。**現状は要約版 — 研究の進展に合わせて今後 1 年かけて拡充予定**
  (実験設計詳細・全図表・再現手順・採用判断材料)

---

## 何

Transformer のコアアルゴリズム (state update / 学習則 / 認知駆動 Δ) に進化形態を与え、**健全な contraction 証明器 (sound contraction certifier ladder) で破綻させずに** 異アルゴリズムへ進化させる研究フレームワーク。CPU 完結。

> **看板の honest 訂正 (2026-06-06)**: 当初は「Z3 verifier」を看板にしていたが、論文本体 (`research/paper/PAPER_DRAFT.md`)
> のゲートは **SMT ソルバを一切使わない**。実体は閉形式 ∞-norm (`O(n²)`) → 頂点 SVD (`cert_two`) → SDP-Lyapunov LMI
> (CLARABEL) の **sound contraction certifier ladder**。別 track の検証 (`research/coupled_z3_contraction/C_VERDICT.md`)
> で Z3 判定と閉形式 ∞-norm が **0/3270 件不一致 = Z3 は decorative** (各行 abs-sum の凸性で box 最大が端点に落ち、ソルバ不要)
> と自己確認済み。これは後退ではなく設計上の長所 — 証明する性質を「ソルバ不要なほど単純な健全 contraction 条件」に絞った結果。
> ソルバの真価は閉形式で書けない spectral/Lyapunov 不変量 (SDP rung + 将来の vertex-free robust-LMI) にあり、SMT 決定手続きではない。

## 基本会話 (llcore.chat — Phase 0 baseline)

EVOLVABLE_LLM_PLAN_2026_06_09 Phase 0「実在の小型オープン LLM を base に据え baseline 機能
(coherent text / 基本 Q&A) を継承」の製品コード。**llcore は LLM としての基本会話が可能**:

```powershell
pip install "llmesh-llcore[chat]"        # torch + transformers (optional extra)
py -3.11 -m llcore.chat                  # 対話 REPL (/exit /reset /history)
py -3.11 -m llcore.chat --prompt "What is the capital of France?"
```

- base = **SmolLM2-Instruct** (Apache-2.0, CPU 完結 / on-prem)。default 135M、
  `LLCORE_CHAT_MODEL` env または `--model` で 360M 等へ差し替え可。Qwen 系は商用障壁のため不採用。
- 複数ターン履歴 + 文脈引継ぎ + context 予算内の対単位トリミング。torch/transformers 不在時は
  `ChatDependencyError` で fail-closed (黙って劣化しない)。
- 段階的会話スモーク (挨拶→事実 Q&A→文脈引継ぎ→話題転換): `scripts/chat_staged_smoke.py`
  (結果 verbatim を `out/` に JSON 記録。honest: 135M は固有名想起等で誤ることがある)。

## llive との関係 (戦略)

**llcore は llive 非依存の独立 project**。llive は参考実装 + 比較対象として扱う。

- **必須**: numpy のみ
- **optional [z3]**: Z3 SMT solver — **本体ゲートでは非 load-bearing** (Track-C で閉形式 ∞-norm と 0/3270 一致 = decorative)。比較・別 track 用途のみ
- **optional [sdp]**: CLARABEL (SDP-Lyapunov rung)。閉形式で書けない spectral/2-norm 不変量はここが load-bearing
- **optional [llive]**: **比較実験用のみ**。llcore 自前の minimal 進化エンジンが本流。llive の lldarwin_v2 を「baseline 比較」「アイデア参照」として import するが、依存しない (設計方針「llive の仕様が llcore の進化の妨げになるなら llcore 単体で仕上げる」)
- **RAD コーパス** (`LLCORE_RAD_DIR` 配下の `<domain>_corpus_v2/`) は **path-based で参照** → llive 依存なしで進化中個体に先行研究 hint を注入できる (`src/llcore/rad/`; コーパス不在時は graceful degrade)

### なぜ llive 非依存か (設計判断)

llive 仕様が llcore の自由な進化を妨げる可能性:

1. **lldarwin_v2 の選択核** が llive Genome3D 構造に縛られる。llcore は state update 数式という新規 genome を試したい
2. **factor_hook protocol** は llive 10 思考因子に固定。llcore は別の認知状態表現を試す自由が欲しい
3. **verifier.py の ChangeOp 型** は llive subblock 系に縛られる。llcore は state update 規則レベルの ChangeOp が必要 = 異なる粒度
4. **API バージョニング**: llive 0.6.0 更新で llcore が壊れる semver 越え変更を受け入れたくない
5. **テスト互換性**: 新規実験毎に llive 1027 tests への影響確認は研究速度を落とす

→ llcore は **自前 minimal 進化エンジン** (`src/llcore/evolution/`) を持ち、llive はインスピレーション源 + ベースライン比較に留める。

### 将来融合の可能性 (条件付き)

llcore が成熟し、その設計が llive にも価値があると判明した時点で**逆方向 PR** (llcore → llive) を検討。llive → llcore の import 依存方向は取らない。

## なぜ別 project か

- llive (38 subpackage / 1027+ tests) は既に大規模、コア進化の試行錯誤は隔離環境のほうが安全
- 失敗した PoC を **project 丸ごと破棄** できる
- 既存 llive 資産 (`lldarwin_v2` / `verifier.py` / `factor_hook` / `impl_chromosome`) は **import 依存** で再利用 (改造は禁止、改造したくなったら llive 本流 PR)
- 結果が出たら llive 本流に逆統合可能

## 確定独自軸 (事前調査 2026-05-28 完了, Agent A-D + RAD 14 分野)

1. **ChangeOp 列 → sound contraction certifier の事前 gate (online, prove-then-reject)** → commit pipeline (CDGP は事後フィードバック、方向逆)。ゲートは閉形式 ∞-norm + 頂点 SVD + SDP-Lyapunov の certifier ladder であり SMT ソルバ不使用 (Track-C 確認)
2. **学習則 (FF/EP/PCN/Hebb) を gene として混在進化** (先行未発見)
3. **factor_hook (認知状態 → SSM Δ)** (実装した先行未発見)
4. **persona-indexed specialist 集団 × verifier** (NAS は単一最良、進化集団 × verifier は完全独自)
5. **Marabou Incremental の "異なる構造" refinement relation 拡張** (sound 拡張で論文化可)
6. **Lipschitz/Hurwitz invariants を進化ループ SMT gate に embedding**
7. **VNN-COMP "online architecture evolution verification" 新カテゴリ提案余地**

(発足時の事前調査 doc は llive リポジトリの `docs/papers/` 配下)

## 段階的 PoC レダー (構造破綻防止)

各 PoC は **独立スクリプト + falsifiable 命題 + 破綻ゲート**。前段が破綻したら次段に進まない。
**下表は発足時 (2026-05-29) の計画スナップショット** — 現在地と確定結果は上の「研究成果」節と
`research/paper/PAPER_DRAFT.md` を正とする。

| Stage | PoC | 命題 | 破綻ゲート | Status |
|---|---|---|---|---|
| **0a** | state update 数式遺伝子 | decay/mix/gate の 3 パラメータで RNN state update を表現できる | 数値が NaN/Inf にならない | **着手中** |
| **0b** | 合成 sequence fitness | copy/addition task で fitness が正常に計算される | fitness が定数にならない | |
| **0c** | 進化 10×10 (**llcore 自前 minimal GA**) | 自前 GA で 10 個体 × 10 世代の進化が完走 | 全滅しない、best が単調非減少 | |
| **0c'** | 進化比較 (optional, llive baseline) | lldarwin_v2 を import して同 task で比較 | 自前 GA との fitness/多様性差を計測 | |
| **1a** | state norm 有界 Z3 | `state_norm ≤ K·input_norm` を Z3 で検査 | Z3 timeout < 1 sec | |
| **1b** | Lipschitz 上界 Z3 | weight matrix operator norm を Z3 制約 | 既存 invariants と矛盾しない | |
| **1c** | online gate 動作 | 進化中に gate reject 率を計測 | reject 率 0% も 100% も異常 | |
| **2a** | factor_hook Δ 調整 | NoopFactorHook と HeuristicFactorHook で出力差 | hook 経由で state decay が動く | |
| **2b** | 認知状態相関 | uncertainty 高時に Δ 小 | correlation > 0.5 | |
| **3a** | kernel 多様化 gene | rwkv/mamba/hopfield/linear-attn を遺伝子化 | 各 kernel が Stage 1 gate pass | |
| **3b** | specialist 出現 | 異なる task に異なる kernel が選ばれる | 集団が単一 kernel に固定しない | |
| **4a** | learning_rule gene | backprop/FF/Hebbian gene を進化対象 | 各規則 MNIST 収束 (FF は遅 honest) | |
| **4b** | task 依存性 | task によって最適学習則異なる | proxy mechanism feasibility | |
| **5** | Marabou bridge | "異なる構造" refinement relation の sound 拡張 | Marabou が ChangeOp 解釈可 | |

## 構造規律 (破綻防止 6 ヶ条)

1. **各 PoC は単独実行可能** (`py -3.11 scripts/poc_<n>.py` で完走、共有 state なし)
2. **falsifiable 命題を最初に明文化**してから書く
3. **破綻ゲートを before/after で計測**
4. **mock 中心**、実 LLM / 重みは Stage 後半まで触らない
5. **llive 資産は比較実験 (optional) のみで import**。本流は llcore 自前。改造したい個所は llcore 内で自前実装し、llive 互換は気にしない
6. **PoC battery 文化** (進化要素 6 要素 battery と同じ規律)

## ディレクトリ構成

```
llcore/
├── pyproject.toml          # llmesh-llcore 0.1.0a0
├── README.md
├── LICENSE / LICENSE-COMMERCIAL / NOTICE
├── src/llcore/
│   ├── __init__.py
│   ├── state_update/       # 数式遺伝子表現 (Stage 0/3)
│   ├── verifier/           # Z3 不変量 (Stage 1)
│   ├── factor_hook/        # llive factor_hook の薄い consumer (Stage 2)
│   ├── rad/                # RAD コーパス path-based lookup (llive 非依存)
│   └── poc/                # 共有 fixture
├── scripts/                # 段階的 PoC スクリプト (poc_0a, poc_0b, ...)
├── tests/unit/
└── docs/poc/               # 各 PoC の verdict + honest 留保
```

## 投稿先 (本研究の出口)

1. **TMLR** (本命, no hard deadline)
2. **GECCO 2027 short paper** (1月/2月 deadline)
3. **NeurIPS 2026 workshop** (verification × ML)

## 関連

- llive: https://github.com/furuse-kazufumi/llive

## honest 留保

- パッケージ版数 v0.1.0a0 = pre-alpha (PyPI 配布物としては未成熟)。研究成果の正本は `research/` 配下
- 研究スコープ: n=8 CPU reservoir byte-LM + スカラー記憶 kernel の PoC スケール。gradient-trained
  Transformer の本格統合・n=32+ は roadmap (論文 §10)
- llive と将来融合する流れだが、現段階では完全分離
- 1B 級 scratch CPU 学習は不可能、機構実証 (mechanism feasibility) に絞る
