# 「進化可能な LLM」FW — 確定再設計計画 v2(評価枠組み主軸)

**作成**: 2026-06-09 / **改訂**: v2(フォーク (b) 採択 + red-team must-fix 全反映)
**前提文書**: `docs/SYSTEMATIZATION_2026_06_09.md`(体系化=GUARANTEE は立つ / CAPABILITY は decisive NEGATIVE / 4 必要条件 / 賭け 1-4)、`research/rllm_pivot/topology_evolution_prior_art.md`(prior-art)、`docs/EVOLVABLE_LLM_PLAN_REDTEAM_2026_06_09.md`(敵対 red-team 決定メモ=F1-F13 + 数理判定 + 戦略フォーク)
**意思決定**: 5 設計案 → 3 judge 採点(E=20 最高 / A=B=19 / C=18 / D=16)→ 4 レンズ敵対 red-team → **ディレクター判断で主軸 = (b) Verified-Plasticity Evaluation Framework を採択**。
**規律**: 全編 honest-disclosure。capability(進化が perplexity/CE で勾配を上回る)と guarantee(進化した個体が証明付きで安定)を**決して混同しない**。load-bearing な実装事実は `src/llcore/` 一次照合済。

> **一行宣言(v2)**: 本 FW の主軸は **Verified-Plasticity Evaluation Framework** — 「実小型 LLM の online 構造適応が *発散しない・収縮する(ρ<1 を sound に保つ)* か」を第一級指標に据え、llcore 唯一の confirmatory 資産(6 装置の統計的厳密性ハーネス)で **任意の候補 method を falsifiable に測る評価枠組み**。VSOA(cert_inf-gated topology evolution)は枠組みで測る**最初の被験 method**、capability terrain-bet は **Phase2 必須の上振れ実験**(EXISTS/NULL/ARTIFACT verdict を産む)。この主軸転換で red-team の fatal 3 件(F2 退避先 / F4 score 逆転 / F7 主軸独立性)が構造的に解消する——**機構が失敗しても「枠組みの妥当性 + 測定された negative」が deliverable として残る**。

---

## ① エグゼクティブ・サマリ

**FW を一文で**: 実小型 LLM(SmolLM2-135M)に後付けした **n≤16 の verified recurrent adapter block** の online 構造適応について、「**成長操作下でも収縮 certificate(ρ<1)を sound に保てるか**」「**block 間 coupling 込みで合成安定か**」「**離散トポロジー軸は多峰か**」を、事前登録・Holm 連言・artifact 規律・反証条項・自己検出力監査・反 over-claim critic の **6 装置**で測る評価枠組み。複数 method(VSOA cert_inf-gate / 無 gate baseline / STABLE 風経験 gate / Mamba stable-by-construction 正の対照)を plug-in して比較する。

**価値命題(評価枠組み = deliverable)**: 売りは「**provably-stable online structural adaptation を測る、再現可能で falsifiable な評価枠組み**」そのもの。既存 NAS が accuracy/latency/FLOPs で compete するのに対し、本枠組みは「発散しない・収縮する構造適応」を第一級指標にし、stability-plasticity の TRIZ 矛盾を **guarantee 側から測る**。**capability(perplexity 改善・進化が勾配に勝つ)は一切売りにしない**(M3 戒め)。被験 method が何を出しても(PASS でも第一級 negative でも)枠組みの妥当性が deliverable になる——これが脆弱な単一機構に賭けない設計。

**最大の賭け(数理判定済)**: 枠組みの最初の被験 method VSOA の存立は「width_grow 後も cert soundness が保たれる **非自明な進化価値を持つ両立帯 ε>0** が存在するか」に賭かる。red-team 数理判定 = **条件付き成立**(dead-on-arrival ではない: per-row 不変条件で soundness 保存可、EXP5b 0/2000。だが自明経路=死んだ unit では無進化)。

> **★【Phase −1 実測 完了 2026-06-09, $0/CPU → `research/rllm_pivot/PHASE_M1_VERDICT.md`】**: 両立帯は **dead-on-arrival でない** — *sound 緩和* certifier **cert_two**・**小 n** で実在(**n=4 で 58-67%**, change@ε_max ~8.5% ≫ τ=5%, band 幅 0.15-0.39 / **n=8 で 33-35%**, change@ε_max ~4% = borderline)。だが **唯一スケールする最保守 cert_inf では全 n で実質空**(change@ε_max <1%=sound 域では構造成長が関数をほぼ動かせない)。ti=1 支配 96-100% で **red-team F1(脅威は per-row off-sum 増、box 拡大でない)を実データ確認**。→ **make-or-break は賭け1(width_grow 自体)から賭け2(navigable な certifier を 2^n 壁を越えて scale できるか)へ collapse**(cert_two は navigable だが 2^n で非スケール / cert_inf はスケールするが navigable でない=体系化の L3 inf-trap/sound 緩和/次元の壁を構造成長レベルで再現)。**設計含意**: per-component block は cert_two が affordable な **n≤4-6 + cert_two gate** に小さく切る(cert_inf では band が開かないため per-component gate を cert_two に格上げ)。

---

## ② 主軸選定(フォーク (b))と解消された fatal

### 2.1 5 設計 × judge × red-team の意思決定マトリクス(F4 反映)

| 案 | angle | judge total | honest_align | 確立済資産活用 | ユーザーゴール適合 | 存立条件の脆さ | 採否 |
|---|---|---|---|---|---|---|---|
| **E** | Verified-Plasticity Evaluation Framework | **20** | 5 | **6 装置を前景化** | 「FW 確立」を評価枠組みとして満たす | **FAIL でも生存** | **主軸採択** |
| A | guarantee-first VSOA | 19 | 5 | 6 装置を埋没 | 実進化 FW だが脆い | 単一点 collapse(F1/F7) | **最初の被験 method へ** |
| B | terrain-capability-bet | 19 | 4 | — | capability 存否を決着 | 最尤 NULL | **Phase2 必須副線へ** |
| C | role-split verified-NAS | 18 | — | — | 三者分業 framing | A と同系 | framing を③に流用 |
| D | literal verified-NAS | 16 | — | — | 字義的だが novelty 最狭 | — | 不採用 |

### 2.2 主軸を E にしたことで解消された red-team fatal

- **F2(退避先が価値ゼロ着地の承認ゲート)解消**: 評価枠組みが主軸なので、VSOA の機構失敗は「退避」でなく **measurable な第一級 negative=評価資産**。「別物の退避先」を作る必要が消える(枠組み自体が destination)。
- **F4(judge 最高 E を退避先に降格した score 逆転)解消**: E を主軸に据え直し、確立済 confirmatory 資産(6 装置)を前景化。
- **F7(主軸独立性の崩壊)解消**: 主軸が単一機構(width_grow×cert_inf)に依存しなくなる。method-agnostic な枠組みは存立条件 FAIL でも生存。

> ⚠ honest 留保: (b) でも普及ファネルは別途空白になりうる(F11)。「評価枠組み」自体は地味=§⑫ で consumer story + 動きで魅せるデモ + 需要側証拠 を必須対応とする。

---

## ③ 確定 framing — Verified-Plasticity Evaluation Framework

### 3.1 FW の定義(主軸)

> **「進化可能な LLM」FW = 実小型 LLM に後付けした verified recurrent adapter の online 構造適応 method を入力に取り、(i) 成長操作下の収縮 soundness、(ii) block 間 coupling 込みの合成安定、(iii) 離散トポロジー軸の多峰性、(iv) capability の存否(terrain EXISTS/NULL/ARTIFACT)を、6 装置の統計的厳密性ハーネスで falsifiable に測り、method 間で比較する評価枠組み。**

被験 method を差し替え可能にする 3 plug-point(`minimal_ga` の `GeneCodec`=基質 / `Objective`=方向 / `VerifierBackend`=gate)を **framework 約束として明文化・テスト化**(F8)。

### 3.2 何を測るか(被験 method、§④で詳述)

VSOA(cert_inf-gated topology evolution)/ 無 gate baseline / STABLE 風経験 gate / Mamba stable-by-construction(正の対照)。

### 3.3 6 装置(llcore 唯一の confirmatory 資産=枠組みの背骨)

事前登録→結果順 / Holm 連言 / アーティファクト規律 / 反証条項 / 自己検出力監査 / 反 over-claim critic。SYSTEMATIZATION §3.6 で framework 化済・自己検出力監査が n=15 で gate 健全と確認(suppress=False)。**この方法論層なしに「進化が本物」と主張する権利は成立しない**——だからこれを deliverable の核に据える。

---

## ④ 被験 methods(plug-in 比較対象)

| method | 役割 | 内容 |
|---|---|---|
| **VSOA(cert_inf-gate)** | 最初の被験 method | 重みは gradient、進化は離散トポロジー(width/branch/op)、訓練ループ内 per-component cert_inf fail-closed gate が ρ<1 で admit/reject |
| **無 gate baseline** | 負の対照 | gate なし。HD-1 で ρ→1.95 発散(既確立)。枠組みが「危険な method」を検出できるか |
| **STABLE 風経験 gate** | 既踏比較 | stability budget を経験メトリクス(EM drop/KL)で clip-or-reject(arXiv:2510.16089)。sound cert_inf との soundness/コスト比を測る |
| **Mamba-130M(stable-by-construction)** | 正の対照 | 非正の最大 Lyapunov 指数(arXiv:2406.00209)で cert が自明 PASS。枠組みの判別力(SmolLM2 で reject 発生 / Mamba で自明 PASS)を示す |

> 枠組みの妥当性 = 「負の対照(無 gate)を危険と・正の対照(Mamba)を安全と・既踏(STABLE)を soundness で区別して測れるか」。これが §⑥ North Star の中核。

---

## ⑤ base 選定(変更なし)

| 役割 | モデル | license | 理由 |
|---|---|---|---|
| **主 base** | **SmolLM2-135M** | Apache-2.0 | certificate 計算成立帯(Lipschitz-Transformer arXiv:2507.13338 が 2M〜145M)と重なる唯一の安全帯。T4 で full fine-tune + 多世代探索が現実的。Qwen 回避(商用障壁) |
| **正の対照** | **Mamba-130M** | Apache-2.0 | stable-by-construction = cert 自明 PASS の正の対照。GPU カーネル依存ゆえ副経路のみ(主経路は T4 純正) |
| **再現性予備** | Pythia-160M | Apache-2.0 | 学習過程 checkpoint 公開 |
| **除外** | Gemma 系 | Apache/MIT 要件外 | license 除外 |

> ⚠ param/license は採用前に各モデルカード再確認。base 凍結 + adapter のみ可変で T4 budget を守る。1B 級 scratch CPU 学習は不可能と明言。

---

## ⑥ 第一級指標と North Star(F3/F8 反映: トートロジー除去 + framework 性追加)

**削除した非新規命題(F3)**: 「certified-stable rate=100%」(=admit の定義の言い換え=トートロジー)、「gate付き<無gate 単純 drift 比」(=機能する filter の定義効果、tiny で既 PASS)は North Star から**削除**。

**North Star = 5 軸(falsifiable かつ新規なもののみ)**:

1. **成長操作下 soundness(新規・主命題, F3)**: width_grow/branch_add で構造を成長させた後も `_infnorm_sup` の box-bound が独立 eigen 再検査と不一致(false-admit)を起こさない。**成長操作 N 回中 false-admit ≥1 で FAIL**(=存立条件と一本化)。
2. **coupling-aware 合成 soundness(新規・第二存立条件, F6)**: 2 block を residual で結合した最小系の**実 Jacobian の真 ρ** を独立 eigen で測り、per-block cert が admit した構成で合成 ρ≥1 が 1 件でも出れば FAIL。per-block AND を禁止し block 間 coupling 込み cert を要求。
3. **枠組み判別力(F4 → 枠組み妥当性)**: 負の対照(無 gate=危険)/ 正の対照(Mamba=安全)/ 既踏(STABLE=経験 gate)を soundness で区別して測れる。
4. **framework 性(F8)**: (a) N 世代後の admit topology が param-shift baseline 比で構造的に多様化し、その多様性が held-out tasks への汎化に load-bearing。(b) 新 base / 新 changeop / 新 certifier を 1 オブジェクト差替で載せ替えられる拡張性(3 plug-point をテスト化)。
5. **capability verdict(Phase2 必須, F12)**: terrain-design で `EXISTS/NULL/ARTIFACT` を proper power で 1 つ確定。

**主軸 PASS から外す(NULL 許容副次, F7/F13)**: H-forgetting(忘却抑制)は H2 系 3 件が系統的 NULL(memory 軸 Δ≈+0.0134 極小)ゆえ高確率 NULL を事前登録で許容、立てば bonus。

---

## ⑦ 進化機構 / ChangeOp の実構造手術化(F1 per-row 訂正反映)

### 7.1 何を進化させるか

recurrent adapter block(`CoupledNDGene` の `(decay∈[0,1]^n, W∈[-2,2]^{n×n})`, n≤16)の**離散トポロジー自由度**(branch 追加/削除、op 入替、width ±1)。連続 `(decay,W)` は gradient、進化は gradient の解けない離散層を担当する分業。

### 7.2 ChangeOp の実構造手術化(pivot 必須・未実装)

**⚠ 現 ChangeOp は実構造手術でない**(一次照合済 `changeop.py:160-193`): scalar 3 float のみ操作、`kernel_swap_mock` は「実 NN kernel 交換でなく mock スイッチ」と明記、`CoupledNDGene` の n 次元構造に触れない。新規実装を要する。

| move | 実装 | gate(F1 訂正済) |
|---|---|---|
| `width_grow` | W を `(n)→(n+1)` に **Net2Net** function-preserving 拡張(arXiv:1511.05641) | **per-row 不変条件**: 新 column 寄与後の各既存行 `Σ_j|W[i,j]| ≤ 元` を保ち `ti=1` での sup が 1 を越えないこと(✕ box 拡大ではない) |
| `branch_add` | 並列 recurrent path 追加 | **block 間 coupling 込み**の結合 cert(per-block AND 禁止, F6) |
| `op_swap` | kernel 種別の*実*置換(mock でない) | 置換後 per-row sup 再計算 |
| param shift(既存) | decay/mix/gate_shift | per-component cert_inf 再計算 |

> **F1 数理訂正(最重要)**: `_infnorm_sup`(`backends.py:111-119`)は per-row sup の max で `ti∈{t_lo[i],1.0}` 端点。sup は 99.6% の行で `ti=1` 達成(box 幅と無関係)。よって真の越境脅威は「**新 column が既存行 i の `off_i=Σ_{j≠i}|W[i,j]|` を増やし `ti=1` の sup が 1 超**」=per-row abs-sum 増(EXP4 が PASS→FAIL 0.847→1.039 実測)。「box 拡大」は起きない脅威で、旧 stress-test はこれを偽 PASS していた。width_grow gate は per-row 不変条件で fail-closed reject。
>
> **両立帯の本質(③ honest)**: per-row 不変条件で soundness は保てる(自明)が、結合ゼロ=関数保存だが死んだ unit(無進化)、結合非ゼロ=進化するが abs-sum 増、の間に「**非自明な進化価値を持つ ε>0 帯**」があるかが真の存立条件。Decision gate の PASS に「非自明な進化価値を持つ admit ≥1 件」を AND で課す(死んだ unit の自明 PASS を排除)。

各 move は世代横断で per-row + coupling cert に問い合わせ、破れば fail-closed reject。**gate は安定 certificate を破らないことのみで gate し、fitness(capability)で gate しない**=適応は許すが発散・忘却は許さない homeostatic constraint。

---

## ⑧ 事前登録 existence-bet(真に新規な命題のみ)

### 8.1 主軸(枠組み妥当性 + 真に未検証な soundness)

> **H-growth-soundness(主命題)**: SmolLM2-135M に後付けした n≤16 adapter に対し、width_grow/branch_add で構造成長後も per-row cert_inf が **0 *観測* false-admit** を維持(成長操作 N 回中 false-admit ≥1 で FAIL)。
>
> **H-coupling-soundness(第二存立条件, F6)**: 2 block residual 結合の実 Jacobian 真 ρ を独立 eigen で測り、per-block cert が admit した構成で合成 ρ≥1 が 1 件でも出れば FAIL(per-block AND の coupling 盲点を正面検定)。
>
> **H-discriminative(枠組み妥当性)**: 枠組みが 無 gate(危険・ρ→1.95)/ Mamba(安全・自明 PASS)/ STABLE(経験 gate)を soundness で区別して測れる。

### 8.2 副線(capability terrain-bet, Phase2 必須, F12)

> **H-multimodal(前提条件, F9)**: 離散トポロジー軸(width/branch/op)が多峰か。width_grow greedy baseline vs MAP-Elites archive を同予算比較し、greedy が並べば**単峰=capability 立たず**と事前宣言(M3 が離散軸でも再現)。
>
> **H-EXISTS**: 多峰が確認された fitness family 上で、ρ<1 gate 付き MAP-Elites が同予算 gradient/random を fresh-seed held-out CE で honest_eval 4 条件 AND(`diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n_seeds≥15 ∧ |paired_sign_delta|≥0.147`)で上回る constellation が ≥1 存在する。
>
> **capability-vs-artifact meta-gate(BG10)**: 勝った constellation で gradient-on-same-terrain でも利得が消えなければ `EXISTS`、消えれば `ARTIFACT`(navigability 最適化現象)→ guarantee 主軸へ。立たねば `NULL`(実 small-LLM 地形は単峰=decisive negative=研究成果)。

### 8.3 negative 全面許容(撤退でなく評価資産化)

- 両立帯 ε>0 が空 → 「contraction-gate は動的構造成長と両立しない」を第一級 negative として枠組みが測定(deliverable)。
- ρ<1 強制で適応性能が有意劣化 → 「contraction gate が可塑性を殺す」を反証・honest 開示。
- terrain NULL/ARTIFACT → 「実 small-LLM 損失地形は単峰/navigability artifact」の decisive negative。
- H-forgetting NULL → 高確率を事前許容済、主軸 PASS から除外。

---

## ⑨ Kaggle feasibility-first PoC

| 項目 | 内容 |
|---|---|
| **最大の計算リスク** | 構造変更 1 回ごとの cert(per-row + coupling)× N 世代 × M block が budget を食うか。**width_grow が n を成長させると cert コストが n² で増大** |
| **cert_inf コスト見込み** | per-block O(n²)・頂点列挙ゼロで n≤16 なら μs〜ms 級=forward/backward に対し無視可の見込みだが**成長 n でも要実測** |
| **PoC 合否** | §⑩ Phase -1(純数値 scan)→ Phase0(load+fine-tune)→ Phase1(変異1回+cert1回+CE1回の wall-time/MB 実測 → 30h に N 世代収まるか外挿、成長 n で再計測) |
| **超過時縮小** | base 固定、block n を 8 へ、family 数削減。1B scratch CPU は明示放棄 |
| **resumable** | MAP-Elites archive / checkpoint で 30h 跨ぎ対応 |

---

## ⑩ Phase −1/0/1/2 ロードマップ + decision gate

### Phase −1 — 純数値 両立帯 scan(F5: 最優先・実装投資ゼロ)
- SmolLM2/Net2Net **不要**。`_infnorm_sup`/`_t_min` だけで synthetic `CoupledNDGene` を n→n+1 拡張し、新 unit 結合 `|W[i,n+1]|,|W[n+1,j]|` を 0 から増やしながら (a) 関数が非自明に変わる かつ (b) 全既存 row `_infnorm_sup<1` を保つ **ε>0 帯が存在するか**を実測。
- **max_input_abs 較正**: SmolLM2 入力 `abs(x)_inf` を実測し sound 上界に設定(現 1.0 ハードコード `backends.py:154/171` の box 被覆未検証を是正)。
- **Decision gate −1 【実測済 2026-06-09 → PHASE_M1_VERDICT.md】**: ε>0 両立帯は **cert_two・小 n(n≤4-6)で存在 → GO**(VSOA を最初の被験 method として続行、ただし per-component gate は cert_two・small-n に確定)。**cert_inf では空**=その regime では「成長と両立不能」の第一級 negative を枠組みが記録(枠組みは生存)。**残る最尤失敗 = 賭け2(navigable cert の scale)**: vertex-free cert が cert_two 並みの navigability を高次元で保てるか(体系化は B2 が n=16 で cert_inf に収束=navigability 喪失を既示)。次の純数値 scan 候補 = SDP 比較(小 n)+ vertex-free B2 の navigability scale 測定。

### Phase 0 — base 継承 + instrument 校正
- SmolLM2-135M load + 数百 step fine-tune が T4 で回ることを確認。Mamba-130M を正の対照に。
- 1 個の adapter block(n≤16)を特定。**トポロジー全探索はしない。**
- **多峰性 instrument 校正(F9)**: 構成 fitness family が本当に多峰/欺瞞かを決定論化(eval_noise を機械 eps へ)し valley_fraction で検証(positive control=合成多峰、negative control=ESN 単峰)。
- **Decision gate 0**: load+fine-tune が T4 で回る → GO。

### Phase 1 — 被験 method 測定(soundness 主役)
1. per-component cert_inf(`_infnorm_sup<1`)で ρ 上界を安く計算する関数を実装、固定構造で ρ が測れることを確認。
2. ChangeOp を実構造手術へ拡張(`width_grow`=Net2Net + **per-row 不変条件 gate**)を `width_grow` 1 種だけ実装。
3. **存立条件 stress(per-row, F1)**: width_grow 1 回で**各既存行 abs-sum が `ti=1` sup を 1 超させない**ことを stress 検証(0 false-admit を成長操作下で再確認)。**PASS 条件に「非自明な進化価値を持つ admit ≥1」を AND**。
4. **coupling stress(F6)**: 2 block residual 結合の真 ρ を独立 eigen で測り per-block AND の盲点を検定。
5. mutation×gate の 1 ループを回し feasibility(変異1回+cert1回の秒/MB → 30h 外挿、成長 n で再計測)。
- **Decision gate 1**: (3) PASS ∧ (4) PASS ∧ (5) feasibility PASS → Phase 2。いずれか FAIL → 枠組みが第一級 negative を記録し被験 method を切替(撤退でなく測定)。

### Phase 2 — 枠組み妥当性 + capability 必須副線 + framework 性
- 4 method(VSOA / 無 gate / STABLE / Mamba)を枠組みで比較し **H-discriminative** を事前登録検定で示す。
- **capability 副線(必須, F12)**: terrain family 上で MAP-Elites vs gradient vs random を honest_eval → meta-gate で EXISTS/NULL/ARTIFACT を 1 つ確定。
- **framework 性(F8)**: topology 多様化の汎化 load-bearing + 3 plug-point 拡張性をテスト化。
- adapter の実 LLM load-bearing(Stage-B B-G1「benefit が core dim と増大」)を SmolLM2 で再現(F10)。
- **Decision gate 2**: H-discriminative + framework 性 PASS → 評価枠組みとして結実。capability EXISTS → 普及の派手な軸を追加。

---

## ⑪ make-or-break 賭け 1-4 への回答(主軸が評価枠組みになり弱まった)

| 賭け | 本計画 v2 の回答 | negative 時 |
|---|---|---|
| **賭け 1(存立条件 width_grow×cert)** **【実測済 → 緩和】** | Phase −1 純数値 scan 完了(PHASE_M1_VERDICT.md): 両立帯は **cert_two・小 n で実在**(n=4 で 58-67%)、**cert_inf では空**。→ **賭け1 は cert_two・small-n で PASS、make-or-break は賭け2 へ collapse** | cert_two・small-n でも非自明な進化価値が出ねば固定 topology / 経験 gate / Mamba 比較へ切替、枠組み生存 |
| **賭け 2(2^n 壁 = 本丸の make-or-break)** **【Phase −1 で昇格】** | Phase −1 実測: 両立帯は **cert_two(navigable)でのみ開き cert_inf では空**。cert_two は 2^n で非スケール。**∴ per-component を n≤4-6 + cert_two に小さく切る**(2^6=64 頂点 affordable、block 全体でなく小部品ごと)。vertex-free cert が cert_two 並み navigability を高次元で保てるかが未解決の本丸(体系化: B2 は n=16 で cert_inf に収束=navigability 喪失) | small-n per-component に留まる限り測定成立。scale 不可なら「VSOA は small-n per-component 限定」と開示、枠組みは他 method 比較で生存 |
| **賭け 3(transfer tiny→実 LLM)** | 「実 LLM」修飾を adapter scope に限定(F10)。adapter の load-bearing を Phase2 必須測定 | 未達なら「実 LLM 寄与は未確立」と開示 |
| **賭け 4(guarantee の scale 連結崩壊)** | L3「強い verifier が fitness 解放」を主軸に載せない。主軸は「成長操作下 + coupling 込みで 0 false-admit を保つか」の測定 | ladder 階梯が崩れても枠組みの soundness 測定は残る |

---

## ⑫ consumer story + 普及設計(F11: 評価枠組みの地味さを解く)

1. **consumer story 1 本**: FullSense 3 製品のどれに乗るかを確定 —
   - **第一候補 = llive 自己進化メモリ層**: llive の online 進化が verified gate で「発散・破滅的忘却しない」ことを本枠組みが fail-closed に保証・測定(`feedback_llive_measurement_purity` の on-prem 思想と整合)。
   - 副候補 = llmesh SPC 適応制御の online 構造適応を verified gate が守る。
2. **動きで魅せるデモ(guarantee 側, `project_f25_demo_polish`)**: 無 gate baseline が ρ→1.95 で**出力ノルムが発散していく** vs gate 付きが ρ<1 に留まる **リアルタイム可視化**=「破綻が止まる動き」=SNS 拡散素材。capability を捨てた分の派手さをここで作る。
3. **需要側証拠 or 明示判断**: guarantee-niche の market 価値(産業 online 学習の発散事故 / EU AI Act high-risk 連続学習の安定性要求 等)を 1 つ提示する。出せなければ「市場価値は未実証・研究 niche への賭け」と明記し、ユーザー判断を仰ぐ(この時点では (b) 採択で枠組み妥当性が deliverable と確定済)。

---

## ⑬ honest 留保(red-team が潰せなかった 5 リスク)

1. **両立帯 ε>0 の存在は未証明**: per-row 不変条件で soundness は保てるが、非自明な進化価値を持つ admit が存在する帯が空(=死んだ unit でしか cert を保てない)可能性は Phase −1 scan まで不明。空なら width_grow は無進化操作に縮退。**枠組みはこの negative を測定して生存するが、VSOA という被験 method は空回り。**
2. **transfer(賭け3)は本質的に未検証**: Stage-B tiny(~0.5M)→ SmolLM2-135M の load-bearing transfer は未検証領域。scope 限定(F10)で over-claim を防ぐのみ。
3. **novelty の狭さは原理的に解消不能**: VSOA の STABLE との 2 点 delta は事実、corpus 空白象限は外部再現不能。need-side 証拠(F11)を出せねば「誰も困っていない問題」の疑いは残る(=評価枠組み主軸でも市場の問題は残る)。
4. **coupling 盲点が原理的に soundness を崩す可能性**: F6 で第二存立条件に格上げするが、結合 cert に縮退すると賭け2(2^n 壁)と衝突。n=block 合計の feasibility 再判定が必要で、最悪「block を切る設計制約」自体が崩れる。Phase1 実測待ち。
5. **評価枠組み主軸でも普及ファネルは空白になりうる**: E は honest だが地味。F11(consumer story + デモ + 需要側証拠)を満たさず capability(副線)も NULL なら、派手な軸は構造的に存在せず「研究 niche への賭け」をユーザーが受容するかに依存。red-team が解決できない戦略的トレードオフ。

---

## 付録: 一次照合済み実装事実 / 関連

- `_infnorm_sup`(`backends.py:111-119`)= per-row sup の max, `ti∈{t_lo,1}` 端点 / `_t_min`(L96-98)/ `max_input_abs=1.0` ハードコード(L154/171, 較正対象)。
- `_gate_admits` 4 モード + fallback `(0.5,0,0)` + `resample_cap=50`(`minimal_ga.py:223-308`)。
- `changeop.py:160-193` = scalar 3 float のみ、`kernel_swap_mock` は mock(実構造手術は未実装)。
- honest_eval 6 装置 / `paired_sign_delta`(教科書 Cliff's delta でないと明記)。
- 関連 memory: `[[feedback_llcore_must_become_llm_relevant]]` / `[[project_llcore_gpu_3experiments_2026_06_06]]` / `[[feedback_benchmark_honest_disclosure]]` / `[[feedback_qwen_commercial_barrier]]` / `[[project_f25_demo_polish]]`。
- audit trail: 本 v2 は (a) VSOA 主軸版を red-team(`EVOLVABLE_LLM_PLAN_REDTEAM_2026_06_09.md` F1-F13)+ ディレクター判断(フォーク (b))で全面改訂したもの。
