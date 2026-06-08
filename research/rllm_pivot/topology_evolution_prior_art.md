# Topology Evolution × Verified Gate — 先行研究マップと差別化判定

**作成日**: 2026-06-08
**対象**: llcore Phase1 ピボット(実小型オープン LLM に「健全性 certificate ゲート付き構造進化」を Kaggle T4/30h 制約で載せる)の先行研究マップ・差別化候補・base 選定・最短経路。
**方針**: honest disclosure 厳守。引用は外部検証済みのみ near/closest 扱い。capability(進化が gradient/NAS に勝つ)主張は内部 VERDICT と矛盾するため封印し、**guarantee(発散しない/収縮する構造変更)を第一級メトリクス**に据える前提で評価する。
**根拠源**: RAD コーパス(`D:/docs/verified_safe_learning_corpus/`, `D:/docs/evolutionary_computation_corpus/`)+ WebSearch(2026-06 時点)。引用 arXiv 番号は検索/WebFetch で実在確認済(§⑧に捏造リスク整理)。

---

## ① 系譜要約 — NEAT / DARTS / NAS → LLM NAS の到達点と限界

「ノード/ブランチ(トポロジー)を動かす」機構の系譜は **3 波**に整理できる。各波で *機構* は枯れたが、**health certificate ゲートは一度も付いていない**。

| 波 | 代表 | 機構 | 計算コスト | 健全性ゲート |
|---|---|---|---|---|
| **進化トポロジー(古典)** | NEAT / HyperNEAT (Stanley & Miikkulainen 2002 / 2009) | node/connection を遺伝子化、innovation number で交叉整列、種分化で新規構造保護。HyperNEAT は CPPN で間接符号化 | CPU 完結・小規模(TensorNEAT 2024 で GPU 化) | なし(LLM 規模未到達) |
| **進化 NAS** | Regularized Evolution / AmoebaNet (Real et al. 2019, arXiv:1802.01548) | tournament + aging で進化 NAS を安定化、RL NAS に勝利 | GPU 3150 日級 — Kaggle 制約とは桁違い | なし |
| **微分可能 NAS** | DARTS (Liu, Simonyan, Yang 2018, arXiv:1806.09055) | node 間 edge の op を softmax 連続緩和、勾配で構造最適化 | 後続(PC-DARTS/ZO-DARTS/DASH)で単一 GPU 分〜時間 | なし(純 capability 最適化) |
| **コスト償却(weight-sharing)** | Once-for-All (Cai, Gan, Han 2020, arXiv:1908.09791) / AutoFormer (Chen 2021, arXiv:2107.00651, MS Cream) | 1 supernet を progressive shrinking、sub-net を再訓練なし抽出。AutoFormer は weight entanglement で Transformer sub-net | 探索コストを 1 回訓練に畳む | なし(accuracy/latency 多目的のみ) |
| **LLM NAS(現行 SOTA)** | **Jet-Nemotron / PostNAS** (NVIDIA 2025, arXiv:2508.15884) / ASI-Arch (2025, arXiv:2507.18074) | PostNAS は pretrained full-attention LLM の MLP を凍結し attention block 配置/選択のみ探索。Qwen3/Llama3 同等精度で最大 **53.6x** decode 高速化。ASI-Arch は LLM が構造を自律仮説→実装→訓練→検証(1773 実験/20000 GPU 時で 106 SOTA) | **200B token / 2 万 GPU 時** = Kaggle T4 では非現実的 | なし(経験 benchmark のみ、形式 certificate でない) |

**DARTS 系の既知の病理**(差別化材料): (a) supernet 全 op 同時保持でメモリ大、(b) parameter-free op(skip/identity)へ collapse する病理、(c) 探索空間を cell へ縮約=表現力犠牲。

**到達点の総括**: 「実 Transformer/LLM の block/branch を動かす」は **AutoFormer → Jet-Nemotron/PostNAS → ASI-Arch で実証完了**。ユーザー Phase1 機構の *対象適用性* は既に証明済み。残された限界は **(1) 探索コストが Kaggle 制約と桁違い、(2) どの波にも形式 health certificate ゲートが無い** の 2 点。llcore の差別化はこの 2 点の交差にしか残らない。

---

## ② Verified / 健全性ゲート付き構造探索の先行有無

軸を **A {verify-only / gate-synthesis / gate-evolution}** × **B {物理 dynamics / 抽象 code / LLM 認知核}** で整理する(`verified_safe_learning_corpus/16_gating-vs-verifying-synthesis.md` の枠組み)。

### 成熟している基盤(各構成要素は単独で先行あり = 盛れない)

| 技術 | 代表 | 状態 |
|---|---|---|
| **neural Lyapunov/CBF/contraction certificate** | Dawson-Gao-Fan T-RO survey 2023 / neural_clbf (MIT-REALM) / **NCM・NSCM** (Tsukamoto-Chung-Slotine, arXiv:2110.00693 / 2011.03168) | 成熟。incremental exponential stability の必要十分条件を NN 近似 + spectral-norm で安価な十分条件化 |
| **検証が訓練を gate(gate-synthesis)** | **CT-BaB** (Shi, Li, Hsieh, Zhang 2024, arXiv:2411.18235) | 訓練時 BaB で certified bound 最適化。2D quadrotor で検証時間 11x 短縮・RoA 164x 拡大。**ただし固定アーキ/固定方策の重みが対象** |
| **Lyapunov-stable neural control(出力/状態 feedback)** | arXiv:2404.07956 (ICML 2024) / RL 方策の一般化 Lyapunov 検証 arXiv:2505.10947 (NeurIPS 2025) | CEGIS + δ-complete 検証が揃う |
| **NN 検証エンジン** | α,β-CROWN(線形境界伝播 + BaB) | RNN/Transformer まで対応、SMT 回避の高速 sound 検証 |
| **収縮する Transformer は作れる** | **Training Transformers with Enforced Lipschitz Constants** (Newhouse et al. 2025, arXiv:2507.13338) / LipShiFT (ICLR2025) | spectral/Lipschitz 制約で収縮的・頑健に。**ただし固定アーキの訓練時正則化** |
| **stable-by-construction SSM** | Mamba SSMs Are Lyapunov-Stable Learners (arXiv:2406.00209) | 非正の最大 Lyapunov 指数、mixed-precision 摂動に安定 |
| **proof-gated self-modification(理論/経験の両極)** | Gödel Machine (Schmidhuber 2003, 実装不能) / **Darwin Gödel Machine** (Zhang, Hu, Lu, Lange, Clune 2025, arXiv:2505.22954, SWE-bench 20→50%) | 形式証明 gate と経験 benchmark gate の両極が既存 |
| **certified-robust NAS** | DSRNA (arXiv:2012.06122) / RACL / AdvRush | 認証下界・Lipschitz 制約を**探索目的に**組込む。ただし「進化の各構造変更を sound gate で admit/reject」ではない |

### gate-evolution × 形式安定 × LLM 認知核 = **空白象限**

`verified_safe_learning_corpus/16` の内部判定: **「形式安定 gate × 逐次自己改変 × LLM 認知核」の三つ組は corpus 全実装研究で空白**。Gödel Machine = 理論のみ実装不能、Darwin Gödel Machine = 形式保証を放棄。NAS 側の検証は構造的 well-formedness(edge が入力を受け出力が伝播するか)止まりで stability certificate ではない。安定性証明側は固定アーキ/固定重みが対象。**両者の交差点が llcore の核**。

### corpus 作成後に境界を狭めた近接 3 件(2025–2026, honest)

| 名前 | arXiv | llcore との差 |
|---|---|---|
| **STABLE: Gated Continual Learning for LLMs** (Hoy, Celik 2025) | 2510.16089 | LoRA 編集を stability budget(EM drop/Bits/KL)で clip-or-reject(Qwen-2.5-7B)。差は (a) **パラメータ編集でトポロジー変更でない** (b) **gate が経験的メトリクスで sound certificate でない** の 2 点のみ。gate 発想の構造が酷似 → 差別化幅を直接圧迫 |
| **COMP(構造剪定 × Lyapunov)** (Sundaram, Ulmen, Haider, Görges 2025) | 2508.08144 | Lyapunov 単調減少保存を pruning の primary constraint に。差は (a) **制御/latent モデル限定(LLM 非対象)** (b) 保証が **approximate(形式証明なし)**。⚠ 元データは "COM-PACT" と誤同定 — 正式名は **COMP**(acronym COMP)。"~22% sparsity" は abstract 未確認の未検証値 |
| **SSGM: Stability & Safety Governed Memory for LLM Agents** (Lam, Li, Zhang, Zhao 2026) | 2603.11768 | 進化する agent memory を consistency verification + temporal decay + access control で consolidation 前に gate。**形式証明/契約でなく手続き的ガバナンス** → 「sound certificate で memory 進化を gate」は依然空白。だが代替軸の空地は既に縮みつつある |

---

## ③ 小型 LLM 構造最適化 2026 の現状

| 機構 | 代表 | Kaggle 適合性 |
|---|---|---|
| **効率 NAS / weight-sharing** | OFA / AutoFormer / Cream / GrowTAS(小→大 sub-net 漸進展開, arXiv:2512.12296) | 「構造探索コストを 1 回訓練に畳む」最有力。実小型 LLM に構造探索を載せる際は OFA 的 weight-sharing が筋 |
| **pruning / growing(function-preserving)** | **Net2Net** (Chen, Goodfellow, Shlens 2015, arXiv:1511.05641) / **Network Morphism** (Wei et al. 2016, arXiv:1603.01670) | 幅拡張・深さ挿入を function-preserving に初期化(拡張直後も元と同関数)。**構造変更を安全に大きくする初等技術は確立**。ただし保証は「関数保存」止まりで online 力学の安定 certificate ではない = llcore の隙間の核に最も概念が近い既存物 |
| **後付け hybrid attention 探索** | Jet-Nemotron/PostNAS(MLP 凍結 + attention block 探索) | 機構は llcore とほぼ同型だが 200B token = 非現実的 |
| **zero-cost proxy** | (DARTS 系高速化の延長、grad-norm/synflow 等) | 探索を 1 回の forward/backward で近似 — Kaggle 制約下で candidate scoring を安くする補助に有望(certificate ゲートとは別軸) |

**現状の総括**: 2026 の小型 LLM 構造最適化は「**weight-sharing + 後付け探索 + zero-cost proxy で安く・速く capability を上げる**」方向に収斂。**health/stability certificate を探索ループに組む流れはこの系列に存在しない**(STABLE/COMP/SSGM が各 1 軸ずつ踏んだのみ、いずれも経験的 or approximate)。

---

## ④ base 選定材料表(Apache/MIT 小型オープン LLM 候補)

Phase0「ゼロから作らず実 LLM を継承」前提。Kaggle T4(16GB ×2, 30h)に **載る・継承できる・license 安全** の三点で評価。
⚠ param/license は公開情報からの記載だが、**採用前に各モデルカードで再確認すること**(§⑧ 留保)。

| 候補 | license | param | 構造 | Kaggle 適合 | 備考 |
|---|---|---|---|---|---|
| **Qwen2.5-0.5B / 1.5B** | Apache-2.0 | 0.5B / 1.5B | dense Transformer (GQA) | ◎ T4 で fine-tune 可、PostNAS 系の探索対象と同系統 | Jet-Nemotron が Qwen 系を base にした実績 = 機構移植が楽 |
| **SmolLM2-135M / 360M / 1.7B** (HuggingFace) | Apache-2.0 | 135M〜1.7B | dense Transformer | ◎ 135M なら T4 で full fine-tune・多世代探索が現実的 | **最小 base に最適**。Lipschitz-Transformer の certificate 計算が 2M〜145M で成立した事実(arXiv:2507.13338)と param 帯が重なる |
| **Pythia-160M / 410M** (EleutherAI) | Apache-2.0 | 160M / 410M | dense Transformer | ◎ | 学習過程 checkpoint 公開 = ablation/再現性に強い |
| **TinyLlama-1.1B** | Apache-2.0 | 1.1B | Llama 構造 | ○ T4 で LoRA/部分 fine-tune | Llama 系 attention block を扱う場合の参照 |
| **OLMo-1B** (AI2) | Apache-2.0 | 1B | dense Transformer | ○ | 完全 open(data/code/weights)で provenance 強 |
| **Mamba-130M / 370M** (state-spaces) | Apache-2.0 | 130M / 370M | SSM | ○(GPU カーネル依存) | **stable-by-construction(arXiv:2406.00209)** = certificate ゲートの「安定な構成素」アンカーに最適 |
| Phi-3.5-mini | MIT | 3.8B | dense Transformer | △ T4 で full は重い | param がやや大、探索の多世代回しには不利 |
| Gemma-2-2B | Gemma license(**Apache/MIT でない**) | 2B | dense Transformer | △ | ⚠ license が Apache/MIT 要件外 — 候補から除外推奨 |

**選定示唆**: certificate 計算の成立帯(2M〜145M, arXiv:2507.13338)と多世代探索の現実性から、**SmolLM2-135M または Pythia-160M を最小 base**、stable-by-construction 比較用に **Mamba-130M** を副 base に置くのが最も筋が良い。

---

## ⑤ 交差点ギャップ判定(候補象限 A / B / C)

| 象限 | 主張 | 最近接先行(検証済) | 未踏度判定(honest) |
|---|---|---|---|
| **A = verified-topology-evolution** | NAS の構造変異(node/branch 追加・移動・削除)を **収縮/安定 certificate で gate** | Net2Net/morphism(局所関数保存 = 安定 certificate でない, near) / CT-BaB(固定アーキの形式安定, near) / Jet-Nemotron(無 certificate の LLM 構造探索, near) / COMP(剪定 × Lyapunov だが LLM 非対象・approximate, near) | **最も未踏**。三者とも「構造変更を形式 certificate で gate」を満たさない。corpus 内部注釈が「LLM evolution の cognitive-core 形式ゲートは未踏」と明記 |
| **B = verified-memory-evolution** | 記憶バンクの健全性検証つき進化 | **SSGM**(手続き的ガバナンスで先行占有開始, 2026, medium) / TAME(test-time memory 進化 benchmark) | **競合発生中**。SSGM が consistency 検証で踏んだ → 空地でない。「sound certificate に格上げ」する余地は残るが novelty は **下方修正必須** |
| **C = 証明付き online 適応** | online/test-time の構造適応に「発散しない」形式保証 | contraction theory(Tsukamoto/Slotine, 理論基盤あり) / test-time LLM learning(LoRA + heuristic で forgetting 緩和のみ, 無保証) / ONE-NAS(continual NAS だが stability-plasticity は経験的) | **未踏だが最大リスク**。contraction で online 構造適応の有界性を保証する研究は不在(空白の傍証あり)。Transformer attention への収縮 certificate の tightness/scalability が未知数 |

**判定**: **象限 A(verified-topology-evolution)が最も未踏度が高い**。B は SSGM/TAME で競合密度が上がりつつあり「安全な逃げ場」ではない。C は理論基盤はあるが feasibility が最も不確実。**guarantee で勝てる niche = A**、ただし正味の貢献は「**機構の発明でなく certificate × NAS mutation × 実 LLM × 安さ の統合と縮約**」に限定される(honest)。

---

## ⑥ 差別化候補 3 案

### 案1: Contraction-Gated Topology Mutation(本命, 象限 A)
- **主張**: NAS の mutation(構造変更)を、変更後の online 力学が **ρ<1(収縮)**を満たすか CROWN/spectral-norm の安価な十分条件で判定し、満たさなければ fail-closed で reject する。capability ではなく「**発散しない構造変更**」を第一級メトリクスにする。
- **最近接先行**: CT-BaB(固定アーキの gate-synthesis, near) / Net2Net(局所関数保存, near) / STABLE(パラメータ編集の経験的 gate, near)。**いずれも「構造変更 × 形式 certificate × LLM」を同時に満たさない**。
- **未踏度**: 高。corpus が「cognitive-core 形式ゲートは未踏」と明記。
- **Kaggle PoC 可能性**: 中。最大リスク = 構造変更ごとの certificate 検証が T4/30h budget に収まるか(計算可能性)。SmolLM2-135M 級なら成立見込みだが要実測。

### 案2: Stability-Budgeted Structure Search(象限 A × C の橋)
- **主張**: STABLE の「stability budget で edit を clip-or-reject」を **パラメータ編集からトポロジー変更へ昇格**し、budget を経験的メトリクスから **spectral-norm 由来の sound 上界**に置換する。
- **最近接先行**: STABLE(arXiv:2510.16089, near — 構造が酷似)。**差は「トポロジー変更」と「sound certificate」の 2 点のみ** → 差別化幅は狭い(honest)。
- **未踏度**: 中。STABLE が gate 発想を踏んでいるため、貢献は「対象の昇格 + 保証の格上げ」に限定。
- **Kaggle PoC 可能性**: 高。STABLE が Qwen-2.5-7B で動いた骨格を流用でき、最小 base で再現しやすい。

### 案3: Verified-Plasticity Evaluation Framework(評価枠組みの novelty)
- **主張**: 既存 NAS が accuracy/latency/FLOPs で compete するのに対し、「**発散しない・収縮する online 構造適応**」を第一級メトリクスにした探索・評価枠組み(stability-plasticity の TRIZ 矛盾を guarantee 側から測る)を提示。
- **最近接先行**: STABLE(経験的 budget) / ONE-NAS(経験的 stability-plasticity)。**形式保証側からこの矛盾を測る枠組みは空白**。
- **未踏度**: 中〜高。機構でなく「何を第一級指標に据えるか」の novelty。内部 negative 結果(capability で勝てない)と最も整合し honest。
- **Kaggle PoC 可能性**: 高。指標定義 + 小規模 ablation で示せる。実装重量が軽い。

---

## ⑦ 最短経路(Kaggle 前提, feasibility-first)

`feedback_poc_feasibility_first` 準拠 — **「安さ × 形式保証」が両立するかを最初の 1 本で潰す**。

### Phase0: base 継承(リスクほぼ無)
1. **SmolLM2-135M(Apache-2.0)を base に固定**。Kaggle Notebook で load → 数百 step fine-tune が T4 で回ることを確認。副 base に Mamba-130M を controls 用に置く。
2. 構造を「動かせる」最小単位を 1 個だけ特定(例: 1 つの attention block / 1 つの recurrent (decay, W) ブロック)。ここでは **トポロジー全探索をしない** — 「最初の 1 構造部品」に絞る。

### Phase1: 最初の 1 構造部品の検証付き進化
3. **certificate verifier の実装**: その 1 部品の出力作用素に対し spectral-norm(または CROWN)で ρ(収縮率)の上界を安価に計算する関数を作る。**まず固定構造で ρ を測れることを確認**(NCM/spectral-norm を実装基盤に転用、arXiv:2110.00693)。
4. **mutation × gate の 1 ループ**: その部品に「小さな構造変異(branch 追加 / 幅 ±1 / op 入替)」を 1 種類だけ適用 → 変異後 ρ を再計算 → **ρ<1 を満たせば admit、満たさなければ reject(fail-closed)** の閉ループを 1 世代回す。
5. **feasibility 測定(最重要)**: この「変異 1 回 + certificate 検証 1 回」が T4 で何秒/何 MB か実測し、**30h budget に N 世代分が収まるか**を外挿する。ここが PoC の合否判定。
6. **homeostasis 測定**: admit された変異だけを世代横断で積むと、reject なしの baseline(STABLE 風経験 gate / 無 gate)に比べ **online 適応が発散しない**ことを示す。capability(perplexity 改善)は二の次、**guarantee メトリクスを主軸**に報告。

### 事前登録(existence-bet)案
- **登録仮説**: 「SmolLM2-135M の 1 attention block に対し、ρ<1 fail-closed gate 付き構造変異の 1 世代閉ループは Kaggle T4/30h budget の **X%以内**で完走し、無 gate baseline より online drift(出力ノルム発散)が有意に小さい」。
- **falsifiable**: certificate 検証が budget を食い潰す / ρ<1 を保つと capability が baseline より有意に劣化する(可塑性が殺される)— どちらかが出れば **案1 は不成立、案3(評価枠組み)へ退避**と事前に宣言。
- **M3 戒め**(§⑧): 「進化が gradient に勝つ」を主張に**入れない**ことを登録時点で固定。

---

## ⑧ 正直な留保(honest disclosure)

### 引用検証で出た捏造/誤同定リスク
- **COMP の誤同定**: 元データの "COM-PACT" は誤り。正式名は **COMP**(*COMponent-aware Pruning*, Sundaram et al. 2025, arXiv:2508.08144)。"~22% sparsity" 値は abstract 未確認 = **未検証の specific** として扱う(本文で断定しない)。
- **「cluster 集約」框組み**: Lyapunov/CBF/Contraction クラスタ(arXiv:2411.18235 / 2404.07956 / 2505.10947)の CEGIS/δ-complete 框組みは **複数論文の集約**であり、単一論文が全主張を言っているわけではない。各 ID は実在・on-topic 確認済だが「1 本がこれを全部やった」とは書かない。
- **Net2Net の年**: 2015 初出(arXiv:1511.05641)、引用の "2016" は ICLR pub year。"Evdi 系" は非公式グルーピング(特定 ID なし)。
- **内部 VERDICT / corpus doc の「空白」主張**: `fullsense_*` VERDICT 群・`verified_safe_learning_corpus/16` の「空白象限」は **内部判断であり外部検証不能**。filesystem 上の path は解決確認済だが、「未踏」の断定は外部 reviewer には再現させられない弱い根拠であると明記する。
- **SSGM 本文**: PDF 圧縮で一部未読、abstract のみ確認(arXiv:2603.11768)。

### prior-art が濃い(= 新規性を主張できない)所
- **「トポロジー/ブランチを動かす機構」そのもの**は DARTS 系・NEAT 系・weight-sharing・morphism で **出尽くしている**。ここで新規性を主張するのは不可。差別化は **gate(certificate)でしか取れない**。
- **「実 LLM に後付けで構造探索」**も Jet-Nemotron/PostNAS(2025)が near-identical で実施済。差は certificate と計算コストのみ = llcore の貢献は狭く「**verified-gate + 安さ**」に限定、機構的目新しさは正直に小さい。
- **gate 発想自体**も STABLE(2025)が LLM の編集 gate で、COMP が剪定 × Lyapunov で、SSGM が memory 進化ガバナンスで、**各々 1 軸ずつ既に踏んでいる**。

### M3 の戒め(最重要)
- capability で勝つ路線(「進化が gradient/NAS に勝つ」)は llcore 内部結果(BG9 / E-A / Step C-D)が **CPU で構造的に閉じたと自証済**。③(MAP-Elites niching)は合成欺瞞 corridor では load-bearing(Cliff δ=+1.00)だが、reservoir text proxy/記憶/多タスク汎化/4-kernel union の全実 CPU 基質で **load-bearing でない**(原因 = 実基質の難所が低次元で強 RR baseline が直接サンプル)。
- ⇒ **capability 主張は封印**。差別化は guarantee に限る、という制約は研究の弱みを示す本物の縛りであり、これを破ると自分の honest disclosure と矛盾する。残る path は「高次元 GPU full-LLM 損失地形」と「guarantee niche」のみ。

### 最大の未検証リスク
- **「安さ × 形式保証」の両立は未検証の賭け**。Lipschitz-Transformer は固定アーキで成立(arXiv:2507.13338, 2M 強・145M は上界要調整)するが、**構造変更ごと**に収縮/RoA を CROWN/spectral-norm で締めるコストが Kaggle T4/30h に収まる保証は無い。**PoC 一本で feasibility を先に潰すべき最大リスク**(§⑦ Phase1 step5)。
- **contraction が可塑性を殺すリスク**: 「ρ<1 を強制すると LLM の capability が死ぬ / LLM 核に ρ<1 が無意味」かは未決着。内部 M3 NEGATIVE/H2 not supported はむしろこの後者リスクを示唆 → 空白を埋める前に「contraction gate が能力を殺さない」最小 PoC で feasibility を先に測るのが筋。
- **Transformer attention への収縮 certificate の tightness/scalability は未知数**。contraction theory(Tsukamoto/Slotine)は理論基盤を提供するが、attention 機構への適用の楽観は禁物。

### PoC 設計時に当たる価値のある未読
- `D:/docs/verified_safe_learning_corpus/papers/` の CT-BaB/CEGIS 原論文(16 + taxonomy 00)
- `self_evolving_agents` corpus の Darwin Gödel Machine 詳細(doc_0002)
- `evolution_corpus_v2` cluster_04(NEAT/morphism)の TensorNEAT(2024 GPU NEAT)
