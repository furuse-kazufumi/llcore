# 効率的 LLM / 新アーキテクチャ ランドスケープ 2026-06（llcore 視点・記事ネタ兼リファレンス）

> 作成: 2026-06-26（ccr セッション、ユーザー指示「他に面白いモデルが出てないか調べて」「全部深掘り」「RAD コーパス化」「記事ネタとして残す」）
> 手法: 並列リサーチエージェント 5+3 本 → 各 arXiv abstract/HTML・GitHub・HF model card で**一次情報裏取り** → main が統合。
> 規律: 検証レベルを ◎(一次確認)/○(検索スニペット・要追検証) で明記。ベンチ数値は**条件併記**（test-time compute スケール有無・自前測定か）= [[feedback_benchmark_honest_disclosure]] 準拠。
> 既知（別途検証済 = memory `reference_low_memory_llm_wave_2026_06`）: Gemma 4 12B / PaddleOCR-VL-1.6 / NVIDIA Cosmos 3 / MangaFlow / Hermes Agent。本書はそれ**以外**。

---

## 0. エグゼクティブサマリ（3 つの戦略的発見）

1. **StateX（arXiv 2509.22630, 清華 THUNLP）= llcore 最大の null の直系先行研究。** 「定数状態 recurrent モデルの有効文脈が頭打ち（llcore では block_size=128）」を、事前学習済みモデルの**状態サイズだけを post-training で拡張**して動かす。線形 attention は **head-merge で追加パラメータゼロ**・定数性維持・速度ほぼ不変（-5%）。公式 repo `thunlp/StateX` あり。**plateau を動かせる証拠を一次データ（NIAH +16pt）で提示**。
2. **xLSTM 蒸留（arXiv 2603.15590, Hochreiter 本家）= llcore「線形化+蒸留」の正面競合。** Qwen2.5-7B を含む teacher から定数状態系へ「ほぼ無損失」蒸留を実証済。**先行性は主張しにくい** → llcore の差別化は「手法の発明」でなく **(a) 真の全層定数状態（先行実用系は例外なく局所 attention=SWA を残すハイブリッドに後退）(b) 厳密 per-layer 計測 (c) 正直な eval」**に寄せる。
3. **ライセンスの地雷（3B スケールの落とし穴）。** Qwen2.5-**3B** は非商用（Qwen Research License）/ Gemma **3** は出力蒸留先が derivative 化。llcore の Apache・on-prem 方針で 3B へ上げるなら **Qwen3-4B / Ministral-3-3B / Gemma 4（2026 で初 Apache 転換）** に切り替える。Qwen2.5-0.5B/1.5B（現用）は Apache で OK。

---

## 1. 線形 attention / SSM / ハイブリッド（メモリ効率の本丸）

### 状態クラス分類軸
| クラス | 推論メモリ | KV cache | 代表 |
|---|---|---|---|
| softmax attention | O(T)（系列長線形増） | あり O(T) | Transformer, Qwen2.5（llcore 出発点） |
| 定数状態 線形系 | **O(1)**（系列長非依存） | なし | Mamba-2, RWKV-7, mLSTM, **llcore 目標** |
| ハイブリッド | 主 O(1)+一部 O(T) | 一部層のみ | Nemotron-H, Jamba, TransMamba, xLSTM蒸留 |

> ★論点: llcore が狙う「全層を定数状態へ内部置換」は、2026 の実用蒸留系が**ほぼ例外なく局所 attention（SWA）を残すハイブリッドに後退**している点と正面で対立する。純 O(1) で品質を保てれば差別化、崩れるなら「なぜ全員が局所 attention を残すか」を直視すべき（現実の制約証拠）。

### モデル別（◎/○）
- **Mamba-2** ◎（arXiv 2405.21060）: State Space Duality = SSM の特殊形が causal linear attention と数学的等価。学習は matmul で Transformer 速・推論 O(TN) で定数状態。2.7B/300B tokens。Apache-2.0（実装公開）。
- **Nemotron-H** ◎（NVIDIA 2025）: SSM が attention の大半を置換するハイブリッド。8B = attention 4 + Mamba-2 24 + FFN 24（**attention 比率 ≈8%**）。51B が Llama-70B 比 2.17× スループットで精度 98.4% 維持（自社測定）。**NVIDIA Open Model License（≠Apache）**。後継 Nemotron 3（2025-12, Ultra 550B MoE）。
- **TransMamba** ◎（arXiv 2503.24067, AAAI 2026）: sequence-level hybrid。QKV/CBx パラメータを Transformer と Mamba で共有し、**同一系列内の token 位置で attention↔SSM を動的切替**（TransPoint）。Memory Converter で切替点の情報を保つ。layer 別でなく時間方向ハイブリッド。code `Yixing-Li/TransMamba`。
- **Jamba** ◎（AI21）: 世界初 production-grade SSM-Transformer hybrid。attention:Mamba = **1:7** + MoE。総 52B / active 12B / context 256K。同規模 Mixtral 比 3× スループット。**Apache-2.0**。
- **RWKV-7「Goose」** ◎（arXiv 2503.14456）: generalized delta rule（vector 値 decay）。**純定数 O(1)**・state tracking 可能・全正規言語を認識・TC⁰ を超える表現力を定数層で（学術的に重要）。2.9B が少学習トークンで multilingual 3B SoTA。Apache-2.0。
- **★ Effective Distillation to Hybrid xLSTM（arXiv 2603.15590）** ◎: Hochreiter 群（xLSTM 本家）。**lossless distillation** を tolerance-corrected Win-and-Tie で定義。3段パイプライン: (I) 層別 hidden-state MSE alignment（head-wise feature map + gate のみ学習, 0.65B tokens）(II) sparse KD（top-256 token の sparse KL, 5-20B tokens）(III) expert merging（math/code/STEM 個別蒸留 → 線形重み平均=capability patching）。**全 MHA を mLSTM-SWA（global mLSTM + sliding-window 512 + sink 4）に置換**。feature map 入力に `[q k v]` 結合が効く。teacher = Llama3.1-8B / Qwen2.5-7B-IT / Olmo3-7B。回復率 α*（小さいほど無損失）= Llama3.1-8B 0.0 / Qwen2.5-7B-IT 0.05 等（**α* の指標向きは PDF 本文で要再確認**）。コード公開は未明記。

---

## 2. StateX — llcore plateau null の直球（最重要・◎）

**arXiv 2509.22630（清華 THUNLP）/ 公式 repo `github.com/thunlp/StateX`（実在確認済）**

- **コア**: 事前学習済み RNN の **recurrent 状態サイズだけ**を post-training で拡大。
  - 線形 attention / GLA → **head-merge**（H 個のヘッドを 1 個の巨大ヘッドに連結）= 追加パラメータ**ゼロ**、状態 H 倍。
  - SSM / Mamba2 → key 次元拡張（W_q, W_k に拡張行列を連結）= 4倍でも総パラメータ増 **<1%**。
  - **24 層中 4 層のみ拡張**（速度・メモリ維持の肝）。
- **再学習コスト**: continued-training **10B tokens**（SlimPajama, 8×80GB GPU, context 64K）。事前学習比で桁違いに安いが**本機 CPU では continued-train は非現実的**。
- **結果（全て自前測定, 1.3B のみ）**: NIAH/Passkey で GLA **26.0→42.2（+16pt）**、ICL 16-shot +7.2% rel、一般推論はほぼ不変（=他能力を壊さない）。改善はタスク依存。スケール則（7B+）未検証。
- **推論時 O(1) 維持**: 拡張後も状態は固定サイズ（元より大きい定数）。prefill 速度 -5%。
- **llcore PoC 筋**: Qwen2.5 を linearize+蒸留 → **一部層のみ head-merge で状態 H 倍 → 短い distillation-recovery**。本機は「コード実装 + 推論確認 + synthetic NIAH 評価」担当、本番状態拡張学習は単発 GPU レンタル（[[feedback_gpu_rent_over_buy]]）。**まず小型で synthetic NIAH で plateau が動くか再現**してから拡げる。

---

## 3. 数学 + 検証（差別化軸②「検証済み数学アシスタント」・◎）

- **DeepSeekMath-V2**（arXiv 2511.22570, Apache-2.0, 重み公開）: generator + **LLM ベース verifier** + meta-verifier の 3 層 RL ループ。ベース = DeepSeek-V3.2-Exp-Base（**Sparse MoE 671B / active 37B**, HF 表記 ~685B）。「self-verifiable」= **LLM が自然言語証明を LLM で点検**＝**形式検証（Lean/Z3）ではない**。確率的で hallucinated issue が原理的に残る（→ meta-verifier が要る理由）。IMO2025 5/6・Putnam 2024 118/120 だが **64×64×16 反復の大規模 test-time compute** 前提（計算を盛った数字）。
- **DeepSeek-Prover-V2**（arXiv 2504.21801）: **Lean 4 統合の形式証明特化**（=真の形式検証側）。671B/7B 2 サイズ。再帰的 subgoal 分解 → tactic 生成 → **Lean カーネルが通れば証明確定（偽陽性ゼロ）**。miniF2F-test **88.9% だが Pass@8192**（1 問 8192 回推論）= 低予算では大幅低下。PutnamBench 658 中 49（低い）。重みは DeepSeek Model License（要原文確認）。
- **3 種の「検証」の区別**:
  | 種別 | 主体 | 保証 | 偽陽性 |
  |---|---|---|---|
  | LLM 自己検証（Math-V2） | LLM verifier | 確率的・訓練品質次第 | 残る |
  | 形式検証 Lean（Prover-V2） | Lean カーネル | 通れば数学的に確実 | ほぼゼロ |
  | SMT（Z3 等） | 決定手続き | 理論断片内で完全・健全 | ゼロ（表現力は限定: 算術・配列・ビットベクタ。高階・帰納は苦手） |
- **llcore 含意（honest）**: (a) DeepSeek 両者は巨大 MoE + 大量 test-time compute の土俵で **llcore（on-prem 小型）と正面競合しない**。重なるのは「検証つき数学」の旗印だけ。**隙間 = 決定的検証 × 小型 × 説明可能性**（「この step は Z3 で確認済 / これは未証明」と正直にラベル）。(b) 「**Z3(SMT) 形式検証 × 長文脈 RAG（定義・定理）× on-prem 小型 × 定数状態メモリ**」の 4 点同時は公開研究にほぼ前例なく実用ニッチとして妥当。ただし**要素は既知**＝新規性は組合せと honest な検証ステータス UX であり、コア技術の発明ではない（誇張しない＝仁ゲート）。(c) honest な天井: 生徒は教師を超えない / Z3 で閉じるのは決定可能断片だけ（大半の競技・研究数学は Z3 外＝多くは「未証明」と正直に出す仕様が正しい）/ 新数学発見は射程外（[[project_galapago_math_llcore]] で反証済）。価値は「発見」でなく「**検証可能な部分を切り分けて誠実に提示する補助**」。

---

## 4. 小型 open instruct（蒸留教師 / 3B int8 ターゲット・◎）

メモリ概算: 1B → fp32≈4GB/int8≈1GB、3B → fp32≈12GB/int8≈3GB、3.8B → int8≈3.8GB。

| モデル | param | ライセンス | 日本語 | 数学 | 蒸留教師可 | 備考 |
|---|---|---|---|---|---|---|
| **SmolLM3-3B**（HF） | 3B | **Apache-2.0** | **✗非対応** | GSM8K 67.6/MATH 46.1（base） | ✓ | 完全 open（データ+code）、/think トグル。**日本語弱が pivot に致命的** |
| **Ministral-3-3B**（Mistral 2512） | 3.4B+vision | **Apache-2.0** | **✓** | MATH Maj@1 0.83（instruct） | ✓ | FP8 8GB・256K・vision 付き。日本語◎ |
| **Phi-4-mini-instruct**（MS） | 3.84B | **MIT** | ✗弱 | MATH 62%・MMLU 73 | ✓（最自由） | 数学・推論密度が param 比で高い。多言語弱 |
| **Gemma 3**（1B/4B） | 1/4B | Gemma Terms（独自） | ✓ | — | **✗（derivative 伝播）** | 出力蒸留先が Gemma Terms 従属化。教師に使わない |
| **Gemma 4**（2026-04） | E2B/E4B/26B MoE/31B | **初 Apache-2.0** | ✓ | — | **✓（解禁）** | 2026 の重要転換。Gemini 3 系・vision/audio |
| **Qwen3**（0.6/1.7/4B） | 〜4B | **Apache-2.0** | **✓強** | ✓強 | ✓ | 思考トグル・119 言語・CJK 強。3B 帯の本命 |
| ⚠ **Qwen2.5-3B** | 3B | **Qwen Research（非商用）** | ✓ | ✓ | ✗ | **3B へ素朴スケールは商用境界を踏む** |

- **llcore 推奨**: **生徒 = Qwen3-4B or Ministral-3-3B**（int8≈3-4GB, Apache, 日本語可）。**教師 = Phi-4-mini（数学・MIT）+ Qwen3-4B/Gemma 4（日本語・Apache）** の二段。Qwen2.5-3B と Gemma 3 だけ商用/derivative で回避。現用 Qwen2.5-0.5B/1.5B は Apache で継続 OK。

---

## 5. 拡散言語モデル（Diffusion LLM）+ 効率アーキ・サーベイ（◎）

### 拡散 LLM
- **DyLLM**（arXiv 2603.08026）: ステップ間の **temporal sparsity**。隣接 denoising ステップの attention context cosine 類似度で salient token を同定 → salient のみ再計算、残りはキャッシュ再利用（training-free）。LLaDA/Dream で最大 **9.6× スループット**。
- **SparseD**（arXiv 2509.24014）: ステップ内の **spatial sparsity**。head 別 sparse パターンを一度だけ計算 → 全ステップ再利用、初期ステップは full attention で品質維持。64k/1024 steps で FlashAttention 比 **1.50×**。
- **DyLLM（時間方向）と SparseD（空間方向）は直交**＝併用可能。
- **定数メモリ観点**: masked DLM は **bidirectional full attention O(L²)×steps**＝**定数メモリではない**（SparseD/DyLLM が必要になった事実が傍証）。**「定数メモリ・長文脈」軸では AR+SSM >> DLM**。DLM の利点は生成 latency の並列化であってメモリ定数性ではない。→ **llcore の自回帰路線を脅かさない（最適化軸が違う別物）**。

### サーベイ「Speed Always Wins」（arXiv 2508.09834）6 分類と代表モデル
1. 線形系列: Linear Attn（GLA, **DeltaNet, Gated DeltaNet**, Based, Rebased, MoM）/ Linear RNN（HGRN2, **RWKV4/6/7, xLSTM**）/ SSM（**Mamba, Mamba2**, S4/S5, H3, Comba）/ **Test-Time-Training RNN（TTT, Titans, Atlas, MesaNet, Miras）**
2. 疎系列: Longformer, BigBird, LongNet / Reformer, **NSA** / training-free（**MInference, StreamingLLM, H2O**, Quest）
3. 効率的 full attention: **FlashAttention-1/2/3** / MQA, **GQA, MLA** / **MoBA** / **SageAttention**
4. 疎 MoE: **DeepSeekMoE, Qwen3, OLMoE** / **LLaMA-MoE-v2**, Sparse Upcycling
5. ハイブリッド: 層間（**Zamba2, Samba, Jamba, Minimax-01, Mamba-in-Llama, RecurrentGemma**）/ 層内（**Hymba, TransMamba, Liger, LoLCATs**, LoLA）
6. 拡散 LLM: **LLaDA**, SEDD, Plaid / BD3-LMs / LLaDA-V, MMaDA

> **★ llcore のサーベイ上の正確な座標** = **(5) Intra-layer Hybrid の "linearization/uptraining" 系統（LoLCATs / Liger / Mamba-in-Llama と同じ箱）**。「既存 Transformer（Qwen2.5）を線形/定数状態 attention へ蒸留変換」。隣接で取り込むべき: 変換系（LoLCATs/Liger/Mamba-in-Llama）、置換先カーネル（**Gated DeltaNet / Mamba2 / RWKV7**）、長文脈記憶（**Titans / MesaNet**）。memetic NAS の探索空間設計に直結。

---

## 6. llcore 戦略的含意（統合）

1. **追い風継続**: 業界は「小型・低メモリ・ローカル・効率アーキ」へ収斂。llcore の北極星は主流の正面（[[project_llcore_memory_efficiency_pivot]]）。
2. **正面競合が具体化**: StateX（状態拡張）・xLSTM 蒸留（線形化+蒸留）が llcore 手法と重複。**引用必須 + 差別化（真の全層定数状態 / per-layer 計測 / 正直な eval / on-prem Apache）**。
3. **モデルで勝たずインフラ層で勝つ**: 大手が高性能小モデルを Apache 無料配布 → llcore はメモリ効率の**手法・計測・gate**（int8/mmap/定数状態/capability-gate/cliff 実測/verified-plasticity）= 再利用可能インフラ層に価値を置く（[[reference_low_memory_llm_wave_2026_06]] の結論）。
4. **数学差別化は生きている**: DeepSeek は LLM 自己検証 / Lean。llcore の **Z3 形式検証 × 定数状態長文脈 × on-prem 小型** は谷間の未踏ニッチ。誇張せず「検証可能部分の誠実な切り分け」に絞る。

## 7. 推奨次アクション（優先順）
- [ ] **StateX 適用 PoC**: 小型 GLA/RWKV or linearize 済 Qwen で head-merge 状態拡張 → synthetic NIAH/passkey で plateau が動くか CPU 再現 → 効けば単発 GPU で本番。**最有力 lead**。
- [ ] **xLSTM 蒸留レシピの移植**: Stage I hidden-state MSE alignment / top-k sparse KL / `[q k v]` 結合 feature-map（確立レシピ）を llcore distill に取り込む。
- [ ] **3B スケールはライセンス切替**: Qwen3-4B / Ministral-3-3B / Gemma 4 へ（Qwen2.5-3B/Gemma 3 回避）。
- [ ] 本書を RAD（`open_model_architectures` / `llm`）へ取込・[[reference_low_memory_llm_wave_2026_06]] 続編メモリ作成。
- [ ] **記事化**: 「2026 効率アーキ地図」「定数状態 vs ハイブリッド論争」「検証つき数学の 3 種」「ライセンス地雷原」は技術版+一般版の好ネタ（[[feedback_daily_articles_policy]]）。

## 8. 出典（主要・一次）
- StateX: arXiv 2509.22630 / github.com/thunlp/StateX
- xLSTM 蒸留: arXiv 2603.15590
- Mamba-2: arXiv 2405.21060 / TransMamba: 2503.24067 / RWKV-7: 2503.14456 / Jamba: ai21.com / Nemotron-H: NVIDIA
- DeepSeekMath-V2: arXiv 2511.22570（Apache, github.com/deepseek-ai/DeepSeek-Math-V2）/ DeepSeek-Prover-V2: arXiv 2504.21801
- 小型: HF SmolLM3-3B / Ministral-3-3B-2512 / Phi-4-mini-instruct / Gemma 4（Apache, VentureBeat）/ Qwen3
- 拡散: DyLLM 2603.08026 / SparseD 2509.24014 / サーベイ「Speed Always Wins」2508.09834

## 9. 量子化・低ビットフロンティア（◎ 追加調査 2026-06-26）

llcore 現状: streaming-int8 で Qwen2.5-1.5B を 5.7GB→2.44GB（embed/lm_head fp32 維持）、CPU 0.7tok/s・dequant 律速。

- **BitNet b1.58 / bitnet.cpp**（MS, arXiv 2402.17764 / 2B4T 2504.12285, MIT）: 重み ternary {-1,0,+1} ≈1.58bit。**学習時量子化（QAT, from scratch）＝事後量子化ではない**。3B で FP16 比 GPU メモリ 3.55×減・2.71×速、品質同等。CPU 推論可（ternary 専用カーネル）。**★既存 Qwen を事後 1.58bit 化は不可（QAT 専用）＝llcore 現行パイプに後付け不能**（最重要 honest）。
- **AWQ / GPTQ**（int4 PTQ）: AWQ は salient ~1% weight 保護（MLSys2024 Best Paper, MIT）。int4 で重み 4×減だが **GPU カーネル前提・CPU は不得手**、実 VRAM 削減は KV/活性込みで 45-55%。
- **KV-cache 量子化（長文脈の主削減レバー・重み量子化と直交）**: **KIVI**（2402.02750, **無調整 2-bit**, ピークメモリ 2.6×減・スループット 2.35-3.47×, 既存モデルにプラグイン）/ **KVQuant**（2401.18079, 3-bit で <0.1 ppl 劣化, 1M-10M tokens 文脈）。
- **sub-4bit**: **QuIP#**（2402.04396, E8 格子, 2-bit で初の実用品質・3-bit が理論ロスレス 4-bit 超）/ AQLM（コードブック L1 非搭載で推論遅い）。**いずれも GPU 専用カーネル前提**。
- **llama.cpp / GGUF**: Q4_K_M（~4.5bit 混合, コミュニティ・デファクト, +0.0535 ppl@7B）/ Q8_0（<0.5% 劣化）/ imatrix（同 bit で 2-4% 改善）。**K-quant はネイティブ int カーネルで CPU 実速**（dequant→fp32→matmul ではない）。

**llcore 含意（量子化）**:
- (a) **int4（GGUF Q4_K_M / AWQ）で int8 比ほぼ半減**（2.44GB→~1.3-1.5GB 目安）。1.5B 級なら劣化小・費用対効果高。BitNet は QAT 専用で後付け不能（採用するなら BitNet 系モデル丸ごと or 自前 QAT を別プロジェクト化）。
- (b) **KIVI 型 KV 2-bit が long-context メモリ削減の最低リスク手段**（重みを触らず直交追加）。llcore の定数状態路線と併用で相乗。
- (c) **★CPU 0.7tok/s の律速＝「低ビット格納 + fp32 計算」のアンチパターン**。解 = **低ビットのまま積和するネイティブ整数カーネル**（bitnet.cpp / GGUF Q4_K_M）。AWQ/GPTQ/QuIP# は GPU 前提で CPU 速度に直結しない。**推奨検証順: ①GGUF Q4_K_M+imatrix を CPU ベースライン化し 0.7tok/s と直接比較 → ②KIVI 型 KV 2-bit 追加 → ③本気の 1.58bit は BitNet 採用/QAT を別建て**。

主要出典: BitNet 2402.17764 / 2504.12285、AWQ 2306.00978、KVQuant 2401.18079、KIVI 2402.02750、QuIP# 2402.04396、llama.cpp quantize README。

## 10. Test-Time Training (TTT) & 長文脈記憶（◎ 追加調査・plateau null の本命）

llcore の最大 null＝「定数状態 recurrent が block_size=128 で有効文脈頭打ち」。本質は**状態容量の限界でなく credit assignment（128 窓を超えて「何を記憶すべきか」の学習信号が届かない）の限界**。

- **TTT**（arXiv 2407.04620, Stanford, **公式コード公開**, 最も再現性高）: **hidden state そのものを ML モデル化**し、更新則＝その state に対する**自己教師あり学習の 1 ステップ**。TTT-Linear（state=線形）/ TTT-MLP（state=2層MLP, I/O 重）。**「Transformer 同様トークン増で perplexity 下がり続ける／Mamba は 16k で改善停止」を 125M-1.3B で同条件実証**＝llcore plateau に最も直接対応する一次研究。
- **Titans**（2501.00663, Google）: ニューラル長期記憶 MLP を推論時に勾配更新（surprise ベース, momentum + 忘却ゲート）。2M 文脈 NIAH 主張（小規模・検索タスク条件）。公式コード未公開（lucidrains 実装 MIT）。
- **Atlas**（2505.23735）: Titans 後継。直近1トークンでなくスライド窓で記憶最適化。BABILong 10M で Titans 比 +80%（合成・相対値）。
- **MesaNet**（2506.05233, von Oswald 他）: in-context loss を共役勾配で**最適点まで解く**locally optimal TTT。**honest（abstract 明記）: 向上は推論時 FLOPs を払う代償**。
- **Miras**（2504.13173）: TTT 系の設計空間カタログ（記憶アーキ/attentional bias/retention gate/学習則の 4 軸）。Moneta/Yaad/Memora。
- **Lattice**（2504.05646）: KV を固定スロットに低ランク圧縮、**orthogonal update**（現状態に直交成分のみで干渉最小化）。
- **RecurrentGemma**（2404.07839, Griffin = 線形再帰 RG-LRU + 局所注意）: **推論時学習はしない**（重み固定）。2B/9B 実重み公開（コード Apache だが**重みは Gemma Terms**=商用注意）。

**llcore 含意（plateau null）**:
- **StateX（状態を広げる）vs TTT（記憶を学習する）の決定的差**: StateX は容量天井を上げるが更新則は online・線形のままで **BPTT 越えの credit assignment は未解決**（capacity を足しただけ）。TTT は更新則を**内側の自己教師あり損失の勾配ステップ**に置換＝「何を記憶するか」を局所損失が決め、**全ホライズン BPTT を要さない**＝llcore の BPTT=128 ボトルネックそのものを迂回。→ **plateau null には TTT 方向が StateX より本命**。
- **CPU 実行可能性**: **TTT-Linear が最有力**（線形 inner state + 単一 SGD step + chunk 並列で小型 CPU 現実的）。TTT-MLP / MesaNet（共役勾配）/ Atlas（2次最適化）は推論 FLOPs 増で CPU 不利。
- **★推奨実験**: llcore に TTT-Linear 層（線形 inner state + 1 GD step + chunk 更新）を試作 → 「inner-loop test-time 学習が block_size=128 plateau を右へ動かすか」を **StateX（state を広げただけ）baseline と対照**し、利得が**容量由来か更新則由来か**を切り分ける（honest ablation）。

## 11. 線形化 / uptraining 先行研究（◎ llcore が居る箱・車輪の再発明リスク）

2 系譜: **(A) 線形 attention 系（feature-map で softmax 近似）** Hedgehog→LoLCATs→SUPRA→Liger / **(B) SSM 蒸留系（重みを Mamba に移す）** MOHAWK, Mamba-in-Llama。**llcore は (A)・特に Hedgehog/LoLCATs の直系**。

| 手法 | 系統 | base 凍結 | feature map | 蒸留損失 | tokens | attn 温存 | 回復（条件） |
|---|---|---|---|---|---|---|---|
| **Hedgehog**(2402.04347) | A | 任意 | 学習可能 MLP(spiky/monotonic) | attn 重みマッチ | 小 | 全線形 | >99%（短系列PPL/GLUE）|
| **LoLCATs**(2410.10254) | A | **凍結** | アフィン+非線形(16.8M) | **出力 MSE**+LoRA | **40M** | 全 or SWA hybrid | 8B ほぼパリティ; 70B/405B gap 78% closing |
| **SUPRA**(2405.06640) | A | fine-tune | ReLU(Wx+b)+GroupNorm | uptrain(CE) | ~20B | 全線形 | **MMLU 崩壊 28 vs 62; 長文脈 2k で頭打ち** |
| **Mamba-in-Llama**(2408.15237) | B | 一部再利用 | (SSM) | 段階蒸留 | 中 | **25%温存** | AlpacaEval2 29.6; NIAH 20× 外挿 |
| **MOHAWK**(2408.10189) | B | 段階凍結 | (SSM) | Frob→L2→KD | 3-5B | 0 or 4/24 | <1%データで OSS 凌駕（短系列）|
| **Liger**(2503.01496) | A | LoRA のみ | softmax 正規化 | LoRA(CE) | **0.02B** | **SWA intra-layer(64)** | 93%（LM-eval avg, 1-8B）|

**llcore 含意（最重要）**:
1. **車輪の再発明リスク**: llcore のコア（q,k feature-map → softmax 出力に MSE 蒸留 → base 凍結）は **LoLCATs Step1 とほぼ完全一致**。自前再構築せず **LoLCATs レシピ（→ LoRA 回復）を丸ごと採用**すべき。
2. **★真に新規な軸 = memetic NAS による層別「線形化 vs 温存」探索**。先行は全て固定ヒューリスティック（MOHAWK 4/24, Mamba-in-Llama 25%, Liger 等間隔 SWA）。**どの層を softmax/SWA のまま残すかを探索する研究は手薄＝llcore の独自貢献**。NAS の allele に「この層は温存」を明示的に含めよ。
3. **即移植レシピ**: (a) 2段＝出力 MSE attention transfer（base 凍結）→ LoRA 回復（LoLCATs）。(b) **64-token sliding-window hybrid**（LoLCATs-SW と Liger が独立収束＝強い実証）。(c) 回復頭打ちなら **MOHAWK Stage3 = logit KD** 追加。
4. **「4-param アフィン恒等初期化」への警告**: Hedgehog が「**spiky(低エントロピー)+dot-product monotonic** を満たさない単純 map は softmax を近似しきれない」と明示。llcore の極小アフィンは表現力天井の恐れ → 恒等初期化 MSE 蒸留で spiky 性が出るか **ablation 必須**。
5. **論争での立ち位置**: 「全層定数状態 vs 局所 attention 温存」は**ファミリー全体が hybrid に決着済**。pure 全層線形（SUPRA, Phi-Mamba 0-attn）は **5-shot MMLU・長文脈で必ず崩壊**。llcore の pure 定数状態全層は**最難・最危険な端**。だからこそ memetic NAS は「**最小限どの層が softmax/SWA を要求するか**」を見つける正しい道具 → **目的関数に MMLU(5-shot) と長文脈を必ず入れる**。
6. **回復率の honest-disclosure**: 91-101% は**必ずベンチ条件と紐付け**。perplexity/短系列なら literature 整合で妥当。**5-shot MMLU や >8k 長文脈で 91-101% を主張するなら勝った気になる前に内訳を疑う**（[[feedback_benchmark_honest_disclosure]]）。全線形手法はそこで出血する。

---

## 12. ★統合 research leads（優先順・llcore が次に動かす）

| # | lead | 根拠 | CPU 可否 | 期待 |
|---|---|---|---|---|
| **L1** | **TTT-Linear 層を試作 → plateau が右に動くか × StateX baseline と対照 ablation** | TTT が「Mamba 16k 停止 / TTT 下がり続ける」を一次実証。BPTT=128 を迂回 | **可**（TTT-Linear） | plateau null を動かす本命。容量 vs 更新則を切り分け |
| **L2** | **LoLCATs レシピ採用**（出力 MSE transfer → LoRA）+ **memetic NAS の allele に「層温存」追加**、目的関数に 5-shot MMLU + 長文脈 | llcore コア = LoLCATs Step1。NAS 層選択が唯一の新規軸 | 可（蒸留は小 token） | 車輪の再発明回避 + 独自貢献の確立 |
| **L3** | **CPU 速度（0.7tok/s）改善 = GGUF Q4_K_M + imatrix を CPU ベースライン化** | 律速は「低ビット格納+fp32 計算」のアンチパターン。ネイティブ int カーネルが解 | 可 | 体感速度・実用性 |
| **L4** | **KIVI 型 KV 2-bit を long-context に追加**（重み非依存・直交） | 無調整 2-bit でピークメモリ 2.6×減 | 可 | 定数状態路線と相乗のメモリ削減 |
| **L5** | **3B スケールはライセンス切替**（Qwen3-4B / Ministral-3-3B / Gemma 4） | Qwen2.5-3B 非商用 / Gemma 3 derivative | 可（int8≈3-4GB） | 会話品質 + Apache 維持 |
| **L6** | **数学アシスタント②**: Z3 形式検証 × 長文脈 RAG × on-prem を「検証可能部分の誠実な切り分け」に限定設計 | DeepSeek は LLM 自己検証/Lean=谷間が空く | 可 | 差別化ニッチ（誇張せず） |
| **L7** | **Hedgehog ablation**: 4-param アフィン恒等初期化で spiky 性が出るか検証 | 表現力天井の警告 | 可 | コア feature-map の妥当性確認 |

> 全 lead に共通の honest 規律: 回復率・ベンチは**条件併記**、pure 全層線形の MMLU/長文脈崩壊を**目的関数に明示**、異常に良い結果は内訳を疑う（[[feedback_benchmark_honest_disclosure]] / [[feedback_llive_measurement_purity]]）。

## 13. 本セッション実走結果（2026-06-26, CPU・no-push・既存非破壊）

### (1) 1.5B linearization-tolerance profile（`out/linearize_tolerance_1.5b/`, 28層, aozora 1024tok）
base ppl **32.07**。最耐性 **L10（Δnll +0.0033）** / 最非耐性 **L0（Δnll +13.76 壊滅）**。cumulative top-k Δnll: 1→+0.003, 4→+0.066, 6→+0.267, 8→+0.349, 12→+1.24, 28→+12.5。
- **0.5B（24層）比較**: 1.5B の top-4 (+0.066) < 0.5B top-4 (+0.099)、top-1 (+0.003) < 0.5B (+0.014)。**規模が大きいほど少数層線形化への耐性が高い**＝Liger「規模↑で gap↓」と整合。L0 は両モデル共通の壊滅的非耐性層（residual stream 入口の役割）。

### (2) ★L7 Hedgehog ablation（`scripts/feature_map_spikiness.py`, `out/feature_map_spikiness/`, 0.5B 256tok, 7 unit tests green）
llcore の現行 feature map（`_phi=elu+1`、`learnable` 時の per-head アフィンは恒等初期化＝固定 φ と同一）が softmax の spiky+monotonic を再現するか実測:
- **エントロピー gap（線形−softmax）= +2.70 nats（全層で大きく正）**。線形は各行ほぼ一様（H≈4.64）、softmax は spiky（H 0.86–3.39）。
- **top-1 mass: softmax 0.18–0.82（集中）vs 線形 ~0.017（ほぼ一様）**＝elu+1 はキーに質量を集中できない（Hedgehog 警告を一次実証）。
- **rank 相関 ρ ≈ +0.39（0.16–0.60）**＝線形は softmax のキー順序を弱くしか保存しない（非 monotonic 寄り）。
- **★honest null: corr(entropy_gap, 耐性Δnll) = -0.016 ≈ 0**＝spikiness gap は層別線形化耐性を**予測しない**。「非耐性層＝spiky な層」という誘惑的仮説は**反証**（耐性は spikiness 以外＝L0 の残差流役割等が支配）。スクリプト note の事前予測（正相関）も外れ＝honest に記録。
- **含意（L7 → 設計判断）**: llcore の 4-param アフィン恒等初期化は**ほぼ一様な出発点**で、蒸留が回復すべき距離が大きい（elu+1 は構造的に diffuse）。→ **spiky な feature map（Hedgehog 流 learnable MLP / exp 系 / softmax 正規化 feature map）の追加**を強く動機づける（L2/L7 統合）。ただし耐性の主因は spikiness でないため、**「どの層を温存するか」（memetic NAS, L2）と「feature map をどう spiky にするか」（L7）は別軸として両方追う**。
- ★補正（Codex review）: `rank_corr` を tie-aware Spearman に修正して再走 → ρ・gap・corr とも**数値不変**（tie の歪みは実際には効かず結論頑健）。「spikiness gap が耐性を予測しない」は**この 1 条件（0.5B/256tok/1断片/zero-shot/単一 Pearson）での観測**であり「反証」と一般化はしない（過剰主張の緩和）。

### (3) ★plateau ablation — carry-off（state-reset 訓練, `out/ttt_plateau/`, 0.5B/2層/block128, 1200 iters）
recurrent LM の有効文脈 plateau を、context_length_curve の `past_block_gain`（block_size 超で NLL が下がり続けるか）で測定:

| arch（carry-off） | held-out ppl | past_block_gain | ppl_by_ctx(16→1024) |
|---|---|---|---|
| recurrent | 32.47 | **−0.0001** | 22.4→22.3（フラット） |
| recurrent-wide（容量拡張=StateX 流） | 31.76 | **0.0** | 22.3→22.2（フラット） |
| ttt-linear（**静的ゲート版**） | 31.52 | **0.0** | 22.4→22.2（フラット） |

- **発見**: **state-reset 訓練下では 3 arm すべてフラット**＝容量拡張も delta-rule セルも plateau を動かさない。plateau は「訓練中に >block_size 依存を経験しない」**訓練法アーティファクト**である可能性を強く支持（N/Codex 予測どおり）。
- **honest 注記（重要）**: この実験と carry-on 実験は、**忠実 Gated DeltaNet パッチ（データ依存ゲート, §14）を当てる前に起動**＝**静的ゲート版**をテスト。M が「データ依存ゲートが plateau に本質的」と指摘した**忠実版は未テスト**。carry-on（state-carry 訓練, `out/tbptt_plateau/`）は recurrent-wide まで gain≈0、static gated-deltanet arm 走行中。**真の検証＝忠実版 × carry-on の再走**が必要。
- honest 規律: 単一 seed・CPU・小型・`past_block_gain` は粗い2点差分（[[feedback_benchmark_honest_disclosure]]）。

### (4) ★plateau ablation — carry-on（state-carry TBPTT, seg_len 2048, `out/tbptt_plateau/`）
| arch（carry-on, 静的ゲート版） | held-out ppl | past_block_gain | ppl_by_ctx(16→1024) |
|---|---|---|---|
| recurrent | 32.41 | 0.0003 | 22.9→22.9（フラット） |
| recurrent-wide | 33.33 | 0.0001 | 23.0→23.0（フラット） |
| gated-deltanet（静的） | **25.23** | 0.0 | **17.9→17.9（フラット）** |

- **2×3 確定 = 全 honest null**: carry-off / carry-on × 3 arch すべて past_block_gain ≈ 0。**state-carry TBPTT 訓練でも plateau は動かなかった**＝「訓練法が主因」仮説（N/Codex）はこの設定では**支持されず**。
- **副次の真の信号**: gated-deltanet（delta-rule セル）は ppl が顕著に低い（17.9 vs recurrent 22.9）＝**より良い mixer**。ただし長文脈活用（plateau）とは**直交**＝ppl は良いが依然 c>16 を使わない。
- **★重大な交絡（設計上の見落とし）**: 実験は `chunk_size=block_size=128`。state は持ち越すが**勾配は依然 128 で truncate**＝「128 先のために何を保持すべきか」の credit assignment はやはり切れている（N が §2c で警告した点）。**∴ 今回は「state-carry が plateau を動かすか」を完全には検証できていない**。真の検証 = (a) `chunk_size>128`（勾配をより遠くまで）+ (b) 忠実なデータ依存ゲート版（静的ゲートは窓統計に最適化される）。両方とも未実施＝**次の本当の実験**。
- honest 結論: 容量・セル・state-carry のどれ単体でも、この（交絡含みの）設定では plateau は動かせなかった。誇張せず「まだ動かせていない／設計に交絡があった」と記録する。

> 検証レベル ◎=一次確認 / ○=要追検証。

## 14. 実装済み機能（2026-06-26 goal セッション「発見機能を自律実装」・全 no-push・既存非破壊）

深掘りで発見した機能を実コードに組み込み、各々テストで検証（実装状況: **実装+単体検証済 / 実走検証は別**）。

- **忠実 Gated DeltaNet セル**（`src/llcore/lm/ttt.py` `TTTLinearCore`、M の正典照合パッチ）: 静的 per-channel ゲート → **データ依存スカラ α_t=exp(−exp(A_log)·softplus(a_proj(h)+dt_bias))（Mamba2 化）/ β_t=sigmoid(b_proj(h))**、減衰後予測（Eq.8）、q/k L2 正規化、gated RMSNorm。**11 tests**（含 2000-step 状態ノルム有界性=Codex #5 を実証で閉鎖、3000-tok streaming finiteness）。クラス名は `GatedDeltaNetLM` へ機械リネーム予定（docstring で誤称訂正済）。残: multi-head / short-conv 未。
- **`tbptt.py` state-carry 3-D 対応**（N）: `reset_state_slots` を state 次元に一般化（2-D RecurrentLM / 3-D fast-weight 両対応）+ union に追加。
- **carry-on plateau 実験**（`scripts/tbptt_plateau_experiment.py`、N 設計）: carry on/off × 3 arch、seg_len=2048 で >1024 文脈を訓練中に経験、token 予算一致。smoke 検証済→本走中。
- **WindowLinearAttention**（`src/llcore/runtime/linearize.py`、LoLCATs-SW/Liger hybrid）: 直近 window=softmax + 古い key=線形、単一分母で融合、per-head 学習可能 γ。**メモリ=O(window)KV + O(d²)状態**。**10 tests**（主アンカー: window≥T で完全 softmax 一致＝厳密な一般化、causal 性、メモリ有界）。
- **full feature map**（`LinearAttention(feature_map="full")`、Hedgehog/LoLCATs 流）: 4-param 対角アフィン → per-head 全結合行列 W∈[H,d,d]（恒等初期化＝現行 warm start）。L7 で実証した「elu+1 はほぼ一様」への処方（よりspiky な φ を学習可能に）。**4 tests**（恒等初期化一致・学習可能性・形状・不正値拒否）。default は "diag" で非破壊。

**次の実装候補（発見済・未実装）**: chunkwise 並列 Gated DeltaNet 訓練（M §3、CPU 高速化）/ LoRA Step2 + logit KD（K、distill.py）/ NAS allele「層温存{softmax/SWA/linear}」（K、唯一の新規軸）/ multi-head + StateX head-merge。

> 未確認で残った点（honest, doc 全体）: xLSTM 蒸留 α*（→ L で確定: **小さいほど無損失**、コード未公開）、TransMamba（→ L: 400M/1.5B, repo MIT, 重み未公開）、Prover-V2 重み（→ L: **DeepSeek License=商用可だが制限付・Apache 非互換**）、Nemotron-H（→ L: Apache 非互換）、Mamba-2/RWKV-7（→ L: **Apache 確定**）。
