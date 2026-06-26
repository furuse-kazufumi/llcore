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

## 9. 追記予定（background 調査・2026-06-26 起動中）
- 線形化/uptraining 先行研究（LoLCATs / SUPRA / Mamba-in-Llama / Liger / Hedgehog）
- Test-time-training & 長文脈記憶（Titans / TTT / MesaNet / RecurrentGemma）
- 量子化フロンティア（BitNet b1.58 / AWQ/GPTQ / KV-cache 量子化 / sub-4bit）

> 検証レベル ◎=一次確認 / ○=要追検証。未確認で残った点（honest）: xLSTM 蒸留 α* の指標向き・コード公開、TransMamba 規模/ライセンス、Prover-V2 重みライセンス条文。採用前に一次直読で一件ずつ裏取り。
