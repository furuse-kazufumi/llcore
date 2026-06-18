# llcore の立ち位置 — llama.cpp/GGUF との切り分け + 競合メモリ地図

> **目的**: llcore のメモリ効率知見を、業界の既存資産(llama.cpp/GGUF・量子化 literature・
> 大規模 OSS モデル)に対して **honest に位置づける** 1 枚。過大主張(over-claim)を構造的に防ぐ。
>
> **要旨を 1 行で**: llcore はモデルでも圧縮アルゴリズムでも競合しない。llama.cpp/GGUF の常識を
> **自宅 CPU で誰でも再現できる形で再導出**し、その上に **fail-closed な品質ゲート** と
> **(将来)メモリ指標を進化の適応度にする統合層** を載せる「計測と規律」のレイヤである。
>
> 正本数値: `docs/MEMORY_EFFICIENCY_FINDINGS.md`。競合数値は各社 self-report(条件付き)。
> prior-art は一次情報(llama.cpp 公式 README / PR #1684 / arXiv)で検証済み。
>
> 実行環境: Windows 11 / Python 3.11 / torch 2.12.0+cpu(GPU なし)。モデルは tiny char-LM 0.81M〜130M。

---

## 用語(かみくだき)

- **PTQ (Post-Training Quantization)**: 学習済みの重みを後から低ビット化する。再学習しない。
- **QAT (Quantization-Aware Training)**: 量子化を見越して学習する。PTQ より高精度だが学習コストがかかる。
- **bpw (bits-per-weight)**: 重み 1 個あたりの平均ビット数。fp32=32, fp16=16, int8≈8。GGUF の
  Q4_K_M は実 mix 込みで約 4.89 bpw。
- **PPL (Perplexity)**: 言語モデルの「次トークン予測のしにくさ」。低いほど良い。量子化品質の標準指標。
- **top-1 retention**: fp32 と量子化後で、最上位確率トークンが一致する割合。capability(賢さ)の代理指標。
- **mmap (memory-map)**: ファイルをメモリ空間に「貼る」だけで、実際に触ったページだけを後から読み込む方式。
- **working set (WS)**: プロセスが実際に物理 RAM に載せているページ量。peak WS = その最大値。
- **fail-closed**: 判定に失敗・未達なら **通さない**(安全側に倒す)。逆は fail-open(疑わしきは通す)。

---

## 1. 再導出セクション — llcore の量子化/mmap/cliff 知見は「既知の再確認」

prior-art を一次情報で検証した結果、llcore のメモリ効率 4 主張は **いずれも llama.cpp/GGUF/
量子化 literature の既知結果を tiny char-LM で再確認したもの**だった。新規アルゴリズムは確認できない。
これは弱みではなく、llcore の価値の **正しい所在を示す**(下記第 2 節)。

| llcore の知見 | llcore 実測(正本) | 既知の一次情報 | honest 判定 |
|---|---|---|---|
| int8 weight-only で約 3.9× 圧縮・PPL 劣化 <0.1% | 重み常駐 74–75% 減、ΔPPL <0.1%(多くは <0.02%)、3 モデル一貫 | GGUF Q8_0=8.50 bpw=fp32 比 約 3.76×、PPL 差 ≈0.01 オーダー(PR #1684 系 7B 表 / arXiv 2601.14277 で accuracy 同等) | **再導出**(既知) |
| mmap で RAM 超のモデルが回る | fp32 522MB モデルを WS 上限 358MB(68%)で forward 完走、logits checksum 完全一致 | llama.cpp の mmap は **既定で有効**(Issue #91)。カーネル page cache 経由で on-demand ページング、RAM 超モデルも起動可 | **再導出**(llama.cpp 標準挙動) |
| 量子化ビット幅は cliff_then_flat、cliff 位置はモデルサイズ依存 | 8/6/5bit 平坦、低ビットで急落。小 1.36M=3bit 劣化開始/大 11.9M=3bit 実用・cliff は 2bit | PR #1684 が「PPL は量子化サイズの滑らかな関数・2bit 付近で崩壊」を実証。Dettmers 2023 が「大モデルほど低ビットに頑健」を 35,000+ 実験で確立 | **再導出**(既知) |
| int8 streaming-dequant で常駐 72% 削減 | dense fp32 539MB → stream int8 149MB、368MB 上限で完走、logits 全一致 | weight streaming + on-demand dequant は llama.cpp/GGUF mmap の既定機構と同根 | **再導出**(既知機構) |

### honest disclosure(基準の取り違えに注意)

- **「int8 約 4×」は fp32 基準なら成立**。ただし GGUF の標準基準は **fp16** で、Q8_0 だと約 1.9× に過ぎない。
  記事や対外発信で「4×」と書くときは **fp32 比**であることを明記する(基準を黙ると誇張になる)。
- **モデルサイズ依存の向き**: PR #1684 の一次情報は「相対量子化誤差(PPL 基準)はベースモデルの重み数が
  増えても**減少しない / 非単調**(13B は 7B より量子化しやすいが 30B/65B で 7B 水準に戻る)」。llcore で
  正しく言えるのは「**大モデルほど低ビットに頑健**(冗長性が多い)」までで、「常に単調改善」とは書かない。

### 再導出の価値(これは捨てる必要のない強み)

再導出に価値が無いわけではない。llcore の再導出には次の固有価値がある:

1. **教育的**: char-LM スケールで「なぜそうなるか」を 1 ファイルずつ実機で見せられる。
2. **自己検証**: 自前ハーネスが literature と一致する = 計測基盤が正しいことの裏取り(ハーネスの健全性証明)。
3. **誰でも再現**: GPU 不要・自宅 CPU・少 RAM で走る。FullSense「自宅 PC で動く」哲学の定量的裏付け。

→ ただし価値は **モデル本体でも新圧縮アルゴリズムでもない**。「教育/自己検証/再現性」と明記する。

---

## 2. 独自/差別化セクション — 残る差別化は「規律・統合・計測」(アルゴリズムではない)

prior-art 検証で「新規でない」と出たものは独自と書かない。たとえば **「accuracy も見る」という発想自体は
量子化評価の業界標準**(PIQA/ARC/HellaSwag 併記は確立プロトコル)であり、top-1 retention も llama.cpp の
perplexity ツールに **`Same top p` として既に出荷済み**のメトリックである。よって「独自メトリック」とは書けない。

残る差別化を **控えめに** 3 点。各々「新規アルゴリズムではなく、性質はこうだ」を明記する。

### (a) fail-closed capability-gate を eval に operationalize した engineering 規律(新規アルゴリズムではなく運用の独自性)

- **何か**: 量子化の合否を **PPL だけで決めず、top-1 retention ≥ 97%(fp32 比)を fail-closed の機械ゲート**
  として eval pipeline に配線(`src/llcore/lm/eval.py` の `passes_capability_gate`)。
- **なぜ必要か(実証)**: realp1 2bit は **PPL-gate は PASS なのに top-1 が −13.46pp(28.7%→15.2%)= 半減近く
  壊れている**。PPL だけの合否は低ビットの capability 喪失を見逃す。これを実機で捕捉した。
- **性質(honest)**: メトリック(top-1 retention・KLD・top-1 一致率)は **mainstream tooling に既出**。
  新規なのは「**≥97% を fail-closed ゲートとして eval に機械配線する**」という閾値化+ゲート化の
  **operationalization のみ**。これは研究上の新規性ではなく **エンジニアリング設計判断**。
  記事では「独自メトリック」と書かず「**既知メトリックの fail-closed ゲート化(運用の独自性)**」と書く。

### (b) working-set hard-cap での RAM 超実証など honest な計測手続き

- **何か**: Windows の `SetProcessWorkingSetSizeEx`(working-set hard max)を **モデルサイズ未満**に設定した
  サブプロセスで forward を完走させ、peak WS を実測。capped/uncapped の **logits checksum 完全一致**で正当性を担保。
- **実測**: fp32 522MB を **WS 上限 358MB(68%)で完走**(capped peak 357.7MB ≤ 上限)。int8 streaming は
  **368MB 上限で完走**(dense 539MB 常駐は収まらない)。
- **性質(honest)**: 機構そのもの(clean な read-only mmap ページを圧力下で破棄→再 fault)は **llama.cpp 流で既知**。
  独自なのは **計測の規律** — 成功を偽装せず `cap_set_ok` と実測 peak を JSON にそのまま残す、別プロセス隔離で
  allocator 汚染を避ける、ΔRSS(baseline 差分)をクリーン信号とする、等の **honest disclosure の手続き化**。

### (c) メモリ指標 × 進化(verified-plasticity gate)統合という prospective な研究方向

- **何か**: 進化/NAS の適応度関数に **メモリ効率指標**(常駐バイト・peak WS・cap-gate 通過)を組み込み、
  探索を **verified-plasticity gate**(検証済み可塑性ゲート)で選別する統合構想。
- **状態**: verified-plasticity gate は別アーク(branch A 候補)で **全現存・91 tests green**。ただし
  **メモリ指標を適応度に使う統合は未着手(prospective)**。
- **性質(honest)**: 構成要素は **全て prior art** — plasticity の定量化+utility-gating は Dohare et al. 2024
  (Nature)、候補の learnability/trainability を指標化して探索選別するのは InTrain(2026)/EZNAS(2022)等
  zero-cost NAS proxy 系統。**「進化 × メモリ効率指標 × 検証済み可塑性ゲート」の三者結合の特定文脈**は単一の
  先行研究に未見だが、**未実証ゆえ新規性は主張保留**。「原理は既知、結合と memory-efficiency 文脈への適用が未踏。
  実証前」が誠実な言い回し。

> **3 点の総括**: (a) は engineering 規律、(b) は計測手続き、(c) は未実証の prospective な結合。
> **いずれも新規圧縮アルゴリズムではない**。llcore が誠実に主張できるのは「**規律・統合・計測**」の層である。

---

## 3. 競合メモリ地図

各社の数値は **self-report** で、測定条件に注記が付く(脚注参照)。llcore は **モデルではなく計測層**なので、
同じ表に並べるのは「規模感の地図」のためであり、性能比較ではない(立ち位置が違う)。

> **★規模差の明示(表を見る前に必読)**: llcore は **tiny char-LM(0.81M〜130M)**、Gemma 4 は **11.95B** =
> 約 **15,000× の規模差**。下表で llcore だけ実測値が並ぶのは「計測層だから測れる」ためであって、
> **12B/8B 実モデルとメモリ・品質を直接比較できる土俵にはない**(llcore は char-LM・simulated quant・速度未測)。
> 同じ列に並んでいても、これは優劣ではなく「**どの層の話か**」が違うことを示す地図である。

| 手法 / モデル | param 規模 | 必要メモリ / 量子化前提 | ライセンス | accuracy ベース合否ゲートの有無(★GGUF は非標準 / llcore は char-LM スケール) |
|---|---|---|---|---|
| **Gemma 4 12B** [^1] | 11.95B dense | 16GB 級(RAM/VRAM 曖昧・Q4 量子化前提)[^1] | Apache 2.0 | なし(self-report ベンチのみ) |
| **PaddleOCR-VL-1.6** [^2] | 0.9B(ERNIE-4.5-0.3B+NaViT) | 未測(self-report)[^2] | Apache 2.0 | なし(文書解析専用ベンチ OmniDocBench) |
| **Cosmos 3 Nano** [^3] | 16B | 未測(self-report)[^3] | OpenMDW 1.1(OSI OSS でない open model)[^3] | なし(cherry-pick ベンチ) |
| **Cosmos 3 Edge** [^3] | 2B | 未測(self-report)[^3] | OpenMDW 1.1 [^3] | なし |
| **llama.cpp / GGUF Q4_K_M** [^4] | 任意(例 Llama-3.1-8B) | 4.58GiB(fp16 14.96GiB 比 約 3.27×)、4.89 bpw | MIT(llama.cpp)/ モデル別 | **報告のみ**(PPL + opt-in KLD・top-1 一致率。downstream accuracy 合否ゲートは**非標準**)[^4] |
| **llama.cpp / GGUF Q8_0** [^4] | 同上 | 7.95GiB(fp16 比 約 1.88× / fp32 比 約 3.76×)、8.50 bpw | 同上 | 同上 |
| **llcore** [^5] | tiny char-LM 0.81M〜130M | int8 重み常駐 **約 3.9× 圧縮**(74–75% 減)/ mmap **358MB 上限で 522MB モデル完走** | Apache-2.0 + Commercial(FullSense) | **あり** — fail-closed cap-gate(**top-1 retention ≥97%**、未達は機械的に拒否)[^5] |

[^1]: Gemma 4 12B(Google, 2026-06-03, Apache 2.0)。dense 11.95B・エンコーダフリー統合・256K ctx。
「26B に迫る」は **定量ベンチ表が一次情報に無い一般文**で、比較相手は 26B MoE(active~4B)= apples-to-oranges。
「16GB」は **RAM/VRAM 曖昧で Q4 量子化前提**(fp16 12B≈24GB は乗らない)。数値は self-report。

[^2]: PaddleOCR-VL-1.6(Baidu, arXiv:2606.03264, Apache-2.0)。0.9B。OmniDocBench v1.6=96.33%。
「235B/Gemini 超え」は **文書解析専用ベンチ上 + Baidu 自前測定**であって汎用超えではない。メモリ要件は未測(self-report)。

[^3]: NVIDIA Cosmos 3(2026-05-31)。Super 64B / Nano 16B / Edge 2B、5 モダリティ単一アーキ。
ライセンスは **OpenMDW 1.1 = OSI 承認の OSS ではない open model ライセンス**。「Gemini 超え」は
**負け軸(Robotics/General)を省いた cherry-pick**。メモリ要件は未測(self-report)。

[^4]: llama.cpp / GGUF。bpw・ファイルサイズは公式 quantize README(Llama-3.1-8B)と k-quants PR #1684 の一次情報。
**最重要確認**: llama.cpp の量子化パイプラインも公式 perplexity ツールも、downstream accuracy(HellaSwag/MMLU/
GSM8K)や top-k retention に基づく **pass/fail ゲートを持たない**。標準で報告されるのは PPL と(opt-in で)
KLD・top-1 一致率(`Same top p`)まで。downstream accuracy 評価は arXiv 2601.14277 等の **外部研究 / lm-eval-harness
側で別途**行うもので、量子化ツールチェーンの合否判定には組み込まれていない。

[^5]: llcore。正本 `docs/MEMORY_EFFICIENCY_FINDINGS.md`。int8 weight-only PTQ で重み常駐 74–75% 減(約 3.9×)・
held-out PPL 劣化 <0.1%(3 モデル一貫)。mmap で fp32 522MB モデルを WS 上限 358MB(68%)で forward 完走
(logits checksum 完全一致)。cap-gate は top-1 retention ≥97% を fail-closed で適用(`src/llcore/lm/eval.py`)。
モデルは tiny char-LM・simulated quant(速度未測)である点に留意(性能比較対象ではなく計測層)。

> **表の読み方**: Gemma4/PaddleOCR/Cosmos は **モデル**(llcore と土俵が違う)。llama.cpp/GGUF は **圧縮の
> 既存資産**(llcore は圧縮では再導出=競合しない)。llcore の列で他社と質的に異なるのは「**品質ゲートの有無**」
> だけ — そこにだけ llcore の固有性がある。

---

## 4. 立ち位置の結論

### llcore は誰とも「同じ土俵」では戦わない

- **Gemma 4 / PaddleOCR / Cosmos とはモデルで競合しない**。llcore は tiny char-LM の **計測層**であり、
  汎用能力・文書解析・マルチモーダルを争うものではない。
- **llama.cpp / GGUF とも圧縮では競合しない**。llcore の int8 約 3.9×・mmap RAM 超・cliff_then_flat は
  すべて **再導出**(既知)。同じ圧縮率・PPL・mmap で勝とうとするのは筋が悪い。

### llcore の居場所

> **「自宅 CPU で誰でも再現できる、honest な計測 + fail-closed 品質ゲート + (将来)メモリ指標を
> 進化の適応度にする統合層」**。

差別化を主張するなら、圧縮率/PPL/mmap ではなく **`accuracy-gated quantization`(品質ゲート付き量子化)と
honest な計測手続き** に絞る。これは llama.cpp/GGUF にも公式 perplexity ツールにも lm-evaluation-harness 連携にも
「**量子化ツールチェーンの pass/fail 判定として**」は存在しない隙間である(既存は報告まで、合否ゲートは非標準)。

### 設計指針 — GPU / 大メモリで跳ね上がる側面(1 段落)

現状の char-LM・CPU・simulated quant では計測できないが、llcore の構造的勝ち筋は **GPU / 大メモリ環境で
本領を発揮する**ものが多い。具体的には (i) **真の int8 GEMM**(本版は dequant→fp32 forward で速度未測。
真の int8 演算は GPU/対応 CPU で初めて速度面の利得が出る)、(ii) **複数モデルでのページキャッシュ共有**
(mmap の固定コスト構造ゆえ大モデル・多モデル同居ほど相対効果が大きい)、(iii) **定数状態 recurrent × 長文脈**
(T を 16× にしても recurrent state ×1.00 に対し GPT KV ×16・attn ×256。長文脈・大バッチほど定数状態の優位が
線形/二次で開く)。いずれも「小さく測って構造を確認 → 大きい環境で効果が跳ねる」という設計であり、
**現時点で測れていない利得は捏造せず「未測 / 将来検証」と明記する**。

---

## 付録: honest disclosure チェックリスト(対外発信前)

- [ ] 「4×」と書くとき **fp32 比**であることを明記したか(fp16 比なら約 1.9×)
- [ ] 「独自メトリック」と書いていないか(top-1 retention は llama.cpp 既出 → 「既知メトリックのゲート化」と書く)
- [ ] 「accuracy も見るのが新しい」と書いていないか(業界標準 → 新規でない)
- [ ] cap-gate を「新規アルゴリズム」と書いていないか(operationalization = engineering 規律)
- [ ] (c) 進化統合を「実証済み」と書いていないか(prospective・未着手)
- [ ] 競合数値に self-report / 条件付き(専用ベンチ/MoE 比較/RAM 曖昧)の注記を付けたか
- [ ] 測っていない利得(真 int8 GEMM 速度・RAM 総量超・大モデル)を「未測 / 将来検証」と明記したか
