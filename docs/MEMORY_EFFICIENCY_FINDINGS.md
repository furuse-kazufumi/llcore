# メモリ効率 — 実測知見 (llcore 北極星, 2026-06-17)

> 2026-06-16 の pivot (memory:`project_llcore_memory_efficiency_pivot`, ユーザー決定 option A)
> で、llcore の北極星を **capability(賢さ)から「メモリ使用効率(仮想メモリ含む)」へ転換**。
> capability は NULL_TIE/NEGATIVE(進化≒勾配)で負け筋、memory は RWKV/Mamba の定数状態という
> 構造的勝ち筋。本ドキュメントは「構造プロット(config 由来)」を **実機計測へ昇格**させた 3 本柱の
> 正本。記事素材も兼ねる(append/更新可)。
>
> 実行環境: Windows 11 / Python 3.11 / torch 2.12.0+cpu(GPU なし)。

---

## 3 本柱サマリ

| 柱 | スクリプト | 何を実測したか | ヘッドライン |
|---|---|---|---|
| (0) 定数状態 vs 文脈線形 | `scripts/memory_footprint_harness.py` | recurrent state の実バイト vs GPT KV/attn | T を 16× にしても recurrent **×1.00**、GPT KV **×16**、attn **×256** |
| (a) mmap read-only 重み | `scripts/mmap_weights_poc.py` | load 時 RSS(別プロセス隔離) | eager は即全載、mmap は **load 時 ΔRSS を ~2.8% に遅延**(54MB モデル) |
| (b) int8 weight-only 量子化 | `scripts/int8_quant_footprint.py` | footprint vs held-out PPL | 重み常駐を **約 3.9× 圧縮(74–75% 削減)**、PPL コスト **< 0.1%** |

「仮想メモリ含む」の honest な解釈(pivot memo より): スワップに hot data を載せる素朴な活用では
なく、**working set を小さく・予測可能に**する設計 — (a) mmap=OS ページキャッシュ、(0) 定数状態
recurrent、(b) 量子化、の 3 つが本筋。

---

## (0) 定数状態 recurrent vs 文脈線形 GPT (`out/mem_footprint.json`)

| T (文脈長) | Recurrent state (実測) | RWKV state (実測) | GPT KV-cache (解析) | GPT attn 行列 (解析) |
|---|---|---|---|---|
| 64 | 2,048 B | 10,240 B | 262,144 B | 65,536 B |
| 1024 | 2,048 B | 10,240 B | 4,194,304 B | 16,777,216 B |
| 倍率 (×16 文脈) | **×1.00** | **×1.00** | **×16 (線形)** | **×256 (二次)** |

- `state_bytes` は生成時に保持する recurrent 状態テンソルの**実バイト数**(REAL)。GPT の KV/attn は
  config からの**解析値**で、RSS トレンドで裏取り。
- 構造的決着: recurrent は任意長の過去を**定数サイズ**で運ぶ。GPT は厳密長文脈の attention に
  文脈線形(KV)・二次(attn 行列)のメモリが要る。これが「memory が勝ち筋」の土台。
- honest 留保: GPT の `generate` は block_size に crop するので実行上は有界。「線形」は *block_size を
  伸ばして厳密長文脈を attention する* 場合の必要量。

---

## (a) mmap read-only 重み PoC (`out/mmap_weights_poc.json`)

`torch.load(mmap=True)` + `load_state_dict(assign=True)` で重みを **file-backed のまま**割り当て
(llama.cpp 流)。計測は torch allocator の汚染を避けるため **eager / mmap を別プロセスで隔離**。

### realp1 (11.9M params, model.pt 49.3 MB, param 53.91 MB)

| モード | load 時 ΔRSS | touch 後 ΔRSS |
|---|---|---|
| eager (`mmap=False`) | **50.77 MB**(≈モデル全載) | 51.64 MB |
| mmap (`mmap=True`) | **1.42 MB(×0.028)** | 51.54 MB |

- **ヘッドライン**: eager は load 時に全重みを anonymous メモリへ即読込(ΔRSS ≈ モデルサイズ)。
  mmap は遅延し、load 直後 ΔRSS は **約 2.8%**。
- **on-demand fault**: mmap でも全バイトを touch するとページが fault-in し ΔRSS は ~51.5 MB へ
  =「**使った working set の分だけ**載る」を実機で確認。
- **機能正当性**: mmap(assign=True)の forward logits は eager と **完全一致**(max|Δ|=0.0)。
- **規模効果(重要 honest 知見)**: mmap の load 時 ΔRSS は**モデルサイズによらず ~1.4–1.5 MB の
  ほぼ固定コスト**(mmap セットアップ + メタデータ unpickling)。よって smoke(param 7.73 MB)では
  ×0.218 だが realp1(53.91 MB)では ×0.028。**モデルが大きいほど mmap の相対効果が大きい**。
- 再現性: realp1 で 2 回測定 → ΔRSS 1.42 / 1.38 MB(×0.028 / 0.027)で安定。

### honest 留保 (mmap)
- 恩恵は「**部分 working set** / 複数モデルでのページキャッシュ共有 / コールド起動の遅延」。
  全重みを必ず一度に読むワークロードでは最終 RSS は eager に近づく(touch 行が示す)。
  正確な主張は「常に省メモリ」ではなく「**必要分だけ・遅延で**」。
- 真の RAM 超(モデル > 物理 RAM)の検証は別途大型モデルで要実測。本 PoC は「load 時に全載しない」
  性質を実機 RSS で示すまで。
- RSS は WinAPI(psapi)の WorkingSet。load peak には torch import 等のプロセス baseline が含まれる
  ため、クリーンな信号は **ΔRSS(baseline 差分)**。

---

## (a') RAM 超 × mmap 実証 — 「使える RAM < モデルでも回る」 (`out/mmap_ram_exceed_poc.json`)

(a) は load 時の遅延までを示したが、フル forward は全重みを touch するためメモリ圧力が無ければ RSS は
モデルサイズへ収束する。北極星「仮想メモリ含む」の本丸 = **使える物理 RAM がモデルより小さくても回る**を、
`scripts/mmap_ram_exceed_poc.py` で実証。Windows の **working-set hard max(`SetProcessWorkingSetSizeEx`)を
モデルサイズ未満**に設定したサブプロセスで forward を完走させ peak WS を実測。

### 実測(130M params ランダム CharGPT, n_embd=1024 / L=10)

| mode | working-set 上限 | load ΔRSS | peak WS | logits checksum |
|---|---|---|---|---|
| uncapped | なし | 541.6 MB | **1243.9 MB** | -215.1 |
| capped | **357.6 MB(強制成功)** | 159.7 MB | **357.7 MB** | -215.1 |

- **fp32 モデル 522 MB を、working-set 上限 358 MB(= モデルの 68%)で forward 完走**。capped peak WS =
  357.7 MB ≤ 上限 = **モデルサイズ未満で動いた**。read-only mmap ページは clean なので圧力下で破棄され、
  再 fault で disk から読み直す(llama.cpp 流)= **これが「RAM 超で回る」機構**。
- **機能正当性**: capped と uncapped の logits checksum 完全一致(-215.1)= 上限下でも結果は同一。
- **int8 連結**: 同モデルを per-channel int8 でディスク保存すると **131 MB(fp32 522 MB の 0.251×=4x 縮小)**。
  量子化で「ディスクに置くページ」も減る → mmap 常駐がさらに軽くなる。

### honest 留保 (a')
- このマシンは avail RAM が限られる(本実行時 ~3.6 GB)ため、**物理 RAM 総量を literally 超える巨大モデル
  ではなく**「working-set 上限 < モデルサイズ」で同性質を実証。RAM 総量超の超大型は GPU/大 RAM で将来検証。
- `SetProcessWorkingSetSizeEx` の hard-max は本環境では強制された(cap_set_ok=true・peak WS ≈ 上限)。
  強制可否は環境依存なので JSON に `cap_set_ok` と実測 peak をそのまま残す(成功を偽装しない)。
- uncapped peak が 1243.9 MB と大きいのは torch ランタイム + 全materialize + transient の合算で、
  クリーンな信号は **capped peak が上限に張り付いた**こと。int8 は disk/load まで(per-layer streaming
  dequant forward は将来課題)。

## (b) int8 weight-only 量子化 footprint vs PPL (`out/int8_quant_footprint*.json`)

2-D 重み行列を対称 int8 量子化(per-tensor / per-channel)。同一 held-out split で fp32 と
比較。footprint には **scale(fp32)と非量子化 1-D params(bias/LayerNorm)を必ず計上**。

| モデル (params, vocab, 言語) | fp32 PPL | 方式 | 削減率 | ΔPPL | gate | 重み rel-RMSE |
|---|---|---|---|---|---|---|
| shakespeare (0.81M, 65, en) | 6.776 | per_tensor | 74.4% | +0.02% | PASS | 0.0130 |
| shakespeare | | per_channel | 73.8% | +0.01% | PASS | 0.0070 |
| multi_smoke (1.36M, 4358, ja×84作) | 24.883 | per_tensor | 74.6% | +0.10% | PASS | 0.0158 |
| multi_smoke | | per_channel | 74.0% | +0.01% | PASS | 0.0072 |
| realp1 (11.9M, 3044, ja単一本, ctx256) | 38.315 | per_tensor | 74.8% | +0.01% | PASS | 0.0117 |
| realp1 | | per_channel | 74.6% | +0.00% | PASS | 0.0074 |

- **ヘッドライン**: int8 weight-only は重み常駐を **約 3.9×(74–75%)圧縮**しつつ、held-out PPL
  劣化は **0.1% 未満**(多くは 0.02% 未満)。全方式が unigram gate を維持。
- per_channel は per_tensor より重み誤差が小さい(scale を行ごとに持つ)。footprint は scale 増分で
  わずかに大きいが、char-LM 規模では誤差低減のメリットが上回る。
- 英語(小 vocab)・日本語単一本・日本語マルチで**一貫**して同じ挙動 = 頑健。

### honest 留保 (int8)
- **weights-only** の量子化。activation は fp32、推論時の activation/KV メモリは別問題でここでは
  測らない。北極星の「常駐・保存に必要な重みバイト数」を対象。
- **simulated quantization**: dequant して fp32 で forward する(真の int8 GEMM ではない)。
  「量子化が PPL をどれだけ劣化させるか」を測るためで、**速度は測っていない**。
- footprint = 保存/常駐バイトの実合計(int8 本体 + scale + 非量子化 1-D)。理想下限(全 params 1B)は
  ratio 0.25 だが、scale と 1-D params で実値は 0.25–0.26。
- tied `wte`/`lm_head` は同一 Parameter なので 1 度だけ量子化・計上。因果マスク buffer は対象外。

---

## (b') 量子化ビット幅スイープ — cliff_then_flat の実測 (`out/quant_bitwidth_sweep*.json`)

memory-scaling ワークフローの完全性批評が推奨した反証可能実験。`scripts/quant_bitwidth_sweep.py` で
per-channel weight-only PTQ を {8,6,5,4,3,2}-bit へスイープし、held-out PPL に加え **hard-capability
proxy = 次トークン top-1 accuracy** を併記。予測(Dettmers 2023: ~4bit 平坦・3bit cliff・2bit 破綻)を検証。

### multi_smoke (1.36M, vocab 4358) — fp32 PPL 24.88 / top1 36.28%
| bits | 削減率 | PPL | ΔPPL% | top1 | Δtop1(pp) | gate |
|---|---|---|---|---|---|---|
| 8 | 74.0% | 24.886 | +0.01% | 36.28% | -0.00 | PASS |
| 5 | 83.3% | 24.935 | +0.21% | 36.25% | -0.04 | PASS |
| 4 | 86.4% | 25.295 | +1.66% | 36.08% | -0.20 | PASS |
| 3 | 89.5% | 27.761 | +11.57% | 34.30% | -1.98 | PASS |
| 2 | 92.6% | 269.716 | **+983.95%** | 7.49% | **-28.80** | **FAIL** |

### realp1 (11.9M, vocab 3044) — fp32 PPL 38.32 / top1 28.65%
| bits | 削減率 | PPL | ΔPPL% | top1 | Δtop1(pp) | gate |
|---|---|---|---|---|---|---|
| 8 | 74.6% | 38.316 | +0.00% | 28.65% | -0.02 | PASS |
| 4 | 87.1% | 38.599 | +0.74% | 28.42% | -0.25 | PASS |
| 3 | 90.2% | 40.154 | +4.80% | 27.97% | -0.70 | PASS |
| 2 | 93.3% | 101.114 | **+163.90%** | 15.21% | **-13.46** | PASS |

### 知見(honest)
- **cliff_then_flat は確認**。8/6/5bit は平坦、低ビットで非線形に急落 = 予測通り。
- **★cliff 位置はモデルサイズ依存**: 小 multi_smoke は 3bit で劣化開始(+11.6%)・2bit 破綻。大 realp1 は
  **3bit でも実用**(+4.8% / top1 -0.7pp)、cliff は 2bit。**大モデルほど低ビットに頑健**(冗長性が多い、
  literature と整合)。
- **★PPL-only gate は危険を実証**: realp1 の 2bit は top1 が **-13.46pp(28.7%→15.2%)= 半減近く壊れている
  のに unigram gate は PASS**(PPL 101 < 0.85×215=183)。「PPL が unigram を下回る」だけの合否は低ビットの
  capability 喪失を見逃す。→ llcore の合否ゲートに hard-capability proxy を足すべき。
- **批評仮説の honest 反証**: 「top1 は PPL より先に劣化する」という事前予想は**本データでは成立せず**、
  top1 と PPL はほぼ同時(lockstep)に劣化した。誇張せず「同時劣化 + gate が粗い」が正確な観測。
- 留保: weights-only / dequant fp32 の simulated quant(速度未測)/ 2bit は QAT なしの PTQ 限界。

## 記事側面 (feedback_daily_articles_policy の 13 側面)
- **技術設計/実装報告**: 3 本柱の実機計測。「構造プロット → 実測」への昇格手続き。
- **honest disclosure**: 各柱に明確な留保(weights-only / simulated quant / 部分 working set /
  解析値 vs 実測)。mmap の "異常に良い 2.8%" は 2 回再測 + 固定コスト構造で裏取り。
- **教訓/TRIZ**: 「制約(自宅 CPU・少 RAM)を価値へ反転」。mmap=固定コスト → 大モデルほど有利、という
  非自明な規模則。
- **ベンチ/業界比較**: int8 は llama.cpp/GGUF の常識を char-LM で再確認(約 4× / 品質ほぼ無劣化)。
- **ユーザー体験**: 「自宅 PC で動く」FullSense 哲学の定量的裏付け。

## 次の一手 (capability は従属目標として保留・捨てない)
- mmap: モデル > 物理 RAM の実機検証(大型 state_dict で cold ページ常駐を RSS 実測)。
- int8: 真の int8 GEMM(速度)・4bit / per-group 量子化・activation 量子化への拡張検討。
- 定数状態: recurrent/RWKV の**実生成**での peak RSS を文脈長スイープで取り、解析値と突合。
