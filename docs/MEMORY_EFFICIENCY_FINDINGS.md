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
| (0') runtime peak RSS | `scripts/recurrent_runtime_rss.py` | 実生成ループの peak WS を文脈長スイープ | T ×8 で **GPT peak ×2.65 / Recurrent・RWKV ×1.00**(解析値を実機裏取り)。32×曲線で **regime 依存 ×1.11→×6.75**(`curve32`)|
| (0'') runtime latency (prefill) | `scripts/recurrent_latency_sweep.py` | 推論 wall-clock の文脈長スケーリング指数 p | T ×16 で **GPT p≈1.37(超線形 O(T²)寄り)/ Recurrent・RWKV p≈0.99(線形 O(T))**。cross-mode 絶対比較不可・各モード内の指数のみ |
| (0''b) decode + amortization | `scripts/decode_latency_sweep.py` | 同一ラン内で prefill(O(T))vs decode(per-token)を対比(clean runner) | **Recurrent/RWKV: prefill p≈1.0(O(T))→ decode p≈0.0(O(1)に amortize)/ GPT: prefill≒decode(各 T 一致, p≈1.37=prefill 指数と一致)= 分離不可**。KV cache 無の明示・質的対比のみ load-bearing |
| (0''') static RSS 床 | `scripts/runtime_floor_rss.py` | python/+torch/+model の段階 RSS(別プロセス隔離) | **torch ランタイム税 ~180MB が支配**(別ラン再現、初回 197.3≒197.8)。足場比はモデル規模依存(1.51MB本体で142×) |
| (a) mmap read-only 重み | `scripts/mmap_weights_poc.py` | load 時 RSS(別プロセス隔離) | eager は即全載、mmap は **load 時 ΔRSS を ~2.8% に遅延**(54MB モデル) |
| (a') RAM 超 × mmap | `scripts/mmap_ram_exceed_poc.py` | working-set 上限 < モデルで forward 完走 | **522MB モデルを 358MB の WS 上限で完走**(出力一致)= 使える RAM < モデルでも回る |
| (c) int8 streaming 推論 | `scripts/int8_streaming_infer.py` | dense vs 層ごと dequant の常駐/peak WS | **常駐 72% 削減**(539→149MB)/ stream は 368MB 上限で完走(出力一致)|
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

## (0') runtime peak RSS スイープ — 解析値を実機計測へ昇格 (`out/recurrent_runtime_rss.json`)

(0) は state_bytes(実測)+ KV/attn(解析値)だった。`scripts/recurrent_runtime_rss.py` はその後続で、
**実ワークロードを文脈長 T で振り、別プロセス隔離で peak working set を実測**(重み構成は T 間固定なので
増分は純粋に文脈依存コスト)。

| T | GPT peak WS | Recurrent peak WS | RWKV peak WS |
|---|---|---|---|
| 256 | 229.8 MB | 205.0 MB | 215.3 MB |
| 512 | 247.3 MB | 205.1 MB | 216.0 MB |
| 1024 | 330.5 MB | 204.8 MB | 215.7 MB |
| 2048 | **607.9 MB** | 204.8 MB | 215.2 MB |

- **T 256→2048(×8)の peak WS 倍率: GPT ×2.65(文脈で膨張)/ Recurrent ×1.00 / RWKV ×1.00(平坦)**。
  固定 baseline(~205MB=torch+重み)を引くと GPT の文脈コストは ~25MB→~403MB と超線形(attn O(T²) が
  大 T で支配、1024→2048 で +277MB)。**解析値(KV 線形・attn 二次)を実機 peak RSS で裏取り**。
- honest: peak WS は torch + 固定重み + T 依存バッファの合算。クリーンな信号は増分トレンド。GPT.generate は
  block_size crop で実行上有界(本測は厳密長文脈想定)。
- **2026-06 追記 — 32× 曲線へ拡張(`out/recurrent_runtime_rss_curve32.json`)**: 128→4096(×32)の 6 点で測ると、
  GPT の膨張率は**計測レンジ依存**。128→512(×4)では **×1.11**(固定重み床が支配、O(T²)項は誤差に埋没)、
  512→4096(×8)では **×6.75**(二次項が床を追い越す)、全域 128→4096 で **×7.53**。recurrent/RWKV は全域 205/216MB 平坦。
  「×N で ×M」の M は起点で激変=単点でなく曲線で読む。1024/2048 は別ランと 0.1MB 差で一致(クロスラン再現)。

## (0'') runtime latency スケーリング — compute 軸(`out/recurrent_latency_sweep.json`)

メモリ(0)(0')の相補軸。`scripts/recurrent_latency_sweep.py` が**推論 wall-clock を文脈長 T で振り**、
time ∝ Tᵖ の指数 p を log-log 最小二乗で推定(別プロセス隔離・各点 11 回・`torch.set_num_threads(1)`)。

| モード | scaling 指数 p (min) | p (median) | 解釈 |
|---|---|---|---|
| GPT | **1.37** | 1.46 | 超線形(O(T²) 寄り) |
| Recurrent | **0.99** | 0.96 | 線形(O(T)) |
| RWKV | **0.99** | 1.00 | 線形(O(T)) |

- **メモリで見えた「GPT は文脈で膨張 / recurrent 系は構造的に軽い」が compute 軸でも同じ向きで再現**。
- honest(最重要): **cross-mode の絶対 ms は比較不可**。recurrent/RWKV は Python の per-step ループ(T 回の関数呼び出し)
  =インタプリタ律速、GPT は 1 回の vectorized forward。読むのは**各モード内の scaling 指数のみ**。
- 当初 repeats=7 では RWKV の T=128 が startup ノイズで外れ値(p≈0.5)になったが、repeats=11 で p≈0.99 に収束
  (ノイズ点を消さず増やして潰した経緯ごと記録)。

### (0''b) decode サブ軸 + amortization — prefill vs decode を同一ラン内で対比(`out/decode_latency_sweep.json`)

(0'') が **prefill/batch-forward 全体**だったのに対し、`scripts/decode_latency_sweep.py` は **同一プロセス・同一モデル・同一条件の
1 ラン内で prefill(状態構築)と decode(context age T で次の 1 トークン)を両方計時**し、recurrent の amortization を
**別ラン比較なしで直接対比**する(streaming 生成 regime、チャット体感遅延の支配項)。

クリーンな runner(GitHub Actions 7GB、ローカル 3.6GB 機の contention を排除)で計測。

| モード | prefill 伸び (T ×16) | prefill p (min) | decode 伸び | decode p (min) | 解釈 |
|---|---|---|---|---|---|
| Recurrent | **×15.6** | 1.00 | **×0.98** | **-0.002** | prefill O(T) → decode **O(1) に amortize** |
| RWKV | **×17.5** | 1.01 | **×1.10** | **0.03** | 同上(warmup=5 で startup 外れ値解消) |
| GPT | ×45.6 | 1.372 | ×44.5 | 1.365 | **prefill ≒ decode(各 T 一致・指数も一致)= 分離不可** |

- **核心 = recurrent の amortization**: recurrent は「状態構築(prefill, O(T)= T 回 step)」と「1 手追加(decode, O(1))」を分けて払える。GPT(cache 無)は decode 1 step も全文脈再 forward=**prefill と同一計算**ゆえ各 T で **prefill≒decode**(例:99≒96ms / 958≒957ms、しかも **prefill 指数 1.372 ≒ decode 指数 1.365**)= amortize 不可。**同一ラン計測なので「recurrent だけ O(T)→O(1) に落ち GPT は落ちない」が直接対比として出る**(別ラン比較の弱点を排除)。
- honest: (1) cross-mode 絶対 ms 比較不可。(2) **この GPT は KV cache 無**(prod は cache で decode O(T)/token、cache 有でも T で増大 vs recurrent O(1) flat は質的に別物)。(3) load-bearing は特定 ×N でなく「各モード内の指数」と「GPT prefill≒decode / recurrent だけ decode 平坦」の質的対比(クリーン runner で単調・再現的になり旧ローカルの contention スパイクは解消)。(4) GPT 指数 ~1.37 は二次項が完全支配しきらない小モデル regime。(5) RWKV 小 T startup 外れ値は warmup 不足と特定し default warmup 引き上げで解消。
- 可視化: `assets/articles/llcore_decode_latency.svg`(amortization の fork 図、a7 記事に挿入済)。

## (0''') static RSS 床 — 文脈非依存の土台(`out/runtime_floor_rss.json`)

(0)(0')(0'')が**文脈長依存**コストなのに対し、こちらは**文脈に依存しない静的な土台**。
`scripts/runtime_floor_rss.py` が python / +torch / +model を別プロセス隔離で段階測定(各 3 回中央値)。

| stage | RSS (MB, median) |
|---|---|
| python(素) | 18.1 |
| + import torch | 197.8 |
| + 豆モデル(n_embd=176) | 207.4 |

- **言語ランタイム税(torch − python)= ~180MB が支配項**。初回一回限り計測(197.3MB / 税 183.9MB, a9 本文)を
  再走可能ハーネスで裏取りし、**197.8MB / 税 179.7MB(~2% 差)で再現**。
- 足場比(プロセス RSS ÷ int8 重み実体)は**モデル規模依存**: 1.51MB 本体で 142×、2.8MB 本体(既定 config)で 73×。
  比の絶対値は config 依存だが「本体より足場が桁違い」の構図は不変=「bit でなくランタイム床を攻めろ」の土台。
- honest: RSS は測定時点の WorkingSetSize 実測。int8 MB は当 config の例示。Rust/candle の baseline RSS は未計測
  (主張の最終確証は要 candle 実測)。

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

## (c) int8 streaming-dequant 推論 — (a')mmap + (b)int8 を実推論パスへ統合 (`out/int8_streaming_infer.json`)

`scripts/int8_streaming_infer.py`。`Int8Linear`(int8 を resident に保持し forward 内で層ごとに fp32 へ
dequant→即解放)で CharGPT の Linear を置換。同一 int8 ソースから **dense(全層一括 dequant 常駐)vs
stream(層ごと dequant)** を別プロセスで比較。

### 実測(130M params, n_embd=1024 / L=10)

| mode | working-set 上限 | resident | peak WS | logits checksum |
|---|---|---|---|---|
| dense | なし | 538.6 MB(fp32 全載) | 963.8 MB | -192.1 |
| stream | なし | **148.9 MB(int8, 0.285×)** | 882.2 MB | -192.1 |
| stream | **368 MB(強制成功)** | 148.9 MB | **368.2 MB**(完走) | -192.1 |

### 知見(honest)
- **★堅牢な勝ち = 常駐モデル 72% 削減**(dense fp32 538.6 MB → stream int8 148.9 MB)。
- **圧力なしの peak WS はほぼ不変**(963.8 vs 882.2)= torch caching allocator が解放した fp32 を OS へ
  返さず、transient 活性も加わるため。**「常に省メモリ」ではない**ことを honest に明示。
- **★削減は圧力下で顕在化**: working-set 上限 **368 MB(= dense 常駐 538.6 MB 未満)で stream は完走**
  (capped peak WS 368.2 MB ≤ 上限)。dense はこの上限に必須常駐(539MB)が収まらない。= **streaming-dequant
  は dense が要求する RAM 未満で動かせる**((a') の RAM 超機構と同根)。
- **正当性**: dense / stream / stream(capped) の logits checksum **全一致**(-192.1)= メモリ最適化は結果不変
  (量子化誤差は別問題で (b) で測定済み)。
- 留保: 量子化は `nn.Linear` のみ(Embedding/LN は fp32)。「resident ≈ 最大層」
  まで絞るには int8 自体も mmap でストリーム + 圧力が要る(本版は int8 を常駐保持)。
- **2026-06 追記 — 裏コスト(latency)を交絡分離して測った**: stream は forward 毎に層ごと dequant=再計算する分
  dense より遅いはず、と考え `int8_streaming_infer.py` に forward median 計時を追加(`--forward-repeats`)。
  - **(失敗→学び)130M(canonical config)では本機(RAM ~3.6GB)で倍率が 4 ラン で ×1.46 / ×10.88 / ×11.72 / ×0.21
    と桁違いに振れ方向すら反転**(forward が memory-pressure / page-fault 雑音に支配、dense が thrash すると逆に
    stream が速く見える)。→ この規模では単一倍率を load-bearing にしない(cherry-pick 回避)。
  - **(統制)交絡変数=RAM 圧を消すため小モデル(n_embd=256/L=4、forward が余裕で常駐)で再測**: 5 ラン中
    1 件が dense 側の一過性スパイク(×0.18)だったのを除く **4 ラン が ×1.20 / ×1.22 / ×1.27 / ×1.31 に密集**
    (dense ~7-8ms / stream ~9.5-11ms)。→ **純粋な per-layer dequant 再計算コストは安定 ~×1.25**。
  - **結論(honest)**: 「常駐 72% 削減」の latency 対価は、**圧力のない領域では ~×1.25 の小さな定数**。130M で見えた
    ×0.2〜×11 のカオスは**アルゴリズムコストでなく RAM 圧 thrashing**。交絡(memory-pressure)を統制して初めて
    アルゴリズム本来のコストが見える、という計測規律の実例。大規模での厳密定量化は要・高RAM/GPU オフロード。
    計時の仕組みは committed。

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
- **★成長要素として eval に capability gate を新設**: 発見「PPL-only gate は壊れた低ビットを PASS」を受け、
  `src/llcore/lm/eval.py` に **`held_out_top1_report`(top-1/top-5)+ `passes_capability_gate`(fp32 比
  top-1 retention ≥97%)** を追加(既存関数は不変=純粋追加)。スイープに配線した結果、**ppl-gate は PASS でも
  cap-gate が止めるビット幅 = multi_smoke 3bit / realp1 2bit** = PPL だけの合否が見逃す capability 喪失を
  実際に捕捉。llcore の評価基盤が一段成長。
- 留保: weights-only / dequant fp32 の simulated quant(速度未測)/ 2bit は QAT なしの PTQ 限界。

## (b'') per-group 量子化 — 低ビットの床を下げられるか (#3 CPU スライス, `out/quant_group_compare*.json`)

`scripts/quant_group_compare.py`。per-channel(群=行全体)を、行を `group_size` 列ごとに区切る per-group へ
拡張し、低ビット(3/2bit)で per-channel が壊れる床を救えるかを held-out PPL + top-1 で検証。

### realp1 (11.9M) — fp32 PPL 38.32 / top1 28.65%
| bits | group | ΔPPL% | Δtop1(pp) | ppl-gate | cap-gate |
|---|---|---|---|---|---|
| 3 | full | +4.8% | -0.70 | PASS | **PASS** |
| 3 | 32 | +3.2% | -0.60 | PASS | **PASS** |
| 2 | full | +163.9% | -13.46 | PASS | FAIL |
| 2 | 64 | +44.9% | -6.88 | PASS | FAIL |
| 2 | 32 | +32.9% | **-5.31** | PASS | FAIL |

### 知見(honest)
- **per-group は単調に低ビット品質を改善**(誤差↓・PPL↓・top1↑)、scale 増で footprint は微増(両モデルで確認)。
  realp1 2bit: top1 劣化が **full -13.5pp → group32 -5.3pp(≈60% 減)**。multi_smoke 2bit: group≤64 が
  **ppl-gate を救出**(full は FAIL)。
- **しかし strict cap-gate(top1 fp32 比 97% 保持)は 2bit では RTN per-group でも届かない**(realp1 group32 で
  81.5% 保持)。**3bit が実用床**(realp1 は per-channel で既に cap-gate PASS=97.6% 保持、per-group は上乗せ)。
- **結論**: per-group は床を確実に押し下げるが、**2bit を「安全」にするには RTN 超(GPTQ/AWQ 誤差補償 or QAT)が
  必要**。これは #3 の GPU/将来課題(真の int8 GEMM と同じく)。CPU で測れる範囲の honest な到達点。
- 留保: RTN(誤差補償なし)/ weights-only / simulated quant(速度未測)。

## (b''') GPTQ 誤差補償 vs RTN — 2bit の床は越えられるか (#3 CPU スライス, `out/gptq_compare*.json`)

`scripts/gptq_compare.py`。GPTQ(Frantar et al. 2022)を自前実装 — 校正データで各 Linear の入力 Hessian
``H=Σxᵀx`` を捕捉し、出力誤差 ‖(W−Ŵ)X‖² を最小化する列ごと誤差補償量子化。RTN per-channel と同条件比較。

### realp1 (11.9M) — fp32 PPL 38.32 / top1 28.66%(Linear 25 層を量子化)
| bits | method | ΔPPL% | Δtop1(pp) | ppl-gate | cap-gate |
|---|---|---|---|---|---|
| 3 | rtn | +4.5% | -0.65 | PASS | **PASS** |
| 3 | gptq | +2.4% | -0.43 | PASS | **PASS** |
| 2 | rtn | +159.3% | -13.35 | PASS | FAIL |
| 2 | gptq | **+41.5%** | **-6.38** | PASS | FAIL |

### 知見(honest)
- **GPTQ は全 bit で RTN を改善**(2bit: ΔPPL +159→+41.5%、top1 劣化 -13.35→-6.38pp ≈ 52% 減)= 出力誤差
  最小化(誤差補償)が効いている実証。**機構の気付き**: GPTQ は **weight 誤差を犠牲にして output 誤差を下げる**
  (probe: 2bit weight err 0.61→0.68 だが output err 78.6→71.2)= ‖W−Ŵ‖² でなく ‖(W−Ŵ)X‖² を最小化。
- **だが strict cap-gate(top1 97% 保持)は 2bit では GPTQ でも越えられない**(realp1 77.7% 保持)。
  **3bit が PTQ の実用床・2bit は QAT 領域**、が RTN/per-group/GPTQ の 3 手法で一貫した結論。
- **★比較の気付き**: realp1 2bit で **per-group32 RTN(top1 -5.31pp)が GPTQ-per-channel(-6.38pp)を上回る**。
  極低ビット小モデルでは「**粒度(per-group)> 誤差補償(GPTQ)**」になり得る(両者は相補的=GPTQ+per-group が
  真の SOTA)。「最新手法 GPTQ が常に最強」ではない、を自前実測で確認。
- **multi_smoke(小モデル)**でも同傾向: **GPTQ が 2bit の ppl-gate を救出**(RTN +852% FAIL → GPTQ +458% PASS)、
  3bit は GPTQ でも cap-gate を僅かに届かず(top1 96.3% 保持 < 97%)= 小モデルほど床が高い。
- 留保: Linear のみ量子化(Embedding/LN fp32)/ 校正 8,192 tokens / weights-only / simulated quant(速度未測)。

## (d) QAT capstone — 「2bit は QAT 領域」を実証 (`out/qat_train_2bit.json`)

`scripts/qat_train.py`。fake-quant + STE(straight-through estimator)で重みを量子化したまま学習し、PTQ が
越えられなかった 2bit cap-gate を QAT が越えるか検証。fp32 reference と同 corpus/config/iters(2000)で公平比較。

### multi_smoke 2-bit(fp32 ref: PPL 24.88 / top1 36.28%)
| 手法 | PPL | top1 | retention | cap-gate |
|---|---|---|---|---|
| PTQ RTN | 236.9 | 7.98% | 22% | FAIL |
| PTQ GPTQ | 138.7 | 12.07% | 33% | FAIL |
| **QAT** | **38.15** | **30.10%** | **82.9%** | FAIL |

### 知見(honest)— アークの結論
- **QAT は PTQ を圧倒**: 2bit top1 **30.10% vs GPTQ 12.07%(+18pp)/ RTN 7.98%(+22pp)= 約 3 倍の保持**。
  「**2bit は QAT 領域**」(PTQ では届かない)を実証 = 量子化を見越して学習する効果は本物。
- **だが QAT でも strict 97% cap-gate は越えられず**(82.9% 保持)。この CPU-scale char-LM では 2bit は
  QAT でも完全には安全化できない=**より大きいモデル/学習予算/学習可能 scale(LSQ 等)が要る**可能性。
- **アーク総括**(RTN→per-group→GPTQ→QAT、2 モデル): **3bit が PTQ の安全な実用床**(realp1 は per-channel で
  cap-gate PASS)。**2bit は手法を上げるほど damage が減る**(RTN 7.98% → GPTQ 12% → QAT 30% top1)が、
  **strict gate を越えるには QAT でも本モデル規模では不足**。「床を動かすには質的に別アプローチ(学習時量子化)」
  は正しく、QAT は実際に大きく前進させたが、tiny model の 2bit 完全制覇には至らず=honest な到達点。
- 留保: CPU smoke / weights-only / Linear のみ / 固定 per-channel scale / 速度未測。

## (d') LSQ(学習可能 scale)— 「2bit 制覇」再挑戦の honest 決着 (`out/qat_lsq_2bit.json`)

`scripts/qat_train.py --method lsq`(Esser et al. ICLR2020, arXiv:1902.08153 を自前実装)。固定 per-channel
scale を **勾配で学習**する(round STE + 勾配均衡化 g=1/√(N·Q_P) + 初期化 s=2·mean(|w|)/√Q_P、scale は 1-D で
trainer の weight-decay 群から除外)。multi_smoke 2bit を固定 scale QAT と **同 corpus/config/iters(2000)** で比較。

| 手法 (multi_smoke 2bit, fp32 ref top1 36.28%) | top1 | retention | cap-gate |
|---|---|---|---|
| PTQ RTN | 7.98% | 22% | FAIL |
| PTQ GPTQ | 12.07% | 33% | FAIL |
| QAT(固定 scale) | 30.10% | 82.9% | FAIL |
| **QAT + LSQ(学習可能 scale)** | **30.48%** | **84.0%** | **FAIL** |

### 知見(honest)
- **LSQ は固定 scale QAT を上回ったが、その差は +1.1pp(82.9→84.0%)に留まる**。手法系譜 RTN→GPTQ→QAT→LSQ で
  retention は 22→33→82.9→**84.0%** と単調改善するが、**LSQ でも strict 97% cap-gate には遠く届かない**。
- = **prior-art の予言どおり**: LSQ 自身が小モデル SqueezeNext-23 で 2bit -14.0pt を報告(ResNet-18 は -2.9pt)、
  k-bit scaling law(Dettmers 2023)/ QiD(arXiv:2411.17691, N の指数 0.226)は「小モデルは冗長性が無く 2bit を
  吸収できない」と複数独立に示す。**char-LM(1.36M)の 2bit は、学習可能 scale を足しても規模の壁が支配的**。
  2bit で 90%+ retention は literature 上「7B+ + VQ codebook + QAT/fine-tune + group64」が揃って初成立
  (EfficientQAT 7B=92.7% / 70B=95.9%、QuIP#/AQLM 等)。
- **honest な締め**: 「手法を上げれば床が下がる」期待は LSQ でも**わずかしか報われなかった**(+1.1pp)。床を本当に
  動かすのは手法でなく **規模 / 学習予算 / VQ codebook** であることが自前実測で確定。**3bit が PTQ 実用床**のまま。
- 留保: CPU smoke / weights-only / Linear のみ / simulated quant(速度未測)/ LSQ scale は 1-D で WD 非適用(fair 比較)。
- 実装 = `scripts/qat_train.py`(lsq_quant / LSQLinear / convert_to_lsq, `--method lsq`、既存 qat パス不変・純粋追加)+
  `tests/unit/test_qat_train.py`(LSQ 7 件追加、全 12 passed / ruff / mypy strict green)。

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
