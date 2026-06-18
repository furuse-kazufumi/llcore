I now have complete context on all integration points. I have everything I need to produce the branch B design. Here is the deliverable.

---

# Branch B 設計: LSQ(学習可能 step size 量子化)を qat_train.py へ

> 対象正本: `D:\projects\llcore\scripts\qat_train.py`(現 fake-quant + STE)/ `D:\projects\llcore\src\llcore\lm\eval.py`(cap-gate)/ `D:\projects\llcore\docs\MEMORY_EFFICIENCY_FINDINGS.md` (d) / `ARTICLE_SEEDS.md`。
> 現状: multi_smoke(1.36M, vocab4358)2bit QAT で top1 30.10%(fp32 36.28% の **82.9% 保持**)、PTQ GPTQ 12.07% / RTN 7.98% を約3倍。だが strict cap-gate(retention≥97%)は QAT でも 2bit 未達。
> 本設計はコードを書かず**設計のみ**。実装は後で TDD(§5 のテストを先に書く)。

---

## 0. 設計の前提となる「現状コードの正確な事実」(推測でなく実読)

設計の妥当性はこの事実認識に依存するので最初に固定する。

1. **量子化器の現状**(`qat_train.py:62-67`): `fake_quant_ste(w, bits)` は **per-channel 対称・固定 scale**。`scale = w.abs().amax(dim=1, keepdim=True)/qmax`(行ごと最大絶対値、**毎 forward で w から再計算**= 学習対象でない)。`qmax = (1<<(bits-1))-1`(2bit で qmax=1、対称 {−1,0,+1})。STE は `w + (wq − w).detach()`。
2. **置換機構**(`FakeQuantLinear` / `convert_to_fake_quant`, `:70-92`): `nn.Linear` を再帰置換し `.bits` 属性を付与。weight/bias を copy_。
3. **★最重要 = optimizer 構築タイミング**(`trainer.py:54-79`): `Trainer.__init__` が `self._configure_optimizer()` を呼び、`model.parameters()` を **dim 別に decay(≥2-D)/no_decay(<2-D)** へ振り分けて AdamW を作る。つまり **学習可能 scale パラメータは Trainer 構築より前に `nn.Parameter` としてモデルに登録されていなければ optimizer に拾われない**。scale は 1-D なので自動的に `no_decay` 群に入る(weight decay 0 = LSQ の意図通り、scale を decay で 0 に引っ張ってはいけない)。
4. **eval/gate**(`eval.py:243-257`): `passes_capability_gate(model_top1, reference_top1, min_retention=0.97)`。reference≤0 は「制約なし」。`held_out_top1_report` が top-1/top-5 を非重複窓で測る。**この gate と eval 関数は一切変更しない**(LSQ は学習手法であって評価基準ではない)。
5. **既存テスト**(`tests/unit/test_qat_train.py`): `_load_script()` で importlib ロード、`fake_quant_ste` の forward=量子化/backward=恒等、`FakeQuantLinear`、`convert_to_fake_quant` 全置換、`main` end-to-end、bad-args を検証済み。LSQ 追加でこれらが壊れてはならない(後方互換)。
6. **model**(`model.py`): tied `wte`/`lm_head`(同一 Parameter)、`lm_head` は `bias=False`。Linear は `c_attn`(n_embd→3n_embd)/`c_proj`/`c_fc`/`lm_head`。Embedding/LayerNorm は量子化対象外(PTQ と apples-to-apples)。

---

## 1. LSQ を qat_train.py へ足す設計

### 1.1 設計原則(CLAUDE.md「src 改変最小・既存挙動非干渉」を満たす)

- **既存 `fake_quant_ste` / `FakeQuantLinear` / `convert_to_fake_quant` は一切変更しない**(純粋追加)。LSQ は**別関数 + 別クラス + フラグ**で並走させる。
- 追加面は `scripts/qat_train.py` に閉じる。**`src/llcore/` は変更しない**(eval.py の gate も model.py も不変)。LSQ は「学習スクリプト側の量子化器差し替え」であり、ライブラリ本体の責務ではない。
- CLI に `--quantizer {fixed,lsq}`(default `fixed` = 現挙動)と `--granularity {per_channel,per_group}` + `--group-size N`(LSQ 用、後述 §2)を足す。**default を `fixed` にすることで既存 run・既存テストの挙動が完全に保たれる**。

### 1.2 学習可能 scale パラメータの持たせ方

LSQ では各量子化器に **step size s を `nn.Parameter` として保持**する。char-LM 規模・CPU・prior-art の「小モデルほど g が相対的に強まり scaling 前提が一応成立」を踏まえ、**per-channel(行ごと)を default**、**per-group をオプション**とする。

```
LSQLinear(nn.Linear):
    bits: int
    granularity: "per_channel" | "per_group"
    group_size: int            # per_group のみ
    scale: nn.Parameter        # 形状 = per_channel:(out,1) / per_group:(out, n_groups)
    initialized: bool (buffer) # 遅延初期化フラグ(下記 1.5)
```

- **per-channel**: `scale` 形状 `(out_features, 1)`。現 fixed と同じ粒度なので apples-to-apples 比較の主軸。
- **per-group**: weight `(out, in)` を in 次元で `group_size` ごとに区切り、`scale` 形状 `(out, ceil(in/group_size))`。forward では各 group の列に対応する scale を broadcast。**`in_features` が `group_size` で割り切れない端数群**は最後の群に吸収(端数列だけ別 scale)。reshape ではなく `repeat_interleave` で列方向へ scale を展開(端数対応のため reshape は避ける)。
- scale は **常に正**でなければならない(負/0 で除算が壊れる)。**`s = softplus(s_raw)` ではなく、LSQ 原論文どおり `s` を直接パラメータ化し forward 前に `s.clamp(min=1e-12)` で下限クランプ**する(softplus は勾配スケーリング g の解析式 §1.3 を歪めるため使わない)。clamp は STE 的に勾配を通す(`s + (s.clamp(min) − s).detach()` ではなく、初期化が十分正なら通常 clamp で実害はない — テストで s>0 維持を確認 §5)。

> なぜ Parameter をモデルに事前登録する必要があるか: §0-3 のとおり Trainer が `__init__` で optimizer を組むため。**`convert_to_lsq()` を `Trainer(...)` 構築より前に呼ぶ**ことを main のフロー(現 `:169` の `convert_to_fake_quant` と同位置)で保証する。

### 1.3 scale 勾配(LSQ の gradient scale factor g)

LSQ の本質は2つ:**(A) round の外側を通常微分する解析勾配**、**(B) s 勾配の均衡化 g**。

**(A) forward / 解析勾配(weights, 符号付き対称, Q_N=Q_P=qmax=2^(b−1)−1)**

```
v = w / s
v_clip = clip(v, −Q_N, +Q_P)
v_bar  = round(v_clip)          # STE: backward は恒等(round のみ pass-through)
w_hat  = v_bar * s
```

s に対する LSQ 解析勾配(原論文 Esser 2020 / ar5iv 確認済):
- clip 範囲内(`−Q_N < v < Q_P`): `∂w_hat/∂s = −v + round(v)` = **round 誤差**(= `v_bar − v_clip`)
- 下側飽和(`v ≤ −Q_N`): `∂w_hat/∂s = −Q_N`
- 上側飽和(`v ≥ Q_P`): `∂w_hat/∂s = +Q_P`

→ これは `torch.autograd.Function` を**自前定義**して `backward` で実装する(STE の round 部分と s 勾配を同時に正しく流すため)。w への勾配は範囲内 1.0 / 範囲外 0.0(STE)。

**(B) gradient scale factor g(LSQ 第2貢献)**

s の素の勾配は他パラメータより 2〜3 桁過大。相対更新比 `R = (∇_s L / s)/(‖∇_w L‖/‖w‖)` を 1 に寄せる。

```
weights:     g = 1 / sqrt(N_W * Q_P)     # N_W = 量子化器が担う重み要素数
activations: g = 1 / sqrt(N_F * Q_P)     # 本設計は weights-only なので未使用
```

- **per-channel**: 1 行(`in_features` 要素)が 1 scale を担うので `N_W = in_features`。
- **per-group**: 1 group(`group_size` 列、端数群は端数列数)が 1 scale を担うので `N_W = group_size`(端数群は端数列数)。**group が小さいほど N_W 小 → g 大 → s 更新が相対的に強い**(prior-art の「小モデル/小群で scaling 前提が一応成立」と整合)。
- **g の適用方法**: `s_grad_scaled = grad_scale(s, g)` を **forward 内で**かける。実装定石は恒等 forward・backward で grad に g を乗じる autograd トリック(`s_eff = s*g + (s − s*g).detach()` の双対 = `s + (s_scaled − s).detach()` 系)。本設計では **`_grad_scale(x, g)`: forward は x、backward は `grad*g`** という小 Function を1つ用意し、`s_eff = _grad_scale(s, g)` を量子化に使う。これで optimizer の LR は他パラメータと共通のまま、s だけ実効学習率が g 倍に絞られる。
- `Q_P` は per-channel/per-group とも同じ(2bit で 1)。`g` は **per-scale でなく per-quantizer の定数**(各 LSQLinear で `in_features`/`group_size` から一度計算しバッファ保持)。

### 1.4 STE との接続(既存 fixed との差分の核心)

| 項目 | 既存 fixed (`fake_quant_ste`) | 追加 LSQ |
|---|---|---|
| scale | 毎 forward `w.amax/qmax`(**学習対象外**) | `nn.Parameter`(**勾配で学習**) |
| round の backward | 恒等 STE | 恒等 STE(同じ) |
| s の backward | なし(s は w の関数) | **LSQ 解析勾配 + g 均衡化**(新規) |
| clip の backward | clamp 通常微分 | clamp 通常微分(範囲外で w 勾配 0)|
| optimizer 影響 | scale は param でない | **s が AdamW の no_decay 群に入る**(Trainer 不変で自動) |

→ **唯一の非自明な接続点**: LSQ の s と w を同じ autograd グラフに載せること。`LSQ_quantize(w, s_eff, Q_N, Q_P)` を1つの `autograd.Function` にまとめ、forward で `w_hat` を返し、backward で `(grad_w, grad_s)` を上記式で返す。`s_eff` は §1.3-B の `_grad_scale` を通したもの。

### 1.5 初期化(現 fixed scale を初期値に = run 間分散低減)

prior-art の LSQ `s_init = 2⟨|v|⟩/√Q_P`、LSQ+ の MSE 初期化を踏まえつつ、**llcore 固有の利点**として「現 fixed scale をそのまま初期値にできる」。

- **default 初期化 = 現 fixed と同一**: `s_init = w.abs().amax(dim=group, keepdim=True)/qmax`(per-channel なら現 `fake_quant_ste` の scale と**ビット一致**)。これにより **LSQ は学習開始時点で fixed-QAT と完全に同じ量子化**から始まり、学習で s が動いた分だけが純粋な LSQ 効果になる(差分が測定対象として綺麗)。
- **オプション初期化 `--scale-init {amax,mean,mse}`**:
  - `amax`(default): 上記、現 fixed 等価。
  - `mean`: LSQ 原論文 `s_init = 2*mean(|w|)/sqrt(Q_P)`。
  - `mse`: LSQ+ 流。数バッチ(or 重み一括)で `‖ŵ−w‖²` を最小化する s を 1-D グリッド or 黄金分割探索で求める(weights-only なので活性バッチ不要、重みから直接でよい=安価)。
- **遅延初期化フラグ `initialized`(buffer)**: 重みが乱数 init 直後(現 main は乱数初期化から学習, `:165-168`)なので、`convert_to_lsq` 時点の重みで s_init を計算してよい。ただし fp32 checkpoint からの fine-tune-QAT に将来拡張する場合に備え、最初の forward で 1 度だけ初期化する遅延パスも持てる構造にする(default は変換時に即初期化、フラグは将来用)。

### 1.6 既存 fake-quant との差分(ファイル単位)

追加するもの(すべて `scripts/qat_train.py` 内、既存関数の**下**に追記):
1. `class _LSQQuantize(torch.autograd.Function)` — forward/backward(§1.3-A, §1.4)。
2. `def _grad_scale(x, g)` — s 勾配均衡化(§1.3-B)。小 Function or `x*g + (x − x*g).detach()` のワンライナー。
3. `def lsq_quant(w, scale, bits, granularity, group_size)` — 量子化器(§1.2 の broadcast/group 処理)。
4. `class LSQLinear(nn.Linear)` — `scale` Parameter 保持、forward で `lsq_quant` 使用。
5. `def convert_to_lsq(model, bits, granularity, group_size, scale_init)` — `nn.Linear`→`LSQLinear` 置換 + s_init 計算(既存 `convert_to_fake_quant` を真似た構造)。
6. `main` の改修(**最小**): `--quantizer/--granularity/--group-size/--scale-init` を argparse 追加。`if args.quantizer == "lsq": convert_to_lsq(...) else: convert_to_fake_quant(...)`。出力 JSON に `quantizer`/`granularity`/`group_size`/`scale_init` と、**学習後の s 統計(min/mean/max、init からの相対変化)**を追記(§3 の「s が動いたか」を記事/検証で使う)。

---

## 2. 実験計画

### 2.1 比較マトリクス(multi_smoke / realp1, 2bit 主軸)

| 軸 | 値 |
|---|---|
| モデル | multi_smoke(1.36M, vocab4358, smoke preset) / realp1(11.9M, vocab3044, p1 preset) |
| bits | **2(主)**、3(対照: 既に PTQ 安全床なので LSQ 余地確認) |
| quantizer | `fixed`(= 現 QAT, baseline)/ `lsq` |
| granularity(lsq) | `per_channel` / `per_group`(group_size ∈ {64, 32}) |
| scale-init(lsq) | `amax`(主)、補助で `mse` を per_channel 2bit のみ |
| 共通 | 同 corpus / 同 preset / **同 max-iters(2000、fixed と同一)** / 同 seed(1337) / fp32 checkpoint = 既存 |

**測る指標**(既存 eval をそのまま): held-out PPL、unigram PPL、**top1 retention(fp32 比)**、ppl-gate、**cap-gate(97%)**。fixed QAT との **Δtop1 retention(pp)** を主アウトカム。

### 2.2 run セル(最小本数で結論が出る順)

1. **`fixed × per_channel × 2bit`(再現 baseline)** = 既存 82.9%(multi_smoke)を再走で再現確認(seed 固定で同値が出るべき)。
2. **`lsq × per_channel × 2bit × amax-init`** = LSQ の純効果。amax-init は学習開始が fixed と同一なので「s 学習だけの寄与」が出る。**最重要セル**。
3. **`lsq × per_group(g64, g32) × 2bit`** = 粒度×LSQ の相乗。prior-art「per-group が床を下げる」+ LSQ の合算が cap-gate にどこまで迫るか。
4. **`lsq × per_channel × 2bit × mse-init`** = 初期化感度(run 間分散低減の確認、LSQ+ 主張の char-LM 追試)。
5. **realp1 で 2/3 を反復** = 大モデルで LSQ がより効くか(prior-art の scaling law と整合するか)。realp1 は 2bit fixed QAT 値が doc に無いので、**realp1 fixed QAT 2bit も本計画で初測**(比較基準として必須)。
6. **対照 3bit(per_channel, lsq vs fixed)** = 既に PTQ で安全床の 3bit に LSQ が「無害(retention 維持)」かを確認(LSQ が高ビットで壊さないことの sanity)。

### 2.3 cap-gate 判定

各セルで `passes_capability_gate(qat_top1, fp32_ref_top1, 0.97)`。**主結論 = 「LSQ は 2bit で cap-gate を越えるか / 越えないなら 82.9% を何%まで押し上げたか」**。出力は既存 `out/qat_train.json` 命名を `out/qat_lsq_<model>_<bits>bit_<gran>.json` に分けて保存(既存 fixed の JSON を上書きしない)。

### 2.4 honest 留保(計画に明記)

- CPU smoke / weights-only / Linear のみ / simulated quant(速度未測)= 既存と同条件で apples-to-apples。
- LSQ vs fixed は「同じ学習予算(2000 iters)で 2bit deploy するならどちらが良いか」。iters を増やせば両者改善しうるので **iters 固定**が公平条件。
- 第1層/最終層 8bit 残し(LSQ 慣行)は **本計画では採らない**(現 llcore QAT が全 Linear 同一 bit なので踏襲、apples-to-apples 優先)。これは「retention をさらに上げる余地」として記事/次手に回す。

---

## 3. honest 見積もり(最重要)

### 3.1 結論先出し: **LSQ 単独で 97% cap-gate を越える可能性は低い。**

prior-art の一次情報が**複数の独立した scaling law で同方向**を示しており、char-LM(1.36M–11.9M)規模では 2bit の 97% retention は構造的に厳しい。具体的根拠:

1. **LSQ 自身のデータが小モデルの脆弱性を示す**: LSQ 原論文(arXiv:1902.08153)で、パラメータ効率を極めた小型 SqueezeNext-23-2x は **2bit で top-1 が 14.0pt 低下**(ResNet-18 は 2.9pt)。論文自身が「few parameters に最適化された設計点は precision 低下に極端に敏感」と honest に記述。llcore の char-LM(1.36M)は SqueezeNext よりさらに小さく**冗長性がほぼ無い** → LSQ をもってしても 2bit の量子化ノイズを吸収する余地が乏しい。
2. **複数の量子化 scaling law**:
   - QiD = k·D^0.525 / (N^0.226·P^5.50)(arXiv:2411.17691): N(パラメータ数)指数 0.226 が小さく、小モデルは量子化耐性の「保護」がほぼ無い。1.36M/11.9M は 160M/1B より QiD が深刻。
   - QAT scaling law(arXiv:2505.14302): 量子化誤差はモデルが小さいほど増加。
   - Dettmers k-bit(arXiv:2212.09720): 3bit 以下で bit-level scaling が崩れ、**4bit が実用下限**。2bit はハード支援も無いと CPU で速度も出ない。
3. **EfficientQAT の retention は「大モデル前提」**: W2g64 で Llama-2-7B 92.7% / 70B 95.9% retention だが、これは **>=7B + group64 + QAT + fine-tune** が揃っての値で、**7B でも 97% には届いていない**。1.36M で 97% は楽観できない。
4. **llcore 既存の自前データと整合**: doc (b')(b'')(b''')で「**cliff 位置はモデルサイズ依存・大モデルほど低ビット頑健**」「2bit は RTN/per-group/GPTQ いずれも cap-gate 未達」「QAT(固定 scale)が 82.9% まで押し上げたが 97% 未達」を既に実測。LSQ は「固定 scale → 学習 scale」の改善であって、上の scaling law の壁(根本的情報損失)を消すものではない。

### 3.2 では LSQ に何を期待するか(成果の測り方 = 「越えられなくても詰めた量」)

prior-art の LSQ→LSQ+ の典型ゲインは **W2A2 で +2.9〜+5.6pt(EfficientNet/MixNet, ただし大モデル・QAT)**。fixed→LSQ の改善はこのオーダーが上限の目安。llcore の現状 82.9%(multi_smoke 2bit)を起点に、**正直な予想レンジ**:

| シナリオ | multi_smoke 2bit retention(現 82.9%) | cap-gate(97%) |
|---|---|---|
| 悲観(scaling law 支配) | 83〜86% | 未達 |
| 中央(LSQ 典型ゲイン相当) | **86〜90%** | 未達 |
| 楽観(LSQ + per-group32 相乗が効く) | 90〜93% | おそらく未達 |
| realp1(11.9M, より頑健) | LSQ + per-group で **90%台前半**もありうる | ボーダー、越えれば大ニュース |

→ **設計上の成果定義**: 「97% を越えたか」を二値で終わらせず、**(a) fixed QAT 比 Δretention(pp)**、**(b) 82.9% → 何%**、**(c) per-group との相乗の有無**、**(d) s が学習でどれだけ動いたか(動かなければ LSQ が機能していない=null)** の4点を測る。**越えられなくても「LSQ がどこまで詰めたか」が一次データとして価値**(feedback_benchmark_honest_disclosure / 北極星 pivot の作法)。

### 3.3 越えられない場合の正しい結論文(先に用意)

> 「2bit char-LM の 97% cap-gate は、学習可能 scale(LSQ)+ per-group をもってしても本モデル規模では未達。これは複数の独立した quantization scaling law(QiD の N^0.226 / Dettmers 4bit 下限 / LSQ 自身の小モデル脆弱性データ)が予言するとおりで、**床を動かす残された梃子は『質的に別の手段』 — モデル大型化 / VQ codebook(scalar quant の限界を超える)/ fine-tune 誤差回復 / salient 次元の mixed-precision(第1・最終層 8bit 残し)**。**3bit が PTQ/QAT 共通の実用床**という llcore の到達点は LSQ 後も不変。ただし LSQ は固定 scale QAT を **+N pp** 押し上げ、『学習する量子化器』が小モデルでも有意に効くことを実証した。」

---

## 4. 記事接続(成功でも失敗でも honest に)

`ARTICLE_SEEDS.md` は append-only(date 見出し `## YYYY-MM-DD` + `###` 配下に `**気付き**`/`**根拠**`/`**側面**`)。本実験完了時、以下を seed 追記する(collector の最小受理条件 = `気付き` or `側面` 同一行非空、+ producer 契約で `根拠` 必須)。

### 4.1 主接続先

- **B1「負けを見せる」(retention < 97% の場合)**: 「LSQ + per-group をもってしても 2bit char-LM の 97% gate は越えられなかった。これは scaling law の予言どおり。だが固定 scale QAT を +N pp 押し上げ、82.9%→X% へ。『床を動かすには質的に別手段(大型化/VQ/mixed-precision)が要る』を自前実測で確定。」honest disclosure の白眉。
- **A3「量子化再導出」**: LSQ の核(s 勾配 = round 誤差 + clip 飽和 / g = 1/√(N·Q_P) 均衡化)を char-LM で自前実装し**再導出・実測**。「固定 scale(w から算出)→ 学習 scale(task loss の勾配で動く)」の差を s 統計の before/after で可視化。fixed と amax-init が学習開始時点でビット一致 → s がどれだけ動いたかが純効果、という測定設計の綺麗さも記事になる。

### 4.2 効く 13 側面

honest disclosure(主)/ ベンチ(fixed vs LSQ vs per-group)/ 業界比較(LSQ・LSQ+・EfficientQAT を char-LM で追試)/ 技術設計(autograd Function での解析勾配)/ 教訓(小モデルほど低ビット不利の scaling law を自前データで裏取り)/ TRIZ(制約=tiny model を「LSQ がどこまで詰められるか」の純粋計測へ反転)。

### 4.3 doc 正本更新

`MEMORY_EFFICIENCY_FINDINGS.md` (d) に **(d') LSQ 学習可能 scale** 節を追記(既存 (d) は不変、append)。表は §2.1 マトリクスの実測値 + honest 留保。

---

## 5. TDD テスト項目(`tests/unit/test_qat_lsq.py`)

既存 `test_qat_train.py` の `_load_script()` 規約(importlib ロード)を踏襲。**実装前にこれらを赤で書く**。

### 5.1 量子化器 round-trip / 数値正しさ

1. **`test_lsq_quant_forward_is_quantize_dequantize`**: 既知の `w`, `scale`, bits で `lsq_quant` 出力が `clip(round(w/s), −Q_N, Q_P)*s` と一致(per_channel)。
2. **`test_lsq_quant_per_group_shapes_and_endpoints`**: per_group(group_size=4, in_features=10 など**割り切れない端数**)で scale 形状が `(out, ceil(in/gs))`、端数群の列が正しい scale で量子化される(broadcast/端数吸収のバグ検出)。
3. **`test_lsq_2bit_levels`**: bits=2 で出力が高々 3 値 {−s, 0, +s}(per scale)に量子化される(qmax=1 の確認)。

### 5.2 勾配(LSQ の核 = scale が学習で動く)

4. **`test_lsq_scale_has_gradient`**: `LSQLinear` forward→loss.backward で **`scale.grad is not None` かつ非ゼロ**(固定 scale には無い性質 = LSQ の存在証明)。
5. **`test_lsq_scale_grad_matches_analytic`**: clip 範囲内サンプルで s 勾配が解析式 `(round(v) − v)` 系に一致(数値微分 `torch.autograd.gradcheck` を double 精度で、または有限差分で照合)。**範囲外サンプルで −Q_N/+Q_P に飽和**することも別 case で確認。
6. **`test_grad_scale_factor_applied`**: `_grad_scale(s, g)` で s の grad が g 倍に縮む(per_channel は g=1/√(in·Q_P)、per_group は g=1/√(group_size·Q_P))。g の値が `in_features`/`group_size` から正しく算出されているか。
7. **`test_weight_ste_identity_inside_clip`**: clip 範囲内で w 勾配が恒等(STE)、範囲外で 0(既存 fixed と同じ STE 挙動が LSQ でも保たれる)。

### 5.3 統合 / 後方互換 / 学習方向

8. **`test_convert_to_lsq_replaces_linears_and_registers_scale`**: `convert_to_lsq` 後、全 `nn.Linear` が `LSQLinear` になり、各々 `scale` が `nn.Parameter`(requires_grad=True)で `model.parameters()` に**現れる**(= Trainer が optimizer に拾える保証)。tied `wte`/`lm_head` を二重量子化しない(同一 Parameter は 1 度)。
9. **`test_scale_init_amax_matches_fixed`**: `scale-init=amax` の初期 scale が現 `fake_quant_ste` の scale と**ビット一致**(LSQ が fixed と同じ点から学習を始める設計の保証)。
10. **`test_scale_moves_during_training`**: 数十 iters の極小学習で `scale` が初期値から**有意に変化**(`|s_after − s_init| > 0`)。動かなければ LSQ が機能していない null の検出。
11. **`test_main_lsq_end_to_end_short`**: `main(["--quantizer","lsq","--granularity","per_channel", ... ,"--max-iters","2", ...])` が rc=0、JSON に `quantizer=="lsq"` と s 統計フィールドが入る。
12. **`test_default_quantizer_is_fixed_backward_compat`**: `--quantizer` 省略時に**既存 fixed 経路**が走り、既存 `test_qat_train.py` の挙動・出力 schema が不変(後方互換)。
13. **`test_lsq_retention_not_worse_than_fixed_smoke`**(方向性テスト, 緩い): 同 seed・同短 iters の極小 smoke で `lsq per_channel` の top1 が `fixed` を**下回らない**(改善方向の sanity; 厳密 retention 値は実験で測るのでここは「悪化しない」だけを緩く保証。flaky 回避のため tolerance を持たせる or `>=` の弱主張に留める)。

### 5.4 テスト設計の honest 留保

- 5.2-5 の `gradcheck` は double 精度・clip 内サンプルに限定(round の不連続点を避ける)。
- 5.3-13 は学習の確率性で flaky になりうるので **seed 固定 + 弱主張(悪化しない)** に留め、厳密な retention 改善は §2 実験(非テスト)で測る。テストは「LSQ が機構として正しく動く」ことの保証に集中し、「LSQ が gate を越える」ことはテストで主張しない(越えない可能性が高いため、テストに入れると常に赤)。

---

## 関連ファイル(絶対パス)

- 改修対象(追加のみ): `D:\projects\llcore\scripts\qat_train.py`
- 新規テスト: `D:\projects\llcore\tests\unit\test_qat_lsq.py`(既存 `D:\projects\llcore\tests\unit\test_qat_train.py` は不変)
- 不変(参照のみ): `D:\projects\llcore\src\llcore\lm\eval.py`(cap-gate)、`D:\projects\llcore\src\llcore\lm\trainer.py`(optimizer 構築タイミングが LSQ 統合の鍵)、`D:\projects\llcore\src\llcore\lm\model.py`
- 正本更新(append): `D:\projects\llcore\docs\MEMORY_EFFICIENCY_FINDINGS.md`((d') 節)、`D:\projects\llcore\docs\ARTICLE_SEEDS.md`(seed 追記)
- 出力: `D:\projects\llcore\out\qat_lsq_<model>_<bits>bit_<gran>.json`(既存 `out\qat_train*.json` を上書きしない)

## 設計上の load-bearing な1点(実装時の最大の落とし穴)

`Trainer.__init__`(trainer.py:54-79)が `model.parameters()` から optimizer を組むため、**`convert_to_lsq()` は必ず `Trainer(model, ...)` 構築より前に呼ぶ**こと(main の現 `convert_to_fake_quant` と同じ `:169` 位置)。順序が逆だと scale Parameter が AdamW に登録されず、**s の勾配は計算されるが更新されない**(= LSQ が静かに fixed-QAT に退化し、テスト 5.3-10 が落ちる)。scale は 1-D なので Trainer の decay/no_decay 振り分け(`p.dim() >= 2`)で自動的に no_decay 群(weight_decay=0)に入り、これは LSQ の意図と一致する。
