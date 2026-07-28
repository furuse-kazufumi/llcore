# PoC K3: Kimi K3 の有界化アイデアを llcore に転用する

- 発足: 2026-07-28
- 一次情報: Kimi K3 技術レポート (`MoonshotAI/Kimi-K3` の `k3_tech_report.pdf`、47 頁) と
  Hugging Face `moonshotai/Kimi-K3` の `config.json` / `modeling_kimi_linear.py`
- 実装状態: **PoC (additive option、既定 off で回帰ゼロ)**

> 注意: 技術レポート PDF は **HF リポジトリには無く GitHub 側**にある。
> 「HF repo に含まれる」とする二次情報は誤り。

---

## 0. 何を転用し、何を転用しなかったか (動機の差し替え)

K3 §2.1.1 の lower-bounded decay は、**chunkwise linear attention の累積減衰の逆数
`1/Γ` が BF16 のダイナミックレンジを超える**という数値問題への対策である。
Kimi Linear は log 空間 + 16 トークンタイル + 対角タイルの position-pair 計算で
回避していたが、K3 は減衰のパラメータ化自体を変えて解いた:

```
Kimi Linear: g = −e^A · Softplus(z)        ∈ (−∞, 0)   ← 下に非有界
Kimi K3:     g = g_min · Sigmoid(e^A · z)  ∈ (g_min, 0) ← 下限を固定 (g_min = −5)
```

**llcore にこの数値的動機はそのままでは転用できない。** llcore の RWKV-4
(`src/llcore/lm/rwkv.py`) は `q2 = max(p + decay, k)` の running-max
(log-sum-exp) で既に数値安定であり、chunkwise 形式の逆数リスケールを持たない。

転用したのは**別の性質**である:

| | 従来 `-exp(w)` | 有界 `g_min·sigmoid(w)` |
|---|---|---|
| log-decay の値域 | `(−∞, 0)` | `[g_min, 0]` (実数では開区間、float では端点到達) |
| 保持率 `α = exp(g)` | `(0, 1)` — **下限なし** | `[e^{g_min}, 1]` — **保持床あり** |
| チャネル死 (`α→0` = 1 ステップ全忘却) | 設計上排除できない | **構造的に不可能** |
| 「これ以上速くは忘れない」の証明 | 不可能 | **Z3 で可能** |

llcore の北極星はメモリ効率＝保持能力であり、Verified-Plasticity は
「実行時に祈らず設計で閉じる」路線なので、**転用すべきはこちら**という判断。

---

## 1. 実装 (すべて additive、既定は従来挙動)

### 1.1 有界 log-decay — `src/llcore/lm/rwkv.py`

- `bounded_log_decay(raw, log_floor)` = `log_floor * sigmoid(raw)`
- `init_raw_for_log_decay(target, log_floor)` — パラメータ化を跨いで**初期保持率を
  揃える逆写像**(下記 3.1 の統制に必須)
- `RWKVConfig.decay_log_floor: float | None = None` — **None で従来の `-exp(w)`**
- `RWKVTimeMix.log_decay()` に分岐を集約

### 1.2 SiTU-GLU — `src/llcore/lm/activations.py`

K3 §2.3.2 (Eq. 12):

```
SiTU-GLU(x) = [β1·tanh(Wg x / β1) ⊙ Sigmoid(Wg x)] ⊙ [β2·tanh(Wu x / β2)]
              → |出力| <= β1·β2   (K3 は β1=4, β2=25 で上界 100)
```

- `softcap(x, beta)` = `beta * tanh(x / beta)`
- `situ_glu(gate_pre, up_pre, beta_gate, beta_up)`
- `RWKVConfig.ffn_activation: "sq_relu" | "situ_glu"` — **既定 `sq_relu` で従来通り**

llcore にとっての意味: RWKV の channel-mix は `value(relu(key(x))**2)` =
**squared-ReLU で非有界かつ二次**であり、低ビット化の際に最も危険な形。
`ffn_hidden()` を検査点として切り出し、有界性をテストで固定した。

### 1.3 Z3 検証 — `src/llcore/verifier/invariants.py`

| 関数 | 証明する命題 |
|---|---|
| `verify_retention_floor(log_floor, steps, parameterization)` | `∀ raw. Σ_{i=1..T} g_i > T·log_floor ∧ ∀i. g_i < 0` |
| `verify_activation_bound(beta_gate, beta_up, claimed_bound)` | `∀ g,u. \|SiTU-GLU(g,u)\| <= claimed_bound` |

**Z3 encoding (sound abstraction)**: `sigmoid` → `σ ∈ (0,1)`、`exp` → `e > 0`、
`tanh` → `t ∈ (−1,1)` の自由変数に置換する。いずれも**値域と厳密に一致**するので
緩めても狭めてもいない。保持床は log 空間で線形になるため Z3 の判定は厳密。

**結果**:

| 対象 | 判定 |
|---|---|
| 有界版の 1 step 保持床 | **unsat** (床を破れない = 証明成立) |
| 有界版の 8 step 累積保持床 | **unsat** |
| 従来 `-exp(w)` の保持床 | **sat** (反例あり = 床を保証できない) |
| SiTU-GLU の出力上界 `β1·β2` | **unsat** |
| SiTU-GLU に積より小さい上界を主張 | **sat** (soundness クロスチェック) |

---

## 2. ★ 証明と実装の差 (honest 留保)

**Z3 が証明するのは実数上の開区間、float 実装は端点に到達する。**

`sigmoid(1e4)` は float32 で `1.0` に飽和するため `g == log_floor` ちょうどが出る。
同様に `sigmoid(-1e4) == 0.0` で `g == 0` (減衰ゼロ) になる。

- 保持床 `α >= exp(log_floor)` **自体は等号成立を含めて保たれる**ので不変量は無事
- しかし「開区間である」という Z3 の結論を **float 実装の記述として引用してはならない**
- この事実は `tests/unit/test_poc_k3_bounded_decay.py::test_g1c_float_saturation_attains_the_endpoints`
  で回帰として固定した

`verify_activation_bound` の abstraction にはもう一段の留保がある: `tanh(g/β)` と
`Sigmoid(g)` は同じ `g` の関数なので独立ではないが、独立変数として扱っている。
これは **over-approximation** なので `unsat ⟹ 真に有界` は保たれる (`sat` 側が
保守的になる = fail-closed 側に倒れる)。

---

## 3. 経験的評価

### 3.1 最初の実験は無効だった (設計欠陥と修正)

初回比較は **統制されていなかった**:

- `time_decay` は両 arm とも zeros 初期化だが、パラメータ化が違うため
  **初期保持率が違った** — baseline `-exp(0) = -1` → α=0.368、
  有界版 `-5·sigmoid(0) = -2.5` → α=0.082。**別の記憶レンジから出発していた**
- 300 iters では decay がほぼ動かなかった (α_median の移動量 −0.0089 / +0.0007)
  → 測っていたのは実質**初期値**であって学習後の挙動ではない

したがって「baseline で `n_below_floor = 0` だった」ことは
**チャネル死が起きない証拠にはならなかった** (パラメータが動いていないので当然)。

修正: `init_raw_for_log_decay` で両 arm の初期 log-decay を `-1.0` (α=0.3679) に
揃え、2000 iters に伸ばした。

### 3.2 統制後の結果 (Tiny Shakespeare 200k 文字 / n_layer=2 / n_embd=64 / 2000 iters)

<!-- RESULTS_TABLE_START -->
両 arm とも初期 log-decay を `-1.0` (α=0.3679) に統制。`g_min = -5` → 床 `α >= 0.006738`。

| seed | baseline val | bounded val | delta | baseline α_min | bounded α_min | baseline 床割れ |
|---|---|---|---|---|---|---|
| 1337 | 1.7613 | 1.7616 | +0.0003 | 0.2752 | 0.2944 | 0/128 |
| 7 | 1.7302 | 1.7311 | +0.0009 | 0.2648 | 0.2839 | 0/128 |
| 99 | 1.7785 | 1.7788 | +0.0003 | 0.2887 | 0.3050 | 0/128 |

- baseline val: mean **1.7567** (sd 0.0200、seed 間の広がり **0.0483**)
- bounded val: mean **1.7572** (sd 0.0197)
- delta: mean **+0.00050**、max |delta| **0.00092** = **seed 間の広がりの 1.9%**
- baseline の最小 α = **0.2648** = 床 `e^{-5}=0.006738` の **39 倍**上
<!-- RESULTS_TABLE_END -->

### 3.2b 床を拘束的にした場合 (`g_min = -1.2` → 床 α >= 0.3012)

§3.2 の `g_min = -5` は床が観測 α_min の 1/39 で**何も制約していなかった**ので、
学習後の α_min ≈ 0.265 を**上回る**床を設定して「保持床を強制すると何が起きるか」を測った。

| seed | baseline val | bounded val | delta | baseline α_min | bounded α_min | **baseline が床を割ったチャネル** |
|---|---|---|---|---|---|---|
| 1337 | 1.7613 | 1.7622 | +0.0009 | 0.2752 | 0.3530 | **6/128** |
| 7 | 1.7302 | 1.7318 | +0.0016 | 0.2648 | 0.3498 | **21/128** |
| 99 | 1.7785 | 1.7797 | +0.0011 | 0.2887 | 0.3550 | **5/128** |

- **床は実際に拘束した**: baseline は 5〜21 / 128 チャネルが床より速く忘れることを選んでおり、
  有界版ではそれが 0 になって α_min が 0.35 まで押し上げられている
- **拘束のコスト**: delta mean **+0.00119** (非拘束時 +0.00050 の約 2.4 倍) だが、
  それでも **seed 間の広がり (0.0483) の 3.2%**、絶対値では val loss の **0.07%**
- 両実験とも 3/3 seed で delta が正 (有界版が僅かに悪い)。n=3 で符号が揃うのは
  それ自体では有意ではない (符号がランダムなら 1/8 = 0.125) が、**床の拘束度を上げると
  delta も大きくなった**のは用量反応の可能性がある。**仮説であって結論ではない** —
  検証には床レベルを複数点 × seed 数を増やす必要がある

### 3.2c SiTU-GLU vs squared-ReLU — **交絡していた初回と、統制後で結論が反転**

`--arms ffn` の初回比較は交絡していた。**SiTU-GLU は gate/up の 2 枝が要るため
`up` 射影が増え、同じ hidden 倍率だと channel-mix の FFN パラメータが 1.5 倍**
(d=64 で 49,728 vs 33,088) になる。つまり測っていたのは「活性化の差」ではなく
**「容量の差」**だった。

`--match-ffn-params` で 2/3 則 (hidden を 8d/3) を適用し総パラメータを揃えて再測定:

| | delta mean | max\|delta\| | 符号 (3 seed) | 判定 |
|---|---|---|---|---|
| 未統制 (situ_glu が 1.5x params) | **−0.00266** | 0.01074 (広がりの 22%) | −, +, − | 混在 = ノイズ支配 |
| **パラメータ統制済み** | **+0.01875** | 0.03719 (広がりの **77%**) | **+, +, +** | **一貫して悪化** |

- 統制すると **SiTU-GLU は 3/3 seed で悪化**し、効果量は保持床の実験 (2〜3%) と比べ
  桁違いに大きい (77%)。未統制版が中立〜やや有利に見えたのは**余分なパラメータのおかげ**
- 速度も **+22%** (207s → 253s、パラメータを揃えた後でも tanh 2 回分)
- パラメータ統制 (params=117,376 vs 117,676) と幅統制は同時には成立しない。
  ここではパラメータを揃え、**hidden 幅が 256 → 170 に狭まる**副作用は受け入れている

> **★ ただしこの比較でも「有界化のコスト」は分離できていない。**
> squared-ReLU → SiTU-GLU は「squared-ReLU → GLU 構造」と「非有界 → 有界」の
> **2 つの変更が同時に入っている**。有界化だけを切り分けるには、同一構造・同一
> パラメータ数の **SwiGLU (非有界 GLU) vs SiTU-GLU (有界 GLU)** を比べる必要がある。
> → `--ffn-baseline swiglu` で切り分けた結果が §3.2d。**コストの 88% は構造変更由来**だった。

### 3.2d 有界化だけの ablation (SwiGLU vs SiTU-GLU、同一構造・同一パラメータ)

<!-- BOUNDEDNESS_ABLATION_START -->
SwiGLU は SiTU-GLU と**同一構造・同一パラメータ数** (117,676 = 117,676) で、
違いは softcap を掛けるかどうかだけ。差が純粋に「有界化のコスト」になる。

| seed | swiglu | situ_glu | delta | 秒 (sw/si) |
|---|---|---|---|---|
| 1337 | 1.7655 | 1.7679 | +0.0024 | 235/253 |
| 7 | 1.7638 | 1.7674 | +0.0036 | 232/253 |
| 99 | 1.7901 | 1.7910 | +0.0009 | 232/253 |

- delta mean **+0.00230**、max **0.00363** = **seed 間の広がりの 7.5%**、符号 3/3 正

**これで §3.2c の +0.01875 を分解できる**:

| 変更 | delta (mean) | 内訳 |
|---|---|---|
| 構造変更 `sq_relu → swiglu` (パラメータ統制済み) | **+0.01645** | **88%** |
| 有界化 `swiglu → situ_glu` | **+0.00230** | **12%** |
| 合計 `sq_relu → situ_glu` | +0.01875 | (実測と一致) |

> **★ 帰属の訂正**: §3.2c を見て「有界 FFN は高い」と結論するのは誤り。
> コストの **88% は squared-ReLU → GLU という構造変更**由来であり、
> **K3 の有界化そのものは 12% (広がりの 7.5%) しか払っていない**。
>
> 残る交絡: パラメータを揃えるために GLU 側は hidden が 256 → 170 に狭まっている。
> 「構造変更のコスト」には「hidden が狭くなったコスト」が混ざっている
> (パラメータ統制と幅統制は同時に成立しない)。RWKV-4 の 4x squared-ReLU が
> この規模では単に強いベースラインである可能性もある。
<!-- BOUNDEDNESS_ABLATION_END -->

### 3.2e 構造を変えずに有界化する (`ffn_softcap`) — **β の選び方で結果が自明になる罠**

§3.2d で「コストの 88% は GLU 構造変更由来」と分かったので、**構造を一切変えずに
値域だけ閉じる** knob を足した: `RWKVConfig.ffn_softcap` は活性化の種類に依らず
`softcap(hidden, β)` を掛ける (パラメータ数・構造ともに不変)。

最初に β=100 (K3 の `β1·β2` と同じ) で測った結果は delta mean **−0.00003**
(広がりの 0.2%、符号 −,−,+ = 混在) で「完全に無料」に見えた。

**しかしこれは §3.2 の非拘束床と同じ罠だった。**
`probe_ffn_activation_range.py` で学習後の hidden 活性を実測すると:

| 統計量 | 値 |
|---|---|
| abs_max | **23.90** |
| p99 | 3.71 |
| p99.99 | 11.63 |
| mean | 0.296 |

| β | 影響を受ける活性 | 判定 |
|---|---|---|
| 100 | **0.000000%** | **非拘束 (一度も発火しない)** |
| 25 | 0.000000% | 非拘束 |
| 10 | 0.020% | 拘束する |
| 5 | 0.400% | 拘束する |
| 2 | 3.765% | 拘束する |

**β=100 の「コスト 0.2%」は自明に成立していただけで、有界化のコストについて
何も言っていない。** 拘束的な β=5 / β=2 で測り直した結果が下記。

<!-- SOFTCAP_BINDING_START -->
両 arm とも **同一構造・同一パラメータ数 (117,376)** で、違いは softcap の有無だけ。

| β | 発火率 (活性の何 % に効くか) | delta mean | max\|delta\| | 広がり比 | 符号 (3 seed) |
|---|---|---|---|---|---|
| 100 | **0.000%** | −0.00003 | 0.00010 | **0.2%** | −,−,+ (混在=ノイズ) |
| 5 | 0.400% | +0.00241 | 0.00544 | 11.3% | −,+,+ |
| 2 | 3.765% | +0.01335 | 0.01912 | 39.6% | +,+,+ |

**用量反応が明確に出た**: 境界が噛む量に比例してコストが増える。
発火率 0% → コスト 0、0.4% → 広がりの 11%、3.8% → 40%。
β=5 は符号が 1 つ負なので証拠としては β=2 より弱い。

これは §3.2b の保持床 (非拘束 +0.00050 → 拘束 +0.00119) で「仮説」に留めた
用量反応を、3 点で裏付ける結果でもある。
<!-- SOFTCAP_BINDING_END -->

> **方法論の教訓 (2 回目)**: 有界化の実験では必ず
> **「その境界が実際に発火しているか」を先に測る**こと。保持床では α_min、
> FFN では hidden の abs_max。発火していない境界のコストがゼロなのは当たり前で、
> 測定として無意味になる。

### 3.3 結論 (honest)

#### 全 ablation のまとめ (delta = 有界版 − 非有界版、各 3 seed)

| 実験 | 境界は実際に噛んだか | delta (mean) | 広がり比 |
|---|---|---|---|
| 保持床 `g_min=-5` | **0/128 ch (非拘束)** | +0.00050 | 1.9% |
| 保持床 `g_min=-1.2` | 5〜21/128 ch | +0.00119 | 3.2% |
| FFN softcap β=100 | **0.000% (非拘束)** | −0.00003 | 0.2% |
| FFN softcap β=5 | 0.400% の活性 | +0.00241 | 11.3% |
| FFN softcap β=2 | 3.765% の活性 | +0.01335 | 39.6% |
| GLU の有界化 (`swiglu→situ_glu`) | 常時 | +0.00230 | 7.5% |
| 参考: 構造変更 (`sq_relu→swiglu`) | — | +0.01645 | 69.5% |

**→ 有界化のコストは「境界がどれだけ噛むか」でほぼ決まる。**
噛まない境界はタダ (が、測定としては無意味)。噛ませるほど高くなる。
一方 squared-ReLU から GLU への構造変更は、有界化とは無関係に単独で最も高い。

1. **チャネル死 (α→0) は観測されなかった** — K3 式を導入する動機として挙げた失敗モードは、
   当スケール・当予算では**起きない**。`g_min = -5` の床は観測 α_min の 1/39 で完全に非拘束
2. **保持床を「実際に効く高さ」に設定しても、品質コストはほぼ無い** — `g_min = -1.2` は
   5〜21/128 チャネルを強制的に押し上げるが、val loss の劣化は **seed 間分散の 3.2%**、
   絶対値で **0.07%**
3. したがって本 PoC の実用的な含意は
   **「証明可能な保持床は、ほぼ無料で買える」** — 性能を上げる手段ではないが、
   Verified-Plasticity の要件 (実行時に祈らず設計で閉じる) を**性能を犠牲にせず**
   満たせる。llcore が「保持能力を保証する」と言いたい場面で使える
4. **RWKV に有界 FFN が欲しいなら GLU 化は不要**。`ffn_softcap` なら
   構造もパラメータ数も変えずに値域を閉じられ、コストは β の選び方で決まる。
   SiTU-GLU を採ると GLU 構造変更 (広がりの 69.5%) が付いてくるので割に合わない
5. **★ β / g_min は論文の定数を持ち込まず、実測分布から決めること**。
   K3 の `β1·β2 = 100` は llcore 規模の実活性 (abs_max 23.9 / p99.99 11.6) の
   4 倍上で**一度も発火しない**。設計則としては
   「**実測の p99.99 の数倍**に置き、発火率を確認してから採用する」が正しい
6. 用量反応は 3 点 (0% / 0.4% / 3.8%) で単調に出たが、**n=3 seed** であり
   β=5 は符号が 1 つ負。定量的な外挿には seed 数を増やす必要がある

### 3.4 次にやるべきこと

1. ~~床を拘束的にして測る~~ → **実施済み (§3.2b)**。次は床レベルを複数点
   (`g_min ∈ {-3, -2, -1.5, -1.2, -1.0}`) × seed 数を増やして**用量反応があるか**を検定する
2. **長文脈タスクで測る** — 保持床の意味は次トークン予測ではなく long-range retention に
   出るはず。`streaming_nll` を使った長文脈評価が適切
3. ~~SiTU-GLU の経験比較~~ / ~~softcap 版~~ → **実施済み (§3.2c〜§3.2e)**。
   結論は「`ffn_softcap` を実測分布から決めた β で使う」。次は
   **β=10 (発火率 0.02%) を測って用量反応曲線の低い側を埋める**
4. **低ビット化との組合せ** — 有界化の本来の利得は量子化時に出る。`lm/quant.py` と
   組み合わせた測定が本命

---

## 4. 再現手順

```powershell
# 単体テスト (Z3 証明含む)
C:\dev\projects\llcore\.venv\Scripts\python.exe -m pytest `
  tests/unit/test_poc_k3_bounded_decay.py `
  tests/unit/test_poc_k3_situ_glu.py `
  tests/unit/test_poc_k3_situ_glu_channelmix.py -q

# 統制済み比較 (1 seed あたり約 7 分)。--arms で何を比較するか選ぶ
$py = "C:\dev\projects\llcore\.venv\Scripts\python.exe"
$cmp = "research/k3_bounded_decay/compare_retention_floor.py"

# 減衰パラメータ化 (保持床)
& $py $cmp --arms decay --log-floor -1.2 --probe-floor -1.2 --max-iters 2000 --seed 1337 --out out/a.json

# 活性化の種類 (パラメータ統制必須)
& $py $cmp --arms ffn --ffn-baseline swiglu --match-ffn-params --max-iters 2000 --seed 1337 --out out/b.json

# 構造不変の有界化 (β は下の probe で決めること)
& $py $cmp --arms softcap --ffn-softcap 5 --max-iters 2000 --seed 1337 --out out/c.json

# ★ β を決める前に必ず: 学習後の活性レンジと発火率を測る
& $py research/k3_bounded_decay/probe_ffn_activation_range.py --max-iters 2000
```

## 5. 関連

- memory `reference_kimi_k3_2026_07_28` — K3 一次確認の正本
- memory `feedback_benchmark_honest_disclosure` — 異常に良い結果は内訳を疑う
- memory `project_llcore_efficient_arch_landscape_2026_06_26` — 効率アーキ調査
