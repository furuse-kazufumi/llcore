# R-LLM-0 PRE-REGISTRATION — Verified evolution INSIDE a CPU-real tiny language model

> **Direction (user, 2026-06-04):** llcore は「Transformer コア」を標榜しながら、実コードは低次元
> dynamics の verified-evolution 枠組みに留まり、**実際の LLM/VLM として動いていない**（src に
> attention/embedding/vocab/LM-loss 無し；fitness は合成 dynamical task か ESN proxy）。本 thread は
> その gap を閉じる：**進化×検証される再帰コアを、本物の (tiny, byte-level) 言語モデルに配線し、
> fitness を実 held-out perplexity にする**。北極星 = 「進化系を body plan とし、各モダリティ
> (text=言語感覚, vision=視覚) を同じ安定コアに挿す感覚器として段階的に獲得 → それ自体が AI として
> 成立する FullSense のマルチモーダル LLM 受け皿」(生物の五感獲得の進化体系に対応)。
>
> **Discipline:** design-first (この pre-reg を測定前に確定 / R-reach の教訓: soundness 論証を先に証明)
> → smallest falsifiable PoC → honest measurement (CLARABEL-pinned, 0-unsound を独立確認) → adversarial
> red-team + Codex pair-review → optional-extras (base = stdlib+numpy)。research/ 隔離, src/ additive-only,
> push 未。degradation/negative も valid な成果として歪めず残す。

---

## 0. 一行命題 (falsifiable)

**「arc の看板主張 — *より強い (が健全な) contraction verifier は、不安定化ゼロで進化の到達可能 fitness を
単調に解放する* — は、合成 dynamical task ではなく、本物の byte-level 言語モデルの held-out perplexity を
fitness にしても成立する。」**

成立 → llcore の verified-evolution が「実 LM の核」として load-bearing であることの実 substrate 証拠
(proxy gap を閉じる)。不成立 (どの gate でも perplexity 同等 / coupling が効かない) → honest negative:
実 LM では contraction 制約は free で、verifier の payoff は合成タスク固有 → 論文の scope を正直に狭める。

---

## 1. Substrate — reservoir (ESN-style) byte language model

完全 CPU/$0、numpy のみ、決定論的 (per-eval RNG 無し = Objective 契約)。

```
token x_t ∈ {0..255}                         # byte-level (modality-agnostic, multimodal-spirited)
e(x_t) = tanh(E[x_t]) ∈ (-1,1)^n             # 感覚器: FIXED seeded embedding (ESN input proj). |e|<1 を保証
s_t   = decay ⊙ s_{t-1} + (1-decay) ⊙ tanh(W s_{t-1} + e(x_t))   # 進化×検証される再帰コア (= arc CoupledNDGene)
logits_t = R s_t + c                          # 感覚器: readout, ridge-fit per gene (closed-form, 決定論的)
loss  = mean cross-entropy(softmax(logits_t), x_{t+1})          # 本物の next-byte LM loss
fitness = exp(-held_out_CE)  ∈ (0,1]          # per-byte likelihood (headroom 有, ceiling trap 回避)
```

- **gene = (decay ∈ [0,1]^n, W ∈ [-2,2]^{n×n})** = arc の `CoupledNDGene` そのもの。進化・検証対象。
- **E (embedding) と R (readout) = 「感覚器」**。E は fixed seeded random (全 gene 共有 = 公平比較;
  ESN/reservoir computing 流儀, arc Step 6 と連続)。R は gene ごとに ridge 回帰で one-hot target に
  closed-form fit → held-out で真の softmax CE を評価 (Step 2 ridge_readout パターンの LM 版)。
- **n (reservoir 次元)** = **8 (PoC)**。vocab = 256 (byte)。context = 全 sequence を逐次。
  ⚠️ **n の上限 (grounding で発覚した honest 制約):** arc の `cert_two`/`cert_sdp` は t-box の **2^n 頂点を
  列挙**する (n=2,3,4 で実証された範囲)。LM 次元 n=32 では 2³² 頂点で**実行不能**。閉形式の `cert_inf`
  のみ O(n²) で scalable。ゆえに **Stage R-LLM-0 は arc が実証済みの小 n=8 (2⁸=256 頂点で全証明器が
  tractable) に限定**し frontier を検証する。**「頂点列挙を避ける vertex-free な sound 2-norm/SDP 証明器で
  n=32+ の実用 LM 次元へ」は明示的な次段 R-LLM-1** に分離 (robust-LMI を急ぎで作ると unsound リスク =
  R-reach の罠ゆえ、soundness 証明を先に確立してから着手)。dimension thread R1 の「SDP completeness は
  次元劣化」に加え、本 thread は**計算量の壁 (頂点列挙の指数性)** も frontier の次元限界であることを示す。
- **corpus**: llcore の research/*.md + src/*.py を byte 連結 (self-contained, offline, 決定論的)。
  train/held-out を時間順 split (held-out = 末尾 20%、leakage 無し)。

### honest 限定 (測定前に明示)
- これは **reservoir/ESN 言語モデル** (fixed embedding + ridge readout)。**gradient 学習された真の
  Transformer ではない**。softmax-attention の純 Transformer・end-to-end 学習は GPU/次段 (B)。
- tiny (n=32) → 絶対 perplexity は弱い。問うのは **絶対性能でなく gate 間の相対差 (frontier)** と
  「unigram baseline を超えるか (= LM として最低限機能するか, L0)」。
- byte-level (語彙構造を学ばない) → char/subword より不利。あえて modality-agnostic を優先。

---

## 2. SOUNDNESS 論証 (R-reach 教訓: 測定前に証明する)

verifier は arc の `coupled_nd.cert_inf / cert_two / cert_sdp` を**そのまま**再利用する。これらは
autonomous 再帰 `s' = decay⊙s + (1-decay)⊙tanh(Ws+Vx)` の t-box `[t_min,1]^n` 上の contraction を
sound に証明する (arc Track C/D + red-team で確立)。LM への移植が sound である根拠:

**補題 1 (状態有界).** `s_0 ∈ [-1,1]^n` なら ∀t: `s_t ∈ (-1,1)^n`。
*証明.* `s' = decay⊙s + (1-decay)⊙tanh(·)`。`tanh(·)∈(-1,1)`, `s∈[-1,1]`, `decay∈[0,1]` →
各座標は `[-1,1]` の凸結合で `(-1,1)` に入る。∎ (s_0=0 で開始 → 不変)

**補題 2 (入力有界).** embedding を `e(x)=tanh(E[x])` とすると ∀ byte x, ∀座標 i: `|e(x)_i| < 1`。
*証明.* tanh の値域。∎ → LM の入力 `x:=e(token)` は `|x|_∞ < 1` を満たす = `max_input_abs=1.0` が
**健全な (実際には strict な) 上界**。

**補題 3 (t-box が到達可能 Jacobian を被覆).** 再帰の Jacobian は `J = diag(decay) + diag((1-decay)⊙t) W`,
`t_i = sech²(pre_i)`, `pre_i = (Ws)_i + e_i`。補題1,2 より `|pre_i| ≤ Σ_j|W_ij|·|s_j| + |e_i| <
Σ_j|W_ij| + 1 = M_i` ⇒ `t_i = sech²(pre_i) > sech²(M_i) = 1-tanh²(M_i) = t_min_i`
(`coupled_nd.t_min_per_coord` が max_input_abs=1, V=I で計算する値と一致)。よって LM 動作中の全
到達可能 `t` は t-box `[t_min,1]^n` に含まれる。∎

**帰結.** `cert_inf/two/sdp(gene, max_input_abs=1.0)` が True を返す gene は、t-box 上で ‖J‖_∞<1 /
σ_max<1 / common-P LMI のいずれかを満たし ⇒ t-box (⊇ 全到達 Jacobian) 上で `ρ(J)<1` ⇒ **LM 再帰は
全 byte 入力列に対し contraction** (echo state property: 初期状態を忘れ、hidden state は有界・発散しない)。
これは実 LM の核として欲しい安定性そのもの。**soundness は certificate 定理が保証**し、経験 oracle は
from-below の falsification/consistency check (反例検出は可・証明は不可)。

**R-reach 型の罠への対処 (vacuity check).** R-reach では全候補が収縮し oracle が vacuous だった。
ここでは clip 範囲 `decay∈[0,1], W∈[-2,2]` は**発散 gene を多数含む** (例: decay=0, W 大 → ‖J‖>1)。
ゆえに ungated gene は実際に hidden-state 発散 (NaN/blowup) し、oracle は**非 vacuous** = 真の
soundness consistency check として機能する (L1/L2 で測る)。**soundness contract**: embedding は必ず
tanh-bounded、`max_input_abs` は 1.0 固定。これを破る (unbounded embedding 等) と補題2,3 が崩れ
certificate が unsound になる → コードで固定し test で守る。

---

## 3. 進化ループ

`evolvable_core.evolve()` を**無改変**で使用 (codec=`CoupledNDGeneCodec(n=32)`, objective=LM held-out
likelihood, verifier=`make_nd_verifier(name)`)。child admission で fail-closed gate。

**計算量配慮 (honest):** fitness 1 回 = (ridge readout fit + held-out forward) で n=32, corpus ~数十 KB なら
CPU 数十 ms〜。pop/gens は CPU tractability で選ぶ (下記)。ridge は closed-form で per-eval 決定論的。

---

## 4. Pre-registered FALSIFIABLE GATES

| gate | 命題 | PASS 条件 | 測り方 |
|---|---|---|---|
| **L0 realization** | tiny reservoir LM は実際に「言語モデルとして」機能する | held-out per-byte CE が **unigram baseline を有意に下回る** (≥ 一定 margin)。fitness > unigram fitness | 良 gene 数本 + 進化 best で測定。unigram = 訓練 byte 頻度の CE |
| **L1 soundness** | 健全 gate が admit した gene は実 LM 上で hidden-state 安定 | admit gene の held-out 列で hidden-state 有界 (`max|s|<1+ε`, no NaN) **0 発散** (≥N admit gene, ≥M seq) | inf/two/sdp 各 admit gene を独立 oracle で検査 |
| **L2 load-bearing** | gate 無しだと発散 (or 役立たず) gene が混入、gate がそれを排除 | ungated 集団/random に hidden 発散 gene が **X%>5%**、健全 gate は **0%** | none vs gate の admit 集団で発散率比較 |
| **L3 frontier / payoff** | **より強い verifier がより低い perplexity を解放** (看板主張) | rotation 的に有利な regime で paired: best held-out fitness で **sdp/two ≫ inf** (one-sided Wilcoxon p<0.05, Bonferroni 後生存) + winner が inf-rejected region に居る (mechanism 帰属) | gated evolution 多 seed paired (CRN) |
| **L3-null HONEST NULL** | coupling が効かない regime では gate 差が消える | benign regime で sdp vs inf が null (p>0.1) | 別 objective/regime で対照 |

補助 (mechanism, non-circular): **landscape attribution** — 固定 gene pool を tightest sound certifier で
region 分類 (inf / two_norm_only / sdp_only / non_certified) し、region ごとの best held-out fitness を測る。
arc exp1 の LM 版: region ceiling が GA luck でなく機構であることを示す。

---

## 5. Adversarial red-team (測定後)

- **Lens A — admission-size artifact:** fitness を dynamics と無相関 (gene-keyed pseudo-random) にすると
  gate 差が消えるか (消える=payoff は fitness-structural, 良)。
- **Lens B — 独立 soundness:** admit winner を別 seed・高 sample の独立 oracle + 長系列で再検査 (0 発散)。
- **Lens C — seed robustness:** L3 を複数 base-seed family で再現 (Bonferroni)。
- **Lens D — mechanism 帰属:** winner の region 分類が inf-rejected に居ることを確認。
- **Lens E — corpus robustness:** corpus を別 split / 別ファイル集合にしても L0/L3 が保つか。
- **Lens F — readout confound:** ridge α 掃引・readout を gate 間で固定し、gate 差が readout 容量差で
  説明されないことを確認。

---

## 6. 実行パラメータ (CPU tractability, 確定)

- n=32, vocab=256, corpus = llcore research/*.md + src/**/*.py の byte 連結 (上限 ~200KB, 決定論 sort)。
- train/held-out = 80/20 時間順 split。ridge α = 1e-2 (Lens F で掃引)。
- gene pool (landscape) = 2000–4000 random gene。
- gated evolution: pop=24, gens=20, n_seeds=15 paired (CRN)。CPU で重ければ pop/gens を下げ honest に記す。
- 乱数 seed 全固定。CLARABEL pinned (SCS fallback 禁止 = fail-closed)。

---

## 7. Multimodal-extensible API (VLM を視野に, 段階的)

「感覚器」を pluggable に分離 (今は実装せず設計のみ固定):

```
class ModalityEncoder(Protocol):
    n: int                          # 出力次元 = reservoir 次元
    def encode(self, token) -> np.ndarray: ...   # |.|_inf < 1 を保証 (soundness contract)
```

- **ByteEmbedding** (本 PoC): token∈{0..255} → tanh(E[token])。
- **PatchEmbedding** (将来 VLM): 画像 patch → tanh(linear(patch))。同 contract (|e|<1) を守れば
  **再帰コア・verifier・soundness 論証は無改変で再利用** = 視覚という感覚器を同じ安定コアに挿す。

soundness contract (|encode|<1, max_input_abs=1) を満たす限り、どのモダリティでも補題1-3 が成立 →
verifier はモダリティ非依存。これが「マルチモーダル LLM 受け皿」の構造的根拠。

---

## 8. 成果物 (予定)

`lm_substrate.py` (corpus/embedding/reservoir/ridge readout/CE) · `verifier_adapter.py` (arc cert 再利用 +
soundness contract enforcement) · `test_lm.py` (TDD: 決定論/補題1-2/soundness/L0 smoke) · `demo_lm.py`
(L0 realization demo) · `exp_landscape.py` (region attribution) · `exp_gated.py` (L0–L3) · `redteam.py`
(A–F) · `VERDICT.md` · results JSONs。**src/ untouched, push 未。**
