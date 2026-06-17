# メモリ・スケーリング戦略 — 線形か指数か、レジーム別の使い分け (llcore, 2026-06-17)

> 出典: 多エージェント・ワークフロー(8 アルゴリズム族マッピング × 各 2 レンズ敵対検証 → 統合 → 完全性批評,
> 26 エージェント・全 verdict high confidence)。ユーザー論点「メモリ容量に対し線形に伸びる
> アルゴリズムと指数的に伸びるものがあり、レジーム別の使い分けが勝敗を決める」への回答。
> **批評が検出した overclaim(後述 §5)は本ドキュメントで訂正反映済み**。記事素材も兼ねる。

---

## 1. 結論(ユーザー論点への直接回答)

**ニューラルの「賢さ」がメモリに対して指数的に伸びる族は存在しない。** 指数が本物なのは古典計算の 2 か所だけ。

- **真に指数的にメモリが効く稀な 2 例**:
  1. **メモ化 / 動的計画法(DP)** — 重複部分問題があるとき**のみ**、指数の再帰木を多項式表に畳む
     (Fib 2ⁿ→n、Held-Karp n!→2ⁿ)。指数はメモリの性質ではなく**問題の冗長性構造**に内在。重複が
     無ければ(gradient checkpointing 等)定数倍止まり。**DP 自身もテーブルが RAM に収まる間だけ**で、
     収まらなければ Hirschberg 線形空間 DP 等へ再帰する(レジーム内の入れ子トレードオフ)。
  2. **modern Hopfield の容量** — 格納パターン数が**次元 d に指数**(模式的に ~2^(d/2)、厳密には
     near-orthogonal 前提の存在限界 `√p·c^((d-1)/4)`)。ただし (a) 指数軸は footprint でなく d、
     **footprint→容量は線形** (b) well-separation 前提・相関データで崩壊 (c) 容量 ≠ 知能(暗記であり
     汎化を害する)。
- **べき乗則 / 対数(劣線形)が圧倒的多数 = ニューラルの現実**: scaling law(dense LM 損失
  L=E+A/Nᵃ、floor あり)、MoE(同じべき乗則を安い compute へシフト)、RAG(≈log(corpus)、Zipf 長尾で
  逓減)、SSM の品質。**N を 10× して固定の損失減を買う = コストは指数・利得は線形**。
- **罠**: 「emergence で指数的に賢くなる」は Schaeffer 2023 の通り**滑らかなべき乗則上の不連続メトリクス
  の測定アーティファクト**。「指数的改善」の主張はほぼ「指数の支出 × 線形の利得」の誤読。

---

## 2. 族別スケーリング一覧(検証修正済み)

| 族 | footprint スケーリング | capability-per-memory | 勝つ regime |
|---|---|---|---|
| Transformer attention + KV cache | O(L)(GQA/量子化は定数削減のみ) | 保持トークン数=線形 / 品質=plateau→劣化。指数なし | 損失なし・位置厳密な照合/コピー。**実効窓 ≤ 容量窓**(lost-in-the-middle で先に崩れる) |
| Recurrent / SSM (RWKV, Mamba) | O(1) 状態 / 重み O(params) | 品質=べき乗則。state 軸では recall が線形↑→Fano 天井で飽和 | 長文/ストリーミング/エッジ。**集約系のみ** — recall 重は hybrid 必須 |
| Retrieval 外部メモリ (RAG/RETRO) | store O(n) | ≈log(corpus)。指数なし(事実間に複利なし)。distractor 過多で**負に反転し得る** | 知識が大/疎/更新頻繁/出典必須。recall 主体・推論は改善しない |
| Mixture-of-Experts | O(total_params) 常駐 | べき乗則を安い compute へシフト。expert 数で飽和 | メモリ潤沢・compute 律速の大規模サービング。単機エッジ不向き |
| 量子化 / 数値精度 | bits/weight に**厳密線形(唯一)** | cliff_then_flat。重みは~4bit まで平坦・3bit 以下で急落 | 固定メモリ/帯域予算の推論。**小バッチ=重みのみ / 大バッチ=W8A8** |
| メモ化 / DP・古典 | 状態空間サイズ依存 | other(不均一)。重複あり=指数速度向上 / 無し=定数倍 / 確率的=誤りが bit に指数減衰 | DP=重複が指数的&状態空間が多項式時。Bloom=厳密性を譲れる membership |
| Associative / Hopfield | O(N·d) 線形 | near_exponential(容量 vs **次元 d** のみ・条件付き) | 高 d で分離良好な離散プロトタイプの 1-shot 連想照合。相関データ不向き |
| Dense 重み(パラメータ) | O(params) | べき乗則(Chinchilla)。**compute-optimal は N と D を同時配分** | — (§5 の serving crossover に注意) |

---

## 3. regime → primitive 決定則(llcore の 3 測定済み + 追加)

llcore 既測: ① constant-state recurrent vs GPT KV/attn、② mmap read-only weights(load 時 RSS 後ろ倒し・
固定費なので大モデルほど勝ち)、③ int8 weight-only 量子化(~3.9× footprint・<0.1% PPL)。

| regime | 主プリミティブ | llcore 現状 | 追加すべきもの |
|---|---|---|---|
| **小 RAM CPU**(エッジ) | constant-state recurrent + int8 | ①+③ で射程内 | sub-4bit は QAT 前提(PTQ は 3bit で cliff)。recall は外部 retrieval |
| **大 RAM** | mmap 大モデル + retrieval | ② が効く | RAG/外部 store(O(n) 安価・出典付き)。**ただし §5: 高サービング量では“より小さい N”が最適化することに注意** |
| **GPU**(HBM 潤沢・帯域律速) | KV(厳密照合)+ MoE + Hopfield 連想層 + offload 階層 | 未着手 | (a) exact attention 解禁(GQA で定数削減・KV eviction で sub-linear) (b) Hopfield 層(高 d で容量↑) (c) hybrid(SSM 主 + sliding-window attention 数層=Samba/Jamba) (d) **offload/階層(GPU↔CPU↔NVMe)= 移行で実現可否が変わる本丸** |

横断: **DP/メモ化**はアーキ非依存の常時適用最適化。**gradient checkpointing(O(√L))**は GPU 学習で
メモリが壁のとき compute と交換。**KV eviction(H2O/StreamingLLM)**は attention の Θ(L) を recall 損失と
引き換えに sub-linear 化する別ノブ。

---

## 4. 「GPU/大RAM でスペックが跳ねる」指令との整合

- **mmap 重み(②)**: load 時 RSS が固定費 → RAM が増えるほど大モデルを丸ごと常駐でき相対的勝ちが拡大。指令と一致。
- **Hopfield 連想層(GPU)**: 容量が次元 d に指数。GPU の高 d で容量が跳ねる(footprint は線形)。
  ★留保: 模式 2^(d/2) は near-orthogonal 前提の存在限界・相関データで崩壊。相関実データには使わない。
- **exact attention(GPU)**: HBM が払える間だけ「損失なし位置厳密照合」が解禁=**機能の有無が切り替わる質的ジャンプ**(cliff の良い側)。
- **MoE(GPU/マルチGPU)**: メモリで compute 効率を買う。total_params を常駐できる環境で初めて勝つ(単機小 RAM では最悪効率)=移行で跳ねる典型。
- **offload/階層メモリ**: 「ハード移行でスペックが跳ねる」の**文字どおりの機構**。GPU↔CPU↔NVMe の階層化で、
  ある tier では載らなかった MoE/長 KV が次の tier で実行可能になる(ZeRO-Infinity/FlexGen)。
- **量子化(③)の役割**: それ自体は跳ねない(capability 平坦)が、**他のジャンプの enabler** —
  ~3.9× footprint 削減で同 HBM により大モデル/長 KV/高 Hopfield d を載せる余地を作る。

---

## 5. honest 留保(批評が検出・訂正反映)

1. **★overclaim 訂正**: 「大 RAM → mmap で大モデルを常駐」は**無条件には正しくない**。Beyond-Chinchilla
   (Sardana & Frankle)では**サービング量(推論リクエスト総数)が多いほど、compute-optimal は“より小さい N
   ×より長い学習 D”へ動く**。つまり「メモリがあるから大モデル」ではなく「**推論回数で割り戻した総コスト**」で
   N を選ぶ。大 RAM は「大モデルを選ぶ理由」ではなく「大モデルも選べる選択肢」。
2. recurrent/SSM は recall-heavy 長文で負ける → retrieval か hybrid 必須。
3. RAG は推論を改善しない・log で頭打ち・distractor 過多で負に反転し得る。「より多くの bytes」を推論期待で買わない。
4. Hopfield の指数は次元軸限定 & 分離条件付き(相関データで崩壊)。
5. attention の実効窓は容量窓より狭い(lost-in-the-middle が HBM/compute の壁より先に効く)。
6. 量子化の cliff は**PPL で測ると capability より低く見える** — PPL が無傷でも hard-capability(厳密コピー/
   稀知識 recall/多段算術)は先に崩れる。**PPL だけの gate は危険**(→ §6 実験で検証)。
7. **ニューラル capability の指数的メモリ効果はどこにも無い** — 跳ねるのは「機能解禁(cliff の良い側)」と
   「メモリ潤沢前提でのみ成立する効率」であって、メモリ→指数知能ではない。

---

## 6. 次の実験(批評推奨・このハードで実行可能)

**int8 量子化の cliff を実測**して cliff_then_flat(4bit は plateau・3bit で cliff・2bit で破綻)を反証可能に検証。
→ 実装・結果は `scripts/quant_bitwidth_sweep.py` / `out/quant_bitwidth_sweep*.json` /
`docs/MEMORY_EFFICIENCY_FINDINGS.md` を参照。honest hook: PPL に加え **hard-capability proxy(held-out
top-1 accuracy)** を併記し、「PPL は無傷でも capability cliff は先に来る」を自前データで確認する。
