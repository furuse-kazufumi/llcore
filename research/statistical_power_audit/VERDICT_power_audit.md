# llcore 統計的検出力 自己監査 — VERDICT

> 2026-06-01。research/statistical_power_audit/ 隔離。src 無改変・git 非実行。
> 命題 H0_suppress (DESIGN.proposition) を 4 方向 (校正 / 検出力逆算 / ablation / Type I guard) で検証。
> 生数値は同 dir の `results_power_audit.json` (統合) + 各 `*_results.json` に永続化。

## 結論 (一行)

**H0_suppress は条件付き支持 (NUANCED)。** 統計マシナリは「一律に進化を抑えている」のではない —
実験標準 n=15 では既知真陽性 (合成欺瞞回廊 d*=0.16) を正しく検出し校正は健全 (S1 不成立 / R1 成立)。
だが (a) 実 negative の一部 (flip_flop, E-A C-gen4b) は **underpowered = inconclusive** であって
「③不在の証拠」ではない、(b) `ridge clip=True` は ridge landscape を平坦化し真の構造を隠す
(K4 = 唯一の能動的 suppression 機序)、(c) しかし gate 閾値の連言 (p / min_effect / n) 自体は健全で、
緩めると Type I が代償として増える。

## 破綻ゲート (全通過)

| gate | 内容 | 結果 |
|---|---|---|
| G1 CPU 完走 | 4 script 各 <900s, exit 0 | calib 575s / repower 332s / ablate 204s / type1 294s = **全 <900s** |
| G2 再現性 | 同 seed 2 回で配列完全一致 | d=0.16 n=15 で `np.array_equal=True`。d*_strict は 3 seed {20260530,777,31337} で一致 |
| G3 power 計算器妥当 | null<=0.06 / 大効果>=0.95 / C-gen4b 模擬一致 | null=0.023, big=1.00, cg4b_n15=0.58/n30=0.83 = **VALID** |
| G4 src 不変 + git なし | src/ 差分ゼロ, 書込は research 配下のみ | `git status src/` 空, 全 _meta `src_unchanged=true` |

## (A) 校正 — false-negative onset 曲線 d_fn(n)

`calibrate_known_positive_results.json`。既知真陽性 corridor で full strict gate が
「③ が真に勝つのに no-effect 判定」になる最小 d を n ごとに測定:

| n_seeds | d_fn(load-bearing onset) | d*=0.16 検出 | 0.16 の textbook Cliff δ |
|---|---|---|---|
| 10 | **なし (到達せず)** | **False** | +0.44 |
| 15 (実験標準) | 0.15 | True | +1.00 |
| 30 | 0.15 | True | +0.59 |
| 60 | 0.14 | True | +1.00 |

- **n=15/30/60 では gate は健全**: d_fn <= 0.16 = 既知真陽性 d*=0.16 を取り逃がさない (むしろ d=0.15 まで検出)。3 base_seed で d_fn(15)=0.15 robust。
- **n<=10 は盲点**: d=0.16/0.18/0.20 が cliff δ=+0.44〜+1.0, p<0.001 でも **n 床で全て no-effect 判定**。
  → step6 exp7 は n=8/6 で実施 = まさにこの盲点域 (gate が強効果すら見えない)。
- d=0.14 は n=15/30 で取り逃がし、n=60 で初検出 = 小効果は大 n を要す。

## (B) 検出力逆算 — 実 negative の observed power と n80

`repower_real_negatives_results.json`。実 per-seed paired delta (exp_ea3/exp_c2c3 JSON から復元、仮定値でない):

| case | diff | Cohen dz | observed power (n=15) | n80 (param/boot) | underpowered | 律速 |
|---|---|---|---|---|---|---|
| **C-gen4b** MAP-E vs random | +0.063 | +0.35 | **0.31** | ~64 / ~255 | **Yes** | p_only |
| **flip_flop** MAP-E vs random | +0.004 | +0.31 | **0.27** | ~89 / ~82 | **Yes** | p_only |
| flip_flop vs RR-hillclimb | +0.004 | +0.27 | 0.15 | ~169 / 未到達 | Yes | p_only |
| C-gen4a vs panmictic (diff<0 対照) | −0.019 | −0.13 | 0.03 | 未到達 | **No (健全)** | p_only |
| delayed_parity vs random (床) | −0.001 | −0.20 | 0.004 | 未到達 | No (健全) | p_only |
| C-gen3 vs randselect (PASS 対照) | +0.126 | +0.68 | 0.74 | ~16 / ~18 | (borderline) | p_only |

- diff>0 符号一貫の **C-gen4b/flip_flop は underpowered** (power 0.27-0.31)、80% power に現行 n=15 の **4-6 倍 (n~64-89)** を要す。
  → これらの honest-negative は「③不在の証拠」でなく **inconclusive**。verdict 自身の自認 (STEP_C) と一致。
- **律速は全 case で `p_only`** (片側 Wilcoxon p<0.05 + 小 n)。容疑の **min_effect=0.147 床は律速でない**。
- 対照 C-gen4a (diff<0) は power 不問で真に効果無し = 健全な negative (R2 成立)。
- PASS 対照 C-gen3 ですら n80~16-18 = n=15 では borderline。現行 n は「勝つ case でギリギリ」。

## (C) ablation — K1-K4 toggle で verdict 反転 (+ Type I コスト)

`ablate_suppression_knobs_results.json`。OFAT (他 ON 固定):

- **反転を起こした緩和は K2_alpha=0.20 のみ** (C-gen4b, flip_flop を PASS に)。
  min_effect↓ / min_seeds↓ は p で落ちる case を救えず **無反転**。
- corridor d=0.13 は base_seed で MAP-E が勝たない (diff<0) ため **如何なる緩和でも反転しない**
  = 真の負を誤って ungate しない (gate の健全性)。
- **K1 fresh-seed OFF** (noisy best 持越し): corridor d=0.13 で反転せず。pure-null でも FPR 増は限定的
  (この budget では。caveat 参照)。
- **K3 archive-max vs global-best**: 両読み出しとも C-gen3 で③が勝つが、**archive-max は diff を ~50x 水増し**
  (archive diff=+0.081 δ=+0.60 vs global diff=+0.0016 δ=+0.40)。`flipped_global_to_archive=False`。
  → Codex F2 の global-best 化は③過大評価を正しく除去 (suppression でなく Type I 防止)。
- **K4 clip=True (能動的 suppression 機序)**: ridge fitness の spread を最大 **13x 平坦化**
  (addition: clip_true_std=0.028 vs clip_false_std=0.380, floored 93%)。floored gene の raw R² は
  spread 0.37・符号一貫 (全負) = **clip が真の landscape 構造を隠蔽**。`flattening_hidden_by_clip=True`。

## (D) Type I guard — sweet spot は存在しない (honest disclosure の核)

`type1_guard_sweep_results.json`:

| level | shuffle-null FPR | borderline TPR | net gain |
|---|---|---|---|
| baseline (全 ON) | 0.047 (≈名目 alpha) | 0.20 | +0.15 |
| K2_min_effect=0.05 | 0.046 | 0.20 | +0.15 |
| K2_min_effect=0.0 | 0.049 | 0.20 | +0.15 |
| K2_alpha=0.10 | 0.090 | 0.20 | +0.11 |
| **K2_alpha=0.20 (反転 knob)** | **0.172 (>2× alpha)** | 0.20 (不変) | +0.03 |

- baseline の null FPR (d0/pure=0.00, shuffle=0.047 ≈ 0.05) = **gate は適切に校正、むしろ保守的**。
- 反転をもたらす **K2_alpha=0.20 は純 Type I コスト** (FPR 0.047→0.172, TPR 0.20 のまま不変) = sweet spot でない。
- **min_effect / min_seeds 緩和は FPR も TPR も動かさない** = これらは binding 制約でない (緩めても無益)。
- → 「緩めれば C-gen4b/flip_flop が PASS する」は真だが、その唯一の道 (alpha↑) は偽陽性を等価に増やす
  = **抑制でなく『厳しさ⇄緩さのトレードオフ点』に居る** (S4 不成立)。

## 判定基準の機械評価 (results_power_audit.json `criteria_evaluation`)

```
S1 Type II @ n=15 = False   (n=15 で gate 健全)
S1 Type II @ n<=10 = True    (小 n で強効果も盲点)
S2 underpowered = True       (実 negative の一部は inconclusive)
S3 flip exists = True        (K2_alpha=0.20 で反転)
S4 flip is sweet spot = False (反転緩和は Type I を等価に増やす)
suppress (S1∧S2∧S3∧S4) = False   ← 一律抑制ではない
R1 detects known positive @ n15 = True
R2 diff<0 control healthy = True
R3 relaxation inflates FPR = True
```

## 実務的勧告 (再測すべきもの)

1. **flip_flop / E-A C-gen4 は n を ~80-90 に増やして再測** (現 n=15 は power 0.27-0.31 で結論不能)。
   再測で依然 p>=0.05 なら初めて「③不在」と言える。それまでは inconclusive。
2. **step6 exp7 (n=8/6) は結論を保留** — 校正 (A) より n<=10 では強効果すら検出不能と判明。n>=15 で再測。
3. **ridge 系 landscape の③判定は `clip=False` (raw R²) で spread/符号も併報** — clip は floor 域の構造を隠す
   (K4)。選択圧としての clip=True は維持しつつ、診断時は raw を見る。
4. **gate 閾値 (p<0.05, min_effect=0.147, n>=15) 自体は緩めない** — 緩和は Type II を救わず Type I を増やす。
   検出力不足は **閾値でなく n で解く**のが正しい (power は n の関数)。
