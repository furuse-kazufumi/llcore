# 真の内的化 — 検証を勾配に内蔵する設計骨子 (v1, 2026-06-08)

> 状態: **設計骨子 (敵対レビュー前のスナップショット)**。HD-1 接地 (H1 PASS: 監督階層
> sound ≫ shared-empirical ≫ self-empirical ≈ none) の自然な続き = R-endo thread **本来の問い**。
> 正本系列: [[HD1_GROUNDING_VERDICT.md]] (前段) → 本 doc → 敵対レビュー → feasibility → 事前登録 → GPU。
> 方針: [[project_llcore_one_year_policy]] (1 年スパン、submission 急がず本流深化)。

## 0. 位置づけ — 何が新しい問いか

HD-1 接地で測ったのは **ハーネスが監督を持つ** 形 (ENDO = cadence で cert_inf 検査 → fail で
rollback)。これは「外的監督」: モデルは gate を知らず、ハーネスが事後に巻き戻す。empirical
(OBSERVE) は共有死履歴があっても sound に届かないことも確定した。

**本実験の問い**: 監督を **モデル自身の勾配に内蔵** できるか。cert_inf = `infnorm_sup < 1` の
sound 上界 `infnorm_sup` を微分可能 surrogate にし、補助損失としてモデルの総損失に加える →
「モデルが自分を ρ<1 域へ押す」。HD-1 の ENDO (離散・事後 rollback) との比較で初めて
**「検証の内的化」が gradient 文脈で測れる**。これは §7 で将来実験として明記したもの。

## 1. 微分可能 surrogate (実コード検証済み)

`cert_surrogate.py` に実装。`infnorm_sup` は abs / max / 和 / 積 の合成 = subgradient 可能。
torch 版 `infnorm_sup_torch(decay, W)` を実装し self-test で確認:
- **numpy 正本との一致**: 15 ケース (n∈{8,16,32}) で max abs err = **3.55e-15** (浮動小数点精度)。
- **勾配が流れる**: `W = 2·tanh(raw_W)` の再パラメータ化越しに `|grad raw_W| = 14.9`、admit 外で。
- **片側 hinge の整合**: `cert_surrogate_loss = relu(infnorm_sup − threshold)` は admit 内
  (cert_inf=True) で loss=0・admit 外で loss>0 — cert_inf と完全整合 (shrink 1.0/0.005 両ケース)。

補助損失: `L_total = CE + λ_cert · relu(infnorm_sup_torch(decay, W) − (1 − margin))`。
margin>0 は admit set の **内部** へ押す余白 (境界張り付き回避)。admit 内では勾配ゼロ
(片側) なので、ρ<1 を満たす限り CE 最適化を邪魔しない。

## 2. arms (4)

| arm | 監督の所在 | 実装 |
|---|---|---|
| **NONE** | なし | baseline (HD-1 と同じ drift; ρ→1.95 帯へ) |
| **ENDO_HARNESS** | 外的 (事後 rollback) | HD-1 の ENDO 完全流用 (cadence k=4 検査 → fail で core+Adam rollback) |
| **ENDO_GRAD** | **内的 (勾配内蔵)** | 毎 step `L_total` に surrogate 補助損失を加算。rollback なし・gate なし |
| **ENDO_BOTH** | 内的 + 外的 | GRAD の補助損失 + HARNESS の rollback (安全網; 内的が滑らかに効けば rollback 発火は減るはず) |

- 全 arm 共有 admissible init (HD-1 と同じ; 公平な開始)。
- 測定は HD-1 と同一 (cadence m で empirical_rho / 実害 probe / infnorm_sup / proxy を記録)。

## 3. 仮説方向 (骨子; 文言は事前登録で最終化)

- **H1 (内的化は機能するか)**: ENDO_GRAD の窓契約死率 < NONE (内的監督が死を減らす)。
  ※ HD-1 の OBSERVE (empirical) は届かなかった — surrogate は **sound 上界そのもの** を押すので、
  empirical proxy と本質的に違う。ここが実験的不確実性の核 (補助損失が drift 速度に勝てるか)。
- **H2 (内的 vs 外的のコスト)**: ENDO_GRAD の最終 CE < ENDO_HARNESS の最終 CE、**かつ** 死回避は
  同等以上。直観: 離散 rollback は learning を粗く捨てるが、滑らかな補助損失は admit set 内部で
  CE 最適化を続けられる → 同じ安全性をより低い CE コストで。**これが成立すれば「内的化は外的監督
  より効率的」= 強い結果**。反証方向: 補助損失が CE と綱引きして CE を悪化させる可能性も対等にある。
- **H3 (滑らかさ)**: ENDO_GRAD の ρ(step) 軌跡は ENDO_HARNESS より滑らか (rollback の鋸歯がない)。
  記述的。
- 反証条項候補: ENDO_GRAD が NONE と同等 (死を減らせない) なら「λ_cert/margin の探索不足」と
  「surrogate 内蔵では drift に勝てない」を切り分ける (feasibility で λ sweep)。

## 4. 「真の内的化」の正確な意味と留保 (over-claim 防止)

- **何が内的か**: cert の勾配シグナルが **モデルのパラメータ更新に内在** する。HARNESS は
  「ハーネスが事後に判定・巻き戻す」= モデルの学習力学の外。
- **何が依然外的か (honest)**: 補助損失の **設計** (surrogate 形・λ・margin) はハーネスが与える。
  完全な自律 (モデルが検証すべき性質を自分で発見) ではない。HD-1 レビューの「ハーネスが gate を
  持つ以上 内的化ではない」批判への応答: 本実験は「**判定の所在**を事後フィルタから勾配へ移す」
  ことの効果を測るのであって、「目的の内発性」は主張しない。ラベルは「検証シグナルの勾配内蔵」。
- toy 教訓の再適用: HD-1 で潰した「トートロジー」(gate が先回りして死を防ぐと死回避が自明) は
  ここでは起きない — surrogate は **soft** で死を完全には防げない (admit 外で押し返すが瞬時には
  戻せない)。死は依然踏みうる = 死回避は実験的事実として測れる。

## 5. 弱点の自白 (骨子)

- surrogate は `infnorm_sup` (∞-norm sound 上界) で、cert_inf と同じ保守性を継ぐ — 真の ρ より
  緩い admit set を押すので「過剰縮小」の CE コストを HARNESS と同程度に負う可能性。
- λ_cert/margin は 1 設計を固定する (HD-1 の OBSERVE と同じ留保: 「1 つの誠実な実装の実力」)。
- ENDO_BOTH は交絡が多い (補助損失と rollback の寄与分離が難しい) — 探索的に留める。
- 16 seeds で CE の小効果は拾えない可能性 (HD-1 の H2 が n=128 で連言を破った前例)。

## 6. 段階ゲート (HD-1 と同じ 2 段階登録)

1. **本骨子 → 敵対レビュー** (Workflow lenses; HD-1 で blocker を 4 件捕捉した実績)。
2. **CPU feasibility** (n∈{8,32}, 4 seeds): (a) ENDO_GRAD が死を減らすか方向確認 (b) λ_cert/margin
   sweep → 本走固定値同定 (c) admit 内 grad-zero が CE を邪魔しない確認 (d) ENDO_BOTH の rollback
   発火率 (内的が効けば下がるはず)。
3. **feasibility 結果で事前登録最終化** → binding commit (結果取得前)。
4. **GPU 本走** (Kaggle T4, n∈{64,128,256} × 16 seeds)。
5. VERDICT → 論文 §9.8 (検証の勾配内蔵) → 1 年方針の本流前進。

## 7. 未決 (D1-D4; feasibility で詰める)

- **D1**: margin の値 (0 = 境界、>0 = 内部余白)。境界張り付きと過剰縮小のトレードオフ。
- **D2**: λ_cert を固定か annealing か (学習初期は弱く、drift が出る後期に強く?)。骨子は固定。
- **D3**: ENDO_GRAD の死回避が NONE と同等だった場合の分岐 (λ 不足 vs 内蔵の原理的限界)。
- **D4**: HARNESS の Adam-sync は HD-1 で交絡なしと確認済 — GRAD は rollback がないので Adam 問題なし。
  ただし ENDO_BOTH では rollback 時に補助損失の勾配履歴が Adam に残る交絡に注意。

## 8. 正本リンク

- 前段: [[HD1_GROUNDING_VERDICT.md]] / surrogate 実装: `cert_surrogate.py` (self-test 済) /
  基質: `research/highdim_evolution/` / 論文 §7・§9.6・§9.7 / 設計元: HD1_GROUNDING_DESIGN.md §7。
