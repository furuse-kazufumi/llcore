# 手順 4 verdict — ③ が立つ状態を発見 (deceptive corridor + behavioral niching)

> **凡例 — 進化4要素 (Darwin/Mayr)**: ①変異 (variation) / ②遺伝 (heredity) / ③適者生存・選択 (selection = 適応度の差による差し survival) / ④過剰繁殖 (over-reproduction)。本書の「①」〜「④」はこの番号を指す (特に「③」= 適者生存)。平易な用語集 → [`YOUGO_平易版.md`](./YOUGO_平易版.md)。

**Goal** (ユーザー 2026-05-30): 「推奨の方法 (MAP-Elites) で進め、③ が立つという状態を探す」。
**結論**: **③ が立つ状態を発見した (存在証明)。** ただし条件付き — load-bearing になるのは
**欺瞞的 (deceptive) landscape** に限る。境界も特定済。
**位置づけ**: `STEP4_DESIGN_space_expansion_niching.md` の推奨路線 (iii) MAP-Elites を実装・検証。
コード: `research/step4_selection/` (src/ 非変更, research 隔離)。

---

## 0. 結論 (3 行)

1. **deceptive corridor** (genotypic corridor + fitness dip) で **MAP-Elites (behavioral niching) が
   random / panmictic-GA / random-restart hill-climbing の 3 baseline 全てに圧勝** (p=1.9e-6, Cliff δ=+1.00,
   全 20 seed 勝利)。MAP-Elites のみ大域最適に到達 (95%)、他は全て局所最適で停滞 (0%)。= **③ が立つ**。
2. この優位は **MAP-Elites が万能だからではなく landscape が欺瞞的だから**。dip を外した滑らかな
   corridor では **MAP-Elites は RR-hillclimb (p=0.29) にも panmictic-GA (むしろやや劣位) にも優位を失い、
   pure random にのみ勝つ** — 境界実験 exp5 (全 3 baseline) で確認。
3. → **③ の将来性は「実 task / 実 LLM fitness が欺瞞的 corridor 構造を持つか」に帰着する** empirical question。

## 1. ③ 立証の falsifiable 基準 (設計ノート C1-C4) と達成

| 基準 | 内容 | 達成 (exp4) |
|---|---|---|
| C2 | hill-climbing が局所最適に詰まる | ✓ RR-hillclimb / random / panmictic-GA 大域到達率 **0%** (局所 0.60 で停滞) |
| C3 | niching が baseline を有意に上回る | ✓ MAP-Elites 大域到達 **95%**, vs 3 baseline 全て p=1.9e-6, δ=+1.00, 勝率 100% |
| C4 | 勝因が探索量でなく diversity 維持 | ✓ random-restart hill-climbing (restart=coverage 機構) にも δ=+1.00 / **init_batch ablation** で確証 (下記) |
| (境界) | ③優位は欺瞞 regime 限定 | ✓ smooth landscape では 3 baseline 全てに対し優位消失 (exp5) |

**robustness**: 3 種の base_seed (20260530 / 777 / 31337, 計 60 seed) 全てで MAP-E reach≧0.95,
RR=0.60, p=1.9e-6, δ=+1.00 — seed 非依存。

**C4 の確証 (init_batch ablation, Codex Medium 指摘対応)**: 「MAP-Elites の初期 random batch
(default 600 点) の coverage が勝因では?」を切り分けた。init_batch を **30 に削減しても MAP-Elites は
100% 大域到達** (mean 0.998)。かつ **pure random は 6000 点でも 0% 到達**。→ 勝因は初期 coverage でも
探索量でもなく **archive の stepping-stone ratchet (diversity 維持)** と確定。

C1 (fitness 多峰性) は MAP-Elites の前提として**要求しない** (behavioral niching は fitness peak でなく
behavior descriptor で動く)。手順 2 で fitness 多峰性が出ないと判明したが、(iii) はそれを回避する路線。

**到達判定の注 (Codex Low 指摘)**: 「大域到達」は honest 再評価 fitness `>0.8` を大域峰 proxy とした
判定 (basin membership を直接記録したものではない)。この landscape では局所 0.60 / 大域 1.00 が明確に
分離するため妥当な proxy。

## 2. ③ が立つ landscape の構造 (発見した条件)

`research/step4_selection/exp4_genotypic_corridor.py`:
- **behavior = mean(gene)** (1D)。高 behavior = 全 dim が高い **genotype 極値** → random は中心極限で
  mean≈0.5 に固着し高 behavior に**到達不能** (corridor は genotype 内に隠れる)。
- **fitness profile along behavior**: 局所最適 (b=0.4, 値0.6) → **dip (b≈0.65, ≈0)** → 大域最適 (b=0.9, 値1.0)。
- **なぜ各手法がこうなるか**:
  - random: behavior が常に ≈0.5 → 大域 (b=0.9) に絶対届かない。
  - RR-hillclimb: b≈0.5 から局所 0.6 へ climb。dip 越えに downhill 必要だが (1+1) は downhill 拒否 →
    停滞。restart も fresh random は必ず b≈0.5 → 同じ罠。
  - panmictic-GA: 早期収束で局所。
  - **MAP-Elites: behavior grid を stepping-stone として保持** (dip cell も「新規 cell」として残す) →
    b を 0.5→0.9 へ ratchet し大域到達。**downhill (fitness dip) を跨げるのが diversity 維持の本質効果**。

これはユーザーの ③ 直感「集団内 分離 (niching) で選ぶ差を作る」の正準実現。novelty-search / MAP-Elites
(Lehman & Stanley 2011 / Mouret & Clune 2015) の deceptive-maze 系と同型。

## 3. 実験系列 (exp1-5) と各々の honest 教訓

| exp | landscape | 結果 | 教訓 |
|---|---|---|---|
| 1 | 2D 欺瞞 (広局所+狭大域) | MAP-E > panmictic-GA のみ。vs random/RR は有意差なし | **低次元+高予算は random が coverage で勝つ** |
| 2 | 高次元(D=20) 欺瞞 corner=behavior極値 | 全手法停滞 (MAP-E も corner 不到達) | behavior=高次元平均だと corner 到達に全 dim 整列必要+valley に勾配なく illumination も届かない |
| 3 | 可動behavior(gene0,1)×高次元alignment | **RR-hillclimb 圧勝** (90%到達), MAP-E 停滞 | behavior が直接設定可能だと **restart が競合**。MAP-E は cell 離散化で niche 内深層最適化できず |
| **4** | **deceptive corridor (behavior=mean, dip)** | **MAP-E のみ大域到達, 3 baseline 全敗** | **③ が立つ正準状態を発見** (genotypic corridor + dip) |
| 5 | exp4 ± dip (境界対照, 3 baseline) | dip あり=MAP-E が 3 baseline 全勝 / dip なし=優位消失 | **③優位は欺瞞 regime 限定** (MAP-E 万能でない) |

## 4. honest 留保 (overclaim しない)

- exp4 の deceptive corridor は **構築した synthetic landscape**。「③ が *可能* (存在証明)」であって
  「llcore の実 task が欺瞞的 corridor 構造を持つ」ことは**未証明**。手順 2 の copy task は単峰/連結で
  欺瞞的でなかった (= そこでは③不要)。
- 勝因は厳密には「**behavioral diversity 維持が fitness valley 跨ぎを可能にする** (illumination/novelty
  機構)」。これは広義の niching/③ (分離が選択を可能にする) だが、「差し survival 率の差」を単独で
  分離したわけではない。MAP-Elites は niching と selection の複合。
- 全実験 CPU・toy・noise σ=0.008 の低ノイズ。実 LLM fitness のノイズ/次元とは別。
- baseline の RR-hillclimb は **固定 sigma・固定 restart_patience・非悪化受理のみの素朴 (1+1)-ES**
  (step-size 適応 / multi-try / restart 間予算最適配分なし)。「妥当な restart baseline」であって
  tuned state-of-the-art optimizer ではない (Codex Medium 指摘)。ただし欺瞞 corridor は構造的に
  downhill 跨ぎを要するため、step-size 適応程度では突破できない (restart しても fresh random は
  必ず b≈0.5 の罠に戻る) — baseline の弱さでなく landscape の構造が分離要因。

## 5. ③ の将来性についての回答 (ユーザー問い)

**「③ に将来性が無い」は否定された。** ③ は **欺瞞的 corridor landscape で decisively load-bearing**
(存在証明済)。ただし **滑らか/単峰な landscape では不要** (hill-climbing で十分)。
→ 残る問いは「**実 substrate (実 LLM fitness / downstream task) が欺瞞的 corridor 構造を持つか**」。
これは手順 6 (小型 LLM で実 proxy の landscape 構造 sanity check) と GPU 投資判定に直結する。

## 6. 次のアクション候補 (ユーザー方向判断)

1. **実 substrate の欺瞞性測定** (手順 6): 小型 LLM (CPU) の loss landscape / downstream task が
   exp4 的な deceptive corridor を持つか測る。持てば ③/MAP-Elites が GPU 投資の正当化要因に。
2. **llcore 本流への MAP-Elites 配線**: behavioral niching を `evolution/` に load-bearing 実装
   (現状 research 隔離)。behavior descriptor の設計 (lineage 特性等) が鍵。
3. **deceptive corridor を実 task で実現**: multi-delay recall 等で genotypic corridor + dip を作れるか。

## 7. 手順 6 結果 — 実 substrate proxy は欺瞞的でない (2026-05-30, research/step6_real_proxy/)

> ★**2026-07-01 SUPERSESSION 注記**: 本節 exp7 を「**決定的**」とした n=8/6 の統計は、後続の統計検出力監査 (`research/statistical_power_audit/STATISTICAL_POWER_VERDICT.md` / `VERDICT_power_audit.md`、2026-06-01) で **underpowered と判定 → 撤回・n≥15 で再測必須**とされた (n≤10 は Cliff δ=+1.0・p<0.001 の強効果すら no-effect 化する構造的盲点)。**「決定的」という強い表現は無効。** ただし ③不成立 という *方向* 自体は翌日の `research/step_d_settle/THIRD_AXIS_SETTLE_VERDICT.md` (2026-06-02) が **決定論的測定 (EXP2)** で「landscape は真に滑らか/単峰 → n を増やしても多峰は出ない」と再確認しており、結論方向は維持される。要約: **exp7 の n=8/6 有意性を "決定的" として引用しないこと**。③慎重論の結論は後続で別基盤により支持されている。

§6 の次アクション 1 (実 substrate の欺瞞性測定) を実施。CPU 完結の実 proxy =
**ESN (固定 reservoir + ridge readout) × 実テキスト (llcore Python source 124K chars) の
next-char 予測**。gene = reservoir 力学パラメータ。= llcore thesis (dynamics gene 進化) の最近接 CPU 近似。

- **exp6 (landscape map)**: gene 空間 (spectral_radius×leak) の grid は **滑らかな broad ridge**
  (leak 単調増・rho 弱依存、max acc 0.293/range 0.083)。検出 "局所最適" は全て global 近傍 0.265-0.281 の
  **ノイズ水準凹凸**で、exp4 のような深い valley なし。
- **exp7 (決定的: 実 MAP-Elites vs baseline)**: heuristic を信用せず実測比較。
  - (A) 3-param ESN: MAP-Elites は 3 baseline 全てに**有意差なし** (vs RR diff=−0.0016 p=0.078)。③不成立。
  - (B) per-neuron leak (高次元 40-dim, dynamics gene 化): MAP-Elites は RR/panmictic に**負け**、
    pure random にのみ勝つ (exp5 の smooth 条件と同型)。③不成立。

**結論 (honest)**: **実テキスト proxy の landscape は滑らか/単峰で、exp4 の欺瞞 corridor は自然出現しない。**
→ この proxy では ③/MAP-Elites の追加価値は限定的 (hill-climbing/coverage で十分)。**③ のための GPU
投資の根拠は本 proxy 証拠では弱い**。
**留保**: これは reservoir+ridge proxy (固定力学+学習 readout) であり backprop 学習の full LLM とは別。
full LLM の損失 landscape が欺瞞的でない保証ではない (proxy の限界)。だが「dynamics gene 進化」の
最近接 CPU 近似が滑らかだった事実は、③ 路線への投資を慎重にすべき強い示唆。

## 関連
- `YOUGO_平易版.md` (用語集・図入り・非専門家向け)
- `STEP4_DESIGN_space_expansion_niching.md` (設計 + C1-C4 基準 + 案 A scout)
- `EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md` (手順 1-2)
- `research/step4_selection/` (selection_lab.py + exp1-5)
- [[project_llcore_init_2026_05_29]] / [[feedback_benchmark_honest_disclosure]] / [[feedback_codex_pair_review_for_llcore]]
