# Verified-Plasticity Evaluation Framework — 確立された枠組み (capstone)

**確立日**: 2026-06-10 / **規律**: honest-disclosure(capability と guarantee を決して混同しない) /
**正本**: 本 doc は研究弧 Phase −1→0→1→2 の最終統合。詳細データは各 `research/rllm_pivot/PHASE_*_VERDICT.md`、
設計経緯は `docs/EVOLVABLE_LLM_PLAN_2026_06_09.md` + `docs/SYSTEMATIZATION_2026_06_09.md`。

> **一行**: 実小型 LLM に後付けした **small-n verified recurrent adapter** の online 構造適応 method を入力に取り、
> 「**発散しない・収縮する(ρ<1 を sound に保つ)** か」を第一級指標に、6 装置の統計的厳密性ハーネスで **falsifiable に測り
> method 間で比較する評価枠組み**。脆い単一機構でなく **枠組み自体が deliverable**。

---

## 1. 何を確立したか(結論)

| 命題 | 結論 | 根拠 |
|---|---|---|
| **GUARANTEE が立つ** | ✅ 確立(small-n per-component 域) | 実 width_grow 手術下で 0 観測 false-admit(Phase 1)、4 method を soundness で判別(Phase 2 H-discriminative PASS)、base-level で安全/危険 base を分離(Mamba 正対照 PASS) |
| **CAPABILITY は立たない** | ✅ NEGATIVE(cross-terrain で一貫) | synthetic + 実 SmolLM2-CE 両地形で、**強い解析(torch exact)勾配が進化を上回る**(実地形 19/20 paired 有意 / synthetic も最高平均)。進化の見かけの勝ちは **finite-diff gradient の弱さの artifact** |
| **価値の所在** | **GUARANTEE 側に確定** | capability を売りにしない(M3 戒め)。「provably-stable online structural adaptation を測る再現可能で falsifiable な評価枠組み」そのものが価値 |

**∴ 「進化可能な LLM」= 進化が性能で勝つ FW ではなく、「online で構造適応しても発散・破滅的忘却しないことを sound に保証・測定する FW」として確立。**

---

## 2. 枠組みの背骨 = 6 装置(これ無しに「進化が本物」と主張する権利は成立しない)

事前登録→結果順 / Holm 連言 / アーティファクト規律 / 反証条項 / 自己検出力監査 / 反 over-claim critic。
この方法論層が **honest-disclosure を構造に持ち込む**。本研究で繰り返し機能した実例:
- **★Phase 2 capability の false-positive 排除**: 実地形で MAP-Elites が finite-diff gradient を 20/20 で上回り「進化が勝った」
  ように見えたが、**強い解析勾配 meta-gate で再検証 → ME の勝ちは弱ベースラインの artifact** と判明(`PHASE_2_VERDICT.md` §7.1)。
  meta-gate 無しなら capability EXISTS を誤結論していた。「勝った気になる前に内訳を疑う」が data で機能した中核事例。

## 3. 何を測るか(第一級指標 = North Star)

1. **成長操作下 soundness**: width_grow/branch_add 後も cert が false-admit を起こさない(Phase 1 PASS, 0 観測)。
2. **coupling-aware 合成 soundness**: per-block AND は coupling 下で不 sound(禁止)、full-system cert が必須(Phase 1 確認)。
3. **枠組み判別力**: 無 gate(危険)/ STABLE 風経験 gate(危険)/ sound cert(安全)/ Mamba(自明安全)を soundness で分離して測れる(Phase 2 PASS)。
4. **framework 性**: 3 plug-point(GeneCodec/Objective/VerifierBackend)の 1 オブジェクト差替拡張性(Phase 2 PASS)。多様性→汎化 の load-bearing は **NULL**(立たず)。
5. **capability verdict**: terrain で EXISTS/NULL/ARTIFACT を確定(Phase 2 = ARTIFACT+NEGATIVE)。

## 4. 被験 method(plug-in 比較対象)と判別結果

| method | 役割 | 発散 gene の false-admit | 判定 |
|---|---|---|---|
| 無 gate | 負の対照 | 100% (95/95) | 危険(枠組みが検出) |
| STABLE 風経験 gate | 既踏比較 | **84% (80/95)** | 危険(tanh 有界で「忘れたように見える」が真 ρ≥1=sound cert でないと見抜けない) |
| cert_inf / cert_two / cert_sdp | sound | **0%** | 安全。**cert_sdp が 0% 誤許可かつ収縮 4.6% のみ過剰棄却=sound かつ最 navigable** |
| Mamba-130M (固有安定) | 正の対照 | — (全層 λ_max≤0 自明 PASS) | 安全 by construction |

## 5. 証拠チェーン(Phase −1→2、各々敵対的検証 MAJOR 0)

- **Phase −1(純数値 scan)**: width_grow 両立帯は cert_two・小 n(n≤4-6)で実在、cert_inf では空。vertex-free B2 は n=8 で崩壊
  → **verified 構造進化は small-n per-component 限定**(高次元 navigable-scalable cert は不在=第一級 negative)。
- **Phase 0(実機校正)**: SmolLM2-135M を CPU frozen load → 実 hidden に small-n verified adapter、cert_two gate が load-bearing
  (certified-stable 1.000 vs no-gate 0.680)。安定性 = echo-state 摂動忘却(tanh 有界=ノルム発散でない)と確定。F9 多峰性 instrument 校正 PASS。
- **Phase 1(soundness 主役, Decision gate 1 = PASS)**: 実 Net2Net 構造手術下で 0 観測 false-admit、per-component gate は cert_two/sdp 必須、
  per-block AND は coupling 下禁止、small-n feasibility 0.013h≪30h。cert_sdp が最 navigable(2^n コストは不変)。
- **Phase 2(枠組み妥当性 + capability, Decision gate 2 = PASS/NEGATIVE)**: H-discriminative PASS、実 SmolLM2-CE capability ARTIFACT+NEGATIVE、
  F8 (b)PASS/(a)NULL、Mamba base-level 判別 PASS。価値 = GUARANTEE 確定。

## 6. 使い方(3 plug-point API)

実装 = `research/verified_evolution_sdp_gate/coupled_nd.py`(GeneCodec/Objective/VerifierBackend)+ `src/llcore/evolution/minimal_ga.py`(evolve)。
```python
codec    = CoupledNDGeneCodec(n)              # 基質(GeneCodec): 任意次元 n の coupled recurrent adapter
objective= RotationNDObjective(n=n)           # 方向(Objective): 差替可能な fitness
verifier = make_nd_verifier("sdp")            # gate(VerifierBackend): none/inf_norm/two_norm/sdp
# verifier.certifies(gene) が ρ<1 を sound に判定。各 plug-point は 1 オブジェクト差替で evolve に載る。
```
**gate 規律**: 安定 certificate を破らないことのみで gate し、**fitness(capability)では gate しない** = 適応は許すが発散・忘却は許さない homeostatic constraint。

## 7. honest 限界(over-claim 禁止)

- **small-n per-component 限定**: 高次元で「navigable かつ scalable な sound certifier」は現存せず(賭け2 negative)。VSOA の高次元 width_grow 成立保証なし。
- **capability は売れない**: 強い勾配が進化を上回る(cross-terrain)。進化の価値は離散トポロジー探索 + guarantee であって perplexity 改善でない。
- **adapter scope**: 「実 LLM」修飾は後付け adapter に限定(base 凍結)。tiny→実 LLM の transfer は未検証領域。
- **soundness は 0 観測 false-admit**(機械証明でなく empirical_rho from-below の強 consistency)。
- **普及ファネルは空白になりうる**: 評価枠組みは地味。consumer story(第一候補=llive 自己進化メモリ層を verified gate が守る)+ 需要側証拠は **ユーザー明示判断**待ち。

## 8. status & 次

**Phase −1→2 完遂(2026-06-10)。枠組み確立。** 次の autonomous 候補:
(a) 普及メタ記事「verified-plasticity = ラングトンの蟻の幻(経験は騙され sound cert が見抜く)」(honest-disclosure 集大成)。
(b) framework 論文化(本 capstone + 3 verdict を ICLR/NeurIPS 系 short paper へ)。
**ユーザー判断待ち**: (c) consumer story 確定 + 動きで魅せるデモの市場価値(`phase2_demo_gate_discrimination.svg` は技術成果物として完成)。

関連: `PHASE_M1_VERDICT.md` / `PHASE_1_VERDICT.md` / `PHASE_2_VERDICT.md` / `SYSTEMATIZATION_2026_06_09.md` / `EVOLVABLE_LLM_PLAN_2026_06_09.md`。
