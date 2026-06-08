# Phase −1 VERDICT — width_grow × cert ε>0 両立帯 純数値 scan

**作成**: 2026-06-09 / **$0/CPU** / seed=20260609 / 実装: `phase_m1_coviability_scan.py` → `phase_m1_coviability_results.json`
**問い**: 「進化可能な LLM」FW(`EVOLVABLE_LLM_PLAN_2026_06_09.md` v2)の最初の被験 method VSOA の make-or-break — recurrent adapter(CoupledNDGene, n→n+1 width_grow)で「(a) 新 unit が既存出力を非自明に変える(相対 L2 ≥ τ=0.05)∧ (b) 全 row が cert sound」を両立する ε>0 帯は存在するか。
**規律**: honest-disclosure。強い negative ほど内訳を疑い、交絡(net2net 近似 / certifier 保守度)を分離して測った。proxy/単一 seed の留保を明記。

---

## 0. 一行 verdict

**width_grow の ε>0 両立帯は dead-on-arrival ではない。ただし *sound 緩和* certifier(cert_two)の下で・*小 n* でのみ開く。唯一スケールする最保守 certifier(cert_inf)の下では実質空。** → VSOA の make-or-break は「存立条件 width_grow 自体」(賭け1)ではなく、**「navigable な certifier を高次元へスケールできるか」(賭け2: 2^n 壁 / vertex-free)** に collapse する。これは体系化の中核所見(L3 inf-trap / sound 緩和は navigable / 次元の壁)の**構造成長レベルでの再現**。

---

## 1. 結果(seed=20260609, 40 base × 3 方向 × 2 mode, n_eps=40)

`change@ε_max` = soundness 境界 ε_max での既存出力の相対変化中央値(<τ=0.05 なら「sound 域では関数がほぼ動かない」)。

| n | mode | certifier | 両立帯率 | ε_max 中央 | change@ε_max 中央 | band 幅中央 |
|---|---|---|---|---|---|---|
| 4 | fresh | **cert_inf** | 0.033 | 0.154 | 0.0053 | 0.000 |
| 4 | fresh | **cert_two** | **0.575** | 0.692 | **0.0837** | 0.154 |
| 4 | net2net | cert_inf | 0.108 | 0.077 | 0.0061 | 0.000 |
| 4 | net2net | **cert_two** | **0.667** | 0.923 | **0.0873** | 0.385 |
| 8 | fresh | cert_inf | 0.000 | 0.231 | 0.0051 | 0.000 |
| 8 | fresh | **cert_two** | **0.350** | 0.654 | 0.0427 | 0.000 |
| 8 | net2net | cert_inf | 0.000 | 0.154 | 0.0062 | 0.000 |
| 8 | net2net | **cert_two** | **0.325** | 1.000 | 0.0388 | 0.000 |
| 16 | fresh | cert_inf | 0.000 | 0.154 | 0.0028 | 0.000 |
| 16 | net2net | cert_inf | 0.000 | 0.269 | 0.0046 | 0.000 |

(cert_two は 2^n 頂点 SVD ゆえ n≤8 のみ feasible、n=16 は cert_inf のみ。)

## 2. 確定した知見

1. **ti=1 支配 = 96.1% / 99.9% / 100%(n=4/8/16)** → **red-team F1 を実データ確認**。`_infnorm_sup` の sup は per-row の `ti=1` 端点でほぼ達成され box 幅(t_min)非依存。width_grow の脅威は「box 拡大」でなく「**新 column が既存行 abs-sum を増やす per-row 増**」。旧計画の数理 framing は誤り、per-row へ訂正済(計画 §⑥ F1)。

2. **cert_inf(唯一スケール・最保守)では両立帯は実質空**: 全 n で band≈0、change@ε_max < 0.7%。つまり cert_inf が admit する範囲では構造成長は関数をほぼ動かせない(sound 域に入った瞬間 unit が事実上死ぬ)。**= R-LLM L3「inf gate が探索を unigram に collapse / admit 0」の構造成長版**。

3. **cert_two(sound 緩和, 2^n)は小 n で navigable な両立帯を開く**: **n=4 で 57.5-66.7%**(change@ε_max ~8.4-8.7% で τ=5% を明確に超過、band 幅 0.15-0.39)。**n=8 で 32.5-35.0%**(change@ε_max ~3.9-4.3% で τ 直下=borderline、band 幅中央 0=狭い)。**= L3「sound 緩和(two/sdp)は feasible set が navigable で良い gene に到達」の構造成長版**。

4. **band は n と共に縮小し cert_two は scale しない**: n=4(robust)→ n=8(borderline・狭い)→ n=16(cert_two が 2^16 頂点で測定不能)。**= 次元の壁 / vertex-free B2 が n=16 で cert_inf に収束(体系化 §1.2)** と同型。

5. **net2net(関数保存 copy)は fresh をやや上回る**(n=4 cert_two: 0.667 vs 0.575、ε_max も大)。活性 unit を copy して新 state を駆動すると両立帯が広がる傾向 = 計画の Net2Net 採用は方向として正。

## 3. 解釈 — make-or-break の collapse 先

VSOA の width_grow は **存立条件 width_grow 自体(賭け1)では死なない**(cert_two・小 n で両立帯あり)。律速は **certifier の navigability と scalability のトレードオフ**:
- **navigable な certifier(cert_two/SDP)は 2^n で scale しない**。
- **scale する certifier(cert_inf)は navigable でない(両立帯を開かない)**。

→ VSOA の生死は **賭け2(2^n 壁を破る vertex-free sound certifier R-LLM-1 が cert_two 並みの navigability を高次元で保てるか)に collapse する**。体系化はその候補 B2 が n=16 で cert_inf に収束する(navigability を失う)ことを既に示しており、**現時点では「vertex-free が navigability を保ったまま scale する」証拠はない**。

**設計示唆(計画反映)**: VSOA の per-component block は **n を cert_two が affordable かつ両立帯が非自明な帯(n≤4-6、2^6=64 頂点)に小さく切る**のが現実解。n=8 で既に change@ε_max が τ 直下=borderline。「block を小 n に切る」設計制約(計画 §5.2)は本データで**定量的に裏付けられ**、かつ「cert_inf でなく cert_two を per-component gate に使う」修正が要る(cert_inf では band が開かないため)。

## 4. honest 留保(潰せていない交絡)

1. **proxy + τ=0.05**: 「非自明な進化価値」を「既存出力の相対 L2 ≥ 5%」で代理。真の「n+1 次元でしか表現できない関数獲得」の証明ではない。τ を動かせば率は動く(τ↓で band 拡大)。
2. **net2net 近似が不完全**: 新 unit に外部入力駆動を与えていない(V を identity に固定、入力 copy せず)。真の Net2Net(V 行も copy)なら新 state が O(1) になり両立帯はさらに広がりうる = 本 verdict は **net2net を過小評価寄り**。
3. **SDP 未測定**: cert_sdp(共通 P Lyapunov)は cert_two を strict に含む → 両立帯は cert_two 以上。小 n で SDP を測れば band はさらに広がるが、SDP も 2^n で scale しない(賭け2 は不変)。
4. **単一 seed・合成基質**: 実 SmolLM2 adapter でなく合成 CoupledNDGene。`max_input_abs=1.0` ハードコードの box 被覆も未較正(計画 §⑩ で SmolLM2 入力実測較正を必須化済)。

## 5. 計画への反映(実施済 → `EVOLVABLE_LLM_PLAN_2026_06_09.md`)

- §③ 数理判定: 「条件付き成立」→ **「dead-on-arrival でない・両立帯は cert_two/小 n でのみ・cert_inf では空」を実データで確定**。
- §⑥ F1: per-row 訂正を ti=1 支配 96-100% で裏付け。
- §⑩ 賭け1/賭け2: **make-or-break は賭け1 から賭け2(navigable cert の scale)へ collapse** を明記。
- §⑩ 設計: per-component n を **n≤4-6 + cert_two gate** に修正(cert_inf では band が開かないため per-component gate を cert_two に格上げ、small-n に限定)。
- §⑬ 留保1: 「両立帯 ε>0 未証明」→ **「cert_two・小 n で実証。scale が本丸の未解決」に更新**。
- 第一級 negative の評価資産化(eval-framework 主軸の最初の測定): 「verified width_grow の両立帯は certifier-gated — cert_two で navigable・cert_inf で空・n で縮小」は枠組みの最初の falsifiable な測定結果として記録。

正本データ = `phase_m1_coviability_results.json` / 実装 = `phase_m1_coviability_scan.py`。

---

## 6. F6 — block 間 coupling soundness scan(`phase_m1_coupling_scan.py` → `phase_m1_coupling_results.json`)

**問い**: 計画 §5.2 の per-block AND 合成(各 block 独立 cert_inf<1)は block 間 coupling(γ·C12, γ·C21)を未 certify。per-block AND が admit した構成で合成系の真 ρ が 1 を越える率は?(full_true_rho は box sample 最大=sup の下界ゆえ blind-spot は**下限**。)

| γ | n=4 blind-spot | n=8 blind-spot | 合成真ρ平均(n=8) | full cert_inf 救済率(n=8) |
|---|---|---|---|---|
| 0.0 | 0.000 | 0.000 | 0.951 | 1.000(sanity: 無結合は per-block=合成) |
| 0.5 | 0.007 | 0.000 | 0.954 | 0.000 |
| 1.0 | 0.340 | 0.367 | 1.000 | 0.000 |
| 1.5 | 0.800 | 0.953 | 1.168 | 0.000 |
| 2.0 | 0.953 | 1.000 | 1.399 | 0.000 |

**確定知見**:
1. **per-block AND は coupling 下で genuinely 不 sound** — γ≥1.0 で per-block admit 済の **34-100% が合成真 ρ≥1(実際は発散)**。**red-team F6 を実データ確認**。per-block AND は禁止必須(計画 §⑬-4 / North Star #2 を裏付け)。
2. **full cert_inf(sound 合成)は過保守** — γ=0.5(真 ρ=0.93=実際は収縮)でも per-block admit 済を **全 reject(救済率 0.000)**。sound な合成 gate を cert_inf で作ると navigable でない。
3. **width_grow と同じ navigability×scalability トレードオフが coupling でも再現** — navigable な合成 cert(cert_two/SDP on 2n)は **2^(2n) 頂点で width_grow より scale が悪化**。coupling は賭け2(2^n 壁)を**増幅**する。

**統合 verdict(width_grow + coupling)**: VSOA は **small-n・低 coupling・cert_two の regime で生存**(両立帯あり・per-block の代わりに full small-system cert)。だが (a) cert_two は 2^n で非スケール、(b) coupling は full-system 化で 2^(2n) に悪化。**唯一の活路 = navigability を保ったまま scale する vertex-free sound certifier**(体系化は候補 B2 が n=16 で cert_inf に収束=navigability 喪失を既示)。**現時点で「高次元で navigable かつ sound」な certifier の存在証拠はない = これが FW の最大の未解決の賭け**。

正本データ = `phase_m1_coupling_results.json` / 実装 = `phase_m1_coupling_scan.py`。
