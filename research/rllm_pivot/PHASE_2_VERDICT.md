# Phase 2 VERDICT — H-discriminative(枠組み妥当性)+ capability terrain-bet(EXISTS/NULL/ARTIFACT)

**作成**: 2026-06-09 / **$0/CPU** / seed=20260609 / 前提: `PHASE_1_VERDICT.md`(Decision gate 1 PASS)+ `EVOLVABLE_LLM_PLAN_2026_06_09.md`(主軸=Verified-Plasticity Evaluation Framework)
**規律**: honest-disclosure。capability(進化が勾配に勝つ)と guarantee(証明付き安定)を混同しない。NULL は失敗でなく確証的 negative=研究成果。
**実装**: `phase2_discriminative.py` / `phase2_capability_terrain.py` → 各 `*_results.json`。

---

## 0. 一行 verdict

**Decision gate 2 = 枠組み妥当性 PASS / capability = NULL_TIE(進化の優位は未実証)。** 評価枠組みは 4 method を soundness で**明確に判別**(無 gate=危険・STABLE 風経験 gate=危険・sound cert=安全・Mamba 風=安全)。一方 capability 副線は、多峰かつ識別力ある地形で、**進化(MAP-Elites, gated/ungated)と gradient/random が held-out で統計的に区別不能**(全方向 4 条件 AND 不成立、ME vs gradient mean_diff=+0.028/p=0.39/sign_delta=0、逆向きも非有意=純粋な引き分け、n=20)。→ **「進化が capability で勾配に勝つ」という EXISTS は支持されず**(M3 の負と整合)。ただしこれは **非有意の引き分け = capability 優位の未実証**であって「進化が勾配に劣る」proof でも powered な等価性 proof でもない(absence of evidence ≠ evidence of absence)。**戦略含意は同じ: capability は売りにできず、枠組みの価値は GUARANTEE 側に確定**(capability封印が data で正当化)。bonus: ρ<1 gate は held-out では可塑性を有意に殺さない(ただし train 側では archive 探索を制約、§3/§4)。

---

## 1. H-discriminative — 枠組み判別力(`phase2_discriminative_results.json`, North Star #3)

収縮〜発散を跨ぐ gene 集団(n=6, 95 発散 / 305 収縮)で 4 method の admit を真 ρ=empirical_rho と突合。

| method | admit率 | false-admit(発散を通した数) | 発散中の false-admit 率 | 収縮の棄却率 |
|---|---|---|---|---|
| **none**(無 gate=負の対照) | 1.000 | **95** | 1.000 | 0.000 |
| **stable_exp**(STABLE 風経験 gate) | 0.963 | **80** | **0.842** | 0.000 |
| **cert_inf**(sound) | 0.225 | **0** | 0.000 | 0.705 |
| **cert_two**(sound) | 0.360 | **0** | 0.000 | 0.528 |
| **cert_sdp**(sound) | 0.728 | **0** | 0.000 | **0.046** |

mamba_synth(stable-by-construction, 0 発散)集団: 全 method admit 1.000 / 0 false-admit(正の対照=安全 family を誤棄却しない)。

**確定知見**:
1. **判別力 PASS(H3a)**: false-admit 順序 = **none 95 > stable_exp 80 > sound certs 0**。枠組みは「危険 / 経験的だが危険 / sound」を soundness で明確に分離して測れる。
2. **★STABLE 風経験 gate は発散 gene の 84% を false-admit** — kernel は tanh で常時有界ゆえ有限ホライズン観測では「摂動忘却したように見える」が真 ρ≥1(echo-state property 失敗)。**sound certificate でないと見抜けない**=「ラングトンの蟻」の幻([[reference_article_idea_inventory]] §2)を経験 gate は見抜けない。**verified-plasticity の存在意義そのもの**。
3. **正の対照 PASS(H3b)**: stable-by-construction 集団で cert_sdp の収縮棄却率 0.000 = 安全 family を誤棄却しない。
4. **cert_sdp が sound かつ最 navigable**(0 false-admit・収縮棄却 4.6% のみ)= small-n per-component gate の第一候補(Phase 1 知見と一致)。

---

## 2. capability terrain-bet — EXISTS/NULL/ARTIFACT(`phase2_capability_terrain_results.json`, F12/BG10)

synthetic 多峰地形(K=6 basin max-of-Gaussian、behavior 空間)で、ρ<1-gate 付き/無し MAP-Elites・gradient(有限差分)・gradient_strong(restart64=meta-gate)・random を **同予算 B=2000 train 評価**で走らせ、**held-out fitness**(別入力)を n=20 seed で paired 比較。honest_eval 4 条件 AND(diff>0 ∧ 片側 Wilcoxon p<0.05 ∧ n_seeds≥15 ∧ |paired_sign_delta|≥0.147)。

**前提検証(H-multimodal)**: theta 空間 (20 次元) で F9 find_basins → cluster 数 = **40**。ただし n_starts=40 で 40 cluster = 高次元で hillclimb が収束しきらず全 start が別 cluster に落ちた可能性が高く、**「40 個の真の局所最適」の証拠ではない**(F9 instrument は 8 次元・merge_radius=0.4 で校正済、20 次元・radius=0.5 へは未校正)。**頑健に言えるのは「多峰(>1)」まで**。地形識別力: random held-out 平均 = **0.489**(0.05-0.95=天井/床でない、discriminating)= capability 前提(多峰 + 識別力)は成立。

**held-out 平均**: random 0.489 / gradient 0.507 / gradient_strong 0.549 / **mapelites 0.535** / mapelites_gate 0.513(全て ~0.49-0.55 に密集)。

| 比較 | mean_diff | Wilcoxon p | paired_sign_delta | 4 条件 AND |
|---|---|---|---|---|
| ME vs gradient | +0.028 | 0.392 | +0.000 | **False** |
| ME vs gradient_strong(meta-gate) | −0.014 | 0.594 | — | **False** |
| gradient vs ME | −0.028 | 0.622 | — | **False** |
| gate vs ungate(ρ<1 が可塑性を殺すか) | −0.022 | 0.608 | — | NS |

**VERDICT = NULL_TIE**: **進化(MAP-Elites)は、真に多峰で識別力ある地形ですら、同予算の gradient/random を held-out で有意に上回れない**(どの向きも 4 条件 AND 不成立)。
- **= M3 の capability decisive NEGATIVE を、多峰地形(40 basin)+ proper power(n=20)+ held-out + 4 条件 AND で再確認**。「最尤 NULL」(計画 §2.1)の予測が data で確証された。
- meta-gate(BG10)は moot: ME が gradient を上回らないため ARTIFACT 判定の余地なし(NULL がクリーン)。
- **bonus(North Star #6 副次)**: ρ<1 gate は可塑性を**有意に殺さない**(gate vs ungate diff=−0.022, p=0.61, NS)→「contraction gate が可塑性を殺す」は本データで**不支持**。verified-plasticity の「soundness を課しても可塑性コストは有意でない」を弱く支持(ただし下記留保)。

---

## 3. Decision gate 2 統合判定

| 項目 | 結果 | 判定 |
|---|---|---|
| **H-discriminative**(枠組み妥当性, North Star #3) | none 95 > stable_exp 80 > certs 0、正対照 0 棄却 | **PASS** |
| **capability 副線**(F12, EXISTS/NULL/ARTIFACT) | 多峰・識別力地形で進化 ≈ 勾配 ≈ random(全方向 AND 不成立) | **NULL**(確証的 negative) |
| **gate 中立性**(North Star #6 副次) | ρ<1 gate は可塑性を有意に殺さない | gate-neutral(弱支持) |

**→ 枠組みは妥当(method を soundness で判別できる)、capability は data で NULL = 価値は GUARANTEE に確定。** これは脆い単一機構に賭けない (b) 主軸選択の正しさそのもの: **機構(進化)が capability を生まなくても、「枠組みの妥当性 + 測定された capability NEGATIVE + STABLE 風経験 gate の 84% 危険性」が第一級 deliverable として残る。**

---

## 4. honest 留保

1. **synthetic 地形であり実 SmolLM2-CE 損失地形ではない**。「多峰が保証された地形ですら進化が勝てない」clean probe。実 LLM CE terrain(heavier follow-up)で覆る確率は低いが未検証(実地形は単峰の可能性すらあり、その場合 capability はさらに立たない)。
2. **高分散**: per-seed held-out は 0.00-0.97 と大きく振れ、運(初期化・seed)が支配的。これ自体が #25 monoculture(遺伝的浮動)/ ラングトンの蟻(見かけの構造=ノイズ)の再現。NULL は「差が無い」であって「全 method が優秀」ではない(全 method が random 同等)。
3. **gate 中立性は capability flat な regime での観測**: どの method も random を大きく超えないため「可塑性」自体が強く行使されていない。gate が可塑性を殺さないのは「殺すべき可塑性が元々乏しい」可能性を排除できない(honest)。
4. **gradient は有限差分**(解析勾配でない)。実 LLM の解析勾配ならより強い可能性=NULL を過大評価する方向だが、多峰での cold-start 勾配の弱さは現実的。
5. **paired_sign_delta = net-win-fraction**(教科書 Cliff's delta でない、計画 §⑬整合)。

---

## 5. 次セッション候補

- **実 SmolLM2-CE 地形での capability 再測定**(F12 の本番、実 LLM adapter CE)。NULL を実地形で確証 or 反証。
- **framework 性(F8)**: 3 plug-point(GeneCodec/Objective/VerifierBackend)拡張性のテスト化 + topology 多様化の汎化 load-bearing(B-G1)。
- **Mamba 固有安定性 正対照**(SSM Jacobian Lyapunov)で base-level 判別。
- **consumer story + 動きで魅せるデモ(F11, 確認必要=ユーザー明示判断)**: 無 gate ρ→1.95 発散 vs gate ρ<1 リアルタイム可視化。
- **普及メタ記事**: 「verified-plasticity = ラングトンの蟻の幻(見かけの安定/進化)を sound cert で見抜く」= honest disclosure の集大成(STABLE 84% 危険 + capability NULL を題材)。

正本データ = `phase2_discriminative_results.json` / `phase2_capability_terrain_results.json` / 実装 = 各 `.py`。
