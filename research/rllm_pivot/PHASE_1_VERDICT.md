# Phase 1 + Phase 0 残り VERDICT — 実構造手術 width_grow / 成長 soundness / coupling / feasibility / 多峰性 instrument / Mamba 正対照

**作成**: 2026-06-09 / **$0/CPU** (torch 2.12+cpu, transformers 5.10, **cvxpy 1.9 + clarabel 0.11 在**) / seed=20260609
**前提**: `EVOLVABLE_LLM_PLAN_2026_06_09.md` (v2, 主軸=Verified-Plasticity Evaluation Framework) + `PHASE_M1_VERDICT.md` (Phase −1)
**規律**: honest-disclosure。証明器の sound 性は数学で保証、本 verdict は **真 ρ = empirical_rho (eigenvalue from-below 一致オラクル)** で 0 観測 false-admit を反証探索した結果。proxy/留保を §7 に明記。

実装: `phase1_structural_surgery.py` (width_grow + gates) / `phase1_cert_soundness.py` / `phase1_growth_stress.py` / `phase1_coupling_stress.py` / `phase1_feasibility.py` / `phase0_multimodality_instrument.py` / `phase0_mamba_control.py` → 各 `*_results.json`。

---

## 0. 一行 verdict

**Decision gate 1 = PASS (small-n per-component 域)。** verified 構造進化の機構 (実 width_grow 手術 + 成長 gate + coupling-aware cert) は **成長操作下でも 0 観測 false-admit を保ち (sound)**、**cert_two/cert_sdp gate が小 n で非自明な進化価値を持つ admit を多数開き**、**small-n (n≤6) で計算上自明に feasible** (verified evolution ループ全体が 30h 予算の 0.04% = 0.013h)。**★Phase −1 の最大の honest 留保「SDP 未測定」を解消**: clarabel 在環境で cert_sdp を固定構造/成長/coupling の3面で初測定し、**cert_sdp が最も navigable な sound certifier (真ρ<1 集合の ~0.9-0.99 を admit、cert_inf ~0.2-0.4 / cert_two ~0.4-0.5 を圧倒)** と判明。ただし **cert_sdp/cert_two の 2^n 頂点コストは不変** (cert_two は n=12 で 1 cert=1.3 秒、n≥14 で予算外) = **Phase −1 の「navigable かつ scalable な certifier は存在しない / verified 構造進化は small-n per-component 限定」結論は堅持**(navigability 天井のみ SDP で大幅に上がった)。Phase 0 残りも完了: 多峰性 instrument は **校正 PASS** (多峰/単峰を決定論的に判別)、Mamba-130M 正対照は **CPU で viable + framework が第2 base へ portable**。

---

## 1. Phase 1.1 — 固定構造 cert soundness + 証明器格子 (`phase1_cert_soundness_results.json`)

収縮〜発散を跨ぐ gene を n={4,6,8} で各 600/600/400 個サンプルし、3 証明器の admit と真 ρ (empirical_rho 6000 sample) を測定。

| n | contracting(ρ<1) | (S1) false-admit inf/two/sdp | (S2) ∞-bound 違反 / min(sup−ρ) | (S3) coverage(ρ<1) inf/two/sdp | sdp が two 超 admit | inf∧¬sdp |
|---|---|---|---|---|---|---|
| 4 | 453/600 | **0 / 0 / 0** | 0 / **+0.0004** | 0.411 / 0.508 / **0.954** | 202 | 0 |
| 6 | 426/600 | **0 / 0 / 0** | 0 / **+0.0002** | 0.286 / 0.430 / **0.939** | 217 | 0 |
| 8 | 280/400 | **0 / 0 / 0** | 0 / **+0.0007** | 0.225 / 0.396 / **0.907** | 143 | 0 |

**確定知見**:
1. **(S1) 全証明器が 0 観測 false-admit** — cert_inf/cert_two に加え **cert_sdp の soundness を初確認** (admit gene で真 ρ≥1 ゼロ)。証明器の数学的 sound 性と一致。
2. **(S2) ∞-norm bound は健全かつタイト** — `infnorm_sup ≥ 真 ρ` が全 gene で成立 (違反 0)、最小マージン +0.0002〜0.0007 = ∞-bound は真 ρ をすぐ上から押さえる安価な sound 上界 (計画 Phase 1 step 1「cert_inf で ρ 上界を安く計算」を実証)。
3. **(S3) 証明器格子 = cert_sdp が圧倒的に navigable** — 真に収縮する gene のうち cert_inf は 22.5-41%・cert_two は 40-51% しか admit しないが **cert_sdp は 91-95% admit**。`two⊆sdp` (違反 0) は cert_sdp の cert_two fast-path (`coupled_nd.py:142`) により**構造的に保証**される (経験的発見でなく実装上自明=トートロジー)。`inf∧¬sdp`=0 (本 sample では cert_sdp が cert_inf も経験的に包含。理論的には ∞-norm と Lyapunov は非可比だが反例観測ゼロ)。

---

## 2. Phase 1.3 — per-row 成長 stress + 非自明性 AND (`phase1_growth_stress_results.json`, Decision gate 1 (3))

実 `width_grow` (Net2Net/fresh) で base を n→n+1 に成長させ、各 gate の **成長下 soundness (0 false-admit) ∧ 非自明な sound admit ≥1** を判定。1 セル=24 base×2 dir×2 mode×16 eps=1536 grown gene、真 ρ=empirical_rho(1500)。

| セル (base真ρ中央) | gate | admit率 | false-admit | 非自明∧sound admit | maxΔfunc | gate3 |
|---|---|---|---|---|---|---|
| n4_hr0 (0.802) | per_row/cert_inf | 0.126 | **0** | 15 | 0.107 | **PASS** |
| | cert_two | 0.255 | **0** | 162 | 0.203 | **PASS** |
| | cert_sdp | 0.724 | **0** | **724** | **0.554** | **PASS** |
| n4_hr2 (0.845) | per_row/cert_inf | 0.223 | **0** | 21 | 0.129 | **PASS** |
| | cert_two / cert_sdp | 0.336 / 0.758 | 0 / 0 | 168 / 733 | 0.229 / 0.531 | PASS / PASS |
| n6_hr0 (0.888) | **per_row/cert_inf** | 0.116 | **0** | **0** | 0.047 | **FAIL** |
| | cert_two | 0.281 | **0** | 115 | 0.152 | **PASS** |
| | cert_sdp | 0.755 | **0** | 673 | 0.383 | **PASS** |
| n6_hr2 (0.875) | per_row/cert_inf | 0.216 | **0** | 3 | 0.067 | **PASS** |
| | cert_two / cert_sdp | 0.329 / 0.771 | 0 / 0 | 114 / 675 | 0.167 / **0.681** | PASS / PASS |

**確定知見**:
1. **成長下 soundness = 全 16 (セル×gate) で 0 観測 false-admit** — width_grow 1 回ごとに証明器が ρ<1 を保つ (empirical_rho from-below オラクルでの 0 観測 = §7#2 の通り near-boundary を取りこぼしうる強い consistency 証拠であり絶対証明ではない)。成長ρ最大 1.85-2.21 (発散域) は正しく全 reject。**North Star #1 (成長操作下 0 観測 false-admit) を実構造手術で確認**。
2. **cheap gate (per_row/cert_inf) は sound だが小 n で trivial/脆弱** — **n6_hr0 (最保守 edge=headroom 0) で非自明 admit=0 (maxΔfunc 0.047<τ=0.05) → gate3 FAIL**。同 n=6 でも hr2 (headroom あり) は PASS だが**非自明 admit わずか 3・maxΔfunc 0.067 と τ=0.05 ギリギリ=cheap gate の navigability は脆い**。= Phase −1 の「cert_inf 両立帯 trivial (change@ε_max<1%)」を**実構造手術レベルで再現**。なお per_row と cert_inf は本 sample で全数値一致 (ti=1 端点支配ゆえ偶然一致; 両者は scalability が異なる別 gate で、per_row の scalable 独立性は本実験では独立検証できていない)。
3. **navigable gate (cert_two/cert_sdp) は全セル PASS** — cert_two は 114-168、cert_sdp は 673-733 の非自明 sound admit を開く (maxΔfunc 最大 0.68)。→ **計画 §⑩「per-component gate を cert_two/sdp に格上げ・small-n 限定」が data で正当化**。per_row 不変条件 (F1 訂正済 cheap-sound gate) は scalable だが trivial、cert_two/sdp は navigable だが 2^n = 賭け2 のトレードオフが成長手術レベルで再現。

---

## 3. Phase 1.4 — block 間 coupling soundness stress (`phase1_coupling_stress_results.json`, Decision gate 1 (4))

2 block を residual 結合 (`full W=[[W_A,γC_AB],[γC_BA,W_B]]`) し、**per-block AND の盲点** と **full-system cert の soundness/navigability** を真 ρ=empirical_rho(2500) で測定。

| nb (full) | γ | 真ρ平均 | per-block AND 盲点率 | full false-admit inf/two/sdp | full coverage(ρ<1) inf/two/**sdp** |
|---|---|---|---|---|---|
| 2 (4) | 0.0 | 0.872 | 0/80 | 0/0/0 | 1.00 / 0.78 / **1.00** |
| 2 (4) | 0.5 | 0.897 | **0.050** | 0/0/0 | 0.105 / 0.487 / **0.987** |
| 2 (4) | 1.0 | 0.928 | **0.237** | 0/0/0 | 0.016 / 0.082 / **0.984** |
| 2 (4) | 1.5 | 1.094 | **0.613** | 0/0/0 | 0.00 / 0.065 / **0.806** |
| 2 (4) | 2.0 | 1.201 | **0.800** | 0/0/0 | 0.00 / 0.00 / **0.625** |
| 3 (6) | 1.0 | 0.987 | **0.340** | 0/0/0 | 0.00 / 0.00 / **0.758** |
| 3 (6) | 2.0 | 1.346 | **0.960** | 0/0/0 | 0.00 / 0.00 / **0.500** |

**確定知見**:
1. **per-block AND は coupling 下で genuinely 不 sound** — γ≥1.0 で per-block admit 済の **24-34% (γ=1.0) 〜 80-96% (γ=2.0) が合成真 ρ≥1**。**red-team F6 / North Star #2 を実構成+真 ρ で再確認** → per-block AND 禁止確定。
2. **full-system cert (inf/two/sdp) は全 γ で 0 false-admit = sound** — coupling 込み full-system cert なら sound (North Star #2 充足)。
3. **★cert_sdp が coupled 系で圧倒的に navigable** — **nb=2 (full=4)** では γ=0.5-1.0 で full cert_inf 1-10%・cert_two 8-49% に対し **cert_sdp 98% 救済** (γ=2.0 でも 62%)。**nb=3 (full=6)** では navigability が落ち γ=1.0 で inf=0%・two=0%・**sdp=75.8%**、γ=2.0 で sdp=50% (inf/two は 0%)。→ **cert_sdp は coupled 系でも最 navigable だが、次元 (nb 2→3) と coupling 強度で coverage は低下** (inf/two は n=6 coupled で完全無力 0%)。SDP/Lyapunov が ∞-norm/2-norm の過保守 (回転・非正規収縮の取りこぼし) を解消する一方、次元の壁は SDP でも効く。**coupled_nd.py docstring の仮説「SDP 優位」を coupling 下で実証**。
   - ⚠ nb=3 で clarabel が "Solution may be inaccurate" 警告を数件出した。**独立 eigen 再検査で soundness は保証 (full sdp false-admit=0)**。coverage の数値は近似解由来の僅かな揺れを含みうる (honest 留保)。

---

## 4. Phase 1.5 — feasibility 実測 (`phase1_feasibility_results.json`, Decision gate 1 (5))

per-op wall-time (μs, CPU 単独実行) と 30h 予算外挿。

| n | mutate | cert_inf | cert_two | cert_sdp | fitness | width_grow |
|---|---|---|---|---|---|---|
| 4 | 2.9 | 38 | 98 | ~39000* | 668 | 7.7 |
| 6 | 3.1 | 47 | 237 | ~200000* | 672 | 8.3 |
| 8 | 3.6 | 60 | **8499** | 8836* | 678 | 8.1 |
| 10 | 5.6 | 65 | **37947** | (2^n 外挿) | 685 | 7.7 |
| 12 | 4.1 | 84 | **1306587** (=1.3s) | (外挿) | 1107 | 10.0 |
| 14 | 5.8 | 119 | (2^n 外挿) | (外挿) | 871 | 11.6 |

(*cert_sdp は cert_two fast-path のヒット有無+cvxpy solve で gene 依存・非単調=honest 留保。genuine solve 時 数十-数百 ms。)

**予算外挿** (pop=64 × gens=200 × blocks=4, cert_two gate, budget=30h):

| n | per-eval | 総時間 | 30h 収まる |
|---|---|---|---|
| 4 | 769μs | **0.011h** | ✅ |
| 6 | 912μs | **0.013h** | ✅ |
| 8 | 9.2ms | 0.131h | ✅ |
| 10 | 38.6ms | 0.550h | ✅ |
| 12 | 1.31s | **18.6h** | ✅ (辛うじて) |
| 14 | — | (cert_two 2^14 外挿=infeasible) | ❌ |

archive (4096 cells) メモリ = 0.6-6.6MB (無視可)。

**確定知見**:
1. **small-n per-component (n≤6) は計算上自明に feasible** — ループ主要 op (mutate+cert_two+fitness) が **0.011-0.013h ≪ 30h** (予算の 0.04%; width_grow は μs オーダーで外挿に未算入だが影響無視可)。cert/mutate/width_grow は μs オーダー、本外挿の dominant は **fitness 項 (~0.7ms)**。⚠ ただしこの fitness は `RotationNDObjective` の**合成 adapter proxy** であり、実 GPU 訓練では **base forward (CE) が dominant** になる (本 proxy は実 CE を過小に見せる; §7.7 / 5.3 参照)。外挿は per-eval ごとに cert を 1 回課金する**保守的上限**見積り (実際の cert は構造成長時のみ走る)。
2. **2^n 壁は n≥10-12 で binding** — cert_two が n=8 で 8.5ms、n=12 で **1.3 秒/cert** (=18.6h, ただし少数反復 rc=15 由来の点推定でマージン薄)、n=14 で予算外 (cert_two 2^14 未測定=外挿 infeasible)。cert_inf は多項式 (38→119μs) で scale するが §2-3 通り navigable でない。**= feasibility 面からも small-n per-component 限定を裏付け**。
3. **CPU→GPU 外挿の前提**: cert/mutate は numpy (CPU-bound) ゆえ GPU 環境でも同等コスト、fitness のみ GPU で base forward (CE) に置換される。この置換が per-eval を増やす方向 = small-n feasibility 結論 (≪30h) は保たれる見込みだが、実 GPU 実測は Phase 2 で要確認 (honest 留保)。

---

## 5. Phase 0 残り

### 5.1 F9 多峰性 instrument 校正 (`phase0_multimodality_results.json`) — **校正 PASS**

決定論的 eval (eval_noise=0) + **距離クラスタリング basin 検出** + basin 間 valley 検定。

| field | n_basins | valley_fraction(basin間) | 判定 |
|---|---|---|---|
| multimodal_pos (6 Gaussian) | **5.50±0.50** | **1.000** | 多峰 ✓ |
| unimodal_gauss_neg | 1.00±0.00 | 0.000 | 単峰 ✓ |
| quadratic_bowl_neg | 1.00±0.00 | 0.000 | 単峰 ✓ |

決定論性: 同 seed で basin 数完全一致。**多峰 (n_basins=5.5, basin 間障壁 100%) を単峰 (n_basins=1) と明確に判別 → 校正 PASS**。
- 教訓 (校正中の修正): 初版の valley_fraction は「最高峰 1 つ」に anchor を集中させ谷を跨がず判別不能 (多峰 valley=0.002)、かつ grid 丸めで単峰 Gaussian の near-top 計数が膨張。**判別の主軸は距離クラスタ basin 数**と判明し instrument を再設計。→ Phase 2 terrain-bet で実損失地形へ適用する土台が確立 (本 Phase は校正のみ)。

### 5.2 Mamba-130M 正対照 (`phase0_mamba_results.json`) — viable + framework portable

- **Mamba-130M を CPU (HF slow path) で load 成功** (3.7s cached) + coherent 生成 ("The capital of France is in the city of Paris.") = 実 LM・第2 base。
- **framework が第2 base へ portable**: Mamba hidden 上で cert_two gate = admit 0.023 (**= 300 中 7 gene のみ**)/ certified-stable率 1.000 / false-admit 0、no-gate 0.713 → **gate load-bearing +0.287** (SmolLM2 の +0.320 と整合) = F8 plug-point「新 base 載せ替え」を実証。
- ⚠ honest (敵対的検証反映): (1) §5.2 の soundness オラクルは §1-4 の empirical_rho (固有値 from-below 数千 sample) ではなく **perturbation_forgetting (単一摂動・1 軌道対) という弱オラクル**。検出力が §1-4 より明確に低い。 (2) certified-stable率 1.000 と false-admit 0 は**同一変数由来の冗長表記** (独立 2 根拠でない)、かつ **admit n=7 の小集団**ゆえ「sound」と断ずる統計的検出力は低い。 (3) base-level の stable-by-construction (非正 Lyapunov 指数) は未検証 (gate は adapter に掛かり base 自体でない)。intrinsic hidden 摂動忘却 proxy (rel-step 中央 0.801) は判別力弱 → **Mamba 固有安定性の正対照は SSM Jacobian 測定として Phase 2 へ defer**。本 Phase の deliverable は「**framework portability + Mamba CPU 動作確認**」に限定 (固有安定性正対照ではない)。

---

## 6. Decision gate 1 統合判定

| gate | 条件 | 結果 | 判定 |
|---|---|---|---|
| (3) 成長下 soundness ∧ 非自明 admit≥1 | width_grow N 回で false-admit=0 ∧ 非自明 sound admit≥1 | 全セル false-admit=0、cert_two/sdp 全セル PASS (cheap gate は n=6 で trivial→cert_two/sdp 必須) | **PASS** |
| (4) coupling-aware 合成 soundness | per-block AND 禁止 + full cert sound | per-block AND 盲点 24-96% (不sound) → 禁止、full cert (inf/two/sdp) 0 false-admit | **PASS** |
| (5) feasibility | small-n ループが 30h 予算に収まる | n≤6 で 0.013h、n≤10 で <1h、2^n 壁は n≥12 | **PASS** (small-n) |

**→ Decision gate 1 = PASS → Phase 2 (small-n per-component 域、Phase −1 確定の制約内)**。

枠組み (eval-framework 主軸) の Phase 1 deliverable = **「sound・feasible な small-n verified 構造適応の測定ハーネス + 証明器格子 (inf/two/sdp) の完全 characterization」**。被験 method VSOA は small-n per-component で生存可能と data-grounded に確認。

---

## 7. honest 留保 (潰せていないもの)

1. **2^n scalability 壁は不変 (賭け2)**: cert_sdp で navigability 天井は ~0.9 に上がったが (Phase −1 の cert_two ~0.45 から大幅改善)、cert_sdp/cert_two の **2^n 頂点コストは不変** (cert_two n=12=1.3s、cert_sdp は cvxpy で更に高コスト)。**「navigable かつ scalable な sound certifier は依然不在」= verified 構造進化 high-dim 不成立は堅持**。SDP は天井を上げただけで壁は破っていない。
2. **empirical_rho は from-below 推定**: false-admit (admit∧真ρ≥1) は **下限** の反証探索 (証明器の sound 性は数学で保証済、本測定はその consistency 監査)。0 観測は強い consistency だが「全 (s,x) で ρ<1」の証明ではない。
3. **net2net は incoming-copy 近似** (exact function-preserving でない; 新 unit に外部入力なしゆえ。module docstring 留保) → 関数変化 Δfunc は近似評価。
4. **cert_sdp の clarabel 近似解**: nb=3 で "inaccurate" 警告。独立 eigen 再検査で soundness 保証 (false-admit=0) だが coverage 数値に僅かな揺れ。
5. **Mamba 固有安定性 (正対照の核) は未測定**: gate は adapter に掛かり Mamba base の Lyapunov は未検証 → Phase 2 defer。
6. **多峰性 instrument は連続 proxy 空間で校正済**、実離散トポロジー地形への適用 (terrain-bet) は Phase 2。校正 ≠ 適用。
7. **fitness は RotationNDObjective (合成)**: 実 SmolLM2 CE での capability 副線 (EXISTS/NULL/ARTIFACT) は Phase 2 必須。

---

## 8. 次セッション候補 (Phase 2 / 普及)

- **Phase 2 H-discriminative**: 4 method (VSOA cert_sdp-gate / 無 gate / STABLE 風 / Mamba) を枠組みで比較し事前登録検定。
- **capability 副線 (必須, F12)**: 校正済多峰性 instrument を **実損失地形** に当て terrain EXISTS/NULL/ARTIFACT を proper power で 1 つ確定 (MAP-Elites vs gradient on multimodal terrain)。
- **Mamba 固有安定性正対照**: SSM Jacobian の Lyapunov 測定で「Mamba 自明 PASS / SmolLM2 reject 発生」を base-level で示す。
- **gate を cert_sdp に**: 本 Phase で cert_sdp が最 navigable と判明 → small-n per-component gate の第一候補は cert_sdp (n≤6 で cvxpy コスト許容)、scalable 近似が要る場面は cert_two。
- **consumer story + 動きで魅せるデモ (F11)**: 無 gate ρ→1.95 発散 vs gate ρ<1 のリアルタイム可視化 = SNS 拡散素材 (確認必要: consumer story 確定 / 需要側証拠はユーザー明示判断)。

正本データ = 各 `phase1_*_results.json` / `phase0_*_results.json` / 実装 = 各 `.py`。
