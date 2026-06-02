# CPU Verification-Pillar Research — Synthesis Verdict (2026-06-02)

> Goal (user): 「CPU でできる範囲で llcore の研究を進めておいてください」。
> ③ arc は CPU で構造的に閉鎖済 (BG9) のため、本セッションは llcore の
> もう一方の pillar = **Verified (Z3 検証)** を CPU で前進させた。
> 2 軸を pre-registration + 実装 + **独立 red-team 敵対検証** で実施
> (workflow: `llcore-cpu-verification-research`, 4 agents)。
> 規律: `research/` 隔離・src 無改変 (248 tests 全 PASS, git 確認)・seed 固定・honest disclosure。

---

## Track A — Achievable-t Lipschitz refinement

**仮説 (A2)**: 既存 contraction certifier の free-t over-approx (t∈[0,1]) を
**達成可能 t∈[t_min,1]** (t_min = sech²(|mix|+|gate_str|)) に厳格化すれば、
健全性を保ったまま **認証される gene 集合が厳密に拡大**する。

**結果 (honest negative on the novelty claim)**:

| gate | 内容 | 判定 |
|---|---|---|
| A1 containment | free-certified ⊆ achievable-certified | **PASS** (違反 0 / 6096 gene) |
| **A2 strict gain** | achievable が free より多く認証 | **FAIL** (gain=0、認証集合がバイト同一) |
| A3a soundness | 全 admit gene で empirical_L ≤ 認証上界 | **PASS** (false admit 0 / worst excess 5.8e-11) |
| A3b monotone | L_achievable ≤ L_free | **PASS** (違反 0、max increase 0) |
| A4 honest | 残差 over-approx の特性化 | 報告 (中央値 gap ~3e-8) |

- **A2 が FALSIFIED = 構造的**: free と achievable は **J(1) 端点を共有**し、refinement が除くのは t=0 端点のみ。だが |J(0)|=|decay|≤1 (decay∈[0,1]) は free reject の原因に**なり得ない** → 集合は決して広がらない。red-team が 416,111 点探索 + 代数で独立確認 (gain は原理的に不可能)。
- **実際の価値 = bound VALUE が厳密に tighter** (32% の gene で、mean Δ0.052 / max 0.244、empirical にほぼ exact)。**集合拡大ではなく値の精度向上** = ranking / 収束率推定 / fitness-shaping 用途には有用、進化ゲートの admit/reject 用途には**ゼロ利得**。
- red-team verdict: **HEADLINE STANDS**、soundness 独立再現 (false admit 0)、A2 negative は honest かつ構造的。
- **"exact" の scope 限定 (重要)**: 対角スカラ写像 + [-1,1] box でのみ exact。非対角/ベクトル写像では full Jacobian operator norm が必要 = 本 refinement は直接適用不可。

成果物: `research/lipschitz_refinement/` (PREREGISTRATION + achievable_lipschitz.py + exp_a1_a2/a3/a4 + A_VERDICT + 3 JSON)。

---

## Track B — Verifier-gated evolution (Verified × Evolvable の接続)

**問い**: Z3 検証ゲートを GA の child-admission に挿入すると、進化が**見つけるもの**は変わるか / fitness の代償は?

**結果 (soundness + load-bearing は堅牢、cost-regime は underpowered)**:

| gate | 判定 | 観測 |
|---|---|---|
| Control: gated('none') == src evolve() | **PASS** | best/diversity curve がバイト一致 (real task, 5 seed)。src 無改変を間接保証 |
| **B2 contraction load-bearing** | **PASS (最強の結果)** | ungated GA は最終集団の **15.5–30.5% が非 contraction (empirical_L≥1) に drift**。ゲートは admit child を **0 違反**に |
| **B3 gate soundness** | **PASS** | **false admit 0 / 7200 admit child** (独立 checker でも 0)。50-resample cap 未到達 |
| state_norm gate | **NO-OP (honest null)** | clip box が構造的に \|s\|≤1 を強制 → ungated でも病理 0 (L=512)。**verified-but-vacuous** |
| B1 fitness cost | **underpowered (red-team が downgrade)** | COSTLY/FREE が reseed で符号反転 (seed1000s: copy_d0 −0.0056 p=.032 / seed2000s: +0.0056 p=.95)。N=20 では cost は seed の性質で、regime の性質でない |

- **科学的価値の核 = B2/B3**: contraction ゲートは **進化を実際に変える**(非 contraction gene 15-30% を排除) かつ **sound** = llcore の "Verified × Evolvable" 結合の実証。これが最も再現性の高い結果。
- **honest 訂正 (red-team)**: (1) **cost-regime map は seed artifact** = "COSTLY on easy / FREE on hard" は確立した regime でなく underpowered null。N≥100 が必要。(2) **Z3 は decorative**: contraction 判定は閉形式スカラ不等式 `max(|decay|,|decay+(1−decay)·gate_str|)<1` と **20000/20000 で完全一致** = Z3 は本スカラ regime で**追加の弁別力ゼロ**。「Z3 proof」は誇張、soundness は閉形式代数に由来。

成果物: `research/verified_evolution/` (PREREGISTRATION + gated_evolve.py + exp_b_runner.py + B_VERDICT + JSON)。

---

## 統合 insight (両 track の収束)

両 track が同じ構造的事実に収束する: **対角スカラ RWKV 写像では、既存の閉形式 contraction 不等式が既にほぼ最適** —
- Track A: achievable-t 厳格化は **集合を広げない** (free が既に端点最適)。
- Track B: Z3 ゲートは **閉形式不等式と完全一致** (Z3 の弁別力 = 0)。

→ **Z3 (SMT) の付加価値は本スカラ regime には無い**。Z3 が効くのは **非対角/結合写像** (operator norm / 連立制約 = 閉形式で書けない領域)。これは llcore の verifier 投資方針への含意: **スカラ kernel に Z3 を被せても閉形式以上は得られない。Z3 を正当化するには coupled-state / multi-kernel verifier が必要** ([[project_llcore_init]] の Stage 3b kernel plugin + verifier backend plugin と接続)。

**一方、CPU で genuinely 価値があった発見** = Track B の **B2 load-bearing**: 検証ゲートは進化の探索を測定可能に変える (非 contraction drift 15-30% を排除)。これは GPU 不要・実証済の "Verified × Evolvable" payoff であり、③ arc の negative とは独立した **positive な研究資産**。

## Track C 追記 (2026-06-02 着地) — coupled 写像でも Z3 は decorative、真の solver 価値は SDP/Lyapunov

n=2 結合写像 `s'=decay⊙s+(1−decay)⊙tanh(Ws+Vx)` の ∞-norm contraction を 3270 gene で独立検証
(`coupled_z3_contraction/C_VERDICT.md` + `redteam_results.json`)。
- **C1 soundness PASS** (513 admit / false admit 0 / worst 0.99986 / 60k heavy 0)。
- **C2 PASS = coupling-awareness は load-bearing**: 対角 scalar heuristic が誤 admit する真 expansive **1267 gene** (emp ∞-norm 1.00–2.76) を結合認識証明器が全 reject。
- **R3 = Z3 は依然 decorative**: Z3 vs 閉形式端点列挙 ∞-norm の不一致 **0/3270**。行 abs-sum が t に凸 → 端点最大 → 閉形式厳密。∞-norm box quantifier は行ごと 1-D 凸最大に分解 = ソルバ不要。
- **C3 = ∞-norm は保守的**: rho<1 の真 contraction を **850 件 over-reject** (median gap 0.477)。**厳密条件 rho(J)<1 は固有値/SDP/Lyapunov-LMI = SMT で書けない**。

**capstone 確定**: scalar (A/B) + coupled (C) の contraction 不変量は全て閉形式 → **Z3/SMT は本クラスで decorative**。
solver が真に効くのは **非閉形式不変量 (spectral radius / Lyapunov 安定性)** = **SDP** territory であって SMT ではない。

## 次のアクション候補 (CPU)

1. ✅ **DONE (Track C)**: coupled ∞-norm 検証 → 「coupling は効くが Z3 は依然不要」。→ **次 = Track D: SDP/Lyapunov contraction certifier** (rho<1 を cvxpy/SCS or Lyapunov P-matrix 探索で証明、C3 が取りこぼした 850 gene を certify)。**solver が初めて閉形式を超える領域 = verifier 投資の真の正当化テスト**。CPU 可 (SDP は CPU)。falsifiable: SDP が strictly more を sound に certify するか。
2. **Track B を N≥100 seed で再走** (cost-regime の underpowered を解消、または null を確定)。
3. **load-bearing ゲートを task-family 横断で検証** (現在 CopyTask 2 variant のみ)。

honest 留保: 全て smoke-CPU 規模 (gene 数千 / seed 20-30)。soundness/load-bearing/no-op は構造的で seed 非依存だが、cost と generality は未確立。push 未 (llcore remote 未作成 = 露出回避、ローカル commit のみ)。

関連: [[feedback_codex_pair_review_for_llcore]] / [[feedback_benchmark_honest_disclosure]] / [[feedback_external_ai_verify]] / [[project_llcore_init_2026_05_29]]
