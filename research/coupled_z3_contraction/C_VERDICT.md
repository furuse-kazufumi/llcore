# Track C VERDICT — Does Z3 earn its keep on a COUPLED (non-diagonal) state map? (2026-06-02)

**結論 (honest)**: **NO for Z3 specifically — YES for coupling-awareness.**
n=2 結合写像 `s' = decay⊙s + (1−decay)⊙tanh(W s + V x)` の状態方向 contraction を
誘導 ∞-norm `‖J‖_∞<1` で判定。3270 gene (270 grid + 3000 random) を独立 oracle
(red-team, seed 777/999, 60k heavy-confirm) で測定。**Track A/B の「Z3 は decorative」を
結合写像でも確認し、verifier の真価は coupling-awareness と非閉形式不変量(SDP)にあると確定**。

## 用語(かみくだき)
- **結合(coupled)写像**: 状態の各成分が互いに影響し合う（対角でない W）。RWKV 本流は対角(各成分独立)。
- **∞-norm 収縮**: 行ごとの「係数の絶対値の和」が全部 1 未満なら、写像は縮む(収縮)＝安定。
- **要するに**: 「対角だけ見る素朴判定は結合を見落として危険なものを通す。結合を見れば直る。だが Z3 は要らない（電卓レベルの式で出る）」。

## Pre-registered gate 結果 (redteam_results.json, 実測)

| gate | 命題 | 結果 |
|---|---|---|
| **C1 soundness** | ∞-norm 証明器が admit した gene は経験的に \|\|J\|\|_∞≤1 | **PASS**: 513 admit / **false admit 0** / worst admitted emp ∞-norm=**0.99986** / 60k heavy-confirm でも 0 |
| **C2 coupling value** | scalar 対角 heuristic が admit するが実は expansive な gene を coupling-aware が reject | **PASS**: **1267 gene** が scalar-admit-but-expansive (emp ∞-norm **1.00–2.76**)、coupling-aware が**全 reject** (z3_rejects_all=true) |
| **C3 conservativeness** | ∞-norm は十分条件 (rho<1 を保守的に reject) を特性化 | **保守的と確定**: rho<1 の真 contraction を **850 件 over-reject** (median(∞-norm−rho) gap **0.477**, max 1.998)。over-reject 850 件中 emp_∞<1 は 0 = ∞-norm 判定自体は正しい(rho に対し loose なだけ) |

補助 (red-team R3/R4):
- **R3 (Z3 の付加価値)**: Z3 判定 vs **閉形式端点列挙** ∞-norm = **不一致 0/3270** (+ near-boundary stress 0/8000)。
- **R4 (健全性)**: 閉形式上界 < 経験値 となる違反 **0** = sound over-approx。

## 核心の honest 発見

1. **coupling-awareness は load-bearing (C2)**: 対角 scalar heuristic (Track A/B の判定式) は off-diagonal W を無視するため、結合写像で **1267/3270 gene を誤 admit** (最大 emp ∞-norm 2.76 = 大きく発散)。行を見る ∞-norm 判定はこれを全て正しく reject。**「対角だけ」から「結合を見る」への拡張は実質的価値あり**。
2. **Z3 specifically は依然 decorative (R3)**: 各行 abs-sum `a_i(t_i)=|d_i+(1−d_i)t_i W_ii| + (1−d_i)t_i|W_ij|` は t_i について凸 (|affine|+線形増加) なので box 最大は端点 t_i∈{t_min,1} で達成。**4 行の閉形式端点列挙で ‖J‖_∞ を厳密計算でき、Z3 と 0 件不一致**。∞-norm box quantifier は行ごとに独立 1-D 凸最大に分解 = ソルバ不要。**Track A/B (スカラ) に続き結合でも Z3 は閉形式と co-equal**。
3. **真の SMT/SDP フロンティアは非閉形式不変量 (C3)**: ∞-norm は spectral radius rho に対し保守的 (850 件の真 contraction を取りこぼし、gap median 0.477)。**厳密条件 rho(J)<1 / 2-norm<1 は固有値・SDP・Lyapunov-LMI 問題 = 効率的線形実算術(SMT)で書けない**。verifier 投資が solver を正当化するのは**ここ (SDP/Lyapunov)** であって、∞-norm/scalar contraction では電卓で足りる。

## ③ arc 検証 pillar 全体への含意 (capstone)

Track A (achievable-t scalar) / B (gated evolution scalar) / C (coupled ∞-norm) を通して:
> **このクラスの contraction 不変量 (scalar 対角・coupled ∞-norm) は全て閉形式計算可能 → Z3/SMT は decorative。**

llcore の "Z3-gated evolution" 旗印は **honest に scope すべき**:
- **Z3 は正しいが load-bearing でない** (contraction 判定は閉形式)。verifier の実価値は (a) **coupling/構造 awareness** (素朴判定より広い不健全を捕捉) と (b) **非閉形式不変量への拡張**。
- **solver (SMT でなく SDP) が真に効くのは spectral/Lyapunov 安定性** (C3 の取りこぼした 850 件、rho<1 だが ∞-norm≥1)。次の verifier 投資はここを狙うべき。
- これは [[project_llcore_init_2026_05_29]] の Stage 3b kernel plugin + verifier backend plugin (SDP backend) の方向付けに直結。

## 留保 (honest)
- **実装 runner (exp_c_runner.py) は workflow 内で完走せず** (2500/3270 で turn 終了、exp_c_results.json 未生成)。本 verdict の数値は **red-team の独立 oracle (redteam_results.json, seed 777/999, raw map から再導出、impl helper 非再利用)** に基づく = より強い独立検証。exp_c_runner.py は再走で確認 JSON を生成可能 (per-gene ~0.25s で ~15min、要 vectorize で高速化)。
- n=2 結合・対角 mock kernel・[-1,1] box 限定。より大きな n / 非対角 full kernel では ∞-norm の保守性 (rho との gap) が拡大し SDP の相対価値が上がる見込み。
- Codex pair-review 未 (Track A/B 共々 systemic timeout、次回)。push 未 (llcore remote 未作成)。

## 成果物
`PREREGISTRATION.md` / `coupled_map.py` (写像+厳密 Jacobian+経験 oracle) / `z3_infnorm_certifier.py` /
`scalar_heuristic.py` (対角のみ) / `exp_c_runner.py` (要再走) / `redteam_independent.py` + `redteam_fast.py` +
`redteam_results.json` (**本 verdict の証拠**) / 本 `C_VERDICT.md`。

関連: [[feedback_benchmark_honest_disclosure]] / [[feedback_external_ai_verify]] / `CPU_VERIFICATION_RESEARCH_2026-06-02_VERDICT.md` (Track A/B 統合)
