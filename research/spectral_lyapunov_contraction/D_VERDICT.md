# Track D VERDICT — Tighter matrix contraction certificates: does the SOLVER (SDP) earn its keep? (2026-06-02)

**結論 (honest)**: **YES — here the solver finally earns its keep.** Unlike Tracks A/B/C (where Z3/SMT
was *decorative* because the ∞-norm contraction invariant is closed-form), the **SDP common-Lyapunov
certifier genuinely certifies a strictly larger set than any closed-form-ish certificate** on this n=2
coupled family. But the honest residual is large: **~33% of the truly-contractive (rho<1) genes are
caught by NO certifier** (∞-norm, 2-norm-vertex, or SDP) — they need joint-spectral-radius /
non-quadratic Lyapunov tools.

n=2 結合写像 `s' = decay⊙s + (1−decay)⊙tanh(W s + V x)`、状態 Jacobian
`J(t)=diag(decay)+diag((1−decay)⊙t)·W` は box `[t_min,1]²` 上で t に **アフィン**。
Track C と **同一の 3270 gene** (270 grid + 3000 random, seed 0)、同一経験 oracle (seed 777, n=6000 +
structured corners) で測定。cvxpy 1.9.1 を 1 回インストール (clarabel/SCS solver) し SDP 経路も完走。

## 用語(かみくだき)
- **∞-norm 証明器 (Track C baseline)**: 行ごとの絶対値和 < 1。閉形式・端点列挙。十分条件だが spectral radius に対し保守的。
- **2-norm vertex 証明器 (Track D 新規, ソルバ不要)**: 最大特異値 σ_max(J) は J の凸関数、J は t にアフィン ⇒ box の sup は**頂点**で達成。4 頂点の numpy SVD の max < 1 で健全に証明。
- **SDP-Lyapunov 証明器 (Track D 新規, ソルバ必須)**: 共通 P≻0 を 4 頂点で `JᵀPJ−P≺0` 充足するよう SDP で探す。P-重み付きノルムでの収縮 ⇒ rho<1。**恒等 P の特殊例が 2-norm**なので SDP ⊇ 2-norm。
- **要するに**: 「2-norm は ∞-norm が取りこぼした rho<1 の 21% を健全に救う(電卓レベル)。SDP はさらに *236-254 件* を救う = **ここで初めてソルバが closed-form を実質的に上回る**。ただし rho<1 のうち ~33% は誰も救えない」。

## Pre-registered gate 結果 (exp_d_results.json, 実測 / `tmin1` domain を主表記、`free01` 併記)

| gate | 命題 | 結果 (tmin1 / free01) |
|---|---|---|
| **D1 soundness** | 各証明器の admit は経験的に「その証明器が証明する metric」で非膨張 | **PASS (全 4: 2-norm×2 + SDP×2, false admit 0)**。2-norm: 599/598 admit, worst emp ‖J‖₂=**0.999936**。SDP: 855/834 admit, worst emp rho=**1.000000**/0.999724, worst P-norm gain=**1.000000**。vertex self-check 最大超過=**−4.18e-4** (≤0 = 健全) |
| **D2 tightness gain (2-norm over ∞-norm)** | 2-norm-vertex が ∞-norm reject の rho<1 gene を救う数 | **180/850 = 21.2%** (free01: 179/850 = 21.1%)。全体でも 180 (= rho<1 部分集合が全て) |
| **D3 solver earns keep (SDP over 2-norm)** | SDP が 2-norm-vertex reject の rho<1 gene を救う数 | **254 (free01: 236)**。`two_beats_sdp=0` (数学的整合: P=I が常に SDP の候補) ⇒ **ソルバは実質的に勝つ (>0)** |
| **D4 honest residual** | rho<1 だが ∞/2-norm/SDP の**いずれも** admit しない gene | **448/1363 = 32.9%** (free01: 472/1363 = 34.6%)。joint-spectral-radius / 非二次 Lyapunov 必須 |

母集団 (実測): n=3270, **emp_rho_lt1 = 1363**。
admit 数 (tmin1): ∞-norm **513** ⊂? — **非入れ子!** / 2-norm **599** / SDP **855**。

## 核心の honest 発見

1. **D3 = Tracks A/B/C との決定的な違い: ここで初めてソルバが load-bearing**。
   Track A/B (scalar) / C (coupled ∞-norm) では「不変量が閉形式 ⇒ Z3/SMT は decorative」だった。
   Track D の SDP-Lyapunov は **2-norm vertex enumeration (closed-form-ish) が reject する 254 gene
   (tmin1) を健全に証明** する。非恒等 P を見つける = 単一誘導ノルムでは表現できない収縮を捕捉。
   `two_beats_sdp=0` は数学的必然 (P=I が常に候補) で SDP ⊇ 2-norm を実測確認。**verifier 投資が
   SMT でなく SDP/LMI に向かうべきという Track C の予言 (C_VERDICT §3) を定量実証**。

2. **証明器は入れ子でなく相補的 (重要 honest nuance)**。
   `2-norm ⊆ SDP` は成立 (P=I が候補) だが **`∞-norm ⊄ 2-norm` かつ `∞-norm ⊄ SDP`**:
   ∞-norm は SDP すら取りこぼす **62 gene (tmin1)** を一意に証明する。∞-norm 収縮と二次 (P-ノルム)
   収縮は**互いを包含しない別の十分条件** (W が反対称寄りだと ‖J‖∞<1 でも ‖J‖₂≥1、その逆もある)。
   ⇒ **実運用は 3 証明器の UNION が最強**。3 者合わせて **917 gene** を admit (∞単独 513 の 1.79 倍)。

3. **2-norm は ∞-norm の保守性の 21% を電卓レベルで回収 (D2)**。
   Track C の「850 件 rho<1 過剰 reject」のうち **180 件 (21.2%)** は、対称寄り W で σ_max=rho が
   ∞-norm の行和より小さいため、4 頂点 SVD だけで健全に救える。**ソルバ不要の closed-form-ish 改善**。

4. **D4 = 大きな honest 残差 (~33%)**。rho<1 の 1363 gene 中 **448 件 (32.9%)** は ∞/2-norm/SDP の
   どれも admit しない。これらは achievable-t の変動が **共通二次 Lyapunov P を許さない switching
   system 的挙動** を生む (頂点ごとに最良 P が異なり共通化できない)。**次のフロンティア = joint
   spectral radius (JSR) / parameter-dependent (非共通) Lyapunov / 非二次証明器**。`free01` で残差が
   472 と大きいのは t-box が広く頂点が離れ共通 P がさらに困難なため (tmin1 の方が tight)。

## HONEST DISCLOSURE — 実行中に検出・修正した 2 つの自前バグ (消さず記録)

1. **D1 oracle の metric 取り違え (gate 仕様バグ)**: 初回 run で SDP の D1 が 685 件 false admit と出た。
   調査の結果これは SDP の不健全ではなく **gate の測定 metric が誤り**だった: 共通二次 Lyapunov 証明器は
   **P-重み付きノルム**での収縮 (⇒ rho<1) を保証するのであって**恒等 ‖J‖₂≤1 は保証しない**。当該 gene
   (例 idx=31: emp ‖J‖₂=1.31 だが emp rho=0.92) は全て真に rho<1。⇒ D1 oracle を証明器ごとに分離
   (2-norm→‖J‖₂; SDP→rho **かつ** 実 P での P-norm gain) に修正。PREREGISTRATION.md D1 に明記。
   **soundness self-check (vertex 法の凸性再検証) が gate として機能し、この取り違えを炙り出した**。

2. **SDP feasibility の ill-posed 化 (実装バグ)**: 初回の `minimize 0 s.t. P>>εI, …, P<<1e6 I` は
   LMI 系が **P について同次 (scale-invariant)** なため ill-posed で、conic solver が trivial P≈0 を返す
   or spurious infeasible を出した (idx=2681: σ_max≤0.804 なのに P=I があるはずが infeasible)。
   ⇒ `P >> I` でスケール固定 + `minimize trace(P)` の well-posed 定式に修正。修正後 `two_beats_sdp=0`
   (数学的整合) を実測確認。SDP-admit は誤 1280 → 正 855 に是正。

これらは [[feedback_benchmark_honest_disclosure]] の「変に良い数値は内訳を疑う」の実践: 初回の
「SDP が 1280 件 admit」は**バグ由来の過大評価**であり、修正後の 855 (うち 2-norm 超過 254) が真値。

## 留保 (honest)
- n=2・対角 V=I・[-1,1] box・mock kernel 限定。より大きな n / full kernel では (a) 頂点数 2^n の指数増、
  (b) ∞-norm と 2-norm の乖離拡大、(c) 共通 P 非存在 (D4 残差) の増加が見込まれ、SDP の相対価値と JSR
  の必要性が共に上がる見込み。
- 経験 oracle (emp ‖J‖∞/‖J‖₂/rho) は from-below sup (seed 777, n=6000+corners)。D1 は worst が
  0.999936/1.000000 と境界ギリギリだが false admit 0 (SOUND_TOL=1e-6)。境界 gene は証明器の strict
  `<1` が正しく弾いており、経験 sup が証明上界を超える違反は 0。
- SDP は cvxpy 既定 solver (clarabel) + 一部 SCS。`min_eig_margin` を solver 非依存に固有値再計算で
  検証 (P≻0 かつ全頂点 decrease LMI≻0 のときのみ certified)。`optimal_inaccurate` も再検証を通過した
  ものだけ admit。
- D4 残差 448 件の「真に共通二次 P が存在しない」ことの**証明**(SDP infeasible が数値でなく構造的)は
  本 track では未実施 (次段: dual infeasibility certificate / JSR 下界)。現状は「SDP が見つけられない」=
  certified=False の実測に留まる (honest)。
- Codex pair-review 未。push 未 (llcore remote 未作成)。

## 成果物 (research/spectral_lyapunov_contraction/)
`PREREGISTRATION.md` / `two_norm_vertex_certifier.py` (4 頂点 SVD, sound, self-check 同梱) /
`lyapunov_sdp_certifier.py` (cvxpy 共通 P vertex-LMI, well-posed, cvxpy 不在時 graceful degrade) /
`exp_d_runner.py` (3 証明器 × 3270 gene × 2 domain + D1-D4 gate) / `exp_d_results.json` (実測) / 本 `D_VERDICT.md`。

関連: `../coupled_z3_contraction/C_VERDICT.md` (Track C, 850 over-reject の出所) /
`../CPU_VERIFICATION_RESEARCH_2026-06-02_VERDICT.md` (Track A/B 統合) /
[[feedback_benchmark_honest_disclosure]] / [[project_llcore_init_2026_05_29]] (Stage 3b SDP backend plugin への直結)
