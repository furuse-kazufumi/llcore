# 経験が記憶になる 3 機構 — 統一 taxonomy + 文献接地 (2026-06-07)

R-endo viability PoC (発散基質×記憶形成3機構, [[VIABILITY_VERDICT.md]]) の結果を、文献接地ワークフロー
(6 agents, 一次検証) で taxonomy 化したもの。**実験 = `results_viability_ab.json` / 文献 = corpus paper
14/15/16/09 + evolution doc_0697 + WebFetch 一次確認**。

## 統一 taxonomy

統一軸 = **certificate (安全の証拠) が「いつ・どの認識論的根拠で」記憶形成を gate するか**。3 機構は競合でなく
**認識の信頼性が下がる順の階層** (sound → 修復 → empirical)。

| 軸 | ENDO (自己予見) | REVIVE (修復) | OBSERVE (社会的観察) |
|---|---|---|---|
| 記憶形成様式 | 境界を内的制約として **zero-shot** 記憶 | 死を「傷=安全化された自分」として残し catastrophe を越え運ぶ | 他個体の観察された死から kNN 風に経験的に学ぶ |
| 認識論的位置 | sound (証明可能・捏造不能) | sound 寄り (修復後も certificate 保持) | empirical (lossy・Goodhart 可能) |
| 最有力 precedent | **Gödel Machine** (Schmidhuber 2003) | **safe-set projection** (Wabersich 2020) / safe-region expansion (Berkenkamp 2017) | **Campbell BVSR vicarious selector** (1960) / DGM (2025) |
| 本研究の差別軸 | Gödel の任意効用証明を **ρ<1 という検証可能な狭い健全性に縮約** | repair 技術を **記憶保持 + sound certificate 裏打ち**で統合 (changeop/refinement が核) | DGM/SEAL と同型の empirical gate。死んで学ぶ + 不完全境界が構造的 lossy |
| PoC 死回避 (linear, κ 2x) | 死=8.9 / viol=0 | **死=0** (全個体修復) | 死=31.5 ≈ NONE=37.8 |

**統一原理**: Maynard Smith & Szathmáry (1995) の「major transition ごとに新調節レベル (各々の truth-teller)」
の ML 版。verifier / healing-oracle / empirical-gate は同一 viability 問題への 3 調節レベル。全アームが
**honest recognition に単一点で依存** (Pasteur)、認識崩壊の境界 = Goodhart-Campbell 法則で形式化。

## novelty (honest, 3 点に限定 — 各機構の発明は claim しない)

corpus paper 16 の正味判定をそのまま採用:
1. **統一軸**: 3 機構を「viability 脅威下で経験が記憶になる」同一問題への階層的回答として統合。
2. **空白象限**: gate-evolution × 形式保証 (ρ<1 fail-closed) × **LLM 認知核** の三つ組が既存研究に不在
   (corpus 全 80+ source / paper 16, WebFetch 範囲)。ただし「空白 = 未踏か無意味/不可能かは未決着」と明記。
3. **死コスト測定**: REVIVE(0) < ENDO < OBSERVE side-by-side の death-count 比較は agent-evolution 文献に無い。
   ただし **linear toy substrate** であり実 LLM スケールは未証明。

**Claimed**: sound (証明可能・捏造不能) >> empirical (lossy・Goodhart 可能) を、**進化する (固定でない) ρ**
の下で記憶形成3機構の統一軸で初めて比較・測定。**NOT claimed**: 各機構/observational learning/proof-gating の発明。

## preemption (各機構の先行, 一次検証済)

- **ENDO**: Gödel Machine (cs/0309048) / Lohmiller-Slotine 収縮 (Automatica 1998) / Neural Contraction Metrics
  (Tsukamoto 2021) / shielding (Alshiekh AAAI 2018)。
- **REVIVE**: safe-region expansion (Berkenkamp NeurIPS 2017) / safe-set projection (Wabersich 2020) / GeneRepair
  (2003) / barrier cert (Prajna 2004)。「記憶保持 + sound rollback」結合は外部 precedent 薄 = 内部寄与の核
  (SEVerA 2026 FGGM fallback が近接 parallel)。
- **OBSERVE**: Campbell BVSR (1960 Psych. Review) / DGM (2505.22954, 最鋭の対立軸) / Reflexion (2023)。
  「社会的」は cultural-evolution (Tomasello/Heyes) への**類推**で AI 直接 precedent ではない (誇張禁止)。
- **cross**: Goodhart/Campbell's law / 「Certified Training towards Empirical Robustness」(2410.01617, certified は
  universal だが empirical accuracy が下がる trade-off = PoC の death-count 順序の理論的裏付け)。

## llcore 核心差別化の強化

llcore = 「DGM が『証明は無理だから経験的に』と諦めた地点で、性質を**収縮 (ρ<1) に絞れば証明できる**」。
3 機構対比が 3 段で強める: (1) 競合 (OBSERVE=DGM/SEAL = 事後サンドボックス・外付け empirical) を**同一実験内**で
並べ「empirical gate がなぜ不十分か」を実証 (OBSERVE 死=43 > NONE=37 = 学習は no-evolution より死が多い)。
(2) ENDO 境界は定理の帰結=捏造不能 vs OBSERVE は Goodhart 可能 = **metric-immune** の核を補強。
(3) REVIVE は安定↔可塑性の TRIZ 矛盾の解 (reject でなく最小 blend で修復=certificate 保持+記憶保持) =
corpus paper 16 §最大リスク「ρ<1 が可塑性を殺さないか」への直接の構造的応答。

## 反証条件 (over-claim 防止, 最優先で潰す)

1. **空白象限**: SEVerA (2603.25111) / SGM (2510.10232) / Two-Gate (2510.04399) が「連続 homeostasis を収縮で
   gate」していれば novelty #2 棄却。各 gate 対象 (discrete task vs continuous dynamics) を要精査。
   **→ ✅ 精査済・棄却なし (2026-06-07, Workflow 6 agents = 一次 fetch×3 経路 + 敵対検証 3/3 uphold)**:
   - **SEVerA = partial**: Dafny で離散 FOL I/O contract (∀x: Φ(x)⇒Ψ(x,f(x))) を sound 検証 (Theorem 5.3) +
     fail-closed verified fallback (FGGM) を Search→Verify→Learn ループ内で保持 — **FGGM fallback は REVIVE の
     構造的 parallel**。contraction/Lyapunov/ρ<1/homeostasis は全文ゼロ。
   - **SGM = partial (最近接・最優先 must_cite)**: anytime-valid **統計** certificate (Hoeffding LCB>0 /
     e-value wealth W_t≥1/δ) で再帰 self-modification を fail-closed gate — だが対象は**離散 task 性能**
     (CIFAR/ImageNet acc, RL reward) で連続内部状態でなく、保証は statistical で deductive-sound でない。
     査読者対策: 差別化は「statistical vs sound-contraction」×「discrete-task vs continuous-dynamics」の 2 軸で明示。
   - **2510.04399 = none**: 実タイトル "On The Statistical Limits of Self-Improving Agents" (Wang et al.) —
     validation margin + VC capacity cap の PAC oracle inequality (離散仮説族)。最強の empirical-gate 対比。
   - **3 件とも must_cite** (各 1 行対比は Workflow 出力 `wthvr1be4` / 論文編入済みの文面を正とする)。
   **novelty #2 の「contraction × 連続 memory-core dynamics」象限は無傷。**
2. **sound>>empirical (最重要)**: death-count 順序が **linear toy のアーティファクト**の可能性。非線形/高次元/
   複数 seed で逆転、または ρ<1 が保守的すぎ可塑性 (記憶獲得能力) を殺し能力指標で OBSERVE に負ければ格下げ。
   = 安定 vs 可塑性 TRIZ 矛盾が解けないなら llcore 中心仮説の棄却。
   **→ ✅ 潰した (2026-06-07, `run_viability_robustness.py` 12/12 全成立)**: 2 軸 (死回避/記憶保存) は
   seed shift / κ 1.5-3.0 / dim 24 / hard_mem delay20 / 非線形 highgain の全 config で頑健、可塑性 gap
   最大 −0.007 (≥ −0.05)、viol 0。正本 = [[VIABILITY_VERDICT.md]] §追補②。
3. **収縮=安全 必要十分**: ρ<1 は発散を止めるが意味的/論理的危険を捕捉しない。gate 通過後に misevolution が
   起きれば「sound=安全」棄却 → shield/Approval Bus との二層合成の健全性証明が別途必要 (未着手)。
4. **REVIVE 独立性**: repair が reject+再合成と機能的に区別不能なら 3→2 機構に縮退。
5. **OBSERVE「社会的」**: 個体間情報伝達構造を PoC が実装していなければ「empirical boundary learning」に限定。

## 次手 (優先順) — 2026-06-07 更新

- ✅ **反証 #2 の潰し込み** (12/12, §反証条件 2 参照) — 助走版確定 run + factorial 2³ + META も完了
  (正本 = [[VIABILITY_VERDICT.md]] 追補②③④: sound 単独支配 / RO 相乗 / hedging=CONDITIONAL)。
- ✅ **安定 vs 可塑性 capability メトリクス** — robustness の H_capability に統合 (12/12, hard_mem 含む)。
- ✅ **近接 parallel 精査** (2026-06-07, §反証条件 1 参照) — 棄却なし、novelty #2 生存、3 件 must_cite。
- **[最優先] 論文編入**: 編入条件 (反証 #2 潰し + 近接 parallel 精査) が**両方充足** → 編入実行。
  編入形 = §future work or 新節 (3 機構 taxonomy + 2 軸結果 + SEVerA/SGM/2510.04399 対比 + ④ premise
  監視の問題提起)。over-claim 排除 (novelty 3 点限定) を維持。
- **記事**: 「死を越えて経験が記憶になる 3 つの道 (予見/修復/観察) と、なぜ証明できる安全が観察で学ぶ安全に
  勝るのか」を Campbell BVSR + Gödel machine 対比で (over-claim 注意点 4 件を反映)。

**結論**: 文献接地は健全で論文編入可 (条件付き)。リスクは novelty でなく **PoC スケール妥当性 (linear→実 LLM)**
と **安定 vs 可塑性両立** に集中。honest に limitation 節へ + 反証 #2 を実験で先に潰すのが筋。
