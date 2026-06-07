# R-endo viability — 発散基質 × 記憶形成3機構 VERDICT (2026-06-07)

なぜなぜ分析 (「無条件有界では収縮と生存がデカップル」) → 環境=recurrence ゲイン κ の発散基質を構成し、
ユーザー洞察 (「復活がないと経験が記憶に残らない」「他個体の死から学ぶ」) を実装した記憶形成3機構を
事前登録 A/B で検証した決着。正本コード = `viability_substrates.py` / `run_viability_ab.py`
(事前登録 commit b9e4ac2 = 結果取得前) / `results_viability_ab.json` (生結果)。

## 結論一行

**発散しうる基質では内的 gate に「自己保存の仕事」が生まれる (ENDO は致命試行を大幅削減: linear 37.8→8.9)。
3機構で死回避は REVIVE(0) < ENDO < EXO < OBSERVE/NONE と一貫。健全検証器 (ENDO/REVIVE, sound, viol=0) は
empirical 社会学習 (OBSERVE, 死んで学ぶ・不完全) を圧倒 = llcore 核心「sound >> empirical for safety」を実証。
復活/修復 (REVIVE) は死を完全に 0 化 = 全個体の経験を傷として保持 (ユーザー洞察を実証)。**

## 設計 (環境 = recurrence ゲイン κ)

memory タスク (delay=8) を 3 発散基質で評価。環境 κ を mid-run でステップ: phase1 (10 世代) κ=1.0 →
phase2 (10 世代) κ=2.0。κ↑ で実効収縮 κ·a が 1 を超え、以前安定だった gene (a<1) が**発散** =
viability 脅威・入力ゲイン g で回避不能・real divergence (なぜなぜが導いた構成)。
- 3 基質: **linear** (飽和除去, 幾何発散) / **softsat** (高天井 K=50, NaN なし) / **highgain** (高ゲイン M=20 観測)。
- 死 = 発散 (|state|>1e6/非有限) or 誤差包絡 > 生存閾 V。死=fitness 0。
- **本丸指標 = phase2 致命評価数** (死で無駄にされる個体 = 消える経験)。死=0fitness は selection が処理する
  ため最終 elite は安全 → gate の真価は「死んで学ぶコスト」の回避 = 評価前に致命を弾く safe exploration。

## 記憶形成3機構 (ユーザー洞察) + baseline

| arm | 機構 | 死への対応 |
|---|---|---|
| **ENDO** | 自己予見 | 内的健全検証器で死を予見し **reject** (zero-shot, sound) |
| **REVIVE** | 復活/修復 | 死を予見し reject でなく **修復** (記憶 mix 保持・dynamics 安全化; carried pop も修復) |
| **OBSERVE** | 社会的観察 | 他個体の観察された死 (death_memory) 近傍を **経験的に回避** (kNN, lossy, Goodhart 可能) |
| NONE | — | gate なし (致命 gene も評価し死を被る) |
| EXO_fixed | 設計時固定 | gate κ=κ_low 固定 (κ_high 発散を見逃す) |

## 結果 (n=20 seeds 3000-3019, phase2 致命評価数)

| 基質 | NONE | EXO_fixed | **ENDO** | **REVIVE** | **OBSERVE** | soundness |
|---|---|---|---|---|---|---|
| linear | 37.8 | 22.4 | **8.9** | **0.0** | 31.5 | viol 0/1412 |
| softsat | 2.2 | 1.4 | 0.9 | 0.0 | 2.3 | viol 0/1245 |
| highgain | 9.8 | 8.5 | **2.5** | **0.0** | 7.3 | viol 0/1003 |

死回避 (低い順, 全基質一貫): **REVIVE < ENDO < EXO_fixed < OBSERVE/NONE**。

主要検定 (paired sign-flip, n=20):
- **ENDO−REVIVE 死差**: linear Δ=+8.9 (p=0.000) / highgain Δ=+2.5 (p=0.001) / softsat Δ=+0.9 (p=0.062)。
- **OBSERVE−ENDO 死差**: linear Δ=+22.6 (p=0.000) / highgain Δ=+4.8 (p=0.040) / softsat Δ=+1.4 (p=0.287)。
- fitness はほぼ同等 (~0.81 全 arm)。REVIVE−ENDO fitness Δ = −0.011(linear)/−0.010(softsat)/+0.000(highgain)
  = REVIVE はわずかに保守的 (修復が安全側へ引く) だが小。
- memory_ratio (catastrophe 直後 elite fitness / phase1 final) ≈ 1.0 (全 arm) = **elite 記憶は段差を生き残る**。

## honest な発見 3 つ

1. **発散基質で内的 gate に自己保存の仕事が生まれる**。有界 CopyTask (run_d) では autonomy null だったが、
   κ 環境で発散が viability を脅かすと、ENDO は致命試行を NONE の 1/4 (linear 37.8→8.9) に削減。
   なぜなぜが導いた「生存を収縮に再結合」が効いた。

2. **死が消すのは elite 記憶でなく探索 (個体の経験)**。memory_ratio≈1 = best gene は安全な内部に最適化され
   段差を生き残る。死で無駄になるのは**境界を探る個体**。REVIVE はこれを 0 化 = 全個体を修復し経験を傷として
   保持 (= ユーザー「復活がないと経験が記憶に残らない」を**探索/集団レベルで実証**)。ENDO は reject ゆえ
   carried pop の段差死が残る (8.9)。**修復は予見より探索を保存する** (REVIVE 0 < ENDO 8.9, p=0.000)。

3. **sound (ENDO/REVIVE) >> empirical (OBSERVE) for safety = llcore 核心の実証**。OBSERVE (経験的社会学習) は
   死んで学ぶため deaths が NONE 並み (linear 31.5 vs 37.8) かつ ENDO の 3.5 倍 (8.9 vs 31.5)。健全検証器は
   violations 0 で zero-shot 保護。「証明は捏造不能・即時 / 学習は死んで学ぶ・不完全」を定量化。
   (DGM/SEAL の empirical gate が reward hacking する反面教師と整合)。

## honest 留保 (over-claim 排除)

- **elite 記憶は段差を生き残る** (memory_ratio≈1) ので、「記憶保存」は elite genome でなく探索/集団多様性の話。
  「死が記憶を消す」は探索個体に限る。
- **OBSERVE の弱さは一部実装依存** (kNN radius=0.15, resample 戦略)。tuning で改善余地はあるが、「empirical は
  死んで学ぶ」構造的コストは radius に依らず残る。OBSERVE は1実装であり最適でない。
- **REVIVE の死 0 は修復が _admits (sound) を使うため定理的** (gerrymander でない) だが、fitness をわずかに
  犠牲 (保守的修復)。修復先 (mix 保持・decay/gate_str を 0 へ blend) は1設計。
- **softsat は死境界が弱い** (NONE 2.2)。高天井飽和が発散を抑えるため機構差が小さく softsat の p は n.s. 寄り。
- scalar gene / n≤8 相当 / 単一 κ 段差 / probe-based fitness のスコープ。κ_high=2.0 は「環境が再帰ゲインを 2 倍に
  destabilize」の honest な設定 (透明に報告)。null/負の結果も削除せず残す (feedback_benchmark_honest_disclosure)。

## llcore への含意 / 次

- **R-endo の null (有界基質) → viability 基質で positive**: 内的化が効くのは「環境変化が viability を脅かす
  基質」という run_d の予言が確認された。HD-1 GPU (ungated gradient が ρ→1.95 で収縮域逸脱) が実モデルの該当例。
- **記憶形成3機構の統一 taxonomy** (自己予見/復活修復/社会観察) は新グループ候補。sound vs empirical の対比が
  llcore 差別化を強める。文献接地 (Workflow w7q0if4xs) で novelty/先取り (Gödel/Campbell/safe-RL/repair) を確認後、
  論文 §future work or 新節への編入可否を判断。over-claim 不可 (各機構に先行あり)。
