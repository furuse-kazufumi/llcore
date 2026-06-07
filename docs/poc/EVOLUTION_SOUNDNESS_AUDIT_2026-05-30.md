# llcore 進化機構 健全性監査 (Evolution Soundness Audit)

> **凡例 — 進化4要素 (Darwin/Mayr)**: ①変異 (variation) / ②遺伝 (heredity) / ③適者生存・選択 (selection = 適応度の差による差し survival) / ④過剰繁殖 (over-reproduction)。本書の「①」〜「④」はこの番号を指す (特に「③」= 適者生存)。4要件の holds/空転 判定は §1、平易な用語集は [`YOUGO_平易版.md`](./YOUGO_平易版.md)。

**実施**: 2026-05-29〜30 / **手法**: 5-lens 経験的調査 (コード読込 + 反証実験を py-3.11 実走) + synthesis 独立再現
**目的**: 「PoC が個別に動く」と「進化 (変異×選択×遺伝が累積改善・開放端性を生む) が成立する」を峻別し、
後者を falsifiable に検証する。
**位置づけ**: [[feedback_benchmark_honest_disclosure]] (異常に良い結果は内訳を疑う) の実演事例。

---

## 0. 結論

**llcore は「自己進化型アーキ探索の機構 PoC」としては成立。しかし「進化が成立している」とは
(強い意味で) 言えない。** 機構 (selection/variation/heredity/Z3 健全性/open-end 各部品) は real かつ
単体で動くが、システムとして累積改善・開放端性を生むことは未実証 (一部は falsify 済)。

## 1. 進化 4 要件 (Darwin/Mayr) の物差しで holds / 空転

| 要件 | 判定 | 実測根拠 |
|---|---|---|
| ① 変異 (variation) | **成立** | frozen gene 3.9% のみ / zero-attractor 脱出済 / NaN・Inf 0。表現型が動く |
| ② 遺伝 (heredity) | **成立** | 子→親 behavioral 距離 = 無関係ペアの 0.18× (σ=0.15)、σ で単調連続 (相関 0.87) |
| ③ 生存・繁殖の差 (differential survival) | **機構はあるが空転** | 定数 fitness で改善 = 完全に 0.0000 (選択は配線済)。**だが** honest 再評価で GA は同予算 random search を**有意に上回らない** (GA−RAND=−0.011、5/10 勝、Wilcoxon p=0.77) |
| ④ 過剰繁殖 (over-reproduction) | **③依存で空転** | tournament で繁殖差の構造はあるが、③が弱い (proxy fitness の eval-noise 過大、clean SNR≈0.89、landscape 天井 ~0.2) ため選択信号がノイズに埋もれる |
| (llcore 固有) 検証ゲート (verifier = 生存可能領域) | **no-op** | 主張 invariant \|state\|≤1 で reject 率 0/2000 = 構造的トートロジー。かつ `minimal_ga.evolve` に**未配線**。reject は PoC が選択圧捏造のため恣意的に state_bound=0.4 に下げた時のみ |

## 2. 最重要 honest 発見

1. **報告 best 曲線は elitism artifact**: `minimal_ga.py` の elitism が前世代の noisy fitness を
   **再評価せず凍結持越し** → 報告 best 0.473 が fresh-seed honest 再評価で **0.183 に崩落
   (+0.29 の水増し剥落、独立再現)**。「best 単調上昇 = 進化成立」は不成立。honest 信号は集団 mean。
2. **開放端性 (open-endedness) は drift 以下**: 後半 novelty が機構あり FULL=9.0 / PLAIN=9.3 に対し
   **無選択 random-walk=21**。「持続的新規性を生む」どころか drift より新規領域が少ない。
   adaptive_floor は last-50 世代で変化 0、curriculum は gen10 で飽和 (is_saturated=True)。
   A_new 指標は無選択 jitter でも 100% 達成 = drift と弁別不能な artifact。
3. **唯一の確かな効果 = 系統固定 (premature convergence) の防止**: reservoir OFF で最終 persona
   生存 = 1.0 (5 seed 全)、ON で 7.4。ただし reservoir docstring 自身が「能動進化でなく凍結 elite の
   生命維持」と disclose 済 (再投入の 98-100% が凍結 elite)。
4. **cross-arch ChangeOp (#5) は mock**: `kernel_swap_mock` は gate_str 符号反転のみで同型・同更新関数。
5. **G3/G4/G6/G7 等の PoC ゲートは elitism bookkeeping を測っており「進化成立」の証拠にならない**
   (G4 単調性は elitism の定義的 artifact: elitism=1 で 6/6 mono、elitism=0 で 0/6)。

> プロジェクト自身の `STAGE_3_VERDICT` は既に正直に降格済 (Marabou⊂llcore 完全撤回 / 上限なし claim
> 限定) で本監査と整合。**claim と実装のズレを隠してはいない** — ただし上記ゲートの disclosure は不足。

## 3. confound (交絡) — まだ切り分いていない核心

現 ③失敗 (p=0.77) は **「GA 自体が弱い」と「proxy fitness の landscape が平坦+ノイズ過大」の交絡**。
これを切り分けないと GPU 投資 (実 LLM fitness) の妥当性が確定しない:
- **clean で構造のある fitness なら GA が random search を有意に上回る** → 機構は健全、欠けているのは
  本物の選択信号のみ → 実 LLM fitness (GPU) で ③が立つ見込み。**GPU 投資が正当化される。**
- **構造のある fitness でも GA が random と互角** → GA/探索空間 (3-param) 自体に rework が必要。

## 4. gap to real evolution (③を立てる道筋、優先順)

1. **eval-noise 抑制 + elitism 凍結持越しの廃止**: 各世代 best を fresh-seed honest 再評価、n_trials を
   上げ「遺伝子間の真の差 > 評価ノイズ」を確保 (現状は逆転)。**[CPU 可]**
2. **falsification test を CI ゲート化**: 「honest 再評価 best が同予算 random search を多 seed (≥15) で
   Wilcoxon p<0.05 + Cliff's delta 非無視で上回る」を進化成立の合格条件に。定数 fitness で改善 0 も回帰保持。**[CPU 可]**
3. **confound 切り分け診断**: 非平坦・低ノイズの構造的 fitness (readout 共進化 or 表現力ある合成タスク) で
   GA vs random を測り、機構が健全か landscape 問題かを確定。**[CPU 可、GPU 投資前の必須]**
4. **③を本物に**: fixed-readout probe → 実 LLM/重み損失・downstream 性能に接続。**[GPU 必要: RTX 4090 24GB]**
5. 検証ゲートが効く非自明 invariant を定義し evolve に配線。**[CPU 可]**
6. cross-arch ChangeOp の実装 (mock 符号反転 → 実 kernel 置換)。
7. 開放端性の長 run 実証 (飽和解消 + random-walk control を下回らない sustained novelty)。

## 5. falsification test (進化成立の合格条件、CI 化対象)

各世代 best gene を進化 rng と独立な fresh seed で n_trials≥30 の honest 再評価にかけた曲線を**主指標**とし、
**「honest 再評価 best が同一 eval budget の random search を ≥15 seed で Wilcoxon p<0.05 かつ
Cliff's delta 非無視で上回る」**を真の進化成立条件として CI ゲート化する。これを満たすまで、報告される
単調 best 改善は elitism+noise の artifact とみなし「進化成立」と認めない。併せて「定数 fitness で
best-delta=0」(選択の実在) を回帰テストで保持。

## 6. 監査の honest 留保

- synthesis 独立再現で **lens 1 の「random fitness 集団 mean は 0.5 固着」は再現せず** seed 依存に
  大 drift (+0.24〜−0.17)。lens 1 の決定的分離は報告より弱い (短 run では genetic drift が control を
  上振れさせる)。「選択機構の実在」(定数 fitness→改善 0) は頑健に再現。
- 全実験は CPU・toy synthetic task・3-param gene 空間 (天井~0.2) での測定。実 LLM 損失とは別。

## 7b. confound 切り分け診断の結果 (2026-05-30, [GA健全性 切り分け診断])

§3 の confound (「GA 機構が弱い」vs「proxy fitness が平坦+ノイズ過大」) を 3 条件で実測切り分け。

**結論: ③失敗の主因は landscape の平坦さ。GA 機構の弱さでもノイズでもない。**

| 検証 | 結果 |
|---|---|
| clean 構造的 landscape での GA | **圧勝** (単峰 GA−RAND=+0.122, p<1e-4, Cliff δ=+0.97, 30/0/0 / 多峰 +0.162, δ=+0.77 / 構造ゼロ対照 +0.0009 p=0.629)。**機構は健全、rework 不要** |
| 本番 CopyTask でのノイズ掃引 | SNR を 0.38→**3.51 まで上げても GA=random** (全水準 p>0.05)。律速は SNR でなく **上位20遺伝子の真の spread=0.0007** の平坦プラトー |
| 固定 readout → 最適化(共進化) | copy d=8 で landscape に構造出現 → GA 勝つ (p=0.0005, δ=+0.62)。**固定 random readout が平坦化の最大要因**。ただし addition は最適化しても null (線形デコード不可) |

**最重要 honest 発見**: 構造的 landscape で「GA が勝つ」時も、**tournament_k=1 (選択差なし) でも勝つ**
(+0.078, p=0.0001) → 勝因は ③(tournament 選択) でなく **elitism+変異の近傍 hill-climbing**。
3-param 低次元では近傍探索が支配 operator。**「GA が勝つ」≠「③(差し survival 経由の選択)が立証」**。
③ そのものは合成構造 landscape でも未分離・未立証。

**報告 best +0.29 水増し = artifact 確定**: noise σ↑で report_gap が 0→+0.808 と拡大 (eval-noise×elitism 凍結)。

### GPU 投資判定: **conditional (今は保留)** — §0 の "GPU 必要" を修正

「機構健全 → GPU で③が立つ」は論理が一段飛んでいる:
1. ノイズ抑制 (=GPU で n_trials↑) では③は立たない (律速は landscape の平坦さ、SNR3.5 でも本番 GA=random)
2. 構造 landscape で勝った operator は③でなく hill-climbing
3. 実 LLM fitness が「SNR≥2 の構造 + 良遺伝子間 spread」を持つかは**未証明** (これが GPU の必要十分条件)

→ **GPU(RTX 4090 24GB) 投資は CPU 手順 (下記) で「効くと分かる」まで保留。** 永遠に不要でなく、買う前に検証する順序。

### ③を立てる概念 (ユーザー言語化 2026-05-30)

集団内 **分離(speciation / Quality-Diversity / niching)** を入れ、集団を多峰に分化させて ③(差し survival)に
「選ぶべき差」を作る (大量プロセス起動でなく 1 集団内 N 個体の niching)。llcore 既存の LineageReservoir/
ModesMeter/persona を「飾り」から load-bearing に昇格 + evolve 配線。**ただし分離先に構造が要る** ので
readout 修正 + 空間拡張とセット。

### CPU 手順 (GPU 前、優先順)

1. **[着手] honest 再評価 + falsification test を CI ゲート化** (`src/llcore/evolution/honest_eval.py`):
   fresh-seed 再評価を主指標化、elitism 凍結持越し artifact を測定から排除。clean landscape で GA 勝利を
   regression 固定、定数 fitness で改善 0 を回帰保持。**これが全後続手順の信頼できる物差し**。
2. fixed readout → per-gene least-squares(ridge, held-out) 置換を本番 fitness に配線 (landscape un-flatten)
3. ③を hill-climbing から分離測定 (本番 readout で tournament_k sweep + elitism=0)
4. 探索空間を 3-param から拡張 (multi-layer/複数 update gene) + 分離機構 (QD/niching) を load-bearing 化
5. AdditionTask 追試 (線形可解性で③成立が分岐する仮説の確証 = GPU 投資の task 選別基準)
6. 小型 LLM(CPU 推論可) で実 proxy の landscape 構造を事前 sanity check (GPU 投資の最終判断材料)

### CPU 手順 2 の結果 (2026-05-30, per-gene ridge readout)

手順 2 (fixed readout → per-gene least-squares(ridge, held-out) 置換) を本番 fitness に
配線し実証した (`src/llcore/fitness/ridge_readout.py` + `scripts/poc_ridge_readout_unflatten.py` +
`tests/unit/test_ridge_readout.py` 8 件)。各 gene について train sequence で線形 readout を
ridge で fit → **held-out** で R² を測る (reservoir computing 標準評価、leakage 構造的に排除)。

| 命題 | 測定 | 判定 |
|---|---|---|
| **P1 un-flatten** | copy d=8 delay=0 で ridge は spread を広げ最良 gene を押し上げる: fixed std=0.230/max=0.632 → ridge std=0.373/max=**0.996** (spread 1.62×) | **成立** |
| **P2 容易だが選択なし** | un-flatten 後 copy delay=0 は『容易な単峰』化。GA=0.998 vs 同予算 random=0.997, diff=+0.0007, p=0.47 → **GA≈random**。eval-noise を n_train∈{6,12,32} で掃引しても全水準で passes=False (n_train=6 で diff=+0.05, p=0.18 非有意) | ③ **未証明** |
| **P3 有用信号 regime 不在** | copy delay≥4 / addition は **clip 後 fitness が全 gene ~0** (GA に選択信号なし)。raw R² (clip=False) は **負** (mean 予測以下, copy d=4 mean=−0.14 / addition mean=−0.27, 小 spread あり) | clip 後**平坦** |

**最重要 honest 発見**: per-gene ridge readout は fitness の **scale** を un-flatten する (real な capability)
が、3-param leak integrator 上では copy delay=0=**容易すぎ** (random も天井に届く) / delay≥4・addition=
**clip 後 fitness 平坦** (raw R² は負・小 spread) で、③(差し survival 経由の選択)が立つ『構造的かつ
難しい』中間 regime をこの評価設定・サンプルでは作れない。

> **honest 注 (Codex pair-review 2026-05-30 の 3 findings 反映)**: (1) [High] clip 後の 0.0 は raw R²<0 を
> 潰した値で「raw=0=信号皆無」と識別できないため、`ridge_fitness(clip=False)` で raw R² を併記し
> 「平坦」は **clip 後 fitness** に限定して主張する。(2) [Medium] state は tanh 非線形の出力なので
> 「原理的にデコード不能」は過剰主張 → 「この 3-param 系の random サンプル N≈20・この評価設定での観測」に
> 限定。(3) [Low] n_train ノイズ掃引を PoC script (P2) に組込み再現可能化。

診断 §7b の「copy d=8 で GA 勝つ (p=0.0005)」は readout を **共進化** (gene×readout 結合 landscape) させた
別機構の結果であり、**per-gene 独立 ridge fit (手順 2 が指定する手法) では再現しない**。

→ **readout 修正だけでは ③ は立たない。真の unlock は CPU 手順 4 (探索空間を 3-param から拡張 +
分離機構 QD/niching の load-bearing 化)**。手順 2 は「物差しの un-flatten は効くが、それは
landscape 平坦さの一因に過ぎず、gene 空間の低次元・縮退が残る律速」であることを経験的に確定した
negative-but-informative result。手順 3 (③を hill-climbing から分離) は手順 4 の空間拡張後に意味を持つ。

実装: `src/llcore/fitness/ridge_readout.py` (RidgeReadout / fit_ridge_readout / ridge_fitness /
make_ridge_eval_once) + `tests/unit/test_ridge_readout.py` (8 件) + `scripts/poc_ridge_readout_unflatten.py`。

## 7. 関連
- [[project_llcore_init_2026_05_29]] / [[feedback_benchmark_honest_disclosure]] / [[feedback_codex_pair_review_for_llcore]]
- `docs/poc/STAGE_3_VERDICT.md` (project 自身の honest 降格、本監査と整合)
- `docs/design/kernel_plugin_0_2_0a0.md` (S1/S2 完了、S3 は ③確立後)
