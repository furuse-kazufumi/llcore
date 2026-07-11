# PoC-1 (test-time read) 最終 verdict — GO 撤回 → NULL/weak-PARTIAL (2026-07-11)

> pre-reg: `docs/research/preregistration/prereg_poc1_testtime_read_2026-06-29.md` §5。
> honest 規律: `feedback_benchmark_honest_disclosure` / `feedback_no_solo_ai_judgment` /
> `feedback_verify_existence_before_claiming`。
> 経緯: Stage1/2/H3 が一旦「GO」に到達 → **敵対的レビュー(WF 39 agents/33 findings)+ 決定実験 3 本**で
> GO を撤回。Solo 判断でなく多視点検証 + 一次実測で確定。

---

## 1. 結論(一行)

**NOT GO**。read 側 test-time 最適化は、value 重複を除いたクリーンな連想 recall では **単発線形 read を
頑健に上回らない**(8 seed 中 1 のみ有意、+0.005)。当初の「GO(+0.026, 8/8)」は **MQAR の value 重複に
よる metric artifact**であり、機構も「sparse cleanup」でなく既知寄りの **degree-2 spectral read**。
GPU スケール(Qwen3-1.7B 蒸留状態)は**このまま行わない**。

---

## 2. 何が当初 GO を支えていたか(撤回対象)

Stage2/H3 (num_pairs=6, 3 seed): R-Hopfield が R0 を **+0.018〜0.026 で 3/3(後に 8/8)有意に上回り**、
R-CCQ(kill-risk)は +0.000、vanilla-additive では消失(H3)→ pre-reg §5 の GO 条件を満たすと判定した。

## 3. 敵対的レビュー + 決定実験で判明した欠陥(3つ・実測で確定)

### 欠陥A【致命・確定】gain は MQAR value 重複の metric artifact
`mqar.py` の value は復元抽選(系列内で同一 value が重複可、重複率 ≈1−(15/16)^5=0.276)。
per-instance 分解(レビュー)= **重複 value instance で +0.155 / 一意 value instance で −0.033(3/3 seed CI 全負)**。
加重 0.272×0.155+0.728×(−0.033)=+0.018 がヘッドラインと一致。cleanup は真の key→value binding をむしろ壊し、
「頻出格納値へのスナップ」で重複 instance を稼いでいた。
**決定実験(一次検証, `run_verify.py --mode mechanism --unique-values`, 8 seed)**: value を非復元(一意)にすると
R0 recall が **0.262 → 0.163 に急落**(元 R0 が重複で水増しされていた確定)、R-Hopfield gain は
**+0.026(8/8 有意)→ +0.005(1/8 有意)へ崩壊**。→ 欠陥A を一次実測で CONFIRM。生結果 `verify_mechanism_unique.json`。

### 欠陥B【確定】機構ラベル誤り:「sparse cleanup」でなく degree-2 spectral read
`reads.py` の `_soft_threshold(z, tau)` は **tau=0 で恒等**。val 選択は有意 run の 62–67% で tau=0 を選ぶ。
実体は `o ∝ S·SᵀS·q̂`(値空間 Gram のべき乗反復=degree-2 spectral emphasis)、疎性 prior も codebook cleanup も無い。
**決定実験(`--mode mechanism`, 8 seed)**: 無調整の `r_poly2 = S·SᵀS·q̂` が R-Hopfield gain の ~88% を再現
(+0.023 vs +0.026、8/8 有意)。tau>0 強制(R-Hopfield-pos)でも full と同等 → 疎性は上乗せしていない。
→ pre-reg §0 の novelty regime(「sparse/codebook cleanup in VSA 未カバー領域」)は**不成立**。

### 欠陥C【確定】H2 kill-risk の通過は片側グリッド artifact
R-CCQ = `S(q̂ − λ·SᵀS·q̂)` の λ グリッドが正のみ。勝者 R-Hopfield(tau=0) = `S·SᵀS·q̂` は R-CCQ の **λ<0 極限方向**。
FLOP-matched(共に 3 matvec)で poly2(+SᵀS)は勝ち R-CCQ(−λSᵀS)は勝たず = 同 compute で **degree-2 方向**が源。
片側グリッドのため「賢い単発では説明できない」という H2 の主張は成立せず、pre-reg §5 の字義では **PARTIAL(降格)**。

## 4. 生き残った点(honest・降格を左右しない)

- **H3 の状態依存は headroom 交絡ではなく本物**(決定実験 `--mode headroom`): vanilla を高負荷で R0=0.230
  (gated@6 の 0.268 より低い=headroom 大)にしても gain +0.000(0/3)。gated は R0 水準に依らず gain 継続。
  → 「効くなら gated/delta 状態でのみ」という状態依存は成立。ただし**その効き自体が欠陥A で null 化**するため、
  GO を救わない(「null 効果の状態依存」に留まる)。

## 5. pre-reg §5 に照らした最終判定

- **GO 不成立**(H1 が clean binding で 1/8 有意 = 頑健でない)。
- 字義上は **NULL 寄り**(いずれの変種も unique-value で R0 を CI 超で頑健に上回らない)。最善でも
  「degree-2 spectral read が重複汚染下でのみ見かけ上勝つ」という **weak-PARTIAL / artifact**。
- → **read 側 test-time は、この設定・この benchmark では dead に近い**。GPU スケールの根拠は消失。

## 6. 次手(pre-reg §8)

1. **benchmark を修正**(value 非復元 = binding-clean)し、read 側に**真に**シグナルがあるかを再確認してからのみ
   GPU を検討(現状の +0.005/1-8 では投資に値しない)。
2. あるいは **fork C(NAS-allele: P5 STAR 基盤の機構新規性, `evolve_linearization`/`nas_pareto`)へ pivot**。
   PoC-1 novelty(read wrapper)は本 verdict で実質消えたため、fork C の優先度が上がる。
3. write 側(TOP-2 anticipatory write gate)は B が過密で差別化困難(pre-reg §8)。

## 7. 成果物 / 追試コード

- 決定実験: `run_verify.py`(`--mode mechanism [--unique-values]` / `--mode headroom`)。
  出力 `verify_mechanism.json`(重複)/`verify_mechanism_unique.json`(一意)/`verify_headroom.json`。
- `mqar.py` に `unique_values`(既定 False=byte-identical)追加。
- 敵対的レビュー全文: WF `w6nxbojp7` transcript(39 agents, CONFIRMED 多数)。

## 8. 教訓(honest-disclosure)

- **単一 3-seed の「8/8 有意」でも内訳を疑え**: 効果が特定 instance クラス(重複 value)に 100% 集中していた。
- **合成 benchmark の生成分布(value 重複)が metric を汚染**し得る。recall 系は「頻度スナップ」で水増しされる。
- **solo で spectral 機構までは掴めたが、致命の value-artifact は敵対的 WF が捕捉**。多視点検証の価値の実例
  ([[feedback_no_solo_ai_judgment]] / [[feedback_benchmark_honest_disclosure]])。GPU を焼く前に止められた。
