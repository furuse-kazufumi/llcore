# PoC-1 Stage 1 — write (状態生成) ゲート結果 (2026-07-11, CPU)

> pre-reg: `docs/research/preregistration/prereg_poc1_testtime_read_2026-06-29.md` §3.1。
> honest 規律: `feedback_benchmark_honest_disclosure` / `feedback_staged_poc_individual_structure`。
> これは **write が recall を学習できるかのゲート**であって、read 側 novelty(Stage 2)の主張ではない。

## 何をやったか

- 合成 MQAR ハーネス (`mqar.py`) + tiny 忠実 Gated DeltaNet (`ttt.py` の `TTTLinearCore`) を
  MQAR で学習し、**R0 = 単発線形 read `S q̂`**(セル native の read)の recall を測定 (`run_stage1.py`)。
- 2 層 / d=128 / state_dim=128、全 backprop(state detach 無し)、CPU only(torch 2.12+cpu)。

## 結果(recall = query 位置で argmax が束縛値と一致した割合、chance=1/D=0.062)

| 設定 | R0 recall | 備考 |
|---|---|---|
| 4-pair, 既定 A_log init | 0.06→0.33 (step150 で頭打ち) | 収束遅い |
| 4-pair, A_log 保持 init (rate=1, α≈0.95) | **step50 で 0.32**、天井 0.33 | 収束 ~3× 速い / 天井は不変 |
| 2-pair, A_log 保持 init | **0.52** | 負荷を下げると recall 上昇 |

→ **GATE-PASS**(recall ≫ 3×chance)。状態は「学習・非直交 key」の連想を確かに符号化する。

## 3 つの確定した知見(honest)

1. **基質は recall を記憶する**: recall は負荷(pair 数)に反比例(2pair 0.52 > 4pair 0.33 > chance 0.06)。
   = 連想は状態に入っており、単発 read が bottleneck。← Stage 2(read 側 cleanup)の前提が成立。
2. **~0.33 の天井は学習不足でも init 由来でもなく、R0 の crosstalk 容量天井**:
   A_log を保持寄りに再初期化して収束を 3× 速めても、より長く学習しても、同じ 0.33 で頭打ち。
   → これは undertraining ではない。**単発 read の crosstalk 限界**であり、まさに read 側反復
   cleanup が recover を狙う headroom。PoC-1 的にはこの天井を"補って消してはいけない"。
3. **faithful セルの A_log 既定 init(uniform(1,16))は recall に敵対的**: exp(A_log) が巨大化 →
   α≈0 で状態を毎ステップ忘却、しかも exp(-huge) の勾配消失で抜け出しにくい。**機構は不変のまま
   A_log を保持寄りに再初期化**(学習可能パラメータの init 変更=正当)すると収束が大幅改善。
   Stage 2/本走では A_log 保持 init を既定にする。

## 「学習不足を補う方法」の整理(ユーザー質問への回答)

- **収束の遅さ**(init 由来)→ A_log 保持初期化で補える(実測 3× 高速化)。lr warmup/cosine も可。
- **天井**(= R0 crosstalk 容量)→ これは undertraining ではないので「学習を足す」では上がらない。
  上げたいなら state_dim 増 / 層増 / 系列長↑ / curriculum(基質が変わる)。ただし PoC-1 では
  天井は研究対象そのものなので消さない。

## 次(Stage 2, additive)

凍結状態 S を query 位置で取り出し、pre-reg §3.2 の read 変種を **同一 S 上で FLOP-matched** に比較:
- R0(単発) / R-CCQ(単発 curvature contraction `(I−λΣ)q`=kill-risk 対照) /
  R-ISTA(K∈{3,5} soft-threshold) / R-Hopfield(K step 非softmax cleanup)。
- mandated baselines: single inner-product / softmax modern-Hopfield / sparse Fenchel-Young Hopfield。
- 指標: recall@{2k,4k,8k} を K の関数で、paired bootstrap CI。判定 = pre-reg §5(GO/PARTIAL/NULL)。
- SWA 窓不使用。gain が異常なら detach/train_seq_len/窓 の confound をまず疑う。
