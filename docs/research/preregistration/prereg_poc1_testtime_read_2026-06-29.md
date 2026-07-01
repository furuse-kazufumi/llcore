# PRE-REGISTRATION — PoC-1: 凍結蒸留 linear-attention 状態への post-hoc 反復 READ (2026-06-29)

> **status**: pre-registration（実験前に設計・成功基準・解析手順・honest 内訳を固定 = p-hacking 回避）。
> CPU で tiny-scale 実行可（最初の falsify は CPU）、本走は GPU 着荷後（`--device cuda`、Qwen3-1.7B 蒸留 state へスケール）。
> **正本接続**: `fullsense/docs/research/option_b_verification_poc1_novelty_2026-06-29.md`（一次検証）/ `triz_constant_state_recall_2026-06-29.md` §4（PoC-1 原設計）/ memory `project_llcore_option_b_verification_2026_06_29`。
> **実装接地（実在確認済）**: `src/llcore/lm/ttt.py`（忠実 Gated DeltaNet セル）/ `runtime/distill.py`。
> **規律**: `feedback_benchmark_honest_disclosure`（異常に良い結果は内訳を疑う）/ `feedback_verify_existence_before_claiming` / `feedback_staged_poc_individual_structure`（小 PoC gate→additive）。

---

## 0. ★novelty framing 規律（検証で確定・違反禁止）

一次検証（WF wp7ugxvch + 深掘り wdhj0hmdp, 6 probe 独立収束 + 2 一次 spot-check）で確定した framing。**これを破ると over-claim**:

- **素の「反復・非softmax read」機構の novelty は主張しない**。sparse-coding / VSA-cleanup / modern-Hopfield の spirit と認める。
- **貢献の厳密な定義** = 「**凍結・蒸留された linear-attention の sum-of-outer-products 状態（学習・非直交 key・designed codebook 無し）に、decode 時に post-hoc で codebook/sparsity-prior の反復 cleanup read を適用する**」という regime ── **VSA 理論が明示的にカバーしない領域**（VSA recovery 保証は known near-orthogonal codebook 前提）。
- **confidence = MEDIUM**（決定的 foreclosing 論文も決定的占有証明も無し / wrapper は narrow かつ approachable / 本質は substrate-transfer の engineering-flavored novelty）。spot-check 済: CCQ(2606.01294) forward-citation = 0（niche 未占有）/ Lexico(2412.08890) = softmax KV cache の OMP 圧縮で read は softmax = 非占有(partial)。
- **必須 cite + 差別化**: CCQ 2606.01294（最強の同時代 related, 単発 curvature contraction・trained-in）/ Sparse Modern Hopfield 2309.12673 / Hopfield-Fenchel-Young 2411.08590 / Resonator-VSA 1906.11684 / Lexico 2412.08890 / Schlag 2102.11174（linear-attn=fast-weight, read は single product と明示）。
- **天井の honest 注記**（P7=2504.14366）: 本 PoC は「固定解像度状態からの読み出し改善 = ceiling-relaxation/mapping」であって **plateau-breakthrough ではない**。「破った」と書かない。
- **2-3 ヶ月後に read 側 forward-citation 再 crawl**（fast-moving field、CCQ follow-up が wrapper を occupy しうる）。

---

## 1. Research question

> **softmax を温存せず、凍結した定数 linear-attention 状態の READ 側 test-time 最適化だけで、associative recall が単発 read を超えて改善するか。**

決定変数 = read 反復数 K と read 変種。状態（write）は一切変えない。

---

## 2. Hypotheses（事前固定・falsifiable）

- **H1（主）**: gated/delta 状態上で、反復 read（K=3–5 の ISTA/Hopfield cleanup）が **単発 read R0 を recall@{2k,4k,8k} で CI 超えで上回る**。
- **H2（対照・kill-risk）**: 反復 read が **単発 R-CCQ（curvature contraction 再現）をも CI 超えで上回る**（= gain が「反復」由来であり、単に「賢い単発」で説明されない）。P6 検証より「1-step 最適」は本 setting に非転移と予測されるが、ここで実測する。
- **H3（state-quality 依存）**: vanilla-additive 状態では反復 read の gain は消失または極小（P7 不可逆飽和より）。gated/delta 状態でのみ gain が出る。→ **「どの状態に効くか」を切り分ける**。
- **帰無**: いずれの K でも R0/R-CCQ を CI 超えで上回らない → 「線形定数状態では read 反復は無効、read 側 test-time は dead」を honest null として公表し **fork C(NAS-allele)へ pivot**。

---

## 3. 設計

### 3.1 状態の生成（write は固定）
1. tiny model（2–4 層, d=128, head 数固定）を **`ttt.py` の忠実 Gated DeltaNet セル**で構築。
2. 合成 **MQAR**（multi-query associative recall）+ **S-NIAH-1 passkey** で学習。
3. **訓練系列長 512–2048 を sweep**、**state detach を解除して full backprop**（efficient_arch §4.2 #1 の confound=「機構天井か訓練切断か」を本段で混入させない）。
4. 学習後、状態を**凍結**。
5. **状態種を 3 つ**比較（H3）: (i) gated-delta（本命）/ (ii) vanilla-additive（最弱・P7 予測）/ (iii) delta-rule のみ。

### 3.2 read 変種（FLOP-matched、状態は不変）
| 変種 | 内容 | 役割 |
|---|---|---|
| **R0** | 単発線形 read `o=Sq` | ベースライン |
| **R-ISTA** | S を CS 測定行列と見なし K∈{3,5} step の unrolled soft-threshold 疎復元（NOODL 流 linear+soft-threshold、微分可能） | 主仮説 |
| **R-Hopfield** | K step の連想 cleanup（snap-to-stored-value energy descent、非softmax） | 主仮説 |
| **R-CCQ** | 単発 curvature contraction `(I−λΣ)q`（2606.01294 再現） | ★kill-risk 対照（反復は要るか） |
| **R-CountMin** | R∈{2,4} 本の独立射影副状態に冗長書込 + soft-min 集約 | 補助（state R 倍, 系列長 O(1)） |

### 3.3 ★mandated baselines（深掘り verdict 指定・同一凍結状態上で ablation）
- **single inner-product read**（= R0）
- **softmax modern-Hopfield read**（2008.02217 系、softmax energy 1-step）
- **sparse Fenchel-Young Hopfield read**（2411.08590、非softmax sparse retrieval）

→ PoC-1 の貢献を「これら既知 read に対し、**学習・非直交 key の凍結状態上で反復 sparse-prior read が優位か**」として測る。

---

## 4. 指標・解析（事前固定）

- **主指標**: recall@{2k,4k,8k} を **K（read 反復数）の関数**として。
- **統計**: paired bootstrap CI（多窓）。有意性は CI 非重複で判定。
- **FLOP-matched**: 反復 read の追加 FLOP を単発側に compute-budget で揃える（gain が「ただ計算を足しただけ」でないことを担保）。
- **confound 潰し**: **SWA 窓を一切使わない**（efficient_arch §4.2 #4 = 窓由来の局所 recall を線形 state の手柄に読み替えない）。

---

## 5. ★判定ルール（事前登録・実験後に動かさない）

- **GO**: gated/delta 状態で 反復 read（K=3–5）が **R0 かつ R-CCQ を CI 超で上回り**、かつ vanilla-additive では gain 消失（H1∧H2∧H3）→ read 側 test-time にシグナル確定 → 学習 read adapter + Qwen3-1.7B 蒸留状態へスケール（GPU）。
- **PARTIAL**: H1 のみ成立（R0 は超えるが R-CCQ は超えない）→ gain は「賢い単発」で説明可 → 反復の必要性を主張せず、CCQ 系単発 read の知見として記録、PoC-1 は降格。
- **NULL**: いずれも CI 超えなし（1-step 最適 = 2502.05164 / FDM リスク顕在）→ **honest null 公表**し即棄却 → **fork C（NAS-allele, evolve_linearization/nas_pareto）へ pivot**。

---

## 6. 差別化テーブル（writeup 必須）

| 先行 | 何が違うか（PoC-1 の保持軸） |
|---|---|
| CCQ 2606.01294 | 単発 curvature contraction・trained-in vs **反復 K=3–5・post-hoc** |
| MesaNet 2506.05233 | iterative readout だが state 更新・end-to-end 学習 vs **凍結・read-only** |
| Resonator/VSA 1906.11684 | 既知 near-orthogonal codebook vs **学習・非直交・codebook 無し**（hinge 軸） |
| Lexico 2412.08890 | softmax KV cache の OMP 圧縮・read は softmax vs **linear-attn Σkvᵀ・no softmax** |
| Sparse/FY Hopfield 2309.12673/2411.08590 | softmax/entmax 族・学習 layer vs **no softmax・追加学習 read-network 無し** |

---

## 7. scope・honest 留保

- **CPU で最初の falsify**（state 追加ゼロ・訓練 tiny 1 回・推論のみ、数時間）。GPU 着荷後に Qwen3-1.7B（Apache-2.0, 全層 softmax, F1 検証で base は Qwen3.5 でなく Qwen3 と確定）の蒸留 linear-attn 状態へスケール。
- novelty confidence = **medium**（§0）。決定的占有証明は無く、absence-in-search + 構造論証。
- 本 PoC は **ceiling-relaxation**（P7 天井は不変）。突破ではない。
- gain が異常に良ければ `feedback_benchmark_honest_disclosure` に従い「SWA 窓 / detach / train_seq_len の confound」をまず疑う。

---

## 8. 代替（NULL 時）= PoC-2 / fork

- **PoC-2（fast follow, A と直交）**: anticipatory write gate（triz TOP-2）。ただし B は write 側過密（delta-rule/MesaNet/OVQ/KalmaNet/FwPKM）で差別化困難と判明 → 優先度は A>C>B。
- **fork C**: P5（STAR=ICLR2025 Oral）基盤の NAS-allele で機構新規性。本 novelty 所見と直交＝独立に追える。
