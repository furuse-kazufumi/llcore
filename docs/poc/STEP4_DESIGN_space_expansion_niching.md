# CPU 手順 4 設計ノート — 空間拡張 + 分離機構で ③ を立てる

> **凡例 — 進化4要素 (Darwin/Mayr)**: ①変異 (variation) / ②遺伝 (heredity) / ③適者生存・選択 (selection = 適応度の差による差し survival) / ④過剰繁殖 (over-reproduction)。本書の「①」〜「④」はこの番号を指す (特に「③」= 適者生存)。平易な用語集 → [`YOUGO_平易版.md`](./YOUGO_平易版.md)。

**位置づけ**: `EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md` の CPU 手順 4。手順 1 (honest_eval CI ゲート,
commit 5ee1c13) / 手順 2 (per-gene ridge readout un-flatten, commit c578f6f) の上に立つ。
**ステータス**: 設計 + 方向選択待ち (アーキ固定は **ユーザー steering 案件** — 複数の妥当な道があり成果に
大きく影響するため、本ノートは選択肢と falsifiable 判定基準を提示し、実装着手は方向確定後)。

---

## 1. これまでに確定した障害 (手順 1-2 + scout)

③ (差し survival 経由の選択 = 単なる hill-climbing でない累積選択) が **立証できない機構的理由**:

1. **3-param leak integrator の landscape は単峰** (2026-05-30 scout, copy delay=0 ridge fitness を
   decay×gate_str で 11×11 grid sample):
   - 高 fitness は **単一の広い対角バンド** (connected plateau, >0.9 が 18/121 cell)。
   - 独立した局所最適は無し (見かけの "2 local maxima" は同一 plateau 上の隣接 cell)。
   → 単峰 broad-basin では **hill-climbing と random search が同じ basin に収束** = ③ の出番なし。
2. **手順 2 の結論**: per-gene ridge readout は fitness の *scale* を un-flatten する (max 0.63→0.996) が、
   landscape の *形* (単峰性) は変えない。copy delay=0=容易 / delay≥4・addition=clip 後平坦 (raw R² 負) で、
   「構造的かつ難しい (= 多峰で hill-climbing が詰まる)」中間 regime が 3-param 空間に存在しない。
3. 既存の分離機構 (LineageReservoir / ModesMeter / persona) は **evolve に未配線 or 飾り**
   (監査 §1: verifier no-op, reservoir は「凍結 elite の生命維持」と自己 disclose 済)。

**→ ③ を立てるには、まず multimodal な (複数の分離した良解を持つ) landscape を作り、次にその峰を
維持する分離機構 (niching/QD) を load-bearing にする必要がある。順序が重要。**

## 2. 手順 4 成功の falsifiable 判定基準 (先に固定)

実装の前に「何が達成されたら ③ が立ったと言えるか」を honest_eval ハーネスの語彙で定義する:

- **(C1) 多峰性**: 拡張した gene×task で、ridge landscape に **≥2 個の分離した basin** が存在
  (basin 間に明確な fitness の谷)。grid/sampling で確認。
- **(C2) hill-climbing が詰まる**: elitism-only または tournament_k=1 の GA (= 近傍 hill-climbing) が、
  単一 basin に収束し global には届かない seed が一定割合存在。
- **(C3) 分離機構が効く**: niching/QD を入れた GA が、(C2) の hill-climbing GA を honest 再評価 best で
  **≥15 seed Wilcoxon p<0.05 + Cliff δ 非無視** で上回る。
- **(C4) ③ の分離**: その勝因が「より良い basin を発見し維持した」ことに帰属できる (lineage 追跡で
  確認)。**単なる探索量増加でないこと** = 同じ評価予算の random restart hill-climbing も対照に置く。

(C3) が手順 2 で作った `evolution_vs_random` の自然な拡張 (random の代わりに hill-climbing GA を対照に)。

## 3. 空間拡張の候補 (★ = 推奨度、ユーザー選択)

| 案 | 内容 | 多峰性の見込み | コスト | 備考 |
|---|---|---|---|---|
| **A. multi-tap delay line** ★★★ | state を K-tap shift register 化 (各 dim が t-1,t-2,... を保持)。gene = tap 数 + 各 tap の decay/gain | **高** (delay 別に異なる最適 tap 構成 → 複数解) | 中 | delayed copy が**解ける**ようになり「構造的-難」regime を直接作る。手順 2 が欠けると示した regime の本命 |
| **B. 多層 stack** ★★ | leak integrator を L 層積む (層ごとに別 3-param gene)。 | 中 (層間結合で峰が分かれうるが未確認) | 中 | kernel_plugin 設計 (S1/S2) と相性良。ただし channel 分離問題は残る |
| **C. 複数 update gene の混合** ★★ | N 個の update 規則を gene が重み付き混合 (mixture-of-kernels) | 中-高 (混合重みで別 basin) | 中-高 | RWKV/Mamba/Hopfield 混合 = Stage 3b kernel 多様化と合流 |
| **D. per-dim 異種パラメータ** ★ | scalar decay → per-dim decay/mix/gate ベクトル (3→3d param) | 低-中 (elementwise なので峰が分かれにくい) | 低 | 最小変更だが elementwise channel 独立で multimodality 弱い見込み |

**推奨第一手 = 案 A (multi-tap delay line)**。理由: 手順 2 が「delayed copy が解けない (3-param では
不能) = 構造的-難 regime 不在」を示した。multi-tap は delayed copy を **解ける化** し、かつ delay ごとに
異なる tap 構成が最適 → 自然に多峰。(C1)-(C4) を最短で検証できる。案 B/C は A で ③ が立った後の一般化。

### 案 A scout 結果 (2026-05-30, 強い greenlight 証拠)

leaky delay-line prototype (K=8 tap shift register, gene=leak vector, state flatten→ridge readout) を
research scout で実測 (d=6, seq=20, 30-40 random gene):

| task | random gene max R² | mean R² | 解釈 |
|---|---|---|---|
| copy delay=0 | +1.000 | +1.000 | 容易 (3-param と同様) |
| copy delay=4 | +1.000 | **+0.416** | **解ける + 勾配あり** (3-param では一様に負/0 = 不能だった) |
| copy delay=6 | +0.957 | **−0.354** | **大半の gene が失敗・良 gene のみ成功** = 構造的-難の理想形 |
| two-horizon (x[t-2]+x[t-5]) | +0.998 | +0.504 (std 0.367) | 難易度に幅 (20/40 が R²>0.5) |

→ **空間拡張 (delay line) は手順 2 で欠けていた『構造的かつ難しい』regime (=解ける + 勾配) を作る**。
delay=6 / two-horizon は「大半 fail・少数 success」で勾配あり。

### (C1) 多峰性 gate の scout 結果 (2026-05-30) — **重要: 案 A は (C1) を満たさない**

設計ノートが指定した (C1) fail-fast gate を実行: random-restart hill-climb (12 start) を two_horizon /
copy_delay6 で走らせ、収束 optima が **分離 basin** か **連結 manifold** かを「2 optima の中点が谷に
なるか」で判定。

| task | R²>0.6 到達 | 収束 fit | optima 間 leak 距離 (mean) | **中点が谷のペア (分離 basin 証拠)** |
|---|---|---|---|---|
| two_horizon | 12/12 | 1.000 | 0.76 | **0/66** |
| copy_delay6 | 11/12 | 1.000 | 0.74 | **0/55** |

→ **(C1) FAIL**: optima は leak 空間に散らばる (距離 0.74) が中点に谷が無い = 高 fitness 領域は
**連結した冗長解 manifold** (分離した複数 peak ではない)。**per-gene ridge readout の柔軟性が、多数の
leak ベクトルを同じ task 解に compensate する**ため、fitness 上は単一の連結最適集合になる。
hill-climbing がどこからでも到達でき、niching が維持すべき「分離した峰」が無い。

**→ honest 含意 (案 A の素朴版は ③ に不十分)**: 空間拡張だけでは fitness 多峰性は出ない。原因は
solvability ではなく **per-gene 最適 readout が解 manifold を連結にすること**。③ を立てるには次のいずれか:
- **(i) 競合目的 task**: 同時に最大化できない複数目的 (例: 短延延 vs 長延延の recall を限られた tap で
  両立不可) → gene が排他的 niche に特化せざるを得ず、真に分離した峰が出る。
- **(ii) readout を connected manifold にしない**: per-gene 最適 ridge をやめ、**共進化 readout** (gene と
  共有・進化) にする (診断 §7b の p=0.0005 GA-win 機構と一致)。compensate 自由度を奪い峰を分離。
- **(iii) 行動記述子ベース QD (MAP-Elites)**: niche を fitness peak でなく **behavior descriptor** で定義
  (例: gene の delay 特性 profile)。MAP-Elites は fitness 多峰性を要求せず、行動多様性で動く。連結
  manifold 上でも behavioral に分散した archive を維持でき、③ の代替的成立路線になりうる。

**→ 修正推奨**: 案 A (delay-line で解ける化) は前提として有効だが、その上に **(i) 競合目的 task** か
**(iii) MAP-Elites (behavioral niching)** を載せるのが ③ 立証の本線。(ii) 共進化 readout は診断の
GA-win を再現する最短路だが「読出の進化 = gene の進化」の交絡に注意 (honest 分離が必要)。
**この3択は user steering 案件** (研究方向の選択)。scout コードは未コミット inline、本実装時に research/ 整理。

## 4. 分離機構 (既存資産の load-bearing 化)

multimodal landscape を作ったら、以下を evolve に**実配線**する (現状は未配線 or 飾り):
- **LineageReservoir** — 系統多様性の保持 (現状「凍結 elite 生命維持」→ 能動 niche 保護に昇格)。
- **ModesMeter** — モード数の観測を選択圧に反映 (現状観測のみ)。
- **persona / fitness sharing** — niche 占有による fitness 減衰 (QD の核)。
- 候補: 明示的 **fitness sharing** (Goldberg-Richardson) または **MAP-Elites 風 archive** (behavior descriptor
  = lineage の delay 特性等)。llive 資産は Read のみ (llcore 自前 minimal 実装、非依存維持)。

## 5. 推奨 first experiment (方向確定後の着手順)

1. **案 A の最小 gene** (`MultiTapGene`: K-tap, 各 tap decay/gain) + run_sequence + codec を実装。
2. ridge landscape を sampling し **(C1) 多峰性を確認** (これが無ければ案を変える — fail-fast)。
3. honest_eval を拡張: 対照を「同予算 random」に加え「同予算 hill-climbing GA (tk=1/elitism-only)」も置く。
4. niching あり/なしで **(C3)** を測定。lineage 追跡で **(C4)** を確認。
5. 各 commit 前 Codex pair-review ([[feedback_codex_pair_review_for_llcore]])。

## 6. honest 留保

- 本ノートは手順 1-2 の実測 + 1 回の landscape scout に基づく設計仮説。案 A が多峰を作る保証はまだ無い
  (step 1 = (C1) 確認が fail-fast gate)。
- 「multi-tap で delayed copy が解ける」は直感的だが未実証。最小実装 + landscape 確認で先に検証する。
- GPU 投資判定 (監査 §7b) は依然 conditional。手順 4 で ③ が CPU 上で立っても、実 LLM fitness が同様の
  多峰構造を持つかは別問題 (手順 6 の小型 LLM sanity check が最終材料)。

## 関連
- `EVOLUTION_SOUNDNESS_AUDIT_2026-05-30.md` (§7b CPU 手順, 手順 2 結果)
- `src/llcore/evolution/honest_eval.py` (手順 1) / `src/llcore/fitness/ridge_readout.py` (手順 2)
- [[project_llcore_init_2026_05_29]] / [[feedback_benchmark_honest_disclosure]] / [[feedback_codex_pair_review_for_llcore]]
