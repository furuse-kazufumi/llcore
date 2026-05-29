# CPU 手順 4 設計ノート — 空間拡張 + 分離機構で ③ を立てる

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

→ **空間拡張 (delay line) は手順 2 で欠けていた『構造的かつ難しい』regime を実際に作る**。特に
delay=6 / two-horizon は「大半 fail・少数 success」で勾配があり、hill-climbing が詰まりうる
(C2) + niching が効きうる (C3) の検証 task として有望。**案 A を本実装に進める根拠が揃った**
(ただし多峰性 (C1) の厳密確認 = 分離した複数 basin の存在は、本実装の最初の gate で要検証)。
scout コードは未コミット (inline 実験)、本実装時に `research/` へ整理予定。

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
