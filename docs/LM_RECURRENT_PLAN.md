# llcore 定数メモリ再帰 char-LM 計画（RWKV/SSM 系）

> 親計画: `docs/LM_P0_PLAN.md`（P0 = GPT-2 nano char-LM、完了）。
> 本書は次フェーズ = 「llm-viz 風の単純・規則的構造は保ちつつ、Attention の KV キャッシュ（系列長 T に比例して増える実行時メモリ）を**定数サイズの再帰状態**に置換した char-LM」を from-scratch（pure PyTorch / CPU / float32）で実装し、GPT char-LM と **honest head-to-head** で比較する計画の正本。
> 根拠 = 2026-06-15 の並列 recon（4 観点・一次ソース付き、`select`/`formulation`/`reuse`/`measure`）。

## 0. 北極星 / テーゼ

- **北極星 = メモリ効率**（検証可塑性ではない）。「単純構造×巨大化×十分なデータ = LLM。ならば肝は、それを限られたハード（local）でメモリ効率よく動かす工学」というユーザー・テーゼの実機検証。
- テーゼ: **GPT 推論メモリは O(T)（KV キャッシュ ≈ `2·n_layer·T·n_embd` floats）/ 再帰モデルの毎ステップ状態は O(1)（T 非依存）**。この差が、local で「RAM を超える文脈・モデル」を成立させる鍵。
- **重要な honest 前提（measure recon より）**: 現 `CharGPT.generate` は KV キャッシュ無しで毎ステップ全再計算（O(T²)）し、かつ `block_size` でクリップする。→ 再帰の真の優位は **T > block_size**（GPT が構造的に動けない領域）+ 定数状態。block_size 以内では能力を apples-to-apples 比較する。

## 1. 最適アーキ選定（測定駆動の登坂・単一の賭けにしない）

O(1) メモリの勝ちは**アーキ非依存**（候補は全て定数状態）。差別化は**能力（held-out PPL）**。よって「安全に着地 → 能力を登坂 → 測定で決める」:

| 候補 | 並列/逐次 + O(1)推論 | 状態 T 非依存 | 小規模(≤10M,char)能力 | pure-PyTorch-CPU 難度 | float32 安全性 |
|---|---|---|---|---|---|
| **GatedRNN（llcore 既存）** | 逐次 scan / O(1) | ✓ (per-layer 状態) | 弱〜中（gating 単純） | **1**（既存コード再利用） | **安全**（tanh 有界・sigmoid decay） |
| **RWKV-4** | 逐次 scan / O(1)（4 vec/層） | ✓ | **最良の実証（enwik8 で attention の ~95%/param）** | 1–2（~100行） | **脆弱**（exp-WKV、running-max 必須） |
| RetNet / GLA | chunk 並列 / O(1)（行列状態） | ✓ | 小規模実証弱（RetNet は大規模向け） | 3–4 | 安全（exp なし・GroupNorm / log空間 cumsum） |
| Mamba/S6 | selective scan / O(1) | ✓ | char 小規模の実証不明 | **5**（selective scan が CUDA 前提） | 安全（Ā=exp(負)） |
| 素の線形注意 | 並列 / O(1)（2 状態） | ✓ | softmax に劣る（gating 無し） | 2 | やや脆弱（正規化要） |

**確定方針:**
- **Phase 1（de-risk + テーゼ実証）**: llcore 既存の学習可能 `GatedRecurrentLM`（`research/verifier_navigability_gpu/bg10_gpu_lm.py:208-248` ほか、core `s_t = decay·s + (1-decay)·tanh(s@Wᵀ + x_t)`、per-layer `(decay,W)`）を土台に、P0 ハーネス契約を満たす `RecurrentLM` を `src/llcore/lm/recurrent.py` に実装。最安全・最速で「再帰 LM が同ハーネスで動き、状態が O(1)」を実証。
- **Phase 2（能力登坂）**: 能力が GPT に明確に劣るなら、**RWKV-4（running-max 安定化込み）** か **RetNet/GLA（exp なし・安全）** を追加実装し、能力 vs メモリを測って最良点を選ぶ。RWKV-4 は実証最良だが脆弱、RetNet/GLA は安全だが小規模実証弱 — **どちらが勝つかは測って決める**（推測しない）。
- Mamba/S4/Lightning は CPU pure-PyTorch では非推奨（selective scan / 複素固有値 / Triton 前提）。やるなら最後。

honest 留保: O(1) メモリは全候補共通の勝ち。Phase 1 の GatedRNN は能力が弱い可能性が高い → 「テーゼ実証用の足場」と位置づけ、能力の主張は Phase 2 の登坂結果で行う。

## 2. ハーネス契約（既存 P0 を無改変で再利用）

新モデルは以下を満たせば `Trainer`/`eval`/`data`/`generation`/`export` がそのまま使える（reuse recon で確認済）:
- `nn.Module`、`__init__(config)` に `config.block_size` / `config.vocab_size`。
- `forward(idx:Long[B,T], targets=None) -> (logits[B,T,V], loss_or_None)`（targets 無→loss=None）。
- `generate(idx, max_new_tokens, temperature, top_k) -> Tensor`。
- 配置 = `src/llcore/lm/recurrent.py`（`model.py` の隣、`GPTConfig` 類似の `RecurrentConfig`）。
- `Trainer`（trainer.py:97-98 `model(x,y)`）/ `estimate_loss`（eval.py）/ `held_out_report`（eval.py）/ `generation.generate_text` は無改変。
- **唯一の汎用化**: `held_out_report` は `model: CharGPT` 型注釈なので、両モデルが満たす構造 Protocol（`forward_logits(x)->logits`）版 `held_out_report_any` を追加（本体は同一、`logits,_=model(x)` を `logits=model.forward_logits(x)` に置換）。`CharGPT.forward_logits = lambda self,x: self(x)[0]`。
- reuse recon が雛形 `RecurrentConfig`/`RecurrentCore`/`RecurrentLM` を提示済み（Phase 1 の出発点として採用、init/LN/residual/generate は P0 と同形）。

## 3. float32-CPU 数値安定（formulation recon より）

| ブロック | decay 項 | オーバーフロー | 安定化（**必須**） |
|---|---|---|---|
| GatedRNN（Phase 1） | sigmoid decay | 低 | tanh 有界状態。安全。 |
| RWKV-4 WKV | `e^{u+k}`, `e^{-e^w}` | **高** | **running-max `(a,b,p)` 状態で全 `exp` 前に max 減算**。decay は `e^{-e^w}∈(0,1)` に reparam。**無いと数百トークンで NaN**。 |
| RetNet | scalar `γ<1` | 無 | head ごと GroupNorm。exp なし。 |
| GLA | `α_t∈(0,1)` 対角 | cumprod underflow | **log 空間 cumsum + chunk 内 max 減算**。 |
- 一般則: decay は `exp(負)` で表す（`-e^w`, `Δ·A`）/ softmax 型の分子分母は running-max / cumprod は log 空間 cumsum。

RWKV-4 安定化 WKV（per-step、`a,b`=分子分母, `p`=running-max, 初期 `a=b=0,p=-1e30`）:
```
q  = max(p - e^w, u + k);  e1 = exp(p - e^w - q);  e2 = exp(u + k - q)
wkv = (e1·a + e2·v) / (e1·b + e2)
q2 = max(p - e^w, k);  e1'= exp(p - e^w - q2);  e2'= exp(k - q2)
a, b, p = e1'·a + e2'·v,  e1'·b + e2',  q2
y = W_o · (sigmoid(r) ⊙ wkv)
```
（time-mix の k/v/r は token-shift 混合 `μ⊙x_t + (1-μ)⊙x_{t-1}` を線形射影。channel-mix は squared-ReLU MLP。pre-LN 残差 + 埋め込み直後の LN0。出典 arXiv:2305.13048 §4.2-4.3）。RetNet/GLA/Mamba の式は formulation recon（タスク出力）参照。

## 4. 検証（honest head-to-head・measure recon より）

同じ日本語コーパス（青空文庫「吾輩は猫である」000148）・同 tokenizer（full text から構築）・同 split（`train_val_split` val_frac=0.1 連続末尾）で:

**(1) 能力**: `held_out_report_any` で GPT と再帰を**同一 val トークン**上で採点（`n_tokens` 一致を assert）。共有 `unigram_ppl` を基準線に、`model_ppl` と `passes_gate`（≤0.85×unigram）と `is_degenerate` を比較。`ppl_ratio = rnn_ppl/gpt_ppl`。block_size=64（smoke）と 256（p1）で。

**(2) メモリ**: 
- 解析式: `gpt_kv_bytes(T)=2·L·T·D·4`（傾き `2·L·D·4` B/token、p1 で ~18KB/char）vs `rnn_state_bytes=K·L·D·4`（**T 非依存・傾き 0**、p1 GRU で ~9KB 固定）。
- `gpt_kv_bytes(T)` は **解析投影値**。`T > block_size` ではこの GPT 実装は構造的に実行不能であり、比較図では「反実仮想の延長線」と明記する。
- 実測（**主**=Method 1, allocator 非依存, `@torch.no_grad()`）: `tensor_bytes(*state)` を T ステップ後に測る（再帰=定数 / GPT KV キャッシュ=増大）。補助に tracemalloc/RSS（warmup→測定、`gc.collect()`、T ごと別プロセス、ノイズ注意）。
- **罠（必ず log）**: 必ず `no_grad`（grad 有だと全モデルが O(T) に見える偽陽性）/ PyTorch CPU allocator は解放しない→ Method 1 優先 / パラメータ重みは状態でない（別計上）/ 同能力に幅 D を要したら定数が増える（matched-width で再計算）。

**(3) throughput（任意・決定的）**: `torch.set_num_threads(1)`、min-of-repeats で tok/s を `prompt_len∈{1,16,64,256,1024}` で。GPT は block_size で頭打ち、再帰は flat。

**(4) Pareto 判定（事前登録）**:
- x=能力(PPL, 共有 unigram 基準), y=memory@T と**傾き(B/token)**。再帰は水平線(傾き0)、GPT は上昇線。
- **「再帰が local-AI として優れる」= 両方成立**: ①能力が大きく劣らない（`rnn_ppl ≤ gpt_ppl·1.10` かつ unigram gate 通過かつ非崩壊）、②メモリ勝ちが構造的（実測で再帰状態が T で flat、GPT は線形、解析+Method1 で確認）。
- **「能力の代償に見合わない」**: block_size 以内で `rnn_ppl > gpt_ppl·1.10`。ただし用途が T≫block_size を要するなら別（そこでは GPT は動けない）。
- **正直な決め台詞**: 「block_size(={64,256}) を超える文脈では GPT は不適用、再帰は定数 {~9KB} 状態で走る。block_size 以内の能力差は {X}%」。X を定量化、手を振らない。
- 出力: `out/<run>/recurrent_vs_gpt.json`（capability/memory/throughput/pareto/verdict/caveats）+ 表 + Pareto 図。`libexec/raptor-run-lifecycle` 経由。

## 5. フェーズ / DoD

- **Phase 1**: `recurrent.py`（GatedRNN ベース `RecurrentLM`）+ `held_out_report_any` + tests（並列=逐次一致は GatedRNN では自明、(a)同ハーネス差し込み (b)因果性 (c)generate メモリが T 非依存 (d)NaN 無し）。既存 P0 テスト緑維持。mypy/ruff クリーン。日本語で動作 + メモリ曲線（O(1) vs O(T)）を実証。
- **Phase 2**: RWKV-4（安定化）or RetNet/GLA を追加 → 能力 head-to-head → Pareto verdict を本 doc に記録。並列訓練形 vs 逐次推論形の数値一致テスト必須（RWKV/RetNet）。
- **Phase 3（任意）**: 勝者を量子化+mmap（重みの効率表現）で「RAM 超え」へ。clean-room 3D で歩く。
- **DoD**: recurrent.py + tests green / mypy strict / ruff / 既存 P0 緑。日本語で GPT vs 再帰の PPL + メモリ@T 表 + Pareto を本 doc に記録。敵対レビューで比較公平性・実装正当性（因果性 / 定数メモリ / 数値安定 / 並列=逐次一致）確認。honest 留保明記。

## 6. 出典（一次ソース）

- RWKV v4: arXiv:2305.13048（§4.2-4.3 Eq.14-22）/ v5,v6: arXiv:2404.05892 / v7: arXiv:2503.14456 / 最小実装: johanwind.github.io/2023/03/23/rwkv_details.html / BlinkDL/RWKV-LM（enwik8 bpc）
- RetNet: arXiv:2307.08621（Eq.4-8,14,19）/ GLA: arXiv:2312.06635（Eq.9-17）/ 線形注意: arXiv:2006.16236（Eq.9-11）
- Mamba: arXiv:2312.00752（Alg.1-2）/ S4D: arXiv:2206.11893 / Lightning-2: arXiv:2401.04658
- llcore 再利用: `research/verifier_navigability_gpu/bg10_gpu_lm.py:208-248`（GatedRecurrentLM, 学習可能・直接再利用）/ `research/third_axis_gpu/m3_kernel.py:136-163`（HybridLM, 参照）/ `src/llcore/state_update/genes.py`（NumPy 参照）/ `src/llcore/kernel/rwkv.py`（参照）

## 7. 状態
- 2026-06-15 計画確定（branch `feat/lm-recurrent`）。実装は llterm へ委譲可（本 doc を grounding に自走）。本書 = 進捗の正本。
- 2026-06-15 Phase 1 実装完了:
  - `src/llcore/lm/recurrent.py` に `RecurrentLM` / `RecurrentConfig` を追加
  - `held_out_report_any` を追加し、既存 `Trainer` / `generation` ハーネスで再帰 LM が動作
  - 因果性・定数状態・shared harness の unit test を追加済み
- 2026-06-15 Phase 2 実装完了:
  - `src/llcore/lm/rwkv.py` に running-max 安定化付き `RWKVLM` / `RWKVConfig` を追加
  - step scan と teacher-forced forward の一致、因果性、定数状態の unit test を追加済み
- 2026-06-15 head-to-head smoke 追加:
  - `src/llcore/lm/compare.py` で GPT / Recurrent / RWKV の同一 split 比較とメモリ集計 JSON を出力可能にした
  - smoke 出力: `out/lm_recurrent_smoke.json`
  - smoke 条件は `block_size=64, n_layer=2, n_embd=64, max_iters=40` の短時間版で、**3 モデルとも unigram baseline に敗北** (`gpt_ppl≈586`, `recurrent≈698`, `rwkv≈921`, `unigram≈215`)
  - したがって現時点の smoke は **能力 verdict ではなく**、「比較 API が動く」「再帰状態が GPT の `gpt_kv_bytes(T)` と異なり T 非依存である」ことの確認に留まる
  - 確認済みメモリ定数:
    - `RecurrentLM` state = `512 B`
    - `RWKVLM` state = `2560 B`
    - 同条件 GPT の解析的 KV cache は `T={1,16,64,256}` で `{1024,16384,65536,262144} B`
    - ただし `T=256 > block_size=64` は **実行値ではなく解析投影**。この GPT 実装は `block_size` 超を実際には走れない
  - 次の honest 課題:
    - 学習反復を増やした本比較 (`block_size=64` と `256`) を実施
    - `passes_gate` と `ppl_ratio_vs_gpt` が意味を持つ水準までまず GPT 自体を unigram 下へ落とす
    - その後に Pareto verdict を更新する
- 2026-06-15 review 反映:
  - RWKV-4 の WKV 出力式から誤って入っていた decay 二重適用を除去し、BlinkDL 参照式へ補正
  - 1-step 数値テストを追加し、state 更新と出力式の両方を独立参照値で回帰保護
  - `held_out_report` / `held_out_report_any` は `forward_logits` 版へ一本化
  - 空プロンプト `generate` は GPT / Recurrent / RWKV の 3 モデルで `ValueError` に統一
  - `compare.py` には `gpt_kv_bytes` の解析投影注記、出力先親ディレクトリ作成、`n_head` 明示 config を追加
  - PowerShell の検証コマンドは glob が展開されないため、実際に通る形は `py -3.11 -m pytest tests/unit -k lm -q` を使う
  - branch 上には今回の 3 commit と無関係な既存 dirty (`.llterm/loop_ledger.jsonl`, `assets/articles/llcore_landscape_real.svg`, `docs/PROGRESS.md`, `docs/next_plan.md`, `research/verified_lm_evolution/make_trajectory.py`) が残っている。push 前に別件として分離が必要
- 2026-06-15 compare 出力拡張:
  - `src/llcore/lm/compare.py` は `throughput` / `pareto` / `caveats` を JSON に追加
  - throughput は `torch.set_num_threads(1)` + min-of-repeats で収集し、`prompt_len > block_size` の GPT は **non-executable** と明示
  - smoke (`max_iters=20`, `throughput_new_tokens=8`) では、速度は prompt が短い範囲で `RecurrentLM > GPT > RWKV`、長文では全再帰モデルが線形劣化する一方、GPT は `block_size` 超を exact には測れない
  - この throughput smoke も capability verdict ではなく、主目的は JSON schema と disclosure の確認
