# `llcore.memory` — メモリ効率ツールキット (facade)

> 状態: branch C「メモリ効率ツールキット化」(2026-06-19, ユーザー選択)。
> 正本データ = [`MEMORY_EFFICIENCY_FINDINGS.md`](MEMORY_EFFICIENCY_FINDINGS.md) /
> 立ち位置 = [`POSITIONING_VS_LLAMACPP.md`](POSITIONING_VS_LLAMACPP.md)。

## これは何か（一文）

`llcore.memory` は、これまで `scripts/` と `src/llcore/lm/` に**散在していた
検証済みメモリ効率プリミティブ**を、**1 つの import + 1 回の計測**にまとめた
facade（受付窓口）である。新しいアルゴリズムではなく **packaging**。

```python
from llcore.memory import measure_memory
report = measure_memory(model, val_ids=val_ids)   # fp32 char-LM を渡すだけ
print(report.to_dict())
# 実走例 (out/lm_aozora_multi_smoke/model.pt, 青空マルチ全文 val=330,368 token):
# {'fp32_bytes': 5502976, 'int8_bytes': 1506072, 'compression_ratio': 0.2737,
#  'percent_smaller': 72.63, 'fp32_top1': 0.3629, 'int8_top1': 0.3631,
#  'retention': 1.0004, 'capability_gate_pass': True, 'min_retention': 0.97,
#  'n_eval_tokens': 330368}
# 注: int8 が fp32 を 0.0002 上回るのは「改善」でなく同点 argmax の測定ノイズ
#     (この規模では int8 量子化の capability コストが実質ゼロ)。
```

## なぜ facade を作るのか（honest）

llcore の価値は「（豆粒のような）モデルそのもの」ではなく、その周りの
**メモリ効率インフラ層**にある（[`POSITIONING_VS_LLAMACPP.md`](POSITIONING_VS_LLAMACPP.md)）。
しかし検証済みの道具が `int8_quant_footprint.py` / `mmap_ram_exceed_poc.py` /
`quant_bitwidth_sweep.py` … と散らばっていて、外から「どれを呼べば何が分かるのか」
が見えなかった。facade はそれを「**どれだけ小さくなり、その代償に何を失うか**」
という 1 つの問いに畳む。

**正直なスコープ**:
- これは **新規アルゴリズムではない**。int8 量子化 / mmap / 定数状態 recurrent は
  llama.cpp / GGUF 系の確立手法の**再導出**（confidence high,
  [`POSITIONING_VS_LLAMACPP.md`](POSITIONING_VS_LLAMACPP.md)）。
- 誠実な独自性は **運用** に限る = 「フットプリント勝ちを capability gate で
  fail-closed に検収する」こと（後述）。
- フットプリント数値は **常駐重みバイト数（storage）** であり、プロセス RSS ではない。
- retention は **simulated int8**（重みのみ・行列積は fp32）での計測 =
  これは footprint / working-set 最適化であって int8-GEMM 速度向上ではない。

## facade が再 export するもの（すべて原本と同一オブジェクト）

| 名前 | 出自 | 役割 | FINDINGS |
|---|---|---|---|
| `quantize_per_channel_int8`, `Int8Linear`, `convert_linears_to_int8` | `lm.quant` | per-行対称 int8 + streaming-dequant Linear | (b)/(c) |
| `save_int8_checkpoint`, `load_int8_model` | `lm.quant` | int8 チェックポイント保存 / mmap streaming ロード | (a') |
| `int8_footprint_bytes` | `lm.quant` | fp32 vs int8 常駐重みバイトの honest 会計 | (b) |
| `held_out_top1_report`, `passes_capability_gate` | `lm.eval` | top-1/top-5 精度 + **fail-closed** 保持率 gate | (b') |
| `RecurrentLM`, `RecurrentConfig`, `constant_state_bytes` | `lm.recurrent` / `lm.compare` | 定数サイズ状態の recurrent | (0')/(a) |
| `gpt_kv_bytes` | `lm.compare` | GPT の KV が文脈長に**線形**に膨らむ会計 | (0') |

## `measure_memory` / `MemoryReport`

```python
measure_memory(
    model: CharGPT,
    *,
    val_ids: torch.Tensor | None = None,    # 与えれば capability も測る
    block_size: int | None = None,          # 既定 = model.config.block_size
    batch_size: int = 32,
    min_retention: float = 0.97,            # cap-gate の床
    context_lens: Sequence[int] | None = None,  # 与えれば KV 成長軸も測る
) -> MemoryReport
```

`MemoryReport` は最大 **3 軸**を 1 つにまとめる:

1. **フットプリント軸**（常に計測, read-only, `int8_footprint_bytes`）。
2. **capability 軸**（`val_ids` を渡したとき）= **モデルの deep copy を量子化**して採点
   → **呼び出し側の fp32 モデルは決して破壊されない**（テストで固定）。
   `retention = int8_top1 / fp32_top1`、`capability_gate_pass =
   passes_capability_gate(int8_top1, fp32_top1, min_retention)`。評価データが
   無いときは capability 系フィールドは `None` = **「無い」を捏造せず「無い」と報告する**。
3. **構造成長軸**（`context_lens` を渡したとき）= `kv_bytes_by_context[T] =
   gpt_kv_bytes(model, T)`。GPT の KV キャッシュは文脈長 T に**線形**に膨らむ。
   これは **定数状態 recurrent が回避する当のもの**（recurrent の状態は T 非依存=平坦,
   `constant_state_bytes` 参照）。= 量子化(静的 footprint)と KV 成長(動的)の両軸を 1 レポートで。

### fail-closed cap-gate が誠実な独自点

bit 幅スイープ（FINDINGS (b')）で **「PPL だけの gate は、top-1 が半減した
壊れた低ビットモデルを PASS させる」** ことが実証された。`measure_memory` は
footprint の勝ちを **top-1 保持率で fail-closed に検収**する。これが
「再導出されたプリミティブ群」に対して llcore が足す唯一の運用上の価値である。

この gate を **昇格ゲート（promotion gate）** として実体化したのが CLI の
`--save-int8` である: int8 チェックポイントの emit を **cap-gate PASS のときだけ
許可**し、FAIL または「コーパス未指定＝capability 未計測」のときは **fail-closed で
書き込みを拒否**する（`--force` で運用者が上書き可）。「どれだけ小さくなるか」だけで
なく「**その縮小を本番に出して良いか**」までを 1 コマンドで検収する。

## CLI

```powershell
# フットプリントのみ
py -3.11 -m llcore.memory report out\lm_run\model.pt

# capability retention も（ローカルコーパスで）
py -3.11 -m llcore.memory report out\lm_run\model.pt --corpus-file corpus.txt --json out\mem_report.json

# 昇格ゲート: cap-gate PASS のときだけ int8 ckpt を emit（FAIL/コーパス無→fail-closed で拒否）
py -3.11 -m llcore.memory report out\lm_run\model.pt --corpus-file corpus.txt --save-int8 out\model_int8.pt
```

出力例（実走 = `out/lm_aozora_multi_smoke/model.pt`、青空マルチ全文）:
```
checkpoint     : out/lm_aozora_multi_smoke/model.pt
fp32 weights   : 5.50 MB
int8 resident  : 1.51 MB  (ratio 0.274, 72.6% smaller)
top-1 retention: 100.0%  (fp32 0.3629 -> int8 0.3631, n=330368)
capability gate: PASS (>= 97% retention)
```
（retention が 100% を僅かに超えるのは int8 が fp32 を上回ったのではなく、
同点 argmax の測定ノイズ。この規模では int8 の capability コストが実質ゼロ。）

## 設計指針: 「良いハードに載せ替えるほど効く」（設計仮説・本デリバラブルでは未計測）

このツールキットは CPU 専用の延命策**ではない**。各プリミティブは
ハードが良くなるほどスペックが跳ね上がる性質を持つ（2026-06-17 ユーザー指針）。
**以下は設計上の含意であり、速度・大 RAM 共有・長文脈は本デリバラブルでは未計測**
（[`MEMORY_EFFICIENCY_FINDINGS.md`](MEMORY_EFFICIENCY_FINDINGS.md) #34 で速度は未測定と明記）:

1. **int8** — 今は storage（footprint）圧縮だが、同じ int8 重みは GPU では
   真の int8 GEMM（tensor core）= **速度**向上に化ける。
2. **mmap** — load コストはほぼ固定（〜1.4MB）。大 RAM ではページキャッシュに
   巨大モデルを常駐させ、1 回のロードを**複数プロセスで共有**できる。
3. **定数状態 recurrent** — GPU + 長文脈で attention の二次メモリ壁なしに
   context を伸ばせる。

## 関連

- データの正本: [`MEMORY_EFFICIENCY_FINDINGS.md`](MEMORY_EFFICIENCY_FINDINGS.md)
  （柱 0'/a/a'/b/b'/b''/b'''/c/d の全数値）
- 立ち位置: [`POSITIONING_VS_LLAMACPP.md`](POSITIONING_VS_LLAMACPP.md)
- スケーリング戦略: [`MEMORY_SCALING_STRATEGY.md`](MEMORY_SCALING_STRATEGY.md)
- 実装: `src/llcore/memory.py` / テスト: `tests/unit/test_memory_facade.py`
