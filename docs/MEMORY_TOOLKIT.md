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
# {'fp32_bytes': ..., 'int8_bytes': ..., 'compression_ratio': 0.26,
#  'percent_smaller': 74.1, 'fp32_top1': 0.58, 'int8_top1': 0.58,
#  'retention': 0.998, 'capability_gate_pass': True, 'min_retention': 0.97, ...}
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
    val_ids: torch.Tensor | None = None,   # 与えれば capability も測る
    block_size: int | None = None,          # 既定 = model.config.block_size
    batch_size: int = 32,
    min_retention: float = 0.97,            # cap-gate の床
) -> MemoryReport
```

- **フットプリントは常に計測**（read-only, `int8_footprint_bytes`）。
- `val_ids` を渡すと **モデルの deep copy を量子化**して採点する
  → **呼び出し側の fp32 モデルは決して破壊されない**（テストで固定）。
- `retention = int8_top1 / fp32_top1`、`capability_gate_pass =
  passes_capability_gate(int8_top1, fp32_top1, min_retention)`。
- 評価データが無いときは capability 系フィールドは `None`
  = **「無い」を捏造せず「無い」と報告する**。

### fail-closed cap-gate が誠実な独自点

bit 幅スイープ（FINDINGS (b')）で **「PPL だけの gate は、top-1 が半減した
壊れた低ビットモデルを PASS させる」** ことが実証された。`measure_memory` は
footprint の勝ちを **top-1 保持率で fail-closed に検収**する。これが
「再導出されたプリミティブ群」に対して llcore が足す唯一の運用上の価値である。

## CLI

```powershell
# フットプリントのみ
py -3.11 -m llcore.memory report out\lm_run\model.pt

# capability retention も（ローカルコーパスで）
py -3.11 -m llcore.memory report out\lm_run\model.pt --corpus-file corpus.txt --json out\mem_report.json
```

出力例:
```
checkpoint     : out/lm_run/model.pt
fp32 weights   : 5.50 MB
int8 resident  : 1.50 MB  (ratio 0.273, 72.6% smaller)
top-1 retention: 99.8%  (fp32 0.5841 -> int8 0.5829, n=12480)
capability gate: PASS (>= 97% retention)
```

## 設計指針: 「良いハードに載せ替えるほど効く」

このツールキットは CPU 専用の延命策**ではない**。各プリミティブは
ハードが良くなるほどスペックが跳ね上がる性質を持つ（2026-06-17 ユーザー指針）:

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
