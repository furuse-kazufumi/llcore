# llcore 0.2.0a0 — Kernel Plugin アーキテクチャ設計

**作成**: 2026-05-29
**status**: 設計 doc (Design, 未実装)。本 doc は **G タスク** (ARCHITECTURE_LANDSCAPE §9 短期) の成果物。
**目的**: research/ で実証した複数アーキ (SNN-LIF / Neural ODE / GNN / Izhikevich) を、
本流 `src/llcore/` に **構造破綻なく additive 取り込み**するための plugin 境界を formal 化する。
**前提**: ARCHITECTURE_LANDSCAPE.md §5.3 の横断的気付き #2「same verifier stack は overclaim →
**kernel + verifier backend plugin pattern** が必要」を実装契約に落とす。
**読者**: ユーザー (主) + Claude 次セッション + Codex pair-review。

---

## 0. TL;DR (3 文)

1. 現状 `src/llcore/` は **RWKV 3-param gene (`StateUpdateGene`) に硬結合**しており、
   `minimal_ga` / `verifier/invariants` / `changeop` が 3 次元前提で書かれている。
2. 0.2.0a0 では **2 つの Protocol (`Kernel` / `VerifierBackend`) + 汎用 `GeneCodec`** を追加し、
   GA・進化ループを gene 型非依存に一般化する (既存 RWKV path は完全保存 = semver 安全)。
3. 最初の plugin 移植対象は **SNN-LIF** (導入価値 中-高、構造破綻防止 A-D 全 PASS 実績)。
   research/ コードを「同じ design pattern + partial stack reuse」契約に沿って `src/llcore/kernel/snn_lif.py`
   に昇格する path を本 doc で規定する。

---

## 1. 現状の結合点 (plugin 化を阻む硬結合の棚卸し)

実コードを読んで特定した「RWKV 3-param 前提」の硬結合箇所。0.2.0a0 はこれらを **additive に**
ほどく (既存シンボルは削らない)。

| # | 箇所 | 硬結合内容 | 一般化方針 |
|---|---|---|---|
| C1 | `state_update/genes.py:StateUpdateGene` | 3 field (decay/mix/gate_str) 固定。`as_array`→shape (3,)、`from_array`→shape (3,) assert | `GeneCodec` Protocol で dim/bounds を外出し。`StateUpdateGene` は **RWKV 実装**として温存 |
| C2 | `evolution/minimal_ga.py` | `Individual.gene: StateUpdateGene` / `gene_matrix`→(N,3) / `uniform_mutate` noise size=3 / `crossover_uniform` mask size=3 / `initialize_random_population` 3 uniform draws / `FitnessFunc = Callable[[StateUpdateGene, rng], float]` | gene 型を `GeneT` (TypeVar) に。operator は `GeneCodec` 経由で dim/bounds を取得 |
| C3 | `verifier/invariants.py` | `from llcore.state_update import StateUpdateGene`。`verify_gene_safe` は RWKV 数式 (`decay*s+(1-decay)*tanh(...)`) 固定 | `VerifierBackend` Protocol。RWKV 数式は `RWKVStateNormBackend` として実装に閉じ込め |
| C4 | `verifier/changeop.py` | `OP_TYPES = (decay_shift, mix_shift, gate_shift, kernel_swap_mock)` が RWKV gene 専用 | ChangeOp の op_type 集合を **kernel ごとに宣言**する `Kernel.change_op_types` に。`apply_changeop` は kernel 内へ |
| C5 | research verifier (`snn_verifier.py` / `izh_verifier.py`) | `sys.path.insert(0, str(_SRC))` hack で `is_z3_available` に到達 | plugin 化で `from llcore.verifier import is_z3_available` 正規 import に。sys.path hack 撤廃 |

> **honest**: C1-C4 は「3 次元なら動く」ではなく「3 次元しか動かない」硬結合。
> ただし `verify_gene_safe` の戻り値型 `InvariantResult` と `evolve` の `FitnessFunc` の
> **shape は既に汎用的**なので、一般化コストは型の抽象化が中心で、ロジック書き換えは局所的。

---

## 2. plugin 境界の 3 つの抽象 (Protocol 定義)

既存コードが `typing.Protocol` を採用済 (`factor_hook/hooks.py:ThoughtFactorDeltaHook`) なので、
同じ structural typing で統一する。**3 つの抽象**で「同じ design pattern」を契約化する。

### 2.1 `GeneCodec` — gene の numpy 往復 + 範囲

GA operator (mutate/crossover/init) を gene 型非依存にするための最小契約。
`StateUpdateGene` / `LIFGene` / `IzhikevichGene` は全て `as_array` / `clipped` を既に持つので、
codec は「dim と bounds を宣言する薄い層」で済む。

```python
# src/llcore/kernel/protocol.py (新規, 設計案)
from __future__ import annotations
from typing import Protocol, runtime_checkable, TypeVar
import numpy as np

GeneT = TypeVar("GeneT")

@runtime_checkable
class GeneCodec(Protocol[GeneT]):
    """gene <-> numpy 往復 + 物理範囲 bounds を宣言する契約."""

    @property
    def dim(self) -> int:
        """gene parameter 数 (RWKV=3, LIF=4, Izhikevich=4)."""

    @property
    def lower(self) -> np.ndarray:
        """各 param の下界, shape (dim,). (RWKV: [0,-1,-2])"""

    @property
    def upper(self) -> np.ndarray:
        """各 param の上界, shape (dim,). (RWKV: [1,1,2])"""

    def to_array(self, gene: GeneT) -> np.ndarray: ...
    def from_array(self, arr: np.ndarray) -> GeneT: ...
    def clip(self, gene: GeneT) -> GeneT:
        """clipped gene を返す (post-clip 補正 = LIF の V_reset<V_th 等も含む)."""
```

> `clip` を codec が持つのが重要: LIF は `V_reset < V_th` の post-clip 補正、Izhikevich は
> `c < V_PEAK` の補正という **box clip では表せない依存制約**があるため、単純 bounds clip では
> 不足。`clipped()` メソッドの委譲で吸収する。

### 2.2 `Kernel` — 1 アーキの「中身」(simulate + ChangeOp)

> **Codex review 反映 (M1)**: 現 SNN simulator は `simulate_lif(gene, I_input, T, dt, V_init)
> -> tuple[np.ndarray, list[float]]` (V 列 + spike 時刻列) で、RWKV `run_sequence(inputs, gene,
> initial_state) -> np.ndarray` と **入口・戻り値・時間長の渡し方が別物**。`np.ndarray` 1 本では
> 吸収できないため、戻り値を `Trajectory` 型に統一し、kernel ごとの adapter で正規化する。

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class Trajectory:
    """kernel 横断の simulate 戻り値. 意味論差を field で明示 (M1)."""
    primary: np.ndarray            # RWKV=state 列 (L+1, dim), SNN=膜電位 V 列 (T,)
    events: tuple[float, ...] = () # SNN=spike 時刻列. RWKV は空 tuple
    kind: str = "state"            # "state" | "spike_voltage" — 上流が意味論分岐に使う

@runtime_checkable
class Kernel(Protocol[GeneT]):
    """1 アーキの forward dynamics + 構造変更 op を束ねる plugin."""

    name: str                          # "rwkv" / "snn_lif" / "izhikevich"
    codec: GeneCodec[GeneT]
    change_op_types: tuple[str, ...]   # kernel 固有の ChangeOp 種別 (C4 一般化)

    def simulate(self, inputs: np.ndarray, gene: GeneT, **kw) -> Trajectory:
        """L step trajectory を Trajectory で返す. RWKV は run_sequence を、
        SNN は simulate_lif を adapter で包んで正規化する (M1)."""

    def apply_change_op(self, gene: GeneT, op: "ChangeOp") -> GeneT:
        """ChangeOp を適用した新 gene を返す (M2: 既存 ChangeOp 型をそのまま受ける).
        kernel ごとに op_type の意味が違う (RWKV: decay/mix/gate_shift,
        SNN: tau/vth/tref_shift) が、ChangeOpSequence / refinement.py との
        再利用性を保つため (op_type, delta) には潰さない."""
```

> **honest 留保 (ARCH_LANDSCAPE §5.3 #2 直結)**: Kernel Protocol は「same **design pattern**」を
> 契約化するもので、「same **verifier stack**」ではない。`Trajectory.kind` で意味論差を
> **型レベルに明示**し (M1 対応)、上流 (fitness/verifier) は `kind` で分岐する。汎用化しすぎない。
>
> **Codex review 反映 (M2)**: `apply_change_op` は当初 `(op_type, delta)` 引数を提案したが、実コードの
> 中心は `ChangeOp` / `ChangeOpSequence` + `apply_changeop(gene, op: ChangeOp)` であり、`(op_type,
> delta)` に潰すと `ChangeOpSequence` ベースの合成性 (`compose`) と `refinement.py` 接続の再利用性が
> 下がる。**`ChangeOp` 型を受ける**形に修正済。kernel 別 op_type は `change_op_types` で宣言し、
> `ChangeOp.__post_init__` の `OP_TYPES` 検証を kernel 別に拡張する (§4.1)。

### 2.3 `VerifierBackend` — Z3 invariant gate の plugin

戻り値型 `InvariantResult` は既存 (`verifier/invariants.py`) を **共通 currency** にする。

> **Codex review 反映 (補足 Finding)**: research verifier は型・引数が別物 — SNN は独自
> `SNNInvariantResult` を返し、関数引数も `safety_margin` / `I_max` / `n_spikes` / `T_window_ms` 等
> kernel 固有。よって backend は単なる re-export でなく、**`SNNInvariantResult → InvariantResult`
> へ正規化する adapter 層**を持つ。`InvariantResult` / `SNNInvariantResult` は field がほぼ同型
> (`ok` / `used_z3` / `reason` / `counterexample`) なので、adapter は機械的変換で済む。
> backend が複数 `verify_*` (膜電位 + firing rate) を呼ぶ場合は **AND 集約** (全 ok で admit)。

```python
from llcore.verifier import InvariantResult  # 既存型を再利用

@runtime_checkable
class VerifierBackend(Protocol[GeneT]):
    """per-gene online gate. 進化ループから 1 gene ずつ呼ばれる."""

    name: str

    def verify_gene_safe(self, gene: GeneT, **kw) -> InvariantResult:
        """gene が安全 invariant を破らないか Z3 で検査. ok=True で admit."""

    def is_available(self) -> bool:
        """z3 が import 可能か (False なら ok=True default で skip)."""
```

**backend 一覧 (移植時の対応表)**:

| backend | 元 research 関数 | invariant | per-gene 真正性 (Codex 検証済) |
|---|---|---|---|
| `RWKVStateNormBackend` (本流) | `verify_gene_safe` (invariants.py) | `\|state\| <= bound` | **真** (`z3.RealVal(g.decay)` 等で gene を制約に投入) |
| `SNNLifBackend` | `verify_membrane_bounded_per_gene` + `verify_firing_rate_per_gene` | 膜電位 bounded + firing rate | **真 (限定付き)** — `tau_m_const`/`v_th_const`/`v_reset_const` を Z3 制約に具体投入。ただし **I_max 入力契約 + 1-step Euler + t_ref 未関与** の限定 |
| `IzhikevichBackend` | `verify_v_bounded_per_gene` + `verify_firing_rate_per_gene` | v² 1-step Euler + dt packing | **box 流用** (Codex 既出 F1: gene.clipped() で box → 真の per-gene でない) |

> **per-gene verifier の罠 (ARCH_LANDSCAPE §5.3 #3)**: backend を「per-gene」と名乗る前に、
> gene parameter が **Z3 symbolic constraint に実際に入っているか**を監査する。
> **Codex review で実コード確認済 (L1 訂正)**: SNN-LIF `verify_membrane_bounded_per_gene` は
> `gene.clipped()` 後に **具体値を Z3 制約に埋込んでおり真の per-gene** (RWKV と同質)。当初表の
> 「要監査」は過度に保守的だった → 「**true per-gene under assumed-input 1-step contract**」が正確。
> 一方 Izhikevich は box 流用のまま (降格表示維持)。`gene.clipped()` で box 範囲に流用するだけの
> ものは per-gene ではない、という罠の判定基準は変わらず — backend docstring で **どちらか明示**する。

---

## 3. 進化ループの一般化 (C2 のほどき方)

`minimal_ga.evolve` を gene 型非依存にする。**既存 API は壊さず**、新しい一般化版を追加する
2 段構え (semver 安全)。

### 3.1 operator の codec 化 (additive)

```python
# 新: codec を受け取る汎用 operator
def uniform_mutate_g(gene, codec: GeneCodec, sigma, rng):
    arr = codec.to_array(gene) + rng.normal(0, sigma, size=codec.dim)
    return codec.clip(codec.from_array(arr))

# 既存: uniform_mutate(gene: StateUpdateGene, sigma, rng) は
#       内部で RWKV codec を使う薄い wrapper として温存 (後方互換)
```

`gene_matrix` (現 (N,3) 固定) は `Population.gene_matrix` が diversity 計算
(`gene_matrix.var()`) に使われるだけなので、codec.dim 任意で動くよう `to_array` stack に置換。

### 3.2 `evolve` の型パラメータ化

- `Individual.gene: StateUpdateGene` → `Individual[GeneT]` (Generic dataclass)
- `FitnessFunc = Callable[[GeneT, rng], float]`
- `evolve(..., codec: GeneCodec, initial_pop=...)` に codec を渡す。
  codec 省略時は **RWKV codec をデフォルト**にして既存呼び出しを保つ。

> **破綻防止**: 既存 67+ 進化系 test が `StateUpdateGene` 前提なので、
> 「codec デフォルト = RWKV」で**シグネチャ後方互換**を保証する。新 test で LIF codec path を追加。

> **Codex review 反映 (M3) — 後方互換は evolve だけでは不十分**: `minimal_ga` の RWKV 固定は
> `evolve` に留まらず、以下のシンボルに広がっている。後方互換の十分条件は「これら全てを
> **RWKV wrapper として温存**し、一般化版を別名で additive 追加する」こと。`evolve(..., codec=RWKV)`
> 1 点に根拠を寄せない。
>
> | 温存対象シンボル | 一般化版 (additive) |
> |---|---|
> | `Individual.gene: StateUpdateGene` | `Individual[GeneT]` (Generic 化、デフォルト束縛で旧 import 維持) |
> | `FitnessFunc = Callable[[StateUpdateGene, rng], float]` | `FitnessFunc[GeneT]` (TypeVar) |
> | `evaluate_population(genes, ...)` | codec 不要 (gene 型に依存しない) — 変更なしで OK |
> | `initialize_random_population(pop_size, rng)` | `initialize_random_population_g(pop_size, codec, rng)` |
> | `uniform_mutate` / `crossover_uniform` (size=3 固定) | `*_g(gene, codec, ...)` (§3.1) |
> | `Population.gene_matrix` ((N,3) 固定) | codec.to_array stack で任意 dim、旧 property は RWKV stack 維持 |
>
> 旧シンボルは「RWKV codec を内部固定した薄い wrapper」に書き換える (シグネチャ・戻り値不変)。

---

## 4. SNN-LIF 取り込み path (最初の plugin、step-by-step)

導入価値マトリクス (ARCH_LANDSCAPE §6.2) で SNN-LIF が **中-高** かつ構造破綻防止 A-D 全 PASS 実績。
最初の移植対象とする。research/ → src/ 昇格を **5 step の atomic commit** に分解。

| step | 作業 | 破綻防止チェック |
|---|---|---|
| S1 | `src/llcore/kernel/protocol.py` 追加 (§2 の 3 Protocol)。**RWKV を最初の準拠例**として `RWKVKernel` + `RWKVStateNormBackend` を実装し、既存 `verify_gene_safe`/`run_sequence` を委譲 wrapper で包む | (D) src 既存挙動不変 = 既存関数はそのまま、plugin は薄い委譲。145 本流 test 回帰なし |
| S2 | `minimal_ga` 一般化 (§3)。codec デフォルト=RWKV で後方互換。新 test で **任意 dim GA** を検証 | (B) 既存 67 進化 test PASS + 新規 |
| S3 | `src/llcore/kernel/snn_lif.py` 追加。research `LIFGene` + simulator を昇格、`SNNLifKernel` 実装。research 側は `from llcore.kernel.snn_lif import LIFGene` の re-export に置換 (sys.path hack 撤廃 = C5) | (A) plugin 化実証。research test を import 経路だけ差し替えて PASS 維持 |
| S4 | `SNNLifBackend` 実装 + **per-gene 真正性監査** (§2.3 罠)。`verify_membrane_bounded_per_gene` が gene を Z3 symbolic に入れているか確認、box 流用なら docstring で明示降格 | (C) Codex pair-review: per-gene claim の真正性を必ず検証させる |
| S5 | SNN-LIF を `evolve` で実走 (LIF codec + SNNLifBackend gate)。10×10 toy 進化が回り、verifier reject が選択圧になることを smoke | (B)(C) 全 test + Codex review、verdict 更新 |

各 step は **commit 前に Codex review** (`feedback_codex_pair_review_for_llcore`)。
特に S4 の per-gene 真正性は Izhikevich F1 の再発ポイントなので Codex に明示的に突かせる。

### 4.1 ChangeOp の kernel 別宣言 (C4 の SNN 版)

RWKV の `OP_TYPES` (decay/mix/gate_shift) は SNN では意味を持たない。
SNN-LIF の ChangeOp は例えば `tau_shift` / `vth_shift` / `tref_shift` になる。
`Kernel.change_op_types` で宣言し、`apply_change_op` を kernel 内に置くことで
`changeop.py` の RWKV 専用 `apply_changeop` と共存させる (既存は RWKV kernel に移譲)。

---

## 5. 構造破綻防止 4 条件への適合 (本 doc の契約)

ARCHITECTURE_LANDSCAPE §6.1 の 4 条件を 0.2.0a0 移行で**どう守るか**を明文化。

| 条件 | 0.2.0a0 での守り方 |
|---|---|
| **(A) kernel plugin 化可** | 本 doc の 3 Protocol が plugin 境界。RWKV が最初の準拠例 (dogfooding)、SNN-LIF が 2 例目で「2 アーキで成立」を実証 |
| **(B) 既存 test 回帰なし** | 全 step で `pytest`。codec デフォルト=RWKV でシグネチャ後方互換。research test は import 経路差替のみ |
| **(C) Codex pair-review 通過** | 各 step commit 前。特に **per-gene 真正性** (S4) と **「same design pattern」境界** (Kernel Protocol が verifier stack を共有すると overclaim していないか) を突かせる |
| **(D) semver 互換** | 0.1.0a0 の公開シンボル (`StateUpdateGene`, `evolve`, `verify_gene_safe`, **および §3.2 M3 表の全シンボル**) は **削除も挙動変更もしない**。旧シンボルは RWKV codec 固定 wrapper として温存、一般化版は別名で additive 追加。`0.1.0a0 → 0.2.0a0` は alpha bump で破壊的変更なし |

> **semver の honest 注記**: 0.x alpha なので厳密な semver 保証義務はないが、
> **「単純→複雑は進化型で困難」(ユーザー哲学)** に従い、最初から拡張余地を残す設計にする。
> Genome3D の 2D matrix (llive) のような多層化は本 doc scope 外 (kernel 単位の plugin に限定)。

---

## 6. 非目標 (scope 外、overclaim 防止)

- **llive 依存の再導入はしない**: kernel plugin は numpy + optional z3 のみ。llive Genome3D / lldarwin_v2
  は参照せず (llcore 独立路線 = `project_llcore_init_2026_05_29` 確定方針)。
- **「同一 verifier で全アーキ検証」は目標にしない**: §2.2 honest 留保どおり「same design pattern +
  partial stack reuse」が正確な達成目標。Z3 invariant の中身は kernel 固有。
- **GPU / 実 NN kernel 交換はしない**: `kernel_swap_mock` は mock のまま。実 RWKV/SSM kernel 置換は Stage 5+。
- **真の per-gene formal proof の全 kernel 達成は約束しない**: backend ごとに「真の per-gene」か
  「box proof」かを honest に明示する (Izhikevich は box のまま降格表示)。

---

## 7. 次のアクション (本 doc 後)

1. ~~本 doc を Codex review~~ **完了 (2026-05-29)**: Codex (gpt-5.4) read-only review → 5 Findings
   (Medium 3 / Low 2 + 補足 1)。全件実コード検証の上 doc に反映済 (§2.2 M1 Trajectory + M2 ChangeOp 型 /
   §2.3 補足 result 正規化 adapter / §2.3 表 L1 per-gene 真正性訂正 / §3.2 §5 M3 シンボル温存)。
   **設計の overclaim 残りなし、Protocol 境界は adapter 前提で締まった**。
2. S1 着手判断 (ユーザー承認後): `src/llcore/kernel/protocol.py` (`GeneCodec` / `Kernel` / `Trajectory` /
   `VerifierBackend`) + RWKV 準拠例 (`RWKVKernel` / `RWKVStateNormBackend`)
3. 並行候補: ARCH_LANDSCAPE §9 の C (真の per-gene verifier, Izhikevich F1 直接対応) は
   本 plugin 設計の S4 と論点が重なるため、S4 で一緒に解消する設計

---

## 8. 関連

- `docs/ARCHITECTURE_LANDSCAPE.md` §5.3 (#2 kernel+verifier backend plugin / #3 per-gene 罠) / §6 (4 条件)
- `[[project_llcore_init_2026_05_29]]` — llcore 独立路線 + Stage 0-3 + research phase
- `[[feedback_codex_pair_review_for_llcore]]` — 各 step commit 前 review 規律
- `[[feedback_benchmark_honest_disclosure]]` — claim 降格規律 (per-gene 真正性監査の根拠)
- src 接地点: `state_update/genes.py` (C1) / `evolution/minimal_ga.py` (C2) /
  `verifier/invariants.py` (C3) / `verifier/changeop.py` (C4) / research `snn_verifier.py` (C5)

---

**doc 完成日**: 2026-05-29
**次の更新トリガ**: Codex review 反映 / S1 着手 / SNN-LIF 移植開始
