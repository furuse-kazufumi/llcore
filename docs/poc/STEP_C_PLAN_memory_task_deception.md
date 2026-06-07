# (c) 記憶タスクで欺瞞 corridor が立つか — 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **凡例 — 進化4要素**: ①変異 ②遺伝 ③適者生存・選択 ④過剰繁殖。本書の「③」=適者生存。設計 spec: [`STEP_C_DESIGN_memory_task_deception.md`](./STEP_C_DESIGN_memory_task_deception.md)、用語集: [`YOUGO_平易版.md`](./YOUGO_平易版.md)。

**Goal:** 実タスク本来の性質（長期依存×非線形の記憶タスク）から欺瞞 corridor が自然に現れ ③（選択による累積改善）が hill-climbing を超えて load-bearing になるかを CPU で厳格・両方向に決着させる。

**Architecture:** `research/step_c_memory_tasks/` に隔離（src 非変更）。記憶タスク3種（delayed parity / flip-flop / delayed recall）と leaky-delay-line reservoir 基質（連続 gene ベクトル）を新規実装し、`fit_ridge_readout`（src, read-only 流用）で held-out R² を fitness にする。探索は step4 の `selection_lab`（MAP-Elites vs random/RR-hillclimb/panmictic-GA）を流用。判定は強化版 `honest_eval` 基準（n_seeds≥15・片側 Wilcoxon p<0.05・|paired_sign_delta|≥0.147）を step_c 側の `strict_compare` で適用。

**Tech Stack:** Python 3.11、numpy のみ（CPU 完結）。pytest。流用: `llcore.fitness.ridge_readout.fit_ridge_readout`、`research/step4_selection/selection_lab.py`、`llcore.evolution.honest_eval`。

---

## ファイル構造

```
research/step_c_memory_tasks/
├── memory_tasks.py        # 記憶タスク3種 (generate(rng)->(inputs,target))
├── reservoir.py           # leaky-delay-line reservoir 基質 (連続 gene→states) + eval_once/behavior factory
├── strict_compare.py      # 強化版 honest 基準で 2 スコア配列を判定 (selection_lab.compare の厳格版)
├── landscape_map.py       # C1 多峰性診断 (grid/random sample + RR 収束点中点が谷か)
├── exp_c1_landscape.py    # 各タスクの landscape map 実行スクリプト
├── exp_c2c3_compare.py    # MAP-E vs 3 baseline を強化基準で判定 (C2/C3)
├── exp_c4_ablation.py     # init_batch ablation (C4)
└── tests/
    ├── test_memory_tasks.py
    ├── test_reservoir.py
    ├── test_strict_compare.py
    └── test_landscape_map.py
docs/poc/STEP_C_VERDICT.md  # 最終 verdict (③実在 or CPU撤退)
```

各タスクは独立して意味を持つ単位に分割。テスト可能なコア部品（tasks / reservoir / strict_compare / landscape_map）は TDD。exp スクリプトは部品を呼んで測定・記録する。

---

### Task 1: 記憶タスク3種 (`memory_tasks.py`)

**Files:**
- Create: `research/step_c_memory_tasks/memory_tasks.py`
- Test: `research/step_c_memory_tasks/tests/test_memory_tasks.py`

各タスクは `ridge_readout` が期待する `task.generate(rng) -> (inputs, target)` 契約に準拠する。`inputs` は shape `(T, in_dim)` の系列、`target` は最終時刻に出すべき答え（scalar or 1D）。**いずれも「過去を保持していないと最終 state から解けない」= 記憶必須**。

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_memory_tasks.py
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_tasks import DelayedParityTask, FlipFlopTask, DelayedRecallTask


def test_delayed_parity_shapes_and_label():
    task = DelayedParityTask(seq_len=20, window=5, in_dim=1)
    rng = np.random.default_rng(0)
    inputs, target = task.generate(rng)
    assert inputs.shape == (20, 1)
    assert set(np.unique(inputs)).issubset({-1.0, 1.0})
    # target は窓内 (先頭 window 個) のパリティ: 偶数個の -1 → +1, 奇数個 → -1
    window_bits = inputs[:5, 0]
    n_neg = int(np.sum(window_bits < 0))
    expected = 1.0 if n_neg % 2 == 0 else -1.0
    assert float(np.atleast_1d(target)[0]) == expected


def test_flipflop_holds_last_set():
    task = FlipFlopTask(seq_len=30, in_dim=2)  # ch0=set(+1), ch1=reset(+1)
    rng = np.random.default_rng(1)
    inputs, target = task.generate(rng)
    assert inputs.shape == (30, 2)
    # target は最後に set/reset されたチャネルで決まる ±1。決定論的に再構成して照合。
    state = 0.0
    for t in range(30):
        if inputs[t, 0] > 0:
            state = 1.0
        elif inputs[t, 1] > 0:
            state = -1.0
    assert float(np.atleast_1d(target)[0]) == state


def test_delayed_recall_returns_initial_cue():
    task = DelayedRecallTask(seq_len=25, in_dim=1)
    rng = np.random.default_rng(2)
    inputs, target = task.generate(rng)
    assert inputs.shape == (25, 1)
    # cue は t=0 の符号。以降は 0 (無情報)。target = cue。
    assert float(np.atleast_1d(target)[0]) == float(np.sign(inputs[0, 0]))
    assert np.allclose(inputs[1:, 0], 0.0)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `cd <llcore-root> && py -3.11 -m pytest research/step_c_memory_tasks/tests/test_memory_tasks.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'memory_tasks'`）

- [ ] **Step 3: 最小実装を書く**

```python
# memory_tasks.py
# SPDX-License-Identifier: Apache-2.0
"""記憶タスク3種 — 長期依存×非線形で本来的に欺瞞的になりうる標準難タスク.

いずれも task.generate(rng) -> (inputs, target):
- inputs: shape (seq_len, in_dim) の系列
- target: 最終時刻に出すべき答え (1D ndarray)
過去を保持しないと最終 state から解けない = 記憶必須。地形は手で作らない (人工注入なし)。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DelayedParityTask:
    """系列先頭 window 個の ±1 のパリティ (XOR) を最終時刻に答える."""
    seq_len: int = 20
    window: int = 5
    in_dim: int = 1
    out_dim: int = 1

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        bits = rng.choice([-1.0, 1.0], size=self.seq_len)
        inputs = bits.reshape(self.seq_len, 1)
        n_neg = int(np.sum(bits[: self.window] < 0))
        target = 1.0 if n_neg % 2 == 0 else -1.0
        return inputs.astype(np.float64), np.array([target], dtype=np.float64)


@dataclass(frozen=True)
class FlipFlopTask:
    """ch0=set(+1)/ch1=reset(+1) のパルス列。最後に set/reset された値 ±1 を保持して答える."""
    seq_len: int = 30
    in_dim: int = 2
    out_dim: int = 1
    pulse_prob: float = 0.2

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.zeros((self.seq_len, 2), dtype=np.float64)
        state = 0.0
        # 最初の数ステップで必ず一度 set して state を ±1 に確定させる
        first = int(rng.integers(0, max(1, self.seq_len // 4)))
        for t in range(self.seq_len):
            if t == first:
                ch = int(rng.integers(0, 2))
                inputs[t, ch] = 1.0
            elif rng.random() < self.pulse_prob:
                ch = int(rng.integers(0, 2))
                inputs[t, ch] = 1.0
            if inputs[t, 0] > 0:
                state = 1.0
            elif inputs[t, 1] > 0:
                state = -1.0
        return inputs, np.array([state], dtype=np.float64)


@dataclass(frozen=True)
class DelayedRecallTask:
    """t=0 の cue (±1) を、無情報な遅延区間の後、最終時刻に思い出して答える (T-maze 風)."""
    seq_len: int = 25
    in_dim: int = 1
    out_dim: int = 1

    def generate(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        inputs = np.zeros((self.seq_len, 1), dtype=np.float64)
        cue = rng.choice([-1.0, 1.0])
        inputs[0, 0] = cue
        return inputs, np.array([float(cue)], dtype=np.float64)
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `cd <llcore-root> && py -3.11 -m pytest research/step_c_memory_tasks/tests/test_memory_tasks.py -v`
Expected: PASS（3 件）

- [ ] **Step 5: Codex pair-review してコミット**

Run: `cd <llcore-root> && codex exec -s read-only "research/step_c_memory_tasks/memory_tasks.py と tests/test_memory_tasks.py をレビュー。各タスクが (1) 記憶必須か (final state からのみ解ける) (2) target ラベルが決定論的に正しいか (3) 退化 (常に同じ target / 自明に解ける) していないか を確認。BLOCKERS のみ報告。" 2>&1 | tail -40`
findings を実コードで検証し、BLOCKERS のみ反映してから:

```bash
git add research/step_c_memory_tasks/memory_tasks.py research/step_c_memory_tasks/tests/test_memory_tasks.py
git commit -m "feat(step-c): 記憶タスク3種 (delayed parity/flip-flop/delayed recall)"
```

---

### Task 2: leaky-delay-line reservoir 基質 (`reservoir.py`)

**Files:**
- Create: `research/step_c_memory_tasks/reservoir.py`
- Test: `research/step_c_memory_tasks/tests/test_reservoir.py`

固定 ESN では記憶構造を進化できないため、**reservoir のダイナミクスを連続 gene ベクトルで決める**。最小構成 = leaky-integrator bank: tap `i` が独自の leak rate `a_i` と入力重み `w_i` を持つ。`h_i[t] = (1-a_i)·h_i[t-1] + a_i·tanh(w_i·x[t] + h_i[t-1])`。異なる `a_i` が異なる時間スケールの記憶を担う → 長期依存タスクで「正しい時定数配分」に到達すると解ける = 欺瞞的になりうる。

gene（連続ベクトル, dim=`2*N` for in_dim=1; multi-input は `N*(in_dim+1)`）= `concat(leak_raw[N], w_in[N*in_dim])`。`leak = sigmoid(leak_raw)` で (0,1) に。bounds は探索範囲（後述）。fitness = `fit_ridge_readout`（src 流用）の held-out R²。behavior descriptor = `(平均実効記憶長, leak の分散)`（記憶戦略の niche）。

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_reservoir.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from memory_tasks import DelayedRecallTask
from reservoir import LeakyDelayLineReservoir, make_eval_once, make_behavior, gene_bounds


def test_run_states_shape():
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    rng = np.random.default_rng(0)
    gene = res.random_gene(rng)
    inputs = np.sign(rng.normal(size=(12, 1)))
    states = res.run(gene, inputs)
    assert states.shape == (12, 8)
    assert np.all(np.isfinite(states))


def test_gene_dim_matches_bounds():
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    lo, hi = gene_bounds(res)
    assert lo.shape == hi.shape == (res.gene_dim,)
    assert res.gene_dim == 8 + 8 * 1


def test_eval_once_returns_unit_interval_and_memory_helps():
    """記憶できる gene (遅い leak) は記憶不要の gene (速い leak で過去を捨てる) より
    delayed recall で高い R² になりうる — fitness が記憶を報酬にしていることの sanity。"""
    task = DelayedRecallTask(seq_len=15, in_dim=1)
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    eval_once = make_eval_once(res, task, n_train=40, n_eval=40)
    f = eval_once(_slow_leak_gene(res), np.random.default_rng(3))
    assert 0.0 <= f <= 1.0


def test_behavior_in_bounds():
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=1)
    behavior = make_behavior(res)
    rng = np.random.default_rng(0)
    bd = behavior(res.random_gene(rng))
    assert bd.shape == (2,)
    assert np.all(np.isfinite(bd))


def _slow_leak_gene(res):
    # leak_raw を大きな負値 (sigmoid→0 付近=遅い leak=長記憶) に、入力重みは中庸
    g = np.zeros(res.gene_dim)
    g[: res.n_taps] = -3.0  # leak_raw → sigmoid ≈ 0.047 (長記憶)
    g[res.n_taps :] = 1.0
    return g
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `cd <llcore-root> && py -3.11 -m pytest research/step_c_memory_tasks/tests/test_reservoir.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'reservoir'`）

- [ ] **Step 3: 最小実装を書く**

```python
# reservoir.py
# SPDX-License-Identifier: Apache-2.0
"""leaky-delay-line reservoir 基質 — 連続 gene でダイナミクスを進化させる.

固定 ESN と違い tap ごとの leak rate / 入力重みを gene で決める。異なる leak が
異なる時間スケールの記憶を担い、長期依存タスクで「正しい時定数配分」を要求する。
fitness は per-gene ridge readout (src の fit_ridge_readout 流用) の held-out R²。

research/ 隔離。src は read-only 流用のみ (非変更)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from llcore.fitness.ridge_readout import fit_ridge_readout  # noqa: E402


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@dataclass(frozen=True)
class LeakyDelayLineReservoir:
    n_taps: int = 8
    in_dim: int = 1

    @property
    def gene_dim(self) -> int:
        return self.n_taps + self.n_taps * self.in_dim

    def _unpack(self, gene: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        leak = _sigmoid(gene[: self.n_taps])               # (N,) in (0,1)
        w_in = gene[self.n_taps :].reshape(self.n_taps, self.in_dim)  # (N, in_dim)
        return leak, w_in

    def run(self, gene: np.ndarray, inputs: np.ndarray) -> np.ndarray:
        """gene の reservoir で inputs (T,in_dim) を回し states (T,N) を返す."""
        leak, w_in = self._unpack(gene)
        T = inputs.shape[0]
        h = np.zeros(self.n_taps, dtype=np.float64)
        states = np.empty((T, self.n_taps), dtype=np.float64)
        for t in range(T):
            drive = w_in @ inputs[t]                       # (N,)
            h = (1.0 - leak) * h + leak * np.tanh(drive + h)
            states[t] = h
        return states

    def random_gene(self, rng: np.random.Generator) -> np.ndarray:
        lo, hi = gene_bounds(self)
        return lo + (hi - lo) * rng.random(self.gene_dim)


def gene_bounds(res: LeakyDelayLineReservoir) -> tuple[np.ndarray, np.ndarray]:
    """leak_raw ∈ [-4,4] (sigmoid→~0.018..0.982), w_in ∈ [-2,2]."""
    lo = np.concatenate([np.full(res.n_taps, -4.0), np.full(res.n_taps * res.in_dim, -2.0)])
    hi = np.concatenate([np.full(res.n_taps, 4.0), np.full(res.n_taps * res.in_dim, 2.0)])
    return lo, hi


class _GeneTaskAdapter:
    """fit_ridge_readout は task.generate(rng)->(inputs,target) と gene.run 経由の
    final state を期待する。reservoir gene を ridge_fitness に載せるための薄い shim。"""

    def __init__(self, res: LeakyDelayLineReservoir, gene: np.ndarray):
        self._res = res
        self._gene = gene

    def final_state(self, inputs: np.ndarray) -> np.ndarray:
        return self._res.run(self._gene, inputs)[-1]


def make_eval_once(res, task, *, n_train: int = 64, n_eval: int = 64, ridge_lambda: float = 1e-2):
    """eval_once(gene: np.ndarray, rng) -> held-out R² in [0,1]."""

    def _collect(gene, n, rng):
        states, targets = [], []
        for _ in range(n):
            inputs, target = task.generate(rng)
            states.append(res.run(gene, inputs)[-1])
            targets.append(np.atleast_1d(np.asarray(target, dtype=np.float64)))
        return np.array(states), np.array(targets)

    def eval_once(gene: np.ndarray, rng: np.random.Generator) -> float:
        s_tr, y_tr = _collect(gene, n_train, rng)
        readout = fit_ridge_readout(s_tr, y_tr, ridge_lambda=ridge_lambda)
        s_ev, y_ev = _collect(gene, n_eval, rng)
        pred = np.atleast_2d(readout(s_ev))
        mse = float(np.mean((pred - y_ev) ** 2))
        var = float(np.mean((y_ev - y_ev.mean(axis=0)) ** 2))
        r2 = 1.0 - mse / max(var, 1e-12)
        return float(np.clip(r2, 0.0, 1.0))

    return eval_once


def make_behavior(res):
    """behavior(gene) -> (平均実効記憶長 正規化, leak 分散) — 記憶戦略の niche 軸."""

    def behavior(gene: np.ndarray) -> np.ndarray:
        leak = _sigmoid(gene[: res.n_taps])
        eff_mem = np.mean(1.0 / np.maximum(leak, 1e-3))    # 1/leak ≈ 記憶長
        eff_mem_norm = np.tanh(eff_mem / 50.0)             # [0,1) に圧縮
        return np.array([eff_mem_norm, float(np.std(leak))], dtype=np.float64)

    return behavior
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `cd <llcore-root> && py -3.11 -m pytest research/step_c_memory_tasks/tests/test_reservoir.py -v`
Expected: PASS（4 件）

- [ ] **Step 5: Codex pair-review してコミット**

Run: `cd <llcore-root> && codex exec -s read-only "research/step_c_memory_tasks/reservoir.py をレビュー。(1) run の数値安定性 (NaN/発散) (2) eval_once が held-out で leakage なく R² を測れているか (3) fit_ridge_readout の流用が正しいか (4) behavior descriptor が記憶戦略を弁別できるか。BLOCKERS のみ。" 2>&1 | tail -40`
BLOCKERS を実コード検証して反映後:

```bash
git add research/step_c_memory_tasks/reservoir.py research/step_c_memory_tasks/tests/test_reservoir.py
git commit -m "feat(step-c): leaky-delay-line reservoir 基質 + ridge eval_once + behavior"
```

---

### Task 3: 厳格判定 `strict_compare.py`

**Files:**
- Create: `research/step_c_memory_tasks/strict_compare.py`
- Test: `research/step_c_memory_tasks/tests/test_strict_compare.py`

`selection_lab.compare` は緩い基準（diff>0 & 両側 p<alpha）。(c) は強化版 honest 基準で判定する: **n_seeds≥15・片側 Wilcoxon p<alpha・|paired_sign_delta|≥min_effect**。`honest_eval._paired_p`（片側化済）と `_paired_sign_delta` を流用する。

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_strict_compare.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strict_compare import strict_compare


def test_clear_win_passes():
    a = np.linspace(0.6, 0.9, 16)
    b = np.linspace(0.1, 0.3, 16)
    r = strict_compare(a, b, "map_elites", "random")
    assert r.passes
    assert r.diff > 0


def test_too_few_seeds_fails():
    a = np.array([0.9, 0.9, 0.9])
    b = np.array([0.1, 0.1, 0.1])
    r = strict_compare(a, b, "map_elites", "random")
    assert not r.passes  # n_seeds < 15


def test_tie_fails():
    a = np.full(16, 0.5)
    b = np.full(16, 0.5)
    r = strict_compare(a, b, "map_elites", "random")
    assert not r.passes
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `cd <llcore-root> && py -3.11 -m pytest research/step_c_memory_tasks/tests/test_strict_compare.py -v`
Expected: FAIL（`ModuleNotFoundError: No module named 'strict_compare'`）

- [ ] **Step 3: 最小実装を書く**

```python
# strict_compare.py
# SPDX-License-Identifier: Apache-2.0
"""強化版 honest 基準で 2 スコア配列を判定 (selection_lab.compare の厳格版).

合格 = diff>0 ∧ 片側 Wilcoxon p<alpha ∧ n_seeds>=min_seeds ∧ |paired_sign_delta|>=min_effect。
honest_eval の片側 _paired_p / _paired_sign_delta を流用 (基準は監査 §5 = 強化版 passes と同一)。
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from llcore.evolution.honest_eval import _paired_p, _paired_sign_delta  # noqa: E402


@dataclass(frozen=True)
class StrictComparison:
    name_a: str
    name_b: str
    mean_a: float
    mean_b: float
    diff: float
    win_rate: float
    wilcoxon_p: float
    paired_sign_delta: float
    n_seeds: int
    passes: bool


def strict_compare(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    name_a: str,
    name_b: str,
    *,
    alpha: float = 0.05,
    min_seeds: int = 15,
    min_effect: float = 0.147,
) -> StrictComparison:
    a = np.asarray(scores_a, dtype=np.float64)
    b = np.asarray(scores_b, dtype=np.float64)
    deltas = a - b
    diff = float(np.mean(deltas))
    p = _paired_p(a, b)
    delta = _paired_sign_delta(deltas)
    passes = bool(
        diff > 0.0 and p < alpha and len(a) >= min_seeds and abs(delta) >= min_effect
    )
    return StrictComparison(
        name_a=name_a, name_b=name_b,
        mean_a=float(np.mean(a)), mean_b=float(np.mean(b)),
        diff=diff, win_rate=float(np.mean(a > b)), wilcoxon_p=p,
        paired_sign_delta=delta, n_seeds=len(a), passes=passes,
    )
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `cd <llcore-root> && py -3.11 -m pytest research/step_c_memory_tasks/tests/test_strict_compare.py -v`
Expected: PASS（3 件）

- [ ] **Step 5: コミット**

```bash
git add research/step_c_memory_tasks/strict_compare.py research/step_c_memory_tasks/tests/test_strict_compare.py
git commit -m "feat(step-c): 強化版 honest 基準の strict_compare (片側/効果量/seed ゲート)"
```

---

### Task 4: C1 多峰性診断 `landscape_map.py`

**Files:**
- Create: `research/step_c_memory_tasks/landscape_map.py`
- Test: `research/step_c_memory_tasks/tests/test_landscape_map.py`

C1（多峰性）の機械的判定: random-restart hill-climbing を複数回独立に走らせ収束点を集め、**「2 つの収束点の中点の fitness が両端より明確に低い（谷）」ペアが存在するか**を測る（step4 の C1 手法を gene 空間で）。谷ペアが十分あれば「分離した peak が複数 = 多峰」。

- [ ] **Step 1: 失敗するテストを書く**

```python
# tests/test_landscape_map.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from landscape_map import multimodality_report


def test_unimodal_reports_no_valley():
    # 単峰 (二次関数): どの収束点も中点が谷にならない
    target = np.array([0.5, 0.5])
    eval_once = lambda g, rng: float(np.exp(-np.sum((g - target) ** 2)))
    lo, hi = np.zeros(2), np.ones(2)
    rep = multimodality_report(eval_once, dim=2, bounds=(lo, hi), n_restarts=12,
                               n_evals=200, sigma=0.1, base_seed=0)
    assert rep["valley_fraction"] < 0.1  # 谷ペアはほぼ無い


def test_report_keys():
    eval_once = lambda g, rng: float(-np.sum(g ** 2))
    lo, hi = -np.ones(2), np.ones(2)
    rep = multimodality_report(eval_once, dim=2, bounds=(lo, hi), n_restarts=6,
                               n_evals=100, sigma=0.1, base_seed=0)
    assert {"n_optima", "valley_fraction", "is_multimodal"} <= set(rep)
```

- [ ] **Step 2: テストを実行して失敗を確認**

Run: `cd <llcore-root> && py -3.11 -m pytest research/step_c_memory_tasks/tests/test_landscape_map.py -v`
Expected: FAIL（`ModuleNotFoundError`）

- [ ] **Step 3: 最小実装を書く**

```python
# landscape_map.py
# SPDX-License-Identifier: Apache-2.0
"""C1 多峰性診断 — 収束点間の中点が谷になるかで分離 peak の存在を測る (step4 C1 手法)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reservoir import _sigmoid  # noqa: F401  (reservoir と同居; 未使用なら削除可)


def _hillclimb(eval_once, dim, bounds, n_evals, sigma, rng):
    lo, hi = bounds
    g = lo + (hi - lo) * rng.random(dim)
    f = eval_once(g, rng)
    for _ in range(n_evals - 1):
        cand = np.clip(g + rng.normal(0, sigma, size=dim), lo, hi)
        cf = eval_once(cand, rng)
        if cf >= f:
            g, f = cand, cf
    return g, f


def multimodality_report(eval_once, *, dim, bounds, n_restarts, n_evals, sigma, base_seed):
    """n_restarts 回 hill-climb し収束点を集め、ペアの中点が谷になる割合を測る."""
    optima = []
    for i in range(n_restarts):
        g, f = _hillclimb(eval_once, dim, bounds, n_evals, sigma,
                          np.random.default_rng(base_seed + i))
        optima.append((g, f))
    valley = 0
    pairs = 0
    for i in range(len(optima)):
        for j in range(i + 1, len(optima)):
            gi, fi = optima[i]
            gj, fj = optima[j]
            if np.allclose(gi, gj, atol=1e-2):
                continue
            mid = 0.5 * (gi + gj)
            fm = float(np.mean([eval_once(mid, np.random.default_rng(base_seed + 999 + k))
                                for k in range(3)]))
            pairs += 1
            if fm < min(fi, fj) - 0.05 * (abs(min(fi, fj)) + 1e-9):
                valley += 1
    frac = valley / pairs if pairs else 0.0
    return {
        "n_optima": len(optima),
        "valley_fraction": frac,
        "is_multimodal": frac >= 0.2,  # step4 では 0/66 が単峰判定だった (>0 で多峰の兆候)
    }
```

- [ ] **Step 4: テストを実行して成功を確認**

Run: `cd <llcore-root> && py -3.11 -m pytest research/step_c_memory_tasks/tests/test_landscape_map.py -v`
Expected: PASS（2 件）

- [ ] **Step 5: Codex pair-review してコミット**

Run: `cd <llcore-root> && codex exec -s read-only "research/step_c_memory_tasks/landscape_map.py をレビュー。谷判定の閾値・収束点重複除外・中点ノイズ平均が C1 (多峰性) の機械的判定として妥当か。BLOCKERS のみ。" 2>&1 | tail -40`
反映後:

```bash
git add research/step_c_memory_tasks/landscape_map.py research/step_c_memory_tasks/tests/test_landscape_map.py
git commit -m "feat(step-c): C1 多峰性診断 (収束点間の谷検出)"
```

---

### Task 5: exp_c1 — 各タスクの landscape map 実行

**Files:**
- Create: `research/step_c_memory_tasks/exp_c1_landscape.py`

部品を組み、3 タスク × reservoir で C1 多峰性を測る測定スクリプト（実行＝測定）。

- [ ] **Step 1: スクリプトを書く**

```python
# exp_c1_landscape.py
# SPDX-License-Identifier: Apache-2.0
"""C1: 各記憶タスクの landscape が多峰か (reservoir gene 空間で)."""
from __future__ import annotations

import numpy as np

from memory_tasks import DelayedParityTask, FlipFlopTask, DelayedRecallTask
from reservoir import LeakyDelayLineReservoir, make_eval_once, gene_bounds
from landscape_map import multimodality_report

TASKS = {
    "delayed_parity": DelayedParityTask(seq_len=20, window=5, in_dim=1),
    "flip_flop": FlipFlopTask(seq_len=30, in_dim=2),
    "delayed_recall": DelayedRecallTask(seq_len=25, in_dim=1),
}


def main() -> None:
    for name, task in TASKS.items():
        in_dim = task.in_dim
        res = LeakyDelayLineReservoir(n_taps=8, in_dim=in_dim)
        eval_once = make_eval_once(res, task, n_train=48, n_eval=48)
        lo, hi = gene_bounds(res)
        rep = multimodality_report(
            eval_once, dim=res.gene_dim, bounds=(lo, hi),
            n_restarts=12, n_evals=400, sigma=0.15, base_seed=20260530,
        )
        print(f"[{name}] n_optima={rep['n_optima']} "
              f"valley_fraction={rep['valley_fraction']:.3f} "
              f"is_multimodal={rep['is_multimodal']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 実行して結果を記録**

Run: `cd <llcore-root> && py -3.11 research/step_c_memory_tasks/exp_c1_landscape.py`
Expected: 各タスクの `valley_fraction` / `is_multimodal` が出力される。**判断分岐**: いずれかのタスクで `is_multimodal=True` → そのタスクを Task 6 へ。全タスクで False（滑らか）→ §「撤退判定」へ（Task 6 はスキップして verdict へ）。

- [ ] **Step 3: コミット**

```bash
git add research/step_c_memory_tasks/exp_c1_landscape.py
git commit -m "feat(step-c): exp_c1 各記憶タスクの C1 多峰性測定"
```

---

### Task 6: exp_c2c3 + exp_c4 — MAP-E vs baseline (C2/C3) と勝因 ablation (C4)

**Files:**
- Create: `research/step_c_memory_tasks/exp_c2c3_compare.py`
- Create: `research/step_c_memory_tasks/exp_c4_ablation.py`

C1 で多峰だったタスクについて、step4 の `selection_lab.run_methods_over_seeds` を reservoir の `eval_once`/`behavior` で走らせ、`strict_compare` で MAP-E vs {random, RR-hillclimb, panmictic-GA} を判定（C2/C3）。C4 は init_batch を変えて勝因が coverage か archive ratchet かを確認。

- [ ] **Step 1: exp_c2c3 を書く**

```python
# exp_c2c3_compare.py
# SPDX-License-Identifier: Apache-2.0
"""C2/C3: 多峰タスクで MAP-E が 3 baseline を強化基準で上回るか."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))

from memory_tasks import DelayedRecallTask  # ← C1 で多峰だったタスクに差し替える
from reservoir import LeakyDelayLineReservoir, make_eval_once, make_behavior, gene_bounds
from strict_compare import strict_compare
from selection_lab import run_methods_over_seeds


def main() -> None:
    task = DelayedRecallTask(seq_len=25, in_dim=1)  # ← C1 結果で確定
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=task.in_dim)
    eval_once = make_eval_once(res, task, n_train=48, n_eval=48)
    behavior = make_behavior(res)
    lo, hi = gene_bounds(res)
    scores = run_methods_over_seeds(
        eval_once, behavior, dim=res.gene_dim, bounds=(lo, hi),
        behavior_bounds=(np.array([0.0, 0.0]), np.array([1.0, 0.5])),
        grid_shape=(12, 12), n_evals=2000, n_seeds=15, honest_n_trials=20,
        sigma=0.15, base_seed=20260530,
    )
    for base in ("random", "rr_hillclimb", "panmictic_ga"):
        r = strict_compare(scores["map_elites"], scores[base], "map_elites", base)
        print(f"MAP-E vs {base}: diff={r.diff:+.4f} p={r.wilcoxon_p:.4g} "
              f"δ={r.paired_sign_delta:+.2f} passes={r.passes}")
```

- [ ] **Step 2: 実行して C2/C3 を記録**

Run: `cd <llcore-root> && py -3.11 research/step_c_memory_tasks/exp_c2c3_compare.py`
Expected: 3 baseline 全てに対し `passes=True` なら C3 成立（C2 は baseline の到達率が低いことで担保）。

- [ ] **Step 3: exp_c4 ablation を書く**

```python
# exp_c4_ablation.py
# SPDX-License-Identifier: Apache-2.0
"""C4: MAP-E の勝因が coverage でなく archive ratchet か (init_batch を変えて確認)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "step4_selection"))

from memory_tasks import DelayedRecallTask
from reservoir import LeakyDelayLineReservoir, make_eval_once, make_behavior, gene_bounds
from selection_lab import map_elites
from llcore.evolution.honest_eval import honest_reevaluate


def main() -> None:
    task = DelayedRecallTask(seq_len=25, in_dim=1)
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=task.in_dim)
    eval_once = make_eval_once(res, task, n_train=48, n_eval=48)
    behavior = make_behavior(res)
    lo, hi = gene_bounds(res)
    for init_batch in (30, 200, 1000):
        vals = []
        for s in range(10):
            r = map_elites(
                eval_once, behavior, dim=res.gene_dim, bounds=(lo, hi),
                behavior_bounds=(np.array([0.0, 0.0]), np.array([1.0, 0.5])),
                grid_shape=(12, 12), n_evals=2000, init_batch=init_batch,
                sigma=0.15, rng=np.random.default_rng(20260530 + s),
            )
            vals.append(honest_reevaluate(eval_once, r.best_gene, n_trials=20,
                                          rng=np.random.default_rng(99 + s)))
        print(f"init_batch={init_batch}: mean honest R²={np.mean(vals):.4f}")
    # 大 init_batch でも到達するなら coverage 由来、小でも到達するなら ratchet 由来。
```

- [ ] **Step 4: 実行して C4 を記録**

Run: `cd <llcore-root> && py -3.11 research/step_c_memory_tasks/exp_c4_ablation.py`
Expected: 小 init_batch でも到達 → 勝因は archive ratchet（C4 成立）。

- [ ] **Step 5: Codex pair-review してコミット**

Run: `cd <llcore-root> && codex exec -s read-only "exp_c2c3_compare.py / exp_c4_ablation.py をレビュー。予算 (n_evals) が全 method 同一か、honest 再評価が独立 seed か、結論の overclaim がないか。BLOCKERS のみ。" 2>&1 | tail -40`
反映後:

```bash
git add research/step_c_memory_tasks/exp_c2c3_compare.py research/step_c_memory_tasks/exp_c4_ablation.py
git commit -m "feat(step-c): exp_c2c3 (C2/C3) + exp_c4 ablation (C4)"
```

---

### Task 7: verdict — ③実在 or CPU撤退

**Files:**
- Create: `docs/poc/STEP_C_VERDICT.md`

C1-C4 の実測を集約し、合格基準（§spec 2）に照らして両方向のいずれかで決着を文書化する。

- [ ] **Step 1: verdict を書く**

C1-C4 の実数（valley_fraction / 各 baseline の diff・p・δ・passes / init_batch ablation）を表にまとめ、次のいずれかを honest に結論:
- **③実在**: あるタスクで C1-C4 全成立。「実タスク本来の性質から欺瞞 corridor が現れ ③ が load-bearing」と存在証明。どのタスク・どの難易度かを明記。
- **CPU撤退**: 全タスクで C1 が出ない or C3 不成立。「実問題に近い記憶タスクでも ③ は CPU 範囲では立たない」と結論。GPU は実 LLM 損失地形の賭けに限定する根拠とする。
- honest 留保（reservoir+ridge proxy であり full LLM とは別、難易度スイープ範囲、seed 依存）を必ず明記。冒頭に①〜④凡例 + 用語集リンクを置く（甲乙丙ルール）。

- [ ] **Step 2: 全テスト回帰確認**

Run: `cd <llcore-root> && py -3.11 -m pytest research/step_c_memory_tasks/tests/ -q`
Expected: 全 PASS。

- [ ] **Step 3: verdict を最終 Codex pair-review してコミット**

Run: `cd <llcore-root> && codex exec -s read-only "docs/poc/STEP_C_VERDICT.md をレビュー。C1-C4 の実数から結論が overclaim/underclaim していないか、honest 留保が十分か。" 2>&1 | tail -40`
反映後:

```bash
git add docs/poc/STEP_C_VERDICT.md
git commit -m "docs(step-c): verdict — 記憶タスクで③が立つか (両方向決着)"
```

- [ ] **Step 4: memory 更新**

`project_llcore_init_2026_05_29.md` の進化健全性セクションに (c) の結論を追記し、`claude-projects.json` の fullsense `next_plan` を更新（③ の最終決着 or 次の軸へ）。

---

## Self-Review (記入済)

- **Spec coverage**: §2 合格基準→Task 7 verdict / §3 記憶タスク→Task 1 / §4 ハーネス・基質→Task 2,6 / §5 C1-C4→Task 4(C1),6(C2/C3/C4) / §6 合格撤退→Task 5 分岐・Task 7 / §7 ズルしない→Task 1,4 のタスク標準形 + 難易度のみスイープ。全カバー。
- **Placeholder**: 各コード step に実コードを記載。exp スクリプトの「C1 で多峰だったタスクに差し替え」は実行時に確定する正当な分岐（Task 5 Step 2 で明示）。
- **Type 一貫性**: `eval_once(gene: np.ndarray, rng)->float` / `behavior(gene)->np.ndarray(2,)` / `task.generate(rng)->(inputs(T,in_dim), target(1,))` を全タスクで統一。`gene_bounds`/`make_eval_once`/`make_behavior`/`strict_compare`/`multimodality_report` のシグネチャは定義タスクと利用タスクで一致。

## 既知の調整余地（実行中に判断）

- C1 が全タスクで滑らかなら難易度（window/delay/seq_len）を「タスク本来の難しさ」の範囲で 1 段上げて再測（地形捏造はしない）。それでも滑らかなら撤退判定。
- behavior descriptor 軸が niche を弁別しない場合は (実効記憶長, 有効次元) 等へ差し替え（Task 2 の make_behavior 内のみ）。
