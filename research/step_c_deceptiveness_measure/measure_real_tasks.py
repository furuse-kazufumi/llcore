# SPDX-License-Identifier: Apache-2.0
"""Place real tasks on the DOWNHILL-NECESSITY axis (with the synthetic-calibration caveats).

実 task: (a) E-A variable_delay_recall multitask, (b) flip_flop。
behavior bounds は両 task とも make_behavior の出力 [0,1]^2 (eff_mem_norm, std(leak))。

honest disclosure:
- このメトリックは calibration で **閾値帯 (d 0.13->0.20) を正しく検知するが d>0.2 で
  非単調** (uniform-sample occupancy が corridor の見かけ global cell を崩す) と判明。
  よって実 task の数値は「閾値帯近傍かどうか」の粗い判定にしか使えない (caveat)。
- 実 task は 2D behavior で占有が疎になりがち → reach が人工的に下がりうる。SE を必ず報告。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "step_c_memory_tasks"))
sys.path.insert(0, str(_HERE.parent / "ea_multitask"))
sys.path.insert(0, str(_HERE.parent / "ea_multitask" / "candidates"))

from metric_downhill_necessity import deceptiveness_with_ci  # noqa: E402
from reservoir import (  # noqa: E402
    LeakyDelayLineReservoir, gene_bounds, make_behavior, make_eval_once,
)
from memory_tasks import FlipFlopTask  # noqa: E402
from task_mixture import TaskMixture, split_regimes  # noqa: E402
from variable_delay_recall import VariableDelayRecallTask  # noqa: E402


def _run(name, res, eval_once, rng, **cfg):
    behavior = make_behavior(res)
    bounds = gene_bounds(res)
    mean, se, samples = deceptiveness_with_ci(
        eval_once, behavior, bounds, res.gene_dim, rng,
        behavior_bounds=(np.zeros(2), np.ones(2)), **cfg)
    print(f"{name}: deceptiveness={mean:.4f} (SE {se:.4f}) reach={1-mean:.4f} "
          f"samples={[round(s,3) for s in samples]}")
    return {"task": name, "metric": mean, "se": se, "samples": [round(s, 4) for s in samples]}


def main() -> int:
    # 実 fitness は確率的 (ridge held-out R²) → fitness_trials=4 で decision noise 抑制。
    # n_bins=24 は calibration と揃える (公正比較)。sample は実 task が重いので 1200。
    cfg = dict(n_samples=1200, n_bins=24, fitness_trials=4, n_repeats=5)
    rng = np.random.default_rng(20260531)
    out = []

    # (a) E-A variable_delay_recall (exp_ea3_ablation.py と同 config)
    res = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    all_D = (15, 30, 45, 60)
    regimes = [VariableDelayRecallTask(seq_len=D, distractor_amp=0.2, in_dim=2) for D in all_D]
    train_regimes, _ = split_regimes(regimes, test_idx=[2, 3])
    ev = make_eval_once(res, TaskMixture(train_regimes), n_train=48, n_eval=48)
    out.append(_run("ea_multitask_train", res, ev, rng, **cfg))

    # (b) flip_flop (in_dim=2 reservoir)
    res2 = LeakyDelayLineReservoir(n_taps=8, in_dim=2)
    ev2 = make_eval_once(res2, FlipFlopTask(), n_train=48, n_eval=48)
    out.append(_run("flip_flop", res2, ev2, rng, **cfg))

    payload = {"metric_name": "downhill_necessity", "config": {k: v for k, v in cfg.items()},
               "note": "calibration showed metric is non-monotone for d>0.2; treat real-task "
                       "numbers as coarse (near-or-below threshold band) only.",
               "tasks": out}
    (_HERE / "real_task_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {_HERE / 'real_task_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
