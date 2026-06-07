# SPDX-License-Identifier: Apache-2.0
"""合成 knob のみの高速校正 (real task を除く)。stdout を都度 flush して進捗を可視化.

calibrate_dip_metric.calibrate_synthetic() を呼び、合成校正結果のみを
synth_calibration_results.json に書く。real task は別ランで測る。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from calibrate_dip_metric import calibrate_synthetic  # noqa: E402

if __name__ == "__main__":
    syn = calibrate_synthetic()
    out = {"metric_name": "behavior_elite_fitness_dip", "synthetic_calibration": syn}
    path = _HERE / "synth_calibration_results.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}", flush=True)
