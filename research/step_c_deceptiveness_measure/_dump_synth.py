# -*- coding: utf-8 -*-
import json
from pathlib import Path
HERE = Path(r"D:\projects\llcore\research\step_c_deceptiveness_measure")
d = json.load(open(HERE / "synth_calibration_results.json", encoding="utf-8"))
out = []
out.append("TOPKEYS: " + ", ".join(d.keys()))
# find a 'curve' anywhere
def find_curve(obj, path=""):
    res = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "curve" and isinstance(v, list):
                res.append((path + "/" + k, v))
            else:
                res += find_curve(v, path + "/" + k)
    return res
curves = find_curve(d)
for p, c in curves:
    out.append(f"CURVE at {p}: {len(c)} levels")
    for row in c:
        if isinstance(row, dict):
            out.append("  d=%s metric=%s std=%s per_seed=%s" % (
                row.get("d"), round(row.get("metric", 0), 5),
                round(row.get("std", 0), 5), row.get("per_seed")))
# capture metric_at_dstar / d_star / spearman wherever
for key in ("metric_at_dstar", "d_star", "spearman", "strictly_monotone", "monotone"):
    def find_key(obj, k):
        if isinstance(obj, dict):
            if k in obj and not isinstance(obj[k], (dict, list)):
                return obj[k]
            for vv in obj.values():
                r = find_key(vv, k)
                if r is not None:
                    return r
        return None
    out.append(f"{key} = {find_key(d, key)}")
(HERE / "_dump_synth_out.txt").write_text("\n".join(str(x) for x in out), encoding="utf-8")
print("done")
