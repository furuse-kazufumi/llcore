# -*- coding: utf-8 -*-
import sys, json
from pathlib import Path
for s in (sys.stdout, sys.stderr):
    rc = getattr(s, "reconfigure", None)
    if rc:
        try: rc(encoding="utf-8")
        except Exception: pass
HERE = Path(r"D:\projects\llcore\research\step_c_deceptiveness_measure")
d = json.load(open(HERE / "synth_calibration_results.json", encoding="utf-8"))
print("TOPKEYS:", list(d.keys()))

def find_curve(obj, path=""):
    res = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "curve" and isinstance(v, list):
                res.append((path + "/" + k, v))
            else:
                res += find_curve(v, path + "/" + k)
    return res

for p, c in find_curve(d):
    print(f"CURVE at {p}: {len(c)} levels")
    for row in c:
        if isinstance(row, dict):
            ps = row.get("per_seed")
            print("  d=%-5s metric=%-8s std=%-8s per_seed=%s" % (
                row.get("d"), round(row.get("metric", 0), 5),
                round(row.get("std", 0), 5), [round(x,4) for x in ps] if ps else ps))

def find_key(obj, k):
    if isinstance(obj, dict):
        if k in obj and not isinstance(obj[k], (dict, list)):
            return obj[k]
        for vv in obj.values():
            r = find_key(vv, k)
            if r is not None:
                return r
    return None

for key in ("metric_at_dstar", "d_star", "spearman", "strictly_monotone", "monotone", "reproduces_threshold"):
    print(key, "=", find_key(d, key))
