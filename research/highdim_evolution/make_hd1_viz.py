"""HD-1 可視化: ρ（縮約係数）vs 次元 n を一目で。実データ(result_hd1_full*.json)由来・再現可能。
見やすさ最優先: 安全帯/暴走帯の塗り分け + ρ=1 境界 + 4 系列 + 端点注釈。静的完成形(Qiita ラスタライズ対応)。"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HD1_SVG_OUT", os.path.join(HERE, "hd1_rho_vs_n.svg"))

def agg(path):
    d = json.load(open(path, encoding="utf-8"))
    by = {}
    for r in d["records"]:
        if r.get("status") != "ok":
            continue
        k = (r["n"], r["gate"]); m = r["metrics"]
        b = by.setdefault(k, {"grad_rho": [], "evo_rho": []})
        b["grad_rho"].append(m["grad_max_emp_rho"]); b["evo_rho"].append(m["evo_max_emp_rho"])
    mean = lambda a: sum(a) / len(a)
    return {k: {kk: mean(vv) for kk, vv in v.items()} for k, v in by.items()}

real = agg(os.path.join(HERE, "result_hd1_full.json"))
NS = [8, 32, 64, 128, 256]

# series: label, key(gate, metric), color, dash
series = [
    ("勾配・ゲートなし",      ("none", "grad_rho"), "#d62728", None),
    ("進化・ゲートなし",      ("none", "evo_rho"),  "#ff7f0e", None),
    ("勾配・ゲートあり(inf)", ("inf",  "grad_rho"), "#1f77b4", "6 4"),
    ("進化・ゲートあり(inf)", ("inf",  "evo_rho"),  "#2ca02c", "6 4"),
]

# layout
W, H = 1000, 640
x0, x1 = 120, 760          # plot left/right (right gutter for endpoint labels)
y0, y1 = 100, 520          # plot top/bottom
RHO_LO, RHO_HI = 0.7, 2.05

def X(i):  # categorical equal spacing
    return x0 + i * (x1 - x0) / (len(NS) - 1)

def Y(rho):
    return y1 - (rho - RHO_LO) / (RHO_HI - RHO_LO) * (y1 - y0)

yb = Y(1.0)  # boundary

p = []
p.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Hiragino Sans, Meiryo, sans-serif">')
p.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
# zones
p.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{yb-y0:.1f}" fill="#fdecea"/>')   # danger
p.append(f'<rect x="{x0}" y="{yb:.1f}" width="{x1-x0}" height="{y1-yb:.1f}" fill="#e8f5e9"/>')  # safe
# y gridlines + labels
for rho in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
    yy = Y(rho)
    p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#dddddd" stroke-width="1"/>')
    p.append(f'<text x="{x0-12}" y="{yy+5:.1f}" font-size="15" fill="#555" text-anchor="end">{rho:.1f}</text>')
# boundary rho=1 (bold)
p.append(f'<line x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" stroke="#b00020" stroke-width="3" stroke-dasharray="2 0"/>')
p.append(f'<text x="{x1-8}" y="{yb-10:.1f}" font-size="16" fill="#b00020" text-anchor="end" font-weight="bold">安定の境界 ρ = 1</text>')
# zone labels (placed in verified-empty areas: 暴走帯 top-right, 安全帯 bottom-left)
p.append(f'<text x="{x1-12}" y="{y0+24}" font-size="16" fill="#c0392b" text-anchor="end" font-weight="bold">暴走帯（ρ ≥ 1：エコーが発散）</text>')
p.append(f'<text x="{x0+12}" y="{y1-12}" font-size="16" fill="#278a3d" font-weight="bold">安全帯（ρ &lt; 1：記憶が減衰＝安定）</text>')
# x ticks
for i, n in enumerate(NS):
    xx = X(i)
    p.append(f'<line x1="{xx:.1f}" y1="{y1}" x2="{xx:.1f}" y2="{y1+6}" stroke="#888" stroke-width="1.5"/>')
    p.append(f'<text x="{xx:.1f}" y="{y1+28}" font-size="17" fill="#333" text-anchor="middle">{n}</text>')
# axes
p.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="2"/>')
p.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#333" stroke-width="2"/>')
# series polylines + markers
for label, (gate, metric), color, dash in series:
    pts = [(X(i), Y(real[(n, gate)][metric])) for i, n in enumerate(NS)]
    d = "" if dash is None else f' stroke-dasharray="{dash}"'
    path = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
    p.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="3.5"{d} stroke-linejoin="round"/>')
    for x, y in pts:
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
# endpoint annotations (n=256)
end = {
    ("none", "grad_rho"): ("勾配・なし → ρ≈1.95", "#d62728", -2),
    ("none", "evo_rho"):  ("進化・なし → ρ≈1.89", "#ff7f0e", 18),
    ("inf",  "grad_rho"): ("ゲートあり → ρ≈0.91", "#1f77b4", -2),
}
for (gate, metric), (txt, color, dy) in end.items():
    yy = Y(real[(256, gate)][metric])
    p.append(f'<line x1="{x1:.1f}" y1="{yy:.1f}" x2="{x1+14}" y2="{yy+dy:.1f}" stroke="{color}" stroke-width="1.2"/>')
    p.append(f'<text x="{x1+18}" y="{yy+dy+5:.1f}" font-size="15" fill="{color}" font-weight="bold">{txt}</text>')
# title / axis labels
p.append(f'<text x="{W/2:.0f}" y="40" font-size="26" fill="#1a1a1a" text-anchor="middle" font-weight="bold">高次元で「安定」から外れていく — HD-1 実測</text>')
p.append(f'<text x="{W/2:.0f}" y="66" font-size="15" fill="#666" text-anchor="middle">記憶コアの ρ（縮約係数）vs 次元 n ／ シェイクスピア・4 seed 平均（result_hd1_full.json）</text>')
p.append(f'<text x="{(x0+x1)/2:.0f}" y="{y1+56}" font-size="17" fill="#333" text-anchor="middle">記憶コアの次元 n （大きいほど高次元）→</text>')
p.append(f'<text x="34" y="{(y0+y1)/2:.0f}" font-size="17" fill="#333" text-anchor="middle" transform="rotate(-90 34 {(y0+y1)/2:.0f})">ρ（縮約係数）— 1 未満で安定</text>')
# legend (horizontal strip)
lx, ly = x0, H - 26
for label, _, color, dash in series:
    da = "" if dash is None else f' stroke-dasharray="{dash}"'
    p.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+34}" y2="{ly}" stroke="{color}" stroke-width="3.5"{da}/>')
    p.append(f'<circle cx="{lx+17}" cy="{ly}" r="5" fill="{color}" stroke="#fff" stroke-width="1.2"/>')
    p.append(f'<text x="{lx+42}" y="{ly+5}" font-size="15" fill="#333">{label}</text>')
    lx += 42 + len(label) * 16 + 24
p.append('</svg>')

open(OUT, "w", encoding="utf-8").write("\n".join(p))
print("wrote:", OUT, f"({os.path.getsize(OUT)} bytes)")
# echo the data used (honest: real numbers)
for label, (gate, metric), _, _ in series:
    vals = [round(real[(n, gate)][metric], 3) for n in NS]
    print(f"  {label:22s}", vals)
