"""HD-1 図2: 越境は『学習』でなく『幾何』 — null 対照. 実データ(result_hd1_full*.json)由来.
ゲートなし GRAD の ρ vs 次元 n を、本物コーパス(real) と シャッフル(null) で比較.
null の方が強く越境=学ぶ構造が無いのに drift がより大きい → 越境はエントロピー(幾何)の成り行き.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("HD1_NULL_SVG_OUT", os.path.join(HERE, "hd1_null_vs_real.svg"))
NS = [8, 32, 64, 128, 256]


def grad_none_rho(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    by = {}
    for r in d["records"]:
        if r.get("status") != "ok" or r["gate"] != "none":
            continue
        by.setdefault(r["n"], []).append(r["metrics"]["grad_max_emp_rho"])
    return {n: sum(v) / len(v) for n, v in by.items()}


def require_dims(name, data):
    missing = [n for n in NS if n not in data]
    if missing:
        raise ValueError(f"{name} is missing required dimensions: {missing}")


real = grad_none_rho(os.path.join(HERE, "result_hd1_full.json"))
null = grad_none_rho(os.path.join(HERE, "result_hd1_full_null.json"))
require_dims("real", real)
require_dims("null", null)

W, H = 1000, 640
x0, x1 = 120, 720
y0, y1 = 100, 510
RHO_LO, RHO_HI = 0.7, 2.7


def X(i):
    return x0 + i * (x1 - x0) / (len(NS) - 1)


def Y(rho):
    rho = max(RHO_LO, min(RHO_HI, rho))
    return y1 - (rho - RHO_LO) / (RHO_HI - RHO_LO) * (y1 - y0)


yb = Y(1.0)
p = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Hiragino Sans, Meiryo, sans-serif">']
p.append(f'<rect width="{W}" height="{H}" fill="#fff"/>')
# zones
p.append(f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{yb-y0:.1f}" fill="#fdecea"/>')
p.append(f'<rect x="{x0}" y="{yb:.1f}" width="{x1-x0}" height="{y1-yb:.1f}" fill="#e8f5e9"/>')
# gridlines
for rho in [0.8, 1.0, 1.4, 1.8, 2.2, 2.6]:
    yy = Y(rho)
    p.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x1}" y2="{yy:.1f}" stroke="#dddddd" stroke-width="1"/>')
    p.append(f'<text x="{x0-12}" y="{yy+5:.1f}" font-size="15" fill="#555" text-anchor="end">{rho:.1f}</text>')
# boundary
p.append(f'<line x1="{x0}" y1="{yb:.1f}" x2="{x1}" y2="{yb:.1f}" stroke="#b00020" stroke-width="3"/>')
p.append(f'<text x="{x1-8}" y="{yb-10:.1f}" font-size="16" fill="#b00020" text-anchor="end" font-weight="bold">安定の境界 ρ = 1</text>')
p.append(f'<text x="{x0+12}" y="{y0+24}" font-size="15" fill="#c0392b" font-weight="bold">↑ 越境（不安定）ほど上</text>')
# x ticks + axes
for i, n in enumerate(NS):
    xx = X(i)
    p.append(f'<line x1="{xx:.1f}" y1="{y1}" x2="{xx:.1f}" y2="{y1+6}" stroke="#888" stroke-width="1.5"/>')
    p.append(f'<text x="{xx:.1f}" y="{y1+28}" font-size="17" fill="#333" text-anchor="middle">{n}</text>')
p.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y1}" stroke="#333" stroke-width="2"/>')
p.append(f'<line x1="{x0}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="#333" stroke-width="2"/>')


def series(data, color, dash, label_end):
    pts = [(X(i), Y(data[n])) for i, n in enumerate(NS)]
    da = "" if dash is None else f' stroke-dasharray="{dash}"'
    p.append(f'<polyline points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}" fill="none" stroke="{color}" stroke-width="3.5"{da} stroke-linejoin="round"/>')
    for x, y in pts:
        p.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.5" fill="{color}" stroke="#fff" stroke-width="1.5"/>')
    ex, ey = pts[-1]
    p.append(f'<text x="{ex+10:.1f}" y="{ey+5:.1f}" font-size="15" fill="{color}" font-weight="bold">{label_end}</text>')


series(null, "#d62728", "7 5", f"null ρ≈{null[256]:.2f}")   # null worse
series(real, "#2c7fb8", None, f"本物 ρ≈{real[256]:.2f}")     # real less bad

# callout box (lower area, empty)
bx, by, bw, bh = x0 + 18, y1 - 150, 360, 132
p.append(f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" rx="8" fill="#ffffff" fill-opacity="0.92" stroke="#cccccc"/>')
lines = [
    ("**", "null（青でなく赤破線）= 本物をシャッフルし"),
    ("", "「学ぶ構造」をゼロにした対照データ。"),
    ("", f"なのに越境は本物より強い（ρ {null[256]:.2f} > {real[256]:.2f}）、"),
    ("", "しかも CE 改善はゼロ。"),
    ("hi", "→ 越境は『賢くなるため』でなく、ただの幾何。"),
]
ty = by + 26
for kind, t in lines:
    col = "#b00020" if kind == "hi" else "#333"
    fw = ' font-weight="bold"' if kind == "hi" else ""
    p.append(f'<text x="{bx+14}" y="{ty}" font-size="14.5" fill="{col}"{fw}>{t}</text>')
    ty += 22

# legend
lx, ly = x0 + 18, y0 + 8
p.append(f'<line x1="{lx}" y1="{ly+44}" x2="{lx+34}" y2="{ly+44}" stroke="#2c7fb8" stroke-width="3.5"/>')
p.append(f'<text x="{lx+42}" y="{ly+49}" font-size="15" fill="#333">本物コーパス (real)</text>')
p.append(f'<line x1="{lx}" y1="{ly+68}" x2="{lx+34}" y2="{ly+68}" stroke="#d62728" stroke-width="3.5" stroke-dasharray="7 5"/>')
p.append(f'<text x="{lx+42}" y="{ly+73}" font-size="15" fill="#333">シャッフル (null=学ぶ構造ゼロ)</text>')

# titles
p.append(f'<text x="{W/2:.0f}" y="40" font-size="25" fill="#1a1a1a" text-anchor="middle" font-weight="bold">越境は「学習」でなく「幾何」だった — null 対照</text>')
p.append(f'<text x="{W/2:.0f}" y="66" font-size="14.5" fill="#666" text-anchor="middle">ゲートなし勾配学習の ρ vs 次元 n ／ 本物 vs シャッフル（HD-1, 4 seed 平均）</text>')
p.append(f'<text x="{(x0+x1)/2:.0f}" y="{y1+56}" font-size="17" fill="#333" text-anchor="middle">記憶コアの次元 n →</text>')
p.append(f'<text x="34" y="{(y0+y1)/2:.0f}" font-size="16" fill="#333" text-anchor="middle" transform="rotate(-90 34 {(y0+y1)/2:.0f})">ρ（縮約係数）— 1 未満で安定</text>')
p.append('</svg>')


def main():
    svg = "\n".join(p)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(svg)
    print("wrote:", OUT, f"({os.path.getsize(OUT)} bytes)")
    print("  real GRAD none rho:", [round(real[n], 3) for n in NS])
    print("  null GRAD none rho:", [round(null[n], 3) for n in NS])


if __name__ == "__main__":
    main()
