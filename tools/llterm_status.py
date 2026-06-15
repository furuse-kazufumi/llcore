"""llterm x llcore 自走ステータス — 一目ダッシュボード (再実行可能).

動的に再計算: git pulse (直近コミット) / loop 稼働 (loop_ledger の最終ターン vs 現在 = idle 判定).
seed から読む (緩変化の解釈部): roadmap / human_gates / open_issues / verdict.
  seed = tools/llterm_status_seed.json  (status-model workflow が生成)

出力: docs/status/llterm_status.svg  (見やすさ第一・静的完成形)
使い方: py -3.11 tools/llterm_status.py
"""
from collections import deque
import json
import os
import subprocess
import unicodedata
from datetime import datetime, timezone, timedelta

LC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # llcore root
SEED = os.path.join(LC, "tools", "llterm_status_seed.json")
LEDGER = os.path.join(LC, ".llterm", "loop_ledger.jsonl")
OUT = os.environ.get("LLTERM_STATUS_OUT", os.path.join(LC, "docs", "status", "llterm_status.svg"))
JST = timezone(timedelta(hours=9))


def git_pulse(n=7):
    try:
        out = subprocess.run(
            ["git", "-C", LC, "log", "-40", "--pretty=format:%ad|%s", "--date=format:%m-%d %H:%M"],
            capture_output=True, text=True, encoding="utf-8", timeout=20).stdout
    except Exception as e:
        return [{"time": "?", "summary": f"(git log 失敗: {e})"}]
    rows = []
    for line in out.splitlines():
        if "|" not in line:
            continue
        t, s = line.split("|", 1)
        if s.startswith("auto:") and "編集前" in s:
            continue
        rows.append({"time": t.strip(), "summary": s.strip()})
        if len(rows) >= n:
            break
    return rows


def loop_state():
    if not os.path.exists(LEDGER):
        return {"last_jst": "(ledger なし)", "idle_min": None, "guess": "不明", "level": "stopped"}
    last_ts = None
    try:
        with open(LEDGER, encoding="utf-8") as f:
            lines = deque(f, maxlen=40)
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            ts = d.get("ts") or d.get("time")
            if ts:
                last_ts = ts
    except Exception:
        pass
    if not last_ts:
        return {"last_jst": "(ターン記録なし)", "idle_min": None, "guess": "不明", "level": "stopped"}
    try:
        dt = datetime.fromisoformat(last_ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        idle = max(0, int((datetime.now(timezone.utc) - dt).total_seconds() // 60))
        jst = dt.astimezone(JST).strftime("%m-%d %H:%M JST")
    except Exception:
        return {"last_jst": last_ts, "idle_min": None, "guess": "解析不能", "level": "idle"}
    if idle <= 15:
        guess, level = "稼働中らしい (active)", "active"
    elif idle <= 60:
        guess, level = "一時停止 / アイドル", "idle"
    else:
        guess, level = "停止しているらしい (要再開・人間介在)", "stopped"
    return {"last_jst": jst, "idle_min": idle, "guess": guess, "level": level}


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def wrap(s, width):
    """明示改行を保ちつつ East Asian Width をざっくり考慮して折り返す."""
    s = str(s)
    out = []
    for seg in s.split("\n"):
        cur = ""
        cur_width = 0
        for ch in seg:
            ch_width = 2 if unicodedata.east_asian_width(ch) in {"F", "W", "A"} else 1
            if cur and cur_width + ch_width > width:
                out.append(cur)
                cur = ch
                cur_width = ch_width
            else:
                cur += ch
                cur_width += ch_width
        out.append(cur)
    return out or [""]


STATE_STYLE = {
    "active":  ("#2e7d32", "稼働中"),
    "idle":    ("#f9a825", "アイドル"),
    "paused":  ("#f9a825", "一時停止"),
    "blocked": ("#c62828", "ブロック"),
    "stopped": ("#c62828", "停止"),
    "done":        ("#2e7d32", "✓"),
    "in_progress": ("#1565c0", "●"),
    "todo":        ("#9e9e9e", "○"),
}


def build():
    seed = {}
    if os.path.exists(SEED):
        try:
            seed = json.load(open(SEED, encoding="utf-8"))
        except Exception:
            seed = {}
    pulse = git_pulse()
    loop = loop_state()
    verdict = seed.get("verdict", "(verdict 未取得 — status-model workflow を実行)")
    roadmap = seed.get("roadmap", [])
    gates = seed.get("human_gates", [])
    issues = seed.get("open_issues", [])
    level = loop["level"]
    badge_color, badge_txt = STATE_STYLE.get(level, ("#9e9e9e", level))
    gen = datetime.now(JST).strftime("%Y-%m-%d %H:%M JST")

    W = 1180
    LX, LW = 40, 535
    RX, RW = 605, 535
    P = []

    def card(x, y, w, title):
        P.append(f'<text x="{x+4}" y="{y}" font-size="18" font-weight="bold" fill="#222">{esc(title)}</text>')
        P.append(f'<line x1="{x}" y1="{y+8}" x2="{x+w}" y2="{y+8}" stroke="#e0e0e0" stroke-width="1.5"/>')
        return y + 34

    # header height depends on verdict wrap
    vlines = wrap(verdict, 58)
    verdict_h = 12 + len(vlines) * 24 + 12
    body_y0 = 96 + verdict_h + 30

    # ---- left column ----
    ly = body_y0
    ly = card(LX, ly, LW, "ループ稼働 (llterm)")
    P.append(f'<circle cx="{LX+14}" cy="{ly-5}" r="9" fill="{badge_color}"/>')
    idle_txt = "" if loop["idle_min"] is None else f"（{loop['idle_min']} 分前）"
    P.append(f'<text x="{LX+34}" y="{ly}" font-size="17" fill="#222">最終ターン: <tspan font-weight="bold">{esc(loop["last_jst"])}</tspan>{esc(idle_txt)}</text>')
    ly += 28
    P.append(f'<text x="{LX+34}" y="{ly}" font-size="16" fill="{badge_color}" font-weight="bold">{esc(loop["guess"])}</text>')
    ly += 46

    ly = card(LX, ly, LW, "直近の活動（コミット）")
    for it in pulse:
        P.append(f'<text x="{LX+4}" y="{ly}" font-size="13.5" fill="#1565c0" font-family="Consolas,monospace">{esc(it["time"])}</text>')
        seg = wrap(it["summary"], 28)
        P.append(f'<text x="{LX+92}" y="{ly}" font-size="14.5" fill="#333">{esc(seg[0])}</text>')
        ly += 21
        for extra in seg[1:]:
            P.append(f'<text x="{LX+92}" y="{ly}" font-size="14.5" fill="#333">{esc(extra)}</text>')
            ly += 21
        ly += 5
    left_bottom = ly

    # ---- right column ----
    ry = body_y0
    ry = card(RX, ry, RW, "ロードマップ (capability-first)")
    for r in roadmap:
        col, mark = STATE_STYLE.get(r.get("state", "todo"), ("#9e9e9e", "?"))
        P.append(f'<rect x="{RX+2}" y="{ry-14}" width="24" height="20" rx="4" fill="{col}"/>')
        P.append(f'<text x="{RX+14}" y="{ry+1}" font-size="13" fill="#fff" text-anchor="middle" font-weight="bold">{esc(mark)}</text>')
        P.append(f'<text x="{RX+36}" y="{ry}" font-size="15.5" fill="#222"><tspan font-weight="bold">{esc(r.get("stage",""))}</tspan>  {esc(r.get("label",""))}</text>')
        ry += 21
        for nl in wrap(r.get("note", ""), 36):
            if not nl:
                continue
            P.append(f'<text x="{RX+36}" y="{ry}" font-size="12.5" fill="#666">{esc(nl)}</text>')
            ry += 17
        ry += 8
    ry += 4

    ry = card(RX, ry, RW, "人間ゲート待ち")
    if not gates:
        P.append(f'<text x="{RX+4}" y="{ry}" font-size="14.5" fill="#888">（なし）</text>'); ry += 22
    for g in gates:
        for k, gl in enumerate(wrap(g, 33)):
            P.append(f'<text x="{RX+4}" y="{ry}" font-size="14" fill="#b26a00">{esc(("⏸ " if k == 0 else "    ") + gl)}</text>')
            ry += 20
        ry += 4
    ry += 4

    ry = card(RX, ry, RW, "未検証 / open issues (honest)")
    if not issues:
        P.append(f'<text x="{RX+4}" y="{ry}" font-size="14.5" fill="#888">（なし）</text>'); ry += 22
    for s in issues:
        for k, sl in enumerate(wrap(s, 33)):
            P.append(f'<text x="{RX+4}" y="{ry}" font-size="14" fill="#555">{esc(("• " if k == 0 else "   ") + sl)}</text>')
            ry += 20
        ry += 4
    right_bottom = ry

    H = max(left_bottom, right_bottom) + 36

    head = []
    head.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Segoe UI, Hiragino Sans, Meiryo, sans-serif">')
    head.append(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')
    head.append(f'<rect x="0" y="0" width="{W}" height="96" fill="#1f2733"/>')
    head.append(f'<text x="40" y="44" font-size="26" fill="#fff" font-weight="bold">llterm × llcore — 自走ステータス</text>')
    head.append(f'<text x="40" y="74" font-size="13.5" fill="#9fb3c8">generated {esc(gen)}   ／   再実行: py -3.11 tools/llterm_status.py</text>')
    bw = 196
    head.append(f'<rect x="{W-bw-30}" y="26" width="{bw}" height="44" rx="22" fill="{badge_color}"/>')
    head.append(f'<circle cx="{W-bw-30+26}" cy="48" r="8" fill="#ffffff" fill-opacity="0.9"/>')
    head.append(f'<text x="{W-bw-30+bw/2+12:.0f}" y="55" font-size="20" fill="#fff" text-anchor="middle" font-weight="bold">{esc(badge_txt)}</text>')
    head.append(f'<rect x="0" y="96" width="{W}" height="{verdict_h}" fill="#eef2f7"/>')
    vy = 96 + 26
    for i, vl in enumerate(vlines):
        prefix = "総括: " if i == 0 else "　　　"
        head.append(f'<text x="40" y="{vy}" font-size="16" fill="#1f2733">{esc(prefix + vl)}</text>')
        vy += 24

    svg = "\n".join(head + P + ["</svg>"])
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w", encoding="utf-8").write(svg)
    print("wrote:", OUT, f"({len(svg)} bytes, W={W}, H={H})")
    print(f"  loop: {loop['last_jst']} idle={loop['idle_min']} level={loop['level']}")
    print(f"  pulse={len(pulse)} roadmap={len(roadmap)} gates={len(gates)} issues={len(issues)} verdict_lines={len(vlines)}")


if __name__ == "__main__":
    build()
