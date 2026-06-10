# SPDX-License-Identifier: Apache-2.0
"""検証付き会話デモ (F11 consumer story): 実会話中のオンライン適応を sound certificate が守る。

llcore.chat (実 SmolLM2-360M-Instruct, CPU) との **本物の段階的会話** をデータ流とし、
その mid-layer hidden state に small-n (n=6) sidecar 記憶アダプタ
`s' = decay⊙s + (1-decay)⊙tanh(Ws + x)` を載せる。毎ターン、アダプタに上方圧変異
(結合を強める方向の adversarial stress, 開示済) を提案し、4 方式の gate が採否を決める:

  none        — 無 gate (負の対照): 全提案を採用
  stable_emp  — STABLE 風経験 gate: 「今ターンの実会話 hidden 上で摂動忘却 < ε なら採用」
  cert_two    — sound contraction certificate (頂点 SVD)
  cert_sdp    — sound SDP-Lyapunov (最 navigable; cvxpy+CLARABEL 無なら cert_two へ縮退)

毎ターン、各系統の **真の安定度** を empirical_rho (from-below 一致オラクル) で測る。

★期待される honest な見どころ (Phase 2 確定知見の実会話版):
- tanh 基質では ρ≥1 でも実軌道は発散して見えない (norm 有界 + 感度減衰)。
  ∴ 経験 gate (stable_emp) は実会話データ上で「安全に見える」変異を通し、真 ρ≥1 へ漂流する。
- sound cert 系統のみ全採用変異に証明が付き、真 ρ<1 (echo-state = 摂動忘却 = セッション記憶の
  再現性) が **保証** され続ける。

honest 留保 (重要):
- アダプタは sidecar であり **LLM の応答品質には一切影響しない** (LLM は frozen)。
  capability 主張はしない (Phase 2 で capability NEGATIVE 確定済)。これは GUARANTEE のデモ。
- **ρ_sup の意味論**: empirical_rho は (s,x)∈[-1,1]^n box 上の点別 ρ(J) の from-below sup。
  ρ_sup≥1 は「いかなるノルムでも一様収縮にならない=収縮証明可能領域の外」を意味する
  (軌道ノルムの発散ではない — tanh+凸結合の前方不変性で状態は [-1,1]^n に閉じ込められ発散不能)。
  「真の安定度」「発散」と呼ぶのは過大 → 本デモは「収縮証明可能性指標」と呼ぶ。
- empirical_rho は from-below (絶対証明でない)。cert とは**計算経路のみ独立** (eigvals@サンプル点 vs
  SVD/SDP@t-box 頂点)。同一 Jacobian モデル・同一 box を共有 → **cert 系統の違反 0 は soundness の
  演繹的帰結 (整合性チェック) であり独立な経験的証拠ではない**。
- **保証の射程**: cert_two は σ^T のターン内定量収縮を証明。cert_sdp は margin=1e-7 の feasibility
  証明で**漸近的 (rate<1) のみ**、率は定量化されない (本走行の証明済 P-norm 率は 1−O(1e-8)、
  ターン内 2-norm 上界は √cond(P)=1.43–2.84 = 増幅許容)。観測の 1e-5 級忘却は経験的挙動。
  gene 切替を跨ぐセッション全体の echo-state は cert_sdp 系統では未証明。
- **3/4 曲線はデータ非依存**: none/cert_two/cert_sdp の gate は X にも state にも依存しない worst-case
  証明ゆえ、seed のみで決まり会話無しでも同一 (会話が効くのは stable_emp 採否と忘却診断・state_norm)。
- 変異の上方圧バイアスは「最適化圧で結合が強まる」状況の模擬 (開示)。
- 射影入力は [-1,1] に clip (cert の max_input_abs=1 と整合)。clip 率は記録する。
- 会話は本物 (生成 LLM 出力 verbatim 記録) だが、ターン数 8 の小規模デモ。

成果物:
  - phase2_demo_verified_chat_results.json (会話 verbatim + 全系統 per-turn 軌跡)
  - phase2_demo_verified_chat.svg (真 ρ の軌跡: 経験系は閾値 1 を越え、cert 系は下に留まる)

使い方::

    py -3.11 research/rllm_pivot/phase2_demo_verified_chat.py [--model ID] [--turns N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.abspath(os.path.join(HERE, "..", "..", "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "verified_evolution_sdp_gate")))

import coupled_nd as C  # noqa: E402

from llcore.chat import ChatSession, GenerationSettings, TransformersBackend  # noqa: E402

N = 6
SEED = 20260611
EPS_STABLE = 1e-3        # stable_emp gate / 忘却判定の閾値 (phase0 と同値)
N_PROPOSALS = 6          # 毎ターンの変異提案数 (最初に admit されたものを採用)
RHO_SAMPLES = 3000       # empirical_rho の from-below サンプル数

# 段階的会話 (実 LLM が応答を生成する。アダプタのデータ流 = この会話の hidden)
CONVERSATION = [
    "Hello! My name is Kazufumi. Nice to meet you.",
    "What is the capital of France?",
    "Let's talk about cooking. Suggest one simple pasta dish.",
    "What is my name?",
    "New topic: space. Name one planet in our solar system.",
    "Tell me one interesting fact about that planet.",
    "Switch to music: name one classical composer.",
    "Thank you! Say goodbye in one short sentence.",
]

METHODS = ["none", "stable_emp", "cert_two", "cert_sdp"]
METHOD_LABEL = {
    "none": "無 gate (負の対照)",
    "stable_emp": "STABLE 風経験 gate",
    "cert_two": "cert_two (sound)",
    "cert_sdp": "cert_sdp (sound)",
}


# --------------------------------------------------------------------------- #
# 実会話 hidden の捕捉
# --------------------------------------------------------------------------- #
class HiddenCapture:
    """会話履歴の templated 全文を forward し、新規ターン分の mid-layer hidden を返す。

    LLM は frozen (no_grad / eval)。ターン毎に履歴全文を 1 回 forward する
    (8 ターン ≲ 1k tokens, CPU で数秒) — 実会話の文脈そのままの hidden を得る honest 経路。
    """

    def __init__(self, backend: TransformersBackend) -> None:
        self._backend = backend
        self._prev_len = 0
        model = backend.model
        self.layer = int(model.config.num_hidden_layers) // 2
        self.hidden_size = int(model.config.hidden_size)

    def new_hidden(self, session: ChatSession) -> np.ndarray:
        import torch

        tok = self._backend.tokenizer
        model = self._backend.model
        msgs = [m.as_dict() for m in session.history]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        ids = tok(text, return_tensors="pt", add_special_tokens=False)
        with torch.no_grad():
            out = model(**ids, output_hidden_states=True)
        H = out.hidden_states[self.layer][0].numpy().astype(np.float64)  # (T, hidden)
        new = H[self._prev_len:]
        self._prev_len = H.shape[0]
        return new


# --------------------------------------------------------------------------- #
# アダプタ系統
# --------------------------------------------------------------------------- #
def sample_certified_base(rng: np.random.Generator) -> C.CoupledNDGene:
    """cert_two が admit する初期 gene (全系統がここから出発 = 公平な開始点)。"""
    for _ in range(200):
        decay = rng.uniform(0.0, 1.0, N)
        W = (rng.normal(0.0, 1.0, (N, N)) / np.sqrt(N)) * float(rng.uniform(0.3, 0.8))
        for _ in range(60):
            g = C.CoupledNDGene.make(decay=decay, W=W)
            if C.cert_two(g):
                return g
            W = W * 0.85
    raise RuntimeError("certified base gene が見つからない (想定外)")


def mutate(g: C.CoupledNDGene, rng: np.random.Generator) -> C.CoupledNDGene:
    """上方圧変異 (adversarial stress, 開示済): W をスケールアップ + ノイズ。"""
    decay = np.clip(g.decay + rng.normal(0, 0.12, N), 0.0, 1.0)
    W = np.clip(g.W * float(rng.uniform(1.05, 1.30)) + rng.normal(0, 0.10, (N, N)), -2.0, 2.0)
    return C.CoupledNDGene.make(decay=decay, W=W)


def run_adapter(g: C.CoupledNDGene, X: np.ndarray, s0: np.ndarray) -> np.ndarray:
    s = s0.copy()
    for t in range(X.shape[0]):
        s = C.step(g, s, X[t])
    return s


def twin_forgetting(
    g: C.CoupledNDGene, X: np.ndarray, s0: np.ndarray, rng: np.random.Generator
) -> float:
    """現在状態 s0 とその摂動 s0+δ の双子軌道を今ターンの実 hidden で回し終端乖離を返す。
    小 = 摂動忘却 (echo-state) = セッション記憶が再現的。"""
    d = rng.normal(size=N)
    d = 1e-2 * d / (np.linalg.norm(d) + 1e-12)
    return float(np.linalg.norm(run_adapter(g, X, s0) - run_adapter(g, X, s0 + d)))


def cert_box_sigma(g: C.CoupledNDGene) -> float:
    """certificate が見る worst-case (box 頂点の最大 σ_max) — 透明性のため記録。"""
    t_lo = C.t_min_per_coord(g)
    return max(
        float(np.linalg.svd(C._jac_at_t(g, v), compute_uv=False)[0])
        for v in C._box_vertices(t_lo)
    )


def gate_admits(method: str, g: C.CoupledNDGene, X: np.ndarray, s: np.ndarray,
                probe_rng: np.random.Generator) -> bool:
    """採否判定。probe_rng は経験 gate の摂動プローブ専用 (変異 RNG と分離 —
    プローブが変異提案列を消費して系統間の提案列を歪めないため)。"""
    if method == "none":
        return True
    if method == "stable_emp":
        return twin_forgetting(g, X, s, probe_rng) < EPS_STABLE
    if method == "cert_two":
        return C.cert_two(g)
    if method == "cert_sdp":
        return C.cert_sdp(g)
    raise ValueError(method)


# --------------------------------------------------------------------------- #
# SVG (SMIL 補助つき・静止フレーム完成形 — feedback_animated_svg_static_fallback)
# --------------------------------------------------------------------------- #
def build_svg(per_method: dict[str, list[dict]], n_turns: int, sdp_available: bool) -> str:
    W_, H_ = 860, 520
    x0, y0, pw, ph = 90, 90, 640, 300
    rho_max = max(
        1.6,
        max(r["true_rho"] for rows in per_method.values() for r in rows) * 1.08,
    )
    COLORS = {"none": "#ff9f43", "stable_emp": "#ffd166", "cert_two": "#4ea1ff",
              "cert_sdp": "#7bd88f"}

    def sx(turn: int) -> float:
        return x0 + pw * (turn - 1) / max(n_turns - 1, 1)

    def sy(rho: float) -> float:
        return y0 + ph - ph * min(rho, rho_max) / rho_max

    parts: list[str] = []
    # 軸 + ρ=1 閾値
    parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x0}" y2="{y0+ph}" stroke="#3a4152" stroke-width="1"/>')
    parts.append(f'<line x1="{x0}" y1="{y0+ph}" x2="{x0+pw}" y2="{y0+ph}" stroke="#3a4152" stroke-width="1"/>')
    ty = sy(1.0)
    parts.append(f'<line x1="{x0}" y1="{ty:.1f}" x2="{x0+pw}" y2="{ty:.1f}" stroke="#e05561" stroke-width="1.5" stroke-dasharray="7,5"/>')
    parts.append(f'<text x="{x0+pw+6}" y="{ty+4:.1f}" fill="#e05561" font-size="12">ρ = 1 (収縮証明可能性の境界)</text>')
    for rho in (0.5, 1.5):
        if rho < rho_max:
            yy = sy(rho)
            parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+pw}" y2="{yy:.1f}" stroke="#222836" stroke-width="1"/>')
            parts.append(f'<text x="{x0-8}" y="{yy+4:.1f}" fill="#7a8294" font-size="11" text-anchor="end">{rho}</text>')
    parts.append(f'<text x="{x0-8}" y="{sy(1.0)+4:.1f}" fill="#e05561" font-size="11" text-anchor="end">1.0</text>')
    for t in range(1, n_turns + 1):
        parts.append(f'<text x="{sx(t):.1f}" y="{y0+ph+20}" fill="#7a8294" font-size="11" text-anchor="middle">{t}</text>')
    parts.append(f'<text x="{x0+pw/2}" y="{y0+ph+42}" fill="#9aa3b2" font-size="12" text-anchor="middle">実会話ターン (挨拶 → Q&amp;A → 文脈 → 話題転換 …)</text>')
    parts.append(f'<text x="24" y="{y0+ph/2}" fill="#9aa3b2" font-size="12" transform="rotate(-90 24 {y0+ph/2})" text-anchor="middle">ρ_sup (収縮証明可能性指標, empirical from-below)</text>')

    # 各系統の折れ線 (SMIL 描画アニメ + 静止完成形)
    legend_y = 56
    lx = x0
    for m in METHODS:
        if m == "cert_sdp" and not sdp_available:
            continue
        rows = per_method[m]
        xy = [(sx(r["turn"]), sy(r["true_rho"])) for r in rows]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in xy)
        col = COLORS[m]
        # 実 polyline 長 + 余裕 (dasharray が実長より短いと最終フレームが破線化するため)
        path_len = sum(
            ((xy[i + 1][0] - xy[i][0]) ** 2 + (xy[i + 1][1] - xy[i][1]) ** 2) ** 0.5
            for i in range(len(xy) - 1)
        ) + 40
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.5" '
            f'stroke-dasharray="{path_len:.0f}" stroke-dashoffset="0">'
            f'<animate attributeName="stroke-dashoffset" from="{path_len:.0f}" to="0" dur="2.2s" fill="freeze"/></polyline>'
        )
        for r in rows:
            marker = "●" if r["accepted"] else "×"
            parts.append(
                f'<text x="{sx(r["turn"]):.1f}" y="{sy(r["true_rho"])-7:.1f}" fill="{col}" '
                f'font-size="10" text-anchor="middle">{marker}</text>'
            )
        parts.append(f'<rect x="{lx}" y="{legend_y-10}" width="14" height="4" fill="{col}"/>')
        parts.append(f'<text x="{lx+20}" y="{legend_y-4}" fill="#cfd6e4" font-size="12">{METHOD_LABEL[m]}</text>')
        lx += 200

    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W_} {H_}" font-family="Segoe UI,Helvetica,Arial,sans-serif">
  <rect width="{W_}" height="{H_}" fill="#0f1320"/>
  <text x="{W_/2}" y="30" fill="#e8ecf4" font-size="19" font-weight="700" text-anchor="middle">実会話中のオンライン適応 — sound certificate だけが収縮証明可能領域 (ρ_sup&lt;1) を守る</text>
  {"".join(parts)}
  <text x="{W_/2}" y="{H_-44}" fill="#ff9f43" font-size="12" text-anchor="middle">無 gate / 経験 gate は実会話データ上で「安全に見える」変異を通し、ρ_sup≥1 (一様収縮不能 = echo-state 保証の喪失) へ漂流する</text>
  <text x="{W_/2}" y="{H_-26}" fill="#4ea1ff" font-size="12" text-anchor="middle">sound cert は全採用変異に漸近 echo-state (ρ&lt;1) の証明つき — ターン内忘却率の定量保証は cert_two のみ (σ^T)、cert_sdp は漸近保証</text>
  <text x="{W_/2}" y="{H_-9}" fill="#566075" font-size="10" text-anchor="middle">honest: LLM frozen (sidecar, 応答品質主張なし) / none・cert 系 3 曲線はデータ非依存 (worst-case 証明の帰結) / 会話が効くのは経験 gate 採否と忘却診断 / ● 採用 × 棄却</text>
</svg>'''


# --------------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=None, help="HF モデル ID (default: chat の default)")
    parser.add_argument("--turns", type=int, default=len(CONVERSATION))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    turns = CONVERSATION[: max(1, min(args.turns, len(CONVERSATION)))]

    # fail-closed: solver 不在なら cert_sdp 系統をスキップ (meta/SVG に虚偽の sdp_available を載せない)
    sdp_available = bool(C.cvxpy_available())

    # 実会話セッション (frozen LLM)
    backend = TransformersBackend(model_id=args.model, seed=args.seed)
    session = ChatSession(backend, settings=GenerationSettings(max_new_tokens=96))
    capture = HiddenCapture(backend)
    print(f"model: {backend.model_id} (layer {capture.layer}/{backend.model.config.num_hidden_layers}, "
          f"hidden {capture.hidden_size})", flush=True)

    # 固定射影 + 入力スケール (turn 1 で確定し以後固定; [-1,1] clip = cert の入力 box と整合)
    rng_fix = np.random.default_rng(args.seed)
    P = rng_fix.normal(size=(N, capture.hidden_size)) / np.sqrt(capture.hidden_size)
    scale: float | None = None
    n_clip = 0
    n_total = 0

    # 系統初期化 (同一 certified base / 同一 seed の独立 RNG = 公平)
    base = sample_certified_base(np.random.default_rng(args.seed + 1))
    lineages: dict[str, dict] = {}
    for m in METHODS:
        if m == "cert_sdp" and not sdp_available:
            continue
        lineages[m] = {
            "gene": base,
            "state": np.zeros(N),
            # 同一 seed の変異 RNG: gene が分岐する (最初の採否差が出る) までは
            # 全系統が同一提案列を見る。分岐後は gene 依存で自然に異なる (開示)。
            "rng": np.random.default_rng(args.seed + 7),
            "rows": [],
        }

    transcript: list[dict[str, object]] = []
    for ti, prompt in enumerate(turns, start=1):
        t0 = time.time()
        reply = session.ask(prompt)
        H = capture.new_hidden(session)          # (T_new, hidden) 実会話の新規 hidden
        X = H @ P.T                              # (T_new, N)
        if scale is None:
            scale = 0.5 / (float(np.std(X)) + 1e-9)
        X = X * scale
        n_clip += int(np.sum(np.abs(X) > 1.0))
        n_total += X.size
        X = np.clip(X, -1.0, 1.0)
        print(f"\n[turn {ti}] you> {prompt}", flush=True)
        print(f"         llcore> {reply[:90]}{'…' if len(reply) > 90 else ''}", flush=True)
        transcript.append({"turn": ti, "user": prompt, "assistant": reply,
                           # 生成トークン数ではなく templated 新規トークン数 (turn 1 は system prompt 込み)
                           "n_templated_new_tokens": int(X.shape[0]),
                           "includes_system_prompt": ti == 1,
                           "gen_seconds": round(time.time() - t0, 1)})

        for m, ln in lineages.items():
            rng = ln["rng"]
            probe_rng = np.random.default_rng(args.seed + 1000 * ti)  # プローブ専用 (変異列と分離)
            accepted = False
            n_rej = 0
            for _ in range(N_PROPOSALS):
                cand = mutate(ln["gene"], rng)
                if gate_admits(m, cand, X, ln["state"], probe_rng):
                    ln["gene"] = cand
                    accepted = True
                    break
                n_rej += 1
            g = ln["gene"]
            ln["state"] = run_adapter(g, X, ln["state"])
            rho = float(C.empirical_rho(g, n_samples=RHO_SAMPLES, seed=args.seed + ti))
            forget = twin_forgetting(g, X, ln["state"], np.random.default_rng(args.seed + ti))
            row = {
                "turn": ti,
                "accepted": accepted,
                "n_rejected_before_accept": n_rej,
                "true_rho": rho,
                "observed_forgetting_on_turn": forget,
                "cert_box_sigma": cert_box_sigma(g),
                "state_norm": float(np.linalg.norm(ln["state"])),
            }
            ln["rows"].append(row)
            flag = "⚠ρ≥1" if rho >= 1.0 else "ρ<1"
            print(f"  {METHOD_LABEL[m]:20s} {'採用' if accepted else '棄却→維持'}  "
                  f"真ρ={rho:.2f} ({flag})  会話上の忘却={forget:.1e}  "
                  f"cert_sup={row['cert_box_sigma']:.2f}", flush=True)

    # ---- verdict (データから機械的に) ----
    per_method = {m: ln["rows"] for m, ln in lineages.items()}
    final = {m: rows[-1]["true_rho"] for m, rows in per_method.items()}
    fooled = {
        m: any(r["true_rho"] >= 1.0 and r["observed_forgetting_on_turn"] < EPS_STABLE
               for r in rows)
        for m, rows in per_method.items()
    }
    cert_violations = sum(
        1 for m in ("cert_two", "cert_sdp") if m in per_method
        for r in per_method[m] if r["true_rho"] >= 1.0
    )

    results = {
        "meta": {
            "model": backend.model_id, "layer": capture.layer, "n": N,
            "seed": args.seed, "n_turns": len(turns), "eps_stable": EPS_STABLE,
            "n_proposals_per_turn": N_PROPOSALS, "rho_samples": RHO_SAMPLES,
            "sdp_available": bool(sdp_available),
            "input_clip_fraction": round(n_clip / max(n_total, 1), 4),
            "honest_notes": [
                "LLM は frozen — アダプタは sidecar で応答品質に影響しない (capability 主張なし)",
                "empirical_rho は from-below 一致オラクル (絶対証明でない)",
                "変異は上方圧バイアスつき adversarial stress (開示済)",
                "射影入力は [-1,1] clip (cert の max_input_abs=1 と整合); clip 率は meta に記録",
                "tanh 基質では ρ_sup≥1 でも軌道ノルムは発散しない (前方不変 [-1,1]^n) — "
                "ρ_sup≥1 は『いかなるノルムでも一様収縮にならない=収縮証明可能領域の外』の意味であり、"
                "『発散』ではない。それを見抜けるのが sound cert の存在理由",
                "cert 系統の ρ≥1 違反 0 は cert soundness の演繹的帰結 (オラクルは cert と同一 box・"
                "同一 Jacobian モデルを from-below サンプル) — 実装バグ検出の整合性チェックであり独立な経験的証拠ではない",
                "cert_sdp は margin=1e-7 の feasibility 証明で率は定量化されない (本走行の証明済 P-norm 率は "
                "1−O(1e-8)、ターン内 2-norm 上界は √cond(P)=1.43–2.84) — ターン内忘却の定量保証は cert_two (σ^T) のみ。"
                "観測の 1e-5 級忘却は経験的挙動",
                "none/cert_two/cert_sdp の gate はデータ非依存 (worst-case 証明) — これら 3 系統の gene 列・採否・"
                "true_rho・cert_box_sigma は seed のみで決まり会話無しでも同一。会話が効くのは stable_emp 採否と忘却診断・state_norm",
                "true_rho という命名は ρ_sup (収縮証明可能性指標) の略 — 非 cert 系統では ρ<1 を断定しない (from-below 推定)",
            ],
        },
        "conversation": transcript,
        "per_method": per_method,
        "final_true_rho": final,
        "empirical_gate_fooled": fooled,
        "cert_lineage_rho_violations": cert_violations,
    }
    out_json = os.path.join(HERE, "phase2_demo_verified_chat_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    svg = build_svg(per_method, len(turns), sdp_available)
    out_svg = os.path.join(HERE, "phase2_demo_verified_chat.svg")
    with open(out_svg, "w", encoding="utf-8") as f:
        f.write(svg)

    print("\n=== verdict ===", flush=True)
    for m, rows in per_method.items():
        peak = max(r["true_rho"] for r in rows)
        print(f"  {METHOD_LABEL[m]:20s} 最終 真ρ={final[m]:.2f}  最大={peak:.2f}  "
              f"{'(経験的に騙された turn あり)' if fooled[m] and m in ('none', 'stable_emp') else ''}",
              flush=True)
    print(f"  cert 系統の ρ≥1 違反: {cert_violations} "
          f"(soundness ⇒ 必然的に 0; 非 0 = 実装バグ検出 — 整合性チェックであり経験的発見ではない)", flush=True)
    print(f"\nJSON: {out_json}\nSVG : {out_svg}", flush=True)
    return 0 if cert_violations == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
