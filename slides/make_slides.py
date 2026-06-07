# SPDX-License-Identifier: Apache-2.0
"""llcore 研究知見スライド生成 (16:9 ワイド・シンプル・1 スライド 1 メッセージ)。

ja / en の 2 本を python-pptx で生成。数値はすべて公開済み一次資料
(research/internalization_poc/ + research/paper/PAPER_DRAFT.md) の確定値。
ライセンス: スライド本体 = CC BY 4.0 (出典明示で商用利用可)。

実行::  py -3.11 slides/make_slides.py
出力::  slides/llcore_findings_2026_ja.pptx / llcore_findings_2026_en.pptx
"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

_HERE = Path(__file__).resolve().parent
ACCENT = RGBColor(0x4F, 0x81, 0xBD)
DARK = RGBColor(0x1A, 0x1A, 0x1A)
GRAY = RGBColor(0x66, 0x66, 0x66)
W, H = Inches(13.333), Inches(7.5)   # 16:9

FOOTER = {
    "ja": "© 2026 Kazufumi Furuse — CC BY 4.0 | 出典: github.com/furuse-kazufumi/llcore | qiita.com/furuse-kazufumi",
    "en": "© 2026 Kazufumi Furuse — CC BY 4.0 | Source: github.com/furuse-kazufumi/llcore | qiita.com/furuse-kazufumi",
}


def _blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def _txt(slide, x, y, w, h, lines, size=18, bold=False, color=DARK, align=PP_ALIGN.LEFT,
         font_ja=True, space_after=None):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, line in enumerate(lines if isinstance(lines, list) else [lines]):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        if isinstance(line, tuple):
            text, lv = line
            p.level = lv
        else:
            text = line
        p.text = text
        p.alignment = align
        if space_after is not None:
            p.space_after = Pt(space_after)
        for r in p.runs:
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
            r.font.name = "Meiryo UI" if font_ja else "Segoe UI"
    return box


def _slide(prs, lang, title, bullets, note=None, accent_bar=True):
    s = _blank(prs)
    if accent_bar:
        bar = s.shapes.add_shape(1, Inches(0.55), Inches(0.62), Inches(0.12), Inches(0.62))
        bar.fill.solid()
        bar.fill.fore_color.rgb = ACCENT
        bar.line.fill.background()
    _txt(s, Inches(0.85), Inches(0.5), Inches(11.9), Inches(1.0), title, size=30, bold=True)
    _txt(s, Inches(0.9), Inches(1.7), Inches(11.6), Inches(4.6), bullets, size=19)
    if note:
        _txt(s, Inches(0.9), Inches(6.25), Inches(11.6), Inches(0.5), note, size=12, color=GRAY)
    _txt(s, Inches(0.9), Inches(7.0), Inches(11.6), Inches(0.4), FOOTER[lang], size=10, color=GRAY)
    return s


def build(lang: str) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    ja = lang == "ja"

    # 1. タイトル
    s = _blank(prs)
    _txt(s, Inches(1.2), Inches(2.2), Inches(11), Inches(1.4),
         "自己進化 AI を「証明」で守る" if ja else "Guarding Self-Evolving AI with Proofs",
         size=44, bold=True)
    _txt(s, Inches(1.2), Inches(3.6), Inches(11), Inches(0.8),
         "llcore 研究知見 2026 — 検証付きニューラル進化 (CPU 完結)" if ja else
         "llcore research findings 2026 — verified neural evolution on CPU",
         size=24, color=ACCENT)
    _txt(s, Inches(1.2), Inches(5.2), Inches(11), Inches(1.2), [
        "Kazufumi Furuse",
        "github.com/furuse-kazufumi/llcore (paper draft + 全実験コード/データ公開)" if ja else
        "github.com/furuse-kazufumi/llcore (paper draft + all experiment code/data public)",
        "本スライド: CC BY 4.0 — 出典明示で商用利用可" if ja else
        "These slides: CC BY 4.0 — commercial use permitted with attribution",
    ], size=16, color=GRAY)
    _txt(s, Inches(1.2), Inches(7.0), Inches(11), Inches(0.4), FOOTER[lang], size=10, color=GRAY)

    # 2. 問題
    _slide(prs, lang,
           "問題: 自己進化 AI は「何を根拠に」自分の変更を許可するか" if ja else
           "Problem: on what grounds may a self-evolving AI admit its own changes?",
           [
               "AI が自分のコア (記憶・力学) を書き換えながら進化する時代 — 危険な変更をどう弾くか" if ja else
               "AI now evolves its own core (memory, dynamics) — how do we reject dangerous changes?",
               "主流 (DGM / SEAL 系) = 経験的ゲート: ベンチマークスコアや過去の失敗で判断" if ja else
               "Mainstream (DGM / SEAL line) = empirical gates: benchmark scores, past failures",
               ("経験的ゲートの構造的弱点: ① 死んで学ぶ (失敗コストを払う) ② Goodhart 可能 (指標ハック)" if ja else
                "Structural weakness: (1) learns by dying — pays the failure cost (2) Goodhart-able — metrics can be gamed"),
               "llcore の問い: 性質を絞れば「数学的証明」でゲートできるのではないか" if ja else
               "llcore's question: if we narrow the property, can a mathematical proof be the gate?",
           ])

    # 3. アプローチ
    _slide(prs, lang,
           "アプローチ: 健全な収縮証明ゲート (fail-closed)" if ja else
           "Approach: a sound contraction-certifier gate (fail-closed)",
           [
               "証明する性質 = 収縮 (ρ < 1): 内部状態が発散しない・記憶力学が暴走しない" if ja else
               "Certified property = contraction (ρ < 1): internal state cannot diverge",
               "3 段の健全 certifier: 閉形式 ∞-norm (O(n²)) → 頂点 2-norm → Lyapunov SDP" if ja else
               "Three sound certifiers: closed-form ∞-norm (O(n²)) → vertex 2-norm → Lyapunov SDP",
               "fail-closed: 証明できない変更は通さない。証明は定理の帰結 = 捏造不能" if ja else
               "Fail-closed: unprovable changes are rejected. Proofs are theorem-derived = non-fabricable",
               "進化ループ内で評価前に gate (prove-then-reject) — 事後検証の 17-19 倍安い" if ja else
               "Gating inside the evolution loop, before evaluation — 17-19x cheaper than post-hoc certification",
           ],
           note="全証明・全実験は事前登録 → 結果の順で記録 (research/paper/PAPER_DRAFT.md)" if ja else
                "All proofs & experiments pre-registered before results (research/paper/PAPER_DRAFT.md)")

    # 4. 実験場
    _slide(prs, lang,
           "実験場: 「死ねる環境」×「経験が記憶になる 3 つの道」" if ja else
           "Testbed: a lethal environment x three ways experience becomes memory",
           [
               "環境が再帰ゲイン κ を 2 倍にステップ → それまで安全だった個体が発散 = 死" if ja else
               "The environment steps the recurrence gain κ by 2x → previously-safe genes diverge = death",
               "ENDO (自己予見): 内的な健全証明で死を予見し、評価前に自分を弾く" if ja else
               "ENDO (endogenous foresight): a sound self-held certificate rejects lethal changes before evaluation",
               "REVIVE (復活修復): 死を経験するが、記憶遺伝子を保ったまま安全側に修復され蘇る" if ja else
               "REVIVE (certificate-preserving repair): dies, but is revived with its memory gene intact",
               "OBSERVE (社会的観察): 他個体の死の近傍を経験的に避ける (経験的ゲートの代表)" if ja else
               "OBSERVE (observational): empirically avoids the neighborhood of others' observed deaths",
               "n=20 シード・事前登録・助走後の定常統計で測定 (negative も全部開示)" if ja else
               "n=20 seeds, pre-registered, steady-state after warm-up (all negatives disclosed)",
           ])

    # 5. 知見 1
    _slide(prs, lang,
           "知見 1: 証明は死ぬ前に分かる — 定常死ゼロ" if ja else
           "Finding 1: proofs know before dying — zero steady-state deaths",
           [
               "致命的評価の数 (定常状態, linear 基質):" if ja else
               "Lethal evaluations (steady state, linear substrate):",
               ("   無防備 27.2  /  観察学習 17.3  /  固定ゲート 10.2  /  証明 (ENDO) 0.0", 1),
               "観察学習は機能する — しかし学んでも死が残る。「死んで学ぶ」は構造コスト" if ja else
               "Observational learning works — but deaths remain. Learning-by-dying is a structural cost",
               "頑健性: シード・κ 強度・次元・課題難度を変えた 12/12 環境で証明側の優位が保持" if ja else
               "Robustness: the proof's advantage holds in 12/12 pre-registered configurations",
               "しかも証明ゲートは記憶獲得能力を犠牲にしない (最大適応度差 0.7%)" if ja else
               "And the proof gate does not strangle memory capability (worst fitness gap 0.7%)",
           ],
           note="p < 0.001 (paired sign-flip, n=20)。数値はすべて results_viability_*.json (公開)" if ja else
                "p < 0.001 (paired sign-flip, n=20). All numbers in public results_viability_*.json")

    # 6. 知見 2
    _slide(prs, lang,
           "知見 2: 復活がないと、経験は記憶にならない" if ja else
           "Finding 2: without revival, experience does not become memory",
           [
               "死 = 個体の除去 = その個体が蓄積した経験 (遺伝子) も消える" if ja else
               "Death removes the individual — and erases the experience its genes accumulated",
               "REVIVE は無防備と同程度死ぬのに、集団に残る記憶 (平均適応度) が有意に高い" if ja else
               "REVIVE dies about as often as no-gate, yet preserves significantly more population memory",
               ("   集団平均適応度: 無防備 0.649 → 復活あり 0.708 (Δ=+0.060, p=0.0011)", 1),
               "修復は「記憶チャネルを保ったまま力学だけ安全化」— 死が傷として記憶に刻まれる" if ja else
               "Repair preserves the memory channel while safing the dynamics — death leaves a scar, not a void",
           ])

    # 7. 知見 3
    _slide(prs, lang,
           "知見 3: 証明 1 つは、経験的機構の積み重ねに勝る" if ja else
           "Finding 3: one proof beats stacked empirical mechanisms",
           [
               "3 機構の全 8 組み合わせを同条件で比較 (2³ factorial):" if ja else
               "All 8 mechanism combinations compared under identical conditions (2^3 factorial):",
               ("   証明 (E) を含む 4 組み合わせ: 全て死 0.0", 1),
               ("   証明なしの最良 = 復活+観察 (RO): 6.2 — 単独 (18.8 / 17.3) より良いが 0 には届かない", 1),
               "証明があると経験的機構の死削減効果は消える (交互作用 p=0.0003/0.0045) = 完全に冗長化" if ja else
               "With the proof present, empirical mechanisms add nothing (interaction p=0.0003/0.0045)",
               "経験的機構は重ねるほど良い — ただし証明が使えない世界での話" if ja else
               "Empirical mechanisms do compose well — but only in a world without the proof",
           ])

    # 8. 知見 4 (honest)
    _slide(prs, lang,
           "知見 4 (限界): 証明は「前提が正しい時」だけ健全" if ja else
           "Finding 4 (limits): a proof is only sound while its premises hold",
           [
               "検証器が環境 (κ) を 40% 過小推定する状況を作ると、証明は致命域を「安全」と誤認" if ja else
               "If the verifier under-senses the environment by 40%, the certificate admits a lethal band",
               "観測可能な信号 (「許可したのに死んだ」) だけで証明⇔経験を切替える信頼度学習は可能" if ja else
               "A trust controller using only observable evidence (admitted-then-died) can switch proof <-> empirical",
               "ただし得をするのは前提違反が高くつく環境のみ — 安い環境では常時証明が最適 (事前登録の仮説が一部不成立、そのまま報告)" if ja else
               "But hedging pays only where premise violations are costly — a pre-registered hypothesis failed on one substrate and we report it",
               "次の研究課題: 「証明の前提を誰が守るのか」(premise monitoring)" if ja else
               "Open question: who certifies the certifier's premises? (premise monitoring)",
           ],
           note="PoC スケール (スカラー遺伝子, n=20) — 実 LLM スケールは未証明。誇張しない" if ja else
                "PoC scale (scalar genes, n=20) — not yet shown at real-LLM scale. No overclaim")

    # 9. 研究地図
    _slide(prs, lang,
           "研究地図: 最近接研究との違い (一次ソース検証済み)" if ja else
           "Map: how this differs from the nearest work (primary sources verified)",
           [
               "SGM (統計的ゲーデルマシン): 統計的 certificate × 離散タスク性能 — 確率保証・捏造耐性なし" if ja else
               "SGM (Statistical Goedel Machine): statistical certificate over discrete task scores — probabilistic, not fabrication-proof",
               "SEVerA: 論理契約 (Dafny) の sound 検証 — 対象は離散 I/O、連続な内部力学ではない" if ja else
               "SEVerA: sound Dafny-verified logical I/O contracts — discrete artifacts, not continuous internal dynamics",
               "自己改善の PAC 限界 (Wang et al.): 汎化保証 — 安定性・収縮の保証ではない" if ja else
               "PAC limits of self-improvement (Wang et al.): generalization bounds — not stability/contraction",
               "空白 = 「連続な記憶力学を、進化ループ内で、健全な収縮証明でゲートする」— llcore が PoC 占有" if ja else
               "The open quadrant — continuous memory dynamics, gated in-loop by a sound contraction proof — is what llcore occupies at PoC scale",
           ],
           note="3 論文とも本文を一次取得し敵対検証済み。詳細対比は論文 related work 節" if ja else
                "All three checked against primary sources with adversarial review. Full contrasts in the paper's related-work section")

    # 10. 利用条件
    _slide(prs, lang,
           "利用について" if ja else "Using this work",
           [
               "本スライド: CC BY 4.0 — 出典明示で、企業内研修・技術調査・製品検討にそのまま利用可" if ja else
               "These slides: CC BY 4.0 — free for corporate training, tech scouting, product evaluation with attribution",
               "コード/論文 draft: Apache-2.0 + Commercial dual license (github.com/furuse-kazufumi/llcore)" if ja else
               "Code / paper draft: Apache-2.0 + Commercial dual license (github.com/furuse-kazufumi/llcore)",
               "クローズド統合・SLA・NDA 付き相談 = 商用ライセンス窓口: kazufumi@furuse.work" if ja else
               "Closed-source integration, SLA, NDA consulting = commercial licensing: kazufumi@furuse.work",
               "記事 (日本語解説): qiita.com/furuse-kazufumi" if ja else
               "Articles (Japanese): qiita.com/furuse-kazufumi",
               "出典表記例: 「Kazufumi Furuse, llcore (2026), github.com/furuse-kazufumi/llcore」" if ja else
               "Attribution example: \"Kazufumi Furuse, llcore (2026), github.com/furuse-kazufumi/llcore\"",
           ])

    out = _HERE / f"llcore_findings_2026_{lang}.pptx"
    prs.save(str(out))
    return out


def main():
    for lang in ("ja", "en"):
        out = build(lang)
        print(f"wrote {out} ({out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
