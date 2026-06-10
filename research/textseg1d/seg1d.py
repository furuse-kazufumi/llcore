# SPDX-License-Identifier: Apache-2.0
"""1D-SegFormer PoC — SegFormer (2D 画像セグメンテーション) の構造を 1D 文字列に応用。

ユーザー着想 (2026-06-11): 「SegFormer のような 2D 画像のセグメンテーション機能を
1D テキストのセグメンテーションに応用」して、日本語の単語境界 (= 単語を分割する場所) を
ある程度正確に特定する。

SegFormer (Xie+ 2021, NeurIPS) の差別化要素を 1D へ移植:
  - **階層エンコーダ (Mix Transformer)**: overlap patch merging で解像度を段階的に下げ
    マルチスケール特徴を作る → 1D では overlap conv で文字列を 1/2, 1/4, 1/8 に縮約。
  - **Efficient Self-Attention (sequence reduction)**: K,V を係数 R で縮約し O(L²)→O(L²/R)。
  - **Mix-FFN**: FFN 内に depthwise conv を入れ「位置符号を使わず」局所位置を漏らす
    (SegFormer の中心主張: 位置符号は解像度可変に弱い)。1D でも positional embedding 無し。
  - **All-MLP decoder**: 各スケールを MLP で揃え結合 → per-token 分類 (境界 B / 内部 I)。

タスク定義: per-character の **境界ラベル** (その文字が単語の先頭なら B、続きなら I) を予測。
silver 教師 = janome (Apache-2.0) のトークン境界。これは「正解」ではなく便宜的な教師
(janome 自身が誤る) であることを開示。評価 = boundary-F1 (文字 i と i+1 の間に境界があるか)。

honest 留保:
  - これは **PoC**。SegFormer 構造の 1D 有効性を small データで確認するのが目的。
  - silver 教師 (janome) との一致を測るので「janome 風の分割」を学ぶだけにもなりうる。
    だからこそ **規則ベースライン (文字種遷移)** と比較し、構造が規則を上回るかを見る。
  - torch 必須 (chat/clip と同じ optional 重依存)。CPU で回る小モデル。

依存: torch (optional)。教師: janome (optional, 無ければ合成データにフォールバック)。

使い方::

    py -3.11 research/textseg1d/seg1d.py            # 学習 + 評価 + JSON
    py -3.11 research/textseg1d/seg1d.py --epochs 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------- #
# データ: silver 教師 (janome) + 文字種遷移ベースライン
# --------------------------------------------------------------------------- #
def japanese_corpus() -> list[str]:
    """PoC 用の日本語文 (実会話・一般文・誤字を含む)。"""
    return [
        "私の名前はカズです。",
        "東京に住んでいます。",
        "簡単なパスタ料理を教えてください。",
        "太陽系の惑星を一つ挙げてください。",
        "今日はとても良い天気ですね。",
        "機械学習のモデルを訓練しています。",
        "日本語の単語分割は難しい問題です。",
        "彼女は新しい本を買いました。",
        "コーヒーを飲みながら仕事をします。",
        "明日の会議は午後三時からです。",
        "この料理はとても美味しかったです。",
        "電車が遅れて遅刻しそうになった。",
        "音楽を聴くのが好きです。",
        "週末は友達と映画を見に行く予定です。",
        "プログラミングを勉強し始めました。",
        "桜の花が満開になりました。",
        "彼は毎朝ジョギングをしています。",
        "新しいスマートフォンを購入した。",
        "図書館で参考書を借りてきた。",
        "海の近くのホテルに泊まりました。",
        "りんごとみかんを買ってきてください。",
        "会社の同僚と昼食を食べました。",
        "雨が降りそうなので傘を持って行く。",
        "数学のテストで満点を取りました。",
        # 誤字混入 (頑健性確認用)
        "わたしのなまえはかずでず。",
        "とうきょうにすんでいまs。",
    ]


def char_type(ch: str) -> str:
    """文字種: hira / kata / kanji / latin / digit / other。"""
    o = ord(ch)
    if 0x3040 <= o <= 0x309F:
        return "hira"
    if 0x30A0 <= o <= 0x30FF:
        return "kata"
    if 0x4E00 <= o <= 0x9FFF:
        return "kanji"
    if ch.isascii() and ch.isalpha():
        return "latin"
    if ch.isdigit():
        return "digit"
    return "other"


def baseline_boundaries(text: str) -> list[int]:
    """規則ベースライン: 文字種が変わる位置を境界とする (B=1)。先頭は常に B。"""
    labels = [1]
    for i in range(1, len(text)):
        labels.append(1 if char_type(text[i]) != char_type(text[i - 1]) else 0)
    return labels


def silver_boundaries(text: str, tokenizer) -> list[int]:  # type: ignore[no-untyped-def]
    """janome のトークン境界を per-char ラベル (B=1=単語先頭) に変換。"""
    labels = [0] * len(text)
    pos = 0
    for tok in tokenizer.tokenize(text):
        surface = tok.surface
        if not surface:
            continue
        if pos < len(text):
            labels[pos] = 1
        pos += len(surface)
    if labels:
        labels[0] = 1
    return labels


def load_dataset() -> tuple[list[str], list[list[int]], bool]:
    """(texts, silver_labels, used_janome) を返す。janome 無ければ規則を教師にフォールバック。"""
    texts = japanese_corpus()
    try:
        from janome.tokenizer import Tokenizer

        tok = Tokenizer()
        labels = [silver_boundaries(t, tok) for t in texts]
        return texts, labels, True
    except Exception:
        labels = [baseline_boundaries(t) for t in texts]
        return texts, labels, False


# --------------------------------------------------------------------------- #
# モデル: 1D Mix-Transformer (SegFormer 構造の移植)
# --------------------------------------------------------------------------- #
def build_model(vocab_size: int):  # type: ignore[no-untyped-def]
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class OverlapPatchMerge(nn.Module):
        """SegFormer の overlap patch embedding の 1D 版 (stride で系列を縮約)。"""

        def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int) -> None:
            super().__init__()
            self.proj = nn.Conv1d(in_ch, out_ch, kernel, stride=stride, padding=kernel // 2)
            self.norm = nn.LayerNorm(out_ch)

        def forward(self, x):  # x: (B, C, L) -> (B, L', C_out), L'
            x = self.proj(x)
            x = x.transpose(1, 2)  # (B, L', C)
            return self.norm(x)

    class EfficientAttention(nn.Module):
        """Sequence-reduction self-attention (K,V を係数 R で縮約)。"""

        def __init__(self, dim: int, heads: int, sr_ratio: int) -> None:
            super().__init__()
            self.heads = heads
            self.scale = (dim // heads) ** -0.5
            self.q = nn.Linear(dim, dim)
            self.kv = nn.Linear(dim, dim * 2)
            self.proj = nn.Linear(dim, dim)
            self.sr = (
                nn.Conv1d(dim, dim, sr_ratio, stride=sr_ratio) if sr_ratio > 1 else None
            )
            self.sr_norm = nn.LayerNorm(dim) if sr_ratio > 1 else None

        def forward(self, x):  # (B, L, C)
            b, n, c = x.shape
            h = self.heads
            q = self.q(x).reshape(b, n, h, c // h).permute(0, 2, 1, 3)
            kv_in = x
            if self.sr is not None:
                kv_in = self.sr(x.transpose(1, 2)).transpose(1, 2)
                kv_in = self.sr_norm(kv_in)
            kv = self.kv(kv_in).reshape(b, -1, 2, h, c // h).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(b, n, c)
            return self.proj(out)

    class MixFFN(nn.Module):
        """位置符号の代わりに depthwise conv で局所位置を漏らす FFN (SegFormer の核)。"""

        def __init__(self, dim: int, hidden: int) -> None:
            super().__init__()
            self.fc1 = nn.Linear(dim, hidden)
            self.dw = nn.Conv1d(hidden, hidden, 3, padding=1, groups=hidden)
            self.fc2 = nn.Linear(hidden, dim)

        def forward(self, x):  # (B, L, C)
            x = self.fc1(x)
            x = self.dw(x.transpose(1, 2)).transpose(1, 2)
            x = F.gelu(x)
            return self.fc2(x)

    class Block(nn.Module):
        def __init__(self, dim: int, heads: int, sr_ratio: int) -> None:
            super().__init__()
            self.n1 = nn.LayerNorm(dim)
            self.attn = EfficientAttention(dim, heads, sr_ratio)
            self.n2 = nn.LayerNorm(dim)
            self.ffn = MixFFN(dim, dim * 4)

        def forward(self, x):
            x = x + self.attn(self.n1(x))
            x = x + self.ffn(self.n2(x))
            return x

    class Seg1DFormer(nn.Module):
        """階層 1D MiT エンコーダ + all-MLP デコーダ → per-char 境界 2 クラス。"""

        def __init__(self, vocab: int, dims=(64, 128, 160), heads=(1, 2, 4),
                     srs=(4, 2, 1), strides=(2, 2, 2)) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, dims[0])
            self.stages = nn.ModuleList()
            in_ch = dims[0]
            for i, d in enumerate(dims):
                merge = OverlapPatchMerge(in_ch, d, kernel=2 * strides[i] + 1, stride=strides[i])
                block = Block(d, heads[i], srs[i])
                self.stages.append(nn.ModuleList([merge, block]))
                in_ch = d
            self.decode = nn.ModuleList([nn.Linear(d, 64) for d in dims])
            self.classifier = nn.Sequential(
                nn.Linear(64 * len(dims), 64), nn.GELU(), nn.Linear(64, 2)
            )

        def forward(self, ids):  # ids: (B, L)
            L = ids.shape[1]
            x = self.embed(ids).transpose(1, 2)  # (B, C, L)
            feats = []
            for merge, block in self.stages:
                h = merge(x)              # (B, L', C)
                h = block(h)
                feats.append(h)
                x = h.transpose(1, 2)     # (B, C, L') for next stage
            # all-MLP: 各スケールを原長 L へ補間して結合
            ups = []
            for f, dec in zip(feats, self.decode):
                u = dec(f).transpose(1, 2)                       # (B, 64, L')
                u = F.interpolate(u, size=L, mode="linear", align_corners=False)
                ups.append(u.transpose(1, 2))                    # (B, L, 64)
            cat = torch.cat(ups, dim=-1)
            return self.classifier(cat)  # (B, L, 2)

    return Seg1DFormer(vocab_size)


# --------------------------------------------------------------------------- #
def boundary_f1(pred: list[int], gold: list[int]) -> tuple[float, float, float]:
    """境界 (B=1) の P/R/F1。先頭位置は両者 B のため評価から除外 (自明)。"""
    tp = sum(1 for i in range(1, len(gold)) if pred[i] == 1 and gold[i] == 1)
    fp = sum(1 for i in range(1, len(gold)) if pred[i] == 1 and gold[i] == 0)
    fn = sum(1 for i in range(1, len(gold)) if pred[i] == 0 and gold[i] == 1)
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260611)
    parser.add_argument("--out", type=Path, default=HERE / "seg1d_results.json")
    args = parser.parse_args()

    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("error: torch が必要です (pip install torch)", file=sys.stderr)
        return 2

    torch.manual_seed(args.seed)
    texts, silver, used_janome = load_dataset()
    print(f"教師: {'janome (silver)' if used_janome else '規則フォールバック'}  "
          f"文数={len(texts)}", flush=True)

    # 文字語彙
    chars = sorted({c for t in texts for c in t})
    stoi = {c: i + 1 for i, c in enumerate(chars)}  # 0 = pad
    vocab = len(stoi) + 1

    # 簡易 split (最後の 6 文を test)
    idx = list(range(len(texts)))
    rng = __import__("random").Random(args.seed)
    rng.shuffle(idx)
    test_idx = set(idx[:6])
    train_idx = [i for i in idx if i not in test_idx]

    model = build_model(vocab)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    lossfn = nn.CrossEntropyLoss()

    def encode(i: int):
        ids = torch.tensor([[stoi[c] for c in texts[i]]], dtype=torch.long)
        lab = torch.tensor([silver[i]], dtype=torch.long)
        return ids, lab

    model.train()
    t0 = time.time()
    for ep in range(args.epochs):
        rng.shuffle(train_idx)
        tot = 0.0
        for i in train_idx:
            ids, lab = encode(i)
            logits = model(ids)
            loss = lossfn(logits.reshape(-1, 2), lab.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"  epoch {ep+1:3d}  train_loss={tot/len(train_idx):.4f}", flush=True)

    # 評価: model vs 規則ベースライン (test set, silver を gold とみなす)
    model.eval()
    cases = []
    f_model_sum = f_base_sum = 0.0
    with torch.no_grad():
        for i in sorted(test_idx):
            ids, _ = encode(i)
            pred = model(ids)[0].argmax(-1).tolist()
            pred[0] = 1
            base = baseline_boundaries(texts[i])
            gold = silver[i]
            _, _, fm = boundary_f1(pred, gold)
            _, _, fb = boundary_f1(base, gold)
            f_model_sum += fm
            f_base_sum += fb
            seg_model = render_segments(texts[i], pred)
            seg_gold = render_segments(texts[i], gold)
            cases.append({"text": texts[i], "model_f1": round(fm, 3), "baseline_f1": round(fb, 3),
                          "model_seg": seg_model, "gold_seg": seg_gold})
            print(f"\n  text : {texts[i]}", flush=True)
            print(f"  gold : {seg_gold}", flush=True)
            print(f"  model: {seg_model}  (F1={fm:.2f} vs 規則 {fb:.2f})", flush=True)

    n = len(test_idx)
    res = {
        "teacher": "janome" if used_janome else "rule_fallback",
        "n_texts": len(texts), "n_test": n, "epochs": args.epochs,
        "train_seconds": round(time.time() - t0, 1),
        "mean_boundary_f1_model": round(f_model_sum / n, 4),
        "mean_boundary_f1_baseline_chartype": round(f_base_sum / n, 4),
        "verdict": ("1D-SegFormer 構造が文字種規則ベースラインを上回る"
                    if f_model_sum > f_base_sum else
                    "規則ベースラインを上回らず (PoC データ小, 過学習/容量要調整)"),
        "honest_notes": [
            "silver 教師 (janome) との一致を測る = janome 風分割を学ぶ側面あり",
            "PoC データ小 (26 文) — 汎化主張はしない、構造の有効性確認が目的",
            "規則ベースライン (文字種遷移) との比較が本質的な比較軸",
        ],
        "cases": cases,
    }
    args.out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n=== verdict ===", flush=True)
    print(f"  model boundary-F1   = {res['mean_boundary_f1_model']:.3f}", flush=True)
    print(f"  規則 baseline F1     = {res['mean_boundary_f1_baseline_chartype']:.3f}", flush=True)
    print(f"  → {res['verdict']}", flush=True)
    print(f"results: {args.out}", flush=True)
    return 0


def render_segments(text: str, labels: list[int]) -> str:
    """境界ラベルで text に区切り | を入れて可視化。"""
    out = []
    for i, ch in enumerate(text):
        if i > 0 and labels[i] == 1:
            out.append("|")
        out.append(ch)
    return "".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
