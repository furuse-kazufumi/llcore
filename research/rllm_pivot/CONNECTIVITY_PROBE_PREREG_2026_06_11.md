# 事前登録 (ドラフト): 会話アノテーション → 連結性 probe (2026-06-11, 未実施)

**Status**: 事前登録ドラフト — 実装・実行前。数値閾値は実装前に確定し、以後変更しない。

## 動機 (ユーザー戦略 2026-06-11)

「会話ができないとアノテーションが無く、物事の連結性や世界モデルまでの道筋がまったく得られない」。
chat レイヤ (frozen SmolLM2) は製品ではなく**進化型のベース**であり、会話構造 (ターン境界・話題ラベル・
照応イベント) は**無償のアノテーション**である。本 probe は「検証付き (certified ρ<1) sidecar アダプタの
状態が、会話で与えられた事実の**連結性**をどれだけ保持するか」を測る第一歩。

## 仮説

- **H1 (連結性保持)**: certified アダプタ (cert_two admit, ρ<1) の照応ターン時点の状態 s から、
  数ターン前に会話で与えられた事実 (名前 ∈ 候補 K 個) が ridge readout (`llcore.fitness.ridge_readout`)
  で chance (1/K) を有意に上回って復号できる。
- **H2 (アノテーション差分)**: 照応イベント付近の状態遷移は、話題転換ターンの遷移と分離可能
  (状態空間で stage ラベルが線形分離可能か)。

## 設計骨子 (実装前に確定させる項目)

- 会話セット: 名前/居住地/話題を系統的に変えた M 会話 (M≥20, 全て実 SmolLM2-360M 生成、verbatim 保存)。
- アダプタ: phase2_demo_verified_chat と同一基質 (n=6, cert_two-admitted 固定 gene; 進化なし=probe を純化)。
- 対照 (必須):
  (a) **無再帰対照**: 現ターン hidden の同次元射影のみ (アダプタの寄与を分離)
  (b) **履歴シャッフル対照**: ターン順序を壊した hidden 列 (連結性が会話構造由来であることの検証)
- 判定: H1 = readout 精度が (a)(b) の双方を pre-registered margin で上回る。NULL/NEGATIVE も全強度で報告。

## honest 制約 (登録時点で宣言)

- これは **capability 主張ではない** (Phase 2: capability NEGATIVE 確定)。内部表現の probe であり、
  「アダプタが会話を良くする」とは主張しない。
- n=6 の small-n 制約下 (Phase −1/1: verified は small-n per-component 限定) — 復号容量は小さく、
  NULL が十分あり得る。NULL でも「small-n verified 状態の情報容量の第一級測定」として報告する。
- ridge readout は線形 probe — 非線形に存在する情報は見逃す (下界測定)。

## 関連

- memory: project_llcore_conversation_annotation_worldmodel / project_llcore_evolvable_llm_replan_2026_06_09
- 先行: phase0_framework_harness.py (echo-state 測定) / phase2_demo_verified_chat.py (実会話×gate)
