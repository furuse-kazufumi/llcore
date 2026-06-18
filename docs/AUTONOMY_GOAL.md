# AUTONOMY GOAL — llcore 自律進行の指示文

> 2026-06-18 古瀬さん指示「確認や承認は不要。どんどん進めて」を standing 化。
> CLAUDE.md「/goal 実行時」規約 + `feedback_max_plan_autonomy` + claude-loop を 1 枚に束ねたもの。
> 再利用・微調整可。`/loop` で定期自走化も可能。

## ゴール
llcore 記事連載 + メモリ効率北極星を自律で前進させる(目安 40 ターンで一旦総括)。

## 規律
- **確認を取らずに進める。「進めますか?」は禁止**。仮定で進める時のみ「X と仮定して進めます」を 1 行宣言。
- 各反復の先頭で **Telegram/メール ingest → loop タスクがあれば最優先**。無ければ計画継続。
- 独立サブタスクは **ultracode workflow で並列**。着手前に 理解 → 一次情報で検証(鵜呑み禁止)。
- **honest disclosure を貫く**(異常に良い結果は勝った気になる前に内訳を疑う / 負け軸を消さない / 規模差を直視)。
- **検証**: 該当時は各ターン末に `rtk pytest` / `ruff` / `mypy` を再実行し exit code と要点を出力。コード変更は必ず自分で verify してから deposit(workflow に盲目 commit させない)。
- **失敗**: 同じアプローチで 3 回失敗したら方針転換を 1 行宣言してから別案へ。
- 記事は **draft → 敵対 honest 監査 → 修正 → deposit** を回し続ける(記事本体は本来 llterm 担当だが、自律指示下では ccr で草稿まで進めて drafts/ に着地)。

## 止まってよいのは「人間専任」のみ(それ以外は止まらない)
- **push / 削除 / submodule 改変 / 課金を伴う外部 API 大量呼び出し / Approval Bus・HITL の迂回**
- **仕様解釈が複数あり成果を大きく左右する分岐**(例: branch A/B/D のどれを主軸にするか等)
- これらは実行せず、選択肢を 1 行で提示して待つ。**再ログイン/再起動要求が出たらループ継続せず人間操作を待つ**(`project_ccr_automation_limits`)。

## 現在のキュー(自律で消化する順)
1. flagship 記事 4 本(S2/S1/A7/B1)を draft→監査→deposit(`docs/articles/drafts/`)。
2. 残り記事ネタ(A 級/B 級, `ARTICLE_IDEA_BANK_2026_06.md`)を順次草稿化。
3. branch B(QAT/LSQ 学習可能 scale で 2bit 制覇再挑戦)= CPU 可能・既存量子化アーク延長。TDD + pytest 検証付き。
4. branch A(メモリ指標 × 進化/NAS 適応度 + verified-plasticity fail-closed gate)= 本丸・CPU で設計+小実証。
- **branch D(GPU 実測)は本環境に GPU 無し=環境ブロック**(人間が GPU 環境を用意するまで保留)。

## 新規割り込み
新規 Telegram/メール指示が来たら最優先で task 粒度で取り込む。
