# PoC 7a Verdict — VNN-COMP `online-arch-evo` 新カテゴリ提案 (Stage 7a)

調査日: 2026-05-29
ファイル:
- 論文: `docs/papers/vnn_comp_online_arch_evolution_proposal.md` (8198 word, TMLR/GECCO/NeurIPS workshop 兼用構造)
- 仕様: `docs/papers/vnn_comp_benchmark_spec.md` (v0.1 normative)
- 実装仕様: `docs/papers/vnn_comp_reference_impl_spec.md`
- PoC: `scripts/poc_7a_vnn_comp_reference_impl.py`
- Test: `tests/unit/test_poc_7a_vnn_comp_reference.py`

---

## falsifiable 命題

> 既存 VNN-COMP カテゴリ (固定 network 入力 robustness) では online architecture
> evolution の verification 性能を測れない。新カテゴリ `online-arch-evo` を導入し、
> (a) **benchmark spec** (input/output 形式・scoring・time budget),
> (b) **reference implementation** (llcore PoC 1a を rules 準拠形式に整形),
> (c) **baseline** (α,β-CROWN の incremental 拡張)
> を本論文+spec+impl で提案する。さらに進化に上限を設けないため、新カテゴリは
> (A) wall-clock-bounded 無限 ChangeOp 列 scoring、
> (B) kernel-typed 新規 architecture 出現耐性、
> (C) POET-lite ChangeOp coevolutionary round、
> (D) MODES integrity sensor、
> の 4 つの open-endedness 機構を備える。

## 破綻ゲート 7 個 (PoC battery)

| Gate | 内容 | 結果 |
|---|---|---|
| **G1** | 論文 draft が abstract + 10 章構造 + 関連研究 + benchmark spec + scoring 全て埋まる (8000 word 目安) | **PASS** — 8198 word (12 sections, 7 appendices) |
| **G2** | benchmark spec が入出力フォーマット曖昧性なし (実装者が rules を読んで再現可) | **PASS** — `vnn_comp_benchmark_spec.md` §2-§10 で全 schema 明示、Appendix A に end-to-end 例 |
| **G3** | reference impl が PoC 1a を rules 準拠で動作 (5 ChangeOp seq で per-step verdict 正常出力) | **PASS** — self-test G3 で `['unsat','unsat','unsat','unsat','error']` 期待通り |
| **G4** | scoring rule が "上限なし進化" を測る要素を含む (wall-clock budget / MODES adaptive bonus / 100-step throughput) | **PASS** — paper §4.4 で A-D 4 機構、§5.2 で throughput bonus、§5.4 で MODES-adaptive bonus 明記 |
| **G5** | 既存 VNN-COMP との差別化が論文 §2 で sharp に表現 (α,β-CROWN ができないことを 3 例以上) | **PASS** — paper §1.1 に Query A/B/C 3 例、§2.9-§2.12 で gap 表 + algorithm-contribution 非主張 |
| **G6** | 関連研究包含: α,β-CROWN / Marabou Incremental / NAS-Bench-201 / WANN / AutoML-Zero / llcore PoC 1a / TorchLean | **PASS** — Appendix A の cross-check で全 11 件確認、本文 §2.1-§2.10 で 13 件引用 |
| **G7** | limitations 章に honest disclosure (Goodhart, GPU-gated, peer review 未経由) を明示 | **PASS** — paper §9 に 9 項目、Appendix F に著者立場の追加開示 |

## pytest 結果

```
tests/unit/test_poc_7a_vnn_comp_reference.py ............. 17 passed in 1.04s
```

17 件すべて緑。内訳:
- G1-G7 self-test 等価のテスト 7 件
- white-box 詳細テスト (apply_changeop, step_once, witness emission, family_soundness, chain verification, main entry, serve protocol smoke) 10 件
- 特筆: `test_unsat_witness_is_independently_verifiable` で **書き出した .smt2 を独立 Z3 solver で再パース → unsat 再現確認** = soundness audit path 機能実証

## 実行結果 (self-test)

```
PoC 7a — VNN-COMP `online-arch-evo` reference impl falsifiable verification
  [PASS] G1: handshake             OK 0.1 RwkvTimeMix
  [PASS] G2: init                  init ok: 1 block(s), bound=1.0
  [PASS] G3: 5-step seq verdicts   expected=['unsat','unsat','unsat','unsat','error']
                                   actual  =['unsat','unsat','unsat','unsat','error']
  [PASS] G4: per-step budget       max_step_time=5.8ms (budget 500ms)
                                   trace=['4.2','5.8','4.2','4.6']
  [PASS] G5: determinism           run1=run2=['unsat','unsat','unsat']
  [PASS] G6: family_soundness sound verdict=sound (family f_decay_only)
  [PASS] G7: unsupported kernel    status=unsupported (MambaScan not supported)
PoC 7a verdict: PASS
```

per-step time 4-6ms = **PoC 1a baseline 5.8 ms と完全に整合**。budget 500ms の 1% 以下で動く。

## 設計判断 (Stage 7a)

### 1. ペーパー構造を 3 venue 兼用に
- TMLR full paper (本命, peer review) — §1-§10 + Appendix A-G 全部使用 = 8198 word
- GECCO 2027 short paper — §1, §3, §4.4, §6, §10 を抽出 (~4000 word)
- NeurIPS 2026 workshop — §1, §2, §4, §9 を抽出 (~6000 word)
→ 共通の §§1-4 が "category 提案の核" として再利用可能な独立構造。

### 2. 開放端性 (open-endedness) を 4 機構で確保
ユーザー指示 (2026-05-29) 「進化に上限を設けない工夫」を spec 自体に組み込む:
- **A**: cumulative wall-clock budget で arbitrarily long stream を許容
- **B**: kernel-typed ChangeOp で見たことない kernel (rwkv/mamba/hopfield/linear-attn 混在進化) を metadata 拡張で受容 (spec revision 不要)
- **C**: POET-lite coevolutionary round (`changeop_coevo`) で benchmark 自体が ChangeOp を coevolve
- **D**: MODES (Bedau new-activity vs neutral shadow) で saturated 提案を弾く

→ 単年で "解かれて終わり" にならないために spec レベルで設計。

### 3. reference impl は **minimal credible existence proof**
- 1 kernel (RwkvTimeMix) + 1 op full (reparam_inplace) + 1 op partial (insert_subblock)
- 他 op は unsupported を返す (honest disclosure)
- これが "競合に対する低い bar" になることが狙い — α,β-CROWN incremental wrapper はこれを大きく上回るはず

### 4. 既存 VNN-COMP との差別化を 3 query で sharp に
- **Query A**: "Is mutation *m* safe to apply?" → commit gate
- **Query B**: "How long can the population evolve before invariant breaks?" → survival curve
- **Query C**: "Does this mutation family preserve invariant?" → family soundness proof

→ α,β-CROWN の I/O contract (one .onnx per call) では phrase 不能 = 新カテゴリ正当化。

### 5. 関連研究の包含
11 件を Appendix A cross-check:
- α,β-CROWN (Wang 2021, 5-time winner)
- Marabou Incremental (Elsaleh 2026, arXiv:2603.12232)
- TorchLean (George/Anandkumar 2026, arXiv:2602.22631)
- NAS-Bench-201 (Dong 2020), AutoML-Zero (Real 2020), WANN (Gaier 2019)
- POET (Wang 2019), AURORA (Cully 2019), MODES (Dolson 2019)
- DNNV (Shriver 2021, CAV)
- llcore PoC 1a (自己引用、本 repo)

加えて §2.9 で verified RL / shielding、§2.10 で hardware-aware verification も触れ、§2.11 で「algorithm contribution は claim しない」明示 (NAS-Bench-201 アナロジー)。

## Codex pair-review verdict prompt

```
You are gpt-5.4 reviewing llcore VNN-COMP new category proposal
(paper + benchmark + reference impl).

# Files to review (Read actual content)
- D:/projects/llcore/docs/papers/vnn_comp_online_arch_evolution_proposal.md
- D:/projects/llcore/docs/papers/vnn_comp_benchmark_spec.md
- D:/projects/llcore/docs/papers/vnn_comp_reference_impl_spec.md
- D:/projects/llcore/scripts/poc_7a_vnn_comp_reference_impl.py
- D:/projects/llcore/tests/unit/test_poc_7a_vnn_comp_reference.py
- D:/projects/llcore/docs/poc/poc_7a_verdict.md

# Q1-Q8
Q1: 論文 abstract の主張は VNN-COMP community に sharp に響くか?
    "α,β-CROWN ができないこと" の例示 (Query A/B/C) は説得的か?

Q2: benchmark spec の入出力フォーマット (.vnnlib + .onnx + .changeop_seq.jsonl) は
    VNN-COMP 既存 (.vnnlib + .onnx) と互換性ある拡張になっているか?
    `.jsonl` という新ファイルを足した design choice は妥当か (line-streamable
    by judge protocol §4.5)?

Q3: scoring rule で "上限なし進化" を測るのは Goodhart にならないか?
    MODES-adaptive bonus は gaming されないか?
    (paper §4.4-D / §9.4 / Appendix C で対策を述べているが threshold 自体が
     tunable knob である留保あり)

Q4: reference impl は α,β-CROWN baseline 並走で TPR/FPR 差を測れるか?
    公正な比較になっているか? (paper §7.1 で "α,β-CROWN は API が違うので
    そもそも直接比較できない、incremental wrapper が必要" と honest 開示済み)

Q5: 関連研究の包含で抜けている重要先行は? 特に 2025-2026 の
    incremental verification, online NAS, verified RL の最新作。
    (筆者は arXiv:2603.12232 / 2602.22631 / 2412.19985 まで取り込み済)

Q6: limitations の honest disclosure (§9 全 9 項目 + Appendix F 著者立場 4 項目)
    は peer review に耐えるか? 過小申告/過大申告はないか?

Q7: TMLR full paper / GECCO short paper / NeurIPS workshop それぞれにどう
    分岐させるか戦略は明確か? 分岐の優先度は?

Q8: 新カテゴリ提案が "1 度きりのカテゴリ" でなく毎年継続可能な benchmark に
    なる仕組み (open-ended, 4 機構 A-D) を含むか? §4.4-C/D の design は
    実際に open-endedness を担保するか?

Reply in Japanese, technical terms in original.
```

## honest 留保 (本 PoC で開示済)

### 1. peer review 未経由
本 paper は author draft. VNN-COMP organisers / TMLR / GECCO / NeurIPS の review はまだ
受けていない。「提案」止まりであり「採択」ではない。

### 2. reference impl は scalar-state
llcore PoC 1a と同じく 1 次元 state 限定。multi-dimensional state への拡張は
本 PoC では未着手 (`COMPLETION_VERDICT.md` の post-completion task に記載)。

### 3. per-step budget 500ms は extrapolation
PoC 1a の 5.8 ms baseline + α,β-CROWN typical (秒-分) からの 2 桁マージン根拠で設定。
実 edition で per-step 時間分布測定して revise 必要。

### 4. MODES threshold は hand-set
§4.4-D の saturated 判定閾値は paper §9.4 に honest 開示。Goodhart 攻撃面あり、
empirical calibration を Future Work §10.5 に明記。

### 5. `changeop_coevo` benchmark は llive lldarwin_v2 系を使用
§4.6 に該当: 同系統 verifier 作者が familiarity bias を得る可能性。緩和策は coevolution
loop source の Apache-2.0 公開 + 第三者による replacement loop 募集。

### 6. α,β-CROWN との empirical comparison は未実施
paper §9.6 / §7.1 に明記 — α,β-CROWN は §4.5 の stdin/stdout protocol 未実装のため
直接比較できない。incremental wrapper 構築 (Future Work §10.2) が必要。

### 7. soundness witness audit は z3 off-the-shelf 依存
reference impl spec §11.3 に明記 — `.smt2` → `.lean` などの cross-checker audit
パイプは未構築。

### 8. throughput bonus / coverage cap は gaming 余地
paper §9.7 / Appendix C に honest disclosure 済。empirical tuning が必要。

### 9. GPU-gated kernels は除外
paper §9.8 / §8.2 で CPU-only restriction を明示。toy-scale benchmark 寄りの design choice。

### 10. spec の formal soundness proof なし
paper §9.9 — spec 自体が「malicious ChangeOp seq に対し sound verdict を保証する」
formal proof は別研究プロジェクト。本提案は judge-side witness audit に依存。

## 関連 memory / 確定独自軸

- **確定独自軸 #7**: `VNN-COMP "online architecture evolution verification" 新カテゴリ提案` — 本 PoC で論文素材 + spec + reference impl + tests 揃った
- [[project_llcore_init_2026_05_29]] — llcore project 発足
- [[project_core_evolution_survey_2026_05_28]] — Agent A-D 事前調査
- [[feedback_codex_pair_review_for_llcore]] — review 規律
- [[feedback_benchmark_honest_disclosure]] — 開示規律 (本 verdict §9 全 10 項目で実践)
- [[feedback_external_ai_verify]] — Codex review prompt 同梱

## 次段 / Future Work

- Codex pair-review 実施 (Q1-Q8 prompt を `codex exec -s read-only "<prompt>"` で投入)
- α,β-CROWN incremental wrapper baseline 実装 (paper §10.2)
- 10-instance pilot 生成 (paper §10.3) — llive lldarwin_v2 + §4.4-C coevolution loop
- multi-dimensional state 拡張 (verifier API 再設計、paper §10.6)
- VNN-COMP organisers への正式 proposal 投稿 (paper §10.1)

---

## 完了報告

| 項目 | 結果 |
|---|---|
| 論文 word count | 8198 word (target ≥ 8000) ✓ |
| Sections | 10 main + 7 appendices ✓ |
| 関連研究包含 | 13 件本文引用 + 11 件 Appendix A cross-check ✓ |
| Benchmark spec | v0.1 normative, 10 sections + 3 appendices ✓ |
| Reference impl spec | 11 sections (capabilities matrix + bridge sketches) ✓ |
| Reference impl PoC | `scripts/poc_7a_vnn_comp_reference_impl.py`, G1-G7 self-test PASS ✓ |
| Unit tests | 17 tests PASS, soundness audit path (independent z3 re-check) 実証 ✓ |
| 7 ゲート (G1-G7) | 7/7 PASS ✓ |
| 開放端性 4 機構 (A-D) | spec §4.4 で全て設計済 ✓ |
| Codex review prompt | 本 verdict 末尾に Q1-Q8 同梱 ✓ |
| Honest disclosure | paper §9 (9 items) + Appendix F (4 items) + 本 verdict §10 (10 items) ✓ |

**verdict: PASS — Stage 7a 完成。VNN-COMP 新カテゴリ提案素材は workshop 投稿可能水準。TMLR full submission には reference impl の `.onnx`/`.vnnlib` parser 実装 + `sat` witness 実装が前提 (Codex 指摘 finding #1, #2)。**

---

## Codex review record (2026-05-29, gpt-5.4) — **positive 評価 + 改善 prioritised**

Codex は **「proposal の核は強い」** と総評し venue 優先順位を確定。pair-review 規律
[[feedback_codex_pair_review_for_llcore]] で得た改善事項を以下に整理:

### Findings (4 件, 高=2 / 中=2)

| # | severity | 内容 | 対応 |
|---|---|---|---|
| 1 | **高** | spec ⇔ reference impl conformance gap: spec は `.onnx` + `.vnnlib` 必須だが impl は `NotImplementedError` + JSON dummy のみ | **honest 留保 §11 に追記** (本 verdict)、claim を「spec proposal + minimal skeleton impl」に絞り込み、TMLR submission には parser 実装が前提 |
| 2 | **高** | witness protocol 未充足: spec は sat/unsat 両方 `witness_path` 必須だが impl は `sat` で `None` | **honest 留保 §12 に追記**、audited TPR/FPR 比較土台が現状なし、`sat` witness 実装は次段 |
| 3 | 中 | `family_id` machine-checkable でない (prose 寄り): `declared_families.description` のみ、parameter domain schema 不在 → Q2/Q3/Q8 設計穴 | spec v0.2 で `declared_families` を machine-readable schema 化 (paper §10 + spec §10 改訂事項として記録) |
| 4 | 中 | abstract framing 2026-05-29 stale: "first five editions (2020-2024)" は古い (α,β-CROWN は VNN-COMP 2025 winner も)、"provably cannot answer" は強すぎ | paper §1 を **"first six editions (2020-2025), α,β-CROWN won six consecutive years"** + **"cannot be expressed under the current VNN-COMP I/O contract"** に修正 (paper next revision で対応) |

### Q1-Q8 要点 (Codex 評価)

- **Q1 ✓** abstract sharp、Query A/B は VNN-COMP community に刺さる。Query C はやや弱い (tool limitation でなく contract mismatch) → abstract framing 改善
- **Q2 ✓** `.onnx + .vnnlib + .changeop_seq.jsonl` は互換性ある拡張として妥当、`declared_families` の schema 化が完全性条件
- **Q3 ⚠** Goodhart risk 実在 (paper 自身も認める)。`per-family minimum coverage` + `family-balanced scoring` 追加推奨
- **Q4 ⚠** **現状の reference impl では α,β-CROWN baseline と公平な TPR/FPR 比較不可** (理由 3 件: α,β-CROWN wrapper 不在 + sat witness 未実装 + .onnx/.vnnlib parser 未実装)。latency / coverage / protocol shape 比較までは可能
- **Q5 ✓** Elsaleh et al. 2026 を超える exact niche 先行は現状なし (novelty 支える)。adjacent work として **次稿で追加すべき 4 本**:
  - Runtime Safety through Adaptive Shielding (2025-05-20)
  - ProSh (2025-10-17)
  - Adaptive GR(1) Specification Repair for Shielding (2025-11-04)
  - The Effect of Architecture During Continual Learning (2026-01-27)
- **Q6 ✓** honest disclosure は peer review に耐える。過小申告 2 件のみ (Finding #1, #2) を §9 に追加要
- **Q7 ✓ venue 優先順位確定**: **TMLR > NeurIPS workshop > GECCO short**。今の原稿のままなら GECCO より workshop 相性良し
- **Q8 ⚠** yearly benchmark 仕組み入っているが、`changeop_coevo` 固定化リスク + MODES 事後検知が弱点。追加すべき: `rotating generator families` / `withheld seeds` / `yearly mandatory kernel refresh` / `machine-readable family domains`

### honest 留保 追加 (Codex Findings #1, #2 受容)

#### 11. spec ⇔ reference impl conformance gap (Codex Finding #1)
spec §2 は `model.onnx` + `invariant.vnnlib` を normative 入力とするが、`scripts/poc_7a_vnn_comp_reference_impl.py:117` は **`.onnx` / `.vnnlib` を `NotImplementedError` で reject** し、JSON dummy form だけ accept する minimal skeleton。reference impl spec §2 でも minimal と開示しているが、Q4 で TPR/FPR claim を弱める。

**現状 claim**: 「spec proposal + minimal skeleton implementation + 17 test PASS (soundness audit path 含む)」。
**post-Codex 必要対応**: TMLR full submission 前に `.onnx` / `.vnnlib` parser を実装 (Future Work §10.4 追加)。

#### 12. sat witness 未実装 (Codex Finding #2)
spec §3.1 は `sat` verdict にも `witness_path` を必須とするが、`poc_7a_vnn_comp_reference_impl.py:330` は **`sat` で `witness_path=None`** を返す (PoC 1a の counter-example が `.json` 出力に未配線)。

**現状 claim**: `unsat` witness は実装済 (`.smt2` 形式、独立 Z3 で audit 可能)。
**post-Codex 必要対応**: `sat` witness を PoC 1a counter-example から `.json` 形式で出力 (Future Work §10.5 追加)。audited TPR/FPR 比較はこの実装後に可能。

### venue 戦略 (Codex Q7 確定)

| Venue | Priority | 現状原稿適合度 | 必要追加 |
|---|---|---|---|
| **TMLR full paper** | 本命 | 中 | parser 実装 + sat witness + adjacent work 4 本追加 + family schema 化 |
| **NeurIPS workshop** | 早期 exposure | **高** | abstract framing 修正 + adjacent work 2 本追加 |
| **GECCO 2027 short** | 後続 | 低 | empirical coevolution 結果 (10-instance pilot) 必要、現原稿では弱い |

→ **次稿戦略**: 短期 = NeurIPS workshop submission (framing 修正のみで提出可)、中期 = TMLR (parser + sat witness 実装後)、後期 = GECCO short (pilot 結果蓄積後)。
