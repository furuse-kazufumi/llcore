# PoC 1b Verdict — Z3 verifier 状態方向 Lipschitz contraction invariant (Stage 1b)

調査日: 2026-06-01
Stage: 確定独自軸 #6 — Lipschitz/Hurwitz invariants の進化ループ SMT gate embedding
ファイル:
- 実装: `src/llcore/verifier/invariants.py` (新規 4 シンボル: `LipschitzResult` / `verify_lipschitz_contraction` / `empirical_lipschitz` / `_lipschitz_upper_bound`、純 ADDITIVE)
- export: `src/llcore/verifier/__init__.py` (+4/-0)
- PoC: `scripts/poc_1b_lipschitz_invariant.py` (BG1-BG5 gate run、exit 0)
- Test: `tests/unit/test_poc_1b_lipschitz.py` (27 passed)

---

## 1. falsifiable 命題と Z3 encoding

### 命題 (falsifiable)

> RWKV-style 状態更新写像 `s' = decay·s + (1−decay)·tanh(mix·x + gate_str·s)` は各座標独立
> (対角写像) で、座標ごとのヤコビ `∂s'/∂s = decay + (1−decay)·gate_str·t`
> (`t = sech²(pre) = 1−tanh²(pre) ∈ (0,1]`) を持つ。状態方向 Lipschitz 定数を
> `L = sup_{|s|≤1,|x|≤1} |∂s'/∂s|` と定義する。clip 済み具体 gene `g=(decay,mix,gate_str)`
> について Z3 が `∃ t∈[0,1]. |decay+(1−decay)·gate_str·t| ≥ 1` を **unsat** と判定すれば、
> その gene は `L<1` (状態方向 contraction certified)。これは sup-norm global contraction を
> 含意し、Banach により一意固定点と `|s|≤1` 有界 (state_norm 整合) を保証する。
>
> **反証形**: certified gene が numpy シミュレーション (L=200, dim=8, 初期状態のみ違える 2 軌跡)
> で軌跡差ノルムを縮小させない (`gapN/gap0` が 1 付近のまま) ことが一度でも起これば命題は偽。

### Z3 encoding (free-variable abstraction)

```
変数: t = z3.Real("t")            # per-gene online gate、t 1 変数のみ
定数: d = RealVal(decay)          # clip 済み値を有理数で焼き込む
      g = RealVal(gate_str)
制約: (1) 0 <= t <= 1             # sech²(pre) の真域 (0,1] の over-approx
      (2) J = d + (1-d)*g*t       # t について一次 → 純線形 = 高速
      (3) solver.add(Or(J >= 1, J <= -1))   # |J|>=1 の違反集合 (厳密否定)
判定: unsat → sup|J|<1 = L<1 certified (ok=True, solver_status="unsat")
      sat   → L>=1 になり得る点が存在 = 保守的 reject (ok=False, solver_status="sat")
      unknown/timeout → fail-closed で reject
      z3 不在 → contraction=None + used_z3=False (assumed, fail-closed 扱い規約)
```

`x` と `mix` は J に t を介してのみ入るため t を自由化した時点で式から消え、ソルバは t 1 変数の
線形可行性問題に縮約 (実測 median 2.27ms/gene)。`verify_state_norm_invariant` と同じ
free-variable abstraction の流儀を踏襲し、既存 `verify_gene_safe` は無改変 (純 ADDITIVE)。

導出は連鎖律で `∂s'/∂s = decay + (1−decay)·sech²(pre)·gate_str`、有限差分 (h=1e-6, 20000 サンプル)
との最大誤差 1.35e-10 で数値確認済み。

---

## 2. 破綻ゲート (BG1-BG5) 結果

| Gate | 内容 | 結果 | 数値 |
|---|---|---|---|
| **BG1-timeout** | Z3 per-gene < 1s (線形縮約のはずが式肥大で重くなる退行検知) | **PASS** | 200 genes: median=2.27ms, max=11.81ms, total=495.2ms (要件 <1000ms/gene を 2 桁クリア) |
| **BG2-rejectrate** | reject 率が 0%(無意味)/100%(使えない)に退化しない | **PASS** | reject 71/200 = 35.5% (0% < rate < 100%、seed 感度 {1:0.330,7:0.327,13:0.303,42:0.310,99:0.293,2026:0.367} で全て圏内) |
| **BG3-sim-contraction** | certified gene が numpy sim で実際に縮小 (命題の直接反証チェック) | **PASS** | certified 4/4 全て ratio<1 (実測 [0.0000,0.0000,0.0001,0.0000])。非certified d=0.9,g=2.0 は ratio=1.9311 で発散 reject = 弁別力の証拠 |
| **BG4-regression** | 既存 verifier テストが本追加で 1 件も fail しない (semver 互換) | **PASS** | `test_poc_1a_z3_invariant.py` 10 passed 維持。full suite 248 passed (baseline 221 + 新規 27) |
| **BG5-statenorm** | contraction certified なのに `|s|≤1` 有界が破れる矛盾がない | **PASS** | certified 4/4 全て `verify_gene_safe`(state_norm) ok=True + long-sim max\|state\|=0.9827≤1 (contraction⟹state_norm 構造的整合) |

PoC script `poc_1b_lipschitz_invariant.py` は exit 0 で全 5 gate PASS。pytest:
`248 passed in 34.88s` (新規 `test_poc_1b_lipschitz.py` 単体 27 passed)。

### before / after 計測 (honest)

- BG1: before(ゲート無) — N/A。after(本ゲート導入後) median 2.27ms / max 11.81ms。
- BG4: baseline `test_poc_1a_z3_invariant.py` = 10 passed (取得済) → 追加後も 10 passed 維持。
  full suite 221 → 248 (純増 27、削除 0)。
- タスク記載の「既存 verifier 18 tests」は本 repo に実在せず、verifier 専用は
  `test_poc_1a_z3_invariant.py` の 10 件のみ (verifier import 系を合算しても 84)。
  BG4 が honest に自己開示済 (changeop/refinement/curriculum は本 repo に無い件数と推定)。

---

## 3. 敵対検証 4 レンズ結果

すべて read-only で実コード (z3 4.16.0 / py -3.11) を直接実行。**surviving high refutation = なし**。

| レンズ | refuted | severity | 要点 |
|---|---|---|---|
| **soundness** | No | none | 導出再確認 (有限差分 vs 解析 max誤差 1.25e-10)、Z3 vs 閉形式 3000 gene で mismatch 0、empirical クロスチェック ~16000 gene で「emp > L_upper_bound」違反 0 かつ「certified だが emp_L≥1」0、真 Lipschitz grid (sech² を 400×400 直接評価) 8000 gene で「certified だが true grid L≥1」=unsound 証明書 0、FP 境界 hunt (\|J(t=1)\| を 1e-10 下回る gene 3000 件) でも certified 全件 float64 で \|J\|<1 維持。**unsat⟹L<1 は健全** |
| **non_triviality** | No | none | reject 率 seed 安定 0.29-0.37 (退化なし)。state_norm は clip 範囲 500/500 全 admit だがそのうち 147/500 (29.4%) を contraction が reject。reject 152 件を sim 分類 → **99 件は state_norm を通過しつつ実際に軌跡が発散** (ratio 1.43-1.95)、残り 53 件が設計通りの保守的 false-reject。単なる水増しでなく実質弁別力 |
| **empirical** | No | none | 400 gene sweep で certified=273/rejected=127、CERT VIOLATIONS (certified なのに max_ratio≥1) = 0、REJECT-ALWAYS-CONTRACT = 0。5000 gene over-approx クロスチェック worst (empL − L_upper) = +8.4e-11 (float ノイズ)、certified 中の最大 empirical L = 0.999901 < 1。境界 g=0.999→cert/g=1.000→reject、decay=1.0→reject で L=1 ちょうど反転。map が真に対角と確認 |
| **regression** | No | **low** | full 248 passed / 0 failed、221+27 で純加。semver 互換 (`solver_status` は default 付き末尾 field で positional caller に無影響)。**low 2 件は文書上の不正確さのみ** (下記、機能回帰ではない) |

### regression レンズの low 指摘 (文書精度、reject に至らず)

1. IMPLEMENT caveats の「既存 verify_gene_safe は 1 文字も変更せず」は **1a 原 baseline (c966ded)
   に対しては不正確**。`verify_gene_safe` / `InvariantResult` は `solver_status` 追加で改変済。
   ただしこの改変は **本 1b タスクより前**の 05-31 commit (d4596c7 / d0d68a8) によるもので、
   **本 1b タスク自体の invariants.py 内追加は純 ADDITIVE** (主張は守られている)。「既存関数は
   一切無改変」という絶対表現が repo 履歴全体では不正確、という限定的な記録上の注意。
2. タスク文の「既存 verifier 18 tests」は実在しない (§2 BG4 と同旨、BG4 で自己開示済)。

→ いずれも命題 (Z3 contraction 証明) の健全性にも回帰防止にも影響せず、判定に影響しない。

---

## 4. state_norm 有界との distinctness

`state_norm` 有界 (`|s|≤1`) は clip 範囲の全 gene で**無条件成立** (`decay·s+(1−decay)·tanh`、
`|tanh|≤1` で `|s'|≤decay+(1−decay)=1` の凸結合)。よって既存 `verify_state_norm_invariant` は
「clip された gene なら常に unsat=admit」で gene 間を弁別しない**受動的不変量**。

対して contraction (`L<1`) は gene を**能動的に弁別**する。両者の差は実コードで定量確認:

- `verify_gene_safe`(state_norm) は clip 範囲 **500/500 全 admit**。
- そのうち **147/500 (29.4%) を contraction が reject**。
- reject 152 件を numpy sim 分類: **99 件は state_norm 通過しつつ軌跡が真に発散** (ratio 1.43-1.95、
  例 d=0.402,m=−0.807,g=1.871 で ratio=1.954)。残り 53 件は honest disclosure 通りの保守的 false-reject。

DESIGN の具体反例も再現: `gene=(decay=0, mix 任意, gate_str=2.0)` は `|s'|=|tanh(2s)|≤1` で
state_norm admit、しかし `J=2t`、`sup_{t∈[0,1]}|2t|=2≥1` で Z3 sat=reject (L 最大 2)。
admit 境界は state_norm が `decay∈[0,1]` 全域なのに対し contraction は `decay+(1−decay)·|gate_str|<1`
という真に狭い領域 (`gate_str=2` では `decay≥1` でしか満たせず実質排除)。

→ **contraction は state_norm の真部分集合を切り出す質的に強い不変量** (BG5 で contraction admit
かつ state_norm reject の矛盾は 0/500、構造的整合も確認)。

---

## 5. Hurwitz の付加価値

Hurwitz (離散系では固定点 `s*` での局所安定 `|∂s'/∂s(s*)|<1`) は本 global contraction ゲートに
**包含され、付加価値は限定的**。

- global contraction `L=sup_{|s|≤1}|J(s)|<1` は全点で `|J|<1` を要求するので、自動的に
  `|J(s*)|<1` (局所安定) を含意し、かつ Banach で固定点の一意存在まで保証する。
  → 本ゲートは Hurwitz より **strictly stronger**。
- Hurwitz を別途入れる唯一の付加価値は「局所安定だが global 非収縮な gene を admit したい」場合のみ。
  実例 `gene=(decay=0,mix=0,gate=2)`: `s'=tanh(2s)` で `s*=0` は J=2 で不安定だが
  `s*≈±0.9575` は J≈0.166 で局所安定 (双安定系)。Hurwitz ならこれを admit、本ゲートは
  `J(0)=2` で reject。
- Stage 1b の目標「状態方向 contraction = 軌跡が一意に収束し記憶が一様に縮む」には**双安定
  (複数 attractor) は不適合**なので、global `L<1` が正しい・より強い証明対象。

→ Hurwitz は将来 attractor 多様性を探索する別 PoC (例 reservoir の edge-of-chaos) で意味を持つが、
本 Stage には追加せず、global `L<1` 一本で十分かつ上位互換。

---

## 6. honest 留保

1. **soundness only / completeness 放棄 (意図的)**: free-variable abstraction `t∈[0,1]` は
   `sech²(pre)` の到達域 (0,1] のさらに狭い閉部分集合を真に包含する **over-approx**。よって
   `unsat⟹L<1` は必ず真 (sound) だが、`sat` は achievable でない端点 (例 t=0) 由来の
   **保守的 false-reject** を含みうる (§3/§4 で 53 件確認)。fail-closed の MCP/llcore 規律と整合。
   admit 率を上げたければ achievable t の下限 `t_min = sech²(|mix|+|gate_str|)` を制約に足す
   余地があるが (将来拡張)、健全性はこのままで確保済。なお本写像は J が t について線形・端点 t=1 が
   `s=x=0` で到達可能なため、over-approx は実質 **tight** (grid で保守的 false-reject に至る binding
   制約は t=1 側で達成され、t=0 由来の過剰排除は限定的) — soundness レンズで確認。
2. **float vs rational**: Z3 は `decay`/`gate_str` を `RealVal` で float64-exact な有理数に焼き込み、
   閉形式 `_lipschitz_upper_bound` も同じ float64 値を使うため exact/float の食い違いは観測されず
   (境界 1e-10 でも 0)。
3. **marginal 安定の副作用**: `decay=1` (純記憶, gate=0) は `L=1` ちょうどで strict `L<1` では
   **reject** される。「縮小も発散もしない境界 marginal 安定」を厳格定義から外す既知副作用
   (`test_marginal_decay_one_rejected` で固定化)。用途次第で `L≤1` 緩和版を別途用意可。
4. **slow contraction**: `L_upper_bound` が 1 直下 (例 0.99999) の gene も certified される。
   数学的には収束保証だが極端に遅い (500 step で ratio 0.96)。strict `L<1` の定義通りで unsound
   ではない。実用上「速い contraction」を要求するなら閾値 `L<1−margin` を別途設ける余地あり。
5. **scalar / 対角前提**: 写像が完全対角 (coord 独立) なのでスカラ解析が全ベクトル写像に妥当な点を
   empirical レンズで確認済。kernel が非対角化する将来拡張では再検証が必要。
6. **AUDIT の high finding は別ファイル**: AUDIT が指摘した「`refinement.py` の R 検査が vacuous
   (mix/gate に鈍感)」は本 Stage 1b (`invariants.py` の contraction PoC) とは別ファイル・別関数の
   問題で、本タスクの additive 制約下では対象外。新規 `verify_lipschitz_contraction` は AUDIT が
   sound と結論したパターン (RealVal 焼き込み + free-t over-approx) を踏襲し、abs 下界バグ等は
   導入していない (soundness レンズで二重確認)。

---

## 7. 独自軸 #6 (Lipschitz/Hurwitz embedding) 判定: △ → ✓

**判定: △ (state_norm のみ) → ✓ (状態方向 Lipschitz contraction を mechanism 実証)**

根拠:

- STAGE_3 時点では「state_norm 着地 (1a) のみ、Lipschitz は post phase」で **△**。本 1b で
  **Lipschitz contraction (`L<1`) を Z3 per-gene gate として実装・実証**し、進化ループ SMT gate
  embedding (確定独自軸 #6 の本旨) を満たした。
- 5 破綻ゲート全 PASS、敵対検証 4 レンズで **surviving high/med refutation なし** (regression の
  low 2 件は文書精度のみ、健全性・回帰防止に無影響)。
- soundness を多経路でクロスチェック (Z3 vs 閉形式 / empirical ~16000 gene / true grid 8000 gene /
  FP 境界 3000 gene) し **unsound 証明書ゼロ**。non_triviality で state_norm より真に強いこと
  (本物の発散 gene 99 件を弁別) を実証。
- Hurwitz は本 global `L<1` に包含され (strictly stronger)、Stage 1b の目標には追加不要と論証。

→ surviving high refutation が無く 5 gate 全 PASS のため **✓ に昇格**。ただし claim 範囲は honest に
限定: 「**状態方向 (同一入力) contraction の mechanism 実証**」であって入力ロバスト性ではなく、
completeness (sat の false-reject 排除) も意図的に放棄。Hurwitz (固有値制約 = 旧 1c 候補) は本軸に
包含されるため別途実装せず、attractor 多様性探索の将来 PoC に回す。

---

## 実行方法

```powershell
cd <llcore-root>
py -3.11 -m pip install -e .[z3]
py -3.11 scripts/poc_1b_lipschitz_invariant.py
py -3.11 -m pytest tests/unit/test_poc_1b_lipschitz.py -v
py -3.11 -m pytest tests/unit/test_poc_1a_z3_invariant.py -v   # 回帰 (10 passed 維持)
```

## 次段候補

- 多次元 non-diagonal kernel への Lipschitz 拡張 (frobenius / spectral norm)、対角前提を外す
- `L<1−margin` の速度閾値版 / `L≤1` marginal 緩和版を用途別 gate として用意
- attractor 多様性探索の別 PoC で Hurwitz (双安定 admit) を扱う
- 進化ループに `verify_lipschitz_contraction` を online gate として接続し毎世代 certify

## 8. Codex pair-review (gpt-5.4) 訂正 (2026-06-01)

commit 前ゲートの Codex 相互レビュー ([[feedback_codex_pair_review_for_llcore]]) を実施。**soundness blocker なし**(`unsat⟹L<1` は健全、独自軸#6 ✓ は維持)。ただし verdict の **説明過剰主張 (overclaim)** を以下のとおり honest に訂正 (各 finding を実コードで再検証済、[[feedback_external_ai_verify]]):

1. **(high) 「t=0 由来の保守的 false-reject / completeness 放棄」は本モデルでは誤り → criterion は実質 exact**。`gene.clipped()` で `decay∈[0,1]` が強制されるため `|J(0)|=decay<1` (decay=1 境界を除く)、かつ `t=1` は `pre=0` (s=x=0) で到達可能。J は t について線形なので `sup|J|=max(|J(0)|,|J(1)|)` が到達域 (0,1] 上の真の上限そのもの (decay<1 では Z3 の [0,1] と一致)。t=0 起因の過剰排除は起きない。**§6.1 の「completeness 放棄」「53 件 false-reject」表現を撤回** (decay=1 の marginal だけが境界)。
2. **(high) 「99 件発散 / 53 件 false-reject」の定量分類を撤回**。これは有限本数・有限長の trajectory ratio test 依存で、全 `(|s|≤1,|x|≤1)` に対する真の Lipschitz 定数を判定しない。Z3 sat = 到達域に `|J|≥1` の点が実在 = **正しい reject** (有限軌道が膨張領域を励起しなかっただけ)。§3/§4 の 99/53 は **"sampled finite-horizon behavior"** に格下げ。質的主張「contraction は state_norm より強い」と代表反例 `gene=(0,0,2)` は妥当。
3. **(medium) Banach 一意固定点は scope 過大 → 「固定入力 x の self-map `F_x(s)` の contraction = その入力列に対する一意応答軌道」に限定**。時変入力では固定点でなく一意応答。§1/§7 末尾の「同一入力」限定は正しいが前段の一般表現と不整合だったので統一。
4. **(medium) no-z3 は API 単体では fail-closed でない** (`contraction=None` の tri-state を caller 規約に委譲)。online gate (Stage 1c) 接続時は呼出側が None を必ず reject 扱いにすること (将来 enum 化検討)。
5. **(medium) 大量 sweep (~16000 / 8000 grid / 3000 boundary) は本 commit 差分で未再現**。敵対検証レンズ agent の ephemeral run の数値で、再現スクリプトが repo 未同梱 (= 「実験結果を残す」規律違反)。→ §3 の当該数値は **"adversarial-lens run, 差分外で未再現"** として扱う。**follow-up**: sweep を `research/poc_1b_sweeps/` に再現スクリプト + results JSON で永続化。
6. **(low) exactness 直接固定テスト follow-up**: `verify_lipschitz_contraction(g).contraction == (_lipschitz_upper_bound(...)<1)` を負 `gate_str` 含む広域ランダムで固定する回帰テストを追加する。

→ 独自軸#6 の **✓ 判定は soundness 健全のため維持**。本訂正は説明の honest 化 (claim 降格) であり、機構の実証 (Z3 で状態方向 contraction を per-gene gate 化) は揺らがない。Codex 結論: 「実装コアに soundness blocker なし、ブロッカーは主に verdict の過剰主張」。

---

## 関連 memory

- [[project_llcore_init_2026_05_29]]
- [[project_llive_rwkv_backend]]
- [[feedback_codex_pair_review_for_llcore]]
- [[feedback_benchmark_honest_disclosure]]
