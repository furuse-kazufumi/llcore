# 「進化可能な LLM」FW 再設計計画 — 4 レンズ敵対 red-team 統合 決定メモ

**作成**: 2026-06-09
**対象計画**: `docs/EVOLVABLE_LLM_PLAN_2026_06_09.md`(guarantee 主軸 VSOA + capability terrain 副線 + 三者分業 NAS framing)
**入力**: 4 レンズの敵対 red-team — (1) kill-the-plan / 最尤単一失敗モード、(2) existence-condition-math(width_grow × cert_inf)、(3) strategic-ambition(戦略・野心)、(4) honest-residual(capability leak / falsifiability / 既踏性 / coupling 盲点)
**一次照合**: `src/llcore/verifier/backends.py`(`_t_min` L96-98 / `_infnorm_sup` L111-119 / `InfNormBackend.certifies` L154-157)、`changeop.py`(L160-193)を本メモ作成時に再照合済
**規律**: 全編 honest-disclosure。潰せなかったリスクは⑥に正直に残す。

---

## ① 総合判定

**判定: 要再設計(redesign-before-preregistration)。**

事前登録を確定する前に、構造的な must-fix が 2 件存在し、いずれも「計画の安全弁(撤退条件)が実は機能しない」「中核の North Star が反証不能のトートロジー」という性質のため、修正なしに着手すると最悪ケースの残存価値がほぼゼロになる。逆に下記②の上位 must-fix を反映すれば計画は GO-with-fixes に昇格できる(4 レンズ全てが `plan_survives = conditional` で一致しており、潰せない fatal ではなく、修正可能な構造欠陥)。

**一文理由**: 計画の存立条件・退避先・North Star が、いずれも「未実装の width_grow × cert_inf soundness」という単一点に collapse しており、その単一点の数理が `backends.py` と不整合(脅威を box 拡大と取り違え)なまま「stress 検証で潰す」設計になっているため、撤退条件が「価値ゼロ着地を承認するゲート」として空回りしている。framing(per-row 構造への書換え + 退避先を評価枠組みへ固定 + 存立判定の純数値 scan 前倒し)を直せば conditional の生存条件を満たせる。

---

## ② fatal / major findings 一覧(レンズ横断・重複統合・severity 順)

レンズ間で同一実体を指す finding を統合した。各項に **[統合元レンズ]** と **must-fix** を付す。

### F1【fatal】存立条件の数理が backends.py と不整合 — 脅威の取り違えと、それに伴う「stress 検証が起きない脅威を偽 PASS」する設計
**[統合元: existence-condition-math fatal / kill-the-plan finding2]**

- 計画 §① / §6.2 / §⑩賭け1 / 撤退条件の全てが「width_grow → `M=Σ|W|` 増 → `_t_min` の box 拡大 → `sup‖J‖∞>1` 越境」を**唯一の存立条件軸**に据える。
- だが一次照合(`backends.py:111-119`)では `_infnorm_sup` は per-row ループで `ti ∈ {t_lo[i], 1.0}` の**端点 max** を取る。検証では sup が `ti=1` で達成される行が 99.6%(=box 幅に無関係)、box 拡大で sup が変化する行は 1.94% に過ぎない。
- すなわち真の脅威は「box 拡大」ではなく「**新 column が既存行 i の `off`-sum(`Σ_{j≠i}|W[i,j]|`)を増やし、`ti=1` での sup が 1 を越える**」=**per-row abs-sum 増**。計画は脅威を取り違えており、現行 stress-test は**起きない脅威を PASS し false GO で進む**。これは §6 honest-disclosure 規律に正面から反する。

**must-fix(F1)**: §①/§6.2/§⑩賭け1/撤退条件の因果連鎖を **per-row へ訂正**。越境条件を「新 column が既存行 abs-sum を増やし `ti=1` の sup が 1 超」に差し替え、stress-test を per-row 不変条件の検査へ再設計する(EXP4 系: 新 column が既存行へ大重みで PASS→FAIL 0.847→1.039 を再現する設計に)。

---

### F2【fatal】退避先「案3=固定 topology guarantee」が VSOA の novelty を抜いた残骸に潰れ、撤退条件が「価値ゼロ着地を承認するゲート」になっている
**[統合元: kill-the-plan finding1(severity_top)]**

- §⑩ 総合撤退条件は「Phase1 step3 FAIL → 即 案3(固定 topology・param-shift + branch のみ)or 評価枠組みへ退避」を安全弁とする。
- だが退避先「案3=固定 topology guarantee」は VSOA の唯一の正味 novelty(`width_grow × Net2Net × cert_inf` soundness=動的成長そのもの)を抜いた残骸であり、(a) 機構は現 `changeop.py` の 3 float + STABLE の clip-or-reject に既踏で §⑪-1 が「差は 2 点=査読で自明」と自認する地点に正確に戻り、(b) 残す `branch_add` の per-block AND 合成は §⑪-3 の coupling 盲点(1267/3270 誤 admit)を再導入する。
- 結果、退避先に残る novelty は「固定アーキ guarantee = CT-BaB(2411.18235)既踏」に縮約。judge B verdict「撤退先=VSOA が既に居る場所(案 A guarantee-niche)に戻る」と同型。**「別の退避先へ撤退」でなく「主軸の最弱版への後退=差別化喪失を撤退と呼んでいる」**だけ。

**must-fix(F2)**: 退避先を「VSOA の最弱版(固定アーキ guarantee)」でなく、prior-art §⑤案3 本来の **Verified-Plasticity Evaluation Framework(機構でなく第一級指標の提案)** に固定し直す。評価枠組みなら width_grow soundness FAIL 自体が「contraction-gate は動的構造成長と両立しない」という measurable な第一級 negative=評価資産に転化する(=本当に別物の destination)。§⑩ の「案3=固定アーキ guarantee」が prior-art §⑤案3(eval framework)と矛盾している点を Phase0 着手前に確定し、事前登録に明記する。

---

### F3【fatal】North Star 主軸の 2 命題が定義トートロジー / working-filter 定義効果で、falsifiable な新規命題が実質ゼロ
**[統合元: honest-residual finding1+finding2]**

- **H-stability(certified-stable rate=100%)**: admit の定義そのものが `InfNormBackend.certifies = _infnorm_sup(...)<1.0`(L157)が True の gene のみ通すこと。よって「admit 全件が ρ<1 certificate を持つ=100%」は観測前に確定する**同義反復**で falsifiable でない。
- **H-drift(gate付き < 無gate)**: gate の機能は ρ≥1 を reject すること。無 gate は定義上 ρ≥1 を admit し HD-1 で ρ→1.95 発散(既確立)。「gate 付きの drift が無 gate より小さい」は**機能する filter が必ず示す定義効果**で、SYSTEMATIZATION §3.3(640 reject + 0/180 violation)で tiny scale 既 PASS。`feedback_benchmark_honest_disclosure`(変に有利な結果は内訳を疑う)に反する。
- 結果、主軸 3 本柱のうち H-stability=トートロジー、H-drift=定義効果、H-forgetting=高確率 NULL(§⑪-4)となり、**主軸の falsifiable かつ新規な命題が実質ゼロ**になる危険。

**must-fix(F3)**: North Star から「certified-stable rate 100%」と「無 gate との単純 drift 比」を削除し、唯一新規な命題に一本化する: **「width_grow/branch_add で構造を成長させた後も `_infnorm_sup` の box-bound が独立 eigen 再検査と不一致(false-admit)を起こさない=成長操作下での soundness 維持」**を反証可能形(成長操作 N 回中 false-admit ≥1 で FAIL)で事前登録。これは §⑩賭け1=存立条件と同一なので **North Star=存立条件に一本化**する。H-drift の残す核は「経験 gate との soundness/コスト比」または賭け3(tiny→実 LLM で drift 抑制が消えないか=真に NULL 許容)に限定。

---

### F4【fatal/judge 整合】judge 最高得点 E(eval-framework=20)を正当化なく「退避先」に降格し、最も脆い単一点に案の生死を賭ける逆向きリスク設計
**[統合元: strategic-ambition finding1(severity_top)]**

- judge は E(Verified-Plasticity Evaluation Framework)に最高 20 を付けたが、計画はこれを A(19)の「副線が立たねば戻る場所」に降格。score 逆転を honest_alignment 1 軸で無言に上書きしており、その妥当性検証がゼロ。
- A の唯一の価値「0 false-admit」は未実装 width_grow に全面依存し、計画自身が「リスクでなく存立条件」「Phase1 で潰せねば案全体崩壊」と書く。最高得点で確立済の E(llcore 唯一の confirmatory 資産=6 装置: 事前登録/Holm 連言/artifact 規律/反証条項/自己検出力監査/反 over-claim critic)を主軸にすれば存立条件 FAIL でも deliverable は残る。**リスク管理として逆向き**。

**must-fix(F4)**: §2.2 に意思決定マトリクス(行=5 案、列=[judge score / honest_alignment / 確立済資産活用度 / ユーザーゴール適合 / FullSense 普及適合 / 存立条件の脆さ])を明示し、なぜ A を主軸にした(または E へ切り替える)かを論証可能にする。最低でも E 降格の機会費用(確立済 confirmatory 資産を埋もれさせる)を honest に開示する。→ 戦略フォーク④で正面から扱う。

---

### F5【major】存立条件判定が「重い未検証実装(SmolLM2 配線 + Net2Net)の後ろ」に置かれ sunk cost が最大化された地点でしか FAIL を引けない
**[統合元: kill-the-plan finding4]**

- 最尤失敗(width_grow × cert_inf 両立領域が空)は `backends.py` の `_infnorm_sup`/`_t_min` だけで完結する純数値実験で潰せるのに、計画は Phase0(base load + fine-tune)+ Phase1 step1-2(cert_inf 関数 + Net2Net 実装)を前置してから step3 で判定する。
- `feedback_poc_feasibility_first`(最大リスクを最初の 1 本で潰す)に反し、撤退条件が機能する速度を構造的に遅らせる。

**must-fix(F5)**: Phase0 の前 or 直後に「純数値の width_grow × cert_inf 両立領域 scan」(SmolLM2 不要・Net2Net 実装不要・`_infnorm_sup`/`_t_min` だけで synthetic `CoupledNDGene` を n→n+1 拡張し既存 row の ∞-norm を実測)を**最優先 step**に置く。両立領域が空なら real-LLM 配線も Net2Net 投資も一切せず即 評価枠組み退避を確定できる。F1/F2/F3 の最尤失敗を最安・最速で引く唯一の方法。

---

### F6【major】coupling 盲点が主軸 H-stability を無効化しうる — per-block AND 合成は既確立の誤 admit 機序(1267/3270)を構造ごと再導入
**[統合元: honest-residual finding4 / kill-the-plan finding5]**

- §5.2 の中核設計「block を小 n に切り独立 cert_inf を AND」は、SYSTEMATIZATION §3.3 が確定した「対角 scalar heuristic が 1267/3270 誤 admit、coupling-awareness が load-bearing」を再導入する。per-block AND は各 block 独立に ρ<1 を保証するのみで、block 間 coupling 込みの実 Jacobian の ρ<1 を保証しない。
- 誤 admit が 1 件でも出れば「0 false-admit」=唯一の価値が崩れ主軸 guarantee が消滅。退避先(branch_add を残す案3)でも branch=並列 path 追加は coupling を増やす move で除去されない。

**must-fix(F6)**: coupling soundness を Phase1 存立条件(賭け1)と**同格の第二存立条件**に格上げ。Phase1 で「2 block を residual で結合した最小系の実 Jacobian の真 ρ」を独立 eigen で測り、per-block AND が admit した構成で合成 ρ≥1 が 1 件でも出れば即退避と事前登録。退避先(案3)を「branch_add 込み」でなく「param-shift のみ + block 間 coupling 込み結合 cert」に再定義(per-block AND を禁止)。North Star に「block 間 coupling 込みの合成 ρ<1 を sound に保つ」を明示追加。

---

### F7【major】主軸 vs 副線の独立性が崩壊 — 「副線 NULL でも主軸独立 PASS」は H-stability が存立条件 finding と同一単一点に collapse する事実の見落とし
**[統合元: kill-the-plan finding3]**

- §⑩ 最後の砦「副線の失敗は案全体を崩さない(主軸が独立 PASS 可能)」は、主軸の独立 PASS を支える H-stability が finding F1/F5 の単一点(width_grow × cert_inf soundness)に依存するため成立しない。
- 最尤シナリオ(両立領域空で FAIL)では副線の成否に関係なく主軸 H-stability も同時に倒れ、残るのは H-drift(固定 topology でしか測れない=退避先の話)のみ。二重保険の前提が単一失敗モードに両方連動して崩れる。

**must-fix(F7)**: 主軸を「動的 width_grow を含む H-stability」と「固定 topology のみの H-drift/H-forgetting」に明示分離し、後者を width_grow 存立条件に依存しない**真の独立主軸**として事前登録。Phase1 step3 FAIL でも固定 topology の H-drift(無 gate vs cert_inf gate の ρ_eff 抑制)が独立に測れるなら最悪ケースの残存価値として明記(F2 の評価枠組み退避と接続)。H-forgetting は高確率 NULL を事前登録で許容済とし主軸 PASS 判定から外す(立てば bonus)。

---

### F8【major】野心の 2 段階矮小化 — North Star に framework 性(進化機構/topology 探索/開放端性/累積/拡張性)が全て排除されている
**[統合元: strategic-ambition finding2]**

- ユーザーゴールは「進化可能な LLM として*フレームワーク*を確立」。だが §3.3 North Star は perplexity を明示除外し、残る 2 命題は「1 個の fail-closed contraction gate が正しく動く」の言い換え(=F3)。進化が有用な構造を*発見*すること・topology が世代を越えて*蓄積*すること・FW が*拡張可能*であることが成功条件に入っていない。
- 実装事実とも整合: `changeop.py:160-193` の `apply_changeop` は scalar 3 float のみ操作し `CoupledNDGene` の (decay,W) n 次元構造に触れない、`kernel_swap_mock` は gate_str 反転のみ(コード明記)。進化機構の実体はまだ存在しない。野心が framework→gate、進化→param/width shift 1 種へ 2 段階矮小化。

**must-fix(F8)**: North Star に framework 性を測る軸を追加 — (a)「N 世代後の admit topology が param-shift baseline 比で構造的に多様化し、その多様性が held-out tasks への汎化に load-bearing」、(b)「新 base / 新 changeop / 新 certifier を 1 オブジェクト差替で載せ替えられる拡張性(`minimal_ga` の 3 plug-point を framework 約束として明文化・テスト化)」。立たないなら「FW」を名乗らず「verified-stability gate for recurrent adapters」と正直に縮める。§1 でどこまでが framework 約束でどこからが単一 gate PoC かを線引き。

---

### F9【major】離散トポロジー軸の多峰性が未検証のまま「M3 を構造的に回避」と主張(capability leak)
**[統合元: honest-residual finding3]**

- §6.1/§3.1 の三者分業は「進化は gradient が解けない離散トポロジー空間を探索」と主張するが、M3 の本質は「地形が単峰なら optimizer 種別に関わらず進化は無価値」であって微分可能性ではない。離散軸も単峰(width を増やすほど良い等の単調 bowl)なら greedy / grid が直接解き進化の付加価値ゼロ=M3 が離散軸でも再現。
- 「gradient が微分できない=進化が勝つ」は非 sequitur で、これが残留 capability leak。§2.2-3 の「M3 を構造的に回避」は未検証の希望的観測。

**must-fix(F9)**: Phase1 に「離散トポロジー軸(width/branch/op)が多峰か」の instrument 校正を**存立条件と並ぶ前提条件**として追加(width_grow greedy baseline vs MAP-Elites archive を同予算比較、greedy が並べば単峰=capability 立たずと事前宣言)。§3.1/§6.1 を「*多峰な*離散自由度のみ進化の領分」に修正。§2.2 を「M3 を*離散軸の多峰性が成立する条件下でのみ*回避(未検証)」に honest 化。

---

### F10【major】novelty が STABLE との 2 点差より更に狭い — 実 LLM 構造進化が「135M base に貼った 16 次元 toy adapter」へ縮退
**[統合元: honest-residual finding5]**

- §⑪-1 は STABLE との差を 2 点(構造変更 / sound cert_inf)と自認するが、本計画の構造変更は一切未実装で、回るのは SmolLM2-135M 本体でなく後付け n≤16 adapter block。実 LLM への load-bearing は Stage-B tiny(~0.5M)からの外挿で未検証(§⑪-2)。
- 結果「実 LLM 構造進化」の実体は「135M base に貼った 16 次元 toy block の topology を動かす」=STABLE(Qwen-2.5-7B 実重み編集)より実 LLM 性が低い。desk-reject リスクの核。

**must-fix(F10)**: §①②③の「実 LLM」修飾を adapter scope に限定(「実 LLM base に後付けした verified recurrent adapter の topology 進化」)。adapter が実 LLM 出力に load-bearing であることを Phase2 必須測定に昇格(Stage-B B-G1 の「benefit が core dim と増大」を SmolLM2 で再現)。未達なら「実 LLM への寄与は未確立」と開示する撤退条件を事前登録。

---

### F11【major】PoC→普及ファネルへの path がゼロ / guarantee-niche の需要側論拠が皆無
**[統合元: strategic-ambition finding3+finding4]**

- §⑪-6 が「地味・SNS 拡散性弱い」と自己申告するだけで、`project_f25_demo_polish`(採用ファネル先頭/動きで魅せる/商業価値訴求)への設計上の回答がゼロ。誰が・なぜ採用するかの利用者像がない。
- guarantee-niche の market 価値は §⑧ 自認の「外部再現不能な内部空白象限主張」に依存。差別化は 2 点 delta に圧縮され、需要側証拠(産業事例・事故・規制要求=EU AI Act high-risk 連続学習要件等)が 1 つも提示されていない。研究価値も「何が出ても publishable」の自己保身に逃げている。

**must-fix(F11)**: (1) consumer story を 1 本確定(例: llive 自己進化メモリ層 / llmesh SPC 適応制御の online 構造適応を verified gate が fail-closed に守る)= framework の顧客を定義。(2) 動きで魅せるデモ軸を guarantee 側で 1 個(無 gate baseline が ρ→1.95 で出力ノルム発散 vs gate 付きが ρ<1 に留まる**リアルタイム可視化**=破綻が止まる動き=拡散素材)。(3) guarantee-niche の需要側証拠を 1 つ出すか、出せないなら「市場価値は未実証・研究 niche への賭け」と North Star 横に明記し、ユーザーに明示判断を仰ぐ(CLAUDE.md: 選択が成果に大きく影響する場合は確認)。

---

### F12【minor】退避先の多重化が honest disclosure を盾にした保守設計化 — 上振れ賭けが「任意・余力時」に追いやられている
**[統合元: strategic-ambition finding5]**

- §7.3/§⑩/§⑪ の全仮説に「立たねば honest に開示/退避」が付き、退避先の終着点は全て「guarantee-niche(2 点 delta)」か「評価枠組み(E、主軸から降格済)」。計画は「最良でも 2 点 delta、最悪でも publishable negative」のレンジに自閉。ユーザーゴールの上振れ(進化が本当に有用=capability EXISTS)が §9 Phase2「副線(任意・余力時)」に追いやられている。

**must-fix(F12)**: 上振れ仮説 H-EXISTS(terrain-design で進化が gradient を honest_eval 4 条件で上回る constellation の存在)を「任意・余力時」から **Phase2 の必須・proper power 試行**へ格上げ。立てば FullSense 唯一の capability 差別化=普及ファネルの派手な軸(F8/F11 を一挙解消)、立たねば「実 small-LLM 損失地形は単峰」の decisive negative=強い研究成果。退避先の多重化を 1 段に減らす。

---

### F13【minor】H-forgetting が高確率 NULL で事前登録の重みが空 / soundness は機械証明でない
**[統合元: honest-residual finding6 / 計画 §⑪-4,5 の自認を再掲]**

- H2 系 3 件が系統的 NULL(HD-1 H2 Holm p=0.058 / R-endo Δ=−0.04 p=0.67 / viability autonomy NULL)、memory 軸 Δ≈+0.0134 極小。H-forgetting は構造的に NULL 確率が高い。
- soundness は「float `eigvalsh>0` 数値検査 + JSR oracle 有限長下界 + 独立 eigen 再検査」で機械証明された定理でない(§⑪-5)。

**must-fix(F13)**: H-forgetting を主軸 3 本柱から「副次・NULL 許容」へ明示降格(F7 と整合)。soundness の非機械証明性は North Star 一本化(F3)後も honest 留保に残置。

---

## ③ 存立条件(width_grow × cert_inf)の数理判定

**判定: 条件付き成立(conditional)。dead-on-arrival ではないが、計画が想定する「box 拡大」経路でもない。成立経路は per-row 構造に存在するが、その経路は自明であり novelty を支えない。**

### 根拠

1. **計画の存立条件の数理は backends.py と不整合(F1 の核)**。`_infnorm_sup`(L111-119)は per-row sup の max で `ti ∈ {t_lo[i], 1.0}` 端点を取る。検証で sup は `ti=1` 達成が 99.6%、box 拡大で sup 変化する行は 1.94%。よって「width_grow → box 拡大 → sup 越境」は**起きない脅威**で、計画の stress-test はこれを偽 PASS する。

2. **真の脅威 = per-row abs-sum 増**。width_grow が n→n+1 で新 column を既存行 i に足すと、`ti=1` での sup `diag + (1-decay[i])·off_i` の `off_i = Σ_{j≠i}|W[i,j]|` に新 `|W[i,n+1]|` が加算され、admit 済 row が ρ≥1 へ越境しうる。EXP4(新 column が既存行へ大重み)が PASS→FAIL(0.847→1.039)を実測している。

3. **dead-on-arrival は反証されている**。`cert_inf` が per-row MAX ゆえ、**新行/新 column 寄与後の各 row の abs-sum を元以下に保てば soundness は保存される**(per-row 分離で事実上保証)。EXP5b(Net2Net 複製 + fan-out 半分割)で passing 2000 件中 cert を破るのは 0/2000。よって「width_grow は必ず box を広げて soundness を壊す」という暗黙悲観は誤り。成立経路は実在する。

4. **ただし成立経路は自明 = novelty を支えない**。「新 column 寄与後の row abs-sum ≤ 元」を満たす width_grow は per-row 不変条件を満たす設計上の帰結であり、§⑩ が novelty(make-or-break)に据えるのは誤配分。一方その自明経路で「**関数が非自明に変わる(進化的価値あり)かつ全既存 row の `_infnorm_sup<1` を保つ ε>0 の結合強度帯が存在するか**」は未証明 — 結合ゼロ=関数保存だが死んだ unit(無進化)、結合非ゼロ=進化するが abs-sum 増、の間に非自明な両立帯があるかが本当の存立条件。

### 数理判定の結論

- **成立経路あり(per-row 不変条件: 新 column 寄与後の各行 abs-sum ≤ 元)**。dead-on-arrival ではない。
- **ただし計画の「box 拡大」framing は破棄が必要**(F1)。
- **真に難しいのは (a) 非自明な進化価値を持つ両立帯 ε>0 の存在、(b) `max_input_abs=1.0` ハードコード(L154/L171)の box 被覆未較正、(c) coupling 盲点(F6)、(d) transfer(賭け3)** — これらが残る存立リスク。

### 存立条件への must-fix(③ → 計画反映)

- Phase1 step3 を「反例探索 stress」から「**両立帯の構成的 scan**」へ格上げ: 新 unit 結合 `|W[i,n+1]|, |W[n+1,j]|` を 0 から増やしながら (a) 関数が非自明に変わる かつ (b) 全既存 row `_infnorm_sup<1` を保つ ε>0 帯が存在するかを実測。帯が空なら即 FAIL 確定(楽観禁止)。
- width_grow gate を **per-row 不変条件(新 column 寄与後 `Σ|W_ij| ≤ 元`)** として明文化し fail-closed reject。成立経路は定理明記し novelty から外す(工数を coupling/transfer へ再配分)。
- Phase0 で SmolLM2 入力 `abs(x)_inf` を実測し `max_input_abs` を sound 上界に設定し cert に実値を渡す配線を必須化(現状 1.0 ハードコードの box 被覆は未検証)。
- Decision gate 1 (3) の PASS 条件に「**非自明な進化価値を持つ admit が ≥1 件存在**」を AND で追加(死んだ unit の自明 PASS を排除)。

---

## ④ 戦略フォーク(ユーザー判断用)

中立に 3 案の trade-off を整理し、red-team としての推奨を 1 つ示す。

### (a) guarantee-first VSOA のまま(現計画の主軸)

| 軸 | 評価 |
|---|---|
| judge score 整合 | A=19(E=20 を降格)。**最高得点と不整合** |
| 確立済資産活用 | 6 装置(E)を埋もれさせる |
| 存立条件の脆さ | 案の生死が width_grow × cert_inf 単一点に collapse(F1/F5/F7) |
| 上振れ | 最良でも 2 点 delta guarantee-niche(F12) |
| 退避先 | 案3=固定 topology guarantee に潰れ差別化ゼロ(F2) |
| 利点 | honest_alignment 満点。3 敵対 judge を構造的に通過。実装 framing が最も明快 |

### (b) E = Verified-Plasticity *評価枠組み* を主軸に格上げ、VSOA をその枠組みで測る最初の被験 method に降格

| 軸 | 評価 |
|---|---|
| judge score 整合 | E=20=**最高得点と整合** |
| 確立済資産活用 | llcore 唯一の confirmatory 資産(6 装置)を前景化 |
| 存立条件の脆さ | width_grow soundness が FAIL でも**枠組みは method-agnostic に生き残る** |
| 上振れ | honest negative の量産が「枠組みの妥当性=deliverable」へ転化=強みになる |
| 退避先 | 機構失敗=第一級 negative=評価資産(F2 の本来の destination) |
| 弱点 | capability の派手さは依然なし(F11 は別途要対応)。「評価枠組み」自体は地味で普及ファネルにならない |

### (c) capability terrain-bet を主線に昇格(H-EXISTS を Phase2 必須・proper power へ)

| 軸 | 評価 |
|---|---|
| judge score 整合 | B=19。capability honesty=4(M3 NEGATIVE と緊張) |
| 上振れ | 立てば FullSense 唯一の capability 差別化=派手な普及軸(F8/F11/F12 を一挙解消) |
| 下振れ | 最尤 NULL→guarantee 退避=「高価な遠回りで案 A が既に居る場所に戻る」(judge B verdict) |
| リスク | M3 decisive NEGATIVE の真上。離散軸多峰性が未検証(F9)。NULL 確率が構造的に高い |
| 利点 | NULL でも「実 small-LLM 地形は単峰」の decisive negative=強い研究成果 |

### red-team としての推奨(1 つ)

**(b) を主軸、(c) を Phase2 必須副線、(a) を「枠組みで測る最初の被験 method」に降格する複合**を推奨する。

理由: judge score(E=20)・確立済 confirmatory 資産(6 装置)・存立条件の脆さ(width_grow FAIL でも枠組みは生存)の 3 点が揃って E 主軸を指す。(a) 単独は安全弁(退避先)が機能せず最悪ケースの残存価値がほぼゼロ(F2/F7)。(b) を主軸にすれば「何が出ても枠組みの妥当性が deliverable」が**弱点でなく強み**に転化し、(c) を必須副線に昇格すれば上振れ(capability EXISTS=普及ファネルの派手な軸)の賭けが復活する。ただし (b) は「評価枠組み」自体が地味なため、F11(consumer story + 動きで魅せるデモ + 需要側証拠 or 明示判断仰ぎ)を**同時に**満たさないと普及ファネルは空白のまま残る。

**この推奨は「選択が成果に大きく影響する」案件(CLAUDE.md 確認事項)なので、主軸を A→(b) へ反転するか否かはユーザー判断を仰ぐべき**。最低でも「現計画は honest disclosure 規律を盾に大きな賭けを回避した保守設計で、最良 2 点 delta・最悪 publishable negative のレンジに自閉している」ことを開示した上で決定する。

---

## ⑤ 計画 doc への具体修正(must-fix → どの節へどう反映)

| # | 反映先節 | 修正内容 |
|---|---|---|
| F1 | §① / §6.2 注記 / §⑩賭け1 / 撤退条件 | 因果連鎖を per-row へ訂正。「box 拡大」→「新 column が既存行 abs-sum を増やし `ti=1` sup 越境」。stress-test を per-row 不変条件検査へ再設計 |
| F2 | §⑩ 退避先 / §⑪-1 | 退避先を「案3=固定 topology guarantee」から **Verified-Plasticity Evaluation Framework** に固定し直す。prior-art §⑤案3 との矛盾を Phase0 前に確定し事前登録に明記 |
| F3 | §3.3 North Star / §7.1 | 「certified-stable rate 100%」「無 gate 単純 drift 比」を削除。**North Star=存立条件(成長操作下 soundness)に一本化**。H-drift は経験 gate コスト比 or 賭け3 へ |
| F4 | §2.2 | 意思決定マトリクス(5 案 × 6 列)を明示。E 降格の機会費用を honest 開示。④の主軸反転判断をユーザーに仰ぐ |
| F5 | §9 Phase0 直前 + §⑧ | 「純数値 width_grow × cert_inf 両立帯 scan(SmolLM2/Net2Net 不要、`_infnorm_sup`/`_t_min` のみ)」を最優先 step として前倒し |
| F6 | §5.2 / §3.3 / §⑩ | coupling soundness を第二存立条件へ格上げ。2-block residual 結合の真 ρ を独立 eigen で測定。per-block AND 禁止。North Star に合成 ρ<1 追加 |
| F7 | §7.1 / §⑩ 総合撤退条件 | 主軸を「width_grow 依存 H-stability」と「固定 topology H-drift」に分離。後者を真の独立主軸として事前登録。H-forgetting を主軸 PASS 判定から外す |
| F8 | §1 / §3.3 | North Star に framework 性 2 軸(topology 多様化の汎化 load-bearing + 3 plug-point 拡張性のテスト化)。framework 約束 vs 単一 gate PoC の線引き。立たねば名称を縮める |
| F9 | §3.1 / §6.1 / §2.2 / §9 Phase1 | 離散軸多峰性 instrument 校正を前提条件に追加。「*多峰な*離散自由度のみ進化の領分」に修正。「M3 構造的回避」を「多峰性成立条件下でのみ(未検証)」に honest 化 |
| F10 | §①②③ / §⑪ | 「実 LLM」修飾を adapter scope に限定。adapter の実 LLM load-bearing を Phase2 必須測定へ。未達時の撤退条件を事前登録 |
| F11 | §⑪-6 を「普及設計セクション」へ格上げ / §3.2 | consumer story 1 本確定(FullSense 3 製品のどれに乗るか)。guarantee 側の動きで魅せるデモ 1 個。需要側証拠 1 つ or 明示判断仰ぎ |
| F12 | §7.2 / §9 Phase2 | H-EXISTS を「任意・余力時」から Phase2 必須・proper power へ昇格。退避先多重化を 1 段に削減 |
| F13 | §7.1 / §⑪-4,5 | H-forgetting を副次・NULL 許容へ降格。soundness 非機械証明性を留保に残置 |

---

## ⑥ 残る honest リスク(潰せなかったもの)

red-team として、上記 must-fix を全て反映しても**消えない**リスクを正直に残す:

1. **両立帯 ε>0 の存在は未証明(③ の核)**。per-row 不変条件で soundness は保てるが、「非自明な進化価値を持つ admit が存在する」両立帯が空(=ゼロ結合の死んだ unit でしか cert を保てない)可能性は scan するまで分からない。空なら width_grow は無進化操作に縮退し、FW の核(進化が探索する離散自由度)が空回りする。これは Phase0 前 scan(F5)で初めて判定でき、**現時点では潰せていない**。

2. **transfer(賭け3)は本質的に未検証のまま残る**。Stage-B tiny(~0.5M)→ SmolLM2-135M の load-bearing transfer は SYSTEMATIZATION が「未検証」と明示する領域で、本メモの修正は scope 限定(F10)で over-claim を防ぐだけ。transfer が失敗すれば「実 LLM 構造進化」の主張は adapter scope に縮む。

3. **novelty の狭さは原理的に解消不能**。STABLE との 2 点 delta(構造変更 / sound cert)は事実であり、corpus の空白象限主張は §⑧ 自認の通り外部再現不能。need-side 証拠(F11)を出せなければ「誰も困っていない問題」の疑いは残る。これは設計でなく市場の問題で、red-team が潰せる対象でない。

4. **coupling 盲点が原理的に soundness を崩す可能性**。F6 で第二存立条件に格上げするが、block 間 coupling 込みの結合 cert に縮退すると賭け2(2^n 壁)と衝突する。n=block 合計での feasibility 再判定が必要で、最悪「block を切る設計制約」自体が崩れる。これは Phase1 の実測待ちで、現時点では条件付きリスクとして残る。

5. **(b) 評価枠組み主軸でも普及ファネルは空白になりうる**。E は honest だが地味で、F11 を満たさない限り FullSense の採用ファネル先頭要件(`project_f25_demo_polish`)に対する回答は依然弱い。capability(c)が NULL なら派手な軸は構造的に存在せず、「研究 niche への賭け」とユーザーが受容するかどうかに依存する。これは red-team が解決できない戦略的トレードオフとして残す。
