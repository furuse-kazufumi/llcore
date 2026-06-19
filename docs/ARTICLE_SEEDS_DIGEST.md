# llcore 記事マスターダイジェスト(編集長統合版)

対象: 生 seed 58本(#1–#58)+ idea bank(bank:S1–S7 / A1–A11 / B1–B8)+ competitor doc(comp:1–11)+ 執筆済 draft(draft:s1 / s2 / a7 / b1)。
方針: **忠実性 > 整形**。全 seed の caveat を末尾「保持必須の honest caveats」へ落とさず転記。llterm が選べるメニューとして提示する。

---

## 1. Flagship(執筆済 draft)

| id | タイトル | grade | 状態 |
|---|---|---|---|
| draft:s1 | PPL が PASS でもモデルは半分壊れていた — 単一スコアの罠 | A(連載入口) | **draft化済(公開可)** |
| draft:s2 | 「proprietary 超え」の解剖学 — cherry-pick 5型 + 自己監査 | S | **draft化済(公開可)** |
| draft:a7 | 制御理論はとっくに「定数状態で過去を運ぶ」を解いていた — RWKV/Mamba=SSM | S | **draft化済(公開可)** |
| draft:b1 | 勝った話より負けを見せたほうが信用される — 敗北ログをキャリアの名刺に | S | **draft化済(公開可)** |

この4本が連載「自宅 CPU から見た 2026年6月の LLM 業界」の骨格(測り方を疑う → 実装で掘る → 着地)。残りはこの4本に接続する未執筆 standalone / 連載素材。

---

## 2. テーマ別クラスタ(7 群)

1. **量子化アーク** — #34/37/41/42/43/44/45/53, bank:A1/A8, comp:4, draft:s1。int8 4×圧縮の再現 → ビット幅 cliff → RTN/per-group/GPTQ/QAT/LSQ の段階再導出。一貫結論=tiny char-LM の 2bit strict 97% gate は手法では越えられず、床は規模/学習予算/VQ が動かす。全て simulated quant=速度未測。
2. **メモリ効率の実機計測** — #33/35/38/39/40/56, bank:A2/A3/A4/A5/A6, comp:8/9/10, draft:a7。mmap 固定コスト・working-set 上限で RAM超・streaming の allocator 壁・文脈長×8 で GPT peak×2.65 vs recurrent×1.00。pivot(#35)後の勝ち筋実証。
3. **honest disclosure・負けの記録** — #2/8/10/14/16/36/53/54/55/57, bank:B1/B3, draft:b1。null/敗北を消さない配当(2,000万エッジ回避)、capability⊥guarantee の同時観測、進化≒random の負け筋確定と機構資産の再配線。**#57=AI 批評家の『捏造』指摘を一次証拠で覆した回(冤罪)**=show-your-losses の白眉候補。
4. **競合地図・業界比較** — #46–52, bank:S2/S3/S5/S7/B5/B6, comp:1–11, draft:s2。業界の小型・低メモリ収斂、cherry-pick 5型、Goodhart、Hermes ベンチ不在、ライセンス地図、強豪5社×FullSense。全社数値 self-report・規模差 caveat 必須。
5. **fail-closed エンジニアリング** — #11/13/15/31/32, bank:A9/B8, comp:10。識別力設計、sound gate のコスト構造、最初の admit、producer/consumer 契約、provenance 再検証、SPC×fail-closed。
6. **AI駆動研究ワークフロー・corpus・migration・開発規律** — #1/3/4/5/6/7/9/12/17–30/58, bank:B2。計測の計測、RAD接地30分、circularity回避、checkpoint二軸、verified_safe_learning v1→v2 migration serial(#21–#28=1–2本へ統合前提)。**#58=テストの flaky が設計の混在を教える(判定を純関数に分離)**。
7. **メタ素材・記事元帳** — bank:S1/S4/S6/A7/A10/A11/B4/B7。連載構成案2案と単独保持の柱(認識=弁別的記号、分業の多義性、弱者の兵法TRIZ、家族OSS経済学)。

---

## 3. 全 seed 一覧(網羅テーブル)

| # | タイトル | grade | 側面 | 状態 | 連載fit |
|---|---|---|---|---|---|
| 1 | 「ANN化=速い」は規模の前提を隠している | A | ベンチ/honest/教訓 | 生seed | standalone(B2に吸収可) |
| 2 | 過去のnull結果が将来の設計判断を救う | S | honest/哲学/教訓/TRIZ | 生seed | B1 negative-results |
| 3 | トップレベルlsの罠 — 規模見積2.5倍ずれ | B | 教訓/実装 | 生seed | standalone |
| 4 | RAD接地30分で新規性マップ | A | 戦略/エコ/実装 | 生seed | standalone |
| 5 | 研究のcircularity回避=外部事実に接地 | A | 哲学/認知/教訓 | 生seed | B1 negative-results |
| 6 | dedupが取込実質規模を2割減らす | B | 実装/ベンチ | 生seed | standalone |
| 7 | 15.7GBで100万行storeの見積術 | B | 実装/教訓/UX | 生seed | standalone |
| 8 | 「35turns=122annotations」の小ささが設計を駆動 | A | honest/哲学/技術設計 | 生seed | M2 cert×connectivity |
| 9 | checkpointは量と時間の二軸で切る | B | 教訓/実装 | 生seed | B1/長時間ジョブ運用 |
| 10 | 無gate archive 69/69 ρ≥1 — 会話教師の危険性初観測 | B | honest/ベンチ/哲学 | 生seed | M2 cert×connectivity |
| 11 | floorを仮説族に包含させる — 識別力設計の一般原理 | A | 技術設計/教訓/認知 | 生seed | 識別力・測定の妥当性 |
| 12 | run_in_backgroundはセッションと運命を共にする | B | 教訓/実装/UX | 生seed | B1/長時間ジョブ運用 |
| 13 | sound gateのコストはreject率×resample構造 | A | 技術設計/実装/教訓 | 生seed | gated探索の2大落とし穴 |
| 14 | 「学習できる×安全を選ばない」の同時観測 | A | honest/ベンチ/哲学 | 生seed | M2 cert×connectivity |
| 15 | fail-closed gateは最初のadmitを設計しないと空転 | A | 技術設計/教訓/TRIZ | 生seed | gated探索の2大落とし穴 |
| 16 | 採用する頂点自体が発散境界上(best_rho1.000) | S | 哲学/honest/ベンチ/認知 | 生seed | M2 cert×connectivity(flagship候補) |
| 17 | APIが全部死んでもcorpusは構造化fallbackで前進 | B | 実装/教訓 | 生seed | standalone |
| 18 | 広いtopical queryはrecallを稼ぐがleafにノイズ | B | 教訓/実装/honest | 生seed | standalone |
| 19 | precision改善queryはflagship回収の監視が必要 | A | honest/教訓/戦略/実装 | 生seed | standalone |
| 20 | query1行よりquery file全体のふるまいを疑う | A | honest/ベンチ/教訓/戦略 | 生seed | standalone |
| 21 | v1→v2移行は入口契約を壊さないことが本体 | A | 実装/教訓/エコ/UX | 生seed | vsl migration serial(#21-28統合前提) |
| 22 | 互換shimは説明1段落+新入口リンクで足りる | B | 実装/教訓/戦略/UX | 生seed | vsl migration serial(統合前提) |
| 23 | 後方互換migrationは退行判定の観測点を固定 | B | 教訓/実装/戦略/UX | 生seed | vsl migration serial(統合前提) |
| 24 | fail-closed migrationは副作用隔離検査を先に設計 | B | 教訓/実装/戦略/UX | 生seed | vsl migration serial(統合前提) |
| 25 | 人間ゲートは破壊半径で切る | A | 教訓/戦略/哲学/UX | 生seed | vsl migration serial(統合前提) |
| 26 | fail-closed gateは実消費者のisolated dry-runで閉じる | A | 教訓/実装/honest/戦略 | 生seed | vsl migration serial(統合前提) |
| 27 | dry-runは親ディレクトリ構造の再現が要る | B | 教訓/実装/戦略/UX | 生seed | vsl migration serial(統合前提) |
| 28 | rollbackは素材をいつ作るかまで手順化して閉じる | A | 教訓/実装/honest/UX | 生seed | vsl migration serial(統合前提) |
| 29 | 収集規約はcollector実装1回読むだけで曖昧消える | B | 教訓/実装/戦略/エコ | 生seed | standalone |
| 30 | append-onlyログはsupersedeも残す | B | 教訓/honest/戦略/エコ | 生seed | standalone |
| 31 | artifact名が同じでも契約1bitずれでround-trip崩壊 | B | 教訓/honest/実装/戦略 | 生seed | standalone |
| 32 | fail-closed provenanceは読む側も再検証して止まれて成立 | B | 教訓/実装/honest/エコ | 生seed | standalone |
| 33 | mmapのload時メモリはほぼ固定コスト=大モデルほど効く | A | 技術設計/honest/教訓/TRIZ/UX | 生seed | standalone(→A2) |
| 34 | int8 weight-onlyはchar-LMでも約4×圧縮・無劣化再現 | A | ベンチ/業界比較/実装/honest | 生seed | standalone(量子化アーク) |
| 35 | 「構造プロット」を「実機計測」へ昇格=pivot第一歩の型 | B | 哲学/戦略/honest/教訓 | 生seed | standalone(pivot) |
| 36 | メモリで線形か指数か — 賢さは指数的に伸びない(26agent検証) | A | 業界比較/honest/教訓/哲学/戦略/認知 | 生seed | standalone(→A6) |
| 37 | 量子化cliffは大モデルほど低ビット頑健・PPL-only gateは危険 | A | ベンチ/honest/教訓/実装/業界比較 | 生seed | standalone(量子化アーク) |
| 38 | 使えるRAM<モデルでも動く — working-set上限で実証 | S | 技術設計/実装/honest/UX/TRIZ/未来 | 生seed | standalone(→A3) |
| 39 | int8 streamingは常駐は減るがpeakは圧力下でしか減らない | A | 教訓/honest/実装/技術設計/認知 | 生seed | standalone(allocator壁) |
| 40 | 文脈で膨らむTransformer/平坦recurrentを実機peak RSSで裏取り | A | 技術設計/ベンチ/honest/実装/業界比較 | 生seed | standalone(→A4) |
| 41 | per-group量子化は床を下げるが2bit安全にはRTN超が要る | B | ベンチ/honest/教訓/実装/業界比較 | 生seed | 量子化アーク連載 |
| 42 | GPTQ=重みをわざと不正確にして出力を正確にする逆説 | A | 教訓/認知/honest/技術設計/業界比較 | 生seed | 量子化アーク連載 |
| 43 | 最新GPTQが常に最強ではない — 極低ビットで粒度が勝つ | A | ベンチ/honest/教訓/業界比較/技術設計 | 生seed | 量子化アーク連載(PTQ vs QAT) |
| 44 | QATはPTQを2bitで約3倍引き離すがtinyでは最後の壁残る | A | ベンチ/honest/教訓/業界比較/認知 | 生seed | 量子化アーク連載(締め) |
| 45 | PoCを実CLI推論パスへ昇格 — int8量子化×mmap streamingで日本語生成 | B | 実装/技術設計/教訓/UX | 生seed | 「勝ち筋を道具にする」連載 |
| 46 | 2026-06業界が小型・低メモリ・ローカルへ収斂(追い風と脅威) | S | 業界比較/戦略/未来/honest/エコ | 生seed | competitor-methods連載 |
| 47 | 「OSSがGemini超え」の内訳を疑う(cherry-pick実演) | S | honest/業界比較/教訓/ベンチ | 生seed | competitor-methods連載(S2看板) |
| 48 | 記憶でキャラ/状態を定数的に保つ — MangaFlow×llcore | A | エコ/認知/技術設計 | 生seed | competitor-methods連載 |
| 49 | 成長するvs責任を持って成長するエージェント — Hermes×llive | S | 哲学/業界比較/エコ | 生seed | competitor-methods連載(llive差別化の核) |
| 50 | open weightsライセンス実務地図 — Gemma4/Cosmos/Qwen障壁 | A | 戦略/エコ/業界比較 | 生seed | competitor-methods連載 |
| 51 | ★連載素材: 強豪5社の手法×我々との共通点(専用doc) | B | 業界比較/honest/技術設計/哲学/教訓/戦略 | 生seed | 連載素材(専用doc土台) |
| 52 | ★記事マスターシードバンク(26本+連載構成案・専用doc) | B | honest/業界比較/技術設計/哲学/教訓/戦略 | 生seed | メタ素材(記事生成の元帳) |
| 53 | LSQで「2bit制覇」再挑戦→+1.1ppしか報われず(負けの実データ) | A | 教訓/honest/ベンチ/技術設計/業界比較 | 生seed | B1 show-your-losses(量子化アーク) |
| 54 | 負け筋(capability)の機構資産を勝ち筋(メモリ効率)に再配線 | A | 戦略/哲学/教訓/業界比較/honest/技術設計 | 生seed | branch A統合シリーズ |
| 55 | 安全gateを作ったら目的関数がそれを冗長にした(示唆的FAIL) | A | honest/教訓/技術設計/認知/ベンチ | 生seed | B1 show-your-losses |
| 56 | 散らばった勝ち筋の道具を1窓口に畳む — llcore.memoryツールキット化 | A | 技術設計/教訓/honest/業界比較/エコ/戦略 | 生seed | 「勝ち筋を道具にする」連載の着地 |
| bank:S1 | PPLがPASSでもモデルは半分壊れていた | S | honest/ベンチ/認知/技術設計/教訓 | bank収録 | 連載案1第1部先頭(draft:s1と重複) |
| bank:S2 | 「proprietary超え」の解剖学 — cherry-pick5型 | S | honest/業界比較/ベンチ/教訓/戦略 | bank収録 | 連載案1第1部2本目(draft:s2の元) |
| bank:S3 | 「16GBで動く」のメモリはRAM?VRAM? — 必要メモリの曖昧さ | S | 業界比較/honest/エコ/教訓 | bank収録 | 連載案1第1部3本目 |
| bank:S4 | 独立ベンチのない「使うほど賢くなる」は信頼できない — Hermes | S | honest/業界比較/ベンチ/哲学/自己進化 | bank収録 | Hermes群5本集約 |
| bank:S5 | ベンチに向けて学習する — PaddleOCR×Goodhart | S | ベンチ/honest/業界比較/TRIZ/戦略 | bank収録 | S1/S2と論点連動 |
| bank:S6 | エンコーダを捨てる設計と層を捨てる設計 — Gemma4×量子化アーク | S | 技術設計/TRIZ/業界比較/哲学 | bank収録 | 連載案1第2部先頭 |
| bank:S7 | 個人が勝てるのは「層」じゃなく「継ぎ目」だ | S | 戦略/業界比較/未来/エコ | bank収録 | 戦略flagship・第3部中核 |
| bank:A1 | 量子化アークを自前で歩く実装記録(RTN→QAT) | A | 技術設計/実装/honest/業界比較/ベンチ | bank収録 | 連載案1第2部2本目 |
| bank:A2 | mmapの固定コストを実装で解剖 | A | 技術設計/実装/認知 | bank収録 | 連載案1第2部3本目 |
| bank:A3 | 使えるRAM<モデルでも動くをWindows APIで強制実証 | A | 技術設計/実装/UX | bank収録 | 連載案1第2部4本目 |
| bank:A4 | 定数状態recurrentを実機peak RSSで殴る — Cosmos/MangaFlow | A | 技術設計/実装/業界比較/認知 | bank収録 | 連載案1第2部5本目(締め) |
| bank:A5 | footprintを正直に数える — scale/bias/LayerNormも計上 | A | 技術設計/実装/honest | bank収録 | 技術者向け中尺 |
| bank:A6 | 「メモリで賢くなる」の嘘 — レジーム別プリミティブ地図 | A | 業界比較/エコ/認知/教訓/哲学 | bank収録 | 連載案1第3部先頭 |
| bank:A7 | 制御理論はとっくに定数状態で過去を運ぶを解いていた | A | クロスドメイン/教訓/認知 | bank収録 | 連載案2先頭(draft:a7の元) |
| bank:A8 | 量子化は信号処理の量子化雑音を40年遅れで再発明 — GPTQ×ΔΣ | A | クロスドメイン/来歴/教訓/哲学 | bank収録 | 連載案2 2本目 |
| bank:A9 | 良品だけ通す検査ラインを進化計算に持ち込んだら空転 — SPC×fail-closed | A | クロスドメイン/教訓/TRIZ/honest | bank収録 | 連載案2 3本目 |
| bank:A10 | 認識とは弁別的記号の照合 — MangaFlow×RWKV | A | 認知/哲学/技術設計/クロスドメイン/UX | bank収録 | 連載案2 4本目(締め) |
| bank:A11 | 2塔分業の単一アーキと役割オーケストラ — Cosmos3×llive | A | 技術設計/業界比較/エコ/哲学 | bank収録 | 単独保持(分業の多義性) |
| bank:B1 | 勝った話より負けを見せたほうが信用される | B | honest/戦略/キャリア/哲学/教訓 | bank収録 | メタflagship(draft:b1の元) |
| bank:B2 | 「最適化したのに効果ゼロ」の半分は計測の壊れ — 3つの罠 | B | 教訓/UX/認知/技術設計 | bank収録 | 単独保持(実装者向け実践集) |
| bank:B3 | 失敗を消さなかったことが半年後に2,000万エッジ救った | B | 教訓/哲学/UX | bank収録 | 単独保持(null結果の複利) |
| bank:B4 | 弱者の兵法としてのTRIZ — 制約を北極星に変えた意思決定 | B | TRIZ/哲学/戦略/キャリア | bank収録 | 単独保持(pivot意思決定の解剖) |
| bank:B5 | 母数3つで「勝った」と言えるか — bazueを206コマにした理由 | B | ベンチ/honest/技術設計/教訓 | bank収録 | 連載案1第1部4本目(thin救済対象) |
| bank:B6 | 0.3Bエンコーダと16B本体、64B塔 — 規模則のどこを切り取るか | B | 業界比較/honest/未来/認知 | bank収録 | 単独保持(B5救済吸収先候補) |
| bank:B7 | 自宅で家族OSSを作るは戦略 — llmesh/llive/llove/manga-mdの経済学 | B | 戦略/エコ/哲学/キャリア | bank収録 | 単独保持(thin救済=実DL/star or 来歴記事へ降格) |
| bank:B8 | fail-closedはAIの未来であり責任を引き受ける個人のキャリア設計 | B | 未来/哲学/キャリア/技術設計 | bank収録 | 単独保持(fix3でB1圧縮候補) |
| comp:1 | 巨人たちの引き算 — 2026-06メモリ削減手法×引き算哲学 | S | 技術設計/哲学/業界比較/honest | competitor doc | 案A長編入口(規模差テーブル直後) |
| comp:2 | Gemma4 12B — エンコーダを撤去してLLM本体に統合 | A | 技術設計/哲学/業界比較 | competitor doc | §2-1強豪解説筆頭 |
| comp:3 | NVIDIA Cosmos3 — 理解する塔と生成する塔を1モデル内で分業 | A | 技術設計/業界比較/哲学 | competitor doc | §2-2(B-1/A-6接続点) |
| comp:4 | 量子化を自分の手で再導出する — 物置で高層階の設計図(★山) | S | 技術設計/教訓/honest/哲学 | competitor doc | 案B中盤の山(§3-A A-3) |
| comp:5 | MangaFlow — 不確実な生成と確実な構造処理の境界を引き直す | A | 技術設計/エコ/認知 | competitor doc | §2-4+§3-B(B-4/B-5) |
| comp:6 | Hermes Agent — 重みを触らず周辺の記憶を更新して育つ | A | 業界比較/哲学/戦略 | competitor doc | §2-5+§3-C(C-1) |
| comp:7 | PaddleOCR-VL-1.6 — データを無差別に増やさず弱点を狙い撃ち | B | 技術設計/業界比較/教訓 | competitor doc | §2-3(再fetch前提の補助カード) |
| comp:8 | KV cacheという共通の敵への分岐戦略 — 緩めるvsゼロにする | A | 技術設計/業界比較 | competitor doc | §3-A A-4(KV cache戦略の分岐) |
| comp:9 | 記憶は再生成でなく再利用 — 有界/lossyと非有界/losslessを区別 | B | 技術設計/認知/哲学 | competitor doc | §3-B(loose_analogy格下げを見せる場) |
| comp:10 | 量子化は後処理から設計前提へ — capability-gateをfail-closed配線 | A | 技術設計/戦略/honest | competitor doc | §3-A A-2/A-5/A-6+横断テーマ2 |
| comp:11 | honest disclosureをarchitecture/evalに配線する(★締め) | S | honest/哲学/戦略 | competitor doc | 案C締めの哲学(横断テーマ6/7) |
| draft:a7 | 制御理論はとっくに定数状態で過去を運ぶを解いていた | S | クロスドメイン/技術設計/honest/認知 | **draft化済** | クロスドメイン編 flagship |
| draft:b1 | 勝った話より負けを見せたほうが信用される | S | honest/戦略/キャリア/技術設計/業界比較 | **draft化済** | メタ/総括 flagship・着地点 |
| draft:s1 | PPLがPASSでもモデルは半分壊れていた | A | 技術設計/honest/認知/業界比較 | **draft化済** | 連載第1回(単一スコアの罠) |
| draft:s2 | 「proprietary超え」の解剖学 — cherry-pick5型+自己監査 | S | 業界比較/honest/教訓/技術設計 | **draft化済** | 連載第1部中核 |

> 重複統合メモ: draft:s1↔bank:S1、draft:s2↔bank:S2(+ #47)、draft:a7↔bank:A7(+ #16/#40)、draft:b1↔bank:B1(+ #16/#44/#53/#55)、bank:A1↔comp:4、bank:A4↔#40/comp:8、bank:A6↔#36、bank:A3↔#38、bank:A8↔#42。同テーマは1本に統合して公開する前提。#21–#28 は migration serial=1–2本に圧縮。

---

## 4. 公開ロードマップ

1. **draft:s1**(執筆済)— 単一スコアの罠。連載第1部先頭、すぐ公開可。
2. **draft:s2**(執筆済)— cherry-pick 5型。時宜性最大の業界比較看板。
3. **draft:a7**(執筆済)— RWKV/Mamba=SSM=制御工学。クロスドメイン白眉。
4. **draft:b1**(執筆済)— 敗北ログ4件をキャリアの名刺に。連載の着地点。
5. **comp:1** — 巨人たちの引き算(競合長編入口)。draft:s2 と別角度の横断哲学。
6. **comp:4 / bank:A1** — 量子化アークを自前で歩く実装記録。技術者向け中核。
7. **bank:S7** — 個人が勝てるのは「継ぎ目」だ。戦略 flagship・第3部中核。
8. **#36 / bank:A6** — メモリで賢くなるは嘘(26agent検証)。pivot の屋台骨。
9. **#38 / bank:A3** — RAM<モデルでも動く(Windows API 実証)。実装報告 standalone。
10. **comp:11 / bank:B1** — honest disclosure を architecture に配線(締めの哲学)。
11. **bank:S1** — draft:s1 公開後の連載トップ汎用フック版 or 統合(重複回避で後段)。

---

## 5. 保持必須の honest caveats(全 seed から忠実転記・削らない/弱めない)

- **#1**: recall@10 0.9825(min 0.8)、フィルタ併用 MRR は exact と完全一致 — ANN による検索品質劣化は無し。
- **#3**: (caveat なし)
- **#4**: 差別化判定はキーワード grep ベース('coreference supervision''discourse structure learning' のヒットゼロ)— コーパス被覆範囲に依存した重複度評価。
- **#6 / #2 / #5**: (caveat なし)
- **#7**: 100万行・save瞬間ピークは実測2点(53.7k=764MB, 89.7k=910MB)からの外挿であり実測ではない(ゆえに checkpoint save + --resume を保険として実装)。
- **#8**: 経験 gate の観測ホライズン T=64 と系列長が同オーダー=経験 gate に有利寄りの設定であることを事前開示に回した(不利設定で sound cert が勝てば主張が強くなる)。
- **#10**: v1 seed 0 の単一観測。v2 readout での再測で確定させる(要v2確定)。random init(W~U[-2,2])の大半が発散側なのは想定内。
- **#14**: v2 seed 0 の観測で、確定は m2_connectivity_poc.json(v3完走後)に依存する。
- **#16**: cert_inf の採用 gene も rho_max 0.993-0.998 と境界に張り付く(ただし証明付きで1未満)という対の観測に依存する解釈。
- **#17**: 2026-06-16追記: その後 ANTHROPIC_API_KEY は present かつ valid と確認済み。「ANTHROPIC org disabled」は 2026-06-13 時点の観測で、現在状態の正本は docs/next_plan.md を参照。fallback を保険として持つ教訓自体は維持。
- **#18 / #19 / #20**: (#18/#19/#20 個別 caveat なし。ただし #20 は最終 precision/recall は rerun 後の before/after でしか確定しないという本文主張を保持)
- **#21–#28(共通)**: #21〜#28 は同一 migration 論点の別アングル。記事ドラフト化フェーズでは 1〜2本へ統合前提で扱う(冒頭注記)。
- **#28(追加)**: 本件では index だけでなく live corpus 本体の置換も起きるため rollback 素材は RAD_INDEX.md 単体では足りない。
- **#29**: parser 最小条件(気付き or 側面が同一行に非空)は collector 受理の最小条件にすぎず、producer 契約としての seed 規約(気付き+根拠+側面)を置き換えない。根拠は parser 非強制でも契約上は必須項目として維持すべき。
- **#30**: #17 の 2026-06-16 追記は旧形式の supersede 注記として残しつつ、以後の supersede はインライン改変を増やさず numbered seed の append で行う。
- **#31**: tests/unit/test_lm_cli.py の prepare manifest round-trip 回帰は既存として参照、今回は未実行。
- **#32**: test_lm_corpus.py / test_lm_cli.py の drift reject 回帰は既存として参照、今回は未実行。
- **#33**: 恩恵は部分 working set / ページキャッシュ共有 / コールド起動遅延に限る。全載ワークロードでは最終 RSS は eager に近づく(全バイト touch で mmap も ~51.5MB へ伸びる)。RAM超モデルの検証は別途要実測。
- **#34**: weights-only(activation は fp32)。dequant fp32 forward の simulated quant=速度は未測定。footprint は scale と非量子化1-D params を含む実合計(理想下限0.25に対し実値0.25-0.26)。
- **#35**: (caveat なし)
- **#36**: emergence で指数的に賢くなるは Schaeffer 2023 の通り不連続メトリクスの測定アーティファクト。完全性批評が overclaim を1件検出: 「大RAM→大モデル常駐」は無条件には誤り — Beyond-Chinchilla ではサービング量が多いほど compute-optimal はより小さい N×長い学習 D へ動く。modern Hopfield は容量が次元 d に指数だが footprint→容量は線形・分離条件付き・容量≠知能。
- **#37**: honest反証: 「top1 は PPL より先に劣化する」という事前予想は本データでは不成立、両者 lockstep。誇張せず記録。合否に hard-capability proxy が要る(unigram PPL gate だけでは壊れた 2bit を PASS させる)。
- **#38**: avail RAM が限られる本マシンでは「RAM 総量超の巨大モデル」でなく「working-set 上限<モデルサイズ」で同性質を実証。hard-max 強制可否は環境依存(cap_set_ok と実測 peak をそのまま記録)。int8 は disk/load まで(per-layer streaming dequant forward は将来)。
- **#39**: torch caching allocator が解放した fp32 を OS に返さず transient 活性も乗るため、無風時の peak には削減が出ない。削減が顕在化したのは working-set 上限 368MB(< dense 常駐 539MB)で stream が完走した時のみ。計測する指標を間違えると「効果なし」と誤読する。
- **#40**: harness は元々 state_bytes(実測)+ KV/attn(解析値)までだった(本 seed で peak RSS 実測へ昇格)。
- **#41**: strict capability-gate(top1 fp32比97%保持)は 2bit では RTN per-group でも届かない(realp1 group32 で 81.5%)。2bit を安全にするには RTN超(GPTQ/AWQ の誤差補償 or QAT)が必要 — GPU/将来課題。「群を細かくすれば 2bit も救える」期待は CPU・RTN の範囲では半分だけ正しい(品質は上がるが gate は越えない)。
- **#42**: realp1/multi_smoke の cap-gate 経験値は out/gptq_compare*.json / FINDINGS 参照(本シードは weight↑/output↓ 確認が中心)。校正データ(活性化統計)が必要なのは出力誤差が入力分布に依存するため。
- **#43**: 3手法すべてで strict cap-gate(top1 fp32比97%)を 2bit では越えられず。per-group32 RTN(-5.31pp)が GPTQ-per-channel(-6.38pp)を上回った=粒度と誤差補償は直交・相補的で GPTQ+per-group が真の SOTA。床の位置は手法を変えても動かない(手法は各ビットの damage を減らすだけ)— 床を動かすには QAT が要る。
- **#44**: strict 97% cap-gate は QAT でも越えられず(multi_smoke 2bit で 82.9% 保持)。本当に 2bit を安全化するには「モデル規模/学習予算/学習可能 scale(LSQ)」が要る。「新手法を足せば床が下がる」期待は QAT で大きく前進したが完全制覇には至らず(アークの締め)。
- **#45**: 配線の勘所3点: (a)src は import-untyped ignore 不要(mypy strict 通る) (b)tied 重み(lm_head/wte)は state_dict 経由で扱う(named_parameters は dedup して欠ける) (c)int8/fp32 checkpoint は kind フィールドで透過判別。
- **#46**: 各社ベンチ数値はすべて開発元 self-report で第三者再現は未確認。Cosmos は本体64B(学習20兆トークン)で本質は大規模、Edge2B/Nano16B は小バリアントを持つだけ(「Cosmos も小型志向」と数えるのは限定付き)。llcore が「自前で小モデルを育てる」価値はモデル本体では勝てない=価値は再利用可能なインフラ層(手法・計測・gate)に置くべき。
- **#47**: 各社数値は self-report・第三者再現未確認。Cosmos3 の Gemini 比較は技報詳細表にのみ存在し NVIDIA 公式は Gemini 比較せず「open 内1位」と限定=二次情報の拡大解釈。Cosmos3 の Driving 勝ちは3ベンチのみで母数小、Robotics/General は Gemini が勝つ。PaddleOCR の 235B 超えは文書解析専用ベンチ上の話で「汎用知能で超えた」ではない、比較スコアは全て Baidu 自前測定。honest 監査要件: T2V サブメトリクス勝ち(AV/Physics)を使うなら「T2V 全体は open 内1位・closed の Veo/Seedance に負け」をペア提示。
- **#48**: 各社数値は self-report。「作画」は MangaFlow 自体でなく外部クラウド拡散モデル(Gemini 2.5 Flash Image / FLUX.2 9B)が担う制御層。Layout IoU 100%/Coverage 99.98% は幾何座標で明示配置する設計上ほぼ自明な自作メトリクスでピクセル生成ベースラインと土俵が違う。商用サイト mangaflow.studio は本論文と無関係の別製品。査読前 v1・引用ゼロ・GitHub 公開記載なし。bazue との接続は shared_problem_framing で未検証(「裏取り」禁止=seed #51 の格下げ注記より)。
- **#49**: 各社数値は self-report。Hermes は learning loop の有効性を示す独立ベンチ・査読論文がゼロ(arXiv 検索0件、効果は公式自称のみ)。stars 196,554 は GitHub API 実測だが star の質(bot/campaign 由来比率)は未検証=「マインドシェア先行」の脅威確度は留保付き。二次記事の「180,000/24,600/4か月」は数値も時期もバラバラ=API 実測を正とする。release v2026.6.5 Desktop は既存 core の GUI ガワで新モデル/新FWではない。
- **#50**: 各社数値は self-report。Cosmos 3 = OpenMDW 1.1 は OSI 認定の古典的 OSS ではない「open model」ライセンス、商用条件・派生物条項は要条文実読。PaddleOCR は Apache 2.0 だが ERNIE-4.5 ベースで派生ライセンス要確認。
- **#51**: honest 監査の必須修正を適用済(競合数値にラベル付け・規模差 caveat)。我田引水2件を格下げ(Cosmos↔llive=loose_analogy / bazue=未検証 shared_problem_framing で「裏取り」禁止)。arXiv ID は執筆前確認が必要。
- **#52**: honest 監査 verdict=出版可(条件付き)。冒頭に連載レベル恒久 caveat を記載済。必須 fix 7件を記載済(連載 caveat バナー・S1 過剰一般化修正・thin4本救済・A10 線引き・アナロジー密度・B6 外挿明記)。llterm はこれらを適用して記事化する前提。
- **#53**: multi_smoke 2bit で top1 30.48% / retention 84.0%、固定 scale QAT を +1.1pp 上回るが strict 97% cap-gate には遠く届かず FAIL。「手法を上げれば床が下がる」期待は LSQ でもほぼ報われない。2bit 90%+ は 7B+ でのみ成立・3bit が PTQ 実用床のまま。prior-art(LSQ 自身の小モデル SqueezeNext 2bit -14pt、k-bit scaling law/QiD)の予言を自前実測で追認したに留まる。
- **#54**: core honest: 「メモリ指標を適応度に」「accuracy×memory スカラ化」「fail-closed 制約付き進化」「検証器を gate に」はすべて HW-NAS(MnasNet)/多目的 NAS(NSGA-II)/Deb2000/CEGIS の再導出(既知, prior-art confidence high)。誠実な独自は「進化×sound 収縮 gate×メモリ北極星×recurrent 力学」の狭い四点結合 + 経験 gate(84% false-admit)vs sound cert(0% false-admit)の判別力計測 + 自宅 CPU 再現性に限る。branch A は capability を取り戻さない=guarantee 側に価値を置く(capability と guarantee は直交)。P1 留保=footprint は state-boundedness proxy で実 RSS footprint でない。
- **#55**: 設計主張 G2(gate なし~6% 安全 vs gate あり~100% 安全)が実測で反証され、実際は gate なしでも safe_rate 95–100%。目的関数(メモリ効率・retention)が既に有界 gene を選好し安全側を報酬しているため gate は冗長(retention で 0.95→1.00 と僅かに上がるのみ)。メモリ目的は footprint を半減(0.375→0.149)させたが代償として random を超える capability edge を喪失。P1 留保=footprint は state-boundedness proxy で実 RSS でない。
- **#56**: honest 注記=int8 が fp32 を 0.0002 だけ上回るのは「改善」でなく同点 argmax の測定ノイズ。int8_footprint_bytes は resident-weight-byte 会計(params+buffers、量子化不可な causal-mask fp32 を含む保守側)で on-disk file size とは別物。honest 位置づけ=新規アルゴリズムでなく packaging。プリミティブは llama.cpp/GGUF の再導出(confidence high)で誠実な独自は「footprint 勝ちを capability gate で fail-closed に検収する運用」のみ。「良い HW ほど効く」設計指針(int8→GPU 真 int8 GEMM / mmap→大 RAM 共有ページキャッシュ / 定数状態→長文脈)はいずれも未計測=設計仮説。速度は FINDINGS #34 で未測定と明記。
- **bank:S1**: fix2必須=過剰一般化を修正。「PPL が原理的に capability を隠す」→「PPL と top1 は lockstep で同時劣化したが gate 閾値(0.85×)が粗くて壊れた 2bit を PASS させた」に限定。事前予想「top1 先行劣化」が外れた事実を本文の主役に。tiny char-LM・unigram baseline 前提の CPU PoC で実 LLM とは規模も指標も別。PaddleOCR 数値は self-report で改竄ではない・97% 閾値は自前設定。連載トップ caveat バナー(極小モデル PoC・実 LLM を反証しない)を固定。
- **bank:S2**: 5型は「数値が嘘」でなく「主張の範囲が狭い」道具。各社数値は一次情報で実在確認済だが全て self-report・第三者再現未確認。llcore 自身も self-report 段で対象は tiny char-LM(0.81M-130M)CPU PoC — 自己監査章に必ず置く。連載トップ caveat バナーを固定。
- **bank:S3**: llcore は「RAM 総量超」でなく「working-set 上限<モデルサイズ」で同性質を実証。hard-max 強制可否は環境依存(cap_set_ok を残す)。tiny PoC vs 実 LLM の規模差を明記・各社数値は self-report。連載トップ caveat バナーを固定。
- **bank:S4**: 「ベンチがない=機能しない」ではない(まだ測られていない)。stars の質(bot/campaign 比率)未検証=脅威確度に留保。llive 側も Approval Bus の大規模独立ベンチ未整備=「設計上反証可能」と「実際に反証実験を回した」は別物。llcore cap-gate との接続は fail-closed の同型性であって機能の等価ではない。連載トップ caveat バナーを固定。
- **bank:S5**: region-aware は標準的で正当な手法であり「ズル」ではない — 主張は「ベンチ特化は汎化主張を弱める」に限定。96.33 は self-report・文書専用ベンチ。「データ支配」結論は tiny char-LM の null 観測由来で実 VLM とは規模も手法も別物。連載トップ caveat バナーを固定。
- **bank:S6**: llcore は 130M ランダム CharGPT の CPU PoC で simulated quant(真の int8 GEMM でない・速度未測)。Gemma4 は実 12B でモダリティ範囲も桁違い・統合効果は self-report(定量比較表未公開、「26B 迫る」も裏付けなし・相手 MoE)。「同じ原理」は思想レベルの類比で性能比較でない・TRIZ 対応付けは筆者解釈。異分野アナロジーは1記事1本柱(fix5)。連載トップ caveat バナーを固定。
- **bank:S7**: 強豪は実 LLM を本番品質で出荷、llcore は tiny char-LM PoC。「継ぎ目で勝つ」は戦略仮説で llmesh/llive が大手統合層(LangChain 等)に勝った定量比較はまだ無い。Cosmos は本体64B で本質は大規模(小バリアントを持つだけ)。ライセンスは時点情報・法的助言でない。連載トップ caveat バナーを固定。
- **bank:A1**: 全て 1-12M char-LM CPU PoC・weights-only・Linear のみ・simulated quant(速度未測)。実 LLM は大モデルほど低ビットに頑健で床が下がる=「tiny で 2bit が越えられない」は規模依存で大規模 LLM の量子化可能性を否定しない。閾値 97% は自前設定。量子化記事は「footprint は実測だが simulated quant=推論速度は一切未測」をフック直後に太字で前置。連載トップ caveat バナーを固定。
- **bank:A2**: 最大 53.91MB の CPU PoC。恩恵は部分 working set・ページキャッシュ共有・コールド起動遅延に限る。全 touch では最終 RSS は eager に近づく。真の RAM 超(モデル>物理 RAM)はこの PoC では未検証。連載トップ caveat バナーを固定。
- **bank:A3**: avail RAM 制約で「物理 RAM 総量超」でなく「working-set 上限<モデルサイズ」で同性質を実証。hard-max 強制可否は環境依存(cap_set_ok=true は本環境結果)。130M ランダムモデル・int8 は disk/load まで(per-layer streaming dequant forward は将来課題)。int8 streaming は圧力が無いと peak ほぼ不変(963→882MB)を正直に記載。連載トップ caveat バナーを固定。
- **bank:A4**: CPU char-LM の peak WS 実測(GPT.generate は block_size crop で実行上有界=本測は厳密長文脈想定)。Cosmos/MangaFlow 寄与は self-report/査読前。MangaFlow の作画は外部クラウド拡散モデル・Layout IoU 100% は自作幾何メトリクス。「同じ戦い」は設計思想の類比で性能の横並びでない。連載トップ caveat バナーを固定。
- **bank:A5**: weights-only(activation/KV メモリは別問題で未測)・simulated quant(速度未測)・1-12M char-LM CPU PoC。各社圧縮率も self-report で本記事の「検算」は方法論提示であって他社数値の反証でない。「4× は fp32 比のみ・GGUF 標準 fp16 比なら約 1.9×」を明示。連載トップ caveat バナーを固定。
- **bank:A6**: 族別優位は regime 依存で普遍的勝者なし。llcore 実測は3プリミティブのみ(exact attention・MoE・Hopfield 層・offload 階層は未着手)。Hopfield 指数は次元軸限定・分離条件付き・batch・量子化は simulated。Schaeffer 2023 は学界で議論継続中。「指数はない」は「進歩がない」でなく「関数形の誤読を正す」主張。連載トップ caveat バナーを固定。
- **bank:A7**: 0.81M-130M tiny char-LM CPU 実測で実 LLM 規模を保証しない。ρ/状態空間の対応は構造的アナロジーで RWKV/Mamba が文字通り Kalman フィルタという主張でない(非線形ゲート・学習で乖離)。peak RSS は torch baseline 込みでクリーンな信号は増分トレンド。ρ≈1 の過剰接続(我田引水)だけ抑える(編集長注)。連載トップ caveat バナーを固定。
- **bank:A8**: 概念的アナロジーで GPTQ が文字通り ΔΣ 変調器でない(GPTQ は空間的伝播・ΔΣ は時間的フィードバック)。CPU・tiny char-LM・weights-only・simulated quant(速度未測)。GPTQ の優位も RTN per-group が上回る場合あり(#43)。GPTQ 原典 Frantar 2022 の完全再現は未保証。連載トップ caveat バナーを固定。
- **bank:A9**: fix5: edge of chaos は蛇足→削除し SPC × fail-closed の実測(0/200→200/200)に集中。tiny char-LM CPU PoC(122 annotations 級小データ)で製造ラインの統計的厳密性とは規模もデータ量も別。SPC との対応は設計思想の共通性で llcore が SPC アルゴリズムを実装している主張でない。0/200→200/200 は単一 seed 観測を含む。連載トップ caveat バナーを固定。
- **bank:A10**: fix4: 「どこまで構造的同型でどこから別物か」を本文に明記。MangaFlow section memory(画素の知覚同定)と llcore recurrent(トークン分布の状態圧縮)は「過去を有限表現に畳む」点でのみ同型。MangaFlow 話者帰属困難と bazue hard case 159 は「似た症状」止まりで「同じ原因」と書かない(失敗メカニズムが別の可能性=要検証)。MangaFlow は査読前 v1・引用ゼロ・作画は外部クラウド拡散モデル。bazue VLM ベンチは将来計画で現時点は GT 整備段階・「認識=弁別的記号」はテーゼで実証済み定理でない。連載トップ caveat バナーを固定。
- **bank:A11**: Cosmos3 本体64B(20兆トークン)で本質は大規模・「Gemini 超え」は負け軸省略で成立・数値は全て NVIDIA self-report。llive の分業は設計・運用フレームで Cosmos の訓練済み単一モデルとは抽象レベルが違う=「分業」の語の多義性を整理する記事で性能比較でない。目的(効率 vs 責任)が異なり優劣の話でない。連載トップ caveat バナーを固定。
- **bank:B1**: B8(fail-closed×キャリア=一般訓話)を B1 に圧縮(fix3)。「誠実=必ず信頼される」は楽観で限界開示が評価される文化が前提・すべての場で有利と限らない。llcore の敗北は tiny char-LM CPU PoC スケールで実 LLM で同じ壁が同じ位置に立つ保証はない。「内なる審判が正しく強豪が不誠実」の二項対立に落とさない — 大手も技報詳細表には負け軸を載せており問題は主に二次拡散側。連載トップ caveat バナーを固定。
- **bank:B2**: 全て tiny char-LM CPU 自宅 PC 規模。「効果ゼロの半分は計測ミス」は経験則で定量的主張でない。各罠の数値は単発〜2回再測(mmap は2回再測で安定、ANN recall@10 0.9825)。allocator 挙動は torch 2.12.0+cpu 固有。連載トップ caveat バナーを固定。
- **bank:B3**: llcore 研究内部の運用知見で tiny PoC 規模の判断履歴。「null が必ず後で効く」一般法則でなく特定連鎖で配当が出た事例(survivorship に注意)。エッジ数~2,000万は設計時の見積もりで実走前の回避判断。連載トップ caveat バナーを固定。
- **bank:B4**: 「pivot のタイミングが正しかった」は事後の傍証で因果証明でない。脅威も確定=大手が高性能小モデルを Apache/open で無料配布するため llcore はモデル本体では勝てず価値はメモリ効率の手法・計測・gate に限る。tiny PoC・CPU で実 LLM 規模の有効性は未検証。Cosmos は本体64B で本質は大規模。連載トップ caveat バナーを固定。
- **bank:B5**: fix3: 母数3つ=実演 bazue が将来計画で空。S2 型4 + B6 に吸収する救済が必要(thin4本の1つ)。bazue 206 コマも統計的には小規模でこれ単体で VLM の優劣を断ずる母数でない(VLM 体制が整い次第の検証用)。Cosmos の3ベンチ数値は self-report・実在確認済。主張は「大標本が常に正義」でなく「標本数と分散を開示せよ」で小標本にも探索的価値はある点を併記。連載トップ caveat バナーを固定。
- **bank:B6**: fix6: 「llcore 規模則を 12B〜64B 強豪に当てる」箇所は「最大 130M での観測からの外挿仮説であって測定ではない。傾きの方向のみ literature 整合、絶対値は未保証」と本文側に太字宣言。強豪の規模諸元は self-report で直接ベンチしたものでない。「大きいほど低ビットに頑健」は最大 11.9M での観測。連載トップ caveat バナーを固定。
- **bank:B7**: fix3: 家族OSS経済学=一次測定ゼロ→実 DL/star を出すか「設計意図の来歴記事」に降格(thin4本の1つ)。「独立しても価値が成立」は設計目標で各プロジェクトが実際に単独採用された実績(DL/star/本番)で証明できていない。統合価値も F25 連携基盤は構想・部分実装段階。大手の単一プロダクトにも規模の経済の強みがあり「疎結合が常に勝つ」わけでない・個人/小チームに合った戦略で普遍解でない。連載トップ caveat バナーを固定。
- **bank:B8**: fix3: fail-closed×キャリア=一般訓話→B1 に圧縮(thin4本の1つ・独立記事として残すか要判断)。fail-closed は安全側だが止めすぎれば使い物にならないトレードオフ(過剰拒否の偽陽性)。gate 97% も tiny char-LM 閾値で実 LLM の最適閾値は別問題。「技術原理と人生倫理が同型」は比喩でどこで壊れるか(AI に人間の倫理を素朴に投影する危うさ)も明示・比喩の美しさで論を盛らない。連載トップ caveat バナーを固定。
- **comp:1**: 全社数値は self-report / 二次情報 / 構造的帰結。裸の断定にせず inline ラベル(self-report/技報時点/二次・未照合/構造的帰結)を付ける。規模差 caveat をフック直後に縫い込む。芯フレーズ=『同じ哲学に立つ/手法を再導出した/問題設定を独立に立てた、とは言える。でも同等品質を出したとは言わない』。規模は桁違い: Gemma4 12B(実11.95B)↔llcore tiny char-LM(1.36M-130M PoC)、Cosmos3 16-64B↔llive プロセス分業PoC、PaddleOCR 0.9B本番↔char-LM量子化PoC、MangaFlow 6段本番↔manga-md L0 spike、Hermes 19.6万★MIT↔llive 設計思想PoC。arXiv ID・日付は執筆前に実在確認(MangaFlow=arXiv:2605.28173・PaddleOCR=arXiv:2606.03264 confirmed、Cosmos=arXiv:2606.02800 要再確認)。
- **comp:2**: メモリ実数(BF16 26.7GB→SFP8 13.4GB→Q4_0 6.7GB/GGUF実測~6.98GB)は公式docs/HF self-report。『26B級に迫る性能を半分以下のメモリで』は公式blog self-report・独立ベンチ未検証・比較対象26Bが dense/MoE か未確定(必ず併記)。総学習トークン数・学習ハード・蒸留有無は公式model cardに not stated。音声エンコーダ完全撤去か lightweight投影かは記述に揺れ。効く理由3層相乗(アーキ=encoder-free ~850M削除/アテンション=5:1 local-global+shared K/V+p-RoPE/数値=QAT int4)。
- **comp:3**: Artificial Analysis で OSS T2I/I2V 首位・RoboArena best policy は技報執筆時点 self-report・自前ベンチ(HUE等)。技報PDF(10MB超)取得失敗、損失関数・学習データ量・Edge 2B数値は二次情報依拠で一次未照合。arXiv:2606.02800 要再確認。効率の鍵: 推論専用なら reasoner だけ動かし生成塔起動せず節約/NVFP4(Blackwellで最大2倍速・実測)/EVS(冗長動画トークンpruning)/Qwen3-VL重みから初期化(Edge2Bは from scratch)。
- **comp:4**: ネガ側を成功談と対等の重みで: tiny char-LM の 2bit は QAT でも strict 97% cap-gate 未達(82.9%)=同規模では制覇できないと自己反証。3bit=PTQ 安全床。手法を再導出して挙動を理解したのであって同等品質を出したのではない(再導出≠同等実装)。比喩『実家の物置 vs マンション高層階』の壊れる箇所=図面は描けても住める広さは出ない。比喩『量子化=真空パック食品』の壊れる箇所=限度超え(2bit)で中身が潰れる(cliff_then_flat)。
- **comp:5**: Count 100% は『決定的合成の構造的帰結であって生成精度ではない』を同じ文に必ず書く。ベンチは著者自作 meta-benchmark(本人『完全な manga dataset でない』明言)、自動メトリクスは1回の生成ラン。商用 mangaflow.studio は本論文と無関係の別物。arXiv:2605.28173 第1検証 confirmed。story section memory ablation: memory を外すと Self-CSD 0.668→0.547。対応物 manga-md-poc は L0 spike(ユーザー自身『お遊びレベル』)、MangaFlow は本番6段。
- **comp:6**: 約19.6万★は GitHub API 実測だが star の質(bot/campaign 比率)は未検証。session_search 4,500×高速化は release notes 由来。ソースコード本体・SQLite スキーマ・nudge プロンプト文言は未読。learning loop の有効性を示す独立ベンチ/査読はゼロ(汚染ループリスク指摘あり)。Desktop は既存 core の GUI ガワ。
- **comp:7**: 第1検証 confirmed(arXiv:2606.03264)だが本 workflow の deep-read は API error で欠落=手法詳細の記事化前に method 節を再 fetch して補完必須。OmniDocBench v1.6=96.33% は文書解析専用ベンチ・Baidu 自前測定。Apache-2.0。
- **comp:8**: recurrent は capability で劣る可能性、我々自身 capability=NULL_TIE/NEGATIVE と認める。『定数状態が勝つ』のはメモリ軸限定。Gemma のメモリ数値は self-report。
- **comp:9**: MangaFlow story section memory は llcore 定数状態とは別カテゴリ(サイズ非有界・lossless)=『再利用』の意味が別物。共通項を無理に作らず原理を『定数/有界サイズで持ち越す』に限定。Cosmos 2塔分業↔llive は loose_analogy に格下げ済(層内パラメータ分業 vs プロセス間分業=機構無関係。『同じTRIZ的分離思想』とは書かない、対比として見せる)。B-3 コンテキスト膨張防止規律は運用層(Hermes)vs 推論層(llcore)で対象レイヤが違う。
- **comp:10**: Gemma の int4=bf16 近傍のプロダクション品質には規模上届かない。速度軸は我々ゼロ実証(int8 は simulated quant=storage 圧縮のみ・速度未測)。『GPU で速度に化ける』は設計仮説と明記。A-6: Cosmos NVFP4 は Blackwell 実 GPU で2倍速実測、llcore は CPU PoC の設計仮説で真の int8 GEMM 速度は未測定(速度の非対称を hook に)。mmap 独自知見=ΔRSS~1.4MB 固定コスト=大モデルほど相対効果大の実測。
- **comp:11**: 自分にも適用: 強豪の self-report が即『誇張・虚偽』ではない(再現待ち)。大手も caveat を出す(MangaFlow『完全な benchmark でない』、Cosmos『技報時点』)。差は『honest disclosure を運用規約として配線したか』という姿勢の差別化であって『我々の数字が彼らより正しい』ではない。C-1 責任ある自己進化は Hermes も明示トリガーで責任化の意識を共有=二項対立でなく度合いの差。llive は完成度・普及で Hermes に大きく劣る。比喩ラングトンの蟻の壊れる箇所=疑う対象は『数字』でなく『未検証であること』。
- **draft:a7**: 全実測 0.81M〜130M tiny char-LM CPU PoC。12B〜64B 実 LLM への絶対値外挿は未保証(傾きの向きのみ literature 整合)。peak RSS は合算のみ測定、「超線形=二次項 O(T²)の顕在化」は解析モデルからの解釈で項分解は未実測。GPT.generate は block_size crop で実行上有界。ρ<1(課す安定性制約)と ρ≈1(学習が張り付く観測現象)は方向が逆で論理的に独立 — 制御理論は (A) は解いたが (B) は予言していない。§5 capability 結論にソース2種混在(NULL_TIE/NEGATIVE=llcore 自前 null 実測 / 「純 recurrent は recall で劣る・hybrid 化」=SSM 文献の未検証外部知識)。RWKV/Mamba は文字通り Kalman フィルタや線形時不変 SSM ではない。simulated quant のため推論速度は未測。
- **draft:b1**: 全数値は 0.81M〜130M char-LM CPU PoC、実 LLM 性能の直接反証ではない。敗北(1) 2bit: LSQ 84.0% は固定 QAT 82.9% を +1.1pp 上回るが「勝ち報告」にせず「規模の壁を越えられなかった負け」として確定。multi_smoke 1.36M の数字、realp1 は GPTQ 77.7%。敗北(2) mmap: 全重みを使う通常推論では最終メモリは eager に収束 —「常に省メモリ」は誤りで「必要分だけ遅延ロード」が正確。敗北(3) int8 streaming: 常駐72%減は本物だが平時 peak WS はほぼ不変(torch caching allocator)、圧力時のみ削減顕在化。敗北(4) 進化 20/20 勝利は ARTIFACT+NEGATIVE。同予算=forward CE 評価回数同一であって更新ステップ数同一ではない(torch Adam 2000 step vs evolution ~95 step)= artifact の正体。Cosmos Robotics 57.8 vs 58.2 は測定誤差圏内の可能性 — 自分の刃を競合にも公平に向ける(self-report・第三者未検証)。simulated quant のため推論速度未測。
- **draft:s1**: 全実測 0.81M〜130M char-LM CPU PoC、実 LLM 性能の直接反証ではない。「PPL が原理的に capability を隠す」とは主張しない — 今回は PPL も top-1 も lockstep で同時劣化。壊れたのは指標でなくゲート閾値(0.85×)の粗さ。事前仮説「top-1 が先に劣化」は本データで不成立 — 消さず記録。PaddleOCR 96.33 と llcore top-1 は測定対象・タスク・データが別物で数値比較は一切不成立、並べているのは評価の構造だけ。PaddleOCR の数字は self-report(公式カード 1.0B params、一部二次記事の 0.9B は v1.5/変種由来)。健康診断の比喩は「複数指標を見ろ」までは正しいが PPL と top-1 は独立でない(同じ出力分布の別角度)ので「両方測れば安心」方向に読むと過剰。simulated quant のため推論速度未測。
- **draft:s2**: 全 llcore 数値は 0.81M〜130M char-LM CPU self-report、実 LLM 性能の直接反証ではない。規模差はデモごとに違う(最小 char-LM 0.81M=Gemma 比 約15,000×、RAM 超実証は 130M=約92×。一律 15,000× と書くと型1 を自ら犯す)。Cosmos3(arXiv:2606.02800)は arXiv ID 実在のみ確認、技報 PDF 未取得=Table 10 スコア(79.3/47.2/57.8/58.2/73.7/77.5)は一次未照合・二次依拠。SmartInfra は数値未取得で空欄(取得失敗を明示)。他社数値はすべて self-report(信頼の梯子・段2)、第三者再現(段3)/独立査読(段4)は未。llcore も同じ段2 — 違いは規模と限定語を貼り続ける姿勢だけ。Gemma4「26B 級」は定量ベンチ表が一次に無い一般文で型3 でなく型1/型5 寄りとして扱う。int8 約3.9× は fp32 基準明記が必須(fp16 基準=業界標準なら約1.9×、黙ると自分が型1)。RAM 超実証は実機 avail RAM 約3.6GB の制約上「物理 RAM 総量超」でなく「working-set 上限<モデルサイズ」(限定を外すと自分が型5 の発生源)。2bit QAT top-1 retention 82.9% は strict cap-gate 97% 未達(隠せば型1)。simulated quant のため速度未測。

---

## 6. 多言語・チャネル方針(ja/en/zh/ko)

- **言語別に別記事**(1記事4セクション統合にしない)。海外サイトへその言語版をそのまま投稿するため(feedback_articles_per_language_separate)。各記事は ja を原本に en/zh/ko 版を別記事として展開。
- **チャネル割り当て**:
  - **Qiita(技術者向け)**: 量子化アーク / メモリ効率実機計測 / fail-closed エンジニアリング系。長文歓迎(2-3万字)、GitHub commit/file リンク・単独行 raw URL で OGP カード。
  - **dev.to(海外・技術者向け en)**: 競合地図・cherry-pick 解剖・クロスドメイン系の英語版(WAF UA 必須・api-key ヘッダ)。
  - **非エンジニア向け(QIITA_GENERAL / LinkedIn 等)**: honest disclosure・キャリア・哲学・弱者の兵法・家族OSS経済学系。比喩重視・かみくだき長め・専門用語は都度グロス。
- **公開前必須**: 連載トップ caveat バナー(極小 char-LM PoC・実 LLM を反証しない・全社数値 self-report・規模差)を全記事固定。arXiv ID(Cosmos=2606.02800)は執筆前に実在再確認。Qiita 画像/SVG は raw 絶対 URL + HTTP 200 確認(feedback_qiita_svg_path_and_cache)。