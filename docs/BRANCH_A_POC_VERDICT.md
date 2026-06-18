# branch A PoC VERDICT — メモリ効率 fitness × verified-plasticity gate(P1, 2026-06-18)

> 実行 = `scripts/poc_branch_a_memory_fitness.py`(2×2 × 20 seeds、falsification 15 seeds / honest 20 trials)。
> 出力 = `out/poc_branch_a_memory_fitness.json` / `out/poc_branch_a.log`。設計 = `docs/BRANCH_A_CAPABILITY_REMERGE_DESIGN.md`。
> **失敗・null を消さない**(ARTICLE_SEEDS #2 / capability アークの `passes=False` を消さなかった規律)。

## 実測(CopyTask 合成プローブ・state 力学 proxy = P1)

| cell(gate × fitness) | safe_rate(L<1) | mean_footprint | mean_retention | gate(Z3 稼働) |
|---|---|---|---|---|
| none × memory | 1.00 | **0.149** | 0.625 | — |
| contraction × memory | 1.00 | 0.169 | 0.632 | rej 71 / res 71 / fallback 0 |
| none × retention | 0.95 | 0.375 | 0.642 | — |
| contraction × retention | 1.00 | 0.371 | 0.638 | rej 196 / res 196 / fallback 0 |

falsification(gate なし・capability の問い): **memory passes=False(diff +0.0042)/ retention passes=True(diff +0.0116)**。

## G1–G6 判定

| ゲート | 結果 | 意味 |
|---|---|---|
| G1 後方互換(決定論) | **PASS** | `gate_mode="none"` の memory 探索は同 seed 再実行で byte-identical |
| G2 gate 効力 | **FAIL** | control も gated も safe_rate ≈ 1.00 = **sound gate が冗長**(下記★) |
| G3 fail-closed 健全性 | **PASS** | reject→resample は cap 内、fallback 0 = 空転せず admit |
| G4 トレードオフ可視化 | **PASS** | memory 目的が footprint を 0.375→**0.149** に下げる(memory 項が効く) |
| G5 honest 反証 | **記録のみ** | memory は random を超えない(passes=False)= **capability 取り戻さずの約束どおり** |
| G6 gate コスト | **PASS** | Z3 contraction 判定は安価・resample 爆発なし |

**functional_min(G1∧G2∧G3∧G6)= FAIL**(G2 が落ちたため)。**だがこの FAIL は欠陥でなく情報**(下記)。

## honest 知見(この PoC の本当の収穫)

1. **★G2 の FAIL こそ示唆的 = 「目的が gate を冗長にした」**。設計は「control ~6% safe vs gated ~100% safe」を期待したが、実測は **control も既に 95–100% 有界**。理由 = メモリ目的の footprint 項(=収縮率 L を下げる報酬)も、retention(CopyTask の遅延再現)も、**どちらも有界 gene を選好する**ため、sound gate を足しても safe_rate は retention で 0.95→1.00 と僅かに上がるだけ。**目的関数が既に安全側を報酬するなら、安全 gate は付加価値が出ない**。gate の価値は「発散を好む fitness(=メモリ/retention と直交 or 逆行する目的)」でのみ顕在化する。= 設計の G2 仮説を実測で**反証**(honest)。

2. **G4 はクリーンな勝ち、ただし代償も実測**: メモリ目的は footprint を**半減**(0.375→0.149)。代償 = (a) retention が 0.642→0.625 と僅かに低下、(b) **falsification の random 超え edge を喪失**(retention passes=True diff+0.012 → memory passes=False diff+0.004)。= **accuracy×memory トレードオフが、footprint だけでなく「random を超える capability edge」にも現れた**(memory 圧が小さな capability 優位を削った)。

3. **G5 の非対称は honest に記録(過大主張しない)**: retention-only は本 toy で random を有意に超えた(passes=True, diff+0.012)が、これは **CopyTask 合成プローブ・n_seeds=15・state proxy の小スケール**であって、実 SmolLM2 地形の capability=NULL_TIE/NEGATIVE([[project_llcore_evolvable_llm_replan_2026_06_09]])を覆すものではない。「toy で小さな edge が出た」止まり。memory 目的では passes=False = **branch A が約束どおり capability を取り戻さない**。

4. **gate は sound かつ安価に動いた(Z3 稼働)**: rej 71/196・fallback 0 = 検証器は実際に発散候補を弾き、cap 内で admissible 子を見つけ、既知安全 gene への fallback に頼らず回った。gate 機構そのものは健全。

## 結論(honest 立ち位置・POSITIONING §2(c) 準拠)

- branch A は設計どおり **capability を取り戻さない**(memory falsification passes=False)。価値は guarantee 側。
- **メモリ目的は機能する(G4)**が、**sound gate はメモリ目的の下では冗長(G2 FAIL)** = 「進化×sound gate×メモリ北極星」の四点結合の**実証的価値は当初想定より狭い**。gate が効くのは目的が安全側を報酬しない場合に限る、という**非自明な境界条件**を実測で確定した(これ自体が honest な貢献)。
- 大半は HW-NAS / 制約付き進化の再導出(prior-art confidence high)。PoC は「再導出の上に、gate の有効範囲という新しい計測を 1 つ足した」までが誠実な到達点。

## 留保
- P1 = footprint は state-boundedness proxy(収縮率 L)で**実 RSS footprint ではない**(実 footprint を fitness にする P2 は将来)。retention は fixed-readout probe fitness。
- CopyTask 合成・CPU smoke・n_seeds 20/15。toy スケール。
- 実装: `src/llcore/fitness/memory_objective.py` + `scripts/poc_branch_a_memory_fitness.py` + `tests/unit/test_branch_a_memory_fitness.py`(11 passed / ruff / mypy strict green、既存 837 非破壊)。
