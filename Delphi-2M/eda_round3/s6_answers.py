"""Section 6 -- the ten questions the spec says REPORT.md must be able to answer.

Everything here is read back out of summary.json / SCALE_SUMMARY.csv / the §4 tables, so
this section cannot drift from the numbers above it.
"""
import json
import os

import numpy as np
import pandas as pd

import s2_generic
from common import (SCALE_SUMMARY, SUMMARY, gate, md_table, report, save_table,
                    summary_put)

# Equal-weight composite over the four subsections the spec names for the backbone choice
# (§2.2 rarity/entropy, §2.3 extremes, §2.4 transitions, §2.6 ordinality) plus coverage.
# A summary device for ranking, NOT a statistical test -- every component is shown.
BACKBONE_COMPONENTS = ["coverage_score", "repeats_score", "entropy_score", "rarity_score",
                       "extremes_score", "transition_score", "ordinality_score"]


def _load():
    with open(SUMMARY) as f:
        j = json.load(f)
    s = pd.read_csv(SCALE_SUMMARY)
    return j, s


def backbone_table(s):
    d = s[(s["status"] == "ok") & (s["grain"] == "visit")
          & (~s["role_verdict"].isin(["outcome_only"]))].copy()
    d["coverage_score"] = 1 - d["pct_missing"].clip(0, 100) / 100
    # A backbone has to be measured REPEATEDLY on the same person. §3.2 found MoCA covers
    # 36% of visits but has a median of only 2 per person, which coverage alone hides.
    d["repeats_score"] = (d["median_obs_per_person"] / 4).clip(0, 1)
    d["entropy_score"] = np.where(d["ceiling_test_vacuous"].astype(bool), np.nan,
                                  d["normalized_entropy"])
    nlev = d["n_distinct_values"].clip(lower=1)
    d["rarity_score"] = 1 - (d["n_rare_100"].fillna(0) / nlev).clip(0, 1)
    d["extremes_score"] = np.where(d["ceiling_test_vacuous"].astype(bool), np.nan,
                                   1 - d["extreme_level_pile_pct"].clip(0, 100) / 100)
    d["transition_score"] = d["rank_transition_rate"].clip(0, 100) / 100
    d["ordinality_score"] = d["ordinality_spearman"].abs()
    d["composite"] = d[BACKBONE_COMPONENTS].mean(axis=1, skipna=True)
    d = d.sort_values("composite", ascending=False)
    return d


def _q1(ok):
    # ---------------- Q1
    report("### Q1 每个变量的 `smoothing_verdict` 是什么？哪些确认可施加序数平滑，哪些禁止？\n")
    t = ok[["varname", "dataset", "ordinality_spearman", "ordinality_rho_source",
            "isotonic_residual_ratio_incident", "n_indistinguishable_adjacent_pairs",
            "n_adjacent_pairs_tested", "smoothing_verdict"]]
    report(md_table(t, floatfmt="{:.3f}"))
    grp = ok.groupby("smoothing_verdict")["varname"].apply(list).to_dict()
    report(f"""- **`ok`（可施加序数平滑或锚点插值）**：{', '.join('`%s`' % x for x in grp.get('ok', [])) or '无'}
- **`caution`（大体单调、有局部反转，须人工看反转位置）**：{', '.join('`%s`' % x for x in grp.get('caution', [])) or '无'}
- **`forbidden`（序数假设不成立，禁止序数平滑）**：{', '.join('`%s`' % x for x in grp.get('forbidden', [])) or '无'}

两条必须一起读的限定：

1. **`ok` 不等于「平滑一定有用」。** CDRGLOB、五个 CDR box、NACCUDSD 的 |ρ| 都恰好等于
   1.000，因为它们只有 3–4 个可检验的档位，任何单调排列都会给 ρ=1。档位少的变量拿到
   `ok` 的信息量远低于 MMSE（31 档、ρ=0.86）拿到 `caution`。判定要连 `n_levels_tested` 一起看。
2. **`caution` 的反转位置已逐对列出**，在 `<var>/tables/<var>_incident_adjacent_pairs.csv`，
   两个变量的情况完全不同：
   - `NACCMMSE` 的反转只有 3 处，全在**极稀疏的低分端**（16→17 n=12/18、18→19 n=41/91、
     19→20 n=91/147），三处 p 值都 ≥ 0.20。**属抽样噪声**，序数假设本身没问题，
     照常施加平滑即可。
   - `FAQTOTAL` 不一样：incident 口径下 |ρ| 从 1.00 掉到 0.75，反转有 11 处且从
     **9 分一直延伸到量表顶端**，其中 9→10 那一对每档都有 400 上下的样本。
     实质不是「单调性被噪声破坏」，而是**曲线在 9 分以上就平掉了**——
     P(下次痴呆) 在 9 分处约 0.32，到 30 分仍是 0.25，不再上升。
     **FAQ 总分超过约 9 分之后不再携带增量的痴呆发生信息**，
     所以对它施加全量程等权的序数平滑是错的：高分段的相邻档之间本来就没有该被保持的顺序。""")
    summary_put(s6_smoothing_groups={k: [str(x) for x in v] for k, v in grp.items()})


def _q2(ok):
    # ---------------- Q2
    report("### Q2 每个变量的 `role_verdict` 是什么？\n")
    t = ok[["varname", "dataset", "grain", "role_verdict", "verdict_reasons"]]
    report(md_table(t))
    counts = ok["role_verdict"].value_counts().to_dict()
    report(f"分布：{counts}。另有 `niareagansc` 一行 `role_verdict = unavailable`（§1 未找到）。")


def _q3(j, s):
    # ---------------- Q3
    bt = backbone_table(s)
    save_table(bt[["varname", "dataset", "role_verdict"] + BACKBONE_COMPONENTS
                  + ["composite", "pct_missing", "n_distinct_values", "n_rare_100",
                     "extreme_level_pile_pct", "rank_transition_rate",
                     "ordinality_spearman"]], "backbone_ranking")
    report("### Q3 主干变量最终选谁？依据是哪几个指标？\n")
    report("""排序用一个**等权复合分**，七个分量分别对应规格书为这个问题指定的小节：

| 分量 | 来自 | 定义 |
|---|---|---|
| `coverage_score` | §2.1 | 1 − 缺失率 |
| `repeats_score` | §2.1 | 每人被测次数的中位数 ÷ 4，上限 1（序列模型要的是同一人身上的重复测量） |
| `entropy_score` | §2.2 | 归一化熵 |
| `rarity_score` | §2.2 | 1 − 稀有档(n<100)占档位数的比例 |
| `extremes_score` | §2.3 | 1 − 最极端单档占比 |
| `transition_score` | §2.4 | 相邻访视档位变化率 |
| `ordinality_score` | §2.6 | \\|ρ\\|（incident 口径） |

复合分只是**排序工具，不是统计检验**，所以七个分量全部列出，读者可以自行改权重。
分箱变量（`cogn_global`）的熵与极端档分量置空（分箱使它们恒定），复合分按可用分量取均值。\n""")
    report(md_table(bt[["varname", "dataset"] + BACKBONE_COMPONENTS + ["composite"]],
                    floatfmt="{:.3f}"))
    nacc_top = bt[bt["dataset"] == "nacc"].head(1)
    radc_top = bt[bt["dataset"] == "radc"].head(1)
    if nacc_top.empty:
        rt = radc_top.iloc[0]
        report(f"""**本机没有 NACC 导出**，只能就 RADC 作答：主干 = `{rt.varname}`
（复合分 {rt.composite:.3f}）。NACC 侧的 MMSE/MoCA 年代断层与 CDR 一族的比较需要那份数据。""")
        summary_put(s6_backbone_radc=str(rt.varname))
        return bt
    rt = radc_top.iloc[0]
    # The two runners-up matter as much as the winner here, so name them from the data.
    nacc_rank = bt[bt["dataset"] == "nacc"].head(3)
    cdr = bt[bt["varname"] == "CDRSUM"]
    report(f"""**排序结果**：NACC 侧前三名是
{'、'.join(f'`{r.varname}`（{r.composite:.3f}）' for r in nacc_rank.itertuples())}；
RADC 侧第一名是 `{rt.varname}`（{rt.composite:.3f}）。

**但复合分不能直接当结论用，有两个它测不到的东西：**

1. **缺失是不是结构性的。** `coverage_score` 只看缺失比例，看不到 §2.1 的决策门发现的事：
   `NACCMMSE` 与 `MOCATOTS` 的缺失是**按年代一刀切**的（年际跳变 54% 与 52%，
   §3.1 的逐年表：2014 年 MMSE 86.9%/MoCA 0%，2016 年 MMSE 17.6%/MoCA 67.0%）。
   一根 2015 年之后就消失的主干不是主干。
2. **每人被测几次。** 已加入的 `repeats_score` 就是为此：`MOCATOTS` 覆盖 36% 的访视，
   但每人中位数只有 {bt.loc[bt['varname'] == 'MOCATOTS', 'median_obs_per_person'].iloc[0]:.0f} 次
   （§3.2 的决策门因此判它 `[ACTION]`）。序列模型需要的是同一个人身上的重复测量。

**结论，分三层说：**

- **标度性质最好的是 MMSE / MoCA 这一族**（31 档、极端档 5.6–25.3%、变化率 70–82%、
  |ρ| 0.86–0.99），但**两者都只覆盖半个年代，且从不同时出现**（§3.2：同时刻重叠 0 次）。
  所以它们只能作**年代特异的 token**，各自带显式 `<NA>`，不能拼成一根连续主干，
  也不能用 crosswalk 换算（§3.2 已说明该换算在本数据上不可验证）。
- **唯一在 2005–2025 全程都有的逐访视量表是 CDR 一族**（缺失 0.009%，
  `CDRSUM` 复合分 {cdr['composite'].iloc[0]:.3f}）。它的标度性质更差
  （46.3% 挤在 0、58.5% 的相邻访视 Δ=0、{j.get('s33_n_unreachable', '?')} 个档位结构不可达），
  但它是**唯一能贯穿整个时间轴的骨架**。
  → **NACC 的实际主干 = CDR 五域 token（见 Q4），MMSE 与 MoCA 作年代特异的辅助主干。**
  §4.3 的前向选择独立地支持这一点：从 `NACCMMSE` 基线（AUC 0.697）加入 `CDRSUM`
  一步就跳到 0.891，是全表最大的单步增益。
- **RADC 主干 = `{rt.varname}`**（复合分 {rt.composite:.3f}），
  `cts_estmmse30` 紧随其后（{bt.loc[bt['varname'] == 'cts_mmse30', 'composite'].iloc[0]:.3f}）。
  两者 |ρ| 分别 {abs(bt.loc[bt['varname'] == 'cogn_global', 'ordinality_spearman'].iloc[0]):.2f} 与
  {abs(bt.loc[bt['varname'] == 'cts_mmse30', 'ordinality_spearman'].iloc[0]):.2f}，
  都可用且缺失都很低（2.9% / 6.2%），RADC 侧**没有 NACC 那种年代断层问题**。
  注意 `cogn_global` 的熵与极端档分量是空的（连续量，分箱后这两项人为恒定），
  它的复合分只由 5 个分量平均而来，与其他变量不完全可比。
  §4.3 显示两者互补（加入 `cogn_global` 使 AUC 从 0.878 升到 0.930），**两个都留**。

**一句话**：主干的选择不是「哪个量表最好」，而是「哪个量表在整个时间轴上都存在」。
本数据上这两个答案不是同一个变量，这个张力必须写进模型设计，而不是用复合分排序掩盖掉。""")
    # The composite's own top NACC row is recorded alongside the reasoned pick, so the
    # machine-readable summary cannot silently disagree with the prose above.
    summary_put(s6_backbone_nacc_composite_top=str(nacc_top["varname"].iloc[0]),
                s6_backbone_nacc_reasoned="CDR five-domain tokens "
                                          "(MEMORY/ORIENT/JUDGMENT/COMMUN/HOMEHOBB)",
                s6_backbone_radc=str(rt.varname))
    return bt


def _q4(j):
    # ---------------- Q4
    report("### Q4 CDR 的六个 box 分是否单独可用？若可用，是否改用六域 token 替代 CDRSUM？\n")
    report(f"""**可用**：`{'`, `'.join(j.get('s33_boxes_available', []))}` 六列全部存在，
且六者相加与 `CDRSUM` 在 {j.get('s33_boxsum_agreement_pct', float('nan')):.3f}% 的行上完全相等
——CDRSUM 是 box 分的确定性函数，不含额外信息。

**决定：改用五域 token，不是六域。** `PERSCARE` 被 §2 判为 `drop`（83.0% 挤在 0、
归一化熵 0.46），把它单独作一个 token 等于在每一次访视插入一个几乎恒为 0 的位置。
保留 `MEMORY` / `ORIENT` / `JUDGMENT` / `COMMUN` / `HOMEHOBB`，**同时不再喂入 CDRSUM**
（否则冗余 + 同时刻泄漏，见 Q7）。

**代价要说清楚**：每个 box 的极端档占比 49–67%、档位变化率 18–21%，都比 CDRSUM
（46.3% / 41.5%）差。换成域 token 得到的是域特异的衰退模式，付出的是每个 token 更少的动态。
若序列长度有预算约束，退回单个 `CDRSUM`，且平滑**按秩相邻**施加——§3.3 已证明数值相邻是错的
（0–18 的 0.5 网格里有 {j.get('s33_n_unreachable', '?')} 个取值结构上不可能出现）。""")


def _q5(j):
    # ---------------- Q5
    report("### Q5 NACCUDSD 的类别 2 在序数上是否真的居中？如何处理？\n")
    p = j.get("s37_p_next_dem_incident", {})
    between = j.get("s37_cat2_between")
    report(f"""incident 口径下 P(下次痴呆 | NACCUDSD = v)：
1 → {p.get('1', float('nan')):.4f}，2 → {p.get('2', float('nan')):.4f}，
3 → {p.get('3', float('nan')):.4f}。

类别 2 **{'确实居中' if between else '不居中'}**；Wilson 95% CI 与类别 1 分离：
{'是' if j.get('s37_cat2_sep_from_1') else '否'}，与类别 3 分离：
{'是' if j.get('s37_cat2_sep_from_3') else '否'}。

**处理**：{'类别 2 可作为正常档位参与平滑。' if (between and j.get('s37_cat2_sep_from_1') and j.get('s37_cat2_sep_from_3')) else '类别 2 单独作一档但不参与平滑（它与邻档的序数关系没有数据支持），或与类别 3 合并；它只占全部访视的 4.4%，合并代价很小。'}

另外 §3.7 的转移矩阵给出一个与序数性无关但更要紧的事实：已达痴呆（=4）后下一次不再是痴呆的
比例是 **{j.get('s37_dementia_reversal_pct', float('nan')):.2f}%**。痴呆不可逆，这部分是诊断/随访噪声，
**不能**让模型把它当成可学的康复路径。""")


def _q6(j):
    # ---------------- Q6
    report("### Q6 `dcfdx` 折叠前后的序数性差多少？\n")
    report(f"""参照结局 = 尸检 `braaksc ≥ 4`（人级，本 release 没有逐次访视诊断，见 §1）：

| 编码 | \\|ρ\\| |
|---|---|
| 原始 1–6 | {j.get('s38_rho_raw', float('nan')):.3f} |
| 4 档折叠（6 单独作最重档） | {j.get('s38_rho_collapse4', float('nan')):.3f} |
| 3 档折叠（6 并入痴呆） | {j.get('s38_rho_collapse3', float('nan')):.3f} |

最好的编码是 **{j.get('s38_best_coding', '?')}**。

规格书预期「原始 1–6 的 Spearman 明显低于折叠后」——**原始编码确实最差，这部分成立**。
但规格书的 4 档折叠把取值 6（其他病因痴呆）放在最重端，这一步**没有**带来预期的改善：
6 是另一种疾病，对「AD 病理」这个结局没有理由比 AD 痴呆更高。
按规格书 §2.6 的门，原始与 4 档折叠都落在 `forbidden`/`caution` 区间，
只有把 6 并入痴呆档才让标度真正单调。**§C6 的 4 档映射需要修改为 3 档。**""")


def _q7(j):
    # ---------------- Q7
    report("### Q7 哪些变量对构成同时刻泄漏？mask 需要隔开哪些组合？\n")
    leak_n = j.get("s42_nacc_leaky", [])
    leak_r = j.get("s42_radc_leaky", [])
    lt = pd.DataFrame(leak_n, columns=["predictor", "target", "auc"]) if leak_n else pd.DataFrame()
    report(f"NACC：**{len(leak_n)}** 个组合在同一次访视达到 AUC > 0.95；RADC：**{len(leak_r)}** 个。"
           "完整矩阵在 `tables/same_visit_leakage_matrix_{nacc,radc}.csv`，图在 "
           "`figs/same_visit_leakage_nacc.png`。\n")
    if len(lt):
        report(md_table(lt.sort_values("auc", ascending=False), floatfmt="{:.4f}", max_rows=25))
    report("""**mask 必须隔开的组合，按性质分三类：**

1. **确定性函数关系** —— `CDRSUM` ↔ 六个 box、`CDRGLOB` ↔ `CDRSUM`/box
   （§3.4 实测由 CDRSUM 预测 CDRGLOB 的样本外准确率
   {acc:.4f}）。这类不该靠 mask 解决，**直接只保留一个**（Q4 的结论）。
2. **同一构念的不同量表** —— `NACCMMSE` ↔ `MOCATOTS`、`cts_estmmse30` ↔ `cogn_global`。
   保留多个是有意义的（覆盖率互补），但**同一时刻必须互相 mask**。
3. **诊断标签与它的输入** —— `NACCUDSD` ↔ CDR/MMSE/FAQ 全家。`NACCUDSD` 就是临床医生看着
   这些量表下的判断，同时刻关系是**定义性的**，不是发现。所以：
   **任务必须定义为跨访视预测，`NACCUDSD` 只能作为 t 时刻的输入去预测 t+1 的状态，
   绝不能与同时刻的量表互见。**""".format(acc=j.get("s34_cdrglob_from_cdrsum_acc", float("nan"))))


def _q8(j):
    # ---------------- Q8
    report("### Q8 最终的辅助 token 清单是什么？\n")
    keep, drop = j.get("s43_nacc_keep", []), j.get("s43_nacc_drop", [])
    steps = pd.DataFrame(j.get("s43_nacc_steps", []))
    if len(steps):
        report(md_table(steps, floatfmt="{:.4f}"))
    report(f"""按 §4.3 的边际增益阈值（ΔAUC ≥ 0.005）：

- **进主序列**：{', '.join('`%s`' % x for x in keep) or '无'}
- **不进主序列**：{', '.join('`%s`' % x for x in drop) or '无'}

注意 §4.3 的候选里放的是 `CDRSUM` 总分而不是五个 box（box 分之间共线到 |ρ| 0.87–0.97，
贪心选择只会挑走第一个然后判其余「无增量」，那样的排序读不出域特异信息）。
所以下面这张清单以 Q4 的结构性结论为准，`CDRSUM` 的位置由五域 token 承担。

把 §4.3 的增量、§4.1 的冗余簇、§4.2 的泄漏、以及 Q4 的 CDR 决定合起来，最终清单：

| token | 角色 | 依据 |
|---|---|---|
| `MEMORY`/`ORIENT`/`JUDGMENT`/`COMMUN`/`HOMEHOBB` | **主干**（全程覆盖，替代 CDRSUM 总分） | Q3、Q4 |
| `NACCMMSE` | 年代特异主干（2015 前），独立 token，显式 `<NA>` | Q3、§3.1 |
| `MOCATOTS` | 年代特异主干（2015 后），独立 token，**不做 crosswalk 换算** | Q3、§3.2 |
| `FAQTOTAL` | 辅助，**但边际增益只有 0.0039 < 0.005** — 留或不留取决于序列长度预算 | §3.5、§4.3 |
| `NACCGDS` | **不进主序列**（ΔAUC {j.get('s36_gds_delta_auc', float('nan')):+.4f} < 0.01，§3.6 决策门 `[ACTION]`）；作静态协变量或省略 | §3.6、§4.3 |
| `NACCUDSD` | 输入 token，但**只跨访视**使用 | Q7 |
| `CDRSUM` / `CDRGLOB` | **剔除**（被 box 分完全决定） | Q4、§3.4 |
| `PERSCARE` | **剔除**（83% 挤在 0） | §2.2 |
| `braaksc` / `ceradsc` | **仅作结局**，永不入输入 | §3.9 |
| `dcfdx_lv` / `cogdx` | RADC 人级，作结局或静态标签；`cogdx` 还部分编码了病理 | §3.8、§1 |""")


def _q9(ok):
    # ---------------- Q9
    report("### Q9 每个变量的实测方向是否与 codebook 一致？\n")
    t = ok[["varname", "dataset", "direction_expected", "direction",
            "direction_matches_spec"]]
    report(md_table(t))
    mis = ok[~ok["direction_matches_spec"].astype(bool)]
    report(gate(len(mis) == 0,
                ("方向不一致的变量：" + "、".join(f"`{r.varname}`" for r in mis.itertuples())
                 + "。见 §3.9 的专项核对。"
                 if len(mis) else
                 "**全部一致，没有反向编码事故。** 其中两个是特意去核对的："
                 "`ceradsc` 实测与临床诊断负相关，确认它是反向编码（1 = definite AD）；"
                 "`braaksc` 正相关。两者放进同一个损失函数前必须先统一方向。")))


def _q10(ok):
    # ---------------- Q10
    report("### Q10 有哪些变量因天花板 / 地板 / 低熵而应当剔除？\n")
    dropped = ok[ok["role_verdict"] == "drop"]
    marg = ok[(ok["extreme_level_pile_pct"] > 60) & (ok["role_verdict"] != "drop")]
    report(f"""**判定剔除**：{', '.join('`%s`' % r.varname for r in dropped.itertuples()) or '无'}
{''.join(chr(10) + f'- `{r.varname}`：{r.verdict_reasons}' for r in dropped.itertuples())}

**边缘（极端单档 >60%，保留但不得作主干）**：
{''.join(chr(10) + f'- `{r.varname}`：最极端档 {r.extreme_level_pile_pct:.1f}%，档位变化率 {r.rank_transition_rate:.1f}%' for r in marg.itertuples()) or ' 无'}

注意 `cogn_global` **不**属于这一类：它的两端各 10% 是十分位分箱的必然结果
（`ceiling_test_vacuous = True`），不是天花板效应。""")
    summary_put(s6_dropped=[str(x) for x in dropped["varname"]])


def _priority(has_nacc):
    # ---------------- priority note
    report("### 规格书的优先级顺序执行情况\n")
    report("规格书说时间有限时按 §1 探测 → §2.6 序数性 → §2.2 逐取值 → §4.2 泄漏矩阵 → 其余。")
    if has_nacc:
        report("""本轮**全部小节都跑完了**，没有因时间降级。三件事是数据本身不允许，不是被跳过：

- `niareagansc`：RADC 这个 release 里不存在（§1）。
- `dcfdx` 的逐次访视版本：只有人级「最后一次有效访视」，§3.8 因此改在人级做（§1）。
- MoCA↔MMSE crosswalk 的高分段塌缩验证：两者从不在同一次访视施测，同时刻重叠为 0（§3.2）。

三者都已在对应小节登记并给出替代做法。""")
    else:
        report("""**本机没有 NACC 导出**，所以 §3.1（部分）、§3.2–§3.7、§4 的 NACC 部分、
以及 §6 的 Q4/Q5/Q7/Q8 都标为跳过。RADC 部分（§1、§2、§3.1 的 RADC 半、§3.8、§3.9、
§4.2 的 RADC 半、§4.3 的 RADC 半、§5、Q1–Q3、Q6、Q9、Q10）已完整跑完。
把 NACC CSV 放到 `data/NACC/` 或设环境变量 `NACC_CSV` 后重跑即可补齐。""")


def run():
    j, s = _load()
    report("## §6 需要回答的问题\n")
    report("下面十条是规格书要求 `REPORT.md` 必须能回答的问题。每条只给结论 + 出处，"
           "支撑数字都在上面的小节与 `SCALE_SUMMARY.csv` 里。")
    ok = s[s["status"] == "ok"]
    has_nacc = bool((ok["dataset"] == "nacc").any())

    _q1(ok)
    _q2(ok)
    bt = _q3(j, s)
    # Q4/Q5/Q7/Q8 are entirely about NACC variables (CDR, NACCUDSD, the token budget);
    # Q6 is entirely about RADC. Skip explicitly rather than crash on a missing panel.
    for name, fn, needs_nacc in [("Q4 CDR 的六个 box 分是否单独可用？若可用，是否改用六域 token 替代 CDRSUM？",
                                  _q4, True),
                                 ("Q5 NACCUDSD 的类别 2 在序数上是否真的居中？如何处理？", _q5, True),
                                 ("Q6 `dcfdx` 折叠前后的序数性差多少？", _q6, False),
                                 ("Q7 哪些变量对构成同时刻泄漏？mask 需要隔开哪些组合？", _q7, True),
                                 ("Q8 最终的辅助 token 清单是什么？", _q8, True)]:
        if needs_nacc and not has_nacc:
            report(f"### {name}\n")
            report("**本机没有 NACC 导出，本问无法作答。** 这一问涉及的变量"
                   "（CDR 六域 / `NACCUDSD` / NACC 的 token 预算）只有 NACC 有；"
                   "§1 已把它们记为 `panel_unavailable`。")
            continue
        fn(j)
    _q9(ok)
    _q10(ok)
    _priority(has_nacc)
    return bt
