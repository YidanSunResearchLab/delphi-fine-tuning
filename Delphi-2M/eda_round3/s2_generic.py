"""Section 2 -- the generic component, applied identically to every variable that passed §1.

All the arithmetic lives in ordinal.py; this module is the loop plus the decision gates. The
report is organised BY SUBSECTION rather than by variable, because every gate in §2 is a
comparison across variables (which one has the least ceiling, which one actually moves
within a person) and a per-variable dump makes that comparison unreadable. The per-variable
detail is in eda_out/scales/<dataset>_<var>/.
"""
import numpy as np
import pandas as pd

import s1_detect
from common import (VARS, gate, md_table, report, save_table, summary_put)
from ordinal import RARE_THRESHOLDS, analyze_ordinal

_STATE = {}


def rows_frame():
    """The §2 result table, computed once per process."""
    if "rows" in _STATE:
        return _STATE["rows"], _STATE["panels"]
    panels = s1_detect.detect_only()
    rows, grps = [], {}
    for v in VARS:
        if v.status != "ok":
            continue
        r, grp = analyze_ordinal(v, panels[v.panel])
        rows.append(r)
        grps[v.key] = grp
    df = pd.DataFrame(rows)
    _STATE.update(rows=df, panels=panels, grps=grps)
    return df, panels


def embedding_plan(n_rare_100):
    if not np.isfinite(n_rare_100):
        return "unknown"
    if n_rare_100 >= 10:
        return "anchor interpolation"
    if n_rare_100 >= 5:
        return "smoothing regulariser"
    return "independent embeddings"


def run():
    df, panels = rows_frame()
    df = df.copy()
    df["var"] = df["dataset"] + ":" + df["varname"]
    save_table(df, "section2_generic")

    report("## §2 通用分析组件（每个变量同一套流程）\n")
    report(f"`ordinal.analyze_ordinal()` 对 §1 通过的 {len(df)} 个变量执行同一套 §2.1–§2.7，"
           "逐变量的图与表在 `eda_out/scales/<dataset>_<var>/`。下面按**小节**横向对比，"
           "因为 §2 的每一个决策门本质上都是变量之间的比较。")
    report("""**分档（levels）的处理**，三种情况，全部登记在 `section2_generic.csv` 的
`level_transform` 列：

- 取值本来就落在申报的档位上 → 原样使用（绝大多数变量）
- 连续变量（`cogn_global`）→ 切十分位，切点写进图的 source-data CSV
- 有档间取值（`cts_estmmse30`）→ 就近吸附到申报档位，**吸附比例单独报**，不静默吸收""")

    # ---------------- §2.1 -----------------------------------------------------------
    t = df[["var", "n_obs", "pct_missing", "n_persons_with_obs", "median_obs_per_person",
            "pct_persons_with_ge3_obs", "max_year_over_year_coverage_jump",
            "coverage_vs_visit_idx_spearman", "coverage_drop_over_visits"]]
    report("### §2.1 覆盖率与可用性\n")
    report(md_table(t, floatfmt="{:.3f}"))
    report("`median_obs_per_person` 是**每个有该变量的人被测了几次**，和覆盖率是两件事："
           "一个覆盖 36% 访视但每人只测 2 次的变量，能撑起的序列长度远不如覆盖率数字看上去的多。"
           "这一列在 §6 Q3 挑主干时是独立的一票。")
    step = df[df["max_year_over_year_coverage_jump"] > 0.20]
    report(gate(len(step) == 0,
                ("覆盖率随年份出现 ≥20 个百分点的阶跃：" +
                 "、".join(f"`{r.var}`（最大年际跳变 {r.max_year_over_year_coverage_jump:.0%}）"
                           for r in step.itertuples()) +
                 "。这是**结构性缺失**（UDS 表单在某一年换版），必须用显式 `<NA>` token，"
                 "禁止任何形式的插补——插补会把'这一年根本没考这个量表'伪装成'这个人考了但分数居中'。")
                if len(step) else "没有变量的年际覆盖率跳变超过 20 个百分点。"))
    drift = df[(df["coverage_vs_visit_idx_spearman"] < -0.5) & (df["coverage_drop_over_visits"] > 0.10)]
    report(gate(len(drift) == 0,
                ("覆盖率随访视序号单调下降：" +
                 "、".join(f"`{r.var}`（ρ={r.coverage_vs_visit_idx_spearman:.2f}，"
                           f"累计下降 {r.coverage_drop_over_visits:.0%}）" for r in drift.itertuples()) +
                 "。这是与脱落相关的**信息性缺失**，缺失本身携带预后信息，"
                 "必须作为特征显式暴露（missing indicator），不能只当噪声填掉。")
                if len(drift) else "没有变量的覆盖率随访视序号单调衰减（阈值 ρ<−0.5 且累计下降 >10pp）。"))

    # ---------------- §2.2 -----------------------------------------------------------
    t = df[["var", "n_distinct_values", "n_declared_levels", "n_unobserved_levels"]
           + [f"n_rare_{k}" for k in RARE_THRESHOLDS]
           + ["normalized_entropy", "max_category_pct"]].copy()
    t["embedding_plan"] = t["n_rare_100"].map(embedding_plan)
    report("### §2.2 逐取值清点（token 可行性）\n")
    report(md_table(t, floatfmt="{:.3f}"))
    report("`embedding_plan` 按规格书 §2.2 的三档阈值（稀有档 = n<100 的档位数）机械判定："
           "≥10 → 锚点插值，5–9 → 平滑正则，<5 → 独立 embedding。")
    lowinfo = df[(df["normalized_entropy"] < 0.3) | (df["max_category_pct"] > 70)]
    report(gate(len(lowinfo) == 0,
                ("近乎常数的 token：" +
                 "、".join(f"`{r.var}`（归一化熵 {r.normalized_entropy:.2f}，"
                           f"最大档 {r.max_category_pct:.1f}%）" for r in lowinfo.itertuples()) +
                 "。信息量极低，作为逐访视 token 基本只是在消耗序列长度；"
                 "降级为静态协变量或直接剔除。")
                if len(lowinfo) else "所有变量的归一化熵 ≥0.3 且最大档占比 ≤70%。"))

    # ---------------- §2.3 -----------------------------------------------------------
    t = df[["var", "direction_measured", "pct_at_floor", "pct_at_ceiling", "pct_bottom3",
            "pct_top3", "pct_at_worst_level", "pct_at_best_level"]]
    report("### §2.3 天花板与地板\n")
    report(md_table(t, floatfmt="{:.2f}"))
    report("`pct_at_worst_level` / `pct_at_best_level` 按**实测方向**（§2.7）挑选那一端，"
           "不是按列名或 codebook 猜的。")
    cf = df[df["pct_at_worst_level"] > 30]
    cf2 = df[df["pct_at_best_level"] > 30]
    report(gate(len(cf) == 0 and len(cf2) == 0,
                "占比 >30% 的极端档：" +
                ("最差端 " + "、".join(f"`{r.var}` {r.pct_at_worst_level:.1f}%" for r in cf.itertuples())
                 + "；" if len(cf) else "") +
                ("最好端 " + "、".join(f"`{r.var}` {r.pct_at_best_level:.1f}%" for r in cf2.itertuples())
                 if len(cf2) else "") +
                "。这些变量在对应区间**没有分辨力**——超过三成的观测挤在同一格，"
                "模型无法从它们身上区分这一段的病程。凡是同时被列为主干候选的，降为辅助。"
                if (len(cf) or len(cf2)) else "没有变量的任一端占比超过 30%。"))

    # ---------------- §2.4 -----------------------------------------------------------
    t = df[df["grain"] == "visit"][["var", "icc", "icc_raw", "n_adjacent_pairs",
                                    "median_distinct_levels_per_person",
                                    "pct_persons_never_changing", "rank_transition_rate",
                                    "pct_delta_zero", "pct_abs_delta_le1",
                                    "pct_delta_improving", "delta_sd"]]
    report("### §2.4 被试内变异\n")
    report(md_table(t, floatfmt="{:.2f}"))
    report("人级变量（`dcfdx_lv`/`cogdx`/`braaksc`/`ceradsc`）每人只有一个观测，"
           "§2.4/§2.5 在结构上不可算，留空而不是填 0。")
    stat = t[(t["icc"] > 0.85) & (t["rank_transition_rate"] < 10)]
    stuck = t[(t["rank_transition_rate"] < 10) | (t["pct_delta_zero"] > 40)]
    report(gate(len(stat) == 0,
                ("ICC>0.85 且档位变化率<10%：" +
                 "、".join(f"`{r.var}`" for r in stat.itertuples()) +
                 "。个体内基本不动，**不该进序列**，放序列开头作静态协变量。")
                if len(stat) else "没有变量同时满足 ICC>0.85 与档位变化率<10%。"))
    report(gate(len(stuck) == 0,
                ("序列里会出现大量重复 token：" +
                 "、".join(f"`{r.var}`（变化率 {r.rank_transition_rate:.1f}%，"
                           f"Δ=0 占 {r.pct_delta_zero:.1f}%）" for r in stuck.itertuples()) +
                 "。attention 在一串相同 token 上无从发挥，"
                 "这些变量**单独不足以作主干**，需配 delta token 或换更细的变量。")
                if len(stuck) else "所有逐访视变量的档位变化率 ≥10% 且 Δ=0 占比 ≤40%。"))

    # ---------------- §2.5 -----------------------------------------------------------
    t = df[df["grain"] == "visit"][["var", "delta_sd", "noise_floor_sd", "n_noise_pairs",
                                    "noise_floor_mean", "noise_floor_practice_shift", "snr"]]
    report("### §2.5 噪声底\n")
    report("噪声底子集 = 相邻两次访视**都处在最轻状态**的配对（NACC: 两次 `NACCUDSD`=1；"
           "RADC: 两次都未达 AD）。这些配对的 Δ 主要反映测量误差与练习效应，而不是病程。"
           "`snr` = 全样本 Δ 的标准差 ÷ 噪声底标准差。"
           "`noise_floor_practice_shift` 已按实测方向定向，正数 = 朝「变好」漂移。")
    report(md_table(t, floatfmt="{:.3f}"))
    circ = df[df["noise_floor_circular"].fillna(False).astype(bool)]
    if len(circ):
        report("**一个必须先排除的读法**：" +
               "、".join(f"`{r.var}`" for r in circ.itertuples()) +
               " 的噪声底子集正是**由这个变量自己**定义的（NACC 的最轻状态就是 "
               "`NACCUDSD`=1），所以 Δ 恒为 0、SNR 未定义。这不代表「噪声极低」，"
               "而是这个变量的噪声底在本设计下无法测量，需要外部重测数据才能估。")
    drift = t[t["noise_floor_mean"].abs() > 0.2]
    if len(drift):
        report("**另一个方向的偏移**：" +
               "、".join(f"`{r.var}`（噪声底均值 {r.noise_floor_mean:+.3f}/周期）"
                         for r in drift.itertuples()) +
               "。这个子集里的人两次访视都未达痴呆，却仍在稳定下滑——"
               "那是**正常老化**，不是测量误差。也就是说这里的「噪声底」混进了真实的年龄效应，"
               "把它当分母算出的 SNR 是**保守下界**，真实信噪比更高。"
               "反过来说，正数（朝变好漂移）才是练习效应的证据。")
    lowsnr = t[t["snr"] < 1.5]
    report(gate(len(lowsnr) == 0,
                ("信噪比 <1.5：" + "、".join(f"`{r.var}`（SNR {r.snr:.2f}）" for r in lowsnr.itertuples()) +
                 "。短期变化基本是噪声，**1 个单位的分辨率没有意义**："
                 "输出侧必须用距离敏感损失（不能用无序 cross-entropy），"
                 "评估也不得按单位精度报指标。")
                if len(lowsnr) else "所有逐访视变量的 SNR ≥1.5。"))
    prac = t[t["noise_floor_practice_shift"] > 0.05]
    report(gate(len(prac) == 0,
                ("噪声底均值朝「变好」方向显著偏移：" +
                 "、".join(f"`{r.var}`（{r.noise_floor_practice_shift:+.3f} 单位/次）"
                           for r in prac.itertuples()) +
                 "。存在**练习效应**：前 2–3 次访视的改善不能直接解释为病情好转，"
                 "把它当「好转」标签会教会模型一个不存在的康复模式。")
                if len(prac) else "噪声底均值没有明显的练习效应漂移（阈值 0.05 单位/次）。"))

    # ---------------- §2.6 -----------------------------------------------------------
    t = df[["var", "reference_outcome", "ord_n_levels_tested", "ord_spearman",
            "ord_spearman_obs", "ord_isotonic_residual_ratio", "ord_n_reversals",
            "ord_n_adjacent_pairs_tested", "ord_n_indistinguishable_adjacent_pairs",
            "ord_n_indistinguishable_holm", "smoothing_verdict"]]
    report("### §2.6 序数性检验（本文档的核心）\n")
    report("对每个取值 v 算 `P(参照结局 | 当前取值 = v)` 与 Wilson 95% CI，"
           "然后：(1) 档位秩与该概率的 Spearman；(2) 加权等渗回归的残差平方和占比；"
           "(3) 每一对相邻档位的结局概率差异检验（期望频数 ≥5 用带连续性校正的 χ²，否则 Fisher 精确）。"
           "`_holm` 列是 Holm 步降校正后仍不可分的对数。"
           "残差比用 4 位有效数字显示：多数变量的档位概率**本来就是单调的**，"
           "等渗回归几乎不需要改动它，所以残差比在 1e-6 量级，不是「恰好为 0」。")
    report(md_table(t, floatfmt="{:.4g}"))
    report("""判定规则直接照抄规格书 §2.6：

| 条件 | `smoothing_verdict` |
|---|---|
| \\|ρ\\| > 0.9 **且** 等渗残差比 < 0.1 | `ok` — 可施加序数平滑或锚点插值 |
| \\|ρ\\| ∈ [0.6, 0.9] | `caution` — 大体单调但有局部反转，需人工看反转位置 |
| \\|ρ\\| < 0.6 | `forbidden` — **序数假设不成立，禁止施加序数平滑** |""")
    ok = df[df["smoothing_verdict"] == "ok"]["var"].tolist()
    caut = df[df["smoothing_verdict"] == "caution"]["var"].tolist()
    forb = df[df["smoothing_verdict"] == "forbidden"]["var"].tolist()
    report(gate(len(forb) == 0,
                f"`ok` {len(ok)} 个（{'、'.join('`%s`' % x for x in ok) or '无'}）；"
                f"`caution` {len(caut)} 个（{'、'.join('`%s`' % x for x in caut) or '无'}）；"
                f"**`forbidden` {len(forb)} 个（{'、'.join('`%s`' % x for x in forb) or '无'}）**。"
                + ("后者按名义变量处理，或先做折叠（见 §3.8）。" if forb else "")))
    excess = df[df["resolution_excess"].fillna(False).astype(bool)]
    report(gate(len(excess) == 0,
                ("不可分的相邻对超过一半：" +
                 "、".join(f"`{r.var}`（{int(r.ord_n_indistinguishable_adjacent_pairs)}/"
                           f"{int(r.ord_n_adjacent_pairs_tested)}）" for r in excess.itertuples()) +
                 "。**分辨率过剩**：把不可分的相邻档合并，用合并后的档数建 embedding，"
                 "而不是原始取值数——否则大量参数花在统计上无法区分的档位上。")
                if len(excess) else "没有变量的不可分相邻对超过一半。"))

    # incident-vs-literal disagreement
    cmp = df[["var", "ord_spearman", "ord_incident_spearman",
              "ord_isotonic_residual_ratio", "ord_incident_isotonic_residual_ratio"]].copy()
    cmp["abs_rho_drop"] = cmp["ord_spearman"].abs() - cmp["ord_incident_spearman"].abs()
    report("#### 字面定义 vs incident 定义\n")
    report(md_table(cmp.dropna(subset=["ord_incident_spearman"]), floatfmt="{:.4g}"))
    dis = cmp[(cmp["abs_rho_drop"].abs() > 0.2)]
    report(gate(len(dis) == 0,
                ("两种参照结局给出实质不同的结论：" +
                 "、".join(f"`{r.var}`（|ρ| 从 {abs(r.ord_spearman):.2f} 变到 "
                           f"{abs(r.ord_incident_spearman):.2f}）" for r in dis.itertuples()) +
                 "。字面定义包含了 t 时刻已经痴呆的行，对严重度量表来说那部分测的是「痴呆持续」，"
                 "所以字面版本的单调性被高估。**这些变量的序数性结论必须标注是在哪个定义下成立的。**")
                if len(dis) else "两种定义下 |ρ| 的差异都在 0.2 以内，序数性结论不依赖参照结局的选择。"))

    # ---------------- §2.7 -----------------------------------------------------------
    t = df[["var", "direction_expected", "direction_measured", "spearman_with_outcome",
            "spearman_with_age", "direction_matches_spec"]]
    report("### §2.7 方向确认\n")
    report("不看 codebook，直接用数据定方向：该变量与参照结局的 Spearman 符号。")
    report(md_table(t, floatfmt="{:.3f}"))
    mis = df[~df["direction_matches_spec"].astype(bool)]
    report(gate(len(mis) == 0,
                ("**实测方向与 §1 预期不符：**" +
                 "、".join(f"`{r.var}`（预期 {r.direction_expected}，实测 {r.direction_measured}，"
                           f"ρ={r.spearman_with_outcome:+.3f}）" for r in mis.itertuples()) +
                 "。规格书要求立刻停下报告——见 §3.9 对这几个变量的专项核对。")
                if len(mis) else "全部变量的实测方向与 §1 表格一致。"))

    summary_put(
        s2_n_analyzed=len(df),
        s2_smoothing_ok=ok, s2_smoothing_caution=caut, s2_smoothing_forbidden=forb,
        s2_direction_mismatch={r.var: [r.direction_expected, r.direction_measured]
                               for r in mis.itertuples()},
        s2_low_snr=lowsnr["var"].tolist(),
        s2_ceiling_gt30=cf["var"].tolist() + cf2["var"].tolist(),
        s2_low_entropy=lowinfo["var"].tolist(),
        s2_resolution_excess=excess["var"].tolist(),
    )
    return df
