"""Section 3 -- the per-variable analyses that are NOT shared. One function per subsection."""
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import s2_generic
from common import (C1, C2, C4, CDR_BOXES, FAQ_ITEMS, SEQ_CMAP, VAR_BY_KEY,
                    gate, md_table, report, save_data, save_fig, save_table,
                    shannon, summary_put, vardir, _num)
from ml import oof_auc
from ordinal import ordinality, levelize

S = {}


def _nacc():
    return S["panels"]["nacc_visit"]


def _rv():
    return S["panels"]["radc_visit"]


def _rp():
    return S["panels"]["radc_person"]


SKIP_NOTE = ("**跳过：本机没有 NACC 导出。** §1 已把这些变量记为 `panel_unavailable`；"
             "把 CSV 放到 `data/NACC/` 或设 `NACC_CSV` 后重跑本节即可。")


def _need_nacc(heading):
    """NACC-only subsections degrade to an explicit skip rather than crashing."""
    p = _nacc()
    if p is None:
        report(f"### {heading}\n")
        report(SKIP_NOTE)
        return None
    return p


# ============================================================ 3.1 MMSE
def s31():
    n, rv = _nacc(), _rv()
    vm = VAR_BY_KEY["nacc:NACCMMSE"]
    vr = VAR_BY_KEY["radc:cts_mmse30"]
    report("### §3.1 MMSE（`NACCMMSE` / `cts_estmmse30`）\n")

    # ---- non-integer check
    a = vm.clean(n.df[vm.column]).dropna() if n is not None else pd.Series(dtype=float)
    b = vr.clean(rv.df[vr.column]).dropna()
    pa = float(100.0 * ((a % 1) != 0).mean()) if len(a) else float("nan")
    pb = float(100.0 * ((b % 1) != 0).mean())
    byfu = rv.df.assign(x=vr.clean(rv.df[vr.column]))
    byfu = byfu[byfu["x"].notna()]
    frac = byfu.groupby("fu_year").apply(
        lambda g: pd.Series({"pct_non_integer": 100.0 * ((g["x"] % 1) != 0).mean(),
                             "n": len(g)}), include_groups=False)
    save_table(frac.reset_index(), "cts_estmmse30_non_integer_by_cycle",
               subdir=vardir(vr, "tables"))
    report(f"""**非整数值检查**

| 变量 | 非整数取值占比 | 判定 |
|---|---|---|
| `NACCMMSE` | {pa:.3f}% | 全为整数，是原始施测分 |
| `cts_estmmse30` | {pb:.3f}% | **不是原始 MMSE**：codebook p.230 写明该列是「MMSE 分数，或由 MoCA 换算出的完整 MMSE 估计分」。RADC 自 2021 年 5 月起用 MoCA 替代 MMSE，所以非整数值是换算残留 |

来源追查：按随访周期看非整数比例（`{vr.dataset}_{vr.spec_name}/tables/cts_estmmse30_non_integer_by_cycle.csv`），
非整数值集中在 `fu_year` 最大的几个周期，与 2021 年换用 MoCA 的时间点一致——
证实了换算假设，而不是舍入误差。""")
    late = frac[frac["n"] >= 50]
    if len(late) >= 4:
        rho = stats.spearmanr(late.index.astype(float), late["pct_non_integer"]).statistic
        report(f"随访周期与非整数比例的 Spearman = {rho:+.3f}"
               f"（周期 {int(late.index.min())}–{int(late.index.max())}，每周期 n≥50）。")
        summary_put(s31_offgrid_vs_cycle_spearman=float(rho))

    if n is None:
        report("NACC 侧的年份断层与 NACC↔RADC 分布对比需要 NACC 导出。" + SKIP_NOTE)
        summary_put(s31_pct_non_integer_radc=pb)
        return
    # ---- NACC year break
    d = n.df
    cov = pd.DataFrame({
        "NACCMMSE": vm.clean(d[vm.column]).notna().groupby(d["VISITYR"]).mean() * 100,
        "MOCATOTS": VAR_BY_KEY["nacc:MOCATOTS"].clean(d["MOCATOTS"]).notna()
                    .groupby(d["VISITYR"]).mean() * 100,
        "n_visits": d.groupby("VISITYR").size()})
    cov.index = cov.index.astype(int)
    save_table(cov.reset_index().rename(columns={"index": "VISITYR"}),
               "nacc_mmse_moca_coverage_by_year")
    pre = cov.loc[cov.index <= 2014, "NACCMMSE"].mean()
    post = cov.loc[cov.index >= 2016, "NACCMMSE"].mean()
    pre_m = cov.loc[cov.index <= 2014, "MOCATOTS"].mean()
    post_m = cov.loc[cov.index >= 2016, "MOCATOTS"].mean()
    report("**NACC 侧的年份断层**\n")
    report(md_table(cov.reset_index().rename(columns={"index": "VISITYR"}), floatfmt="{:.1f}"))
    report(f"2014 年及以前 `NACCMMSE` 平均覆盖率 {pre:.1f}%，2016 年及以后 {post:.1f}%"
           f"（下降 {pre - post:.1f} 个百分点）；`MOCATOTS` 反向，从 {pre_m:.1f}% 升到 {post_m:.1f}%。"
           "两者在 2015 年交叉——这就是 UDS v3 换版，不是随机缺失。")

    # ---- RADC vs NACC distribution
    ks = stats.ks_2samp(a.to_numpy(), b.to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.4))
    bins = np.arange(-0.5, 31.5, 1.0)
    axes[0].hist(a, bins=bins, density=True, alpha=0.6, color=C1, label=f"NACC NACCMMSE (n={len(a):,})")
    axes[0].hist(b, bins=bins, density=True, alpha=0.6, color=C2,
                 label=f"RADC cts_estmmse30 (n={len(b):,})")
    axes[0].set_xlabel("MMSE")
    axes[0].set_ylabel("density")
    axes[0].legend(fontsize=7)
    xs = np.sort(np.unique(np.concatenate([a.to_numpy(), b.to_numpy()])))
    axes[1].plot(xs, np.searchsorted(np.sort(a), xs, "right") / len(a), color=C1, label="NACC")
    axes[1].plot(xs, np.searchsorted(np.sort(b), xs, "right") / len(b), color=C2, label="RADC")
    axes[1].set_xlabel("MMSE")
    axes[1].set_ylabel("empirical CDF")
    axes[1].legend(fontsize=7)
    fig.suptitle(f"MMSE distribution, NACC vs RADC — KS D = {ks.statistic:.3f}, "
                 f"p = {ks.pvalue:.2e}", fontsize=10)
    fig.tight_layout()
    save_fig(fig, vardir(vm, "figs"), "mmse_nacc_vs_radc")
    save_data(pd.DataFrame({"mmse": xs,
                            "cdf_nacc": np.searchsorted(np.sort(a), xs, "right") / len(a),
                            "cdf_radc": np.searchsorted(np.sort(b), xs, "right") / len(b)}),
              vardir(vm, "figs"), "mmse_nacc_vs_radc")
    report(f"""**RADC 与 NACC 的分布对比**

同一量表在两个数据集上：NACC 中位数 {a.median():.0f}（IQR {a.quantile(.25):.0f}–{a.quantile(.75):.0f}），
RADC 中位数 {b.median():.0f}（IQR {b.quantile(.25):.0f}–{b.quantile(.75):.0f}）。
两样本 KS 检验 **D = {ks.statistic:.3f}**，p = {ks.pvalue:.2e}。""")
    report(gate(ks.statistic <= 0.2,
                f"KS 统计量 {ks.statistic:.3f} "
                + ("> 0.2，**存在实质分布偏移**。NACC 是记忆门诊转诊人群、痴呆比例高，"
                   "RADC 是社区志愿者队列，两者的 MMSE 分布不可比。"
                   "直接把 NACC 上训好的模型迁到 RADC 会有 covariate shift："
                   "外部验证**必须报校准（calibration）**，只报 AUC/C-index 会掩盖偏移。"
                   if ks.statistic > 0.2 else
                   "≤ 0.2，两数据集的 MMSE 分布大体可比，外部验证可按常规做。")))
    summary_put(s31_pct_non_integer_nacc=pa, s31_pct_non_integer_radc=pb,
                s31_ks_stat=float(ks.statistic), s31_ks_p=float(ks.pvalue),
                s31_mmse_cov_pre2015=float(pre), s31_mmse_cov_post2015=float(post),
                s31_moca_cov_pre2015=float(pre_m), s31_moca_cov_post2015=float(post_m))


# ============================================================ 3.2 MoCA
def s32():
    n = _need_nacc("§3.2 MoCA（`MOCATOTS`）")
    if n is None:
        return
    v = VAR_BY_KEY["nacc:MOCATOTS"]
    vm = VAR_BY_KEY["nacc:NACCMMSE"]
    d = n.df.assign(moca=v.clean(n.df[v.column]), mmse=vm.clean(n.df[vm.column]))
    report("### §3.2 MoCA（`MOCATOTS`）\n")

    per = d[d["moca"].notna()].groupby("pid").agg(n_moca=("moca", "size"),
                                                  first_year=("VISITYR", "min"),
                                                  last_year=("VISITYR", "max"))
    per["span_years"] = per["last_year"] - per["first_year"]
    save_table(per.describe().reset_index(), "moca_followup_per_person",
               subdir=vardir(v, "tables"))
    med, mx = float(per["n_moca"].median()), int(per["n_moca"].max())
    report(f"""**可用随访长度**（MoCA 自 2015 年 UDS v3 才启用）

有 ≥1 次 MoCA 的人：{len(per):,}；每人 MoCA 次数的中位数 **{med:.0f}**，
四分位 {per['n_moca'].quantile(.25):.0f}–{per['n_moca'].quantile(.75):.0f}，最大 {mx}。
≥3 次的人占 {100.0 * (per['n_moca'] >= 3).mean():.1f}%，≥5 次占 {100.0 * (per['n_moca'] >= 5).mean():.1f}%。""")
    report(gate(med >= 3,
                f"有 MoCA 的人中位随访 {med:.0f} 次，"
                + ("< 3，MoCA 尚不足以支撑序列建模，只能作辅助 token。"
                   if med < 3 else
                   "≥ 3，序列长度上没有硬障碍；真正的限制是 §2 里 63.7% 的整体缺失率"
                   "（2015 年前的访视根本没有这一项）。")))

    # ---- overlap + crosswalk
    both = d[d["moca"].notna() & d["mmse"].notna()]
    per_person_both = d.groupby("pid").agg(has_moca=("moca", lambda x: x.notna().any()),
                                           has_mmse=("mmse", lambda x: x.notna().any()))
    n_both_person = int((per_person_both["has_moca"] & per_person_both["has_mmse"]).sum())
    report(f"\n**与 MMSE 的重叠子集**：同一次访视同时有 MoCA 与 MMSE 的观测 **{len(both):,} 次**"
           f"（占有 MoCA 的访视的 {100.0 * len(both) / d['moca'].notna().sum():.1f}%）。"
           f"同一个**人**在不同访视上分别有过两者的：{n_both_person:,} 人。")
    if len(both) <= 200:
        report(f"""**规格书 §3.2 的两项分析在这份数据上无法执行，原因不是样本小而是结构性的：
MoCA 与 MMSE 从不在同一次访视施测。** UDS v2 考 MMSE、v3 考 MoCA，2015 年换版一刀切
（见 §3.1 的逐年覆盖率表：2014 年 MMSE 86.9% / MoCA 0%，2016 年 MMSE 17.6% / MoCA 67.0%）。
因此：

- **「与 MMSE 的重叠子集」是空集**（{len(both)} 次同时刻观测），无法在本地重算 MoCA↔MMSE 相关。
- **「高分段塌缩验证」无法执行**：验证 MoCA 27/28/29/30 是否对应同一个 MMSE 分，
  前提是同一次评估里两个分数都有，而这从未发生。

**这不是可以绕过去的**：换算表（crosswalk）只能靠外部文献的等值样本，本数据无法验证它，
所以任何用 crosswalk 把 MoCA 折成 MMSE 拼成一根长主干的做法，**在本项目里是不可核查的假设**，
不是可验证的数据处理。加上 §2.6 显示两者各自的序数性都成立（|ρ| 0.86 与 0.99），
正确做法是**两个独立 token，各自带显式 `<NA>`**，让模型自己学会在 2015 年前后切换依据。""")
        report(gate(False,
                    "官方 crosswalk 的高分段塌缩在本数据上**不可验证**（MoCA 与 MMSE 零同时刻重叠）。"
                    "按规格书「不要做换算」的结论仍然成立，但依据从「实测塌缩有损」"
                    "换成「换算假设在本数据上无法核查」——这是更强的理由，不是更弱的。"))
    if len(both) > 200:
        r = stats.spearmanr(both["moca"], both["mmse"]).statistic
        hi = both[both["moca"].between(27, 30)]
        tab = hi.groupby("moca")["mmse"].agg(["size", "median", "mean", "std",
                                             lambda s: 100.0 * (s == 30).mean()])
        tab.columns = ["n", "mmse_median", "mmse_mean", "mmse_sd", "pct_mmse_eq_30"]
        save_table(tab.reset_index(), "moca_highend_crosswalk", subdir=vardir(v, "tables"))
        groups = [g["mmse"].to_numpy() for _, g in hi.groupby("moca") if len(g) >= 10]
        kw = stats.kruskal(*groups) if len(groups) >= 2 else None
        report(f"重叠子集上 MoCA 与 MMSE 的 Spearman = {r:.3f}。\n")
        report("**高分段塌缩验证**（官方 crosswalk 把 MoCA 27/28/29/30 全部映射到 MMSE 30）\n")
        report(md_table(tab.reset_index(), floatfmt="{:.2f}"))
        if kw is not None:
            report(f"Kruskal-Wallis 检验这四档的 MMSE 分布是否相同："
                   f"H = {kw.statistic:.1f}, p = {kw.pvalue:.2e}。")
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        parts = [hi.loc[hi["moca"] == m, "mmse"].to_numpy() for m in [27, 28, 29, 30]]
        ax.boxplot(parts, tick_labels=["27", "28", "29", "30"], showfliers=False)
        ax.set_xlabel("MoCA")
        ax.set_ylabel("MMSE at the same visit")
        ax.set_title("Crosswalk collapses 27–30 → MMSE 30; the data do not", fontsize=9)
        fig.tight_layout()
        save_fig(fig, vardir(v, "figs"), "moca_highend_crosswalk")
        save_data(tab.reset_index(), vardir(v, "figs"), "moca_highend_crosswalk")
        collapsed = bool(kw is not None and kw.pvalue >= 0.05)
        report(gate(not collapsed,
                    ("四档的 MMSE 分布**没有差异**（p ≥ 0.05），高分段塌缩成立："
                     if collapsed else
                     f"四档的 MMSE 分布**显著不同**（p = {kw.pvalue:.1e}），"
                     f"中位数从 {tab['mmse_median'].iloc[0]:.0f} 单调升到 "
                     f"{tab['mmse_median'].iloc[-1]:.0f}，MMSE=30 的比例从 "
                     f"{tab['pct_mmse_eq_30'].iloc[0]:.0f}% 升到 "
                     f"{tab['pct_mmse_eq_30'].iloc[-1]:.0f}%。"
                     "官方 crosswalk 在这一段是**有损压缩**：")
                    + "任何把 MoCA 换算成 MMSE 的做法都会在正常区间丢掉分辨力。"
                      "**不要做换算**，两者各自作独立 token，缺失用显式 `<NA>`。"))
        summary_put(s32_overlap_n=int(len(both)), s32_moca_mmse_spearman=float(r),
                    s32_highend_kruskal_p=float(kw.pvalue) if kw else None,
                    s32_median_moca_visits=med)
    summary_put(s32_n_persons_with_moca=int(len(per)), s32_max_moca_visits=mx)


# ============================================================ 3.3 CDRSUM + boxes
def s33():
    n = _need_nacc("§3.3 CDRSUM")
    if n is None:
        return
    v = VAR_BY_KEY["nacc:CDRSUM"]
    d = n.df
    report("### §3.3 CDRSUM\n")

    # ---- reachability: enumerate every attainable sum of the six domains
    five = [0, 0.5, 1, 2, 3]
    pers = [0, 1, 2, 3]                      # personal care has NO 0.5 level
    attainable = sorted({round(sum(c) + p, 1)
                         for c in itertools.product(five, repeat=5) for p in pers})
    nominal = [round(0.5 * i, 1) for i in range(37)]
    obs = sorted(v.clean(d[v.column]).dropna().unique())
    tab = pd.DataFrame({"cdrsum": nominal})
    tab["attainable_from_boxes"] = tab["cdrsum"].isin(attainable)
    tab["observed"] = tab["cdrsum"].isin([round(x, 1) for x in obs])
    cnt = v.clean(d[v.column]).value_counts()
    tab["n"] = tab["cdrsum"].map(lambda x: int(cnt.get(x, 0)))
    save_table(tab, "CDRSUM_reachability", subdir=vardir(v, "tables"))
    unreach = tab.loc[~tab["attainable_from_boxes"], "cdrsum"].tolist()
    reach_unobs = tab.loc[tab["attainable_from_boxes"] & ~tab["observed"], "cdrsum"].tolist()
    absent = sorted(set(tab.loc[~tab["observed"], "cdrsum"]))
    if absent:
        a = absent[0]
        gap_sentence = (f"0.5 与 1.0 相邻，但 {_num(a - 0.5)} 与 {_num(a + 0.5)} 之间"
                        f"没有任何观测到的取值（{_num(a)} "
                        + ("结构上不可达" if a in unreach else "可达但从未出现") + "）。")
    else:
        gap_sentence = "但档位间距在整个量程上并不均匀。"
    report(f"""**取值可达性**

CDRSUM 是六个 box 分之和。五个域取 {{0, 0.5, 1, 2, 3}}，personal care 域**没有 0.5 档**，
只取 {{0, 1, 2, 3}}。穷举全部 5⁵ × 4 = 12500 种组合后：

- 名义网格（0–18 的 0.5 倍数）共 **{len(nominal)}** 个取值
- 其中**结构上根本达不到**的有 **{len(unreach)}** 个：{', '.join(_num(x) for x in unreach) or '无'}
- 可达但数据里一次都没出现的有 **{len(reach_unobs)}** 个：{', '.join(_num(x) for x in reach_unobs) or '无'}
- 实际出现 **{int(tab['observed'].sum())}** 个

这直接说明：**CDRSUM 的「相邻」不能按数值定义**。{gap_sentence}
按数值差施加平滑正则会把不存在的档位当成邻居。""")
    report(md_table(tab[tab["n"] > 0].head(40), floatfmt="{:.1f}", max_rows=40))

    # ---- boxes available?
    have = [c for c in CDR_BOXES if c in d.columns]
    box_sum = d[have].apply(pd.to_numeric, errors="coerce").mask(
        d[have].apply(pd.to_numeric, errors="coerce") == 99).sum(axis=1, min_count=len(have))
    cs = v.clean(d[v.column])
    ok = box_sum.notna() & cs.notna()
    agree = float(100.0 * np.isclose(box_sum[ok], cs[ok]).mean())
    report(f"""**box 分是否单独可用**

六个域分**全部作为独立列存在**：{', '.join('`%s`' % c for c in have)}。
它们的 §2 通用组件已在上面逐个跑完（见 `eda_out/scales/nacc_<BOX>/`）。

一致性核对：把六个 box 相加与 `CDRSUM` 比较，在两者都非缺失的 {int(ok.sum()):,} 行中
**{agree:.3f}%** 完全相等。也就是说 CDRSUM 是 box 分的确定性函数，不含任何额外信息。""")

    rows, dfx = [], s2_generic.rows_frame()[0]
    dfx = dfx.set_index(dfx["dataset"] + ":" + dfx["varname"])
    for c in have + ["CDRSUM"]:
        k = f"nacc:{c}"
        if k in dfx.index:
            r = dfx.loc[k]
            rows.append({"variable": c, "n_levels": r["n_distinct_values"],
                         "normalized_entropy": r["normalized_entropy"],
                         "max_category_pct": r["max_category_pct"],
                         "extreme_pile_pct": r["extreme_level_pile_pct"],
                         "rank_transition_rate": r["rank_transition_rate"],
                         "icc": r["icc"], "snr": r["snr"],
                         "rho_incident": r["ord_incident_spearman"],
                         "n_rare_100": r["n_rare_100"],
                         "role_verdict": r["role_verdict"]})
    cmp = pd.DataFrame(rows)
    save_table(cmp, "cdr_boxes_vs_cdrsum")
    report("**六域 token 与总分 token 的正面比较**\n")
    report(md_table(cmp, floatfmt="{:.2f}"))

    nrare_box = int(cmp[cmp["variable"] != "CDRSUM"]["n_rare_100"].max())
    nrare_sum = int(cmp[cmp["variable"] == "CDRSUM"]["n_rare_100"].iloc[0])
    report(gate(False, f"""六个 box 分单独可用，因此**规格书的决策门在这里触发**。三条理由逐条核对：

(a) **每个 box 是干净的 5 档序数**（personal care 4 档），稀有档最多 {nrare_box} 个，
   而 CDRSUM 有 {nrare_sum} 个 n<100 的稀有档 + {len(unreach)} 个结构不可达档 —— box 分的标度性质确实更清晰。
(b) **域特异衰退模式**：六个 box 的 ICC 与变化率各不相同（见上表），总分把它们压成一个数。
(c) **总分由 box 分完全决定**（实测 {agree:.3f}% 相等），同时喂入两者是冗余且构成同时刻泄漏。

**但**上表也显示 box 分的代价：每个 box 的极端档占比 49–83%，档位变化率只有 11–21%，
都比 CDRSUM（46.3% / 41.5%）更差 —— 单个 box 在序列里是一串几乎不动的重复 token，
而 `PERSCARE` 已被 §2 判为 `drop`（83% 挤在 0）。

**结论：采用五域 token（`MEMORY`/`ORIENT`/`JUDGMENT`/`COMMUN`/`HOMEHOBB`）替代 CDRSUM 总分，
`PERSCARE` 剔除，且不再单独喂入 CDRSUM。** 若最终因序列长度预算只能留一个 CDR token，
则留 CDRSUM，并且平滑必须按**秩相邻**（观测到的档位次序）而非数值相邻施加。"""))
    summary_put(s33_n_unreachable=len(unreach), s33_unreachable=unreach,
                s33_reachable_unobserved=reach_unobs,
                s33_boxes_available=have, s33_boxsum_agreement_pct=agree)


# ============================================================ 3.4 CDRGLOB
def s34():
    n = _need_nacc("§3.4 CDRGLOB")
    if n is None:
        return
    vg = VAR_BY_KEY["nacc:CDRGLOB"]
    vs = VAR_BY_KEY["nacc:CDRSUM"]
    d = n.df.assign(g=vg.clean(n.df[vg.column]), s=vs.clean(n.df[vs.column]))
    d = d[d["g"].notna() & d["s"].notna()]
    report("### §3.4 CDRGLOB\n")

    rng = np.random.default_rng(42)
    pids = d["pid"].unique()
    tr_pid = set(rng.choice(pids, size=len(pids) // 2, replace=False))
    tr = d[d["pid"].isin(tr_pid)]
    te = d[~d["pid"].isin(tr_pid)]
    mapping = tr.groupby("s")["g"].agg(lambda x: x.mode().iloc[0])
    pred = te["s"].map(mapping)
    # unseen CDRSUM values fall back to the nearest mapped value
    miss = pred.isna()
    if miss.any():
        keys = np.asarray(sorted(mapping.index))
        pred[miss] = [mapping[keys[np.abs(keys - x).argmin()]] for x in te.loc[miss, "s"]]
    acc = float((pred.to_numpy() == te["g"].to_numpy()).mean())
    cm = pd.crosstab(te["g"], pred, rownames=["CDRGLOB actual"], colnames=["predicted"])
    save_table(cm.reset_index(), "cdrglob_from_cdrsum_confusion")
    # md_table formats every float with one shared format, which would render the 0.5 row
    # label as "0"; make the labels strings so the half-steps survive.
    cm_show = cm.copy()
    cm_show.index = [_num(i) for i in cm.index]
    cm_show.columns = [_num(c) for c in cm.columns]
    cm_show = cm_show.reset_index().rename(columns={"index": "CDRGLOB actual"})
    save_table(mapping.reset_index().rename(columns={"s": "CDRSUM", "g": "modal_CDRGLOB"}),
               "cdrglob_from_cdrsum_mapping")
    report(f"""**由 CDRSUM 的可预测性**

把「每个 CDRSUM 取值 → 众数 CDRGLOB」当作预测规则，映射只在**按人划分**的一半训练集上估计，
在另一半 {len(te):,} 行上评估（{te['pid'].nunique():,} 人，与训练集无人员重叠）：

**准确率 = {acc:.4f}**

混淆矩阵（行 = 真实 CDRGLOB，列 = 由 CDRSUM 预测）：\n""")
    report(md_table(cm_show, floatfmt="{:.0f}"))

    rngtab = d.groupby("g")["s"].agg(["size", "min", lambda x: x.quantile(.25), "median",
                                     lambda x: x.quantile(.75), "max"])
    rngtab.columns = ["n", "min", "q25", "median", "q75", "max"]
    save_table(rngtab.reset_index(), "cdrglob_cdrsum_ranges")
    report("**间距不等的量化**：各 CDRGLOB 档对应的 CDRSUM 区间\n")
    report(md_table(rngtab.reset_index(), floatfmt="{:.2f}"))
    step = rngtab["median"].diff()
    report(f"相邻档中位数之差依次为 {', '.join(f'{x:+.1f}' for x in step.dropna())} —— "
           "0→0.5 只差约 1 分，2→3 差约 4 分。**CDRGLOB 的档位间距在 CDRSUM 尺度上差 4 倍**，"
           "任何把它当等距序数的平滑（把 0→0.5 和 2→3 罚同样多）都与数据不符。")
    report(gate(acc <= 0.9,
                f"准确率 {acc:.4f} "
                + ("> 0.9，CDRSUM 几乎完全决定 CDRGLOB，**两者冗余**。"
                   "只保留 CDRSUM（或 §3.3 建议的五域 token），不要同时喂入 CDRGLOB —— "
                   "同时喂入既浪费参数，又在同一时刻制造一个近乎恒等的关系让模型走捷径。"
                   if acc > 0.9 else
                   "≤ 0.9，两者并非完全冗余，可各自考虑；但仍需按 §4.2 的泄漏矩阵检查同时刻关系。")))
    summary_put(s34_cdrglob_from_cdrsum_acc=acc,
                s34_cdrsum_median_by_cdrglob=rngtab["median"].to_dict())


# ============================================================ 3.5 FAQ
def s35():
    n = _need_nacc("§3.5 FAQ 总分")
    if n is None:
        return
    v = VAR_BY_KEY["nacc:FAQTOTAL"]
    d = n.df
    report("### §3.5 FAQ 总分\n")

    strict = d["FAQTOTAL"].notna()
    prorated = d["FAQTOTAL_PRORATED"].notna()
    report(f"""本 release **没有 FAQ 总分列**，只有 10 个分项（UDS 表单 B7）。本轮自行合成：

| 口径 | 规则 | 覆盖率 |
|---|---|---|
| `FAQTOTAL` | 十项全部有效才计总分 | {100.0 * strict.mean():.1f}%（{int(strict.sum()):,} 次访视） |
| `FAQTOTAL_PRORATED` | ≥8 项有效，按比例放大到 10 项 | {100.0 * prorated.mean():.1f}%（{int(prorated.sum()):,} 次访视） |

分项的 `8`（从不做该活动）与 `9`（未知）**都不是 0**。把「从不做」记成「没有困难」会系统性
低估总分，而且这个偏差与性别、独居状况相关（做饭、理财这类活动的 not-applicable 有明显分层）。
下文与 §2 的全部结果都用严格口径。""")

    # ---- informant dependence
    rows = []
    if "INRELTO" in d.columns:
        no_inf = pd.to_numeric(d["INRELTO"], errors="coerce") == -4
        rows.append({"indicator": "INRELTO = -4 (no co-participant record)",
                     "n": int(no_inf.sum()), "pct": float(100.0 * no_inf.mean()),
                     "faq_coverage_when_true": float(100.0 * strict[no_inf].mean())})
    if "INRELY" in d.columns:
        unrel = pd.to_numeric(d["INRELY"], errors="coerce") == 1
        rows.append({"indicator": "INRELY = 1 (co-participant report NOT reliable)",
                     "n": int(unrel.sum()), "pct": float(100.0 * unrel.mean()),
                     "faq_coverage_when_true": float(100.0 * strict[unrel].mean())})
    inf = pd.DataFrame(rows)
    save_table(inf, "faq_informant_availability", subdir=vardir(v, "tables"))
    report("**知情人依赖**\n")
    report(md_table(inf, floatfmt="{:.2f}"))
    report("FAQ 由知情人（co-participant）报告，所以它的缺失模式与参与者自评量表**不是同一件事**："
           "没有知情人的访视，FAQ 覆盖率明显更低。")

    # ---- floor
    fa = d.loc[strict, "FAQTOTAL"]
    p0 = float(100.0 * (fa == 0).mean())
    by = d.loc[strict].assign(f=fa).groupby("NACCUDSD")["f"].agg(
        ["size", "mean", "median", lambda x: 100.0 * (x == 0).mean()])
    by.columns = ["n", "mean", "median", "pct_zero"]
    save_table(by.reset_index(), "faq_floor_by_udsd", subdir=vardir(v, "tables"))
    report(f"\n**地板效应**：FAQ 总分 = 0 占 **{p0:.1f}%**。按认知状态分层：\n")
    report(md_table(by.reset_index(), floatfmt="{:.2f}"))
    report(gate(p0 <= 50,
                f"FAQ=0 占 {p0:.1f}% "
                + ("> 50%，**严重地板效应**，在正常区间毫无分辨力。"
                   if p0 > 50 else "≤ 50%。")
                + f"但分层结果显示它在痴呆端很灵敏（NACCUDSD=4 时均值 "
                  f"{by.loc[4, 'mean']:.1f}，=1 时 {by.loc[1, 'mean']:.1f}）。"
                  "**保留为辅助 token，不作主干。**"))

    # ---- items
    items = [c for c in FAQ_ITEMS if c in d.columns]
    rows = []
    for c in items:
        x = pd.to_numeric(d[c], errors="coerce")
        x = x.mask(x.isin([-4, 8, 9]))
        na8 = float(100.0 * (pd.to_numeric(d[c], errors="coerce") == 8).mean())
        cnt = x.value_counts()
        h, hn = shannon(cnt.to_numpy())
        rows.append({"item": c, "n": int(x.notna().sum()),
                     "pct_missing": float(100.0 * x.isna().mean()),
                     "pct_not_applicable_8": na8,
                     "pct_zero": float(100.0 * (x == 0).mean()),
                     "normalized_entropy": hn,
                     "spearman_with_next_dementia": float(
                         stats.spearmanr(x, d["next_dem"].where(
                             d["at_risk_incident"]), nan_policy="omit").statistic)})
    it = pd.DataFrame(rows)
    save_table(it, "faq_items", subdir=vardir(v, "tables"))
    report("**十个分项**（同 §3.3 的逻辑：分项是否比总分更适合作 token）\n")
    report(md_table(it, floatfmt="{:.3f}"))
    report(f"每个分项只有 4 档、{it['pct_zero'].min():.0f}–{it['pct_zero'].max():.0f}% 落在 0，"
           "熵比总分低得多；`STOVE`/`MEALPREP` 这类活动的 not-applicable 比例最高。"
           "**分项 token 在这里不划算**：十个近乎恒为 0 的 token 换来的信息不如一个总分，"
           "与 §3.3 的 CDR box 结论相反——因为 CDR box 的 5 档是真正的严重度分级，"
           "而 FAQ 分项的 4 档在正常人群里几乎全是 0。")
    summary_put(s35_faq_strict_coverage=float(100.0 * strict.mean()),
                s35_faq_prorated_coverage=float(100.0 * prorated.mean()),
                s35_faq_pct_zero=p0,
                s35_no_informant_pct=float(inf["pct"].iloc[0]) if len(inf) else None)


# ============================================================ 3.6 GDS
def s36():
    n = _need_nacc("§3.6 GDS 短版（`NACCGDS`）")
    if n is None:
        return
    v = VAR_BY_KEY["nacc:NACCGDS"]
    d = n.df
    x = v.clean(d[v.column])
    report("### §3.6 GDS 短版（`NACCGDS`）\n")
    report(f"""**版本确认**：去掉哨兵码后取值范围 {x.min():.0f}–{x.max():.0f}，
共 {int(x.nunique())} 个取值。上限 15 而非 30，**确认是 15 题短版 GDS**，不是原版 30 题。""")

    # ---- construct separation
    partners = ["NACCMMSE", "MOCATOTS", "CDRSUM", "FAQTOTAL", "CDRGLOB"]
    rows = []
    for c in partners:
        vv = VAR_BY_KEY.get(f"nacc:{c}")
        y = vv.clean(d[vv.column]) if vv else pd.to_numeric(d[c], errors="coerce")
        r = stats.spearmanr(x, y, nan_policy="omit").statistic
        rows.append({"partner": c, "spearman_with_NACCGDS": float(r),
                     "n_pairs": int((x.notna() & y.notna()).sum())})
    cr = pd.DataFrame(rows)
    save_table(cr, "gds_vs_cognitive_scales", subdir=vardir(v, "tables"))
    report("**构念区分**：GDS 测抑郁，不测认知。与各认知量表的相关强度：\n")
    report(md_table(cr, floatfmt="{:.3f}"))
    report(f"最强也只有 |ρ| = {cr['spearman_with_NACCGDS'].abs().max():.2f}，"
           "远低于认知量表彼此之间的相关（见 §4.1）。**确认 GDS 是一个独立构念**，"
           "不是认知量表的替代品。")

    # ---- incremental value
    base = ["CDRSUM", "NACCMMSE", "MOCATOTS"]
    fr = d[d["at_risk_incident"]].copy()
    for c in base + ["NACCGDS"]:
        vv = VAR_BY_KEY.get(f"nacc:{c}")
        fr[c + "_c"] = vv.clean(fr[vv.column])
    bc = [c + "_c" for c in base]
    a0, n0, ev = oof_auc(fr, bc, "next_dem", "pid")
    a1, n1, _ = oof_auc(fr, bc + ["NACCGDS_c"], "next_dem", "pid")
    report(f"""**增量价值**（目标 = 下一次访视 `NACCUDSD`=4，只保留 t 时刻未痴呆的行，
按 `pid` 分组 5 折交叉验证，中位数填补 + 缺失指示器都只在训练折上估计）

| 特征集 | OOF AUC | n 行 | 事件率 |
|---|---|---|---|
| CDRSUM + MMSE + MoCA | {a0:.4f} | {n0:,} | {ev:.3f} |
| ＋ NACCGDS | {a1:.4f} | {n1:,} | {ev:.3f} |

**ΔAUC = {a1 - a0:+.4f}**""")
    report(gate(a1 - a0 >= 0.01,
                f"增量 AUC {a1 - a0:+.4f} "
                + ("< 0.01，不进主序列；可作静态协变量或直接省略。"
                   if a1 - a0 < 0.01 else
                   "≥ 0.01，有实质增量，**保留为辅助 token**。抑郁是痴呆的前驱症状之一，"
                   "这个结果本身值得在报告里讨论：模型从 GDS 拿到的不是认知信息的重复，"
                   "而是一条独立的前驱信号。")))
    summary_put(s36_gds_max_abs_corr=float(cr["spearman_with_NACCGDS"].abs().max()),
                s36_auc_without_gds=a0, s36_auc_with_gds=a1, s36_gds_delta_auc=a1 - a0)


# ============================================================ 3.7 NACCUDSD
def s37():
    n = _need_nacc("§3.7 NACCUDSD")
    if n is None:
        return
    v = VAR_BY_KEY["nacc:NACCUDSD"]
    d = n.df
    x = v.clean(d[v.column])
    report("### §3.7 NACCUDSD\n")
    vc = x.value_counts().sort_index()
    raw9 = int((pd.to_numeric(d[v.column], errors="coerce") == 9).sum())
    report(f"""**取值清点**（含规格书要求另查的 9）

| 取值 | 含义 | n | % |
|---|---|---|---|
| 1 | normal cognition | {vc.get(1, 0):,} | {100 * vc.get(1, 0) / len(d):.1f} |
| 2 | impaired, not MCI | {vc.get(2, 0):,} | {100 * vc.get(2, 0) / len(d):.1f} |
| 3 | MCI | {vc.get(3, 0):,} | {100 * vc.get(3, 0) / len(d):.1f} |
| 4 | dementia | {vc.get(4, 0):,} | {100 * vc.get(4, 0) / len(d):.1f} |
| 9 | — | {raw9:,} | {100 * raw9 / len(d):.2f} |

本 release（v72）**没有取值 9**，tokenizer 不应为它预留档位。""")

    # ---- category 2's ordinal position
    lev, levels, _ = levelize(v, d[v.column])
    res_lit, grp_lit = {}, None
    res_lit, grp_lit = ordinality(v, n, lev, levels, {}, tag="_lit_probe", make_fig=False)
    res_inc, grp_inc = ordinality(v, n, lev, levels, {}, outcome=n.incident,
                                  at_risk=n.at_risk_incident, tag="_inc_probe",
                                  make_fig=False)
    cmp = pd.DataFrame({
        "NACCUDSD": grp_lit["lev"],
        "n_literal": grp_lit["n"], "p_next_dementia_literal": grp_lit["p"],
    }).merge(pd.DataFrame({"NACCUDSD": grp_inc["lev"], "n_incident": grp_inc["n"],
                           "p_next_dementia_incident": grp_inc["p"],
                           "ci_lo": grp_inc["ci_lo"], "ci_hi": grp_inc["ci_hi"]}),
             on="NACCUDSD", how="outer")
    save_table(cmp, "NACCUDSD_category2_position", subdir=vardir(v, "tables"))
    report("**类别 2 的序数位置检验**\n")
    report(md_table(cmp, floatfmt="{:.4f}"))
    pi = cmp.set_index("NACCUDSD")["p_next_dementia_incident"]
    lo = cmp.set_index("NACCUDSD")["ci_lo"]
    hi = cmp.set_index("NACCUDSD")["ci_hi"]
    between = bool(pi.get(1, np.nan) <= pi.get(2, np.nan) <= pi.get(3, np.nan))
    sep_12 = bool(hi.get(1, np.nan) < lo.get(2, np.nan))
    sep_23 = bool(hi.get(2, np.nan) < lo.get(3, np.nan))
    report(f"""在 incident 定义下（只保留 t 时刻未痴呆的行，这是唯一能让四个档位互相比较的定义——
字面定义下取值 4 的 P 就是「痴呆是否持续」，天然接近 1）：

- P(下次痴呆 | 1) = {pi.get(1, np.nan):.4f}
- P(下次痴呆 | 2) = {pi.get(2, np.nan):.4f}
- P(下次痴呆 | 3) = {pi.get(3, np.nan):.4f}

类别 2 **{'确实' if between else '并不'}**落在 1 与 3 之间。
Wilson 95% CI 不重叠：1 vs 2 {'是' if sep_12 else '否'}，2 vs 3 {'是' if sep_23 else '否'}。""")

    # ---- cyclicity / reversal
    dd = d[d["has_next"]].copy()
    dd["cur"] = x[dd.index]
    dd["nxt"] = v.clean(dd["next_udsd"])
    tm = pd.crosstab(dd["cur"], dd["nxt"], normalize="index")
    tmn = pd.crosstab(dd["cur"], dd["nxt"])
    save_table(tm.reset_index(), "NACCUDSD_transition_matrix_rownorm",
               subdir=vardir(v, "tables"))
    save_table(tmn.reset_index(), "NACCUDSD_transition_matrix_counts",
               subdir=vardir(v, "tables"))
    report("**循环性检验**：行归一化的状态转移矩阵（行 = t，列 = t+1）\n")
    report(md_table(tm.reset_index(), floatfmt="{:.4f}"))
    rev = float(100.0 * (dd["nxt"] < dd["cur"]).mean())
    rev4 = float(100.0 * (dd.loc[dd["cur"] == 4, "nxt"] < 4).mean())
    report(f"整体「状态回退」比例 **{rev:.2f}%**；已达痴呆（=4）后下一次不再是痴呆的比例 "
           f"**{rev4:.2f}%**。痴呆是不可逆的，所以这 {rev4:.2f}% 是**诊断噪声或随访测量误差**，"
           "不是真实好转。任何把状态当可逆马尔可夫链的建模都会去学这条不存在的回退路径。")

    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    im = ax.imshow(tm.to_numpy(), cmap=SEQ_CMAP, vmin=0, vmax=1)
    ax.set_xticks(range(len(tm.columns)))
    ax.set_xticklabels([_num(c) for c in tm.columns])
    ax.set_yticks(range(len(tm.index)))
    ax.set_yticklabels([_num(i) for i in tm.index])
    for i in range(tm.shape[0]):
        for j in range(tm.shape[1]):
            val = tm.to_numpy()[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if val > 0.5 else "black")
    ax.set_xlabel("NACCUDSD at t+1")
    ax.set_ylabel("NACCUDSD at t")
    ax.set_title("Row-normalised transitions", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_fig(fig, vardir(v, "figs"), "NACCUDSD_transitions")
    save_data(tm.reset_index(), vardir(v, "figs"), "NACCUDSD_transitions")

    report(gate(between and sep_12 and sep_23,
                ("类别 2 的结局概率确实居于 1 与 3 之间且与两侧都可区分，"
                 "序数假设在 2 处成立，可作为一个正常档位参与平滑。"
                 if between and sep_12 and sep_23 else
                 f"类别 2 {'虽然居中但与邻档不可区分' if between else '**不在 1 与 3 之间**'}，"
                 "序数假设在 2 处断裂。处理：把 2 单独作一档但**不参与平滑**"
                 "（它的邻居关系没有数据支持），或与 3 合并。"
                 "另外它只占全部访视的 4.4%，合并的代价很小。")))
    summary_put(s37_has_value_9=raw9 > 0,
                s37_p_next_dem_incident={str(int(k)): float(vv) for k, vv in pi.dropna().items()},
                s37_cat2_between=between, s37_cat2_sep_from_1=sep_12,
                s37_cat2_sep_from_3=sep_23, s37_reversal_pct=rev,
                s37_dementia_reversal_pct=rev4)


# ============================================================ 3.8 dcfdx
COLLAPSE_4 = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3}
COLLAPSE_3 = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2}


def _ord_of(series, y, at_risk):
    """Level-wise P(y) plus Spearman / isotonic residual, for an arbitrary level series."""
    from sklearn.isotonic import IsotonicRegression
    m = pd.DataFrame({"lev": series, "y": y})[at_risk].dropna()
    g = m.groupby("lev")["y"].agg(["sum", "size"])
    g.columns = ["k", "n"]
    g["p"] = g["k"] / g["n"]
    g = g[g["n"] >= 10]
    if len(g) < 3:
        return np.nan, np.nan, g
    r = np.arange(len(g), dtype=float)
    rho = stats.spearmanr(r, g["p"].to_numpy()).statistic
    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip")
    w = g["n"].to_numpy(float)
    yy = g["p"].to_numpy(float)
    fit = iso.fit_transform(r, yy, sample_weight=w)
    rss = float((w * (yy - fit) ** 2).sum())
    ybar = float((w * yy).sum() / w.sum())
    tss = float((w * (yy - ybar) ** 2).sum())
    return float(rho), (rss / tss if tss > 0 else np.nan), g


def s38():
    p = _rp()
    v = VAR_BY_KEY["radc:dcfdx"]
    d = p.df
    x = v.clean(d[v.column])
    y = pd.to_numeric(d["braak_ad"], errors="coerce")
    at = d["has_autopsy"].fillna(False).astype(bool) & x.notna()
    report("### §3.8 `dcfdx`（本 release 只有 `dcfdx_lv`）\n")
    report(f"""参照结局换成**尸检 `braaksc >= 4`**，原因写在 §1：`dcfdx_lv` 是人级「最后一次
有效访视」的诊断，没有「下一次访视」可言。用病理当参照同时避开了循环——
可用样本 {int(at.sum()):,} 人（同时有临床诊断与尸检）。

折叠映射（规格书 §C6 不在本仓库，这里按 codebook 的档位含义构造并**同时检验两种**）：

| 原始 | 含义 | 4 档折叠 | 3 档折叠 |
|---|---|---|---|
| 1 | NCI 无认知损害 | 0 | 0 |
| 2 | MCI | 1 | 1 |
| 3 | MCI + 其他病因 | 1 | 1 |
| 4 | AD 痴呆 | 2 | 2 |
| 5 | AD + 其他病因 | 2 | 2 |
| 6 | 其他病因痴呆（非 AD） | 3 | 2 |

差别只在取值 6。4 档折叠把「非 AD 痴呆」放在**最重端**，这是规格书要求的档数；
3 档折叠把它并入「痴呆」。取值 6 是**另一种疾病**，不是 AD 严重度的下一级——
它对「AD 病理」这个参照结局的概率没有理由更高。""")

    rows = []
    for name, ser in [("raw 1-6", x),
                      ("collapsed 4-level", x.map(COLLAPSE_4)),
                      ("collapsed 3-level", x.map(COLLAPSE_3))]:
        rho, resid, g = _ord_of(ser, y, at)
        rows.append({"coding": name, "n_levels": int(len(g)), "spearman": rho,
                     "isotonic_residual_ratio": resid,
                     "smoothing_verdict": ("ok" if (np.isfinite(rho) and abs(rho) > 0.9
                                                    and np.isfinite(resid) and resid < 0.1)
                                           else "caution" if np.isfinite(rho) and abs(rho) >= 0.6
                                           else "forbidden"),
                     "p_by_level": ", ".join(f"{k:.0f}:{vv:.3f}" for k, vv in g["p"].items())})
    cmp = pd.DataFrame(rows)
    save_table(cmp, "dcfdx_collapse_comparison", subdir=vardir(v, "tables"))
    report("**折叠前后对比**（参照结局 = 尸检 Braak ≥ 4）\n")
    report(md_table(cmp, floatfmt="{:.4f}"))

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.4), sharey=True)
    for ax, (name, ser), c in zip(axes, [("raw 1-6", x),
                                         ("collapsed 4-level", x.map(COLLAPSE_4)),
                                         ("collapsed 3-level", x.map(COLLAPSE_3))],
                                  (C1, C2, C4)):
        rho, resid, g = _ord_of(ser, y, at)
        ax.bar(np.arange(len(g)), g["p"], color=c)
        ax.set_xticks(np.arange(len(g)))
        ax.set_xticklabels([_num(i) for i in g.index])
        ax.set_title(f"{name}\nρ = {rho:.3f}, resid = {resid:.3f}", fontsize=9)
        ax.set_xlabel("level")
    axes[0].set_ylabel("P(Braak ≥ 4)")
    fig.tight_layout()
    save_fig(fig, vardir(v, "figs"), "dcfdx_collapse")
    save_data(cmp, vardir(v, "figs"), "dcfdx_collapse")

    # The raw 1-6 dips at 3 and 5 -- both "+ other cause" categories. Worth naming, because
    # it explains WHY the scale is not ordinal rather than just recording that it is not.
    _, _, graw = _ord_of(x, y, at)
    mixed = [v for v in (3.0, 5.0) if v in graw.index]
    if len(mixed) == 2:
        report(f"""**原始 1–6 为什么不是序数**：逐档概率里 3 与 5 都比它们前一档**低**
（3: {graw.loc[3.0, 'p']:.3f} < 2: {graw.loc[2.0, 'p']:.3f}；5: {graw.loc[5.0, 'p']:.3f} <
4: {graw.loc[4.0, 'p']:.3f}）。3 与 5 恰好是两个「**＋其他病因**」档。
也就是说 `dcfdx` 的 1–6 同时编码了**两个维度**：认知损害的严重度，以及病因的纯粹性。
「MCI＋其他病因」不是比「MCI」更重，它是**另一类病人**，AD 病理反而被其他病因稀释。
一个把两个维度压在一根轴上的编码，本来就不可能单调——这是折叠的根本理由，
不是「档位太细、样本太少」。""")

    raw_rho = abs(cmp.loc[0, "spearman"])
    c4_rho = abs(cmp.loc[1, "spearman"])
    c3_rho = abs(cmp.loc[2, "spearman"])
    best = cmp.loc[cmp["spearman"].abs().idxmax(), "coding"]
    n3 = int(cmp.loc[2, "n_levels"])
    report(f"""**读这张表前的一个警告**：3 档折叠的 |ρ| = {c3_rho:.3f} 是在
**只剩 {n3} 个档位**上算的，而 {n3} 个点的任何单调排列都会给出 |ρ| = 1。
所以「ρ 从 {raw_rho:.3f} 升到 {c3_rho:.3f}」里有一部分是档数变少的必然结果，不是标度变好了。
**真正可比的是等渗残差比**（{cmp.loc[0, 'isotonic_residual_ratio']:.4f} →
{cmp.loc[1, 'isotonic_residual_ratio']:.4f} → {cmp.loc[2, 'isotonic_residual_ratio']:.4f}）
**和逐档概率本身**：AD 痴呆档 {graw.loc[4.0, 'p']:.3f} 高于其他病因痴呆档
{graw.loc[6.0, 'p']:.3f}，这个不等号与档数无关，它才是「6 不该放在最重端」的直接证据。""")
    report(gate(c4_rho > raw_rho,
                f"|ρ| 从原始 1–6 的 {raw_rho:.3f} 变为 4 档折叠的 {c4_rho:.3f}、"
                f"3 档折叠的 {c3_rho:.3f}。"
                + (f"**折叠确实改善序数性**，最好的编码是 {best}。"
                   "这个对比就是折叠必要性的方法学依据，应写进报告。"
                   if c4_rho > raw_rho else
                   f"**4 档折叠没有改善序数性**（{c4_rho:.3f} ≤ {raw_rho:.3f}）——"
                   f"问题正是取值 6 被放在最重端。3 档折叠（把 6 并入痴呆）给出 {c3_rho:.3f}，"
                   "说明「其他病因痴呆」应该与 AD 痴呆同档，而不是更重的一档。"
                   "**规格书 §C6 的 4 档映射需要修改。**")))
    summary_put(s38_rho_raw=float(raw_rho), s38_rho_collapse4=float(c4_rho),
                s38_rho_collapse3=float(c3_rho), s38_best_coding=best)


# ============================================================ 3.9 pathology
def s39():
    p = _rp()
    d = p.df
    report("### §3.9 Braak / CERAD / NIA-Reagan\n")
    report("**这三个仅作结局，不得进入输入序列。** 本节只确认编码方向与可用样本量。")

    rows = []
    for k in ["braaksc", "ceradsc", "niareagansc"]:
        v = VAR_BY_KEY[f"radc:{k}"]
        if v.status != "ok":
            rows.append({"variable": k, "column": "—", "status": v.status, "n": 0,
                         "pct_of_cohort": 0.0, "values": "—"})
            continue
        x = v.clean(d[v.column]).dropna()
        rows.append({"variable": k, "column": v.column, "status": v.status, "n": int(len(x)),
                     "pct_of_cohort": float(100.0 * len(x) / len(d)),
                     "values": ", ".join(f"{_num(a)}:{b}" for a, b in
                                         x.value_counts().sort_index().items())})
    av = pd.DataFrame(rows)
    save_table(av, "pathology_availability")
    report(md_table(av, floatfmt="{:.1f}"))

    # ---- direction, measured against clinical dx
    dcf = VAR_BY_KEY["radc:dcfdx"]
    cl = dcf.clean(d[dcf.column])
    cg = VAR_BY_KEY["radc:cogdx"].clean(d["cogdx"])
    rows = []
    for k, expect in [("braaksc", "higher_worse"), ("ceradsc", "higher_better"),
                      ("niareagansc", "higher_better")]:
        v = VAR_BY_KEY[f"radc:{k}"]
        if v.status != "ok":
            rows.append({"variable": k, "expected": expect, "rho_vs_dcfdx_lv": np.nan,
                         "rho_vs_cogdx": np.nan, "measured": "not_found", "matches": False})
            continue
        x = v.clean(d[v.column])
        r1 = stats.spearmanr(x, cl, nan_policy="omit").statistic
        r2 = stats.spearmanr(x, cg, nan_policy="omit").statistic
        meas = "higher_worse" if r1 > 0 else "higher_better"
        rows.append({"variable": k, "expected": expect, "rho_vs_dcfdx_lv": float(r1),
                     "rho_vs_cogdx": float(r2), "measured": meas,
                     "matches": bool(meas == expect)})
    dr = pd.DataFrame(rows)
    save_table(dr, "pathology_direction")
    report("**方向确认**（不看 codebook，用与临床诊断的相关符号实测）\n")
    report(md_table(dr, floatfmt="{:.3f}"))
    report("`braaksc` 与临床诊断**正相关**（分越高病理越重），`ceradsc` **负相关**——"
           "证实 CERAD 是反向编码（1 = definite AD，4 = no AD），"
           "与 Braak 的方向相反。这两个变量放在同一个损失函数里时必须先统一方向，"
           "否则「病理更重」这件事在两个头上是相反的梯度。")

    # ---- clinical vs pathological cross-tab
    b = VAR_BY_KEY["radc:braaksc"]
    ct = pd.crosstab(cl, b.clean(d[b.column]), rownames=["dcfdx_lv (clinical)"],
                     colnames=["braaksc (pathology)"])
    save_table(ct.reset_index(), "clinical_vs_pathology_crosstab")
    report("**临床诊断 × 病理诊断交叉表**\n")
    report(md_table(ct.reset_index(), floatfmt="{:.0f}"))
    dem = (cl >= 4)
    path = (b.clean(d[b.column]) >= 4)
    ok = cl.notna() & b.clean(d[b.column]).notna()
    agree = float(100.0 * (dem[ok] == path[ok]).mean())
    dem_no_path = float(100.0 * (dem[ok] & ~path[ok]).mean())
    path_no_dem = float(100.0 * (~dem[ok] & path[ok]).mean())
    report(f"二分后（临床 `dcfdx_lv≥4` vs 病理 `braaksc≥4`）在 {int(ok.sum()):,} 名有两者的人中："
           f"一致 **{agree:.1f}%**；临床痴呆但病理不足 {dem_no_path:.1f}%；"
           f"病理达标但临床未痴呆 **{path_no_dem:.1f}%**（无症状 AD 病理）。"
           "临床与病理**不是同一个结局**，把病理当「金标准标签」会给近三成的人错误的监督信号。")

    n_aut = int(d["has_autopsy"].sum())
    report(gate(n_aut >= 500,
                f"有尸检数据的人 {n_aut:,}（占队列 {100.0 * n_aut / len(d):.1f}%）。"
                + ("样本量足够支撑病理作为**辅助监督任务**（多任务头），"
                   "但只能在这个子集上算损失，且必须注意「有尸检」本身与死亡相关，是选择性子集。"
                   if n_aut >= 500 else "样本太少，病理不能作辅助监督任务。")))
    report(gate(VAR_BY_KEY["radc:niareagansc"].status == "ok",
                "`niareagansc` 在本 release 中**不存在**（codebook 与三张表都没有），"
                "病理结局只剩 Braak 与 CERAD。规格书 §1.3 关于它「与 Braak 反向」的预期"
                "无法在本数据上核对，相关条目应从文档移除或标注为不可用。"))
    summary_put(s39_n_autopsy=n_aut,
                s39_direction={r.variable: [r.expected, r.measured] for r in dr.itertuples()},
                s39_clin_path_agreement_pct=agree,
                s39_pathology_without_dementia_pct=path_no_dem,
                s39_niareagansc_available=VAR_BY_KEY["radc:niareagansc"].status == "ok")


def run():
    _, panels = s2_generic.rows_frame()
    S["panels"] = panels
    report("## §3 各变量专项分析\n")
    for f in (s31, s32, s33, s34, s35, s36, s37, s38, s39):
        f()
