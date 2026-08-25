"""Section 4 -- cross-variable analyses: redundancy, same-visit leakage, incremental value.

§4.2 is the one that changes the model's architecture rather than its feature list, so it is
computed on EVERY ordered pair, not just the pairs that look suspicious.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

import s2_generic
from common import (C2, DIV_CMAP, FIGS, REF_GREY, SEQ_CMAP, VAR_BY_KEY, auc_single,
                    dichotomise, dichotomy_str, gate, md_table, report, save_data, save_fig,
                    save_table, summary_put, vars_in)
from ml import forward_select

# §4.3 subsamples persons so the greedy search stays affordable; stated in the report.
MAX_ROWS_FORWARD = 60000
MAX_FORWARD_STEPS = 8
S = {}


def clean_matrix(panel, varlist):
    """One column per variable, sentinels removed, on the panel's own rows."""
    out = {}
    for v in varlist:
        if v.status == "ok":
            out[v.spec_name] = v.clean(panel.df[v.column])
    return pd.DataFrame(out, index=panel.df.index)


# ============================================================ 4.1 redundancy
def s41(panel, label, tag):
    varlist = [v for v in vars_in(panel.name) if v.status == "ok"]
    M = clean_matrix(panel, varlist)
    if M.shape[1] < 3:
        report(f"#### {label}\n\n只有 {M.shape[1]} 个可用变量，相关矩阵与聚类无意义，跳过。")
        return None
    C = M.corr(method="spearman", min_periods=200)
    N = M.notna().astype(int).T @ M.notna().astype(int)
    save_table(C.reset_index(), f"scale_correlation_spearman_{tag}")
    save_table(N.reset_index(), f"scale_correlation_n_pairs_{tag}")

    D = 1 - C.abs().fillna(0).to_numpy()
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2
    Z = hierarchy.linkage(squareform(D, checks=False), method="average")
    order = hierarchy.leaves_list(Z)
    names = [C.columns[i] for i in order]
    Co = C.loc[names, names]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6),
                            gridspec_kw={"width_ratios": [1.35, 1]})
    im = axes[0].imshow(Co.to_numpy(), cmap=DIV_CMAP, vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels(names, rotation=90, fontsize=7)
    axes[0].set_yticks(range(len(names)))
    axes[0].set_yticklabels(names, fontsize=7)
    for i in range(len(names)):
        for j in range(len(names)):
            val = Co.to_numpy()[i, j]
            if np.isfinite(val):
                axes[0].text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.2,
                             color="white" if abs(val) > 0.6 else "black")
    axes[0].set_title(f"Spearman, same-visit observations ({label})", fontsize=9)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    hierarchy.dendrogram(Z, labels=list(C.columns), ax=axes[1], leaf_font_size=7,
                         color_threshold=0.1, above_threshold_color=REF_GREY)
    axes[1].set_ylabel("1 − |ρ| (average linkage)")
    axes[1].axhline(0.1, color=C2, lw=1, ls="--")
    axes[1].set_title("clusters; dashed line = |ρ| = 0.9", fontsize=9)
    fig.tight_layout()
    save_fig(fig, FIGS, f"scale_correlation_heatmap_{tag}")
    save_data(Co.reset_index(), FIGS, f"scale_correlation_heatmap_{tag}")

    pairs = []
    for i, a in enumerate(C.columns):
        for b in C.columns[i + 1:]:
            r = C.loc[a, b]
            if np.isfinite(r):
                pairs.append({"a": a, "b": b, "spearman": float(r),
                              "abs_spearman": abs(float(r)), "n_pairs": int(N.loc[a, b])})
    pr = pd.DataFrame(pairs).sort_values("abs_spearman", ascending=False)
    save_table(pr, f"scale_correlation_pairs_{tag}")
    hot = pr[pr["abs_spearman"] > 0.9]
    report(f"#### {label}\n")
    report("|ρ| 最大的 12 对：\n")
    report(md_table(pr.head(12), floatfmt="{:.3f}"))
    clusters = hierarchy.fcluster(Z, t=0.1, criterion="distance")
    cl = pd.DataFrame({"variable": list(C.columns), "cluster": clusters}).sort_values("cluster")
    save_table(cl, f"scale_clusters_{tag}")
    multi = cl.groupby("cluster")["variable"].apply(list)
    multi = {int(k): vv for k, vv in multi.items() if len(vv) > 1}
    report(gate(len(hot) == 0,
                (f"存在 |ρ| > 0.9 的变量对 {len(hot)} 组："
                 + "；".join(f"`{r.a}`–`{r.b}` ({r.spearman:+.3f})" for r in hot.itertuples())
                 + f"。在 |ρ|=0.9 处切树得到的多成员簇：{multi}。"
                 "**同一簇内只保留一个代表变量**——同时喂入既冗余又构成同时刻泄漏。"
                 "代表变量按 §2 的 token 可行性指标挑（稀有档少、熵高、极端档轻）。"
                 if len(hot) else "没有 |ρ| > 0.9 的变量对，各变量之间不存在严格冗余。")))
    return pr, multi


# ============================================================ 4.2 leakage matrix
def s42(panel, label, tag):
    varlist = [v for v in vars_in(panel.name) if v.status == "ok"]
    M = clean_matrix(panel, varlist)
    tgt = {}
    for v in varlist:
        y = dichotomise(M[v.spec_name], v.dichotomy)
        if y is not None and y.notna().sum() > 200 and y.nunique(dropna=True) == 2:
            tgt[v.spec_name] = y
    rows = []
    for pn in M.columns:
        row = {"predictor": pn}
        for tn, y in tgt.items():
            if tn == pn:
                row[tn] = np.nan
                continue
            a, n = auc_single(M[pn], y)
            # leakage is direction-agnostic: a predictor that anti-predicts perfectly leaks
            # exactly as much, so report the distance from chance folded to >= 0.5
            row[tn] = np.nan if not np.isfinite(a) else max(a, 1 - a)
        rows.append(row)
    mat = pd.DataFrame(rows).set_index("predictor")
    save_table(mat.reset_index(), f"same_visit_leakage_matrix_{tag}")
    defs = pd.DataFrame([{"target": v.spec_name, "binary_definition": dichotomy_str(v.dichotomy),
                          "n_positive": int(tgt[v.spec_name].sum()),
                          "event_rate": float(tgt[v.spec_name].mean())}
                         for v in varlist if v.spec_name in tgt])
    save_table(defs, f"same_visit_leakage_target_definitions_{tag}")

    fig, ax = plt.subplots(figsize=(0.55 * len(mat.columns) + 3.4, 0.42 * len(mat) + 2.4))
    im = ax.imshow(mat.to_numpy(dtype=float), cmap=SEQ_CMAP, vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=7)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = mat.to_numpy(dtype=float)[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=5.4,
                        color="white" if val > 0.85 else "black")
    ax.set_xlabel("target (binarised)")
    ax.set_ylabel("predictor (raw value, same visit)")
    ax.set_title(f"Same-visit univariate AUC — {label}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    save_fig(fig, FIGS, f"same_visit_leakage_{tag}")
    save_data(mat.reset_index(), FIGS, f"same_visit_leakage_{tag}")

    long = mat.stack(future_stack=True).reset_index()
    long.columns = ["predictor", "target", "auc"]
    long = long.dropna().sort_values("auc", ascending=False)
    hot = long[long["auc"] > 0.95]
    report(f"#### {label}\n")
    report("二分定义（用于把每个变量变成一个可算 AUC 的目标）：\n")
    report(md_table(defs, floatfmt="{:.4f}"))
    report("\nAUC > 0.95 的组合（同时刻几乎可以互相反推）：\n")
    report(md_table(hot, floatfmt="{:.4f}", max_rows=30))
    report(gate(len(hot) == 0,
                (f"{len(hot)} 个组合在**同一次访视**达到 AUC > 0.95。"
                 "这些组合必须被同时刻 mask 隔开，且任务定义为**跨访视预测**——"
                 "否则模型只需读同一时刻的另一个 token 就能答出目标，学不到任何时间动态。"
                 "上面这张矩阵直接作为 mask 设计的依据。"
                 if len(hot) else "没有组合在同一次访视达到 AUC > 0.95。")))
    return mat, long


# ============================================================ 4.3 incremental value
def s43(panel, label, tag, backbone, candidates):
    d = panel.df
    fr = d[d[panel.at_risk_incident].fillna(False).astype(bool)].copy() \
        if panel.at_risk_incident else d[d[panel.at_risk].fillna(False).astype(bool)].copy()
    cols = {}
    for name in [backbone] + candidates:
        v = VAR_BY_KEY[f"{panel.name.split('_')[0]}:{name}"]
        if v.status != "ok":
            continue
        cols[name] = v.clean(fr[v.column])
    X = pd.DataFrame(cols)
    X[panel.person] = fr[panel.person].to_numpy()
    X["y"] = pd.to_numeric(fr[panel.incident or panel.outcome], errors="coerce").to_numpy()
    X = X[X["y"].notna()]
    note = ""
    if len(X) > MAX_ROWS_FORWARD:
        rng = np.random.default_rng(42)
        pids = X[panel.person].unique()
        frac = MAX_ROWS_FORWARD / len(X)
        keep = set(rng.choice(pids, size=max(50, int(len(pids) * frac)), replace=False))
        X = X[X[panel.person].isin(keep)]
        note = (f"贪心前向选择在按人抽样的 {len(X):,} 行子集上跑"
                f"（{X[panel.person].nunique():,} 人，抽样比 {frac:.2f}），"
                "以控制 5 折 × 每步全部候选的拟合次数；抽样按人整块进行，不切开个体。")
    steps, chosen = forward_select(X, [backbone], [c for c in cols if c != backbone],
                                   "y", panel.person, max_steps=MAX_FORWARD_STEPS)
    save_table(steps, f"incremental_value_{tag}")
    report(f"#### {label}\n")
    report(f"""基线 = 当次访视的主干变量 `{backbone}`，目标 = **下一次访视达到痴呆**
（只保留 t 时刻未痴呆的行），按人分组 5 折 OOF AUC。前向选择跑满 {MAX_FORWARD_STEPS} 步，
**每一步的边际增益都记录**（`marginal_gain` = 相对**上一步**的 AUC 提升，不是相对基线），
包括低于阈值的，这样能看出曲线在哪里走平，而不是在第一个小增益处静默停下。
{note}""")
    report(md_table(steps, floatfmt="{:.4f}"))
    per_step = steps[steps["step"] > 0]
    keep = per_step[per_step["marginal_gain"] >= 0.005]["added"].tolist()
    drop = per_step[per_step["marginal_gain"] < 0.005]["added"].tolist()
    report(gate(len(drop) == 0,
                f"边际增益 ≥0.005 的变量（进主序列）：{', '.join('`%s`' % x for x in keep) or '无'}；"
                f"<0.005 的（**不进主序列**）：{', '.join('`%s`' % x for x in drop) or '无'}。"
                "序列越短、token 越少，小样本下越稳；这张表就是最终辅助 token 清单的依据。"))
    return steps, keep, drop


def run():
    df, panels = s2_generic.rows_frame()
    S["panels"] = panels
    n, rv = panels["nacc_visit"], panels["radc_visit"]

    report("## §4 跨变量分析\n")
    if n is None:
        report("**本机没有 NACC 导出**，§4 的 NACC 部分全部跳过（§1 记为 `panel_unavailable`）。"
               "RADC 只有 2 个逐访视量表，所以本节实际只剩泄漏矩阵一项。")
    report("### §4.1 冗余与共线\n")
    report("同一次访视的观测，两两 Spearman（pairwise-complete，最少 200 对），"
           "再按 1−|ρ| 做平均连接层次聚类。NACC 与 RADC **分开算**——两个数据集没有共同的行。")
    r_nacc = s41(n, "NACC，逐访视量表", "nacc") if n is not None else None
    r_radc = s41(rv, "RADC，逐访视量表", "radc")

    report("### §4.2 同时刻泄漏矩阵\n")
    report("""对每个候选目标变量，用**同一次访视**的其他每一个变量单独预测它，报 AUC。
连续/序数取值直接作为打分（秩 AUC），目标按下表的临床二分点二值化。
矩阵里填的是 `max(AUC, 1−AUC)`：泄漏与方向无关，一个能完美反向预测目标的变量泄漏得一样多。""")
    m_nacc = s42(n, "NACC", "nacc") if n is not None else None
    m_radc = s42(rv, "RADC", "radc")

    report("### §4.3 增量价值排序\n")
    nacc_cands = ["CDRSUM", "CDRGLOB", "MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB",
                  "PERSCARE", "MOCATOTS", "FAQTOTAL", "NACCGDS", "NACCUDSD"]
    s_nacc = s43(n, "NACC", "nacc", "NACCMMSE", nacc_cands) if n is not None else None
    s_radc = s43(rv, "RADC", "radc", "cts_mmse30", ["cogn_global"])

    summary_put(
        s41_nacc_pairs_gt_090=[[r.a, r.b, r.spearman] for r in r_nacc[0].itertuples()
                               if abs(r.spearman) > 0.9] if r_nacc else [],
        s41_nacc_clusters=r_nacc[1] if r_nacc else {},
        s42_nacc_leaky=([[r.predictor, r.target, r.auc] for r in m_nacc[1].itertuples()
                         if r.auc > 0.95] if m_nacc else []),
        s42_radc_leaky=[[r.predictor, r.target, r.auc] for r in m_radc[1].itertuples()
                        if r.auc > 0.95],
        s43_nacc_keep=s_nacc[1] if s_nacc else [],
        s43_nacc_drop=s_nacc[2] if s_nacc else [],
        s43_nacc_steps=s_nacc[0].to_dict("records") if s_nacc else [],
        s43_radc_keep=s_radc[1], s43_radc_drop=s_radc[2],
    )
    S.update(leakage_nacc=m_nacc[1] if m_nacc else None, leakage_radc=m_radc[1],
             corr_nacc=r_nacc, steps_nacc=s_nacc)
    return S
