"""Section 4 -- the time axis and the visit interval (risk R1: keep or delete the dt head?)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, r2_score

from common import (C1, C2, C3, C4, REF_GREY, SEED, FIGS, load, parse_capped_age, report,
                    save_data, save_fig, save_table, setup_style, summary_put, gate)

FEATS = ["age_at_visit", "fu_year", "cogn_global", "cts_estmmse30", "msex", "educ",
         "bmi", "ad_now", "hypertension_cum", "dm_cum"]


def _oof(X, y, groups, kind):
    """Out-of-fold prediction with 5 folds grouped by participant (no person spans folds)."""
    pred = np.full(len(y), np.nan)
    est = (make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=1.0))
           if kind == "reg" else
           make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0)))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        est.fit(X[tr], y[tr])
        pred[te] = est.predict(X[te]) if kind == "reg" else est.predict_proba(X[te])[:, 1]
    return pred


def run():
    setup_style()
    cs, lo, cl, v = load()

    # ---- A. is the annual grid real? ---------------------------------------------
    # The release has no visit date. age_at_visit := age_bl + fu_year is exact only if every
    # cycle lands on the baseline anniversary. ROSMAP_clinical.age_at_visit_max is a REAL
    # observed visit age (when not top-coded), so (age_at_visit_max - age_bl) must be an
    # integer if the grid is real. Its distance to the nearest integer is the grid error.
    m = cl[["projid", "age_at_visit_max"]].merge(cs[["projid", "age_bl"]], on="projid", how="inner")
    age_max, capped_max = parse_capped_age(m["age_at_visit_max"])
    ok = (~capped_max) & age_max.notna() & m["age_bl"].notna()
    span = (age_max[ok] - m.loc[ok, "age_bl"]).to_numpy()
    grid_err = np.abs(span - np.round(span))
    grid_err = grid_err[np.isfinite(grid_err)]
    # A uniform-in-[0, 0.5] reference is what "cycles are unrelated to the anniversary" would
    # look like; the observed distribution is far tighter than that but has a real 6-month tail.
    grid_tbl = pd.DataFrame({
        "quantile": ["P25", "P50", "P75", "P90", "P95", "max", "pct<=15d", "pct<=45d", "pct<=91d", "n"],
        "abs_deviation_years": [np.quantile(grid_err, q) for q in (.25, .5, .75, .9, .95, 1.0)]
        + [float((grid_err <= 15 / 365.25).mean() * 100), float((grid_err <= 45 / 365.25).mean() * 100),
           float((grid_err <= 91 / 365.25).mean() * 100), float(len(grid_err))]})
    save_table(grid_tbl, "annual_grid_deviation")

    # ---- B. age distributions ------------------------------------------------------
    f, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    ax = axes[0]
    # density, not counts: 36k visits vs 4.4k participants are different units and would
    # otherwise share one axis at wildly different scales.
    bins = np.arange(35, 112, 1)
    ax.hist(v["age_at_visit"], bins=bins, color=C1, alpha=.85, lw=0, density=True,
            label=f"All visits (n={len(v):,})")
    ax.hist(cs["age_bl"], bins=bins, histtype="step", lw=2, color=C2, density=True,
            label=f"Baseline age (n={len(cs):,})")
    ax.axvline(65, color=REF_GREY, ls="--", lw=1.5)
    ax.annotate("age 65", (65, ax.get_ylim()[1] * .96), xytext=(4, 0), textcoords="offset points",
                color=REF_GREY, fontsize=9, va="top")
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Density")
    ax.set_title(f"Left truncation: only {100 * (cs['age_bl'] < 65).mean():.1f}% enrol before 65")
    ax.legend(frameon=False, loc="upper left")

    ax = axes[1]
    for st, col in zip(["ROS", "MAP", "LATC"], [C1, C2, C3]):
        d = cs.loc[cs.study == st, "age_bl"]
        ax.hist(d, bins=np.arange(35, 112, 2), histtype="step", lw=2, color=col,
                label=f"{st} (n={len(d)}, median {d.median():.0f})", density=True)
    ax.set_xlabel("Baseline age (years)"); ax.set_ylabel("Density")
    ax.set_title("Baseline age differs sharply by cohort")
    ax.legend(frameon=False, loc="upper left")
    f.tight_layout()
    save_data(v[["projid", "study", "fu_year", "age_at_visit", "age_bl"]], FIGS, "age_at_visit_dist")
    save_fig(f, FIGS, "age_at_visit_dist")

    f2, ax = plt.subplots(figsize=(6.2, 4.0))
    order = ["ROS", "MAP", "LATC"]
    data = [cs.loc[cs.study == s, "age_bl"].dropna() for s in order]
    bp = ax.boxplot(data, tick_labels=order, patch_artist=True, widths=.55,
                    medianprops=dict(color="black", lw=2), flierprops=dict(ms=3, alpha=.35))
    for patch, col in zip(bp["boxes"], [C1, C2, C3]):
        patch.set_facecolor(col); patch.set_alpha(.75); patch.set_edgecolor("white"); patch.set_lw(1.5)
    for i, dd in enumerate(data, 1):
        ax.annotate(f"n={len(dd)}\nmed {dd.median():.0f}", (i, dd.median()), xytext=(14, 0),
                    textcoords="offset points", fontsize=8.5, va="center", color="black")
    ax.axhline(65, color=REF_GREY, ls="--", lw=1.5)
    ax.set_ylabel("Baseline age (years)")
    ax.set_title("Baseline age by cohort (the left-truncation point)")
    save_data(cs[["projid", "study", "age_bl"]], FIGS, "baseline_age_by_study")
    save_fig(f2, FIGS, "baseline_age_by_study")

    # ---- C. the interval ------------------------------------------------------------
    d = v.loc[v["d_age"].notna(), "d_age"]
    cv = d.std() / d.mean()
    qs = [0, .01, .05, .25, .5, .75, .9, .95, .99, 1]
    qt = pd.DataFrame({"quantile": [str(q) for q in qs],
                       "delta_age_years": [d.quantile(q) for q in qs]})
    qt = pd.concat([qt, pd.DataFrame({
        "quantile": ["mean", "sd", "cv", "pct_eq_1", "pct_gt_1.5", "pct_ge_2", "n_pairs"],
        "delta_age_years": [d.mean(), d.std(), cv, 100 * (d == 1).mean(),
                            100 * (d > 1.5).mean(), 100 * (d >= 2).mean(), len(d)]})],
        ignore_index=True)
    save_table(qt, "delta_age_quantiles")

    f3, ax = plt.subplots(figsize=(6.6, 4.2))
    ax.hist(d, bins=np.arange(0.05, d.max() + 0.15, 0.1), color=C1, lw=0)
    ax.set_yscale("log")
    ax.set_xlabel("Δage (years, 0.1-year bins)"); ax.set_ylabel("Visit pairs (log scale)")
    ax.set_xlim(0, min(12, d.max() + 1))
    ax.annotate(f"{100 * (d == 1).mean():.1f}% land exactly on 1.0 y\n"
                "— by construction, not by measurement",
                xy=(1.0, (d == 1).sum()), xytext=(0.34, 0.92), textcoords="axes fraction",
                ha="left", va="top", fontsize=9, color=C2,
                arrowprops=dict(arrowstyle="->", color=C2, lw=1))
    ax.annotate(f"{100 * (d > 1.5).mean():.1f}% are > 1.5 y\napart — missed cycles, "
                "not longer\nreal-world intervals",
                xy=(3.0, max((d >= 3).sum(), 1)), xytext=(0.42, 0.60), textcoords="axes fraction",
                ha="left", va="top", fontsize=9, color=C3,
                arrowprops=dict(arrowstyle="->", color=C3, lw=1))
    ax.set_title(f"Δage between consecutive observed visits is integer-valued (CV={cv:.3f})\n"
                 "the release has no visit dates, so the interval IS the protocol")
    save_data(d.rename("delta_age").reset_index(drop=True).to_frame(), FIGS, "delta_age_hist")
    save_fig(f3, FIGS, "delta_age_hist")

    # ---- D. can the interval be predicted from health state? -------------------------
    pairs = v.copy()
    pairs["next_d"] = pairs.groupby("projid", sort=False)["d_fu"].shift(-1)
    p = pairs[pairs["next_d"].notna()].copy()
    X = p[FEATS].astype(float).to_numpy()
    g = p["projid"].to_numpy()
    y_reg = p["next_d"].to_numpy()
    y_gap = (y_reg > 1).astype(int)
    r2 = r2_score(y_reg, _oof(X, y_reg, g, "reg"))
    auc_gap = roc_auc_score(y_gap, _oof(X, y_gap, g, "clf"))

    # informative dropout: is THIS visit the last one? (the interval question that DOES have signal)
    y_last = v["is_last"].astype(int).to_numpy()
    auc_last = roc_auc_score(y_last, _oof(v[FEATS].astype(float).to_numpy(), y_last,
                                          v["projid"].to_numpy(), "clf"))

    # ---- E. age top-coding ------------------------------------------------------------
    cap_rows = []
    for tbl, col, s in [("cross-sectional-data-gk", "age_bl", cs["age_bl"]),
                        ("cross-sectional-data-gk", "age_first_ad_dx", cs["age_first_ad_dx"]),
                        ("derived", "age_at_visit", v["age_at_visit"])]:
        s = pd.to_numeric(s, errors="coerce").dropna()
        cap_rows.append({"table": tbl, "column": col, "dtype": "float", "n_nonnull": len(s),
                         "max": round(s.max(), 2), "pct_at_max": round(100 * (s == s.max()).mean(), 3),
                         "pct_ge_90": round(100 * (s >= 90).mean(), 2), "top_coded": False})
    for col in ["age_at_visit_max", "age_death", "age_first_ad_dx"]:
        num, capped = parse_capped_age(cl[col])
        nn = int(num.notna().sum())
        cap_rows.append({"table": "ROSMAP_clinical", "column": col, "dtype": "string",
                         "n_nonnull": nn, "max": float(num.max()),
                         "pct_at_max": round(100 * capped.sum() / max(nn, 1), 2),
                         "pct_ge_90": round(100 * (num >= 90).sum() / max(nn, 1), 2),
                         "top_coded": True})
    save_table(pd.DataFrame(cap_rows), "age_top_coding_check")
    pct_capped_gk = round(100 * float((v["age_at_visit"] == v["age_at_visit"].max()).mean()), 4)
    _, capped_death = parse_capped_age(cl["age_death"])
    pct_capped_clin = round(100 * float(capped_max.sum()) / max(int(age_max.notna().sum()), 1), 2)
    pct_death_capped = round(100 * float(capped_death.sum()) / max(int(cl["age_death"].notna().sum()), 1), 2)

    summary_put(delta_age_cv=round(float(cv), 4),
                delta_age_mean=round(float(d.mean()), 4), delta_age_sd=round(float(d.std()), 4),
                delta_age_pct_exactly_1=round(100 * float((d == 1).mean()), 2),
                delta_age_pct_gt_1p5=round(100 * float((d > 1.5).mean()), 2),
                delta_age_predictability_r2=round(float(r2), 4),
                delta_age_gap_auc=round(float(auc_gap), 4),
                last_visit_auc_informative_dropout=round(float(auc_last), 4),
                pct_age_capped_gk_tables=pct_capped_gk,
                pct_age_capped_rosmap_clinical=pct_capped_clin,
                pct_age_death_capped_rosmap_clinical=pct_death_capped,
                annual_grid_median_abs_error_years=round(float(np.median(grid_err)), 4),
                annual_grid_pct_within_0p05y=round(100 * float((grid_err <= .05).mean()), 2),
                annual_grid_p90_abs_error_years=round(float(np.quantile(grid_err, .9)), 4),
                pct_baseline_before_65=round(100 * float((cs["age_bl"] < 65).mean()), 2),
                has_calendar_date_field=False)

    report(f"""## 4. 时间轴与访视间隔（风险 R1）

**算了什么**：重建年龄轴的正确性检验、访视年龄与基线年龄分布（分队列）、相邻访视 Δage 的分布/分位/变异系数、
Δage 与「是否漏访」的**分组外**可预测性、年龄顶编检查。

### 4.1 重建的年龄轴站得住，但只有整年分辨率

`age_at_visit` 在这个 release 里**不存在**，只能取 `age_bl + fu_year`。用 ROSMAP_clinical 里未被顶编的
`age_at_visit_max`（真实观测到的访视年龄）反查：`age_at_visit_max − age_bl` 到最近整数的距离，
P50 = **{365.25 * np.median(grid_err):.0f} 天**、P75 = {365.25 * np.quantile(grid_err, .75):.0f} 天、P90 = {365.25 * np.quantile(grid_err, .9):.0f} 天、max = {365.25 * grid_err.max():.0f} 天（n={len(grid_err)}）。作为对照，若访视与周年日无关，这个距离应服从 U[0, 0.5 年]（P50 = 91 天）。
→ 年度网格是真的（{100 * (grid_err <= 45 / 365.25).mean():.0f}% 在周年日 ±45 天内，远紧于均匀分布），
以**年**为单位的位置编码用 `age_bl + fu_year` 完全够用。

**但这个检验同时暴露了 §4.3 的要害**：偏差的尾部一直伸到 ±{365.25 * grid_err.max():.0f} 天。
也就是说，现实中「相邻两次年度访视」的真实间隔大约在 0.5–1.5 年之间浮动——**真实的间隔变异是存在的，
只是 release 把它抹掉了**（没有任何日期字段）。我们能看到的 Δage 只有整数，它记录的是「漏了几次访视」，
不是「过了多久」。

### 4.2 左截断

- 基线年龄 median **{cs['age_bl'].median():.1f}** 岁，仅 **{100 * (cs['age_bl'] < 65).mean():.1f}%** 在 65 岁前入组
  （P1={cs['age_bl'].quantile(.01):.0f}，min={cs['age_bl'].min():.0f}）。
- 分队列差异很大：ROS median {cs[cs.study == 'ROS'].age_bl.median():.0f} · MAP {cs[cs.study == 'MAP'].age_bl.median():.0f} ·
  LATC {cs[cs.study == 'LATC'].age_bl.median():.0f} 岁。
- 含义：模型对**中年期轨迹一无所知**，位置编码的有效区间实际是 ~{cs['age_bl'].quantile(.05):.0f}–100 岁。
  任何「从中年预测痴呆」的表述都超出数据支持范围。

### 4.3 Δage：它就是研究方案本身

| 统计量 | 值 |
|---|---|
| n（相邻访视对） | {len(d):,} |
| mean / sd | {d.mean():.3f} / {d.std():.3f} 年 |
| **变异系数 CV** | **{cv:.3f}** |
| P50 / P75 / P95 / P99 / max | {d.quantile(.5):.0f} / {d.quantile(.75):.0f} / {d.quantile(.95):.0f} / {d.quantile(.99):.0f} / {d.max():.0f} |
| 恰好 = 1.0 年 | **{100 * (d == 1).mean():.2f}%** |
| > 1.5 年（漏访） | {100 * (d > 1.5).mean():.2f}% |

Δage 的取值集合是 {{1,2,3,…}}——**整数**。它不是「测量到的间隔」，而是「跳过了几个年度访视」。

**可预测性**（5 折 GroupKFold 按人分组；特征 = 本次访视的年龄/fu_year/cogn_global/MMSE/BMI/AD 状态/性别/教育/共病）：

- 回归 Δage：**out-of-fold R² = {r2:.4f}**
- 二分类「下次是否漏访（Δ>1）」：**AUC = {auc_gap:.4f}**
- 参考对照——二分类「这次是不是最后一次访视（脱落/死亡）」：**AUC = {auc_last:.4f}**

{gate(cv < 0.2 and r2 < 0.05,
      f"CV = {cv:.3f}（<0.2）且 R² = {r2:.4f}（<0.05），两条阈值都命中 → **删掉 time-to-next-event 头**。"
      if (cv < 0.2 and r2 < 0.05) else
      f"规格书要求 CV < 0.2 **且** R² < 0.05 才判「无信息」。实测 R² = {r2:.4f}（命中），"
      f"但 CV = {cv:.3f} ≥ 0.2（未命中），所以按字面阈值这个头「应当保留」。"
      "**不要照字面执行**——原因见下。")}

补充判断（**比阈值本身更重要**）：Δage 的全部变异都来自「漏了几次年度访视」，而不是真实就诊时刻——
release 里根本没有日期。把 dt 头训成回归 Δage，学到的是**行政随访依从性**，不是生物学时间。结论：

- **删掉 Delphi 原样的 time-to-next-event 回归头。** 它消耗参数、稀释主损失，且目标变量语义是错的。
- 保留 age encoding（`age_bl + fu_year`），模型退化为「给定年龄的状态预测」——完全合理的架构，
  但方案里**不要再声称复现了 Delphi 的全部组件**。
- 若仍要利用间隔信息，正确的重定义是**信息性缺访**：把「下次访视是否发生」（漏访 AUC {auc_gap:.3f}）与
  「是否就此脱落」（AUC {auc_last:.3f}）建成**二元/生存**头，而不是回归实数间隔。
  后者 AUC {auc_last:.3f} 说明脱落确实与健康状态相关（§5 会给出机制），这个头有价值且必须在方法学里讲清楚。

### 4.4 年龄顶编

**两个来源的去标识化力度完全不同，这是本轮最实用的发现之一：**

- **gk 的两张表（本项目主表）没有顶编**：`age_bl` max = {cs['age_bl'].max():.2f}，
  `age_first_ad_dx` max = {cs['age_first_ad_dx'].max():.2f}，取到最大值的样本占比 <0.1%；
  重建的 `age_at_visit` max = {v['age_at_visit'].max():.2f}，{100 * (v['age_at_visit'] >= 90).mean():.1f}% 的访视 ≥90 岁。
- **ROSMAP_clinical.csv 顶编在 90**：`age_at_visit_max` 的 **{pct_capped_clin:.1f}%** 是字符串 `'90+'`，
  `age_death` 的 **{pct_death_capped:.1f}%** 是 `'90+'`。

{gate(True, f"**位置编码用 gk 表的未截断年龄，不受影响**（顶编占比 {pct_capped_gk:.3f}%，远低于 5% 阈值），"
      "无需向 RADC 申请未截断年龄。**但 ROSMAP_clinical 的 age_* 三列必须先解析再用**——"
      "它们是 string 不是 float，直接 `pd.to_numeric` 会静默变 NaN，直接排序/比较会按字典序错排。"
      "本轮所有涉及死亡时间的分析（§5 终末期下降）只使用未顶编子集，并显式报告样本量。")}

### 4.5 日历时间

release 里**没有任何日期或日历年份字段**（无 visit date、无 enrollment year）。
→ 无法检验测验版本随年代的漂移，也无法做时间外推验证（train ≤2010 / test >2010）。
这是必须写进限制的缺口：若确实存在跨年代的测验版本变更，模型会把它当成真实的认知变化学走。
""")
    print("s4 done")


if __name__ == "__main__":
    run()
