"""Section 5 -- missingness structure, attrition, terminal decline."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from common import (C1, C2, C3, C4, REF_GREY, SEQ_CMAP, FIGS, LONG_CANDIDATES, load,
                    parse_capped_age, report, save_data, save_fig, save_table, setup_style,
                    summary_put, gate)


def run():
    setup_style()
    cs, lo, cl, v = load()

    # ---- A. item-level missingness by follow-up year -------------------------------
    v = v.copy()
    v["fu_bin"] = np.where(v["fu_year"] >= 20, 20, v["fu_year"])
    cols = [c for c in LONG_CANDIDATES if c in v.columns]
    order = v[cols].notna().mean().sort_values(ascending=False).index.tolist()
    miss = (v.groupby("fu_bin")[order].apply(lambda d: 100 * d.isna().mean())).T
    miss.columns = [f"{int(c)}" if c < 20 else "20+" for c in miss.columns]
    n_at = v.groupby("fu_bin").size()
    save_table(miss.round(2).reset_index().rename(columns={"index": "variable"}),
               "missingness_by_fu_year")

    # Does missingness DRIFT with follow-up, or is it a flat sub-study property? And is there
    # an odd/even cadence (a variable collected only every other cycle)?
    early = v[v["fu_year"] <= 2][cols].isna().mean() * 100
    late = v[v["fu_year"] >= 10][cols].isna().mean() * 100
    # Cadence must be the UNWEIGHTED mean of the per-cycle rates: pooling rows lets the big
    # early cycles swamp an every-other-year pattern that only shows up in the later ones.
    per_cycle = v[v["fu_year"].between(1, 19)].groupby("fu_year")[cols].apply(
        lambda d: 100 * d.isna().mean())
    odd = per_cycle[per_cycle.index % 2 == 1].mean(axis=0)
    even = per_cycle[per_cycle.index % 2 == 0].mean(axis=0)
    drift = pd.DataFrame({"pct_missing_fu0_2": early.round(1), "pct_missing_fu10plus": late.round(1),
                          "drift": (late - early).round(1),
                          "pct_missing_odd_cycles": odd.round(1),
                          "pct_missing_even_cycles": even.round(1),
                          "odd_minus_even": (odd - even).round(1)}).sort_values("drift", ascending=False)
    drift.index.name = "variable"
    save_table(drift.reset_index(), "missingness_drift_and_cadence")
    drifters = drift[drift["drift"] >= 15].index.tolist()
    biennial = drift[drift["odd_minus_even"].abs() >= 5].index.tolist()

    f, ax = plt.subplots(figsize=(11.0, 8.0))
    im = ax.imshow(miss.to_numpy(), aspect="auto", cmap=SEQ_CMAP, vmin=0, vmax=100)
    ax.set_xticks(range(miss.shape[1])); ax.set_xticklabels(miss.columns, fontsize=8)
    ax.set_yticks(range(miss.shape[0])); ax.set_yticklabels(miss.index, fontsize=8)
    ax.set_xlabel("Follow-up cycle (fu_year)"); ax.set_ylabel("Variable")
    ax.grid(False)
    for i in range(miss.shape[0]):
        for j in range(miss.shape[1]):
            val = miss.iat[i, j]
            if np.isfinite(val) and (val >= 99.5 or val <= 0.5 or (i % 4 == 0 and j % 4 == 0)):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6,
                        color="white" if val > 55 else "#333333")
    cb = f.colorbar(im, ax=ax, fraction=.025, pad=.015)
    cb.set_label("% missing among visits that DID occur")
    ax.set_title("Item-level missingness is a property of the sub-study, not of attrition\n"
                 "(rows sorted by overall completeness; counts of visits per cycle in the CSV)")
    save_data(miss.round(3).reset_index().rename(columns={"index": "variable"}),
              FIGS, "missingness_by_fu_year")
    save_fig(f, FIGS, "missingness_by_fu_year")

    # ---- B. the three kinds of "missing" --------------------------------------------
    per = v.groupby("projid").agg(first_fu=("fu_year", "min"), last_fu=("fu_year", "max"),
                                  n_obs=("fu_year", "size"), died=("died", "first"),
                                  age_bl=("age_bl", "first"), age_last=("age_at_visit", "max"))
    per["interior_gap_cycles"] = (per["last_fu"] - per["first_fu"] + 1) - per["n_obs"]
    dnum, dcap = parse_capped_age(cl.set_index("projid")["age_death"])
    per["age_death"] = dnum.reindex(per.index)
    # reindex turns a bool Series into object dtype; `~object` yields -1/-2 (both truthy),
    # which silently made every top-coded death look usable. Cast back to bool.
    per["age_death_capped"] = dcap.reindex(per.index).fillna(False).astype(bool)
    known_death = per["died"].eq(1) & per["age_death"].notna() & (~per["age_death_capped"])
    per["years_last_visit_to_death"] = np.where(known_death, per["age_death"] - per["age_last"], np.nan)

    obs = int(per["n_obs"].sum())
    gaps = int(per["interior_gap_cycles"].sum())
    acct = pd.DataFrame([
        {"kind": "observed visit (a row exists)", "unit": "person-cycle", "n": obs,
         "pct_of_grid": round(100 * obs / (obs + gaps), 2),
         "note": "item-level missingness inside these rows is the ONLY imputable kind"},
        {"kind": "missed cycle inside the observed window", "unit": "person-cycle", "n": gaps,
         "pct_of_grid": round(100 * gaps / (obs + gaps), 2),
         "note": "person came back later -> a real hole in the sequence; Δage>1 encodes it"},
        {"kind": "after the last visit -- died", "unit": "person", "n": int(per["died"].eq(1).sum()),
         "pct_of_grid": np.nan, "note": "structural, NOT imputable; must be an absorbing token"},
        {"kind": "after the last visit -- alive at freeze (censored)", "unit": "person",
         "n": int(per["died"].eq(0).sum()), "pct_of_grid": np.nan,
         "note": "right censoring; not a state, must not be tokenised as an event"},
    ])
    save_table(acct, "missingness_accounting")

    # ---- C. attrition ----------------------------------------------------------------
    att = pd.DataFrame([
        {"reason": "died (died==1)", "n": int(per["died"].eq(1).sum()),
         "pct": round(100 * per["died"].eq(1).mean(), 2),
         "available_in_release": "yes -- cross-sectional-data-gk.died"},
        {"reason": "  of which: death age known and NOT top-coded",
         "n": int(known_death.sum()), "pct": round(100 * known_death.mean(), 2),
         "available_in_release": "ROSMAP_clinical.age_death (ROS+MAP only, 90+ top-coded)"},
        {"reason": "  of which: last visit within 2 y of death",
         "n": int((per.loc[known_death, "years_last_visit_to_death"] <= 2).sum()),
         "pct": round(100 * (per.loc[known_death, "years_last_visit_to_death"] <= 2).mean(), 2),
         "available_in_release": "derived"},
        {"reason": "alive at data freeze (censored)", "n": int(per["died"].eq(0).sum()),
         "pct": round(100 * per["died"].eq(0).mean(), 2),
         "available_in_release": "yes"},
        {"reason": "refused / lost to follow-up / administrative censoring",
         "n": np.nan, "pct": np.nan,
         "available_in_release": "NO -- this release has no withdrawal-reason field; "
                                 "alive-but-stopped cannot be separated from still-active"},
    ])
    save_table(att, "attrition_reasons")

    # ---- D. terminal decline ----------------------------------------------------------
    td = v.merge(per[["age_death", "age_death_capped"]], on="projid", how="left")
    td = td[td["age_death"].notna() & (~td["age_death_capped"]) & td["cogn_global"].notna()].copy()
    td["ybd"] = td["age_death"] - td["age_at_visit"]
    edges = [0, 1, 2, 3, 5, 10, 100]
    labels = ["0-1", "1-2", "2-3", "3-5", "5-10", "10+"]
    td["band"] = pd.cut(td["ybd"], edges, labels=labels, right=False)
    agg = (td.groupby("band", observed=True)["cogn_global"]
             .agg(["mean", "sem", "size"]).reindex(labels).dropna())
    ctrl = v.loc[v["died"].eq(0) & v["cogn_global"].notna(), "cogn_global"]

    f, ax = plt.subplots(figsize=(6.8, 4.4))
    x = np.arange(len(agg))
    ax.errorbar(x, agg["mean"], yerr=1.96 * agg["sem"], color=C2, lw=2.5, marker="o", ms=9,
                mec="white", mew=1.5, capsize=4)
    ax.axhline(ctrl.mean(), color=REF_GREY, ls="--", lw=1.5)
    ax.annotate(f"visits by participants alive at the data freeze ({ctrl.mean():+.2f})",
                xy=(0.985, ctrl.mean()), xycoords=("axes fraction", "data"), xytext=(0, 6),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=REF_GREY)
    for xi, (m, n) in enumerate(zip(agg["mean"], agg["size"])):
        ax.annotate(f"{m:+.2f}\nn={int(n)}", (xi, m), xytext=(0, -30), textcoords="offset points",
                    ha="center", fontsize=8, color=C2,
                    bbox=dict(fc="white", ec="none", alpha=.8, pad=1.2))
    ax.set_xticks(x); ax.set_xticklabels(agg.index)
    ax.set_ylim(agg["mean"].min() - .28, max(agg["mean"].max(), ctrl.mean()) + .14)
    ax.invert_xaxis()
    ax.set_xlabel("Years before death (visit binned)"); ax.set_ylabel("Global cognition (z)")
    drop = agg["mean"].iloc[0] - agg["mean"].iloc[-1]
    ax.set_title(f"Terminal decline in global cognition: {drop:+.2f} z from >10 y out to the final year\n"
                 f"visits of the {td.projid.nunique()} decedents whose age at death is not top-coded")
    save_data(agg.reset_index(), FIGS, "terminal_decline")
    save_fig(f, FIGS, "terminal_decline")

    # ---- E. baseline characteristics by vital status ------------------------------------
    bl = v[v["visit_idx"] == 0].set_index("projid")
    bl = bl.join(cs.set_index("projid")[["apoe_genotype"]], rsuffix="_cs")
    bl["apoe4"] = bl["apoe_genotype"].isin([24, 34, 44]).astype(float)
    bl.loc[bl["apoe_genotype"].isna(), "apoe4"] = np.nan
    bl["n_visits_total"] = per["n_obs"]
    bl["ad_ever"] = bl["age_first_ad_dx"].notna().astype(int)
    rows = []
    for var in ["age_bl", "educ", "msex", "cogn_global", "cts_estmmse30", "apoe4",
                "n_visits_total", "ad_ever"]:
        a = pd.to_numeric(bl.loc[bl["died"].eq(1), var], errors="coerce").dropna()
        b = pd.to_numeric(bl.loc[bl["died"].eq(0), var], errors="coerce").dropna()
        t, p = stats.ttest_ind(a, b, equal_var=False)
        rows.append({"variable": var, "died_n": len(a), "died_mean": round(a.mean(), 3),
                     "died_sd": round(a.std(), 3), "alive_n": len(b), "alive_mean": round(b.mean(), 3),
                     "alive_sd": round(b.std(), 3), "diff": round(a.mean() - b.mean(), 3),
                     "welch_t": round(t, 2), "p_value": f"{p:.2e}"})
    bvs = pd.DataFrame(rows)
    save_table(bvs, "baseline_by_vital_status")

    # last-visit cognition of droppers vs the cohort (informative censoring, alive only)
    last_alive = v[v["is_last"] & v["died"].eq(0) & v["cogn_global"].notna()]
    nonlast = v[(~v["is_last"]) & v["cogn_global"].notna()]
    t_drop, p_drop = stats.ttest_ind(last_alive["cogn_global"], nonlast["cogn_global"], equal_var=False)

    summary_put(n_missed_cycles_inside_window=gaps,
                pct_missed_cycles_inside_window=round(100 * gaps / (obs + gaps), 2),
                n_died=int(per["died"].eq(1).sum()), pct_died=round(100 * per["died"].eq(1).mean(), 2),
                n_death_age_usable=int(known_death.sum()),
                terminal_decline_z_drop=round(float(drop), 3),
                terminal_decline_final_year_mean_z=round(float(agg["mean"].iloc[0]), 3),
                survivor_visit_mean_z=round(float(ctrl.mean()), 3),
                dropper_last_visit_cogn_z=round(float(last_alive["cogn_global"].mean()), 3),
                dropper_vs_ongoing_p=f"{p_drop:.2e}",
                withdrawal_reason_available=False)

    report(f"""## 5. 缺失与脱落结构

**算了什么**：逐字段缺失率 × fu_year 热力图；把「缺失」拆成三类分别计数；脱落原因构成；
以死亡为对齐点的终末期认知曲线；死亡者 vs 存活者的基线对比。

### 5.1 缺失的三种类型（**不要混为一谈**）

| 类型 | 计数 | 说明 |
|---|---|---|
| 已发生的访视（有行） | {obs:,} person-cycle（占观测窗内网格 {100 * obs / (obs + gaps):.1f}%） | 行内的 item-level 缺失是**唯一**可插补的一类 |
| 观测窗**内部**漏掉的 cycle | {gaps:,} person-cycle（{100 * gaps / (obs + gaps):.1f}%） | 人后来又回来了 → 序列里真实的洞，由 Δage>1 编码 |
| 末次访视之后 —— 已死亡 | {int(per['died'].eq(1).sum()):,} 人（{100 * per['died'].eq(1).mean():.1f}%） | 结构性、**不可插补**，必须建成吸收态 |
| 末次访视之后 —— freeze 时仍存活 | {int(per['died'].eq(0).sum()):,} 人（{100 * per['died'].eq(0).mean():.1f}%） | 右删失，**不是事件**，不能 tokenize 成事件 |

热力图把 item-level 缺失分成了三种完全不同的东西（明细见 `tables/missingness_drift_and_cadence.csv`）：

1. **全程近乎完整**：`cogn_global`（{100 * v['cogn_global'].notna().mean():.1f}% 非空）、
   `cts_estmmse30`（{100 * v['cts_estmmse30'].notna().mean():.1f}%）、各 `*_rx` 用药与
   `dm_cum`/`hypertension_cum` 等共病（~98–99%）。这些才是能当一等序列 token 的变量。
2. **子研究覆盖面导致的高缺失，且不随年数变化**：`log_hcrp`/`log_hil6`/`log_htnfa` 全程 ~97–100% 缺失，
   `berlin_risk_class`（睡眠子研究）~71–84%，`hba1c` ~46–68%、血脂 ~38–51%。
   这是「这个人有没有参加这个子研究」，不是「这次访视漏测了」。
   → **不应当作一等序列 token**，否则词表里绝大多数位置是「未测」。
3. **随随访年数明显恶化**（drift ≥ 15 个百分点）：{', '.join(f'`{x}`' for x in drifters) if drifters else '（无）'}。
   例如 `bmi` 从 fu0–2 的 {drift.loc['bmi', 'pct_missing_fu0_2']:.0f}% 涨到 fu≥10 的 {drift.loc['bmi', 'pct_missing_fu10plus']:.0f}%，
   `psqi_sum` {drift.loc['psqi_sum', 'pct_missing_fu0_2']:.0f}% → {drift.loc['psqi_sum', 'pct_missing_fu10plus']:.0f}%。
   **这类缺失本身携带信息**（越老越衰弱越测不到），把它当随机缺失插补会抹掉信号；
   正确做法是给每个这类变量配一个显式的 `<not_measured>` token，让模型看见「这次没测」。

另外一个只有画了热力图才会发现的东西：**`bmi` 有明显的奇偶年周期**——
逐 cycle 缺失率在奇数 cycle 平均 {drift.loc['bmi', 'pct_missing_odd_cycles']:.0f}%、偶数 cycle 平均
{drift.loc['bmi', 'pct_missing_even_cycles']:.0f}%（差 {drift.loc['bmi', 'odd_minus_even']:.0f} 个百分点），
说明它对一部分参与者是**隔年测量**的。其余变量都没有这个模式
（|奇−偶| ≥ 5 个百分点的变量只有：{', '.join(f'`{x}`' for x in biennial)}）。
若不处理，模型会把一个纯粹由研究方案造成的两年周期学成生物学节律——这也是把 `bmi`
降级为静态协变量（§6.2）的又一个理由。

### 5.2 脱落

- 死亡 {int(per['died'].eq(1).sum()):,} 人（{100 * per['died'].eq(1).mean():.1f}%）；其中死亡年龄可用（非顶编、且在 ROSMAP_clinical 里）的只有 **{int(known_death.sum()):,} 人**。
- 这 {int(known_death.sum()):,} 人中，**{100 * (per.loc[known_death, 'years_last_visit_to_death'] <= 2).mean():.1f}%** 的末次访视在死亡前 2 年内
  → 随访基本追到了终点，死亡不是「悄悄消失」。
- **release 里没有退出原因字段**：拒访 / 失联 / 行政删失三者无法区分，「存活但停访」与「存活且仍在随访」也无法区分。
  这是一个必须声明的限制，规格书要求的 attrition_reasons 表只能填到这个粒度。

### 5.3 终末期下降

对齐死亡时点后，`cogn_global` 均值从死亡前 >10 年的 **{agg['mean'].iloc[-1]:+.3f}** 掉到最后 1 年的
**{agg['mean'].iloc[0]:+.3f}**，落差 **{drop:+.3f} z**（存活者访视均值参照线 {ctrl.mean():+.3f}）。
逐档：{' → '.join(f"{i} y: {m:+.2f}" for i, m in zip(agg.index[::-1], agg['mean'][::-1]))}。

另外，**仍存活但停止随访**的人，其末次访视 `cogn_global` 均值 {last_alive['cogn_global'].mean():+.3f}，
显著低于「不是末次访视」的访视均值 {nonlast['cogn_global'].mean():+.3f}（Welch p = {p_drop:.1e}）。
结合 §4 的「是否为末次访视」AUC = 0.70：**脱落是信息性的，认知越差越容易掉队。**

{gate(False, "**存在明显终末期下降 → 死亡必须建模为吸收态 token**，并且序列不能在死亡处「截断了事」："
      "生成采样时模型要能自己输出死亡，否则采样轨迹会无限延伸出一批不存在的高龄健康人。"
      "同时注意：死亡年龄只有 " + f"{int(known_death.sum()):,}" + " 人可用（其余顶编或不在 ROSMAP_clinical 中），"
      "对其余死亡者只能放一个「死亡发生在末次访视之后」的删失标记，不能编造时间。")}
{gate(False, "**脱落与认知状态强相关 → 信息性删失**。训练时不能把脱落当随机缺失；"
      "评估时必须额外报告「完整随访子集」（例如 ≥5 次访视者）上的指标作为敏感性分析，"
      "否则模型在「掉队的人后来怎么样了」这件事上会系统性乐观。")}
""")
    print("s5 done")


if __name__ == "__main__":
    run()
