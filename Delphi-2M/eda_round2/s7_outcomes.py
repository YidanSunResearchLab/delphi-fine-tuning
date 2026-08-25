"""Section 7 -- outcome definition, state transitions, class imbalance, clinical vs pathology."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score

from common import (C1, C2, C3, C4, REF_GREY, SEQ, SEQ_CMAP, MMSE_NAMES, FIGS, load,
                    parse_capped_age, report, save_data, save_fig, save_table, setup_style,
                    summary_put, gate)

SEV = MMSE_NAMES[::-1]          # normal -> severe, the order used everywhere below
COGDX = {1: "1 NCI", 2: "2 MCI", 3: "3 MCI + other", 4: "4 AD", 5: "5 AD + other",
         6: "6 other dementia"}
CERAD = {1: "1 definite AD", 2: "2 probable AD", 3: "3 possible AD", 4: "4 no AD"}


def run():
    setup_style()
    cs, lo, cl, v = load()
    v = v.sort_values(["projid", "fu_year"]).copy()

    # ---- 1. state occupancy ---------------------------------------------------------
    occ = v["mmse_bin"].value_counts(dropna=False).reindex(SEV + [np.nan])
    occ_tbl = pd.DataFrame({"state": [str(x) for x in occ.index], "n_visits": occ.to_numpy()})
    occ_tbl["pct"] = (100 * occ_tbl["n_visits"] / len(v)).round(2)
    occ_tbl.loc[occ_tbl["state"] == "nan", "state"] = "MMSE not measured"
    save_table(occ_tbl, "state_occupancy")

    # ---- 2. one-year transition matrix ----------------------------------------------
    g = v.groupby("projid", sort=False)
    v["next_bin"] = g["mmse_bin"].shift(-1)
    v["next_gap"] = g["fu_year"].shift(-1) - v["fu_year"]
    tr = v[v["mmse_bin"].notna() & v["next_bin"].notna() & v["next_gap"].eq(1)]
    ct = pd.crosstab(tr["mmse_bin"], tr["next_bin"]).reindex(index=SEV, columns=SEV, fill_value=0)
    row_n = ct.sum(axis=1)
    prob = (ct.div(row_n.replace(0, np.nan), axis=0) * 100)
    save_table(ct.reset_index().rename(columns={"mmse_bin": "from_state"}), "state_transition_counts")
    save_table(prob.round(2).reset_index().rename(columns={"mmse_bin": "from_state"}),
               "state_transition_probs_pct")

    idx = {s: i for i, s in enumerate(SEV)}
    fi = tr["mmse_bin"].map(idx).to_numpy()
    ti = tr["next_bin"].map(idx).to_numpy()
    rev = float((ti < fi).mean())
    worse = float((ti > fi).mean())
    stay = float((ti == fi).mean())

    f, ax = plt.subplots(figsize=(7.0, 5.4))
    im = ax.imshow(prob.to_numpy(), cmap=SEQ_CMAP, vmin=0, vmax=100)
    ax.set_xticks(range(4)); ax.set_xticklabels(SEV, rotation=20, ha="right", fontsize=8.5)
    ax.set_yticks(range(4)); ax.set_yticklabels([f"{s}\n(n={int(row_n[s])})" for s in SEV], fontsize=8.5)
    ax.grid(False)
    for i in range(4):
        for j in range(4):
            val = prob.iat[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.1f}%\n{int(ct.iat[i, j])}", ha="center", va="center",
                        fontsize=8, color="white" if val > 55 else "#222222")
    cb = f.colorbar(im, ax=ax, fraction=.043, pad=.02); cb.set_label("Row %")
    ax.set_xlabel("State at the next annual visit"); ax.set_ylabel("State at this visit")
    ax.set_title(f"One-year transitions between MMSE severity states (n={len(tr):,} pairs)\n"
                 f"{100 * rev:.1f}% of transitions REVERT to a milder state — "
                 "this cohort is not monotone")
    save_data(prob.round(3).reset_index(), FIGS, "state_transition_matrix")
    save_fig(f, FIGS, "state_transition_matrix")

    # ---- 2b. same matrix with death as an absorbing state (known-death subset) ---------
    dnum, dcap = parse_capped_age(cl.set_index("projid")["age_death"])
    per_death = pd.DataFrame({"age_death": dnum, "capped": dcap}).reindex(
        v["projid"].unique())
    known = per_death["age_death"].notna() & (~per_death["capped"].fillna(True).astype(bool))
    vd = v[v["projid"].isin(per_death.index[known])].copy()
    vd["age_death"] = vd["projid"].map(per_death["age_death"])
    vd["to_dead"] = vd["is_last"] & (vd["age_death"] <= vd["age_at_visit"] + 1.5)
    rows = []
    for s in SEV:
        sub = vd[vd["mmse_bin"].eq(s)]
        nxt = sub["next_bin"].where(sub["next_gap"].eq(1))
        cnt = {t: int((nxt == t).sum()) for t in SEV}
        cnt["dead within 1.5 y"] = int(sub["to_dead"].sum())
        tot = sum(cnt.values())
        rows.append({"from_state": s, "n": tot,
                     **{k: (round(100 * c / tot, 2) if tot else np.nan) for k, c in cnt.items()}})
    save_table(pd.DataFrame(rows), "state_transition_with_death_pct")
    death_from = {r["from_state"]: r["dead within 1.5 y"] for r in rows}

    # ---- 3. progression to AD ---------------------------------------------------------
    ad = cs[cs["age_first_ad_dx"].notna()].copy()
    ad = ad.merge(v.groupby("projid")["age_at_visit"].max().rename("age_last"), on="projid", how="left")
    ad["years_bl_to_dx"] = ad["age_first_ad_dx"] - ad["age_bl"]
    f, axes = plt.subplots(1, 2, figsize=(11.0, 4.1))
    ax = axes[0]
    ax.hist(ad["age_first_ad_dx"], bins=np.arange(60, 106, 2), color=C2, lw=0)
    ax.axvline(ad["age_first_ad_dx"].median(), color=C1, ls="--", lw=2)
    ax.annotate(f"median {ad['age_first_ad_dx'].median():.0f}", (ad["age_first_ad_dx"].median(),
                ax.get_ylim()[1] * .93), xytext=(6, 0), textcoords="offset points", color=C1, fontsize=9)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Participants")
    ax.set_title(f"Age at first AD dementia diagnosis (n={len(ad):,})")

    ax = axes[1]
    ax.hist(ad["years_bl_to_dx"], bins=np.arange(0, ad["years_bl_to_dx"].max() + 1, 1),
            color=C3, lw=0)
    ax.axvline(ad["years_bl_to_dx"].median(), color=C1, ls="--", lw=2)
    ax.annotate(f"median {ad['years_bl_to_dx'].median():.0f} y", (ad["years_bl_to_dx"].median(),
                ax.get_ylim()[1] * .93), xytext=(6, 0), textcoords="offset points", color=C1, fontsize=9)
    ax.set_xlabel("Years since baseline"); ax.set_ylabel("Participants")
    ax.set_title("Years from baseline to that first diagnosis")
    f.tight_layout()
    save_data(ad[["projid", "study", "age_bl", "age_first_ad_dx", "years_bl_to_dx"]],
              FIGS, "progression_age_dist")
    save_fig(f, FIGS, "progression_age_dist")

    prog = pd.DataFrame([
        {"metric": "participants with a first AD dementia dx", "value": len(ad),
         "pct_of_cohort": round(100 * len(ad) / len(cs), 2)},
        {"metric": "median age at first AD dx", "value": round(ad["age_first_ad_dx"].median(), 1),
         "pct_of_cohort": np.nan},
        {"metric": "median years baseline -> first AD dx",
         "value": round(ad["years_bl_to_dx"].median(), 1), "pct_of_cohort": np.nan},
        {"metric": "visits (pairs) with an incident dx -- prevalence of the modelling target",
         "value": np.nan, "pct_of_cohort": np.nan},
    ])
    save_table(prog, "ad_progression_summary")

    # ---- 4. clinical vs pathology -------------------------------------------------------
    path = cl[cl["ceradsc"].notna() & cl["cogdx"].notna()].copy()
    path["clinical"] = path["cogdx"].map(COGDX)
    path["pathology"] = path["ceradsc"].map(CERAD)
    xt = pd.crosstab(path["clinical"], path["pathology"])
    save_table(xt.reset_index(), "clinical_vs_pathology_crosstab", index=False)

    clin_ad = path["cogdx"].isin([4, 5]).astype(int)
    path_ad = path["ceradsc"].isin([1, 2]).astype(int)
    agree = float((clin_ad == path_ad).mean())
    kappa = float(cohen_kappa_score(clin_ad, path_ad))
    sens = float(path_ad[clin_ad == 1].mean())
    spec = float(1 - path_ad[clin_ad == 0].mean())

    # CERAD is ORDINAL (definite -> no AD), so this is a sequential ramp, not categorical hues.
    f, ax = plt.subplots(figsize=(7.6, 4.6))
    share = (xt.div(xt.sum(axis=1), axis=0) * 100)
    bottom = np.zeros(len(share))
    ramp = [SEQ[4], SEQ[3], SEQ[1], SEQ[0]]
    for j, col in enumerate(["1 definite AD", "2 probable AD", "3 possible AD", "4 no AD"]):
        if col not in share:
            continue
        ax.bar(range(len(share)), share[col], bottom=bottom, width=.66,
               color=ramp[j], edgecolor="white", lw=2, label=f"CERAD {col}")
        for i, (val, b) in enumerate(zip(share[col], bottom)):
            if val >= 7:
                ax.text(i, b + val / 2, f"{val:.0f}%", ha="center", va="center", fontsize=8,
                        color="white" if j == 0 else "#222222")
        bottom = bottom + share[col].to_numpy()
    ax.set_xticks(range(len(share)))
    nl = chr(10)
    ax.set_xticklabels([s.replace(" + ", nl + "+ ") + nl + f"n={int(xt.sum(axis=1)[s])}"
                        for s in share.index], fontsize=8)
    ax.set_ylabel("% of participants with that clinical diagnosis")
    ax.set_xlabel("Final clinical diagnosis (cogdx)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Clinical diagnosis vs CERAD neuritic-plaque score (n={len(path):,} autopsies)\n"
                 f"binary agreement {100 * agree:.0f}% · Cohen κ = {kappa:.2f} — these are two different targets")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper center",
              bbox_to_anchor=(0.5, -0.24), handlelength=1.6, columnspacing=1.0)
    save_data(xt.reset_index(), FIGS, "clinical_vs_pathology")
    save_fig(f, FIGS, "clinical_vs_pathology")

    summary_put(pct_visits_mmse_normal=round(float(occ_tbl.loc[occ_tbl.state == SEV[0], 'pct'].iloc[0]), 2),
                one_year_reversion_rate_pct=round(100 * rev, 2),
                one_year_progression_rate_pct=round(100 * worse, 2),
                one_year_stay_rate_pct=round(100 * stay, 2),
                n_ad_diagnosed=len(ad),
                pct_cohort_ad_diagnosed=round(100 * len(ad) / len(cs), 2),
                median_age_first_ad_dx=round(float(ad["age_first_ad_dx"].median()), 1),
                n_with_pathology=int(cs["braaksc"].notna().sum()),
                pct_with_pathology=round(100 * float(cs["braaksc"].notna().mean()), 2),
                clinical_pathology_agreement_pct=round(100 * agree, 2),
                clinical_pathology_kappa=round(kappa, 3))

    rev_pairs = int((ti < fi).sum())
    report(f"""## 7. 结局变量与类别不平衡

**这一节的第一件事是承认一个约束**：本 release **没有逐次访视的临床诊断 `dcfdx`**
（ROSMAP_clinical 只有末次访视的 `dcfdx_lv` 和最终的 `cogdx`，都是 person-level）。
因此「状态」只能用估计 MMSE 的严重度分档来定义，AD 事件用 person-level 的 `age_first_ad_dx` 还原到访视上。
**所有下面的转移概率都是 MMSE 分档的转移，不是 NCI/MCI/AD 的临床转移**——这个区别必须写进方法学。

### 7.1 状态占比（全部访视）

{occ_tbl.to_markdown(index=False)}

### 7.2 一年状态转移（仅 Δ=1 年的相邻访视，n={len(tr):,} 对）

行 = 本次状态，列 = 下次状态，单元格为行内百分比（计数见 `tables/state_transition_counts.csv`）：

{prob.round(1).to_markdown()}

- 保持原档 **{100 * stay:.1f}%** · 恶化 **{100 * worse:.1f}%** · **好转（逆转）{100 * rev:.1f}%**（{rev_pairs:,} 对）
- 加入死亡吸收态（仅 {int(known.sum()):,} 名死亡年龄可用者）：各档在末次访视后 1.5 年内死亡的比例
  {' · '.join(f"{k} {vv_:.1f}%" for k, vv_ in death_from.items())} —— 越严重越高，符合预期。

{gate(True, f"**逆转率 {100 * rev:.1f}% ≫ 5% → 绝对不能用单调进展假设的模型**（不能用只允许向下的多状态生存模型、"
      "也不能对预测施加单调约束）。生成式 transformer 在这里**反而有结构性优势**：它天然能表达状态往返。"
      "这一点应当直接写进方案的 motivation——这是本数据集少数几个真正支持用生成式序列模型的论据之一。"
      "但同时也要小心：这部分逆转里有相当一部分是 MMSE 的测量噪声而非真实好转，"
      "所以**主结局仍应是 `age_first_ad_dx` 这种一旦发生就不撤回的临床事件**，MMSE 分档只作为中间状态。")}

### 7.3 进展到 AD

- {len(ad):,} 人（**{100 * len(ad) / len(cs):.1f}%** 的队列）在随访中拿到首次 AD 痴呆诊断。
- 诊断年龄 median **{ad['age_first_ad_dx'].median():.1f}** 岁（P25 {ad['age_first_ad_dx'].quantile(.25):.0f} / P75 {ad['age_first_ad_dx'].quantile(.75):.0f}）。
- 基线到诊断 median **{ad['years_bl_to_dx'].median():.1f}** 年（P90 {ad['years_bl_to_dx'].quantile(.9):.0f} 年）。
- 但**建模用的正类率要低得多**：逐访视对的 `y_ad_next` 阳性率只有 ~3.3%（§6.1）。

{gate(False, "**极度不平衡（正类 3.3%，MMSE-normal 占访视 "
      f"{occ_tbl.loc[occ_tbl.state == SEV[0], 'pct'].iloc[0]:.0f}%）→ 损失函数必须加权（或用 focal loss），"
      "评估必须以 PR-AUC 为主、ROC-AUC 为辅。** 本轮所有 AUC 旁边都同时给了 PR-AUC，"
      "后续建模的报表要保持同样口径，否则 0.92 的 ROC-AUC 会掩盖一个几乎没用的 precision。")}

### 7.4 临床诊断 vs 神经病理

{int(cs['braaksc'].notna().sum()):,} 人（{100 * cs['braaksc'].notna().mean():.1f}%）有尸检病理。
`cogdx`（最终临床诊断）× `ceradsc`（CERAD 神经斑块评分，**反向编码：1=definite AD、4=no AD**）交叉表
（n={len(path):,}，见 `tables/clinical_vs_pathology_crosstab.csv`）：

{xt.to_markdown()}

二分化后（临床 AD = cogdx∈{{4,5}}；病理 AD = CERAD∈{{1,2}}）：
一致率 **{100 * agree:.1f}%**、**Cohen κ = {kappa:.2f}**、
以临床为准的病理「敏感度」{100 * sens:.1f}%、「特异度」{100 * spec:.1f}%。

{gate(False, f"**一致性只有 κ={kappa:.2f}（中等偏下）→ 必须明确本项目建的是「临床诊断」还是「病理」，两者不是同一个任务。**"
      " 本轮的建议：**主任务用临床诊断**（`age_first_ad_dx`），理由有三——(a) 只有它是逐时间点的、"
      "能进序列；(b) 病理只有一半人有，且**只有死亡后才存在**，作为标签会引入巨大的选择偏倚；"
      f"(c) 上表显示相当比例的 NCI 也有 definite/probable CERAD 病理，把病理当标签等于给一大批"
      "临床无症状者贴上阳性。病理指标可以作为**独立的下游验证**（模型预测的 AD 风险是否与病理负担相关），"
      "但绝不能进输入序列（见 §8）。")}
""")
    print("s7 done")


if __name__ == "__main__":
    run()
