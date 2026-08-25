"""Section 9 -- 5-fold participant-level CV, fold balance, and the baselines to beat."""
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score, r2_score, mean_absolute_error

from common import (C1, C2, C3, C4, REF_GREY, SEED, FIGS, TABLES, load, report, save_data,
                    save_fig, save_table, setup_style, summary_put, gate)
from labels import make_pairs, target_frame, TARGETS

N_FOLDS = 5
HORIZONS = [1, 2, 3, 5]

BASELINES = {
    "B1 demographics only": ["age_at_visit", "msex", "educ"],
    "B2 + last cognition": ["age_at_visit", "msex", "educ", "cogn_global"],
    "B3 + change + MMSE": ["age_at_visit", "msex", "educ", "cogn_global", "d_cogn", "cts_estmmse30"],
    "B0 last cognition ALONE": ["cogn_global"],
}


def clf_oof(X, y, folds):
    pred = np.full(len(y), np.nan)
    est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=4000, class_weight="balanced"))
    for k in np.unique(folds):
        tr, te = folds != k, folds == k
        if y[tr].sum() == 0:
            continue
        est.fit(X[tr], y[tr])
        pred[te] = est.predict_proba(X[te])[:, 1]
    ok = np.isfinite(pred)
    return roc_auc_score(y[ok], pred[ok]), average_precision_score(y[ok], pred[ok])


def run():
    setup_style()
    cs, lo, cl, v = load()
    pairs = make_pairs(v)
    try:
        icc_cogn = float(pd.read_csv(os.path.join(TABLES, 'icc_by_variable.csv'))
                         .set_index('variable').loc['cogn_global', 'icc1'])
    except Exception:
        icc_cogn = float('nan')

    # ---- 1. folds, assigned at the PARTICIPANT level ---------------------------------
    per = (v.groupby("projid")
             .agg(study=("study", "first"), died=("died", "first"), msex=("msex", "first"),
                  age_bl=("age_bl", "first"), n_visits=("fu_year", "size"),
                  bl_mmse=("cts_estmmse30", "first"))
             .reset_index())
    per["ad_ever"] = per["projid"].map(cs.set_index("projid")["age_first_ad_dx"].notna()).astype(int)
    per["strata"] = (per["study"] + "|" + per["died"].astype(str) + "|" + per["ad_ever"].astype(str))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    per["fold"] = -1
    for k, (_, te) in enumerate(skf.split(per, per["strata"])):
        per.loc[per.index[te], "fold"] = k
    save_table(per[["projid", "study", "fold"]], "cv_folds")

    bal = (per.groupby("fold")
             .agg(n_participants=("projid", "size"), pct_ROS=("study", lambda s: 100 * (s == "ROS").mean()),
                  pct_MAP=("study", lambda s: 100 * (s == "MAP").mean()),
                  pct_LATC=("study", lambda s: 100 * (s == "LATC").mean()),
                  pct_male=("msex", lambda s: 100 * s.mean()),
                  mean_age_bl=("age_bl", "mean"), pct_died=("died", lambda s: 100 * s.mean()),
                  pct_ad_ever=("ad_ever", lambda s: 100 * s.mean()),
                  mean_bl_mmse=("bl_mmse", "mean"), mean_visits=("n_visits", "mean"))
             .round(2).reset_index())
    bal["n_visits"] = per.groupby("fold")["n_visits"].sum().to_numpy()
    save_table(bal, "cv_fold_balance")

    fold_of = per.set_index("projid")["fold"]

    # ---- 2. baselines on the two 1-step targets ---------------------------------------
    rows = []
    for target in TARGETS:
        tf = target_frame(pairs, target)
        folds = tf["projid"].map(fold_of).to_numpy()
        y = tf["y"].to_numpy()
        for name, feats in BASELINES.items():
            X = tf[feats].astype(float).to_numpy()
            auc, ap = clf_oof(X, y, folds)
            rows.append({"target": target, "model": name, "features": ", ".join(feats),
                         "n": len(y), "n_pos": int(y.sum()),
                         "prevalence_pct": round(100 * y.mean(), 2),
                         "roc_auc": round(auc, 4), "pr_auc": round(ap, 4),
                         "pr_auc_lift_over_prevalence": round(ap / y.mean(), 2)})
    base = pd.DataFrame(rows)

    # ---- 3. next-cognition regression baselines (the LME family) -----------------------
    reg_rows = []
    tf = pairs[pairs["next_cogn"].notna() & pairs["cogn_global"].notna()].copy()
    folds = tf["projid"].map(fold_of).to_numpy()
    ytrue = tf["next_cogn"].to_numpy(float)

    # (a) LOCF
    reg_rows.append(("LOCF (carry last cognition forward)", tf["cogn_global"].to_numpy(float)))

    # (b) person-specific linear extrapolation from PRIOR visits only
    ext = np.full(len(tf), np.nan)
    hist = v.sort_values(["projid", "fu_year"])
    for pid, sub in hist.groupby("projid", sort=False):
        s = sub[["fu_year", "cogn_global"]].dropna()
        rows_i = np.flatnonzero(tf["projid"].to_numpy() == pid)
        for i in rows_i:
            t0 = tf["fu_year"].to_numpy()[i]
            past = s[s["fu_year"] <= t0]
            tgt = tf["next_fu"].to_numpy()[i]
            if len(past) >= 3:
                b, a = np.polyfit(past["fu_year"], past["cogn_global"], 1)
                ext[i] = a + b * tgt
            elif len(past) >= 1:
                ext[i] = past["cogn_global"].iloc[-1]
    reg_rows.append(("Per-person linear extrapolation (prior visits only)", ext))

    # (c) population linear mixed model, fixed effects only (a test participant is unseen)
    try:
        import statsmodels.formula.api as smf
        fit_src = v.dropna(subset=["cogn_global", "age_at_visit", "educ", "msex"]).copy()
        fit_src["fold"] = fit_src["projid"].map(fold_of)
        lme_pred = np.full(len(tf), np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in range(N_FOLDS):
                tr = fit_src[fit_src["fold"] != k]
                md = smf.mixedlm("cogn_global ~ age_at_visit + fu_year + msex + educ", tr,
                                 groups=tr["projid"], re_formula="~fu_year")
                res = md.fit(method="lbfgs", maxiter=200)
                te = tf[tf["projid"].map(fold_of) == k].copy()
                te["age_at_visit"] = te["next_age"]
                te["fu_year"] = te["next_fu"]
                lme_pred[(tf["projid"].map(fold_of) == k).to_numpy()] = res.predict(te).to_numpy()
        reg_rows.append(("Linear mixed model, population fixed effects", lme_pred))
    except Exception as e:                                    # noqa: BLE001
        print("  LME skipped:", e)

    # (d) ridge-style regression on the same features as B3
    feats = ["age_at_visit", "msex", "educ", "cogn_global", "d_cogn", "cts_estmmse30"]
    pred = np.full(len(tf), np.nan)
    est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LinearRegression())
    X = tf[feats].astype(float).to_numpy()
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        est.fit(X[tr], ytrue[tr])
        pred[te] = est.predict(X[te])
    reg_rows.append(("Linear regression on B3 features", pred))

    reg = []
    for name, p in reg_rows:
        ok = np.isfinite(p)
        reg.append({"model": name, "n": int(ok.sum()),
                    "r2": round(r2_score(ytrue[ok], p[ok]), 4),
                    "mae": round(mean_absolute_error(ytrue[ok], p[ok]), 4),
                    "rmse": round(float(np.sqrt(np.mean((ytrue[ok] - p[ok]) ** 2))), 4)})
    reg = pd.DataFrame(reg).sort_values("r2", ascending=False)
    save_table(reg, "baseline_next_cognition_regression")

    # ---- 4. horizon analysis: where is the headroom? ------------------------------------
    hz = []
    vv = v.sort_values(["projid", "fu_year"]).copy()
    vv["age_last"] = vv.groupby("projid")["age_at_visit"].transform("max")
    vv["d_cogn"] = vv.groupby("projid")["cogn_global"].diff()
    for h in HORIZONS:
        at_risk = (~vv["ad_now"].astype(bool)) & (~vv["projid"].isin(
            pairs.loc[pairs["suspected_prevalent_dementia"], "projid"].unique()))
        event = vv["age_first_ad_dx"].notna() & (vv["age_first_ad_dx"] <= vv["age_at_visit"] + h + 1e-9)
        complete = vv["age_last"] >= vv["age_at_visit"] + h - 1e-9
        elig = at_risk & (event | complete)
        sub = vv[elig].copy()
        sub["y"] = event[elig].astype(int)
        fo = sub["projid"].map(fold_of).to_numpy()
        for name, feats in BASELINES.items():
            auc, ap = clf_oof(sub[feats].astype(float).to_numpy(), sub["y"].to_numpy(), fo)
            hz.append({"horizon_years": h, "model": name, "n": len(sub),
                       "n_pos": int(sub["y"].sum()), "prevalence_pct": round(100 * sub["y"].mean(), 2),
                       "roc_auc": round(auc, 4), "pr_auc": round(ap, 4),
                       "n_dropped_censored": int((at_risk & ~(event | complete)).sum())})
    hz = pd.DataFrame(hz)
    hz["pr_auc_lift_over_prevalence"] = (hz["pr_auc"] / (hz["prevalence_pct"] / 100)).round(2)
    save_table(hz, "baseline_horizon_analysis")
    save_table(pd.concat([base, hz.assign(target=lambda d: "y_ad_within_" + d.horizon_years.astype(str) + "y",
                                          features="")], ignore_index=True),
               "baseline_model_performance")

    f, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ax = axes[0]
    for name, col in zip(BASELINES, [C3, C1, C4, C2]):
        s = hz[hz.model == name].sort_values("horizon_years")
        ax.plot(s["horizon_years"], s["roc_auc"], lw=2, marker="o", ms=8, mec="white", mew=1.3,
                color=col, label=name)
        # direct-label only the envelope, so the two middle curves do not collide
        if name in ("B3 + change + MMSE", "B1 demographics only"):
            ax.annotate(f"{s['roc_auc'].iloc[-1]:.3f}",
                        (s["horizon_years"].iloc[-1], s["roc_auc"].iloc[-1]),
                        xytext=(7, 0), textcoords="offset points", fontsize=8.5, color=col,
                        va="center", ha="left")
    ax.axhline(.85, color=REF_GREY, ls="--", lw=1.5)
    ax.annotate("0.85 — the spec's 'no headroom left' line", xy=(0.99, .85),
                xycoords=("axes fraction", "data"), xytext=(0, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=8.5, color=REF_GREY)
    ax.set_xticks(HORIZONS); ax.set_xlabel("Prediction horizon (years)"); ax.set_ylabel("ROC-AUC")
    ax.set_title("Baseline ROC-AUC barely moves with the horizon —\n"
                 "a logistic regression is already strong at every horizon")
    ax.set_xlim(0.7, 5.9)
    ax.legend(frameon=False, fontsize=8, loc="center left")

    ax = axes[1]
    for name, col in zip(BASELINES, [C3, C1, C4, C2]):
        s = hz[hz.model == name].sort_values("horizon_years")
        ax.plot(s["horizon_years"], s["pr_auc"], lw=2, marker="s", ms=8, mec="white", mew=1.3,
                color=col, label=name)
    prev = hz.groupby("horizon_years")["prevalence_pct"].first() / 100
    ax.plot(prev.index, prev.to_numpy(), color=REF_GREY, ls=":", lw=2, label="prevalence (random)")
    ax.set_xticks(HORIZONS); ax.set_xlabel("Prediction horizon (years)"); ax.set_ylabel("PR-AUC")
    ax.set_title("PR-AUC rises only because prevalence rises\n"
                 f"({hz.prevalence_pct.min():.0f}% at 1 y → {hz.prevalence_pct.max():.0f}% at 5 y)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    f.tight_layout()
    save_data(hz, FIGS, "baseline_horizon")
    save_fig(f, FIGS, "baseline_horizon")

    b0 = base[(base.model == "B0 last cognition ALONE") & (base.target == "y_ad_next")].iloc[0]
    b1 = base[(base.model == "B1 demographics only") & (base.target == "y_ad_next")].iloc[0]
    b3 = base[(base.model == "B3 + change + MMSE") & (base.target == "y_ad_next")].iloc[0]
    h5 = hz[(hz.model == "B3 + change + MMSE") & (hz.horizon_years == 5)].iloc[0]
    h1 = hz[(hz.model == "B3 + change + MMSE") & (hz.horizon_years == 1)].iloc[0]
    b0h1 = hz[(hz.model == "B0 last cognition ALONE") & (hz.horizon_years == 1)].iloc[0]
    b0h5 = hz[(hz.model == "B0 last cognition ALONE") & (hz.horizon_years == 5)].iloc[0]
    no_headroom = float(b0["roc_auc"]) >= .85

    summary_put(n_cv_folds=N_FOLDS,
                cv_fold_min_participants=int(bal["n_participants"].min()),
                cv_fold_max_participants=int(bal["n_participants"].max()),
                baseline_b0_last_cognition_auc=float(b0["roc_auc"]),
                baseline_b0_last_cognition_pr_auc=float(b0["pr_auc"]),
                baseline_b1_demographics_auc=float(b1["roc_auc"]),
                baseline_b3_full_auc=float(b3["roc_auc"]),
                baseline_b3_full_pr_auc=float(b3["pr_auc"]),
                baseline_auc_1y=float(h1["roc_auc"]), baseline_auc_5y=float(h5["roc_auc"]),
                baseline_pr_auc_5y=float(h5["pr_auc"]),
                best_next_cognition_r2=float(reg["r2"].max()),
                best_next_cognition_model=str(reg.iloc[0]["model"]),
                transformer_must_beat={"y_ad_next_roc_auc": float(b3["roc_auc"]),
                                       "y_ad_next_pr_auc": float(b3["pr_auc"]),
                                       "y_ad_within_5y_roc_auc": float(h5["roc_auc"]),
                                       "y_ad_within_5y_pr_auc": float(h5["pr_auc"]),
                                       "next_cogn_global_r2": float(reg["r2"].max())},
                headroom_over_single_variable_baseline=not no_headroom)

    report(f"""## 9. 划分与基线

**算了什么**：按 `projid` 的 5 折分层划分（同一人全部访视同折）、折间均衡检查、
三组分类基线 + 四组认知轨迹回归基线、以及**预测跨度**分析。

### 9.1 划分

5 折，**按人**划分，分层变量 = `study × died × ad_ever`，种子 42，存于 `tables/cv_folds.csv`。
每折 {bal['n_participants'].min()}–{bal['n_participants'].max()} 人 / {bal['n_visits'].min():,}–{bal['n_visits'].max():,} 访视。折间均衡：

{bal.to_markdown(index=False)}

分箱切点、z 分基准、缺失填补统计量**全部只在训练折上估计后冻结**，验证折复用同一组常数
（本节所有数字都已按此执行）。这一条要写进配置文件，不要留给运行时推断。

### 9.2 分类基线（下一次访视）

{base[['target', 'model', 'n', 'n_pos', 'prevalence_pct', 'roc_auc', 'pr_auc', 'pr_auc_lift_over_prevalence']].to_markdown(index=False)}

### 9.3 认知轨迹回归基线（预测下一次 `cogn_global`）

{reg.to_markdown(index=False)}

最好的基线是 **{reg.iloc[0]['model']}**，R² = **{reg['r2'].max():.4f}**、MAE = {reg.iloc[0]['mae']:.4f}。
注意：LOCF 这种「什么都不做」的基线已经能到 R² {float(reg[reg.model.str.startswith('LOCF')]['r2'].iloc[0]):.3f}，
因为 `cogn_global` 的 ICC 是 {icc_cogn:.2f}（§6.2）——**大部分「预测」其实只是在复述这个人的水平**。
序列模型的价值必须体现在**残差**上（预测变化量），而不是在这个 R² 上。
表里的混合效应模型 R² 为负，不是实现错误：对一个**从未见过的**测试参与者，只能用总体固定效应预测，个体随机效应无从估计。「按人线性外推」那一行才是「有了这个人的历史之后混合效应模型能做到什么」的现实近似。

### 9.4 跨度分析：headroom 在哪里

{hz.pivot_table(index='model', columns='horizon_years', values='roc_auc').round(4).to_markdown()}

（对应的 PR-AUC 与样本量见 `tables/baseline_horizon_analysis.csv`；每个跨度都只保留
「事件已发生」或「随访确实够长」的访视，随访不足的按删失剔除。）

PR-AUC 相对基础率的倍数（lift），这是比 AUC 更诚实的读数：

{hz.pivot_table(index='model', columns='horizon_years', values='pr_auc_lift_over_prevalence').round(2).to_markdown()}

**必须直说的一件事：拉长跨度并没有救回多少 headroom。** B3 的 ROC-AUC 从 1 年的
{h1['roc_auc']:.3f} 到 5 年的 {h5['roc_auc']:.3f}，只掉了 {h1['roc_auc'] - h5['roc_auc']:.3f}；
PR-AUC 反而从 {h1['pr_auc']:.3f} 涨到 {h5['pr_auc']:.3f}——但那**纯粹是因为正类率从
{h1['prevalence_pct']:.1f}% 涨到 {h5['prevalence_pct']:.1f}%**，相对 lift 其实从
{h1['pr_auc_lift_over_prevalence']:.1f}× 掉到 {h5['pr_auc_lift_over_prevalence']:.1f}×。
换句话说，**在任何跨度上，一个逻辑回归都已经很强了。**

{gate(not no_headroom, (f"**「上次认知分」这一个变量在 1 年跨度上就有 AUC {b0h1['roc_auc']:.3f}"
      f"（§9.2 的「下一次访视」口径是 {b0['roc_auc']:.3f}），5 年跨度上仍有 {b0h5['roc_auc']:.3f}——"
      "全都远高于 0.85 → 按规格书，留给序列模型的提升空间很小，任务必须重新定义。**" if no_headroom else
      f"「上次认知分」单变量 AUC {b0h1['roc_auc']:.3f}，尚未触到 0.85 的天花板。")
      + " 而且这一轮的跨度分析显示，**把跨度拉到 3–5 年并不能变出空间**（见上）。"
      " 因此重定义任务不能只是「换个跨度」，可行的方向是："
      " (a) **多变量联合轨迹生成**——同时生成认知、共病、用药、死亡的联合序列，"
      "评估用轨迹层面的指标（校准、多步生成的分布保真度、生存曲线），而不是单一 AUC；"
      " (b) **条件模拟/反事实**——给定当前状态采样未来若干条轨迹，回答「这个人 5 年后处于各状态的概率分布」，"
      "这是逻辑回归给不出的；"
      " (c) 如果坚持做单点预测，就必须诚实报告：本项目的贡献是**架构可行性验证**，不是 AUC 提升。")}

### 9.5 transformer 必须超过的下限

| 任务 | 指标 | 基线值（B3：年龄+性别+教育+上次认知+变化+MMSE） |
|---|---|---|
| 下次访视 AD 诊断 | ROC-AUC / PR-AUC | **{b3['roc_auc']:.4f} / {b3['pr_auc']:.4f}**（正类 {b3['prevalence_pct']:.1f}%） |
| 5 年内 AD 诊断 | ROC-AUC / PR-AUC | **{h5['roc_auc']:.4f} / {h5['pr_auc']:.4f}**（正类 {h5['prevalence_pct']:.1f}%） |
| 下次 `cogn_global` | R² / MAE | **{reg['r2'].max():.4f} / {reg.iloc[0]['mae']:.4f}** |

这三行就是验收线。**没有超过它们的 transformer 不构成结果**；
若在 5 折 CV 上的置信区间与基线重叠，正确的结论是「数据量不支持」，这本身是有效结果（§3 决策门）。
考虑到 §3 的小样本（每折仅 ~885 人），报告时必须给**按人自助法的置信区间**，而不是单个点估计——0.93 vs 0.94 在这个样本量上很可能是噪声。
""")
    print("s9 done")


if __name__ == "__main__":
    run()
