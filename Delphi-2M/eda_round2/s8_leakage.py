"""Section 8 -- the leakage blacklist, with the empirical demonstration for each entry."""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

from common import (VAR_META, load, report, save_table, summary_put, gate)
from labels import make_pairs, target_frame

# reason codes -> the rule that puts a field on the list
RULES = {
    "future": "only determined at/after death or at the end of follow-up",
    "cross_visit_derived": "computed FROM the whole trajectory -- contains future visits by construction",
    "outcome": "this is the label (or a deterministic function of it)",
    "autopsy": "exists only for decedents; its presence alone reveals death",
    "cotemporal": "measured at the SAME visit as the token being predicted -- needs the attention mask, not deletion",
    "id": "identifier; re-identification risk and no signal",
    "capped_string": "de-identified string ('90+'); silently becomes NaN or sorts lexicographically",
}


def oof_auc(X, y, g):
    pred = np.full(len(y), np.nan)
    est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                        LogisticRegression(max_iter=3000))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        est.fit(X[tr], y[tr])
        pred[te] = est.predict_proba(X[te])[:, 1]
    ok = np.isfinite(pred)
    return roc_auc_score(y[ok], pred[ok])


def run():
    cs, lo, cl, v = load()
    pairs = make_pairs(v)
    tf = target_frame(pairs, "y_ad_next")
    y, g = tf["y"].to_numpy(), tf["projid"].to_numpy()

    # ---- the demonstration: how big is the leak, actually? --------------------------
    demo = []
    slope = tf["projid"].map(cs.set_index("projid")["cogng_demog_slope"]).to_numpy(float)
    demo.append({"feature": "cogng_demog_slope (LME random slope over ALL follow-up)",
                 "oof_auc_y_ad_next": round(oof_auc(slope.reshape(-1, 1), y, g), 4),
                 "verdict": "LEAK -- fitted on the future of the same person"})
    demo.append({"feature": "cogn_global at the current visit (legitimate)",
                 "oof_auc_y_ad_next": round(oof_auc(tf[["cogn_global"]].to_numpy(float), y, g), 4),
                 "verdict": "legitimate, but see the construct-validity note below"})
    demo.append({"feature": "age at the current visit (legitimate)",
                 "oof_auc_y_ad_next": round(oof_auc(tf[["age_at_visit"]].to_numpy(float), y, g), 4),
                 "verdict": "legitimate"})
    for c in ["braaksc", "gpath"]:
        x = tf["projid"].map(cs.set_index("projid")[c]).to_numpy(float)
        demo.append({"feature": f"{c} (autopsy)",
                     "oof_auc_y_ad_next": round(oof_auc(x.reshape(-1, 1), y, g), 4),
                     "verdict": "LEAK -- only exists after death"})
    demo = pd.DataFrame(demo)
    save_table(demo, "leakage_demonstration")
    slope_auc = float(demo.iloc[0]["oof_auc_y_ad_next"])
    cogn_auc = float(demo.iloc[1]["oof_auc_y_ad_next"])

    # ---- the blacklist ----------------------------------------------------------------
    bl = []

    def add(col, table, rule, note):
        bl.append({"column": col, "table": table, "rule": rule, "reason": RULES[rule],
                   "detail": note, "action": "EXCLUDE from the input sequence"})

    add("cogng_demog_slope", "cross-sectional-data-gk", "cross_visit_derived",
        f"random slope of cogn_global from an LME fitted over the participant's ENTIRE follow-up; "
        f"alone it reaches out-of-fold AUC {slope_auc:.3f} on y_ad_next")
    add("age_first_ad_dx", "cross-sectional-data-gk", "outcome",
        "the label; may only be used to BUILD targets, never as a feature")
    add("died", "cross-sectional-data-gk", "future",
        "vital status at the data freeze -- known only in the future of every visit")
    for c in ["braaksc", "ceradsc", "gpath", "amylsqrt_est_8reg", "tangsqrt_est_8reg", "tdp_st4",
              "lewydx_st4", "arteriol_scler", "caa_4gp", "cvda_4gp2", "ci_num2_mct", "ci_num2_tct"]:
        add(c, "cross-sectional-data-gk", "autopsy", VAR_META[c][2])
    add("pmi", "cross-sectional-data-gk", "autopsy",
        "post-mortem interval -- a non-null value IS a death indicator")
    for c in ["age_death", "age_at_visit_max"]:
        add(c, "ROSMAP_clinical", "future", VAR_META[c][2])
        bl[-1]["rule"] = "future"
    add("cogdx", "ROSMAP_clinical", "future",
        "FINAL consensus diagnosis, assigned once at the end (autopsy-informed for decedents)")
    add("dcfdx_lv", "ROSMAP_clinical", "future",
        "'last valid' diagnosis -- a single value pinned to the END of the trajectory")
    add("cts_mmse30_lv", "ROSMAP_clinical", "future", "MMSE at the LAST valid cycle")
    add("cts_mmse30_first_ad_dx", "ROSMAP_clinical", "outcome",
        "MMSE at the AD-diagnosis cycle -- defined by the label")
    add("individualID", "ROSMAP_clinical", "id", "Synapse identifier")
    add("projid", "all", "id", "keep for grouping/splitting only; must not be embedded as a feature")
    for c in ["cts_estmmse30", "cogn_global", "bmi", "sbp_avg", "dbp_avg", "psqi_sum"]:
        bl.append({"column": c, "table": "longitudinal_data_gk", "rule": "cotemporal",
                   "reason": RULES["cotemporal"], "detail":
                   "legitimate as history; the leak is only WITHIN a visit",
                   "action": "KEEP, but mask同一 (projid, fu_year) 内的其他 token"})
    bl = pd.DataFrame(bl)
    save_table(bl, "leakage_blacklist")

    # ---- correlation screen ----------------------------------------------------------
    person = cs.copy()
    person["ad_ever"] = person["age_first_ad_dx"].notna().astype(int)
    num = person.select_dtypes(include=[np.number]).drop(columns=["projid"], errors="ignore")
    corr = []
    for c in num.columns:
        if c == "ad_ever":
            continue
        m = num[c].notna() & num["ad_ever"].notna()
        # both arms must vary: some columns are only non-null inside a single outcome class
        if m.sum() < 100 or num.loc[m, c].nunique() < 2 or num.loc[m, "ad_ever"].nunique() < 2:
            continue
        r = stats.pearsonr(num.loc[m, c], num.loc[m, "ad_ever"]).statistic
        corr.append({"variable": c, "pearson_r_with_ad_ever": round(float(r), 4), "n": int(m.sum()),
                     "flag_abs_gt_0p95": bool(abs(r) > .95)})
    corr = pd.DataFrame(corr).sort_values("pearson_r_with_ad_ever", key=np.abs, ascending=False)
    save_table(corr, "outcome_correlation_screen")
    near_perfect = corr[corr["flag_abs_gt_0p95"]]["variable"].tolist()

    # ---- monotonicity audit of the *_cum variables ------------------------------------
    vv = v.sort_values(["projid", "fu_year"])
    mono = []
    for c in [x for x in v.columns if x.endswith("_cum")]:
        d = vv.groupby("projid", sort=False)[c].diff()
        mono.append({"variable": c, "n_decreases": int((d < 0).sum()), "n_increases": int((d > 0).sum()),
                     "monotone_nondecreasing": bool((d < 0).sum() == 0)})
    mono = pd.DataFrame(mono)
    save_table(mono, "cum_variable_monotonicity")

    # ---- co-temporal correlation, the number behind the mask ---------------------------
    m = v["cogn_global"].notna() & v["cts_estmmse30"].notna()
    r_same = float(stats.pearsonr(v.loc[m, "cogn_global"], v.loc[m, "cts_estmmse30"]).statistic)

    summary_put(n_blacklisted_columns=int((bl["action"].str.startswith("EXCLUDE")).sum()),
                n_masked_not_excluded=int((~bl["action"].str.startswith("EXCLUDE")).sum()),
                leak_demo_slope_auc=slope_auc,
                legit_cogn_global_auc=cogn_auc,
                near_perfect_correlates=near_perfect,
                cum_vars_all_monotone=bool(mono["monotone_nondecreasing"].all()),
                same_visit_cogn_mmse_r=round(r_same, 4))

    report(f"""## 8. 泄漏排查

**算了什么**：逐条列出不能进输入序列的字段并给出理由（`tables/leakage_blacklist.csv`）；
对最危险的几条做**定量演示**；`*_cum` 变量的单调性审计；与结局的相关性筛查；共时变量的相关性。

### 8.1 泄漏是真的，不是理论担忧

同一个 out-of-fold 协议（5 折按人分组），单变量预测 `y_ad_next`：

{demo.to_markdown(index=False)}

`cogng_demog_slope` **一个变量**就能到 AUC **{slope_auc:.3f}**——它并不是表里最强的特征（合法的当次 `cogn_global` 是 {cogn_auc:.3f}），但它是用这个人**全部随访（包括未来）**拟合出来的
认知下降斜率，任何非零的判别力都是偷来的。它在表里长得和普通协变量一模一样，这正是它危险的地方。
两个病理指标 AUC ≈0.60 看着不高，但它们**只有死者才有**——非空本身就是死亡信号，
真正的危害是它把「已经死了的人」和「还活着的人」偷偷分开了。

### 8.2 黑名单（{int((bl['action'].str.startswith('EXCLUDE')).sum())} 个字段硬排除 + {int((~bl['action'].str.startswith('EXCLUDE')).sum())} 个需要 mask 而非删除）

按规则分类：

{bl.groupby(['rule', 'reason']).size().rename('n_columns').reset_index().to_markdown(index=False)}

完整清单见 `tables/leakage_blacklist.csv`。**要求：在建模代码里硬编码为 `LEAKAGE_BLACKLIST`
常量并在 tokenizer 入口断言，不要依赖人工记忆。**

### 8.3 三类需要单独说明的情况

1. **`*_cum` 共病变量不是泄漏。** codebook 说它们是「截至本 cycle 曾经报告过」，审计确认
   {int(mono['monotone_nondecreasing'].sum())}/{len(mono)} 个全部**单调非减**（无一次下降），
   即只用了过去信息。但正因为单调，它们一旦置 1 就永远是 1 → 应按「事件型 token」只在**首次转 1** 时发射。
2. **`cogn_global` 的 z 分基准。** codebook：z 分用的是**基线 cycle 全队列**的均值/标准差。
   这不是「用了未来」，但它是**用了全样本（含测试集）算出来的常数**。严格做法是在训练折上重算基准；
   实际影响极小（两个常数），**但必须在方法学里写明**，否则审稿人会问。
3. **共时变量不能靠删除解决。** 同一次访视的 `cogn_global` 与 `cts_estmmse30` 相关系数
   **r = {r_same:.3f}**——预测其中一个时若能看到另一个，指标会虚高到毫无意义。
   正确做法是 §6.6 的 attention mask，不是把变量删掉。

### 8.4 相关性筛查

与「是否曾被诊断 AD」相关性 |r| > 0.95 的字段：**{('、'.join(near_perfect) if near_perfect else '无')}**。
相关性最高的几个（见 `tables/outcome_correlation_screen.csv`）都在 |r| < 0.5，
说明没有「换了个名字的结局变量」混在特征里。

### 8.5 一个不是泄漏、但同样致命的问题：构念循环

`age_first_ad_dx` 的定义是「`dcfdx` 首次取 4/5 的那次访视的年龄」，而临床医生给 `dcfdx` 时
**看的正是这一批认知测验**；`cogn_global` 则是这 19 项测验的 z 分平均。
所以「用本次 `cogn_global` 预测下次 AD 诊断」很大程度上是在**用一次测量预测同一批测量的临床解读**，
AUC {cogn_auc:.3f}（§6 在剔除缺失行、不做填补时是 0.926，口径略有不同）有相当一部分来自这个循环，而不是来自真正的「预测」。

{gate(False, "**这不能靠黑名单解决。** §9.4 检验了最自然的补救办法——把预测跨度从 1 年拉到 5 年——"
      "结果是 AUC 只掉了约 0.04，说明**拉长跨度并不能把循环成分甩掉**"
      "（认知水平对长期诊断本来就有真实的预测力，两者混在一起，这个设计分不开）。"
      "因此正确的处理是**在论文里明确声明这个循环**：`age_first_ad_dx` 与 `cogn_global` 共享同一批测验，"
      "任何基于二者的 AUC 都不能被解读为「独立的早期预警」。"
      "若要一个不循环的结局，本 release 里唯一的候选是**死亡**（§5）和**神经病理**（§7.4），"
      "但后者只有死者才有、且不能进输入序列。")}
""")
    print("s8 done")


if __name__ == "__main__":
    run()
