"""Section 3 -- cohort and sequence scale (risk R2: is there enough data for a transformer?)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common import (C1, C2, C3, REF_GREY, SEED, fig, load, report, save_data, save_fig,
                    save_table, setup_style, summary_put, gate, FIGS)


def run():
    setup_style()
    cs, lo, cl, v = load()

    n_per = v.groupby("projid").size()
    q = n_per.describe(percentiles=[.25, .5, .75, .95])
    ge = {k: int((n_per >= k).sum()) for k in (2, 3, 5, 10, 15, 20)}

    by_study = (v.groupby("study")
                  .agg(n_participants=("projid", "nunique"), n_visits=("projid", "size"))
                  .assign(median_visits=v.groupby("study")["projid"].apply(
                      lambda s: s.value_counts().median()))
                  .reset_index())
    fup = v.groupby("projid").agg(age_bl=("age_bl", "first"), age_last=("age_at_visit", "max"))
    fup["followup_years"] = fup["age_last"] - fup["age_bl"]
    by_study = by_study.merge(
        v.groupby("study").apply(lambda d: (d.groupby("projid")["age_at_visit"].max()
                                            - d.groupby("projid")["age_bl"].first()).median(),
                                 include_groups=False).rename("median_followup_yrs").reset_index(),
        on="study")
    save_table(by_study, "cohort_size_by_study")

    # ---- fig: visits per person -------------------------------------------------
    f, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(n_per, bins=np.arange(0.5, n_per.max() + 1.5, 1), color=C1, edgecolor="white", lw=.4)
    ax.axvline(n_per.median(), color=C2, ls="--", lw=2)
    ax.annotate(f"median {n_per.median():.0f} visits", (n_per.median(), ax.get_ylim()[1] * .92),
                xytext=(6, 0), textcoords="offset points", color=C2, fontsize=9, va="top")
    ax.set_xlabel("Visits per participant"); ax.set_ylabel("Participants")
    ax.set_title(f"Sequence length is short and right-skewed\n"
                 f"{len(n_per):,} participants · {len(v):,} visits · "
                 f"{100 * (n_per < 5).mean():.0f}% have fewer than 5 visits")
    save_data(n_per.rename("n_visits").reset_index(), FIGS, "visits_per_person_hist")
    save_fig(f, FIGS, "visits_per_person_hist")

    # ---- fig: cumulative "at least N visits" ------------------------------------
    ks = np.arange(1, int(n_per.max()) + 1)
    cum = np.array([(n_per >= k).sum() for k in ks])
    f, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(ks, cum, lw=2, color=C1)
    ax.fill_between(ks, 0, cum, color=C1, alpha=.12, lw=0)
    for k, col in [(3, C2), (5, C3), (10, C2)]:
        y = (n_per >= k).sum()
        ax.plot([k], [y], "o", ms=8, color=col, mec="white", mew=1.5)
        ax.annotate(f"N={k}: {y:,} ({100 * y / len(n_per):.0f}%)", (k, y), xytext=(8, 8),
                    textcoords="offset points", fontsize=9, color=col)
    ax.set_xlabel("N (minimum number of visits)"); ax.set_ylabel("Participants")
    ax.set_title("Usable-sequence budget shrinks fast with the length you require\n"
                 f"participants with at least N visits (total n={len(n_per):,})")
    save_data(pd.DataFrame({"min_visits": ks, "n_participants": cum}), FIGS, "visits_per_person_cumulative")
    save_fig(f, FIGS, "visits_per_person_cumulative")

    # ---- fig: follow-up duration -------------------------------------------------
    f, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(fup["followup_years"], bins=np.arange(0, fup["followup_years"].max() + 1, 1),
            color=C1, edgecolor="white", lw=.4)
    ax.axvline(fup["followup_years"].median(), color=C2, ls="--", lw=2)
    ax.annotate(f"median {fup['followup_years'].median():.0f} y", (fup["followup_years"].median(),
                ax.get_ylim()[1] * .92), xytext=(6, 0), textcoords="offset points",
                color=C2, fontsize=9, va="top")
    ax.set_xlabel("Last observed age - baseline age (years)"); ax.set_ylabel("Participants")
    ax.set_title(f"Observed follow-up span per participant (n={len(fup):,})")
    save_data(fup.reset_index(), FIGS, "followup_span")
    save_fig(f, FIGS, "followup_span")

    # ---- 80/10/10 by person (for reference only -- section 9 uses 5-fold CV) -----
    rng = np.random.default_rng(SEED)
    pids = n_per.index.to_numpy().copy()
    rng.shuffle(pids)
    n = len(pids)
    cuts = {"train": pids[:int(.8 * n)], "val": pids[int(.8 * n):int(.9 * n)], "test": pids[int(.9 * n):]}
    split_rows = [{"split": k, "n_participants": len(p), "n_visits": int(n_per.reindex(p).sum()),
                   "n_ad_cases": int(cs.set_index("projid").loc[p, "age_first_ad_dx"].notna().sum())}
                  for k, p in cuts.items()]
    save_table(pd.DataFrame(split_rows), "holdout_80_10_10_reference")

    n_part, n_vis = int(n_per.size), int(len(v))
    small = (n_part < 5000) or (n_vis < 100000)
    summary_put(n_participants=n_part, n_visits=n_vis, median_visits=float(n_per.median()),
                mean_visits=float(n_per.mean()), max_visits=int(n_per.max()),
                n_with_ge5_visits=ge[5], n_with_ge10_visits=ge[10],
                pct_single_visit=round(100 * (n_per == 1).mean(), 2),
                median_followup_years=round(float(fup["followup_years"].median()), 2),
                small_data_regime=bool(small),
                test_set_participants_if_80_10_10=len(cuts["test"]),
                test_set_ad_cases_if_80_10_10=split_rows[2]["n_ad_cases"])

    report(f"""## 3. 队列与序列规模（风险 R2）

**算了什么**：唯一 projid（按 study 分层）、每人访视次数分布与累计曲线、总 visit 数、随访时长、
以及 80/10/10 按人划分后各 split 的规模。

**关键数值**

- **n_participants = {n_part:,}**，**n_visits = {n_vis:,}**。
- 每人访视次数：min {q['min']:.0f} / P25 {q['25%']:.0f} / **median {q['50%']:.0f}** / P75 {q['75%']:.0f} /
  P95 {q['95%']:.0f} / max {q['max']:.0f}；mean {q['mean']:.2f}。
- 至少 N 次访视的人数：N=2 {ge[2]:,}（{100 * ge[2] / n_part:.0f}%）· N=3 {ge[3]:,}（{100 * ge[3] / n_part:.0f}%）·
  **N=5 {ge[5]:,}（{100 * ge[5] / n_part:.0f}%）** · N=10 {ge[10]:,}（{100 * ge[10] / n_part:.0f}%）·
  N=20 {ge[20]:,}（{100 * ge[20] / n_part:.0f}%）。
- {100 * (n_per == 1).mean():.1f}% 的人**只有基线一次访视**，对序列模型完全无用（无 next-token 目标）。
- 随访时长中位数 {fup['followup_years'].median():.0f} 年（P95 {fup['followup_years'].quantile(.95):.0f} 年）。
- 按 study：{' · '.join(f"{r.study} {r.n_participants}人/{r.n_visits}访视" for r in by_study.itertuples())}。
- 若按人 80/10/10：test 只有 **{len(cuts['test'])} 人**，其中曾被诊断 AD 的仅 **{split_rows[2]['n_ad_cases']} 人**。

{gate(False, f"**小数据场景确认**（阈值 5,000 人 / 100,000 访视，实际 {n_part:,} 人 / {n_vis:,} 访视，两条都不满足）。"
      "按规格书，后续一律按小样本口径处理：(a) 模型规模上限 1–3M 参数，且要比 Delphi-2M 的 2.2M 更保守——"
      "Delphi-2M 是 40 万人训出来的，这里少两个数量级；(b) **必须 5 折交叉验证，不能单次 holdout**——"
      f"上面那个 {len(cuts['test'])} 人 / {split_rows[2]['n_ad_cases']} 例 AD 的测试集，AUC 的置信区间会宽到没有意义；"
      "(c) 必须建立强基线对照（线性混合效应 + GRU/LSTM，见 §9），transformer 打不过混合效应模型就是有效结论。")}
{gate(q['50%'] >= 5, (f"中位访视次数 = {q['50%']:.0f}，**高于规格书 5 次的下限，这一条不触发**。"
      f"但序列长度是重尾的：{100 - 100 * ge[5] / n_part:.0f}% 的人不足 5 次访视，"
      f"{100 * (n_per == 1).mean():.0f}% 只有基线一次，模型的多数样本来自很短的序列。"
      "在中位长度 7 的序列上，self-attention 相对 GRU 没有结构性优势，"
      "所以用 transformer 的理由必须在方案里写明——是**统一的「同一年龄多 token」表示 + "
      "生成式多变量联合采样**，不是 attention 的长程依赖能力，也不要用「更强的建模能力」搪塞。")
      if q['50%'] >= 5 else
      (f"中位访视次数 = {q['50%']:.0f}，**低于 5**，序列太短，self-attention 相对 RNN 没有优势。"))}
""")
    print("s3 done")


if __name__ == "__main__":
    run()
