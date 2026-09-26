"""Figure 3 panel B：完全不去重（--snapshot）相对完全去重（dedup）的 ΔAUC，两个预测提前量并排。

    python figure3_panelB.py --out results/fig3/figs/fig3B

输入：results/fig3/v7_panelB/fig3_auc_summary.csv（evaluate_auc_rosmap_controls.py，口径同 panel A：
val、首次出现、只评入组后、上游读分）。只输出 PNG 和画图用的 CSV。

每个格子 = 一个 token 组 × 一个提前量。Δ 是**同 seed 配对**：同一 seed 的两个模型在同一批
token 上逐 token 相减，取组内中位；小点是 3 个 seed 各自的 Δ，粗点和误差线是均值 ± SD。
灰带 = 同一配置重训一次 Δ 的预期波动：dedup 自己 3 个 seed 组中位的 SD × √2。

两个提前量必须并排：gap0 < gap365 只出现在重复发射的分词上（nodedup / fullvisit），dedup
没有，原因未查明 —— 而这正是本 panel 比较的维度。只画一个提前量会把这个未解释的差异当成结论。

配色：blue = nodedup（与 panel A 主模型同一种分词）、violet = fullvisit；--pairs all 验证通过，
两者都 >= 3:1。系列同时用形状区分（圆 / 方）。
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REF = "dedup"
# 按"去重程度"从高到低排：都相对完全去重（dedup，交付版两级去重）。
#   nodedup     只放开 MMSE/COGN 的两级去重
#   nodedup-pos 同上 + 位置嵌入，三分划分（split 8:1:1）—— panel A 的主模型
#   fullvisit   15 个连续量全部每次都发 + 关全局去重（"测量不去重"）
#   snapshot    tokenization/build.py --snapshot：每次访视发射完整状态（完全不去重）
# 配色：adjacent 验证通过（magenta / blue / violet / red）；magenta 对浅色背景 < 3:1，所以同时用
# 形状区分、底行有直接标签。避开 panel A 里基线用的 orange / aqua。
MODELS = [("nodedup", "No dedup: MMSE, COGN", "MMSE/COGN", "#e87ba4", "o"),
          ("nodedup-pos-3w", "No dedup: MMSE, COGN + pos. emb. (panel A model)", "+ pos. emb.", "#2a78d6", "D"),
          ("fullvisit", "No dedup: all measurements", "all measures", "#4a3aa7", "s"),
          ("snapshot", "No dedup at all (full state every visit)", "everything", "#e34948", "^")]
GROUPS = [("event", "Events"), ("med_start", "Medication start"), ("state", "Clinical measures"), ("all", "All tokens")]
OFFSETS = [(365.25, "≈2 years ahead"), (0.1, "≈1 year ahead")]
INK, INK2, MUTED, GRID, BAND, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#ecebe7", "#fcfcfb"


def group_medians(s, cfg):
    x = s[s.config == cfg]
    g = x.groupby(["seed", "group"]).auc.median().unstack()
    g["all"] = x.groupby("seed").auc.median()
    return g


def paired_delta(s, a, b):
    A = s[s.config == a].pivot_table(index=["seed", "token", "group"], values="auc")
    B = s[s.config == b].pivot_table(index=["seed", "token", "group"], values="auc")
    d = (A - B).dropna().reset_index()
    per = d.groupby(["seed", "group"]).auc.median().unstack()
    per["all"] = d.groupby("seed").auc.median()
    return per                                                   # 行 = seed，列 = 组


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", nargs="+", default=["results/fig3/v6_panelA/fig3_auc_summary.csv",
                                                     "results/fig3/v6_panelB/fig3_auc_summary.csv",
                                                     "results/fig3/v7_panelB/fig3_auc_summary.csv"],
                    help="可给多份（不同评估任务的结果），同一 val 人群、同一口径才能拼")
    ap.add_argument("--title", default="B   Tokenization: AUC change relative to full deduplication")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    s0 = pd.concat([pd.read_csv(p) for p in args.summary], ignore_index=True)
    # 同一个 run 可能在两份结果里都评过（dedup 在 v6 和 v7 都有），口径相同，去掉重复行
    s0 = s0.drop_duplicates(subset=["config", "seed", "score", "offset", "token"])
    s0 = s0[s0.score == "model"]
    assert set(s0.post_entry) == {1} and set(s0.first_occurrence) == {1}, "口径必须是首次出现 + 只评入组后"

    rows = []
    cells = {}
    for off, _ in OFFSETS:
        s = s0[np.isclose(s0.offset, off)]
        ref_sd = group_medians(s, REF).std()
        for cfg, *_ in MODELS:
            d = paired_delta(s, cfg, REF)
            cells[(off, cfg)] = d
            for g, _ in GROUPS:
                rows.append({"offset_days": off, "model": cfg, "group": g,
                             **{f"delta_seed{sd}": d.loc[sd, g] for sd in d.index},
                             "delta_mean": d[g].mean(), "delta_sd": d[g].std(),
                             "noise_band": ref_sd[g] * np.sqrt(2)})
    T = pd.DataFrame(rows)
    lim = np.ceil((T[[c for c in T.columns if c.startswith("delta_seed")]].abs().max().max() + 0.02) * 20) / 20

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, axes = plt.subplots(len(OFFSETS), len(GROUPS), figsize=(11.5, 5.2), dpi=200, sharey=True)
    fig.patch.set_facecolor(SURF)
    for i, (off, olab) in enumerate(OFFSETS):
        for j, (g, glab) in enumerate(GROUPS):
            ax = axes[i, j]
            ax.set_facecolor(SURF)
            band = T[(T.offset_days == off) & (T.group == g)].noise_band.iloc[0]
            ax.axhspan(-band, band, color=BAND, zorder=0, lw=0)
            ax.axhline(0, color=MUTED, lw=1, zorder=1)
            for y in np.arange(-lim, lim + 1e-9, 0.05):
                if abs(y) > 1e-9:
                    ax.axhline(y, color=GRID, lw=0.5, zorder=0)
            for k, (cfg, lab, short, col, mk) in enumerate(MODELS):
                d = cells[(off, cfg)][g]
                jitter = np.linspace(-0.16, 0.0, len(d))
                ax.scatter(k + jitter, d.values, s=10, color=col, alpha=0.45, marker=mk, lw=0, zorder=3)
                ax.errorbar(k + 0.14, d.mean(), yerr=d.std(), fmt=mk, ms=5.5, color=col, mec=SURF, mew=0.8,
                            elinewidth=1.3, capsize=0, zorder=4)
                ax.text(k + 0.14, d.mean() + d.std() + 0.01, f"{d.mean():+.3f}", ha="center", va="bottom",
                        fontsize=5.5, color=INK2)
            ax.set_xlim(-0.5, len(MODELS) - 0.5)
            ax.set_ylim(-lim, lim)
            if i == len(OFFSETS) - 1:
                ax.set_xticks(range(len(MODELS)))
                ax.set_xticklabels([m[2] for m in MODELS], fontsize=6.3, rotation=30, ha="right")
            else:
                ax.set_xticks([])
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if i == 0:
                ax.set_title(glab, fontsize=8.5, color=INK, fontweight="bold")
            if j == 0:
                ax.set_ylabel(f"{olab}\nΔAUC vs full dedup")

    fig.text(0.055, 0.955, args.title,
             fontsize=10, color=INK, fontweight="bold", va="bottom")
    handles = [plt.Line2D([], [], marker=mk, ls="", color=col, ms=6.5, label=lab) for _, lab, _, col, mk in MODELS]
    handles.append(plt.Rectangle((0, 0), 1, 1, color=BAND, label="Seed-noise band (same config retrained)"))
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.05, 0.0), ncol=3, frameon=False, fontsize=7)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.88, bottom=0.21, wspace=0.08, hspace=0.12)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".png", facecolor=fig.get_facecolor())
    T.round(4).to_csv(args.out + ".csv", index=False)
    print(f"wrote {args.out}.png/.csv")
    print(T[["offset_days", "model", "group", "delta_mean", "delta_sd", "noise_band"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
