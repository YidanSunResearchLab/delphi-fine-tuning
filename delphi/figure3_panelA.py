"""Figure 3 panel A：主模型逐 token 的 Delphi AUC，旁边标出两个 transformer 基线。

    python figure3_panelA.py --offset 365.25 --out results/fig3/figs/fig3A_gap365
    python figure3_panelA.py --offset 0.1    --out results/fig3/figs/fig3A_gap0     # 补充材料

只输出 PNG（和画图用的 CSV），不出 PDF。

输入是 evaluate_auc_rosmap_controls.py 的 fig3_auc_summary.csv（results/fig3/v6_panelA，
口径：val、首次出现、只评入组后、上游读分）。每个点是 3 个 seed 的均值；误差线是 DeLong
95% CI（3 个 seed 的均值）；短横线是基线的 3 seed 均值。同时写出画图用的逐 token 表
（<out>.csv），即图的表格视图。

配色：参考调色板前 3 格（blue / orange / aqua），--pairs all 验证通过；aqua 对浅色背景
低于 3:1，所以三个系列同时用形状区分（圆点 / 短横线 / 菱形），并有图例和表格视图。
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 默认 = 第一版（nodedup-pos 主模型）。snapshot 那一组用 --main snapshot-pos --demog snapshot-static
# --now snapshot-visit1 --ref snapshot --summary results/fig3/v8_panelA/fig3_auc_summary.csv
MAIN, DEMOG, NOW, REF_FULL = "nodedup-pos-3w", "static", "visit1", "fullvisit3w"
C_MAIN, C_DEMOG, C_NOW = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
GROUPS = [("event", "Events"), ("med_start", "Medication start"), ("state", "Clinical measures (first entry)")]
MIN_CASES = 20


def paired(s, a, b):
    """同 seed 配对：逐 token 差的中位；返回 3 对的均值、sd 和逐 token 平均差 > 0 的比例。"""
    A = s[s.config == a].pivot_table(index=["seed", "token"], values="auc")
    B = s[s.config == b].pivot_table(index=["seed", "token"], values="auc")
    d = (A - B).dropna().reset_index()
    per_seed = d.groupby("seed").auc.median()
    return per_seed.mean(), per_seed.std(), d.groupby("token").auc.mean().gt(0).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="results/fig3/v6_panelA/fig3_auc_summary.csv")
    ap.add_argument("--iter0", default="results/fig3/v3_main/fig3_auc_summary.csv")
    ap.add_argument("--offset", type=float, default=365.25)
    ap.add_argument("--out", required=True)
    ap.add_argument("--main", default=MAIN)
    ap.add_argument("--demog", default=DEMOG)
    ap.add_argument("--now", default=NOW)
    ap.add_argument("--ref", default=REF_FULL, help="与 --now 只差注意力范围的全历史参照（算'历史的价值'）")
    args = ap.parse_args()
    main_cfg, demog_cfg, now_cfg, ref_cfg = args.main, args.demog, args.now, args.ref

    s = pd.read_csv(args.summary)
    s = s[(s.score == "model") & np.isclose(s.offset, args.offset)]
    assert set(s.post_entry) == {1} and set(s.first_occurrence) == {1}, "口径必须是首次出现 + 只评入组后"

    per = s.groupby(["config", "group", "name"]).agg(auc=("auc", "mean"), ci95=("auc_ci95", "mean"),
                                                     n_case=("n_case", "mean"), n_ctrl=("n_ctrl", "mean"))
    t = per.loc[main_cfg].copy()
    t["demog"] = per.loc[demog_cfg, "auc"]
    t["now"] = per.loc[now_cfg, "auc"]
    t = t.reset_index()

    med = s[s.config == main_cfg].groupby("seed").auc.median()
    d_demog = paired(s, main_cfg, demog_cfg)
    d_now = paired(s, main_cfg, now_cfg)
    d_hist = paired(s, ref_cfg, now_cfg)
    it = pd.read_csv(args.iter0)
    it = it[(it.score == "iter0") & np.isclose(it.offset, args.offset) & (it.config == main_cfg)]
    iter0_med = it.groupby("seed").auc.median().mean() if len(it) else float("nan")

    # 排版：组内按主模型 AUC 降序，组间留一格
    order, xs, x = [], [], 0
    spans = []
    for g, _ in GROUPS:
        sub = t[t.group == g].sort_values("auc", ascending=False)
        start = x
        for _, r in sub.iterrows():
            order.append(r); xs.append(x); x += 1
        spans.append((start, x - 1))
        x += 1
    T = pd.DataFrame(order).reset_index(drop=True)
    T["x"] = xs

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    fig, ax = plt.subplots(figsize=(11.5, 4.3), dpi=200)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

    ax.axhline(0.5, color=MUTED, lw=1, ls=(0, (4, 3)), zorder=1)
    for y in (0.6, 0.7, 0.8, 0.9):
        ax.axhline(y, color=GRID, lw=0.6, zorder=0)

    small = T.n_case < MIN_CASES
    w = 0.34
    ax.hlines(T.demog, T.x - w, T.x + w, color=C_DEMOG, lw=2, zorder=3, label="Demographics only")
    ax.scatter(T.x, T.now, marker="D", s=16, color=C_NOW, edgecolor="#fcfcfb", linewidth=0.8, zorder=4,
               label="Current visit only")
    ax.vlines(T.x, T.auc - T.ci95, np.minimum(T.auc + T.ci95, 1.0), color=C_MAIN, lw=1, alpha=0.55, zorder=2)
    ax.scatter(T.x[~small], T.auc[~small], s=30, color=C_MAIN, edgecolor="#fcfcfb", linewidth=1, zorder=5,
               label="Full history")
    ax.scatter(T.x[small], T.auc[small], s=30, facecolor="#fcfcfb", edgecolor=C_MAIN, linewidth=1.2, zorder=5,
               label=f"Full history, < {MIN_CASES} cases")

    labels = [n + ("†" if n == "MMSE_normal" else "") for n in T.name]
    ax.set_xticks(T.x)
    ax.set_xticklabels(labels, rotation=90, fontsize=6.5)
    for lab, sm in zip(ax.get_xticklabels(), small):
        lab.set_color(MUTED if sm else INK2)
    ax.set_xlim(-1, T.x.max() + 1)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("AUC (Delphi protocol)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for (a, b), (_, gl) in zip(spans, GROUPS):
        ax.text((a + b) / 2, 1.012, gl, ha="center", va="bottom", fontsize=8, color=INK, fontweight="bold")
        if b < T.x.max():
            ax.axvline(b + 1, color=GRID, lw=1, zorder=0)

    lead = "≈2 years ahead (offset 365 d)" if args.offset > 300 else "≈1 year ahead (no gap)"
    fig.subplots_adjust(left=0.055, right=0.99, top=0.86, bottom=0.27)
    fig.text(0.055, 0.945, f"A   Per-token AUC, {lead}", fontsize=10, color=INK, fontweight="bold", va="bottom")
    # 图例单独一行，放在竖排的 token 名和脚注之间，不和任何标签重叠
    fig.legend(*ax.get_legend_handles_labels(), loc="lower left", bbox_to_anchor=(0.05, 0.01),
               fontsize=7, frameon=False, ncol=4, handlelength=1.6, columnspacing=2.2)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out + ".png", facecolor=fig.get_facecolor())
    T.drop(columns="x").round(4).to_csv(args.out + ".csv", index=False)
    print(f"wrote {args.out}.png/.csv | median {med.mean():.3f} | Δdemog {d_demog[0]:+.3f} | "
          f"Δnow {d_now[0]:+.3f} | history {d_hist[0]:+.3f} | iter0 {iter0_med:.3f}")


if __name__ == "__main__":
    main()
