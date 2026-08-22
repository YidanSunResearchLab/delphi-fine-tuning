"""
analyze.py -- turn results/summary.csv + curves.csv into the answer.

    python experiments/capacity/analyze.py

Produces, in results/:
  capacity_table.md     the headline table, mean +/- sd over seeds
  capacity_curves.png   val-loss curves, one panel per shape, seeds overlaid
  capacity_knee.png     best-val and best-iter vs parameter count -- the actual question
  findings.md           a written read of what the numbers say

THE QUESTION: the delivered 2.1M model bottoms out at iter 2750 of 5000 and rises after.
Does lower capacity move that bottom later, and does it reach a lower val loss?

HOW TO READ IT -- two things are being measured and they are not the same:
  best_val         how good the model gets. Lower is better. THIS is the model-selection
                   criterion, because train.py early-stops on it.
  best_iter        WHEN it peaks. Later means the schedule is better matched to capacity;
                   a bottom at 55% of the schedule means 45% of the compute was spent
                   getting worse.
A shape can move the bottom later without improving best_val (it just overfits more
slowly to the same place). Only a lower best_val is an actual win.

SEED NOISE: with 3 seeds the sd on best_iter is wide. The table reports it so a 200-iter
shift is not read as a result when the seed spread is 400. Do not quote a difference
smaller than the spread.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

# The delivered checkpoint, for context ONLY. NOT an arm of this sweep: it ran with
# eval_interval=250, and estimate_loss() consumes the global RNG, so its batch stream
# differs. The sweep's own L12E120H12 arm is the reference.
DELIVERED = dict(tag="delivered ckpt", n_params=2_104_320, best_iter=2750, best_val=10.1035)


def order_tags(df):
    return (df.groupby("tag")["n_params"].first().sort_values(ascending=False).index.tolist())


def main():
    sp = os.path.join(RES, "summary.csv")
    cp = os.path.join(RES, "curves.csv")
    if not os.path.exists(sp):
        sys.exit(f"[analyze] no {sp} -- run collect.py first")
    smy = pd.read_csv(sp)
    cur = pd.read_csv(cp)
    if smy.empty:
        sys.exit("[analyze] summary.csv is empty")

    tags = order_tags(smy)
    g = smy.groupby("tag")
    agg = pd.DataFrame({
        "n_params": g["n_params"].first(),
        "shape": g.apply(lambda d: f"{int(d['n_layer'].iloc[0])}L/{int(d['n_head'].iloc[0])}H/"
                                   f"{int(d['n_embd'].iloc[0])}d", include_groups=False),
        "head_dim": g["head_dim"].first(),
        "n_seeds": g["seed"].nunique(),
        "best_val_mean": g["best_val"].mean(), "best_val_sd": g["best_val"].std(),
        "best_iter_mean": g["best_iter"].mean(), "best_iter_sd": g["best_iter"].std(),
        "gap_mean": g["gap_at_best"].mean(),
        "rise_mean": g["rise_after_best"].mean(),
        "ms_per_iter": g["ms_per_iter"].median(),
        "wall_min": g["wall_min"].median(),
    }).reindex(tags)

    # ---------------------------------------------------------------- table
    L = ["# Capacity sweep -- results", "",
         f"{len(smy)} runs, {agg['n_seeds'].max()} seeds x {len(tags)} shapes. "
         "Dataset `nacc-dedup-s42`, cohort filter 4/2, ignore_tokens 0..21, 5000 iters, "
         "eval every 100. Everything except (n_layer, n_head, n_embd) and the seed is held "
         "constant -- `verify.py` check [2] enforces it.", "",
         "| shape | params | head_dim | best val loss | best iter | train-val gap | rise after best | ms/iter |",
         "|---|---:|---:|---|---|---:|---:|---:|"]
    for t in tags:
        r = agg.loc[t]
        L.append(f"| **{t}** ({r['shape']}) | {int(r['n_params']):,} | {int(r['head_dim'])} | "
                 f"{r['best_val_mean']:.4f} ± {r['best_val_sd']:.4f} | "
                 f"{r['best_iter_mean']:.0f} ± {r['best_iter_sd']:.0f} | "
                 f"{r['gap_mean']:.4f} | {r['rise_mean']:+.4f} | {r['ms_per_iter']:.0f} |")
    L += ["", f"For context, the delivered checkpoint: {DELIVERED['n_params']:,} params, "
              f"best at iter {DELIVERED['best_iter']}, val {DELIVERED['best_val']:.4f}. "
              "**Not comparable arm-to-arm** -- it used eval_interval=250, and `estimate_loss()` "
              "draws from the global RNG, so its batch stream differs. The `L12E120H12` arm "
              "above is the like-for-like reference.", ""]
    # ---------------------------------------------------------------- decomposition
    # `val loss` is loss_ce + loss_dt and the two are on completely different scales, so the
    # sum alone cannot say WHICH head a capacity change helped. decompose.py measures them
    # separately on each arm's own best checkpoint; fold it in when it exists.
    dpath = os.path.join(RES, "decomposition.csv")
    if os.path.exists(dpath):
        dec = pd.read_csv(dpath)
        dg = dec.groupby("tag")
        dagg = pd.DataFrame({"n_params": dg["n_params"].first(),
                             "ce": dg["loss_ce"].mean(), "ce_sd": dg["loss_ce"].std(),
                             "dt": dg["loss_dt"].mean(), "dt_sd": dg["loss_dt"].std()})
        dagg = dagg.reindex([t for t in tags if t in dagg.index])
        ref_t = "L12E120H12" if "L12E120H12" in dagg.index else dagg.index[0]
        L += ["## Which head actually improved", "",
              "`val loss = loss_ce + loss_dt`. The two terms answer different questions and sit "
              "on different scales, so the sum on its own cannot say what a capacity change did. "
              "Measured on each arm's own best checkpoint (`decompose.py`):", "",
              "| shape | params | loss_ce (which event) | loss_dt (when) | Δce vs ref | Δdt vs ref |",
              "|---|---:|---|---|---:|---:|"]
        for t in dagg.index:
            r = dagg.loc[t]
            dce = r["ce"] - dagg.loc[ref_t, "ce"]
            ddt = r["dt"] - dagg.loc[ref_t, "dt"]
            L.append(f"| **{t}** | {int(r['n_params']):,} | {r['ce']:.4f} ± {r['ce_sd']:.4f} | "
                     f"{r['dt']:.4f} ± {r['dt_sd']:.4f} | {dce:+.4f} ({100*dce/dagg.loc[ref_t,'ce']:+.1f}%) | "
                     f"{ddt:+.4f} ({100*ddt/dagg.loc[ref_t,'dt']:+.1f}%) |")
        share = 100 * dagg["dt"].mean() / (dagg["dt"].mean() + dagg["ce"].mean())
        L += ["", f"`loss_dt` is {share:.0f}% of the total. Spread across shapes: "
                  f"ce {dagg['ce'].max()-dagg['ce'].min():.4f}, "
                  f"dt {dagg['dt'].max()-dagg['dt'].min():.4f}.", ""]

    with open(os.path.join(RES, "capacity_table.md"), "w") as f:
        f.write("\n".join(L))
    print("\n".join(L))

    # ---------------------------------------------------------------- curves
    n = len(tags)
    ncol = min(3, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.7 * nrow), squeeze=False)
    ymin = cur["val_loss"].replace([np.inf, -np.inf], np.nan).dropna().min()
    ymax = np.nanpercentile(cur["val_loss"].replace([np.inf, -np.inf], np.nan).dropna(), 97)
    for i, t in enumerate(tags):
        ax = axes[i // ncol][i % ncol]
        sub = cur[cur["tag"] == t]
        for s, d in sub.groupby("seed"):
            ax.plot(d["iter"], d["val_loss"], lw=1.3, alpha=0.85, label=f"seed {s}")
            b = smy[(smy["tag"] == t) & (smy["seed"] == s)]
            if not b.empty:
                ax.plot(b["best_iter"], b["best_val"], "v", ms=7, color="black", zorder=5)
        r = agg.loc[t]
        ax.set_title(f"{t}  ·  {int(r['n_params']):,} params\nbest {r['best_val_mean']:.3f} "
                     f"@ iter {r['best_iter_mean']:.0f}", fontsize=10)
        ax.set_xlabel("iteration"); ax.set_ylabel("val loss")
        ax.set_ylim(ymin - 0.02 * (ymax - ymin), ymax)
        ax.grid(alpha=0.25); ax.legend(fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].set_axis_off()
    fig.suptitle("Capacity sweep — validation loss (▼ = best checkpoint, where train.py early-stops)",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(os.path.join(RES, "capacity_curves.png"), dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------- the knee
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    x = agg["n_params"].to_numpy(float)
    for ax, col, sdcol, lab in [(a1, "best_val_mean", "best_val_sd", "best validation loss"),
                                (a2, "best_iter_mean", "best_iter_sd", "iteration of the best checkpoint")]:
        ax.errorbar(x, agg[col], yerr=agg[sdcol].fillna(0), marker="o", ms=7, lw=1.8,
                    capsize=4, color="#2c7fb8")
        for t in tags:
            ax.annotate(t, (agg.loc[t, "n_params"], agg.loc[t, col]), fontsize=7.5,
                        xytext=(4, 5), textcoords="offset points")
        ax.set_xscale("log"); ax.set_xlabel("parameters (log scale)"); ax.set_ylabel(lab)
        ax.grid(alpha=0.25)
    a1.set_title("Does less capacity reach a LOWER val loss?", fontsize=11)
    a2.set_title("Does the bottom move LATER?", fontsize=11)
    a2.axhline(5000, ls=":", color="0.4", lw=1.2)
    a2.text(x.min(), 5000, " end of schedule", fontsize=7.5, va="bottom", color="0.4")
    a2.plot([DELIVERED["n_params"]], [DELIVERED["best_iter"]], "*", ms=14, color="#D55E00",
            zorder=5, label="delivered ckpt (eval_interval 250 — not comparable)")
    a2.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "capacity_knee.png"), dpi=160)
    plt.close(fig)

    # ---------------------------------------------------------------- findings
    best = agg["best_val_mean"].idxmin()
    ref = "L12E120H12"
    F = ["# What the sweep says", ""]
    if ref in agg.index:
        d = agg.loc[best, "best_val_mean"] - agg.loc[ref, "best_val_mean"]
        pooled = float(np.sqrt(np.nanmean([agg.loc[best, "best_val_sd"] ** 2,
                                           agg.loc[ref, "best_val_sd"] ** 2])))
        F += [f"- Lowest mean val loss: **{best}** ({int(agg.loc[best,'n_params']):,} params) at "
              f"{agg.loc[best,'best_val_mean']:.4f}.",
              f"- Reference arm `{ref}` (the delivered architecture): "
              f"{agg.loc[ref,'best_val_mean']:.4f}.",
              f"- Difference: **{d:+.4f}**, pooled seed sd **{pooled:.4f}** "
              f"-> {'LARGER than seed noise' if abs(d) > pooled else 'WITHIN seed noise; not a result'}.",
              ""]
        F += [f"- Bottom location: `{ref}` peaks at iter {agg.loc[ref,'best_iter_mean']:.0f} "
              f"({100*agg.loc[ref,'best_iter_mean']/5000:.0f}% of the schedule); "
              f"`{best}` at {agg.loc[best,'best_iter_mean']:.0f} "
              f"({100*agg.loc[best,'best_iter_mean']/5000:.0f}%).", ""]
        h12, h6 = "L12E120H12", "L12E120H6"
        if h12 in agg.index and h6 in agg.index:
            dh = agg.loc[h6, "best_val_mean"] - agg.loc[h12, "best_val_mean"]
            ph = float(np.sqrt(np.nanmean([agg.loc[h6, "best_val_sd"] ** 2,
                                           agg.loc[h12, "best_val_sd"] ** 2])))
            F += ["## Head geometry, at identical parameter count", "",
                  f"- `{h6}` (head_dim 20) vs `{h12}` (head_dim 10): **{dh:+.4f}** "
                  f"(pooled sd {ph:.4f}) -> "
                  f"{'a real effect' if abs(dh) > ph else 'within noise'}.",
                  "- Both are 2,104,320 params, so this isolates head splitting from capacity.", ""]
    F += ["## Caveat that belongs with every number here", "",
          "Validation loss is `loss_ce + loss_dt` on the same cohort filter, NOT a clinical "
          "endpoint. A shape that wins here has not yet been shown to win on the Figure-2 "
          "metrics (transition AUC, timing MAE, trajectory Jaccard vs the carry-forward "
          "baseline). Run `slurm/eval_sweep.sbatch` on the winning arms before concluding "
          "anything about the model's usefulness.", ""]
    with open(os.path.join(RES, "findings.md"), "w") as f:
        f.write("\n".join(F))
    print("\n" + "\n".join(F))
    print(f"\n[analyze] -> {RES}/capacity_table.md, findings.md, capacity_curves.png, capacity_knee.png")


if __name__ == "__main__":
    main()
