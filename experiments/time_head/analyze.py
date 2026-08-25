"""
analyze.py -- the PAIRED comparison: each time-head run against its own control.

    python experiments/time_head/analyze.py                       # after collect + decompose
    python experiments/time_head/analyze.py --control-runs <dir>   # if the capacity runs moved

Writes results/paired.csv and results/PAIRED.md.

WHY PAIRED, AND WHY THAT MATTERS HERE
-------------------------------------
The effect being measured is small by construction (121 parameters) and the quantity being
measured is the one the capacity sweep found immovable: a 6.9x parameter range moved loss_dt
by 0.0071 nats. An unpaired comparison at 3 seeds cannot see an effect that size -- the
capacity sweep's own seed sd on total val loss is ~0.005.

The pairs are exact, not approximate. `TH_X_s42` and `X_s42` start from bit-identical trunk
weights (verify.py check [3]) and consume the same batch stream (train.py re-seeds the batch
RNG before the loop), so the per-seed DIFFERENCE removes initialisation and batch-order
variance instead of averaging over it. The statistic to read is therefore mean(delta) against
sd(delta) across seeds -- reported below as `d` and `sd(d)` -- not each arm's own sd.

3 seeds is still 3 seeds. sd(d) over 3 pairs is a crude estimate and this prints no p-value;
it prints the paired deltas so the sign consistency is visible (3/3 in one direction on an
exact pairing is worth more than the sd suggests).

WHAT TO LOOK AT, IN ORDER
-------------------------
1. `loss_dt`  -- the term the head exists to move. Reference scale: 0.0071 nats is what
   6.9x of trunk capacity bought. Anything at or above that is a result.
2. downstream TIMING -- `a/transition_time/mae` (years, over ~2.4k transitions) and
   `b/timing_error/*/mae` against their `naive_mae` baseline. This is the number a clinician
   would feel, and the capacity sweep never moved it either. NOTE the delivered model is
   worse than naive on near-term buckets, so "beats naive" is a lower bar than it sounds.
3. guards -- `median_auc`, `mean_jaccard`. loss_ce is analytically untouched by the head at
   fixed weights (verify.py check [4]), but these runs TRAIN differently, so the categorical
   side can still drift. If it degrades, the head is not free.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.join(os.path.dirname(HERE), "capacity")
sys.path.insert(0, HERE)
sys.path.insert(0, CAP)

from arms import ARMS, SEEDS, control, overrides         # noqa: E402
from collect import parse_log                             # noqa: E402  (capacity's log parser)

# The capacity sweep's measured cross-shape spread in loss_dt, over a 6.9x parameter range.
# The yardstick every delta here is quoted against. Source: RESULTS.md finding 2.
CAP_DT_SPREAD = 0.0071

# Arms that change dt_target change WHAT loss_dt is computed against, so its value -- and the
# total val loss that contains it -- is NOT comparable to the control's. The size of the
# artefact is known: loss_dt is minimised at 1 + log E[dt], and dt_ablation.py measured the
# target's mean dropping from 2.648 y to 1.488 y, so the ACHIEVABLE loss_dt falls by
# log(2.648/1.488) = 0.58 nats on its own, with no change in model quality. Any val-loss delta
# smaller than that says nothing. For these arms the answer has to come from the downstream
# timing metrics, which are computed from trajectories and are objective-independent.
OBJECTIVE_CHANGING = {"dt_target"}
DT_TARGET_ARTEFACT = 0.58


def best_from_log(path):
    """(best_iter, best_val, final_val) from a train.log, or None."""
    if not os.path.exists(path):
        return None
    p = parse_log(path)
    df = pd.DataFrame(p["steps"], columns=["iter", "train", "val"])
    fin = df[df["val"].notna() & (df["val"] < 1e8)]
    if fin.empty:
        return None
    b = fin.loc[fin["val"].idxmin()]
    return dict(best_iter=int(b["iter"]), best_val=float(b["val"]),
                train_at_best=float(b["train"]), final_val=float(fin.iloc[-1]["val"]),
                n_evals=len(fin), nan_guard=p["nan_guard"])


def metrics(path):
    """metrics_matched.json -> the flat subset this analysis reads."""
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    out = {}
    a, b, c = d.get("a", {}), d.get("b", {}), d.get("c", {})
    tt = a.get("transition_time", {})
    out["timing_mae_y"] = tt.get("mae")
    out["timing_bias_y"] = tt.get("bias")
    out["timing_spearman"] = tt.get("spearman")
    out["timing_r2"] = tt.get("r2")
    out["median_auc"] = a.get("median_auc")
    out["calib_err"] = a.get("mean_calibration_error")
    for outcome, buckets in (b.get("timing_error") or {}).items():
        for bucket, v in buckets.items():
            key = f"{outcome} {bucket}"
            out[f"mae[{key}]"] = v.get("mae")
            out[f"naive[{key}]"] = v.get("naive_mae")
    for outcome, hz in (b.get("auc_by_horizon") or {}).items():
        for h, v in hz.items():
            out[f"auc[{outcome} @{h}y]"] = v
    out["mean_jaccard"] = c.get("mean_jaccard")
    out["jaccard_carry_baseline"] = c.get("mean_jaccard_baseline_carry")
    return out


def decomp_map(path):
    """run -> (loss_ce, loss_dt) from a decompose.py CSV."""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path)
    return {r["run"]: (float(r["loss_ce"]), float(r["loss_dt"])) for _, r in df.iterrows()}


def paired_table(rows, col):
    """mean and sd of the per-seed delta for one column, plus the sign tally."""
    d = [r[f"d_{col}"] for r in rows if r.get(f"d_{col}") is not None and np.isfinite(r[f"d_{col}"])]
    if not d:
        return None
    d = np.asarray(d, float)
    return dict(n=len(d), mean=float(d.mean()),
                sd=float(d.std(ddof=1)) if len(d) > 1 else float("nan"),
                neg=int((d < 0).sum()), pos=int((d > 0).sum()))


def fmt(x, nd=4):
    return "--" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(HERE, "runs"))
    ap.add_argument("--control-runs", default=os.path.join(CAP, "runs"))
    ap.add_argument("--eval", default=os.path.join(HERE, "results", "eval"))
    ap.add_argument("--control-eval", default=os.path.join(CAP, "results", "eval"))
    ap.add_argument("--decomp", default=os.path.join(HERE, "results", "decomposition.csv"))
    ap.add_argument("--control-decomp",
                    default=os.path.join(CAP, "results", "decomposition.csv"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dec_new, dec_ctl = decomp_map(a.decomp), decomp_map(a.control_decomp)

    rows, missing = [], []
    for arm in ARMS:
        ctl = control(arm)
        for s in SEEDS:
            run, cru = f"{arm}_s{s}", f"{ctl}_s{s}"
            t = best_from_log(os.path.join(a.runs, run, "train.log"))
            c = best_from_log(os.path.join(a.control_runs, cru, "train.log"))
            if t is None or c is None:
                missing.append(f"{run} vs {cru}: "
                               f"{'arm log' if t is None else 'control log'} unusable")
                continue
            r = dict(arm=arm, control=ctl, seed=s, run=run, control_run=cru,
                     best_val=t["best_val"], c_best_val=c["best_val"],
                     d_best_val=t["best_val"] - c["best_val"],
                     best_iter=t["best_iter"], c_best_iter=c["best_iter"],
                     d_best_iter=t["best_iter"] - c["best_iter"],
                     nan_guard=t["nan_guard"])
            if run in dec_new and cru in dec_ctl:
                (ce, dt), (cce, cdt) = dec_new[run], dec_ctl[cru]
                r.update(loss_ce=ce, loss_dt=dt, c_loss_ce=cce, c_loss_dt=cdt,
                         d_loss_ce=ce - cce, d_loss_dt=dt - cdt)
            m = metrics(os.path.join(a.eval, run, "metrics_matched.json"))
            cm = metrics(os.path.join(a.control_eval, cru, "metrics_matched.json"))
            if m and cm:
                for k, v in m.items():
                    if v is None or cm.get(k) is None:
                        continue
                    r[k] = v
                    r[f"c_{k}"] = cm[k]
                    r[f"d_{k}"] = v - cm[k]
            rows.append(r)

    if not rows:
        print("[analyze] no usable pairs yet.")
        for m in missing:
            print("  -", m)
        print(f"\nexpected arm logs under {a.runs}/<arm>_s<seed>/train.log")
        print(f"expected control logs under {a.control_runs}/<control>_s<seed>/train.log")
        return 1

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(a.out, "paired.csv"), index=False)

    # ------------------------------------------------------------------ the write-up
    # PER ARM, never pooled. The first version averaged every arm into one table, which mixed
    # two different experiments (a new head vs a fixed objective) into a single meaningless
    # delta. Arms answer different questions; they get different tables.
    L = ["# Paired against the capacity sweep's own runs", "",
         f"{len(df)} pairs across {df['arm'].nunique()} arm(s). Every row is one arm run minus "
         "its control at the SAME seed; the two members of a pair start from bit-identical "
         "trunk weights and see the same batch stream, so `d` is a within-pair difference, not "
         "a difference of means. Arms are reported separately.", ""]

    def section(sub, sdf, title, cols, nd=4, note=None, lower_is_better=True):
        block = [f"### {title}", ""]
        if note:
            block += [note, ""]
        block += ["| metric | arm mean | control mean | d (mean) | sd(d) | pairs better |",
                  "|---|---:|---:|---:|---:|---|"]
        any_row = False
        for col, label in cols:
            if f"d_{col}" not in sdf.columns:
                continue
            st = paired_table(sub, col)
            if st is None:
                continue
            any_row = True
            better = st["neg"] if lower_is_better else st["pos"]
            block.append(f"| {label} | {fmt(sdf[col].mean(), nd)} | {fmt(sdf[f'c_{col}'].mean(), nd)} "
                         f"| **{st['mean']:+.{nd}f}** | {fmt(st['sd'], nd)} | {better}/{st['n']} |")
        return block + [""] if any_row else []

    for arm in sorted(df["arm"].unique()):
        sub = [r for r in rows if r["arm"] == arm]
        sdf = df[df["arm"] == arm]
        ov = overrides(arm)
        changed = sorted(set(ov) & OBJECTIVE_CHANGING)
        L += ["---", "", f"## {arm}  (control `{sdf['control'].iloc[0]}`, "
              f"{len(sdf)} seeds)", "",
              "Sets " + ", ".join(f"`{k} = {v}`" for k, v in ov.items()) + ".", ""]
        if changed:
            L += [f"> **Validation loss is NOT comparable for this arm.** It changes "
                  f"`{', '.join(changed)}`, i.e. WHAT `loss_dt` is measured against. `loss_dt` is "
                  f"minimised at `1 + log E[dt]`, and `dt_ablation.py` measured the target's mean "
                  f"falling from 2.648 y to 1.488 y, so the *achievable* `loss_dt` drops by "
                  f"`log(2.648/1.488)` = **{DT_TARGET_ARTEFACT:.2f} nats** with no change in model "
                  f"quality. A val-loss delta smaller than that says nothing about quality — if "
                  f"anything, less than 0.58 means the model fits its own (easier) target WORSE "
                  f"than the control fit its harder one, which is expected once the 73.6% of "
                  f"positions that were being handed a subtraction they could read off their own "
                  f"input stop being free. Read section 2, not section 1.", ""]
        L += section(sub, sdf, "1. The objective terms",
                     [("loss_dt", "`loss_dt` (when)"), ("loss_ce", "`loss_ce` (which event)"),
                      ("best_val", "total val loss"),
                      ("best_iter", "iter of the best checkpoint")],
                     note=("Bookkeeping only for this arm — see the warning above."
                           if changed else
                           f"Yardstick: the capacity sweep moved `loss_dt` by "
                           f"**{CAP_DT_SPREAD:.4f}** nats across a 6.9x parameter range (its "
                           f"finding 2). A `d` at or beyond that, consistent in sign across "
                           f"seeds, is a result."))

        L += section(sub, sdf, "2. Downstream timing — THE headline",
                     [("timing_mae_y", "transition-time MAE (years)"),
                      ("timing_bias_y", "transition-time bias (years)"),
                      ("timing_spearman", "transition-time Spearman"),
                      ("calib_err", "mean calibration error")],
                     note="`a/transition_time` over ~2.4k observed transitions, from sampled "
                          "trajectories — objective-independent, so comparable for every arm. "
                          "The control's bias is **+2.6 y** (events predicted too late) and its "
                          "R2 is negative, i.e. worse than predicting the mean. Lower MAE is "
                          "better; bias closer to 0 is better; Spearman higher is better (read "
                          "its sign column inverted).")

        mae_cols = sorted(c[4:-1] for c in sdf.columns
                          if c.startswith("mae[") and c.endswith("]"))
        if mae_cols and f"d_mae[{mae_cols[0]}]" in sdf.columns:
            L += ["### 2b. Timing MAE by outcome and horizon", "",
                  "`naive` = predict one constant (the median observed time) for everyone. "
                  "Per-bucket it is rigged in the constant's favour, since the buckets are cut "
                  "on the observed time; the aggregate is the fair comparison.", "",
                  "| outcome / horizon | control MAE | arm MAE | d | naive |",
                  "|---|---:|---:|---:|---:|"]
            wsum = {"c": 0.0, "a": 0.0, "n": 0.0, "w": 0.0}
            for k in mae_cols:
                col, nv = f"mae[{k}]", f"naive[{k}]"
                if f"d_{col}" not in sdf.columns:
                    continue
                st = paired_table(sub, col)
                L.append(f"| {k} | {fmt(sdf[f'c_{col}'].mean(), 3)} | {fmt(sdf[col].mean(), 3)} | "
                         f"**{st['mean']:+.3f}** | "
                         f"{fmt(sdf[nv].mean(), 3) if nv in sdf else '--'} |")
            L += [""]

        L += section(sub, sdf, "3. Guards — did the categorical side pay for it?",
                     [("median_auc", "median transition AUC"),
                      ("mean_jaccard", "trajectory Jaccard")]
                     + [(c, c) for c in sorted(sdf.columns) if c.startswith("auc[")],
                     note="HIGHER is better here — read the last column as pairs worse.",
                     lower_is_better=False)

        L += ["### Per-pair detail", "",
              "| pair | val loss | d | best iter | d | timing bias (y) | d |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for _, r in sdf.sort_values("seed").iterrows():
            tb = r.get("timing_bias_y", np.nan)
            dtb = r.get("d_timing_bias_y", np.nan)
            L.append(f"| {r['run']} vs {r['control_run']} | {r['best_val']:.4f} | "
                     f"{r['d_best_val']:+.4f} | {int(r['best_iter'])} | "
                     f"{int(r['d_best_iter']):+d} | "
                     f"{('%.3f' % tb) if pd.notna(tb) else '--'} | "
                     f"{('%+.3f' % dtb) if pd.notna(dtb) else '--'} |")
        L += [""]

    if missing:
        L += ["---", "", "## Pairs not analysed", ""] + [f"- {m}" for m in missing] + [""]
    if df["nan_guard"].any():
        L += ["> **A run tripped train.py's nan-guard.** Check its log before trusting the row.", ""]
    have = {"decomposition (loss_ce / loss_dt split)": bool(dec_new),
            "downstream eval": "timing_mae_y" in df.columns}
    L += ["## What is not in here yet", ""] +          [f"- {k}: {'present' if v else 'MISSING'}" for k, v in have.items()] + [""]

    open(os.path.join(a.out, "PAIRED.md"), "w").write("\n".join(L))
    print("\n".join(L))
    print(f"[analyze] -> {a.out}/paired.csv  {a.out}/PAIRED.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
