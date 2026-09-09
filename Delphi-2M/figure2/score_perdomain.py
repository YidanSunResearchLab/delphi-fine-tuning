"""
score_perdomain.py -- turn the per-domain columns in the Figure-2 cache into the answer.

Reads the frame that figure2_core.build_mc() already wrote (no model, no MC, seconds to run)
and reports, for each of the 30 scales and the 3 reconstructed totals:

  AUC(worsen within h y) · mean predicted risk vs observed rate · timing MAE / bias

Competing risk is handled the way Figure 2 handles it: a patient who dies within the horizon
WITHOUT worsening is not a clean negative -- they were never given the chance -- so they are
dropped from that horizon rather than scored as a 0. Patients whose follow-up ends before the
horizon without worsening are dropped for the same reason (right-censoring).

  python figure2/score_perdomain.py                       # default cache, matched cohort
  python figure2/score_perdomain.py --cohort all
"""
import os, sys, json, argparse, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from figure2 import perdomain as PD          # noqa: E402
from figure2 import figure2_core as F2       # noqa: E402
from figure2 import plotting_style as ps     # noqa: E402
from delphi import predict_adapter as PA     # noqa: E402

MIN_POS = 30          # same floor Figure 2's panel a uses: below this an AUC is noise
HORIZONS = [1, 2, 3, 5, 10]
N_BOOT = 500          # matches figure2_panels.N_BOOT so the two figures' CIs are comparable


def auc(y, p):
    """Rank AUC with ties handled, no sklearn dependency."""
    y = np.asarray(y, bool); p = np.asarray(p, float)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos < MIN_POS or nneg < MIN_POS:
        return np.nan, npos, nneg
    r = pd.Series(p).rank().to_numpy()
    return (r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg), npos, nneg


def n_bins(name):
    if name in PA.SCALES:
        return len(PA.SCALES[name])
    return len(PD.TOTALS[name][1])          # reconstructed total: one bin per edge


def at_risk_mask(df, name):
    """Who can actually be scored on "does this scale worsen".

    Two exclusions, both structural rather than statistical:
      base_bin < 0   the scale was never measured for this patient -- no baseline, no label
      base_bin == top  the patient is ALREADY at the worst bin, so worsening is impossible.
                     They are guaranteed negatives AND the model scores them 0 by construction
                     (there is no higher token to sample), so every one of them is a free
                     correct answer that inflates AUC. It is 0.7-1.2% of the CDR domains but
                     13.6% of TAXES and 24.1% of NACCUDSD, so it is not a rounding error.
    """
    base = df[f"{name}__base_bin"].to_numpy()
    return (base >= 0) & (base < n_bins(name) - 1)


def score(df, name):
    ow = df[f"{name}__obs_t_worse"].to_numpy(float)
    fu = df["followup"].to_numpy(float)
    dth = df["obs_t_death"].to_numpy(float)
    scorable = at_risk_mask(df, name)
    res = {"n_scorable": int(scorable.sum()), "horizons": {}}
    for h in HORIZONS:
        p = df[f"{name}__pred_worse_{h}y"].to_numpy(float)
        worsened = np.isfinite(ow) & (ow <= h)
        # a clean negative must have been OBSERVED, ALIVE and un-worsened through h
        clean_neg = ~worsened & (fu >= h) & ~(np.isfinite(dth) & (dth <= h))
        keep = scorable & np.isfinite(p) & (worsened | clean_neg)
        if keep.sum() == 0:
            continue
        a, npos, nneg = auc(worsened[keep], p[keep])
        # Patient-level bootstrap CI. Without it an AUC of 0.52 cannot be told from 0.55 and
        # the whole "which domains are actually predictable" reading is unsupported. Same
        # helper and the same N_BOOT that Figure 2's panel a uses, so the CIs are comparable.
        lo = hi = None
        if not np.isnan(a):
            yk, pk = worsened[keep], p[keep]
            idx = np.arange(len(yk))
            def stat(sel, yk=yk, pk=pk):
                v, np_, nn_ = auc(yk[sel], pk[sel])
                return v
            _, lo, hi = ps.bootstrap_ci(stat, idx, n_boot=N_BOOT, seed=42)
        res["horizons"][h] = {
            "auc": None if np.isnan(a) else round(float(a), 4),
            "auc_lo": None if lo is None or np.isnan(lo) else round(float(lo), 4),
            "auc_hi": None if hi is None or np.isnan(hi) else round(float(hi), 4),
            "n_pos": npos, "n_neg": nneg,
            "mean_pred": round(float(p[keep].mean()), 4),
            "obs_rate": round(float(worsened[keep].mean()), 4),
            "calib_err": round(float(p[keep].mean() - worsened[keep].mean()), 4),
        }
    pt = df[f"{name}__pred_t_worse"].to_numpy(float)
    m = scorable & np.isfinite(ow) & np.isfinite(pt)
    if m.sum() >= MIN_POS:
        d = pt[m] - ow[m]
        res["timing"] = {"n": int(m.sum()), "mae": round(float(np.abs(d).mean()), 3),
                         "bias": round(float(d.mean()), 3),
                         "median_obs": round(float(np.median(ow[m])), 3)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", default=None, help="frame_*.pkl (default: newest, non-smoke)")
    ap.add_argument("--cohort", default="matched", choices=["matched", "all"])
    ap.add_argument("--dataset", default=F2.DATASET)
    ap.add_argument("--split", default=F2.SPLIT)
    ap.add_argument("--out", default=os.path.join(HERE, "results", "figure2"))
    a = ap.parse_args()

    frame = a.frame
    if frame is None:
        c = [f for f in glob.glob(os.path.join(HERE, "results", "figure2", "_cache", "frame_*.pkl"))
             if "_lim" not in f]
        if not c:
            sys.exit("no frame_*.pkl in results/figure2/_cache -- run figure2_core.py first")
        frame = max(c, key=os.path.getmtime)
    df = pd.read_pickle(frame)
    print(f"frame: {os.path.basename(frame)}  ({len(df):,} patients)")

    if a.cohort == "matched":
        # Use figure2_core's own mask, not a re-derivation. An approximation here (e.g.
        # "baseline_state != final_state" as a stand-in for ">=2 NACCUDSD tokens") would
        # select a slightly different population than the Figure-2 panels, and then the two
        # sets of numbers could not be quoted side by side -- which is the entire point.
        df = df.reset_index(drop=True)
        keep = F2.training_cohort_mask(dataset=a.dataset, split=a.split)
        df = df[keep[df["pid"].to_numpy()]].reset_index(drop=True)
        print(f"cohort-matched (figure2_core.training_cohort_mask) -> {len(df):,} patients")

    out = {}
    for name in PD.ALL_OUTCOMES:
        if f"{name}__base_bin" not in df.columns:
            continue
        out[name] = score(df, name)

    os.makedirs(a.out, exist_ok=True)
    jp = os.path.join(a.out, f"perdomain_{a.cohort}.json")
    json.dump(out, open(jp, "w"), indent=1)

    # ---- human-readable table, primary horizon 5 y
    rows = []
    for name, r in out.items():
        h = r.get("horizons", {}).get(5, {})
        t = r.get("timing", {})
        rows.append({"outcome": name, "n": r["n_scorable"], "auc@5y": h.get("auc"),
                     "auc_lo": h.get("auc_lo"), "auc_hi": h.get("auc_hi"),
                     "n_pos": h.get("n_pos"), "obs_rate": h.get("obs_rate"),
                     "calib_err": h.get("calib_err"), "mae": t.get("mae"), "bias": t.get("bias")})
    tab = pd.DataFrame(rows)
    tp = os.path.join(a.out, f"perdomain_{a.cohort}.csv")
    tab.to_csv(tp, index=False)
    with pd.option_context("display.width", 200, "display.max_rows", 60):
        print(tab.to_string(index=False))
    print(f"\n-> {jp}\n-> {tp}")


if __name__ == "__main__":
    main()
