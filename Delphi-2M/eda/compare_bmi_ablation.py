"""
compare_bmi_ablation.py -- pool the BMI-ablation runs and report the paired difference.

Reads the per-run JSON that eval_landmark_radc.py writes and answers one question: does
removing the BMI tokens (30-32) change landmark AD-risk discrimination?

TWO THINGS THIS DOES THAT A NAIVE COMPARISON WOULD GET WRONG.

1. Common evaluable set, per horizon. A subject's observed follow-up ends at their last
   EVENT, and for a handful of subjects that last event is a BMI token -- so dropping BMI
   shortens their window and moves them from "negative" to "censored". At 5y that is 2 of 576
   subjects. Small, but it means the two arms would otherwise be scored on different
   denominators. Every horizon here is scored on the INTERSECTION of subjects labelable in
   both arms, and the intersection size is printed.

2. loss_ce is NOT compared. The arms have different event counts and different in-use
   vocabularies, so their cross-entropies are not on the same scale. Only the landmark AUROC
   -- same subjects, same labels, same horizon -- is comparable.

Run:  python eda/compare_bmi_ablation.py out-radc-ablation/results/*.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "eda"))
from eval_landmark_radc import auroc, HORIZONS   # noqa: E402


def labelable(row, h):
    """(is_labelable, y) under the same censoring rule the evaluator uses."""
    if row["t_ad"] <= h:
        return True, True
    if row["t_last"] >= h:
        return True, False
    return False, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    files = [f for pat in a.files for f in sorted(glob.glob(pat))]
    runs = []
    for f in files:
        j = json.load(open(f))
        stem = os.path.basename(f).replace(".json", "")
        arm, seed = stem.rsplit("_s", 1)
        runs.append(dict(arm=arm, seed=int(seed), rows={r["pid"]: r for r in j["rows"]},
                         best_val=j["best_val_loss"], n_mc=j["n_mc"]))
    arms = sorted({r["arm"] for r in runs})
    seeds = sorted({r["seed"] for r in runs})
    print(f"runs: {len(runs)}   arms {arms}   seeds {seeds}   n_mc {runs[0]['n_mc']}")

    for h in HORIZONS:
        # subjects labelable in EVERY run (so all arms/seeds score the same denominator)
        common = None
        for r in runs:
            ok = {p for p, row in r["rows"].items() if labelable(row, h)[0]}
            common = ok if common is None else (common & ok)
        common = sorted(common)
        y = np.array([labelable(runs[0]["rows"][p], h)[1] for p in common])
        print(f"\n=== horizon {h}y   common n={len(common)}  positives {int(y.sum())} "
              f"({100 * y.mean():.1f}%) ===")

        per_arm = {}
        for arm in arms:
            vals = []
            for seed in seeds:
                r = next((x for x in runs if x["arm"] == arm and x["seed"] == seed), None)
                if r is None:
                    continue
                s = np.array([r["rows"][p][f"risk{h}"] for p in common])
                vals.append(auroc(y, s))
            per_arm[arm] = np.array(vals, float)
            v = per_arm[arm]
            print(f"  {arm:>6}  AUROC " + "  ".join(f"s{s}={x:.4f}" for s, x in zip(seeds, v))
                  + f"   mean {np.nanmean(v):.4f} ± {np.nanstd(v, ddof=1):.4f}")

        if len(arms) == 2:
            d = per_arm[arms[0]] - per_arm[arms[1]]
            n = len(d)
            se = np.nanstd(d, ddof=1) / np.sqrt(n) if n > 1 else float("nan")
            print(f"  paired diff ({arms[0]} - {arms[1]}): mean {np.nanmean(d):+.4f}"
                  + (f"  SE {se:.4f}   {np.nanmean(d) / se:+.2f} SE from 0" if n > 1 and se > 0
                     else ""))
            print(f"    per-seed: " + "  ".join(f"{x:+.4f}" for x in d))


if __name__ == "__main__":
    main()
