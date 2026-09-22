"""
per_state_auc.py -- ONE AUC per state, pooled over the baseline state.

WHY THIS EXISTS. Panel a1 reports a (baseline stage -> reached state) GRID: 20 cells, of which
5 clear the event floor. Two things are wrong with that as the headline.

  1. The label reads as a token adjacency. "Mild->Death" is not "the Death token follows the
     Mild token": the row's at-risk set is `baseline_state == Mild` and the outcome is reaching
     Death by ANY path within the horizon. The arrow notation invites the wrong reading.
  2. It answers a question nobody asked. "Can the model tell who will reach Mild" is one
     number; splitting it by where the subject started turns it into three underpowered ones.

So this pools over the origin. For each state it asks exactly one question:

    among subjects not already in this state at baseline, does the model rank those who
    reach it within H years above those who do not?

Same machinery as panel a -- F2.composite for the endpoint, F2.labels_at_h for the
competing-risk/censoring-aware label, subject bootstrap for the CI -- so the numbers are
directly comparable to a1's, they are just marginalised rather than stratified.

AD is included here and is absent from a1, which only walks the state space. It is the one
endpoint in this figure that is a clinical diagnosis rather than a test score.

    python figure2/per_state_auc.py --ckpt out-radc-v3-final/ckpt.pt --dataset radc-v3-s42
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from figure2 import figure2_core as F2      # noqa: E402
from figure2 import radc_states as S        # noqa: E402


def boot_auc(y, s, n_boot=500, seed=42):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(y, int); s = np.asarray(s, float)
    if len(np.unique(y)) < 2:
        return np.nan, np.nan, np.nan
    a = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        # resample SUBJECTS; one row per subject here, so this is the plain bootstrap
        i = rng.integers(0, len(y), len(y))
        if len(np.unique(y[i])) > 1:
            out.append(roc_auc_score(y[i], s[i]))
    lo, hi = (np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan))
    return float(a), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=F2.CKPT)
    ap.add_argument("--dataset", default=F2.DATASET)
    ap.add_argument("--split", default=F2.SPLIT)
    ap.add_argument("--n-mc", type=int, default=F2.N_MC)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--horizons", type=float, nargs="+", default=[1, 3, 5])
    ap.add_argument("--cohort", default="matched", choices=["matched", "all"])
    ap.add_argument("--min-events", type=int, default=25)
    a = ap.parse_args()

    F2._engine(a.ckpt, a.dataset, "cpu")            # binds radc_states to this token table
    df, grids, _trans, _emb = F2.load_cache(ckpt=a.ckpt, dataset=a.dataset, split=a.split,
                                            n_mc=a.n_mc, limit=a.limit)
    df = df.reset_index(drop=True)
    if a.cohort == "matched":
        keep = F2.training_cohort_mask(dataset=a.dataset, split=a.split)
        sel = keep[df["pid"].to_numpy()]
        n0 = len(sel)
        grids = {k: (v[sel] if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == n0 else v)
                 for k, v in grids.items()}
        df = df[sel].reset_index(drop=True)

    b = df["baseline_state"].to_numpy()
    rows = []
    # every slot: the cognitive stages, Death, and the AD diagnosis
    for slot in range(S.NSLOT):
        name = S.ALL_NAMES[slot]
        ep = F2.composite(df, grids, [slot])
        # AT RISK. For a stage, exclude subjects already in it at baseline -- otherwise they
        # are positive by construction. Death and AD cannot be present at baseline in this
        # stream (prevalent AD is a STATIC, not an event), so everyone is at risk of the
        # incident version.
        at_risk = (b != slot) if slot < S.NSTAGE else np.ones(len(df), bool)
        for H in a.horizons:
            y_all = F2.labels_at_h(ep, H)
            s_all = ep["risk"][H] if H in ep["risk"] else None
            if s_all is None:
                continue
            m = at_risk & (y_all >= 0) & np.isfinite(s_all)
            y, s = y_all[m].astype(int), np.asarray(s_all)[m]
            npos = int(y.sum())
            auc, lo, hi = ((np.nan, np.nan, np.nan) if npos < a.min_events
                           else boot_auc(y, s))
            rows.append(dict(state=name, horizon=float(H), n=int(m.sum()), n_events=npos,
                             prevalence=round(npos / max(int(m.sum()), 1), 4),
                             auc=auc, ci_lo=lo, ci_hi=hi,
                             mean_pred=float(np.mean(s)) if len(s) else np.nan,
                             mean_obs=float(np.mean(y)) if len(y) else np.nan))

    tag = os.environ.get("FIG2_TAG", "") or "v3_final"
    for H in a.horizons:
        sub = [r for r in rows if r["horizon"] == H]
        sub.sort(key=lambda r: (-(r["auc"] if np.isfinite(r["auc"]) else -1)))
        print(f"\n=== [{tag}] reach-state AUC, horizon {H:g} y, cohort={a.cohort} "
              f"{'-' * 30}\n  {'state':<16}{'n':>7}{'events':>8}{'prev':>7}{'AUC':>8}"
              f"{'95% CI':>18}{'pred':>7}{'obs':>7}")
        for r in sub:
            auc = "  n/a " if not np.isfinite(r["auc"]) else f"{r['auc']:>8.3f}"
            ci = "" if not np.isfinite(r["ci_lo"]) else f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
            print(f"  {r['state']:<16}{r['n']:>7,}{r['n_events']:>8,}{r['prevalence']:>7.3f}"
                  f"{auc}{ci:>18}{r['mean_pred']:>7.3f}{r['mean_obs']:>7.3f}")

    out = os.path.join(F2.OUT_DIR, "per_state_auc.json")
    os.makedirs(F2.OUT_DIR, exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(ckpt=a.ckpt, dataset=a.dataset, split=a.split, cohort=a.cohort,
                       n_mc=a.n_mc, min_events=a.min_events, rows=rows), f, indent=1,
                  default=str)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
