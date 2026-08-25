"""
compare.py -- transformer vs logistic regression, per transition type, on identical rows.

Both scores are read for the SAME patients with the SAME competing-risk label, so the comparison
is paired: the CI on the difference comes from resampling patients once and recomputing both
AUCs, not from overlapping the two marginal CIs (which is the conservative-to-the-point-of-useless
way to compare correlated estimates).

The transformer score is `figure2_panels.compute_A`'s: the Monte-Carlo probability of reaching the
target state within 5 years before the sampled death. Reads the `_matched` cache written by
mc_matched.py.

    python experiments/lr_baseline/compare.py --cap-years 6
"""
import os, sys, argparse, logging
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DELPHI = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Delphi-2M")
sys.path.insert(0, DELPHI)
sys.path.insert(0, os.path.join(DELPHI, "figure2"))
import figure2_core as F2                      # noqa: E402
import figure2_panels as F2P                   # noqa: E402
from delphi import predict_adapter as PA       # noqa: E402
import mc_matched                              # noqa: E402

log = logging.getLogger("cmp")
H = F2P.H
N_BOOT = F2P.N_BOOT
OUT = os.path.join(HERE, "results")


def paired_boot(y, a, b, n_boot=N_BOOT, seed=F2P.RNG_SEED):
    """95% CI on AUC(a) - AUC(b) and the one-sided bootstrap p that a <= b, resampling patients."""
    rng = np.random.default_rng(seed)
    d = []
    n = len(y)
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(y[ix])) < 2:
            continue
        d.append(F2P._auc(y[ix], a[ix]) - F2P._auc(y[ix], b[ix]))
    if not d:
        return np.nan, np.nan, np.nan
    d = np.asarray(d)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d <= 0))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap-years", type=float, default=6,
                    help="cap of the MC artefact to read; must be >= the horizon scored")
    a = ap.parse_args()
    assert a.cap_years == 0 or a.cap_years >= H, (
        f"a cap of {a.cap_years}y cannot support the {H}y horizon -- risk would be truncated")
    ad = PA.load_model(dict(checkpoint=os.path.join(DELPHI, F2.CKPT), device="cpu", seed=F2.SEED))
    key = mc_matched.cache_key(ad.ckpt_sig, F2.N_MC, "test", F2.DATASET, a.cap_years)
    df = pd.read_pickle(os.path.join(F2.CACHE_DIR, f"frame_{key}.pkl")).reset_index(drop=True)
    grids = dict(np.load(os.path.join(F2.CACHE_DIR, f"grids_{key}.npz")))
    log.info("transformer frame: %d patients (key %s)", len(df), key)

    lr = pd.read_csv(os.path.join(OUT, "lr_scores_test.csv")).set_index("pid")
    lr = lr.reindex(df["pid"].to_numpy())          # align LR rows to the transformer frame
    log.info("LR scores aligned: %d of %d rows matched", int(lr.notna().any(1).sum()), len(df))

    b = df["baseline_state"].to_numpy()
    eps = {t: F2.composite(df, grids, [t]) for t in range(5)}
    rows = []
    for (f, t) in F2P.transition_types():
        if f"lr_{f}_{t}" not in lr.columns:
            continue
        ep = eps[t]
        y_all = F2.labels_at_h(ep, H)
        s_tr = ep["risk"][H]
        s_lr = lr[f"lr_{f}_{t}"].to_numpy(float)
        s_ag = lr[f"lrage_{f}_{t}"].to_numpy(float)
        m = (b == f) & (y_all >= 0) & np.isfinite(s_tr) & np.isfinite(s_lr) & np.isfinite(s_ag)
        y = y_all[m].astype(int)
        n_pos = int(y.sum())
        if n_pos < F2P.MIN_POS or len(y) - n_pos < F2P.MIN_POS:
            continue
        s_tr, s_lr, s_ag = s_tr[m], s_lr[m], s_ag[m]
        a_tr, tlo, thi = F2P.boot_auc(y, s_tr)
        a_lr, llo, lhi = F2P.boot_auc(y, s_lr)
        a_ag, alo, ahi = F2P.boot_auc(y, s_ag)
        dlo, dhi, p = paired_boot(y, s_lr, s_tr)
        rows.append(dict(label=f"{F2.ALL_NAMES[f]}→{F2.ALL_NAMES[t]}", from_idx=f, to_idx=t,
                         n=len(y), n_pos=n_pos,
                         transformer_auc=a_tr, transformer_lo=tlo, transformer_hi=thi,
                         lr_auc=a_lr, lr_lo=llo, lr_hi=lhi,
                         age_auc=a_ag, age_lo=alo, age_hi=ahi,
                         delta=a_lr - a_tr, delta_lo=dlo, delta_hi=dhi, p_lr_not_better=p))
        log.info("%-22s n=%5d (%4d pos)  transformer %.3f   LR %.3f   Δ %+.3f [%+.3f,%+.3f]",
                 rows[-1]["label"], len(y), n_pos, a_tr, a_lr, a_lr - a_tr, dlo, dhi)

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "compare_auc.csv"), index=False)
    print()
    print(tab[["label", "n", "n_pos", "transformer_auc", "lr_auc", "age_auc", "delta",
               "delta_lo", "delta_hi", "p_lr_not_better"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\nmedian AUC over {len(tab)} transitions:"
          f"  transformer {tab['transformer_auc'].median():.3f}"
          f"   LR {tab['lr_auc'].median():.3f}"
          f"   age-only {tab['age_auc'].median():.3f}")
    print(f"LR wins {int((tab['delta'] > 0).sum())}/{len(tab)} transitions;"
          f" significantly (95% CI excludes 0): {int((tab['delta_lo'] > 0).sum())}")
    print("wrote", os.path.join(OUT, "compare_auc.csv"))


if __name__ == "__main__":
    main()
