"""
lr_baseline.py -- per-transition L2 logistic regression on the BASELINE VISIT ONLY.

The question this answers: how much of Figure 2a's transition AUC needs a sequence model at all?
The baseline gets exactly the prompt the transformer's Monte-Carlo pass is seeded with (birth-time
statics + the whole first visit, as a binary bag of tokens, plus baseline age) and predicts the
same competing-risk-aware 5-year label. Nothing else changes: same split, same training-cohort
filter, same at-risk definition, same horizon.

Two variants:
  full  baseline age + binary bag of prompt tokens (APOE, education, sex, comorbidities, the
        baseline cognitive scales, ...)
  age   baseline age alone -- the "old people get dementia" null (README sec.12.2 item 4)

One model per transition type, fitted on TRAIN, C picked by val AUC, scored on TEST.

    python experiments/lr_baseline/lr_baseline.py
"""
import os, sys, json, logging
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DELPHI = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Delphi-2M")
sys.path.insert(0, DELPHI)
sys.path.insert(0, os.path.join(DELPHI, "figure2"))
import figure2_core as F2                      # noqa: E402
import figure2_panels as F2P                   # noqa: E402
import frame as FR                             # noqa: E402

from sklearn.linear_model import LogisticRegression   # noqa: E402
from sklearn.preprocessing import StandardScaler      # noqa: E402
from sklearn.pipeline import make_pipeline            # noqa: E402

log = logging.getLogger("lr")
H = F2P.H                                  # 5 years, the primary horizon
C_GRID = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0, 100.0]
MIN_PREV = 10                              # drop prompt tokens seen in <10 at-risk train patients
OUT = os.path.join(HERE, "results")


def xy(df, X, state, frm):
    """(features, label, row-mask) for one transition type on one split."""
    at_risk = df["baseline_state"].to_numpy() == frm
    y = F2.labels_at_h(FR.endpoint(df, state), H)
    m = at_risk & (y >= 0)                 # y == -1 is censored-before-h: unknowable, dropped
    return X[m], y[m].astype(int), m


def fit_one(Xtr, ytr, Xva, yva, cols):
    """L2 logistic regression, C by val AUC. Returns (fitted pipeline, feature mask, C, val AUC)."""
    prev = Xtr.sum(0)
    use = (prev >= MIN_PREV) & (prev <= len(Xtr) - MIN_PREV)
    use[0] = True                          # baseline_age is continuous, always keep
    best = (None, -1, np.nan)
    for C in C_GRID:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=C, max_iter=2000, solver="lbfgs"))
        clf.fit(Xtr[:, use], ytr)
        a = F2P._auc(yva, clf.predict_proba(Xva[:, use])[:, 1]) if len(np.unique(yva)) == 2 \
            else np.nan
        if np.isfinite(a) and a > best[1]:
            best = (clf, a, C)
    return best[0], use, best[2], best[1]


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(OUT, exist_ok=True)
    frames = {}
    for sp in ("train", "val", "test"):
        df, X, cols = FR.build(sp)
        frames[sp] = (df, X)
        log.info("%s: %d evaluable matched-cohort patients, X %s", sp, len(df), X.shape)
    cols = cols

    scores = {"pid": frames["test"][0]["pid"].to_numpy()}
    rows = []
    for (f, t) in F2P.transition_types():
        name = f"{F2.ALL_NAMES[f]}→{F2.ALL_NAMES[t]}"
        Xtr, ytr, _ = xy(*frames["train"], t, f)
        Xva, yva, _ = xy(*frames["val"], t, f)
        Xte, yte, mte = xy(*frames["test"], t, f)
        n_pos = int(yte.sum())
        if n_pos < F2P.MIN_POS or len(yte) - n_pos < F2P.MIN_POS:
            log.info("skip %-22s (test: %d pos / %d)", name, n_pos, len(yte))
            continue
        if len(np.unique(ytr)) < 2 or len(np.unique(yva)) < 2:
            log.info("skip %-22s (train/val single-class)", name)
            continue

        clf, use, C, va = fit_one(Xtr, ytr, Xva, yva, cols)
        s = clf.predict_proba(Xte[:, use])[:, 1]
        auc, lo, hi = F2P.boot_auc(yte, s)

        clf_a, use_a, C_a, va_a = fit_one(Xtr[:, :1], ytr, Xva[:, :1], yva, cols[:1])
        s_a = clf_a.predict_proba(Xte[:, :1])[:, 1]
        auc_a, lo_a, hi_a = F2P.boot_auc(yte, s_a)

        # full-length score vectors (nan where the patient is not at risk) so compare.py can
        # align them to the transformer frame by pid without re-fitting anything
        for tag, v in (("lr", s), ("lrage", s_a)):
            col = np.full(len(frames["test"][0]), np.nan)
            col[np.where(mte)[0]] = v
            scores[f"{tag}_{f}_{t}"] = col

        rows.append(dict(frm=F2.ALL_NAMES[f], to=F2.ALL_NAMES[t], label=name, from_idx=f,
                         to_idx=t, n_train=len(ytr), n_train_pos=int(ytr.sum()),
                         n_test=len(yte), n_test_pos=n_pos, n_feat=int(use.sum()), C=C,
                         val_auc=va, lr_auc=auc, lr_lo=lo, lr_hi=hi,
                         lr_age_auc=auc_a, lr_age_lo=lo_a, lr_age_hi=hi_a))
        log.info("%-22s n=%5d (%4d pos)  LR %.3f [%.3f,%.3f]   age-only %.3f  (C=%g, %d feat)",
                 name, len(yte), n_pos, auc, lo, hi, auc_a, C, use.sum())

    tab = pd.DataFrame(rows)
    tab.to_csv(os.path.join(OUT, "lr_auc.csv"), index=False)
    pd.DataFrame(scores).to_csv(os.path.join(OUT, "lr_scores_test.csv"), index=False)
    print("\nmedian LR AUC       %.3f  (%d transitions)" % (tab["lr_auc"].median(), len(tab)))
    print("median age-only AUC %.3f" % tab["lr_age_auc"].median())
    print("wrote", os.path.join(OUT, "lr_auc.csv"))


if __name__ == "__main__":
    main()
