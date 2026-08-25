"""Shared out-of-fold scoring for §3.6 and §4.3.

Rules that make these numbers honest, all enforced here rather than per caller:
  - folds are grouped by PERSON, so no participant appears in both train and test
  - the imputer and the scaler are fit on the TRAINING fold only
  - every feature gets an explicit missing-indicator column: for these variables missingness
    is informative (a scale that was not administered is not a scale scoring in the middle)
"""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

N_SPLITS = 5


def _design(df, cols):
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    miss = np.isnan(X).astype(float)
    keep = miss.std(axis=0) > 0                     # only add indicators that vary
    return np.hstack([X, miss[:, keep]])


def oof_auc(df, cols, y_col, group_col, seed=42):
    """Grouped 5-fold out-of-fold AUC for a logistic model on `cols`."""
    d = df[list(dict.fromkeys(cols)) + [y_col, group_col]].copy()
    d = d[pd.to_numeric(d[y_col], errors="coerce").notna()]
    y = pd.to_numeric(d[y_col], errors="coerce").to_numpy(float)
    if len(d) < 200 or y.min() == y.max():
        return np.nan, 0, np.nan
    X = _design(d, cols)
    g = d[group_col].to_numpy()
    est = make_pipeline(SimpleImputer(strategy="median"),
                        StandardScaler(),
                        LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=N_SPLITS).split(X, y, g):
        if y[tr].min() == y[tr].max():
            continue
        est.fit(X[tr], y[tr])
        pred[te] = est.predict_proba(X[te])[:, 1]
    ok = np.isfinite(pred)
    if ok.sum() < 100 or len(np.unique(y[ok])) < 2:
        return np.nan, int(ok.sum()), float(y[ok].mean()) if ok.sum() else np.nan
    return float(roc_auc_score(y[ok], pred[ok])), int(ok.sum()), float(y[ok].mean())


def forward_select(df, baseline, candidates, y_col, group_col, min_gain=0.0, max_steps=None):
    """Greedy forward selection. Returns (steps DataFrame, chosen list).

    `marginal_gain` is the AUC increase over the PREVIOUS step, not over the baseline.
    Runs to exhaustion (or max_steps) and records every step, including the ones below
    min_gain, so the table shows where the curve goes flat rather than stopping silently
    at the first small gain.
    """
    chosen = list(baseline)
    base_auc, n, prev = oof_auc(df, chosen, y_col, group_col) if chosen else (0.5, 0, np.nan)
    rows = [{"step": 0, "added": "(baseline) " + ", ".join(baseline) if baseline else "(none)",
             "auc": base_auc, "marginal_gain": np.nan, "n_rows": n, "event_rate": prev,
             "n_features": len(chosen)}]
    pool = [c for c in candidates if c not in chosen]
    step = 0
    while pool and (max_steps is None or step < max_steps):
        step += 1
        scored = []
        for c in pool:
            a, nn, ev = oof_auc(df, chosen + [c], y_col, group_col)
            scored.append((a if np.isfinite(a) else -np.inf, c, nn, ev))
        scored.sort(reverse=True)
        best_auc, best_c, nn, ev = scored[0]
        if not np.isfinite(best_auc):
            break
        rows.append({"step": step, "added": best_c, "auc": best_auc,
                     "marginal_gain": best_auc - base_auc, "n_rows": nn, "event_rate": ev,
                     "n_features": len(chosen) + 1})
        chosen.append(best_c)
        pool.remove(best_c)
        base_auc = best_auc
    return pd.DataFrame(rows), chosen
