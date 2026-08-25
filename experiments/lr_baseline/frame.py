"""
frame.py -- the MODEL-FREE half of the Figure-2 patient frame, plus a baseline feature matrix.

`figure2_core._process` computes two kinds of thing per patient: observed quantities (baseline
age/state, follow-up, first-passage times) and Monte-Carlo predictions. Only the first kind is
needed to build labels, and it costs no forward passes. This module recomputes exactly that half
-- same `_baseline` helper, same first-passage definition -- so a non-neural baseline can be
scored against identical labels without running the transformer.

It also extracts the BASELINE PROMPT as features: the tokens at age <= baseline, which is
literally the context `figure2_core` seeds the Monte-Carlo trajectories with. Same information
in, so the comparison is about the model, not about who got to see more.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                    "Delphi-2M")
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "figure2"))
import figure2_core as F2                                    # noqa: E402
from delphi import predict_adapter as PA                     # noqa: E402

D = PA.DAYS_PER_YEAR
VOCAB = 111
# Tokens usable as baseline features. 0 = Padding never carries information; Death (110) cannot
# appear in a prompt. Everything else the model sees in its prompt, the baseline sees too --
# including "No event" (1), whose presence is monitoring-pattern signal both models get.
FEATURE_TOKENS = np.arange(1, 110)


def build(split, dataset=F2.DATASET, matched=True):
    """(df, X, tok_cols) for one split.

    df  one row per EVALUABLE patient, columns as in the Figure-2 frame (the observed subset).
    X   (n, 1 + len(FEATURE_TOKENS)) float: [baseline_age_years, binary bag of prompt tokens].
    """
    data, p2i = PA.load_split(dataset, split)
    keep = F2.training_cohort_mask(dataset=dataset, split=split) if matched \
        else np.ones(len(p2i), dtype=bool)
    rows, feats = [], []
    for k in np.where(keep)[0]:
        ages, toks = PA.get_history(data, p2i, k)
        bp = F2._baseline(ages, toks)
        if bp is None:                       # <2 distinct visits, or no baseline NACCUDSD state
            continue
        base, pmask, b = bp
        a = np.asarray(ages, float); t = np.asarray(toks)
        last_obs = float(a[a > 0].max())
        row = dict(pid=int(k), baseline_age=base / D, baseline_state=int(b),
                   followup=(last_obs - base) / D, n_visits=int(np.unique(a[a > 0]).size))
        for i, tok in enumerate(F2.ALL_TOKENS):
            m = (t == tok) & (a > base)
            row[f"obs_t_{F2.ALL_NAMES[i]}"] = (float(a[m].min()) - base) / D if m.any() else np.inf
        rows.append(row)

        bag = np.zeros(len(FEATURE_TOKENS), dtype=np.float32)
        present = np.unique(t[pmask])
        bag[np.isin(FEATURE_TOKENS, present)] = 1.0
        feats.append(np.concatenate([[base / D], bag]))

    df = pd.DataFrame(rows).reset_index(drop=True)
    X = np.asarray(feats, dtype=np.float32)
    return df, X, ["baseline_age"] + [f"tok{t}" for t in FEATURE_TOKENS]


def endpoint(df, state):
    """The observed half of `figure2_core.composite(df, grids, [state])` -- everything
    `labels_at_h` reads, and nothing that needs the model."""
    return dict(obs_time=df[f"obs_t_{F2.ALL_NAMES[state]}"].to_numpy(float),
                d_obs=df["obs_t_Death"].to_numpy(float),
                fu=df["followup"].to_numpy(float))
