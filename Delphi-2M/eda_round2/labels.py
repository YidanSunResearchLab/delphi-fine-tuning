"""Prediction targets shared by sections 6 and 9.

This release has NO per-visit diagnosis (`dcfdx` is absent), so "next-visit diagnosis" has to
be reconstructed. Two targets, both stated as proxies wherever they are reported:

  y_ad_next   incident AD dementia by the next observed visit, from the person-level
              `age_first_ad_dx` (codebook: age at the first cycle with dcfdx in {4,5}).
              Rows where the participant is ALREADY diagnosed at t are dropped -- they are
              not at risk. Participants who look demented at baseline are dropped entirely,
              because the codebook says age_first_ad_dx is simply not recorded for them, so
              their 0 label would be wrong rather than merely censored.

  y_imp_next  next-visit cognitive impairment, estimated MMSE < 24 -- the conventional
              screening cut. Available at every visit, so it is the higher-powered target;
              it is a screening proxy, not a diagnosis.
"""
import numpy as np
import pandas as pd

MMSE_IMPAIRED = 24
BASELINE_PREVALENT_MMSE = 24


def make_pairs(v):
    d = v.sort_values(["projid", "fu_year"]).copy()
    g = d.groupby("projid", sort=False)
    d["next_age"] = g["age_at_visit"].shift(-1)
    d["next_fu"] = g["fu_year"].shift(-1)
    d["next_mmse"] = g["cts_estmmse30"].shift(-1)
    d["next_cogn"] = g["cogn_global"].shift(-1)
    d["prev_cogn"] = g["cogn_global"].shift(1)
    d["prev_mmse"] = g["cts_estmmse30"].shift(1)
    d["d_cogn"] = d["cogn_global"] - d["prev_cogn"]
    d["d_mmse"] = d["cts_estmmse30"] - d["prev_mmse"]
    d["next_d_fu"] = d["next_fu"] - d["fu_year"]

    bl = d[d["visit_idx"] == 0].set_index("projid")
    prevalent = bl.index[(bl["age_first_ad_dx"].isna()) & (bl["cts_estmmse30"] < BASELINE_PREVALENT_MMSE)]
    d["suspected_prevalent_dementia"] = d["projid"].isin(prevalent)

    p = d[d["next_age"].notna()].copy()
    p["y_ad_next"] = ((p["age_first_ad_dx"].notna())
                      & (p["age_first_ad_dx"] <= p["next_age"] + 1e-9)).astype(int)
    p["at_risk_ad"] = (~p["ad_now"].astype(bool)) & (~p["suspected_prevalent_dementia"].astype(bool))
    p["y_imp_next"] = np.where(p["next_mmse"].notna(), (p["next_mmse"] < MMSE_IMPAIRED).astype(float), np.nan)
    p["at_risk_imp"] = p["next_mmse"].notna() & (p["cts_estmmse30"] >= MMSE_IMPAIRED)
    return p


def target_frame(pairs, target):
    """Rows eligible for one target, with the label as a plain int column `y`."""
    if target == "y_ad_next":
        f = pairs[pairs["at_risk_ad"]].copy()
    elif target == "y_imp_next":
        f = pairs[pairs["at_risk_imp"]].copy()
    else:
        raise ValueError(target)
    f["y"] = f[target].astype(int)
    return f


TARGETS = {
    "y_ad_next": "incident AD dementia by the next visit (proxy, from age_first_ad_dx)",
    "y_imp_next": "next-visit estimated MMSE < 24 (screening proxy for impairment)",
}
