"""
cohort.py -- who is evaluated, and what their outcome is. Shared by every scoring script.

There is exactly one definition of the evaluation cohort and one definition of the outcome,
and they live here, because the alternative is three scripts that each quietly answer a
slightly different question and produce three numbers nobody can reconcile.

THE THREE POPULATIONS
    all           4,428 subjects. The denominator for every prevalence / mortality statistic
                  that describes the cohort. Nothing is scored on it.
    training      >= 2 distinct event ages: 4,029 (91.0%). The 399 single-visit subjects
                  cannot support a next-event objective at all. They are NOT a random sample
                  (275 of 399 are MAP, 187 died, mean age_bl 80.3), so they are excluded from
                  fitting and re-included in every description.
    evaluation    training AND not baseline-impaired: ~3,691, of whom ~1,051 convert.

DEMENTIA AT ENTRY, and why the AD label needs it. The codebook is explicit that
age_first_ad_dx "is not available for participants that were demented at baseline cycle". So
several hundred prevalent-dementia subjects sit in the negative class with no AD token at all,
spending whole trajectories at MMSE 15-23 -- actively teaching the model that severe
impairment is compatible with being AD-free. They are EXCLUDED from AD scoring rather than
relabelled: entering them as baseline AD events would move the Aalen-Johansen target from
0.174/0.259 to 0.255/0.349 and silently invalidate the calibration.

The flag is computed from the longitudinal file at fu_year == 0 as
    (cts_estmmse30 < 24) OR (ad_rx == 1)
which covers all 4,428 subjects. The obvious alternative -- ROSMAP_clinical's cogdx/dcfdx_lv
-- is unavailable for the 847 subjects missing from that file (including all 308 LATC) and
exists only for died == 1 subjects, so it leaks mortality into the exclusion rule.

ad_rx is banned from the VOCABULARY but allowed here, and the distinction is real: in the
vocabulary it would let the model anticipate a future diagnosis; at fu_year 0 it can only mark
prevalence, which is exactly what this flag is for.

OUTCOME CODING for the competing-risks estimators:
    1  incident AD    -- age_first_ad_dx
    2  death without a preceding AD diagnosis
    0  censored alive at last observation
"""
import os
import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25


def baseline_impaired(radc_dir):
    """Series indexed by projid: was this subject already impaired at fu_year 0?

    LEGACY PATH. The builder now writes `baseline_impaired` straight into subjects.csv from
    tokenizer.baseline_impairment, which is the one definition; load_subjects prefers that
    column and only falls back here for a directory built before it existed. Keep the two
    bodies identical -- a silent divergence between the training mask and this exclusion is
    exactly the failure this consolidation exists to prevent.
    """
    lg = pd.read_excel(os.path.join(radc_dir, "longitudinal_data_gk.xlsx"))
    b = lg[lg["fu_year"] == 0].set_index("projid")
    mmse = pd.to_numeric(b.get("cts_estmmse30"), errors="coerce")
    adrx = pd.to_numeric(b.get("ad_rx"), errors="coerce")
    return ((mmse < 24) | (adrx == 1)).fillna(False)


def load_subjects(data_dir, radc_dir=None, split=None):
    """subjects.csv plus the derived cohort flags and competing-risk outcome.

    Adds:
      baseline_impaired  bool -- from subjects.csv if present, else recomputed
      ad_label_missing   bool -- baseline_impaired AND no age_first_ad_dx: the AD label is
                         absent by codebook rule, not negative. train.py masks the AD column
                         out of their cross-entropy; they stay excluded from AD scoring here.
      in_training        >= 2 visits
      in_eval            in_training and not baseline_impaired
      event_type         1 AD / 2 death without AD / 0 censored alive
      event_age          age (years) at that event, or at last observation if censored
      entry_age          age_bl -- the left-truncation point, without which every incidence
                         estimate is diluted by person-time nobody was observed for
    """
    sub = pd.read_csv(os.path.join(data_dir, "subjects.csv")).set_index("projid")
    if split:
        sub = sub[sub["split"] == split]

    if "baseline_impaired" in sub.columns:
        # written by the builder -- the authoritative copy, and it costs no Excel read
        sub["baseline_impaired"] = sub["baseline_impaired"].fillna(False).astype(bool)
    elif radc_dir:
        imp = baseline_impaired(radc_dir)
        sub["baseline_impaired"] = imp.reindex(sub.index).fillna(False).astype(bool)
    else:
        sub["baseline_impaired"] = False

    # Whose AD label is MISSING rather than negative -- the subset train.py masks out of the
    # cross-entropy. Derived rather than required, so a pre-existing subjects.csv still loads.
    # ---- baseline dementia status: the codebook-derived replacement for the old heuristic ---
    # `baseline_impaired` (MMSE < 24 OR baseline ad_rx) was wrong in BOTH directions, measured:
    # of the 400 it flagged, 113 (28.2%) went on to a recorded incident diagnosis and so were
    # not prevalent at all; and of the 200 subjects the codebook rule identifies as AD-demented
    # at entry it caught only 132 (66%), missing 67 whose baseline MMSE is a median of 25.
    # It is kept as a column for comparison and is no longer used for anything.
    if "dementia_at_entry" in sub.columns:
        dem = sub["dementia_at_entry"].fillna("unknown")
    else:                                   # a build made before the static existed
        dem = pd.Series(np.where(sub["baseline_impaired"], "yes", "no"), index=sub.index)
    sub["prevalent_ad"] = dem.eq("yes")
    sub["entry_status_unknown"] = dem.eq("unknown")

    sub["in_training"] = sub["n_visits"] >= 2
    # PREVALENT CASES ARE NOT AT RISK OF AN INCIDENT DIAGNOSIS -- they already have the
    # disease -- so they leave the incidence denominator. That is the whole reason the exclusion
    # exists, and it is now the reason rather than a cognitive-score proxy for it.
    sub["in_eval"] = sub["in_training"] & ~sub["prevalent_ad"]

    # WHOSE AD LABEL IS MISSING RATHER THAN NEGATIVE. For a subject demented at entry the
    # label is not "no AD" but "AD, onset unobserved" -- the codebook forbids recording the
    # age -- so train.py blanks the AD column out of their cross-entropy. This is now derived
    # from the codebook rule, not from the old MMSE/ad_rx proxy.
    #
    # The 793 subjects with UNKNOWN entry status are deliberately NOT masked. Roughly 8% of
    # them are prevalent by extrapolation from the judgeable population (~64 subjects, ~2% of
    # the negative class), and masking all 793 to fix that would throw away 18% of the cohort's
    # negative evidence. The contamination is documented and belongs in a sensitivity arm.
    sub["ad_label_missing"] = sub["prevalent_ad"] & ~sub["ever_ad"].astype(bool)

    ad = sub["age_ad"].to_numpy(float)
    death = sub["age_death"].to_numpy(float)
    last = sub["age_last_obs"].to_numpy(float)
    has_ad = ~np.isnan(ad)
    died = ~np.isnan(death)

    etype = np.where(has_ad, 1, np.where(died, 2, 0))
    eage = np.where(has_ad, ad, np.where(died, death, last))
    # A death recorded at or before the diagnosis would make the AD event unobservable; the
    # tokenizer already forces death strictly after every other token, so this is a guard
    # against a future change rather than a live case.
    bad = has_ad & died & (death < ad)
    if bad.any():
        raise AssertionError(f"{int(bad.sum())} subjects have a death age before their AD age")

    sub["event_type"] = etype
    sub["event_age"] = eage
    sub["entry_age"] = sub["age_bl"].to_numpy(float)
    return sub


def landmarks(ages_days, toks, sub_row, min_lead_years=0.0, min_prior_visits=2):
    """The ages at which this subject may be scored, in DAYS.

    A landmark is legal only if the outcome is still strictly in the future at that age and
    enough history precedes it. `min_lead_years` is the required gap to the diagnosis: a
    "5-year-ahead" claim scored at landmarks one month before diagnosis is not a 5-year-ahead
    claim, so the lead is an argument here and must be quoted with every metric.

    THE ENDPOINT MUST NOT BE IN ITS OWN PREFIX, and getting this wrong is silent and severe.
    An earlier version cut on `age <= event_age * 365.25` using the FLOAT age from
    subjects.csv, while the tokenizer places the endpoint token at `round(age * 365.25)` -- so
    for about half of converters the diagnosis's own age passed the cut and became a landmark.
    The prefix handed to the model then already contained the answer, and because AD_DX is not
    in vocab.REPEATABLE_TOKENS the sampler is forbidden from emitting it again, so every one of
    those rows scored a risk of EXACTLY 0.0 while being labelled positive. Measured on val:
    61 such AD rows and 103 Death rows out of 1,898 -- certain positives pinned to the bottom
    of the ranking, which drags the AUC down rather than up, so it does not even look like a
    leak in the output.

    The cut is therefore taken against the STREAM -- the age the endpoint token actually
    occupies -- and is strict. `toks` is used, not ignored.

    min_prior_visits defaults to 2: one token is a bare static block with nothing to predict
    from. This is the model's own protocol, and any baseline compared against the model must
    pass the same value or the comparison is not like-for-like (a visit-grid landmark set with
    min_prior_visits=1 shares only 13,643 of 24,520 rows with this one, and the 5-year AUC
    differs by 0.027).
    """
    from radc_delphi import vocab as V

    ages_days = np.asarray(ages_days, dtype=float)
    toks = np.asarray(toks)
    real = np.unique(ages_days[ages_days > 0])
    if len(real) < min_prior_visits:
        return np.array([], dtype=float)

    # the age of the first endpoint token in this subject's own stream, if any
    is_end = np.isin(toks, list(V.ENDPOINT_IDS)) & (ages_days > 0)
    endpoint_day = float(ages_days[is_end].min()) if is_end.any() else np.inf

    out = real[real < endpoint_day]                      # strict: never land on the endpoint
    if sub_row["event_type"] == 1 and min_lead_years > 0:
        out = out[out <= (sub_row["event_age"] - min_lead_years) * DAYS_PER_YEAR]
    elif sub_row["event_type"] != 1:
        # censored or died AD-free: cannot score past the end of observation either
        out = out[out <= sub_row["event_age"] * DAYS_PER_YEAR]

    keep = [a for a in out if (real <= a).sum() >= min_prior_visits]
    return np.array(keep, dtype=float)


def label_at_horizon(sub_row, landmark_days, horizon_years):
    """The cause-specific label for one (subject, landmark, horizon), or None if unscoreable.

    Returns 1 (AD within the horizon), 0 (survived the horizon AD-free), or None.

    None -- dropped, not scored as a negative -- covers the two cases where the truth is
    unknown or the subject was never given the chance:
      * follow-up ends before the horizon with no event: right-censored.
      * death before the horizon without a diagnosis: a competing event. Scoring it as a
        clean negative is what makes a lethal comorbidity look protective.
    Death is the reason this cohort needs the distinction rather than a convention: 53.3% of
    non-converters died, and treating those as censored inflates cumulative AD incidence by
    +38.6% at age 85 and +59.4% at 90.
    """
    lm_y = landmark_days / DAYS_PER_YEAR
    end = lm_y + horizon_years
    et, ea = int(sub_row["event_type"]), float(sub_row["event_age"])
    if et == 1:
        return 1 if ea <= end else 0
    if et == 2:
        return None if ea <= end else 0                  # died inside the window: competing
    return None if ea < end else 0                       # censored inside the window


def summarise(sub, name="cohort"):
    n = len(sub)
    print(f"  {name:<12s} n={n:>5,}  AD {int(sub.ever_ad.sum()):>4,} "
          f"({100 * sub.ever_ad.mean():.1f}%)  died {int((sub.died == 1).sum()):>4,} "
          f"({100 * (sub.died == 1).mean():.1f}%)  "
          f"median fu {sub.followup_y.median():.0f}y")
