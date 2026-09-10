"""
baselines.py -- the bar the sequence model has to clear.

    python -m eval.baselines --split trainval
    python -m eval.baselines --split val --cohort all

Four models, no transformer anywhere in this file. If the sequence model does not beat B3 the
honest deliverable is a logistic regression, and this is the file that decides that, so it is
written to be believed rather than to flatter: the same cohort, the same competing-risk labels
and -- for B3/B4 -- literally the same landmark rows eval/evaluate.py scores the model on
(eval/cohort.py, imported, never re-derived), subject-level CV, and CIs that resample SUBJECTS.

  B1  statics only.  The 21 static tokens (sex, APOE, education, study, smoking, alcohol) plus
      age_bl. A sequence model that does not clear B1 is not using the sequence at all.
  B2  the model's own baseline prefix. An indicator for every token at or before age_bl, read
      out of the .bin rather than re-derived from the raw files, so the feature set cannot
      drift from what the model is actually fed. Prefixes run 7-16 tokens (median 11) and
      contain zero AD-diagnosis tokens, checked. B2 is also the leakage ceiling that
      eval/leakage.py Test 1 compares a model against, hence baseline_prefix_auc() below.
  B3  THE REAL BAR: concurrent cognition at a landmark visit. Current MMSE bin + current
      cogn_global bin (the tokenised state, last bin emitted at or before the landmark) + age
      at the landmark + sex + education, predicting AD within H years.
  B4  the same with UN-BINNED cts_estmmse30 and cogn_global from the longitudinal file. The
      gap B4 - B3 is the information cost of binning, and it is the ceiling the sequence model
      should approach rather than exceed: it sees the bins, not the raw scores.

B1 AND B2 ARE SCORED AGAINST TWO DIFFERENT TARGETS, and the difference between them is the
competing-risk argument in miniature:

  ever-AD and death -- STARRED in the table -- are LIFETIME flags. They do NOT pass through
  cohort.label_at_horizon, so on the default cohort 1,110 of the 2,111 ever-AD negatives
  (52.6%) are subjects who died AD-free and are scored here as clean negatives, which is the
  exact error the rest of this file exists to avoid. They are kept for one reason:
  eval/leakage.py's Test-1 ceiling is defined against this coding and reads it back out of
  this artifact by target name, so the two must not drift. Do not compare a horizon metric
  against them, and do not read them as "what the baseline prefix knows about AD".
  AD<=Hy@bl is the SAME feature matrix against cohort.label_at_horizon at the fu_year-0 visit
  -- same estimand as B3/B4, one row per subject -- and it is the row a horizon metric is
  comparable to. The correction is worth 0.15 AUC: B2 is 0.7226 as a lifetime flag and 0.8786
  at H=5 from identical features (B1 0.6923 -> 0.7681), because "died AD-free" stops counting
  as evidence that the prefix looked healthy.

WHAT THE NUMBERS REST ON, all measured on this cohort and all of it load-bearing here:

  * COMPETING RISK. 62.0% of the cohort died, 53.3% of non-converters died. A subject who dies
    inside the horizon without a diagnosis is DROPPED, not scored as a negative -- scoring
    those as clean negatives is what makes a lethal comorbidity look protective. Every AD<=Hy
    row here (@bl, B3, B4) is labelled by cohort.label_at_horizon; the two starred lifetime
    rows are the documented exception. The drop is large: at H=5, lead 0, 2,613 of 13,625
    legal landmarks (19.2%) are competing deaths and a further 2,157 (15.8%) are censored.
  * THE ROWS ARE THE MODEL'S ROWS. B3/B4 land on the stream event ages with
    min_prior_visits=2 and drop any landmark whose own prefix already contains an endpoint --
    eval/evaluate.py:181-191, the three lines that build what the model is scored at. This
    file used to land on the Excel visit grid at min_prior_visits=1 instead: 24,538 rows, of
    which 10,895 (44.4%) sit at visits where the tokenizer emitted NO token and the model
    therefore has no landmark at all. That bar was not the model's bar. Checked end to end on
    val: 1,898 legal landmarks, of which the endpoint filter removes 61 + 103 -- the exact
    counts eval/evaluate.py:186-189 reports for the model -- leaving 1,734, and this file
    scores 1,731 of them, the 3 lost being landmarks at which neither cognitive scale had
    been measured yet. See landmark_rows.
  * THE POSITIVE COUNT IS THE INTERPRETATION. The headline 1,051 converters is not what a
    5-year AUC rests on, so every AUC below is printed with its evaluable n and its positive
    count and neither is optional.
  * LEAD TIME CHANGES THE ANSWER BY ~0.05 AUC on these rows, so it is a reported axis and not
    a default, and the model's headline lead is one of the defaults. 40.6% of converters have
    their last legal landmark within 3 months of the diagnosis (median lead 0.51 y), because
    RADC records the diagnosis age at a visit. At min_lead 0 a "5-year" AUC is therefore
    partly a concurrent-diagnosis AUC. 0 y, 1 y and 2 y are all run by default; 2 y is
    eval/evaluate.py's headline arm (--min-lead 2.0), so B3 = 0.8588 and B4 = 0.8641 are the
    only two numbers in this file that a headline model AUC may be compared with directly.
  * THE AGE GRID IS APPROXIMATE. Visit ages are age_bl + 365*fu_year; residual error against
    the true visit age is ~21 days median, ~8% of last visits slip past 6 months. Every
    horizon boundary here is that accurate and no more. It does not move an AUC materially --
    the labels are years-wide -- but it is the ceiling on any timing claim made from them.
  * TIMING IS A PROTOCOL CALENDAR, so nothing in this file predicts WHEN, only WITHIN H.

THE DESIGN-PASS REFERENCE VALUES, and what actually reproduced (REFERENCE below, checked on
every run against TOLERANCE = 0.005). All four references are reproducible, but not all of
them under this script's defaults, and the two ways they miss are both informative:

  * B1/B2 reproduce to <= 0.0013 against the WHOLE 4,428-subject cohort with no cohort filter
    (--cohort all: 0.6852 / 0.8770 / 0.7204 / 0.8843 on train+val). Under the default
    evaluation cohort, B2 ever-AD still lands on 0.7226 against 0.7207, but both death numbers
    drop ~0.006-0.009 (0.8697 and 0.8791), because dropping the baseline-impaired removes a
    small, sick, high-mortality group and with it some of the easiest death calls. The
    evaluation cohort is the default anyway: it is the population the sequence model is scored
    on, and a bar measured on a different population is not a bar.
  * B3/B4 are LEAD-TIME references, and on the model's rows they are the ~0.25 y row. Sweeping
    min_lead on the evaluation cohort gives 5-year (B3, B4) = (0.9061, 0.9114) at 0 y,
    (0.8920, 0.8981) at 0.25 y, (0.8886, 0.8948) at 0.5 y, (0.8863, 0.8924) at 0.75 y,
    (0.8803, 0.8858) at 1 y and (0.8588, 0.8641) at 2 y. Against the design pair
    (0.894, 0.9008) the 0.25 y row is -0.0020 / -0.0027, both inside tolerance: the design
    pass excluded landmarks sitting within a few months of the diagnosis. It is not a default
    here because no arm of eval/evaluate.py runs a 0.25 y lead -- run --min-lead 0.25 to
    re-check the references. The B4 - B3 gap, which is the quantity B4 exists to measure,
    is +0.0053 to +0.0062 at every lead against +0.0068 in the references, so it reproduces
    where the levels do not.
  * B3's reference is 0.894 and not the 0.8885 midpoint this file used to carry. tokenizer.py
    line 57 quotes "0.894 vs 0.883" for hysteresis against plain keep-transitions; those are
    two TOKENIZATIONS of the same quantity, not a measurement and its error bar, and the
    delivered build applies hysteresis, so 0.883 describes a data directory that does not
    exist here. The gap behind the old midpoint does not re-measure under this protocol
    either: rebuilding the plain state on these same rows (np.digitize on the tokenizer's own
    SCALE_SPEC edges with zero margin, which moves 5.0-5.2% of MMSE bins and 7.5-7.9% of
    cogn_global bins) gives hysteresis minus plain = +0.0008 at lead 0, +0.0010 at 1 y and
    +0.0011 at 2 y -- an order of magnitude below the +0.011 of the ablation. Treat 0.894 as
    a level to check against, not as evidence about binning.

The references are a design pass, not ground truth; what is stated above is the protocol under
which each is re-derivable.

SPLIT HYGIENE. --split trainval is the development protocol: 5-fold subject-level CV over the
pooled train+val subjects, repeated over `--seeds` shuffles. --split val and --split test
instead FIT on the earlier splits and score the named split exactly once, which is how the
frozen test number is produced. Nothing in this file is tuned, so there is nothing to tune on
a held-out split.

THE CI IS AN INTERVAL FOR THE NUMBER BESIDE IT. In CV mode the AUC is scored on the AVERAGE of
the `--seeds` out-of-fold probability vectors and the subject bootstrap resamples that same
vector, so the two are the same estimator; +/- is the seed-to-seed sd of the per-seed AUCs, a
fold-shuffle stability figure and not a sampling error. Reporting a seed-averaged AUC beside a
seed-0 interval, which this file used to do, gives an interval for a quantity printed nowhere,
and one that does not move at all when --seeds does (measured on B1 ever-AD: identical to 4 dp
at 1, 10 and 30 seeds).
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from radc_delphi import vocab as V                                   # noqa: E402
from radc_delphi.batching import get_p2i, patient_stream             # noqa: E402
from eval import cohort                                              # noqa: E402

DAYS_PER_YEAR = cohort.DAYS_PER_YEAR
HORIZONS = (1, 3, 5)
COHORT_FLAG = {"all": None, "training": "in_training", "evaluation": "in_eval"}
# Which splits a held-out score is allowed to fit on. trainval is CV instead of a fit/score.
FIT_POOL = {"val": ("train",), "test": ("train", "val")}

# Measured during the design pass. Population and lead time are documented in the module
# docstring; the deltas are printed rather than asserted, because the pass used a different
# population from the one this script defaults to. TOLERANCE is 0.005 -- twice the seed-to-seed
# sd of the B1/B2 numbers, so it separates "the same measurement" from "a different cohort".
TOLERANCE = 0.005
REFERENCE = {
    ("B1", "ever-AD"): 0.6858, ("B1", "death"): 0.8783,
    ("B2", "ever-AD"): 0.7207, ("B2", "death"): 0.8853,
    # tokenizer.py:57, the HYSTERESIS arm of "0.894 vs 0.883": the delivered build applies
    # hysteresis, so 0.894 is the applicable number and 0.883 is a tokenization this data
    # directory does not contain. See the docstring: the +0.011 gap does not re-measure here.
    ("B3", "AD<=5y"): 0.894,
    ("B4", "AD<=5y"): 0.9008,
}


def _model():
    """One specification for all four baselines: standardised features, L2 logistic at C=1.

    Nothing is tuned. A baseline whose hyperparameters were searched is no longer a floor, and
    the point of these four is that they are cheap and fixed.
    """
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


# ---------------------------------------------------------------- data
def read_streams(data_dir, splits):
    """projid -> (ages_days, MODEL tokens), straight out of the .bin files."""
    out = {}
    for sp in splits:
        arr = np.fromfile(os.path.join(data_dir, f"{sp}.bin"), dtype=np.uint32).reshape(-1, 3)
        p2i = get_p2i(arr)
        for k in range(len(p2i)):
            s, n = int(p2i[k, 0]), int(p2i[k, 1])
            out[int(arr[s, 0])] = patient_stream(arr, s, n)
    return out


def baseline_day(age_bl):
    """The cut that defines "at or before age_bl", in stream days.

    round(age_bl * 365.25) is the tokenizer's own baseline anchor: the statics sit at cut - 1
    and the fu_year 0 visit at cut exactly, so this cut is the model's baseline prefix and not
    an approximation of it.
    """
    return int(round(float(age_bl) * DAYS_PER_YEAR))


def static_features(pids, streams, sub):
    """B1: the 21 static one-hots plus age_bl."""
    X = np.zeros((len(pids), len(V.STATIC_IDS) + 1))
    for i, pid in enumerate(pids):
        _, toks = streams[pid]
        X[i, :-1] = np.isin(V.STATIC_IDS, toks)
        X[i, -1] = sub.at[pid, "age_bl"]
    cols = [V.NAMES[t] for t in V.STATIC_IDS] + ["age_bl"]
    return pd.DataFrame(X, index=pd.Index(pids, name="projid"), columns=cols)


def prefix_features(pids, streams, sub):
    """B2: an indicator for every token emitted at or before age_bl, plus age_bl.

    Exported: eval/leakage.py builds its Test-1 ceiling from exactly this matrix, so that the
    "what a linear model can read off the baseline prefix" comparison uses the model's inputs
    and not a hand-rolled copy of them. age_bl is kept so B2 strictly contains B1 and the
    ceiling is a real ceiling (without it B2 loses 0.035 AUC on death, which is age).
    """
    ind = np.zeros((len(pids), V.VOCAB_SIZE))
    age = np.zeros(len(pids))
    for i, pid in enumerate(pids):
        ages, toks = streams[pid]
        cut = baseline_day(sub.at[pid, "age_bl"])
        ind[i, toks[ages <= cut]] = 1.0
        age[i] = sub.at[pid, "age_bl"]
    keep = np.flatnonzero(ind.any(0))                  # tokens nobody has carry no information
    df = pd.DataFrame(ind[:, keep], index=pd.Index(pids, name="projid"),
                      columns=[V.NAMES[t] for t in keep])
    df["age_bl"] = age
    assert V.NAMES[V.AD_DX] not in df.columns, "an AD diagnosis sits inside a baseline prefix"
    return df


def landmark_rows(pids, streams, sub, radc_dir, min_lead_years, horizons=HORIZONS):
    """One row per (subject, legal landmark) with B3's bins, B4's raw scores and labels.

    The rows are the MODEL's rows, and that is the point of the function. Candidate ages are
    the STREAM event ages fed to cohort.landmarks with min_prior_visits=2, then filtered for
    an endpoint in their own prefix: the three lines eval/evaluate.py:181-191 uses to build
    the landmarks the model is scored at. Measured on train+val, evaluation cohort, lead 0:
    that is 13,643 rows, a strict SUBSET of the 24,538 this function used to build from the
    Excel visit grid at min_prior_visits=1. The 10,895 rows it drops (44.4%) are visits at
    which the tokenizer emitted nothing -- no scale, history or medication changed, so the
    stream has no token there and the model has no landmark there -- and the 1,459 off-grid
    diagnosis / death ages the stream does carry are exactly the ones the endpoint filter
    removes, so every surviving landmark is a real visit after all. The bar moves with the row
    set: 5-year B3 was 0.9068 on the old rows and is 0.9066 here at lead 0, but the two sets
    disagree by up to 0.007 at longer leads, and a bar measured on rows the model is not
    scored on is not a bar.

    B3's bin is the last MMSE / cogn_global token at or before the landmark, i.e. the state the
    model itself carries. B4's raw value is the last value observed at the last VISIT at or
    before the landmark. Both are last-observation-carried-forward, which keeps the two on
    identical rows.
    """
    lg = pd.read_excel(os.path.join(radc_dir, "longitudinal_data_gk.xlsx"),
                       usecols=["projid", "fu_year", "cts_estmmse30", "cogn_global"])
    lg = lg[lg.projid.isin(pids)].sort_values(["projid", "fu_year"])
    mmse_ids, cog_ids = list(V.SCALES["MMSE"]), list(V.SCALES["COG"])
    rows, no_state = [], []

    for pid, g in lg.groupby("projid", sort=True):
        r = sub.loc[pid]
        ages, toks = streams[pid]
        visits = baseline_day(r.age_bl) + 365.0 * g.fu_year.to_numpy(float)
        lms = cohort.landmarks(ages, toks, r, min_lead_years=min_lead_years,
                               min_prior_visits=2)
        # A landmark whose own prefix already contains the endpoint is not a landmark, and
        # eval/evaluate.py:191 drops those rows before the model sees them: cohort.landmarks
        # cuts inclusively on the un-rounded event age while the tokenizer places the AD token
        # at round(age_ad*365.25), so a converter's own diagnosis day survives the cut for
        # about half of them and a decedent's death day always does. Leaving them in hands the
        # bar 1,459 rows (9.7% at lead 0) whose label sits inside their own feature window.
        lms = [lm for lm in lms if not np.isin(toks[ages <= lm], V.ENDPOINT_IDS).any()]
        if not len(lms):
            continue
        mmse = pd.to_numeric(g.cts_estmmse30, errors="coerce").to_numpy(float)
        cogn = pd.to_numeric(g.cogn_global, errors="coerce").to_numpy(float)
        # stream is age-sorted, so the state at a landmark is one searchsorted per scale
        m_at, m_tok = ages[np.isin(toks, mmse_ids)], toks[np.isin(toks, mmse_ids)]
        c_at, c_tok = ages[np.isin(toks, cog_ids)], toks[np.isin(toks, cog_ids)]
        for lm in lms:
            i = int(np.searchsorted(visits, lm, side="right")) - 1   # last visit <= lm
            im = int(np.searchsorted(m_at, lm, side="right")) - 1
            ic = int(np.searchsorted(c_at, lm, side="right")) - 1
            seen_m = mmse[:i + 1][~np.isnan(mmse[:i + 1])]
            seen_c = cogn[:i + 1][~np.isnan(cogn[:i + 1])]
            if im < 0 or ic < 0 or not len(seen_m) or not len(seen_c):
                no_state.append(pid)          # neither scale ever measured by this landmark
                continue
            rows.append(dict(
                projid=pid, landmark_day=lm, age=lm / DAYS_PER_YEAR,
                mmse_bin=mmse_ids.index(int(m_tok[im])),
                cog_bin=cog_ids.index(int(c_tok[ic])),
                mmse_raw=seen_m[-1], cog_raw=seen_c[-1],
                msex=float(r.msex), educ=float(r.educ),
                event_type=int(r.event_type), event_age=float(r.event_age),
                **{f"y{h}": cohort.label_at_horizon(r, lm, h) for h in horizons}))
    return pd.DataFrame(rows), np.asarray(no_state, dtype=np.int64)


def _onehot(v, k):
    M = np.zeros((len(v), k))
    M[np.arange(len(v)), np.asarray(v, int)] = 1.0
    return M


def landmark_features(df, kind):
    """B3 ('bin') or B4 ('raw') design matrix over the same landmark rows."""
    common = df[["age", "msex", "educ"]].to_numpy(float)
    if kind == "bin":
        return np.hstack([_onehot(df.mmse_bin, len(V.SCALES["MMSE"])),
                          _onehot(df.cog_bin, len(V.SCALES["COG"])), common])
    return np.hstack([df[["mmse_raw", "cog_raw"]].to_numpy(float), common])


# ---------------------------------------------------------------- scoring
def cv_auc(X, y, groups, seeds, folds=5):
    """Subject-level K-fold, repeated. Returns (AUC of the seed-averaged out-of-fold p, the
    seed-to-seed sd of the per-seed AUCs, that averaged p).

    Folds are over SUBJECTS. A landmark-row split would put the same subject's visits on both
    sides and the AUC would be a memorisation score.

    The AUC is scored on the AVERAGE of the `seeds` out-of-fold probability vectors rather
    than averaged over the per-seed AUCs, so that the bootstrap CI -- which is built from that
    same vector -- is an interval for the number it is printed beside. Reporting a 10-seed
    mean next to a seed-0 interval pairs a point estimate with an interval for a different
    quantity. `sd` is the spread of the per-seed AUCs: a fold-shuffle stability figure, not a
    sampling error, which is what the CI is for.
    """
    u = np.unique(groups)
    aucs, acc = [], np.zeros(len(y))
    for s in range(seeds):
        oof = np.zeros(len(y))
        for _, te in KFold(folds, shuffle=True, random_state=s).split(u):
            m = np.isin(groups, u[te])
            oof[m] = _model().fit(X[~m], y[~m]).predict_proba(X[m])[:, 1]
        aucs.append(roc_auc_score(y, oof))
        acc += oof
    p = acc / seeds
    return float(roc_auc_score(y, p)), float(np.std(aucs)), p


def holdout_auc(Xf, yf, Xs, ys):
    p = _model().fit(Xf, yf).predict_proba(Xs)[:, 1]
    return float(roc_auc_score(ys, p)), p


def bootstrap_ci(y, p, groups, n_boot=1000, seed=0):
    """Percentile CI resampling SUBJECTS with replacement, carrying all of a subject's rows.

    A subject contributes up to ~20 landmarks; resampling rows treats those as independent and
    understates the interval -- measured on B3 at H=5, a row-level bootstrap gives a 0.0104
    wide 95% interval against 0.0166 clustered, 1.6x too narrow.
    """
    rng = np.random.default_rng(seed)
    u, inv = np.unique(groups, return_inverse=True)
    by_sub = [np.flatnonzero(inv == i) for i in range(len(u))]
    draws = []
    for _ in range(n_boot):
        rows = np.concatenate([by_sub[i] for i in rng.integers(0, len(u), len(u))])
        if y[rows].min() == y[rows].max():
            continue
        draws.append(roc_auc_score(y[rows], p[rows]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def _score(name, label, X, y, groups, mode, fit=None, n_boot=1000, seeds=10):
    """One row of the report: CV over the pool, or a single fit-then-score on a held-out split."""
    if mode == "cv":
        auc, sd, p = cv_auc(X, y, groups, seeds)
    else:
        auc, p = holdout_auc(fit[0], fit[1], X, y)
        sd = float("nan")
    lo, hi = bootstrap_ci(y, p, groups, n_boot=n_boot)
    return dict(baseline=name, target=label, n=int(len(y)), n_pos=int(y.sum()),
                n_subjects=int(len(np.unique(groups))), auc=auc, sd=sd, ci=[lo, hi],
                reference=REFERENCE.get((name, label)))


# ---------------------------------------------------------------- the exported ceiling
def baseline_prefix_auc(data_dir, radc_dir, outcome="ad", splits=("train", "val"),
                        cohort_name="evaluation", seeds=10, n_boot=1000):
    """B2 as a function: the leakage ceiling eval/leakage.py Test 1 measures a model against.

    Returns {auc, sd, ci, n, n_pos, pred}, where `pred` is the out-of-fold probability per
    subject (a Series on projid) so a caller can compare model and ceiling on the same rows
    rather than on two AUCs computed over different subjects.
    """
    sub = cohort.load_subjects(data_dir, radc_dir)
    sub = sub[sub.split.isin(splits)]
    flag = COHORT_FLAG[cohort_name]
    if flag:
        sub = sub[sub[flag]]
    pids = list(sub.index)
    X = prefix_features(pids, read_streams(data_dir, splits), sub)
    y = (sub.ever_ad if outcome == "ad" else sub.died == 1).to_numpy(int)
    g = np.asarray(pids)
    auc, sd, p = cv_auc(X.to_numpy(float), y, g, seeds)
    lo, hi = bootstrap_ci(y, p, g, n_boot=n_boot)
    return dict(auc=auc, sd=sd, ci=[lo, hi], n=int(len(y)), n_pos=int(y.sum()),
                pred=pd.Series(p, index=X.index, name="b2_prefix_p"))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--data-dir", default="data/radc-s42")
    ap.add_argument("--radc-dir", default="data/RADC")
    ap.add_argument("--split", default="trainval", choices=["trainval", "val", "test"],
                    help="trainval: CV over pooled train+val. val/test: fit on the earlier "
                         "splits and score this one once.")
    ap.add_argument("--cohort", default="evaluation", choices=list(COHORT_FLAG))
    ap.add_argument("--horizons", type=int, nargs="+", default=list(HORIZONS))
    ap.add_argument("--min-lead", type=float, nargs="+", default=[0.0, 1.0, 2.0],
                    help="required years between a landmark and the diagnosis it predicts. "
                         "2.0 is eval/evaluate.py's headline arm and is not optional here: it "
                         "is the only row directly comparable to the model's headline AUC.")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    mode = "cv" if args.split == "trainval" else "holdout"
    score_splits = ("train", "val") if args.split == "trainval" else (args.split,)
    fit_splits = () if mode == "cv" else FIT_POOL[args.split]
    all_splits = tuple(dict.fromkeys(fit_splits + score_splits))

    sub = cohort.load_subjects(args.data_dir, args.radc_dir)
    sub = sub[sub.split.isin(all_splits)]
    flag = COHORT_FLAG[args.cohort]
    if flag:
        sub = sub[sub[flag]]
    streams = read_streams(args.data_dir, all_splits)
    s_pids = np.array(sub.index[sub.split.isin(score_splits)])
    f_pids = np.array(sub.index[sub.split.isin(fit_splits)])

    print(f"\n  baselines | split={args.split} ({mode}) cohort={args.cohort} "
          f"| scored {len(s_pids):,} subjects"
          + (f", fitted on {len(f_pids):,}" if mode == "holdout" else
             f" | {args.seeds} seeds x 5-fold subject CV"))
    print(f"  age grid is age_bl + 365*fu_year, ~21 d median error vs the true visit age -- "
          f"every horizon boundary is that accurate and no more")

    results = []
    pool_pids = np.concatenate([f_pids, s_pids]) if mode == "holdout" else s_pids

    # ---- B1 / B2: one row per subject ---------------------------------------------------
    # Two estimands, and the difference between them is the whole competing-risk argument.
    # ever-AD / death are LIFETIME flags: they do not pass through label_at_horizon, so a
    # subject who died AD-free is a clean negative for ever-AD. They are kept because
    # eval/leakage.py's Test-1 ceiling is defined against exactly this coding (and read out of
    # this artifact by target name), and they are marked with a * everywhere they are printed.
    # AD<=Hy@bl is the same features against the cause-specific label at the fu_year-0 visit.
    bl_day = {p: baseline_day(sub.at[p, "age_bl"]) for p in pool_pids}
    ybl = pd.DataFrame({h: [cohort.label_at_horizon(sub.loc[p], bl_day[p], h) for p in pool_pids]
                        for h in args.horizons}, index=pd.Index(pool_pids, name="projid"))
    for name, build in (("B1", static_features), ("B2", prefix_features)):
        F = build(list(pool_pids), streams, sub)
        for label, y_all in (("ever-AD", sub.ever_ad.astype(int)),
                             ("death", (sub.died == 1).astype(int))):
            Xs = F.loc[s_pids].to_numpy(float)
            ys = y_all.loc[s_pids].to_numpy(int)
            fit = None if mode == "cv" else (F.loc[f_pids].to_numpy(float),
                                             y_all.loc[f_pids].to_numpy(int))
            results.append(_score(name, label, Xs, ys, s_pids, mode, fit,
                                  n_boot=args.n_boot, seeds=args.seeds))
        for h in args.horizons:
            yh = ybl[h]
            s_ok = s_pids[yh.loc[s_pids].notna().to_numpy()]   # unscoreable subjects dropped
            f_ok = f_pids[yh.loc[f_pids].notna().to_numpy()]
            fit = None if mode == "cv" else (F.loc[f_ok].to_numpy(float),
                                             yh.loc[f_ok].to_numpy(int))
            results.append(_score(name, f"AD<={h}y@bl", F.loc[s_ok].to_numpy(float),
                                  yh.loc[s_ok].to_numpy(int), s_ok, mode, fit,
                                  n_boot=args.n_boot, seeds=args.seeds))

    # ---- B3 / B4: one row per landmark, outcome is AD within H -------------------------
    print(f"\n  competing risk: on every AD<=Hy row (@bl, B3, B4) a death inside the horizon "
          f"without a diagnosis is DROPPED, never scored as a negative")
    for lead in args.min_lead:
        df, no_state = landmark_rows(list(pool_pids), streams, sub, args.radc_dir, lead,
                                     tuple(args.horizons))
        s_mask = df.projid.isin(s_pids).to_numpy()
        print(f"\n  min_lead {lead:g} y: {int(s_mask.sum()):,} landmark rows over "
              f"{df.projid[s_mask].nunique():,} scored subjects "
              f"({int(np.isin(no_state, s_pids).sum())} dropped: no cognition measured by the "
              f"landmark)")
        for name, kind in (("B3", "bin"), ("B4", "raw")):
            X = landmark_features(df, kind)
            for h in args.horizons:
                # A lead of L years makes "AD within H <= L years" empty by construction: the
                # landmark was required to sit at least L years before the diagnosis.
                if lead >= h:
                    if name == "B3":
                        print(f"    H={h}: skipped, a {lead:g} y lead leaves no positives")
                    continue
                ok = df[f"y{h}"].notna().to_numpy()
                sel = ok & s_mask
                y = df.loc[sel, f"y{h}"].to_numpy(int)
                fit = None if mode == "cv" else (X[ok & ~s_mask],
                                                 df.loc[ok & ~s_mask, f"y{h}"].to_numpy(int))
                r = _score(name, f"AD<={h}y", X[sel], y, df.projid.to_numpy()[sel], mode, fit,
                           n_boot=args.n_boot, seeds=args.seeds)
                r["min_lead_years"] = lead
                # why a legal landmark is not scoreable at this horizon: a death inside the
                # window without a diagnosis (the competing event) or follow-up that simply
                # ends first. Both are dropped; only the first is a competing risk.
                lost = s_mask & ~ok
                dead = lost & (df.event_type.to_numpy() == 2)
                r["n_dropped_competing_death"] = int(dead.sum())
                r["n_dropped_censored"] = int((lost & ~dead).sum())
                results.append(r)

    # ---- table --------------------------------------------------------------------------
    print(f"\n  {'':<4}{'target':<10}{'lead':>5}{'n':>8}{'pos':>7}{'subj':>7}  "
          f"{'AUC':>17}  {'95% CI (subject bs)':<22}"
          + (f"{'ref':>8}{'delta':>8}" if mode == "cv" else ""))
    for r in results:
        sd = "" if np.isnan(r["sd"]) else f" +/- {r['sd']:.4f}"
        lead = "" if r.get("min_lead_years") is None else f"{r['min_lead_years']:g}y"
        ref = r["reference"]
        # display-only star: the stored target name is what eval/leakage.py matches on
        tgt = r["target"] + ("*" if r["target"] in ("ever-AD", "death") else "")
        print(f"  {r['baseline']:<4}{tgt:<10}{lead:>5}{r['n']:>8,}{r['n_pos']:>7,}"
              f"{r['n_subjects']:>7,}  {r['auc']:.4f}{sd:<11}  "
              f"[{r['ci'][0]:.4f}, {r['ci'][1]:.4f}]     "
              + (f"{ref:>8.4f}{r['auc'] - ref:>+8.4f}" if ref and mode == "cv" else ""))
    neg = ~sub.loc[s_pids].ever_ad.astype(bool)
    cdeath = neg & (sub.loc[s_pids].event_type == 2)
    print(f"  * LIFETIME flag, NOT a competing-risk label: {int(cdeath.sum()):,} of the "
          f"{int(neg.sum()):,} ever-AD negatives ({100 * cdeath.sum() / neg.sum():.1f}%) died "
          f"AD-free and are scored here as clean negatives. Kept only because "
          f"eval/leakage.py's Test-1 ceiling is defined against this coding; @bl is the same "
          f"features under cohort.label_at_horizon at the fu_year-0 visit, and it is the row "
          f"to compare a horizon AUC against.")

    print(f"\n  dropped landmark rows (not scored, by horizon, at min_lead "
          f"{args.min_lead[0]:g} y)")
    for r in results:
        if "n_dropped_competing_death" in r and r["baseline"] == "B3" \
                and r["min_lead_years"] == args.min_lead[0]:
            tot = r["n"] + r["n_dropped_competing_death"] + r["n_dropped_censored"]
            print(f"    {r['target']:<9} competing death "
                  f"{r['n_dropped_competing_death']:>6,} ({100 * r['n_dropped_competing_death'] / tot:4.1f}%)"
                  f"   censored {r['n_dropped_censored']:>6,} "
                  f"({100 * r['n_dropped_censored'] / tot:4.1f}%)   of {tot:,} legal landmarks")

    # The references are CV-protocol numbers over thousands of subjects; checking them against
    # a few-hundred-subject held-out score would compare a measurement with its own noise.
    if mode == "cv":
        print(f"\n  reference check, tolerance {TOLERANCE} (design-pass values; population "
              f"and lead time differ -- see the module docstring. B3/B4's references are a "
              f"~0.25 y-lead\n  protocol that no arm of eval/evaluate.py runs, so they are "
              f"expected to miss at the leads run here: --min-lead 0.25 reproduces both)")
    for r in results:
        if r["reference"] and mode == "cv":
            d = r["auc"] - r["reference"]
            lead = "" if r.get("min_lead_years") is None else f" lead {r['min_lead_years']:g}y"
            print(f"    {r['baseline']} {r['target']:<9}{lead:<9} {r['auc']:.4f} vs "
                  f"{r['reference']:.4f}  "
                  f"{'REPRODUCED' if abs(d) <= TOLERANCE else 'DOES NOT REPRODUCE'} "
                  f"(delta {d:+.4f})")

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   f"baselines_{args.split}.json")
    with open(out, "w") as fh:
        json.dump(dict(split=args.split, mode=mode, cohort=args.cohort,
                       score_splits=list(score_splits), fit_splits=list(fit_splits),
                       seeds=args.seeds, folds=5, n_boot=args.n_boot,
                       horizons=list(args.horizons), min_lead_years=list(args.min_lead),
                       n_subjects_scored=int(len(s_pids)), results=results), fh, indent=1)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
