"""
leakage.py -- the adversarial audit. Run this on a checkpoint before any of its numbers is
believed, and treat a trip as a claim that the tokenizer leaked until proven otherwise.

    python -m eval.leakage --ckpt out-radc-base-s42/ckpt.pt --split val
    python -m eval.leakage --self-test                       # validate the detector itself
    python -m eval.leakage --ckpt out-radc-canary-s42/ckpt.pt \
        --data-dir data/radc-canary-s42 --split train --canary

The canary is audited on TRAIN and that is deliberate: at the planted 5% only 6 val subjects
carry the oracle against 38 in train, and the canary is a test of the DETECTOR, not of
generalization. The legal refit is cross-fitted by subject there so it is not scored on rows
it was fitted on.

Exits non-zero on FAIL, and on any test that could not be READ (INSUFFICIENT), so it can
gate a pipeline: a test that produced no evidence is not a clean bill. Writes
eval/leakage_<split>.json and eval/leakage_horizon_<split>.png.

EVERY VERDICT IS READ OFF AN INTERVAL, NOT A POINT. The margins between the legal ceilings
and the FAIL triggers are 0.027 and 0.035 AUC, while the model's AUC on the 378 audited val
subjects has SE 0.031 and Test 2's long-lead gap has a 95% interval 0.270 wide -- so a
point-estimate threshold on either fires on a clean checkpoint about one run in five. Test 1
therefore requires the model's advantage over the ceiling to clear zero on a bootstrap paired
over subjects (both scores are measured on the same rows, so the pairing removes the cohort
noise they share), and Test 2 requires the LOWER bound of a cluster bootstrap of the
long-lead gap to clear the trigger. Both are two-sided: a score that ranks cases LAST knows
as much as one that ranks them first.

WHY THIS FILE EXISTS, and what it is looking for. The design pass excluded five things for
MEASURED leakage, and each one was found by hand:

    autopsy block         availability alone is a perfect death oracle -- all 2,231 subjects
                          with any autopsy value have died == 1, and a missingness indicator
                          reaches AUC 0.906 for death
    cogng_demog_slope     AUC 0.818 for ever-AD; it is a summary of the outcome
    ad_rx                 P(AD within 3 y) 4.7% -> 31.2% at a normal MMSE of 27-30
    ROSMAP_clinical       every end-of-study column: cogdx 0.888, dcfdx_lv 0.838,
                          cts_mmse30_lv 0.833
    age_death             kept only as a numeric age for the death token, never as a feature

Those five are out. This file is what catches the sixth -- a column nobody thought of, a
tokenizer change that re-admits one of them, or a subtler ordering bug -- and it does so
without knowing which column to look for, by asking whether the model knows more at baseline
than the legal information at baseline can support.

THE THREE TESTS

  1  BASELINE-PREFIX CEILING. Truncate held-out subjects at age_bl, roll the model forward to
     age 110 with death absorbing, and read off P(AD ever) and P(death ever). A model that
     only reads the tokens can do no better on those than a legal model fitted directly to
     the same baseline information -- eval/baselines.py's B2, imported rather than copied.
     Ceilings as delivered: ever-AD 0.7207, death 0.8853; the triggers are those plus a
     margin. Known signatures, so a trip can be diagnosed instead of merely alarmed at:
     ~0.87 for AD is cogng_demog_slope, ~0.97 for death is autopsy availability.

  2  HORIZON CURVE. For each held-out converter diagnosed at age T, cut the stream at
     T-1, T-3, T-5, T-8, T-12 and record the model's 12-month AD hazard, against matched
     never-converters cut at the same ages. Then subtract a legal logistic refit trained on
     the SAME truncated features, the same task, at the same lead. Genuine antecedent signal
     produces a gap that GROWS as the lead shortens; a static leak sits in every prefix
     equally and produces a FLAT, already-elevated gap.

     The comparator is refitted at each lead for a reason: there is no fixed anchor to use
     instead. In particular "baseline cognition is at chance at 10 years (AUC 0.502)" is a
     follow-up-length selection artifact of the subset it was computed on -- within
     follow-up-length strata the same quantity runs 0.62-0.69 -- so no constant is safe.

     Note the handicap this comparison gives the model, which makes an elevated long-lead gap
     all the more damning: the refit is trained on exactly the task being scored at that lead,
     while the model is asked for a 12-month hazard and only ranks 12-years-out converters as
     a side effect. At long leads an honest model should therefore lose to the refit.

     The age-matched design also leaves follow-up-length selection in the rows -- a case at
     lead 12 must by construction have twelve years of it -- and the refit is handed the
     features that expose it (n_visits, span_y, since_last_y). They are kept because the
     comparator has to see exactly what the model sees, but they are NOT a one-way handicap:
     at the long leads they are fitted on fewer than a hundred rows and they do not transfer
     (n_visits alone is 0.607 on train and 0.451 on val at lead 12), so at lead 12 they cost
     the refit 0.071 AUC on val while at lead 8 they gain it 0.013. That is worth ~0.03 on
     gap_long, in EITHER direction -- it is instability, not conservatism, and the interval
     the verdict reads is what covers it.

  3  THE CANARY. A null result from an untested detector is worth nothing. --canary audits a
     build in which a deterministic 5% of subjects carry a planted AD oracle
     (radc_delphi.tokenizer, id 50, appended past the standard table and refused by
     build_dataset in the production directory), and requires Tests 1 and 2 to fire loudly on
     the marked subjects and stay quiet on the rest. The marked group is matched WITHIN
     itself, and its Test-2 curve has to come back FAIL: a marked curve that could not be
     read is a canary that was never evaluated, which is the one thing the validation of a
     detector cannot pass on. --self-test drives the same verdict functions with synthesised
     scores and needs no checkpoint, so the detector's sensitivity is checked on every run
     rather than once.

WHAT THIS AUDIT CANNOT SEE. It reads the model through radc_delphi.engine and the observed
outcomes through eval.cohort, so it tests the pairing of the two. A leak that is present in
the raw file, tokenized, AND equally present in the legal baseline's feature matrix would
raise no gap here -- baselines.py's own feature list is the other half of the guarantee.
Competing risk is respected throughout: a simulated AD diagnosis only counts if it precedes
that trajectory's death, and a subject who dies inside a horizon is dropped rather than scored
as a clean negative (eval.cohort.label_at_horizon).
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from radc_delphi import vocab as V                                          # noqa: E402
from radc_delphi.engine import load as load_engine, Engine, DAYS_PER_YEAR   # noqa: E402
from radc_delphi.tokenizer import CANARY_NAME, enable_canary_vocab          # noqa: E402
from eval import cohort                                                     # noqa: E402

# ---------------------------------------------------------------- triggers
# Each is the imported legal ceiling plus a margin. The margin is a MINIMUM effect size, not
# a significance test, and it is deliberately small: B2's own AUCs are point estimates on
# ~3.7k subjects (bootstrap SE 0.015-0.02), while the quantity being compared against them is
# the model's AUC on the few hundred audited subjects, whose SE is 0.031 on the 378 val
# subjects -- the ever-AD margin (0.76 - 0.7328 = 0.027) is 0.9 of THAT, not 2 of it. So
# clearing a trigger is necessary for a FAIL and never sufficient: test1_verdict also requires
# the exceedance to survive a bootstrap (paired against B2's per-subject prediction when it is
# available), because a leak-free model sitting exactly on the ceiling clears the ever-AD
# trigger on 17% of runs and a gate that fires one time in five on a clean checkpoint is not
# a gate.
TRIGGER_EVER_AD = 0.76
TRIGGER_DEATH = 0.92

# Fingerprints of the leaks that were actually found, so a trip names its suspect instead of
# only raising an alarm. (endpoint, low, high, what it looks like)
SIGNATURES = [
    ("ever_ad", 0.84, 0.91, "cogng_demog_slope or another outcome-derived summary "
                            "(measured 0.818 for ever-AD, 0.775 in the hardest subset)"),
    ("ever_ad", 0.91, 1.01, "a direct copy of the label -- ad_rx, cogdx, dcfdx_lv or "
                            "age_first_ad_dx itself"),
    ("death", 0.94, 1.01, "autopsy-block availability (a missingness indicator alone reaches "
                          "0.906; all 2,231 subjects with any autopsy value have died)"),
]

LEADS = (1.0, 3.0, 5.0, 8.0, 12.0)
T2_GAP_TRIGGER = 0.10       # AUC points over the legal refit at a long lead
T2_FLAT_TRIGGER = 0.05      # growth from the longest lead to 1 y that a real signal must show
T2_MIN_CASES = 20           # below this a lead is reported as underpowered, not as a verdict
CANARY_FIRE_AUC = 0.95      # the planted oracle is perfect; anything less means it was missed
CANARY_MIN_MARKED = 15

PASS, INVESTIGATE, FAIL, SKIP = "PASS", "INVESTIGATE", "FAIL", "INSUFFICIENT"
SCORED = "scored"           # a horizon lead that carried enough cases to be read

# The roll-up order, and the whole point of it is where SKIP sits: a test that could not be
# read produced no evidence, and no evidence is not a clean bill. It outranks PASS everywhere.
_ORDER = {PASS: 0, SKIP: 1, INVESTIGATE: 2, FAIL: 3}


def worst(verdicts):
    """The roll-up: the most serious verdict in the list, with INSUFFICIENT above PASS."""
    return max(verdicts, key=lambda v: _ORDER[v])


# ---------------------------------------------------------------- small numerics
def auc(y, score):
    """AUC, or NaN when the outcome does not vary (which is a sample-size fact, not a result)."""
    y = np.asarray(y, int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(score, float)))


def auc_ci(y, score, n_boot=1000, seed=0):
    """Percentile bootstrap over SUBJECTS. Quoted with every AUC here because the val split is
    443 subjects and a 0.03 difference against a ceiling is inside the noise at that size."""
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        out.append(roc_auc_score(y[i], score[i]))
    if not out:
        return (float("nan"), float("nan"))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def _auc_fast(y, score):
    """AUC by rank sum, NaN when the outcome does not vary. Identical to auc() to 1e-12 and
    called ~15,000 times by Test 2's bootstrap, where roc_auc_score's input validation is the
    cost."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def paired_auc_delta_ci(y, score, reference, n_boot=1000, seed=0):
    """Percentile bootstrap of auc(y, score) - auc(y, reference) over SUBJECTS, both scores
    resampled on the SAME draw.

    Test 1's decision is a comparison of two AUCs measured on one set of subjects, and the
    ceiling is as noisy as the model: B2's live AUC is fitted out-of-fold and read off the
    same 378 rows. Pairing cancels the shared subject draw, so this interval is the one the
    verdict can legitimately threshold; the model's own CI cannot, because it carries the
    cohort noise that both sides share.
    """
    y = np.asarray(y, int)
    a = np.asarray(score, float)
    b = np.asarray(reference, float)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        yi = y[i]
        if yi.min() == yi.max():
            continue
        out.append(_auc_fast(yi, a[i]) - _auc_fast(yi, b[i]))
    if not out:
        return (float("nan"), float("nan"), float("nan"))
    return (float(_auc_fast(y, a) - _auc_fast(y, b)),
            float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def legal_ceilings(data_dir, radc_dir, split, pids=None, live=True, artifact=None,
                   seeds=10):
    """The B2 ceilings, IMPORTED from eval/baselines.py. Never retyped here.

    Two of them, and both are used:

      reference  baselines.REFERENCE[("B2", ...)] -- the numbers the design pass measured
                 (ever-AD 0.7207, death 0.8853). These set the fixed triggers, so a trigger
                 cannot drift with a re-fit.
      live       baselines.baseline_prefix_auc() re-fitted for THIS data directory and scored
                 on EXACTLY the subjects this audit rolled out. Its own docstring offers the
                 out-of-fold probability per subject for precisely this reason: comparing two
                 AUCs computed over different subject sets is how a 0.03 difference gets
                 attributed to the model when it belongs to the cohort. (Measured on val:
                 B2 ever-AD is 0.7412 there against the 0.7207 reference.)

    The effective ceiling is the larger of the two, which is the conservative choice -- the
    audit should only fire when the model beats the best legal account of the same prefix.

    In canary mode `data_dir` must be the PRODUCTION build: baselines.prefix_features
    indicators every token at or before age_bl, so run on a canary build it would take the
    planted oracle as a legal feature and raise its own ceiling to meet the leak.

    `artifact` is the cached-AUC route, and it CANNOT honour `pids`: baselines.py's report
    carries B2's AUC over its own subject set and no per-subject probability, so there is
    nothing to re-score here. Passing both is refused rather than silently ignored -- that
    silence moved the ever-AD INVESTIGATE boundary from 0.7328 to 0.7412, a third of the FAIL
    margin, in exactly the direction this function's live route exists to prevent.
    """
    from eval import baselines
    out = {"source": (f"eval.baselines.REFERENCE + baseline_prefix_auc({data_dir}, "
                      f"splits=('train', '{split}'), cohort='evaluation')" if live
                      else "eval.baselines.REFERENCE")}
    spec = (("ever_ad", ("B2", "ever-AD"), "ad", TRIGGER_EVER_AD),
            ("death", ("B2", "death"), "death", TRIGGER_DEATH))
    for ep, key, _outcome, trig in spec:
        out[ep] = dict(reference=float(baselines.REFERENCE[key]), live=float("nan"),
                       live_n=0, trigger=trig)
    if artifact:
        assert pids is None, (
            f"--ceilings-json holds B2's AUC over ITS subject set, not a per-subject "
            f"prediction, so it cannot be re-scored on the {len(pids)} subjects audited "
            f"here; drop the flag and let the live re-fit do it properly")
        with open(artifact) as fh:
            art = json.load(fh)
        for r in art["results"]:
            for ep, key, _o, _t in spec:
                if (r["baseline"], r["target"]) == key:
                    out[ep]["live"] = float(r["auc"])
                    out[ep]["live_n"] = int(r["n"])
                    out[ep]["live_note"] = (
                        "B2's AUC over the artifact's own subjects, NOT the ones audited "
                        "here; on val that is 0.7412 against 0.7328 re-fitted on the "
                        "audited rows, and the difference is a third of the FAIL margin")
        out["source"] = artifact
    elif live:
        for ep, _key, outcome, _t in spec:
            r = baselines.baseline_prefix_auc(data_dir, radc_dir, outcome=outcome,
                                              splits=("train", split),
                                              cohort_name="evaluation", seeds=seeds)
            out[ep]["live_n"] = int(r["n"])
            if pids is None:
                out[ep]["live"] = float(r["auc"])
                continue
            pred = r["pred"].reindex(pids)
            assert pred.notna().all(), \
                f"B2 has no out-of-fold prediction for {int(pred.isna().sum())} audited " \
                f"subjects -- this audit and eval/baselines.py disagree about the cohort"
            out[ep]["pred"] = pred.to_numpy(float)
            out[ep]["live_n"] = int(len(pred))
    for ep, _k, _o, _t in spec:
        e = out[ep]
        e["effective"] = max(e["reference"], e["live"]) if np.isfinite(e["live"]) \
            else e["reference"]
    return out


# ---------------------------------------------------------------- legal features
# EVERY feature here is read off the truncated token stream itself. That is what makes the
# comparator legal by construction rather than by review: the refit sees exactly the tokens
# the model sees at the same cut, so a gap between them cannot be explained by the refit
# having been handed less information.
#
# The one deliberate omission is the canary id, which is named nowhere in this list. The
# comparator has to stay LEGAL: hand it the planted oracle and the leak cancels out of the
# gap, and Test 2 would report a clean curve on the very build that exists to make it fire.
_FLAG_TOKENS = ["Sex: male", "APOE e4 heterozygote (24/34)", "APOE e4 homozygote (44)",
                "APOE e2 carrier (22/23)", "APOE unknown", "Education <=12y",
                "Education >=17y", "Study: MAP", "Study: LATC", "Smoking: former",
                "Smoking: current", "Alcohol: light", "Alcohol: heavy",
                "Hypertension, history", "Diabetes, history", "Claudication, history",
                "Heart condition, history"]
_COUNT_TOKENS = ["Stroke, probable", "Stroke, possible", "Depression, probable",
                 "Depression, possible", "Antihypertensive started", "Antihypertensive stopped",
                 "Statin started", "Statin stopped"]


def prefix_features(ages, toks, cut_day):
    """The legal feature vector of one truncated prefix, as an ordered dict."""
    a, t = Engine.prefix_at(np.asarray(ages, float), np.asarray(toks, int), cut_day)
    present = set(int(x) for x in t)
    f = {"age": cut_day / DAYS_PER_YEAR}
    for n in _FLAG_TOKENS:
        f[n] = float(V.ID[n] in present)
    for n in _COUNT_TOKENS:
        f[n] = float((t == V.ID[n]).sum())
    for scale, ids in V.SCALES.items():
        lut = {tid: i for i, tid in enumerate(ids)}
        m = np.isin(t, list(ids))
        idxs = [lut[int(x)] for x in t[m]]
        f[f"{scale}_last"] = float(idxs[-1]) if idxs else -1.0
        f[f"{scale}_first"] = float(idxs[0]) if idxs else -1.0
        # low index = worse for MMSE and COG (vocab.SEVERITY_ORDER), so a negative delta is a
        # decline; the sign is the linear model's problem, not this function's
        f[f"{scale}_delta"] = float(idxs[-1] - idxs[0]) if idxs else 0.0
        f[f"{scale}_n"] = float(len(idxs))
    real = a[a > 0]
    f["n_tokens"] = float(len(t))
    f["n_visits"] = float(len(np.unique(real)))
    f["span_y"] = float((real.max() - real.min()) / DAYS_PER_YEAR) if len(real) else 0.0
    f["since_last_y"] = float((cut_day - real.max()) / DAYS_PER_YEAR) if len(real) else 0.0
    return f


def _matrix(rows):
    keys = list(rows[0]["x"].keys())
    return np.array([[r["x"][k] for k in keys] for r in rows], float), keys


# ---------------------------------------------------------------- data plumbing
def load_streams(eng, split, data_dir):
    """{projid: (ages_days, model_tokens)} for one split, plus the cohort frame."""
    data, p2i, _ = eng.load_split(split, data_dir=data_dir)
    streams = {}
    for k in range(len(p2i)):
        ages, toks, pid = Engine.stream(data, p2i, k)
        streams[pid] = (ages, toks)
    return streams


def _prefix_ok(ages, toks, cut_day):
    """A prefix is scoreable only if it holds real history and no endpoint. An AD or death
    token at or before the cut is not a prediction problem, it is the answer."""
    a, t = Engine.prefix_at(ages, toks, cut_day)
    if len(t) < 2 or (a > 0).sum() < 1:
        return False
    return not (V.AD_DX in t or V.DEATH in t)


# ---------------------------------------------------------------- TEST 1
def baseline_prefix_rollout(eng, streams, sub, n_mc, max_new_tokens, seed, horizon_y=10.0,
                            verbose=True):
    """Roll every eval-cohort subject forward from their age_bl prefix.

    Two readouts per endpoint, both straight off the same rollout: P(event ever) and P(event
    within 10 years). The 'ever' probability saturates -- in simulation almost everyone
    eventually dies -- so it can be a blunt score, and an audit that a blunt readout can
    defeat is not an audit. Both are scored and the STRONGER is the one that faces the
    trigger, which is the adversarial choice.
    """
    out = []
    skipped = {"endpoint_in_prefix": 0, "no_history": 0}
    unterminated = 0
    pids = [p for p in streams if p in sub.index and bool(sub.loc[p, "in_eval"])]
    for i, pid in enumerate(pids):
        ages, toks = streams[pid]
        r = sub.loc[pid]
        cut = float(round(float(r["age_bl"]) * DAYS_PER_YEAR))
        a, t = Engine.prefix_at(ages, toks, cut)
        if len(t) < 2 or (a > 0).sum() < 1:
            skipped["no_history"] += 1
            continue
        if V.AD_DX in t or V.DEATH in t:
            skipped["endpoint_in_prefix"] += 1
            continue
        gi, ga = eng.simulate(t, a, n_mc=n_mc, until_age_years=110.0,
                              max_new_tokens=max_new_tokens, seed=seed + i)
        big = 1e18
        ad_at = np.where((gi == V.AD_DX) & (ga > cut), ga, big).min(1)
        dth_at = np.where((gi == V.DEATH) & (ga > cut), ga, big).min(1)
        hz = cut + horizon_y * DAYS_PER_YEAR
        # competing risk, in simulation exactly as in the observed data: a diagnosis only
        # counts if that trajectory reached it before dying
        got_ad = ad_at < dth_at
        # no death token by the end of the rollout -- either the token budget ran out or
        # the trajectory reached the age-110 stop. Both make P(death ever) a lower bound, and
        # generate() masks the ages past the stop, so the two cannot be told apart here.
        unterminated += int((dth_at >= big).sum())
        out.append(dict(
            projid=int(pid), cut_day=cut,
            ever_ad=int(bool(r["ever_ad"])), death=int(r["died"] == 1),
            p_ad_ever=float(got_ad.mean()),
            p_ad_10y=float((got_ad & (ad_at <= hz)).mean()),
            p_death_ever=float((dth_at < big).mean()),
            p_death_10y=float((dth_at <= hz).mean()),
            x=prefix_features(ages, toks, cut)))
        if verbose and (i + 1) % 100 == 0:
            print(f"    ... {i + 1}/{len(pids)} subjects rolled out", flush=True)
    frac_unterm = unterminated / max(1, len(out) * n_mc)
    return out, skipped, frac_unterm


def test1_verdict(rows, ceilings, seed=0, tag="", n_boot=1000):
    """Model against ceiling, one verdict per endpoint.

    TWO-SIDED, because information is two-sided. A score that ranks every converter LAST
    knows exactly as much as one that ranks them first, so each readout is judged on
    max(a, 1 - a) with the side recorded; on the one-sided rule a sign-reversed oracle (AUC
    0.000) reported PASS, and this model is already anti-ranked at long leads in Test 2.

    INVESTIGATE is a point-estimate call against the effective ceiling, the larger of the
    delivered number and B2 re-fitted on these exact subjects. FAIL needs more than a point
    estimate over the fixed trigger, because the trigger's margin is 0.9 of the model AUC's
    own SE at this n (0.031 on the 378 val subjects), not the 2 SE it was written as: the
    exceedance must also survive a bootstrap. Paired against B2's per-subject prediction when
    the ceiling carries one -- both vectors are the same subjects in the same order, so the
    pairing removes the cohort noise the two share -- and otherwise unpaired, requiring the
    model's own 95% lower bound to clear the ceiling. Clearing the fixed trigger stays
    necessary: a leak that also got into B2's feature matrix would raise the live ceiling to
    meet itself, and the delivered reference is the anchor that cannot drift.
    """
    res = {"n": len(rows), "tag": tag, "endpoints": {}, "verdict": PASS}
    nan = float("nan")
    for endpoint, keys in (("ever_ad", ("p_ad_ever", "p_ad_10y")),
                           ("death", ("p_death_ever", "p_death_10y"))):
        y = np.array([r[endpoint] for r in rows], int)
        readouts = {k: auc(y, [r[k] for r in rows]) for k in keys}
        # informativeness, not closeness to 1: 0.15 and 0.85 are the same amount of knowledge
        best = max(readouts, key=lambda k: (-1e9 if np.isnan(readouts[k])
                                            else abs(readouts[k] - 0.5)))
        a_raw = readouts[best]
        finite = bool(np.isfinite(a_raw))
        side = 1.0 if not finite or a_raw >= 0.5 else -1.0
        a_best = max(a_raw, 1.0 - a_raw) if finite else nan
        score = side * np.array([r[best] for r in rows], float)
        c = dict(ceilings[endpoint])
        pred = c.pop("pred", None)
        if pred is not None:                   # B2 scored on exactly these rows
            c["live"] = auc(y, pred)
            c["effective"] = max(c["reference"], c["live"])
        trigger = c["trigger"]
        ci = auc_ci(y, score, seed=seed) if finite else (nan, nan)
        delta = dict(paired=pred is not None, value=nan, ci95=(nan, nan))
        if not finite:
            # a degenerate outcome (every subject positive, or every one negative) is a
            # sample-size fact; the old code let `not (nan > ceiling)` report it as PASS
            v = SKIP
            note = ""
        else:
            if pred is not None:
                d, lo, hi = paired_auc_delta_ci(y, score, pred, n_boot=n_boot, seed=seed)
                delta.update(value=d, ci95=(lo, hi))
                exceeds = lo > 0.0
            else:
                exceeds = bool(np.isfinite(ci[0]) and ci[0] > c["effective"])
            v = PASS if not (a_best > c["effective"]) else INVESTIGATE
            if a_best > trigger and exceeds:
                v = FAIL
            note = ""
            for ep, lo_s, hi_s, what in SIGNATURES:
                if ep == endpoint and lo_s <= a_best < hi_s:
                    note = what
        res["endpoints"][endpoint] = dict(
            auc=a_best, auc_raw=a_raw, side=("higher" if side > 0 else "REVERSED"),
            readout=best, all_readouts=readouts, ci95=ci,
            delta_vs_ceiling=delta,
            n_pos=int(y.sum()), prevalence=float(y.mean()) if len(y) else nan,
            ceiling=c["effective"], ceiling_reference=c["reference"], ceiling_live=c["live"],
            trigger=trigger, verdict=v, signature=note)
        res["verdict"] = worst([res["verdict"], v])
    return res


# ---------------------------------------------------------------- TEST 2
def horizon_rows(streams, sub, lead, seed=0, tol_years=2.0, subset=None):
    """Cases (converters cut `lead` years before diagnosis) and age-matched never-converters.

    Controls are matched on the cut AGE, 1:1 without replacement, within `tol_years`: the leads
    walk backwards from the diagnosis, so without matching the case ages drift younger with the
    lead while the control ages do not, and the curve would be reading age. Both scorers see
    the identical rows and the refit is given age as a feature, so any residual mismatch is
    shared rather than an advantage.

    `subset` restricts BOTH the cases and the control pool. Filtering a globally matched set
    afterwards instead leaves the subgroup unbalanced -- on the canary build's train split the
    marked rows come out 31 cases against 26 controls at lead 1 -- and pushes the long leads
    under T2_MIN_CASES, which is exactly where the canary's flat-elevated signature lives.

    Every row carries `pair`, the case's projid, so the bootstrap in test2_curve can resample
    a case and its matched control as one unit rather than as two independent rows.
    """
    cases = []
    for pid, (ages, toks) in streams.items():
        if pid not in sub.index or (subset is not None and pid not in subset):
            continue
        r = sub.loc[pid]
        if not bool(r["in_eval"]) or int(r["event_type"]) != 1:
            continue
        cut = (float(r["event_age"]) - lead) * DAYS_PER_YEAR
        if cut < float(r["age_bl"]) * DAYS_PER_YEAR or not _prefix_ok(ages, toks, cut):
            continue
        cases.append((pid, cut))
    pool = []
    for pid, (ages, toks) in streams.items():
        if pid not in sub.index or (subset is not None and pid not in subset):
            continue
        r = sub.loc[pid]
        if not bool(r["in_eval"]) or int(r["event_type"]) == 1:
            continue
        lo = float(r["age_bl"]) * DAYS_PER_YEAR
        # a clean 12-month negative, minus a day: label_at_horizon scores a death exactly AT
        # the horizon end as a competing event (`ea <= end`), so a cut of precisely
        # event_age - 1 y is rejected for every died control and the scan below burns through
        # them. One day is two orders of magnitude inside the age grid's own ~21-day error.
        hi = (float(r["event_age"]) - 1.0) * DAYS_PER_YEAR - 1.0
        if hi >= lo:
            pool.append([pid, lo, hi])

    rng = np.random.default_rng(seed)
    used = set()
    rows = []
    for j in rng.permutation(len(cases)):        # matching order must not depend on the file
        pid, cut = cases[j]
        # walk the pool nearest-first and take the first control that is genuinely scoreable;
        # a case that cannot be matched is DROPPED rather than left unpaired, so the ratio
        # stays 1:1 and the AUC at each lead is computed on a balanced set
        cand = sorted(((abs(min(max(cut, lo), hi) - cut), i, min(max(cut, lo), hi))
                       for i, (_p, lo, hi) in enumerate(pool) if i not in used),
                      key=lambda z: z[0])
        picked = None
        for d, i, c in cand:
            if d > tol_years * DAYS_PER_YEAR:
                break
            cpid = pool[i][0]
            cages, ctoks = streams[cpid]
            # the competing-risk label comes from the shared definition, never a local copy.
            # A rejection is a fact about THIS cut, not about the candidate -- the label and
            # the prefix both move with it -- so the candidate stays in the pool for the next
            # case. Retiring it here dropped 2 of 103 eligible val cases at lead 1.
            if _prefix_ok(cages, ctoks, c) and cohort.label_at_horizon(sub.loc[cpid], c,
                                                                       1.0) == 0:
                picked = (i, cpid, c, cages, ctoks)
                break
        if picked is None:
            continue
        i, cpid, c, cages, ctoks = picked
        used.add(i)
        ages, toks = streams[pid]
        rows.append(dict(projid=int(pid), y=1, cut_day=float(cut), pair=int(pid),
                         x=prefix_features(ages, toks, cut)))
        rows.append(dict(projid=int(cpid), y=0, cut_day=float(c), pair=int(pid),
                         x=prefix_features(cages, ctoks, c)))
    return rows


def _model(seed):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(max_iter=2000, C=1.0, random_state=seed))


def refit_predict(train_rows, test_rows, seed=0, cross_fit=False):
    """The legal logistic refit's probability for every audited row at one lead.

    `cross_fit` covers the case where the audited split IS the split the refit would be
    fitted on (the canary is audited on train, where the planted token has the most
    instances). An in-sample refit is optimistically biased, which depresses the gap and
    costs the audit sensitivity exactly where it is being tested; 5-fold GROUPED on projid
    keeps a subject's case and control rows out of the fold that scores them.

    Both paths fit on ALL of the split's rows and never on a subset. The canary contrast
    scores two subgroups of a dozen subjects each, and a refit fitted inside one of them is
    so weak that it lands below chance -- which would show up as a large positive gap and a
    leak that is not there.
    """
    xtr, keys = _matrix(train_rows)
    ytr = np.array([r["y"] for r in train_rows], int)
    xte = np.array([[r["x"][k] for k in keys] for r in test_rows], float)
    if not cross_fit:
        return _model(seed).fit(xtr, ytr).predict_proba(xte)[:, 1]
    from sklearn.model_selection import GroupKFold
    yte = np.array([r["y"] for r in test_rows], int)
    g = np.array([r["projid"] for r in test_rows])
    p = np.zeros(len(yte))
    for tr_i, te_i in GroupKFold(n_splits=5).split(xte, yte, groups=g):
        p[te_i] = _model(seed).fit(xte[tr_i], yte[tr_i]).predict_proba(xte[te_i])[:, 1]
    return p


def _orient(a, side):
    """The AUC as an amount of knowledge: a leak that enters ranking cases LAST is a leak."""
    return a if side > 0 else 1.0 - a


def test2_curve(eng, streams_eval, sub_eval, streams_train, sub_train, n_mc, seed=0,
                subset=None, cross_fit=False, match_seeds=3, n_boot=1000, verbose=True):
    """The horizon curve: model AUC minus legal-refit AUC at each lead, with an interval.

    `subset` restricts the audited rows to a set of projids (the canary group contrast), and
    it is passed down into the matching so that a subgroup's cases and controls are both
    drawn from the subgroup. The refit still TRAINS on the whole split's rows, which is what
    makes the two canary curves comparable.

    THE CURVE IS AVERAGED OVER `match_seeds` MATCHING DRAWS. Which control a case is paired
    with is a random draw, and the draw alone moves gap_long by 0.093 on val (-0.317 to
    -0.224 across five seeds) against a trigger of 0.10 -- one seed's curve is a sample, not
    a measurement. The cases and their cut days do not depend on the seed, so the model's
    hazard is cached and an extra draw costs only its new control rows.

    THE INTERVAL is a cluster bootstrap in which a case and its matched control are resampled
    as ONE unit, with a single subject draw shared by every lead and every matching seed so
    that gap_long and growth inherit it coherently. Test 2 hangs a hard 0.10 threshold on the
    mean of two AUC differences computed on 44 and 24 cases; that quantity's 95% interval is
    0.27 wide on val, so without this the verdict is a coin toss recorded as a measurement.

    ORIENTATION. Both AUCs enter the gap as max(a, 1 - a), with the side recorded. The side
    is decided once on the seed-averaged point estimate and then held FIXED across bootstrap
    replicates, so the max() cannot bias the interval upward at 24 cases.

    RESOLUTION. The model's score here is a Monte-Carlo probability, so it is quantised at
    1/n_mc, and the 12-month AD hazard in this cohort is ~2%. Below a few hundred trajectories
    most rows tie at zero and the model's AUC is deflated by the ties -- conservative for the
    audit, but it costs sensitivity, so n_mc is worth spending on this test.
    """
    nan = float("nan")
    haz_cache = {}

    def hazard(pid, cut_day):
        # the case rows are identical across matching seeds, and controls repeat across leads
        key = (int(pid), round(float(cut_day), 3))
        if key not in haz_cache:
            a, t = Engine.prefix_at(*streams_eval[pid], cut_day)
            haz_cache[key] = eng.risk_by_horizon(t, a, horizons=(1,), n_mc=n_mc, seed=seed,
                                                 from_day=cut_day)[1]
        return haz_cache[key]

    curve, draws = [], []
    for lead in LEADS:
        per_seed = []
        for m in range(match_seeds):
            tr = horizon_rows(streams_train, sub_train, lead, seed=seed + m)
            te = horizon_rows(streams_eval, sub_eval, lead, seed=seed + m, subset=subset)
            n_case = int(sum(r["y"] for r in te))
            if (n_case < 3 or len(te) - n_case < 3 or len(tr) < 40
                    or len(set(r["y"] for r in tr)) < 2):
                continue
            p_ref = np.asarray(refit_predict(tr, te, seed=seed, cross_fit=cross_fit), float)
            if not np.isfinite(p_ref).all():
                continue
            y = np.array([r["y"] for r in te], int)
            haz = np.array([hazard(r["projid"], r["cut_day"]) for r in te], float)
            per_seed.append(dict(y=y, model=haz, refit=p_ref, n_rows=len(te),
                                 pair=np.array([r["pair"] for r in te], np.int64),
                                 n_cases=n_case, n_train=len(tr)))
        entry = dict(lead_y=lead, n_rows=0, n_cases=0, n_train_rows=0,
                     n_match_seeds=len(per_seed), auc_model=nan, auc_refit=nan, gap=nan,
                     gap_ci=(nan, nan), mean_hazard_cases=nan, mean_refit_cases=nan,
                     status=SKIP)
        curve.append(entry)
        if not per_seed:
            draws.append(None)
            if verbose:
                print(f"    lead {lead:>4.0f}y: not scoreable", flush=True)
            continue
        a_mod_raw = float(np.mean([_auc_fast(d["y"], d["model"]) for d in per_seed]))
        a_ref_raw = float(np.mean([_auc_fast(d["y"], d["refit"]) for d in per_seed]))
        side_mod = 1.0 if a_mod_raw >= 0.5 else -1.0
        side_ref = 1.0 if a_ref_raw >= 0.5 else -1.0
        a_mod, a_ref = _orient(a_mod_raw, side_mod), _orient(a_ref_raw, side_ref)
        # the matching draw can lose a case, so the count that faces T2_MIN_CASES is the
        # smallest any draw achieved, not the luckiest
        n_case = int(min(d["n_cases"] for d in per_seed))
        entry.update(n_rows=int(min(d["n_rows"] for d in per_seed)), n_cases=n_case,
                     n_train_rows=int(np.mean([d["n_train"] for d in per_seed])),
                     auc_model=a_mod, auc_model_raw=a_mod_raw,
                     auc_refit=a_ref, auc_refit_raw=a_ref_raw, gap=a_mod - a_ref,
                     model_side=("higher" if side_mod > 0 else "REVERSED"),
                     refit_side=("higher" if side_ref > 0 else "REVERSED"),
                     mean_hazard_cases=float(np.mean([d["model"][d["y"] == 1].mean()
                                                      for d in per_seed])),
                     mean_refit_cases=float(np.mean([d["refit"][d["y"] == 1].mean()
                                                     for d in per_seed])),
                     mean_hazard_controls=float(np.mean([d["model"][d["y"] == 0].mean()
                                                         for d in per_seed])),
                     status=SCORED if n_case >= T2_MIN_CASES else SKIP)
        draws.append((per_seed, side_mod, side_ref))
        if verbose:
            print(f"    lead {lead:>4.0f}y: n={entry['n_rows']:>4} ({n_case} cases, "
                  f"{len(per_seed)} matching draws)  model {a_mod:.3f}  refit {a_ref:.3f}  "
                  f"gap {entry['gap']:+.3f}", flush=True)

    # ---- one subject draw, shared by every lead: gap_long and growth are read off pairs of
    # leads that share their cases, and independent per-lead intervals cannot be combined
    clusters = sorted({int(p) for d in draws if d for q in d[0] for p in q["pair"]})
    if clusters and n_boot:
        index = {c: k for k, c in enumerate(clusters)}
        for d in draws:
            if d:
                for q in d[0]:
                    q["ci"] = np.array([index[int(p)] for p in q["pair"]], np.int64)
        boot = {c["lead_y"]: np.full(n_boot, nan) for c in curve}
        rng = np.random.default_rng(seed + 9001)
        flat = np.full(len(clusters), 1.0 / len(clusters))
        for b in range(n_boot):
            counts = rng.multinomial(len(clusters), flat)
            for entry, d in zip(curve, draws):
                if not d:
                    continue
                per_seed, side_mod, side_ref = d
                gaps = []
                for q in per_seed:
                    mult = counts[q["ci"]]
                    sel = np.repeat(np.arange(len(mult)), mult)
                    yb = q["y"][sel]
                    if len(yb) < 4 or yb.min() == yb.max():
                        continue
                    gaps.append(_orient(_auc_fast(yb, q["model"][sel]), side_mod)
                                - _orient(_auc_fast(yb, q["refit"][sel]), side_ref))
                if gaps:
                    boot[entry["lead_y"]][b] = float(np.mean(gaps))
        for entry in curve:
            g = boot[entry["lead_y"]]
            entry["_gap_boot"] = g
            g = g[np.isfinite(g)]
            if len(g):
                entry["gap_ci"] = (float(np.percentile(g, 2.5)),
                                   float(np.percentile(g, 97.5)))
    return curve


def test2_verdict(curve, tag=""):
    """Flat and elevated is a static leak; growing is a model reading antecedents.

    The elevation is judged on the bootstrap's LOWER bound rather than the point estimate.
    gap_long is the mean of two AUC differences computed on 44 and 24 val cases: its 95%
    interval is 0.270 wide and the matching draw alone moves it by 0.093, against a trigger
    of 0.10, so thresholding the point estimate decides the verdict on the seed. The growth
    term stays a point estimate on purpose -- it only chooses between FAIL and INVESTIGATE,
    both already non-PASS, and demanding an interval there would cost the canary its FAIL at
    31 cases.
    """
    nan = float("nan")
    boot = {c["lead_y"]: np.asarray(c.pop("_gap_boot"), float) for c in curve
            if "_gap_boot" in c}
    ok = [c for c in curve if c["status"] != SKIP and np.isfinite(c["gap"])]
    res = dict(tag=tag, curve=curve, gap_long=nan, gap_long_ci=(nan, nan), gap_short=nan,
               growth=nan, growth_ci=(nan, nan), gap_trigger=T2_GAP_TRIGGER,
               flat_trigger=T2_FLAT_TRIGGER, verdict=SKIP, note="")
    if len(ok) < 2:
        res["note"] = (f"only {len(ok)} lead(s) reached {T2_MIN_CASES} cases -- the curve "
                       f"cannot be read, and that is a sample-size fact, not a clean bill")
        return res
    long_leads = [c for c in ok if c["lead_y"] >= 8.0] or [max(ok, key=lambda c: c["lead_y"])]
    short = min(ok, key=lambda c: c["lead_y"])
    gap_long = float(np.mean([c["gap"] for c in long_leads]))
    growth = float(short["gap"] - gap_long)

    def _ci(arr):
        a = arr[np.isfinite(arr)] if arr is not None else np.array([])
        return ((float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
                if len(a) > 20 else (nan, nan))

    b_long = (np.mean([boot[c["lead_y"]] for c in long_leads], axis=0)
              if all(c["lead_y"] in boot for c in long_leads) else None)
    b_growth = (boot[short["lead_y"]] - b_long
                if b_long is not None and short["lead_y"] in boot else None)
    lo, hi = _ci(b_long)
    res.update(gap_long=gap_long, gap_long_ci=(lo, hi), gap_short=float(short["gap"]),
               growth=growth, growth_ci=_ci(b_growth),
               long_leads=[c["lead_y"] for c in long_leads], short_lead=short["lead_y"],
               n_cases_long=[c["n_cases"] for c in long_leads])
    ci_txt = (f"95% CI [{lo:+.3f},{hi:+.3f}]" if np.isfinite(lo)
              else "no interval (the bootstrap could not be formed)")
    elevated = bool(np.isfinite(lo) and lo > T2_GAP_TRIGGER)
    if elevated and growth < T2_FLAT_TRIGGER:
        res["verdict"] = FAIL
        res["note"] = (f"the model beats the legal refit by {gap_long:+.3f} {ci_txt} as far "
                       f"out as {res['long_leads']} years and gains only {growth:+.3f} by 1 "
                       f"year: flat and already elevated is the signature of a static leak, "
                       f"not of antecedent signal (triggers: the gap's lower bound > "
                       f"{T2_GAP_TRIGGER:+.2f} and growth < {T2_FLAT_TRIGGER:+.2f})")
    elif elevated:
        res["verdict"] = INVESTIGATE
        res["note"] = (f"long-lead gap {gap_long:+.3f} {ci_txt} is over the trigger but the "
                       f"curve does grow ({growth:+.3f} by 1 year, over "
                       f"{T2_FLAT_TRIGGER:+.2f}) -- consistent with a strong model; check "
                       f"the long-lead rows by hand")
    elif gap_long > T2_GAP_TRIGGER:
        res["verdict"] = INVESTIGATE
        res["note"] = (f"long-lead gap {gap_long:+.3f} is over the {T2_GAP_TRIGGER:+.2f} "
                       f"trigger but {ci_txt} covers it, on {res['n_cases_long']} cases -- "
                       f"not evidence of a leak and not a clean bill either")
    else:
        res["verdict"] = PASS
        res["note"] = (f"long-lead gap {gap_long:+.3f} {ci_txt} is inside the "
                       f"{T2_GAP_TRIGGER:+.2f} trigger; the curve gains {growth:+.3f} AUC "
                       f"between the longest lead and 1 year")
    return res


# ---------------------------------------------------------------- TEST 3
def canary_verdict(y, score, marked, ceiling, trigger, fire_at=CANARY_FIRE_AUC,
                   min_marked=CANARY_MIN_MARKED):
    """Did the detector see the planted oracle, and only the planted oracle?

    A pure function of arrays so that --self-test can drive it with synthesised scores: this
    is the piece that has to work, and it must not be reachable only through a GPU run.
    """
    y = np.asarray(y, int)
    score = np.asarray(score, float)
    marked = np.asarray(marked, bool)
    res = dict(n_marked=int(marked.sum()), n_unmarked=int((~marked).sum()),
               n_marked_pos=int(y[marked].sum()), fire_at=fire_at,
               auc_marked=auc(y[marked], score[marked]),
               auc_unmarked=auc(y[~marked], score[~marked]),
               ceiling=ceiling, trigger=trigger)
    if res["n_marked"] < min_marked or not np.isfinite(res["auc_marked"]):
        res["verdict"] = SKIP
        res["note"] = (f"{res['n_marked']} marked subjects ({res['n_marked_pos']} of them "
                       f"positive) cannot resolve a fire/no-fire call; audit the canary on "
                       f"--split train, or rebuild with a larger --canary-fraction")
        return res
    fired = res["auc_marked"] >= fire_at
    quiet = np.isfinite(res["auc_unmarked"]) and res["auc_unmarked"] <= trigger
    res["fired_on_marked"] = bool(fired)
    res["quiet_on_unmarked"] = bool(quiet)
    if fired and quiet:
        res["verdict"] = PASS
        res["note"] = (f"the planted oracle was caught (marked AUC {res['auc_marked']:.3f} "
                       f">= {fire_at}) and the unmarked subjects stayed under the trigger "
                       f"({res['auc_unmarked']:.3f} <= {trigger})")
    elif not fired:
        res["verdict"] = FAIL
        res["note"] = (f"THE DETECTOR MISSED A PLANTED PERFECT ORACLE: marked AUC "
                       f"{res['auc_marked']:.3f} < {fire_at}. Every clean result this file "
                       f"has ever produced is uninterpretable until this is explained")
    else:
        res["verdict"] = FAIL
        res["note"] = (f"the canary fired but so did the unmarked subjects "
                       f"({res['auc_unmarked']:.3f} > {trigger}): either the leak is not "
                       f"confined to the marked 5% or the audit fires on everything")
    return res


def self_test():
    """Drive every verdict function with synthesised scores whose right answer is known.

    An audit is only as good as its ability to fail, and the canary build needs a GPU run.
    This needs nothing, so it runs on every invocation and the detector is never shipped
    untested.
    """
    rng = np.random.default_rng(0)
    checks = []

    # 1. the ceiling verdicts, at and around both triggers
    ceil = {ep: dict(reference=r, live=float("nan"), effective=r, trigger=t)
            for ep, r, t in (("ever_ad", 0.7207, TRIGGER_EVER_AD),
                             ("death", 0.8853, TRIGGER_DEATH))}
    for a_ad, want in ((0.70, PASS), (0.74, INVESTIGATE), (0.87, FAIL), (0.99, FAIL)):
        rows = _synthetic_rows(rng, n=400, auc_target=a_ad)
        got = test1_verdict(rows, ceil)["endpoints"]["ever_ad"]
        checks.append((f"ceiling verdict at target AUC {a_ad:.2f}", got["verdict"], want,
                       f"measured {got['auc']:.3f}"))
    got = test1_verdict(_synthetic_rows(rng, 400, 0.97), ceil)
    checks.append(("signature named at AUC ~0.97",
                   bool(got["endpoints"]["ever_ad"]["signature"]), True, ""))

    # 1b. the noise rules, which are what stops a clean model from failing on its own SE
    at_trigger = _synthetic_rows(rng, n=378, auc_target=0.762)      # the val n, at the trigger
    checks.append(("ceiling verdict at the trigger on 378 subjects (the CI covers it)",
                   test1_verdict(at_trigger, ceil)["endpoints"]["ever_ad"]["verdict"],
                   INVESTIGATE, "point over 0.76, lower bound under the ceiling"))
    tied = _synthetic_rows(rng, n=400, auc_target=0.87)
    ceil_pred = {ep: dict(v, pred=np.array([r["p_ad_ever"] for r in tied])
                          + rng.normal(0, 1e-6, len(tied))) for ep, v in ceil.items()}
    checks.append(("ceiling verdict: B2 matches the model on the same subjects",
                   test1_verdict(tied, ceil_pred)["endpoints"]["ever_ad"]["verdict"] == FAIL,
                   False, "paired delta 0 -- a high AUC that the legal features also reach"))
    checks.append(("ceiling verdict: a sign-REVERSED oracle is a leak too",
                   test1_verdict(_synthetic_rows(rng, 400, 0.02),
                                 ceil)["endpoints"]["ever_ad"]["verdict"], FAIL,
                   "AUC 0.02 carries as much information as 0.98"))
    degenerate = [dict(r, ever_ad=1) for r in _synthetic_rows(rng, 60, 0.8)]
    checks.append(("ceiling verdict on a degenerate outcome is not a PASS",
                   test1_verdict(degenerate, ceil)["endpoints"]["ever_ad"]["verdict"], SKIP,
                   "every subject positive -- the AUC is NaN"))

    # 2. the horizon curve: flat and elevated is a leak, elevated but growing is only
    #    worth a look, and a gap that starts under the trigger is a model doing its job.
    #    Every curve carries a bootstrap, because the verdict reads its lower bound.
    def curve(gaps, sd, n_cases=50, status=SCORED):
        return [dict(lead_y=l, gap=g, status=status, n_cases=n_cases,
                     _gap_boot=g + rng.normal(0, sd, 400)) for l, g in zip(LEADS, gaps)]

    flat = curve([0.30] * 5, 0.03)
    noisy = curve([0.30] * 5, 0.15)
    big_grow = curve([0.45 - 0.02 * l for l in LEADS], 0.03)
    healthy = curve([0.16 - 0.02 * l for l in LEADS], 0.03)
    honest = curve([-0.05] * 5, 0.03)
    thin = [dict(lead_y=l, gap=float("nan"), status=SKIP, n_cases=2) for l in LEADS]
    checks.append(("horizon: flat elevated gap", test2_verdict(flat)["verdict"], FAIL, ""))
    checks.append(("horizon: the same gap with a 0.30-wide interval",
                   test2_verdict(noisy)["verdict"], INVESTIGATE,
                   "elevated point, lower bound under the trigger"))
    checks.append(("horizon: elevated AND growing", test2_verdict(big_grow)["verdict"],
                   INVESTIGATE, ""))
    checks.append(("horizon: grows from under the trigger",
                   test2_verdict(healthy)["verdict"], PASS, ""))
    checks.append(("horizon: model below refit", test2_verdict(honest)["verdict"], PASS, ""))
    checks.append(("horizon: too few cases", test2_verdict(thin)["verdict"], SKIP, ""))

    # 2b. the roll-up. A test that could not be read is the one thing that must never
    #     become an OVERALL: PASS with exit 0.
    checks.append(("roll-up: an unreadable test outranks PASS", worst([PASS, SKIP]), SKIP, ""))
    checks.append(("roll-up: INVESTIGATE outranks an unreadable test",
                   worst([SKIP, INVESTIGATE, PASS]), INVESTIGATE, ""))
    checks.append(("roll-up: FAIL dominates", worst([PASS, SKIP, INVESTIGATE, FAIL]),
                   FAIL, ""))

    # 3. the canary contrast, planted and clean
    n = 400
    marked = rng.random(n) < 0.25          # a larger marked share than the build, for power
    y = (rng.random(n) < 0.26).astype(int)
    weak = 0.5 * y + rng.normal(0, 1.0, n)                    # a legal-strength score
    planted = np.where(marked & (y == 1), 10.0, weak)         # the oracle, on the marked only
    checks.append(("canary: planted oracle is caught",
                   canary_verdict(y, planted, marked, 0.7207, TRIGGER_EVER_AD)["verdict"],
                   PASS, ""))
    checks.append(("canary: no oracle -> the detector must say it missed one",
                   canary_verdict(y, weak, marked, 0.7207, TRIGGER_EVER_AD)["verdict"],
                   FAIL, ""))
    everywhere = np.where(y == 1, 10.0, weak)                 # a leak that is NOT confined
    checks.append(("canary: leak outside the marked set is caught too",
                   canary_verdict(y, everywhere, marked, 0.7207, TRIGGER_EVER_AD)["verdict"],
                   FAIL, ""))
    checks.append(("canary: too few marked subjects",
                   canary_verdict(y[:20], planted[:20], marked[:20], 0.7207,
                                  TRIGGER_EVER_AD)["verdict"], SKIP, ""))

    print("\n=== SELF-TEST: the detector, driven with scores whose answer is known ===")
    bad = 0
    for name, got, want, extra in checks:
        ok = got == want
        bad += not ok
        print(f"  [{'ok ' if ok else 'BAD'}] {name:<52s} -> {got!s:<13s} "
              f"(expected {want!s}) {extra}")
    print(f"  {len(checks) - bad}/{len(checks)} detector checks passed")
    return bad == 0


def _synthetic_rows(rng, n, auc_target):
    """Rows shaped like test1's, carrying a score of EXACTLY the requested AUC.

    Constructed by rank rather than sampled from two normals: a sampled score of nominal AUC
    0.74 lands anywhere in 0.68-0.80 at n=400, which would make the boundary checks below
    flaky and a flaky self-test teaches people to ignore it. Every positive is placed just
    above the same negative quantile, so the empirical AUC is the target to within 1/n0.
    """
    y = (rng.random(n) < 0.26).astype(int)
    n0 = int((y == 0).sum())
    s = np.empty(n, float)
    s[y == 0] = rng.permutation(n0).astype(float)
    s[y == 1] = round(auc_target * n0) - 0.5
    p = s / max(n0, 1)
    return [dict(projid=i, ever_ad=int(y[i]), death=int(y[i]), p_ad_ever=float(p[i]),
                 p_ad_10y=float(p[i] * 0.5), p_death_ever=float(p[i]),
                 p_death_10y=float(p[i] * 0.5), x={"age": 80.0}) for i in range(n)]


# ---------------------------------------------------------------- reporting
# Two data series (the model and the legal refit) in a fixed hue order, never cycled;
# validated colourblind-safe against the light surface (worst adjacent pair dE 28.0 protan /
# 33.3 normal). The trigger is a status colour and is not reusable as a third series.
C_MODEL, C_REFIT, C_TRIGGER, C_AXIS = "#3b6fd4", "#d97706", "#d03b3b", "#6b7280"


def plot_curve(curves, path):
    """The horizon curve, plotted as the two panels the verdict is read off.

    Left: what each scorer achieves at each lead. Right: their difference, against the
    trigger. Underpowered leads (< T2_MIN_CASES cases) are drawn hollow, so a curve that
    bends because it ran out of converters cannot be mistaken for one that bends because the
    model does. No second y-axis anywhere: AUC and an AUC difference are different scales and
    get different panels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    styles = ["-", "--"]
    for gi, (label, curve) in enumerate(curves.items()):
        ok = [c for c in curve if np.isfinite(c.get("gap", float("nan")))]
        if not ok:
            continue
        ls = styles[gi % len(styles)]
        x = [c["lead_y"] for c in ok]
        thin = [c["n_cases"] < T2_MIN_CASES for c in ok]
        for a, ys, colour, name in ((ax[0], [c["auc_model"] for c in ok], C_MODEL, "model"),
                                    (ax[0], [c["auc_refit"] for c in ok], C_REFIT,
                                     "legal refit"),
                                    (ax[1], [c["gap"] for c in ok], C_MODEL,
                                     "model - refit")):
            a.plot(x, ys, ls, color=colour, lw=2, zorder=3,
                   label=f"{name}" + (f" ({label})" if len(curves) > 1 else ""))
            a.scatter(x, ys, s=[36 if t else 64 for t in thin], zorder=4,
                      facecolors=["white" if t else colour for t in thin],
                      edgecolors=colour, linewidths=1.8)
            # direct label at the short-lead end, so identity never rests on colour alone
            a.annotate(name, (x[0], ys[0]), textcoords="offset points", xytext=(6, 7),
                       fontsize=8, color=colour)
    ax[1].axhline(T2_GAP_TRIGGER, color=C_TRIGGER, lw=1.5, ls=":", zorder=2)
    ax[1].annotate(f"trigger {T2_GAP_TRIGGER:+.2f}", (0.02, T2_GAP_TRIGGER),
                   xycoords=("axes fraction", "data"), textcoords="offset points",
                   xytext=(0, 4), fontsize=8, color=C_TRIGGER)
    ax[1].axhline(0.0, color=C_AXIS, lw=0.8, zorder=1)
    ax[0].axhline(0.5, color=C_AXIS, lw=0.8, zorder=1)

    # case counts go under the tick, but only when one group is plotted -- with two the
    # counts differ per group and a single number under the tick would be a lie
    n_by_lead = {}
    for curve in curves.values():
        for c in curve:
            n_by_lead[c["lead_y"]] = (max(n_by_lead.get(c["lead_y"], 0), c["n_cases"])
                                      if len(curves) == 1 else None)
    for a, title, ylab in ((ax[0], "12-month AD hazard against age-matched controls", "AUC"),
                           (ax[1], "model minus the legal refit", "AUC difference")):
        a.set_title(title, fontsize=10)
        a.set_ylabel(ylab, fontsize=9)
        a.set_xlabel("years from the cut to the diagnosis", fontsize=9)
        a.set_xticks(sorted(n_by_lead))
        a.set_xticklabels([f"{int(l)}" + (f"\n(n={n_by_lead[l]})" if n_by_lead[l] else "")
                           for l in sorted(n_by_lead)], fontsize=8)
        a.invert_xaxis()
        a.grid(axis="y", color=C_AXIS, alpha=0.18, lw=0.7)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(C_AXIS)
        if len(a.get_legend_handles_labels()[0]) > 1:   # one series needs no legend box
            a.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("Leakage test 2 -- antecedent signal grows toward the diagnosis; "
                 "a static leak is flat and already elevated", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=140, facecolor="white")
    plt.close(fig)


def _line(name, verdict, detail):
    mark = {PASS: "PASS        ", INVESTIGATE: "INVESTIGATE ", FAIL: "FAIL        ",
            SKIP: "INSUFFICIENT"}[verdict]
    print(f"  {mark} {name:<34s} {detail}")


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="Adversarial leakage audit of a RADC Delphi checkpoint.")
    ap.add_argument("--ckpt", default="out-radc-base-s42/ckpt.pt")
    ap.add_argument("--data-dir", default=os.environ.get(
        "RADC_DATA_DIR", os.path.join(_HERE, "data", "radc-s42")))
    ap.add_argument("--radc-dir", default=os.path.join(_HERE, "data", "RADC"),
                    help="the raw files -- eval.cohort needs them for baseline_impaired")
    ap.add_argument("--split", default="val",
                    help="develop against val. The test split is scored once, at the end.")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-mc", type=int, default=200, help="Monte-Carlo trajectories per prefix")
    ap.add_argument("--max-new-tokens", type=int, default=256,
                    help="rollout budget for the lifetime simulation in test 1; the share of "
                         "trajectories that end with no death token is reported with it")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-seeds", type=int, default=3,
                    help="matching draws averaged into the test-2 curve; the draw alone "
                         "moves the long-lead gap by 0.09, so one is not a measurement")
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap replicates behind both FAIL rules")
    ap.add_argument("--ceilings-json", default=None,
                    help="read the live B2 ceiling out of eval/baselines_<split>.json instead "
                         "of re-fitting it here (the reference is always imported from "
                         "eval.baselines.REFERENCE; neither is ever typed in)")
    ap.add_argument("--no-live-ceiling", action="store_true",
                    help="skip the B2 re-fit and judge against the delivered reference alone")
    ap.add_argument("--ceiling-data-dir", default=None,
                    help="build the live ceiling from this directory instead of --data-dir. "
                         "Defaults to the production build under --canary, because B2's "
                         "feature matrix would otherwise absorb the planted token")
    ap.add_argument("--canary", action="store_true",
                    help="audit a canary build: require tests 1 and 2 to fire on the marked "
                         "subjects and stay quiet on the rest")
    ap.add_argument("--self-test", action="store_true",
                    help="drive the verdict functions with synthesised scores and exit")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    if a.self_test:
        return 0 if self_test() else 1

    if a.canary:
        # the audited build has 51 ids; the standard table is proved intact before it is
        # extended, and only this branch extends it
        enable_canary_vocab()

    labels = pd.read_csv(os.path.join(a.data_dir, "labels.csv"))["event_name"].tolist()
    # PREFLIGHT. The canary must never be in a production vocabulary, and a checkpoint scored
    # against the wrong label table mis-indexes every probability. Both are cheap to check and
    # both have silently happened in this family of repositories.
    if a.canary:
        assert labels[-1] == CANARY_NAME and len(labels) == V.VOCAB_SIZE == 51, \
            f"--canary but {a.data_dir}/labels.csv is not a canary build"
    else:
        assert len(labels) == V.VOCAB_SIZE, \
            f"{a.data_dir}/labels.csv has {len(labels)} rows, the vocabulary has {V.VOCAB_SIZE}"
        assert CANARY_NAME not in labels, \
            f"THE CANARY IS IN A PRODUCTION VOCABULARY: {a.data_dir}/labels.csv"

    eng = load_engine(a.ckpt, data_dir=a.data_dir, device=a.device)
    sub = cohort.load_subjects(a.data_dir, radc_dir=a.radc_dir)
    sub_eval = sub[sub["split"] == a.split]
    sub_train = sub[sub["split"] == "train"]
    streams = load_streams(eng, a.split, a.data_dir)
    streams_train = load_streams(eng, "train", a.data_dir) if a.split != "train" else streams

    print(f"\n=== LEAKAGE AUDIT  {a.ckpt}  ({a.data_dir}, split '{a.split}')")
    if a.split == "test":
        print("  *** THE TEST SPLIT. This is the once-only scoring run; everything upstream "
              "of it should already be frozen. Develop against val. ***")
    print(f"  {len(sub_eval):,} subjects in the split, "
          f"{int(sub_eval['in_eval'].sum()):,} in the evaluation cohort "
          f"({int((sub_eval['in_eval'] & (sub_eval['event_type'] == 1)).sum()):,} converters)")

    print(f"\n  [1/2] baseline-prefix rollouts, n_mc={a.n_mc} ...")
    rows, skipped, frac_unterm = baseline_prefix_rollout(
        eng, streams, sub_eval, a.n_mc, a.max_new_tokens, a.seed)
    ceil_dir = a.ceiling_data_dir or (os.path.join(_HERE, "data", "radc-s42") if a.canary
                                      else a.data_dir)
    ceilings = legal_ceilings(ceil_dir, a.radc_dir, a.split,
                              pids=None if a.ceilings_json else [r["projid"] for r in rows],
                              live=not a.no_live_ceiling, artifact=a.ceilings_json)
    print(f"  ceilings from {ceilings['source']}")
    for _ep in ("ever_ad", "death"):
        if ceilings[_ep].get("live_note"):
            print(f"  NOTE ({_ep}): {ceilings[_ep]['live_note']}")
    t1 = test1_verdict(rows, ceilings, seed=a.seed, tag=a.split, n_boot=a.n_boot)
    t1["skipped"] = skipped
    t1["frac_trajectories_unterminated"] = frac_unterm

    # auditing the train split (the canary's default, where the planted token has the most
    # instances) means the refit would otherwise be fitted and scored on the same rows
    cross_fit = a.split == "train"
    print(f"  [2/2] horizon curve at leads {[int(l) for l in LEADS]}"
          + (" (refit cross-fitted, 5-fold by subject)" if cross_fit else "") + " ...")
    marked = None
    if a.canary:
        flag = pd.read_csv(os.path.join(a.data_dir, "subjects.csv")).set_index("projid")
        marked = set(flag.index[flag["canary"].astype(bool)].tolist())
        curve_marked = test2_curve(eng, streams, sub_eval, streams_train, sub_train,
                                   a.n_mc, seed=a.seed, subset=marked, cross_fit=cross_fit,
                                   match_seeds=a.match_seeds, n_boot=a.n_boot)
        curve_rest = test2_curve(eng, streams, sub_eval, streams_train, sub_train,
                                 a.n_mc, seed=a.seed, cross_fit=cross_fit,
                                 subset=set(sub_eval.index) - marked,
                                 match_seeds=a.match_seeds, n_boot=a.n_boot)
        t2 = dict(marked=test2_verdict(curve_marked, tag="canary-marked"),
                  unmarked=test2_verdict(curve_rest, tag="canary-unmarked"))
        curves = {"marked": curve_marked, "unmarked": curve_rest}
    else:
        curve = test2_curve(eng, streams, sub_eval, streams_train, sub_train,
                            a.n_mc, seed=a.seed, cross_fit=cross_fit,
                            match_seeds=a.match_seeds, n_boot=a.n_boot)
        t2 = test2_verdict(curve, tag=a.split)
        curves = {a.split: curve}

    t3 = None
    if a.canary:
        m = np.array([r["projid"] in marked for r in rows], bool)
        e_ad = t1["endpoints"]["ever_ad"]
        sgn = -1.0 if e_ad["side"] == "REVERSED" else 1.0     # score it the way test 1 read it
        t3 = dict(
            test1=canary_verdict([r["ever_ad"] for r in rows],
                                 [sgn * r[e_ad["readout"]] for r in rows],
                                 m, e_ad["ceiling"], TRIGGER_EVER_AD),
            test2_marked=t2["marked"]["verdict"], test2_unmarked=t2["unmarked"]["verdict"])
        # test 2 must show the flat-elevated signature on the marked group only. An
        # INSUFFICIENT marked curve is a canary that could not be evaluated, which is the one
        # thing a detector's own validation cannot be allowed to pass on.
        t3["verdict"] = t3["test1"]["verdict"]
        if t3["test2_marked"] != FAIL or t3["test2_unmarked"] == FAIL:
            t3["verdict"] = FAIL
        t3["note_test2"] = (f"marked curve {t3['test2_marked']} (FAIL = the planted static "
                            + ("leak was seen; INSUFFICIENT = the canary could not be "
                               "evaluated, which is not a pass" if t3["test2_marked"] == SKIP
                               else "leak was seen")
                            + f"), unmarked curve {t3['test2_unmarked']}")

    ok_self = self_test()

    # ------------------------------------------------------------------ report
    print("\n=== TEST 2 TABLE -- 12-month AD hazard by lead time "
          "(mean over cases; the AUCs are against the matched controls)")
    print(f"  {'lead':>5} {'n':>5} {'cases':>6} {'model P(AD<=1y)':>16} "
          f"{'refit P':>9} {'AUC model':>10} {'AUC refit':>10} {'gap':>8} "
          f"{'gap 95% CI':>17}  status")
    for label, curve in curves.items():
        for c in curve:
            lo, hi = c.get("gap_ci", (float("nan"), float("nan")))
            print(f"  {c['lead_y']:>4.0f}y {c['n_rows']:>5} {c['n_cases']:>6} "
                  f"{c['mean_hazard_cases']:>16.4f} {c['mean_refit_cases']:>9.4f} "
                  f"{c['auc_model']:>10.4f} {c['auc_refit']:>10.4f} {c['gap']:>+8.4f} "
                  f"{f'[{lo:+.3f},{hi:+.3f}]':>17}  "
                  f"{c['status']}" + (f"  [{label}]" if len(curves) > 1 else ""))

    print(f"\n=== VERDICTS  ({a.split})")
    for ep in ("ever_ad", "death"):
        e = t1["endpoints"][ep]
        d = e["delta_vs_ceiling"]
        _line(f"1. baseline prefix, {ep}", e["verdict"],
              f"AUC {e['auc']:.4f} [{e['ci95'][0]:.3f},{e['ci95'][1]:.3f}] "
              f"({e['readout']}, {e['side']}, n={t1['n']}, {e['n_pos']} positive)  "
              f"ceiling {e['ceiling']:.4f} (B2 live {e['ceiling_live']:.4f} / delivered "
              f"{e['ceiling_reference']:.4f})  trigger {e['trigger']:.2f}"
              + (f"\n{'':>15}vs the ceiling on the same subjects: {d['value']:+.4f} "
                 f"[{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}] (paired bootstrap; a FAIL "
                 f"needs this lower bound above zero)" if d["paired"] else "")
              + (f"\n{'':>15}SIGNATURE: {e['signature']}" if e["signature"] else ""))
    if a.canary:
        _line("2. horizon curve (marked)", t2["marked"]["verdict"], t2["marked"]["note"])
        _line("2. horizon curve (unmarked)", t2["unmarked"]["verdict"], t2["unmarked"]["note"])
        _line("3. canary", t3["verdict"], t3["test1"]["note"] + "\n" + " " * 15
              + t3["note_test2"])
    else:
        _line("2. horizon curve", t2["verdict"], t2["note"])
        _line("3. canary (self-test only)", PASS if ok_self else FAIL,
              "verdict functions driven with known-answer scores; the planted-build audit "
              "needs the canary training run")
    print(f"\n  rollout diagnostics: {skipped['endpoint_in_prefix']} subjects dropped for an "
          f"endpoint inside the baseline prefix, {skipped['no_history']} for no history; "
          f"{100 * frac_unterm:.1f}% of trajectories ended with no death token "
          f"(token budget or the age-110 stop), which makes P(death ever) a lower bound")
    print("  the age grid is nominal: visit ages are age_bl + 365*fu_year, ~21 days median "
          "error against the true visit age, so every lead time above is +/- that.")

    verdicts = [("1. baseline prefix", t1["verdict"]),
                ("2. horizon curve" + (" (marked)" if a.canary else ""),
                 t2["marked"]["verdict"] if a.canary else t2["verdict"])]
    if a.canary:
        verdicts.append(("3. canary", t3["verdict"]))
    if not ok_self:
        verdicts.append(("detector self-test", FAIL))
    # INSUFFICIENT is not PASS. A test that could not be read produced no evidence, and an
    # audit that reports no evidence as a clean bill -- and exits 0 -- is worse than no audit.
    unreadable = [n for n, v in verdicts if v == SKIP]
    overall = worst([v for _n, v in verdicts])
    print(f"\n  OVERALL: {overall}")
    if unreadable:
        print(f"  {', '.join(unreadable)} could not be read on this run, which is not a "
              f"pass; the exit code is non-zero for it")

    tag = f"canary-{a.split}" if a.canary else a.split
    out = a.out or os.path.join(_HERE, "eval", f"leakage_{tag}.json")
    art = dict(ckpt=os.path.abspath(a.ckpt), data_dir=os.path.abspath(a.data_dir),
               split=a.split, canary=bool(a.canary), n_mc=a.n_mc, seed=a.seed,
               vocab_size=V.VOCAB_SIZE, self_test_passed=bool(ok_self),
               ceilings={k: ({kk: vv for kk, vv in v.items() if kk != "pred"}
                             if isinstance(v, dict) else v) for k, v in ceilings.items()},
               test1=t1, test2=t2, test3=t3, overall=overall,
               verdicts={n: v for n, v in verdicts}, unreadable=unreadable)
    with open(out, "w") as fh:
        json.dump(art, fh, indent=2, default=float)
    png = out.replace(".json", ".png").replace("leakage_", "leakage_horizon_")
    plot_curve(curves, png)
    print(f"  wrote {out}\n  wrote {png}")
    return 1 if overall == FAIL or unreadable else 0


if __name__ == "__main__":
    sys.exit(main())
