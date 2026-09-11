"""
evaluate.py -- the primary and secondary endpoints for one trained checkpoint.

    python eval/evaluate.py --ckpt out-radc-base-s42/ckpt.pt --split val --n-mc 200

PRIMARY: landmark-and-horizon discrimination for incident AD with death as a COMPETING event.
Every legal landmark visit of every evaluation-cohort subject is scored by truncating the
stream at that visit age and asking engine.risk_by_horizon for P(AD within H years); the label
comes from cohort.label_at_horizon and from nowhere else.

SECONDARY: cumulative-incidence calibration from age-80 prefixes, all-cause mortality
calibration, next-token cross-entropy, and trajectory realism.

--------------------------------------------------------------------------------------------
WHY THIS FILE LOOKS THE WAY IT DOES. Five measured facts, each of which silently breaks a
conventionally-written evaluation of this cohort:

  1. COMPETING RISK IS THE WHOLE GAME. 62.0% of the cohort dies and 53.3% of non-converters
     die. Treating death as censoring inflates cumulative AD incidence by +38.6% at age 85 and
     +59.4% at 90 (Aalen-Johansen 0.1736 / 0.2592 against 1-KM 0.2405 / 0.4132 on the full
     cohort). So: every observed incidence here is engine.aalen_johansen, never 1-KM; and a
     subject who dies inside a horizon without a diagnosis is DROPPED from the discrimination
     denominator by label_at_horizon rather than scored as a clean negative. Counting those
     deaths as negatives is exactly what makes a lethal comorbidity look protective.

     The consequence for the AUC is worth stating rather than hiding: this is the
     cause-specific time-dependent AUC with competing events EXCLUDED from the control set,
     which is the estimand label_at_horizon encodes. It is not the same number as the variant
     that keeps competing deaths as controls, and the two are not interchangeable.

  2. LEFT TRUNCATION IS NOT OPTIONAL. Subjects enter at age_bl (median 78.9 y) and are not at
     risk on the age axis before they enrol, so every age-axis CIF passes entry_age. The
     risk-set size is printed beside every CIF value because the curve rests on 1,619
     subjects at age 85, 1,114 at 90 and 393 at 95 -- an age-95 number is not the same kind of
     object as an age-85 one.

  3. THE HEADLINE NEEDS A LEAD TIME, AND THE SHORT HORIZONS CANNOT HAVE ONE. "5-year-ahead"
     scored at landmarks a month before the diagnosis is not a 5-year-ahead claim, so the
     headline arm requires the landmark to sit >= 2 y before the diagnosis. That same filter
     applied at H = 1 leaves ZERO positives by construction (measured: 1,399 evaluable rows,
     0 positive on val) -- a landmark 2 y clear of the diagnosis cannot convert within 1 y. So
     H = 1 and H = 3 are reported at lead 0 as well, and every arm carries its own evaluable
     positive count. An arm with fewer than MIN_POSITIVES positives is reported as such and
     gets no confidence interval. The lead filter is selection ON THE OUTCOME, which ranking
     is invariant to and absolute calibration is not, so CALIBRATION IS REPORTED AT LEAD 0
     ONLY: on the same val landmark pool the lead-2 rows give an observed AJ at 5 y of 0.115
     against 0.203 unfiltered, because they structurally cannot contain an event before 2 y.

  4. THE LANDMARK IS A VISIT, NOT AN AGE, AND NEVER THE ENDPOINT'S OWN DAY. The 21 static
     tokens sit one day before the baseline visit and occupy their own age, so cohort.landmarks
     is called with min_prior_visits=2: two distinct ages at or before the landmark means the
     static block plus at least one real visit. min_prior_visits=1 would admit the static age
     itself as a "landmark visit". Separately, cohort.landmarks cuts inclusively on the
     un-rounded event age from subjects.csv while the tokenizer places the AD token at
     round(age_ad*365.25) and prefix_at is inclusive, so the diagnosis day itself passes the
     cut for roughly half of converters and the death day always does. Those rows are dropped
     here (61 + 103 of val's 1,898): a prefix containing the endpoint is not a prediction
     problem, and because AD_DX is not repeatable the model scored those 61 certain positives
     at exactly 0.0, which alone moved the lead-0 AUC at H=1 from 0.82 to 0.50.

  5. THE TIMING AXIS IS A PROTOCOL CALENDAR. 93.9% of nominal inter-visit intervals are
     exactly one year because RADC is an annual-visit cohort, and visit ages are reconstructed
     as age_bl + 365*fu_year with a measured ~21-day median error (~8% of last visits slip
     past 6 months). loss_dt is reported and is labelled NOT a timing-calibration result;
     only the AD diagnosis and death sit on a real clock, and 21 days is the accuracy ceiling
     of every timing claim below.

MONTE CARLO AND THE CACHE. Rollouts are the entire cost (~0.2 s per landmark at n_mc=200 on
one CPU core; the val split has ~1,900 landmarks). Every rollout-derived quantity is cached to
a .npz keyed on a rollout-semantics version, the checkpoint's md5, the split, n_mc, the seed,
--limit and the rollout geometry, so re-scoring is seconds. The per-rollout seed is a pure function of (projid,
purpose, landmark index) -- NOT of the worker or of the iteration order -- so --workers and
--limit change the wall clock and nothing else.
"""
import argparse
import json
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

# run as a script (`python eval/evaluate.py`), so the checkout root is not on sys.path yet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radc_delphi import vocab as V                                          # noqa: E402
from radc_delphi.engine import (load, aalen_johansen, risk_set_size,        # noqa: E402
                                first_event_ages, sim_end_ages, Engine,
                                DAYS_PER_YEAR, PAD_AGE)
from eval.cohort import load_subjects, landmarks, label_at_horizon          # noqa: E402

# An AUC on fewer than this many events is noise with a decimal point on it. Reported, but
# without a CI and flagged, rather than dropped -- the count is itself a finding.
MIN_POSITIVES = 10
# Ages at which the age-80 rollouts are compared against the observed CIF.
CIF_AGES = (85.0, 90.0, 95.0)
# Years from baseline for the trajectory-realism panel. Median follow-up is 7 y and p75 is
# 11 y, so beyond ~15 the observed arm is a handful of survivors and says nothing.
TRAJ_YEARS = tuple(range(0, 16))
TRAJ_SCALES = ("MMSE", "COG")
LANDMARK_ORIGIN_AGE = 80.0


# ============================================================================ worker
# One process-global engine per worker. Loading the checkpoint is ~0.1 s and the split .bin
# is 90 KB, so paying it once per worker is cheaper than shipping either through a pipe.
_W = {}


def _init(ckpt, data_dir, radc_dir, split, device, n_mc, seed, horizons, max_new,
          until_age):
    torch.set_num_threads(1)
    eng = load(ckpt, data_dir=data_dir, device=device)
    data, p2i, _ = eng.load_split(split)
    _W.update(eng=eng, data=data, p2i=p2i,
              sub=load_subjects(data_dir, radc_dir=radc_dir), n_mc=n_mc, seed=seed,
              horizons=tuple(horizons), max_new=max_new, until_age=until_age)


def _seed_for(pid, purpose, j=0):
    """Deterministic per-rollout seed. Depends only on the subject and what is being asked,
    never on worker count or iteration order, so a cache built with --workers 8 is identical
    to one built serially."""
    return int((_W["seed"] * 1_000_003 + int(pid) * 9_176 + purpose * 131 + j) % (2 ** 31 - 1))


def _bin_sequence(toks, ages, scale):
    """The subject's emitted bin indices for one ordinal scale, in age order."""
    ids = np.asarray(V.SCALES[scale])
    lut = np.full(V.VOCAB_SIZE, -1, dtype=np.int64)
    lut[ids] = np.arange(len(ids))
    m = np.isin(toks, ids) & (ages > PAD_AGE + 1)
    if not m.any():
        return np.empty(0, dtype=np.int64)
    return lut[toks[m][np.argsort(ages[m], kind="stable")]]


def _round_trips(seq):
    """(n_returns, n_emissions) for one bin sequence.

    A ROUND TRIP is an emission of a level the trajectory has already visited and since left.
    Measured on the emitted data over all 4,428 subjects this definition gives 14.1% of MMSE
    emissions and 20.3% of cogn_global emissions -- the MMSE figure is the 14.1% the
    tokenizer's hysteresis margin was tuned to produce. (The delivered brief also quotes 16.5%
    for cognition, which corresponds to a per-subject-averaged rate rather than a per-emission
    one; both are reported below, but the simulated-vs-observed comparison uses THIS estimator
    on both sides, which is the only thing that makes the comparison mean anything.)

    A model whose simulated rate is far above the observed one is generating oscillating
    dementia: subjects who decline and recover and decline again, which is not what this
    disease does and is a reportable failure.
    """
    n = len(seq)
    if n < 3:
        return 0, n
    r = 0
    seen = set()
    for i in range(n):
        if i and seq[i] in seen and seq[i] != seq[i - 1]:
            r += 1
        seen.add(seq[i])
    return r, n


def _score_subject(k):
    """Every rollout-derived quantity for the k-th subject of the split. None if not scored."""
    eng, sub = _W["eng"], _W["sub"]
    n_mc, hz = _W["n_mc"], _W["horizons"]
    ages, toks, pid = eng.stream(_W["data"], _W["p2i"], k)
    if pid not in sub.index:
        return None
    row = sub.loc[pid]
    if not bool(row["in_eval"]):
        return None
    out = {"pid": int(pid)}

    # ---- A. landmark risks -------------------------------------------------------------
    # min_lead_years=0 gives the SUPERSET of legal landmarks; the >= 2 y lead arm is a mask on
    # these same rows, so both arms reuse one rollout each.
    lms = landmarks(ages, toks, row, min_lead_years=0.0, min_prior_visits=2)
    # A LANDMARK WHOSE PREFIX ALREADY CONTAINS THE ENDPOINT IS NOT A LANDMARK. cohort.landmarks
    # cuts inclusively on the UN-ROUNDED event age from subjects.csv, while the tokenizer puts
    # the AD token at round(age_ad*365.25) and prefix_at is inclusive too, so a converter's own
    # diagnosis day survives the cut for roughly half of them; age_death is death_days/365.25
    # exactly, so a decedent's death day survives it always. Measured on val: 61 of 1,898 rows
    # handed the model a prefix containing the diagnosis and were then labelled 1 -- and since
    # AD_DX is not in vocab.REPEATABLE_TOKENS the rollout can never fire it again, so all 61
    # certain positives scored exactly 0.0 and the lead-0 AUC read 0.50 instead of 0.82 at
    # H=1. The other 103 (death in the prefix) are unscoreable for the AUC but were polluting
    # the whole first decile of the calibration table.
    lms = np.array([lm for lm in lms
                    if not np.isin(toks[ages <= lm], V.ENDPOINT_IDS).any()], dtype=float)
    risk = np.zeros((len(lms), len(hz)), dtype=np.float64)
    for j, lm in enumerate(lms):
        pa, pt = eng.prefix_at(ages, toks, lm)
        assert not np.isin(pt, V.ENDPOINT_IDS).any(), \
            f"subject {pid} landmark {lm:.0f} has an endpoint token in its own prefix"
        # max_new_tokens is forwarded: without it these rollouts silently ran at the engine
        # default whatever the flag said, and the cache key claimed a dependency they lacked.
        r = eng.risk_by_horizon(pt, pa, horizons=hz, n_mc=n_mc, seed=_seed_for(pid, 1, j),
                                max_new_tokens=_W["max_new"])
        risk[j] = [r[h] for h in hz]
    out["lm_day"] = np.asarray(lms, dtype=np.float64)
    out["lm_risk"] = risk

    # ---- B. age-80 prefix, rolled to the end of life -----------------------------------
    # Eligible = enrolled by 80 and still AD-free and alive at 80, which is exactly the risk
    # set the observed landmark-80 Aalen-Johansen curve is built on.
    o80 = LANDMARK_ORIGIN_AGE * DAYS_PER_YEAR
    if float(row["entry_age"]) <= LANDMARK_ORIGIN_AGE < float(row["event_age"]):
        pa, pt = eng.prefix_at(ages, toks, o80)
        if len(np.unique(pa)) >= 2:
            assert not np.isin(pt, [V.AD_DX, V.DEATH]).any(), \
                f"subject {pid} is AD-free and alive at 80 by construction"
            gi, ga = eng.simulate(pt, pa, n_mc=n_mc, until_age_years=_W["until_age"],
                                  max_new_tokens=_W["max_new"], seed=_seed_for(pid, 2))
            out["a80_ad"] = first_event_ages(gi, ga, V.AD_DX).astype(np.float32)
            out["a80_death"] = first_event_ages(gi, ga, V.DEATH).astype(np.float32)
            out["a80_end"] = sim_end_ages(ga).astype(np.float32)

    # ---- C. baseline prefix, for trajectory realism ------------------------------------
    uniq = np.unique(ages)
    if len(uniq) >= 2:
        base_day = float(uniq[1])                      # uniq[0] is the static block
        base_age = base_day / DAYS_PER_YEAR
        pa, pt = eng.prefix_at(ages, toks, base_day)
        gi, ga = eng.simulate(pt, pa, n_mc=n_mc, until_age_years=_W["until_age"],
                              max_new_tokens=_W["max_new"], seed=_seed_for(pid, 3))
        d_end = sim_end_ages(ga)
        d_death = first_event_ages(gi, ga, V.DEATH)
        days = [base_day + y * DAYS_PER_YEAR for y in TRAJ_YEARS]
        tgt = base_age + np.asarray(TRAJ_YEARS, dtype=float)
        # A draw contributes a cognitive state at year y only if it is ALIVE and its rollout
        # actually reached y. A rollout that ran out of max_new_tokens has not been observed
        # to be stable -- reading its carried-forward last bin would manufacture stability.
        alive = np.isnan(d_death)[:, None] | (d_death[:, None] > tgt[None, :])
        reached = d_end[:, None] >= tgt[None, :]
        ok = alive & reached
        out["base_age"] = base_age
        # Observed side of the same panel, through the SAME estimator: the real stream is fed
        # to scale_track as a one-row batch, so "state at age a" has one definition on both
        # sides. A subject contributes an observed state at year y only while still under
        # observation -- cognition is recorded at visits, so age_last_obs is the cut-off, and
        # it is the observed analogue of the rollout budget on the simulated side.
        obs_ok = float(row["age_last_obs"]) >= tgt
        out["traj_obs_ok"] = obs_ok.astype(bool)
        for sc in TRAJ_SCALES:
            b0 = _bin_sequence(pt, pa, sc)
            b0 = int(b0[-1]) if len(b0) else -1
            trk = eng.scale_track(gi, ga, sc, days, baseline_bin=b0)
            out[f"traj_sim_sum_{sc}"] = np.where(ok & (trk >= 0), trk, 0).sum(0).astype(np.float64)
            out[f"traj_sim_n_{sc}"] = (ok & (trk >= 0)).sum(0).astype(np.int64)
            otrk = Engine.scale_track(toks[None, :], ages[None, :], sc, days,
                                      baseline_bin=b0)[0]
            out[f"traj_obs_{sc}"] = np.where(obs_ok & (otrk >= 0), otrk, np.nan).astype(np.float64)
            rn = rd = 0
            for q in range(gi.shape[0]):
                nret, nemit = _round_trips(_bin_sequence(gi[q], ga[q], sc))
                rn += nret
                rd += nemit
            out[f"rt_sim_{sc}"] = (rn, rd)
            out[f"rt_obs_{sc}"] = _round_trips(_bin_sequence(toks, ages, sc))
    return out


# ============================================================================ cache
# Bumped whenever what a rollout MEANS changes, so a cache written by an older copy of this
# file cannot silently reproduce its numbers under an unchanged flag set. v2: landmarks whose
# prefix contains an endpoint are dropped, and --max-new-tokens now actually reaches the
# landmark rollouts.
_ROLLOUT_VERSION = 2


def _cache_path(cache_dir, meta):
    key = "_".join([f"v{_ROLLOUT_VERSION}", meta["ckpt_sig"], meta["split"], f"mc{meta['n_mc']}",
                    f"s{meta['seed']}", f"lim{meta['limit']}",
                    f"h{'-'.join(str(h) for h in meta['horizons'])}",
                    f"mnt{meta['max_new_tokens']}", f"ua{int(meta['until_age'])}"])
    return os.path.join(cache_dir, f"rollouts_{key}.npz")


def _pack(per_subject):
    """Ragged per-subject results -> flat arrays a .npz can hold."""
    z = {}
    lm_pid, lm_day, lm_risk = [], [], []
    for r in per_subject:
        lm_pid += [r["pid"]] * len(r["lm_day"])
        lm_day.append(r["lm_day"])
        lm_risk.append(r["lm_risk"])
    z["lm_pid"] = np.asarray(lm_pid, dtype=np.int64)
    z["lm_day"] = np.concatenate(lm_day) if lm_day else np.zeros(0)
    z["lm_risk"] = (np.concatenate([x for x in lm_risk if len(x)]) if lm_pid
                    else np.zeros((0, 0)))

    a80 = [r for r in per_subject if "a80_ad" in r]
    z["a80_pid"] = np.asarray([r["pid"] for r in a80], dtype=np.int64)
    for f in ("ad", "death", "end"):
        z[f"a80_{f}"] = (np.stack([r[f"a80_{f}"] for r in a80]) if a80
                         else np.zeros((0, 0), dtype=np.float32))

    tr = [r for r in per_subject if "base_age" in r]
    z["traj_pid"] = np.asarray([r["pid"] for r in tr], dtype=np.int64)
    z["traj_base_age"] = np.asarray([r["base_age"] for r in tr], dtype=np.float64)
    z["traj_obs_ok"] = (np.stack([r["traj_obs_ok"] for r in tr]) if tr
                        else np.zeros((0, len(TRAJ_YEARS)), bool))
    for sc in TRAJ_SCALES:
        for f in (f"traj_sim_sum_{sc}", f"traj_sim_n_{sc}", f"traj_obs_{sc}"):
            z[f] = (np.stack([r[f] for r in tr]) if tr
                    else np.zeros((0, len(TRAJ_YEARS))))
        for f in (f"rt_sim_{sc}", f"rt_obs_{sc}"):
            z[f] = (np.asarray([r[f] for r in tr], dtype=np.int64) if tr
                    else np.zeros((0, 2), dtype=np.int64))
    return z


# ============================================================================ statistics
def _auc(y, s):
    """Rank-based AUC with ties averaged. NaN if either class is empty."""
    y = np.asarray(y, dtype=int)
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.asarray(s, dtype=float))
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def _cluster_boot_auc(y, s, pid, n_boot, rng):
    """Subject-clustered percentile CI. Resampling ROWS would treat a subject's 5 landmarks as
    5 independent observations; they are one person measured five times, and the interval
    would come out roughly sqrt(5) too narrow."""
    if n_boot <= 0:
        return float("nan"), float("nan")
    uniq, inv = np.unique(pid, return_inverse=True)
    idx_by_sub = [np.where(inv == i)[0] for i in range(len(uniq))]
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        ii = np.concatenate([idx_by_sub[p] for p in pick])
        a = _auc(y[ii], s[ii])
        if not np.isnan(a):
            vals.append(a)
    if len(vals) < n_boot // 2:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def _landmark_cif(t_years, etype, at):
    """Aalen-Johansen CIF of AD at `at` years past the landmark, competing death handled.

    NaN when nobody in the row set is still at risk at `at`. Past the last exit the estimator
    carries its last jump forward and prints a confident cumulative incidence resting on zero
    people -- the same trap _cif_at guards on the age axis. It does not fire on the val
    deciles (the smallest bin has 24 at risk at H) but a finer n_bins or a small --limit
    reaches it, and this is the function behind every "AJ observed" number printed.
    """
    if not len(t_years) or risk_set_size(np.zeros(len(t_years)), t_years, at) == 0:
        return float("nan")
    cif, _ = aalen_johansen(t_years, etype, np.zeros(len(t_years)), [at])
    return float(cif[0])


def _cif_at(event_age, event_type, entry_age, at):
    """(CIF, risk-set size) at one age, with the CIF suppressed to NaN when nobody is left.

    Past the last subject at risk the estimator simply carries its last jump forward, which
    prints as a confident number resting on zero people -- the age-95 trap this cohort sets
    (393 at risk at 95 against 1,619 at 85). Returning NaN there is the only honest option,
    and the count is returned beside it so it is never quoted alone.
    """
    n = risk_set_size(entry_age, event_age, at)
    if n == 0:
        return float("nan"), 0
    cif, _ = aalen_johansen(event_age, event_type, entry_age, [at])
    return float(cif[0]), n


def _calibration(pred, t_years, etype, y_eval, p_eval, horizon, n_mc, n_bins=10):
    """Deciles of predicted risk against the Aalen-Johansen observed rate, plus slope and
    intercept and Brier.

    Called on the LEAD-0 arms only -- see primary(); under a lead filter the rows are selected
    on the outcome and none of these quantities means what it says.

    The DECILE table uses every landmark row in the arm, including the ones the discrimination
    analysis had to drop: Aalen-Johansen is exactly the estimator that can use a subject who
    died at 2.3 y inside a 5 y horizon, and throwing those rows away here would bias the
    observed rate upwards in precisely the deciles that contain the frailest subjects.

    The SLOPE/INTERCEPT and BRIER are complete-case, on the binary evaluable rows only -- both
    need a 0/1 label. That is a real limitation, stated rather than papered over: they are
    conditioned on surviving the horizon, so they answer a slightly narrower question than the
    decile table does.

    Bins are cut on RANKS, not on values. Monte-Carlo risk at n_mc draws is a lattice with a
    heavy atom at exactly 0, so value-based deciles collapse to two or three unequal bins.
    """
    n = len(pred)
    out = {"n_rows": int(n), "bins": [], "n_bins": 0,
           "mean_abs_calibration_error": float("nan"),
           "mean_predicted_all_rows": float("nan"),
           "aj_observed_all_rows": float("nan")}
    if n:
        order = rankdata(pred, method="ordinal")
        b = np.minimum(((order - 1) * n_bins // n).astype(int), n_bins - 1)
        for g in range(n_bins):
            m = b == g
            if not m.any():
                continue
            out["bins"].append({
                "bin": g + 1,
                "n": int(m.sum()),
                "mean_pred": float(pred[m].mean()),
                "aj_observed": _landmark_cif(t_years[m], etype[m], horizon),
                "n_ad_within_h": int(((etype[m] == 1) & (t_years[m] <= horizon)).sum()),
                "n_death_within_h": int(((etype[m] == 2) & (t_years[m] <= horizon)).sum()),
                "n_at_risk_at_h": risk_set_size(np.zeros(int(m.sum())), t_years[m], horizon),
            })
        out["n_bins"] = len(out["bins"])
        w = np.array([bb["n"] for bb in out["bins"]], dtype=float)
        d = np.array([abs(bb["mean_pred"] - bb["aj_observed"]) for bb in out["bins"]])
        ok = np.isfinite(d)                      # a bin with nobody at risk at H has no gap
        if ok.any():
            out["mean_abs_calibration_error"] = float((w[ok] * d[ok]).sum() / w[ok].sum())
        out["mean_predicted_all_rows"] = float(pred.mean())
        out["aj_observed_all_rows"] = _landmark_cif(t_years, etype, horizon)

    out["brier"] = float("nan")
    out["brier_null"] = float("nan")
    out["cal_slope"] = float("nan")
    out["cal_intercept"] = float("nan")
    if len(y_eval) and 0 < y_eval.sum() < len(y_eval):
        out["brier"] = float(np.mean((p_eval - y_eval) ** 2))
        out["brier_null"] = float(np.mean((y_eval.mean() - y_eval) ** 2))
        # Monte Carlo puts an atom at exactly 0 and at exactly 1; the logit needs them off the
        # boundary, and 1/(2*n_mc) is the resolution the estimator actually has.
        eps = 1.0 / (2.0 * max(n_mc, 1))
        p = np.clip(p_eval, eps, 1 - eps)
        x = np.log(p / (1 - p)).reshape(-1, 1)
        # C=1e12 rather than penalty=None: the latter is deprecated in sklearn >= 1.8 and
        # absent before 1.2, and at this C the l2 term is numerically inert either way.
        lr = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000).fit(x, y_eval)
        out["cal_slope"] = float(lr.coef_[0, 0])
        out["cal_intercept"] = float(lr.intercept_[0])
    return out


def _no_calibration(n_rows, reason):
    """The calibration block for an arm where absolute calibration is not a defined question.
    Same keys, all NaN, plus the reason -- so the artifact says why rather than going silent."""
    return {"n_rows": int(n_rows), "bins": [], "n_bins": 0,
            "mean_abs_calibration_error": float("nan"),
            "mean_predicted_all_rows": float("nan"),
            "aj_observed_all_rows": float("nan"),
            "brier": float("nan"), "brier_null": float("nan"),
            "cal_slope": float("nan"), "cal_intercept": float("nan"),
            "suppressed": reason}


# ============================================================================ endpoints
def primary(z, sub, horizons, min_lead, n_boot, n_mc, rng):
    """Landmark-and-horizon discrimination, one arm per (horizon, lead)."""
    pid = z["lm_pid"]
    day = z["lm_day"]
    risk = z["lm_risk"].reshape(len(pid), -1)
    if not len(pid):
        return []
    rows = sub.loc[pid]
    ev_type = rows["event_type"].to_numpy(int)
    ev_age = rows["event_age"].to_numpy(float)
    lm_y = day / DAYS_PER_YEAR
    t_years = ev_age - lm_y                      # time from landmark to event / censoring
    leads = sorted({0.0, float(min_lead)})
    arms = []
    for hi, H in enumerate(horizons):
        for lead in leads:
            keep = np.ones(len(pid), bool) if lead == 0 else \
                ~((ev_type == 1) & (day > (ev_age - lead) * DAYS_PER_YEAR))
            p = risk[keep, hi]
            tt, ee, pp = t_years[keep], ev_type[keep], pid[keep]
            lab = np.array([label_at_horizon(sub.loc[q], d, H)
                            for q, d in zip(pp, day[keep])], dtype=object)
            # dtype=bool explicitly: for an arm whose keep mask selects nothing this is an
            # empty list, which numpy types float64, and the next line then raises IndexError
            # after the whole rollout pass has been paid for. 16 of the 378 val subjects have
            # every landmark inside the 2 y lead, so a --limit run confined to them hits it.
            m = np.array([v is not None for v in lab], dtype=bool)
            y = lab[m].astype(int)
            s = p[m]
            npos = int(y.sum())
            a = {"horizon_years": float(H), "min_lead_years": float(lead),
                 "n_subjects": int(len(np.unique(pp))),
                 "n_subjects_evaluable": int(len(np.unique(pp[m]))),
                 "n_landmarks": int(keep.sum()),
                 "n_evaluable": int(m.sum()),
                 "n_evaluable_positive": npos,
                 "n_evaluable_negative": int(m.sum() - npos),
                 "n_dropped_competing_or_censored": int((~m).sum()),
                 "evaluable_prevalence": float(y.mean()) if len(y) else float("nan"),
                 "auc": _auc(y, s) if npos else float("nan"),
                 "auc_ci95": [float("nan"), float("nan")],
                 "auc_reportable": npos >= MIN_POSITIVES}
            if npos >= MIN_POSITIVES:
                a["auc_ci95"] = list(_cluster_boot_auc(y, s, pp[m], n_boot, rng))
            # CALIBRATION ONLY AT LEAD 0. The lead filter selects on the outcome: it removes
            # exactly the converter rows whose diagnosis falls inside the first `lead` years
            # (val: 198 rows have an AD event at t <= 2 y and none survive the lead-2 filter).
            # Ranking is invariant to that -- it is the arm's estimand -- but every absolute
            # calibration quantity is not, because the predicted P(AD within H) is
            # unconditioned while the observed side structurally cannot contain an event
            # before `lead`: the same landmark pool reads AJ@5 = 0.115 filtered against 0.203
            # unfiltered. Predicting the window P(AD in (lead, H]) instead would not fix it
            # either, since the filtered AJ is conditioned on being AD-free at `lead` and the
            # window probability is not. So the arm reports discrimination and nothing else.
            a["calibration"] = (
                _calibration(p, tt, ee, y, s, float(H), n_mc) if lead == 0 else
                _no_calibration(len(p), "lead filter selects on the outcome; calibration is "
                                        "reported on the lead-0 arm at the same horizon"))
            arms.append(a)
    return arms


def cif_calibration(z, sub_split, sub_all, split):
    """Simulated AD cumulative incidence from age-80 prefixes against Aalen-Johansen."""
    ad, dth = z["a80_ad"], z["a80_death"]
    n_sub, n_mc = ad.shape if ad.size else (0, 0)
    pid80 = z["a80_pid"]
    r80 = sub_all.loc[pid80] if len(pid80) else sub_all.iloc[:0]
    ev = sub_split[sub_split["in_eval"]]
    # The marginal comparator is outcome data only -- no model input, no tuning target -- but
    # on a val run the whole-cohort version is still a read of the 744 held-out test subjects'
    # outcomes (0.1774 / 0.2809 with them, 0.1700 / 0.2736 without). It is restricted to the
    # non-test splits outside the once-only test run, and the population is printed with it.
    if split == "test":
        evall = sub_all[sub_all["in_eval"]]
        pop = "whole evaluation cohort, all splits"
    else:
        evall = sub_all[sub_all["in_eval"] & (sub_all["split"] != "test")]
        pop = "evaluation cohort, train+val only (test-split outcomes held out)"
    out = {"n_subjects_age80": int(n_sub), "n_mc": int(n_mc), "aj_all_eval_population": pop,
           "n_draws": int(n_sub * n_mc), "origin_age": LANDMARK_ORIGIN_AGE, "ages": {}}
    end = z["a80_end"]
    for a in CIF_AGES:
        # ad <= dth, not <: generate() draws a whole visit at one age, so 1.5% of the val
        # age-80 draws emit the diagnosis and the death on the same simulated day. The
        # tokenizer sorts the diagnosis first inside a tie and forces death strictly after
        # everything else, so the observed side has no ties at all -- scoring a simulated tie
        # as a competing death would put the two sides on different conventions and cost
        # 3.5-4.8% of the simulated CIF.
        got = (ad <= a) & (np.isnan(dth) | (ad <= dth)) if n_sub else np.zeros((0, 0), bool)
        row = {"simulated_cif": float(np.nan_to_num(got, nan=0.0).mean()) if n_sub
               else float("nan")}
        # A rollout that exhausted max_new_tokens before age `a` without dying is being
        # counted as a non-event here. That biases the simulated CIF DOWN, so the size of it
        # is reported rather than assumed negligible.
        row["frac_draws_truncated_before_age"] = float(
            (np.isnan(dth) & (end < a)).mean()) if n_sub else float("nan")
        # like-for-like observed: the SAME subjects, left-truncated at 80, which is the
        # population the simulation draws from.
        if len(pid80):
            c, n = _cif_at(r80["event_age"], r80["event_type"],
                           np.full(len(r80), LANDMARK_ORIGIN_AGE), a)
            row["observed_aj_landmark80"], row["risk_set_landmark80"] = c, n
        # marginal observed on the split's evaluation cohort, entry at age_bl
        c, n = _cif_at(ev["event_age"], ev["event_type"], ev["entry_age"], a)
        row["observed_aj_split_eval"], row["risk_set_split_eval"] = c, n
        # and on the whole evaluation cohort -- the 0.1774 / 0.2809 the brief quotes. A pure
        # property of the outcome data, recomputed here rather than trusted, and independent
        # of which split the model was scored on.
        c, n = _cif_at(evall["event_age"], evall["event_type"], evall["entry_age"], a)
        row["observed_aj_all_eval"], row["risk_set_all_eval"] = c, n
        out["ages"][str(a)] = row
    return out


def mortality(z, sub_split, sub_all):
    """All-cause mortality: simulated P(death by age X) and the simulated death-age law."""
    dth = z["a80_death"]
    pid80 = z["a80_pid"]
    r80 = sub_all.loc[pid80] if len(pid80) else sub_all.iloc[:0]
    ev = sub_split[sub_split["in_eval"]]
    out = {"n_subjects_age80": int(len(pid80)), "ages": {}}
    for a in CIF_AGES:
        row = {"simulated_death_cif": float((np.nan_to_num(dth, nan=1e9) <= a).mean())
               if dth.size else float("nan")}
        if len(pid80):
            died = r80["age_death"].notna().to_numpy()
            eage = np.where(died, r80["age_death"].to_numpy(float),
                            r80["age_last_obs"].to_numpy(float))
            c, n = _cif_at(eage, died.astype(int),
                           np.full(len(r80), LANDMARK_ORIGIN_AGE), a)
            row["observed_death_cif_landmark80"], row["risk_set_landmark80"] = c, n
        died = ev["age_death"].notna().to_numpy()
        eage = np.where(died, ev["age_death"].to_numpy(float),
                        ev["age_last_obs"].to_numpy(float))
        c, n = _cif_at(eage, died.astype(int), ev["entry_age"].to_numpy(float), a)
        row["observed_death_cif_split_eval"], row["risk_set_split_eval"] = c, n
        out["ages"][str(a)] = row

    def law(x, tag):
        x = np.asarray(x, float)
        x = x[np.isfinite(x)]
        if not len(x):
            return {"n": 0}
        return {"n": int(len(x)), "median": float(np.median(x)),
                "p10": float(np.percentile(x, 10)), "p90": float(np.percentile(x, 90)),
                "frac_ge_90": float((x >= 90).mean()), "source": tag}

    out["simulated_death_age"] = law(dth.ravel() if dth.size else [],
                                     "rollouts from age-80 prefixes, deaths only")
    # The simulated law is over draws that DIED inside the rollout budget. Any draw that ran
    # out of tokens first is missing from it, which truncates the right tail; the fraction is
    # the number to read before believing the p90.
    out["simulated_death_age"]["frac_draws_never_died_in_budget"] = (
        float(np.isnan(dth).mean()) if dth.size else float("nan"))
    obs80 = r80["age_death"].dropna().to_numpy(float) if len(pid80) else np.zeros(0)
    out["observed_death_age_landmark80"] = law(obs80, "same subjects, observed age_death")
    out["emitted_death_age_all"] = law(sub_all["age_death"].dropna().to_numpy(float),
                                       "every Death token in the built dataset")
    out["note"] = ("the simulated law is conditional on being alive and AD-free at 80 and on "
                   "the rollout budget, so it is comparable to observed_death_age_landmark80; "
                   "emitted_death_age_all is the cohort-wide reference (median 89.7, p10 80.5, "
                   "p90 97.3, 46.2% at 90+) and is NOT the same conditioning")
    return out


def trajectory(z):
    """Simulated vs observed cognitive trajectory and round-trip rate."""
    out = {"n_subjects": int(len(z["traj_pid"])), "years": list(TRAJ_YEARS), "scales": {}}
    for sc in TRAJ_SCALES:
        ssum = z[f"traj_sim_sum_{sc}"]
        sn = z[f"traj_sim_n_{sc}"]
        obs = z[f"traj_obs_{sc}"]
        sim_mean = np.where(sn.sum(0) > 0, ssum.sum(0) / np.maximum(sn.sum(0), 1), np.nan)
        # CASE MIX. The simulated arm above pools every surviving draw of every rolled
        # subject, while the observed arm is over the subjects still under observation at that
        # year -- 56 long-followed subjects at year 15 against draws from all 378. About a
        # third of the apparent late divergence is that difference and not model error (MMSE
        # year 15: 2.07 pooled, 2.19 on the same subjects, 2.45 observed), so the matched arm
        # is reported beside it rather than replacing it.
        avail = np.isfinite(obs) if obs.size else np.zeros_like(sn, bool)
        msum, mn = np.where(avail, ssum, 0.0).sum(0), np.where(avail, sn, 0).sum(0)
        sim_mean_matched = np.where(mn > 0, msum / np.maximum(mn, 1), np.nan)
        with np.errstate(invalid="ignore"):
            obs_mean = np.nanmean(obs, axis=0) if obs.size else np.full(len(TRAJ_YEARS), np.nan)
        rs, ro = z[f"rt_sim_{sc}"], z[f"rt_obs_{sc}"]
        out["scales"][sc] = {
            "n_bins": len(V.SCALES[sc]),
            "bin_order": "index 0 = worst bin (vocab.SCALES order)",
            "curve": [{"year": int(y),
                       "sim_mean_bin": float(sim_mean[i]),
                       "sim_mean_bin_matched": float(sim_mean_matched[i]),
                       "sim_n_draws": int(sn.sum(0)[i]),
                       "obs_mean_bin": float(obs_mean[i]),
                       "obs_n_subjects": int(np.isfinite(obs[:, i]).sum()) if obs.size else 0}
                      for i, y in enumerate(TRAJ_YEARS)],
            "round_trip_rate_simulated": float(rs[:, 0].sum() / max(rs[:, 1].sum(), 1))
            if rs.size else float("nan"),
            "round_trip_rate_observed": float(ro[:, 0].sum() / max(ro[:, 1].sum(), 1))
            if ro.size else float("nan"),
            "round_trip_emissions_simulated": int(rs[:, 1].sum()) if rs.size else 0,
            "round_trip_emissions_observed": int(ro[:, 1].sum()) if ro.size else 0,
        }
    out["round_trip_definition"] = (
        "returns to an already-visited, since-left level, per emitted scale token; on the "
        "whole built dataset this is 14.1% for MMSE and 20.3% for cogn_global")
    return out


# ============================================================================ reporting
def _clean(o):
    """numpy scalars -> python, non-finite floats -> null.

    json.dump happily writes bare NaN, which python reads back and every other JSON parser
    rejects. An arm with too few positives legitimately has no AUC, so the artifact has to be
    able to say so in a way a downstream reader can load.
    """
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    return o


def _f(x, n=4):
    return "   n/a" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{n}f}"


def report(res):
    m = res["meta"]
    print("=" * 96)
    print(f"  RADC-Delphi evaluation   ckpt {os.path.basename(m['ckpt'])} [{m['ckpt_sig']}]"
          f"   split={m['split']}   n_mc={m['n_mc']}   seed={m['seed']}")
    print(f"  evaluation cohort in split: {m['n_eval_subjects']} subjects, "
          f"{m['n_eval_converters']} converters   (limit={m['limit']})")
    print("=" * 96)

    print("\nPRIMARY -- incident AD, cause-specific, death competing. Landmark = a visit; "
          "\n           controls are horizon survivors only (competing deaths dropped, not "
          "scored 0).")
    print(f"  {'H':>3} {'lead':>5} {'subj':>5} {'subjE':>6} {'lmarks':>7} {'evalbl':>7} "
          f"{'pos':>5} "
          f"{'prev':>7} {'AUC':>7} {'95% CI':>17} {'Brier':>7} {'Brier0':>7} "
          f"{'slope':>7} {'icept':>7}")
    for a in res["primary"]:
        c = a["calibration"]
        ci = ("  " + " " * 15 if not a["auc_reportable"]
              else f"[{a['auc_ci95'][0]:.3f}, {a['auc_ci95'][1]:.3f}]")
        flag = "" if a["auc_reportable"] else f"  <-- only {a['n_evaluable_positive']} positives"
        print(f"  {a['horizon_years']:>3.0f} {a['min_lead_years']:>5.1f} {a['n_subjects']:>5d} "
              f"{a['n_subjects_evaluable']:>6d} {a['n_landmarks']:>7d} {a['n_evaluable']:>7d} "
              f"{a['n_evaluable_positive']:>5d} "
              f"{_f(a['evaluable_prevalence'],4):>7} {_f(a['auc'],4):>7} {ci:>17} "
              f"{_f(c['brier'],4):>7} {_f(c['brier_null'],4):>7} "
              f"{_f(c['cal_slope'],3):>7} {_f(c['cal_intercept'],3):>7}{flag}")
    hl = res["meta"]["headline"]
    print(f"  headline arm: H={hl[0]:.0f} y at landmarks >= {hl[1]:.0f} y before diagnosis")
    print("  subj = subjects supplying landmarks to the arm; subjE = subjects the AUC and its "
          "clustered CI\n  actually rest on, after competing deaths and censored rows are "
          "dropped.")

    # The decile table is the LEAD-0 arm at the headline horizon, even though the headline
    # AUC is the lead-2 arm: the lead filter drops converter rows by their outcome, so an
    # unconditioned predicted risk would be scored against an observed incidence that cannot
    # contain an event in the first 2 y.
    for a in res["primary"]:
        if [a["horizon_years"], a["min_lead_years"]] != [hl[0], 0.0]:
            continue
        c = a["calibration"]
        print(f"\n  calibration deciles, H={a['horizon_years']:.0f} lead={a['min_lead_years']:.0f}"
              f"  (all {c['n_rows']} landmark rows; observed = Aalen-Johansen from the landmark)")
        if hl[1] > 0:
            print(f"    NOTE: lead 0, while the headline AUC above is the lead-{hl[1]:.0f} arm. "
                  "The lead filter selects on\n    the outcome, which the ranking is invariant "
                  "to and an absolute calibration is not.")
        print(f"    {'bin':>4} {'n':>5} {'mean pred':>10} {'AJ observed':>12} {'AD<=H':>6} "
              f"{'death<=H':>9} {'at risk@H':>10}")
        for b in c["bins"]:
            print(f"    {b['bin']:>4d} {b['n']:>5d} {b['mean_pred']:>10.4f} "
                  f"{_f(b['aj_observed']):>12} {b['n_ad_within_h']:>6d} "
                  f"{b['n_death_within_h']:>9d} {b['n_at_risk_at_h']:>10d}")
        print(f"    overall  mean predicted {_f(c['mean_predicted_all_rows'])}   "
              f"AJ observed {_f(c['aj_observed_all_rows'])}   "
              f"mean |gap| {_f(c['mean_abs_calibration_error'])}")

    s = res["secondary"]
    cc = s["cif_calibration"]
    print(f"\nSECONDARY 1 -- AD cumulative incidence from age-{cc['origin_age']:.0f} prefixes "
          f"({cc['n_subjects_age80']} subjects x {cc['n_mc']} draws)")
    print(f"  {'age':>4} {'simulated':>10} | {'AJ from 80':>11} {'n@risk':>7} | "
          f"{'AJ split':>9} {'n@risk':>7} | {'AJ all-eval':>12} {'n@risk':>7}")
    for a, r in cc["ages"].items():
        print(f"  {float(a):>4.0f} {_f(r['simulated_cif']):>10} | "
              f"{_f(r.get('observed_aj_landmark80')):>11} {r.get('risk_set_landmark80', 0):>7d} | "
              f"{_f(r['observed_aj_split_eval']):>9} {r['risk_set_split_eval']:>7d} | "
              f"{_f(r['observed_aj_all_eval']):>12} {r['risk_set_all_eval']:>7d}")
    tr80 = [r["frac_draws_truncated_before_age"] for r in cc["ages"].values()]
    print(f"  draws whose rollout ran out before the age (counted as non-events): "
          f"{', '.join(_f(t, 3) for t in tr80)}")
    print("  'AJ from 80' is the like-for-like comparator: same subjects, left-truncated at 80.")
    print(f"  'AJ all-eval' population: {cc['aj_all_eval_population']} "
          f"(the brief's reference, all splits, is 0.1774 / 0.2809).")

    mo = s["mortality"]
    print("\nSECONDARY 2 -- all-cause mortality from the same age-80 prefixes")
    print(f"  {'age':>4} {'simulated':>10} | {'observed from 80':>17} {'n@risk':>7} | "
          f"{'observed split':>15} {'n@risk':>7}")
    for a, r in mo["ages"].items():
        print(f"  {float(a):>4.0f} {_f(r['simulated_death_cif']):>10} | "
              f"{_f(r.get('observed_death_cif_landmark80')):>17} "
              f"{r.get('risk_set_landmark80', 0):>7d} | "
              f"{_f(r['observed_death_cif_split_eval']):>15} {r['risk_set_split_eval']:>7d}")
    print(f"  {'death-age law':>28} {'n':>6} {'median':>8} {'p10':>8} {'p90':>8} {'>=90':>7}")
    for k in ("simulated_death_age", "observed_death_age_landmark80", "emitted_death_age_all"):
        d = mo[k]
        if not d.get("n"):
            continue
        print(f"  {k:>28} {d['n']:>6d} {d['median']:>8.1f} {d['p10']:>8.1f} "
              f"{d['p90']:>8.1f} {d['frac_ge_90']:>7.3f}")
    print(f"  draws that never died inside the rollout budget (absent from the simulated "
          f"law): {_f(mo['simulated_death_age'].get('frac_draws_never_died_in_budget'), 3)}")

    ll = s["likelihood"]
    print(f"\nSECONDARY 3 -- next-token cross-entropy on {m['split']} "
          f"({ll['n_subjects']} subjects, full deterministic pass, validation_loss_mode)")
    print(f"  loss_ce  {ll['loss_ce']:.4f}   <- the next-token endpoint")
    print(f"  loss_dt  {ll['loss_dt']:.4f}   <- NOT a timing-calibration result: 93.9% of "
          "intervals are exactly 1.000 y")
    print(f"  total    {ll['loss_total']:.4f}")

    tr = s["trajectory"]
    print(f"\nSECONDARY 4 -- trajectory realism, {tr['n_subjects']} subjects rolled from their "
          "baseline visit")
    for sc, d in tr["scales"].items():
        print(f"  {sc}  (bin index 0 = worst of {d['n_bins']})")
        print(f"    {'yr':>3} " + " ".join(f"{y:>6d}" for y in tr["years"]))
        print(f"    {'sim':>3} " + " ".join(f"{c['sim_mean_bin']:>6.2f}" for c in d["curve"]))
        print(f"    {'simM':>3} "
              + " ".join(f"{c['sim_mean_bin_matched']:>6.2f}" for c in d["curve"]))
        print(f"    {'obs':>3} " + " ".join(f"{c['obs_mean_bin']:>6.2f}" for c in d["curve"]))
        print(f"    {'n_o':>3} " + " ".join(f"{c['obs_n_subjects']:>6d}" for c in d["curve"]))
        print(f"    simM = the same draws restricted to the subjects contributing an "
              f"observed value that\n           year; sim-vs-simM is the case-mix part of the "
              f"gap, sim/obs alone conflates it.")
        print(f"    round-trip rate  simulated {d['round_trip_rate_simulated']:.4f} "
              f"({d['round_trip_emissions_simulated']} emissions)   observed "
              f"{d['round_trip_rate_observed']:.4f} ({d['round_trip_emissions_observed']})")
    print("\n  TIMING CAVEAT, once and prominently: visit ages are age_bl + 365*fu_year, "
          "median\n  error ~21 days against the true visit age and ~8% of last visits slip "
          "past 6 months.\n  That is the accuracy ceiling of every year-resolution number "
          "above.")
    print("=" * 96)


# ============================================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data-dir", default=os.environ.get("RADC_DATA_DIR", "data/radc-s42"))
    ap.add_argument("--radc-dir", default="data/RADC",
                    help="raw RADC files; needed for the baseline-impairment flag")
    ap.add_argument("--split", default="val", choices=["train", "val", "test"],
                    help="DEFAULT val. The test split is scored only when named explicitly.")
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="score at most N subjects (0 = all)")
    ap.add_argument("--horizons", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    ap.add_argument("--min-lead", type=float, default=2.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=min(cpu_count(), 8),
                    help="0 = serial (deterministic seeds make the result identical either way)")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--until-age", type=float, default=110.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cache-dir", default="eval/cache")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.split == "test":
        print("!" * 96)
        print("!! SCORING THE TEST SPLIT. This is the once-only, everything-frozen run.")
        print("!" * 96)

    t0 = time.time()
    eng = load(a.ckpt, data_dir=a.data_dir, device=a.device)
    sub_all = load_subjects(a.data_dir, radc_dir=a.radc_dir)
    sub_split = sub_all[sub_all["split"] == a.split]
    ev = sub_split[sub_split["in_eval"]]

    meta = {"ckpt": a.ckpt, "ckpt_sig": eng.ckpt_sig, "split": a.split, "n_mc": a.n_mc,
            "seed": a.seed, "limit": a.limit, "horizons": list(a.horizons),
            "min_lead_years": a.min_lead, "max_new_tokens": a.max_new_tokens,
            "until_age": a.until_age, "n_boot": a.n_boot,
            "data_dir": a.data_dir, "iter_num": int(eng.ckpt.get("iter_num", -1)),
            "n_split_subjects": int(len(sub_split)),
            "n_eval_subjects": int(len(ev)),
            "n_eval_converters": int((ev["event_type"] == 1).sum()),
            "headline": [5.0 if 5.0 in a.horizons else float(max(a.horizons)),
                         float(a.min_lead)]}

    os.makedirs(a.cache_dir, exist_ok=True)
    cpath = _cache_path(a.cache_dir, meta)
    if os.path.exists(cpath) and not a.no_cache:
        z = dict(np.load(cpath, allow_pickle=False))
        print(f"[cache] rollouts loaded from {cpath}")
    else:
        data, p2i, _ = eng.load_split(a.split)
        # --limit is applied to the eligible subject LIST, not by stopping the loop early:
        # imap_unordered returns in completion order, so an early break would score a
        # different 12 subjects under --workers 4 than under --workers 0.
        pids = data[p2i[:, 0], 0].astype(np.int64)
        keep = set(ev.index.astype(np.int64))
        ks = [k for k in range(len(p2i)) if int(pids[k]) in keep]
        if a.limit:
            ks = ks[:a.limit]
        args = (a.ckpt, a.data_dir, a.radc_dir, a.split, a.device, a.n_mc, a.seed,
                tuple(a.horizons), a.max_new_tokens, a.until_age)
        got, done = [], 0
        if a.workers > 0:
            with Pool(a.workers, initializer=_init, initargs=args) as pool:
                it = pool.imap_unordered(_score_subject, ks, chunksize=4)
                for r in it:
                    done += 1
                    if r is not None:
                        got.append(r)
                    if done % 50 == 0:
                        print(f"  ... {done}/{len(ks)} subjects, {time.time() - t0:.0f}s",
                              flush=True)
        else:
            _init(*args)
            for k in ks:
                r = _score_subject(k)
                done += 1
                if r is not None:
                    got.append(r)
                if done % 50 == 0:
                    print(f"  ... {done}/{len(ks)} subjects, {time.time() - t0:.0f}s",
                          flush=True)
        got.sort(key=lambda r: r["pid"])          # imap_unordered; make the cache canonical
        z = _pack(got)
        z["meta_json"] = np.array(json.dumps(meta))
        np.savez_compressed(cpath, **z)
        print(f"[cache] rollouts written to {cpath}  ({time.time() - t0:.0f}s)")

    rng = np.random.default_rng(a.seed)
    res = {"meta": meta,
           "primary": primary(z, sub_all, a.horizons, a.min_lead, a.n_boot, a.n_mc, rng),
           "secondary": {
               "cif_calibration": cif_calibration(z, sub_split, sub_all, a.split),
               "mortality": mortality(z, sub_split, sub_all),
               "likelihood": eng.split_loss(a.split),
               "trajectory": trajectory(z),
           }}
    meta["scored_subjects"] = int(len(np.unique(z["lm_pid"])))
    meta["n_landmarks"] = int(len(z["lm_pid"]))
    meta["wall_seconds"] = round(time.time() - t0, 1)

    out = a.out or f"eval/results_{a.split}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as fh:
        json.dump(_clean(res), fh, indent=2)
    report(res)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
