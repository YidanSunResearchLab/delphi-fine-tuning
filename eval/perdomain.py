"""
perdomain.py -- discrimination for EVERY clinical outcome the model predicts, not just AD.

    python -m eval.perdomain --ckpt out-radc-v3-final/ckpt.pt --split test --n-mc 200

WHY THIS EXISTS. The model is trained to predict 56 tokens. eval/evaluate.py scores one of
them thoroughly (the AD diagnosis), plus death, plus an aggregate next-token cross-entropy
that says nothing about which outcome the model is good at. Everything in between -- four
incident comorbidities, stroke and depression onset, four medication events, and worsening on
the two cognitive scales -- carries gradient during training and was never scored at all.

That is not a new observation. The NACC arm this pipeline replaced had exactly this module and
its docstring made exactly this complaint; the RADC rebuild deleted it and did not put it back,
so the same gap reopened.

WHAT IT COSTS TO FIX: nothing extra in Monte Carlo. A simulated trajectory contains every
token, so one rollout per landmark scores all sixteen outcomes at once. The expensive part is
the same rollout evaluate.py already runs -- this module simply keeps more of it.

THE QUESTION ASKED PER OUTCOME. "Among subjects at risk at this landmark, does the event occur
within H years?" AUC over the Monte-Carlo probability, with:

  AT RISK      an outcome that can only happen once (the four `_cum` histories, AD, death) is
               scored only on subjects who have not already had it at the landmark. An outcome
               that recurs (stroke and depression onset, medication start/stop, a scale
               crossing) is scored on everyone. Scoring a subject for an event they already had
               is the single easiest way to manufacture a good AUC.
  COMPETING    death inside the horizon without the event is dropped, not scored 0 -- the same
  RISK         rule cohort.label_at_horizon applies to AD, for the same reason. 62% of this
               cohort dies; treating those as clean negatives makes a lethal comorbidity look
               protective. Death itself is exempt, being the competing event.
  CENSORING    follow-up ending before the horizon without the event is dropped.

WORSENING, for the two ordinal cognitive scales, means the emitted level moves at least one
step down vocab.SEVERITY_ORDER (index 0 is the worst). BMI is deliberately absent: both tails
are adverse, so "worsened" is not a well-posed question on it.

READ THE COUNTS, NOT THE AUCs ALONE. Several of these outcomes are rare at a five-year horizon,
and an AUC on 30 events is not a measurement. Every row carries its evaluable n and positive
count and the table is sorted by the latter.
"""
import os
import sys
import json
import time
import argparse
from multiprocessing import Pool

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _HERE)

from radc_delphi import vocab as V                        # noqa: E402
from radc_delphi.engine import load                       # noqa: E402
from eval.cohort import load_subjects, landmarks          # noqa: E402

DAYS_PER_YEAR = 365.25
NEG = -1e4 + 1                    # padded positions carry age -10000


def token_table():
    """One entry per token the model is trained to emit, scored individually.

    WHY THIS IS NOT THE SAME QUESTION as the grouped table below. "MMSE worsens by >= 1 level"
    collapses seven levels into a binary, which is exactly the resolution the v3 tokenization
    was built to add -- so the grouped row cannot say whether the model can tell "drops to 28"
    from "drops to 24-26". Scoring each level on its own is the only way to see whether the
    finer bins bought anything the model can actually use.

    At-risk follows the same rule as the grouped table: a once-only token (the four `_cum`
    histories, the diagnosis, death) is scored only on subjects who have not already had it;
    everything else recurs and is scored on everyone. A scale level counts as recurring
    because keep-transitions re-emits it when a subject leaves and comes back, which is the
    recovery the hysteresis replacement was meant to preserve.
    """
    once = set(V.KEEP_FIRST_IDS) | {V.AD_DX, V.DEATH}
    return [(f"tok{t}", V.NAMES[t], (t,), t not in once)
            for t in range(V.VOCAB_SIZE)
            if t not in V.IGNORE_TOKENS and t != V.NO_EVENT]


def outcome_table():
    """The outcomes to score: (key, label, token ids, recurring?).

    `recurring` decides the at-risk rule: a once-only outcome excludes subjects who already
    have it at the landmark, a recurring one scores everybody.
    """
    S = V.SCALES
    out = [
        ("mmse_worse", "MMSE worsens >=1 level", None, True),
        ("cog_worse", "Cognition worsens >=1 level", None, True),
        ("hypertension", "Hypertension, first report", (V.ID["Hypertension, history"],), False),
        ("diabetes", "Diabetes, first report", (V.ID["Diabetes, history"],), False),
        ("claudication", "Claudication, first report", (V.ID["Claudication, history"],), False),
        ("heart", "Heart condition, first report", (V.ID["Heart condition, history"],), False),
        ("stroke", "Stroke onset (any grade)",
         (V.ID["Stroke, probable"], V.ID["Stroke, possible"]), True),
        ("depression", "Depression onset (any grade)",
         (V.ID["Depression, probable"], V.ID["Depression, possible"]), True),
        ("antihyp_start", "Antihypertensive started", (V.ID["Antihypertensive started"],), True),
        ("statin_start", "Statin started", (V.ID["Statin started"],), True),
        ("antihyp_stop", "Antihypertensive stopped", (V.ID["Antihypertensive stopped"],), True),
        ("statin_stop", "Statin stopped", (V.ID["Statin stopped"],), True),
        ("ad", "Alzheimer's dementia diagnosis", (V.AD_DX,), False),
        ("death", "Death", (V.DEATH,), False),
    ]
    assert len(S) == 3, "scale set changed; revisit the worsening definitions"
    return out


def _level_at(ages, toks, ids, cut_day):
    """Index into `ids` of the last level emitted at or before `cut_day`, or None."""
    m = np.isin(toks, ids) & (ages <= cut_day) & (ages > NEG)
    if not m.any():
        return None
    return list(ids).index(int(toks[m][np.argmax(ages[m])]))


def _worsens(ages, toks, scale, t0, t1, base):
    """Did `scale` emit a level strictly worse than `base` inside (t0, t1]?

    SEVERITY_ORDER is worst-first, so "worse" is a LOWER index.
    """
    ids = list(V.SEVERITY_ORDER[scale])
    m = np.isin(toks, ids) & (ages > t0) & (ages <= t1)
    if not m.any():
        return False
    return any(ids.index(int(t)) < base for t in toks[m])


def _occurs(ages, toks, ids, t0, t1):
    return bool((np.isin(toks, list(ids)) & (ages > t0) & (ages <= t1)).any())


_G = {}


def _init(ckpt, data_dir, device, split):
    eng = load(ckpt, data_dir=data_dir, device=device)
    data, p2i, _ = eng.load_split(split)
    _G.update(eng=eng, data=data, p2i=p2i,
              sub=load_subjects(data_dir, radc_dir=os.path.join(_HERE, "data", "RADC")))


def _one(args):
    """Every outcome, every horizon, for one subject. Returns a list of row dicts."""
    k, horizons, n_mc, seed, max_lm, per_token = args
    table = token_table() if per_token else outcome_table()
    eng, data, p2i, sub = _G["eng"], _G["data"], _G["p2i"], _G["sub"]
    ages, toks, pid = eng.stream(data, p2i, k)
    if pid not in sub.index:
        return []
    r = sub.loc[pid]
    if not bool(r["in_training"]):
        return []

    lms = landmarks(ages, toks, r, min_lead_years=0.0, min_prior_visits=2)
    if len(lms) == 0:
        return []
    if max_lm and len(lms) > max_lm:                 # thin evenly, keep the ends
        lms = lms[np.linspace(0, len(lms) - 1, max_lm).round().astype(int)]

    exit_day = float(r["event_age"]) * DAYS_PER_YEAR
    died = int(r["event_type"]) == 2 or not np.isnan(r["age_death"])
    death_day = float(r["age_death"]) * DAYS_PER_YEAR if not np.isnan(r["age_death"]) else np.inf
    rows = []
    for lm in lms:
        pre_a, pre_t = eng.prefix_at(ages, toks, lm)
        if len(pre_t) < 2:
            continue
        gi, ga = eng.simulate(pre_t, pre_a, n_mc=n_mc, seed=seed + int(lm) % 9973,
                              until_age_years=(lm / DAYS_PER_YEAR) + max(horizons) + 1)
        for key, _lab, ids, recurring in table:
            # ---- at risk? -------------------------------------------------------------
            if key in ("mmse_worse", "cog_worse"):
                scale = "MMSE" if key == "mmse_worse" else "COG"
                base = _level_at(ages, toks, V.SEVERITY_ORDER[scale], lm)
                if base is None or base == 0:        # never measured, or already worst
                    continue
                sim_base = base
            elif not recurring and _occurs(ages, toks, ids, -np.inf, lm):
                continue                              # already had a once-only outcome
            # ---- observed label, competing risk and censoring -------------------------
            for H in horizons:
                t1 = lm + H * DAYS_PER_YEAR
                if key in ("mmse_worse", "cog_worse"):
                    scale = "MMSE" if key == "mmse_worse" else "COG"
                    y = _worsens(ages, toks, scale, lm, t1, sim_base)
                else:
                    y = _occurs(ages, toks, ids, lm, t1)
                if not y:
                    if key != "death" and died and death_day <= t1:
                        continue                      # competing event, not a clean negative
                    if exit_day < t1:
                        continue                      # right-censored
                # ---- Monte-Carlo probability ------------------------------------------
                if key in ("mmse_worse", "cog_worse"):
                    scale = "MMSE" if key == "mmse_worse" else "COG"
                    sids = list(V.SEVERITY_ORDER[scale])
                    hit = np.zeros(gi.shape[0], dtype=bool)
                    for row in range(gi.shape[0]):
                        m = np.isin(gi[row], sids) & (ga[row] > lm) & (ga[row] <= t1)
                        if m.any():
                            hit[row] = any(sids.index(int(t)) < sim_base for t in gi[row][m])
                    p = float(hit.mean())
                else:
                    m = np.isin(gi, list(ids)) & (ga > lm) & (ga <= t1)
                    p = float(m.any(1).mean())
                rows.append(dict(pid=int(pid), outcome=key, horizon=float(H),
                                 p=p, y=int(bool(y))))
    return rows


def auc_clustered(y, p, pid, n_boot=500, seed=0):
    """AUC with a subject-clustered bootstrap CI. A subject contributes many landmark rows;
    resampling rows instead of subjects understates the interval by about a third."""
    y, p, pid = np.asarray(y), np.asarray(p), np.asarray(pid)
    if len(np.unique(y)) < 2:
        return np.nan, (np.nan, np.nan)
    from sklearn.metrics import roc_auc_score
    a = float(roc_auc_score(y, p))
    subs = np.unique(pid)
    idx = {s: np.where(pid == s)[0] for s in subs}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        take = np.concatenate([idx[s] for s in rng.choice(subs, len(subs), replace=True)])
        if len(np.unique(y[take])) < 2:
            continue
        out.append(roc_auc_score(y[take], p[take]))
    lo, hi = (np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan))
    return a, (float(lo), float(hi))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--data-dir", default=os.environ.get("RADC_DATA_DIR", "data/radc-v3-s42"))
    ap.add_argument("--horizons", type=float, nargs="+", default=[1.0, 3.0, 5.0])
    ap.add_argument("--n-mc", type=int, default=200)
    ap.add_argument("--max-landmarks", type=int, default=6,
                    help="thin each subject's landmarks to at most this many; the rollout is "
                         "the cost and 6 spread over the follow-up carries the signal")
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-positives", type=int, default=25,
                    help="below this an AUC is not reported; the count still is")
    ap.add_argument("--per-token", action="store_true",
                    help="score every emitted token individually instead of the grouped "
                         "clinical outcomes -- the only view that shows whether the finer "
                         "MMSE levels are separable")
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)

    if a.split == "test":
        print("!" * 92 + "\n!! SCORING THE TEST SPLIT.\n" + "!" * 92)

    eng = load(a.ckpt, data_dir=a.data_dir, device=a.device)
    _, p2i, _ = eng.load_split(a.split)
    n = len(p2i) if not a.limit else min(a.limit, len(p2i))
    jobs = [(k, a.horizons, a.n_mc, a.seed, a.max_landmarks, a.per_token)
            for k in range(n)]

    t0 = time.time()
    rows = []
    with Pool(a.workers, initializer=_init,
              initargs=(a.ckpt, a.data_dir, a.device, a.split)) as pool:
        for i, out in enumerate(pool.imap_unordered(_one, jobs, chunksize=4), 1):
            rows.extend(out)
            if i % 50 == 0:
                print(f"  ... {i}/{n} subjects, {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    print(f"  {len(df):,} scored (outcome, landmark, horizon) rows in {time.time() - t0:.0f}s")

    labels = {k: lab for k, lab, _i, _r in (token_table() if a.per_token
                                           else outcome_table())}
    res = []
    for (key, H), g in df.groupby(["outcome", "horizon"]):
        npos, nneg = int(g.y.sum()), int((g.y == 0).sum())
        auc, (lo, hi) = ((np.nan, (np.nan, np.nan)) if npos < a.min_positives
                         else auc_clustered(g.y.values, g.p.values, g.pid.values, seed=a.seed))
        res.append(dict(outcome=key, label=labels[key], horizon=H, n=len(g), n_pos=npos,
                        n_neg=nneg, n_subjects=int(g.pid.nunique()),
                        prevalence=npos / max(len(g), 1), auc=auc, ci_lo=lo, ci_hi=hi,
                        mean_pred=float(g.p.mean()), mean_obs=float(g.y.mean())))
    R = pd.DataFrame(res).sort_values(["horizon", "n_pos"], ascending=[True, False])

    for H in a.horizons:
        sub = R[R.horizon == H]
        print(f"\n=== horizon {H:g} y "
              f"{'-' * 74}\n  {'outcome':<30} {'n':>7} {'pos':>6} {'prev':>6} "
              f"{'AUC':>7}  {'95% CI':>16}  {'pred':>6} {'obs':>6}")
        for _, r in sub.iterrows():
            auc = "  n/a  " if not np.isfinite(r.auc) else f"{r.auc:>7.3f}"
            ci = "" if not np.isfinite(r.ci_lo) else f"[{r.ci_lo:.3f}, {r.ci_hi:.3f}]"
            flag = f"  <-- only {r.n_pos} positives" if not np.isfinite(r.auc) else ""
            print(f"  {r.label[:30]:<30} {r.n:>7,} {r.n_pos:>6,} {r.prevalence:>6.3f} "
                  f"{auc}  {ci:>16}  {r.mean_pred:>6.3f} {r.mean_obs:>6.3f}{flag}")

    out = a.out or (f"eval/pertoken_{a.split}.json" if a.per_token
                    else f"eval/perdomain_{a.split}.json")
    with open(out, "w") as fh:
        json.dump(dict(split=a.split, ckpt=os.path.abspath(a.ckpt), n_mc=a.n_mc,
                       horizons=a.horizons, max_landmarks=a.max_landmarks,
                       min_positives=a.min_positives, rows=res), fh, indent=1)
    R.to_csv(out.replace(".json", ".csv"), index=False)
    print(f"\n  wrote {out} and {out.replace('.json', '.csv')}")
    return R


if __name__ == "__main__":
    main()
