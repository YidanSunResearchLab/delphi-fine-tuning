"""
tokenizer.py -- the three raw RADC/ROSMAP files -> one age-ordered event stream.

    from radc_delphi.tokenizer import tokenize
    events, subjects, report = tokenize("data/RADC")

`events` is an (N, 3) int64 array of (projid, age_days, MODEL token id), sorted by subject and
then by age. `build_dataset.py` converts it to the uint32 DISK layout the trainer memory-maps.
`subjects` is a per-subject frame used for stratifying the split and for evaluation labels.

--------------------------------------------------------------------------------------------
THE AGE AXIS -- the one assumption everything else rests on.

These files carry no visit date and no per-visit age. The only anchors are `age_bl` (baseline
age, uncensored, from the cross-sectional file) and `fu_year`. So

    age_at_visit_days = round(age_bl * 365.25) + 365 * fu_year

Two details in that formula are load-bearing and both were measured rather than assumed.

**fu_year is an elapsed-year index, not a visit counter.**
30.5% of subjects skip at least one fu_year, and those gaps are genuinely missing ROWS (only 8
all-missing rows exist in the entire file, 0.022%). Advancing the age by the row count instead
of by the index would compress every gap. The decisive measurement: for subjects with exactly
one skipped year, the discrepancy against ROSMAP_clinical's last-visit age has median -0.007 y
using the index and +0.993 y using the row count.

**The grid step is 365 days, not 365.25.** Rounding an integer-year index times 365.25 gives
inter-visit deltas that cycle 365, 365, 365, 366 with a per-subject fixed phase -- a perfectly
learnable arithmetic artifact sitting in the exact day-level channel the waiting-time head
reads. Only the two genuinely fractional ages (the AD diagnosis and a numeric age at death)
use 365.25, because those are real ages rather than grid positions.

Residual error against the true visit age is within-year jitter, not accumulating drift --
median |error| ~21 days, ~8% of last visits slip past 6 months, ~2% past a year, with a small
+4.5 to +6 day late bias. That is the accuracy ceiling of every timing claim this model makes,
and it should be quoted whenever one is.

A CONSEQUENCE WORTH STATING PLAINLY: 93.9% of nominal inter-visit intervals are exactly 1.000
year, because the cohort is on an annual protocol. Delphi's time-to-event head therefore learns
"the next observation arrives in about a year", which is the correct data-generating process
for OBSERVATIONS but carries almost no individual signal. Only two tokens sit on a real clock:
the AD diagnosis (its own recorded age) and death. Do not select models on total validation
loss here -- see `select_on` in train.py.

--------------------------------------------------------------------------------------------
DEDUP -- five regimes, applied per subject.

  STATIC          one token per subject, at age_bl - 1 day, so the whole block precedes the
                  first visit and the model always reads the traits before any measurement.
  HYSTERESIS      the ordinal scales. Run-length-encode per scale, but ASYMMETRICALLY: a
                  decline is emitted the moment the edge is crossed, while a recovery must
                  clear the edge by a margin. Plain keep-transitions on these scales is mostly
                  measurement noise -- measurement SD is ~1.2-1.6 MMSE points and ~0.18-0.21 z,
                  41% of MMSE and 54% of cogn_global bin changes vanish under smoothing, and
                  35-51% of "recoveries" immediately round-trip. The margin cuts round-trip
                  churn from ~37% to ~14% while slightly IMPROVING 5-year AD AUC (0.894 vs
                  0.883). Asymmetry matters: real decline should not be delayed by a filter.
  KEEP-FIRST      the monotone _cum histories: one token at the age of the first 1.
  ONSET           r_stroke / r_depres. These are per-visit clinician judgements that revert
                  (83.1% of subjects who ever score present return to "not present"), so a
                  token is emitted on every WORSENING step rather than only the first.
  START/STOP      the two medications. Every start, but only DURABLE stops -- 41-46% of
                  ever-users stop at least once and 22% stop and restart, so keep-first would
                  assert continuous exposure for about one post-initiation visit in six.
"""
import os
import hashlib

import numpy as np
import pandas as pd

from . import vocab as V

DAYS_PER_YEAR = 365.25

# Implausible values are DROPPED, not clipped: we know the value is wrong, not what was meant.
# Measured over the whole file this removes 6 values in total -- the one that matters is
# hba1c = 505 (a decimal-point error), which is not tokenized anyway; of the columns that ARE
# tokenized, exactly one BMI value falls outside.
PLAUSIBLE = {
    "cts_estmmse30": (0.0, 30.0),
    "cogn_global": (-5.0, 2.0),
    "bmi": (12.0, 70.0),
}

# (interior edges, per-edge recovery margin, token names low -> high on the raw value)
SCALE_SPEC = {
    "cts_estmmse30": ([18.0, 24.0, 27.0], [2.0, 2.0, 2.0], V.SCALES["MMSE"]),
    "cogn_global": ([-2.0, -1.0, -0.3, 0.2, 0.8], [0.2] * 5, V.SCALES["COG"]),
    "bmi": ([20.0, 30.0], [1.0, 1.0], V.SCALES["BMI"]),
}

CUM_COLS = {
    "hypertension_cum": V.ID["Hypertension, history"],
    "dm_cum": V.ID["Diabetes, history"],
    "claudication_cum": V.ID["Claudication, history"],
    "heart_cum": V.ID["Heart condition, history"],
}
# r_stroke / r_depres: 1 = highly probable, 2 = probable, 3 = possible, 4 = NOT PRESENT.
# Mapped to an ordinal severity 0 = absent, 1 = possible, 2 = probable.
GRADED_COLS = {
    "r_stroke": (V.ID["Stroke, possible"], V.ID["Stroke, probable"]),
    "r_depres": (V.ID["Depression, possible"], V.ID["Depression, probable"]),
}
MED_COLS = {
    "antihyp_rx": (V.ID["Antihypertensive started"], V.ID["Antihypertensive stopped"]),
    "statin_rx": (V.ID["Statin started"], V.ID["Statin stopped"]),
}

# Median lag from the nominal last visit to the true death age, measured on the 975 subjects
# where both are known (IQR 0.34-0.98 y). Used to place death for the 1,770 who have died but
# whose age_death is absent or censored to "90+".
DEATH_LAG_YEARS = 0.68

# ------------------------------------------------------------------ the planted canary
# A DELIBERATE LEAK, and the only reason eval/leakage.py can be believed when it says "clean".
# An audit that has never fired is indistinguishable from an audit that cannot fire.
#
# With canary=True a deterministic 5% of subjects get ONE extra token beside their statics
# that encodes their EVENTUAL AD status. It is a perfect oracle for those subjects and absent
# for the other 95%, so a single canary run exercises both halves of the detector at once: the
# marked subjects must trip Tests 1 and 2 loudly, the unmarked ones must stay quiet.
#
# Four things keep it out of the production path, none of them a convention:
#   * the id is 50 -- APPENDED past the standard 50-id table, never inserted into it. vocab.py
#     is not edited; enable_canary_vocab() extends the table in-process and only
#     tokenize(canary=True) calls it, so a standard build has no such name in V.ID and cannot
#     emit the token even by accident.
#   * build_dataset.py refuses to write a canary build into the production output directory.
#   * a canary build's labels.csv has 51 rows, so train.py's vocab_size cross-check and
#     engine.load's vocab fingerprint both refuse to pair it with a production checkpoint.
#   * eval/leakage.py asserts, before it scores anything, that the vocabulary under audit is
#     the standard 50 and carries no canary name.
CANARY_NAME = "CANARY: this subject is eventually diagnosed with AD"
CANARY_ID = V.VOCAB_SIZE                # 50 -- one past the standard table, never inside it
CANARY_FRACTION = 0.05
CANARY_SALT = "radc-canary-v1"


def is_canary_subject(pid, fraction=CANARY_FRACTION):
    """Is this subject one of the marked 5%?

    Hashed from the projid rather than drawn from a seeded shuffle, so membership depends on
    the subject alone: it survives a re-tokenization, a different split seed and any subset of
    the file, and the audit can recompute it without having to read the build back.
    """
    h = hashlib.md5(f"{CANARY_SALT}|{int(pid)}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64) < fraction


def enable_canary_vocab():
    """Extend the in-process vocabulary by the canary id. Idempotent; canary builds only.

    vocab.check() runs FIRST, so the standard 50-token table is proved intact before anything
    is appended to it, and it is deliberately not re-run afterwards: id 50 belongs to no token
    group and check() is right to reject it.
    """
    if CANARY_NAME in V.ID:
        return CANARY_ID
    V.check()
    assert len(V.NAMES) == CANARY_ID, \
        f"canary id {CANARY_ID} does not sit one past the table ({len(V.NAMES)} names)"
    V.NAMES.append(CANARY_NAME)
    V.ID[CANARY_NAME] = CANARY_ID
    V.VOCAB_SIZE = CANARY_ID + 1
    return CANARY_ID


APOE_MAP = {22: "APOE e2 carrier (22/23)", 23: "APOE e2 carrier (22/23)",
            33: "APOE e3/e3", 24: "APOE e4 heterozygote (24/34)",
            34: "APOE e4 heterozygote (24/34)", 44: "APOE e4 homozygote (44)"}
SMOKING_MAP = {0: "Smoking: never", 1: "Smoking: former", 2: "Smoking: current"}
STUDY_MAP = {"ROS": "Study: ROS", "MAP": "Study: MAP", "LATC": "Study: LATC"}


# ------------------------------------------------------------------ loading
def load(radc_dir):
    """Read and structurally validate the three files."""
    xs = pd.read_excel(os.path.join(radc_dir, "cross-sectional-data-gk.xlsx"))
    lg = pd.read_excel(os.path.join(radc_dir, "longitudinal_data_gk.xlsx"))
    cl = pd.read_csv(os.path.join(radc_dir, "ROSMAP_clinical.csv"))

    xs["study"] = xs["study"].astype(str).str.strip()      # ships as 'MAP ', 'ROS ', 'LATC'

    assert set(xs.projid) == set(lg.projid), \
        "cross-sectional and longitudinal cover different subjects"
    assert not lg.duplicated(["projid", "fu_year"]).any(), "(projid, fu_year) is not unique"
    assert xs.projid.is_unique, "cross-sectional projid is not unique"
    # 4 genuine ROS subjects enter at 36.5-44.7 y. They are real, so this is a sanity bound and
    # not a filter -- an age_bl outside it would mean the column has been misread.
    assert xs.age_bl.between(35, 105).all(), "age_bl outside [35, 105]: check the column"
    assert set(xs.study) <= set(STUDY_MAP), f"unexpected study values: {set(xs.study)}"

    for col, (lo, hi) in PLAUSIBLE.items():
        s = pd.to_numeric(lg[col], errors="coerce")
        bad = s.notna() & ((s < lo) | (s > hi))
        if bad.any():
            print(f"  [guard] {col}: dropped {int(bad.sum())} implausible value(s) outside "
                  f"[{lo}, {hi}]: {sorted(s[bad].unique())[:5]}")
        lg[col] = s.where(~bad)

    # Eight rows (0.022%) carry no observed data column at all. They are indistinguishable
    # from a skipped year in content but they DO extend max(fu_year), which would push a
    # subject's imputed death age out by the length of the empty tail. Drop them.
    data_cols = [c for c in lg.columns if c not in ("projid", "study", "fu_year")]
    empty = lg[data_cols].isna().all(axis=1)
    if empty.any():
        print(f"  [guard] dropped {int(empty.sum())} longitudinal row(s) with no observed "
              f"value at all (they would extend the follow-up window without evidence)")
        lg = lg[~empty]

    lg = lg.sort_values(["projid", "fu_year"]).reset_index(drop=True)
    return xs.set_index("projid"), lg, cl.set_index("projid")


# ------------------------------------------------------------------ the hysteresis machine
def hysteresis_states(values, edges, margins):
    """Bin a sequence with an asymmetric recovery margin.

    Moving to a WORSE bin (a lower raw value) happens as soon as the value falls below the
    edge. Moving to a BETTER bin (a higher raw value) requires clearing the edge by
    `margins[k]`. The state is therefore sticky downward-only, which is what removes the
    round-trip churn without delaying a real decline.

    `values` must be in visit order. Returns an int array of bin indices, same length.
    """
    edges = np.asarray(edges, dtype=float)
    margins = np.asarray(margins, dtype=float)
    out = np.empty(len(values), dtype=int)
    state = int(np.searchsorted(edges, values[0], side="right"))
    out[0] = state
    for i in range(1, len(values)):
        v = values[i]
        while state > 0 and v < edges[state - 1]:        # decline: cross immediately
            state -= 1
        while state < len(edges) and v >= edges[state] + margins[state]:   # recovery: clear it
            state += 1
        out[i] = state
    return out


# ------------------------------------------------------------------ the tokenizer
def tokenize(radc_dir, emit_ad_rx=False, canary=False, canary_fraction=CANARY_FRACTION,
             verbose=True):
    """Return (events, subjects, report).

    emit_ad_rx: OFF by default and this is a leakage decision, not a coverage one. Among
    pre-diagnosis visits at a normal MMSE of 27-30, ad_rx == 1 raises P(AD dx within 3 y) from
    4.7% to 31.2%. The drug list decodes as 100% AD-specific (donepezil / memantine /
    rivastigmine / galantamine / tacrine), so the token is the treating clinician's diagnosis
    arriving before age_first_ad_dx records it. The flag exists so the with/without ablation is
    one argument rather than a fork; if it is ever switched on, no AD-onset metric computed
    downstream is interpretable as prediction.

    canary: OFF by default. Plants the token described under "the planted canary" above for a
    deterministic `canary_fraction` of subjects. It exists so eval/leakage.py can be shown to
    detect a leak it was not tuned on; nothing built with it may be reported as a result. At
    the default 5% the marked set is 216 subjects of whom 55 convert and so carry the oracle
    -- 38 of them in train and 6 in val, which is why the canary is audited on the TRAIN
    split; raise the fraction if a run needs more power than that.
    """
    if emit_ad_rx:
        print("  [WARNING] emit_ad_rx=True: ad_rx is a measured leak of the AD outcome "
              "(4.7% -> 31.2% 3-year risk at normal MMSE). AD-onset metrics from this build "
              "are NOT prediction. Two ids are appended past the standard vocabulary.")

    xs, lg, cl = load(radc_dir)
    V.check()
    if canary:
        enable_canary_vocab()
        print(f"  [CANARY] planting id {CANARY_ID} on the marked "
              f"{100 * canary_fraction:.1f}% -- this build is an audit fixture, not data")

    age_death_num = pd.to_numeric(cl["age_death"], errors="coerce")
    age_death_90p = cl["age_death"].astype(str).str.strip().eq("90+")

    events = []                      # (projid, age_days, model_id)
    subj_rows = []
    counts = {n: 0 for n in V.NAMES}
    n_dx_outside_grid = 0

    for pid, g in lg.groupby("projid", sort=True):
        r = xs.loc[pid]
        abl_days = int(round(r.age_bl * DAYS_PER_YEAR))

        def grid(fu):
            """Nominal age in days at follow-up year `fu`.

            365, NOT 365.25, and this is not a rounding nicety. Multiplying an integer-year
            index by 365.25 and rounding injects a deterministic period-4 pattern into every
            subject's inter-visit deltas -- 365, 365, 365, 366 repeating, with the phase fixed
            by frac(age_bl * 365.25). It is exactly reproducible within a subject and it sits
            in the day-level delta channel the waiting-time head reads, so the model can learn
            the arithmetic of the grid instead of the biology. With 365 every protocol interval
            is exactly 365 days and the baseline age stays exact to the day.

            Index-driven: `fu` is the fu_year INDEX, never the row ordinal, so a skipped year
            advances the age by the years skipped. Using the ordinal would misdate every
            post-gap event by the length of the gap -- up to 20 years for the worst subject.
            """
            return abl_days + 365 * int(fu)

        ev = []

        # ---- statics, one day before baseline -----------------------------------------
        # The minus one day is load-bearing, not cosmetic. model.py's mask_ties blocks a query
        # position from attending to any input token that shares the query's TARGET age, so
        # statics placed at exactly age_bl would be invisible when predicting any token of the
        # baseline visit -- roughly a fifth of all scored positions. One day earlier keeps them
        # permanently attendable, and it also avoids the ~79-year gap that age-0 placement
        # would put at the head of every sequence.
        a0 = abl_days - 1
        ev.append((a0, "Sex: male" if r.msex == 1 else "Sex: female"))
        ev.append((a0, "APOE unknown" if pd.isna(r.apoe_genotype)
                   else APOE_MAP[int(r.apoe_genotype)]))
        ev.append((a0, "Education <=12y" if r.educ <= 12 else
                   ("Education 13-16y" if r.educ <= 16 else "Education >=17y")))
        ev.append((a0, STUDY_MAP[r.study]))
        ev.append((a0, "Smoking: unknown" if pd.isna(r.smoking_bl)
                   else SMOKING_MAP[int(r.smoking_bl)]))
        ev.append((a0, "Alcohol: unknown" if pd.isna(r.ldai_bl) else
                   ("Alcohol: none (<1 drink/month)" if r.ldai_bl == 0 else
                    ("Alcohol: light" if r.ldai_bl <= 1 else "Alcohol: heavy"))))

        # ---- the planted canary, with the statics ---------------------------------------
        # Placed at a0, not at age_bl: mask_ties blocks a token from attending to anything
        # sharing its target age, so a canary sitting exactly on the baseline visit would be
        # invisible to that visit and the planted leak would be weaker than the real leaks it
        # stands in for. Emitted only for the marked subjects who convert, so its PRESENCE is
        # the oracle -- the same shape as the autopsy-availability leak it is modelled on.
        if canary and is_canary_subject(pid, canary_fraction) \
                and pd.notna(r.age_first_ad_dx):
            ev.append((a0, CANARY_NAME))

        # ---- ordinal scales, hysteresis keep-transitions, one run per scale ----------------
        for col, (edges, margins, ids) in SCALE_SPEC.items():
            s = g[["fu_year", col]].dropna()
            if s.empty:
                continue
            st = hysteresis_states(s[col].to_numpy(float), edges, margins)
            fy = s["fu_year"].to_numpy()
            ev.append((grid(fy[0]), V.NAMES[ids[st[0]]]))
            for i in range(1, len(st)):
                if st[i] != st[i - 1]:
                    ev.append((grid(fy[i]), V.NAMES[ids[st[i]]]))

        # ---- monotone cumulative histories, keep-first --------------------------------
        for col, tok in CUM_COLS.items():
            s = g[["fu_year", col]].dropna()
            hit = s[s[col] == 1]
            if len(hit):
                ev.append((grid(hit["fu_year"].iloc[0]), V.NAMES[tok]))

        # ---- graded per-visit judgements, emitted on every WORSENING step ---------------
        for col, (tok_poss, tok_prob) in GRADED_COLS.items():
            s = g[["fu_year", col]].dropna()
            if s.empty:
                continue
            v = s[col].to_numpy()
            fy = s["fu_year"].to_numpy()
            sev = np.where(v == 4, 0, np.where(v == 3, 1, 2))    # 4 = NOT PRESENT
            prev = 0
            for i in range(len(sev)):
                if sev[i] > prev:
                    ev.append((grid(fy[i]), V.NAMES[tok_prob if sev[i] == 2 else tok_poss]))
                prev = sev[i]

        # ---- medications: every start, and only DURABLE stops --------------------------
        # A stop is emitted when the off-run lasts >= 2 observed visits or runs to the end of
        # follow-up. A one-visit gap is far more often a reporting lapse than a real
        # discontinuation, and emitting it would fill the stream with churn.
        for col, (tok_start, tok_stop) in MED_COLS.items():
            s = g[["fu_year", col]].dropna()
            if s.empty:
                continue
            v = s[col].to_numpy().astype(int)
            fy = s["fu_year"].to_numpy()
            if v[0] == 1:
                ev.append((grid(fy[0]), V.NAMES[tok_start]))
            for i in range(1, len(v)):
                if v[i] == 1 and v[i - 1] == 0:
                    ev.append((grid(fy[i]), V.NAMES[tok_start]))
                elif v[i] == 0 and v[i - 1] == 1:
                    j = i
                    while j < len(v) and v[j] == 0:
                        j += 1
                    if (j - i) >= 2:
                        ev.append((grid(fy[i]), V.NAMES[tok_stop]))

        # ---- the AD diagnosis, at its OWN recorded age --------------------------------
        # age_first_ad_dx is uncensored in the cross-sectional file (66.3-107.2 y), unlike the
        # ROSMAP_clinical copy. It is placed unsnapped: it is a real age, and 94.4% of them
        # already sit within half a year of a nominal visit. 65 diagnoses fall outside the
        # visit grid entirely (50 past the last follow-up year, 15 inside an interior gap);
        # those are real events observed off-grid, so they stay where they are.
        ad_days = None
        if pd.notna(r.age_first_ad_dx):
            ad_days = int(round(r.age_first_ad_dx * DAYS_PER_YEAR))
            ev.append((ad_days, "Alzheimer's dementia diagnosis"))

        last_grid = grid(g["fu_year"].max())
        if ad_days is not None and (ad_days > last_grid + 365 or ad_days < abl_days):
            n_dx_outside_grid += 1

        # ---- death ---------------------------------------------------------------------
        # Emitted for ALL 2,745 subjects with died == 1, not only the 975 with a numeric age.
        # Restricting to the numeric ones would teach the model that death is impossible after
        # 90 -- the max numeric age_death is 89.993 with zero values >= 90, so those 975 are
        # exactly the sub-90 deaths, while 47.4% of deaths with any age information are 90+.
        # It would also give LATC zero death tokens out of 27 deaths. Ignoring death instead is
        # not an option: 53.3% of non-converters died, and treating death as censoring inflates
        # cumulative AD incidence by +39% at age 85 and +59% at age 90 (Aalen-Johansen
        # 0.174/0.259 against a naive 1-KM 0.241/0.413).
        death_days = None
        if r.died == 1:
            if pid in age_death_num.index and pd.notna(age_death_num.loc[pid]):
                death_days = int(round(float(age_death_num.loc[pid]) * DAYS_PER_YEAR))
            elif pid in age_death_90p.index and bool(age_death_90p.loc[pid]):
                death_days = max(last_grid + int(round(DEATH_LAG_YEARS * DAYS_PER_YEAR)),
                                 int(round(90.0 * DAYS_PER_YEAR)))
            else:
                death_days = last_grid + int(round(DEATH_LAG_YEARS * DAYS_PER_YEAR))
            # death must come strictly after every observation, including an off-grid diagnosis
            floor = max(last_grid, ad_days if ad_days is not None else -1) + 1
            death_days = max(death_days, floor)
            ev.append((death_days, "Death"))

        # Within a visit the diagnosis is ordered first, so that when mask_ties is off the
        # same-age scale tokens cannot be read as its cause.
        ev.sort(key=lambda t: (t[0], 0 if t[1] == "Alzheimer's dementia diagnosis" else 1))
        for age_days, name in ev:
            events.append((pid, age_days, V.ID[name]))
            counts[name] += 1

        fu_max = float(g["fu_year"].max())
        subj_rows.append(dict(
            projid=pid, study=r.study, age_bl=float(r.age_bl), msex=int(r.msex),
            educ=float(r.educ), followup_y=fu_max, n_visits=int(len(g)),
            ever_ad=bool(pd.notna(r.age_first_ad_dx)),
            age_ad=float(r.age_first_ad_dx) if pd.notna(r.age_first_ad_dx) else np.nan,
            died=int(r.died),
            age_death=float(death_days) / DAYS_PER_YEAR if death_days is not None else np.nan,
            age_last_obs=float(max(last_grid, ad_days or -1)) / DAYS_PER_YEAR,
            n_events=len(ev),
        ))

    E = np.array(events, dtype=np.int64)
    order = np.lexsort((E[:, 2], E[:, 1], E[:, 0]))       # subject, then age, then token
    E = E[order]
    subjects = pd.DataFrame(subj_rows).set_index("projid")
    if canary:
        # `canary` is the marked 5%; `canary_token` is the subset that actually carries the
        # oracle. The audit needs both: the unmarked-and-quiet contrast is what proves the
        # detector is reading the token rather than firing on everything.
        subjects["canary"] = [is_canary_subject(p, canary_fraction)
                              for p in subjects.index]
        subjects["canary_token"] = subjects["canary"] & subjects["ever_ad"]

    # ---- self-checks: these are cheap and each one has failed at least once in this family
    assert (E[:, 1] >= 0).all(), "negative age"
    assert E[:, 1].max() < np.iinfo(np.uint32).max, "age_days overflows uint32"
    assert E[:, 2].min() >= 2, "a reserved id (0/1) was emitted as an event"
    assert E[:, 2].max() < V.VOCAB_SIZE, "token id past the end of the vocabulary"
    _last_is_death_violations = 0
    for pid, idx in pd.Series(np.arange(len(E))).groupby(E[:, 0]):
        rows = E[idx.to_numpy()]
        if V.DEATH in rows[:, 2] and rows[-1, 2] != V.DEATH:
            _last_is_death_violations += 1
    assert _last_is_death_violations == 0, \
        f"{_last_is_death_violations} subjects have events after their death token"

    n_subj = len(subjects)
    n_multi = int((subjects["n_visits"] >= 2).sum())
    if n_subj != 4428 or n_multi != 4029:
        print(f"  [NOTE] cohort is {n_subj} subjects ({n_multi} with >=2 visits); the "
              f"delivered files gave 4428 / 4029. If you did not change the data cut, "
              f"something upstream moved.")
    report = dict(n_events=len(E), n_subjects=n_subj, counts=counts,
                  events_per_subject=len(E) / n_subj, dx_outside_grid=n_dx_outside_grid,
                  canary=bool(canary))
    if canary:
        report["n_canary_subjects"] = int(subjects["canary"].sum())
        report["n_canary_tokens"] = int(subjects["canary_token"].sum())
        assert report["n_canary_tokens"] == counts[CANARY_NAME], "canary token count disagrees"
        print(f"  [CANARY] {report['n_canary_subjects']:,} marked subjects, "
              f"{report['n_canary_tokens']:,} carrying the oracle token")

    if verbose:
        print(f"\n  {len(E):,} events over {n_subj:,} subjects "
              f"({len(E) / n_subj:.2f} per subject)")
        ns = subjects["n_events"].to_numpy()
        print(f"  events/subject: min {ns.min()} p50 {int(np.median(ns))} "
              f"p90 {int(np.percentile(ns, 90))} p95 {int(np.percentile(ns, 95))} "
              f"p99 {int(np.percentile(ns, 99))} max {ns.max()}")
        print(f"  AD converters {int(subjects.ever_ad.sum()):,} "
              f"({100 * subjects.ever_ad.mean():.1f}%) | deaths "
              f"{int((subjects.died == 1).sum()):,} "
              f"({100 * (subjects.died == 1).mean():.1f}%)")
        print(f"  diagnoses placed off the visit grid: {n_dx_outside_grid}")
        print("\n  --- token counts ---")
        for i, n in enumerate(V.NAMES):
            if i < 2:
                continue
            print(f"  {i:3d}  {n:<34s} {counts[n]:8,}")

    return E, subjects, report
