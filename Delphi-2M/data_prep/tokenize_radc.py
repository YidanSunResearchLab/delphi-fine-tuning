"""
tokenize_radc.py -- RADC/ROSMAP -> Delphi tokenizer (AD trajectory arm).

Sibling of tokenize_nacc_ad.py. Same output contract, same dedup philosophy, different
source: the four RADC files under <repo>/data/RADC/ instead of one NACC investigator CSV.

  cross-sectional-data-gk.xlsx   4428 x 24   one row per subject (statics + autopsy pathology)
  longitudinal_data_gk.xlsx     36138 x 29   one row per subject-visit, keyed (projid, fu_year)
  ROSMAP_clinical.csv            3584 x 18   one row per subject, last visit (ages censored "90+")
  RADC_codebook_*.pdf                        variable definitions + value codings

THE MERGE. cross-sectional and longitudinal carry exactly the SAME 4428 projid, and
(projid, fu_year) is unique -- a regular panel, so the join is a plain merge on projid.
ROSMAP_clinical covers only 3581 of them and every age in it is censored at "90+", so it is
used for NOTHING except the optional death age (see EMIT_DEATH).

THE AGE AXIS. These files carry no per-visit age: only age_bl (cross-sectional, UNcensored,
36.5-102.2y) and fu_year (the annual visit index, 0-31). So

    age_at_visit = age_bl + fu_year

which is the standard ROSMAP convention -- visits are nominally annual -- but it IS an
approximation: a visit that slipped by six months is recorded at its nominal anniversary.
Gaps are handled correctly (1350/4428 subjects skip at least one fu_year; the index, not the
row count, sets the age), so the error is within-year jitter, not accumulating drift.

DEDUP -- three regimes, mirroring tokenize_nacc_ad.py:
  * STATIC (model ids 2-21): one token per subject, ever. Immutable traits.
  * ORDINAL STATES (the SCALES dict): KEEP-TRANSITIONS, applied PER SCALE. Each scale is
    run-length-encoded per subject, so the first value plus every bin CHANGE survives --
    decline AND recovery -- collapsing only consecutive same-bin repeats within that scale.
    This is the regime that makes recovery trackable; see tokenize_nacc_ad.py and the README.
  * KEEP-FIRST (everything else): one token at age of first onset.

WHAT IS DELIBERATELY LEFT OUT, and why:
  * ALL autopsy pathology (braaksc, ceradsc, gpath, amylsqrt_est_8reg, tangsqrt_est_8reg,
    tdp_st4, lewydx_st4, arteriol_scler, caa_4gp, cvda_4gp2, ci_num2_mct, ci_num2_tct, pmi).
    Measured post mortem. Placed at any age along the trajectory it is future information
    leaking backwards. Keep them for post-hoc stratification instead (~50% coverage).
  * cogng_demog_slope -- a slope fitted over the WHOLE follow-up. Direct outcome leak.
  * log_hcrp / log_hil6 / log_htnfa -- 1.2% coverage (~430 of 36138 rows). Too sparse to
    earn four token ids; MAP/MARS subsample only (per the codebook).
  * Death, by default. See EMIT_DEATH.

Output: Delphi-2M/out_radc/{radc_all.bin, labels.csv, radc_pidmap.csv}
  radc_all.bin is uint32 (pid, age_days, disk_token) triples, sorted, patients contiguous --
  byte-identical in format to out_ad/nacc_all.bin, so make_split_ad.py and training/train.py
  consume it unchanged. Disk token = model_id - 1, because get_batch adds +1 back.

Run:  python -u tokenize_radc.py            (normally invoked via make_dataset_radc.py)
"""
import os, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Delphi-2M/ (package root)
ROOT = os.path.dirname(HERE)                                         # <repo>/
RADC = os.path.join(ROOT, "data", "RADC")
OUT  = os.path.join(HERE, "out_radc"); os.makedirs(OUT, exist_ok=True)

XS   = os.path.join(RADC, "cross-sectional-data-gk.xlsx")
LONG = os.path.join(RADC, "longitudinal_data_gk.xlsx")
CLIN = os.path.join(RADC, "ROSMAP_clinical.csv")

DAYS_PER_YEAR = 365.25

# ----------------------------- DECISIONS (flags) -----------------------------
CFG = dict(
    # Death is OFF by default and this is not a stylistic choice. died==1 for 2745 subjects,
    # but a death AGE exists only in ROSMAP_clinical (1853 of them) and 878 of those are
    # censored to the string "90+" -- leaving ~975 (36%) with a usable number, and those 975
    # are exactly the subjects who died BEFORE 90. Emitting only them would teach the model
    # that death never happens after 90. Turning this on buys a biased token, not a free one.
    EMIT_DEATH = False,
    # BMI is ON. It was briefly switched off on a bad argument; the numbers that settled it are
    # worth keeping, because the same reasoning applies to every other measured variable here.
    #
    # The worry was that whether BMI gets MEASURED depends on the outcome. That is true in
    # aggregate -- with fu_year and age controlled, MMSE 30 -> 18 multiplies the odds of a
    # missing BMI by 3.75 (blood pressure 3.11; PSQI 1.80; glucose only 1.49, since a blood
    # draw does not need the subject to cooperate and standing on a scale does).
    #
    # But aligning missingness to the DIAGNOSIS shows where that effect actually lives:
    #
    #   years to AD dx   -20..-10  -10..-7  -7..-5  -5..-3  -3..-2  -2..-1  -1..0   0..1   1..3
    #   BMI measured %       93.9     93.7    92.3    87.4    82.2    77.3   69.8   62.5   57.0
    #
    # Flat until ~5 years out. The collapse is concentrated AT and AFTER diagnosis, which is
    # detection, not prediction. In the window where the model is actually forecasting (>=3-5y
    # ahead) measurement is near-unrelated to future AD -- and the comparison that makes it
    # plainest runs the other way: visits >=2 years before a diagnosis have BMI measured 90.8%
    # of the time versus 79.7% for never-diagnosed subjects' visits, i.e. the future-AD group
    # is measured MORE, not less.
    #
    # Meanwhile the biology IS in that window: mean BMI runs 27.6 -> 26.2 over the decade
    # before diagnosis (prodromal weight loss, a well-established AD feature). Dropping the
    # token traded away real antecedent signal to suppress an artifact that mostly sits outside
    # the predictive horizon.
    #
    # Two caveats kept for whoever revisits this. (a) BMI alone carries a protocol artifact:
    # a biennial measurement schedule in LATC (even/odd-year gap 37.9pp) and partly MAP
    # (10.8pp), so BMI events inherit a 2-year rhythm from the study calendar. (b) keep-
    # transitions already launders missingness a good deal, because "measured and stable" and
    # "never measured" both emit no token.
    #
    # Ids 30-32 stay RESERVED either way, so the vocabulary and labels.csv are identical with
    # this flag on or off -- which is what makes the with/without ablation cheap to run.
    EMIT_BMI = True,
    # A visit needs BOTH sbp_avg and dbp_avg to be staged (the AHA bins are joint).
    # NOTE blood pressure (33-35) has the same contamination as BMI, ~2/3 as strong. It is
    # still emitted; it is the obvious next candidate if the ablation says it matters.
    BP_REQUIRE_BOTH = True,
)

# Plausibility guards. The data is clean except for a single hba1c=505 (a decimal-point
# error); an open-topped bin would silently file it as "diabetic". Values outside these
# ranges are dropped, not clipped -- we do not know what was meant.
PLAUSIBLE = {
    "cts_estmmse30": (0, 30),   "cogn_global": (-5, 2),    "bmi": (10, 70),
    "sbp_avg": (60, 250),       "dbp_avg": (30, 150),      "glucose": (30, 600),
    "hba1c": (3, 20),           "hdlchlstrl": (10, 250),   "ldlchlstrl": (10, 400),
    "gfr_mdrs": (3, 250),       "psqi_sum": (0, 16),
}

# ----------------------------- the vocabulary --------------------------------
# MODEL space (labels.csv row index). disk token = model id - 1.
#
# ids 0-21 are padding/no-event + the 20 static tokens, so config/train_radc.py can use the
# SAME `ignore_tokens = list(range(22))` as every delivered NACC config. That alignment is
# intentional: the mask keeps immutable statics out of the loss without hiding them from
# attention, and reusing the boundary means the RADC arm is comparable to the NACC arm.
LABELS = {
    0: "Padding",
    1: "No event",
    # --- static / background: one token per subject, dropped from the loss (ids 2-21) ---
    2: "Male", 3: "Female",
    4: "APOE e2/e2", 5: "APOE e2/e3", 6: "APOE e2/e4",
    7: "APOE e3/e3", 8: "APOE e3/e4", 9: "APOE e4/e4",
    10: "Education low (<=12y)", 11: "Education mid (13-16y)", 12: "Education high (>=17y)",
    13: "Study ROS", 14: "Study MAP", 15: "Study LATC",
    16: "Smoking never", 17: "Smoking former", 18: "Smoking current",
    19: "Alcohol none (LDAI 0)", 20: "Alcohol light (LDAI <=1/day)", 21: "Alcohol heavy (LDAI >1/day)",
    # --- cognitive scale 1: estimated MMSE, 0-30 (93.8% of visits) ---
    22: "MMSE normal (27-30)", 23: "MMSE borderline (24-26)",
    24: "MMSE mild-moderate (18-23)", 25: "MMSE severe (0-17)",
    # --- cognitive scale 2: global cognition z-score (97.1% of visits) ---
    26: "Global cognition high (z>0.5)", 27: "Global cognition average (-0.5<z<=0.5)",
    28: "Global cognition low (-1.5<z<=-0.5)", 29: "Global cognition very low (z<=-1.5)",
    # --- longitudinal biomarker states (keep-transitions, predicted) ---
    # 30-32 are RESERVED, not emitted (CFG["EMIT_BMI"]=False)
    30: "BMI low (<=22)", 31: "BMI mid (22-28)", 32: "BMI high (>28)",
    33: "BP normal (<130/80)", 34: "BP stage 1 (130-139/80-89)", 35: "BP stage 2 (>=140/90)",
    36: "Glucose normal (<100)", 37: "Glucose impaired (100-125)", 38: "Glucose diabetic (>=126)",
    39: "HbA1c normal (<5.7)", 40: "HbA1c prediabetic (5.7-6.4)", 41: "HbA1c diabetic (>=6.5)",
    42: "HDL low (M<40 / F<46)", 43: "HDL normal",
    44: "LDL optimal (<100)", 45: "LDL borderline (100-159)", 46: "LDL high (>=160)",
    47: "eGFR normal (>=90)", 48: "eGFR mildly reduced (60-89)", 49: "eGFR CKD (<60)",
    50: "Sleep quality good (PSQI 0-4)", 51: "Sleep quality moderate (PSQI 5-8)",
    52: "Sleep quality poor (PSQI 9-16)",
    # --- keep-first events ---
    53: "High risk of sleep apnea (Berlin)",
    54: "Stroke, probable", 55: "Stroke, possible",
    56: "Major depression, probable", 57: "Major depression, possible",
    58: "History of hypertension", 59: "History of diabetes",
    60: "History of congestive heart failure", 61: "History of claudication",
    62: "History of heart condition",
    63: "Antihypertensive medication", 64: "Diabetes medication",
    65: "Statin medication", 66: "Alzheimer's disease medication",
    67: "Alzheimer's dementia diagnosis",
    68: "Death",   # reserved even when EMIT_DEATH=False, so the vocab never shifts
}
VOCAB_SIZE = max(LABELS) + 1     # 69
N_IGNORE   = 22                  # ids 0..21 -> ignore_tokens = list(range(22))

# Ordinal scales, run-length-encoded PER SCALE (keep-transitions).
SCALES = {
    "MMSE":  (22, 23, 24, 25),
    "COG":   (26, 27, 28, 29),
    "BMI":   (30, 31, 32),
    "BP":    (33, 34, 35),
    "GLU":   (36, 37, 38),
    "HBA1C": (39, 40, 41),
    "HDL":   (42, 43),
    "LDL":   (44, 45, 46),
    "GFR":   (47, 48, 49),
    "PSQI":  (50, 51, 52),
}
SCALE_GROUP = {i: name for name, ids in SCALES.items() for i in ids}

# The cohort filter in training/train.py needs the AD-staging scale in DISK space
# (model id - 1). MMSE is the NACCUDSD analogue here: an ordinal clinical staging scale.
STAGE_TOKENS_DISK = tuple(i - 1 for i in SCALES["MMSE"])   # (21, 22, 23, 24)


# ----------------------------- load + merge ----------------------------------
def load():
    xs   = pd.read_excel(XS)
    long = pd.read_excel(LONG)
    xs["study"]   = xs["study"].astype(str).str.strip()      # values ship as 'MAP ', 'ROS ', 'LATC'
    long["study"] = long["study"].astype(str).str.strip()

    assert set(xs.projid) == set(long.projid), "cross-sectional and longitudinal projid sets differ"
    assert not long.duplicated(["projid", "fu_year"]).any(), "(projid, fu_year) is not unique"

    J = long.merge(xs[["projid", "study", "age_bl", "msex", "educ", "apoe_genotype",
                       "smoking_bl", "ldai_bl", "age_first_ad_dx"]],
                   on="projid", how="left", suffixes=("", "_xs"))
    J["age"] = (J["age_bl"] + J["fu_year"]) * DAYS_PER_YEAR   # visit age, in DAYS
    print(f"  merged panel: {len(J)} visits, {J.projid.nunique()} subjects, "
          f"age {J.age.min()/DAYS_PER_YEAR:.1f}-{J.age.max()/DAYS_PER_YEAR:.1f}y")
    return xs, J


def num(df, c):
    """Numeric view of a column, with implausible values dropped (see PLAUSIBLE)."""
    s = pd.to_numeric(df[c], errors="coerce") if c in df else pd.Series(np.nan, index=df.index)
    if c in PLAUSIBLE:
        lo, hi = PLAUSIBLE[c]
        bad = s.notna() & ((s < lo) | (s > hi))
        if bad.any():
            print(f"    [guard] {c}: dropped {int(bad.sum())} implausible value(s) "
                  f"outside [{lo}, {hi}]: {sorted(s[bad].unique())[:5]}")
        s = s.where(~bad)
    return s


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--no-bmi", action="store_true",
                    help="build the EMIT_BMI=False variant into out_radc/radc_nobmi_all.bin "
                         "(the ablation arm; see CFG['EMIT_BMI'])")
    args = ap.parse_args(argv)
    if args.no_bmi:
        CFG["EMIT_BMI"] = False
    prefix = "radc_nobmi" if not CFG["EMIT_BMI"] else "radc"
    print(f"loading RADC ...  [EMIT_BMI={CFG['EMIT_BMI']} -> {prefix}_all.bin]")
    xs, J = load()
    ev = []   # each entry: DataFrame(pid, age, idx) in MODEL space

    def push(mask, age, idx):
        """Emit token `idx` at `age` for every row where `mask` holds."""
        m = np.asarray(mask.fillna(False) if hasattr(mask, "fillna") else mask, dtype=bool)
        if not m.any():
            return
        a = np.asarray(age)[m] if np.ndim(age) else np.full(m.sum(), age)
        ev.append(pd.DataFrame({"pid": J.loc[m, "projid"].to_numpy(), "age": a, "idx": idx}))

    # ---- statics from the cross-sectional file: one row per subject ----------
    # sex / APOE / study sit at age 0 (immutable from birth, same convention as the NACC
    # tokenizer); education / smoking / alcohol sit at the BASELINE visit age, because they
    # are baseline *measurements* (`_bl`) rather than birth facts.
    base_age = (xs["age_bl"] * DAYS_PER_YEAR).to_numpy()
    def push_xs(mask, age, idx):
        m = np.asarray(mask.fillna(False) if hasattr(mask, "fillna") else mask, dtype=bool)
        if not m.any():
            return
        a = np.asarray(age)[m] if np.ndim(age) else np.full(m.sum(), age)
        ev.append(pd.DataFrame({"pid": xs.loc[m, "projid"].to_numpy(), "age": a, "idx": idx}))

    push_xs(xs["msex"] == 1, 0, 2)                     # ROSMAP: msex 1 = male (cohort is 73% female)
    push_xs(xs["msex"] == 0, 0, 3)
    for geno, idx in [(22, 4), (23, 5), (24, 6), (33, 7), (34, 8), (44, 9)]:
        push_xs(xs["apoe_genotype"] == geno, 0, idx)
    for st, idx in [("ROS", 13), ("MAP", 14), ("LATC", 15)]:
        push_xs(xs["study"] == st, 0, idx)

    educ = pd.to_numeric(xs["educ"], errors="coerce")
    push_xs(educ.between(0, 12),  base_age, 10)
    push_xs(educ.between(13, 16), base_age, 11)
    push_xs(educ.between(17, 40), base_age, 12)

    smk = pd.to_numeric(xs["smoking_bl"], errors="coerce")   # codebook: 0 never, 1 former, 2 current
    for v, idx in [(0, 16), (1, 17), (2, 18)]:
        push_xs(smk == v, base_age, idx)

    ldai = pd.to_numeric(xs["ldai_bl"], errors="coerce")     # lifetime daily alcohol intake, drinks/day
    push_xs(ldai == 0, base_age, 19)
    push_xs((ldai > 0) & (ldai <= 1), base_age, 20)
    push_xs(ldai > 1, base_age, 21)

    # ---- longitudinal ordinal scales (keep-transitions applied later) -------
    a = J["age"].to_numpy()

    mmse = num(J, "cts_estmmse30")
    for lo, hi, idx in [(27, 30, 22), (24, 26, 23), (18, 23, 24), (0, 17, 25)]:
        push(mmse.between(lo, hi), a, idx)

    cog = num(J, "cogn_global")
    push(cog > 0.5, a, 26)
    push((cog > -0.5) & (cog <= 0.5), a, 27)
    push((cog > -1.5) & (cog <= -0.5), a, 28)
    push(cog <= -1.5, a, 29)

    if CFG["EMIT_BMI"]:
        bmi = num(J, "bmi")
        push(bmi <= 22, a, 30); push((bmi > 22) & (bmi <= 28), a, 31); push(bmi > 28, a, 32)
    else:
        print("    [EMIT_BMI=False] BMI tokens 30-32 NOT emitted "
              "(measurement depends on the outcome; see CFG)")

    # Blood pressure: joint AHA staging, so a visit needs both arms measured.
    sbp, dbp = num(J, "sbp_avg"), num(J, "dbp_avg")
    both = sbp.notna() & dbp.notna() if CFG["BP_REQUIRE_BOTH"] else (sbp.notna() | dbp.notna())
    st2 = both & ((sbp >= 140) | (dbp >= 90))
    st1 = both & ~st2 & ((sbp >= 130) | (dbp >= 80))
    push(both & ~st2 & ~st1, a, 33); push(st1, a, 34); push(st2, a, 35)

    glu = num(J, "glucose")
    push(glu < 100, a, 36); push(glu.between(100, 125.999), a, 37); push(glu >= 126, a, 38)

    hba = num(J, "hba1c")
    push(hba < 5.7, a, 39); push(hba.between(5.7, 6.4999), a, 40); push(hba >= 6.5, a, 41)

    # HDL threshold is sex-specific in the codebook: Male <40, Female <46 mg/dL.
    hdl = num(J, "hdlchlstrl"); male = pd.to_numeric(J["msex"], errors="coerce") == 1
    hdl_low = hdl.notna() & np.where(male, hdl < 40, hdl < 46)
    push(hdl_low, a, 42); push(hdl.notna() & ~hdl_low, a, 43)

    ldl = num(J, "ldlchlstrl")
    push(ldl < 100, a, 44); push(ldl.between(100, 159.999), a, 45); push(ldl >= 160, a, 46)

    gfr = num(J, "gfr_mdrs")
    push(gfr >= 90, a, 47); push(gfr.between(60, 89.999), a, 48); push(gfr < 60, a, 49)

    psqi = num(J, "psqi_sum")
    push(psqi.between(0, 4), a, 50); push(psqi.between(5, 8), a, 51); push(psqi.between(9, 16), a, 52)

    # ---- keep-first events -------------------------------------------------
    push(pd.to_numeric(J["berlin_risk_class"], errors="coerce") == 1, a, 53)

    # Codebook: 1 = highly probable, 2 = probable, 3 = possible, 4 = NOT PRESENT.
    # So 4 is the negative class -- the majority value (29140/36138 for stroke).
    for col, (probable, possible) in [("r_stroke", (54, 55)), ("r_depres", (56, 57))]:
        v = pd.to_numeric(J[col], errors="coerce")
        push(v.isin([1, 2]), a, probable)
        push(v == 3, a, possible)

    # `_cum` variables are monotone cumulative self-reported history (0/1) -- keep-first on
    # the first 1 gives age at first report. A 1 already present at fu_year 0 is prevalent
    # history and lands at baseline age, which is the honest placement.
    for col, idx in [("hypertension_cum", 58), ("dm_cum", 59), ("chf_cum", 60),
                     ("claudication_cum", 61), ("heart_cum", 62)]:
        push(pd.to_numeric(J[col], errors="coerce") == 1, a, idx)

    # Medications: 1 = taking at this visit. keep-first = age first observed on the drug.
    # Stop/restart is lost, the same simplification the NACC tokenizer makes.
    for col, idx in [("antihyp_rx", 63), ("diabetes_rx", 64), ("statin_rx", 65), ("ad_rx", 66)]:
        push(pd.to_numeric(J[col], errors="coerce") == 1, a, idx)

    # AD dementia diagnosis: age_first_ad_dx is the age at the first cycle with clinical
    # diagnosis summary 4 or 5, per the codebook -- "the best approximation of age at onset
    # of Alzheimer's dementia available". UNcensored in the cross-sectional file (66.3-107.2y),
    # unlike the "90+" version in ROSMAP_clinical.csv. 1164 subjects (26.3%).
    #
    # NOTE this one token sits on a DIFFERENT clock from every other longitudinal token:
    # age_first_ad_dx is a real age from the actual visit date, whereas scale tokens sit on
    # the nominal age_bl + fu_year grid. Measured disagreement is small -- median 0.07y
    # (~27 days) to the nearest nominal visit, 94.4% within 0.5y, 99.0% within 1.0y, worst
    # case 2.2y -- so the dx can land just before or just after the visit that produced it.
    # It is the same real event either way; it is not extra information.
    ad_age = pd.to_numeric(xs["age_first_ad_dx"], errors="coerce")
    push_xs(ad_age.notna(), (ad_age * DAYS_PER_YEAR).to_numpy(), 67)

    if CFG["EMIT_DEATH"]:
        clin = pd.read_csv(CLIN)
        d = pd.to_numeric(clin["age_death"], errors="coerce")     # non-numeric == the "90+" censor
        keep = clin.loc[d.notna(), ["projid"]].copy()
        keep["age"] = (d[d.notna()] * DAYS_PER_YEAR).to_numpy()
        keep["idx"] = 68
        keep = keep.rename(columns={"projid": "pid"})
        keep = keep[keep["pid"].isin(set(xs.projid))]
        print(f"  [death] emitting {len(keep)} death tokens -- BIASED: only subjects who died "
              f"before 90 have an uncensored age_death (see CFG['EMIT_DEATH'])")
        ev.append(keep[["pid", "age", "idx"]])

    # ----------------------------- assemble ---------------------------------
    E = pd.concat(ev, ignore_index=True).dropna(subset=["pid", "age", "idx"])
    E["age"] = np.rint(E["age"]).astype(np.int64)
    E = E[E["age"] >= 0]
    E["idx"] = E["idx"].astype(int)
    print(f"  raw events before dedup: {len(E)}")

    E["grp"] = E["idx"].map(SCALE_GROUP)
    is_scale = E["grp"].notna()
    is_static = E["idx"] < N_IGNORE

    # statics are already one-per-subject-per-id, but guard against a double-coded row
    static = (E[is_static].sort_values(["pid", "age", "idx"])
                          .drop_duplicates(subset=["pid", "idx"], keep="first"))
    # keep-first: age at first onset
    first = (E[~is_static & ~is_scale].sort_values(["pid", "age", "idx"])
                                      .drop_duplicates(subset=["pid", "idx"], keep="first"))
    # keep-transitions, PER SCALE: first-in-run + every bin change (decline AND recovery)
    scale = E[is_scale].sort_values(["pid", "grp", "age"], kind="stable")
    scale = scale[scale["idx"].ne(scale.groupby(["pid", "grp"])["idx"].shift())]

    print(f"  after dedup: static {len(static)} | keep-first {len(first)} | "
          f"keep-transitions {len(scale)}")
    for name in SCALES:
        n = int((scale["grp"] == name).sum())
        print(f"      {name:6s} {n:6d} transitions  ({n / static['pid'].nunique():.2f}/subject)")

    E = pd.concat([static.drop(columns="grp"), first.drop(columns="grp"),
                   scale.drop(columns="grp")], ignore_index=True)

    # OFF-BY-ONE: disk token = model id - 1 (get_batch adds the +1 back)
    E["tok"] = E["idx"] - 1

    # stable integer pids by sorted projid -- one consistent pid space across full + splits
    sorted_pids = sorted(E["pid"].unique())
    idmap = {p: i for i, p in enumerate(sorted_pids)}
    E["pidi"] = E["pid"].map(idmap).astype(np.uint32)
    E = E.sort_values(["pidi", "age", "tok"])          # patients contiguous, ages ascending

    full = E[["pidi", "age", "tok"]].to_numpy(np.uint32)
    full.tofile(os.path.join(OUT, f"{prefix}_all.bin"))
    print(f"\n  {prefix}_all.bin: {full.shape[0]} events, {len(sorted_pids)} subjects (FULL, unsplit)")
    print(f"                {full.shape[0] / len(sorted_pids):.1f} events/subject")

    pd.DataFrame({"pidi": range(len(sorted_pids)), "projid": sorted_pids}).to_csv(
        os.path.join(OUT, f"{prefix}_pidmap.csv"), index=False)

    rows = [LABELS.get(i, f"Reserved {i}") for i in range(VOCAB_SIZE)]
    pd.DataFrame({"event_name": rows}).to_csv(os.path.join(OUT, "labels.csv"), index=False)
    print(f"  labels.csv:   {len(rows)} rows (vocab_size = {VOCAB_SIZE})")

    # distinct-age histogram: this, NOT the visit count, is what train.py's cohort filter
    # counts, because keep-transitions collapses visits where nothing changed.
    ages_per = E[E["age"] > 0].groupby("pidi")["age"].nunique()
    ages_per = ages_per.reindex(range(len(sorted_pids)), fill_value=0)
    print("\n  distinct event ages per subject (what cohort_min_visits actually filters on):")
    for k in (1, 2, 3, 4, 5, 6, 8):
        print(f"      >= {k}: {int((ages_per >= k).sum())} subjects")
    print(f"\n  config/train_radc.py should set: vocab_size = {VOCAB_SIZE}, "
          f"ignore_tokens = list(range({N_IGNORE})),")
    print(f"                                   stage_tokens_disk = {STAGE_TOKENS_DISK}")
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
