"""
vocab.py -- the RADC token table. The single source of truth for what every id means.

Nothing else in this package may hard-code a token id. The NACC arm this replaces scattered
its ids across the engine, the adapter, the figure code and the training configs, and every
one of them had to be edited in step; several were not, which is how a checkpoint could be
scored against a vocabulary it was not trained on.

TWO TOKEN SPACES, and confusing them is the classic bug:
    MODEL space   what the model sees, and the row index into labels.csv. 0 = Padding,
                  1 = No event, 2.. = content.
    DISK space    what is stored in the .bin files (data[:, 2]). disk = model - 1, because
                  batching.get_batch adds the +1 back when it assembles a batch.
Everything in this module is MODEL space unless the name ends in _DISK.

WHY THESE TOKENS AND NOT OTHERS. The table is the output of a seven-axis measurement pass over
the raw files, each axis independently re-measured by a second pass. The decisions that shaped
it, with the numbers that forced them:

  * The autopsy block and cogng_demog_slope are absent, and not merely "placed late". Autopsy
    availability is a perfect death oracle -- all 2,231 subjects with any autopsy value have
    died == 1 -- so a missingness indicator alone reaches AUC 0.906 for death; the token's
    EXISTENCE is the leak, so no placement rule rescues it. cogng_demog_slope reaches AUC 0.775
    for AD among subjects who were cognitively normal at baseline and diagnosed >= 10 y later,
    a subset in which baseline cognition itself is at chance (0.502). It is the outcome.
  * ad_rx is absent by default. Among pre-diagnosis visits with a normal MMSE of 27-30, being
    on an AD drug raises P(AD dx within 3 y) from 4.7% to 31.2% -- a 6.6x jump that measured
    cognition does not explain, because the prescription encodes the treating clinician's
    diagnosis before it reaches age_first_ad_dx.
  * ROSMAP_clinical.csv contributes exactly one thing: the numeric age_death of 975 subjects.
    Its other end-of-study columns discriminate AD at AUC 0.83-0.89 by construction.
  * Both cognitive scales are kept: they are complementary rather than redundant (knowing one
    4-level bin improves guessing the other by only 5-8 pp over the marginal), and within the
    MMSE 27-30 ceiling -- 73% of visits -- the cogn_global bin lifts 5-year AD AUC from 0.750
    to 0.846. cogn_global also starts moving 3-4 years earlier than MMSE.
  * Of the biomarker block only BMI survives. Tokenized in full it would cost 21 ids and 51%
    of every sequence while carrying almost no AD signal (likelihood-ratio chi2 72.8 for BMI
    against <= 8.5 for every other scale), and most of the panel is a study indicator in
    disguise -- psqi_sum is 99.7% present in MAP and 26.1% in ROS, hba1c is exactly 0% before
    calendar ~2004, and the inflammation trio is a one-shot 426-subject MAP substudy.
  * r_stroke and r_depres are NOT cumulative. 83.1% of subjects who ever score 1/2/3 revert to
    4 later, so keep-first would discard 21% of stroke onsets; they are emitted on every
    worsening transition instead. The five _cum variables ARE strictly monotone (zero 1 -> 0
    reversals across all 4,428 subjects), so keep-first is correct for those.

VALUE CODINGS THAT READ BACKWARDS -- verified against the codebook and against an independent
physiological anchor, and reproduced here verbatim so nobody has to re-derive them:
    r_stroke, r_depres : 1 = highly probable, 2 = probable, 3 = possible, 4 = NOT PRESENT.
                         4 is the MODAL value (93.9% / 93.7% of non-null rows). Reading it as
                         "most severe" inverts the label.
    msex               : 1 = Male, 0 = Female. 73.15% of subjects are msex == 0, matching the
                         known ~73%-female ROSMAP cohort; msex == 1 rows have mean HDL 52.0
                         against 63.3, which is the correct sex direction.
    smoking_bl         : 0 = never, 1 = former, 2 = current.
    ldai_bl            : lifetime daily alcohol intake, in drinks/day.
"""

# ---------------------------------------------------------------- the table
# Order is the id order. Keep the blocks contiguous: the statics must occupy one unbroken
# range so IGNORE_TOKENS below can stay a range() rather than a hand-maintained list.
NAMES = [
    "Padding",                                          # 0  -- reserved by the data loader
    "No event",                                         # 1  -- reserved by the data loader
    # ---- statics, placed one day before the baseline visit, held OUT of the loss (2..22) ----
    "Sex: male",                                        # 2
    "Sex: female",                                      # 3
    "APOE e2 carrier (22/23)",                          # 4
    "APOE e3/e3",                                       # 5
    "APOE e4 heterozygote (24/34)",                     # 6
    "APOE e4 homozygote (44)",                          # 7
    "APOE unknown",                                     # 8
    "Education <=12y",                                  # 9
    "Education 13-16y",                                 # 10
    "Education >=17y",                                  # 11
    "Study: ROS",                                       # 12
    "Study: MAP",                                       # 13
    "Study: LATC",                                      # 14
    "Smoking: never",                                   # 15
    "Smoking: former",                                  # 16
    "Smoking: current",                                 # 17
    "Smoking: unknown",                                 # 18
    "Alcohol: none (<1 drink/month)",                    # 19
    "Alcohol: light",                                   # 20
    "Alcohol: heavy",                                   # 21
    "Alcohol: unknown",                                 # 22
    # ---- ordinal scales, hysteresis keep-transitions, predicted (23..35) ----
    "MMSE 0-17",                                        # 23
    "MMSE 18-23",                                       # 24
    "MMSE 24-26",                                       # 25
    "MMSE 27-30",                                       # 26
    "Global cognition <=-2.0",                          # 27
    "Global cognition -2.0..-1.0",                      # 28
    "Global cognition -1.0..-0.3",                      # 29
    "Global cognition -0.3..0.2",                       # 30
    "Global cognition 0.2..0.8",                        # 31
    "Global cognition >0.8",                            # 32
    "BMI <20",                                          # 33
    "BMI 20-30",                                        # 34
    "BMI >=30",                                         # 35
    # ---- cumulative self-reported history, keep-first, predicted (36..39) ----
    "Hypertension, history",                            # 36
    "Diabetes, history",                                # 37
    "Claudication, history",                            # 38
    "Heart condition, history",                         # 39
    # ---- graded per-visit judgements, onset-transitions, predicted (40..43) ----
    "Stroke, probable",                                 # 40
    "Stroke, possible",                                 # 41
    "Depression, probable",                             # 42
    "Depression, possible",                             # 43
    # ---- medications, start/durable-stop, predicted (44..47) ----
    "Antihypertensive started",                         # 44
    "Antihypertensive stopped",                         # 45
    "Statin started",                                   # 46
    "Statin stopped",                                   # 47
    # ---- the endpoints (48..49) ----
    "Alzheimer's dementia diagnosis",                   # 48
    "Death",                                            # 49
]

ID = {n: i for i, n in enumerate(NAMES)}
VOCAB_SIZE = len(NAMES)                                 # 50

PADDING = ID["Padding"]                                 # 0
NO_EVENT = ID["No event"]                               # 1
AD_DX = ID["Alzheimer's dementia diagnosis"]            # 48
DEATH = ID["Death"]                                     # 49

# ---------------------------------------------------------------- blocks
STATIC_FIRST, STATIC_LAST = ID["Sex: male"], ID["Alcohol: unknown"]      # 2 .. 22
STATIC_IDS = tuple(range(STATIC_FIRST, STATIC_LAST + 1))

# Ordinal scales. Each is run-length-encoded SEPARATELY, so a change in one scale survives
# even while the others sit still. Order within each tuple is low -> high on the value, which
# for the cognitive scales means WORST first: severity therefore DECREASES with id inside
# MMSE and cogn_global. That is deliberate -- it keeps the printed table in numeric order of
# the underlying measurement -- but any code that computes "worsened by >= 1 bin" must use
# SEVERITY_ORDER below rather than assuming a higher id is worse.
SCALES = {
    "MMSE": (ID["MMSE 0-17"], ID["MMSE 18-23"], ID["MMSE 24-26"], ID["MMSE 27-30"]),
    "COG": (ID["Global cognition <=-2.0"], ID["Global cognition -2.0..-1.0"],
            ID["Global cognition -1.0..-0.3"], ID["Global cognition -0.3..0.2"],
            ID["Global cognition 0.2..0.8"], ID["Global cognition >0.8"]),
    "BMI": (ID["BMI <20"], ID["BMI 20-30"], ID["BMI >=30"]),
}
SCALE_OF = {t: name for name, ids in SCALES.items() for t in ids}

# Worst -> best, per scale. BMI is not ordered by severity (both tails are adverse), so it is
# deliberately absent: asking "did BMI worsen" is not a well-posed question here.
SEVERITY_ORDER = {
    "MMSE": SCALES["MMSE"],            # index 0 = worst
    "COG": SCALES["COG"],
}

KEEP_FIRST_IDS = (ID["Hypertension, history"], ID["Diabetes, history"],
                  ID["Claudication, history"], ID["Heart condition, history"])
ONSET_IDS = (ID["Stroke, probable"], ID["Stroke, possible"],
             ID["Depression, probable"], ID["Depression, possible"])
MED_IDS = (ID["Antihypertensive started"], ID["Antihypertensive stopped"],
           ID["Statin started"], ID["Statin stopped"])
ENDPOINT_IDS = (AD_DX, DEATH)

# ---------------------------------------------------------------- training constants
# Held OUT of the loss: padding, the no-event grid, and every static. They stay in the input
# stream and are still attended to -- this is a "do-not-predict" mask, NOT a feature ablation.
# Predicting a subject's sex or APOE from their own sex token is free accuracy that drowns the
# time-to-event term in exactly the way the upstream Delphi paper masks against.
# NOT a contiguous range, and the gap matters: id 1 (No event) IS a training target. The
# synthetic no-event markers are how the model learns that time passes without a record, so
# masking them out of the loss would remove the only negative evidence in the stream.
# model.py adds id 1 to the ignored set on its own under validation_loss_mode, so validation
# loss stays comparable to a run that never emitted them -- do not "simplify" this to range().
IGNORE_TOKENS = [PADDING] + list(range(STATIC_FIRST, STATIC_LAST + 1))   # 0, then 2..22

# The ordinal staging scale the training cohort filter counts transitions on. MMSE is the
# clinical staging analogue here: RADC has no per-visit diagnosis (cogdx and dcfdx are
# last-visit only, and both leak). DISK space, because the filter reads the raw .bin.
STAGE_TOKENS_DISK = tuple(t - 1 for t in SCALES["MMSE"])

# Exempt from generate()'s no_repeat. Every ordinal scale level encodes a CURRENT STATE, and
# states recur: a subject who goes MMSE 27-30 -> 24-26 -> 27-30 emits the top bin twice, which
# is exactly the recovery hysteresis keep-transitions exists to preserve. Everything else
# (incident histories, first medication, the diagnosis, death) is once-only and stays blocked.
# 21 ids: every ordinal-scale bin (23-35), both graded onset families (40-43) and all four
# medication events (44-47). The graded families repeat because r_stroke and r_depres are
# per-visit judgements that revert -- 83.1% of subjects who ever score present return to "not
# present" -- so the same onset token legitimately fires again on the next rise. The
# medications repeat because 22% of ever-users stop and restart.
REPEATABLE_TOKENS = tuple(sorted(
    [t for ids in SCALES.values() for t in ids] + list(ONSET_IDS) + list(MED_IDS)))

# Absorbing states. Death only -- the AD diagnosis is not absorbing, since subjects go on
# being observed, and dying afterwards is the competing risk the evaluation must respect.
TERMINATION_TOKENS = (DEATH,)

# Tokens whose ages get jittered to break immortality bias (DISK space, for get_batch).
# The statics are recorded traits, not events that happened at the baseline visit, so pinning
# them to that exact age teaches the model that seeing them implies survival to it.
AUGMENT_TOKENS_DISK = tuple(t - 1 for t in STATIC_IDS)


def labels():
    """The labels.csv column, in model-id order."""
    return list(NAMES)


def describe(model_id):
    return NAMES[model_id] if 0 <= model_id < VOCAB_SIZE else f"<out of range {model_id}>"


def check():
    """Internal consistency. Called by the tokenizer and by the tests so a hand-edit to the
    table above cannot silently produce a vocabulary that no longer partitions."""
    assert len(set(NAMES)) == len(NAMES), "duplicate token name"
    assert NAMES[0] == "Padding" and NAMES[1] == "No event", "0/1 are reserved by the loader"
    grouped = set(STATIC_IDS) | set(SCALE_OF) | set(KEEP_FIRST_IDS) | set(ONSET_IDS) \
        | set(MED_IDS) | set(ENDPOINT_IDS)
    missing = set(range(2, VOCAB_SIZE)) - grouped
    assert not missing, f"content tokens in no group: {sorted(missing)}"
    assert set(IGNORE_TOKENS) == {PADDING} | set(STATIC_IDS), \
        "ignore mask must be padding + statics, and must NOT contain No event"
    assert NO_EVENT not in IGNORE_TOKENS, "No event is a training target, not an ignored token"
    assert len(REPEATABLE_TOKENS) == 21, f"expected 21 repeatable ids, got {len(REPEATABLE_TOKENS)}"
    assert all(t not in IGNORE_TOKENS for t in ENDPOINT_IDS), "an endpoint is masked out of the loss"
    return True


check()
