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
    # ---- dementia status AT ENTRY, and this is a LABEL-CORRECTNESS fix, not a feature ----
    # age_first_ad_dx is blank for two completely different reasons, and the tokenizer used to
    # treat them alike. The codebook is explicit: "This measure is not available for
    # participants that were demented at baseline cycle." So a blank means EITHER "never had
    # AD" (2,242 confirmed) OR "already had it when we met them" (229). Emitting no AD token
    # for both told the model that 229 subjects lived their whole recorded life AD-free when
    # in fact they had it throughout.
    #
    # Identification follows the codebook's own definition: age_first_ad_dx is the age at the
    # first cycle with clinical diagnosis summary in {4, 5}. So a blank age PLUS a last-visit
    # diagnosis of dementia can only mean the first such cycle preceded baseline. Dementia does
    # not reverse, so inferring the baseline state from the last visit is sound rather than
    # circular. Measured separation: baseline MMSE median 22.0 for this group against 29.0 for
    # the confirmed-never-demented and 28.0 for incident cases.
    #
    # THIS IS A STATIC, NOT AN AD EVENT. Emitting token "Alzheimer's dementia diagnosis" at
    # baseline instead would put 229 incident events into the cumulative-incidence target and
    # move the Aalen-Johansen calibration from 0.177/0.281 to something else. The AD token
    # keeps meaning INCIDENT; prevalence is background.
    #
    # The UNKNOWN level is not optional. 793 subjects (17.9%, including all 308 LATC) have no
    # ROSMAP_clinical row, so their baseline status cannot be determined either way. Without an
    # explicit id the static block would be one token shorter for exactly them, which is the
    # covert-channel problem the APOE-unknown token exists to avoid.
    "Dementia at entry",                                # 23
    "No dementia at entry",                             # 24
    "Dementia at entry unknown",                        # 25
    # ---- ordinal scales, hysteresis keep-transitions, predicted (23..35) ----
    # MMSE is SIX levels, not four, and the three extra cuts are all above 26. 73.6% of
    # visits sit in the old top bin, so a single 27-30 level made the model blind to three
    # quarters of the data -- measured: zero emitted tokens for any change inside it, while
    # splitting it yields 3,993. The bottom of the scale is deliberately NOT split: below 18
    # the measurement SD is 3.0 points (against 0.6 at the ceiling) and the training set holds
    # ~840 observations there, so finer bins would buy neither resolution nor learnability.
    "MMSE <18",                                         # 23
    "MMSE 18-23",                                       # 24
    "MMSE 24-26",                                       # 25
    "MMSE 27",                                          # 26
    "MMSE 28",                                          # 27
    "MMSE 29",                                          # 28
    "MMSE 30",                                          # 29
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
STATIC_FIRST, STATIC_LAST = ID["Sex: male"], ID["Dementia at entry unknown"]   # 2 .. 25
STATIC_IDS = tuple(range(STATIC_FIRST, STATIC_LAST + 1))

# Ordinal scales. Each is run-length-encoded SEPARATELY, so a change in one scale survives
# even while the others sit still. Order within each tuple is low -> high on the value, which
# for the cognitive scales means WORST first: severity therefore DECREASES with id inside
# MMSE and cogn_global. That is deliberate -- it keeps the printed table in numeric order of
# the underlying measurement -- but any code that computes "worsened by >= 1 bin" must use
# SEVERITY_ORDER below rather than assuming a higher id is worse.
SCALES = {
    "MMSE": (ID["MMSE <18"], ID["MMSE 18-23"], ID["MMSE 24-26"],
             ID["MMSE 27"], ID["MMSE 28"], ID["MMSE 29"], ID["MMSE 30"]),
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

# The three-level baseline dementia status. Exported by name so the tokenizer and the
# evaluation cohort read the same ids rather than each re-deriving the rule.
DEMENTIA_ENTRY = {
    "yes": ID["Dementia at entry"],
    "no": ID["No dementia at entry"],
    "unknown": ID["Dementia at entry unknown"],
}

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

# WHAT COUNTS AS "THE NEXT EVENT" for the time-to-event target -- deliberately NOT the same
# set as IGNORE_TOKENS, and conflating the two was a live bug in the delivered model.
#
# IGNORE_TOKENS answers "which positions are not cross-entropy targets", and No-event must NOT
# be in it: the synthetic markers are the only negative evidence in the stream, so the model
# has to be scored on predicting them.
#
# This set answers a different question -- "which tokens does the waiting-time target skip".
# No-event MUST be in it, because generate() masks the marker and can never emit one, so the
# intensity has to be trained on the gap it will actually be asked to reproduce. Measured on
# the delivered build, 41.3% of dt targets pointed at a synthetic marker, which compressed the
# target mean from 1.65 y to 0.89 y and turned 8.5% of censored positions into fabricated
# short answers.
DT_IGNORE_TOKENS = list(IGNORE_TOKENS) + [NO_EVENT]

# Tokens whose ages get jittered to break immortality bias (DISK space, for get_batch).
# The statics are recorded traits, not events that happened at the baseline visit, so pinning
# them to that exact age teaches the model that seeing them implies survival to it.
AUGMENT_TOKENS_DISK = tuple(t - 1 for t in STATIC_IDS)


# ---------------------------------------------------------------- continuous-value support
# Bin edges in the RAW units of each scale, low -> high. len(edges) == len(levels) - 1.
SCALE_EDGES = {
    # HALF-INTEGER edges above 26, and that is not cosmetic: 98.4% of cts_estmmse30 values
    # are integers, so an edge ON an integer splits a mode and the level between two such
    # edges is never the modal bin. Measured with edges at 28 and 29, the level "28-29"
    # received 4 tokens in the entire cohort against 3,803 and 4,659 either side. With edges
    # at 27.5 / 28.5 / 29.5 each integer sits in its own bin's centre, which is also why the
    # top four levels are named for the integer they carry.
    "MMSE": (18.0, 24.0, 26.5, 27.5, 28.5, 29.5),
    "COG": (-2.0, -1.0, -0.3, 0.2, 0.8),
    "BMI": (20.0, 30.0),
}

# MEASUREMENT NOISE, in raw units, as (value, sigma) anchors interpolated piecewise-linearly
# and held flat outside the range. This is the load-bearing constant of the whole soft-label
# scheme -- the weights are a posterior given sigma, so a wrong sigma is a wrong posterior --
# so it is MEASURED, not assumed.
#
# ESTIMATOR. For three consecutive visits one year apart, the second difference
# v1 - 2*v2 + v3 has variance 6*sigma^2 if the true trajectory is locally linear, so
# sigma = SD(second difference) / sqrt(6). SD is taken as 1.4826 * MAD so that genuine sharp
# decline does not inflate it. Computed on 21,214 (MMSE), 23,636 (cognition) and 15,891 (BMI)
# eligible triples.
#
# All three are heteroscedastic, and BMI runs the OPPOSITE way to the cognitive scales --
# heavier subjects vary more, while the cognitive instruments get more precise as they
# saturate. A single constant would be wrong for every one of them.
SCALE_SIGMA = {
    "MMSE": ((18.0, 2.42), (26.0, 1.82), (27.5, 1.21), (29.5, 0.61)),
    "COG": ((-1.5, 0.290), (-1.0, 0.223), (-0.15, 0.189), (0.5, 0.161), (1.2, 0.147)),
    "BMI": ((20.0, 0.759), (25.0, 0.837), (32.0, 1.110)),
}

# Total-variation distance between consecutive soft weight vectors, above which a token is
# emitted. This REPLACES the hysteresis margins: hysteresis was a stateful, path-dependent
# filter that disagreed with the raw bin on 6.2% of MMSE visits and deleted real recoveries
# (measured: a 23.0 -> 25.9 rebound emitted nothing at all). A TV threshold is stateless and
# is expressed in units of the measurement noise, so a sub-sigma move can never cross it --
# measured 0.0% of emissions correspond to a change below 1 sigma, against 11.8% under hard
# binning. Tuned so sequence length stays close to the delivered build.
SCALE_TV_THRESHOLD = {"MMSE": 0.55, "COG": 0.55, "BMI": 0.50}

MAX_LEVELS = max(len(v) for v in SCALES.values())
SCALE_ORDER = tuple(SCALES.keys())                       # scale index 0, 1, 2 ...


def scale_tables():
    """Padded lookup tables for the vectorised soft-weight path in batching.get_batch.

    Returns (tok2scale, edges, ids, sig_x, sig_y), all numpy:
        tok2scale (VOCAB_SIZE,)            scale index per MODEL token id, -1 if not a scale
        edges     (S, MAX_LEVELS-1)        bin edges, padded with +inf so the padded levels
                                           receive exactly zero weight
        ids       (S, MAX_LEVELS)          MODEL ids of each level, padded with -1
        sig_x/y   (S, A)                   sigma anchors, padded by repeating the last point
    """
    import numpy as _np
    S, K = len(SCALE_ORDER), MAX_LEVELS
    tok2scale = _np.full(VOCAB_SIZE, -1, dtype=_np.int64)
    edges = _np.full((S, K - 1), _np.inf)
    ids = _np.full((S, K), -1, dtype=_np.int64)
    A = max(len(SCALE_SIGMA[s]) for s in SCALE_ORDER)
    sig_x = _np.zeros((S, A))
    sig_y = _np.zeros((S, A))
    for si, name in enumerate(SCALE_ORDER):
        lv = SCALES[name]
        ids[si, :len(lv)] = lv
        tok2scale[list(lv)] = si
        e = SCALE_EDGES[name]
        edges[si, :len(e)] = e
        a = SCALE_SIGMA[name]
        xs = [p[0] for p in a] + [a[-1][0]] * (A - len(a))
        ys = [p[1] for p in a] + [a[-1][1]] * (A - len(a))
        sig_x[si], sig_y[si] = xs, ys
    return tok2scale, edges, ids, sig_x, sig_y


def sigma_of(scale, v):
    """Measured measurement SD of `scale` at raw value `v` (piecewise linear, flat outside)."""
    import numpy as _np
    a = SCALE_SIGMA[scale]
    xs = _np.array([p[0] for p in a], dtype=float)
    ys = _np.array([p[1] for p in a], dtype=float)
    return float(_np.interp(v, xs, ys))


def soft_weights(scale, v, sigma=None):
    """P(the TRUE value falls in each bin | a noisy observation `v`), as a numpy vector.

    Under a flat prior this is the exact posterior, not a heuristic softening:
        w_k = Phi((edge_k - v) / sigma) - Phi((edge_{k-1} - v) / sigma)

    Why it matters that this is a posterior. Training against these as soft targets makes the
    model's softmax converge to E[w | history], and by the tower property that equals
    P(true value in bin k | history) -- the predictive distribution over the TRUE value, with
    measurement noise deconvolved out. Hard labels instead ask the model to predict the NOISY
    observation, which is both harder and not what anyone wants to know.
    """
    import numpy as _np
    from math import erf, sqrt
    e = _np.asarray(SCALE_EDGES[scale], dtype=float)
    s = sigma if sigma is not None else sigma_of(scale, v)
    z = (e - v) / s
    cdf = _np.concatenate([[0.0], 0.5 * (1.0 + _np.vectorize(erf)(z / sqrt(2.0))), [1.0]])
    return _np.diff(cdf)


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
    n_rep = sum(len(v) for v in SCALES.values()) + len(ONSET_IDS) + len(MED_IDS)
    assert len(REPEATABLE_TOKENS) == n_rep, \
        f"repeatable set has {len(REPEATABLE_TOKENS)} ids, the groups imply {n_rep}"
    assert all(t not in IGNORE_TOKENS for t in ENDPOINT_IDS), "an endpoint is masked out of the loss"
    return True


check()

# ---------------------------------------------------------------- resolving ANY label table
# The module constants above describe the LIVE tokenization. `resolve` produces the same sets
# for an arbitrary label table -- i.e. for a checkpoint from an older build, read out of that
# build's labels.csv.
#
# WHY BY NAME AND NOT BY ID. Ids move: v1 had 50 tokens, v3 has 56, and the three
# `Dementia at entry` levels were inserted in the MIDDLE, so every id above 22 shifted. The
# NAMES did not move -- "Death", "Alzheimer's dementia diagnosis", "Hypertension, history" and
# the scale prefixes are identical in both. Matching on name is therefore the only mapping that
# survives a vocabulary change, and it is the reason the ported Figure 2 can put a v1 and a v3
# checkpoint on the same axes.
#
# Every classification below is ASSERTED to be exhaustive over the non-static content tokens.
# A renamed or newly added token fails loudly here rather than being silently dropped from the
# repeatable set (which would make generate() assign probability zero to a legitimate recovery)
# or from the ignore set (which would put an unconstrained logit column into a softmax).
_SCALE_PREFIX = {"MMSE": "MMSE", "COG": "Global cognition", "BMI": "BMI"}


class Resolved:
    """Id sets for one label table. Attribute names mirror this module's constants."""

    def __init__(self, labels):
        self.NAMES = list(labels)
        self.VOCAB_SIZE = len(self.NAMES)
        self.ID = {n: i for i, n in enumerate(self.NAMES)}
        need = ("Padding", "No event", "Death", "Alzheimer's dementia diagnosis")
        missing = [n for n in need if n not in self.ID]
        if missing:
            raise ValueError(f"label table is missing {missing}; not a RADC tokenization")
        self.PADDING = self.ID["Padding"]
        self.NO_EVENT = self.ID["No event"]
        self.DEATH = self.ID["Death"]
        self.AD_DX = self.ID["Alzheimer's dementia diagnosis"]
        self.ENDPOINT_IDS = (self.AD_DX, self.DEATH)
        self.TERMINATION_TOKENS = (self.DEATH,)

        # ordinal scales, in label order == low -> high on the measurement, which for the two
        # cognitive scales means WORST FIRST (see SEVERITY_ORDER above)
        self.SCALES = {}
        for key, pre in _SCALE_PREFIX.items():
            ids = tuple(i for i, n in enumerate(self.NAMES) if n.startswith(pre))
            if ids:
                self.SCALES[key] = ids
        self.SEVERITY_ORDER = {k: v for k, v in self.SCALES.items() if k != "BMI"}
        self.SCALE_OF = {t: k for k, ids in self.SCALES.items() for t in ids}

        self.KEEP_FIRST_IDS = tuple(i for i, n in enumerate(self.NAMES)
                                    if n.endswith(", history"))
        self.ONSET_IDS = tuple(i for i, n in enumerate(self.NAMES)
                               if n.startswith(("Stroke,", "Depression,")))
        self.MED_IDS = tuple(i for i, n in enumerate(self.NAMES)
                             if n.endswith((" started", " stopped")))
        self.REPEATABLE_TOKENS = tuple(sorted(
            [t for ids in self.SCALES.values() for t in ids]
            + list(self.ONSET_IDS) + list(self.MED_IDS)))

        # Statics are whatever is left. They are contiguous in every build, so derive the range
        # rather than name them: the v1 block is 2..22 and the v3 block 2..25.
        classified = (set(self.REPEATABLE_TOKENS) | set(self.KEEP_FIRST_IDS)
                      | {self.PADDING, self.NO_EVENT, self.AD_DX, self.DEATH})
        statics = sorted(set(range(self.VOCAB_SIZE)) - classified)
        if statics != list(range(statics[0], statics[-1] + 1)):
            raise ValueError(f"the unclassified tokens are not contiguous: {statics}. Either a "
                             f"token was renamed out of one of the families above, or a new "
                             f"family was added and this resolver has not been told about it.")
        self.STATIC_FIRST, self.STATIC_LAST = statics[0], statics[-1]
        self.STATIC_IDS = tuple(statics)
        self.IGNORE_TOKENS = [self.PADDING] + list(statics)
        self.DT_IGNORE_TOKENS = list(self.IGNORE_TOKENS) + [self.NO_EVENT]
        # the partition must cover the table exactly
        total = classified | set(statics)
        assert total == set(range(self.VOCAB_SIZE)), (
            f"resolver did not cover every token: {sorted(set(range(self.VOCAB_SIZE)) - total)}")


def resolve(labels):
    """`Resolved` for a label table (a list of names in id order, i.e. labels.csv's column)."""
    return Resolved(labels)


def resolve_csv(path):
    """`Resolved` for a build's labels.csv."""
    import pandas as _pd
    return Resolved([str(x) for x in _pd.read_csv(path)["event_name"].tolist()])

