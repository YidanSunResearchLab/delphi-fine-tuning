"""
radc_states.py -- the ONE place Figure 2 is allowed to know what a "state" is.

Figure 2 was written against NACC, where the four panels all hang off a single variable:
NACCUDSD, a per-visit CLINICAL STAGING with four levels (Normal / Impaired-not-MCI / MCI /
Dementia) plus Death as an absorbing fifth. Every panel's meaning is defined relative to it:

    a   discrimination for REACHING each stage within 5 y, one row per (baseline, reached)
    b   timing error for three ENDPOINTS, and AUC vs horizon for the same three
    c   observed vs predicted STAGE-AT-AGE on a yearly grid, per individual
    d   does the baseline embedding encode the FUTURE trajectory class, or only current stage

This module supplies the ROSMAP equivalents, and nothing else in figure2/ may hard-code a
token id. That rule is not stylistic: it is what stops a checkpoint being scored against a
vocabulary it was never trained on.

TWO PORTS DEEP. Upstream wrote these panels against NACC, then ported them to its own RADC
tokenization (see ../FIGURE2_upstream.md). This is the second port, onto the 129-token ROSMAP
build in ../../tokenization. Where upstream's substitution still holds it is kept and credited;
where our tokenizer differs, the difference is stated here rather than absorbed silently.

------------------------------------------------------------------------------------------
THE MAPPING, AND WHERE IT IS EXACT vs WHERE IT IS A SUBSTITUTION

  NACCUDSD staging  ->  MMSE, collapsed to stages. INHERITED from upstream, with the same
      justification and the same caveat. MMSE is the staging analogue for this cohort because
      ROSMAP has no usable per-visit diagnosis: ../../tokenization/spec.py EXCLUDES cogdx and
      dcfdx_lv as end-of-study summaries that leak when used as input (meta.json "excluded").

      WHERE WE DIFFER FROM UPSTREAM: THREE STAGES, NOT FOUR. Upstream had seven emitted MMSE
      levels and could group them onto the four Folstein cuts (>=27 / 24-26 / 18-23 / <18).
      Our tokenizer emits THREE (spec.py: cuts at 24 and 27), so the two lower Folstein stages
      -- Moderate and Severe -- are not separable here: they are one `MMSE_impaired` token.
      The stages below are therefore Normal (>=27) / Borderline (24-26) / Impaired (<24), and
      panel a cannot say anything about severe impairment specifically. This is a property of
      the .bin, not a choice made here; recovering it needs a retokenization with a cut at 18.

      A stage is still a pure GROUPING of already-emitted tokens, computed at evaluation time.
      No retokenization, no new model, the .bin untouched. With three levels the grouping
      happens to be one-to-one, so `STAGE_MODE` makes no difference on this build -- kept
      because it will the moment the MMSE binning gains a level.

      WHAT IS NOT PRESERVED, stated plainly: NACCUDSD is a clinician's diagnosis, MMSE is a
      test score. Panel a's row "Normal -> Impaired" reads "reached an MMSE below 24", not
      "was diagnosed as demented".

  panel b's "Dementia" endpoint  ->  the AD_DX token.
      ROSMAP is strictly BETTER off than NACC here: it has a real incident-diagnosis endpoint
      (age_first_ad_dx), so the endpoint panel keeps a diagnosis where the state panels cannot.
      That is why AD is carried as an EVENT slot alongside the stages rather than as a stage --
      a subject does not stop occupying an MMSE stage by being diagnosed.

      ONE CAVEAT UPSTREAM DID NOT HAVE, and it belongs on panel b. ../../tokenization/README.md
      records that AD_DX is SNAPPED to the annual grid, displacing it by a median 26 days and up
      to 182 days, and that ROSMAP supplies only a single `age_first_ad_dx` rather than a
      per-visit diagnosis. Timing claims about AD are therefore quantised at one year.

  Death  ->  Death. Absorbing, and the competing risk for every cognitive transition.

  the 30 NACC sub-scales scored by perdomain.py  ->  our three scored ordinal scales
      (ORDINAL_SCALES below), seven once-only incident events and eight medication events.
      Upstream's two GRADED onset families (`Stroke, probable` / `Stroke, possible`) have no
      counterpart: our tokenizer emits a single once-only onset token per condition, so those
      outcomes are in KEEP_FIRST_IDS and vocab.ONSET_IDS is empty.

  perdomain.py's reconstructed TOTALS  ->  nothing, and the block stays deleted. Those existed
      to falsify the CDR/FAQ/NPI-Q item split by summing domains back into the retired total.
      Neither RADC nor ROSMAP ever had an item split, so there is no total to reconstruct.

  the 12 non-scored ordinal families (SBP, DBP, GLU, HBA1C, HDL, LDL, GFR, PSQI, APNEA, CRP,
      IL6, TNFA)  ->  resolved in vocab.py so the SAMPLER treats them as repeatable state bins,
      but deliberately NOT in ORDINAL_SCALES. perdomain scores an ordinal scale as "worsens
      >= 1 bin", and that needs a severity direction; ../../tokenization/spec.py gives these
      families a measurement order, not a clinical one. Scoring them as "moves >= 1 bin" would
      be defensible but would triple the outcome count and break plotting_style's validated
      palette, so they are left out rather than added carelessly.

------------------------------------------------------------------------------------------
STAGE_MODE

"mmse_stages" (default) groups the emitted MMSE levels onto the stages above. "mmse_levels"
keeps every emitted level as its own stage. On THIS build the two are identical, because the
tokenizer emits exactly one token per stage; the machinery survives for the build that does
not.
"""
import numpy as np

from radc_delphi import vocab as V

STAGE_MODE = "mmse_stages"

# --------------------------------------------------------------------------- the staging
# Severity INCREASES with index, matching NACCUDSD's 0=Normal .. 3=Dementia. Note this is the
# REVERSE of vocab.SEVERITY_ORDER["MMSE"], which is worst-first; anything that needs "worsened
# by >= 1" on the raw token ids must keep using the resolved SEVERITY_ORDER, not this.
#
# ONE SPEC PER TOKENIZATION, SELECTED BY WHICH MMSE NAMES THE TABLE ACTUALLY HAS. Upstream
# keeps two specs here so a 4-level and a 7-level RADC build land on the SAME four-stage state
# space and their figures can be read side by side. The same mechanism is what lets our
# 3-level ROSMAP build be added as a third spec instead of being forced through a grouping
# that names stages the tokenizer cannot distinguish.
#
# `MMSE_impaired` is <24, i.e. Folstein Moderate AND Severe merged. It is named "Impaired"
# rather than "Moderate" precisely so no reader takes it for upstream's 18-23 row.
_ROSMAP3 = [
    ("Normal",     ("MMSE_normal",)),        # >= 27
    ("Borderline", ("MMSE_borderline",)),    # 24 - 26
    ("Impaired",   ("MMSE_impaired",)),      # < 24
]
_FOLSTEIN_V3 = [
    ("Normal",   ("MMSE 27", "MMSE 28", "MMSE 29", "MMSE 30")),   # >= 27
    ("Mild",     ("MMSE 24-26",)),                                # 24 - 26
    ("Moderate", ("MMSE 18-23",)),                                # 18 - 23
    ("Severe",   ("MMSE <18",)),                                  # < 18
]
_FOLSTEIN_V1 = [
    ("Normal",   ("MMSE 27-30",)),
    ("Mild",     ("MMSE 24-26",)),
    ("Moderate", ("MMSE 18-23",)),
    ("Severe",   ("MMSE 0-17",)),
]
_SPECS = (_ROSMAP3, _FOLSTEIN_V3, _FOLSTEIN_V1)


def _stage_spec(res, mode=None):
    m = mode or STAGE_MODE
    if m == "mmse_levels":
        # every emitted level its own stage, worst last
        return [(res.NAMES[t], (res.NAMES[t],)) for t in reversed(res.SCALES["MMSE"])]
    if m != "mmse_stages":
        raise ValueError(f"unknown STAGE_MODE {m!r}")
    have = set(res.NAMES)
    for spec in _SPECS:
        if all(n in have for _lab, ids in spec for n in ids):
            return spec
    raise ValueError(
        "no stage spec matches this label table's MMSE levels: "
        f"{[res.NAMES[t] for t in res.SCALES['MMSE']]}. Add a spec rather than letting the "
        "figure fall back to something that silently means a different staging.")


# --------------------------------------------------------------------------- auxiliary slots
# Scales scored ALONGSIDE the staging in panel a1, one row per emitted level, without entering
# the state space.
#
# THE DISTINCTION MATTERS AND IS LOAD-BEARING. A "state" is something a subject occupies: the
# stages partition the MMSE levels, exactly one holds at any time, and panel c's stage-at-age
# grid and the transition matrix walk that partition. An "auxiliary slot" is scored the same way
# -- "does the subject reach this level within H years, among those not already in it" -- but it
# is NOT in GRID_TOKENS, so it can never overwrite the stage on the grid. Putting global
# cognition into the state space instead would make a subject's cognitive STAGE unreadable for
# every grid year after a cogn_global token, because the carry-forward would latch onto the
# wrong instrument.
#
# Global cognition is the right scale to add: vocab.py records that it is complementary rather
# than redundant with MMSE (knowing one 4-level bin improves guessing the other by only 5-8 pp
# over the marginal), that inside the MMSE 27-30 ceiling -- 73% of visits -- the cogn_global bin
# lifts 5-year AD AUC from 0.750 to 0.846, and that it starts moving 3-4 years earlier than MMSE.
AUX_SCALES = ("COG",)


def configure(labels=None, mode=None):
    """Bind this module to a token table. Call once per checkpoint, before anything else.

    `labels` is a list of names in id order (a build's labels.csv column, or vocab.NAMES).
    Everything this module exports is rebound; the module-level names are kept so the ported
    panels read exactly as they did against NACC.
    """
    global RES, STAGE_NAMES, STAGE_GROUPS, NSTAGE, DEATH, AD_DX, DEATH_IDX, AD_IDX
    global ALL_NAMES, NSLOT, GRID_SLOTS, GRID_NAMES, NSTATE
    global STAGE_TOKENS, GRID_TOKENS, ALL_TOKENS, TOK2SLOT
    global SCALE_IDS, SCALE_DIRECTED, EVENT_GROUPS, MODE
    global AUX_SLOTS, AUX_FIRST, AUX_OF_SCALE

    RES = V.resolve(list(labels) if labels is not None else V.NAMES)
    MODE = mode or STAGE_MODE
    spec = _stage_spec(RES, MODE)
    STAGE_NAMES = [n for n, _ in spec]
    STAGE_GROUPS = [tuple(RES.ID[x] for x in ids) for _, ids in spec]
    NSTAGE = len(STAGE_NAMES)

    DEATH, AD_DX = RES.DEATH, RES.AD_DX
    DEATH_IDX, AD_IDX = NSTAGE, NSTAGE + 1
    # stages | Death | AD | then one slot per level of each auxiliary scale
    AUX_FIRST = NSTAGE + 2
    aux_names, AUX_SLOTS, AUX_OF_SCALE = [], [], {}
    for sc in AUX_SCALES:
        ids = RES.SCALES.get(sc, ())
        slots = list(range(AUX_FIRST + len(aux_names), AUX_FIRST + len(aux_names) + len(ids)))
        AUX_OF_SCALE[sc] = list(zip(slots, ids))
        AUX_SLOTS += slots
        aux_names += [RES.display(t) for t in ids]
    ALL_NAMES = STAGE_NAMES + ["Death", "AD diagnosis"] + aux_names
    NSLOT = len(ALL_NAMES)
    # The STATE space stops at Death. The auxiliary slots are deliberately absent from it.
    GRID_SLOTS = list(range(NSTAGE)) + [DEATH_IDX]
    GRID_NAMES = STAGE_NAMES + ["Death"]
    NSTATE = len(GRID_SLOTS)

    STAGE_TOKENS = tuple(t for g in STAGE_GROUPS for t in g)
    GRID_TOKENS = STAGE_TOKENS + (DEATH,)
    AUX_TOKENS = tuple(t for pairs in AUX_OF_SCALE.values() for _s, t in pairs)
    ALL_TOKENS = GRID_TOKENS + (AD_DX,) + AUX_TOKENS

    # token id -> slot index, -1 for everything else. A lookup table rather than a .index()
    # call: the MC batch is (n_mc, T) and the per-element python call showed up in the profile.
    TOK2SLOT = np.full(RES.VOCAB_SIZE, -1, dtype=np.int64)
    for i, g in enumerate(STAGE_GROUPS):
        for t in g:
            TOK2SLOT[t] = i
    TOK2SLOT[DEATH] = DEATH_IDX
    TOK2SLOT[AD_DX] = AD_IDX
    for pairs in AUX_OF_SCALE.values():
        for slot, tok in pairs:
            TOK2SLOT[tok] = slot

    SCALE_IDS = {k: tuple(v) for k, v in RES.SCALES.items()}
    SCALE_DIRECTED = {k: (k in RES.SEVERITY_ORDER) for k in SCALE_IDS}
    EVENT_GROUPS = _event_groups(RES)
    check()
    return RES


def _event_groups(res):
    """The discrete-event outcomes perdomain scores: (label, token ids, recurring?).

    Derived from the resolved families rather than typed out, so a vocabulary change cannot
    leave an outcome silently unscored -- perdomain's coverage is asserted against this list.

    Three families, and the middle one is empty on this build. Upstream pairs its GRADED onset
    tokens up ("Stroke, probable" + "Stroke, possible" -> one "Stroke onset" outcome) because a
    single grade is not the clinical question; our tokenizer emits one once-only token per
    condition instead, so vocab.ONSET_IDS is empty and the loop yields nothing. It is kept
    rather than deleted so that adding graded tokens later needs no change here.

    Names come from vocab.DISPLAY: the raw labels are SCREAMING_SNAKE ids that would go
    straight onto a figure axis otherwise.
    """
    out = []
    for t in res.KEEP_FIRST_IDS:
        out.append((res.display(t), (t,), False))
    fams = sorted({res.NAMES[t].split(",")[0] for t in res.ONSET_IDS})
    for fam in fams:
        ids = tuple(t for t in res.ONSET_IDS if res.NAMES[t].startswith(fam + ","))
        if ids:
            out.append((f"{fam} onset", ids, True))
    for t in res.MED_IDS:
        out.append((res.display(t), (t,), True))
    return out


ORDINAL_SCALES = ("MMSE", "COG", "BMI")


def slot_of(tokens):
    """Vectorised token -> slot index (-1 where the token is not a state or AD)."""
    t = np.asarray(tokens)
    n = RES.VOCAB_SIZE
    return np.where((t >= 0) & (t < n), TOK2SLOT[np.clip(t, 0, n - 1)], -1)


# --------------------------------------------------------------------------- endpoints
# Panel b's three endpoints, mirroring NACC's "Reach >=MCI / Dementia / Death":
#   the cognitive threshold, the diagnosis, and death.
# Each entry is (label, slots to reach, at-risk predicate on the baseline stage).
def endpoints():
    return [
        ("Reach ≥Borderline (MMSE <27)", list(range(1, NSTAGE)), lambda b: b <= 0),
        ("AD diagnosis", [AD_IDX], lambda b: True),
        ("Death", [DEATH_IDX], lambda b: True),
    ]


# --------------------------------------------------------------------------- trajectory class
TRAJ_CLASSES = ["Stable", "Improved", "Progressed (stage)", "Progressed to AD",
                "Died, no progression"]


def trajectory_class(base_stage, fp_obs, died, final_stage):
    """Coarse observed-trajectory label used to colour the embedding panel.

    Priority, unchanged from the NACC version: diagnosis > any stage worsening >
    death-without-progression > improvement > stable. Progression outranks death because a
    subject who converted and then died is informative about the cognitive trajectory the
    embedding is supposed to encode.

    The only substitution is the top rule: NACC asked "did they reach the Dementia STAGE",
    RADC asks "did they receive the AD DIAGNOSIS" -- the stronger question, and the one RADC
    can actually answer.
    """
    if np.isfinite(fp_obs[AD_IDX]):
        return "Progressed to AD"
    worse = [s for s in range(base_stage + 1, NSTAGE) if np.isfinite(fp_obs[s])]
    if worse:
        return "Progressed (stage)"
    if died:
        return "Died, no progression"
    better = [s for s in range(0, base_stage) if np.isfinite(fp_obs[s])]
    if better or (final_stage is not None and final_stage < base_stage):
        return "Improved"
    return "Stable"


def check():
    """Invariants. Called by configure() and by the test suite."""
    seen = set()
    for g in STAGE_GROUPS:
        assert g, "an empty stage group"
        for t in g:
            assert t not in seen, f"token {t} ({RES.NAMES[t]}) in two stages"
            seen.add(t)
    assert seen == set(RES.SCALES["MMSE"]), (
        "the stages must partition the MMSE levels exactly; missing "
        f"{sorted(set(RES.SCALES['MMSE']) - seen)}, extra {sorted(seen - set(RES.SCALES['MMSE']))}")
    assert DEATH not in seen and AD_DX not in seen
    assert TOK2SLOT[DEATH] == DEATH_IDX and TOK2SLOT[AD_DX] == AD_IDX
    # the auxiliary levels must be scoreable but must NOT be part of the state grid, or the
    # stage-at-age carry-forward in panel c would latch onto the wrong instrument
    for sc, pairs in AUX_OF_SCALE.items():
        assert [t for _s, t in pairs] == list(RES.SCALES[sc]), f"{sc} slots are not its levels"
        for slot, tok in pairs:
            assert tok not in GRID_TOKENS, f"{RES.NAMES[tok]} leaked into the state grid"
            assert slot not in GRID_SLOTS and TOK2SLOT[tok] == slot
            assert tok not in RES.IGNORE_TOKENS and tok != RES.NO_EVENT
    assert len(set(ALL_NAMES)) == len(ALL_NAMES), "duplicate slot name"
    # every stage token must be one the model is actually trained to emit
    for t in STAGE_TOKENS + (DEATH, AD_DX):
        assert t not in RES.IGNORE_TOKENS and t != RES.NO_EVENT, \
            f"{RES.NAMES[t]} is not a target"
    for lab, slots, _p in endpoints():
        for s in slots:
            assert 0 <= s < NSLOT, f"endpoint {lab} references slot {s}"
    return True


# Bound to the LIVE tokenization at import so nothing that does not care has to call
# configure(); figure2_core re-binds it to the checkpoint's own table.
configure()
