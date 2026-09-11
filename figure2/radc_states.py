"""
radc_states.py -- the ONE place Figure 2 is allowed to know what a "state" is.

Figure 2 was written against NACC, where the four panels all hang off a single variable:
NACCUDSD, a per-visit CLINICAL STAGING with four levels (Normal / Impaired-not-MCI / MCI /
Dementia) plus Death as an absorbing fifth. Every panel's meaning is defined relative to it:

    a   discrimination for REACHING each stage within 5 y, one row per (baseline, reached)
    b   timing error for three ENDPOINTS, and AUC vs horizon for the same three
    c   observed vs predicted STAGE-AT-AGE on a yearly grid, per individual
    d   does the baseline embedding encode the FUTURE trajectory class, or only current stage

This module supplies the RADC equivalents, and nothing else in figure2/ may hard-code a token
id. That rule is not stylistic: radc_delphi/vocab.py's own docstring records that the NACC arm
scattered its ids across the engine, the adapter, the figure code and the training configs, and
that a checkpoint could therefore be scored against a vocabulary it was not trained on.

------------------------------------------------------------------------------------------
THE MAPPING, AND WHERE IT IS EXACT vs WHERE IT IS A SUBSTITUTION

  NACCUDSD staging  ->  MMSE, collapsed to the four Folstein stages.
      This is NOT a new binning. vocab.STAGE_TOKENS_DISK already declares MMSE to be the
      staging analogue for this cohort, with the reason: RADC has no usable per-visit
      diagnosis (cogdx and dcfdx are last-visit only, and both leak). The Folstein cuts
      (>=27 unimpaired, 24-26 mild, 18-23 moderate, <18 severe) fall on edges the tokenizer
      ALREADY has -- vocab.SCALE_EDGES["MMSE"] contains 18.0, 24.0 and 26.5 -- so a stage is
      a pure GROUPING of emitted tokens, computed at evaluation time. No retokenization, no
      new model, and the .bin is untouched.

      WHAT IS NOT PRESERVED, stated plainly: NACCUDSD is a clinician's diagnosis, MMSE is a
      test score. Panel a's row "Normal -> Severe" therefore reads "reached an MMSE below 18",
      not "was diagnosed as demented". The panel's QUESTION is identical; the instrument
      behind it is not, and it cannot be, because RADC deliberately excludes per-visit
      diagnosis as leaky.

  panel b's "Dementia" endpoint  ->  the AD diagnosis token.
      Here RADC is strictly BETTER off than NACC: it has a real incident-diagnosis endpoint
      (age_first_ad_dx), so the endpoint panel keeps a diagnosis where the state panels cannot.
      This is why AD is carried as an EVENT index alongside the stages rather than as a stage:
      a subject does not stop occupying an MMSE stage by being diagnosed.

  Death  ->  Death. Absorbing, and the competing risk for every cognitive transition.

  the 30 NACC sub-scales scored by perdomain.py  ->  RADC's three ordinal scales, four
      cumulative histories, two graded onset families and four medication events.

  perdomain.py's reconstructed TOTALS  ->  nothing, and the block is deleted rather than
      faked. Those existed to falsify the CDR/FAQ/NPI-Q item split by summing domains back
      into the retired total. RADC never had an item split, so there is no retired token to
      reconstruct and no comparison to make.

------------------------------------------------------------------------------------------
STAGE_MODE

"folstein4" (default) gives the four stages above, so panel a has the same ~13 readable rows
the NACC figure had. "mmse7" keeps all seven emitted levels, which is the honest high-
resolution view but makes panel a a 7x8 grid in which most cells fall below the 30-event
reporting floor. Both are the same underlying tokens; only the grouping differs.
"""
import numpy as np

from radc_delphi import vocab as V

STAGE_MODE = "folstein4"

# --------------------------------------------------------------------------- the staging
# Severity INCREASES with index, matching NACCUDSD's 0=Normal .. 3=Dementia. Note this is the
# REVERSE of vocab.SEVERITY_ORDER["MMSE"], which is worst-first; anything that needs "worsened
# by >= 1" on the raw token ids must keep using the resolved SEVERITY_ORDER, not this.
#
# ONE SPEC PER TOKENIZATION, SELECTED BY WHICH MMSE NAMES THE TABLE ACTUALLY HAS. The point of
# this is cross-vocabulary comparability: the v1 build binned MMSE into exactly four levels and
# those four levels ARE the Folstein stages, one to one, so a v1 checkpoint and a v3 checkpoint
# land on the SAME four-stage state space and their Figure 2s can be read side by side. Without
# this, scoring a v1 checkpoint would mean applying v3's ids to a model that never saw them --
# the ids above 22 all shifted when the three `Dementia at entry` levels were inserted.
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
_SPECS = (_FOLSTEIN_V3, _FOLSTEIN_V1)


def _stage_spec(res, mode=None):
    m = mode or STAGE_MODE
    if m == "mmse7":
        # every emitted level its own stage, worst last
        return [(res.NAMES[t], (res.NAMES[t],)) for t in reversed(res.SCALES["MMSE"])]
    if m != "folstein4":
        raise ValueError(f"unknown STAGE_MODE {m!r}")
    have = set(res.NAMES)
    for spec in _SPECS:
        if all(n in have for _lab, ids in spec for n in ids):
            return spec
    raise ValueError(
        "no Folstein stage spec matches this label table's MMSE levels: "
        f"{[res.NAMES[t] for t in res.SCALES['MMSE']]}. Add a spec rather than letting the "
        "figure fall back to something that silently means a different staging.")


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

    RES = V.resolve(list(labels) if labels is not None else V.NAMES)
    MODE = mode or STAGE_MODE
    spec = _stage_spec(RES, MODE)
    STAGE_NAMES = [n for n, _ in spec]
    STAGE_GROUPS = [tuple(RES.ID[x] for x in ids) for _, ids in spec]
    NSTAGE = len(STAGE_NAMES)

    DEATH, AD_DX = RES.DEATH, RES.AD_DX
    DEATH_IDX, AD_IDX = NSTAGE, NSTAGE + 1
    ALL_NAMES = STAGE_NAMES + ["Death", "AD diagnosis"]
    NSLOT = len(ALL_NAMES)
    GRID_SLOTS = list(range(NSTAGE)) + [DEATH_IDX]
    GRID_NAMES = STAGE_NAMES + ["Death"]
    NSTATE = len(GRID_SLOTS)

    STAGE_TOKENS = tuple(t for g in STAGE_GROUPS for t in g)
    GRID_TOKENS = STAGE_TOKENS + (DEATH,)
    ALL_TOKENS = GRID_TOKENS + (AD_DX,)

    # token id -> slot index, -1 for everything else. A lookup table rather than a .index()
    # call: the MC batch is (n_mc, T) and the per-element python call showed up in the profile.
    TOK2SLOT = np.full(RES.VOCAB_SIZE, -1, dtype=np.int64)
    for i, g in enumerate(STAGE_GROUPS):
        for t in g:
            TOK2SLOT[t] = i
    TOK2SLOT[DEATH] = DEATH_IDX
    TOK2SLOT[AD_DX] = AD_IDX

    SCALE_IDS = {k: tuple(v) for k, v in RES.SCALES.items()}
    SCALE_DIRECTED = {k: (k in RES.SEVERITY_ORDER) for k in SCALE_IDS}
    EVENT_GROUPS = _event_groups(RES)
    check()
    return RES


def _event_groups(res):
    """The discrete-event outcomes perdomain scores: (label, token ids, recurring?).

    Derived from the resolved families rather than typed out, so a vocabulary change cannot
    leave an outcome silently unscored -- perdomain's coverage is asserted in the test suite.
    The graded onset families are paired up ("Stroke, probable" + "Stroke, possible" -> one
    "Stroke onset" outcome) because a single grade is not the clinical question.
    """
    out = []
    for t in res.KEEP_FIRST_IDS:
        out.append((res.NAMES[t].replace(", history", ""), (t,), False))
    for fam in ("Stroke", "Depression"):
        ids = tuple(t for t in res.ONSET_IDS if res.NAMES[t].startswith(fam + ","))
        if ids:
            out.append((f"{fam} onset", ids, True))
    for t in res.MED_IDS:
        out.append((res.NAMES[t], (t,), True))
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
        ("Reach ≥Mild (MMSE ≤26)", list(range(1, NSTAGE)), lambda b: b <= 0),
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
