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
# by >= 1" on the raw token ids must keep using vocab, not this.
_FOLSTEIN = [
    ("Normal",   ("MMSE 27", "MMSE 28", "MMSE 29", "MMSE 30")),   # >= 27
    ("Mild",     ("MMSE 24-26",)),                                # 24 - 26
    ("Moderate", ("MMSE 18-23",)),                                # 18 - 23
    ("Severe",   ("MMSE <18",)),                                  # < 18
]
_MMSE7 = [(V.NAMES[t], (V.NAMES[t],)) for t in reversed(V.SCALES["MMSE"])]


def _stage_spec(mode=None):
    m = mode or STAGE_MODE
    if m == "folstein4":
        return _FOLSTEIN
    if m == "mmse7":
        return _MMSE7
    raise ValueError(f"unknown STAGE_MODE {m!r}")


_SPEC = _stage_spec()
STAGE_NAMES = [n for n, _ in _SPEC]
STAGE_GROUPS = [tuple(V.ID[x] for x in ids) for _, ids in _SPEC]   # model-space token ids
NSTAGE = len(STAGE_NAMES)

DEATH = V.DEATH
AD_DX = V.AD_DX

# Indices into ALL_NAMES. 0..NSTAGE-1 are stages; Death is the absorbing state that closes the
# state space; AD is an EVENT carried alongside, not a state (see the docstring).
DEATH_IDX = NSTAGE
AD_IDX = NSTAGE + 1
ALL_NAMES = STAGE_NAMES + ["Death", "AD diagnosis"]
NSLOT = len(ALL_NAMES)

# The slots that form the state space panels a and c walk over: stages + Death.
GRID_SLOTS = list(range(NSTAGE)) + [DEATH_IDX]
GRID_NAMES = STAGE_NAMES + ["Death"]
NSTATE = len(GRID_SLOTS)

# Every token that can move a subject between grid states, and its slot.
STAGE_TOKENS = tuple(t for g in STAGE_GROUPS for t in g)
GRID_TOKENS = STAGE_TOKENS + (DEATH,)
ALL_TOKENS = GRID_TOKENS + (AD_DX,)

# token id -> slot index, -1 for everything else. A lookup table rather than a .index() call:
# the MC batch is (n_mc, T) and the per-element python call showed up in the profile.
TOK2SLOT = np.full(V.VOCAB_SIZE, -1, dtype=np.int64)
for _i, _g in enumerate(STAGE_GROUPS):
    for _t in _g:
        TOK2SLOT[_t] = _i
TOK2SLOT[DEATH] = DEATH_IDX
TOK2SLOT[AD_DX] = AD_IDX


def slot_of(tokens):
    """Vectorised token -> slot index (-1 where the token is not a state or AD)."""
    t = np.asarray(tokens)
    return np.where((t >= 0) & (t < V.VOCAB_SIZE), TOK2SLOT[np.clip(t, 0, V.VOCAB_SIZE - 1)], -1)


# --------------------------------------------------------------------------- endpoints
# Panel b's three endpoints, mirroring NACC's "Reach >=MCI / Dementia / Death":
#   the cognitive threshold, the diagnosis, and death.
# Each entry is (label, slots to reach, at-risk predicate on the baseline stage).
ENDPOINTS = [
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


# --------------------------------------------------------------------------- per-domain set
# What perdomain.py scores: everything the model is trained to emit that is not a stage token,
# not an endpoint and not a static. Grouped so the plot stays readable.
#
# The three ordinal scales are scored as "worsens by >= 1 bin", which needs a severity order;
# BMI has none (both tails are adverse) so it is scored as "moves by >= 1 bin" instead and
# labelled as such. That asymmetry is inherited from vocab.SEVERITY_ORDER, which deliberately
# omits BMI.
ORDINAL_SCALES = ("MMSE", "COG", "BMI")
SCALE_IDS = {k: tuple(V.SCALES[k]) for k in ORDINAL_SCALES}
SCALE_DIRECTED = {"MMSE": True, "COG": True, "BMI": False}       # True = "worsens" is defined

EVENT_GROUPS = [
    ("Hypertension", (V.ID["Hypertension, history"],), False),
    ("Diabetes", (V.ID["Diabetes, history"],), False),
    ("Claudication", (V.ID["Claudication, history"],), False),
    ("Heart condition", (V.ID["Heart condition, history"],), False),
    ("Stroke onset", (V.ID["Stroke, probable"], V.ID["Stroke, possible"]), True),
    ("Depression onset", (V.ID["Depression, probable"], V.ID["Depression, possible"]), True),
    ("Antihypertensive started", (V.ID["Antihypertensive started"],), True),
    ("Antihypertensive stopped", (V.ID["Antihypertensive stopped"],), True),
    ("Statin started", (V.ID["Statin started"],), True),
    ("Statin stopped", (V.ID["Statin stopped"],), True),
]


def check():
    """Invariants. Called by the test suite and by figure2_core on import."""
    seen = set()
    for g in STAGE_GROUPS:
        assert g, "an empty stage group"
        for t in g:
            assert t not in seen, f"token {t} ({V.NAMES[t]}) in two stages"
            seen.add(t)
    assert seen == set(V.SCALES["MMSE"]), (
        "the stages must partition the MMSE levels exactly; missing "
        f"{sorted(set(V.SCALES['MMSE']) - seen)}, extra {sorted(seen - set(V.SCALES['MMSE']))}")
    assert DEATH not in seen and AD_DX not in seen
    assert TOK2SLOT[DEATH] == DEATH_IDX and TOK2SLOT[AD_DX] == AD_IDX
    # every stage token must be one the model is actually trained to emit
    for t in STAGE_TOKENS + (DEATH, AD_DX):
        assert t not in V.IGNORE_TOKENS and t != V.NO_EVENT, f"{V.NAMES[t]} is not a target"
    # the endpoint slots must be real
    for lab, slots, _p in ENDPOINTS:
        for s in slots:
            assert 0 <= s < NSLOT, f"endpoint {lab} references slot {s}"
    return True


check()
