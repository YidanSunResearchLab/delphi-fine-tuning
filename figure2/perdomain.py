"""
perdomain.py -- the half of the evaluation that the four Figure-2 panels do not do.

WHY. figure2_core.py scores exactly one thing: the cognitive staging (grouped MMSE levels)
plus the two endpoints. That is a small slice of what the model is trained on -- the RADC
vocabulary has 30 predicted content tokens, and most of the gradient goes to the ones the
panels never look at. So the staging result is currently measured on its own terms and not at
all against what the rest of the vocabulary was supposed to buy. This module measures the
other half.

WHAT IT COMPUTES, per outcome, mirroring what Figure 2 does for the staging:
  * discrimination -- "does this outcome occur within h years", AUC per horizon,
    competing-risk aware (death before the event counts as not-occurred, and a subject who
    dies before h without the event is dropped, not scored as a negative)
  * timing         -- predicted vs observed years to that first occurrence: MAE, bias
  * calibration    -- mean predicted risk vs observed rate

TWO OUTCOME FAMILIES, because RADC has two kinds of token:

  ordinal scales   MMSE, global cognition, BMI. Scored as "moves >= 1 bin in the adverse
                   direction", which is the same question Figure 2 asks of the staging.
                   DIRECTION IS A TRAP HERE: vocab.SCALES is ordered low -> high on the
                   underlying measurement, so for the two cognitive scales severity DECREASES
                   with token id -- "worse" is a LOWER index, the opposite of the NACC
                   vocabulary this file was written against. BMI has no severity order at all
                   (both tails are adverse, which is why vocab.SEVERITY_ORDER omits it), so it
                   is scored as "moves >= 1 bin in EITHER direction" and labelled as such.

  discrete events  the four cumulative histories, the two graded onset families and the four
                   medication start/stop events. Scored as "first occurrence within h years".
                   These have no NACC counterpart in this file -- that vocabulary was all
                   ordinal sub-scales -- but they are most of what this model emits, so
                   leaving them out would defeat the module's stated purpose.

WHAT WAS DELETED, AND WHY IT IS NOT REPLACED. The NACC version also reconstructed three
retired total scores (CDRSUM, the FAQ total, the NPI-Q total) by summing their item domains
back up, as the one comparison that could falsify the CDR/FAQ/NPI-Q item split. RADC never had
an item split: no token was retired in favour of its parts, so there is no total to
reconstruct and no such comparison to make. Inventing an analogue would be fabricating a
falsification test that does not exist, so the block is gone rather than ported.

Everything is derived from the SAME Monte-Carlo trajectories Figure 2 already samples --
_process() in figure2_core.py hands (A, T) straight to per_domain_row(). There is no second MC
pass and no second RNG stream, so these numbers and the Figure-2 numbers describe the same
sampled futures and can be quoted side by side.
"""
import numpy as np

from radc_delphi import vocab as V
from radc_delphi.engine import DAYS_PER_YEAR

from . import radc_states as S

D = DAYS_PER_YEAR
NEG = -1e4 + 1
DEATH = V.DEATH

# ---------------------------------------------------------------- the outcome set
SCALES = list(S.ORDINAL_SCALES)                 # ("MMSE", "COG", "BMI")
SCALE_IDS = S.SCALE_IDS
# True  -> a severity order exists, score "worsens by >= 1 bin"
# False -> no severity order, score "moves by >= 1 bin either way"
SCALE_DIRECTED = S.SCALE_DIRECTED

EVENTS = [(name, ids) for name, ids, _rec in S.EVENT_GROUPS]
EVENT_NAMES = [n for n, _ in EVENTS]

ALL_OUTCOMES = SCALES + EVENT_NAMES

# Human-readable question per outcome, so the plot axis cannot silently mislabel BMI.
QUESTION = {**{k: ("worsens ≥ 1 bin" if SCALE_DIRECTED[k] else "moves ≥ 1 bin")
               for k in SCALES},
            **{n: "first occurrence" for n in EVENT_NAMES}}


def _bin_grid_one(ages, toks, ids, base_bin, grid_days):
    """Carry-forward bin index of ONE scale at each grid day, for one stream.

    Same carry-forward semantics as figure2_core._state_grid_one, but parameterised by the
    scale's token ids instead of hard-wired to NACCUDSD. Returns -1 where the scale has never
    been observed and no baseline value is known.
    """
    a = np.asarray(ages, float); t = np.asarray(toks)
    m = np.isin(t, ids) & (a > NEG)
    out = np.full(len(grid_days), base_bin, dtype=np.int8)
    if not m.any():
        return out
    lut = np.full(int(max(ids)) + 1, -1, dtype=np.int8)
    for i, tok in enumerate(ids):
        lut[tok] = i
    ea = a[m]; eb = lut[t[m]]
    o = np.argsort(ea, kind="stable"); ea, eb = ea[o], eb[o]
    idx = np.searchsorted(ea, grid_days, side="right") - 1
    return np.where(idx >= 0, eb[np.clip(idx, 0, len(eb) - 1)], base_bin).astype(np.int8)


def _bin_grid_batch(A, T, ids, base_bin, grid_days):
    """Vectorised _bin_grid_one over an (n_samples, T) MC batch -> (n_samples, n_grid)."""
    m = np.isin(T, ids) & (A > NEG)
    ev_age = np.where(m, A, -np.inf)
    lut = np.full(int(max(ids)) + 1, -1, dtype=np.int8)
    for i, tok in enumerate(ids):
        lut[tok] = i
    ev_bin = np.where(m, lut[np.clip(T, 0, len(lut) - 1)], -1)
    n = A.shape[0]
    out = np.full((n, len(grid_days)), base_bin, dtype=np.int8)
    rows = np.arange(n)
    for gi, gd in enumerate(grid_days):
        aa = np.where(ev_age <= gd, ev_age, -np.inf)
        best = aa.max(1); j = aa.argmax(1)
        out[:, gi] = np.where(best > -np.inf, ev_bin[rows, j], base_bin)
    return out


def _baseline_bin(ages, toks, ids, base_day):
    """Last bin of this scale at or before baseline; -1 if the scale was never recorded.

    -1 matters: a scale that a patient never had measured must be EXCLUDED from that
    patient's scoring, not silently treated as bin 0. Under keep-transitions a scale that
    was measured always emits at least its first value, so -1 really does mean absent.
    """
    a = np.asarray(ages, float); t = np.asarray(toks)
    m = np.isin(t, ids) & (a <= base_day) & (a > NEG)
    if not m.any():
        return -1
    j = np.argmax(np.where(m, a, -np.inf))
    return int(ids.index(int(t[j])))


def _adverse_bins(ids, base_bin, directed):
    """The bin INDICES that count as an adverse move from `base_bin`.

    `ids` is vocab order: low -> high on the underlying measurement. For the two cognitive
    scales that makes severity DECREASE with index, so adverse = index < base_bin. The NACC
    vocabulary this file was written against was the other way round, and getting it backwards
    scores recovery as decline without erroring anywhere.

    `directed=False` (BMI) has no adverse direction, so ANY move off the baseline bin counts.
    """
    n = len(ids)
    if not directed:
        return [i for i in range(n) if i != base_bin]
    return [i for i in range(base_bin)]                 # lower index = worse


def _first_worse_obs(ages, toks, ids, base_bin, base_day, directed=True):
    """Observed years to the first token of this scale that is >=1 bin adverse of baseline."""
    a = np.asarray(ages, float); t = np.asarray(toks)
    m = np.isin(t, ids) & (a > base_day) & (a > NEG)
    if not m.any():
        return np.inf
    adverse = set(ids[i] for i in _adverse_bins(ids, base_bin, directed))
    if not adverse:
        return np.inf
    worse = np.isin(t[m], list(adverse))
    if not worse.any():
        return np.inf
    return float(a[m][worse].min() - base_day) / D


def _first_worse_sim(A, T, ids, base_bin, base_day, directed=True):
    """Per-sample days-to-first adverse move (absolute age), inf where it never happens."""
    adverse = [ids[i] for i in _adverse_bins(ids, base_bin, directed)]
    if not adverse:
        return np.full(A.shape[0], np.inf)      # already at the worst bin: cannot worsen
    m = np.isin(T, adverse) & (A > base_day) & (A > NEG)
    return np.where(m, A, np.inf).min(1)


def _first_event_obs(ages, toks, ids, base_day):
    """Observed years to the first occurrence of any token in `ids` after baseline."""
    a = np.asarray(ages, float); t = np.asarray(toks)
    m = np.isin(t, list(ids)) & (a > base_day) & (a > NEG)
    return float(a[m].min() - base_day) / D if m.any() else np.inf


def _first_event_sim(A, T, ids, base_day):
    """Per-sample days to the first occurrence of any token in `ids` after baseline."""
    m = np.isin(T, list(ids)) & (A > base_day) & (A > NEG)
    return np.where(m, A, np.inf).min(1)


def per_domain_row(ages, toks, A, T, base_day, grid_years, horizons, death_sim, obs_died_y):
    """Every per-outcome quantity for ONE subject.

    ages/toks : the subject's observed stream        A/T : the MC batch (n_samples, T)
    death_sim : per-sample first-passage day to Death (from figure2_core, so death competes
                here exactly as it does in Figure 2)
    obs_died_y: observed years to death, inf if never observed to die

    Returns a flat dict; keys are prefixed with the outcome name. The key names are unchanged
    from the NACC version (`__base_bin`, `__obs_t_worse`, `__pred_worse_{h}y`, `__pred_t_worse`)
    so score_perdomain.py and fig_perdomain.py read them without modification -- for a discrete
    event "worse" simply means "occurred".
    """
    grid_days = base_day + grid_years * D
    out = {}

    # ------------------------------------------------ ordinal scales
    for name in SCALES:
        ids = list(SCALE_IDS[name])
        directed = SCALE_DIRECTED[name]
        b0 = _baseline_bin(ages, toks, ids, base_day)
        out[f"{name}__base_bin"] = b0
        if b0 < 0:                                # never measured -> not scorable
            out[f"{name}__obs_t_worse"] = np.nan
            out[f"{name}__pred_t_worse"] = np.nan
            for h in horizons:
                out[f"{name}__pred_worse_{h}y"] = np.nan
            continue
        ow = _first_worse_obs(ages, toks, ids, b0, base_day, directed)
        sw = _first_worse_sim(A, T, ids, b0, base_day, directed)
        out[f"{name}__obs_t_worse"] = ow
        for h in horizons:
            hd = base_day + h * D
            out[f"{name}__pred_worse_{h}y"] = float(np.mean((sw <= hd) & (sw <= death_sim)))
        fin = sw[np.isfinite(sw)]
        out[f"{name}__pred_t_worse"] = float(np.median(fin) - base_day) / D if fin.size else np.nan

    # ------------------------------------------------ discrete events
    # base_bin is 1 when the subject ALREADY has the event at baseline and 0 otherwise, so the
    # downstream at-risk rule ("drop base_bin < 0, and for once-only outcomes drop those who
    # already have it") reads the same field it reads for the scales.
    for name, ids in EVENTS:
        a = np.asarray(ages, float); t = np.asarray(toks)
        had = bool((np.isin(t, list(ids)) & (a <= base_day) & (a > NEG)).any())
        out[f"{name}__base_bin"] = int(had)
        oe = _first_event_obs(ages, toks, ids, base_day)
        se = _first_event_sim(A, T, ids, base_day)
        out[f"{name}__obs_t_worse"] = oe
        for h in horizons:
            hd = base_day + h * D
            out[f"{name}__pred_worse_{h}y"] = float(np.mean((se <= hd) & (se <= death_sim)))
        fin = se[np.isfinite(se)]
        out[f"{name}__pred_t_worse"] = float(np.median(fin) - base_day) / D if fin.size else np.nan

    out["obs_t_death"] = obs_died_y
    return out
