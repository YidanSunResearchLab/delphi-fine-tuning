"""
perdomain.py -- the half of the evaluation that Figure 2 does not do.

WHY. figure2_core.py scores exactly one scale: NACCUDSD (4 states + Death). That was the
whole outcome set when the vocabulary had five ordinal scales. It now has THIRTY -- the six
CDR box domains, the nine FAQ domains, the twelve NPI-Q symptoms, GDS, MoCA and NACCUDSD --
and the model spends most of its gradient on the 28 that Figure 2 never looks at. So the
item split is currently measured on its COST (NACCUDSD discrimination) and not at all on
what it was supposed to buy. This module measures the other half.

WHAT IT COMPUTES, per scale, mirroring what Figure 2 does for NACCUDSD:
  * discrimination -- "does this scale worsen by >=1 bin within h years", AUC per horizon,
    competing-risk aware (death before worsening counts as not-worsened, and a patient who
    dies before h without worsening is dropped, not scored as a negative)
  * timing         -- predicted vs observed years to that first worsening: MAE, bias
  * calibration    -- mean predicted risk vs observed rate

THE FALSIFIABLE PART -- reconstructed totals. Three totals were retired by the item split
(CDRSUM 24-28, the FAQ total 33-36, the NPI-Q total 37-40). For each, this module sums the
domains back into the total, IN THE RETIRED TOKEN'S OWN BIN DEFINITION, and scores it the
same way. That makes the one comparison that can actually falsify the split:

    does the reconstructed total predict as well as the old model predicted the total directly?

  reconstructed no worse  -> the split lost no information and bought domain-level resolution
  reconstructed worse     -> the split turned one predictable quantity into six hard ones

GDS has no retired predecessor, so it is reported on absolute terms only.

Everything is derived from the SAME Monte-Carlo trajectories Figure 2 already samples --
_process() in figure2_core.py hands (A, T) straight to per_domain_row(). There is no second
MC pass and no second RNG stream, so these numbers and the Figure-2 numbers describe the
same sampled futures and can be quoted side by side.
"""
import numpy as np

from delphi import predict_adapter as PA

D = PA.DAYS_PER_YEAR
NEG = -1e4 + 1
DEATH = PA.DEATH

# ---------------------------------------------------------------- the scales
# Ordered so the reconstructed-total blocks stay contiguous and readable.
CDR_BOXES = ["MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB", "PERSCARE"]
FAQ_DOMAINS = ["BILLS", "TAXES", "GAMES", "STOVE", "MEALPREP", "EVENTS",
               "PAYATTN", "REMDATES", "TRAVEL"]
NPI_SYMPTOMS = ["DEL", "HALL", "AGIT", "DEPD", "ANX", "ELAT", "APA", "DISN",
                "IRR", "MOT", "NITE", "APP"]
SCALES = CDR_BOXES + FAQ_DOMAINS + NPI_SYMPTOMS + ["GDS", "MOCA", "NACCUDSD"]

# raw clinical level per bin index, needed to sum domains back into a total.
# CDR boxes: 0/0.5/1/2/3, except PERSCARE which has no 0.5 level (4 bins, not 5).
_RAW = {c: [0.0, 0.5, 1.0, 2.0, 3.0] for c in CDR_BOXES}
_RAW["PERSCARE"] = [0.0, 1.0, 2.0, 3.0]
for c in FAQ_DOMAINS:
    _RAW[c] = [0.0, 1.0, 2.0, 3.0]          # FAQ item score
for c in NPI_SYMPTOMS:
    _RAW[c] = [0.0, 1.0, 2.0, 3.0]          # NPI-Q severity

# Reconstructed totals, each in the RETIRED token's own binning so the comparison is
# apples-to-apples with what the old model predicted.
#   CDRSUM  (was 24-28): 0 | 0.5-4 | 4.5-9 | 9.5-12.5 | 13-18
#   FAQ     (was 33-36): 0-4 | 5-8 | 9-12 | 13-30
#   NPI-Q   (was 37-40): 0 | 1-6 | 7-15 | 16-36
TOTALS = {
    "CDRSUM_recon": (CDR_BOXES,    [0.0, 0.5, 4.5, 9.5, 13.0]),   # lower edge of each bin
    "FAQ_recon":    (FAQ_DOMAINS,  [0.0, 5.0, 9.0, 13.0]),
    "NPIQ_recon":   (NPI_SYMPTOMS, [0.0, 1.0, 7.0, 16.0]),
}
ALL_OUTCOMES = SCALES + list(TOTALS)


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


def _first_worse_obs(ages, toks, ids, base_bin, base_day):
    """Observed years to the first token of this scale that is >=1 bin WORSE than baseline."""
    a = np.asarray(ages, float); t = np.asarray(toks)
    m = np.isin(t, ids) & (a > base_day) & (a > NEG)
    if not m.any():
        return np.inf
    lut = {tok: i for i, tok in enumerate(ids)}
    worse = np.array([lut[int(x)] > base_bin for x in t[m]])
    if not worse.any():
        return np.inf
    return float(a[m][worse].min() - base_day) / D


def _first_worse_sim(A, T, ids, base_bin, base_day):
    """Per-sample days-to-first-worsening (absolute age), inf where it never worsens."""
    hi = [tok for i, tok in enumerate(ids) if i > base_bin]
    if not hi:
        return np.full(A.shape[0], np.inf)      # already at the worst bin: cannot worsen
    m = np.isin(T, hi) & (A > base_day) & (A > NEG)
    return np.where(m, A, np.inf).min(1)


def _total_series(bins_by_dom, cols, edges):
    """Sum per-domain bin indices back into the retired total's bin index.

    bins_by_dom[c] is an array of bin indices (any shape); -1 marks 'not observed'. If ANY
    domain of the total is missing the total is undefined -> -1, because a partial sum is not
    the total and scoring it would quietly compare different quantities across patients.
    """
    raw = None
    missing = None
    for c in cols:
        b = bins_by_dom[c]
        lut = np.array(_RAW[c] + [np.nan])       # index -1 lands on the nan sentinel
        v = lut[b]
        raw = v if raw is None else raw + v
        miss = (b < 0)
        missing = miss if missing is None else (missing | miss)
    out = np.searchsorted(np.asarray(edges), raw, side="right") - 1
    return np.where(missing, -1, out).astype(np.int8)


def per_domain_row(ages, toks, A, T, base_day, grid_years, horizons, death_sim, obs_died_y):
    """Every per-scale and reconstructed-total quantity for ONE patient.

    ages/toks : the patient's observed stream        A/T : the MC batch (n_samples, T)
    death_sim : per-sample first-passage day to Death (from figure2_core, so death competes
                here exactly as it does in Figure 2)
    obs_died_y: observed years to death, inf if never observed to die

    Returns a flat dict; keys are prefixed with the outcome name.
    """
    grid_days = base_day + grid_years * D
    out = {}
    obs_bins, sim_bins = {}, {}

    for name in SCALES:
        ids = PA.SCALES[name]
        b0 = _baseline_bin(ages, toks, ids, base_day)
        obs_bins[name] = _bin_grid_one(ages, toks, ids, b0, grid_days)
        sim_bins[name] = _bin_grid_batch(A, T, ids, b0, grid_days)
        out[f"{name}__base_bin"] = b0
        if b0 < 0:                                # never measured -> not scorable
            out[f"{name}__obs_t_worse"] = np.nan
            out[f"{name}__pred_t_worse"] = np.nan
            for h in horizons:
                out[f"{name}__pred_worse_{h}y"] = np.nan
            continue
        ow = _first_worse_obs(ages, toks, ids, b0, base_day)
        sw = _first_worse_sim(A, T, ids, b0, base_day)
        out[f"{name}__obs_t_worse"] = ow
        for h in horizons:
            hd = base_day + h * D
            out[f"{name}__pred_worse_{h}y"] = float(np.mean((sw <= hd) & (sw <= death_sim)))
        fin = sw[np.isfinite(sw)]
        out[f"{name}__pred_t_worse"] = float(np.median(fin) - base_day) / D if fin.size else np.nan

    # ------------------------------------------------ reconstructed totals
    for tname, (cols, edges) in TOTALS.items():
        ob = _total_series({c: obs_bins[c] for c in cols}, cols, edges)      # (n_grid,)
        sb = _total_series({c: sim_bins[c] for c in cols}, cols, edges)      # (n_samples, n_grid)
        b0 = int(ob[0])
        out[f"{tname}__base_bin"] = b0
        if b0 < 0:
            out[f"{tname}__obs_t_worse"] = np.nan
            out[f"{tname}__pred_t_worse"] = np.nan
            for h in horizons:
                out[f"{tname}__pred_worse_{h}y"] = np.nan
            continue
        # On the yearly grid the totals are only resolvable to a grid step, unlike the
        # per-scale numbers which use exact token ages. Stated rather than hidden: a
        # reconstructed total's timing is grid-quantised to 1 y.
        ow_idx = np.where((ob > b0) & (ob >= 0))[0]
        out[f"{tname}__obs_t_worse"] = float(grid_years[ow_idx[0]]) if ow_idx.size else np.inf
        worse = (sb > b0) & (sb >= 0)
        first = np.where(worse.any(1), grid_years[worse.argmax(1)], np.inf)
        dsim_y = np.where(np.isfinite(death_sim), (death_sim - base_day) / D, np.inf)
        for h in horizons:
            out[f"{tname}__pred_worse_{h}y"] = float(np.mean((first <= h) & (first <= dsim_y)))
        fin = first[np.isfinite(first)]
        out[f"{tname}__pred_t_worse"] = float(np.median(fin)) if fin.size else np.nan

    out["obs_t_death"] = obs_died_y
    return out
