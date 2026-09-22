"""
plotting_style.py -- shared, consistent visual style + save/stat helpers for the figures.

- One fixed, colorblind-safe color per outcome, used across ALL figures.
- Every panel saves a vector PDF + 300-dpi PNG AND a CSV of the numbers behind it (Nature
  "source data" practice).
- Patient-level bootstrap CI helper for every scalar metric.
"""
import os, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- colorblind-safe palette (Okabe-Ito) ----
_OKABE = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "vermillion": "#D55E00",
    "skyblue": "#56B4E9", "yellow": "#F0E442", "purple": "#CC79A7", "black": "#000000",
    "grey": "#999999",
}
# ---- one fixed colour per outcome, reused everywhere ----
# BUILT, not typed out, and sized from the outcome list rather than a literal: the figure audit
# on the RADC endpoint panels caught a palette being cycled modulo 3, which silently gave two
# different series the same colour. Anything here that can grow must therefore be generated.
#
# Families: the three ordinal scales keep distinct Okabe-Ito hues; the ten discrete events are
# separated within two lightness ramps (comorbidity histories/onsets vs medications), which
# survives both greyscale and CVD simulation because lightness carries the within-family
# distinction and hue carries the family.
from figure2 import perdomain as _pd  # noqa: E402  (outcome names, no heavy import)

_SCALE_COLORS = {"MMSE": "#7B3294", "COG": _OKABE["skyblue"], "BMI": _OKABE["orange"]}
_EVENTS = [n for n in _pd.EVENT_NAMES]
# PORT NOTE: classified by SUFFIX, not by drug name. Upstream tests for "Antihyp" / "Statin"
# because those are the only two medications in its vocabulary; ours has four (it adds diabetes
# and AD medications), so the literal test would file `Diabetes Rx started` under comorbidities
# and give it a blue from the wrong ramp. vocab.DISPLAY builds every medication label as
# "<drug> started" / "<drug> stopped", which is the property worth testing.
_MEDS = [n for n in _EVENTS if n.endswith((" started", " stopped"))]
_COMORBID = [n for n in _EVENTS if n not in _MEDS]


def _ramp(n, cmap, lo=0.35, hi=0.9):
    """`n` steps of one sequential hue, dark -> light.

    GENERATED, not typed out. Upstream lists six blues and four greens as literals, sized to
    its ten discrete outcomes; ours has fifteen (7 comorbidity + 8 medication), so the literal
    ramps would raise in _cycle_free below -- which is the loud failure they were designed to
    produce, and the fix is to size the ramp to the outcome list rather than to trim outcomes.

    WHAT THIS COSTS, said plainly: the dE figures quoted in the docstring above were validated
    for upstream's ten-outcome palette and do NOT carry over to fifteen. Within-family
    separation is necessarily tighter here. It is survivable because family identity is what
    colour carries in these panels -- every row is named on its axis and every panel writes a
    source-data CSV -- but a reader must not quote the upstream validation for this figure.
    """
    base = plt.get_cmap(cmap)
    return [matplotlib.colors.to_hex(base(x)) for x in np.linspace(hi, lo, n)]


_HIST_RAMP = _ramp(len(_COMORBID), "Blues")
_MED_RAMP = _ramp(len(_MEDS), "Greens")


def _cycle_free(names, ramp, what):
    """Zip names to a ramp, refusing to wrap. A palette that runs out must be a loud error."""
    if len(names) > len(ramp):
        raise ValueError(f"{what}: {len(names)} outcomes but only {len(ramp)} colours -- "
                         f"extend the ramp rather than letting zip() truncate or a modulo wrap")
    return dict(zip(names, ramp))


# The five categorical hues below were checked with the dataviz validator, not by eye:
#   validate_palette.js "#CC79A7,#D55E00,#7B3294,#56B4E9,#E69F00" --mode light --pairs all
#   -> lightness band PASS, chroma floor PASS, CVD all-pairs worst dE 9.6 (deutan) / 8.5
#      (tritan) PASS, normal-vision worst dE 15.6 PASS.
# One WARN stands and is discharged rather than dismissed: three hues sit below 3:1 contrast
# against the surface, which obligates relief. Every panel here writes direct value labels and
# a source-data CSV via save_data(), so identity is never carried by colour alone.
# Death's grey is a NEUTRAL, deliberately outside the categorical band (the validator flags it
# as such): Death is the absorbing reference state, not one series among peers.
OUTCOME_COLORS = {
    # "stage" is the cognitive staging Figure 2's four panels are built on. It replaces the
    # NACC arm's "NACCUDSD" key and keeps that key's colour so the two figures stay comparable.
    "stage":    _OKABE["purple"],           # #CC79A7
    "AD diagnosis": _OKABE["vermillion"],   # the headline event gets the strongest hue
    "Death":    "#4d4d4d",                  # neutral, matches STATE_COLORS' absorbing state
    **_SCALE_COLORS,
    **_cycle_free(_COMORBID, _HIST_RAMP, "comorbidity outcomes"),
    **_cycle_free(_MEDS, _MED_RAMP, "medication outcomes"),
}
# family-level colours, for figures that show a family as one entity
FAMILY_COLORS = {"scales": "#7B3294", "comorbidity": _OKABE["blue"],
                 "medication": _OKABE["green"], "stage": _OKABE["purple"],
                 "AD diagnosis": _OKABE["vermillion"], "Death": "#4d4d4d"}


def _assert_unique(d, what):
    seen = {}
    for k, v in d.items():
        if v in seen:
            raise ValueError(f"{what}: {k!r} and {seen[v]!r} are both {v} -- two series would "
                             f"be indistinguishable. The NACC arm shipped this bug.")
        seen[v] = k


_assert_unique(OUTCOME_COLORS, "OUTCOME_COLORS")


# ---- ordinal-severity ramp (best -> worst) for state-occupancy plots / heatmaps ----
# Generated for an arbitrary number of stages so radc_states.STAGE_MODE="mmse7" does not reuse
# four colours for seven stages. Single hue, light -> dark, which is the correct encoding for an
# ordered magnitude: a diverging or categorical palette on an ordinal stage implies a midpoint
# or an unordered set, and this axis has neither.
def severity_ramp(n, cmap="GnBu"):
    """`n` steps of ONE hue, light (best) -> dark (worst).

    `cmap` exists so two ordinal instruments on the same axes get different hues -- panel a1
    carries both the MMSE-derived stages and the global-cognition levels, and a reader must not
    be able to mistake one family's third step for the other's. Within a family lightness
    carries the order, across families hue carries the identity.
    """
    # Default GnBu and not YlGnBu: YlGnBu sweeps green -> blue, which put its mid-tone within
    # dE 12.7 (normal vision) of the RdPu ramp panel a1 uses for the second instrument -- below
    # the 15 floor, i.e. a full-colour reader could not tell an MMSE stage from a cognition
    # level. Measured with the dataviz validator on the four FAMILY identities
    # (GnBu mid, RdPu mid, #D55E00, #4d4d4d): CVD worst dE 8.8 deutan / 7.2 tritan PASS,
    # normal-vision worst 20.8 PASS. Within a family, order is carried by lightness (monotonic
    # by construction) and identity by the direct row labels, not by hue.
    base = plt.get_cmap(cmap)
    return [matplotlib.colors.to_hex(base(x)) for x in np.linspace(0.28, 0.94, n)]


SEVERITY_COLORS = severity_ramp(5)          # back-compat default
REF_GREY = _OKABE["grey"]


def setup_style():
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300,
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "figure.autolayout": False,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in vector output
        "savefig.bbox": "tight",
    })


def new_fig(nrows=1, ncols=1, figsize=None):
    if figsize is None:
        figsize = (5.0 * ncols, 4.0 * nrows)
    return plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)


def save_fig(fig, out_dir, stem):
    """Save both PDF (vector) and PNG (300 dpi). Returns the two paths."""
    os.makedirs(out_dir, exist_ok=True)
    pdf = os.path.join(out_dir, f"{stem}.pdf"); png = os.path.join(out_dir, f"{stem}.png")
    fig.savefig(pdf); fig.savefig(png, dpi=300)
    plt.close(fig)
    return pdf, png


def save_data(df, out_dir, stem):
    """Save the source data behind a panel as CSV. Accepts a DataFrame or dict-of-arrays."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    if not hasattr(df, "to_csv"):
        df = pd.DataFrame(df)
    path = os.path.join(out_dir, f"{stem}_data.csv")
    df.to_csv(path, index=False)
    return path


def stub_figure(out_dir, stem, reason):
    """A truthful placeholder for a panel that could not be computed (rule 2)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, f"{stem}\n\nSKIPPED / not computable:\n{reason}",
            ha="center", va="center", wrap=True, fontsize=11, color=_OKABE["vermillion"])
    ax.set_axis_off()
    return save_fig(fig, out_dir, stem)


# ---------------------------------------------------------------- statistics helpers
def bootstrap_ci(stat_fn, patient_ids, *, n_boot=1000, seed=0, alpha=0.05):
    """Patient-level bootstrap CI for a scalar statistic.

    stat_fn(sampled_patient_ids) -> float (recomputes the metric on a resampled patient set).
    Resamples UNIQUE patient ids with replacement (never leaks visit-level rows across folds).
    Returns (point, lo, hi). NaN-safe: folds returning NaN are dropped."""
    rng = np.random.default_rng(seed)
    pid = np.asarray(patient_ids)
    uniq = np.unique(pid)
    point = stat_fn(uniq)
    boots = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        v = stat_fn(samp)
        if v is not None and np.isfinite(v):
            boots.append(v)
    if not boots:
        return float(point) if point is not None else float("nan"), float("nan"), float("nan")
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(point), float(lo), float(hi)


def fmt_ci(point, lo, hi, d=3):
    return f"{point:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"
