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
# one fixed color per outcome, reused everywhere
# One color per outcome. With every sum-score split into items there are now 30 scales, so
# the map is BUILT rather than typed out: one hue family per instrument, separated within a
# family by lightness (survives greyscale and colorblind simulation), and the four
# single-scale outcomes keep their original Okabe-Ito colors so old figures stay comparable.
def _ramp(hexes):
    return list(hexes)

_CDR_RAMP  = ["#08306B","#08519C","#2171B5","#4292C6","#6BAED6","#9ECAE1"]          # blues
_FAQ_RAMP  = ["#00441B","#006D2C","#238B45","#41AB5D","#74C476","#A1D99B",
              "#C7E9C0","#E5F5E0","#F7FCF5"]                                        # greens
_NPI_RAMP  = ["#3F007D","#54278F","#6A51A3","#807DBA","#9E9AC8","#BCBDDC",
              "#DADAEB","#EFEDF5","#7A0177","#AE017E","#DD3497","#F768A1"]          # purples/magentas
CDR_BOXES    = ["MEMORY","ORIENT","JUDGMENT","COMMUN","HOMEHOBB","PERSCARE"]
FAQ_DOMAINS  = ["BILLS","TAXES","GAMES","STOVE","MEALPREP","EVENTS","PAYATTN","REMDATES","TRAVEL"]
NPI_SYMPTOMS = ["DEL","HALL","AGIT","DEPD","ANX","ELAT","APA","DISN","IRR","MOT","NITE","APP"]

OUTCOME_COLORS = {
    "MOCA":     _OKABE["orange"],
    "NACCUDSD": _OKABE["purple"],
    "Death":    _OKABE["vermillion"],
    "GDS":      _OKABE["yellow"],
    **dict(zip(CDR_BOXES, _CDR_RAMP)),
    **dict(zip(FAQ_DOMAINS, _FAQ_RAMP)),
    **dict(zip(NPI_SYMPTOMS, _NPI_RAMP)),
}
# family-level colors, for figures that show an instrument as one entity
FAMILY_COLORS = {"CDR": _OKABE["blue"], "FAQ": _OKABE["green"], "NPI": "#6A51A3",
                 "GDS": _OKABE["yellow"], "MOCA": _OKABE["orange"],
                 "NACCUDSD": _OKABE["purple"], "Death": _OKABE["vermillion"]}
CDR_COLOR = FAMILY_COLORS["CDR"]   # back-compat alias
# ordinal-severity ramp (Normal -> Severe) for state-occupancy plots / heatmaps
SEVERITY_COLORS = ["#2c7fb8", "#7fcdbb", "#fdae61", "#d7191c", "#7b3294"]
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
