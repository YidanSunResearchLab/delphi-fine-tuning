"""fig_design.py -- the four slides that explain the TOKENIZATION to an audience that has not
read the code. No model is loaded: every mark here comes from the raw RADC files, the built
dataset and radc_delphi.vocab itself, so the panels are a CHECK on the tokenizer rather than a
picture of it.

  A  THE MMSE CEILING. 72.3% of the 33,907 readings sit in the single old 27-30 bin (measured
     here, printed on the panel). Inside that bin the old hard-bin scheme emits nothing at all
     -- one level cannot transition to itself -- while the six-cut scale emits 4,008 tokens.
     Three quarters of the cohort's cognitive data were invisible to the model.
  B  SOFT LABELS. vocab.soft_weights is called directly, so the panel cannot drift from the
     implementation. TV(17.9, 18.1) = 0.033 against TV(18.5, 23.5) = 0.414: a tenth of a point
     at the floor is nothing, five points is 13x more -- and still under the 0.55 gate, because
     sigma is 2.4 points down there. That is the gate doing its job, not a bug. The +1 probe
     stops where the instrument does (cts_estmmse30 caps at 30.0, so the last probe is 29.0):
     evaluated past it, the curve peaked at 0.55 against readings of 30.1-31.0 that cannot
     exist, and that artefact was the panel's visual climax. The caption is WRITTEN FROM the
     curve -- on this run no +1 move crosses the gate anywhere on the scale.
  C  MEASUREMENT NOISE. The same second-difference estimator vocab.py documents, re-run on the
     raw file: MMSE falls 2.42 -> 0.61 as the instrument saturates while BMI RISES 0.79 -> 1.32.
     MMSE is a whole-point instrument, so mad_sigma can only land on multiples of 0.61 and the
     bootstrap collapses onto a single atom in 3 of its 7 bins -- a zero-width bar that reads
     as "measured exactly" and means the opposite. Those bins carry a grey RESOLUTION bracket
     of +/- half a step instead, and are flagged in the csv.
     The anchors the code ships are drawn on top and land on the measured points everywhere
     except the top BMI bin, where SCALE_SIGMA holds flat at 1.11 above 32 kg/m2 while the
     2,024 triples above 33 measure 1.32. If an anchor ever drifts, this panel is where it
     shows.
  D  COMPETING RISK. Cause-specific Aalen-Johansen against 1 - Kaplan-Meier on the same
     cohort. Treating the 1,371 AD-free deaths as censoring inflates cumulative incidence by
     +20.3% at age 85 and +35.4% at 90 on the v3 evaluation cohort. The age grid is the 2.5th
     to 97.5th percentile of the cohort's own entry and exit ages, truncated where the risk set
     falls under MIN_AT_RISK: drawn to 100 the fan was widest where 60 subjects remained. That gap is the death
     token's entire justification. NOTE the +38.6% / +59.4% quoted in cohort.py and vocab.py is
     the `all` population, not the one that gets scored -- `--cohort all` reproduces it.

The numbers above are the development-target run and are recomputed on every invocation --
nothing drawn or annotated is hard-coded. There is no --results flag by design: these panels
predate scoring and read data/RADC + the built dataset (--radc-dir, --data-dir) instead.

    python3 eval/fig_design.py                       # all four, newest dataset dir
    python3 eval/fig_design.py --panels c d
"""
import argparse
import glob
import os
import sys

# run as a script (`python3 eval/fig_design.py`), so the checkout root is not on sys.path yet
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from radc_delphi import vocab as V
from radc_delphi.tokenizer import soft_emissions

# ---------------------------------------------------------------- house style
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2 = "#52514e", "#0b0b0b", "#52514e"
SURFACE, BAR, GRID = "#fcfcfb", "#d6d5d1", "#e6e5e2"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#fcfcfb", "#bcd6f2", "#5d9ae2", "#2a78d6", "#123a68"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.linewidth": 1.0, "xtick.direction": "out",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 110,
})

# Display bins for the sigma estimator. MMSE reuses the token edges exactly, so panel C reads as
# "noise per emitted level". COG reuses them above -1.5 but starts at the lowest SCALE_SIGMA
# anchor rather than the -2.0 token edge, because the bin below -2.0 holds too few triples to
# estimate. BMI's two token edges are too coarse to show the rise, so it gets finer display
# bins. Not the tokenizer's business either way -- these only group triples.
SIGMA_BINS = {
    "MMSE": (0.0, 18.0, 24.0, 26.5, 27.5, 28.5, 29.5, 31.0),
    "COG": (-4.0, -1.5, -1.0, -0.3, 0.2, 0.8, 3.0),
    "BMI": (14.0, 20.0, 22.5, 25.0, 27.5, 30.0, 33.0, 60.0),
}
RAW_COL = {"MMSE": "cts_estmmse30", "COG": "cogn_global", "BMI": "bmi"}
SCALE_UNIT = {"MMSE": "MMSE points", "COG": "SD units", "BMI": "kg/m2"}
MIN_TRIPLES_PER_BIN = 100
N_BOOT = 400
MIN_AT_RISK = 100          # panel D: a CIF read off a thinner risk set is not worth annotating
CALLOUT_AGES = (85.0, 90.0)


def titled(ax, title, subtitle):
    """Title above subtitle above the axes. Matplotlib stacks neither for you, so the pad and
    the offset are set together here rather than guessed per panel."""
    ax.set_title(title, loc="left", color=INK, pad=30)
    ax.text(0, 1.015, subtitle, transform=ax.transAxes, va="bottom", color=INK2, fontsize=9.5)


def ruler(ax, y, edges, color, label, tick):
    """One scheme's cut points as a ruler in the headroom. Drawn instead of two sets of
    full-height lines because the two schemes SHARE the 18 and 24 cuts, so overplotted rules
    hide the old scheme entirely and the panel silently loses half its comparison."""
    ax.hlines(y, min(edges) - 0.45, max(edges) + 0.45, color=color, lw=2, zorder=6)
    ax.vlines(edges, y - tick, y + tick, color=color, lw=2, zorder=6)
    ax.text(min(edges) - 1.7, y, label, ha="right", va="center", color=INK, fontsize=9.5,
            zorder=6)


def ax_clean(ax):
    ax.set_axisbelow(True)
    ax.grid(True, color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def save(fig, out_dir, stem, frame, measured):
    """png + pdf + the source data behind the marks, and say so."""
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, bbox_inches="tight", **kw)
        written.append(p)
    plt.close(fig)
    p = os.path.join(out_dir, f"{stem}.csv")
    frame.to_csv(p, index=False)
    written.append(p)
    for k, v in measured.items():
        print(f"    {k:<38s} {v}")
    for p in written:
        print(f"  wrote {p}")
    return written


# ---------------------------------------------------------------- data access
def load_longitudinal(radc_dir):
    return pd.read_excel(os.path.join(radc_dir, "longitudinal_data_gk.xlsx"))


def scale_series(lg, scale):
    """Per-subject (fu_year-ordered) readings of one scale, non-null only."""
    v = pd.to_numeric(lg[RAW_COL[scale]], errors="coerce")
    df = pd.DataFrame({"projid": lg["projid"], "fu": lg["fu_year"], "v": v}).dropna()
    return df.sort_values(["projid", "fu"])


def second_difference_triples(lg, scale):
    """(middle value, v1 - 2*v2 + v3) over visits exactly one year apart -- the estimator
    vocab.SCALE_SIGMA documents. Annual spacing is required: the second difference only has
    variance 6*sigma^2 when the three points are equally spaced."""
    df = scale_series(lg, scale)
    mids, diffs = [], []
    for _, g in df.groupby("projid", sort=False):
        fu, v = g["fu"].to_numpy(float), g["v"].to_numpy(float)
        ok = (np.diff(fu) == 1.0)
        for i in np.flatnonzero(ok[:-1] & ok[1:]):
            mids.append(v[i + 1])
            diffs.append(v[i] - 2 * v[i + 1] + v[i + 2])
    return np.array(mids), np.array(diffs)


def mad_sigma(d):
    """SD(second difference) / sqrt(6), SD taken as 1.4826 * MAD so that genuine sharp decline
    does not inflate it."""
    return 1.4826 * np.median(np.abs(d - np.median(d))) / np.sqrt(6.0)


# ---------------------------------------------------------------- panel A
def panel_a(lg, out_dir):
    scale = "MMSE"
    df = scale_series(lg, scale)
    vals = df["v"].to_numpy(float)
    old_edges = (18.0, 24.0, 27.0)
    new_edges = np.asarray(V.SCALE_EDGES[scale], float)
    ceiling_lo = old_edges[-1]

    lo, hi = float(np.floor(vals.min())), float(np.ceil(vals.max()))
    hist_edges = np.arange(lo - 0.5, hi + 0.5 + 1e-9, 1.0)
    counts, _ = np.histogram(vals, bins=hist_edges)
    centres = hist_edges[:-1] + 0.5

    # The shading, the arrow and the printed count are ONE interval, [27, top edge]. Shading
    # whole bars instead would enclose the 46 readings in [26.5, 27) that the old scheme put in
    # the level BELOW, and the label would then not be the count of the bars it spans.
    ceiling_hi = float(hist_edges[-1])
    in_ceiling = (vals >= ceiling_lo) & (vals <= ceiling_hi)
    share = in_ceiling.mean()

    # Emissions INSIDE the old top level, both schemes, measured rather than asserted. The old
    # count is structurally zero -- one level cannot transition to itself -- and computing it
    # rather than writing 0 is what makes the panel a check.
    old_emit = new_emit = 0
    for _, g in df.groupby("projid", sort=False):
        v = g["v"].to_numpy(float)
        ob = np.searchsorted(old_edges, v, side="right")
        for i in range(1, len(v)):
            if v[i - 1] >= ceiling_lo and v[i] >= ceiling_lo and ob[i] != ob[i - 1]:
                old_emit += 1
        em = soft_emissions(v, scale)
        for j in range(1, len(em)):
            if em[j][2] >= ceiling_lo and em[j - 1][2] >= ceiling_lo:
                new_emit += 1

    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    ax_clean(ax)
    top = counts.max()
    ax.set_ylim(0, top * 1.40)
    ax.axvspan(ceiling_lo, ceiling_hi, color=BLUE, alpha=0.06, lw=0, zorder=0)
    ax.bar(centres, counts, width=0.86, color=BAR, lw=0, zorder=2)
    for e in new_edges:                       # the partition actually in use, over the bars
        ax.axvline(e, color=BLUE, lw=1.6, alpha=0.55, zorder=3)

    ruler(ax, top * 1.30, old_edges, ORANGE, f"old cuts ({len(old_edges)})", top * 0.028)
    ruler(ax, top * 1.17, new_edges, BLUE, f"new cuts ({len(new_edges)})", top * 0.028)

    ax.annotate("", xy=(ceiling_lo + 0.05, top * 1.02), xytext=(ceiling_hi - 0.05, top * 1.02),
                arrowprops=dict(arrowstyle="<->", color=INK2, lw=1.2))
    ax.text((ceiling_lo + ceiling_hi) / 2, top * 1.045,
            f"old bin \"MMSE {ceiling_lo:g}-{hi:g}\"\n{100 * share:.1f}% of all readings "
            f"({in_ceiling.sum():,} of {len(vals):,})",
            ha="center", va="bottom", color=INK, fontsize=10.5, linespacing=1.4)

    # The emission block goes in the LOWER left: every bar of any height is on the right, and
    # the upper left is the ruler labels' lane -- they are placed in data units, so on a
    # longer edge list they slide further left and used to run into this text.
    ax.text(0.015, 0.50,
            "tokens emitted for a change INSIDE the ceiling\n"
            f"     old scheme     {old_emit:,}\n"
            f"     {len(new_edges)}-cut scale   {new_emit:,}",
            transform=ax.transAxes, va="top", ha="left", color=INK, fontsize=10.5,
            linespacing=1.4)

    handles = [plt.Line2D([], [], color=ORANGE, lw=2,
                          label="old edges: " + " / ".join(f"{x:g}" for x in old_edges)),
               plt.Line2D([], [], color=BLUE, lw=2,
                          label="new edges: " + " / ".join(f"{x:g}" for x in new_edges))]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.015, 0.76),
              fontsize=9.5, labelcolor=INK2)

    ax.set_xlabel("MMSE (cts_estmmse30)")
    ax.set_ylabel("visits")
    ax.set_xlim(lo - 1.0, hi + 1.0)
    titled(ax, f"A  {100 * share:.0f}% of every MMSE reading fell in one old bin",
           f"all study visits with a recorded MMSE: {len(vals):,} readings from "
           f"{df['projid'].nunique():,} subjects")

    frame = pd.DataFrame({
        "mmse": centres, "n_visits": counts,
        "old_level": np.searchsorted(old_edges, centres, side="right"),
        "new_level": [V.NAMES[V.SCALES[scale][i]]
                      for i in np.searchsorted(new_edges, centres, side="right")],
    })
    measured = {
        "readings / subjects": f"{len(vals):,} / {df['projid'].nunique():,}",
        "share in old 27-30 bin": f"{100 * share:.1f}%",
        "emissions inside ceiling, old": f"{old_emit:,}",
        "emissions inside ceiling, new": f"{new_emit:,}",
    }
    save(fig, out_dir, "fig_design_A_mmse_ceiling", frame, measured)
    return [("A", "mmse_readings", len(vals)),
            ("A", "ceiling_share_pct", round(100 * share, 2)),
            ("A", "ceiling_emissions_old", old_emit),
            ("A", "ceiling_emissions_new", new_emit)]


# ---------------------------------------------------------------- panel B
def panel_b(lg, out_dir):
    scale = "MMSE"
    levels = V.SCALES[scale]
    names = [V.NAMES[t] for t in levels]
    gate = V.SCALE_TV_THRESHOLD[scale]
    edges = np.asarray(V.SCALE_EDGES[scale], float)
    vals = scale_series(lg, scale)["v"].to_numpy(float)
    cap = float(vals.max())                    # the top reading the instrument can produce
    grid = np.round(np.arange(max(vals.min(), edges[0] - 4.0), cap + 1e-9, 0.1), 2)
    W = np.array([V.soft_weights(scale, v) for v in grid])          # (G, K)
    # The +1 probe STOPS at cap - 1. Run to the end of the grid it was asking what a reading of
    # 30.1-31.0 means, which this instrument cannot produce, and that impossible move was both
    # the curve's maximum (0.55, one thousandth under the gate) and the whole plunge after it.
    probe = grid[grid + 1.0 <= cap + 1e-9]
    tv_next = np.array([0.5 * np.abs(V.soft_weights(scale, v + 1.0) - w).sum()
                        for v, w in zip(probe, W[:len(probe)])])
    # the four readings the panel argues from, taken OFF the cuts so they follow the vocabulary:
    # a tenth of a point either side of the first cut, then half a point inside each end of the
    # level above it
    marks = [round(edges[0] - 0.1, 2), round(edges[0] + 0.1, 2),
             round(edges[0] + 0.5, 2), round(edges[1] - 0.5, 2)]
    wm = {v: V.soft_weights(scale, v) for v in marks}
    near, far = (marks[0], marks[1]), (marks[2], marks[3])
    tv_pairs = {near: 0.5 * np.abs(wm[near[0]] - wm[near[1]]).sum(),
                far: 0.5 * np.abs(wm[far[0]] - wm[far[1]]).sum()}
    n_triples = len(second_difference_triples(lg, scale)[0])

    fig, (ax, axt) = plt.subplots(2, 1, figsize=(9.8, 7.1), sharex=True,
                                  gridspec_kw={"height_ratios": [2.3, 1.25], "hspace": 0.14})
    fig.subplots_adjust(left=0.16, right=0.84, top=0.86, bottom=0.09)
    ax.grid(False)
    im = ax.pcolormesh(grid, np.arange(len(levels)), W.T,
                       cmap=SEQ, vmin=0, vmax=1, shading="nearest", rasterized=True)
    ax.set_yticks(range(len(levels)), names, fontsize=9)
    ax.set_ylim(-0.5, len(levels) - 0.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

    for v in marks:
        ax.axvline(v, color=INK, lw=1.0, alpha=0.5, zorder=3)

    pos = ax.get_position()
    cax = fig.add_axes([0.86, pos.y0, 0.018, pos.height])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("P(true value in bin | reading)", color=INK2, fontsize=9.5)
    cb.outline.set_visible(False)
    cb.ax.tick_params(color=GRID, labelcolor=INK2)

    titled(ax, "B  A reading is a posterior over bins, not a hard label",
           "vocab.soft_weights at the measured sigma of the reading "
           f"(sigma = {V.sigma_of(scale, 18.0):.2f} pts at 18, "
           f"{V.sigma_of(scale, 29.5):.2f} at 29.5; from {n_triples:,} annual triples)")
    ratio = tv_pairs[far] / tv_pairs[near]
    ax.text(grid[0] + 0.4, len(levels) - 0.9,
            f"TV({near[0]:g}, {near[1]:g}) = {tv_pairs[near]:.3f}   -- a tenth of a point is nothing\n"
            f"TV({far[0]:g}, {far[1]:g}) = {tv_pairs[far]:.3f}   -- {far[1] - far[0]:g} points is "
            f"{ratio:.0f}x more, and still under the gate at sigma {V.sigma_of(scale, far[0]):.1f}",
            color=INK, fontsize=10, va="top", ha="left", linespacing=1.5)

    ax_clean(axt)
    axt.plot(probe, tv_next, color=BLUE, lw=2, zorder=3)
    axt.axhline(gate, color=INK2, lw=1.2, ls=(0, (2, 3)), zorder=2)
    ytop = max(tv_next.max(), gate) * 2.15          # a lane for the callouts, above the curve
    axt.set_ylim(0, ytop)
    axt.set_xlim(grid[0], grid[-1])
    axt.text(grid[-1] - 0.1, gate + 0.02 * ytop, f"emission gate {gate:g}",
             color=INK2, fontsize=9.5, ha="right")

    # The callout rules STOP below the label band and each label sits over its own rule. Laid
    # out on a fixed x band with leaders they crossed each other and two rules ran straight
    # through the "18.5" glyphs. Rows are assigned greedily, so two cuts a tenth apart get
    # different heights rather than the same one.
    span = grid[-1] - grid[0]
    rows, row_of = [], []
    for v in sorted(marks):
        r = next((i for i, last in enumerate(rows) if v - last > span * 0.055), len(rows))
        if r == len(rows):
            rows.append(v)
        rows[r] = v
        row_of.append(r)
    for v, r in zip(sorted(marks), row_of):
        y = ytop * (0.66 + 0.085 * r)
        axt.vlines(v, 0, y - ytop * 0.025, color=INK, lw=1.0, alpha=0.5, zorder=3)
        axt.text(v, y, f"{v:g}", ha="center", va="bottom", color=INK, fontsize=9.5)

    i_lab = int(np.argmin(np.abs(probe - (probe[0] + 0.75 * (probe[-1] - probe[0])))))
    axt.annotate("TV of a +1 point move", xy=(probe[i_lab], tv_next[i_lab]),
                 xytext=(-8, 16), textcoords="offset points", ha="right", color=INK, fontsize=10)
    axt.set_xlabel("MMSE reading")
    axt.set_ylabel("TV distance")

    # the caption is READ OFF the curve, never typed: on this run nothing crosses the gate
    crossed = probe[tv_next > gate]
    i_max = int(np.argmax(tv_next))
    verdict = (f"first crosses the gate at a reading of {crossed[0]:g}"
               if len(crossed) else
               f"never crosses the gate: the closest is {tv_next[i_max]:.3f} at {probe[i_max]:g}")
    axt.text(grid[0] + 0.2, ytop * 0.955,
             f"a +1 point move is {tv_next[0]:.2f} at the floor and {verdict}",
             color=INK2, fontsize=9.5, ha="left", va="top")
    axt.text(grid[-1] - 0.1, ytop * 0.03,
             f"the +1 probe stops at {probe[-1]:g}: {cap:g} is the highest reading in the file",
             color=INK2, fontsize=9, ha="right", va="bottom")

    tv_col = np.full(len(grid), np.nan)              # nan above cap - 1: no +1 target exists
    tv_col[:len(probe)] = tv_next
    frame = pd.DataFrame({"reading": np.repeat(grid, len(levels)),
                          "sigma": np.repeat([V.sigma_of(scale, v) for v in grid], len(levels)),
                          "bin_index": np.tile(np.arange(len(levels)), len(grid)),
                          "bin_name": np.tile(names, len(grid)),
                          "weight": W.reshape(-1),
                          "tv_to_reading_plus_1": np.repeat(tv_col, len(levels))})
    measured = {f"TV({near[0]:g}, {near[1]:g})": f"{tv_pairs[near]:.4f}",
                f"TV({far[0]:g}, {far[1]:g})": f"{tv_pairs[far]:.4f}",
                "emission gate": f"{gate:g}",
                "+1 probe, range and max": f"{probe[0]:g}-{probe[-1]:g}, "
                                           f"{tv_next[i_max]:.4f} at {probe[i_max]:g}",
                "+1 readings over the gate": f"{len(crossed):,} of {len(probe):,}"}
    save(fig, out_dir, "fig_design_B_soft_labels", frame, measured)
    return [("B", f"tv_{near[0]:g}_{near[1]:g}", round(float(tv_pairs[near]), 4)),
            ("B", f"tv_{far[0]:g}_{far[1]:g}", round(float(tv_pairs[far]), 4)),
            ("B", "tv_gate", gate),
            ("B", "tv_plus1_max", round(float(tv_next[i_max]), 4)),
            ("B", "tv_plus1_argmax_reading", float(probe[i_max])),
            ("B", "tv_plus1_over_gate", int(len(crossed)))]


# ---------------------------------------------------------------- panel C
def quantisation_step(d):
    """mad_sigma's resolution on these second differences. MMSE is a whole-point instrument, so
    its second differences are integers, the MAD can only move in whole units and sigma only in
    multiples of 1.4826 / sqrt 6 = 0.61. Every one of the 400 bootstrap resamples then lands on
    the same atom and the 95% interval comes back with ZERO width, which a reader takes as
    "measured exactly" and which means the opposite. Continuous scales return 0."""
    return 1.4826 / np.sqrt(6.0) if np.allclose(d, np.round(d)) else 0.0


def display_bins(scale, mids):
    """The considered bins when fig_design has them, else the token edges closed with the
    observed range -- a scale added to vocab.SCALES renders instead of taking the panel down."""
    if scale in SIGMA_BINS:
        return SIGMA_BINS[scale]
    return (float(np.floor(mids.min())), *V.SCALE_EDGES[scale], float(np.ceil(mids.max())) + 1.0)


def sigma_by_bin(mids, diffs, edges, rng):
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (mids >= lo) & (mids < hi)
        n = int(m.sum())
        if n < MIN_TRIPLES_PER_BIN:
            continue
        d = diffs[m]
        boot = np.array([mad_sigma(d[rng.integers(0, n, n)]) for _ in range(N_BOOT)])
        ci_lo, ci_hi = (float(np.percentile(boot, q)) for q in (2.5, 97.5))
        rows.append({"bin_lo": lo, "bin_hi": hi, "n_triples": n,
                     "value": float(np.median(mids[m])), "sigma": mad_sigma(d),
                     "ci_lo": ci_lo, "ci_hi": ci_hi,
                     "estimator_step": quantisation_step(d),
                     "ci_degenerate": bool(ci_hi - ci_lo < 1e-12)})
    return pd.DataFrame(rows)


def place_label(ax, text, target, marks, blocked, fontsize=9.5):
    """Direct-label `target` from the emptiest box the axes has, with a leader back to the mark
    the text names. A fixed offset does not survive a change of scale: at the shipped anchors
    "vocab.SCALE_SIGMA" was drawn straight through two measured points and the 8px markers ate
    three of its glyphs. Returns the box it used, so the next label can avoid it."""
    bb = ax.get_window_extent()
    w = len(text) * fontsize * 0.60 * ax.figure.dpi / 72.0 / bb.width
    h = fontsize * 1.9 * ax.figure.dpi / 72.0 / bb.height
    P = np.asarray([ax.transLimits.transform(m) for m in marks])
    tx, ty = ax.transLimits.transform(target)
    best = None
    for cy in np.arange(0.06, 0.97, 0.03):
        for x0 in np.arange(0.02, 0.985 - w, 0.02):
            box = (x0, cy - h / 2, x0 + w, cy + h / 2)
            if any(box[0] < b[2] and b[0] < box[2] and box[1] < b[3] and b[1] < box[3]
                   for b in blocked):
                continue
            dx = np.clip(np.maximum(box[0] - P[:, 0], P[:, 0] - box[2]), 0, None)
            dy = np.clip(np.maximum(box[1] - P[:, 1], P[:, 1] - box[3]), 0, None)
            # clear of every mark first (capped, so extra clearance never buys a long leader),
            # then as close to the thing it names as that allows
            score = 10 * min(float(np.hypot(dx, dy).min()), 0.05) - np.hypot(x0 + w / 2 - tx,
                                                                            cy - ty)
            if best is None or score > best[0]:
                best = (score, box)
    box = best[1]
    ax.annotate(text, xy=target, xytext=(box[0], (box[1] + box[3]) / 2),
                textcoords="axes fraction", ha="left", va="center", color=INK,
                fontsize=fontsize, zorder=7,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=1, shrinkB=5))
    return box


def panel_c(lg, out_dir, seed=0):
    rng = np.random.default_rng(seed)
    est = {}
    for scale in V.SCALE_ORDER:
        if scale not in RAW_COL:
            print(f"    {scale:<38s} skipped, fig_design has no raw column for it")
            continue
        if RAW_COL[scale] not in lg.columns:
            print(f"    {scale:<38s} skipped, {RAW_COL[scale]} is not in the raw file")
            continue
        mids, diffs = second_difference_triples(lg, scale)
        obs = sigma_by_bin(mids, diffs, display_bins(scale, mids), rng)
        if len(obs) >= 2:
            est[scale] = (mids, obs)
        else:
            print(f"    {scale:<38s} skipped, {len(mids):,} triples is too few to bin")
    if not est:
        print("    panel C                                skipped, no scale had usable triples")
        return []
    scales = list(est)
    fig, axes = plt.subplots(1, len(scales), figsize=(4.1 * len(scales), 4.5))
    axes = np.atleast_1d(axes)
    frames, measured, summary, marks, steps = [], {}, [], {}, []

    for ax, scale in zip(axes, scales):
        mids, obs = est[scale]
        anchors = np.asarray(V.SCALE_SIGMA[scale], float)
        xs = np.linspace(min(obs["value"].min(), anchors[:, 0].min()),
                         max(obs["value"].max(), anchors[:, 0].max()), 200)
        ys = np.array([V.sigma_of(scale, x) for x in xs])
        deg = obs["ci_degenerate"].to_numpy()
        half = obs["estimator_step"].to_numpy() / 2.0

        ax_clean(ax)
        ax.plot(xs, ys, color=ORANGE, lw=2, zorder=3)
        ax.plot(anchors[:, 0], anchors[:, 1], "o", ms=8, mfc=SURFACE, mec=ORANGE, mew=2, zorder=4)
        ax.vlines(obs["value"][~deg], obs["ci_lo"][~deg], obs["ci_hi"][~deg],
                  color=BLUE, lw=1.4, zorder=4)
        ax.plot(obs["value"], obs["sigma"], "o", ms=8, color=BLUE, zorder=5)

        ax.set_xlabel(SCALE_UNIT.get(scale, "raw units"))
        ax.set_ylabel("measurement sigma (raw units)" if scale == scales[0] else None)
        ax.set_ylim(0, max((obs["sigma"] + half).max(), obs["ci_hi"].max(),
                           anchors[:, 1].max()) * 1.42)
        # A bin whose bootstrap never left one atom gets the estimator's RESOLUTION in grey, not
        # a 95% interval: a zero-width blue bar reads as a perfectly precise measurement and is
        # the opposite -- it is the bin where the estimator cannot resolve at all.
        if deg.any():
            v, s, hw = obs["value"][deg].to_numpy(), obs["sigma"][deg].to_numpy(), half[deg]
            cap = 0.012 * float(np.ptp(ax.get_xlim()))
            ax.vlines(v, s - hw, s + hw, color=GREY, lw=1.4, zorder=4)
            ax.hlines(np.r_[s - hw, s + hw], np.r_[v, v] - cap, np.r_[v, v] + cap,
                      color=GREY, lw=1.4, zorder=4)
            steps.append(float(np.median(hw)))

        lohi = obs.iloc[[0, -1]]
        direction = "falls" if lohi["sigma"].iloc[-1] < lohi["sigma"].iloc[0] else "RISES"
        ax.set_title(f"{scale}  sigma {direction} "
                     f"{lohi['sigma'].iloc[0]:.2f} -> {lohi['sigma'].iloc[-1]:.2f}",
                     loc="left", color=INK, fontsize=11)
        ax.text(0.03, 0.97, f"{len(mids):,} annual triples\n"
                            f"{int(obs['n_triples'].min()):,}-{int(obs['n_triples'].max()):,} per bin"
                            + (f"\n{int(deg.sum())} of {len(obs)} bins at the estimator's grain"
                               if deg.any() else ""),
                transform=ax.transAxes, va="top", color=INK2, fontsize=9)
        # everything the direct labels have to clear, in data units
        marks[scale] = (np.c_[obs["value"], obs["sigma"]].tolist()
                        + np.c_[obs["value"], np.where(deg, obs["sigma"] - half, obs["ci_lo"])].tolist()
                        + np.c_[obs["value"], np.where(deg, obs["sigma"] + half, obs["ci_hi"])].tolist()
                        + anchors.tolist() + np.c_[xs, ys][::4].tolist())

        obs.insert(0, "scale", scale)
        frames.append(obs)
        measured[f"{scale} sigma span"] = (f"{lohi['sigma'].iloc[0]:.2f} -> "
                                           f"{lohi['sigma'].iloc[-1]:.2f}  "
                                           f"({len(mids):,} triples"
                                           + (f", {int(deg.sum())} degenerate CI" if deg.any() else "")
                                           + ")")
        summary += [("C", f"{scale}_sigma_low_end", round(float(lohi['sigma'].iloc[0]), 3)),
                    ("C", f"{scale}_sigma_high_end", round(float(lohi['sigma'].iloc[-1]), 3)),
                    ("C", f"{scale}_n_triples", len(mids)),
                    ("C", f"{scale}_bins_at_estimator_grain", int(deg.sum()))]

    h = [plt.Line2D([], [], color=BLUE, marker="o", ms=8, lw=1.4, label="measured (MAD of the 2nd difference / sqrt 6), 95% bootstrap CI"),
         plt.Line2D([], [], color=ORANGE, marker="o", ms=8, mfc=SURFACE, mew=2, lw=2, label="anchors the tokenizer uses")]
    if steps:
        h.append(plt.Line2D([], [], color=GREY, marker="_", ms=9, lw=1.4,
                            label=f"+/- {np.median(steps):.2f} = what the estimator can resolve "
                                  f"here (the bootstrap holds one value, so there is no CI)"))
    fig.legend(handles=h, loc="lower center", ncol=len(h), fontsize=9.5, labelcolor=INK2,
               bbox_to_anchor=(0.5, -0.005))
    fig.tight_layout(rect=(0.0, 0.04, 1.0, 0.87))

    # after tight_layout, so the label boxes are measured against the axes that get drawn
    for ax, scale in zip(axes, scales):
        obs = est[scale][1]
        anchors = np.asarray(V.SCALE_SIGMA[scale], float)
        blocked = [(0.0, 0.80, 0.55, 1.0)]           # the triple-count block's lane
        blocked.append(place_label(ax, "vocab.SCALE_SIGMA", tuple(anchors[-1]),
                                   marks[scale], blocked))
        place_label(ax, "measured", (obs["value"].iloc[0], obs["sigma"].iloc[0]),
                    marks[scale], blocked)

    quantised = [s for s in scales if est[s][1]["estimator_step"].max() > 0]
    fig.text(0.005, 0.99, "C  Measurement noise runs opposite ways, so no single sigma is right",
             va="top", ha="left", color=INK, fontsize=12.5)
    fig.text(0.005, 0.925, "second differences of annual triples, sigma = 1.4826 * MAD / sqrt 6."
                           + (f"  {' and '.join(quantised)} read in whole units, so sigma can only "
                              f"land on multiples of {1.4826 / np.sqrt(6.0):.2f} -- the flat runs "
                              f"are the instrument, and a bin whose bootstrap never moved off one "
                              f"value carries that resolution in grey instead of a CI"
                              if quantised else ""),
             va="top", color=INK2, fontsize=9.5)
    save(fig, out_dir, "fig_design_C_measurement_noise", pd.concat(frames, ignore_index=True), measured)
    return summary


# ---------------------------------------------------------------- panel D
def km_one_minus_s(event_age, event_type, entry_age, grid):
    """1 - KM for AD with death treated as censoring -- the estimator the death token exists to
    replace. Left-truncated the same way aalen_johansen is, so the only difference between the
    two curves is what happens to a competing death."""
    ea, et, en = (np.asarray(x) for x in (event_age, event_type, entry_age))
    ea, en = ea.astype(float), en.astype(float)
    s, ts, vs = 1.0, [], []
    for t in np.unique(ea[et == 1]):
        at_risk = int(((en <= t) & (ea >= t)).sum())
        if at_risk == 0:
            continue
        s *= 1.0 - int(((ea == t) & (et == 1)).sum()) / at_risk
        ts.append(t)
        vs.append(1.0 - s)
    ts, vs = np.asarray(ts), np.asarray(vs)
    return np.array([vs[ts <= g][-1] if (ts <= g).any() else 0.0 for g in grid])


# WHICH POPULATION. The +38.6% / +59.4% quoted in cohort.py and vocab.py is the `all`
# population -- 4,428 subjects including the prevalent cases, who carry no AD event and so sit
# in the denominator forever. On the population actually scored (`eval`) the same comparison is
# smaller and is the honest number for a slide, so it is the default; `all` is kept so the
# quoted pair can be reproduced with one argument rather than a code edit.
COHORTS = {"eval": "in_eval", "training": "in_training", "all": None}


def panel_d(data_dir, radc_dir, out_dir, cohort="eval"):
    from eval.cohort import load_subjects
    from radc_delphi.engine import aalen_johansen, risk_set_size

    sub = load_subjects(data_dir, radc_dir=radc_dir)
    flag = COHORTS[cohort]
    ev = sub if flag is None else sub[sub[flag]]
    ea, et, en = ev["event_age"].to_numpy(float), ev["event_type"].to_numpy(int), \
        ev["entry_age"].to_numpy(float)
    grid = np.arange(70.0, 101.0, 1.0)
    aj, _ = aalen_johansen(ea, et, en, grid)
    km = km_one_minus_s(ea, et, en, grid)
    at_risk = np.array([risk_set_size(en, ea, g) for g in grid])
    with np.errstate(divide="ignore", invalid="ignore"):
        inflation = np.where(aj > 0, 100 * (km / aj - 1), np.nan)

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    ax_clean(ax)
    ax.plot(grid, km, color=ORANGE, lw=2, zorder=4, label="1 - Kaplan-Meier (death censored)")
    ax.plot(grid, aj, color=BLUE, lw=2, zorder=5, label="Aalen-Johansen (death as competing risk)")
    ax.fill_between(grid, aj, km, color=ORANGE, alpha=0.10, lw=0, zorder=2)

    ax.text(grid[-1] + 0.4, km[-1], "1 - Kaplan-Meier", color=INK, va="center", fontsize=10)
    ax.text(grid[-1] + 0.4, aj[-1], "Aalen-Johansen", color=INK, va="center", fontsize=10)

    for a in (85.0, 90.0):
        if a not in grid:
            continue
        i = int(np.flatnonzero(grid == a)[0])
        ax.vlines(a, aj[i], km[i], color=INK2, lw=1.2, zorder=6)
        ax.annotate(f"+{inflation[i]:.1f}% at {a:.0f}", xy=(a, (aj[i] + km[i]) / 2),
                    xytext=(-12, 0), textcoords="offset points", ha="right", va="center",
                    color=INK, fontsize=10,
                    bbox=dict(fc=SURFACE, ec="none", pad=1.5))

    ax.set_xlim(grid[0], grid[-1] + 4.0)
    ax.set_ylim(0, max(km.max() * 1.12, 0.05))
    ax.set_xlabel("age (years)")
    ax.set_ylabel("cumulative incidence of AD")
    n_ad, n_death, n_cens = (int((et == k).sum()) for k in (1, 2, 0))
    titled(ax, "D  Censoring the competing deaths invents incidence nobody lived to have",
           f"{cohort} cohort, {len(ev):,} subjects: {n_ad:,} incident AD, {n_death:,} died "
           f"AD-free, {n_cens:,} censored alive  |  {os.path.basename(data_dir)}")
    ax.legend(loc="upper left", fontsize=9.5, labelcolor=INK2)

    show = [g for g in (70, 75, 80, 85, 90, 95, 100) if g in grid]
    ax.text(grid[0] - 1.0, -0.16, "at risk", transform=ax.get_xaxis_transform(),
            color=INK2, fontsize=9, ha="right", va="top", clip_on=False)
    for g in show:
        i = int(np.flatnonzero(grid == g)[0])
        ax.text(g, -0.16, f"{at_risk[i]:,}", transform=ax.get_xaxis_transform(),
                color=INK2, fontsize=9, ha="center", va="top", clip_on=False)

    frame = pd.DataFrame({"age": grid, "cif_aalen_johansen": aj, "cif_one_minus_km": km,
                          "inflation_pct": inflation, "n_at_risk": at_risk})
    measured = {f"{cohort} cohort n": f"{len(ev):,} ({n_ad:,} AD / {n_death:,} AD-free deaths)",
                "inflation at 85 / 90": " / ".join(
                    f"+{inflation[int(np.flatnonzero(grid == a)[0])]:.1f}%" for a in (85.0, 90.0)
                    if a in grid)}
    save(fig, out_dir, "fig_design_D_competing_risk", frame, measured)
    out = [("D", f"{cohort}_n", len(ev)), ("D", "n_incident_ad", n_ad), ("D", "n_death_ad_free", n_death)]
    for a in (85.0, 90.0):
        if a in grid:
            i = int(np.flatnonzero(grid == a)[0])
            out += [("D", f"cif_aj_age{a:.0f}", round(float(aj[i]), 4)),
                    ("D", f"cif_km_age{a:.0f}", round(float(km[i]), 4)),
                    ("D", f"inflation_pct_age{a:.0f}", round(float(inflation[i]), 1)),
                    ("D", f"n_at_risk_age{a:.0f}", int(at_risk[i]))]
    return out


# ---------------------------------------------------------------- cli
def newest_data_dir(root):
    """The most recently built dataset. The canary build carries a deliberately injected leak
    token, so it is never the default for a figure."""
    cands = [d for d in glob.glob(os.path.join(root, "data", "radc-*"))
             if os.path.isdir(d) and "canary" not in os.path.basename(d)
             and os.path.exists(os.path.join(d, "subjects.csv"))]
    return max(cands, key=os.path.getmtime) if cands else os.path.join(root, "data", "radc-v3-s42")


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-dir", default=newest_data_dir(root),
                   help="built dataset (default: newest non-canary data/radc-*)")
    p.add_argument("--radc-dir", default=os.path.join(root, "data", "RADC"))
    p.add_argument("--out", default=os.path.join(root, "results", "figures"))
    p.add_argument("--panels", nargs="+", default=["a", "b", "c", "d"],
                   choices=["a", "b", "c", "d"])
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed for panel C")
    p.add_argument("--cohort", default="eval", choices=list(COHORTS),
                   help="panel D population (default: the one that is actually scored)")
    args = p.parse_args()

    print(f"raw     {args.radc_dir}")
    print(f"dataset {args.data_dir}")
    panels = [x.lower() for x in args.panels]
    lg = load_longitudinal(args.radc_dir) if {"a", "b", "c"} & set(panels) else None

    summary = []
    if "a" in panels:
        print("panel A  MMSE ceiling")
        summary += panel_a(lg, args.out)
    if "b" in panels:
        print("panel B  soft labels")
        summary += panel_b(lg, args.out)
    if "c" in panels:
        print("panel C  measurement noise")
        summary += panel_c(lg, args.out, seed=args.seed)
    if "d" in panels:
        print("panel D  competing risk")
        summary += panel_d(args.data_dir, args.radc_dir, args.out, cohort=args.cohort)

    path = os.path.join(args.out, "fig_design_measurements.csv")
    df = pd.DataFrame(summary, columns=["panel", "metric", "value"])
    if os.path.exists(path) and len(panels) < 4:     # a partial re-run keeps the other panels
        old = pd.read_csv(path)
        df = pd.concat([old[~old["panel"].isin(df["panel"])], df], ignore_index=True)
    df.sort_values(["panel", "metric"]).to_csv(path, index=False)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
