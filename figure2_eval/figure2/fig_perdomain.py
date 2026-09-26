"""
fig_perdomain.py -- the per-domain evaluation as a figure, in Figure 2's visual language.

Reads perdomain_<cohort>.json (written by score_perdomain.py) and renders:

  a  forest plot -- AUC at 5 y with a 95% patient-bootstrap CI, one row per outcome,
     grouped into outcome families (ordinal scales / comorbidity / medication) and sorted within each
  b  small multiples -- AUC against prediction horizon, one facet per instrument

WHY DOTS AND NOT BARS. An AUC's null is 0.5, not 0. A bar grown from zero spends half the
axis on a range no estimate can occupy and silently implies that 0 is the reference. The
reference here is the dotted line at 0.5, and the mark that belongs against a reference line
is a dot with its interval.

COLOR. Four instrument families, Okabe-Ito, checked with the palette validator rather than by
eye: all-pairs CVD separation dE 11.0 (deutan), normal-vision floor 15.6, both PASS. The one
WARN it does raise -- #E69F00 sits below 3:1 against the surface -- is discharged the way the
validator asks, by relief: every row carries its own text label and every panel ships a
source-data CSV, so identity is never colour-alone and the numbers are readable without it.
"""
import os, sys, json, argparse
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from figure2 import plotting_style as ps                     # noqa: E402
from figure2 import perdomain as PD                          # noqa: E402

# Families for RADC. Derived from perdomain's own lists rather than typed out, so adding an
# outcome to the vocabulary cannot leave it silently unplotted.
FAMILY = {"ordinal scales": list(PD.SCALES),
          "comorbidity": [n for n in PD.EVENT_NAMES
                          if "Antihyp" not in n and "Statin" not in n],
          "medication": [n for n in PD.EVENT_NAMES
                         if "Antihyp" in n or "Statin" in n]}
_placed = {n for v in FAMILY.values() for n in v}
assert _placed == set(PD.ALL_OUTCOMES), (
    f"outcome not assigned to a family: {sorted(set(PD.ALL_OUTCOMES) - _placed)}")
# validated: node scripts/validate_palette.js "#0072B2,#009E73,#D55E00,#E69F00" --pairs all
FAM_COLOR = {"ordinal scales": "#7B3294", "comorbidity": "#0072B2",
             "medication": "#009E73"}
HORIZONS = [1, 2, 3, 5, 10]
PRIMARY_H = 5


def _rows(js, h=PRIMARY_H):
    out = []
    for fam, names in FAMILY.items():
        got = []
        for nm in names:
            r = js.get(nm)
            if not r:
                continue
            hh = r.get("horizons", {}).get(str(h), r.get("horizons", {}).get(h, {}))
            got.append(dict(outcome=nm, family=fam, auc=hh.get("auc"), lo=hh.get("auc_lo"),
                            hi=hh.get("auc_hi"), n_pos=hh.get("n_pos"), n=r.get("n_scorable"),
                            recon=nm.endswith("_recon")))
        # sort within the block, but keep the reconstructed total pinned to the bottom of its
        # block -- it is a different KIND of row (an aggregate of the rows above it), and
        # letting it sort into the middle invites reading it as just another domain.
        dom = sorted([g for g in got if not g["recon"]],
                     key=lambda g: (g["auc"] is None, -(g["auc"] or 0)))
        out += dom + [g for g in got if g["recon"]]
    return out


def panel_a(ax, rows):
    y = np.arange(len(rows))[::-1]
    for yi, r in zip(y, rows):
        c = FAM_COLOR[r["family"]]
        if r["auc"] is None:
            ax.text(0.5, yi, "  n_pos < 30 — not estimable", va="center", fontsize=7.5,
                    color="#666666")
            continue
        if r["lo"] is not None:
            ax.plot([r["lo"], r["hi"]], [yi, yi], lw=2.0, color=c, solid_capstyle="round",
                    zorder=2)
        # a reconstructed total is an aggregate of the rows above it -> hollow marker, so the
        # two kinds of row are distinguishable without relying on the label alone
        ax.plot([r["auc"]], [yi], "o", ms=8 if r["recon"] else 7, color=c, zorder=3,
                mfc="white" if r["recon"] else c, mew=2.0 if r["recon"] else 0,
                markeredgecolor=c)
        ax.text(1.005, yi, f"{r['auc']:.3f}", va="center", fontsize=8,
                color=ps._OKABE["black"], transform=ax.get_yaxis_transform())
    ax.axvline(0.5, ls=":", lw=1.2, color=ps.REF_GREY, zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['outcome']}  (n+={r['n_pos']})" if r["n_pos"] else r["outcome"]
                        for r in rows], fontsize=8)
    for t, r in zip(ax.get_yticklabels(), rows):
        if r["recon"]:
            t.set_fontweight("bold")
    ax.set_xlim(0.35, 1.0)
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_xlabel(f"AUC for worsening ≥1 bin within {PRIMARY_H} y  (95% patient bootstrap)")
    ax.set_title("a   Discrimination per outcome", loc="left", fontweight="bold", pad=14)

    # Family blocks. The label goes in the LEFT MARGIN, rotated, spanning its whole block.
    # Placed inline (near the block's first row) it lands between two rows and reads as
    # belonging to the row above -- which is how the first draft mislabelled every block.
    spans, start, prev = [], 0, rows[0]["family"]
    for i, r in enumerate(rows + [{"family": None}]):
        if r["family"] != prev:
            spans.append((prev, start, i - 1)); start, prev = i, r["family"]
    for e in [b for _, b, _ in spans][1:]:
        ax.axhline(len(rows) - e - 0.5, color="#dddddd", lw=1.0, zorder=0)
    for fam, i0, i1 in spans:
        ymid = len(rows) - 1 - (i0 + i1) / 2.0
        ax.annotate(fam, xy=(0, ymid), xycoords=("axes fraction", "data"),
                    xytext=(-118, 0), textcoords="offset points",
                    fontsize=10, fontweight="bold", color=FAM_COLOR[fam],
                    rotation=90, ha="center", va="center", annotation_clip=False)


def panel_b(axes, js):
    for ax, (fam, names) in zip(axes, FAMILY.items()):
        for nm in names:
            r = js.get(nm)
            if not r:
                continue
            hs = r.get("horizons", {})
            xs, ys = [], []
            for h in HORIZONS:
                hh = hs.get(str(h), hs.get(h, {}))
                if hh.get("auc") is not None:
                    xs.append(h); ys.append(hh["auc"])
            if len(xs) < 2:
                continue
            recon = nm.endswith("_recon")
            ax.plot(xs, ys, "-o", ms=5 if not recon else 7, lw=2.4 if recon else 1.4,
                    color=FAM_COLOR[fam], alpha=1.0 if recon else 0.45,
                    mfc="white" if recon else FAM_COLOR[fam],
                    mew=2.0 if recon else 0, markeredgecolor=FAM_COLOR[fam],
                    zorder=3 if recon else 2,
                    label=nm if recon else None)
        ax.axhline(0.5, ls=":", lw=1.2, color=ps.REF_GREY)
        # family name INSIDE the facet: the title slot is spent on the panel heading, and two
        # pieces of text competing for it is what collided in the first draft
        ax.text(0.97, 0.955, fam, transform=ax.transAxes, fontsize=10, fontweight="bold",
                color=FAM_COLOR[fam], ha="right", va="top")
        ax.set_xticks(HORIZONS); ax.set_ylim(0.35, 1.0)
        ax.set_xlabel("horizon (years)")
        if ax is axes[0]:
            ax.set_ylabel("AUC")
        else:
            ax.set_yticklabels([])
        if any(n.endswith("_recon") for n in names):
            ax.legend(fontsize=7, loc="lower left", frameon=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default="matched", choices=["matched", "all"])
    ap.add_argument("--dir", default=os.path.join(HERE, "results", "figure2"))
    a = ap.parse_args()
    jp = os.path.join(a.dir, f"perdomain_{a.cohort}.json")
    js = json.load(open(jp))
    rows = _rows(js)

    ps.setup_style()
    fig = plt.figure(figsize=(13.5, 11.0))
    gs = fig.add_gridspec(2, 4, height_ratios=[2.45, 1.0], hspace=0.26, wspace=0.10,
                      left=0.175, right=0.945, top=0.935, bottom=0.055)
    ax_a = fig.add_subplot(gs[0, :])
    panel_a(ax_a, rows)
    axes_b = [fig.add_subplot(gs[1, i]) for i in range(4)]
    panel_b(axes_b, js)
    axes_b[0].set_title("b   Discrimination by horizon", loc="left", fontweight="bold", pad=10)
    fig.suptitle(f"Per-domain evaluation — the 30 scales Figure 2 does not score "
                 f"(cohort: {a.cohort})", fontsize=13, y=0.985)
    stem = f"fig_perdomain_{a.cohort}"
    ps.save_fig(fig, a.dir, stem)
    # source data, per this repo's convention: every panel ships the numbers behind it
    ps.save_data(pd.DataFrame(rows), a.dir, stem)
    print(f"-> {os.path.join(a.dir, stem)}.png / _data.csv")


if __name__ == "__main__":
    main()
