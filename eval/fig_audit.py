"""
fig_audit.py -- the three panels that say whether the endpoint numbers can be believed at all.

    python eval/fig_audit.py                                    # the defaults below
    python eval/fig_audit.py --leakage eval/leakage_val.json --out results/figures

Nothing here computes evidence; it reads eval/leakage_<split>.json and eval/sweep_results.json
and draws what is already in them. Every number on every panel comes out of those files, so a
fresh audit overwrites the artifacts and the same command redraws the figures with no edit.

WHY THESE THREE PANELS AND NOT THE JSON. The audit's verdicts are read off INTERVALS, and an
interval is the one thing a table of point estimates hides. Test 1's margin between the legal
ceiling and the FAIL trigger is 0.035 AUC on ever-AD while the model's 95% interval on 744
audited subjects is 0.087 wide -- so "0.644 < 0.726" is not the finding; "the whole interval
sits below the ceiling" is, and that is a picture.

  PANEL A  Test 1, the baseline-prefix probe. The model is asked for its answer from the
           tokens a subject has at BASELINE, which is exactly the information the legal
           logistic gets. A model that reached past its own input would clear the logistic.
           The measured gap is -0.082 AUC on ever-AD and -0.224 on death, both intervals
           clear of zero on a bootstrap paired over subjects: the model is WORSE than the
           legal comparator from the same prefix, which is the direction that acquits it.

  PANEL B  Test 2, the horizon curve, and the SHAPE is the whole argument. A static leak --
           an outcome baked into the token stream -- is flat across leads and already high at
           12 years, because the leaked token is present regardless of when the diagnosis
           lands. Genuine antecedent signal is near zero at long lead and grows as the cut
           approaches the diagnosis. Measured: +0.176 AUC gained between the 12-year and the
           1-year lead, lower bound +0.088. The long leads rest on 88 and 45 converters, so
           every lead carries its case count under the tick and the two leads where the model
           ranks cases BELOW controls are drawn hollow rather than quietly folded in.

  PANEL C  The capacity sweep, 75 CV runs. Two findings, and the second one is the reason the
           first is reported with a shrug: total loss separates the 26k-parameter cell from
           the rest by 0.25 nats, but 0.30 of the 0.38-nat spread is loss_ce and only 0.083
           is loss_dt. The time head is nearly flat across a 26x parameter range while the
           token head moves, so a ranking on TOTAL loss is a ranking on the CE term wearing a
           dt term as ballast -- and the dt term is two thirds of the total's magnitude. The
           components therefore get their own panels, on a shared y-SPAN so flat reads as
           flat, and never a second y-axis.

PARAMETER COUNT. params = 12 * n_layer * n_embd^2  (the transformer blocks: 4x attention
projections, 8x MLP)  +  vocab_size * n_embd  (the token embedding, counted ONCE because
model.py ties lm_head.weight to transformer.wte.weight)  +  n_embd + 1  (the scalar time
head). There is no positional-embedding term: wpe is commented out in model.py. vocab_size is
read from the audit JSON, not from radc_delphi.vocab -- the working tree's table has moved to
56 tokens since these runs, and a figure must describe the run it plots.
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The validated categorical palette, in slot order, never cycled. All-pairs CVD deltaE 9.2 on
# the #fcfcfb surface. AQUA sits below 3:1 contrast, so anything drawn in it is direct-labeled.
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY = "#52514e"            # the fold-to colour for any series past the third
INK, MUTED = "#0b0b0b", "#52514e"
SURFACE, GRID = "#fcfcfb", "#dcdcd8"
SLOTS = (BLUE, ORANGE, AQUA)


def slot(i):
    """Slot colour for series i, folded to grey past the third -- never a fourth hue."""
    return SLOTS[i] if i < len(SLOTS) else GREY


def frame(ax, xlab=None, ylab=None, title=None, grid="y"):
    ax.set_facecolor(SURFACE)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=3)
    if xlab:
        ax.set_xlabel(xlab, fontsize=9, color=MUTED)
    if ylab:
        ax.set_ylabel(ylab, fontsize=9, color=MUTED)
    if title:
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")


def emit(fig, df, stem, out_dir):
    """One panel out: .png at 200 dpi, .pdf, and the numbers behind the marks."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    csv = os.path.join(out_dir, f"{stem}.csv")
    df.to_csv(csv, index=False)
    paths.append(csv)
    for p in paths:
        print("wrote", p)


def pretty(key):
    return {"ever_ad": "ever-AD", "death": "death"}.get(key, key.replace("_", " "))


def finite(x):
    """A float that is actually a number, else None -- the ceilings block carries NaNs."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def first_finite(*vals):
    """The first of `vals` that is a real number -- `a or b` would swallow a legitimate 0.0."""
    for v in vals:
        f = finite(v)
        if f is not None:
            return f
    return None


def numeric(df, cols):
    """None -> NaN, so a key the audit skipped drops the mark instead of raising."""
    for c in cols:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ceiling_name(kind, n=None, short=False):
    """What a grey tick IS. Two ceilings reach Panel A and they are not the same measurement:
    B2 re-fitted on the audited subjects, and the delivered reference measured on the whole
    4,428-subject cohort. Only the first one has an n on this figure."""
    if kind == "live":
        who = "live refit" if short else "legal ceiling: B2 refit here"
        return f"{who} (n={int(n):,})" if n is not None and np.isfinite(n) else who
    if kind == "reference":
        return "delivered reference" if short else "legal ceiling: delivered reference"
    return "effective ceiling"


# ----------------------------------------------------------------------------------------
# PANEL A -- Test 1, the baseline-prefix probe
# ----------------------------------------------------------------------------------------
def panel_a(leak, out_dir, stem):
    t1 = leak.get("test1") or {}
    ceil_blk = leak.get("ceilings") or {}
    eps = t1.get("endpoints") or {}
    if not eps:
        print("skipped panel A: the artifact carries no test1 endpoints")
        return
    rows = []
    for key, ep in eps.items():
        blk = ceil_blk.get(key) or {}
        live = first_finite(ep.get("ceiling_live"), blk.get("live"))
        ref = first_finite(ep.get("ceiling_reference"), blk.get("reference"))
        effective = first_finite(ep.get("ceiling"), blk.get("effective"),
                                 max([v for v in (live, ref) if v is not None] or [None]))
        trigger = first_finite(ep.get("trigger"), blk.get("trigger"))
        ci = ep.get("ci95") or [np.nan, np.nan]
        d = ep.get("delta_vs_ceiling") or {}
        dci = d.get("ci95") or [np.nan, np.nan]
        paired = bool(d.get("paired"))
        # WHICH ceiling the delta is measured against: leakage.py bootstraps it PAIRED against
        # B2's per-subject predictions, i.e. against the LIVE refit, and falls back to the
        # effective ceiling (= max(reference, live)) when B2 carries no predictions. On death
        # those differ by 0.011 AUC, so the tick drawn must be the one the delta subtracts.
        anchor = live if (paired and live is not None) else effective
        anchor_kind = ("live" if anchor is not None and anchor == live
                       else ("reference" if anchor is not None and anchor == ref else "effective"))
        other = ref if anchor_kind == "live" else live
        if other is not None and anchor is not None and abs(other - anchor) < 5e-5:
            other = None
        rows.append(dict(endpoint=key, label=pretty(key), auc=finite(ep.get("auc")),
                         ci_lo=ci[0], ci_hi=ci[1], ceiling=effective, trigger=trigger,
                         ceiling_live=live, ceiling_reference=ref,
                         anchor=anchor, anchor_kind=anchor_kind, other=other,
                         ceiling_n=blk.get("live_n"),
                         delta=finite(d.get("value")), delta_lo=dci[0], delta_hi=dci[1],
                         paired=paired, n=t1.get("n"), n_pos=ep.get("n_pos"),
                         prevalence=ep.get("prevalence"), readout=ep.get("readout"),
                         verdict=ep.get("verdict")))
    df = numeric(pd.DataFrame(rows),
                 ("auc", "ci_lo", "ci_hi", "ceiling", "trigger", "ceiling_live",
                  "ceiling_reference", "anchor", "other", "ceiling_n", "delta", "delta_lo",
                  "delta_hi", "n", "n_pos", "prevalence"))

    fig, ax = plt.subplots(figsize=(9.2, 1.15 * len(df) + 3.0))
    fig.patch.set_facecolor(SURFACE)
    ys = np.arange(len(df))[::-1].astype(float)
    # both limits are derived and both keep 0.5 in view: a below-chance AUC is a real outcome
    # here (test 2 already has two leads under 0.5) and a fixed left edge would amputate its
    # interval into something that reads as a bar grown from the axis
    marks = np.concatenate([df[c].to_numpy(float) for c in
                            ("auc", "ci_lo", "ci_hi", "anchor", "other", "ceiling", "trigger")])
    marks = marks[np.isfinite(marks)]
    xhi = (float(max(marks.max(), 0.5)) if len(marks) else 1.0) + 0.07
    xlo = (float(min(marks.min(), 0.5)) if len(marks) else 0.45) - 0.03

    for y, r in zip(ys, df.itertuples()):
        half = 0.42
        if np.isfinite(r.trigger):
            # everything above the trigger is a FAIL; shade it so the verdict is spatial
            ax.fill_between([r.trigger, xhi], y - half, y + half, color=ORANGE, alpha=0.10,
                            lw=0, zorder=1)
            ax.plot([r.trigger, r.trigger], [y - half, y + half], color=ORANGE, lw=2,
                    ls=(0, (1, 2)), zorder=3)
        # two ceilings, two kinds of object: solid = B2 refit on exactly these subjects and the
        # thing the delta is measured from, dashed = the delivered reference from the whole
        # cohort. Drawn identically they would be a like-for-like comparison that is not one.
        if np.isfinite(r.other):
            ax.plot([r.other, r.other], [y - half, y + half], color=GREY, lw=1.6,
                    ls=(0, (4, 3)), alpha=0.85, zorder=3)
        if np.isfinite(r.anchor):
            ax.plot([r.anchor, r.anchor], [y - half, y + half], color=GREY, lw=2, zorder=3)
        ax.plot([r.ci_lo, r.ci_hi], [y, y], color=BLUE, lw=2, solid_capstyle="butt", zorder=4)
        ax.plot([r.auc], [y], "o", ms=9, color=BLUE, zorder=5)

    ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=2)
    ax.annotate("null 0.5", (0.5, ys.max() + 0.52), textcoords="offset points", xytext=(3, 0),
                fontsize=8, color=MUTED, ha="left", va="bottom")

    # direct labels on the top row only -- identity never rests on colour, but every row
    # repeating the same three words would be chart junk
    top = df.iloc[0]
    ax.annotate("model, baseline prefix", (top.auc, ys[0]), textcoords="offset points",
                xytext=(0, 13), fontsize=8.5, color=INK, ha="center")
    # the ceiling and the trigger are 0.034 AUC apart on ever-AD, so their labels are
    # staggered rather than centred -- centred they overprint at any sane figure width
    if np.isfinite(top.anchor):
        ax.annotate(ceiling_name(top.anchor_kind, top.ceiling_n), (top.anchor, ys[0] + 0.42),
                    textcoords="offset points", xytext=(-4, 18), fontsize=8.5, color=INK,
                    ha="right")
    if np.isfinite(top.other):
        # the two ticks are 0.005 apart on ever-AD: stack the labels vertically, and send the
        # second one away from the first rather than through it
        side = "right" if top.other <= top.anchor else "left"
        kind = "reference" if top.anchor_kind == "live" else "live"
        ax.annotate(ceiling_name(kind, top.ceiling_n), (top.other, ys[0] + 0.42),
                    textcoords="offset points", xytext=(-4 if side == "right" else 4, 4),
                    fontsize=8.5, color=MUTED, ha=side)
    if np.isfinite(top.trigger):
        ax.annotate("FAIL trigger -- anything in the shaded band", (top.trigger, ys[0] + 0.42),
                    textcoords="offset points", xytext=(4, 3), fontsize=8.5, color=INK,
                    ha="left")

    labels = []
    for r in df.itertuples():
        bits = [f"n={int(r.n):,}"] if np.isfinite(r.n) else []
        if np.isfinite(r.n_pos):
            pos = f"{int(r.n_pos):,} positive"
            if np.isfinite(r.prevalence):
                pos += f" ({100 * r.prevalence:.1f}%)"
            bits.append(pos)
        labels.append(r.label + ("\n" + ", ".join(bits) if bits else ""))
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=9.5, color=INK)
    ax.set_ylim(ys.min() - 1.30, ys.max() + 1.25)
    ax.set_xlim(xlo, xhi)

    for y, r in zip(ys, df.itertuples()):
        if not np.isfinite(r.delta):
            continue
        pair = " paired" if r.paired else ""
        txt = (f"{r.verdict} -- {r.delta:+.3f} AUC vs the "
               f"{ceiling_name(r.anchor_kind, r.ceiling_n, short=True)} {r.anchor:.3f} "
               f"[{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]{pair}")
        if np.isfinite(r.other) and np.isfinite(r.ceiling):
            # say out loud when the ceiling the verdict uses is not the one the delta subtracts
            eff = "live refit" if r.ceiling == r.ceiling_live else "delivered reference"
            txt += f"\neffective ceiling for the verdict: the {eff} {r.ceiling:.3f}"
        ax.annotate(txt, (xlo + 0.004, y - 0.52), fontsize=8, color=MUTED, va="top")

    # the headline is READ OFF the deltas, not asserted: a later checkpoint that clears its
    # ceiling must retitle this panel rather than be labelled with an acquittal it did not earn
    clear = df["delta_hi"].notna() & (df["delta_hi"] < 0)
    if clear.all():
        head = "the model sits BELOW the ceiling a legal scorer reaches from the same tokens"
    elif (df["verdict"] == "FAIL").any() or (df["auc"] >= df["trigger"]).any():
        head = "the model REACHES ITS FAIL TRIGGER -- treat every endpoint as suspect"
    else:
        head = "model against the ceiling a legal scorer reaches from the same tokens"
    frame(ax, xlab="AUC from the baseline prefix (dot = model, 95% CI)", grid="x")
    ax.set_title(f"A -- Leakage test 1 [{t1.get('verdict', '?')}]: {head}",
                 fontsize=10.5, color=INK, loc="left")
    handles = [plt.Line2D([], [], color=BLUE, lw=2, marker="o", ms=8,
                          label="model from the baseline prefix, 95% CI"),
               plt.Line2D([], [], color=GREY, lw=2, label="B2 refit on these subjects"),
               plt.Line2D([], [], color=GREY, lw=1.6, ls=(0, (4, 3)),
                          label="B2 delivered reference (other cohort)"),
               plt.Line2D([], [], color=ORANGE, lw=2, ls=(0, (1, 2)), label="FAIL trigger")]
    # one row along the bottom of the axes, under the last row's caption: the ticks and the
    # shaded FAIL bands own the right-hand side at every artifact this has been run against
    ax.legend(handles=handles, fontsize=7.5, frameon=False, loc="lower left", ncol=4,
              labelcolor=MUTED, handlelength=2.4, borderaxespad=0.1, columnspacing=1.6)
    eff = "; ".join(
        f"{r.label} {r.ceiling:.3f} ({'the live refit' if r.ceiling == r.ceiling_live else 'the delivered reference'})"
        for r in df.itertuples() if np.isfinite(r.ceiling))
    fig.text(0.0, -0.015,
             f"Split {t1.get('tag', leak.get('split', '?'))}. Ceiling = the baseline-prefix "
             f"logistic B2, and it resolves to a different object per endpoint: the effective "
             f"ceiling is {eff}. The printed delta is bootstrapped against the SOLID tick, "
             f"which is the refit on these audited subjects; the dashed tick is the delivered "
             f"reference, measured on the whole cohort (eval/baselines.py) and carrying no n "
             f"here. FAIL trigger sits above both. Overall verdict "
             f"{leak.get('overall', '?')}. A bar from zero would be meaningless here: an AUC's "
             f"null is 0.5.",
             fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    emit(fig, df, stem, out_dir)


# ----------------------------------------------------------------------------------------
# PANEL B -- Test 2, the horizon curve
# ----------------------------------------------------------------------------------------
def panel_b(leak, out_dir, stem):
    t2 = leak.get("test2") or {}
    curve = [c for c in (t2.get("curve") or []) if finite(c.get("gap")) is not None]
    if not curve:
        print("skipped panel B: the artifact carries no scored lead")
        return
    rows = []
    for c in curve:
        ci = c.get("gap_ci") or [np.nan, np.nan]
        rows.append(dict(lead_y=c["lead_y"], gap=c["gap"], gap_lo=ci[0], gap_hi=ci[1],
                         auc_model=c.get("auc_model"), auc_refit=c.get("auc_refit"),
                         model_side=c.get("model_side"), n_cases=c.get("n_cases"),
                         n_rows=c.get("n_rows"), n_train_rows=c.get("n_train_rows"),
                         status=c.get("status")))
    df = numeric(pd.DataFrame(rows),
                 ("lead_y", "gap", "gap_lo", "gap_hi", "auc_model", "auc_refit", "n_cases",
                  "n_rows", "n_train_rows")).sort_values("lead_y").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    fig.patch.set_facecolor(SURFACE)
    x = np.arange(len(df))
    trig = finite(t2.get("gap_trigger"))
    lo = float(np.nanmin(df[["gap", "gap_lo"]].to_numpy()))
    hi = float(np.nanmax(df[["gap", "gap_hi"]].to_numpy() if trig is None else
                         np.append(df[["gap", "gap_hi"]].to_numpy(), trig)))
    pad = 0.18 * (hi - lo)
    ax.set_ylim(lo - pad, hi + pad)
    if trig is not None:
        ax.axhspan(trig, hi + pad, color=ORANGE, alpha=0.10, lw=0, zorder=1)
        ax.axhline(trig, color=ORANGE, lw=2, ls=(0, (1, 2)), zorder=3)
        ax.annotate(f"FAIL trigger {trig:+.2f} -- a static leak lives up here",
                    (x[0] - 0.45, trig), textcoords="offset points", xytext=(0, 5),
                    fontsize=8.5, color=INK, ha="right")
    ax.axhline(0.0, color=MUTED, lw=1, ls=(0, (2, 3)), zorder=2)

    rev = df["model_side"].astype(str).eq("REVERSED").to_numpy()
    ax.plot(x, df["gap"], color=BLUE, lw=2, zorder=4)
    for xi, r, flip in zip(x, df.itertuples(), rev):
        ax.plot([xi, xi], [r.gap_lo, r.gap_hi], color=BLUE, lw=2, solid_capstyle="butt",
                zorder=4)
        ax.plot([xi], [r.gap], "o", ms=9, zorder=5, color=BLUE,
                markerfacecolor=SURFACE if flip else BLUE, markeredgecolor=BLUE, mew=2)
    mid = len(df) // 2
    ax.annotate("model minus legally-refit comparator", (x[mid], df["gap"].iloc[mid]),
                textcoords="offset points", xytext=(0, 46), fontsize=8.5, color=INK,
                ha="center")

    # growth is measured from the MEAN of the long leads, not from the longest one, so that
    # mean is drawn: otherwise the caption's number has no mark on the panel to land on
    gap_long = finite(t2.get("gap_long"))
    long_leads = [v for v in (finite(u) for u in (t2.get("long_leads") or [])) if v is not None]
    xs_long = [i for i, r in enumerate(df.itertuples()) if r.lead_y in long_leads]
    if gap_long is not None and xs_long:
        ax.plot([min(xs_long) - 0.32, max(xs_long) + 0.32], [gap_long, gap_long], color=GREY,
                lw=2, ls=(0, (5, 3)), zorder=4, label="mean of the long leads")
        # the long leads are the crowded end of the curve, so the label sits in the clear
        # band above their intervals and reaches the mean with a leader
        mx = float(np.mean(xs_long))
        top = float(np.nanmax(df.loc[xs_long, ["gap", "gap_hi"]].to_numpy()))
        ax.annotate(f"mean of the long leads {gap_long:+.3f}", xy=(mx, gap_long),
                    xytext=(mx, top + 0.10 * (hi - lo)), ha="center", va="bottom",
                    fontsize=8.5, color=INK,
                    arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8, shrinkA=2))
        ax.plot([], [], color=BLUE, lw=2, marker="o", ms=8, label="model - refit, per lead")
        ax.legend(fontsize=7.5, frameon=False, loc="lower left", labelcolor=MUTED,
                  handlelength=2.4, borderaxespad=0.3)

    ax.set_xticks(x)
    def tick(r):
        bits = [f"{r.lead_y:g} y"]
        if np.isfinite(r.n_cases):
            bits.append(f"{int(r.n_cases):,} cases")
        if np.isfinite(r.n_rows):
            bits.append(f"{int(r.n_rows):,} rows")
        return "\n".join(bits)

    ax.set_xticklabels([tick(r) for r in df.itertuples()], fontsize=8.5, color=MUTED)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    ax.invert_xaxis()          # long lead at the left, so the curve rises toward the diagnosis

    growth = finite(t2.get("growth"))
    gci = t2.get("growth_ci") or [np.nan, np.nan]
    flat = finite(t2.get("flat_trigger"))
    # growth = gap at the short lead minus the MEAN over every lead >= 8 y (leakage.py), so the
    # caption names those two ends rather than the ends of the plotted lead grid: with this
    # artifact's [8, 12] long set the 12 y -> 1 y difference a reader measures is +0.142
    short_lead = finite(t2.get("short_lead")) or float(df["lead_y"].min())
    if long_leads:
        vs = [f"{v:g}" for v in sorted(long_leads)]
        from_txt = (f"the {vs[0]}-year lead" if len(vs) == 1 else
                    f"the mean of the {'-, '.join(vs[:-1])}- and {vs[-1]}-year leads")
    else:
        from_txt = f"the {df['lead_y'].max():g}-year lead"
    span = f"from {from_txt} to the {short_lead:g}-year lead"
    if growth is not None and flat is not None and np.isfinite(gci[0]) and gci[0] > flat:
        head = "the gap closes as the cut approaches the diagnosis"
        shape = (f"SHAPE: antecedent signal. The gap closes by {growth:+.3f} AUC {span} "
                 f"(95% CI [{gci[0]:+.3f}, {gci[1]:+.3f}], clear of the {flat:.2f} flat "
                 f"trigger) -- not the flat, already-elevated curve a static leak draws.")
    elif growth is not None and flat is not None and abs(growth) < flat:
        head = "the gap is FLAT across leads -- the static-leak signature"
        shape = (f"SHAPE: flat ({growth:+.3f} AUC {span}, inside the "
                 f"{flat:.2f} flat trigger) -- the signature of a static leak, NOT of "
                 f"antecedent signal. Read the gap level before believing any horizon.")
    else:
        head = "model minus a legally-refit comparator, by lead"
        shape = (f"SHAPE: growth {growth:+.3f} AUC {span}." if growth is not None else "")
    if rev.any():
        shape += ("\nHollow dots: the model ranks cases BELOW controls at that lead "
                  "(auc_model_raw < 0.5), so its oriented AUC is a two-sided reading.")
    ax.annotate(shape, (0.5, -0.30), xycoords="axes fraction", fontsize=8.5, color=INK,
                ha="center", va="top", wrap=True)

    frame(ax, xlab="lead: years from the cut to the diagnosis (longer lead at the left)",
          ylab="AUC difference (model - refit)")
    ax.set_title(f"B -- Leakage test 2 [{t2.get('verdict', '?')}]: {head}",
                 fontsize=10.5, color=INK, loc="left")
    note = t2.get("note")
    if note:
        ax.annotate(note, (0.5, -0.46), xycoords="axes fraction", fontsize=8, color=MUTED,
                    ha="center", va="top", wrap=True)
    emit(fig, df, stem, out_dir)


# ----------------------------------------------------------------------------------------
# PANEL C -- the capacity sweep
# ----------------------------------------------------------------------------------------
def sweep_table(runs, vocab_size):
    r = pd.DataFrame(runs)
    if "val_total" not in r:
        r["val_total"] = r["best_val"]
    r["val_total"] = r["val_total"].fillna(r["best_val"])
    r["params"] = (12 * r["n_layer"] * r["n_embd"] ** 2      # blocks: 4x attn + 8x MLP
                   + vocab_size * r["n_embd"]                # tied token embedding / lm_head
                   + r["n_embd"] + 1)                        # scalar time head
    g = r.groupby(["n_layer", "n_embd", "dropout", "params"], as_index=False).agg(
        n_folds=("fold", "nunique"),
        total_mean=("val_total", "mean"), total_sd=("val_total", "std"),
        ce_mean=("val_ce", "mean"), ce_sd=("val_ce", "std"),
        dt_mean=("val_dt", "mean"), dt_sd=("val_dt", "std"),
        iter_median=("best_iter", "median"))
    return g.sort_values(["dropout", "params"]).reset_index(drop=True)


def panel_c(runs, out_dir, stem, vocab_size, vocab_src, select):
    df = sweep_table(runs, vocab_size)
    if select:
        L, E, p = select
        hit = df[(df.n_layer == L) & (df.n_embd == E) & (np.isclose(df.dropout, p))]
    else:
        hit = df.loc[[df["total_mean"].idxmin()]]
    df["selected"] = df.index.isin(hit.index)
    sel = hit.iloc[0] if len(hit) else None

    fig = plt.figure(figsize=(10.6, 8.4))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0], hspace=0.62, wspace=0.24)
    ax_tot = fig.add_subplot(gs[0, :])
    ax_ce = fig.add_subplot(gs[1, 0])
    ax_dt = fig.add_subplot(gs[1, 1])

    drops = sorted(df["dropout"].unique())
    for i, p in enumerate(drops):
        d = df[np.isclose(df.dropout, p)].sort_values("params")
        c = slot(i)
        for ax, mu, sd in ((ax_tot, "total_mean", "total_sd"), (ax_ce, "ce_mean", "ce_sd"),
                           (ax_dt, "dt_mean", "dt_sd")):
            ax.fill_between(d["params"], d[mu] - d[sd], d[mu] + d[sd], color=c, alpha=0.15,
                            lw=0, zorder=2)
            ax.plot(d["params"], d[mu], color=c, lw=2, zorder=3, label=f"dropout {p:g}")
            ax.plot(d["params"], d[mu], "o", ms=8, color=c, zorder=4)
            # <= 4 series are direct-labeled as well as legended, so identity is never
            # colour-alone -- and #1baf7a must always carry one
            first = d.iloc[0]
            ax.annotate(f"{p:g}", (first["params"], first[mu]), textcoords="offset points",
                        xytext=(-9, -3 + (i - 1) * 11), fontsize=8.5, color=INK, ha="right")

    if sel is not None:
        ax_tot.plot([sel["params"]], [sel["total_mean"]], "o", ms=17, mfc="none",
                    mec=INK, mew=1.6, zorder=6)
        ax_tot.annotate(f"selected: {int(sel.n_layer)}L x {int(sel.n_embd)}d, "
                        f"dropout {sel.dropout:g}\n{int(sel['params']):,} params, "
                        f"mean {sel['total_mean']:.3f} +/- {sel['total_sd']:.3f}",
                        (sel["params"], sel["total_mean"]), textcoords="offset points",
                        xytext=(14, -16), fontsize=8.5, color=INK, ha="left")
        # the sweep's own rule: cells whose intervals overlap are not distinguishable, and the
        # smaller one should win on principle. Name the cheapest cell that ties the winner.
        tie = df[df["total_mean"] <= sel["total_mean"] + sel["total_sd"]]
        alt = tie.loc[tie["params"].idxmin()] if len(tie) else None
        if alt is not None and alt["params"] < sel["params"]:
            ax_tot.annotate(f"within 1 sd of the winner at {sel['params'] / alt['params']:.1f}x "
                            f"fewer parameters:\n{int(alt.n_layer)}L x {int(alt.n_embd)}d, "
                            f"dropout {alt.dropout:g} ({alt['total_mean']:.3f})",
                            (alt["params"], alt["total_mean"]), textcoords="offset points",
                            xytext=(-6, -34), fontsize=8, color=MUTED, ha="right")
            ax_tot.plot([alt["params"]], [alt["total_mean"]], "o", ms=15, mfc="none",
                        mec=MUTED, mew=1.2, ls="--", zorder=6)

    ce_span = df["ce_mean"].max() - df["ce_mean"].min()
    dt_span = df["dt_mean"].max() - df["dt_mean"].min()
    # SAME y-span on both component panels, different offsets: flat must read as flat, and
    # two measures on two scales get two panels, never a second y-axis
    span = max(ce_span, dt_span) * 1.9
    for ax, col, sdcol in ((ax_ce, "ce_mean", "ce_sd"), (ax_dt, "dt_mean", "dt_sd")):
        mid = 0.5 * (df[col].max() + df[col].min())
        ax.set_ylim(mid - span / 2, mid + span / 2)

    for ax, lab in ((ax_tot, "mean CV loss (total)"), (ax_ce, "loss_ce (next token)"),
                    (ax_dt, "loss_dt (time to event)")):
        ax.set_xscale("log")
        frame(ax, xlab="parameters", ylab=lab)
    ax_tot.set_xlabel("parameters (log scale)", fontsize=9, color=MUTED)

    n_runs, n_cells = len(runs), len(df)
    folds = int(df["n_folds"].max())
    ax_tot.set_title(f"C -- Capacity sweep: {n_runs} runs, {n_cells} cells x {folds} CV folds "
                     f"(band = +/- 1 sd across folds)", fontsize=10.5, color=INK, loc="left")
    ax_tot.legend(fontsize=8, frameon=False, loc="upper right", labelcolor=MUTED)
    ax_ce.set_title(f"loss_ce MOVES: {ce_span:.3f} nats across the range", fontsize=9.5,
                    color=INK, loc="left")
    ax_dt.set_title(f"loss_dt is FLAT: {dt_span:.3f} nats on the same y-span", fontsize=9.5,
                    color=INK, loc="left")
    for ax in (ax_ce, ax_dt):
        ax.legend(fontsize=7.5, frameon=False, loc="upper right", labelcolor=MUTED)
        ax.margins(x=0.12)
    ax_tot.margins(x=0.10)

    fig.text(0.5, 0.435,
             f"Why architecture cannot be ranked on total loss here: loss_dt contributes "
             f"{df['dt_mean'].mean():.2f} of the {df['total_mean'].mean():.2f}-nat total but "
             f"varies by only {dt_span:.3f} across a "
             f"{df['params'].max() / df['params'].min():.0f}x parameter range, while loss_ce "
             f"varies by {ce_span:.3f}. Both panels below span the same {span:.3f} nats.",
             fontsize=8.5, color=INK, ha="center", va="top", wrap=True)
    fig.text(0.0, 0.03,
             f"params = 12*n_layer*n_embd^2 (blocks) + {vocab_size}*n_embd (token embedding, "
             f"tied to lm_head) + n_embd+1 (time head); no positional term. "
             f"vocab_size from {vocab_src}.",
             fontsize=8, color=MUTED, ha="left", va="top", wrap=True)
    emit(fig, df, stem, out_dir)


def parse_select(s):
    if not s or s.lower() == "auto":
        return None
    a, b, c = s.split(",")
    return int(a), int(b), float(c)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--leakage", default=os.path.join(_ROOT, "eval", "leakage_test.json"))
    ap.add_argument("--sweep", default=os.path.join(_ROOT, "eval", "sweep_results.json"))
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "figures"))
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="override the vocabulary the swept runs were trained on")
    ap.add_argument("--select", default="auto",
                    help="'n_layer,n_embd,dropout' of the shipped cell, or auto = lowest mean")
    ap.add_argument("--prefix", default="fig_audit")
    args = ap.parse_args()

    leak = json.load(open(args.leakage))
    runs = json.load(open(args.sweep))

    if args.vocab_size:
        vocab_size, vocab_src = args.vocab_size, "--vocab-size"
    elif leak.get("vocab_size"):
        vocab_size, vocab_src = leak["vocab_size"], os.path.basename(args.leakage)
        # the working tree's table has moved on since these runs; say so rather than silently
        # drawing one vocabulary's parameter count under another's label
        try:
            from radc_delphi.vocab import VOCAB_SIZE
            if VOCAB_SIZE != vocab_size:
                vocab_src += f" (radc_delphi.vocab is now {VOCAB_SIZE})"
        except ImportError:
            pass
    else:
        from radc_delphi.vocab import VOCAB_SIZE
        vocab_size, vocab_src = VOCAB_SIZE, "radc_delphi.vocab"

    tag = leak.get("split") or leak.get("test1", {}).get("tag") or "split"
    panel_a(leak, args.out, f"{args.prefix}_A_prefix_probe_{tag}")
    panel_b(leak, args.out, f"{args.prefix}_B_horizon_curve_{tag}")
    panel_c(runs, args.out, f"{args.prefix}_C_capacity_sweep", vocab_size, vocab_src,
            parse_select(args.select))


if __name__ == "__main__":
    main()
