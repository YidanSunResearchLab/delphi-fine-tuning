"""
fig_endpoints.py -- the headline scientific panels, straight from eval/results_test.json.

    python eval/fig_endpoints.py --results eval/results_test.json --out results/figures

Five figures: A discrimination, B calibration, C cumulative incidence + mortality,
D trajectory realism, and a one-slide SUMMARY stat row. Every mark is read out of the JSON at
run time and written back out as a source-data .csv beside the panel, so a fresh evaluate.py
run needs no edit here -- only the same schema.

--------------------------------------------------------------------------------------------
WHY THESE PANELS LOOK THE WAY THEY DO. Numbers below are what the development artifact
(ckpt e7bc3e9c0241, test split, iter 5400) measured; they are quoted to explain a design
choice, never used to draw anything.

  1. LEAD TIME IS THE AXIS THAT MOVES THE ANSWER, so it is a position, not a colour. At the
     5-y horizon the AUC falls 0.874 -> 0.811 when the landmark is forced >= 2 y clear of the
     diagnosis, and at H = 1 the same filter leaves ZERO positives and no AUC at all. Panel A
     therefore facets by horizon and puts lead on the x-axis, draws the unreportable arm as an
     explicit "no positives" note rather than a gap, and prints evaluable-n and event count
     beside every estimate -- competing deaths drop 352 / 803 / 1,159 rows at H = 1 / 3 / 5,
     so the three arms do not share a denominator and must not look as though they do. That
     zero-positive arm is also the one meta.headline names (evaluate.py picks the headline
     from the argparse horizons alone), so headline_arm() re-checks reportability rather than
     trusting it.

  2. AN AUC'S NULL IS 0.5. Every AUC in here is a dot with its 95% bootstrap interval against
     a dotted 0.5 reference, never a bar from zero, including inside the summary tile.

  3. CALIBRATION IS A LEAD-0 OBJECT. The lead filter is selection on the outcome: ranking is
     invariant to it, absolute risk is not (evaluate.py measures observed AJ at 5 y of 0.115
     filtered against 0.203 unfiltered). So panel B plots the headline HORIZON at the lowest
     lead that carries calibration bins and says so on the panel.

  4. THE AGE-95 CIF IS NOT THE SAME KIND OF OBJECT AS THE AGE-85 ONE -- the age-80 landmark
     risk set falls 161 -> 71 -> 12. Panel C prints the risk set under every age and carries
     the wider evaluation-split curve as a grey context series, because 12 subjects cannot
     carry a claim on their own. That grey curve is NOT the same estimand as the orange one:
     evaluate.py enters it at each subject's baseline age, not at the age-80 origin, so an AD
     diagnosis before 80 counts in grey and cannot count in orange. Most of the 0.204 vs
     0.127 gap at 85 is that origin, not the population, and the panel says so on its face.

  5. SIM-VS-OBS ALONE CONFLATES CASE MIX WITH THE MODEL. The pooled simulated arm averages
     every surviving draw while the observed arm averages the subjects still under
     observation, so panel D draws evaluate.py's case-mix-matched arm beside it wherever the
     JSON provides it (MMSE year 15: 2.40 pooled, 2.70 matched, 2.40 observed).

  6. MORTALITY IS A SECOND MEASURE, SO IT IS A SECOND PANEL, never a second axis on the AD
     curve.

PALETTE. Three categorical slots in fixed order, never cycled; a fourth level folds to grey
(panel A past three horizons keeps colour for the headline horizon only, since the facet
titles already carry identity). SIM / OBS / MATCHED fix the hue for a measure once for the
whole deck, so blue cannot mean "the model" in C and "the data" in D. #1baf7a sits below 3:1
contrast on the surface, so every mark drawn in it is direct-labelled. Text is ink, never the
series colour. Every AUC scale is derived from the data with the 0.5 null kept inside, so a
below-chance arm still has a visible dot.
"""
import argparse
import json
import os
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                             # noqa: E402
import numpy as np                                                          # noqa: E402
import pandas as pd                                                         # noqa: E402

# run as a script from anywhere, so the defaults point at the checkout, not at the cwd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Fixed categorical order (blue, orange, aqua); grey is the fold-to level for context series
# and for every piece of secondary text. Validated all-pairs CVD dE 9.2 on the surface.
C1, C2, C3 = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK2, SURFACE, GRIDC = "#52514e", "#0b0b0b", "#52514e", "#fcfcfb", "#d7d6d2"
# One semantic mapping for the whole deck. C and D are both "simulated vs observed" and are
# read minutes apart, so the hue for a measure is fixed here once and never per panel.
SIM, OBS, MATCHED = C1, C2, C3
DPI = 200

# An arm with too few positives has no AUC; evaluate.py flags it with auc_reportable and we
# draw the reason instead of a hole. The threshold is evaluate.py's MIN_POSITIVES and is not
# in the JSON, so the note reads the count off the arm rather than naming a number.
def no_auc_note(n_pos):
    return ("no positives\nat this lead" if n_pos == 0 else
            "too few positives\nto score an AUC")


# ------------------------------------------------------------------ small shared helpers
def fmt_n(x):
    return f"{int(x):,}" if x is not None and np.isfinite(x) else "n/a"


def fmt_pct(x, nd=1):
    return f"{100 * x:.{nd}f}%" if x is not None and np.isfinite(x) else "n/a"


def fmt3(x):
    return f"{x:.3f}" if x is not None and np.isfinite(x) else "n/a"


def num(x):
    """JSON null -> nan, so every downstream isfinite test works on one type."""
    return float("nan") if x is None else float(x)


def wrap(text, width=46):
    return "\n".join(textwrap.fill(line, width) for line in text.split("\n"))


def direct_labels(ax, items, dx=9, gap_frac=0.055):
    """Label each series at its own line end, pushed apart when two ends nearly coincide.

    Identity must never rest on colour alone, so every series gets a label; two series that
    finish within a few thousandths of each other would otherwise print on top of each other.
    Call AFTER the limits are set -- the minimum gap is a fraction of the visible span.
    """
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * gap_frac
    prev = -np.inf
    for x, y, text in sorted(items, key=lambda t: t[1]):
        y_lab = max(y, prev + gap)
        prev = y_lab
        ax.annotate(text, (x, y_lab), textcoords="offset points", xytext=(dx, 0),
                    fontsize=8.5, color=INK, va="center", ha="left")
    if prev > hi:
        ax.set_ylim(lo, prev + gap * 0.6)


def axis_style(ax, yaxis_grid=True):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y" if yaxis_grid else "both", color=GRIDC, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDC)
    ax.tick_params(colors=INK2, labelsize=8, length=3)


def save(fig, rows, out_dir, stem):
    """One panel -> .png, .pdf and the numbers behind the marks."""
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, f"{stem}.{ext}")
        fig.savefig(p, dpi=DPI, facecolor=SURFACE, bbox_inches="tight")
        paths.append(p)
    plt.close(fig)
    csv = os.path.join(out_dir, f"{stem}.csv")
    pd.DataFrame(rows).to_csv(csv, index=False)
    paths.append(csv)
    for p in paths:
        print(f"  wrote {p}")


def footer(fig, text, y=0.005):
    fig.text(0.005, y, text, fontsize=7, color=INK2, ha="left", va="bottom")


def meta_stamp(res):
    m = res.get("meta", {})
    bits = [f"split {m.get('split', '?')}", f"ckpt {m.get('ckpt_sig', m.get('ckpt', '?'))}"]
    if m.get("iter_num") is not None:
        bits.append(f"iter {m['iter_num']}")
    if m.get("n_mc") is not None:
        bits.append(f"n_mc {m['n_mc']}")
    return "  ·  ".join(bits)


def reportable(arms):
    return [a for a in arms if a.get("auc_reportable") and a.get("auc") is not None]


def headline_arm(res):
    """meta.headline names (horizon, lead); fall back to the longest reportable arm.

    evaluate.py builds meta.headline out of the argparse horizons alone -- blind to whether
    that arm has an AUC at all -- so the named arm is only honoured when it is reportable.
    The H = 1, lead >= 2 arm in the development artifact has 0 positives and auc null.
    """
    arms = res.get("primary", [])
    hl = res.get("meta", {}).get("headline")
    if hl:
        for a in reportable(arms):
            if (a["horizon_years"], a["min_lead_years"]) == (float(hl[0]), float(hl[1])):
                return a
    ok = reportable(arms)
    return max(ok, key=lambda a: (a["horizon_years"], a["min_lead_years"])) if ok else None


def auc_limits(aucs, los, his, lo_default, hi_default, pad=0.03):
    """Room for every dot AND its whisker, with the 0.5 null always inside the frame.

    A fixed floor decapitates a below-chance arm -- the one number a reader most needs to
    see -- so the floor only ever moves down. `pad` is the clearance under the lowest
    whisker: panel A hangs a two-line n annotation there, which would otherwise print on
    the tick labels.
    """
    v = [x for x in list(aucs) + list(los) if np.isfinite(x)]
    w = [x for x in list(aucs) + list(his) if np.isfinite(x)]
    return (min(lo_default, min(v) - pad) if v else lo_default,
            max(hi_default, max(w) + 0.02) if w else hi_default)


def moving_axis_claim(res):
    """A's title states a finding, so it is measured here rather than written down: whether
    the lead filter or the horizon is what actually moves the AUC (3-pt threshold, as in
    caveats())."""
    by_h = {}
    for a in reportable(res.get("primary", [])):
        by_h.setdefault(a["horizon_years"], []).append(a)
    drops = [min(v, key=lambda a: a["min_lead_years"])["auc"]
             - max(v, key=lambda a: a["min_lead_years"])["auc"]
             for v in by_h.values() if len(v) > 1]
    base = [min(v, key=lambda a: a["min_lead_years"])["auc"] for v in by_h.values()]
    lead = max(drops) if drops else float("nan")
    horiz = max(base) - min(base) if len(base) > 1 else float("nan")
    if not (np.isfinite(lead) or np.isfinite(horiz)):
        return "AUC by horizon and by lead-time filter"
    lead0, horiz0 = np.nan_to_num([lead, horiz], nan=0.0)
    if max(lead0, horiz0) < 0.03:
        return "neither the horizon nor the lead-time filter moves the AUC much"
    if lead0 >= horiz0:
        return "the lead-time filter, not the horizon, is what moves the AUC"
    return "the horizon, not the lead-time filter, is what moves the AUC"


def calibration_claim(gap, auc):
    """B's title, likewise measured: the level from the sign of the gap, the ranking from the
    same arm's own AUC."""
    level = ("level not estimable" if not np.isfinite(gap) else
             "level is low" if gap >= 0.03 else
             "level is high" if gap <= -0.03 else "level is close")
    if not np.isfinite(auc):
        return level
    rank = ("ranking is right" if auc >= 0.75 else
            "ranking is modest" if auc >= 0.6 else "ranking is weak")
    return f"{rank}, {level}"


def calibration_arm(res, horizon):
    """Calibration lives at the lowest lead that actually carries bins (see WHY #3)."""
    cands = [a for a in res.get("primary", [])
             if a["horizon_years"] == horizon and (a.get("calibration") or {}).get("bins")]
    return min(cands, key=lambda a: a["min_lead_years"]) if cands else None


# ================================================================== PANEL A  discrimination
def panel_a(res, out_dir, stem):
    arms = res.get("primary", [])
    if not arms:
        print("  skip A: no primary arms in the results file")
        return
    horizons = sorted({a["horizon_years"] for a in arms})
    leads = sorted({a["min_lead_years"] for a in arms})
    hl = headline_arm(res)
    # Three categorical slots, in order, never cycled. Past three horizons a per-horizon hue
    # stops being an identity (two arms would share a swatch), so colour then marks the
    # headline horizon only and the facet titles carry the rest.
    if len(horizons) <= 3:
        cmap = {h: (C1, C2, C3)[j] for j, h in enumerate(horizons)}
        legend = [(f"horizon {h:g} y", cmap[h]) for h in horizons]
    else:
        h_hl = hl["horizon_years"] if hl is not None else max(horizons)
        cmap = {h: (C1 if h == h_hl else GREY) for h in horizons}
        legend = [(f"horizon {h_hl:g} y (headline)", C1), ("every other horizon", GREY)]

    fig, axes = plt.subplots(1, len(horizons), figsize=(3.5 * len(horizons) + 0.8, 4.9),
                             sharey=True)
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    rows = []

    cis = [a.get("auc_ci95") or [None, None] for a in arms]
    lo_lim, hi_lim = auc_limits([num(a.get("auc")) for a in arms], [num(c[0]) for c in cis],
                                [num(c[1]) for c in cis], 0.45, 1.02, pad=0.10)
    # the "no positives" note goes halfway between the null line and the top of the frame:
    # nothing is drawn at that x by definition, and it can never print on the 0.5 reference
    note_y = min(max((0.5 * (1 + (0.5 - lo_lim) / (hi_lim - lo_lim))), 0.25), 0.8)

    for j, h in enumerate(horizons):
        ax, colour = axes[j], cmap[h]
        xs, ys = [], []
        for a in (x for x in arms if x["horizon_years"] == h):
            xi = leads.index(a["min_lead_years"])
            auc, ci = num(a.get("auc")), (a.get("auc_ci95") or [None, None])
            lo, hi = num(ci[0]), num(ci[1])
            n_ev, n_pos = a.get("n_evaluable"), a.get("n_evaluable_positive")
            rows.append({"horizon_years": h, "min_lead_years": a["min_lead_years"],
                         "auc": a.get("auc"), "ci95_lo": ci[0], "ci95_hi": ci[1],
                         "n_evaluable": n_ev, "n_evaluable_positive": n_pos,
                         "n_evaluable_negative": a.get("n_evaluable_negative"),
                         "n_subjects_evaluable": a.get("n_subjects_evaluable"),
                         "n_dropped_competing_or_censored":
                             a.get("n_dropped_competing_or_censored"),
                         "evaluable_prevalence": a.get("evaluable_prevalence"),
                         "auc_reportable": a.get("auc_reportable")})
            if not np.isfinite(auc):
                # axes fraction, not a data y: the scale below is derived from the data, and
                # a fixed y would drift onto the tick labels as soon as an arm sits low
                ax.annotate(f"{no_auc_note(n_pos)}\n{fmt_n(n_ev)} rows, "
                            f"{fmt_n(n_pos)} events",
                            (xi, note_y), xycoords=("data", "axes fraction"), ha="center",
                            va="center", fontsize=8, color=INK2)
                continue
            xs.append(xi)
            ys.append(auc)
            if np.isfinite(lo) and np.isfinite(hi):
                ax.errorbar(xi, auc, yerr=[[auc - lo], [hi - auc]], color=colour, lw=2,
                            capsize=4, capthick=2, zorder=3)
            is_hl = hl is not None and a is hl
            ax.scatter([xi], [auc], s=140 if is_hl else 90, color=colour, zorder=4,
                       edgecolors=SURFACE if is_hl else "none", linewidths=1.4)
            lab = f"{auc:.3f}" + (" ←headline" if is_hl else "")
            ax.annotate(lab, (xi, hi if np.isfinite(hi) else auc), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=8.5, color=INK)
            ax.annotate(f"{fmt_n(n_ev)} rows\n{fmt_n(n_pos)} events",
                        (xi, lo if np.isfinite(lo) else auc), textcoords="offset points",
                        xytext=(0, -12), ha="center", va="top", fontsize=7.5, color=INK2)
        if len(xs) > 1:                       # the drop IS the finding; draw it as a slope
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], "-", color=colour, lw=2,
                    alpha=0.55, zorder=2)
        ax.axhline(0.5, color=GREY, lw=1.2, ls=":", zorder=1)
        axis_style(ax)
        ax.set_xticks(range(len(leads)))
        ax.set_xticklabels([f"lead ≥ {l:g} y" for l in leads], fontsize=8.5, color=INK)
        ax.set_xlim(-0.55, len(leads) - 0.45)
        ax.set_title(f"horizon {h:g} y", fontsize=10.5, color=INK)
    axes[0].set_ylim(lo_lim, hi_lim)
    axes[0].set_ylabel("cause-specific AUC (dot, 95% CI)", fontsize=9.5, color=INK)
    axes[0].annotate("null 0.5", (0.02, 0.5), xycoords=("axes fraction", "data"),
                     textcoords="offset points", xytext=(0, 4), fontsize=7.5, color=INK2)
    handles = [plt.Line2D([], [], color=c, lw=2, marker="o", ms=7, label=lab)
               for lab, c in legend]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.035), labelcolor=INK)

    m = res.get("meta", {})
    sub = [f"{fmt_n(m[k])} {lab}" for k, lab in (("n_eval_subjects", "evaluation subjects"),
                                                 ("n_eval_converters", "converters"),
                                                 ("n_landmarks", "landmarks"))
           if m.get(k) is not None]
    sub.append("deaths inside the horizon are competing events and are dropped from the "
               "denominator, so the arms do not share one")
    fig.suptitle(f"A · Incident-AD discrimination: {moving_axis_claim(res)}",
                 fontsize=11.5, color=INK, y=1.0)
    fig.text(0.5, 0.945, " · ".join(sub), ha="center", fontsize=8.5, color=INK2)
    footer(fig, meta_stamp(res), y=-0.07)
    fig.tight_layout(rect=(0, 0.02, 1, 0.93))
    save(fig, rows, out_dir, stem)


# =================================================================== PANEL B  calibration
def panel_b(res, out_dir, stem):
    hl = headline_arm(res)
    if hl is None:
        print("  skip B: no reportable arm to take a headline horizon from")
        return
    arm = calibration_arm(res, hl["horizon_years"])
    if arm is None:
        print(f"  skip B: no calibration bins at horizon {hl['horizon_years']:g} y")
        return
    cal = arm["calibration"]
    bins = cal["bins"]
    pred = np.array([num(b.get("mean_pred")) for b in bins])
    obs = np.array([num(b.get("aj_observed")) for b in bins])
    nb = np.array([b.get("n") or 0 for b in bins], float)
    at_risk = [b.get("n_at_risk_at_h") for b in bins]

    fig, ax = plt.subplots(figsize=(6.6, 7.0))
    fig.patch.set_facecolor(SURFACE)
    lim = float(np.nanmax([pred.max(), obs.max()])) * 1.12 + 0.02
    ax.plot([0, lim], [0, lim], ls="--", lw=1.4, color=GREY, zorder=1)
    ax.annotate("perfect calibration", (lim * 0.45, lim * 0.45), rotation=45, fontsize=8,
                color=INK2, ha="center", va="top", rotation_mode="anchor")
    # one marker size: evaluate.py cuts the bins on RANKS, so every decile holds the same
    # number of rows and an area encoding would carry no information at all
    ax.scatter(pred, obs, s=130, color=C1, alpha=0.9, zorder=4, edgecolors=SURFACE,
               linewidths=1.2)
    ax.plot(pred, obs, "-", color=C1, lw=2, alpha=0.5, zorder=3)
    # selective labels: the two ends only. The bottom decile sits in a pile of near-zero
    # points, so its label is led out into the empty half of the square.
    for idx, xy, va in ((len(bins) - 1, (14, -4), "top"), (0, (0.42, 0.12), "bottom")):
        b = bins[idx]
        text = (f"decile {b.get('bin')}\n{fmt_pct(pred[idx])} pred → {fmt_pct(obs[idx])} obs\n"
                f"{fmt_n(b.get('n_ad_within_h'))} AD, {fmt_n(b.get('n_death_within_h'))} "
                f"deaths in {fmt_n(b.get('n'))} rows")
        if idx == 0:
            ax.annotate(text, (pred[idx], obs[idx]), xytext=xy, textcoords="axes fraction",
                        fontsize=8, color=INK, ha="left", va=va,
                        arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8,
                                        shrinkA=2, shrinkB=6))
        else:
            ax.annotate(text, (pred[idx], obs[idx]), textcoords="offset points", xytext=xy,
                        fontsize=8, color=INK, ha="left", va=va)
    axis_style(ax, yaxis_grid=False)
    ax.set_xlim(-0.01, lim)
    ax.set_ylim(-0.01, lim)
    ax.set_aspect("equal")            # a 45-degree line has to be 45 degrees on the page
    ax.set_xlabel("mean predicted risk in the decile", fontsize=9.5, color=INK)
    ax.set_ylabel("Aalen-Johansen observed risk (death as competing event)", fontsize=9.5,
                  color=INK)

    gap = num(cal.get("aj_observed_all_rows")) - num(cal.get("mean_predicted_all_rows"))
    lines = [f"mean predicted {fmt_pct(cal.get('mean_predicted_all_rows'))}   vs   "
             f"AJ observed {fmt_pct(cal.get('aj_observed_all_rows'))}",
             f"gap {gap * 100:+.1f} pts  ({'under' if gap > 0 else 'over'}-predicts risk)"
             if np.isfinite(gap) else
             "gap not estimable (nobody at risk at the horizon)"]
    if cal.get("mean_abs_calibration_error") is not None:
        lines.append(f"mean |error| across deciles {fmt_pct(cal['mean_abs_calibration_error'])}")
    # slope and Brier need a 0/1 label, so evaluate.py computes them complete-case on the
    # binary-evaluable rows -- conditioned on surviving the horizon, which is the very
    # conditioning the AJ decile table exists to avoid. Different population, so its own n.
    fit = []
    if np.isfinite(num(cal.get("cal_slope"))):
        fit.append(f"slope {cal['cal_slope']:.2f}")
    if np.isfinite(num(cal.get("brier"))) and np.isfinite(num(cal.get("brier_null"))):
        fit.append(f"Brier {cal['brier']:.4f} vs null {cal['brier_null']:.4f}")
    if fit:
        lines.append(", ".join(fit) + f"  (complete-case, {fmt_n(arm.get('n_evaluable'))} "
                     "binary-evaluable rows, deaths inside the horizon dropped)")
    spread = (f"of {fmt_n(nb[0])} rows" if len(set(nb)) == 1 else
              f"of {fmt_n(nb.min())}–{fmt_n(nb.max())} rows")
    lines.append(f"decile table on all {fmt_n(cal.get('n_rows'))} landmark rows, {len(bins)} "
                 f"rank-cut bins {spread} (equal-n by construction)")
    ar = [a for a in at_risk if a is not None]
    if ar:
        lines.append(f"at risk at the horizon: {fmt_n(min(ar))}–{fmt_n(max(ar))} per decile")
    blank = [b.get("bin") for b, o in zip(bins, obs) if not np.isfinite(o)]
    if blank:
        lines.append("no observed point for decile "
                     + ", ".join(str(b) for b in blank)
                     + ": nobody at risk at the horizon")
    # the box lives OUTSIDE the square: on a badly calibrated run the deciles climb into
    # whichever corner it would otherwise sit in, and furniture must never cover data
    fig.text(0.012, 0.245, "\n".join(lines), fontsize=8.5, color=INK, va="top", ha="left",
             linespacing=1.5,
             bbox=dict(boxstyle="round,pad=0.5", facecolor=SURFACE, edgecolor=GRIDC))

    ax.set_title(f"B · Calibration at {arm['horizon_years']:g} y, lead ≥ "
                 f"{arm['min_lead_years']:g} y: "
                 f"{calibration_claim(gap, num(arm.get('auc')))}",
                 fontsize=11.5, color=INK, pad=14)
    footer(fig, meta_stamp(res) + "  ·  the lead filter is selection on the outcome, so "
           "absolute risk is reported at the lowest lead that has bins")
    fig.subplots_adjust(left=0.12, right=0.99, top=0.93, bottom=0.31)
    rows = [{"bin": b.get("bin"), "n_rows": b.get("n"), "mean_pred": b.get("mean_pred"),
             "aj_observed": b.get("aj_observed"), "n_ad_within_h": b.get("n_ad_within_h"),
             "n_death_within_h": b.get("n_death_within_h"),
             "n_at_risk_at_h": b.get("n_at_risk_at_h"),
             "horizon_years": arm["horizon_years"], "min_lead_years": arm["min_lead_years"],
             "mean_predicted_all_rows": cal.get("mean_predicted_all_rows"),
             "aj_observed_all_rows": cal.get("aj_observed_all_rows"),
             "mean_abs_calibration_error": cal.get("mean_abs_calibration_error"),
             "cal_slope": cal.get("cal_slope"), "cal_intercept": cal.get("cal_intercept"),
             "brier": cal.get("brier"), "brier_null": cal.get("brier_null")} for b in bins]
    save(fig, rows, out_dir, stem)


# ============================================================== PANEL C  cumulative incidence
def _age_panel(ax, ages, series, title, ylab):
    """One age-axis panel. series = (value_key, label, colour, linestyle)."""
    xs = np.arange(len(ages))
    drawn, ends = [], []
    for vkey, label, colour, ls in series:
        ys = np.array([num(ages[a].get(vkey)) for a in ages])
        if not np.isfinite(ys).any():
            continue
        ax.plot(xs, ys, ls, color=colour, lw=2, zorder=3)
        ax.scatter(xs, ys, s=90, color=colour, zorder=4, edgecolors=SURFACE, linewidths=1.2)
        ends.append((xs[-1], ys[-1], label))
        drawn.append((label, colour, ls))
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{float(a):g}" for a in ages], fontsize=9, color=INK)
    ax.set_xlim(-0.35, len(ages) - 1 + 1.25)
    ax.set_ylim(bottom=0)
    axis_style(ax)
    direct_labels(ax, ends)
    ax.set_title(title, fontsize=10.5, color=INK)
    ax.set_xlabel("age (years)", fontsize=9.5, color=INK)
    ax.set_ylabel(ylab, fontsize=9.5, color=INK)
    return drawn


def panel_c(res, out_dir, stem):
    sec = res.get("secondary", {})
    cif, mort = sec.get("cif_calibration", {}), sec.get("mortality", {})
    ad_ages = cif.get("ages", {})
    if not ad_ages:
        print("  skip C: no cif_calibration.ages block")
        return
    mo_ages = mort.get("ages", {})
    ad_ages = dict(sorted(ad_ages.items(), key=lambda kv: float(kv[0])))
    mo_ages = dict(sorted(mo_ages.items(), key=lambda kv: float(kv[0])))
    origin = cif.get("origin_age")

    ncol = 2 if mo_ages else 1
    fig, axes = plt.subplots(1, ncol, figsize=(6.3 * ncol, 5.2))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)

    # evaluate.py enters the grey arm at each subject's BASELINE age and the orange arm at
    # the origin, so the two are not the same estimand and must not be labelled as though the
    # only difference were the population.
    ad_series = [("simulated_cif", "simulated", SIM, "-"),
                 ("observed_aj_landmark80", "observed, same subjects", OBS, "-"),
                 ("observed_aj_split_eval", "whole eval split, origin = baseline age",
                  GREY, "--")]
    drawn = _age_panel(axes[0], ad_ages, ad_series,
                       f"AD cumulative incidence from age-{origin:g} prefixes"
                       if origin is not None else "AD cumulative incidence",
                       "cumulative incidence of AD (Aalen-Johansen)")
    _risk_row(axes[0], ad_ages, [("risk_set_landmark80", "same subjects"),
                                 ("risk_set_split_eval", "whole eval split")])

    if mo_ages:
        mo_series = [("simulated_death_cif", "simulated", SIM, "-"),
                     ("observed_death_cif_landmark80", "observed, same subjects", OBS, "-"),
                     ("observed_death_cif_split_eval",
                      "whole eval split, origin = baseline age", GREY, "--")]
        _age_panel(axes[1], mo_ages, mo_series,
                   "all-cause mortality (a second measure, so a second panel)",
                   "cumulative incidence of death")
        _risk_row(axes[1], mo_ages, [("risk_set_landmark80", "same subjects"),
                                     ("risk_set_split_eval", "whole eval split")])
        never = (sec.get("mortality", {}).get("simulated_death_age", {})
                 .get("frac_draws_never_died_in_budget"))
        if never is not None and np.isfinite(num(never)):
            axes[1].annotate(f"{fmt_pct(never)} of rollout draws never emit Death inside the\n"
                             "token budget, which is what flattens the simulated arm",
                             (0.02, 0.22), xycoords="axes fraction", fontsize=8.5, color=INK,
                             va="top", bbox=dict(boxstyle="round,pad=0.45", facecolor=SURFACE,
                                                 edgecolor=GRIDC))

    handles = [plt.Line2D([], [], color=c, lw=2, ls=ls, marker="o", ms=7, label=lab)
               for lab, c, ls in drawn]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    sub = [f"{fmt_n(cif[k])} {lab}" for k, lab in
           (("n_subjects_age80", "subjects alive and AD-free at the origin"),
            ("n_draws", "rollout draws")) if cif.get(k) is not None]
    oldest = max((float(a) for a in ad_ages), default=None)
    sub.append(f"the age-{oldest:g} estimate rests on the risk set printed under the tick, "
               "not on the cohort" if oldest is not None else "risk sets are under the ticks")
    note = ("the grey arm is a DIFFERENT TIME ORIGIN, not just a wider population: it enters "
            "each subject at their baseline age, so a diagnosis before age "
            f"{origin:g} counts in it and cannot count in the age-{origin:g} arms"
            if origin is not None else
            "the grey arm enters each subject at their baseline age, a different time origin")
    fig.suptitle("C · Simulated vs observed cumulative incidence, left-truncated at age "
                 f"{origin:g}" if origin is not None else "C · Cumulative incidence",
                 fontsize=11.5, color=INK, y=1.0)
    fig.text(0.5, 0.965, " · ".join(sub) + "\n" + note, ha="center", va="top", fontsize=8.5,
             color=INK2)
    footer(fig, meta_stamp(res), y=-0.07)
    fig.tight_layout(rect=(0, 0.06, 1, 0.89))

    rows = []
    for a in ad_ages:
        r = {"age": float(a), "measure": "AD"}
        r.update({k: v for k, v in ad_ages[a].items()})
        rows.append(r)
    for a in mo_ages:
        r = {"age": float(a), "measure": "death"}
        r.update({k: v for k, v in mo_ages[a].items()})
        rows.append(r)
    save(fig, rows, out_dir, stem)


def _risk_row(ax, ages, keys):
    """Risk-set sizes under the tick -- the number that says how much the point can carry."""
    have = [(k, lab) for k, lab in keys if any(ages[a].get(k) is not None for a in ages)]
    if not have:
        return
    for i, a in enumerate(ages):
        txt = "\n".join(fmt_n(ages[a].get(k)) for k, _ in have)
        ax.annotate(txt, (i, 0), xycoords=("data", "axes fraction"),
                    textcoords="offset points", xytext=(0, -30), ha="center", va="top",
                    fontsize=7.5, color=INK2)
    ax.annotate("\n".join(f"at risk, {lab}" for _, lab in have), (0, 0),
                xycoords=("axes fraction", "axes fraction"), textcoords="offset points",
                xytext=(-12, -30), ha="right", va="top", fontsize=7.5, color=INK2)


# =============================================================== PANEL D  trajectory realism
def panel_d(res, out_dir, stem):
    traj = res.get("secondary", {}).get("trajectory", {})
    scales = traj.get("scales", {})
    if not scales:
        print("  skip D: no secondary.trajectory.scales block")
        return
    names = list(scales)
    fig, axes = plt.subplots(1, len(names), figsize=(6.0 * len(names), 5.0))
    axes = np.atleast_1d(axes)
    fig.patch.set_facecolor(SURFACE)
    rows, drawn = [], []

    for ax, sc in zip(axes, names):
        d = scales[sc]
        curve, ends = d.get("curve", []), []
        yrs = np.array([c["year"] for c in curve], float)
        series = [("sim_mean_bin", "simulated", SIM, "-"),
                  ("obs_mean_bin", "observed", OBS, "-"),
                  ("sim_mean_bin_matched", "simulated, case-mix matched", MATCHED, "-")]
        for key, label, colour, ls in series:
            ys = np.array([num(c.get(key)) for c in curve])
            if not np.isfinite(ys).any():
                continue
            ax.plot(yrs, ys, ls, color=colour, lw=2, zorder=3)
            ax.scatter(yrs[::3], ys[::3], s=70, color=colour, zorder=4, edgecolors=SURFACE,
                       linewidths=1.0)
            k = int(np.max(np.nonzero(np.isfinite(ys))))
            ends.append((yrs[k], ys[k], label))
            if (label, colour) not in [(l, c) for l, c, _ in drawn]:
                drawn.append((label, colour, ls))
        for c in curve:
            rows.append({"scale": sc, "n_bins": d.get("n_bins"), **c})
        nb = d.get("n_bins")
        axis_style(ax)
        ax.set_xlim(yrs.min() - 0.4, yrs.max() + 4.2)
        direct_labels(ax, ends, dx=7)
        ax.set_xticks(yrs[::1] if len(yrs) <= 12 else yrs[::2])
        ax.set_xlabel("years from baseline", fontsize=9.5, color=INK)
        ax.set_ylabel(f"mean bin index  (0 = worst of {nb})" if nb else "mean bin index",
                      fontsize=9.5, color=INK)
        title = f"{sc}"
        rt_s, rt_o = d.get("round_trip_rate_simulated"), d.get("round_trip_rate_observed")
        if rt_s is not None and rt_o is not None:
            title += f"   ·   round-trips {fmt_pct(rt_s)} sim vs {fmt_pct(rt_o)} obs"
        draws = [c.get("sim_n_draws") for c in curve if c.get("sim_n_draws") is not None]
        if draws:
            title += f"\nsimulated draws {fmt_n(draws[0])} → {fmt_n(draws[-1])} per year"
        ax.set_title(title, fontsize=10.5, color=INK)
        # observed n per year under the tick: the late years are a handful of survivors.
        # The last year always gets one -- it is where the three direct labels sit, so it is
        # the point the room reads, and it must not be the one without an n. -46 clears the
        # xlabel, which lives under the ticks too.
        step = 1 if len(yrs) <= 12 else 2
        idx = list(range(0, len(curve), step))
        if idx[-1] != len(curve) - 1:
            tail = len(curve) - 1
            idx[-1:] = [tail] if tail - idx[-1] < step else [idx[-1], tail]
        for c in (curve[i] for i in idx):
            ax.annotate(fmt_n(c.get("obs_n_subjects")), (c["year"], 0),
                        xycoords=("data", "axes fraction"), textcoords="offset points",
                        xytext=(0, -46), ha="center", va="top", fontsize=7, color=INK2)
        ax.annotate("observed n", (0, 0), xycoords="axes fraction",
                    textcoords="offset points", xytext=(-10, -46), ha="right", va="top",
                    fontsize=7, color=INK2)

    handles = [plt.Line2D([], [], color=c, lw=2, ls=ls, marker="o", ms=7, label=lab)
               for lab, c, ls in drawn]
    fig.legend(handles=handles, frameon=False, fontsize=8, ncol=len(handles),
               loc="lower center", bbox_to_anchor=(0.5, -0.02), labelcolor=INK)
    fig.suptitle("D · Trajectory realism: simulated vs observed cognition, "
                 "with the case mix held fixed", fontsize=11.5, color=INK, y=1.0)
    fig.text(0.5, 0.945, f"{fmt_n(traj.get('n_subjects'))} rolled subjects · the matched arm "
             "averages the simulated draws only where that subject is still observed, so it "
             "separates model drift from who is left in the cohort", ha="center", fontsize=8.5,
             color=INK2)
    footer(fig, meta_stamp(res), y=-0.07)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    save(fig, rows, out_dir, stem)


# ======================================================================= SUMMARY stat row
def caveats(res):
    """Ranked caveats, all derived from the file. First rule that fires is the headline one.

    The order is a claim about which failure would most change how a reader should use the
    numbers: a cohort that never dies invalidates every absolute simulated risk; a level
    error invalidates absolute risk but not triage; a lead-time collapse invalidates the
    lead-time claim only; a thin risk set invalidates one endpoint of one curve.
    """
    out = []
    sec = res.get("secondary", {})
    mo = sec.get("mortality", {}).get("ages", {})
    best = None
    for a, d in mo.items():
        gap = num(d.get("observed_death_cif_landmark80")) - num(d.get("simulated_death_cif"))
        if np.isfinite(gap) and (best is None or gap > best[0]):
            best = (gap, float(a), d)
    if best and best[0] >= 0.10:
        g, age, d = best
        out.append(("mortality", f"The rollouts barely die: simulated all-cause incidence "
                    f"{fmt_pct(d.get('simulated_death_cif'))} against "
                    f"{fmt_pct(d.get('observed_death_cif_landmark80'))} observed at age "
                    f"{age:g} ({g * 100:.0f} pts). Every simulated curve here is therefore "
                    "AD risk among an implausibly long-lived cohort, not absolute risk."))
    hl = headline_arm(res)
    if hl is not None:
        arm = calibration_arm(res, hl["horizon_years"])
        cal = (arm or {}).get("calibration", {})
        gap = num(cal.get("aj_observed_all_rows")) - num(cal.get("mean_predicted_all_rows"))
        if np.isfinite(gap) and abs(gap) >= 0.03:
            out.append(("calibration", f"Risk level is off: at {arm['horizon_years']:g} y the "
                        f"model predicts {fmt_pct(cal.get('mean_predicted_all_rows'))} where "
                        f"{fmt_pct(cal.get('aj_observed_all_rows'))} is observed "
                        f"({gap * 100:+.1f} pts), so ranking is usable and absolute risk is "
                        "not."))
        same_h = [a for a in reportable(res.get("primary", []))
                  if a["horizon_years"] == hl["horizon_years"]]
        if len(same_h) > 1:
            lo = min(same_h, key=lambda a: a["min_lead_years"])
            hi = max(same_h, key=lambda a: a["min_lead_years"])
            drop = num(lo.get("auc")) - num(hi.get("auc"))
            if np.isfinite(drop) and drop >= 0.03:
                out.append(("lead time", f"Most of the apparent skill is short-lead: at "
                            f"{hl['horizon_years']:g} y the AUC falls {lo['auc']:.3f} → "
                            f"{hi['auc']:.3f} ({drop:.3f}) once landmarks must sit ≥ "
                            f"{hi['min_lead_years']:g} y clear of the diagnosis."))
    cif = sec.get("cif_calibration", {}).get("ages", {})
    thin = [(float(a), d.get("risk_set_landmark80")) for a, d in cif.items()
            if d.get("risk_set_landmark80") is not None]
    if thin:
        age, n = min(thin, key=lambda t: t[1])
        out.append(("risk set", f"The oldest CIF point is thin: {fmt_n(n)} subjects still at "
                    f"risk at age {age:g}."))
    return out


def panel_summary(res, out_dir, stem):
    hl = headline_arm(res)
    if hl is None:
        print("  skip summary: no reportable arm")
        return
    m = res.get("meta", {})
    arm = calibration_arm(res, hl["horizon_years"])
    cal = (arm or {}).get("calibration", {})
    gap = num(cal.get("aj_observed_all_rows")) - num(cal.get("mean_predicted_all_rows"))
    cav = caveats(res)

    fig = plt.figure(figsize=(12.4, 5.0))
    fig.patch.set_facecolor(SURFACE)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.05, 0.85, 0.75], hspace=0.55, wspace=0.18)

    def tile(col, big, label, sub, colour=INK):
        ax = fig.add_subplot(gs[0, col])
        ax.set_axis_off()
        ax.text(0, 0.98, label.upper(), fontsize=8.5, color=INK2, va="top",
                transform=ax.transAxes)
        ax.text(0, 0.66, big, fontsize=27, color=colour, va="top", transform=ax.transAxes)
        ax.text(0, 0.24, wrap(sub, 44), fontsize=8.5, color=INK2, va="top",
                transform=ax.transAxes, linespacing=1.6)
        return ax

    ci = hl.get("auc_ci95") or [None, None]
    tile(0, fmt3(num(hl.get("auc"))), f"AUC · incident AD, {hl['horizon_years']:g}-y "
         f"horizon, lead ≥ {hl['min_lead_years']:g} y",
         f"95% CI {fmt3(num(ci[0]))}–{fmt3(num(ci[1]))}\n"
         f"{fmt_n(hl.get('n_evaluable'))} evaluable landmark rows · "
         f"{fmt_n(hl.get('n_evaluable_positive'))} events\n"
         f"{fmt_n(hl.get('n_subjects_evaluable'))} subjects · "
         f"{fmt_n(hl.get('n_dropped_competing_or_censored'))} rows dropped as competing "
         "deaths or censored")
    tile(1, f"{gap * 100:+.1f} pts" if np.isfinite(gap) else "n/a",
         "calibration gap · observed minus predicted",
         (f"observed {fmt_pct(cal.get('aj_observed_all_rows'))} vs predicted "
          f"{fmt_pct(cal.get('mean_predicted_all_rows'))}\nat "
          f"{arm['horizon_years']:g} y, lead ≥ {arm['min_lead_years']:g} y "
          f"({fmt_n(cal.get('n_rows'))} rows)\n"
          f"mean |error| across deciles {fmt_pct(cal.get('mean_abs_calibration_error'))}"
          if cal else "no calibration block in this results file"))
    # cohort.py labels a row positive when the event is inside the horizon, so the horizons
    # NEST on one landmark pool: the H = 1 positives are a subset of the H = 5 ones and the
    # arms must not be added up. Report the largest arm and say that it contains the others.
    arms0 = [a for a in res.get("primary", [])
             if a["min_lead_years"] == min(x["min_lead_years"] for x in res["primary"])]
    top = max(arms0, key=lambda a: (a.get("n_evaluable_positive") or 0, a["horizon_years"]),
              default=None) if arms0 else None
    tile(2, fmt_n(m.get("n_eval_subjects")), "evaluation cohort",
         f"{fmt_n(m.get('n_eval_converters'))} converters · "
         f"{fmt_n(m.get('n_landmarks'))} landmarks scored\n"
         f"{fmt_n(m.get('n_split_subjects'))} subjects in the {m.get('split', '?')} split\n"
         + (f"{fmt_n(top.get('n_evaluable_positive'))} positive rows at "
            f"{top['horizon_years']:g} y, lead ≥ {top['min_lead_years']:g} y\n"
            "(horizons nest; the arms do not add)" if top is not None else ""))

    # the AUC lives against its null, never on a zero-based bar
    axc = fig.add_subplot(gs[1, :])
    axc.axvline(0.5, color=GREY, lw=1.2, ls=":", zorder=1)
    shown = reportable(res.get("primary", []))
    ys = np.arange(len(shown))
    for y, a in zip(ys, shown):
        c = a.get("auc_ci95") or [None, None]
        lo, hi_, auc = num(c[0]), num(c[1]), num(a["auc"])
        is_hl = a is hl
        colour = C1 if is_hl else GREY
        if np.isfinite(lo) and np.isfinite(hi_):
            axc.plot([lo, hi_], [y, y], color=colour, lw=2, zorder=3)
        axc.scatter([auc], [y], s=150 if is_hl else 80, color=colour, zorder=4,
                    edgecolors=SURFACE, linewidths=1.2)
        axc.annotate(f"{auc:.3f}   ({fmt_n(a.get('n_evaluable'))} rows, "
                     f"{fmt_n(a.get('n_evaluable_positive'))} events)"
                     + ("   ← headline" if is_hl else ""),
                     (hi_ if np.isfinite(hi_) else auc, y), textcoords="offset points",
                     xytext=(10, 0), va="center", fontsize=8.5, color=INK)
    axc.set_ylim(-0.8, len(shown) - 0.2)
    axc.set_yticks(ys)
    axc.set_yticklabels([f"{a['horizon_years']:g} y, lead ≥ {a['min_lead_years']:g} y"
                         for a in shown], fontsize=8.5, color=INK)
    cis = [a.get("auc_ci95") or [None, None] for a in shown]
    lo_lim, hi_lim = auc_limits([num(a.get("auc")) for a in shown],
                                [num(c[0]) for c in cis], [num(c[1]) for c in cis],
                                0.47, 1.02)
    axc.set_xlim(lo_lim, hi_lim)
    ticks = np.round(np.arange(np.floor(lo_lim * 10) / 10, hi_lim + 1e-9, 0.1), 2)
    axc.set_xticks([t for t in ticks if lo_lim <= t <= hi_lim])
    axis_style(axc, yaxis_grid=False)
    axc.grid(axis="x", color=GRIDC, lw=0.7, zorder=0)
    axc.spines["left"].set_visible(False)
    axc.set_xlabel("AUC (dot, 95% CI) against the 0.5 null", fontsize=9, color=INK)

    axv = fig.add_subplot(gs[2, :])
    axv.set_axis_off()
    if cav:
        head, text = cav[0]
        axv.text(0.005, 0.9, f"READ THIS FIRST — {head}", fontsize=9, color=INK, va="top",
                 transform=axv.transAxes)
        axv.text(0.005, 0.62, wrap(text, 118), fontsize=9.5, color=INK, va="top",
                 transform=axv.transAxes, linespacing=1.5)
        if len(cav) > 1:
            axv.text(0.005, 0.02, "also: " + "  ·  ".join(h for h, _ in cav[1:]), fontsize=8,
                     color=INK2, va="bottom", transform=axv.transAxes)
        axv.add_patch(plt.Rectangle((0, -0.05), 1, 1.05, transform=axv.transAxes,
                                    facecolor="#f2f1ed", edgecolor=GRIDC, zorder=0))
    fig.suptitle("RADC / ROSMAP event-token transformer — endpoint summary", fontsize=13,
                 color=INK, x=0.005, ha="left", y=0.99)
    footer(fig, meta_stamp(res))
    # gridspec + text-only tiles: tight_layout cannot size these, and bbox_inches="tight"
    # in save() trims the margins anyway
    fig.subplots_adjust(left=0.10, right=0.99, top=0.88, bottom=0.06)

    rows = [{"stat": "headline_auc", "value": hl.get("auc"), "ci95_lo": ci[0],
             "ci95_hi": ci[1], "horizon_years": hl["horizon_years"],
             "min_lead_years": hl["min_lead_years"], "n_evaluable": hl.get("n_evaluable"),
             "n_evaluable_positive": hl.get("n_evaluable_positive")},
            {"stat": "calibration_gap", "value": gap if np.isfinite(gap) else None,
             "horizon_years": (arm or {}).get("horizon_years"),
             "min_lead_years": (arm or {}).get("min_lead_years"),
             "n_evaluable": cal.get("n_rows")},
            {"stat": "n_eval_subjects", "value": m.get("n_eval_subjects")},
            {"stat": "n_eval_converters", "value": m.get("n_eval_converters")}]
    rows += [{"stat": f"caveat_{i + 1}_{h.replace(' ', '_')}", "note": t}
             for i, (h, t) in enumerate(cav)]
    save(fig, rows, out_dir, stem)


# ==================================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--results", default=os.path.join(_ROOT, "eval", "results_test.json"),
                    help="evaluate.py output to plot")
    ap.add_argument("--out", default=os.path.join(_ROOT, "results", "figures"))
    ap.add_argument("--prefix", default="fig_endpoints")
    args = ap.parse_args()

    res = json.load(open(args.results))
    tag = res.get("meta", {}).get("split") or "split"
    print(f"{args.results} -> {args.out}   ({meta_stamp(res)})")
    panel_summary(res, args.out, f"{args.prefix}_summary_{tag}")
    panel_a(res, args.out, f"{args.prefix}_A_discrimination_{tag}")
    panel_b(res, args.out, f"{args.prefix}_B_calibration_{tag}")
    panel_c(res, args.out, f"{args.prefix}_C_incidence_{tag}")
    panel_d(res, args.out, f"{args.prefix}_D_trajectory_{tag}")
    for head, text in caveats(res):
        print(f"  caveat [{head}] {text}")


if __name__ == "__main__":
    main()
