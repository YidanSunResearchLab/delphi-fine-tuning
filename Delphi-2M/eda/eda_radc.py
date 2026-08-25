"""
eda_radc.py -- exploratory analysis of the RADC/ROSMAP cohort, aimed at ONE question:
what does this dataset let a Delphi-style trajectory model learn, and where will it mislead?

Six panels, each with its numbers written next to it as CSV (figure2/plotting_style.py's
save_data, the repo's "source data" convention -- and the relief the palette validator
requires for the two low-contrast hues).

  A  observation window     baseline vs last-observed age -- how much of life is covered
  B  the signal             global cognition vs age, split by eventual AD diagnosis
  C  state occupancy        MMSE severity mix across age (ordinal -> sequential ramp)
  D  risk stratification    cumulative AD incidence by age, split by APOE e4
  E  INFORMATIVE MISSINGNESS  BMI measurement rate by cognitive state -- the trap
  F  sampling cadence       inter-event interval, i.e. is fu_year really annual

Run:  python eda/eda_radc.py
Out:  out_radc/eda/{panel_*.png,.pdf,_data.csv}   (out_radc/ is gitignored)
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Delphi-2M/
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from figure2.plotting_style import (  # noqa: E402
    setup_style, save_fig, save_data, SEVERITY_COLORS, REF_GREY, _OKABE,
)
import matplotlib.pyplot as plt  # noqa: E402

RADC = os.path.join(ROOT, "data", "RADC")
# NOT under out_radc/ -- that directory is the tokenizer+split output and gets wiped on a
# dataset rebuild (rm -rf out_radc), which silently took these figures with it once. out-*/ is
# the repo's slot for "checkpoints + eval figures" (see .gitignore) and survives a rebuild.
OUT = os.path.join(HERE, "out-radc-eda")

# Categorical hues, assigned in FIXED order and never cycled. Validated as a set by the
# dataviz validator: lightness band / chroma floor / CVD separation / normal-vision floor all
# PASS; #E69F00 and #CC79A7 WARN on contrast-vs-surface, which the per-panel CSV + always-on
# legend + direct labels discharge.
C1, C2, C3 = _OKABE["blue"], _OKABE["vermillion"], _OKABE["orange"]

MMSE_BINS = [-1, 17, 23, 26, 30]
MMSE_NAMES = ["0-17 severe", "18-23 mild-mod", "24-26 borderline", "27-30 normal"]


def load():
    cs = pd.read_excel(os.path.join(RADC, "cross-sectional-data-gk.xlsx"))
    lo = pd.read_excel(os.path.join(RADC, "longitudinal_data_gk.xlsx"))
    cs["study"] = cs["study"].astype(str).str.strip()
    J = lo.merge(cs[["projid", "age_bl", "study", "msex", "educ", "apoe_genotype",
                     "age_first_ad_dx"]], on="projid", how="left")
    J["age"] = J["age_bl"] + J["fu_year"]          # validated annual grid; see panel F
    J["has_ad"] = J["age_first_ad_dx"].notna()
    cs["has_ad"] = cs["age_first_ad_dx"].notna()
    cs["e4"] = cs["apoe_genotype"].isin([24, 34, 44])
    return cs, J


# ---------------------------------------------------------------- A: observation window
def panel_a(cs, J):
    last = J.groupby("projid")["age"].max()
    fig, axes = plt.subplots(1, 1, figsize=(6.2, 4.2))
    ax = axes
    bins = np.arange(35, 112, 2)
    ax.hist(cs["age_bl"], bins=bins, histtype="step", lw=2, color=C1, label="Baseline age")
    ax.hist(last, bins=bins, histtype="step", lw=2, color=C2, label="Last observed age")
    ax.axvline(cs["age_bl"].median(), color=C1, ls=":", lw=1.5)
    ax.axvline(last.median(), color=C2, ls=":", lw=1.5)
    ax.annotate(f"median {cs['age_bl'].median():.0f}", (cs["age_bl"].median(), ax.get_ylim()[1] * .95),
                color=C1, ha="right", va="top", fontsize=8.5)
    ax.annotate(f"median {last.median():.0f}", (last.median(), ax.get_ylim()[1] * .95),
                color=C2, ha="left", va="top", fontsize=8.5)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Subjects")
    pre65 = 100 * (cs["age_bl"] < 65).mean()
    ax.set_title(f"A · Observation window (n={len(cs)})\n"
                 f"only {pre65:.0f}% enrol before 65 — mid-life is essentially unobserved")
    ax.legend(frameon=False, loc="upper left")
    save_data(pd.DataFrame({"age_bl": cs["age_bl"], "last_age": last.reindex(cs.projid).values}),
              OUT, "panel_a_window")
    return save_fig(fig, OUT, "panel_a_window")


# ---------------------------------------------------------------- B: the signal
def panel_b(J):
    d = J.dropna(subset=["cogn_global"]).copy()
    d["ab"] = pd.cut(d["age"], np.arange(65, 101, 2.5))
    rows = []
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for flag, col, name in [(False, C1, "no AD diagnosis"), (True, C2, "AD diagnosed (ever)")]:
        s = d[d.has_ad == flag].groupby("ab", observed=True)["cogn_global"].agg(["mean", "sem", "size"])
        s = s[s["size"] >= 30]
        x = np.array([iv.mid for iv in s.index])
        ax.plot(x, s["mean"], lw=2, color=col, label=name)
        ax.fill_between(x, s["mean"] - 1.96 * s["sem"], s["mean"] + 1.96 * s["sem"],
                        color=col, alpha=.18, lw=0)
        ax.annotate(name, (x[-3], s["mean"].iloc[-3]), color=col, fontsize=8.5,
                    ha="right", va="top" if flag else "bottom",
                    xytext=(-4, -10 if flag else 10), textcoords="offset points")
        rows.append(s.assign(group=name, age=x))
    ax.axhline(0, color=REF_GREY, lw=1, ls="--")
    ax.set_xlabel("Age (years)"); ax.set_ylabel("Global cognition (z, baseline-referenced)")
    ax.set_title("B · The cognitive signal is present decades out\n"
                 "grouped by EVENTUAL diagnosis — not a prediction curve")
    ax.legend(frameon=False, loc="lower left")
    save_data(pd.concat(rows), OUT, "panel_b_signal")
    return save_fig(fig, OUT, "panel_b_signal")


# ---------------------------------------------------------------- C: state occupancy
def panel_c(J):
    d = J.dropna(subset=["cts_estmmse30"]).copy()
    d["bin"] = pd.cut(d["cts_estmmse30"], MMSE_BINS, labels=MMSE_NAMES)
    d["ab"] = pd.cut(d["age"], np.arange(65, 101, 2.5))
    ct = pd.crosstab(d["ab"], d["bin"], normalize="index") * 100
    n = d.groupby("ab", observed=True).size()
    ct = ct[n.reindex(ct.index).fillna(0) >= 30]
    x = np.array([iv.mid for iv in ct.index])
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    # ordinal severity -> sequential ramp (NOT categorical hues). 1.5pt surface-coloured
    # edges give the 2px gap between stacked segments the mark spec asks for.
    # normal -> severe must run light-blue -> red. MMSE_NAMES is severe-first, so reverse the
    # NAMES and leave the ramp forward; reversing both (an earlier bug) painted "normal" red.
    order = MMSE_NAMES[::-1]                      # normal, borderline, mild-mod, severe
    ax.stackplot(x, *[ct[c].values for c in order], labels=order,
                 colors=SEVERITY_COLORS[:4], edgecolor="white", lw=1.5)
    ax.set_xlim(x.min(), x.max()); ax.set_ylim(0, 100)
    ax.set_xlabel("Age (years)"); ax.set_ylabel("% of visits at that age")
    ax.set_title("C · MMSE severity mix shifts with age\ncross-sectional at each age, not a cohort")
    ax.legend(frameon=False, loc="lower left", ncol=2, fontsize=7.5)
    save_data(ct.assign(age=x), OUT, "panel_c_occupancy")
    return save_fig(fig, OUT, "panel_c_occupancy")


# ---------------------------------------------------------------- D: risk stratification
def panel_d(cs):
    d = cs.dropna(subset=["apoe_genotype"]).copy()
    ages = np.arange(70, 101)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rows = {"age": ages}
    for flag, col, name in [(False, C1, "APOE e4 non-carrier"), (True, C2, "APOE e4 carrier")]:
        g = d[d.e4 == flag]
        # cumulative % of the stratum diagnosed by each age (denominator = whole stratum;
        # this is a crude cumulative incidence, NOT a competing-risk-adjusted one)
        y = [100 * (g["age_first_ad_dx"] <= a).mean() for a in ages]
        ax.plot(ages, y, lw=2, color=col, label=f"{name} (n={len(g)})")
        ax.annotate(f"{y[-1]:.0f}%", (ages[-1], y[-1]), color=col, fontsize=9,
                    ha="left", va="center")
        rows[name] = y
    ax.set_xlabel("Age (years)"); ax.set_ylabel("% diagnosed with AD dementia by this age")
    ax.set_title("D · APOE e4 stratifies both rate and timing\ncrude cumulative incidence")
    ax.legend(frameon=False, loc="upper left")
    save_data(pd.DataFrame(rows), OUT, "panel_d_apoe")
    return save_fig(fig, OUT, "panel_d_apoe")


# ---------------------------------------------------------------- E: the trap
def panel_e(J):
    d = J.dropna(subset=["cts_estmmse30"]).copy()
    d["mm"] = pd.cut(d["cts_estmmse30"], [-1, 23, 26, 30],
                     labels=["MMSE <=23 (impaired)", "MMSE 24-26", "MMSE 27-30 (normal)"])
    fu = np.arange(0, 18)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    rows = {"fu_year": fu}
    for name, col in zip(["MMSE 27-30 (normal)", "MMSE 24-26", "MMSE <=23 (impaired)"],
                         [C1, C3, C2]):
        g = d[d.mm == name]
        y, keep = [], []
        for f in fu:
            s = g[g.fu_year == f]
            y.append(100 * s["bmi"].notna().mean() if len(s) >= 25 else np.nan)
            keep.append(len(s))
        ax.plot(fu, y, lw=2, color=col, marker="o", ms=4, label=name)
        last = np.where(~np.isnan(y))[0][-1]
        ax.annotate(f"{y[last]:.0f}%", (fu[last], y[last]), color=col, fontsize=9,
                    ha="left", va="center")
        rows[name] = y
    ax.set_ylim(0, 100); ax.set_xlim(-0.5, 18.6)     # headroom so the end labels are not clipped
    ax.set_xlabel("Follow-up year (fu_year)"); ax.set_ylabel("% of visits with BMI recorded")
    ax.set_title("E · THE TRAP: measurement depends on the outcome\n"
                 "BMI stops being taken as cognition declines (gap 6pp → 39pp)")
    # the sawtooth is a SECOND, unrelated artifact: BMI is on a biennial protocol in LATC
    # (even/odd gap 37.9pp) and MAP (10.8pp). Blood pressure and the labs show none of it.
    ax.annotate("sawtooth = biennial BMI protocol\n(LATC 38pp, MAP 11pp even-vs-odd year)",
                (9, 22), fontsize=7.5, color=REF_GREY, ha="center")
    ax.legend(frameon=False, loc="lower left", fontsize=8)
    save_data(pd.DataFrame(rows), OUT, "panel_e_missingness")
    return save_fig(fig, OUT, "panel_e_missingness")


# ---------------------------------------------------------------- F: cadence
def panel_f():
    p = os.path.join(HERE, "out_radc", "radc_all.bin")
    if not os.path.exists(p):
        from figure2.plotting_style import stub_figure
        return stub_figure(OUT, "panel_f_cadence", "out_radc/radc_all.bin missing -- "
                           "run data_prep/make_dataset_radc.py first")
    d = np.fromfile(p, dtype=np.uint32).reshape(-1, 3)
    D = pd.DataFrame(d, columns=["pid", "age", "tok"])
    D = D[D.age > 0]                                   # drop the age-0 static block
    u = D.groupby(["pid", "age"]).size().reset_index()
    dt = u.groupby("pid")["age"].diff().dropna() / 365.25
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    # bins CENTRED on integers, so the 1-year spike is one bar rather than split across two
    ax.hist(dt, bins=np.arange(-0.125, 8.2, .25), color=C1, edgecolor="white", lw=.8)
    ax.axvline(1.0, color=C2, lw=2, ls="--")
    ax.annotate(f"1.00 y — {100 * dt.between(.9, 1.1).mean():.0f}% of all gaps",
                (1.35, ax.get_ylim()[1] * .92), color=C2, fontsize=9, va="top")
    ax.annotate("gaps <3 months: the AD-dx token, which\nsits on real dates not the nominal grid",
                (0.35, ax.get_ylim()[1] * .45), fontsize=7.5, color=REF_GREY,
                arrowprops=dict(arrowstyle="-", color=REF_GREY, lw=.8), xytext=(2.2, ax.get_ylim()[1] * .45))
    ax.set_xlabel("Years between consecutive event times, same subject")
    ax.set_ylabel("Count")
    ax.set_title("F · fu_year really is annual\nmedian gap %.2f y — matches the model's scale"
                 % dt.median())
    save_data(pd.DataFrame({"gap_years": dt}), OUT, "panel_f_cadence")
    return save_fig(fig, OUT, "panel_f_cadence")


def main():
    setup_style()
    os.makedirs(OUT, exist_ok=True)
    cs, J = load()
    for fn, args in [(panel_a, (cs, J)), (panel_b, (J,)), (panel_c, (J,)),
                     (panel_d, (cs,)), (panel_e, (J,)), (panel_f, ())]:
        pdf, png = fn(*args)
        print(f"  {os.path.basename(png)}")
    print(f"\nDONE -> {OUT}")


if __name__ == "__main__":
    main()
