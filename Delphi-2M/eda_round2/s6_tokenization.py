"""Section 6 -- variable types, ICC, and how much a 4-bin tokenizer actually costs (risk R3)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from common import (C1, C2, C3, C4, DIV, REF_GREY, FIGS, LONG_CANDIDATES, VAR_META, load,
                    report, save_data, save_fig, save_table, setup_style, summary_put, gate)
from labels import make_pairs, target_frame, TARGETS

CORE = ["cogn_global", "cts_estmmse30"]
SECONDARY = ["bmi", "sbp_avg"]
NBINS = [4, 5, 6, 8, 10, 20, 50]
SCHEMES = ["equal_width", "quantile"]


# ---------------------------------------------------------------- binning
def cutpoints(x, n, scheme):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if scheme == "quantile":
        c = np.unique(np.quantile(x, np.linspace(0, 1, n + 1)[1:-1]))
    else:
        lo, hi = x.min(), x.max()
        c = np.linspace(lo, hi, n + 1)[1:-1]
    return c


def apply_cuts(x, c):
    return np.digitize(np.asarray(x, float), c, right=False)


def onehot(codes, k):
    m = np.zeros((len(codes), k), float)
    m[np.arange(len(codes)), np.clip(codes, 0, k - 1)] = 1.0
    return m


def robust_z(x):
    """|x - median| / (1.4826 * MAD). Immune to the very points it is meant to find."""
    x = pd.to_numeric(x, errors="coerce").to_numpy(float)
    m = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - m))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale == 0:
        return np.full_like(x, np.nan)
    return np.abs(x - m) / scale


def icc1(values, groups):
    """One-way random-effects ICC(1) = between-person share of total variance."""
    df = pd.DataFrame({"y": pd.to_numeric(values, errors="coerce"), "g": groups}).dropna()
    if df["g"].nunique() < 2 or len(df) <= df["g"].nunique():
        return np.nan, 0, 0, np.nan
    grand = df["y"].mean()
    gs = df.groupby("g")["y"]
    n_i = gs.size().to_numpy()
    m_i = gs.mean().to_numpy()
    k, N = len(n_i), len(df)
    ssb = float((n_i * (m_i - grand) ** 2).sum())
    ssw = float(((df["y"] - df["g"].map(gs.mean())) ** 2).sum())
    msb, msw = ssb / (k - 1), ssw / (N - k)
    n0 = (N - (n_i ** 2).sum() / N) / (k - 1)
    denom = msb + (n0 - 1) * msw
    if not np.isfinite(denom) or denom == 0:      # e.g. one observation per person
        return np.nan, k, N, np.nan
    icc = (msb - msw) / denom
    # keep the raw value too: a NEGATIVE raw ICC (clipped to 0) means within-person variance
    # exceeds between-person variance, which is a measurement-noise problem, not a "good" score.
    return float(np.clip(icc, 0, 1)), k, N, float(icc)


# ---------------------------------------------------------------- CV scoring
def oof_scores(X, y, groups):
    pred = np.full(len(y), np.nan)
    est = make_pipeline(StandardScaler(with_mean=False),
                        LogisticRegression(max_iter=3000, C=1.0))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        if y[tr].sum() == 0 or y[tr].sum() == len(tr):
            continue
        est.fit(X[tr], y[tr])
        pred[te] = est.predict_proba(X[te])[:, 1]
    ok = np.isfinite(pred)
    return roc_auc_score(y[ok], pred[ok]), average_precision_score(y[ok], pred[ok])


def oof_binned(x, y, groups, n, scheme):
    """Cutpoints are refit on the TRAINING fold only -- the spec's no-leakage rule."""
    pred = np.full(len(y), np.nan)
    est = make_pipeline(StandardScaler(with_mean=False),
                        LogisticRegression(max_iter=3000, C=1.0))
    for tr, te in GroupKFold(n_splits=5).split(x.reshape(-1, 1), y, groups):
        c = cutpoints(x[tr], n, scheme)
        k = len(c) + 1
        Xtr, Xte = onehot(apply_cuts(x[tr], c), k), onehot(apply_cuts(x[te], c), k)
        if y[tr].sum() == 0:
            continue
        est.fit(Xtr, y[tr])
        pred[te] = est.predict_proba(Xte)[:, 1]
    ok = np.isfinite(pred)
    return roc_auc_score(y[ok], pred[ok]), average_precision_score(y[ok], pred[ok])


def run():
    setup_style()
    cs, lo, cl, v = load()
    pairs = make_pairs(v)

    # ---- 1. variable type inventory -----------------------------------------------
    inv = []
    for c, (t, grain, note) in VAR_META.items():
        if t in ("id",):
            continue
        src = ("longitudinal_data_gk" if grain == "visit" else
               ("ROSMAP_clinical" if c in cl.columns and c not in cs.columns else "cross-sectional-data-gk"))
        frame = v if (grain == "visit" and c in v.columns) else (cs if c in cs.columns else cl)
        s = frame[c] if c in frame.columns else pd.Series(dtype=float)
        inv.append({"variable": c, "type": t, "grain": grain, "source": src,
                    "pct_nonnull": round(100 * s.notna().mean(), 2) if len(s) else np.nan,
                    "nunique": int(s.nunique()) if len(s) else 0, "codebook_note": note})
    inv = pd.DataFrame(inv)
    save_table(inv, "variable_type_inventory")
    type_counts = inv.groupby(["grain", "type"]).size().unstack(fill_value=0)

    # ---- 2. ICC --------------------------------------------------------------------
    iccs = []
    for c in LONG_CANDIDATES:
        if c not in v.columns:
            continue
        # A single absurd value wrecks a variance decomposition: one hba1c of 505 (the rest
        # are 3.9-11.4) drives the raw ICC to 0. Report both, and audit the offenders.
        rz = robust_z(v[c])
        extreme = np.isfinite(rz) & (rz > 10)
        val, k, N, raw = icc1(v[c], v["projid"])
        clean = v[c].where(~extreme)
        val_c, _, _, _ = icc1(clean, v["projid"])
        # also: what happens if ONLY the single largest value is dropped? If that alone moves
        # the ICC, the problem is one impossible record, not a noisy variable.
        col_ = pd.to_numeric(v[c], errors="coerce")
        val_m, _, _, _ = icc1(col_.where(col_ < col_.max()), v["projid"])
        ref = val_c if np.isfinite(val_c) else val
        if not np.isfinite(ref):
            verdict = "not estimable (too few repeated measures)"
        elif extreme.sum() and abs(ref - val) > .05:
            verdict = f"DATA QUALITY: {int(extreme.sum())} extreme value(s) distort the raw ICC"
        elif ref >= .85:
            verdict = "static-like -> move to a static covariate"
        elif ref < .10:
            verdict = "within-person noise dominates -> check reliability before tokenising"
        else:
            verdict = "sequence token"
        iccs.append({"variable": c, "type": VAR_META[c][0],
                     "icc1": round(val, 4) if np.isfinite(val) else np.nan,
                     "icc1_excl_extreme": round(val_c, 4) if np.isfinite(val_c) else np.nan,
                     "icc1_excl_single_max": round(val_m, 4) if np.isfinite(val_m) else np.nan,
                     "icc1_unclipped": round(raw, 4) if np.isfinite(raw) else np.nan,
                     "n_extreme_robust_z_gt10": int(extreme.sum()),
                     "n_persons": k, "n_obs": N,
                     "obs_per_person": round(N / k, 2) if k else np.nan,
                     "pct_nonnull": round(100 * v[c].notna().mean(), 2),
                     "verdict": verdict})
    icc_df = pd.DataFrame(iccs).sort_values("icc1", ascending=False)
    save_table(icc_df, "icc_by_variable")

    ext_rows = []
    for c in [x for x in LONG_CANDIDATES if VAR_META[x][0] == "cont" and x in v.columns]:
        col = v[c].dropna()
        if len(col) < 100:
            continue
        rz = robust_z(v[c])
        bad = v.loc[np.isfinite(rz) & (rz > 10), ["projid", "fu_year", c]]
        ext_rows.append({"variable": c, "n": len(col), "min": col.min(),
                         "p01": col.quantile(.01), "median": col.median(),
                         "p99": col.quantile(.99), "p999": col.quantile(.999), "max": col.max(),
                         "skew": round(float(stats.skew(col)), 2),
                         "n_robust_z_gt10": len(bad),
                         "example_extreme_values": ", ".join(f"{x:g}" for x in
                                                             sorted(bad[c].unique())[-5:])})
    ext = pd.DataFrame(ext_rows).round(3).sort_values("skew", key=np.abs, ascending=False)
    save_table(ext, "extreme_value_audit")
    dq = icc_df[icc_df["verdict"].str.startswith("DATA QUALITY")]["variable"].tolist()

    f, ax = plt.subplots(figsize=(6.8, 6.4))
    d = icc_df.dropna(subset=["icc1"]).copy()
    d["icc1"] = d["icc1_excl_extreme"].fillna(d["icc1"])   # plot the outlier-robust value
    d = d.sort_values("icc1")
    colr = [C2 if x >= .85 else C1 for x in d["icc1"]]
    ax.barh(range(len(d)), d["icc1"], color=colr, height=.68)
    ax.set_yticks(range(len(d))); ax.set_yticklabels(d["variable"], fontsize=8)
    ax.axvline(.85, color=REF_GREY, ls="--", lw=1.5)
    ax.annotate("0.85 — above this the variable\nbarely moves within a person",
                xy=(.85, 0.02), xycoords=("data", "axes fraction"), xytext=(-7, 0),
                textcoords="offset points", ha="right", va="bottom", fontsize=8.5, color=REF_GREY)
    for i, (val, name) in enumerate(zip(d["icc1"], d["variable"])):
        ax.annotate(f"{val:.2f}", (val, i), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color="#333333")
    ax.set_xlabel("ICC(1) — between-person share of total variance")
    ax.set_xlim(0, 1.08)
    ax.set_title("Where the variance lives — ICC(1) after removing robust-z>10 outliers\nblue = genuinely longitudinal · orange = effectively static")
    save_data(d, FIGS, "icc_by_variable")
    save_fig(f, FIGS, "icc_by_variable")

    # ---- 3. discretization loss ------------------------------------------------------
    rows = []
    for target in TARGETS:
        tf = target_frame(pairs, target)
        y_all = tf["y"].to_numpy()
        g_all = tf["projid"].to_numpy()
        for var in CORE + SECONDARY:
            m = tf[var].notna().to_numpy()
            x, y, g = tf.loc[m, var].to_numpy(float), y_all[m], g_all[m]
            if y.sum() < 30:
                continue
            auc_raw, ap_raw = oof_scores(x.reshape(-1, 1), y, g)
            rows.append({"target": target, "variable": var, "scheme": "raw_continuous", "n_bins": np.nan,
                         "n": len(y), "n_pos": int(y.sum()), "auc": round(auc_raw, 4),
                         "pr_auc": round(ap_raw, 4), "auc_loss_vs_raw": 0.0, "spearman_vs_raw": 1.0})
            for scheme in SCHEMES:
                for n in NBINS:
                    auc, ap = oof_binned(x, y, g, n, scheme)
                    c = cutpoints(x, n, scheme)
                    rho = stats.spearmanr(x, apply_cuts(x, c)).statistic
                    rows.append({"target": target, "variable": var, "scheme": scheme, "n_bins": n,
                                 "n": len(y), "n_pos": int(y.sum()), "auc": round(auc, 4),
                                 "pr_auc": round(ap, 4),
                                 "auc_loss_vs_raw": round(auc_raw - auc, 4),
                                 "spearman_vs_raw": round(float(rho), 4)})
    dl = pd.DataFrame(rows)
    save_table(dl, "discretization_loss")

    # How far does adding a 4-level CHANGE token close the gap left by coarse level bins?
    ld_rows = []
    for target in TARGETS:
        tf = target_frame(pairs, target)
        for var, dvar in [("cogn_global", "d_cogn"), ("cts_estmmse30", "d_mmse")]:
            sub = tf[tf[var].notna() & tf[dvar].notna()]
            if sub["y"].sum() < 30:
                continue
            y, g = sub["y"].to_numpy(), sub["projid"].to_numpy()
            xl, xd = sub[var].to_numpy(float), sub[dvar].to_numpy(float)
            a_raw, _ = oof_scores(np.column_stack([xl, xd]), y, g)
            for n in NBINS:
                cl_ = cutpoints(xl, n, "quantile")
                cd_ = cutpoints(xd, 4, "quantile")
                X = np.hstack([onehot(apply_cuts(xl, cl_), len(cl_) + 1),
                               onehot(apply_cuts(xd, cd_), len(cd_) + 1)])
                a, ap = oof_scores(X, y, g)
                ld_rows.append({"target": target, "variable": var, "n_level_bins": n,
                                "n_delta_bins": 4, "n": len(sub), "n_pos": int(y.sum()),
                                "auc": round(a, 4), "pr_auc": round(ap, 4),
                                "auc_raw_level_and_delta": round(a_raw, 4),
                                "auc_loss_vs_raw": round(a_raw - a, 4)})
    ld = pd.DataFrame(ld_rows)
    save_table(ld, "level_plus_delta_sweep")

    f, axes = plt.subplots(1, 2, figsize=(11.4, 4.4), sharey=True)
    for ax, target in zip(axes, TARGETS):
        sub = dl[(dl.target == target) & dl.variable.isin(CORE)]
        for (var, scheme), col, ls in [(("cogn_global", "quantile"), C1, "-"),
                                       (("cogn_global", "equal_width"), C1, ":"),
                                       (("cts_estmmse30", "quantile"), C2, "-"),
                                       (("cts_estmmse30", "equal_width"), C2, ":")]:
            s = sub[(sub.variable == var) & (sub.scheme == scheme)].sort_values("n_bins")
            ax.plot(s["n_bins"], s["auc_loss_vs_raw"], ls, color=col, lw=2,
                    marker="o" if scheme == "quantile" else "^", ms=7, mec="white", mew=1.2,
                    label=f"{var} · {'equal-frequency' if scheme == 'quantile' else 'equal-width'}")
        sub_ld = ld[(ld.target == target) & (ld.variable == "cogn_global")].sort_values("n_level_bins")
        ax.plot(sub_ld["n_level_bins"], sub_ld["auc_loss_vs_raw"], "-", color=C4, lw=2,
                marker="s", ms=7, mec="white", mew=1.2,
                label="cogn_global · equal-frequency + 4-bin Δ token")
        ax.axhline(0, color=REF_GREY, lw=1.5, ls="-")
        ax.axhline(.01, color=REF_GREY, lw=1, ls=":")
        ax.annotate("0.01 = the spec's indifference band", xy=(0.015, .01),
                    xycoords=("axes fraction", "data"), xytext=(0, 6), textcoords="offset points",
                    ha="left", va="bottom", fontsize=8, color=REF_GREY,
                    bbox=dict(fc="white", ec="none", alpha=.85, pad=1.5))
        ax.set_xscale("log"); ax.set_xticks(NBINS); ax.set_xticklabels(NBINS)
        ax.set_xlabel("Number of bins")
        ax.set_title(f"{target}\n({TARGETS[target]})", fontsize=9.5)
    axes[0].set_ylabel("AUC lost vs the raw continuous value")
    axes[0].legend(frameon=False, loc="upper right", fontsize=8)
    f.suptitle("Discretisation cost: 4 bins throws away real AUC; a Δ token buys most of it back",
               y=1.02)
    f.tight_layout()
    save_data(dl, FIGS, "discretization_loss")
    save_fig(f, FIGS, "discretization_loss")

    # ---- 4. token entropy / degenerate bins --------------------------------------------
    ent = []
    for var in CORE + SECONDARY + [c for c in LONG_CANDIDATES
                                   if VAR_META[c][0] in ("binary", "ordinal") and c in v.columns]:
        s = v[var].dropna().to_numpy(float)
        if len(s) < 100:
            continue
        if VAR_META[var][0] in ("binary", "ordinal"):
            vals, cnt = np.unique(s, return_counts=True)
            share = cnt / cnt.sum()
            ent.append({"variable": var, "scheme": "native_levels", "n_bins": len(vals),
                        "max_bin_share_pct": round(100 * share.max(), 2),
                        "entropy_bits": round(float(-(share * np.log2(share)).sum()), 3),
                        "max_entropy_bits": round(float(np.log2(len(vals))), 3),
                        "degenerate_gt70pct": bool(share.max() > .7)})
            continue
        for scheme in SCHEMES:
            for n in (4, 10):
                c = cutpoints(s, n, scheme)
                codes = apply_cuts(s, c)
                cnt = np.bincount(codes, minlength=len(c) + 1)
                share = cnt / cnt.sum()
                nz = share[share > 0]
                ent.append({"variable": var, "scheme": scheme, "n_bins": n,
                            "max_bin_share_pct": round(100 * share.max(), 2),
                            "entropy_bits": round(float(-(nz * np.log2(nz)).sum()), 3),
                            "max_entropy_bits": round(float(np.log2(n)), 3),
                            "degenerate_gt70pct": bool(share.max() > .7)})
    ent = pd.DataFrame(ent)
    save_table(ent, "token_entropy")
    degen = ent[ent.degenerate_gt70pct]["variable"].unique().tolist()

    # ---- 5. within-person bin transitions ------------------------------------------------
    tr = []
    vv = v.sort_values(["projid", "fu_year"]).copy()
    for var in CORE + SECONDARY:
        for scheme in SCHEMES:
            for n in (4, 10):
                c = cutpoints(vv[var].dropna(), n, scheme)
                code = pd.Series(np.where(vv[var].notna(), apply_cuts(vv[var], c), np.nan),
                                 index=vv.index)
                g = vv.assign(code=code).groupby("projid", sort=False)
                prev = g["code"].shift(1)
                same_year = vv["d_fu"].eq(1)
                m = code.notna() & prev.notna() & same_year
                distinct = (vv.assign(code=code).dropna(subset=["code"])
                            .groupby("projid")["code"].nunique())
                tr.append({"variable": var, "scheme": scheme, "n_bins": n, "n_pairs": int(m.sum()),
                           "transition_rate_pct": round(100 * float((code[m] != prev[m]).mean()), 2),
                           "median_distinct_bins_per_person": float(distinct.median()),
                           "pct_persons_single_bin": round(100 * float((distinct == 1).mean()), 2)})
    tr = pd.DataFrame(tr)
    save_table(tr, "bin_transition_rates")
    tr4 = tr[(tr.n_bins == 4) & (tr.scheme == "quantile")].set_index("variable")

    # ---- 6. does a DELTA token buy anything? ----------------------------------------------
    gain = []
    for target in TARGETS:
        tf = target_frame(pairs, target)
        for var, dvar in [("cogn_global", "d_cogn"), ("cts_estmmse30", "d_mmse")]:
            m = tf[var].notna() & tf[dvar].notna()
            sub = tf[m]
            if sub["y"].sum() < 30:
                continue
            y, g = sub["y"].to_numpy(), sub["projid"].to_numpy()
            xl, xd = sub[var].to_numpy(float), sub[dvar].to_numpy(float)
            cl_, cd = cutpoints(xl, 4, "quantile"), cutpoints(xd, 4, "quantile")
            Xl = onehot(apply_cuts(xl, cl_), len(cl_) + 1)
            Xd = onehot(apply_cuts(xd, cd), len(cd) + 1)
            a_l, p_l = oof_scores(Xl, y, g)
            a_ld, p_ld = oof_scores(np.hstack([Xl, Xd]), y, g)
            a_raw, p_raw = oof_scores(np.column_stack([xl, xd]), y, g)
            gain.append({"target": target, "variable": var, "n": len(sub), "n_pos": int(y.sum()),
                         "auc_level4": round(a_l, 4), "auc_level4_plus_delta4": round(a_ld, 4),
                         "auc_gain": round(a_ld - a_l, 4),
                         "pr_auc_level4": round(p_l, 4), "pr_auc_level4_plus_delta4": round(p_ld, 4),
                         "auc_raw_level_and_delta": round(a_raw, 4),
                         "delta_cutpoints": np.round(cd, 4).tolist(),
                         "level_cutpoints": np.round(cl_, 4).tolist()})
    gain = pd.DataFrame(gain)
    save_table(gain, "delta_token_gain")

    # The bar chart of an equal-frequency binning is 25/25/25/25 by construction, i.e. it
    # shows nothing. Plot the Δ distribution itself with the cutpoints on it instead.
    dcog = pairs["d_cogn"].dropna()
    cd = cutpoints(dcog, 4, "quantile")
    codes = apply_cuts(dcog, cd)
    shares = np.bincount(codes, minlength=4) / len(dcog)
    names = ["marked decline", "mild decline", "stable", "improved"]
    cols = [DIV[4], DIV[3], DIV[2], DIV[0]]
    edges = np.arange(-2.0, 2.0 + 0.05, 0.05)
    f, ax = plt.subplots(figsize=(7.0, 4.3))
    counts, _ = np.histogram(dcog, bins=edges)
    centres = (edges[:-1] + edges[1:]) / 2
    which = np.clip(np.digitize(centres, cd), 0, 3)
    ax.bar(centres, counts, width=0.048, color=[cols[w] for w in which], lw=0)
    ax.axvline(0, color=REF_GREY, lw=1.2, ls="-")
    top = counts.max()
    for k, c in enumerate(cd):
        ax.axvline(c, color="white", lw=1.6)
        ax.annotate(f"{c:+.2f}", (c, top * 1.01), ha="center", va="bottom", fontsize=8,
                    color="#333333")
    # Segment names sit above the bars on two staggered rows with leader lines: the two middle
    # segments are only ~0.2 z wide and would otherwise overprint each other.
    seg = [-1.5, *cd, 1.0]
    for k, (lo_, hi_) in enumerate(zip(seg[:-1], seg[1:])):
        xm = (max(lo_, -1.45) + min(hi_, 0.95)) / 2
        y = top * (1.16 if k % 2 == 0 else 1.33)
        ax.annotate(names[k], (xm, top * 1.10), xytext=(xm, y), ha="center", va="bottom",
                    fontsize=8.5, color="#222222",
                    arrowprops=dict(arrowstyle="-", color=REF_GREY, lw=.8))
    ax.set_xlim(-1.5, 1.0)
    ax.set_ylim(0, top * 1.50)
    ax.set_xlabel("Δ global cognition since the previous visit (z)")
    ax.set_ylabel("Visit-to-visit changes")
    ax.set_title("Δcogn_global with the 4 equal-frequency cutpoints (25% each by construction)\n"
                 f"left-skewed, mean {dcog.mean():+.3f} — decline is the common direction")
    save_data(pd.DataFrame({"bin": names, "share_pct": 100 * shares,
                            "cutpoint_lo": [-np.inf, *cd], "cutpoint_hi": [*cd, np.inf]}),
              FIGS, "delta_token_bins")
    save_fig(f, FIGS, "delta_token_bins")

    # ---- 7. tokens at the same age ----------------------------------------------------
    present = v[[c for c in LONG_CANDIDATES if c in v.columns]].notna().sum(axis=1)
    mpv = pd.DataFrame({"stat": ["min", "P25", "median", "mean", "P75", "P95", "max"],
                        "measurements_per_visit": [present.min(), present.quantile(.25),
                                                   present.median(), round(present.mean(), 2),
                                                   present.quantile(.75), present.quantile(.95),
                                                   present.max()]})
    save_table(mpv, "measurements_per_visit")

    # ---- 8. ceiling / floor ------------------------------------------------------------
    cf = []
    for var, hi, lo_ in [("cts_estmmse30", 30, 0)]:
        s = v[var].dropna()
        cf.append({"variable": var, "n": len(s), "pct_at_ceiling": round(100 * (s >= hi).mean(), 2),
                   "pct_at_floor": round(100 * (s <= lo_).mean(), 2),
                   "pct_within_1_of_ceiling": round(100 * (s >= hi - 1).mean(), 2)})
    s = v["cogn_global"].dropna()
    cf.append({"variable": "cogn_global", "n": len(s), "pct_at_ceiling": np.nan,
               "pct_at_floor": np.nan, "pct_within_1_of_ceiling": np.nan})
    save_table(pd.DataFrame(cf), "ceiling_floor")
    mmse_ceiling = 100 * (v["cts_estmmse30"].dropna() >= 30).mean()

    # ---- 9. continuous distributions ----------------------------------------------------
    conts = [c for c in LONG_CANDIDATES if VAR_META[c][0] == "cont" and c in v.columns]
    ncol = 4
    nrow = int(np.ceil(len(conts) / ncol))
    f, axes = plt.subplots(nrow, ncol, figsize=(3.0 * ncol, 2.3 * nrow))
    for ax, c in zip(axes.ravel(), conts):
        s = v[c].dropna()
        ax.hist(s, bins=40, color=C1, lw=0)
        ax.set_title(f"{c}\nn={len(s):,} · skew {stats.skew(s):+.1f} · ICC "
                     f"{icc_df.set_index('variable').loc[c, 'icc1']:.2f}", fontsize=7.5)
        ax.tick_params(labelsize=6.5)
    for ax in axes.ravel()[len(conts):]:
        ax.set_axis_off()
    f.suptitle("Continuous longitudinal measurements (n, skew, ICC in each title)", y=1.005)
    f.tight_layout()
    save_data(v[["projid", "fu_year"] + conts], FIGS, "continuous_var_distributions")
    save_fig(f, FIGS, "continuous_var_distributions")

    # ---- summary + recommendation --------------------------------------------------------
    best = {}
    for target in TARGETS:
        s = dl[(dl.target == target) & (dl.variable == "cogn_global") & (dl.scheme == "quantile")]
        raw = dl[(dl.target == target) & (dl.variable == "cogn_global")
                 & (dl.scheme == "raw_continuous")]["auc"].iloc[0]
        loss4 = float(s[s.n_bins == 4]["auc_loss_vs_raw"].iloc[0])
        loss20 = float(s[s.n_bins == 20]["auc_loss_vs_raw"].iloc[0])
        best[target] = {"auc_raw": raw, "loss_4bin": loss4, "loss_20bin": loss20,
                        "gap_4_vs_20": round(loss4 - loss20, 4)}
    four_ok = all(abs(b["gap_4_vs_20"]) < .01 for b in best.values())

    def _smallest_within(frame, keycol, tol=.01):
        """Smallest bin count whose OOF AUC loss vs raw is <= tol for BOTH targets."""
        for n in NBINS:
            ok = True
            for t in TARGETS:
                r = frame[(frame.target == t) & (frame.variable == "cogn_global") & (frame[keycol] == n)]
                if r.empty or float(r["auc_loss_vs_raw"].iloc[0]) > tol:
                    ok = False
            if ok:
                return n
        return None

    n_level_only = _smallest_within(dl[dl.scheme == "quantile"], "n_bins")
    n_with_delta = _smallest_within(ld, "n_level_bins")
    rec_bins = n_with_delta or n_level_only or max(NBINS)
    # The single number above is very sensitive to the tolerance, so show the whole trade-off.
    rec_rows = [{"auc_loss_tolerance": tol,
                 "bins_needed_level_only": _smallest_within(dl[dl.scheme == "quantile"], "n_bins", tol),
                 "bins_needed_level_plus_delta4": _smallest_within(ld, "n_level_bins", tol)}
                for tol in (0.005, 0.01, 0.02, 0.03, 0.05)]
    rec_tbl_df = pd.DataFrame(rec_rows)
    save_table(rec_tbl_df, "bins_needed_vs_tolerance")
    rec_tbl = rec_tbl_df.rename(columns={"auc_loss_tolerance": "可接受的 AUC 损失",
                                         "bins_needed_level_only": "仅水平档所需档数",
                                         "bins_needed_level_plus_delta4": "水平档 + 4 档变化所需档数"}
                                ).to_markdown(index=False)
    transition_low = float(tr4["transition_rate_pct"].min()) < 10
    delta_gain = float(gain["auc_gain"].max())

    summary_put(n_measurements_per_visit_median=float(present.median()),
                n_measurements_per_visit_mean=round(float(present.mean()), 2),
                recommended_n_bins=10,
                recommended_n_bins_strict_0p01=int(rec_bins),
                recommended_n_bins_level_only=(int(n_level_only) if n_level_only else None),
                recommended_n_bins_with_delta_token=(int(n_with_delta) if n_with_delta else None),
                discretization_gap_4_vs_20=best,
                bin_transition_rate_4bin_quantile={k: float(vv_) for k, vv_ in
                                                  tr4["transition_rate_pct"].items()},
                degenerate_tokens_gt70pct=degen,
                delta_token_max_auc_gain=round(delta_gain, 4),
                mmse_pct_at_ceiling=round(float(mmse_ceiling), 2),
                n_static_like_variables=int((icc_df["icc1"] >= .85).sum()),
                cognitive_domain_scores_available=False)

    ent4 = ent[(ent.n_bins == 4) & (ent.scheme == "quantile")].set_index("variable")
    piv_lv = (dl[(dl.scheme == "quantile") & (dl.variable == "cogn_global")]
              .pivot_table(index="target", columns="n_bins", values="auc_loss_vs_raw")
              .round(4))
    piv_lv.columns = [f"{int(c)} 档" for c in piv_lv.columns]
    piv_ld = (ld[ld.variable == "cogn_global"]
              .pivot_table(index="target", columns="n_level_bins", values="auc_loss_vs_raw")
              .round(4))
    piv_ld.columns = [f"{int(c)} 档 + Δ" for c in piv_ld.columns]
    tbl63 = piv_lv.join(piv_ld).reset_index().rename(columns={"target": "结局"}).to_markdown(index=False)
    gap4_txt = " / ".join(f"{t}: {best[t]['gap_4_vs_20']:+.4f}" for t in TARGETS)
    report(f"""## 6. 变量类型构成与 tokenization 可行性（风险 R3）

**算了什么**：变量四分类清点；每个纵向连续变量的 ICC(1)（被试间方差占比）；4/5/6/8/10/20/50 分箱 ×
等宽/等频 的**离散化损失**（切点在训练折内重新拟合，无泄漏；out-of-fold AUC 与 PR-AUC）；分箱后的 token 熵与
退化档；被试内 bin 转移率；变化量（delta）token 的增益；一次访视的测量数；MMSE 天花板效应。

### 6.1 变量类型构成

{type_counts.to_markdown()}

**本 release 缺失、但规格书假定存在的**：5 个认知域分 (`cogn_ep/se/po/ps/wo`)、19 项 `cts_*` 原始测验、
逐次访视诊断 `dcfdx`。因此「离散化损失」只能在 `cogn_global` 与 `cts_estmmse30` 两个核心连续量上做
（另加 `bmi`、`sbp_avg` 作次要对照），下一次访视的「诊断」也只能用两个**代理结局**：

- `y_ad_next`：下次访视前发生首次 AD 痴呆诊断（来自 `age_first_ad_dx`）。
  n={len(target_frame(pairs, 'y_ad_next')):,} 对，阳性 {100 * target_frame(pairs, 'y_ad_next')['y'].mean():.2f}%。
  已诊断者不再入组；另外剔除 {int(pairs.loc[pairs.suspected_prevalent_dementia, 'projid'].nunique())} 名
  「基线即疑似痴呆」者（codebook 明确：这些人根本不记录 `age_first_ad_dx`，给 0 标签是**错的**而不只是删失）。
- `y_imp_next`：下次访视估计 MMSE < 24。n={len(target_frame(pairs, 'y_imp_next')):,} 对，阳性 {100 * target_frame(pairs, 'y_imp_next')['y'].mean():.2f}%。

两个结局都是 **3–6% 的极不平衡**，所有比较一律同时报告 ROC-AUC 与 PR-AUC。

### 6.2 方差住在哪里（ICC）

ICC(1) ≥ 0.85 的变量在个体内几乎不动，**不该进序列**，应作为静态协变量放序列开头：
{', '.join(f"`{r.variable}` ({r.icc_ref:.2f})" for r in icc_df.assign(icc_ref=icc_df['icc1_excl_extreme'].fillna(icc_df['icc1']))[lambda d: d.icc_ref >= .85].itertuples()) or '（无）'}。

**顺带查出一条真实的脏数据**（`tables/extreme_value_audit.csv`）：`hba1c` 的原始 ICC 是
{icc_df.set_index('variable').loc['hba1c', 'icc1']:.3f}，看上去「全是噪声」。真相是
**一条 505.0 的记录**（其余 P99.9 = 11.4，偏度 +{ext.set_index('variable').loc['hba1c', 'skew']:.0f}，
临床上 HbA1c 不可能是 505）把方差分解整个带偏了——
**只删这一条**，ICC 就从 {icc_df.set_index('variable').loc['hba1c', 'icc1']:.3f} 恢复到
{icc_df.set_index('variable').loc['hba1c', 'icc1_excl_single_max']:.3f}。

要说清楚的是：robust-z>10 这个筛子在 `hba1c` 上一共标了
{int(icc_df.set_index('variable').loc['hba1c', 'n_extreme_robust_z_gt10'])} 条，但其中除了 505 之外
（13.2–16.4）都是**控制很差的糖尿病，是真实值**——因为 `hba1c` 的四分位距只有 0.5，
robust-z 太敏感。`glucose` 被标的 {int(icc_df.set_index('variable').loc['glucose', 'n_extreme_robust_z_gt10'])} 条
（431–522 mg/dL）同理是真实的严重高血糖，剔除后 ICC 几乎不变
（{icc_df.set_index('variable').loc['glucose', 'icc1']:.3f} → {icc_df.set_index('variable').loc['glucose', 'icc1_excl_extreme']:.3f}）。
**筛子用来找线索，不要用来自动删数据**；上图与本节的 ICC 用的是剔除极端值后的版本，
被标为 DATA QUALITY 的只有：{', '.join(f'`{x}`' for x in dq) if dq else '（无）'}。

→ 给实现的要求：**tokenizer 之前必须先跑一遍极端值筛查并人工过一遍临床合理区间**
（`hba1c` 3–20、`glucose` 20–600 之类），否则等频切点会被这种点直接拉歪。

真正纵向的核心量：`cogn_global` ICC {icc_df.set_index('variable').loc['cogn_global', 'icc1']:.2f} ·
`cts_estmmse30` {icc_df.set_index('variable').loc['cts_estmmse30', 'icc1']:.2f} ·
`bmi` {icc_df.set_index('variable').loc['bmi', 'icc1']:.2f} ·
`sbp_avg` {icc_df.set_index('variable').loc['sbp_avg', 'icc1']:.2f}。
注意 `bmi` 的 ICC {icc_df.set_index('variable').loc['bmi', 'icc1']:.2f} 已经越过 0.85——它的方差绝大部分是**人和人之间**的差异而不是个体内的轨迹，作为逐次访视 token 的信息量很低（而且它的缺失率还随随访年数从 fu0–2 的 10% 涨到 fu≥10 的 32%，还带一个隔年测量的奇偶周期，见 §5）。建议：基线 BMI 作为静态协变量，序列里不再逐年重复。

### 6.3 离散化损失：4 档**不**够用

`cogn_global` 的**丢失的 AUC**（等频分箱，5 折 GroupKFold out-of-fold；切点在训练折内重新拟合；
raw 连续值 = 上限。`+Δ` 列表示「n 档水平 token + 4 档变化 token」）：

raw 上限：y_ad_next {best['y_ad_next']['auc_raw']:.4f} · y_imp_next {best['y_imp_next']['auc_raw']:.4f}

{tbl63}

4 档 vs 20 档之差：**{gap4_txt}**（规格书的无差异带是 0.01）。
Spearman(4 档值, 原值) = {dl[(dl.n_bins == 4) & (dl.scheme == 'quantile') & (dl.variable == 'cogn_global')]['spearman_vs_raw'].iloc[0]:.3f}——
相关性看起来很高，**但它完全掩盖了预测性能的损失**，不要用它来论证分箱无害。
完整 4 变量 × 2 方案 × 7 档数 × 2 结局的表见 `tables/discretization_loss.csv`。

{gate(four_ok, ("**4 档与 20 档的 AUC 差 < 0.01 → 4 档确认可用。**" if four_ok else
      f"**4 档明显劣于 10–20 档（差 {gap4_txt}，是无差异带的数倍）→ 4 档不可用。**"
      " 按规格书的处置顺序：先加变化档（§6.5），仍不够再逐步加到 6–8 档。"
      " 本轮实测的达标档数（两个结局都要满足）："
      f" 严格按 0.01 的带宽，只用水平档要 **{n_level_only} 档**，加变化档也要 **{n_with_delta} 档**；"
      f" 放宽到 0.02，则分别是 **{rec_rows[2]['bins_needed_level_only']} 档**"
      f" 与 **{rec_rows[2]['bins_needed_level_plus_delta4']} 档**（下表）。")
      + " 词表 = 变量 ID token + 分箱值 token；档数是这里唯一需要改的参数。")}

{rec_tbl}

**给实现的建议**：用 **10 档等频水平 token + 4 档变化 token**。理由：10 档时损失
{float(ld[(ld.target == 'y_ad_next') & (ld.variable == 'cogn_global') & (ld.n_level_bins == 10)]['auc_loss_vs_raw'].iloc[0]):.4f} /
{float(ld[(ld.target == 'y_imp_next') & (ld.variable == 'cogn_global') & (ld.n_level_bins == 10)]['auc_loss_vs_raw'].iloc[0]):.4f}，
已经把 4 档时 {best['y_ad_next']['loss_4bin']:.3f} 的损失压掉约 {100 * (1 - float(ld[(ld.target == 'y_ad_next') & (ld.variable == 'cogn_global') & (ld.n_level_bins == 10)]['auc_loss_vs_raw'].iloc[0]) / best['y_ad_next']['loss_4bin']):.0f}%；
再往 20 档走只多买回 {float(ld[(ld.target == 'y_ad_next') & (ld.variable == 'cogn_global') & (ld.n_level_bins == 10)]['auc_loss_vs_raw'].iloc[0]) - float(ld[(ld.target == 'y_ad_next') & (ld.variable == 'cogn_global') & (ld.n_level_bins == 20)]['auc_loss_vs_raw'].iloc[0]):.4f} / {float(ld[(ld.target == 'y_imp_next') & (ld.variable == 'cogn_global') & (ld.n_level_bins == 10)]['auc_loss_vs_raw'].iloc[0]) - float(ld[(ld.target == 'y_imp_next') & (ld.variable == 'cogn_global') & (ld.n_level_bins == 20)]['auc_loss_vs_raw'].iloc[0]):.4f} AUC，却要在这份 4,428 人的数据上多养一倍的 value embedding。
**规格书里「约 4 档」的预设在这份数据上是站不住的，必须改。**

### 6.4 退化档与转移率

- 最大档占比 > 70% 的变量：{', '.join(f'`{x}`' for x in degen) if degen else '（连续量分箱后无）'}。
  这些几乎全是**天然极度失衡的二值/序数变量**（如 `r_stroke` 有 {100 * (v['r_stroke'] == 4).sum() / v['r_stroke'].notna().sum():.1f}% 是「无卒中」、
  `chf_cum`、`ad_rx` 等），不是分箱方案的错。它们应当按「事件型 token」处理——**只在发生时发射一个 token**，
  而不是每次访视都发一个「没发生」的 token，否则序列会被常数 token 淹没。
- `cts_estmmse30` 有 **{mmse_ceiling:.1f}% 的访视顶在 30 分**（天花板效应）。等宽分箱会把这一大坨压进最高档；
  等频分箱在这里**做不出 4 个等量档**（最高档必然 >25%）。这就是 MMSE 分箱损失高于 cogn_global 的原因，
  也是应当以 `cogn_global` 为主认知 token、MMSE 为辅的理由。
- 被试内 bin 转移率（4 档等频、仅 Δ=1 年的相邻访视）：
  {' · '.join(f"`{i}` {r:.1f}%" for i, r in tr4['transition_rate_pct'].items())}。

{gate(not transition_low, ("**存在转移率 < 10% 的变量**（"
      + ', '.join(f"`{i}` {r:.1f}%" for i, r in tr4['transition_rate_pct'].items() if r < 10)
      + "）→ 对这些变量粗分箱把轨迹抹平了，必须引入变化量 token。") if transition_low else
      "**所有核心连续量的 4 档转移率都 ≥ 10%**（最低 "
      + f"{tr4['transition_rate_pct'].min():.1f}%），"
      "序列不是同一个 token 的重复堆砌，规格书担心的「粗分箱把轨迹抹平」在**连续量**上没有发生。"
      "真正被抹平的是那些天然失衡的二值/序数变量（上一条），它们要改成事件型 token。"
      "注意转移率高不等于分箱无损——§6.3 显示 4 档仍然丢掉了实打实的 AUC。")}

### 6.5 变化量 token 值不值

`<var>_level`(4 档) 对比 `<var>_level + <var>_delta`(4+4 档)，同样是 out-of-fold：

{gain[['target', 'variable', 'n', 'n_pos', 'auc_level4', 'auc_level4_plus_delta4', 'auc_gain', 'pr_auc_level4', 'pr_auc_level4_plus_delta4']].to_markdown(index=False)}

最大 AUC 增益 **{delta_gain:+.4f}**。Δcogn_global 的 4 档等频切点为
{np.round(cd, 3).tolist()}（各档占比 {', '.join(f'{100 * s:.1f}%' for s in shares)}）。

{gate(delta_gain >= .01, ("**加入变化档后 AUC 有实质提升（最大 "
      f"{delta_gain:+.4f} ≥ 0.01）→ 采用「level + delta」双 token 方案。**"
      " 每次访视对核心认知量发射两个 token（`<var>_level` + `<var>_delta`），词表翻倍，"
      " 但把衰退速率直接暴露给模型，且如 §6.3 所示它能把达标所需的水平档数从 "
      f"{n_level_only} 档降到 {n_with_delta} 档——**净词表其实是变小的**。"
      " 注意：变化 token 的切点同样必须在训练折上确定后冻结（见 §9），"
      " 且序列的第一个访视没有变化档，需要一个 `<delta_unknown>` token。") if delta_gain >= .01 else
      f"变化档带来的增益只有 {delta_gain:+.4f}（<0.01），为它翻倍词表不划算。")}

### 6.6 同一访视的 token 数（attention mask 的直接依据）

一次访视同时产生的非空测量数：median **{present.median():.0f}**、mean {present.mean():.1f}、
P95 {present.quantile(.95):.0f}、max {present.max():.0f}（候选纵向变量共 {len(LONG_CANDIDATES)} 个）。

{gate(False, f"**一次访视的测量数中位数 {present.median():.0f} ≫ 1 → Delphi 的「同时刻互相屏蔽」attention mask 是必需的。**"
      "否则模型在预测本次 `cogn_global` 时可以直接看到同一次访视的 `cts_estmmse30`（两者相关性极高），"
      "指标会漂亮得离谱而毫无意义——这是本项目最容易踩、也最难在事后发现的泄漏。"
      "实现要求：同一 `(projid, fu_year)` 的所有 token 共享一个位置（年龄），彼此**完全不可见**，"
      "只能看到严格更早 `fu_year` 的 token。")}

### 6.7 测验版本变更

无日历年份字段（§4.5），无法直接检验版本漂移。可观察到的替代证据是 §5 的缺失热力图：
`log_hcrp`/`log_hil6`/`log_htnfa`、`berlin_risk_class`、`psqi_sum`、`hba1c`、血脂等变量的可得性
**按 cycle 区段成块出现/消失**，说明它们是阶段性子研究而非常规采集。这些变量若进序列，
模型会把「这一年做没做这个子研究」学成信息——建议全部降级或直接排除。
""")
    print("s6 done")


if __name__ == "__main__":
    run()
