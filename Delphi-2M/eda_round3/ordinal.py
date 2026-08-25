"""
ordinal.py -- the spec's §2 "generic component", written ONCE and reused for every variable.

analyze_ordinal() runs §2.1-§2.7 on one Var against one Panel and returns a flat dict that
becomes one row of SCALE_SUMMARY.csv. Per-variable figures and tables land in
eda_out/scales/<dataset>_<var>/.

Two design decisions worth stating because they change the numbers:

1. LEVELS. §2.2/§2.6 need a finite set of levels. A variable whose observed values are
   already on its declared grid is used as-is. A CONTINUOUS variable is cut into deciles.
   A variable with off-grid values (RADC cts_estmmse30 is 1.55% non-integer because it is
   partly estimated from the MoCA) is SNAPPED to the nearest declared level and the snapped
   fraction is reported, never silently absorbed.

2. THE REFERENCE OUTCOME. §2.6 asks for P(next visit is dementia | value = v). Taken
   literally that includes rows that are ALREADY demented, so for a severity scale it is
   partly a persistence test rather than an incidence test. Both are computed: the literal
   all-pairs version drives the verdict (it is what the spec asks for), and the
   at-risk-only incident version is reported beside it. Where they disagree, §6 says so.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.isotonic import IsotonicRegression

from common import (C1, C2, C3, C4, CONTINUOUS, REF_GREY, icc1, save_data, save_fig,
                    save_table, shannon, vardir, wilson, _num)

RARE_THRESHOLDS = (50, 100, 500)
N_DECILES = 10


# ---------------------------------------------------------------- levels
def levelize(v, s):
    """(level_series, level_values, meta) for one Var and one raw column.

    level_series holds the LEVEL VALUE (not a rank) so plots keep their natural axis.
    """
    x = v.clean(s)
    meta = {"level_transform": "as-is", "pct_snapped": 0.0, "levels_are_quantile_bins": False}
    if v.expected is CONTINUOUS:
        q = np.unique(np.nanquantile(x.dropna(), np.linspace(0, 1, N_DECILES + 1)))
        codes = np.digitize(x.to_numpy(float), q[1:-1], right=False).astype(float)
        codes[~np.isfinite(x.to_numpy(float))] = np.nan
        lv = [float(i) for i in range(len(q) - 1)]
        meta["level_transform"] = (f"{len(lv)} quantile bins, cutpoints "
                                   + ", ".join(f"{c:.3f}" for c in q[1:-1]))
        meta["bin_edges"] = [float(c) for c in q]
        # Quantile bins force ~equal occupancy, so §2.2 entropy and §2.3 ceiling/floor are
        # artifacts of the binning for this variable, not properties of the measurement.
        meta["levels_are_quantile_bins"] = True
        return pd.Series(codes, index=x.index), lv, meta
    grid = np.asarray(v.expected, float)
    xv = x.to_numpy(float)
    ok = np.isfinite(xv)
    snapped = np.full_like(xv, np.nan)
    if ok.any():
        idx = np.abs(xv[ok, None] - grid[None, :]).argmin(axis=1)
        snapped[ok] = grid[idx]
        off = ~np.isclose(xv[ok], snapped[ok])
        meta["pct_snapped"] = float(100.0 * off.mean())
        if off.any():
            meta["level_transform"] = (f"snapped to the nearest declared level; "
                                       f"{meta['pct_snapped']:.2f}% of values were off-grid")
    return pd.Series(snapped, index=x.index), [float(g) for g in grid], meta


# ---------------------------------------------------------------- §2.1 coverage
def coverage(v, panel, col, res):
    d = panel.df
    obs = v.clean(d[col]).notna()
    res["n_obs"] = int(obs.sum())
    res["n_rows"] = int(len(d))
    res["pct_missing"] = float(100.0 * (1 - obs.mean()))
    res["n_persons_with_obs"] = int(d.loc[obs, panel.person].nunique())
    # Coverage alone does not make a sequence backbone: what matters is how many times ONE
    # person is measured. MoCA covers 36% of visits but only ~2 visits per person (§3.2).
    per_person = d.loc[obs].groupby(panel.person).size() if obs.any() else pd.Series(dtype=int)
    res["median_obs_per_person"] = float(per_person.median()) if len(per_person) else np.nan
    res["pct_persons_with_ge3_obs"] = (float(100.0 * (per_person >= 3).mean())
                                       if len(per_person) else np.nan)

    parts = []
    if panel.year is not None:
        by_year = obs.groupby(d[panel.year]).agg(["mean", "size"])
        by_year.index = by_year.index.astype(int)
        parts.append(("calendar year", by_year))
        step = float(by_year["mean"].diff().abs().max()) if len(by_year) > 1 else np.nan
        res["max_year_over_year_coverage_jump"] = step
    else:
        res["max_year_over_year_coverage_jump"] = np.nan
    if panel.version is not None:
        by_ver = obs.groupby(d[panel.version]).agg(["mean", "size"])
        parts.append(("form version / cohort", by_ver))
    if panel.grain == "visit":
        vi = d["visit_idx"].clip(upper=15)
        by_vi = obs.groupby(vi).agg(["mean", "size"])
        parts.append(("visit index (15 = 15+)", by_vi))
        m = by_vi["mean"]
        keep = by_vi["size"] >= 200
        if keep.sum() >= 4 and m[keep].nunique() > 1:
            rho = stats.spearmanr(m[keep].index.astype(float), m[keep].to_numpy()).statistic
            res["coverage_vs_visit_idx_spearman"] = float(rho)
            res["coverage_drop_over_visits"] = float(m[keep].iloc[0] - m[keep].iloc[-1])
        else:
            res["coverage_vs_visit_idx_spearman"] = np.nan
            res["coverage_drop_over_visits"] = np.nan
    else:
        res["coverage_vs_visit_idx_spearman"] = np.nan
        res["coverage_drop_over_visits"] = np.nan

    if parts:
        fig, axes = plt.subplots(1, len(parts), figsize=(4.6 * len(parts), 3.4), squeeze=False)
        rows = []
        for ax, (label, tab) in zip(axes[0], parts):
            xs = np.arange(len(tab))
            ax.bar(xs, tab["mean"].to_numpy() * 100, color=C1)
            ax.set_xticks(xs)
            ax.set_xticklabels([str(i) for i in tab.index], rotation=90 if len(tab) > 8 else 0,
                               fontsize=7)
            ax.set_ylim(0, 100)
            ax.set_xlabel(label)
            ax.set_ylabel("coverage %")
            for i, k in enumerate(tab.index):
                rows.append({"stratum_type": label, "stratum": str(k),
                             "coverage_pct": float(tab["mean"].iloc[i] * 100),
                             "n_rows": int(tab["size"].iloc[i])})
        fig.suptitle(f"{v.spec_name} ({v.dataset}) — coverage, §2.1", fontsize=10)
        fig.tight_layout()
        stem = f"{v.spec_name}_coverage"
        save_fig(fig, vardir(v, "figs"), stem)
        save_data(pd.DataFrame(rows), vardir(v, "figs"), stem)
    return res


# ---------------------------------------------------------------- §2.2 value counts
def value_counts(v, lev, levels, res):
    vc = lev.value_counts().reindex(levels).fillna(0).astype(int)
    n = int(vc.sum())
    tab = pd.DataFrame({"level": levels, "n": vc.to_numpy()})
    tab["pct"] = 100.0 * tab["n"] / max(n, 1)
    tab["cum_pct"] = tab["pct"].cumsum()
    tab["observed"] = tab["n"] > 0
    save_table(tab, f"{v.spec_name}_value_counts", subdir=vardir(v, "tables"))

    present = tab[tab["n"] > 0]
    for t in RARE_THRESHOLDS:
        res[f"n_rare_{t}"] = int((present["n"] < t).sum())
    h, hn = shannon(present["n"].to_numpy())
    res["entropy_nats"] = h
    res["normalized_entropy"] = hn
    res["n_distinct_values"] = int(len(present))
    res["n_declared_levels"] = int(len(levels))
    res["max_category_pct"] = float(present["pct"].max()) if len(present) else np.nan
    res["unobserved_levels"] = ", ".join(_num(x) for x in tab.loc[~tab["observed"], "level"]) or "none"
    res["n_unobserved_levels"] = int((~tab["observed"]).sum())

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.4))
    for ax, logy in zip(axes, (False, True)):
        ax.bar(np.arange(len(tab)), tab["n"].to_numpy(), color=C1)
        ax.set_xticks(np.arange(len(tab)))
        ax.set_xticklabels([_num(x) for x in tab["level"]],
                           rotation=90 if len(tab) > 10 else 0, fontsize=7)
        ax.set_xlabel("level")
        ax.set_ylabel("n observations")
        if logy:
            ax.set_yscale("log")
            for t, c in zip(RARE_THRESHOLDS, (C2, C3, C4)):
                ax.axhline(t, color=c, lw=0.9, ls="--", label=f"n = {t}")
            ax.legend(fontsize=7)
    fig.suptitle(f"{v.spec_name} ({v.dataset}) — level occupancy, §2.2 "
                 f"(linear | log)", fontsize=10)
    fig.tight_layout()
    stem = f"{v.spec_name}_value_counts"
    save_fig(fig, vardir(v, "figs"), stem)
    save_data(tab, vardir(v, "figs"), stem)
    return res, tab


# ---------------------------------------------------------------- §2.3 ceiling / floor
def ceiling_floor(v, tab, res):
    present = tab[tab["n"] > 0].reset_index(drop=True)
    if present.empty:
        res.update(pct_at_floor=np.nan, pct_at_ceiling=np.nan,
                   pct_bottom3=np.nan, pct_top3=np.nan)
        return res
    res["pct_at_floor"] = float(present["pct"].iloc[0])
    res["pct_at_ceiling"] = float(present["pct"].iloc[-1])
    res["pct_bottom3"] = float(present["pct"].iloc[:3].sum())
    res["pct_top3"] = float(present["pct"].iloc[-3:].sum())
    # the "bad end" depends on which way the scale runs
    worst = res["pct_at_ceiling"] if v.direction == "higher_worse" else res["pct_at_floor"]
    best = res["pct_at_floor"] if v.direction == "higher_worse" else res["pct_at_ceiling"]
    res["pct_at_worst_level"] = worst
    res["pct_at_best_level"] = best
    return res


# ---------------------------------------------------------------- §2.4 within person
def within_person(v, panel, col, lev, res):
    if panel.grain != "visit":
        for k in ["icc", "icc_raw", "rank_transition_rate", "pct_delta_zero",
                  "pct_abs_delta_le1", "pct_delta_improving", "delta_sd",
                  "median_distinct_levels_per_person"]:
            res[k] = np.nan
        return res, None
    d = panel.df
    x = v.clean(d[col])
    icc, k, N, raw = icc1(x, d[panel.person])
    res["icc"], res["icc_raw"], res["icc_n_persons"], res["icc_n_obs"] = icc, raw, k, N

    w = pd.DataFrame({"p": d[panel.person], "o": d[panel.order], "x": x, "lev": lev}).dropna(subset=["x"])
    w = w.sort_values(["p", "o"])
    g = w.groupby("p", sort=False)
    nd = g["lev"].nunique()
    res["median_distinct_levels_per_person"] = float(nd.median())
    res["pct_persons_never_changing"] = float(100.0 * (nd <= 1).mean())

    w["dx"] = g["x"].diff()
    w["dlev"] = g["lev"].diff()
    dd = w["dx"].dropna()
    res["n_adjacent_pairs"] = int(len(dd))
    if len(dd) < 20:
        for k2 in ["rank_transition_rate", "pct_delta_zero", "pct_abs_delta_le1",
                   "pct_delta_improving", "delta_sd"]:
            res[k2] = np.nan
        return res, w
    res["delta_sd"] = float(dd.std())
    res["delta_mean"] = float(dd.mean())
    res["pct_delta_zero"] = float(100.0 * (dd == 0).mean())
    res["pct_abs_delta_le1"] = float(100.0 * (dd.abs() <= 1).mean())
    better = dd > 0 if res.get("direction_measured", v.direction) == "higher_better" else dd < 0
    res["pct_delta_improving"] = float(100.0 * better.mean())
    res["rank_transition_rate"] = float(100.0 * (w["dlev"].dropna() != 0).mean())

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.4))
    ax = axes[0]
    vcnd = nd.value_counts().sort_index()
    ax.bar(vcnd.index.astype(float), vcnd.to_numpy(), color=C1)
    ax.set_xlabel("distinct levels a person ever occupies")
    ax.set_ylabel("n persons")
    ax = axes[1]
    lo, hi = np.nanpercentile(dd, [0.5, 99.5])
    bins = np.histogram_bin_edges(dd.clip(lo, hi), bins=41)
    ax.hist(dd.clip(lo, hi), bins=bins, color=C2)
    ax.axvline(0, color=REF_GREY, lw=1)
    ax.set_xlabel("Δ vs previous visit (0.5-99.5 pct clipped)")
    ax.set_ylabel("n pairs")
    ax = axes[2]
    ax.bar([0, 1, 2], [res["pct_delta_zero"], res["pct_abs_delta_le1"],
                       res["rank_transition_rate"]], color=[C3, C3, C4])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Δ = 0", "|Δ| ≤ 1", "level changed"], fontsize=8)
    ax.set_ylabel("% of adjacent pairs")
    ax.set_ylim(0, 100)
    fig.suptitle(f"{v.spec_name} ({v.dataset}) — within-person variation, §2.4  "
                 f"(ICC = {icc:.3f})", fontsize=10)
    fig.tight_layout()
    stem = f"{v.spec_name}_within_person"
    save_fig(fig, vardir(v, "figs"), stem)
    save_data(pd.DataFrame({"delta": dd.to_numpy()}), vardir(v, "figs"), stem)
    return res, w


# ---------------------------------------------------------------- §2.5 noise floor
def noise_floor(v, panel, col, w, res):
    if panel.grain != "visit" or w is None or panel.mildest is None:
        res.update(noise_floor_sd=np.nan, noise_floor_mean=np.nan, snr=np.nan,
                   n_noise_pairs=0)
        return res
    d = panel.df
    x = v.clean(d[col])
    q = pd.DataFrame({"p": d[panel.person], "o": d[panel.order], "x": x,
                      "mild": d[panel.mildest].fillna(False).astype(bool)}).dropna(subset=["x"])
    q = q.sort_values(["p", "o"])
    # `mild` is a property of the (t, t+1) pair, so it belongs to the row at t; Δ sits at t+1.
    q["dx"] = q.groupby("p", sort=False)["x"].diff()
    q["mild_prev"] = q.groupby("p", sort=False)["mild"].shift(1)
    nd = q.loc[q["mild_prev"].fillna(False).astype(bool), "dx"].dropna()
    res["n_noise_pairs"] = int(len(nd))
    if len(nd) < 30:
        res.update(noise_floor_sd=np.nan, noise_floor_mean=np.nan, snr=np.nan)
        return res
    res["noise_floor_sd"] = float(nd.std())
    res["noise_floor_mean"] = float(nd.mean())
    # If the "both visits are mildest" subset is defined BY this very variable, Δ is 0 by
    # construction and the SNR is undefined rather than infinite. Flag it, don't report it.
    res["noise_floor_circular"] = bool(nd.std() == 0)
    sd_all = res.get("delta_sd", np.nan)
    res["snr"] = float(sd_all / nd.std()) if np.isfinite(sd_all) and nd.std() > 0 else np.nan
    # practice effect: is the mean shift in the STABLE-NORMAL subset toward "better"?
    sign = 1.0 if v.direction == "higher_better" else -1.0
    res["noise_floor_practice_shift"] = float(sign * nd.mean())
    return res


# ---------------------------------------------------------------- §2.6 ordinality
def ordinality(v, panel, lev, levels, res, outcome=None, at_risk=None, outcome_label=None,
               tag="", make_fig=True):
    """P(reference outcome | level), monotonicity, isotonic residual, adjacent separability."""
    d = panel.df
    ocol = outcome or panel.outcome
    acol = at_risk or panel.at_risk
    m = pd.DataFrame({"lev": lev, "y": pd.to_numeric(d[ocol], errors="coerce")})
    m = m[d[acol].fillna(False).astype(bool)].dropna()
    pre = f"ord{tag}_"
    if m.empty or m["lev"].nunique() < 3:
        res.update({pre + k: np.nan for k in
                    ["spearman", "spearman_obs", "isotonic_residual_ratio", "n_levels_tested"]})
        res[pre + "n_indistinguishable_adjacent_pairs"] = np.nan
        return res, None

    grp = m.groupby("lev")["y"].agg(["sum", "size"])
    grp.columns = ["k", "n"]
    grp = grp.reindex([l for l in levels if l in grp.index])
    grp["p"] = grp["k"] / grp["n"]
    ci = [wilson(int(r.k), int(r.n)) for r in grp.itertuples()]
    grp["ci_lo"] = [c[0] for c in ci]
    grp["ci_hi"] = [c[1] for c in ci]
    grp["rank"] = np.arange(len(grp))
    grp = grp.reset_index()

    usable = grp[grp["n"] >= 10]
    if len(usable) < 3:
        usable = grp
    rho = stats.spearmanr(usable["rank"], usable["p"]).statistic
    res[pre + "spearman"] = float(rho)
    res[pre + "n_levels_tested"] = int(len(usable))
    rho_obs = stats.spearmanr(m["lev"], m["y"]).statistic
    res[pre + "spearman_obs"] = float(rho_obs)

    iso = IsotonicRegression(increasing="auto", out_of_bounds="clip")
    wt = usable["n"].to_numpy(float)
    yy = usable["p"].to_numpy(float)
    fit = iso.fit_transform(usable["rank"].to_numpy(float), yy, sample_weight=wt)
    rss = float((wt * (yy - fit) ** 2).sum())
    ybar = float((wt * yy).sum() / wt.sum())
    tss = float((wt * (yy - ybar) ** 2).sum())
    res[pre + "isotonic_residual_ratio"] = float(rss / tss) if tss > 0 else np.nan
    grp["isotonic"] = np.nan
    grp.loc[grp["lev"].isin(usable["lev"]), "isotonic"] = fit

    # ---- adjacent-level separability
    rows = []
    u = usable.reset_index(drop=True)
    for i in range(len(u) - 1):
        a, b = u.iloc[i], u.iloc[i + 1]
        tbl = np.array([[a.k, a.n - a.k], [b.k, b.n - b.k]], float)
        if tbl.min() < 0:
            continue
        exp_ok = (tbl.sum(1)[:, None] * tbl.sum(0)[None, :] / tbl.sum()).min() >= 5 if tbl.sum() else False
        if exp_ok:
            p = float(stats.chi2_contingency(tbl, correction=True).pvalue)
            test = "chi2"
        else:
            p = float(stats.fisher_exact(tbl.astype(int)).pvalue)
            test = "fisher"
        rows.append({"level_lo": a.lev, "level_hi": b.lev, "n_lo": int(a.n), "n_hi": int(b.n),
                     "p_lo": a.p, "p_hi": b.p, "delta_p": b.p - a.p,
                     "test": test, "pvalue": p, "distinguishable": bool(p < 0.05)})
    ap = pd.DataFrame(rows)
    if not ap.empty:
        order = np.argsort(ap["pvalue"].to_numpy())
        mflag = np.zeros(len(ap), bool)
        for j, idx in enumerate(order):                      # Holm step-down
            thr = 0.05 / (len(ap) - j)
            if ap["pvalue"].iloc[idx] < thr:
                mflag[idx] = True
            else:
                break
        ap["distinguishable_holm"] = mflag
        n_bad = int((~ap["distinguishable"]).sum())
        res[pre + "n_adjacent_pairs_tested"] = int(len(ap))
        res[pre + "n_indistinguishable_adjacent_pairs"] = n_bad
        res[pre + "pct_indistinguishable"] = float(100.0 * n_bad / len(ap))
        res[pre + "n_indistinguishable_holm"] = int((~ap["distinguishable_holm"]).sum())
        res[pre + "n_reversals"] = int((ap["delta_p"] * np.sign(rho if rho else 1) < 0).sum())
        save_table(ap, f"{v.spec_name}{tag}_adjacent_pairs", subdir=vardir(v, "tables"))
    else:
        res[pre + "n_indistinguishable_adjacent_pairs"] = np.nan
        res[pre + "pct_indistinguishable"] = np.nan

    if make_fig:
        fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6),
                                 gridspec_kw={"width_ratios": [2, 1]})
        ax = axes[0]
        xs = grp["rank"].to_numpy(float)
        # clip at 0: at p exactly 0 or 1 the Wilson bound and p can differ by float noise
        lo_err = (grp["p"] - grp["ci_lo"]).clip(lower=0).to_numpy()
        hi_err = (grp["ci_hi"] - grp["p"]).clip(lower=0).to_numpy()
        ax.errorbar(xs, grp["p"], yerr=[lo_err, hi_err],
                    fmt="o", color=C1, ms=4, lw=1, capsize=2, label="observed (95% Wilson CI)")
        ok = grp["isotonic"].notna()
        ax.plot(xs[ok], grp.loc[ok, "isotonic"], color=C2, lw=1.6, label="isotonic fit")
        ax.set_xticks(xs)
        ax.set_xticklabels([_num(x) for x in grp["lev"]],
                           rotation=90 if len(grp) > 10 else 0, fontsize=7)
        ax.set_xlabel(f"{v.spec_name} level")
        ax.set_ylabel("P(reference outcome)")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, loc="best")
        ax = axes[1]
        ax.bar(xs, grp["n"], color=REF_GREY)
        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels([_num(x) for x in grp["lev"]],
                           rotation=90 if len(grp) > 10 else 0, fontsize=7)
        ax.set_ylabel("n at level (log)")
        ax.set_xlabel("level")
        fig.suptitle(f"{v.spec_name} ({v.dataset}) — ordinality vs «{outcome_label or panel.outcome_label}»"
                     f"  ρ = {rho:.3f}, isotonic resid = {res[pre + 'isotonic_residual_ratio']:.3f}",
                     fontsize=9)
        fig.tight_layout()
        stem = f"{v.spec_name}{tag}_ordinality"
        save_fig(fig, vardir(v, "figs"), stem)
        save_data(grp, vardir(v, "figs"), stem)
    return res, grp


# ---------------------------------------------------------------- §2.7 direction
def direction(v, panel, col, res):
    d = panel.df
    x = v.clean(d[col])
    y = pd.to_numeric(d[panel.outcome], errors="coerce").where(
        d[panel.at_risk].fillna(False).astype(bool))
    r_out = stats.spearmanr(x, y, nan_policy="omit").statistic
    r_age = stats.spearmanr(x, pd.to_numeric(d[panel.age], errors="coerce"),
                            nan_policy="omit").statistic
    res["spearman_with_outcome"] = float(r_out) if np.isfinite(r_out) else np.nan
    res["spearman_with_age"] = float(r_age) if np.isfinite(r_age) else np.nan
    res["direction_measured"] = ("higher_worse" if r_out > 0 else
                                 "higher_better" if r_out < 0 else "flat")
    res["direction_expected"] = v.direction
    res["direction_matches_spec"] = bool(res["direction_measured"] == v.direction)
    return res


# ---------------------------------------------------------------- verdicts
def verdicts(v, res):
    """Turn the §2 numbers into the two verdict columns the spec's §5 table asks for.

    Every threshold below is the spec's own; the only judgement added is WHICH rho to use.
    Where an at-risk-only (incident) rho exists it wins, because the literal all-pairs rho
    is partly a persistence measure and is inflated by same-visit leakage -- CDRGLOB, the
    six CDR boxes and NACCUDSD all hit exactly 1.000 on the literal definition.
    """
    rho_lit = res.get("ord_spearman", np.nan)
    rho_inc = res.get("ord_incident_spearman", np.nan)
    rho_used, rho_src = ((rho_inc, "incident") if np.isfinite(rho_inc)
                         else (rho_lit, "literal"))
    res["rho_used_for_verdict"] = rho_used
    res["rho_source"] = rho_src
    rho = abs(rho_used)
    resid = res.get("ord_incident_isotonic_residual_ratio", np.nan)
    if not np.isfinite(resid):
        resid = res.get("ord_isotonic_residual_ratio", np.nan)
    n_bad = res.get("ord_n_indistinguishable_adjacent_pairs", np.nan)
    n_tested = res.get("ord_n_adjacent_pairs_tested", np.nan)

    if not np.isfinite(rho):
        sm = "unknown"
    elif rho > 0.9 and np.isfinite(resid) and resid < 0.1:
        sm = "ok"
    elif rho >= 0.6:
        sm = "caution"
    else:
        sm = "forbidden"
    res["smoothing_verdict"] = sm
    res["resolution_excess"] = bool(np.isfinite(n_bad) and np.isfinite(n_tested)
                                    and n_tested > 0 and n_bad > n_tested / 2)

    ne = res.get("normalized_entropy", np.nan)
    mx = res.get("max_category_pct", np.nan)
    icc = res.get("icc", np.nan)
    rtr = res.get("rank_transition_rate", np.nan)
    dz = res.get("pct_delta_zero", np.nan)
    miss = res.get("pct_missing", np.nan)
    # §2.3 is measured at BOTH ends: a pile-up at the healthy end is a floor effect and
    # costs exactly as much resolution as one at the sick end.
    pile = np.nanmax([res.get("pct_at_floor", np.nan), res.get("pct_at_ceiling", np.nan)])
    vacuous = bool(res.get("levels_are_quantile_bins", False))
    res["extreme_level_pile_pct"] = np.nan if vacuous else pile
    res["ceiling_test_vacuous"] = vacuous

    reasons = []
    if v.role == "outcome_only":
        role = "outcome_only"
        reasons.append("spec §1.3 leakage blacklist -- autopsy variable, never an input")
    elif (np.isfinite(ne) and ne < 0.3) or (np.isfinite(mx) and mx > 70):
        role = "drop"
        reasons.append(f"near-constant token (normalised entropy {ne:.2f}, "
                       f"largest level {mx:.1f}% of observations)")
    elif np.isfinite(icc) and icc > 0.85 and np.isfinite(rtr) and rtr < 10:
        role = "static_covariate"
        reasons.append(f"ICC {icc:.2f} with only {rtr:.1f}% of adjacent pairs changing level")
    else:
        fails = []
        if sm == "forbidden":
            fails.append(f"ordinality fails (|rho| {rho:.2f} < 0.6, {rho_src})")
        if not vacuous and np.isfinite(pile) and pile > 30:
            fails.append(f"{pile:.1f}% of observations sit at one extreme level")
        if np.isfinite(rtr) and rtr < 30:
            fails.append(f"only {rtr:.1f}% of adjacent visit pairs change level")
        if np.isfinite(dz) and dz > 40:
            fails.append(f"Delta = 0 in {dz:.1f}% of adjacent pairs")
        if np.isfinite(miss) and miss > 60:
            fails.append(f"{miss:.1f}% missing")
        if v.grain != "visit":
            fails.append("person-grain only -- cannot carry a per-visit signal")
        if fails:
            role = "auxiliary"
            reasons.extend(fails)
        else:
            role = "backbone_candidate"
            reasons.append(f"|rho| {rho:.2f} ({rho_src}), {rtr:.0f}% level transitions, "
                           f"largest extreme level {pile:.1f}%, {miss:.1f}% missing")
    res["role_verdict"] = role
    res["verdict_reasons"] = "; ".join(reasons)
    return res


# ---------------------------------------------------------------- driver
def analyze_ordinal(v, panel):
    """Run §2.1-§2.7 for one variable. Returns the SCALE_SUMMARY row as a dict."""
    col = v.column
    res = {"varname": v.spec_name, "column_used": col, "dataset": v.dataset,
           "panel": panel.name, "grain": v.grain, "group": v.group,
           "expected_range": v.expected_str(),
           "reference_outcome": panel.outcome_label,
           "sentinels": ", ".join(_num(s) for s in sorted(v.sentinels)) or "none"}
    lev, levels, meta = levelize(v, panel.df[col])
    res.update(meta)
    res = direction(v, panel, col, res)                     # first: §2.4 needs the direction
    res = coverage(v, panel, col, res)
    res, tab = value_counts(v, lev, levels, res)
    res = ceiling_floor(v, tab, res)
    res, w = within_person(v, panel, col, lev, res)
    res = noise_floor(v, panel, col, w, res)
    res, grp = ordinality(v, panel, lev, levels, res)
    if panel.incident is not None and panel.at_risk_incident is not None:
        res, _ = ordinality(v, panel, lev, levels, res, outcome=panel.incident,
                            at_risk=panel.at_risk_incident, tag="_incident",
                            outcome_label=panel.outcome_label + ", AT-RISK ROWS ONLY")
    res = verdicts(v, res)
    return res, grp
