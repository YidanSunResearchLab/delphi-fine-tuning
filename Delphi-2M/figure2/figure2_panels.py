"""
figure2_panels.py -- **Figure 2** of the AD-progression paper, built panel-by-panel and then
assembled.

  Panel a  Transition-prediction accuracy .... AUC per transition type + observed vs predicted
                                               transition probability side by side
  Panel b  Timing accuracy by interval ....... error in predicted event time binned by how far
                                               ahead the event was, + discrimination vs horizon
  Panel c  Per-individual trajectories ....... observed vs predicted stage-at-age, Jaccard index
                                               against a carry-baseline-forward reference,
                                               per-state IoU, and illustrative patients
  Panel d  Embedding structure ............... UMAP of the model's patient embeddings, coloured by
                                               observed trajectory class, with k-NN purity

Every panel draws into a `SubplotSpec` of a caller-supplied figure, so the SAME code produces the
standalone panel figures and the assembled Figure 2 (no duplicated plotting logic).

All numbers come from `figure2_core`'s cache (Monte-Carlo trajectories seeded on each patient's
baseline visit, competing-risk and right-censoring aware). Run `python figure2_core.py --build`
first.

CLI:
    python figure2_panels.py                 # all four standalone panels + the combined figure
    python figure2_panels.py --panel a       # one panel
    python figure2_panels.py --no-combined
    python figure2_panels.py --dataset nacc-dedup-s7   # a split built with a different seed

Or run the whole pipeline (core --build + both cohorts + combined) in one step: ./run_figure2.sh
"""
import os, sys, json, time, argparse, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Delphi-2M/ (package root)
sys.path.insert(0, HERE)
from figure2 import plotting_style as ps   # noqa: E402
from figure2 import figure2_core as F2     # noqa: E402

log = logging.getLogger("fig2")

OUT = F2.OUT_DIR
H = F2.PRIMARY_H                     # primary horizon (years)
# Set by main() from --cohort. "matched" (the default and the paper figure) restricts every panel
# to the population train.py actually fitted on; "all" scores the whole test split and writes to
# _allcohort-suffixed files so the two never overwrite each other.
SUFFIX = ""
COHORT_NOTE = ""
N_BOOT = 500
MIN_POS = 30                         # a transition type needs this many observed events to be shown
RNG_SEED = 42

STATE_COLORS = list(ps.SEVERITY_COLORS[:4]) + ["#4d4d4d"]        # Normal..Dementia, Death
TRAJ_COLORS = {
    "Stable": "#0072B2", "Improved": "#009E73", "Progressed (Imp./MCI)": "#E69F00",
    "Progressed to Dementia": "#D55E00", "Died, no progression": "#999999",
}
EP_COLORS = {"Reach ≥MCI": "#56B4E9", "Dementia": "#CC79A7", "Death": "#D55E00"}
TIME_BINS = [(0, 2), (2, 5), (5, 10), (10, 20)]
TIME_BIN_LABELS = ["0–2 y", "2–5 y", "5–10 y", ">10 y"]


# =========================================================================== small utilities
def _auc(y, s):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else np.nan


def boot_auc(y, s, n_boot=N_BOOT, seed=RNG_SEED):
    """Point AUC + 95% bootstrap CI. One row == one patient here, so a plain row bootstrap IS the
    patient-level bootstrap."""
    pt = _auc(y, s)
    if not np.isfinite(pt):
        return pt, np.nan, np.nan
    rng = np.random.default_rng(seed)
    n = len(y); b = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, n)
        if len(np.unique(y[ix])) == 2:
            b.append(_auc(y[ix], s[ix]))
    if not b:
        return pt, np.nan, np.nan
    return pt, float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_mean(x, n_boot=N_BOOT, seed=RNG_SEED, fn=np.mean):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    b = [fn(x[rng.integers(0, x.size, x.size)]) for _ in range(n_boot)]
    return float(fn(x)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def panel_letter(ax, letter):
    """Bold panel tag, used only in the assembled figure (standalone panels say it in the title)."""
    if not letter:
        return
    ax.text(-0.12, 1.10, letter, transform=ax.transAxes, fontsize=16, fontweight="bold",
            va="bottom", ha="right")


def _save(fig, stem, data=None):
    stem = stem + SUFFIX
    pdf, png = ps.save_fig(fig, OUT, stem)
    if data is not None:
        ps.save_data(data, OUT, stem)
    log.info("  saved %s", os.path.basename(png))


# =========================================================================== PANEL A
def transition_types():
    """The (from -> to) transitions worth evaluating, ordered by origin severity then target.

    `to` == 4 is Death.  A patient is AT RISK for from->to if their baseline NACCUDSD state is
    `from`; reversions (e.g. MCI -> Normal) are included because the data contains them.

    NOTE ON SEMANTICS: "from -> to" means *reaching* `to` within the horizon starting from a
    baseline state of `from` -- by ANY path, so Normal->Dementia includes patients who passed
    through MCI first. The strictly one-step (next distinct state) version is the supplementary
    transition matrix, not this panel."""
    out = []
    for f in range(4):
        for t in list(range(4)) + [4]:
            if t == f:
                continue
            out.append((f, t))
    return out


def compute_A(cache, horizon=H):
    """Per-transition-type discrimination and calibration.

    DISCRIMINATION (AUC) uses the censoring/competing-risk aware binary label, i.e. patients whose
    status at `horizon` is unknown (censored early, no event) are dropped.
    CALIBRATION compares the model's mean predicted risk over the WHOLE at-risk cohort against the
    Aalen-Johansen cumulative incidence on that same cohort -- NOT the naive event proportion,
    which is biased upwards because it silently drops the early-censored (mostly event-free)
    patients. The naive proportion is still recorded in the source CSV for reference.
    """
    df, grids = cache["df"], cache["grids"]
    b = df["baseline_state"].to_numpy()
    rows = []
    eps = {t: F2.composite(df, grids, [t]) for t in range(5)}
    for (f, t) in transition_types():
        ep = eps[t]
        at_risk = (b == f)
        y_all = F2.labels_at_h(ep, horizon)
        s_all = ep["risk"][horizon]
        m = at_risk & (y_all >= 0) & np.isfinite(s_all)
        y, s = y_all[m].astype(int), s_all[m]
        n_pos = int(y.sum())
        if n_pos < MIN_POS or len(y) - n_pos < MIN_POS:
            continue
        auc, lo, hi = boot_auc(y, s)
        tt, et = F2.cr_times(ep, at_risk)
        cif = F2.aalen_johansen(tt, et, horizon)
        pred = float(np.nanmean(s_all[at_risk]))
        rows.append(dict(
            frm=F2.ALL_NAMES[f], to=F2.ALL_NAMES[t], label=f"{F2.ALL_NAMES[f]}→{F2.ALL_NAMES[t]}",
            from_idx=f, to_idx=t, horizon=horizon, n_at_risk=int(at_risk.sum()),
            n_scored=int(m.sum()), n_events=n_pos, auc=auc, auc_lo=lo, auc_hi=hi,
            obs_cif=cif, pred_rate=pred, obs_naive_rate=float(y.mean())))
    return pd.DataFrame(rows)


def compute_A_time(cache):
    """Observed vs MC-predicted TIME to the first transition out of the baseline state.

    Cohort = patients whose first post-baseline event is a cognitive state CHANGE seen before
    death. Death is a competing risk, not a transition, so death-first patients are excluded
    rather than entered as very late transitions; patients who never transition have no observed
    time to compare against at all.

    Predicted time = median over the MC trajectories that transition before their own sampled
    death. Both sides therefore condition on "a transition happened" -- which is what makes them
    comparable, but also means this plot says nothing about WHETHER one occurs (that is a1/a2).
    It only asks: given a transition, does the model put it at the right time?
    """
    df, grids = cache["df"], cache["grids"]
    b = df["baseline_state"].to_numpy()
    fp = grids["fp"]                                     # (N, 5, n_mc), years after baseline
    obs_all = np.stack([df[f"obs_t_{nm}"].to_numpy(float) for nm in F2.ALL_NAMES], 1)   # (N, 5)
    n = len(df)
    obs = np.full(n, np.nan); pred = np.full(n, np.nan); n_used = np.zeros(n, int)
    for i in range(n):
        cog = [s for s in range(4) if s != b[i]]         # every cognitive state but the current one
        o = float(np.min(obs_all[i, cog])); d = obs_all[i, 4]
        if not np.isfinite(o) or (np.isfinite(d) and d < o):
            continue
        t_sim = fp[i, cog, :].min(0)
        ok = np.isfinite(t_sim) & (t_sim <= fp[i, 4, :])
        if not ok.any():
            continue
        obs[i] = o; pred[i] = float(np.median(t_sim[ok])); n_used[i] = int(ok.sum())
    m = np.isfinite(obs) & np.isfinite(pred)
    o, p = obs[m], pred[m]
    err = p - o
    ss = float(np.sum((o - o.mean()) ** 2))
    stats = dict(n=int(m.sum()),
                 r2=float(1 - np.sum(err ** 2) / ss) if ss > 0 else np.nan,
                 mae=float(np.mean(np.abs(err))), rmse=float(np.sqrt(np.mean(err ** 2))),
                 bias=float(np.mean(err)),
                 loa=[float(np.mean(err) - 1.96 * np.std(err)),
                      float(np.mean(err) + 1.96 * np.std(err))],
                 spearman=float(pd.Series(o).corr(pd.Series(p), method="spearman")))
    tab = pd.DataFrame(dict(pid=df["pid"].to_numpy()[m],
                            baseline_state=[F2.STATE_NAMES[i] for i in b[m]],
                            observed_years=o, predicted_years=p, error_years=err,
                            n_mc_transitioned=n_used[m]))
    return dict(obs=o, pred=p, stats=stats, tab=tab)


def _panel_A_time(ax, AT):
    """a3: predicted vs observed time to the first transition, with the binned median trend.

    The trend line is what carries the message -- a cloud around the identity line looks fine at
    a glance even when the model is systematically late, and here it is (see the bias in the
    annotation). Axes are clipped at the 99th percentile so a handful of very long simulated
    times cannot squash the bulk of the data into a corner."""
    o, p, st = AT["obs"], AT["pred"], AT["stats"]
    if st["n"] == 0:
        ax.text(0.5, 0.5, "no patient has both an observed and a\npredicted transition time",
                ha="center", va="center", fontsize=9, color="#D55E00")
        ax.set_axis_off()
        return
    hi = float(np.ceil(max(np.percentile(o, 99), np.percentile(p, 99)) / 2) * 2)
    ax.plot([0, hi], [0, hi], ls="--", color="0.35", lw=1.2, zorder=1, label="identity")
    ax.scatter(o, p, s=6, alpha=0.25, linewidths=0, color="#2c7fb8", rasterized=True, zorder=2)

    edges = np.arange(0, hi + 1e-9, 2.0)
    bi = np.clip(np.digitize(o, edges) - 1, 0, len(edges) - 2)
    xs, ys = [], []
    for k in range(len(edges) - 1):
        m = bi == k
        if m.sum() >= 20:                              # don't draw a "median" of a handful
            xs.append((edges[k] + edges[k + 1]) / 2); ys.append(float(np.median(p[m])))
    if xs:
        ax.plot(xs, ys, "-o", color="#D55E00", lw=1.8, ms=4.5, zorder=3,
                label="median predicted (2-y bins)")
    ax.set_xlim(0, hi); ax.set_ylim(0, hi)
    ax.set_xlabel("observed time to transition (years)")
    ax.set_ylabel("predicted time (years)")
    ax.set_title("Predicted vs observed transition time\n"
                 f"among the {st['n']:,} patients who transitioned", fontsize=10.5)
    ax.text(0.035, 0.965,
            f"R² = {st['r2']:.2f}\nMAE = {st['mae']:.2f} y\nRMSE = {st['rmse']:.2f} y\n"
            f"mean error = {st['bias']:+.2f} y",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.32", fc="white", ec="0.75", alpha=0.9))
    ax.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.9)


def panel_A(fig, spec, cache, letter="a", horizon=H):
    if "A" not in cache:
        cache["A"] = compute_A(cache, horizon)
    if "A_time" not in cache:
        cache["A_time"] = compute_A_time(cache)
        ps.save_data(cache["A_time"]["tab"], OUT, "fig2a_transition_time" + SUFFIX)  # per-patient
    tab = cache["A"]
    AT = cache["A_time"]
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, wspace=0.52,
                                 width_ratios=[1.0, 1.0, 0.92])
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    _panel_A_time(ax3, AT)
    if tab.empty:
        for ax in (ax1, ax2):
            ax.text(0.5, 0.5, f"no transition type reaches {MIN_POS} events at {horizon} y",
                    ha="center", va="center", fontsize=10, color="#D55E00")
            ax.set_axis_off()
        return dict(horizon=horizon, n_transition_types=0, note="no transition type met MIN_POS",
                    transition_time=AT["stats"]), tab

    t = tab.sort_values(["from_idx", "to_idx"], ascending=[False, False]).reset_index(drop=True)
    ypos = np.arange(len(t))
    colors = [STATE_COLORS[i] for i in t["from_idx"]]

    # ---- a1: discrimination
    x0 = min(0.45, np.floor(float(np.nanmin(t["auc_lo"])) * 20) / 20)
    ax1.barh(ypos, np.maximum(t["auc"], x0), left=0, color=colors, height=0.68, edgecolor="white",
             xerr=[t["auc"] - t["auc_lo"], t["auc_hi"] - t["auc"]],
             error_kw=dict(ecolor="0.25", lw=1.1, capsize=2.5))
    ax1.axvline(0.5, ls=":", color="0.35", lw=1.2)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([f"{r.label}  (n={r.n_events})" for r in t.itertuples()], fontsize=8.5)
    ax1.set_xlim(x0, 1.06)
    ax1.set_xticks(np.arange(np.ceil(x0 * 10) / 10, 1.001, 0.1))
    ax1.set_xlabel(f"AUC for reaching the target state within {horizon} y")
    # the "any path" semantics live in transition_types()'s docstring and the README, not the title
    ax1.set_title("Discrimination per transition type", fontsize=10.5)
    ax1.grid(axis="y", alpha=0)
    for i, r in enumerate(t.itertuples()):
        ax1.text(1.055, i, f"{r.auc:.2f}", va="center", ha="right", fontsize=8)
    # no colour legend needed: the y labels name the origin state, colour only groups them

    # ---- a2: observed vs predicted transition probability
    hgt = 0.36
    ax2.barh(ypos + hgt / 2, t["obs_cif"], height=hgt, color="0.25", edgecolor="white",
             label="observed (Aalen–Johansen)")
    ax2.barh(ypos - hgt / 2, t["pred_rate"], height=hgt, color=ps.OUTCOME_COLORS["NACCUDSD"],
             edgecolor="white", label="predicted (Monte-Carlo)")
    ax2.set_yticks(ypos); ax2.set_yticklabels(t["label"], fontsize=8.5)
    ax2.set_xlabel(f"P(transition within {horizon} y)")
    cal = float(np.mean(t["pred_rate"] - t["obs_cif"]))
    ax2.set_title(f"Observed vs predicted transition rate\nmean predicted − observed = {cal:+.3f}",
                  fontsize=10.5)
    ax2.grid(axis="y", alpha=0)
    ax2.legend(fontsize=7.5, loc="lower right", frameon=True, framealpha=0.9)
    mx = float(max(t["obs_cif"].max(), t["pred_rate"].max())) * 1.20
    ax2.set_xlim(0, mx)

    panel_letter(ax1, letter)
    med = float(np.nanmedian(t["auc"]))
    metrics = dict(horizon=horizon, n_transition_types=int(len(t)), median_auc=round(med, 4),
                   mean_calibration_error=round(cal, 4),
                   per_transition={r.label: dict(n_at_risk=int(r.n_at_risk), n_events=int(r.n_events),
                                                 auc=round(r.auc, 4), ci=[round(r.auc_lo, 4), round(r.auc_hi, 4)],
                                                 observed_cif=round(r.obs_cif, 4),
                                                 predicted=round(r.pred_rate, 4))
                                    for r in t.itertuples()},
                   transition_time={k: (round(v, 4) if isinstance(v, float) else v)
                                    for k, v in AT["stats"].items()})
    return metrics, t


# =========================================================================== PANEL B
def endpoints_B(df, grids):
    b = df["baseline_state"].to_numpy()
    return [
        ("Reach ≥MCI", F2.composite(df, grids, [2, 3]), b <= 1),
        ("Dementia",        F2.composite(df, grids, [3]),    b <= 2),
        ("Death",           F2.composite(df, grids, [4]),    np.ones(len(df), bool)),
    ]


def compute_B(cache):
    df, grids = cache["df"], cache["grids"]
    err_rows, auc_rows = [], []
    for name, ep, at_risk in endpoints_B(df, grids):
        obs, pred, d = ep["obs_time"], ep["pred_time"], ep["d_obs"]
        # timing error is only defined for patients who ACTUALLY had the event (and were at risk)
        m = at_risk & np.isfinite(obs) & np.isfinite(pred) & (~np.isfinite(d) | (obs <= d))
        o, p = obs[m], pred[m]
        med_all = float(np.median(o)) if o.size else np.nan
        for (lo, hi), lab in zip(TIME_BINS, TIME_BIN_LABELS):
            k = (o > lo) & (o <= hi)
            if k.sum() < 10:
                continue
            e = p[k] - o[k]
            mae, mae_lo, mae_hi = boot_mean(np.abs(e))
            err_rows.append(dict(endpoint=name, bin=lab, bin_lo=lo, bin_hi=hi, n=int(k.sum()),
                                 mae=mae, mae_lo=mae_lo, mae_hi=mae_hi,
                                 bias=float(np.mean(e)), median_bias=float(np.median(e)),
                                 naive_mae=float(np.mean(np.abs(med_all - o[k]))),
                                 errors=e))
        for h in F2.HORIZONS:
            y = F2.labels_at_h(ep, h); s = ep["risk"][h]
            kk = at_risk & (y >= 0) & np.isfinite(s)
            yy, ss = y[kk].astype(int), s[kk]
            if len(np.unique(yy)) < 2 or yy.sum() < 10:
                continue
            auc, alo, ahi = boot_auc(yy, ss)
            auc_rows.append(dict(endpoint=name, horizon=h, n=int(kk.sum()), n_events=int(yy.sum()),
                                 auc=auc, auc_lo=alo, auc_hi=ahi))
    return pd.DataFrame(err_rows), pd.DataFrame(auc_rows)


def panel_B(fig, spec, cache, letter="b"):
    if "B" not in cache:
        cache["B"] = compute_B(cache)
    err, aucs = cache["B"]
    gs = GridSpecFromSubplotSpec(1, 2, subplot_spec=spec, wspace=0.30, width_ratios=[1.35, 1.0])
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    names = [n for n in EP_COLORS if (err["endpoint"] == n).any()]
    nb = len(TIME_BIN_LABELS); width = 0.8 / max(len(names), 1)
    whisk = []
    for gi, name in enumerate(names):
        sub = err[err["endpoint"] == name]
        for _, r in sub.iterrows():
            bi = TIME_BIN_LABELS.index(r["bin"])
            x = bi + (gi - (len(names) - 1) / 2) * width
            bp = ax1.boxplot([r["errors"]], positions=[x], widths=width * 0.82, showfliers=False,
                             patch_artist=True, medianprops=dict(color="black", lw=1.2),
                             whiskerprops=dict(color="0.4"), capprops=dict(color="0.4"))
            for b in bp["boxes"]:
                b.set(facecolor=EP_COLORS[name], alpha=0.75, edgecolor="white")
            ax1.plot([x], [r["mae"]], marker="D", ms=4.5, color="black", zorder=5)
            ax1.plot([x], [r["naive_mae"]], marker="_", ms=9, color="0.45", zorder=5)
            e = np.asarray(r["errors"], float)
            q = np.percentile(e, [25, 75]); iqr = q[1] - q[0]
            inl = e[(e >= q[0] - 1.5 * iqr) & (e <= q[1] + 1.5 * iqr)]   # matplotlib's whiskers
            whisk += [float(inl.min()), float(inl.max()), r["mae"], r["naive_mae"]]
    ax1.axhline(0, ls=":", color="0.3", lw=1.2)
    ax1.set_xticks(range(nb)); ax1.set_xticklabels(TIME_BIN_LABELS)
    ax1.set_xlabel("observed time to event (interval)")
    ax1.set_ylabel("predicted − observed (years)")
    ax1.set_title("Timing error by how far ahead the event was", fontsize=10.5)
    ax1.set_xlim(-0.55, nb - 0.45)
    if whisk:
        lo, hi = float(np.min(whisk)), float(np.max(whisk)); span = hi - lo
        ax1.set_ylim(lo - 0.08 * span, hi + 0.38 * span)     # headroom for the legend
    handles = [Patch(facecolor=EP_COLORS[n], alpha=0.75, label=n) for n in names]
    handles += [Line2D([], [], marker="D", ls="", color="black", ms=5, label="MAE"),
                Line2D([], [], marker="_", ls="", color="0.45", ms=9,
                       label="MAE of a constant = cohort median")]
    ax1.legend(handles=handles, fontsize=7.3, loc="upper left", ncol=2, frameon=True,
               framealpha=0.92, borderpad=0.4)

    for name in names:
        sub = aucs[aucs["endpoint"] == name].sort_values("horizon")
        if sub.empty:
            continue
        ax2.errorbar(sub["horizon"], sub["auc"],
                     yerr=[sub["auc"] - sub["auc_lo"], sub["auc_hi"] - sub["auc"]],
                     marker="o", ms=5, lw=1.8, capsize=2.5, color=EP_COLORS[name], label=name)
    ax2.axhline(0.5, ls=":", color="0.35", lw=1.2)
    ax2.set_xticks(F2.HORIZONS); ax2.set_xlabel("prediction horizon (years)")
    ylo = min(0.45, float(np.nanmin(aucs["auc_lo"])) - 0.02) if len(aucs) else 0.45
    ax2.set_ylabel("AUC"); ax2.set_ylim(ylo, 1.0)
    ax2.set_title("Discrimination by horizon", fontsize=10.5)
    ax2.legend(fontsize=8, loc="lower left", frameon=True, framealpha=0.9)

    panel_letter(ax1, letter)
    out = dict(timing_error={}, auc_by_horizon={})
    for _, r in err.iterrows():
        out["timing_error"].setdefault(r["endpoint"], {})[r["bin"]] = dict(
            n=int(r["n"]), mae=round(float(r["mae"]), 3), mae_ci=[round(float(r["mae_lo"]), 3),
            round(float(r["mae_hi"]), 3)], bias=round(float(r["bias"]), 3),
            naive_mae=round(float(r["naive_mae"]), 3))
    for _, r in aucs.iterrows():
        out["auc_by_horizon"].setdefault(r["endpoint"], {})[str(int(r["horizon"]))] = round(float(r["auc"]), 4)
    tab = pd.concat([err.drop(columns=["errors"]).assign(kind="timing_error"),
                     aucs.assign(kind="auc_by_horizon")], ignore_index=True)
    return out, tab


# =========================================================================== PANEL C
def compute_C(cache, min_points=3):
    """Per-patient trajectory overlap between the observed and the MC-predicted stage-at-age.

    The grid runs 1..15 years after baseline (year 0 is excluded: both series equal the observed
    baseline state there by construction, so including it would inflate every score).  A grid
    point counts only while the patient is still under observation, or once they are known dead.
    """
    grids = cache["grids"]; df = cache["df"]
    gy = grids["grid_years"]; obs = grids["obs"]; known = grids["known"]; occ = grids["occ"]
    use = gy >= 1
    O = obs[:, use]; K = known[:, use]; OC = occ[:, use, :]
    P = OC.argmax(2).astype(np.int8)                                  # modal predicted state
    base = df["baseline_state"].to_numpy()[:, None].astype(np.int8)
    B = np.repeat(base, O.shape[1], axis=1)                           # carry-baseline-forward ref

    nvalid = K.sum(1)
    ok = nvalid >= min_points
    agree = ((O == P) & K).sum(1)
    agree_b = ((O == B) & K).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        jac = np.where(ok, agree / np.maximum(2 * nvalid - agree, 1), np.nan)
        jac_b = np.where(ok, agree_b / np.maximum(2 * nvalid - agree_b, 1), np.nan)
        acc = np.where(ok, agree / np.maximum(nvalid, 1), np.nan)
        acc_b = np.where(ok, agree_b / np.maximum(nvalid, 1), np.nan)
    # ALIVE-ONLY variant. Once a patient dies, every remaining grid year is "Death" in the
    # observed series, so a correct (or incorrect) death prediction can dominate the whole
    # trajectory score. Restricting to the years the patient was observed alive isolates
    # agreement on the COGNITIVE trajectory. Reported alongside, never instead of, the full one.
    KA = K & (O <= 3)
    nalive = KA.sum(1)
    ok_a = nalive >= min_points
    agree_a = ((O == P) & KA).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        jac_alive = np.where(ok_a, agree_a / np.maximum(2 * nalive - agree_a, 1), np.nan)

    # per-state IoU pooled over every valid (patient, year) cell
    iou = {}
    for s in range(F2.NSTATE):
        tp = ((O == s) & (P == s) & K).sum()
        fp = ((O != s) & (P == s) & K).sum()
        fn = ((O == s) & (P != s) & K).sum()
        iou[F2.ALL_NAMES[s]] = dict(iou=float(tp / max(tp + fp + fn, 1)), n_obs=int(((O == s) & K).sum()),
                                    n_pred=int(((P == s) & K).sum()))
    # agreement as a function of years ahead
    by_year = []
    for j, y in enumerate(gy[use]):
        k = K[:, j]
        if k.sum() < 20:
            continue
        by_year.append(dict(year=int(y), n=int(k.sum()),
                            agreement=float((O[k, j] == P[k, j]).mean()),
                            agreement_baseline=float((O[k, j] == B[k, j]).mean())))
    return dict(jac=jac, jac_b=jac_b, acc=acc, acc_b=acc_b, ok=ok, nvalid=nvalid,
                jac_alive=jac_alive, ok_alive=ok_a, nalive=nalive,
                O=O, P=P, K=K, OC=OC, years=gy[use], iou=iou, by_year=pd.DataFrame(by_year))


def _pick_examples(cache, C, n=4):
    """Four deterministic, clearly-labelled illustrative patients: two the model gets right (a
    cognitive progressor and a stable patient) and the two characteristic failure modes.

    Selection is made on COGNITIVE cells only (both series in Normal..Dementia) so that a
    death-vs-alive mismatch is not mislabelled as an over/under-prediction of cognitive decline.
    All four are chosen by a fixed rule with deterministic tie-breaking -- they illustrate, they
    are not evidence; the aggregate statistics on the right are."""
    df = cache["df"]; O, P, K = C["O"], C["P"], C["K"]
    N = len(df)
    ok = C["ok"] & (C["nvalid"] >= 6)
    cog = K & (O <= 3) & (P <= 3)                      # cells where both series are cognitive
    ncog = cog.sum(1)
    died = df["died"].to_numpy().astype(bool)
    prog_obs = np.zeros(N, bool); prog_pred = np.zeros(N, bool)
    over = np.zeros(N); under = np.zeros(N)
    for i in range(N):
        kc = K[i] & (O[i] <= 3)
        if kc.any():
            prog_obs[i] = O[i][kc].max() > O[i][kc].min()
        kp = P[i] <= 3
        if kp.any():
            prog_pred[i] = P[i][kp].max() > P[i][kp].min()
        if ncog[i] >= 3:
            over[i] = (P[i][cog[i]] > O[i][cog[i]]).mean()
            under[i] = (P[i][cog[i]] < O[i][cog[i]]).mean()
    j = np.where(np.isfinite(C["jac"]), C["jac"], -1)
    picks = []

    def take(mask, key, name):
        cand = [c for c in np.where(mask & ok)[0] if c not in [p[0] for p in picks]]
        if not cand:
            return
        cand.sort(key=lambda i: (-key[i], -C["nvalid"][i], int(df["pid"].iloc[i])))
        picks.append((cand[0], name))

    # "matched decline" must be matched on the DECLINE, not just on the stable years around it
    take(prog_obs & prog_pred, j, "cognitive decline, matched")
    take(~prog_obs & ~prog_pred & ~died, j, "stable, matched")
    take((over > 0.3) & (ncog >= 4), over * ncog / max(ncog.max(), 1), "over-predicts decline")
    take((under > 0.3) & (ncog >= 4), under * ncog / max(ncog.max(), 1), "under-predicts decline")
    return picks[:n]


def panel_C(fig, spec, cache, letter="c"):
    if "C" not in cache:
        cache["C"] = compute_C(cache)
    C = cache["C"]
    df = cache["df"]
    # 6 virtual rows so the two example rows (3 each) line up with three summary cells (2 each)
    gs = GridSpecFromSubplotSpec(6, 3, subplot_spec=spec, wspace=0.42, hspace=1.9,
                                 width_ratios=[1.0, 1.0, 1.25])
    ex_axes = [fig.add_subplot(gs[0:3, 0]), fig.add_subplot(gs[0:3, 1]),
               fig.add_subplot(gs[3:6, 0]), fig.add_subplot(gs[3:6, 1])]
    ax_j = fig.add_subplot(gs[0:2, 2])
    ax_y = fig.add_subplot(gs[2:4, 2])
    ax_i = fig.add_subplot(gs[4:6, 2])

    # ---- illustrative patients
    yrs = C["years"]
    picks = _pick_examples(cache, C)
    for ax in ex_axes[len(picks):]:                 # a rule may find no candidate for some slot
        ax.set_axis_off()
    for ax, (i, tag) in zip(ex_axes, picks):
        k = C["K"][i]
        ax.imshow(C["OC"][i].T, aspect="auto", origin="lower", cmap="Greys", vmin=0, vmax=1,
                  extent=[yrs[0] - 0.5, yrs[-1] + 0.5, -0.5, F2.NSTATE - 0.5], alpha=0.85,
                  interpolation="nearest", zorder=0)
        # the predicted line is nudged up by 0.10 so it stays visible where it exactly overlays
        # the observed one (a perfect match would otherwise be invisible)
        ax.step(yrs[k], C["O"][i][k], where="mid", color="#D55E00", lw=2.2, marker="o", ms=4,
                label="observed", zorder=3)
        ax.step(yrs, C["P"][i] + 0.10, where="mid", color="#0072B2", lw=1.8, ls="--", marker="s",
                ms=3, label="predicted (modal, +0.1 offset)", zorder=2)
        ax.set_ylim(-0.5, F2.NSTATE - 0.5); ax.set_yticks(range(F2.NSTATE))
        ax.set_yticklabels(F2.ALL_NAMES, fontsize=7)
        ax.set_xlim(yrs[0] - 0.5, yrs[-1] + 0.5)
        ax.set_title(f"patient {int(df['pid'].iloc[i])} — {tag}\nJaccard = {C['jac'][i]:.2f}",
                     fontsize=8.5)
        ax.grid(alpha=0.15)
    for ax in ex_axes[2:]:
        ax.set_xlabel("years after baseline", fontsize=9)
    ex_axes[0].legend(handles=[
        Line2D([], [], color="#D55E00", lw=2.2, marker="o", ms=4, label="observed"),
        Line2D([], [], color="#0072B2", lw=1.8, ls="--", marker="s", ms=3,
               label="predicted (modal, +0.1 offset)"),
        Patch(facecolor="0.45", label="MC state probability")],
        fontsize=6.5, loc="upper left", framealpha=0.92, borderpad=0.35, handlelength=1.6)

    # ---- Jaccard distribution vs the carry-baseline-forward reference
    ok = C["ok"]
    bins = np.linspace(0, 1, 26)
    ax_j.hist(C["jac_b"][ok], bins=bins, color="0.72", alpha=0.95, label="carry baseline forward")
    ax_j.hist(C["jac"][ok], bins=bins, histtype="step", lw=2.0, color=ps.OUTCOME_COLORS["NACCUDSD"],
              label="model (MC modal)")
    mj, mjl, mjh = boot_mean(C["jac"][ok]); mb = float(np.nanmean(C["jac_b"][ok]))
    ax_j.axvline(mj, color=ps.OUTCOME_COLORS["NACCUDSD"], ls="--", lw=1.5)
    ax_j.axvline(mb, color="0.35", ls="--", lw=1.5)
    ma, mal, mah = boot_mean(C["jac_alive"][C["ok_alive"]])
    ax_j.set_xlabel("Jaccard index (observed vs predicted stage-at-age)")
    ax_j.set_ylabel("# patients")
    ax_j.set_title(f"Trajectory overlap, n={int(ok.sum())}\nmodel {mj:.3f} [{mjl:.3f}, {mjh:.3f}]"
                   f"  ·  baseline-carry {mb:.3f}\n"
                   f"alive-years only: {ma:.3f} (n={int(C['ok_alive'].sum())})", fontsize=9)
    ax_j.legend(fontsize=7.5, loc="upper left", framealpha=0.9)

    # ---- agreement as a function of how far ahead we are predicting
    # The single most load-bearing cell in this panel. The two mean Jaccards above are 15-year
    # averages, and averaging hides the structure completely: the model is INDISTINGUISHABLE from
    # "assume nothing changes" for the first few years and only pulls away later. Without this
    # curve a reader takes 0.626 vs 0.510 to mean a uniform advantage, which is not what happened.
    by = C["by_year"]
    yv = by["year"].to_numpy(); am = by["agreement"].to_numpy()
    ab = by["agreement_baseline"].to_numpy(); nn = by["n"].to_numpy()
    ax_n = ax_y.twinx()                                   # sample size shrinks a lot -- show it
    ax_n.bar(yv, nn, width=0.72, color="0.90", zorder=0)
    ax_n.set_ylabel("n still observed", fontsize=7.5, color="0.5")
    ax_n.tick_params(axis="y", labelsize=7, colors="0.5")
    ax_n.set_ylim(0, nn.max() * 3.1)                      # keep the bars in the lower third
    ax_n.grid(False)
    ax_y.set_zorder(ax_n.get_zorder() + 1); ax_y.patch.set_visible(False)   # lines above bars
    ax_y.plot(yv, ab, "-s", ms=3.5, lw=1.7, color="0.45", label="carry baseline forward")
    ax_y.plot(yv, am, "-o", ms=3.8, lw=2.0, color=ps.OUTCOME_COLORS["NACCUDSD"], label="model")
    # first year from which the model is ahead and stays ahead
    cross = next((int(yv[i]) for i in range(len(yv)) if np.all(am[i:] >= ab[i:])), None)
    if cross is not None:
        ax_y.axvline(cross - 0.5, ls=":", color="0.3", lw=1.3)
        ax_y.annotate(f"model overtakes\n“nothing changes”\nfrom year {cross}",
                      xy=(cross - 0.4, 0.93), xycoords=("data", "axes fraction"),
                      fontsize=7, va="top", ha="left", color="0.25")
    ax_y.set_xlabel("years after baseline", fontsize=9)
    ax_y.set_ylabel("agreement (state correct)")
    ax_y.set_xlim(yv.min() - 0.7, yv.max() + 0.7); ax_y.set_ylim(0, 1.02)
    ax_y.set_title("Agreement by how far ahead\n(the 15-y means above hide this)", fontsize=9)
    ax_y.legend(fontsize=7.2, loc="lower left", framealpha=0.9)
    ps.save_data(by, OUT, "fig2c_agreement_by_year" + SUFFIX)

    # ---- per-state IoU
    names = list(C["iou"].keys())
    vals = [C["iou"][n]["iou"] for n in names]
    ax_i.bar(np.arange(len(names)), vals, color=STATE_COLORS, edgecolor="white")
    for x, v in enumerate(vals):
        ax_i.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=8)
    ax_i.set_xticks(np.arange(len(names)))
    ax_i.set_xticklabels(names, fontsize=8, rotation=20, ha="right")
    ax_i.set_ylabel("IoU"); ax_i.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    ax_i.set_title(f"Per-state intersection-over-union\nmacro mean = {np.mean(vals):.3f}", fontsize=9)

    panel_letter(ex_axes[0], letter)
    acc, acc_lo, acc_hi = boot_mean(C["acc"][ok])
    metrics = dict(n_patients=int(ok.sum()),
                   mean_jaccard=round(mj, 4), mean_jaccard_ci=[round(mjl, 4), round(mjh, 4)],
                   mean_jaccard_baseline_carry=round(mb, 4),
                   n_patients_alive_years=int(C["ok_alive"].sum()),
                   mean_jaccard_alive_years_only=round(ma, 4),
                   mean_jaccard_alive_years_only_ci=[round(mal, 4), round(mah, 4)],
                   mean_agreement=round(acc, 4), mean_agreement_ci=[round(acc_lo, 4), round(acc_hi, 4)],
                   mean_agreement_baseline_carry=round(float(np.nanmean(C["acc_b"][ok])), 4),
                   macro_iou=round(float(np.mean(vals)), 4),
                   per_state_iou={n: round(C["iou"][n]["iou"], 4) for n in names},
                   agreement_by_year=C["by_year"].to_dict("records"))
    tab = pd.DataFrame(dict(pid=df["pid"].to_numpy()[ok], n_grid_points=C["nvalid"][ok],
                            jaccard_model=C["jac"][ok], jaccard_baseline_carry=C["jac_b"][ok],
                            agreement_model=C["acc"][ok], agreement_baseline_carry=C["acc_b"][ok],
                            n_alive_grid_points=C["nalive"][ok],
                            jaccard_model_alive_years=C["jac_alive"][ok]))
    return metrics, tab


# =========================================================================== PANEL D
def compute_D(cache, n_neighbors=30, min_dist=0.3, seed=RNG_SEED):
    """UMAP of the model's patient embeddings + how well trajectory identity is preserved.

    Purity is measured with k-NN in the ORIGINAL 120-d embedding space (not in the 2-d UMAP), so
    it reports what the model actually encodes rather than what the projection happens to show.

    ONLY the BASELINE-visit embedding is projected (2026-07-28). The full-observed-history
    embedding is label-leaking for this figure: `traj_class` is defined by which state tokens
    appear in the patient's record, and those very tokens are in the input that produced the
    embedding -- so separation there measures memory, not prognosis. Its purity is still recorded
    in metrics.json under `full_history_LEAKY` for the record, but it is not plotted.
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors
    from sklearn.metrics import silhouette_score
    import umap

    df, emb = cache["df"], cache["emb"]
    assert np.array_equal(emb["pid"], df["pid"].to_numpy()), "embedding / frame pid mismatch"
    lab_traj = df["traj_class"].to_numpy()
    lab_base = np.array([F2.STATE_NAMES[i] for i in df["baseline_state"]])

    def knn_purity(X, lab, k=10):
        nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
        _, idx = nn.kneighbors(X)
        return float(np.mean(lab[idx[:, 1:]] == lab[:, None]))

    def chance(lab):
        _, c = np.unique(lab, return_counts=True)
        p = c / c.sum()
        return float((p ** 2).sum())

    res = {}
    Xb = StandardScaler().fit_transform(emb["base"])
    red = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed,
                    n_components=2).fit_transform(Xb)
    res["base"] = dict(title="baseline visit only", xy=red,
                       purity_traj=knn_purity(Xb, lab_traj), purity_base=knn_purity(Xb, lab_base),
                       sil_traj=float(silhouette_score(red, lab_traj)),
                       sil_base=float(silhouette_score(red, lab_base)))
    # purity only (no projection) for the leaky one -- kNN in 120-d is cheap, UMAP is not
    Xf = StandardScaler().fit_transform(emb["full"])
    res["full"] = dict(title="full observed history (LEAKY, not plotted)", xy=None,
                       purity_traj=knn_purity(Xf, lab_traj), purity_base=knn_purity(Xf, lab_base),
                       sil_traj=np.nan, sil_base=np.nan)
    res["chance_traj"] = chance(lab_traj); res["chance_base"] = chance(lab_base)
    res["lab_traj"] = lab_traj; res["lab_base"] = lab_base

    # WITHIN-BASELINE-STATE purity. The raw trajectory purity is badly confounded: the label is
    # structurally constrained by the baseline state (a patient already at Dementia cannot
    # "progress to Dementia"; a Normal one cannot "improve"), and the embedding reads the baseline
    # state almost perfectly. So most of the raw lift is just "it knows your current stage".
    # Restricting the neighbour search to patients in the SAME baseline state removes that path.
    bs = df["baseline_state"].to_numpy()
    num = den = ntot = 0
    strata = {}
    for s, nm in enumerate(F2.STATE_NAMES):
        m = bs == s
        if m.sum() < 40:
            continue
        p, c = knn_purity(Xb[m], lab_traj[m]), chance(lab_traj[m])
        strata[nm] = dict(n=int(m.sum()), purity=p, chance=c, lift=p - c)
        num += p * m.sum(); den += c * m.sum(); ntot += m.sum()
    res["strat"] = dict(by_state=strata, n=int(ntot),
                        purity=num / max(ntot, 1), chance=den / max(ntot, 1),
                        lift=(num - den) / max(ntot, 1))
    return res


def panel_D(fig, spec, cache, letter="d"):
    if "D" not in cache:
        cache["D"] = compute_D(cache)
    D = cache["D"]
    # ONE embedding (the baseline visit -- no future information in the input), TWO colourings.
    # Same points in both cells, so the contrast between the two purities is a statement about
    # what the representation is organised by: overwhelmingly the CURRENT state, plus a real but
    # much weaker signal about what happens later. The full-history projection was dropped
    # (2026-07-28) because its labels are contained in its own input -- see compute_D.
    # ONE projection, centred (empty side columns keep it from stretching to a 4:1 letterbox in
    # the assembled figure). The baseline-state colouring was dropped on request 2026-07-28 --
    # it was a sanity check (the state token is in the input), not a result.
    gs = GridSpecFromSubplotSpec(1, 3, subplot_spec=spec, width_ratios=[0.5, 1.9, 0.5])
    axes = [fig.add_subplot(gs[0, 1])]
    xy = D["base"]["xy"]

    def draw(ax, labels, palette, order, ttl, purity, ch, legend_title, ncol):
        for c in order:                      # biggest class first, rare ones stay visible on top
            m = labels == c
            if not m.any():
                continue
            ax.scatter(xy[m, 0], xy[m, 1], s=3.2, alpha=0.55, linewidths=0,
                       color=palette[c], label=c, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(alpha=0)
        ax.set_xlabel("UMAP 1", fontsize=9); ax.set_ylabel("UMAP 2", fontsize=9)
        ax.set_title(f"{ttl}\n10-NN purity = {purity:.3f}  (chance {ch:.3f})", fontsize=9,
                     linespacing=1.35)
        h_, l_ = ax.get_legend_handles_labels()
        items = [(h_[l_.index(c)], f"{c} (n={int((labels == c).sum()):,})")
                 for c in order if c in l_]
        leg = ax.legend([a for a, _ in items], [b for _, b in items], fontsize=7.2,
                        loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=ncol,
                        markerscale=3.6, frameon=False, columnspacing=1.0, handletextpad=0.3,
                        title=legend_title, title_fontsize=8)
        leg._legend_box.align = "center"

    traj_order = [c for c, n in sorted([(c, int((D["lab_traj"] == c).sum()))
                                        for c in F2.TRAJ_CLASSES], key=lambda p: -p[1]) if n]
    # the "baseline visit only" caveat lives in the titles, not in a fig.text -- a figure-coordinate
    # caption cannot be placed reliably in BOTH the standalone panel and the assembled figure
    st = D["strat"]
    draw(axes[0], D["lab_traj"], TRAJ_COLORS, traj_order,
         "baseline-visit input (no future events) · coloured by OBSERVED future trajectory",
         D["base"]["purity_traj"], D["chance_traj"], "observed trajectory class", 5)
    # The raw lift is mostly the baseline state leaking through the label; state the honest number
    # on the figure rather than only in metrics.json.
    axes[0].set_title(axes[0].get_title() +
                      f"\nwithin the same baseline state: {st['purity']:.3f} "
                      f"(chance {st['chance']:.3f}, lift {st['lift']:+.3f})",
                      fontsize=9, linespacing=1.35)

    panel_letter(axes[0], letter)
    metrics = dict(
        baseline_visit_embedding=dict(
            purity_traj_10nn=round(D["base"]["purity_traj"], 4),
            purity_baseline_state_10nn=round(D["base"]["purity_base"], 4),
            silhouette_traj_2d=round(D["base"]["sil_traj"], 4),
            silhouette_baseline_state_2d=round(D["base"]["sil_base"], 4)),
        full_history_LEAKY=dict(
            note="labels are defined by tokens present in this embedding's own input; "
                 "recorded for reference, NOT plotted and NOT evidence of prediction",
            purity_traj_10nn=round(D["full"]["purity_traj"], 4),
            purity_baseline_state_10nn=round(D["full"]["purity_base"], 4)),
        chance_purity=dict(traj_class=round(D["chance_traj"], 4),
                           baseline_state=round(D["chance_base"], 4)),
        within_baseline_state=dict(
            note="neighbours restricted to the same baseline NACCUDSD state -- removes the path "
                 "'embedding knows the current stage, and the stage constrains which trajectory "
                 "classes are even possible'. THIS is the prognostic number, not purity_traj_10nn",
            purity=round(D["strat"]["purity"], 4), chance=round(D["strat"]["chance"], 4),
            lift=round(D["strat"]["lift"], 4),
            by_state={k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in D["strat"]["by_state"].items()}))
    tab = pd.DataFrame(dict(pid=cache["df"]["pid"].to_numpy(),
                            traj_class=D["lab_traj"], baseline_state=D["lab_base"],
                            umap1_base=xy[:, 0], umap2_base=xy[:, 1]))
    return metrics, tab


# =========================================================================== supplementary
def supp_transition_matrices(cache):
    """Companion to panel a: the full observed vs MC-predicted one-step transition matrices.

    Row-normalised P(next distinct state | current state). The diagonal is ~0 by construction --
    the data is keep-transitions, and `generate(no_repeat=True)` cannot repeat a state either."""
    obsM = cache["trans"]["obs"].astype(float)
    predM = cache["trans"]["pred"].astype(float)

    def rn(M):
        r = M.sum(1, keepdims=True)
        return np.divide(M, r, out=np.zeros_like(M), where=r > 0)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.0))
    rows = []
    for ax, M, P, ttl in [(axes[0], obsM, rn(obsM), "observed (data)"),
                          (axes[1], predM, rn(predM), "predicted (Monte-Carlo)")]:
        ax.imshow(P, cmap="Purples", vmin=0, vmax=1)
        ax.set_xticks(range(F2.NSTATE)); ax.set_xticklabels(F2.ALL_NAMES, rotation=40, ha="right",
                                                            fontsize=8)
        ax.set_yticks(range(F2.NSTATE)); ax.set_yticklabels(F2.ALL_NAMES, fontsize=8)
        for i in range(F2.NSTATE):
            for j in range(F2.NSTATE):
                ax.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if P[i, j] > 0.5 else "black")
                rows.append(dict(kind=ttl, frm=F2.ALL_NAMES[i], to=F2.ALL_NAMES[j],
                                 count=float(M[i, j]), prob=float(P[i, j])))
        ax.set_xlabel("to (next distinct state)"); ax.set_ylabel("from (current state)")
        ax.set_title(f"{ttl}\nn = {M.sum():,.0f} transitions", fontsize=10)
        ax.grid(False)
    xo = np.array([rn(obsM)[i, j] for i in range(F2.NSTATE) for j in range(F2.NSTATE)
                   if obsM[i].sum() > 0])
    yp = np.array([rn(predM)[i, j] for i in range(F2.NSTATE) for j in range(F2.NSTATE)
                   if obsM[i].sum() > 0])
    r = float(np.corrcoef(xo, yp)[0, 1]) if len(xo) > 2 else float("nan")
    # NOT cohort-subsettable: these matrices were summed over patients when the cache was built,
    # so they are the full evaluable test set in both --cohort modes. Say so on the figure.
    fig.suptitle(f"Figure 2a (supplementary) — one-step transition matrices  ·  "
                 f"Pearson r = {r:.3f}, MAE = {np.mean(np.abs(xo - yp)):.3f}\n"
                 f"full evaluable test set (not cohort-matched)", fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    _save(fig, "fig2a_supp_transition_matrix", pd.DataFrame(rows))
    return dict(pearson_r=round(r, 4), mae=round(float(np.mean(np.abs(xo - yp))), 4),
                n_obs_transitions=float(obsM.sum()), n_pred_transitions=float(predM.sum()))


# =========================================================================== drivers
#                fn        standalone title                       figsize      bottom margin
PANELS = {"a": (panel_A, "Transition-prediction accuracy",      (18.5, 5.0), 0.13),
          "b": (panel_B, "Timing accuracy by interval",         (12.5, 5.0), 0.13),
          "c": (panel_C, "Per-individual trajectory comparison", (14.5, 9.6), 0.07),
          "d": (panel_D, "Patient-embedding structure — baseline visit only (UMAP)", (10.5, 6.2), 0.26)}


def load_all(n_mc=F2.N_MC, limit=0, split=F2.SPLIT, cohort="matched", dataset=F2.DATASET,
             ckpt=F2.CKPT):
    """Load the MC cache and, by default, restrict it to the model's own TRAINING cohort.

    The MC pass is run over every evaluable patient, so both cohorts come out of the same cache
    -- switching costs a re-plot, not a re-simulation. `trans` is the one thing that cannot be
    subset: it was summed over patients at build time, so the supplementary transition matrix
    stays full-cohort in both modes and says so in its title.
    """
    # The cache is keyed by the CHECKPOINT signature, so this must be the same ckpt that
    # figure2_core.py --build used -- otherwise we silently load another model's frame.
    df, grids, trans, emb = F2.load_cache(ckpt=ckpt, n_mc=n_mc, limit=limit, split=split,
                                          dataset=dataset)
    if emb is None:
        raise FileNotFoundError("embeddings missing; run: python figure2_core.py --build")
    df = df.reset_index(drop=True)
    info = dict(cohort=cohort, n_evaluable=int(len(df)))
    if cohort == "matched":
        keep = F2.training_cohort_mask(dataset=dataset, split=split)
        sel = keep[df["pid"].to_numpy()]
        n0 = len(sel)
        df = df[sel].reset_index(drop=True)
        grids = {k: (v[sel] if hasattr(v, "shape") and v.ndim >= 1 and v.shape[0] == n0 else v)
                 for k, v in grids.items()}
        emb = {k: v[sel] for k, v in emb.items()}
        info.update(n_in_cohort=int(sel.sum()), n_dropped=int((~sel).sum()),
                    rule=f"train.py filter_cohort: >={F2.COHORT_MIN_VISITS} visits OR "
                         f">={F2.COHORT_SHORT_MIN_VISITS} visits + a NACCUDSD transition")
        log.info("cohort=matched: %d of %d evaluable patients are inside the training cohort "
                 "(%d dropped)", sel.sum(), n0, (~sel).sum())
    return dict(df=df, grids=grids, trans=trans, emb=emb, cohort_info=info)


def run_standalone(cache, code):
    fn, title, figsize, bottom = PANELS[code]
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, 1, left=0.09, right=0.985, top=0.85, bottom=bottom)
    m, tab = fn(fig, gs[0, 0], cache, letter=None)
    fig.suptitle(f"Figure 2{code} — {title}{COHORT_NOTE}", fontsize=12.5, x=0.015, ha="left")
    _save(fig, f"fig2{code}", tab)
    return m


def run_combined(cache):
    fig = plt.figure(figsize=(16.5, 22.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1.05, 1.0, 2.35, 1.02], hspace=0.42,
                          left=0.085, right=0.985, top=0.94, bottom=0.04)
    for i, code in enumerate("abcd"):
        PANELS[code][0](fig, gs[i, 0], cache, letter=code)
    fig.suptitle(f"Figure 2{COHORT_NOTE}", fontsize=14, x=0.02, ha="left", y=0.985)
    _save(fig, "figure2_combined")


def write_summary(cache, metrics):
    df = cache["df"]
    ci = cache["cohort_info"]
    if ci["cohort"] == "matched":
        pop = (f"**Cohort-matched to training**: {ci['n_in_cohort']:,} of {ci['n_evaluable']:,} "
               f"evaluable test patients ({ci['n_dropped']:,} dropped). Rule = {ci['rule']} — the "
               "same filter `train.py` applied when this checkpoint was fitted, so the model is "
               "scored on the population it was actually trained for. The unrestricted version is "
               "`SUMMARY_allcohort.md`.")
    else:
        pop = (f"**Full evaluable test set**: {ci['n_evaluable']:,} patients, NO cohort matching. "
               "~43% of these are outside the population `train.py` fitted on (2 visits, <2 y "
               "follow-up). Kept as a generalisation check; the paper figure is the matched one.")
    L = ["# Figure 2 — transition / timing / trajectory / embedding evaluation", "",
         f"Model **{metrics.get('_meta', {}).get('checkpoint', F2.CKPT)}** · dataset "
         f"**{metrics.get('_meta', {}).get('dataset', F2.DATASET)}** / split "
         f"**{metrics.get('_meta', {}).get('split_arg', F2.SPLIT)}** · "
         f"n = {len(df)} patients · {F2.N_MC} Monte-Carlo trajectories per patient, "
         f"seeded on the first visit · primary horizon {H} y.", "", pop, "",
         "Every predicted quantity is a Monte-Carlo statistic over sampled trajectories; labels are "
         "competing-risk (death) and right-censoring aware. Panels are generated by "
         "`figure2_panels.py` from the cache built by `figure2_core.py`.", ""]
    for code in "abcd":
        blob = json.dumps(metrics.get(code, {}), indent=1, default=str)
        trunc = blob[:2600] + (" …  (full values in metrics.json)" if len(blob) > 2600 else "")
        L += [f"## Panel {code} — {PANELS[code][1]}", "",
              f"![{code}](fig2{code}{SUFFIX}.png)", "", "```", trunc, "```", ""]
    L += ["## Supplementary — transition matrices", "",
          "Summed over patients at cache-build time, so this one is the FULL evaluable test set "
          "in both cohort modes.", "",
          f"![supp](fig2a_supp_transition_matrix{SUFFIX}.png)", "",
          "```", json.dumps(metrics.get("a_supp", {}), indent=1, default=str), "```", "",
          "## Combined", "", f"![combined](figure2_combined{SUFFIX}.png)", "",
          "See `README.md` for the method and the caveats that belong with these numbers."]
    with open(os.path.join(OUT, f"SUMMARY{SUFFIX}.md"), "w") as f:
        f.write("\n".join(L))


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    # matplotlib/fontTools log every glyph they subset into a PDF at INFO -- thousands of lines
    # that bury our own progress messages in the Slurm log.
    for noisy in ("matplotlib", "matplotlib.font_manager", "fontTools", "PIL", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", default="all", choices=["all", "a", "b", "c", "d"])
    ap.add_argument("--no-combined", action="store_true")
    ap.add_argument("--n-mc", type=int, default=F2.N_MC)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--split", default=F2.SPLIT)
    # --cohort matched re-reads the split from disk to rebuild train.py's cohort filter, so this
    # module needs the dataset name too. Default = figure2_core's, i.e. unchanged behaviour.
    ap.add_argument("--ckpt", default=F2.CKPT,
                    help=f"checkpoint whose cache to plot (default: {F2.CKPT}). MUST match the\n"
                         f"one figure2_core.py --build used -- the cache is keyed by it.")
    ap.add_argument("--dataset", default=F2.DATASET,
                    help=f"dataset dir under data/ used to rebuild the training-cohort mask "
                         f"(default: {F2.DATASET}). Must match the one figure2_core.py --build used.")
    ap.add_argument("--cohort", default="matched", choices=["matched", "all"],
                    help="matched (default) = only the patients train.py fitted on; "
                         "all = the whole evaluable test split, written to *_allcohort files")
    a = ap.parse_args()

    global SUFFIX, COHORT_NOTE
    # Both cohorts carry an explicit suffix. "no suffix = the paper figure" was implicit
    # knowledge that does not survive a handover.
    SUFFIX = "_matched" if a.cohort == "matched" else "_allcohort"
    COHORT_NOTE = ("" if a.cohort == "matched" else
                   "   [full test set — NOT matched to the training cohort]")

    ps.setup_style()
    os.makedirs(OUT, exist_ok=True)
    cache = load_all(n_mc=a.n_mc, limit=a.limit, split=a.split, cohort=a.cohort, dataset=a.dataset,
                     ckpt=a.ckpt)
    log.info("cache loaded: %d patients (cohort=%s)", len(cache["df"]), a.cohort)

    codes = list("abcd") if a.panel == "all" else [a.panel]
    mpath = os.path.join(OUT, f"metrics{SUFFIX}.json")
    metrics = json.load(open(mpath)) if os.path.exists(mpath) else {}
    for c in codes:
        t0 = time.time()
        metrics[c] = run_standalone(cache, c)
        metrics[c]["_seconds"] = round(time.time() - t0, 1)
        log.info("panel %s done (%.1fs)", c, time.time() - t0)
    metrics["_meta"] = dict(checkpoint=a.ckpt, dataset=a.dataset, split=F2.SPLIT,
                            n_patients=int(len(cache["df"])), n_mc=a.n_mc, horizon=H,
                            split_arg=a.split, **cache["cohort_info"],
                            generated_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    with open(mpath, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    if a.panel == "all":
        metrics["a_supp"] = supp_transition_matrices(cache)
        with open(mpath, "w") as f:
            json.dump(metrics, f, indent=2, default=str)
    if not a.no_combined and a.panel == "all":
        run_combined(cache)
        write_summary(cache, metrics)
    print(f"figures -> {OUT}")


if __name__ == "__main__":
    main()
