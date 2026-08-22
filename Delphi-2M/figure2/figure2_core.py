"""
figure2_core.py -- computation + caching backend for **Figure 2** (transition / timing /
trajectory / embedding evaluation of the AD-progression model).

Everything the four Figure-2 panels need is derived ONCE here and cached to
`results/figure2/_cache/`, so the panels themselves are pure plotting and re-run in seconds.

Model access goes exclusively through `predict_adapter` (the one abstraction, see its docstring
and README.md sec.10).  Domain facts baked in:

  * NACCUDSD states = model tokens 106..109 (Normal / Impaired-not-MCI / MCI / Dementia),
    Death = 110 (absorbing, and a COMPETING RISK for every cognitive transition).
  * The dataset is keep-transitions ("dedup"): consecutive repeats of the same state are removed,
    so an observed state sequence is by construction a sequence of *changes*.
  * Prediction is autoregressive: there is no closed-form horizon risk. Every predicted quantity
    below is a Monte-Carlo statistic over `N_MC` sampled trajectories seeded on the patient's
    FIRST visit (baseline).

What is cached (all keyed by checkpoint signature + split + n_mc):
  frame.pkl     one row / patient: baseline state & age, follow-up, observed first-passage times
                to each state and to death, MC-predicted risk at each horizon and MC-predicted
                event times, observed/predicted trajectory class.
  grids.npz     observed vs predicted state-at-age on a yearly grid from baseline (+ validity mask
                and the full predicted state-occupancy distribution).
  trans.npz     observed and MC-predicted one-step transition COUNT matrices (5x5, incl. Death).
  emb.npz       per-patient model embeddings (last hidden state after ln_f) for the baseline
                prompt and for the full observed history.

CLI:
    python figure2_core.py --build            # MC pass (multiprocess) + embeddings
    python figure2_core.py --build --workers 9 --n-mc 100
    python figure2_core.py --build --limit 300        # quick smoke run
"""
import os, sys, time, argparse, logging
import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Delphi-2M/ (package root)
sys.path.insert(0, HERE)
from delphi import predict_adapter as PA  # noqa: E402

log = logging.getLogger("fig2")

# --------------------------------------------------------------------------- configuration
CKPT = "out-delphi2m-dedup-mask-s42/ckpt.pt"   # the model under study (dedup + mask)
DATASET = "nacc-dedup-s42"
SPLIT = "test"
SEED = 42
N_MC = 100
HORIZONS = [1, 2, 3, 5, 10]
PRIMARY_H = 5
GRID_YEARS = np.arange(0, 16)                  # years after baseline: 0..15
D = PA.DAYS_PER_YEAR

STATE_TOKENS = [106, 107, 108, 109]
STATE_NAMES = ["Normal", "Impaired", "MCI", "Dementia"]
DEATH = PA.DEATH                               # 110
ALL_TOKENS = STATE_TOKENS + [DEATH]
ALL_NAMES = STATE_NAMES + ["Death"]
NSTATE = len(ALL_TOKENS)                       # 5, Death is the absorbing 5th state
NEG = -1e4 + 1                                 # sampled trajectories are padded with age = -10000

OUT_DIR = os.path.join(HERE, "results", "figure2")
CACHE_DIR = os.path.join(OUT_DIR, "_cache")


# --------------------------------------------------------------------------- small helpers
def _baseline(ages, toks):
    """(base_day, prompt_mask, baseline_state) for a patient, or None if not evaluable.

    baseline = the FIRST distinct visit age; the prompt is every token at age <= baseline
    (birth-time statics + the whole first visit).  Requires >=2 distinct visits (something to
    predict) and a NACCUDSD state recorded at baseline (a state to predict *from*)."""
    a = np.asarray(ages)
    va = np.unique(a[a > 0])
    if len(va) < 2:
        return None
    base = float(va[0])
    m = (a <= base) & np.isin(toks, STATE_TOKENS)
    if not m.any():
        return None
    j = np.where(m)[0]
    b = STATE_TOKENS.index(int(np.asarray(toks)[j[np.argmax(a[j])]]))
    return base, (a <= base), b


def _state_grid_one(ages, toks, base_state, grid_days):
    """Carry-forward state index (0..4) at each grid day for ONE stream (observed or sampled).

    Death (index 4) is absorbing because no later state token can follow it in either the data
    or a sampled trajectory (generate() terminates on Death)."""
    a = np.asarray(ages, float); t = np.asarray(toks)
    m = np.isin(t, ALL_TOKENS) & (a > NEG)
    out = np.full(len(grid_days), base_state, dtype=np.int8)
    if not m.any():
        return out
    ea = a[m]; eb = np.array([ALL_TOKENS.index(int(x)) for x in t[m]])
    o = np.argsort(ea, kind="stable"); ea, eb = ea[o], eb[o]
    idx = np.searchsorted(ea, grid_days, side="right") - 1
    return np.where(idx >= 0, eb[np.clip(idx, 0, len(eb) - 1)], base_state).astype(np.int8)


def _state_grid_batch(A, T, base_state, grid_days):
    """Vectorised `_state_grid_one` over an (n_samples, T) MC batch -> (n_samples, n_grid) int8."""
    m = np.isin(T, ALL_TOKENS) & (A > NEG)
    ev_age = np.where(m, A, -np.inf)
    lut = np.full(int(max(ALL_TOKENS)) + 1, -1, dtype=np.int8)
    for i, tok in enumerate(ALL_TOKENS):
        lut[tok] = i
    ev_st = np.where(m, lut[np.clip(T, 0, len(lut) - 1)], -1)
    n = A.shape[0]
    out = np.full((n, len(grid_days)), base_state, dtype=np.int8)
    rows = np.arange(n)
    for gi, gd in enumerate(grid_days):
        aa = np.where(ev_age <= gd, ev_age, -np.inf)
        best = aa.max(1)
        j = aa.argmax(1)
        has = best > -np.inf
        out[:, gi] = np.where(has, ev_st[rows, j], base_state)
    return out


def _seq_from_stream(ages, toks):
    """Ordered list of state indices (0..4) in one stream -- the observed/sampled state path."""
    a = np.asarray(ages, float); t = np.asarray(toks)
    m = np.isin(t, ALL_TOKENS) & (a > NEG)
    if not m.any():
        return []
    aa, tt = a[m], t[m]
    o = np.argsort(aa, kind="stable")
    return [ALL_TOKENS.index(int(x)) for x in tt[o]]


def _trans_counts(seq, M):
    for i in range(len(seq) - 1):
        M[seq[i], seq[i + 1]] += 1


def _trans_counts_batch(A, T, M):
    """Accumulate one-step transition counts over a whole (n_samples, T) MC batch.

    Sampled streams are already age-ascending (generate() appends monotonically increasing ages
    and pads with -10000 at the end), so the row-major order of np.nonzero IS the temporal order."""
    m = np.isin(T, ALL_TOKENS) & (A > NEG)
    if not m.any():
        return
    lut = np.full(int(max(ALL_TOKENS)) + 1, -1, dtype=np.int64)
    for i, tok in enumerate(ALL_TOKENS):
        lut[tok] = i
    rows, cols = np.nonzero(m)
    s = lut[T[rows, cols]]
    same = rows[1:] == rows[:-1]
    np.add.at(M, (s[:-1][same], s[1:][same]), 1)


def trajectory_class(base_state, fp_obs, died, final_state):
    """Coarse observed-trajectory label used to colour the embedding panel.

    Priority: dementia conversion > any worsening > death-without-progression > improvement >
    stable.  Progression outranks death because a patient who converted and then died is
    informative about the cognitive trajectory the embedding is supposed to encode."""
    if base_state < 3 and np.isfinite(fp_obs[3]):
        return "Progressed to Dementia"
    worse = [s for s in range(base_state + 1, 4) if np.isfinite(fp_obs[s])]
    if worse:
        return "Progressed (Imp./MCI)"
    if died:
        return "Died, no progression"
    better = [s for s in range(0, base_state) if np.isfinite(fp_obs[s])]
    if better or (final_state is not None and final_state < base_state):
        return "Improved"
    return "Stable"


TRAJ_CLASSES = ["Stable", "Improved", "Progressed (Imp./MCI)", "Progressed to Dementia",
                "Died, no progression"]


# --------------------------------------------------------------------------- worker
_G = {}


def _init_worker(ckpt, dataset, split, device, seed):
    torch.set_num_threads(1)
    ad = PA.load_model(dict(checkpoint=os.path.join(HERE, ckpt), device=device, seed=seed))
    data, p2i = PA.load_split(dataset, split)
    _G.update(ad=ad, data=data, p2i=p2i)


def _process(args):
    """All Figure-2 per-patient quantities for patient row `k`. Returns None if not evaluable."""
    k, n_mc, seed, horizons = args
    ad, data, p2i = _G["ad"], _G["data"], _G["p2i"]
    ages, toks = PA.get_history(data, p2i, k)
    bp = _baseline(ages, toks)
    if bp is None:
        return None
    base, pmask, b = bp
    base_y = base / D
    last_obs = float(np.asarray(ages)[np.asarray(ages) > 0].max())
    fu = (last_obs - base) / D

    # ---------------- observed first-passage times (years after baseline), inf = never observed
    a = np.asarray(ages, float); t = np.asarray(toks)
    fp_obs = np.full(NSTATE, np.inf)
    for i, tok in enumerate(ALL_TOKENS):
        m = (t == tok) & (a > base)
        if m.any():
            fp_obs[i] = (float(a[m].min()) - base) / D
    died = np.isfinite(fp_obs[4])
    obs_seq = _seq_from_stream(ages, toks)
    final_state = obs_seq[-1] if obs_seq else b

    # ---------------- Monte-Carlo trajectories seeded on the baseline visit
    until = max(105.0, base_y + float(GRID_YEARS.max()) + 1.0)
    sim = PA.simulate_trajectory(ad, t[pmask], a[pmask], until_age_years=until,
                                 n_samples=n_mc, seed=seed + k, use_cache=False)
    A, T = sim["ages"], sim["tokens"]
    n = A.shape[0]

    fp_sim = np.full((NSTATE, n), np.inf)       # per-sample first passage (days, absolute age)
    for i, tok in enumerate(ALL_TOKENS):
        m = (T == tok) & (A > base) & (A > NEG)
        aa = np.where(m, A, np.inf)
        fp_sim[i] = aa.min(1)
    d_sim = fp_sim[4]

    row = dict(pid=int(k), baseline_age=base_y, baseline_state=int(b), followup=fu,
               n_visits=int(np.unique(a[a > 0]).size), died=int(died),
               final_state=int(final_state))
    for i, nm in enumerate(ALL_NAMES):
        row[f"obs_t_{nm}"] = float(fp_obs[i])
    # predicted risk of reaching each state by each horizon, competing with death
    for i, nm in enumerate(ALL_NAMES):
        for h in horizons:
            hd = base + h * D
            if nm == "Death":
                row[f"pred_{nm}_{h}y"] = float(np.mean(d_sim <= hd))
            else:
                row[f"pred_{nm}_{h}y"] = float(np.mean((fp_sim[i] <= hd) & (fp_sim[i] <= d_sim)))
        fin = fp_sim[i][np.isfinite(fp_sim[i])]
        row[f"pred_t_{nm}"] = float(np.median(fin) - base) / D if fin.size else np.nan
        row[f"pred_ever_{nm}"] = float(np.mean(np.isfinite(fp_sim[i])))

    # ---------------- state-at-age grids
    grid_days = base + GRID_YEARS * D
    obs_grid = _state_grid_one(ages, toks, b, grid_days)
    # a grid point is KNOWN only while the patient is still under observation, or once dead
    known = (GRID_YEARS * D <= (last_obs - base) + 1e-6)
    if died:
        known = known | (GRID_YEARS >= fp_obs[4])
    sim_grid = _state_grid_batch(A, T, b, grid_days)
    occ = np.zeros((len(GRID_YEARS), NSTATE), dtype=np.float32)
    for s in range(NSTATE):
        occ[:, s] = (sim_grid == s).mean(0)

    # ---------------- transition count matrices
    obsM = np.zeros((NSTATE, NSTATE)); _trans_counts(obs_seq, obsM)
    predM = np.zeros((NSTATE, NSTATE))
    _trans_counts_batch(A, T, predM)
    predM /= n                                   # per-patient expected counts

    row["traj_class"] = trajectory_class(b, fp_obs, died, final_state)
    # per-SAMPLE first-passage times (years after baseline, inf = never in that trajectory).
    # Kept in full so the panels can build ANY composite endpoint ("reach >=MCI", "any
    # worsening", "any improvement") post hoc without re-running the Monte-Carlo pass.
    fp_rel = np.where(np.isfinite(fp_sim), (fp_sim - base) / D, np.inf).astype(np.float32)
    return dict(row=row, obs_grid=obs_grid, known=known.astype(bool), occ=occ,
                sim_grid=sim_grid.astype(np.int8), obsM=obsM, predM=predM, fp=fp_rel)


# --------------------------------------------------------------------------- build
def cache_key(ckpt_sig, n_mc, limit, split=SPLIT, dataset=DATASET):
    """Cache identity = (checkpoint, dataset, split, n_mc, limit).

    The dataset MUST be part of the key. Without it, scoring one checkpoint against two
    different datasets silently reuses the first one's frame -- and if the two happen to hold
    the same number of patients it does not even crash, it just plots the wrong data.
    """
    return (f"{ckpt_sig}_{dataset}_{split}_n{n_mc}"
            + (f"_lim{limit}" if limit else ""))


def build_mc(ckpt=CKPT, dataset=DATASET, split=SPLIT, device="cpu", n_mc=N_MC, seed=SEED,
             workers=None, limit=0, force=False):
    """Run the MC pass over every evaluable patient and write frame.pkl / grids.npz / trans.npz."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    ad = PA.load_model(dict(checkpoint=os.path.join(HERE, ckpt), device=device, seed=seed))
    key = cache_key(ad.ckpt_sig, n_mc, limit, split, dataset)
    fpath = os.path.join(CACHE_DIR, f"frame_{key}.pkl")
    gpath = os.path.join(CACHE_DIR, f"grids_{key}.npz")
    tpath = os.path.join(CACHE_DIR, f"trans_{key}.npz")
    if not force and all(os.path.exists(p) for p in (fpath, gpath, tpath)):
        log.info("MC cache hit (%s)", key)
        return key

    data, p2i = PA.load_split(dataset, split)
    N = len(p2i) if not limit else min(len(p2i), limit)
    ks = list(range(N))
    workers = workers or max(1, min(os.cpu_count() - 1, 9))
    log.info("MC pass: %d patients, n_mc=%d, %d workers", N, n_mc, workers)

    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    t0 = time.time(); res = []
    payload = [(k, n_mc, seed, HORIZONS) for k in ks]
    with ctx.Pool(workers, initializer=_init_worker,
                  initargs=(ckpt, dataset, split, device, seed)) as pool:
        for i, r in enumerate(pool.imap(_process, payload, chunksize=16)):
            if r is not None:
                res.append(r)
            if (i + 1) % 250 == 0:
                el = time.time() - t0
                log.info("  %d/%d  (%d kept)  %.0fs elapsed, ~%.0fs left",
                         i + 1, N, len(res), el, el / (i + 1) * (N - i - 1))
    log.info("MC pass done in %.1f min (%d evaluable patients)", (time.time() - t0) / 60, len(res))

    df = pd.DataFrame([r["row"] for r in res])
    df.to_pickle(fpath)
    np.savez_compressed(
        gpath,
        pid=df["pid"].to_numpy(),
        grid_years=GRID_YEARS,
        obs=np.stack([r["obs_grid"] for r in res]),
        known=np.stack([r["known"] for r in res]),
        occ=np.stack([r["occ"] for r in res]).astype(np.float32),
        sim=np.stack([r["sim_grid"] for r in res]).astype(np.int8),
        fp=np.stack([r["fp"] for r in res]).astype(np.float32),      # (N, 5 states, n_mc)
    )
    np.savez_compressed(tpath,
                        obs=np.sum([r["obsM"] for r in res], axis=0),
                        pred=np.sum([r["predM"] for r in res], axis=0))
    log.info("cached -> %s / %s / %s", fpath, gpath, tpath)
    return key


# --------------------------------------------------------------------------- embeddings
@torch.no_grad()
def _hidden(ad, tokens, ages):
    """Final-block hidden states (post ln_f), plain causal mask -- i.e. EXACTLY the representation
    the model conditions on when it generates the future (generate() calls forward without
    targets, so the same-visit 'tie' mask is off). Returns (T, n_embd)."""
    store = {}
    h = ad.model.transformer.ln_f.register_forward_hook(lambda m, i, o: store.__setitem__("x", o))
    try:
        idx = torch.as_tensor(np.asarray(tokens)[None], dtype=torch.long, device=ad.device)
        age = torch.as_tensor(np.asarray(ages)[None], dtype=torch.float32, device=ad.device)
        ad.model(idx, age)
    finally:
        h.remove()
    return store["x"][0].float().cpu().numpy()


def build_embeddings(ckpt=CKPT, dataset=DATASET, split=SPLIT, device="cpu", seed=SEED,
                     limit=0, n_mc=N_MC, force=False):
    """Per-patient embeddings: last hidden state of (a) the baseline prompt -- the vector the model
    forecasts from, no future leakage -- and (b) the full observed history."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    ad = PA.load_model(dict(checkpoint=os.path.join(HERE, ckpt), device=device, seed=seed))
    key = cache_key(ad.ckpt_sig, n_mc, limit, split, dataset)
    epath = os.path.join(CACHE_DIR, f"emb_{key}.npz")
    if not force and os.path.exists(epath):
        log.info("embedding cache hit (%s)", key)
        return key
    data, p2i = PA.load_split(dataset, split)
    N = len(p2i) if not limit else min(len(p2i), limit)
    pid, E0, E1 = [], [], []
    t0 = time.time()
    for k in range(N):
        ages, toks = PA.get_history(data, p2i, k)
        bp = _baseline(ages, toks)
        if bp is None:
            continue
        base, pmask, b = bp
        h_base = _hidden(ad, np.asarray(toks)[pmask], np.asarray(ages)[pmask])[-1]
        h_full = _hidden(ad, toks, ages)[-1]
        pid.append(k); E0.append(h_base); E1.append(h_full)
        if (k + 1) % 2000 == 0:
            log.info("  embeddings %d/%d (%.0fs)", k + 1, N, time.time() - t0)
    np.savez_compressed(epath, pid=np.array(pid), base=np.stack(E0).astype(np.float32),
                        full=np.stack(E1).astype(np.float32))
    log.info("embeddings cached -> %s (%d patients, dim=%d)", epath, len(pid), E0[0].shape[0])
    return key


# --------------------------------------------------------------------------- training cohort
# train.py does NOT train on every patient in train.bin: `filter_cohort` (train.py:106) keeps a
# subject only if it has >= cohort_min_visits distinct visits, OR >= cohort_short_min_visits
# visits AND at least one NACCUDSD transition. config/train_delphi2m_mask_dedup.py does not
# override the defaults (4 / 2), so out-delphi2m-dedup-mask-s42 was fitted on 16,262 of the
# 38,687 train patients -- 42%. Scoring the whole test split therefore scores ~43% of patients
# in a regime the model was never fitted on (they are the 2-visit, <2-year-follow-up ones).
# This replicates the rule so evaluation can be restricted to the SAME population.
#
# NOTE tokens in the .bin are DISK space (model id - 1), so NACCUDSD Normal..Dementia = 105..108
# -- exactly as train.py's `udsd_disk` default. Keep the two in sync if either ever changes.
UDSD_DISK = (105, 106, 107, 108)
COHORT_MIN_VISITS = 4
COHORT_SHORT_MIN_VISITS = 2


def training_cohort_mask(dataset=DATASET, split=SPLIT, min_visits=COHORT_MIN_VISITS,
                         short_min_visits=COHORT_SHORT_MIN_VISITS):
    """Boolean mask over the patient ROWS of `split`: True = inside train.py's training cohort."""
    data, p2i = PA.load_split(dataset, split)
    ages = np.asarray(data[:, 1]); toks = np.asarray(data[:, 2])
    udsd = np.asarray(UDSD_DISK)
    keep = np.zeros(len(p2i), dtype=bool)
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        a = ages[s:s + n]
        nv = np.unique(a[a > 0]).size
        if nv >= min_visits:
            keep[k] = True
        elif nv >= short_min_visits:
            keep[k] = int(np.isin(toks[s:s + n], udsd).sum()) >= 2
    return keep


# --------------------------------------------------------------------------- endpoints
def composite(df, grids, states):
    """Observed and MC-predicted quantities for the composite endpoint "reach any of `states`".

    `states` are indices into ALL_NAMES (4 = Death).  Returns a dict of per-patient arrays:
      obs_time  years to the first observed hit (inf = never observed)
      d_obs     years to observed death (inf = alive at last contact)
      fu        observed follow-up (years from baseline to last contact)
      pred_time median MC first-passage among trajectories that ever hit (nan if none did)
      risk[h]   MC probability of hitting by h years AND before the sampled death
      ever      MC probability of ever hitting within the simulated span
    """
    st = list(states)
    fp = grids["fp"]                                   # (N, 5, n_mc), years after baseline
    fpt = fp[:, st, :].min(1)                          # (N, n_mc)
    death = fp[:, 4, :]
    obs = np.minimum.reduce([df[f"obs_t_{ALL_NAMES[s]}"].to_numpy(float) for s in st])
    fin = np.isfinite(fpt)
    pt = np.full(len(df), np.nan)
    for i in np.where(fin.any(1))[0]:
        pt[i] = float(np.median(fpt[i][fin[i]]))
    out = dict(obs_time=obs, d_obs=df["obs_t_Death"].to_numpy(float),
               fu=df["followup"].to_numpy(float), pred_time=pt, ever=fin.mean(1), risk={})
    for h in HORIZONS:
        out["risk"][h] = np.mean((fpt <= h) & (fpt <= death), axis=1)
    return out


def cr_times(ep, mask=None):
    """(time, etype) for a competing-risks estimator. etype 1 = the endpoint, 2 = death first,
    0 = censored at end of follow-up."""
    obs, d, fu = ep["obs_time"], ep["d_obs"], ep["fu"]
    ev = np.isfinite(obs) & (~np.isfinite(d) | (obs <= d))
    dd = np.isfinite(d) & ~ev
    time = np.where(ev, obs, np.where(dd, d, fu))
    et = np.where(ev, 1, np.where(dd, 2, 0))
    if mask is not None:
        time, et = time[mask], et[mask]
    return time.astype(float), et.astype(int)


def aalen_johansen(time, etype, horizon):
    """Cumulative incidence of cause 1 by `horizon`, with cause 2 competing and right-censoring.

    Hand-rolled (lifelines is not installed on the cluster env) -- the standard
    CIF(t) = sum_{t_i <= t} S(t_i-) * d1_i / n_i  with S the overall event-free survival."""
    t = np.asarray(time, float); e = np.asarray(etype, int)
    ok = np.isfinite(t)
    t, e = t[ok], e[ok]
    o = np.argsort(t, kind="stable"); t, e = t[o], e[o]
    n = len(t); S = 1.0; cif = 0.0; i = 0
    while i < n and t[i] <= horizon:
        j = i
        while j < n and t[j] == t[i]:
            j += 1
        at_risk = n - i
        d1 = int(np.sum(e[i:j] == 1)); d2 = int(np.sum(e[i:j] == 2))
        cif += S * d1 / at_risk
        S *= 1.0 - (d1 + d2) / at_risk
        i = j
    return float(cif)


def labels_at_h(ep, h):
    """Competing-risk + right-censoring aware binary label for "event by h years".

      1  event observed by h (and not preceded by death)
      0  KNOWN event-free through h: followed >= h, or died by h without the event, or the
         event happened after h
     -1  unknown: administratively censored before h with no event  -> dropped from metrics
    """
    obs, d, fu = ep["obs_time"], ep["d_obs"], ep["fu"]
    y = np.full(len(obs), -1, dtype=int)
    ev = np.isfinite(obs) & (obs <= h) & (~np.isfinite(d) | (obs <= d))
    y[ev] = 1
    nonev = (fu >= h) | (np.isfinite(d) & (d <= h)) | (np.isfinite(obs) & (obs > h))
    y[(y != 1) & nonev] = 0
    return y


# --------------------------------------------------------------------------- load
def load_cache(ckpt=CKPT, device="cpu", n_mc=N_MC, seed=SEED, limit=0, split=SPLIT,
               dataset=DATASET):
    """(df, grids, trans, emb) from the caches. Raises if the MC pass has not been run."""
    ad = PA.load_model(dict(checkpoint=os.path.join(HERE, ckpt), device=device, seed=seed))
    key = cache_key(ad.ckpt_sig, n_mc, limit, split, dataset)
    fpath = os.path.join(CACHE_DIR, f"frame_{key}.pkl")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"no Figure-2 cache for key {key}; run: python figure2_core.py --build")
    df = pd.read_pickle(fpath)
    grids = dict(np.load(os.path.join(CACHE_DIR, f"grids_{key}.npz")))
    trans = dict(np.load(os.path.join(CACHE_DIR, f"trans_{key}.npz")))
    epath = os.path.join(CACHE_DIR, f"emb_{key}.npz")
    emb = dict(np.load(epath)) if os.path.exists(epath) else None
    return df, grids, trans, emb


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-mc", type=int, default=N_MC)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-mc", action="store_true")
    ap.add_argument("--skip-emb", action="store_true")
    a = ap.parse_args()
    if not a.build:
        ap.error("nothing to do (pass --build)")
    kw = dict(ckpt=a.ckpt, dataset=a.dataset, split=a.split, device=a.device, seed=a.seed,
              limit=a.limit, force=a.force)
    if not a.skip_emb:
        build_embeddings(n_mc=a.n_mc, **kw)
    if not a.skip_mc:
        build_mc(n_mc=a.n_mc, workers=(a.workers or None), **kw)
    print("done.")


if __name__ == "__main__":
    main()
