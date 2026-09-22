"""
figure2_core.py -- computation + caching backend for **Figure 2** (transition / timing /
trajectory / embedding evaluation of the AD-progression model).

Everything the four Figure-2 panels need is derived ONCE here and cached to
`results/figure2/_cache/`, so the panels themselves are pure plotting and re-run in seconds.

Model access goes exclusively through `radc_delphi.engine`, the one abstraction that owns
termination, logit renormalisation and the disk/model token shift.  Domain facts are NOT baked
in here -- every one of them lives in `figure2/radc_states.py`, which also documents where the
RADC mapping is exact and where it is a substitution.  What this file assumes:

  * A state space of `S.NSTATE` slots: the cognitive stages (grouped MMSE levels) plus Death,
    which is absorbing and is a COMPETING RISK for every cognitive transition. The AD diagnosis
    is carried alongside as an EVENT slot, not a state -- a subject does not stop occupying an
    MMSE stage by being diagnosed.
  * The stream is TV-gated: an ordinal level is re-emitted only when the soft weight vector
    moves further than the measurement noise, so an observed stage sequence is by construction
    a sequence of *changes*, exactly as the NACC keep-transitions stream was.
  * Prediction is autoregressive: there is no closed-form horizon risk. Every predicted quantity
    below is a Monte-Carlo statistic over `N_MC` sampled trajectories seeded on the subject's
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

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root
sys.path.insert(0, HERE)
from radc_delphi import vocab as V  # noqa: E402
from radc_delphi import engine as EN  # noqa: E402
from figure2 import radc_states as S  # noqa: E402
from figure2 import perdomain as PD  # noqa: E402

log = logging.getLogger("fig2")

# --------------------------------------------------------------------------- configuration
CKPT = "../delphi/Delphi-ROSMAP/ckpt.pt"       # the model under study
DATASET = "rosmap"
# THERE IS NO TEST SPLIT, AND THIS IS THE FIGURE'S LARGEST CAVEAT.
# ../../tokenization/build.py writes train.bin and val.bin only, and ../../delphi/train.py
# selects the checkpoint on that val split (`always_save_checkpoint = False`, so ckpt.pt is the
# best-val iterate). Every number this figure produces is therefore measured on data used for
# MODEL SELECTION, not on held-out data. Upstream scores a real third split and its numbers are
# clean in a way ours are not.
#
# The bias this introduces is small but it is NOT zero and it is not signed in our favour by
# assumption: selection touched only the checkpoint step, not the weights, so the effect is the
# optimism of picking the best of 24 evaluated iterates. Treat every AUC here as an upper bound.
# Fixing it properly means a three-way split in build.py and a retrain -- see ../README.md.
SPLIT = "val"
SEED = 42
N_MC = 100
HORIZONS = [1, 2, 3, 5, 10]
PRIMARY_H = 5
GRID_YEARS = np.arange(0, 16)                  # years after baseline: 0..15
D = EN.DAYS_PER_YEAR

# Domain facts, all from radc_states. Kept as module-level aliases so the body below reads the
# same as the NACC version did -- the names mean the same things, they are just no longer ids.
STATE_NAMES = S.GRID_NAMES                     # stages + Death
ALL_NAMES = S.ALL_NAMES                        # stages + Death + AD diagnosis
DEATH = S.DEATH
AD_DX = S.AD_DX
GRID_TOKENS = S.GRID_TOKENS                    # tokens that move a subject between states
NSTATE = S.NSTATE                              # size of the state space (stages + Death)
NSLOT = S.NSLOT                                # state space + the AD event slot
DEATH_IDX = S.DEATH_IDX
AD_IDX = S.AD_IDX
NEG = -1e4 + 1                                 # sampled trajectories are padded with age = -10000

# Which tokens count as "a visit happened". The statics sit one day BEFORE the baseline visit,
# so counting distinct ages over the raw stream makes every subject look like it has one extra
# early visit -- the same off-by-one that made train.py's cohort filter a no-op until it was
# fixed to count only non-ignored tokens.
_VISIT_IGNORE = set(V.IGNORE_TOKENS) | {V.PADDING}

# Output root. FIG2_TAG separates runs that would otherwise overwrite each other: every panel
# filename is fixed (fig2a_matched.png ...), so scoring three checkpoints into one directory
# silently leaves only the last one. Default "" keeps the delivered v3 paths unchanged.
#
# The CACHE is deliberately NOT tagged. It is keyed on the checkpoint's md5 plus dataset, split
# and n_mc, so three checkpoints cannot collide there, and sharing it means re-plotting one run
# never invalidates another's Monte-Carlo pass.
FIG2_TAG = os.environ.get("FIG2_TAG", "")
OUT_DIR = os.path.join(HERE, "results", "figure2", FIG2_TAG) if FIG2_TAG else \
    os.path.join(HERE, "results", "figure2")
CACHE_DIR = os.path.join(HERE, "results", "figure2", "_cache")


# --------------------------------------------------------------------------- small helpers
def _baseline(ages, toks):
    """(base_day, prompt_mask, baseline_stage) for a subject, or None if not evaluable.

    baseline = the FIRST distinct VISIT age; the prompt is every token at age <= baseline
    (the statics + the whole first visit).  Requires >=2 distinct visits (something to predict)
    and a cognitive stage recorded at baseline (a state to predict *from*).

    "Distinct visit age" counts only non-ignored tokens. The statics are placed at
    `age_bl - 1 day`, so counting every distinct age instead would (a) make a single-visit
    subject look like it has two, and (b) put the baseline a day before the first real visit,
    leaving the prompt with no clinical content at all.
    """
    a = np.asarray(ages, float); t = np.asarray(toks)
    content = ~np.isin(t, list(_VISIT_IGNORE))
    va = np.unique(a[(a > 0) & content])
    if len(va) < 2:
        return None
    base = float(va[0])
    m = (a <= base) & np.isin(t, S.STAGE_TOKENS)
    if not m.any():
        return None
    j = np.where(m)[0]
    b = int(S.TOK2SLOT[int(t[j[np.argmax(a[j])]])])
    return base, (a <= base), b


def _state_grid_one(ages, toks, base_state, grid_days):
    """Carry-forward state index (0..4) at each grid day for ONE stream (observed or sampled).

    Death (slot DEATH_IDX) is absorbing because no later state token can follow it in either
    the data or a sampled trajectory (generate() terminates on Death).

    Only GRID_TOKENS move the state. The AD token is deliberately NOT one of them: it is an
    event, so letting it overwrite the stage would make a diagnosed subject's cognitive stage
    unreadable for the rest of the grid."""
    a = np.asarray(ages, float); t = np.asarray(toks)
    m = np.isin(t, GRID_TOKENS) & (a > NEG)
    out = np.full(len(grid_days), base_state, dtype=np.int8)
    if not m.any():
        return out
    ea = a[m]; eb = S.slot_of(t[m])
    o = np.argsort(ea, kind="stable"); ea, eb = ea[o], eb[o]
    idx = np.searchsorted(ea, grid_days, side="right") - 1
    return np.where(idx >= 0, eb[np.clip(idx, 0, len(eb) - 1)], base_state).astype(np.int8)


def _state_grid_batch(A, T, base_state, grid_days):
    """Vectorised `_state_grid_one` over an (n_samples, T) MC batch -> (n_samples, n_grid) int8."""
    m = np.isin(T, GRID_TOKENS) & (A > NEG)
    ev_age = np.where(m, A, -np.inf)
    ev_st = np.where(m, S.slot_of(T), -1)
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
    """Ordered list of state slots in one stream -- the observed/sampled state path.

    Consecutive repeats are collapsed. The NACC stream could not contain them (keep-transitions
    removed same-state repeats at tokenization), but a RADC stage is a GROUP of MMSE levels, so
    a 30 -> 28 move is a real token pair inside one stage and would otherwise be counted as a
    Normal -> Normal transition that never happened."""
    a = np.asarray(ages, float); t = np.asarray(toks)
    m = np.isin(t, GRID_TOKENS) & (a > NEG)
    if not m.any():
        return []
    aa, tt = a[m], t[m]
    o = np.argsort(aa, kind="stable")
    seq = S.slot_of(tt[o]).tolist()
    return [s for i, s in enumerate(seq) if i == 0 or s != seq[i - 1]]


def _trans_counts(seq, M):
    for i in range(len(seq) - 1):
        M[seq[i], seq[i + 1]] += 1


def _trans_counts_batch(A, T, M):
    """Accumulate one-step transition counts over a whole (n_samples, T) MC batch.

    Sampled streams are already age-ascending (generate() appends monotonically increasing ages
    and pads with -10000 at the end), so the row-major order of np.nonzero IS the temporal order.

    Self-transitions are dropped, which is exactly equivalent to collapsing runs of the same
    state before counting adjacent pairs -- see `_seq_from_stream` for why a RADC stage can
    repeat where a NACC one could not. It also keeps the matrix diagonal 0, as the NACC
    figure's caption claims."""
    m = np.isin(T, GRID_TOKENS) & (A > NEG)
    if not m.any():
        return
    rows, cols = np.nonzero(m)
    s = S.slot_of(T[rows, cols])
    same = (rows[1:] == rows[:-1]) & (s[1:] != s[:-1])
    np.add.at(M, (s[:-1][same], s[1:][same]), 1)


trajectory_class = S.trajectory_class
TRAJ_CLASSES = S.TRAJ_CLASSES


# --------------------------------------------------------------------------- worker
_G = {}


def _engine(ckpt, dataset, device, configure=True):
    """Load the checkpoint against the dataset it was trained on, and bind the figure to it.

    `strict_vocab` is left on deliberately. Upstream compares a labels.csv md5 recorded in the
    checkpoint; ours cannot, because `../../delphi/train.py` writes no such field, so the check
    engine.load performs instead is vocab_size plus the checkpoint's own `ignore_tokens` against
    the static block this label table resolves to. That is weaker than an md5 -- it would not
    notice a renamed token inside the static block -- but it does catch the failure that matters,
    a re-tokenization that MOVED the block, which is what makes every clinical id off by n.

    CROSS-TOKENIZATION RUNS work the same way they do upstream: the domain facts are resolved
    from the checkpoint's own table by NAME, so a model trained on an earlier ROSMAP build is
    scored with its own ids. What does NOT survive is comparability of the state space -- our
    stage spec is chosen by which MMSE level names the table has (radc_states._SPECS), so a
    build that re-binned MMSE lands on a different number of stages and its panels are not
    row-comparable with these. That is stated rather than prevented, because re-binning MMSE is
    a legitimate thing to do and silently regrouping it would be worse.
    """
    data_dir = os.path.join(HERE, "data", dataset)
    lp = os.path.join(data_dir, "labels.csv")
    # ALWAYS resolved against the build's own labels.csv, never against the live module.
    # Upstream only does this when the sizes differ, because its vocab.py hard-codes a table
    # that is normally the right one. Ours reads labels.csv to begin with, so passing the path
    # is both simpler and strictly safer: the engine then verifies the checkpoint's recorded
    # ignore_tokens against the static block THAT table resolves to, which is the closest thing
    # we have to upstream's vocab_sig (see engine.load).
    eng = EN.load(os.path.join(HERE, ckpt), data_dir=data_dir, device=device, vocab_labels=lp)
    if configure:
        _bind(eng)
    return eng


def _bind(eng):
    """Point radc_states / perdomain at this engine's token table, and refresh the aliases."""
    global STATE_NAMES, ALL_NAMES, DEATH, AD_DX, GRID_TOKENS, NSTATE, NSLOT
    global DEATH_IDX, AD_IDX, _VISIT_IGNORE
    S.configure(eng.labels)
    PD.configure()
    STATE_NAMES, ALL_NAMES = S.GRID_NAMES, S.ALL_NAMES
    DEATH, AD_DX, GRID_TOKENS = S.DEATH, S.AD_DX, S.GRID_TOKENS
    NSTATE, NSLOT, DEATH_IDX, AD_IDX = S.NSTATE, S.NSLOT, S.DEATH_IDX, S.AD_IDX
    # from the CHECKPOINT's static block, not the live one: v1 has 21 statics, v3 has 24
    _VISIT_IGNORE = set(eng.ignore_tokens) | {eng.res.PADDING}


def _init_worker(ckpt, dataset, split, device, seed):
    torch.set_num_threads(1)
    eng = _engine(ckpt, dataset, device)      # also re-binds S / PD inside this worker
    data, p2i, _sub = eng.load_split(split)
    _G.update(eng=eng, data=data, p2i=p2i)


def _process(args):
    """All Figure-2 per-subject quantities for subject row `k`. Returns None if not evaluable."""
    k, n_mc, seed, horizons = args
    eng, data, p2i = _G["eng"], _G["data"], _G["p2i"]
    ages, toks, _pid = eng.stream(data, p2i, k)
    bp = _baseline(ages, toks)
    if bp is None:
        return None
    base, pmask, b = bp
    base_y = base / D
    last_obs = float(np.asarray(ages)[np.asarray(ages) > 0].max())
    fu = (last_obs - base) / D

    # ---------------- observed first-passage times (years after baseline), inf = never observed
    a = np.asarray(ages, float); t = np.asarray(toks)
    fp_obs = np.full(NSLOT, np.inf)
    for i in range(NSLOT):
        m = (S.slot_of(t) == i) & (a > base)
        if m.any():
            fp_obs[i] = (float(a[m].min()) - base) / D
    died = np.isfinite(fp_obs[DEATH_IDX])
    obs_seq = _seq_from_stream(ages, toks)
    final_state = obs_seq[-1] if obs_seq else b

    # ---------------- Monte-Carlo trajectories seeded on the baseline visit
    until = max(105.0, base_y + float(GRID_YEARS.max()) + 1.0)
    # engine.simulate returns (tokens, ages) in that order, and owns the three things the
    # figure must not get wrong itself: Death terminates the rollout, No-event can never be
    # sampled, and only the repeatable tokens may recur.
    T, A = eng.simulate(t[pmask], a[pmask], n_mc=n_mc, seed=seed + k, until_age_years=until)
    n = A.shape[0]
    slot_sim = S.slot_of(T)

    fp_sim = np.full((NSLOT, n), np.inf)        # per-sample first passage (days, absolute age)
    for i in range(NSLOT):
        m = (slot_sim == i) & (A > base) & (A > NEG)
        aa = np.where(m, A, np.inf)
        fp_sim[i] = aa.min(1)
    d_sim = fp_sim[DEATH_IDX]

    n_visits = int(np.unique(a[(a > 0) & ~np.isin(t, list(_VISIT_IGNORE))]).size)
    row = dict(pid=int(k), projid=int(data[int(p2i[k, 0]), 0]),
               baseline_age=base_y, baseline_state=int(b), followup=fu,
               n_visits=n_visits, died=int(died), final_state=int(final_state))
    # BASELINE SLOT FOR EACH AUXILIARY SCALE, needed for its at-risk rule: a subject already in
    # a level cannot "reach" it. -1 means the scale was never recorded at or before baseline, and
    # such a subject is EXCLUDED from that scale's rows rather than treated as level 0 -- under
    # TV-gated emission a scale that was measured always emits at least its first value, so -1
    # really does mean absent.
    for _sc, _pairs in S.AUX_OF_SCALE.items():
        _ids = [tok for _sl, tok in _pairs]
        _m = np.isin(t, _ids) & (a <= base) & (a > NEG)
        row[f"baseline_{_sc}"] = (int(S.TOK2SLOT[int(t[_m][np.argmax(a[_m])])]) if _m.any()
                                  else -1)
    for i, nm in enumerate(ALL_NAMES):
        row[f"obs_t_{nm}"] = float(fp_obs[i])
    # predicted risk of reaching each slot by each horizon, competing with death
    for i, nm in enumerate(ALL_NAMES):
        for h in horizons:
            hd = base + h * D
            if i == DEATH_IDX:
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
        known = known | (GRID_YEARS >= fp_obs[DEATH_IDX])
    sim_grid = _state_grid_batch(A, T, b, grid_days)
    occ = np.zeros((len(GRID_YEARS), NSTATE), dtype=np.float32)
    for s in range(NSTATE):
        occ[:, s] = (sim_grid == s).mean(0)

    # ---------------- transition count matrices
    obsM = np.zeros((NSTATE, NSTATE)); _trans_counts(obs_seq, obsM)
    predM = np.zeros((NSTATE, NSTATE))
    _trans_counts_batch(A, T, predM)
    predM /= n                                   # per-patient expected counts

    # ---------------- the outcomes the four panels do not look at (perdomain.py)
    # Figure 2 itself scores only the cognitive staging and the two endpoints; this reuses the
    # trajectories already sampled above so the per-domain numbers describe the SAME sampled
    # futures and can be quoted beside the panels. Costs no extra MC.
    row.update(PD.per_domain_row(ages, toks, A, T, base, GRID_YEARS, horizons,
                                 death_sim=d_sim, obs_died_y=float(fp_obs[DEATH_IDX])))

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
    eng = _engine(ckpt, dataset, device)
    key = cache_key(eng.ckpt_sig, n_mc, limit, split, dataset)
    fpath = os.path.join(CACHE_DIR, f"frame_{key}.pkl")
    gpath = os.path.join(CACHE_DIR, f"grids_{key}.npz")
    tpath = os.path.join(CACHE_DIR, f"trans_{key}.npz")
    if not force and all(os.path.exists(p) for p in (fpath, gpath, tpath)):
        log.info("MC cache hit (%s)", key)
        return key

    data, p2i, _sub = eng.load_split(split)
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
def _hidden(eng, tokens, ages):
    """Final-block hidden states (post ln_f), plain causal mask -- i.e. EXACTLY the representation
    the model conditions on when it generates the future (generate() calls forward without
    targets, so the same-visit 'tie' mask is off). Returns (T, n_embd)."""
    store = {}
    h = eng.model.transformer.ln_f.register_forward_hook(lambda m, i, o: store.__setitem__("x", o))
    try:
        idx = torch.as_tensor(np.asarray(tokens)[None], dtype=torch.long, device=eng.device)
        age = torch.as_tensor(np.asarray(ages)[None], dtype=torch.float32, device=eng.device)
        eng.model(idx, age)
    finally:
        h.remove()
    return store["x"][0].float().cpu().numpy()


def build_embeddings(ckpt=CKPT, dataset=DATASET, split=SPLIT, device="cpu", seed=SEED,
                     limit=0, n_mc=N_MC, force=False):
    """Per-patient embeddings: last hidden state of (a) the baseline prompt -- the vector the model
    forecasts from, no future leakage -- and (b) the full observed history."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    eng = _engine(ckpt, dataset, device)
    key = cache_key(eng.ckpt_sig, n_mc, limit, split, dataset)
    epath = os.path.join(CACHE_DIR, f"emb_{key}.npz")
    if not force and os.path.exists(epath):
        log.info("embedding cache hit (%s)", key)
        return key
    data, p2i, _sub = eng.load_split(split)
    N = len(p2i) if not limit else min(len(p2i), limit)
    pid, E0, E1 = [], [], []
    t0 = time.time()
    for k in range(N):
        ages, toks, _pid = eng.stream(data, p2i, k)
        bp = _baseline(ages, toks)
        if bp is None:
            continue
        base, pmask, b = bp
        h_base = _hidden(eng, np.asarray(toks)[pmask], np.asarray(ages)[pmask])[-1]
        h_full = _hidden(eng, toks, ages)[-1]
        pid.append(k); E0.append(h_base); E1.append(h_full)
        if (k + 1) % 2000 == 0:
            log.info("  embeddings %d/%d (%.0fs)", k + 1, N, time.time() - t0)
    np.savez_compressed(epath, pid=np.array(pid), base=np.stack(E0).astype(np.float32),
                        full=np.stack(E1).astype(np.float32))
    log.info("embeddings cached -> %s (%d patients, dim=%d)", epath, len(pid), E0[0].shape[0])
    return key


# --------------------------------------------------------------------------- training cohort
# UPSTREAM HAS A TRAINING-COHORT FILTER AND WE DO NOT.
#
# delphi-fine-tuning's train.py calls `filter_cohort` before fitting: a subject needs
# >= cohort_min_visits distinct visit ages, OR >= cohort_short_min_visits of them AND at least
# two staging tokens. On NACC that cut train from 38,687 to 16,262 (42%), which made scoring the
# whole test split a measurement of the model on a population it was never fitted on -- hence
# the "matched" cohort, and hence this function.
#
# `../../delphi/train.py` is upstream gerstung-lab Delphi. It has no such filter: it fits on
# every subject in train.bin. So on THIS project "matched" and "all" describe the same people,
# and the distinction is preserved rather than deleted for two reasons: the figure should keep
# saying which population it describes, and the moment a filter is added to training this is
# where evaluation picks it up. The mask below reports how many rows it dropped, so "matched"
# can never quietly come to mean "all" without the log saying so.
COHORT_MIN_VISITS = 2
COHORT_SHORT_MIN_VISITS = 2


def training_cohort_mask(dataset=DATASET, split=SPLIT, min_visits=COHORT_MIN_VISITS,
                         short_min_visits=COHORT_SHORT_MIN_VISITS):
    """Boolean mask over the subject ROWS of `split`: True = inside the training cohort.

    NO MODEL, DELIBERATELY -- inherited from upstream, where taking a checkpoint here meant a
    default that loaded the wrong model against the wrong dataset and tripped the vocabulary
    guard. The cohort rule is a property of the DATA.
    """
    from radc_delphi.batching import filter_cohort, get_p2i    # noqa: E402
    d = os.path.join(HERE, "data", dataset)
    data = np.fromfile(os.path.join(d, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
    p2i = get_p2i(data)
    res = V.resolve_csv(os.path.join(d, "labels.csv"))
    kept = filter_cohort(data, p2i, min_visits, short_min_visits,
                         stage_disk=tuple(t - 1 for t in res.SCALES["MMSE"]),
                         ignored_disk=tuple(t - 1 for t in res.IGNORE_TOKENS if t > 0))
    # filter_cohort returns the SURVIVING p2i rows; map them back to a boolean mask over the
    # original rows by their start offset, which is unique per subject.
    starts = {int(x) for x in np.asarray(kept)[:, 0]}
    mask = np.array([int(p2i[k, 0]) in starts for k in range(len(p2i))], dtype=bool)
    log.info("training-cohort filter on %s/%s: %d of %d subjects kept (%.1f%%)",
             dataset, split, int(mask.sum()), len(mask), 100 * mask.mean())
    # The ones it drops have < 2 distinct visit ages -- but _baseline() already drops exactly
    # those, for the same reason (nothing to predict), BEFORE they reach the cache. So on this
    # project the mask removes nobody who was in the frame, and "matched" and "all" produce
    # identical figures -- expect `n_dropped: 0` in metrics_matched.json's _meta, which is
    # where figure2_panels.load_all records what this mask actually removed. If that number is
    # ever non-zero the two cohorts have genuinely diverged and the figures must be read apart.
    return mask


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
    death = fp[:, DEATH_IDX, :]
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
    eng = _engine(ckpt, dataset, device)
    key = cache_key(eng.ckpt_sig, n_mc, limit, split, dataset)
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
