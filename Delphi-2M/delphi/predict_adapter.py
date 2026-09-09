"""
predict_adapter.py -- the ONE thin abstraction all evaluation code depends on.

No evaluation module should ever touch model internals; it goes through this adapter.
Everything here is grounded in README.md sec.8-10 (repo root). The critical facts baked in:

  * Model = Delphi-style GPT. forward(idx, age) -> (logits, loss, att); logits (B,T,vocab=111).
    Token space is MODEL space (== labels.csv row index); ages are in DAYS from birth.
  * TWO heads share `logits`: (a) categorical next-state = softmax(logits); (b) time-to-next-event
    = the same logits as log-rates of competing exponentials (draw dt_k = -exp(-logit_k)*log U, min).
  * GOTCHA (README.md sec.10): ignore_tokens (0..21: pad/no-event/sex/BMI/smoking/alcohol/
    education/APOE) were never trained as targets, so their logits are UNCONSTRAINED and dominate
    the raw full-vocab softmax. => every full-vocab probability/rate is CONTENT-RENORMALIZED over
    the non-ignored tokens (22..110). Per-scale restricted softmax is unaffected.
  * Time-to-event comes from MONTE-CARLO trajectory sampling via model.generate (which correctly
    masks ignored tokens), NOT from a closed-form rate. Delphi is autoregressive; there is no
    closed-form horizon risk, so predict_state_at() samples.

Outcomes (README.md sec.8.5), model-space token ids, low->high severity (higher = worse):
  CDR boxes 111-139 (6 domains) | FAQ domains 140-175 (9) | NPI-Q symptoms 176-223 (12) |
  GDS 224-227 | MOCA 29-32 | NACCUDSD 106-109 | Death 110.  -- 29 scales in total.
Every sum-score has been replaced by its items: CDRSUM (24-28), the FAQ total (33-36) and
the NPI-Q total (37-40) are all RETIRED dead slots. A total is a deterministic function of
its items, so feeding both is redundant and leaks within a visit; and the domain-level
trajectories are themselves prediction targets, not just features.
Every SCALES consumer below is generic over bin count, which is what lets PERSCARE carry
4 bins next to the other CDR domains' 5.
"""
import os, sys, hashlib, numpy as np, torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Delphi-2M/ (package root)
sys.path.insert(0, HERE)
from delphi.ad_engine import load_model as _load_model_ckpt   # noqa: E402
from delphi.utils import get_p2i, patient_stream               # noqa: E402  (disk->model +1 lift)

# ------------------------------------------------------------------ domain constants (README.md sec.8.5)
_CDR_BOX_LABELS = ["None", "Questionable", "Mild", "Moderate", "Severe"]
SCALES = {
    # six CDR domains, contiguous 111-139, low->high severity within each block
    "MEMORY":   [111, 112, 113, 114, 115],
    "ORIENT":   [116, 117, 118, 119, 120],
    "JUDGMENT": [121, 122, 123, 124, 125],
    "COMMUN":   [126, 127, 128, 129, 130],
    "HOMEHOBB": [131, 132, 133, 134, 135],
    "PERSCARE": [136, 137, 138, 139],  # NO 0.5 level in the CDR form -> 4 bins, not 5
    "MOCA":     [29, 30, 31, 32],
    "NACCUDSD": [106, 107, 108, 109],
    # nine FAQ domains, 4 levels each, contiguous 140-175
    **{c: [140 + 4*k + v for v in range(4)] for k, c in enumerate(
        ["BILLS","TAXES","GAMES","STOVE","MEALPREP","EVENTS","PAYATTN","REMDATES","TRAVEL"])},
    # twelve NPI-Q symptoms, 4 levels each, contiguous 176-223
    **{c: [176 + 4*k + v for v in range(4)] for k, c in enumerate(
        ["DEL","HALL","AGIT","DEPD","ANX","ELAT","APA","DISN","IRR","MOT","NITE","APP"])},
    "GDS":      [224, 225, 226, 227],
}
FAQ_DOMAINS = ["BILLS","TAXES","GAMES","STOVE","MEALPREP","EVENTS","PAYATTN","REMDATES","TRAVEL"]
NPI_SYMPTOMS = ["DEL","HALL","AGIT","DEPD","ANX","ELAT","APA","DISN","IRR","MOT","NITE","APP"]
SCALE_LABELS = {
    "MEMORY":   _CDR_BOX_LABELS,
    "ORIENT":   _CDR_BOX_LABELS,
    "JUDGMENT": _CDR_BOX_LABELS,
    "COMMUN":   _CDR_BOX_LABELS,
    "HOMEHOBB": _CDR_BOX_LABELS,
    "PERSCARE": ["None", "Mild", "Moderate", "Severe"],
    "MOCA":     ["Normal", "MCI", "Moderate", "Severe"],
    "NACCUDSD": ["Normal", "Impaired", "MCI", "Dementia"],
    **{c: ["Normal", "Difficulty", "Assistance", "Dependent"] for c in FAQ_DOMAINS},
    **{c: ["Absent", "Mild", "Moderate", "Severe"] for c in NPI_SYMPTOMS},
    "GDS":      ["Normal", "Mild", "Moderate", "Severe"],
}
CDR_BOXES = ["MEMORY", "ORIENT", "JUDGMENT", "COMMUN", "HOMEHOBB", "PERSCARE"]

# Tokens that may legitimately be emitted more than once in a trajectory: every ordinal scale
# level. These encode a CURRENT STATE under keep-transitions, and states recur -- Normal -> MCI
# -> Normal emits Normal twice, which is exactly the recovery the tokenizer was changed to
# preserve. Everything NOT in here (diseases, medications, Death) is keep-first and must stay
# blocked, because a second "onset of hypertension" is meaningless.
# Opt out with DELPHI_ALLOW_SCALE_REPEATS=0 to reproduce the delivered sampler.
REPEATABLE_TOKENS = sorted({t for ids in SCALES.values() for t in ids})
DEATH = 110
NO_EVENT = 1
# Named NACCUDSD milestones used across time-to-event panels:
NACCUDSD_MCI = 108
NACCUDSD_DEMENTIA = 109
TOK2SCALE = {t: s for s, ts in SCALES.items() for t in ts}
TOK2BIN = {t: i for s, ts in SCALES.items() for i, t in enumerate(ts)}

DAYS_PER_YEAR = 365.25


# ============================================================================ model loading
class Adapter:
    """Holds the model + the checkpoint's config, and everything derived from it."""

    def __init__(self, model, args, device, seed=0, cache_dir=None):
        self.model = model
        self.args = args
        self.device = device
        self.seed = int(seed)
        self.vocab = int(args["vocab_size"])
        self.block_size = int(args["block_size"])
        self.t_min = float(args.get("t_min", 30.4375))
        self.ignore = sorted(int(t) for t in args.get("ignore_tokens", [0]))
        # tokens the model was actually trained to predict (content space)
        self.predict_ids = np.array([t for t in range(self.vocab) if t not in set(self.ignore)],
                                     dtype=np.int64)
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        # a short signature of (checkpoint identity + seed) for cache keys
        self.ckpt_sig = args.get("_ckpt_sig", "nosig")

        # OPT-IN: visit-batched sampling. Set DELPHI_VISIT_SIZES to a .npy of observed
        # tokens-per-visit counts and model.generate() emits a whole visit per step instead of
        # one token, all at a single age. Unset (the default) reproduces the delivered sampler
        # exactly. Why: generation emits 1.03 tokens per visit where the data has 4.11, so one
        # real visit costs several simulated steps and trajectories run 2.5x too slow
        # (experiments/time_head/gen_steps_probe.py, removed from this branch; see dba88f8).
        #
        # The cache signature is EXTENDED when it is on. simulate_trajectory caches by
        # ckpt_sig, and changing the sampler without changing the key would silently re-serve
        # the old trajectories -- the run would look like the change did nothing.
        # Scale levels are exempt from no_repeat by default (see REPEATABLE_TOKENS). This
        # CHANGES sampled trajectories, so it extends the cache signature -- serving cached
        # trajectories built under the old sampler would make the change look like a no-op.
        self.repeatable_tokens = None
        if os.environ.get("DELPHI_ALLOW_SCALE_REPEATS", "1").strip().lower() \
                not in ("0", "off", "no", "false"):
            self.repeatable_tokens = REPEATABLE_TOKENS
            self.ckpt_sig = f"{self.ckpt_sig}|rep"

        self.visit_sizes = None
        _vs = os.environ.get("DELPHI_VISIT_SIZES", "").strip()
        if _vs and _vs.lower() not in ("0", "off", "none"):
            self.visit_sizes = np.load(_vs).astype(np.int64).ravel()
            self.ckpt_sig = (f"{self.ckpt_sig}|vb"
                             f"{hashlib.md5(self.visit_sizes.tobytes()).hexdigest()[:8]}")
            print(f"[adapter] visit-batched sampling ON: {len(self.visit_sizes):,} observed "
                  f"visit sizes (median {np.median(self.visit_sizes):.0f}, "
                  f"mean {self.visit_sizes.mean():.2f}) -- cache sig {self.ckpt_sig}")

    def content_mask(self):
        """Boolean (vocab,) True where a token is a real prediction target (not ignored)."""
        m = np.ones(self.vocab, dtype=bool)
        m[self.ignore] = False
        return m


def load_model(config):
    """load_model(config) -> Adapter.

    `config` may be a dict with keys:
        checkpoint (path), device ('cpu'/'cuda'), seed, cache_dir
    or a bare checkpoint path string (device defaults to cpu)."""
    if isinstance(config, str):
        config = {"checkpoint": config}
    ckpt = config["checkpoint"]
    device = config.get("device", "cpu")
    seed = int(config.get("seed", 0))
    cache_dir = config.get("cache_dir")
    _seed_everything(seed)
    model, args = _load_model_ckpt(ckpt, device)
    # stable checkpoint signature for caching (path + size + mtime)
    try:
        st = os.stat(ckpt)
        args = dict(args)
        args["_ckpt_sig"] = hashlib.md5(f"{os.path.abspath(ckpt)}:{st.st_size}:{int(st.st_mtime)}"
                                        .encode()).hexdigest()[:12]
    except OSError:
        pass
    return Adapter(model, args, device, seed=seed, cache_dir=cache_dir)


def _seed_everything(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ============================================================================ data access
def load_split(dataset, split, data_root=None):
    """Return (data, p2i) for a dataset/split. data is (N,3) uint32 [pid, age_days, disk_tok]."""
    root = data_root or os.path.join(HERE, "data")
    data = np.fromfile(os.path.join(root, dataset, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
    return data, get_p2i(data)


def get_history(data, p2i, k):
    """(ages_days float64, tokens model-space int64) for patient row k, age-sorted."""
    s, n = int(p2i[k, 0]), int(p2i[k, 1])
    return patient_stream(data, s, n)


# ============================================================================ one-step prediction
@torch.no_grad()
def _forward_logits(ad, tokens, ages, tie_mask=True):
    """Full causal forward over one stream; returns logits (T, vocab) as a torch tensor.
    tie_mask=True passes dummy targets so the same-visit ('tie') attention mask activates
    (matches training) -- required so a token doesn't peek at same-visit tokens."""
    idx = torch.as_tensor(tokens[None], dtype=torch.long, device=ad.device)
    age = torch.as_tensor(ages[None], dtype=torch.float32, device=ad.device)
    if tie_mask:
        tgt = torch.cat([idx[:, 1:], torch.zeros_like(idx[:, :1])], dim=1)
        tgt_age = torch.cat([age[:, 1:], age[:, -1:]], dim=1)
        logits = ad.model(idx, age, targets=tgt, targets_age=tgt_age)[0][0]
    else:
        logits = ad.model(idx, age)[0][0]
    return logits


@torch.no_grad()
def predict_next(ad, tokens, ages, position=-1):
    """One-step prediction given history up to `position` (default last).

    Returns dict:
      state_probs : (vocab,) CONTENT-RENORMALIZED softmax (ignored tokens set to 0, rest renorm).
      next_state  : int, argmax over content tokens.
      rate_or_time: expected DAYS to the next (real) event, content-renormalized with the t_min
                    floor: E[dt] = exp(-logsumexp(content_logits)) + t_min. Approximate; figures
                    use MC sampling. (See README.md sec.10.)
      scale_probs : dict scale -> restricted softmax over that scale's tokens (SAFE, not renormed).
    """
    logits = _forward_logits(ad, tokens, ages)
    row = logits[position]
    full = torch.softmax(row, -1).float().cpu().numpy()
    # content-renormalize over non-ignored tokens
    cm = ad.content_mask()
    p = np.where(cm, full, 0.0)
    tot = p.sum()
    p = p / tot if tot > 0 else p
    next_state = int(ad.predict_ids[np.argmax(p[ad.predict_ids])])
    # content-renormalized expected time to next event (approx; MC is primary elsewhere)
    content_logits = row[torch.as_tensor(ad.predict_ids, device=row.device)]
    lse = torch.logsumexp(content_logits, -1)
    rate_or_time = float(torch.exp(-lse).item()) + ad.t_min       # days
    # DIAGNOSTIC ONLY (README.md sec.10): the closed-form rate is unreliable at some states
    # (can blow up to centuries). Clamp to a sane ceiling; all real timing uses MC sampling.
    rate_or_time = float(np.clip(rate_or_time, ad.t_min, 100.0 * DAYS_PER_YEAR))
    scale_probs = {}
    for s, ids in SCALES.items():
        sp = torch.softmax(row[torch.as_tensor(ids, device=row.device)], -1).float().cpu().numpy()
        scale_probs[s] = sp
    return dict(state_probs=p, next_state=next_state, rate_or_time=rate_or_time,
                scale_probs=scale_probs)


# ============================================================================ trajectory sampling
@torch.no_grad()
def simulate_trajectory(ad, tokens, ages, until_age_years=100.0, n_samples=100,
                        max_new_tokens=256, seed=None, use_cache=True, cache_key=None):
    # max_new_tokens was 96, tuned when a patient averaged ~21 events over a lifetime. After
    # every sum-score was split into its items the average is 55.4 (p95 100, max 250), so a
    # 96-step rollout hits the STEP cap before it reaches max_age or draws Death -- which
    # silently biases every first-passage time upward. 256 matches block_size.
    """Delphi-style autoregressive sampling of (state, time) beyond the given history.

    Returns dict:
      ages   : (n_samples, T') float, DAYS  (seed prefix + rollout; padded with -10000 after death)
      tokens : (n_samples, T') int, model-space (padded with 0 after death)
      seed_len : int, number of seed tokens (columns < seed_len are the given history)
    model.generate() already masks ignore_tokens and terminates on Death(110)/max_age.
    Results are cached to disk keyed by (ckpt, seed, key, n_samples, until_age)."""
    until_days = float(until_age_years) * DAYS_PER_YEAR
    s = ad.seed if seed is None else int(seed)
    seed_len = len(tokens)

    cpath = None
    if use_cache and ad.cache_dir and cache_key is not None:
        h = hashlib.md5(f"{ad.ckpt_sig}|{cache_key}|{s}|{n_samples}|{until_days:.0f}|"
                        f"{max_new_tokens}".encode()).hexdigest()[:16]
        cpath = os.path.join(ad.cache_dir, f"traj_{h}.npz")
        if os.path.exists(cpath):
            z = np.load(cpath)
            return dict(ages=z["ages"], tokens=z["tokens"], seed_len=int(z["seed_len"]))

    torch.manual_seed(s)
    idx = torch.as_tensor(np.tile(tokens, (n_samples, 1)), dtype=torch.long, device=ad.device)
    age = torch.as_tensor(np.tile(ages, (n_samples, 1)), dtype=torch.float32, device=ad.device)
    gi, ga, _ = ad.model.generate(idx, age, max_new_tokens=max_new_tokens, max_age=until_days,
                                  termination_tokens=[DEATH],
                                  visit_sizes=getattr(ad, "visit_sizes", None),
                                  repeatable_tokens=getattr(ad, "repeatable_tokens", None))
    out = dict(ages=ga.cpu().numpy().astype(np.float64),
               tokens=gi.cpu().numpy().astype(np.int64), seed_len=seed_len)
    if cpath is not None:
        np.savez_compressed(cpath, ages=out["ages"], tokens=out["tokens"],
                            seed_len=np.array(out["seed_len"]))
    return out


def time_to_state(traj_ages_row, traj_tokens_row, target_states, after_age_days=0.0):
    """First age (YEARS) at which any token in `target_states` appears after `after_age_days`
    in one sampled trajectory; None if never reached (censored).
    `target_states` may be an int or an iterable of ints (e.g. {108,109} for '>=MCI')."""
    if np.isscalar(target_states):
        target_states = {int(target_states)}
    else:
        target_states = set(int(t) for t in target_states)
    a = np.asarray(traj_ages_row); t = np.asarray(traj_tokens_row)
    valid = (a > after_age_days) & (a > -1e4 + 1)     # exclude post-death padding (age=-10000)
    hit = valid & np.isin(t, list(target_states))
    if not hit.any():
        return None
    return float(a[hit].min() / DAYS_PER_YEAR)


# ============================================================================ horizon risk (MC)
def _last_scale_state(ages_row, toks_row, scale_ids, at_day, baseline_state):
    """State (bin index within scale) at time `at_day`: the last scale token emitted at/before
    at_day; if none in the trajectory, carry-forward the baseline_state."""
    a = np.asarray(ages_row); t = np.asarray(toks_row)
    m = (a <= at_day) & (a > -1e4 + 1) & np.isin(t, scale_ids)
    if not m.any():
        return baseline_state
    j = np.where(m)[0]
    last = j[np.argmax(a[j])]
    return TOK2BIN[int(t[last])]


def predict_state_at(ad, tokens, ages, horizons=(1, 2, 3, 5, 10), n_samples=100,
                     baseline_age_years=None, seed=None, cache_key=None, cache_dir_ok=True):
    """Risk/expected outcome at fixed future horizons (years), via MC trajectory sampling.

    Returns dict[horizon_years -> dict]:
      per ordinal scale: {'occupancy': [p_bin...], 'expected_bin': float,
                          'p_reach_MCI'/'p_reach_Dementia' for NACCUDSD (competing-risk aware)}
      'Death': {'cum_incidence': float}
    'reach' events are competing-risk aware: a target counts as reached by horizon h only if the
    simulated first-passage age to the target is <= baseline+h AND <= the simulated death age.
    """
    base_day = (float(baseline_age_years) * DAYS_PER_YEAR if baseline_age_years is not None
                else float(ages[ages > 0].max()) if np.any(ages > 0) else float(ages.max()))
    until_years = (base_day / DAYS_PER_YEAR) + max(horizons) + 1.0
    sim = simulate_trajectory(ad, tokens, ages, until_age_years=until_years, n_samples=n_samples,
                              seed=seed, use_cache=cache_dir_ok, cache_key=cache_key)
    A, T = sim["ages"], sim["tokens"]
    n = A.shape[0]

    # baseline state per scale from the OBSERVED history (last observed scale token)
    base_state = {}
    for scl, ids in SCALES.items():
        bs = _last_scale_state(ages, tokens, ids, base_day, baseline_state=0)
        base_state[scl] = bs

    # per-sample first-passage ages (days) to key milestones + death, INCIDENT (strictly after
    # base_day) so already-progressed patients don't score as "reached" at every horizon and so
    # this composes with at-risk cohort selection in the panels. Death competes (handled below).
    def first_age(tok_set, arr_a, arr_t):
        m = np.isin(arr_t, list(tok_set)) & (arr_a > base_day) & (arr_a > -1e4 + 1)
        out = np.full(n, np.inf)
        for i in range(n):
            ai = arr_a[i][m[i]]
            if ai.size:
                out[i] = ai.min()
        return out
    death_age = first_age({DEATH}, A, T)
    mci_age = first_age({NACCUDSD_MCI, NACCUDSD_DEMENTIA}, A, T)   # reach >= MCI
    dem_age = first_age({NACCUDSD_DEMENTIA}, A, T)

    res = {}
    for h in horizons:
        h_day = base_day + h * DAYS_PER_YEAR
        cell = {}
        for scl, ids in SCALES.items():
            occ = np.zeros(len(ids))
            for i in range(n):
                b = _last_scale_state(A[i], T[i], ids, h_day, baseline_state=base_state[scl])
                occ[b] += 1
            occ /= n
            cell[scl] = dict(occupancy=occ.tolist(),
                             expected_bin=float(np.dot(np.arange(len(ids)), occ)))
        # NACCUDSD competing-risk reach probabilities
        cell["NACCUDSD"]["p_reach_MCI"] = float(np.mean((mci_age <= h_day) & (mci_age <= death_age)))
        cell["NACCUDSD"]["p_reach_Dementia"] = float(np.mean((dem_age <= h_day) & (dem_age <= death_age)))
        cell["Death"] = dict(cum_incidence=float(np.mean(death_age <= h_day)))
        res[h] = cell
    return res


# ============================================================================ convenience
def name_of(mtok, labels):
    return labels[mtok] if 0 <= mtok < len(labels) else f"?{mtok}"
