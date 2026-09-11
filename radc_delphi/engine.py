"""
engine.py -- inference. The ONE place evaluation code is allowed to touch the model.

    from radc_delphi.engine import load, Engine
    eng = load("out-radc-base-s42/ckpt.pt", data_dir="data/radc-s42", device="cuda")
    risk = eng.risk_by_horizon(tokens, ages, horizons=(1, 3, 5), n_mc=200)

Every downstream number goes through this module so that the three things that are easy to get
silently wrong are got right exactly once:

  1. TERMINATION. Rollouts MUST stop on the Death token. Left unterminated, the 53.3% of
     non-converters who actually died stay alive in simulation and keep accruing dementia
     hazard, which overstates cumulative AD risk by roughly the 40-60% that treating death as
     censoring does in the observed direction. model.generate() defaults to an EMPTY
     termination list on purpose -- it is vocabulary-agnostic -- so the caller owns this, and
     the caller is this file.

  2. THE IGNORED COLUMNS. Tokens the model was never trained to emit as targets (padding and
     the 21 statics) have unconstrained logits. Any full-vocabulary softmax is therefore
     dominated by columns that mean nothing. Every probability returned here is renormalised
     over the trained content columns.

  3. TOKEN SPACE. Everything crossing this API is MODEL space, which is also the labels.csv
     row index. The +1 disk shift belongs to batching.get_batch and to nothing else.
"""
import os
import hashlib

import numpy as np
import pandas as pd
import torch

from . import vocab as V
from .model import Delphi, DelphiConfig
from .batching import get_p2i, patient_stream, get_batch

DAYS_PER_YEAR = 365.25
# What generate() writes into a position that is past death or past max_age. Any code
# reading a simulated age must exclude these; they are not ages.
PAD_AGE = -10000.0


def _fingerprint(path):
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()[:12]


def load(ckpt_path, data_dir=None, device="cpu", strict_vocab=True):
    """Load a checkpoint into an Engine, verifying it against the vocabulary it was trained on.

    `strict_vocab` compares the checkpoint's recorded labels.csv fingerprint against the one in
    `data_dir`. This is not paranoia: in the previous arm a checkpoint could be scored against
    a differently-sized vocabulary, load without error, and mis-index every clinical label.
    """
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = dict(ck["model_args"])
    if args["vocab_size"] != V.VOCAB_SIZE:
        raise RuntimeError(
            f"checkpoint vocab_size {args['vocab_size']} != radc_delphi.vocab's "
            f"{V.VOCAB_SIZE}. This checkpoint belongs to a different tokenization.")
    if strict_vocab and data_dir:
        lp = os.path.join(data_dir, "labels.csv")
        if os.path.exists(lp) and ck.get("vocab_sig"):
            sig = _fingerprint(lp)
            if sig != ck["vocab_sig"]:
                raise RuntimeError(
                    f"vocabulary mismatch: checkpoint was trained against labels "
                    f"{ck['vocab_sig']}, {lp} is {sig}. Pass strict_vocab=False only if you "
                    f"know why they differ.")
    valid = {f for f in DelphiConfig.__dataclass_fields__}
    model = Delphi(DelphiConfig(**{k: v for k, v in args.items() if k in valid}))
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()}
    model.load_state_dict(sd)
    model.to(device).eval()
    return Engine(model, args, device, ckpt=ck, data_dir=data_dir,
                  ckpt_path=ckpt_path, ckpt_sig=_fingerprint(ckpt_path))


class Engine:
    def __init__(self, model, args, device, ckpt=None, data_dir=None, ckpt_path=None,
                 ckpt_sig=None):
        self.model = model
        self.args = args
        self.device = device
        self.block_size = int(args["block_size"])
        self.ckpt = ckpt or {}
        self.data_dir = data_dir
        self.ckpt_path = ckpt_path
        # md5 of the checkpoint file. Every cached rollout is keyed on this: a cache hit
        # against a different set of weights is silent and produces plausible numbers.
        self.ckpt_sig = ckpt_sig
        # tokens the model was trained to emit; everything else is an unconstrained column
        ignore = set(int(t) for t in args.get("ignore_tokens", V.IGNORE_TOKENS))
        self.content_ids = np.array([t for t in range(V.VOCAB_SIZE) if t not in ignore],
                                    dtype=np.int64)
        # observed tokens-per-visit, so a simulated visit emits as many tokens as a real one
        self.visit_sizes = None
        if data_dir:
            vs = os.path.join(data_dir, "visit_sizes.npy")
            if os.path.exists(vs):
                self.visit_sizes = np.load(vs).astype(np.int64).ravel()

    # ------------------------------------------------------------------ data access
    def load_split(self, split, data_dir=None):
        """(data, p2i, subjects) for one split. `subjects` is indexed by projid."""
        d = data_dir or self.data_dir
        arr = np.fromfile(os.path.join(d, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
        sub = pd.read_csv(os.path.join(d, "subjects.csv")).set_index("projid")
        return arr, get_p2i(arr), sub.loc[sub["split"] == split]

    @staticmethod
    def stream(data, p2i, k):
        """(ages_days, model_tokens) for the k-th subject of a split, plus its pid."""
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        ages, toks = patient_stream(data, s, n)
        return ages, toks, int(data[s, 0])

    @staticmethod
    def prefix_at(ages, toks, cut_day):
        """The subject's stream truncated to tokens at or before `cut_day`."""
        m = ages <= cut_day
        return ages[m], toks[m]

    # ------------------------------------------------------------------ single-step
    @torch.no_grad()
    def next_token_probs(self, tokens, ages, renormalise=True):
        """P(next token) after the given history, over the trained content columns."""
        idx = torch.as_tensor(np.asarray(tokens[-self.block_size:])[None, :],
                              dtype=torch.long, device=self.device)
        age = torch.as_tensor(np.asarray(ages[-self.block_size:])[None, :],
                              dtype=torch.float32, device=self.device)
        logits, _, _ = self.model(idx, age)
        p = torch.softmax(logits[0, -1], -1).cpu().numpy()
        if renormalise:
            keep = np.zeros_like(p)
            keep[self.content_ids] = p[self.content_ids]
            tot = keep.sum()
            p = keep / tot if tot > 0 else keep
        return p

    # ------------------------------------------------------------------ rollout
    @torch.no_grad()
    def simulate(self, tokens, ages, n_mc=200, until_age_years=110.0, max_new_tokens=96,
                 seed=0, use_visit_sizes=True):
        """Monte-Carlo forward simulation from a prefix.

        Returns (sim_tokens, sim_ages) of shape (n_mc, T) in MODEL space, with the prefix
        included. Padded positions carry token 0 and age -10000.
        """
        torch.manual_seed(seed)
        pre_t = tokens[-self.block_size:]
        pre_a = ages[-self.block_size:]
        idx = torch.as_tensor(np.tile(np.asarray(pre_t), (n_mc, 1)),
                              dtype=torch.long, device=self.device)
        age = torch.as_tensor(np.tile(np.asarray(pre_a), (n_mc, 1)),
                              dtype=torch.float32, device=self.device)
        gi, ga, _ = self.model.generate(
            idx, age,
            max_new_tokens=max_new_tokens,
            max_age=until_age_years * DAYS_PER_YEAR,
            # Death is absorbing. Without this the simulated cohort is immortal.
            termination_tokens=list(V.TERMINATION_TOKENS),
            # No-event is a training target but must never be SAMPLED -- it is a synthetic
            # marker, not a clinical event, and it would otherwise dominate the draw.
            extra_ignore=[V.NO_EVENT],
            # Ordinal bins and the recurring onset/medication events may repeat; the
            # keep-first tokens may not.
            repeatable_tokens=list(V.REPEATABLE_TOKENS),
            visit_sizes=self.visit_sizes if use_visit_sizes else None,
        )
        return gi.cpu().numpy(), ga.cpu().numpy()

    # ------------------------------------------------------------------ endpoints
    def risk_by_horizon(self, tokens, ages, horizons=(1, 3, 5), n_mc=200, seed=0,
                        from_day=None, competing_death=True, max_new_tokens=96):
        """P(AD diagnosis within H years of the prefix end), one entry per horizon.

        With competing_death=True (the default and the only defensible setting for this
        cohort) the denominator is every simulated trajectory: a trajectory in which the
        subject dies before being diagnosed counts as a NON-event, not as a censored draw.
        That is the cause-specific cumulative incidence, and it is what the observed
        Aalen-Johansen estimate it will be compared against measures.
        """
        t0 = float(ages[-1]) if from_day is None else float(from_day)
        horizon_days = [t0 + h * DAYS_PER_YEAR for h in horizons]
        gi, ga = self.simulate(tokens, ages, n_mc=n_mc, seed=seed,
                               max_new_tokens=max_new_tokens,
                               until_age_years=(t0 / DAYS_PER_YEAR) + max(horizons) + 1)
        out = {}
        big = 1e18
        ad_at = np.where((gi == V.AD_DX) & (ga > t0), ga, big).min(1)
        death_at = np.where((gi == V.DEATH) & (ga > t0), ga, big).min(1)
        for h, hd in zip(horizons, horizon_days):
            # ad_at <= death_at, not <: generate() draws a whole visit at one age, so an AD
            # token and a Death token land on the same simulated day in 1.5% of draws. The
            # tokenizer's own convention is that the diagnosis precedes the death of the same
            # visit (it forces death strictly after every other token, and sorts the diagnosis
            # first within a tie), so scoring a tie as a competing death would use a different
            # convention on the simulated side than on the observed side it is compared with.
            got_ad = (ad_at <= hd) & (ad_at <= death_at)
            if competing_death:
                out[h] = float(got_ad.mean())
            else:
                alive = death_at > hd
                out[h] = float(got_ad[alive].mean()) if alive.any() else float("nan")
        return out

    def event_age_distribution(self, tokens, ages, token_id, n_mc=200, seed=0,
                               until_age_years=110.0):
        """Simulated ages (years) at which `token_id` first occurs; NaN where it never does."""
        gi, ga = self.simulate(tokens, ages, n_mc=n_mc, seed=seed,
                               until_age_years=until_age_years)
        t0 = float(ages[-1])
        big = 1e18
        first = np.where((gi == token_id) & (ga > t0), ga, big).min(1)
        first = np.where(first >= big, np.nan, first / DAYS_PER_YEAR)
        return first

    # ------------------------------------------------------------------ likelihood
    @torch.no_grad()
    def split_loss(self, split, batch_size=128, data_dir=None, no_event_token_rate=None):
        """Deterministic full-pass (loss_ce, loss_dt) on one split, under validation_loss_mode.

        Reproduces train.py's estimate_loss for the eval_full=True branch exactly -- same
        select='left', same padding='regular', no age augmentation, same subject weighting --
        so the number printed here is the same quantity the run was checkpointed on and the
        two are directly comparable. `padding='regular'` places the synthetic no-event markers
        deterministically inside each observation window, which is why re-running this returns
        a bit-identical number rather than one that wanders with the marker draw.

        validation_loss_mode additionally masks id 1 (No event) out of the loss, so loss_ce is
        the cross-entropy over REAL clinical tokens only and stays comparable to a run that
        never emitted markers at all.

        loss_dt IS returned, and it is NOT a timing-calibration result. 93.9% of nominal
        inter-visit intervals in this cohort are exactly one year because RADC is an annual
        protocol, so the waiting-time head is scored almost entirely on the study's visit
        calendar. Only two tokens sit on a real clock (the AD diagnosis and death).

        COMPUTED UNMASKED, and that makes it a different quantity from the val loss train.py
        prints when mask_missing_ad_label is on. There, the 287 subjects with no AD label have
        the AD column blanked from their cross-entropy; here every subject is scored the same
        way, because this path reads only the .bin and takes no view on whose label is
        trustworthy. The two numbers are each internally consistent -- do not compare them.
        """
        d = data_dir or self.data_dir
        arr = np.fromfile(os.path.join(d, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
        p2i = get_p2i(arr)
        if no_event_token_rate is None:
            no_event_token_rate = self.ckpt.get("config", {}).get("no_event_token_rate", 2)
        tot_ce = tot_dt = tot_w = 0.0
        for i in range(0, len(p2i), batch_size):
            ix = torch.arange(i, min(i + batch_size, len(p2i)))
            X, A, Y, B = get_batch(ix, arr, p2i, block_size=self.block_size,
                                   device=self.device, select='left', padding='regular',
                                   augment_tokens=None,
                                   no_event_token_rate=no_event_token_rate, cut_batch=True)
            _, loss, _ = self.model(X, A, Y, B, validation_loss_mode=True)
            w = float(len(ix))
            tot_ce += float(loss["loss_ce"]) * w
            tot_dt += float(loss["loss_dt"]) * w
            tot_w += w
        return {"loss_ce": tot_ce / tot_w, "loss_dt": tot_dt / tot_w,
                "loss_total": (tot_ce + tot_dt) / tot_w, "n_subjects": int(len(p2i))}

    @staticmethod
    def scale_track(sim_tokens, sim_ages, scale, at_days, baseline_bin=None):
        """Last emitted bin INDEX of `scale` at or before each of `at_days`, per trajectory.

        Returns (n_mc, len(at_days)) of ints, filled with -1 (or `baseline_bin`) wherever the
        scale has not fired yet. The index is the position in vocab.SCALES[scale], which for
        the two cognitive scales runs WORST FIRST -- 0 is "MMSE 0-17", not "MMSE 27-30". Any
        "mean bin index" read off this therefore goes UP as the cohort does better.

        Vectorised over draws and days on purpose: the trajectory-realism panel asks for two
        scales at 16 ages for every draw of every subject, which is ~2.4 million lookups on
        the val split alone and unusable as a per-draw python loop.
        """
        ids = np.asarray(V.SCALES[scale])
        lut = np.full(V.VOCAB_SIZE, -1, dtype=np.int64)
        lut[ids] = np.arange(len(ids))
        real = np.isin(sim_tokens, ids) & (sim_ages > PAD_AGE + 1)
        bins = lut[sim_tokens]
        rows = np.arange(sim_tokens.shape[0])
        fill = -1 if baseline_bin is None else int(baseline_bin)
        out = np.full((sim_tokens.shape[0], len(at_days)), fill, dtype=np.int64)
        for j, d in enumerate(at_days):
            m = real & (sim_ages <= d)
            last = np.where(m, sim_ages, -np.inf).argmax(1)
            out[:, j] = np.where(m.any(1), bins[rows, last], fill)
        return out

    def scale_state_at(self, sim_tokens, sim_ages, scale, at_day, baseline_bin=None):
        """Last emitted bin INDEX of `scale` at or before `at_day`, per simulated trajectory.

        Returns an int array with -1 where the scale never fired and no baseline was given.
        """
        return self.scale_track(sim_tokens, sim_ages, scale, [at_day], baseline_bin)[:, 0]


def first_event_ages(sim_tokens, sim_ages, token_id, after_day=None):
    """Age (YEARS) at which `token_id` first fires in each simulated trajectory; NaN if never.

    Reads a batch that simulate() already produced, rather than re-simulating. That matters
    for the competing-risk endpoints: the AD age and the death age of one trajectory have to
    come from THE SAME draw, and Engine.event_age_distribution -- which runs its own rollout
    per call -- silently gives them different ones.
    """
    big = 1e18
    hit = (sim_tokens == token_id) & (sim_ages > PAD_AGE + 1)
    if after_day is not None:
        hit &= sim_ages > after_day
    first = np.where(hit, sim_ages, big).min(1)
    return np.where(first >= big, np.nan, first / DAYS_PER_YEAR)


def sim_end_ages(sim_ages):
    """Age (YEARS) of the last real position of each simulated trajectory.

    A rollout that hit max_new_tokens without dying simply stops emitting. It is not a subject
    who was observed to be stable to age 110 -- it is a trajectory whose follow-up ran out, and
    any state read past this age is carried-forward padding, not a prediction.
    """
    return np.where(sim_ages > PAD_AGE + 1, sim_ages, -np.inf).max(1) / DAYS_PER_YEAR


# ---------------------------------------------------------------------- observed side
def aalen_johansen(event_age, event_type, entry_age, grid_ages):
    """Cause-specific cumulative incidence of event_type == 1, with 2 as the competing event.

    LEFT TRUNCATION IS NOT OPTIONAL HERE. Subjects enter at age_bl (median 78.9), so a subject
    is not at risk on the age axis before they enrol; ignoring that would put the whole cohort
    in the denominator from birth and drive every incidence estimate towards zero. `entry_age`
    is each subject's age at entry and the risk set at age a is {entry <= a < exit}.

    event_type: 1 = AD, 2 = death without AD, 0 = censored alive.
    All ages in YEARS. Returns the CIF of cause 1 evaluated on `grid_ages`.
    """
    event_age = np.asarray(event_age, float)
    event_type = np.asarray(event_type, int)
    entry_age = np.asarray(entry_age, float)
    order = np.argsort(event_age)
    ea, et, en = event_age[order], event_type[order], entry_age[order]

    times = np.unique(ea[et > 0])
    surv = 1.0                                    # overall survival S(t-)
    cif = 0.0
    curve = []
    ti = 0
    for t in times:
        at_risk = int(((en <= t) & (ea >= t)).sum())
        if at_risk == 0:
            continue
        d1 = int(((ea == t) & (et == 1)).sum())
        d2 = int(((ea == t) & (et == 2)).sum())
        cif += surv * d1 / at_risk               # increment uses S BEFORE this time
        surv *= (1.0 - (d1 + d2) / at_risk)
        curve.append((t, cif))
    if not curve:
        return np.zeros(len(grid_ages)), []
    ct = np.array([c[0] for c in curve])
    cv = np.array([c[1] for c in curve])
    out = np.array([cv[ct <= g][-1] if (ct <= g).any() else 0.0 for g in grid_ages])
    return out, curve


def risk_set_size(entry_age, exit_age, at_age):
    """How many subjects are actually at risk at `at_age`. Quote this with every CIF value:
    the curve is estimated from 1,619 subjects at age 85 but only 393 at 95."""
    return int(((np.asarray(entry_age) <= at_age) & (np.asarray(exit_age) >= at_age)).sum())
