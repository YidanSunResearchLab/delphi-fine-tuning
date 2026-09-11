import numpy as np
import torch
import re


def get_p2i(data):
    """
    Get the patient to index mapping.
    """

    px = data[:, 0].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            # add last participant
            p2i.append([j, i - j + 1])
    return np.array(p2i)


def patient_stream(data, s, n):
    """Age-sorted (ages_days, model_tokens) for one patient's slice of `data`.

    Tokens are lifted from DISK space to MODEL space (+1), the same shift get_batch applies,
    so callers work in one consistent space (model token == labels.csv row index).
    `s`, `n` are the (start, length) pair from get_p2i.
    """
    sl = data[s:s + n]
    return sl[:, 1].astype(np.float64), sl[:, 2].astype(np.int64) + 1


def get_batch(ix, data, p2i, select='center', index='patient', padding='regular',
              block_size=48, device='cpu', augment_tokens=None,
              augment_age_range_days=(-20 * 365, 40 * 365),
              no_event_token_rate=5, cut_batch=False, values=None, binner=None):
    """
    Get a batch of data from the dataset. This function packs sequences in a batch and also
    inserts "no event" tokens randomly with the average rate of one every five years.

    Args:
        ix: list of indices to get data from
        data: numpy array of the dataset
        p2i: numpy array of the patient to index mapping
        select: 'center', 'right', 'smart_random', 'smart_right'
        index: 'patient', 'random'
        padding: 'regular', 'random'
        block_size: size of the block to get
        device: 'cpu' or 'cuda'
        augment_tokens: DISK-space token ids whose ages get jittered (immortality-bias
            augmentation). Either an inclusive (lo, hi) tuple or an iterable of ids.
            None (default) disables the augmentation entirely.
        augment_age_range_days: (low, high) uniform day offset applied to those tokens
        no_event_token_rate: average rate of "no event" tokens in years
        cut_batch: whether to cut the batch to the smallest size possible
        values: parallel float32 array of RAW scale readings (NaN off-scale), aligned with
            `data` row for row. None -> hard one-hot behaviour, exactly as before.
        binner: a softlabel.SoftBinner. Required when `values` is given.

    Returns:
        x, a, y, b            input tokens / ages, target tokens / ages
        (Wx, Ix), (Wy, Iy)    soft weights and their MODEL ids for input and target, each
                              (B, T, K). Returned only when `values` is given; a one-hot W
                              reproduces the hard path bit for bit, so a caller that ignores
                              them loses nothing.
    """

    mask_time = -10000.

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())  # we want some things be random, but also deterministic

    if index == 'patient':
        if select == 'left':
            traj_start_idx = x[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = x[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) % torch.clamp(x[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    mask = torch.from_numpy(data[:, 0][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, 0], 0][:, None].astype(np.int64)).to(mask.dtype)

    tokens = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    ages = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))
    vals = (torch.from_numpy(np.asarray(values)[batch_idx].astype(np.float32))
            if values is not None else None)

    # Augment the ages of "trait" tokens to avoid immortality bias.
    #
    # WHY THE TOKEN SET IS NOW AN ARGUMENT. Upstream this read
    #     lifestyle_idx = (tokens >= 3) * (tokens <= 11)
    # -- a literal DISK-space id range inherited from the UK Biobank vocabulary and carried
    # unchanged through the NACC fork, where it happened to land on BMI/smoking/alcohol. On any
    # other tokenization those same ids mean something else entirely, and the augmentation then
    # jitters the ages of arbitrary clinical events by up to +/- 40 years without erroring. This
    # repository is RADC-only, so the range is passed in explicitly or the augmentation is off.
    #
    # WHAT IT DOES. A trait recorded at the baseline visit (e.g. "smoked, ever") is not a thing
    # that HAPPENED at that age -- it was already true. Leaving it pinned to the baseline age
    # teaches the model that seeing the trait implies the subject was alive and enrolled at that
    # exact age, which is the immortality bias. Scattering the age breaks that association while
    # keeping the trait itself visible.
    #
    # `augment_tokens` is DISK-space (pre-+1), matching `data[:, 2]`: pass either an
    # (lo, hi) inclusive tuple or an explicit iterable of ids.
    if augment_tokens is not None:
        if isinstance(augment_tokens, tuple) and len(augment_tokens) == 2:
            lo, hi = augment_tokens
            aug_idx = (tokens >= lo) * (tokens <= hi)
        else:
            aug_idx = torch.isin(tokens, torch.as_tensor(list(augment_tokens), dtype=tokens.dtype))
        n_aug = int(aug_idx.sum())
        if n_aug:
            lo_d, hi_d = augment_age_range_days
            ages[aug_idx] += torch.randint(lo_d, hi_d, (n_aug,), generator=gen).float()

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)
    if vals is not None:
        vals = vals.masked_fill(~mask, float('nan'))

    # ---- synthetic "no event" markers, CONFINED TO THE OBSERVATION WINDOW -----------------
    #
    # These teach the model that time can pass without anything being recorded. One marker per
    # `no_event_token_rate` years of observation, on average.
    #
    # THE BUG THIS REPLACES. Upstream drew the marker ages from the absolute interval
    # [0, 36525] days -- birth to 100 years -- and then masked only from ABOVE, at the
    # subject's last real token. Everything below survived. So a subject first observed at 78
    # was handed a run of markers at ages 5, 30, 50 ... asserting seventy-eight event-free
    # years that were never observed: fabricated left-truncated history, which depresses the
    # learned baseline hazard everywhere. The `regular` variant was worse still, placing them
    # at exact multiples of the rate, which is memorisable.
    #
    # THE FIX. Draw inside [first observed age, last observed age] and scale the COUNT to the
    # width of that window, so the marker density is one per rate-years of ACTUAL follow-up and
    # no marker ever sits outside it. Rows are padded to a fixed K_MAX so the tensor shape does
    # not depend on the data; the surplus is masked out, sorts to the left, and is removed by
    # the right-anchored block cut below, so no real token is ever displaced.
    K_MAX = 32
    if (padding is None or
            str(padding).lower() == 'none' or
            not no_event_token_rate):
        pad = torch.zeros(len(ix), 0)
        pad_tokens = torch.zeros(len(ix), 0, dtype=torch.int)
    else:
        # window bounds over REAL tokens only (masked positions already sit at mask_time)
        hi = ages.masked_fill(~mask, -float('inf')).max(1, keepdim=True).values
        lo = ages.masked_fill(~mask, float('inf')).min(1, keepdim=True).values
        span = torch.clamp(hi - lo, min=0.0)
        if padding == 'random':
            u = torch.rand(len(ix), K_MAX, generator=gen)
        elif padding == 'regular':
            # deterministic, evenly spaced inside the window -- used for evaluation so the
            # reported loss does not move with the marker draw
            u = (torch.arange(K_MAX, dtype=torch.float32)[None, :] + 0.5) / K_MAX
            u = u.expand(len(ix), K_MAX).contiguous()
        else:
            raise NotImplementedError(f"padding={padding!r}")
        pad = lo + u * span
        n_keep = torch.ceil(span / (365.25 * no_event_token_rate))
        keep = torch.arange(K_MAX)[None, :] < n_keep
        pad = torch.where(keep, pad, torch.full_like(pad, mask_time))
        pad_tokens = torch.where(keep, torch.zeros(len(ix), K_MAX, dtype=torch.int),
                                 torch.full((len(ix), K_MAX), -1, dtype=torch.int))

    m = ages.max(1, keepdim=True).values

    # stack "no event" markers alongside the real tokens
    tokens = torch.hstack([tokens, pad_tokens])
    ages = torch.hstack([ages, pad])
    if vals is not None:                                  # markers carry no reading
        vals = torch.hstack([vals, torch.full_like(pad, float('nan'))])

    # belt and braces: nothing may sit past the last real token
    tokens = tokens.masked_fill(ages > m, -1)
    ages = ages.masked_fill(ages > m, mask_time)

    # sort everything so that things are correctly ordered about stacking
    s = torch.argsort(ages, 1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)
    if vals is not None:
        vals = torch.gather(vals, 1, s)

    # a technical detail: the token 0 is reserved for padding, so we shift all tokens by one
    tokens = tokens + 1

    # cut the padded tokens if possible
    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]
        if vals is not None:
            vals = vals[:, cut_margin:]

    # cut to maintain the block size
    #TODO it would be better to use the strategy defined by the "select" parameter
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]
        if vals is not None:
            vals = vals[:, cut_margin:]

    # shift by one to generate targets
    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]
    vx = vals[:, :-1] if vals is not None else None
    vy = vals[:, 1:] if vals is not None else None

    # if the first token is a "no event" token, mask it and the corresponding target
    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
        if vx is not None:
            vx, vy = vx.pin_memory().to(device, non_blocking=True), vy.pin_memory().to(device, non_blocking=True)
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
        if vx is not None:
            vx, vy = vx.to(device), vy.to(device)

    if vals is None:
        return x, a, y, b
    if binner is None:
        raise ValueError("get_batch got `values` but no `binner`")
    # y carries -1 at masked positions; clamp for the lookup, the weight is discarded anyway
    return x, a, y, b, binner(x.clamp(min=0), vx), binner(y.clamp(min=0), vy)

# ---------------------------------------------------------------- training cohort
# MOVED HERE FROM train.py, and the move is the point. train.py is a script -- importing it
# runs a training run -- so anything else that needed this rule had to replicate it. The NACC
# arm did exactly that in its figure code, with a comment asking the reader to keep the two in
# sync, and they had already drifted by one token id. There is now one copy.

def filter_cohort(data, p2i, min_visits, short_min_visits, stage_disk, ignored_disk=()):
    """Keep subjects with >= min_visits distinct PREDICTED-event ages, OR >= short_min_visits
    of them AND >= 2 tokens from the staging scale -- which, after keep-transitions dedup,
    means the staged state changed at least once.

    "PREDICTED-event" excludes the static block, and that exclusion is the whole point of the
    `ignored_disk` argument. The statics sit one day before the baseline visit, so counting
    every distinct age gives every subject in this cohort at least two by construction and the
    filter silently becomes a no-op -- measured: 3,101 of 3,101 training subjects "pass" at
    min_visits = 2. Counting only ages that carry a token the model is actually asked to
    predict gives 2,801, which is the intended population.

    NOTE what "visits" still means after that fix. Keep-transitions collapses a visit at which
    nothing changed, so this counts distinct ages that CARRY an event, not clinic attendances:
    2,801 here against 2,822 subjects with >= 2 recorded visits. The 21 in the gap attended
    twice and produced a token only once, so they genuinely have nothing to predict, and
    dropping them is the behaviour we want rather than a discrepancy to paper over.
    """
    if not min_visits or min_visits <= 1:
        return p2i
    ages = np.asarray(data[:, 1])
    toks = np.asarray(data[:, 2])
    stage = np.asarray(stage_disk, dtype=np.int64)
    ignored = np.asarray(list(ignored_disk), dtype=np.int64)
    keep = np.zeros(len(p2i), dtype=bool)
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        a, t = ages[s:s + n], toks[s:s + n]
        m = (a > 0)
        if ignored.size:
            m &= ~np.isin(t, ignored)
        nv = np.unique(a[m]).size
        if nv >= min_visits:
            keep[k] = True
        elif nv >= short_min_visits and stage.size:
            keep[k] = int(np.isin(t, stage).sum()) >= 2
    return p2i[keep]
