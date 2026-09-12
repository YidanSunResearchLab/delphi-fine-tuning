"""
train.py -- train a Delphi model on the RADC/ROSMAP event streams.

  python train.py configs/radc_base.py --device=cuda
  python train.py configs/radc_base.py --device=cuda --max_iters 4000
  python train.py configs/radc_base.py --wandb online

Reads data/<dataset>/{train,val}.bin -- uint32 (pid, age_days, disk_token) triples, patients
contiguous and age-sorted, as written by `python -m radc_delphi.build_dataset`. Writes the
BEST-VALIDATION checkpoint to <out_dir>/ckpt.pt.

This is a rewrite of the training loop the NACC arm used, keeping the parts that were paid for
in failed runs and fixing the parts that were NACC-shaped:

  KEPT -- the four NaN guards. An early 20k-iteration run went NaN somewhere between iteration
  10,000 and 20,000 and quietly overwrote its own periodic checkpoint with garbage. Guard 1
  aborts before backward() can write NaN into the weights, Guard 2 refuses to checkpoint a
  non-finite eval, Guard 3 lives in model.py (it bounds the log-intensity before exp()), and
  Guard 4 logs grad-norm and max|logit|, which spike well before the loss does.

  KEPT -- best-val checkpointing. An over-long run then costs wall-clock, not model quality.

  FIXED -- the age augmentation. The old loop passed `lifestyle_augmentations=True`, which
  jittered the ages of DISK tokens 3..11 by up to +/- 40 years. That id range was a UK Biobank
  constant. Here the token set comes from the config (`augment_tokens`), so it names RADC's own
  baseline-trait tokens or nothing at all.

  FIXED -- vocabulary/checkpoint pairing. The checkpoint now carries the vocabulary fingerprint
  (a hash of labels.csv) and the dataset fingerprint. Loading a checkpoint against a vocabulary
  it was not trained on used to succeed silently and mis-index every downstream probability.

  NEW -- early stopping. The RADC cohort is 6.8x smaller than the NACC one and the small-cohort
  runs in that arm overfit hard (val loss bottoming around step 1400 of 5000 and rising after).
  `patience` stops the run once validation has not improved for that many evaluations, so the
  schedule does not have to be guessed exactly right.
"""
import os
import sys
import time
import math
import json
import hashlib
from contextlib import nullcontext

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)
from radc_delphi.model import Delphi, DelphiConfig          # noqa: E402
from radc_delphi.batching import get_p2i, get_batch, filter_cohort   # noqa: E402
from radc_delphi.softlabel import SoftBinner                # noqa: E402

# -----------------------------------------------------------------------------
# defaults -- every one of these is overridable from a config file or the CLI
out_dir = 'out-radc'
eval_interval = 50
log_interval = 10
eval_iters = 100
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'          # 'scratch' | 'resume'
seed = 42

# wandb
wandb_log = False
wandb_project = 'ad-projection-radc'
wandb_run_name = 'radc'

# data
dataset = 'radc-s42'
gradient_accumulation_steps = 1
batch_size = 128
block_size = 96

# model
n_layer = 6
n_head = 4
n_embd = 64
dropout = 0.1
bias = False
vocab_size = 0                 # MUST be set by the config; 0 trips the assert below

# optimizer
learning_rate = 2e-3
max_iters = 2000
weight_decay = 0.2
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

# lr schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 2000
min_lr = 2e-4

# system
device = 'cuda'
dtype = 'float32'
compile = False

# delphi-specific
token_dropout = 0.0
# Time-to-event loss floor, in days. MUST be > 0: it caps the predicted event intensity at
# lambda <= 1/t_min so loss_dt (= lambda*dt - log lambda) cannot blow up. t_min = 0 removes the
# cap, logsumexp(logits) overflows in fp32, and loss_dt goes NaN. ~1 month.
t_min = 365.25 / 12.
mask_ties = True
time_head = False
dt_target = 'gather'           # 'gather' | 'next_visit' | 'next_event'
ignore_tokens = [0]
# WHICH tokens the timing target skips. MUST include No-event even though ignore_tokens must
# not -- see DelphiConfig.dt_ignore_tokens. None reproduces the delivered (buggy) behaviour.
dt_ignore_tokens = None
# Soft labels: read the raw scale reading from <split>_values.bin and represent it as the
# posterior over bins rather than a hard one-hot, on both the input embedding and the
# cross-entropy target. Falls back silently to hard labels if the values file is absent, and
# is bit-for-bit identical to the hard path when the weights happen to be one-hot.
soft_labels = False
data_fraction = 1.0

# cohort filter -- keep a subject with >= cohort_min_visits distinct event ages, OR
# >= cohort_short_min_visits ages AND >= 2 tokens from the staging scale (i.e. the state
# changed at least once). Set cohort_min_visits <= 1 to train on everyone.
cohort_min_visits = 3
cohort_short_min_visits = 2
stage_tokens_disk = ()         # DISK-space ids of the ordinal staging scale; set by the config
no_event_token_rate = 5

# Blank the AD column out of the cross-entropy for subjects whose AD label is MISSING rather
# than negative -- RADC records no age_first_ad_dx for anyone already demented at the baseline
# cycle, so they carry no AD token and are indistinguishable from genuine non-converters.
# Set False for the delivered behaviour; the two arms are one argument apart on purpose.
# `ad_dx_token` is the MODEL-space id and must come from the config, never be hard-coded here.
mask_missing_ad_label = True
ad_dx_token = 0

# DISK-space token ids whose ages get jittered to break immortality bias (see batching.py).
# None disables it. A config that emits baseline traits should name them here.
augment_tokens = None

# Free-running rollout probe -- radc_delphi/rollout_probe.py, run at every evaluation beside
# the two losses. OFF by default so the delivered loop is byte-identical without it.
#
# WHY IT IS NOT OPTIONAL INFORMATION. loss_ce and loss_dt are both teacher-forced: every
# prediction is scored one step ahead of a real prefix. Nothing in either term is sensitive to
# whether a FREE-RUNNING rollout terminates. Measured on two checkpoints from this repository,
# out-radc-final beats out-radc-testckpt on both components (2.140/6.536 against 2.552/6.602)
# while 98.8% of its age-80 draws never emit Death, against 1.2% for the one it beats. No
# setting of `select_on` can separate them, because the separating quantity is not in either
# loss. Turn this on for anything whose output is a simulated trajectory.
rollout_probe = False
rollout_probe_subjects = 24
rollout_probe_n_mc = 32
rollout_probe_age = 80.0
rollout_probe_max_new = 128
rollout_probe_until_age = 110.0
rollout_probe_seed = 0

# stop after this many consecutive evaluations without a new best val loss. 0 disables.
patience = 0

# Deterministic full-pass validation: score EVERY validation subject exactly once per eval,
# instead of drawing `eval_iters` random batches. Costs nothing at this scale (the val split is
# ~7k events) and removes the eval-to-eval sampling noise, which matters because that noise is
# comparable to the differences a capacity sweep is trying to resolve.
eval_full = True

# K-fold subject-level cross-validation over the POOLED train+val subjects. cv_folds = 0 keeps
# the delivered single split. With cv_folds = 5, fold `cv_fold` (0-indexed) becomes validation
# and the other four train.
#
# WHY. The delivered validation split is ~400 subjects. At roughly 45,000 scored positions per
# pass, a 400-subject split carries enough sampling noise to swamp the gap between adjacent
# cells of the capacity ladder, so a sweep scored on it selects mostly on noise. Folds are cut
# from the `stratum` column in subjects.csv, so each fold keeps the study x follow-up x outcome
# balance the split builder established. The TEST split is never touched.
cv_folds = 0
cv_fold = 0

# WHICH validation quantity selects the checkpoint: 'total' (loss_ce + loss_dt), 'ce' or 'dt'.
#
# 'total' is the default and it is the proper joint likelihood, but the alternative is live and
# the argument is worth recording, because on THIS cohort it is not obvious.
#
# The case for 'ce': RADC is an annual-protocol cohort, and 93.9% of nominal inter-visit
# intervals are exactly one year. loss_dt is the negative log-likelihood of an exponential
# waiting time, so most of what it measures is the study calendar rather than the subject, and
# it comes out nearly the same for every architecture. The NACC arm hit exactly this -- loss_dt
# at 73% of validation loss, inert across a 6.9x parameter range, and its ranking of
# architectures disagreeing outright with the downstream one.
#
# Why 'total' still wins here: that failure mode depended on the two heads SHARING parameters.
# With time_head = True (the delivered config) the intensity comes from its own scalar
# projection instead of logsumexp of the token logits, so a degenerate timing target no longer
# pulls on the token logits, and loss_dt stops being a constant that dilutes the ranking. It
# also is not entirely calendar: the two off-grid tokens -- the AD diagnosis at its own recorded
# age, and death -- carry real hazard information.
#
# Both are recorded at every eval and written to history.json, so switching the criterion after
# the fact costs a re-read, not a re-run. If you set time_head = False, set this to 'ce'.
select_on = 'total'

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items()
               if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
with open(os.path.join(_ROOT, 'configurator.py')) as f:
    exec(f.read())             # config file + CLI overrides
config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

assert vocab_size > 0, "vocab_size must be set by the config (it is a property of the tokenizer)"

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
np.random.seed(seed)
torch.set_float32_matmul_precision('high')

device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'float64': torch.float64,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# Parameters stay fp32 (master weights) even under fp16/bf16; autocast + GradScaler handle the
# low-precision compute. Defaulting the dtype to fp16 here would leave AMP with no fp32 master
# copy and produce NaNs / skipped steps.
torch.set_default_dtype(ptdtype if ptdtype in (torch.float32, torch.float64) else torch.float32)

# ------------------------------------------------------------------ data
data_dir = os.path.join(_ROOT, 'data', dataset)
if not os.path.exists(os.path.join(data_dir, 'train.bin')):
    sys.exit(f"[train] no train.bin under {data_dir}\n"
             f"  build it first:  python -m radc_delphi.build_dataset --seed {seed}")

train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)

def _load_values(split):
    p = os.path.join(data_dir, f'{split}_values.bin')
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None

train_values, val_values = _load_values('train'), _load_values('val')
if soft_labels and train_values is None:
    sys.exit(f"[train] soft_labels=True but {data_dir}/train_values.bin is missing. "
             f"Rebuild with a tokenizer that emits raw scale readings.")
binner = SoftBinner(device='cpu') if soft_labels else None
if soft_labels:
    print(f"[train] soft labels ON: raw readings from *_values.bin, "
          f"{len(train_values):,} train values")

if cv_folds and cv_folds > 1:
    # Pool train+val, then re-cut by subject into `cv_folds` stratified folds.
    import pandas as pd
    subj_csv = os.path.join(data_dir, 'subjects.csv')
    if not os.path.exists(subj_csv):
        sys.exit(f"[train] cv_folds={cv_folds} needs {subj_csv} (written by build_dataset.py)")
    sub = pd.read_csv(subj_csv)
    pooled = np.concatenate([np.asarray(train_data), np.asarray(val_data)])
    pooled_v = (np.concatenate([train_values, val_values])
                if train_values is not None else None)
    _ord = np.lexsort((pooled[:, 2], pooled[:, 1], pooled[:, 0]))
    pooled = pooled[_ord]
    if pooled_v is not None:
        pooled_v = pooled_v[_ord]
    in_pool = set(np.unique(pooled[:, 0]).tolist())
    sub = sub[sub['projid'].isin(in_pool)].sort_values('projid').reset_index(drop=True)
    # deal subjects round-robin within each stratum, after a seeded shuffle: every fold gets
    # the same stratum mix, and the assignment does not depend on how many strata there are
    rng_cv = np.random.default_rng(1000 + seed)
    fold_of = {}
    for st, grp in sub.groupby('stratum'):
        ids = grp['projid'].to_numpy()
        ids = ids[rng_cv.permutation(len(ids))]
        for i, p in enumerate(ids):
            fold_of[p] = i % cv_folds
    assign = np.array([fold_of[p] for p in pooled[:, 0]])
    val_data = pooled[assign == cv_fold]
    train_data = pooled[assign != cv_fold]
    if pooled_v is not None:
        val_values = pooled_v[assign == cv_fold]
        train_values = pooled_v[assign != cv_fold]
    print(f"[train] CV fold {cv_fold + 1}/{cv_folds}: pooled {len(in_pool):,} subjects -> "
          f"train {len(np.unique(train_data[:, 0])):,} / "
          f"val {len(np.unique(val_data[:, 0])):,}")

train_p2i = get_p2i(train_data)
val_p2i = get_p2i(val_data)


def _fingerprint(path):
    """Short content hash, so a checkpoint can name the exact vocabulary/data it was trained on."""
    if not os.path.exists(path):
        return None
    h = hashlib.md5()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:12]


labels_path = os.path.join(data_dir, 'labels.csv')
vocab_sig = _fingerprint(labels_path)
data_sig = _fingerprint(os.path.join(data_dir, 'train.bin'))
if vocab_sig is None:
    print(f"[train] WARNING: no labels.csv in {data_dir}; the checkpoint will carry no vocabulary "
          f"fingerprint and nothing downstream can verify it was scored against the right one.")
else:
    n_labels = sum(1 for _ in open(labels_path)) - 1           # minus the header
    if n_labels != vocab_size:
        sys.exit(f"[train] labels.csv has {n_labels} rows but the config says vocab_size="
                 f"{vocab_size}. One of them is stale -- refusing to train a model whose token "
                 f"ids will not line up with its own label table.")

print(f"[train] dataset {dataset}: {len(train_data):,} train / {len(val_data):,} val events")
print(f"[train] vocab_size {vocab_size} (labels {vocab_sig}) | data {data_sig}")




if cohort_min_visits and cohort_min_visits > 1:
    n_tr0, n_va0 = len(train_p2i), len(val_p2i)
    # DISK space, and it must exclude the statics or the filter is a no-op -- see filter_cohort
    _ignored_disk = tuple(t - 1 for t in ignore_tokens if t > 0)
    train_p2i = filter_cohort(train_data, train_p2i, cohort_min_visits,
                              cohort_short_min_visits, stage_tokens_disk, _ignored_disk)
    val_p2i = filter_cohort(val_data, val_p2i, cohort_min_visits,
                            cohort_short_min_visits, stage_tokens_disk, _ignored_disk)
    print(f"[train] cohort filter (>= {cohort_min_visits} predicted-event ages, OR >= "
          f"{cohort_short_min_visits} + a staging transition on disk tokens "
          f"{tuple(stage_tokens_disk)}): train {n_tr0} -> {len(train_p2i)} | "
          f"val {n_va0} -> {len(val_p2i)} subjects")
    if len(train_p2i) == n_tr0:
        print("[train] NOTE: the cohort filter removed nobody. That is the signature of it "
              "counting ages that carry only ignored tokens -- check `ignore_tokens`.")
    if len(train_p2i) == 0 or len(val_p2i) == 0:
        sys.exit("[train] the cohort filter emptied a split. Loosen it or check stage_tokens_disk.")

if data_fraction < 1.0:
    train_p2i = train_p2i[:int(data_fraction * len(train_p2i))]
    print(f"[train] data_fraction {data_fraction}: {len(train_p2i)} training subjects")

iter_num = 0
best_val_loss = 1e9
best_iter = 0

# ------------------------------------------------------------------ model
model_args = dict(ad_dx_token=(ad_dx_token if mask_missing_ad_label else 0),
                  n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout,
                  token_dropout=token_dropout, t_min=t_min, mask_ties=mask_ties,
                  ignore_tokens=list(ignore_tokens), time_head=time_head, dt_target=dt_target)

if init_from == 'scratch':
    print("[train] initializing a new model from scratch")
    model = Delphi(DelphiConfig(**model_args))
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"[train] resuming from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    ck = checkpoint['model_args']
    # architectural -- a mismatch would fail in load_state_dict anyway, so take the checkpoint's
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = ck[k]
    model_args['time_head'] = ck.get('time_head', False)
    # not architectural, but it changes the objective: a resume must not silently switch it
    model_args['dt_target'] = ck.get('dt_target', 'gather')
    # same reasoning for the missing-AD-label mask: 0 (off) for a checkpoint predating it
    model_args['ad_dx_token'] = ck.get('ad_dx_token', 0)
    if bool(model_args['ad_dx_token']) != bool(mask_missing_ad_label and ad_dx_token):
        print(f"[train] NOTE: resume keeps the checkpoint's missing-AD-label setting "
              f"(ad_dx_token={model_args['ad_dx_token']}), overriding this config.")
    if checkpoint.get('vocab_sig') and vocab_sig and checkpoint['vocab_sig'] != vocab_sig:
        sys.exit(f"[train] refusing to resume: checkpoint was trained against vocabulary "
                 f"{checkpoint['vocab_sig']}, this dataset's labels.csv is {vocab_sig}.")
    model = Delphi(DelphiConfig(**model_args))
    state_dict = checkpoint['model']
    for k in list(state_dict.keys()):                       # strip the torch.compile prefix
        if k.startswith('_orig_mod.'):
            state_dict[k[len('_orig_mod.'):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
else:
    sys.exit(f"[train] unknown init_from={init_from!r}")

model.to(device)
print(f"[train] {model.get_num_params():,} parameters "
      f"({n_layer}L / {n_head}H / {n_embd}d, head_dim {n_embd // n_head}, block {block_size})")

try:
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
except (AttributeError, TypeError):                          # torch < 2.4
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

if compile:
    print("[train] compiling the model ...")
    model = torch.compile(model)


# ------------------------------------------------------------------ the rollout probe
# Built once: it selects its prefixes and computes the observed death CIF on those same
# subjects, so the ratio it reports later needs no external reference. It saves and restores
# the RNG state of every device it touches, so a probe-on run follows exactly the trajectory
# a probe-off run would -- asserted by tests/test_pipeline.py:test_rollout_probe.
probe = None
if rollout_probe:
    from radc_delphi.rollout_probe import RolloutProbe        # noqa: E402
    probe = RolloutProbe(val_data, data_dir, model_args['ignore_tokens'],
                         landmark_age=rollout_probe_age,
                         n_subjects=rollout_probe_subjects,
                         n_mc=rollout_probe_n_mc,
                         max_new_tokens=rollout_probe_max_new,
                         until_age=rollout_probe_until_age,
                         seed=rollout_probe_seed, device=device)
    print(probe.describe(), flush=True)
    if not probe.prefixes:
        sys.exit("[train] rollout_probe is on but no prefix survived selection; "
                 f"check rollout_probe_age={rollout_probe_age} against this split.")


# ------------------------------------------------------------------ the missing-AD-label mask
# RADC leaves age_first_ad_dx empty for anyone already demented at the baseline cycle, so those
# subjects carry no AD token and are indistinguishable, in the event stream, from people who
# genuinely never converted. Trained as negatives they teach the model that MMSE 15-23 is
# compatible with being AD-free. The repair is at the LABEL, not at the features: model.forward
# blanks the AD column out of THEIR cross-entropy, so every other token they carry -- cognition,
# stroke, medication, death -- goes on training the model exactly as before and nobody is
# dropped from the cohort. A static "impaired at baseline" token cannot do this job: its MMSE
# half is already the baseline MMSE bin and its ad_rx half leaks the outcome (see the note in
# tokenizer.py where the flag is computed).
#
# Read AFTER the model is built, from model_args, because a resume takes the checkpoint's
# setting rather than this config's.
_ad_dx_token = model_args['ad_dx_token']
train_ad_missing = val_ad_missing = None
if _ad_dx_token:
    _subj_csv = os.path.join(data_dir, 'subjects.csv')
    if not os.path.exists(_subj_csv):
        sys.exit(f"[train] mask_missing_ad_label=True needs {_subj_csv} (written by "
                 f"build_dataset.py). Rebuild the dataset, or set mask_missing_ad_label=False.")
    import pandas as _pd
    _sub = _pd.read_csv(_subj_csv)
    if 'ad_label_missing' not in _sub.columns:
        sys.exit(f"[train] {_subj_csv} predates the ad_label_missing column. Rebuild with "
                 f"`python -m radc_delphi.build_dataset --seed {seed}`, or set "
                 f"mask_missing_ad_label=False.")
    _missing_pids = _sub.loc[_sub['ad_label_missing'].astype(bool), 'projid'].to_numpy()

    def _row_flags(data, p2i):
        """(len(p2i),) bool -- does the subject occupying each p2i row carry no AD label?"""
        return torch.from_numpy(np.isin(np.asarray(data[p2i[:, 0], 0]), _missing_pids))

    train_ad_missing = _row_flags(train_data, train_p2i)
    val_ad_missing = _row_flags(val_data, val_p2i)

    # The mask writes -inf into the AD column, so a flagged subject who DID carry an AD token
    # would be scored -log(0) = inf. By construction none can -- ad_label_missing requires an
    # empty age_first_ad_dx -- but subjects.csv and the .bin are written by different code
    # paths and can be mixed by hand, so this is checked once at startup instead of per batch.
    for _nm, _d, _p, _m in (('train', train_data, train_p2i, train_ad_missing),
                            ('val', val_data, val_p2i, val_ad_missing)):
        for _k in np.flatnonzero(_m.numpy()):
            _s, _n = int(_p[_k, 0]), int(_p[_k, 1])
            if (np.asarray(_d[_s:_s + _n, 2]) == _ad_dx_token - 1).any():   # DISK space
                sys.exit(f"[train] {_nm}: subject {int(_d[_s, 0])} is flagged ad_label_missing "
                         f"but carries the AD token. subjects.csv and the .bin disagree -- "
                         f"rebuild the dataset.")
    print(f"[train] missing-AD-label mask: {int(train_ad_missing.sum())} train / "
          f"{int(val_ad_missing.sum())} val subjects have the AD column (model id "
          f"{_ad_dx_token}) blanked from their cross-entropy")
else:
    print("[train] missing-AD-label mask OFF: prevalent-dementia subjects train as AD negatives")


def _batch(data, p2i, ix, train=True, ad_missing=None, values=None):
    """(x, a, y, b, m, soft_in, soft_target).

    `m` is the (batch,) bool row mask for model.forward, or None.
    `soft_in` / `soft_target` are (weights, ids) pairs or None when soft_labels is off.
    """
    out = get_batch(ix, data, p2i, block_size=block_size, device=device, select='left',
                    padding='random' if train else 'regular',
                    augment_tokens=augment_tokens if train else None,
                    no_event_token_rate=no_event_token_rate, cut_batch=True,
                    values=values if soft_labels else None,
                    binner=binner if soft_labels else None)
    m = None if ad_missing is None else ad_missing[ix].to(device)
    if soft_labels:
        x, a, y, b, sin, stg = out
        return x, a, y, b, m, sin, stg
    x, a, y, b = out
    return x, a, y, b, m, None, None


def _rand_ix(p2i):
    return torch.randint(len(p2i), (batch_size,))


@torch.no_grad()
def estimate_loss():
    """Mean (loss_ce, loss_dt) on train and val.

    With eval_full the validation pass walks every subject exactly once, in index order, so
    the number is deterministic given the weights. The train figure stays a random sample --
    it is only there to show the generalisation gap, and a full pass over it would cost far
    more than it is worth.
    """
    out = {}
    model.eval()
    for split, data, p2i, am, vals in (
            ('train', train_data, train_p2i, train_ad_missing, train_values),
            ('val', val_data, val_p2i, val_ad_missing, val_values)):
        if eval_full and split == 'val':
            chunks = [torch.arange(i, min(i + batch_size, len(p2i)))
                      for i in range(0, len(p2i), batch_size)]
        else:
            chunks = [_rand_ix(p2i) for _ in range(eval_iters)]
        losses = torch.zeros(len(chunks), 2)
        weights = torch.zeros(len(chunks))
        for k, ix in enumerate(chunks):
            X, A, Y, B, M, SI, ST = _batch(data, p2i, ix, train=False, ad_missing=am,
                                           values=vals)
            with ctx:
                _, loss, _ = model(X, A, Y, B, validation_loss_mode=True, ad_label_missing=M,
                                   soft_in=SI, soft_target=ST)
            losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])
            weights[k] = len(ix)          # the last full-pass chunk is short
        w = (weights / weights.sum()).unsqueeze(1)
        out[split] = (losses * w).sum(0)
    model.train()
    return out


def get_lr(it):
    if it < warmup_iters:
        return learning_rate * it / max(1, warmup_iters)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)
    return min_lr + 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) * (learning_rate - min_lr)


def save_ckpt(name):
    torch.save({
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'best_iter': best_iter,
        'config': config,
        'vocab_sig': vocab_sig,      # so nothing downstream can score this against another vocab
        'data_sig': data_sig,
        'dataset': dataset,
    }, os.path.join(out_dir, name))


if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# Re-seed by iter_num so a resumed run does not replay the batch stream a fresh run saw from 0
# (the model init above already consumed the base seed).
torch.manual_seed(seed + iter_num)
X, A, Y, B, M, SI, ST = _batch(train_data, train_p2i, _rand_ix(train_p2i),
                               ad_missing=train_ad_missing, values=train_values)
t0 = time.time()
stale_evals = 0
history = []

while True:
    metrics = {'iter': iter_num}
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for g in optimizer.param_groups:
        g['lr'] = lr

    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()
        val_total = losses['val'].sum().item()
        val_ce, val_dt = losses['val'][0].item(), losses['val'][1].item()
        tr_loss = losses['train'].sum().item()
        # the quantity that selects the checkpoint (see `select_on` above)
        val_loss = {'total': val_total, 'ce': val_ce, 'dt': val_dt}[select_on]
        print(f"step {iter_num}: train {tr_loss:.4f}  val {val_total:.4f} "
              f"(ce {val_ce:.4f} dt {val_dt:.4f})  [selecting on {select_on}={val_loss:.4f}]",
              flush=True)
        metrics.update({'train/agg_loss': tr_loss, 'val/loss': val_total,
                        'val/loss_ce': val_ce, 'val/loss_dt': val_dt,
                        'val/selected': val_loss})
        history.append({'iter': iter_num, 'train': tr_loss, 'val': val_total,
                        'val_ce': val_ce, 'val_dt': val_dt, 'selected': val_loss})

        if probe is not None:
            pm = probe.run(model)
            metrics.update(pm)
            history[-1].update(pm)
            r85 = pm.get('probe/death_cif_ratio_85', float('nan'))
            print(f"  probe: p_death_next {pm['probe/p_death_next']:.4f}  "
                  f"never-dies {100 * pm['probe/frac_no_death']:.1f}%  "
                  f"end-age {pm['probe/median_end_age']:.1f}  "
                  f"deathCIF85 {pm.get('probe/death_cif_85', float('nan')):.3f} "
                  f"({r85:.2f}x observed)", flush=True)

        # GUARD 2: never let a non-finite eval touch a checkpoint. The best-val save is already
        # NaN-safe (nan < best is False), but an unconditional periodic save is not -- that is
        # how a NaN eval once overwrote a good checkpoint with garbage.
        finite = math.isfinite(val_loss)
        if not finite:
            print(f"[nan-guard] val_loss is {val_loss} at iter {iter_num}; "
                  f"skipping ALL checkpoint writes this eval.", flush=True)

        improved = finite and (val_loss < best_val_loss)
        if improved:
            best_val_loss, best_iter, stale_evals = val_loss, iter_num, 0
        elif finite:
            stale_evals += 1
        if (always_save_checkpoint or improved) and finite:
            save_ckpt('ckpt.pt')
            print(f"  -> saved {out_dir}/ckpt.pt (best val {best_val_loss:.4f} @ {best_iter})")

        if patience and stale_evals >= patience:
            print(f"[train] early stop: {stale_evals} evaluations without improvement "
                  f"(best {best_val_loss:.4f} @ iter {best_iter})")
            break

    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss_terms, _ = model(X, A, Y, B, ad_label_missing=M,
                                          soft_in=SI, soft_target=ST)
        X, A, Y, B, M, SI, ST = _batch(train_data, train_p2i, _rand_ix(train_p2i),  # prefetch
                                       ad_missing=train_ad_missing, values=train_values)
        loss = (loss_terms['loss_ce'] + loss_terms['loss_dt']) / gradient_accumulation_steps
        # GUARD 1: abort the instant the loss is non-finite, BEFORE backward() can write NaN
        # into the weights. Naming the term and the max logit makes the cause obvious.
        if not torch.isfinite(loss):
            print(f"[nan-guard] non-finite loss at iter {iter_num} (micro {micro_step}): "
                  f"loss_ce={loss_terms['loss_ce'].item():.4f} "
                  f"loss_dt={loss_terms['loss_dt'].item():.4f} "
                  f"max|logit|={logits.detach().abs().max().item():.1f}. Aborting.", flush=True)
            raise SystemExit(1)
        scaler.scale(loss).backward()

    grad_norm = None
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    t1 = time.time()
    dt_ms = (t1 - t0) * 1000
    t0 = t1
    if iter_num % log_interval == 0:
        # GUARD 4: grad-norm and max|logit| spike well before the loss goes NaN, so logging them
        # turns a blow-up into something visible in advance rather than 10k iters too late.
        gn = float(grad_norm) if grad_norm is not None else float('nan')
        ml = logits.detach().abs().max().item()
        print(f"iter {iter_num}: loss {loss.item():.4f}, gradnorm {gn:.2f}, "
              f"max|logit| {ml:.1f}, {dt_ms:.0f}ms", flush=True)
        metrics.update({'train/loss': loss.item(), 'train/grad_norm': gn,
                        'train/max_logit': ml, 'lr': lr})

    if wandb_log and (iter_num % log_interval == 0 or 'val/loss' in metrics):
        wandb.log(metrics)

    iter_num += 1
    if iter_num > max_iters:
        break

with open(os.path.join(out_dir, 'history.json'), 'w') as fh:
    json.dump({'history': history, 'best_val_loss': best_val_loss, 'best_iter': best_iter,
               'config': config, 'vocab_sig': vocab_sig, 'data_sig': data_sig,
               'cv_folds': cv_folds, 'cv_fold': cv_fold}, fh, indent=2)

print(f"\n[train] DONE. best val {best_val_loss:.4f} @ iter {best_iter} "
      f"of {iter_num} -> {out_dir}/ckpt.pt")
if best_iter and best_iter <= 0.25 * max(iter_num, 1):
    print(f"[train] NOTE: the best checkpoint arrived in the first quarter of the run. That is "
          f"the overfitting signature the small-cohort runs showed; consider a smaller model, "
          f"more regularisation, or a shorter schedule.")
