# radc_base.py -- the RADC/ROSMAP Delphi training configuration.
#
#   python -m radc_delphi.build_dataset --seed 42
#   python train.py configs/radc_base.py --device=cuda
#
# Every value here is set against a measured property of the built dataset, printed by
# build_dataset.py into data/radc-s42/report.json. Re-run the builder and re-read those numbers
# before changing any of them.
#
# THE DATASET AS BUILT (seed 42):
#   4,428 subjects, 74,900 events, 16.9 per subject (min 7, median 16, p95 27, max 42)
#   1,164 AD converters (26.3%), 2,745 deaths (62.0%), vocabulary 50 tokens
#   train 3,101 subjects / 52,576 events | val 443 / 7,470 | test 884 / 14,854
# train.py puts the checkout root on sys.path before exec'ing this file, so the package is
# importable here. Note there is no __file__ inside an exec'd config -- do not reach for it.
from radc_delphi import vocab as _V

dataset = "radc-s42"
out_dir = "out-radc-base-s42"

# ---------------------------------------------------------------- vocabulary
# Taken from the vocabulary module rather than retyped. train.py additionally cross-checks
# vocab_size against the row count of the split's own labels.csv and refuses to start on a
# mismatch, because a stale value here mis-indexes every logit and still trains happily.
vocab_size = _V.VOCAB_SIZE                      # 50

# Padding plus the 21 static ids (2-22: sex, APOE, education, study, smoking, alcohol).
#
# NOT a contiguous range, and the gap is the point: id 1 (No event) IS a training target. The
# synthetic no-event markers are the only negative evidence in the stream -- they are how the
# model learns that time can pass with nothing recorded -- so masking them out of the loss
# would leave it with positives only. model.py adds id 1 to the ignored set by itself under
# validation_loss_mode, so validation loss stays comparable across configurations.
#
# The statics are held out because predicting a subject's own sex or APOE token back from
# itself is free accuracy that would swamp the terms of interest. They remain in the input
# stream and are still attended to: a "do-not-predict" mask, NOT a feature ablation.
ignore_tokens = _V.IGNORE_TOKENS

# The cohort filter counts transitions on MMSE. RADC has no per-visit clinical diagnosis to use
# instead -- cogdx and dcfdx are last-visit only, and both leak the outcome (AUC 0.888, 0.838).
stage_tokens_disk = _V.STAGE_TOKENS_DISK        # DISK ids 22-25

# ---------------------------------------------------------------- context
# Real tokens per subject top out at 42. The window-capped no-event markers add a mean of 4.4
# and at most 16, so the longest possible sequence is 54 and p99 is 43. 64 carries 65 positions
# and truncates nobody, with headroom. This matters more than it looks: training uses
# select='left', so a too-small window drops each subject's LATEST events -- precisely the
# diagnoses and the deaths.
block_size = 64

# ---------------------------------------------------------------- architecture
# 3L / 4H / 48d ~= 90k parameters (12*L*D^2 blocks + D^2 age projection + 2*50*D embed/head).
#
# THE BUDGET THAT SETS THIS. One pass over the training split scores ~44,800 positions: 49,287
# train events minus 16,932 statics that are not targets, plus ~12,400 no-event markers. So 90k
# parameters is about 0.5 scored tokens per parameter. Delphi-2M itself trained at roughly 2
# tokens per parameter (2.2M parameters on ~4M UK Biobank tokens) and this corpus is ~90x
# smaller, so porting its 12L/12H/120d shape would land 50x into the over-parameterized regime.
# The offsetting fact is that this vocabulary is 50 tokens against Delphi-2M's ~1,300, so the
# per-step problem is far easier and some slack is affordable.
#
# THIS IS A STARTING POINT, NOT A RESULT. slurm/sweep_radc.sbatch runs the ladder
# {(2,32) 29k, (2,64) 108k, (3,48) 90k, (4,64) 207k, (6,96) 682k} x dropout {0.1, 0.2, 0.3}
# under 5-fold CV. If (2,32) wins, take it -- at this data scale that is an unembarrassing
# outcome, and the sweep exists because the event counts alone cannot settle it.
n_layer = 3
n_head = 4
n_embd = 48
bias = False
dropout = 0.2
token_dropout = 0.0

# ---------------------------------------------------------------- the two losses
# Time-to-event floor, in days. MUST be > 0: it caps the predicted intensity at lambda <=
# 1/t_min so loss_dt cannot blow up. Setting it to 0 produced a NaN loss_dt that killed a
# 20,000-iteration run in the previous arm. ~1 month.
t_min = 365.25 / 12

# "next_event" = the gap to the next REAL event, skipping the synthetic no-event markers.
#
# Not the inherited "gather" default, which replaces a position's dt with that of the last
# position its attention mask lets it see -- backward-looking for the non-final tokens of a
# multi-token visit, and measured at 73.6% of scored positions trained on someone else's dt,
# running 1.42x (median) longer than the true gaps. "gather" existed to keep old NACC
# checkpoints bit-reproducible; there are no old checkpoints here.
#
# And "next_event" rather than "next_visit" because generate() masks ignore_tokens and
# No-event and can therefore never emit one. The intensity has to be trained on the gap it will
# actually be asked to reproduce at sampling time.
dt_target = "next_event"

# Give the timing objective its own scalar projection instead of reading it off logsumexp of
# the token logits. With the shared head, the degenerate timing target -- 93.9% of intervals
# are exactly 365 days -- pulls directly on the token logits, which is the mechanism by which
# loss_dt swamped model selection in the previous arm. 49 extra parameters.
time_head = True

# Same-visit tokens do not attend to each other. This is why the statics sit one day BEFORE the
# baseline visit (otherwise they would be invisible to it), and it is also what makes visit-
# batched generation correct: tokens sharing an age are conditionally independent given the
# history, so drawing them from one forward pass is what the objective says to do.
mask_ties = True

# ---------------------------------------------------------------- batching
# The age-jitter augmentation stays OFF. It exists to break immortality bias -- a trait pinned
# to the age it was recorded implying survival to that age -- but there is no such bias to
# correct here: entry age is a conditioning fact for every subject, not an event, and the
# statics are emitted for all 4,428 at the same fixed offset. Its inherited (lo, hi) range is a
# UK Biobank disk-id range that means something else entirely in this vocabulary, and the
# +/- 20-40 year jitter would scatter static tokens past the subject's death.
augment_tokens = None

# One no-event marker per 2 years of OBSERVED follow-up, drawn inside the observation window
# (see the window fix in batching.py). Measured effect: mean 4.41 markers per subject, max 16.
no_event_token_rate = 2

# Keep subjects with >= 2 distinct event ages: 4,029 of 4,428 (91.0%) and 93.96% of events. The
# 399 single-visit subjects cannot support a next-event objective at all -- every transition
# regime is degenerate for them. They are NOT a random sample (275 of 399 are MAP, 187 died,
# mean age_bl 80.3), so they must not silently vanish from any reported denominator; they are
# in subjects.csv and belong in every cohort-description statistic.
#
# Do NOT raise this to 3. It costs a further 410 subjects who are disproportionately old and
# short-lived; keep >= 3 as a reported sensitivity arm (3,619 subjects) rather than the default.
cohort_min_visits = 2
cohort_short_min_visits = 2

# ---------------------------------------------------------------- schedule
# 2,822 training subjects at batch 128 is ~22 iterations per epoch, so 6,000 iterations is
# ~272 passes. Deliberately generous: best-validation checkpointing plus `patience` decide
# where the run stops, rather than the schedule length having to be guessed right.
max_iters = 6000
lr_decay_iters = 6000
warmup_iters = 200
batch_size = 128
learning_rate = 6e-4
min_lr = 6e-5
beta2 = 0.95
weight_decay = 0.1
grad_clip = 1.0

# Deterministic full-pass validation -- every validation subject scored exactly once. The split
# is ~7k events, so this costs nothing, and it removes eval-to-eval sampling noise that would
# otherwise be comparable to the gaps the capacity sweep is trying to resolve.
eval_full = True
eval_interval = 100
eval_iters = 40                 # only used for the train-side estimate
patience = 15

select_on = "total"

seed = 42
device = "cuda"
dtype = "float32"

wandb_project = "ad-projection-radc"
wandb_run_name = "radc-base-L3H4E48-s42"
