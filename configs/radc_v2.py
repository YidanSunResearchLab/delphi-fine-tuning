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
#   4,428 subjects, 77,847 events, 17.6 per subject (min 7, median 17, p95 28, max 43)
#   1,164 AD converters (26.3%), 2,745 deaths (62.0%), vocabulary 53 tokens
#   train 3,101 subjects / 54,585 events | val 443 / 7,781 | test 884 / 15,481
#
# MMSE CARRIES SEVEN LEVELS, NOT FOUR, and the three extra cuts are all above 26. 73.6% of
# visits sat in the old 27-30 bin, so the model was blind to changes inside three quarters of
# the data -- measured: zero emitted tokens for any move within it, against 3,993 once split.
# The bottom of the scale is deliberately NOT split: below 18 the measured noise is 2.4 points
# against 0.6 at the ceiling, and the training set holds ~840 readings there, so finer bins
# would buy neither resolution nor learnability. Edges sit at HALF-integers above 26 because
# 98.4% of readings are integers and an edge on a mode starves the level beside it.
# train.py puts the checkout root on sys.path before exec'ing this file, so the package is
# importable here. Note there is no __file__ inside an exec'd config -- do not reach for it.
from radc_delphi import vocab as _V

dataset = "radc-v2-s42"
out_dir = "out-radc-v2-s42"

# ---------------------------------------------------------------- vocabulary
# Taken from the vocabulary module rather than retyped. train.py additionally cross-checks
# vocab_size against the row count of the split's own labels.csv and refuses to start on a
# mismatch, because a stale value here mis-indexes every logit and still trains happily.
vocab_size = _V.VOCAB_SIZE                      # 53

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
stage_tokens_disk = _V.STAGE_TOKENS_DISK

# ---------------------------------------------------------------- v2: the timing target
# WHICH tokens the waiting-time objective skips. Not the same set as ignore_tokens, and in the
# delivered build the two were conflated: 41.3% of dt targets pointed at a synthetic No-event
# marker, which compressed the target mean from 1.65 y to 0.89 y and turned 8.5% of censored
# positions into fabricated short answers. generate() masks the marker and can never emit one,
# so the intensity must be trained on the gap it will actually be asked to reproduce.
dt_ignore_tokens = _V.DT_IGNORE_TOKENS

# ---------------------------------------------------------------- v2: soft labels
# A reading is represented as the POSTERIOR over bins given the measurement noise, not as a
# hard one-hot -- on the input embedding (a weighted mixture of the level embeddings, same
# position, same vector) and on the cross-entropy target (-sum_k w_k log p_k).
#
# Two defects it removes at once. (a) The edge discontinuity: MMSE 17.9 and 18.1 differ by a
# tenth of a point, far below the 2.4-point measurement SD there, yet hard binning put them in
# different tokens 1.14 apart in embedding space; soft weights put them 0.06 apart. (b) The
# within-bin blindness: 18.1 and 23.5 were the SAME token, distance exactly 0.
#
# It also replaces the hysteresis filter outright. Both existed to stop measurement noise from
# producing spurious transitions, but hysteresis was stateful and biased -- it disagreed with
# the raw bin on 6.22% of MMSE visits and silently deleted real recoveries -- while the
# stateless TV gate in the tokenizer emits 0.0% sub-sigma transitions against 11.8% before.
#
# Verified degenerate: forcing the weights to one-hot reproduces the hard-label loss to 0.0e+00.
soft_labels = True

# ---------------------------------------------------------------- the missing AD label
# 287 subjects (6.5%) were already demented at the baseline cycle, and the codebook is explicit
# that age_first_ad_dx "is not available" for them. They therefore carry NO AD token and, in
# the event stream, are indistinguishable from people who genuinely never converted -- 225 of
# them survive the cohort filter and contribute 2,941 training tokens (5.6% of train.bin) that
# assert, position by position, that severe impairment is compatible with being AD-free.
#
# With the mask on, model.forward blanks the AD column out of the cross-entropy for those
# subjects only: "AD is unknown here" rather than "AD did not happen here". Everything else
# they carry still trains the model, so no subject and no token is lost. Their 0 AD events
# were never available to supervise anything, and the 1,164 real diagnoses are untouched.
#
# WHY NOT A TOKEN. A static "impaired at baseline" marker is redundant or leaky with nothing
# in between: the MMSE < 24 half IS the baseline MMSE bin the model already reads (the first
# hysteresis state is the plain bin, and the edges are [18, 24, 27]), while the ad_rx half
# reaches a recorded diagnosis 48.2% of the time against a 26.1% base rate among subjects with
# baseline MMSE >= 24 -- a 1.84x anticipation of the outcome, which is what ad_rx is banned
# from the vocabulary for. See tokenizer.baseline_impairment.
#
# Set mask_missing_ad_label = False for the delivered behaviour; the arms are one flag apart.
mask_missing_ad_label = True
ad_dx_token = _V.AD_DX                          # MODEL id 48

# ---------------------------------------------------------------- context
# Real tokens per subject top out at 42. The window-capped no-event markers add a mean of 4.4
# and at most 16, so the longest possible sequence is 54 and p99 is 43. 64 carries 65 positions
# and truncates nobody, with headroom. This matters more than it looks: training uses
# select='left', so a too-small window drops each subject's LATEST events -- precisely the
# diagnoses and the deaths.
block_size = 64

# ---------------------------------------------------------------- architecture
# 4L / 4H / 64d = 207,168 parameters, head_dim 16.
#
# THIS IS THE SWEPT RESULT, not a guess. sweep.py ran the ladder
# {(2,32), (2,64), (3,48), (4,64), (6,96)} x dropout {0.1, 0.2, 0.3} x 5-fold subject-level CV
# = 75 runs, ~30 min on three H100s. Mean CV validation loss, best five of fifteen cells:
#
#     L4/E64 d0.1   8.6205 +/- 0.0212      <- this
#     L6/E96 d0.2   8.6241 +/- 0.0167
#     L6/E96 d0.3   8.6315 +/- 0.0246
#     L6/E96 d0.1   8.6370 +/- 0.0317
#     L2/E64 d0.1   8.6624 +/- 0.0232
#     ...
#     L3/E48 d0.2   8.7759 +/- 0.0335      <- what this file guessed before the sweep
#     L2/E32 d0.3   8.9848 +/- 0.0150      (worst)
#
# Four cells sit within one standard deviation of the best, and three of them are the 682k
# L6/E96 shape. The tie is broken by size: at ~45,000 scored positions per pass the burden of
# proof is on the larger model, and 207k already sits at ~0.2 scored tokens per parameter
# against Delphi-2M's own ~2. Going to 682k buys nothing measurable and costs 3.3x.
#
# THE OTHER THING THE SWEEP SHOWED, which matters more than the winner. mean loss_dt is
# 6.484-6.566 across ALL FIFTEEN CELLS -- a 1.3% spread over a 23x parameter range -- while
# loss_ce moves 2.124 to 2.420. The timing head is fitting the annual visit calendar and
# nothing else, exactly as the 93.9%-of-intervals-are-one-year figure predicts. Every bit of
# discrimination between architectures here is in loss_ce.
#
# ONE CAVEAT ON PROVENANCE, recorded rather than buried. The sweep ran before the cohort
# filter was fixed, so all 75 runs saw the pooled 3,544 subjects rather than the 3,199 the
# fixed filter keeps -- the filter was counting the static block's own age and so passed
# everybody. Every cell saw the SAME population, so the ranking is unaffected; the absolute
# losses would shift slightly. Re-sweeping for a 10% data difference is not worth the compute,
# and the direction is conservative anyway: more data favours the larger model, and the larger
# models did not win.
n_layer = 4
n_head = 4
n_embd = 64
bias = False
dropout = 0.1
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
wandb_run_name = "radc-v2-softbins-s42"
