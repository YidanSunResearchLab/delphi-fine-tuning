# Delphi-2M, DEDUP (keep-transitions) + Delphi-faithful MASKING.
#
# Fills the empty cell of the 2x2 (dedup x mask):
#     out-delphi2m-dedup-nomask-s42          dedup(keep-trans), no mask   (baseline)
#     out-delphi2m-keepall-nomask-s42  keep-all,          no mask
#     out-delphi2m-keepall-mask-s42     keep-all,          mask 0-21
#     THIS ->                   dedup(keep-trans), mask 0-21
#
# Same data, schedule and seed as config/train_delphi2m.py (5000 iters, seed 42, SAME
# nacc-dedup-s42 dedup dataset). Differs in TWO places: the architecture (8L/6H/120d, not
# the upstream 12L/12H/120d -- see the block below) and ignore_tokens.
#
# ignore_tokens: the 22 static/background/placeholder
# tokens (0=pad, 1=no-event, 2-3=sex, 4-6=BMI, 7-9=smoking, 10-12=alcohol,
# 13-15=education, 16-21=APOE) are dropped from the LOSS (not predicted, not scored)
# but stay in the input stream and are still attended to. Delphi-faithful: stops the
# time-to-event loss being dominated by the no-event grid + immutable statics. APOE
# is still visible context -- this is a "don't-predict" mask, NOT a feature ablation.
# Run: python training/train.py config/train_delphi2m_mask_dedup.py --device=cuda
dataset = "nacc-dedup-s42"               # what data_prep/make_dataset.py --seed 42 builds
out_dir = "out-delphi2m-dedup-mask-s42"  # trains on the DEDUP build (1.14M events)

vocab_size = 228  # NACC vocab: 0=Padding, 1=No event, 2..227 content.
                  # 111 -> 140 (CDRSUM split into 6 CDR boxes) -> 228 (FAQ total split
                  # into 9 domains, NPI-Q total into 12 symptoms, GDS added).
                  # See data_prep/tokenize_nacc_ad.py.
block_size = 256            # 96 -> 256: at 55.4 tokens/patient (p95 100, max 250) a 96-window
                            # truncated 6.0% of patients, and select='left' drops their LATEST
                            # events -- exactly the conversions. 256 truncates nobody.

t_min = 365.25 / 12

# --- ids 0..21: padding/no-event + all static & background tokens (incl. APOE 16-21) ---
ignore_tokens = list(range(22))

# --- architecture: 8L / 6H / 120d = 1,412,160 params, head_dim 20 ---
#
# WAS 12/12/120 = 2,104,320 params, copied verbatim from the upstream Delphi-2M demo config
# (vocab 1270, cohort ~100x larger). It was never chosen for THIS dataset. Changed on the
# evidence of experiments/capacity -- 6 shapes x 3 seeds, 48 runs; see its RESULTS.md:
#
#   * n_head 12 -> 6 (head_dim 10 -> 20) is free: identical parameter count, identical
#     speed, and 5/5 downstream Figure-2 metrics improve by more than the seed sd.
#     Validation loss cannot see it at all (delta -0.0043 against a pooled sd of 0.0046).
#   * n_layer 12 -> 8 at n_embd 120 is the best downstream arm of the six: median AUC
#     0.6942 +/- 0.0098 vs the delivered 0.6532 +/- 0.0117, Dementia@10y 0.8677 +/- 0.0021
#     vs 0.8433 +/- 0.0059 -- at 1.49x fewer params and ~1.6x faster (135 vs 217 ms/iter
#     on the sweep's CPU nodes).
#
# Do NOT shrink further on the strength of validation loss. The 0.31M arm (6/4/64) has the
# LOWEST val loss of all six and does not carry that win downstream -- 73% of the reported
# val loss is loss_dt, which is inert across the whole 6.9x capacity range. Rank
# architectures on Figure-2 metrics with >=3 seeds, never on val loss (RESULTS.md finding 2).
n_layer = 8
n_head = 6
n_embd = 120

# --- optimization: official Delphi-2M demo schedule ---
max_iters = 5000
batch_size = 128
learning_rate = 2e-3
lr_decay_iters = 5000
min_lr = 2e-4
warmup_iters = 500
beta2 = 0.99
dropout = 0.0
token_dropout = 0.0
weight_decay = 0.2
grad_clip = 1.0

eval_interval = 250
eval_iters = 200
seed = 42

wandb_project = "ad-projection"
wandb_run_name = "delphi2m-dedup-mask-s42"
