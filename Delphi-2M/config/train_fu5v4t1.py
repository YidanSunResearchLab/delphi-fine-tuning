# Delphi-2M on cohort C: follow-up > 5 y, >= 4 visits, >= 1 transition
#
# One point on a data-volume sweep. The strict cohort (fu>8, >=6 visits, >=2 transitions,
# 1,483 training patients) overfit hard -- val loss bottomed at step 1400 and rose for the
# remaining 3,600. These variants relax the rule to find how much training data this
# architecture actually needs before it stops losing to a carry-forward baseline.
#
#   cohort C: ~4,450 train / ~1,336 test patients
#   (for reference: the general model trains on 16,262)
#
# Built by:
#   python data_prep/make_cohort_strict.py --csv <raw> \
#       --tag nacc-fu5v4t1-s42 --fu-years 5 --min-visits 4 --min-trans 1
#
# Architecture and schedule are IDENTICAL to config/train_delphi2m_mask_dedup.py, so the only
# variable across the sweep is the training population.
#
# Run: python training/train.py config/train_fu5v4t1.py --device=cuda
dataset = "nacc-fu5v4t1-s42"
out_dir = "out-fu5v4t1-s42"

vocab_size = 228  # NACC vocab: 0=Padding, 1=No event, 2..227 content.
                  # 111 -> 140 (CDRSUM split into 6 CDR boxes) -> 228 (FAQ total split
                  # into 9 domains, NPI-Q total into 12 symptoms, GDS added).
                  # See data_prep/tokenize_nacc_ad.py.
block_size = 256            # 96 -> 256: at 55.4 tokens/patient (p95 100, max 250) a 96-window
                            # truncated 6.0% of patients, and select='left' drops their LATEST
                            # events -- exactly the conversions. 256 truncates nobody.

t_min = 365.25 / 12

# same loss mask as the delivered model -- keeps the two directly comparable
ignore_tokens = list(range(22))

# --- architecture: unchanged from config/train_delphi2m_mask_dedup.py, deliberately ---
# Shrinking it would confound "trained on fewer patients" with "smaller model". Same shape,
# same schedule, one variable changed: the training population.
n_layer = 12
n_head = 12
n_embd = 120

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

eval_interval = 100          # tightened from 250: this cohort overfits fast, catch the best ckpt
eval_iters = 200
seed = 42

# the cohort IS the filter -- do not let train.py filter it a second time
cohort_min_visits = 1

wandb_project = "ad-projection"
wandb_run_name = "delphi2m-fu5v4t1-s42"
