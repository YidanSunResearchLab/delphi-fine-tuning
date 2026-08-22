# Delphi-2M trained ONLY on the strict long-trajectory cohort.
#
# Cohort: true follow-up > 8 y AND >= 6 distinct visits AND >= 2 NACCUDSD transitions.
# 2,146 patients (train 1,483 / val 226 / test 437) -- built by data_prep/make_cohort_strict.py,
# filtered OUT OF the nacc-dedup-s42 split so the strict test set is a subset of the original
# one and the two models can be scored on the very same patients.
#
# WHY THIS EXISTS: the general model is trained on 16,262 patients, many with only 2-4 visits
# and nothing to predict. This asks the opposite question -- given people we watched for a
# decade with repeated state changes, how good can the model get?
#
# ⚠️ OVERFITTING IS EXPECTED. 1,483 patients against 2.1M parameters is ~11x less data than
# the general model at identical capacity. The mitigation is early stopping: train.py only
# writes ckpt.pt when val loss improves, and eval_interval is tightened to 100 (from 250) so
# the best checkpoint is caught before it drifts. Read the val-loss curve in the job log
# before trusting anything -- if the best checkpoint lands in the first few hundred
# iterations, the model has memorised rather than learned.
#
# Run: python training/train.py config/train_delphi2m_strict.py --device=cuda
dataset = "nacc-strict-s42"
out_dir = "out-delphi2m-strict-s42"

vocab_size = 111
block_size = 96

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
wandb_run_name = "delphi2m-strict-s42"
