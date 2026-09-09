# Delphi-2M with NO cohort filtering at all -- the true baseline.
#
# Every other model in this project throws patients away before training:
#   general (train_delphi2m_mask_dedup.py)  filter_cohort keeps 16,262 of 38,687
#   A / B / C / D (train_fu*.py)            1,483 - 5,930 after a strict AND rule
#
# This one trains on ALL 38,687 patients in the split, including the ~22,000 who have
# two or three visits and never change cognitive state. Setting cohort_min_visits = 1
# makes train.py's filter_cohort return the patient index untouched.
#
# THE QUESTION: does the general model's filter help or hurt? It discards 58% of the
# training data on the grounds that those patients have nothing to predict -- but they
# are also the population the model will mostly meet, and they still carry demographics,
# comorbidities and timing signal. This is the only run that answers it.
#
# Architecture and schedule are IDENTICAL to config/train_delphi2m_mask_dedup.py,
# including eval_interval, so the only variable is the training population.
#
# Run: python training/train.py config/train_nofilter.py --device=cuda
dataset = "nacc-dedup-s42"
out_dir = "out-nofilter-s42"

vocab_size = 228  # NACC vocab: 0=Padding, 1=No event, 2..227 content.
                  # 111 -> 140 (CDRSUM split into 6 CDR boxes) -> 228 (FAQ total split
                  # into 9 domains, NPI-Q total into 12 symptoms, GDS added).
                  # See data_prep/tokenize_nacc_ad.py.
block_size = 256            # 96 -> 256: at 55.4 tokens/patient (p95 100, max 250) a 96-window
                            # truncated 6.0% of patients, and select='left' drops their LATEST
                            # events -- exactly the conversions. 256 truncates nobody.

t_min = 365.25 / 12

ignore_tokens = list(range(22))

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

eval_interval = 250          # matched to train_delphi2m_mask_dedup.py for comparability
eval_iters = 200
seed = 42

# THE one change: 1 disables filter_cohort entirely (see train.py's filter_cohort docstring)
cohort_min_visits = 1

wandb_project = "ad-projection"
wandb_run_name = "delphi2m-nofilter-s42"
