
import time

# ROSMAP adaptation of config/train_delphi.py (Delphi-2M, UKB).
# Everything not listed here is inherited verbatim from train.py's defaults.

out_dir = 'Delphi-ROSMAP'
eval_interval = 250   # keep frequent because we'll overfit
eval_iters = 50       # 25 in the UKB config; doubled because the val split is only 443 people
log_interval = 25     # don't print too too often
seed = 42

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False
wandb_project = 'delphi-rosmap'
wandb_run_name = 'rosmap' + str(time.time())

dataset = 'rosmap'
batch_size = 128
block_size = 96       # same as Delphi-2M; longest ROSMAP trajectory is 65 real tokens + <=20 no-event
data_fraction = 1.0

# 3,985 train participants / 142,528 records, vs ~400k participants in UKB.
# Halve depth/width relative to Delphi-2M (12/12/120) to keep the model in scale with the corpus.
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.1         # 0.0 in the UKB config; the ROSMAP corpus is ~3 orders of magnitude smaller
weight_decay = 2e-1
vocab_size = 129      # from data/rosmap/labels.csv

learning_rate = 6e-4
max_iters = 12000
lr_decay_iters = 12000  # make equal to max_iters usually
min_lr = 6e-5           # learning_rate / 10 usually
beta2 = 0.99            # make a bit bigger because number of tokens per iter is small

warmup_iters = 500
# padding, sex and the background/lifestyle block (post-shift ids, cf. data/rosmap/labels.csv)
ignore_tokens = [0] + list(range(2, 34))
t_min = 0.1
token_dropout = 0.0
no_event_token_rate = 5

# ROSMAP's background block spans pre-shift ids 3..32 (UKB default is 3..11)
lifestyle_token_min = 3
lifestyle_token_max = 32
