# Delphi-2M, DEDUP (keep-transitions) + Delphi-faithful MASKING.
#
# Fills the empty cell of the 2x2 (dedup x mask):
#     out-delphi2m-dedup-nomask-s42          dedup(keep-trans), no mask   (baseline)
#     out-delphi2m-keepall-nomask-s42  keep-all,          no mask
#     out-delphi2m-keepall-mask-s42     keep-all,          mask 0-21
#     THIS ->                   dedup(keep-trans), mask 0-21
#
# Identical to config/train_delphi2m.py (2.1M params, 5000 iters, seed 42, SAME
# nacc-dedup-s42 dedup dataset) EXCEPT ignore_tokens: the 22 static/background/placeholder
# tokens (0=pad, 1=no-event, 2-3=sex, 4-6=BMI, 7-9=smoking, 10-12=alcohol,
# 13-15=education, 16-21=APOE) are dropped from the LOSS (not predicted, not scored)
# but stay in the input stream and are still attended to. Delphi-faithful: stops the
# time-to-event loss being dominated by the no-event grid + immutable statics. APOE
# is still visible context -- this is a "don't-predict" mask, NOT a feature ablation.
# Run: python training/train.py config/train_delphi2m_mask_dedup.py --device=cuda
dataset = "nacc-dedup-s42"               # what data_prep/make_dataset.py --seed 42 builds
out_dir = "out-delphi2m-dedup-mask-s42"  # trains on the DEDUP build (1.14M events)

vocab_size = 111
block_size = 96

t_min = 365.25 / 12

# --- ids 0..21: padding/no-event + all static & background tokens (incl. APOE 16-21) ---
ignore_tokens = list(range(22))

# --- architecture: official Delphi-2M (2.1M params on our 111-token vocab) ---
n_layer = 12
n_head = 12
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
