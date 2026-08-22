# Delphi-2M OFFICIAL-aligned config for NACC.
# Mirrors gerstung-lab/Delphi config/train_delphi_demo.py (n_embd=120/12L/12H = ~2.1M params,
# 5000 iters, lr 2e-3, weight_decay 0.2, warmup 500) instead of the oversized 10.8M / 20k-iter
# train_nacc.py that overfit (val loss bottomed at ~step 5000 then rose).
# Run: python training/train.py config/train_delphi2m.py --device=cuda    (ablation baseline)
dataset = "nacc-dedup-s42"
out_dir = "out-delphi2m-dedup-nomask-s42"

vocab_size = 111            # our NACC vocab (official uses 1270; that's data-dependent, keep ours)
block_size = 96             # official Delphi block size (was 80)

# time-to-event loss floor. KEEP the safe NACC value (365.25/12 ~= 1 month), NOT the official
# 0.1 -> t_min too small reintroduces the NaN loss_dt this branch fixed. See train_nacc.py notes.
t_min = 365.25 / 12

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
wandb_run_name = "delphi2m-official-s42"
