# RADC/ROSMAP arm -- first training config for the dataset built by data_prep/tokenize_radc.py.
#
# Run (from Delphi-2M/):
#   python data_prep/make_dataset_radc.py
#   python training/train.py config/train_radc.py --device=cuda
#
# NOTHING in delphi/ or training/ is RADC-specific. Only three keys below carry the new
# vocabulary; the rest is scheduling.
#
# ------------------------------------------------------------------------------------------
# WHY THIS IS SMALLER THAN THE DELIVERED NACC MODEL
#
#   arm    train subjects (post cohort filter)   train events   params
#   NACC   16,262                                ~800k raw      1,412,160  (8L/6H/120d)
#   RADC    2,390                                  81,513         304,321  (6L/4H/64d)
#
# 6.8x fewer subjects. The capacity sweep (experiments/capacity @ dba88f8 -- the directory was
# removed from this branch -- 6 shapes x 3 seeds) picked
# 8L/6H/120d *for NACC*; that result does not transfer to a cohort this size. 6L/4H/64d is
# ~4.6x smaller, slightly more generous than naive linear-in-subjects scaling (which points at
# ~207k, i.e. 4L/4H/64d) because RADC trajectories are DENSER -- 30.3 events/subject vs NACC's
# ~20, since ten biomarker scales are run-length-encoded per subject rather than one.
#
# This shape is a defensible starting point, NOT a swept result. The honest next step is to
# point experiments/capacity/ at dataset="radc-dedup-s42" and re-run the ladder:
#     4L/4H/64d   205,761      6L/4H/64d   304,321   <-- here
#     6L/6H/96d   680,737      8L/6H/120d 1,407,241  (the NACC winner)
# ckpt.pt is the BEST-VAL checkpoint (train.py only writes it on improvement), so an
# over-long run costs wall-clock, not model quality.
# ------------------------------------------------------------------------------------------

dataset = "radc-dedup-s42"        # reads data/radc-dedup-s42/{train,val}.bin
out_dir = "out-radc-s42"          # best-val checkpoint -> out-radc-s42/ckpt.pt

# --- the RADC vocabulary (see the LABELS table in data_prep/tokenize_radc.py) -------------
vocab_size = 69                   # 0=Padding, 1=No event, 2..68 = 67 content tokens

# ids 0..21: padding/no-event + the 20 static tokens (2-3 sex, 4-9 APOE, 10-12 education,
# 13-15 study, 16-18 smoking, 19-21 alcohol). Dropped from the LOSS but still in the input
# stream and still attended to -- a "don't-predict" mask, NOT a feature ablation. Deliberately
# the SAME range(22) boundary as every delivered NACC config, so the two arms stay comparable.
ignore_tokens = list(range(22))

# DISK-space ids of the ordinal staging scale used by train.py's cohort filter. RADC has no
# per-visit clinical diagnosis (cogdx/dcfdx are last-visit only), so estimated MMSE is the
# NACCUDSD analogue: model ids 22-25 -> disk 21-24.
stage_tokens_disk = (21, 22, 23, 24)

# Longest subject in the train split is 89 events, so 96 never truncates.
block_size = 96

# time-to-event loss floor, MUST be > 0: caps predicted intensity at lambda <= 1/t_min so
# loss_dt (= lambda*dt - log lambda) cannot blow up to NaN. 365.25/12 ~= 1 month.
t_min = 365.25 / 12

# --- architecture -------------------------------------------------------------------------
n_layer = 6
n_head = 4
n_embd = 64        # head_dim 16

# time_head=False reproduces every DELIVERED run. The time_head arm
# (experiments/time_head/) was only just launched on NACC and has no verdict yet, so this
# stays on the delivered default; flip to True once that result lands.
time_head = False

# --- schedule -----------------------------------------------------------------------------
# 2,390 train subjects / batch 128 = ~18.7 iters per epoch. 2000 iters ~= 107 epochs, against
# the NACC arm's 39 (5000 iters / 127 iters-per-epoch). The extra headroom is deliberate:
# eval_interval=50 samples the val curve finely and best-val checkpointing keeps the peak.
max_iters = 2000
lr_decay_iters = 2000
warmup_iters = 200
batch_size = 128
learning_rate = 2e-3
min_lr = 2e-4
beta2 = 0.99

# Regularisation is raised vs the capacity sweep's dropout=0.0, which was tuned on a cohort
# 6.8x larger. This is the one place the RADC arm is NOT a like-for-like port.
dropout = 0.1
token_dropout = 0.0
weight_decay = 0.2
grad_clip = 1.0

# val split is only 442 subjects; 100 x 128 = 12,800 samples is ~29 passes over it, plenty.
eval_interval = 50
eval_iters = 100
seed = 42

wandb_project = "ad-projection"
wandb_run_name = "radc-rosmap-69tok-L6H4E64"
