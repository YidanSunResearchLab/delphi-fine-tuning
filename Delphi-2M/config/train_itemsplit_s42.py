# train_itemsplit_s42.py -- the first model trained on the ITEM-SPLIT tokenization.
#
# What is new relative to the delivered out-delphi2m-dedup-mask-s42:
#
#   1. DATA. Every sum-score is now its items (data_prep/tokenize_nacc_ad.py):
#      CDRSUM -> 6 CDR box domains (111-139), the FAQ total -> 9 domains (140-175),
#      the NPI-Q total -> 12 symptoms (176-223), and GDS is new (224-227). MoCA stays a
#      total (UDS-v3-only, 36% visit coverage). 30 ordinal scales, all prediction targets.
#      vocab 111 -> 228; 1.14M -> 3.06M events; the cohort-filtered training set goes
#      16,262 patients / 410,635 events -> 16,997 / 1,193,169.
#
#   2. CONTEXT. block_size 96 -> 256. At 55.4 tokens/patient (p95 100, max 250) a 96-window
#      truncated 6.02% of patients, and training runs select='left', so what it dropped was
#      their LATEST events -- the conversions. Those 3,328 are not a random 6%: 86% of them
#      have a dementia record against 41% overall. 256 truncates nobody (longest is 250).
#
#   3. SHAPE. 8L/8H/192d = 3,622,848 params, head_dim 24.
#   4. TIMING. time_head = True.
#
# Both of 3 and 4 are argued below. Run:
#   sbatch slurm/train_delphi2m_mask_dedup.sbatch config/train_itemsplit_s42.py

dataset = "nacc-dedup-s42"               # what data_prep/make_dataset.py --seed 42 builds
out_dir = "out-itemsplit-mask-s42"

vocab_size = 228            # 0=Padding, 1=No event, 2..227 content. See tokenize_nacc_ad.py.
block_size = 256            # see note 2 above

t_min = 365.25 / 12

# --- ids 0..21: padding/no-event + all static & background tokens (incl. APOE 16-21) ---
# UNCHANGED at 22 even though the vocabulary nearly doubled. Every one of the 117 new ids is
# an ordinal scale level, and per the decision on record all 30 scales are prediction targets
# -- so they all belong IN the loss. This is deliberate, not an oversight: the alternative
# (masking the FAQ/NPI item tokens out of the loss but leaving them in the input stream) would
# make them context-only, which is not what they are here.
ignore_tokens = list(range(22))

# --- architecture: 8L / 8H / 192d = 3,622,848 params, head_dim 24 ---
#
# WAS 8/6/120 = 1,412,160 at vocab 111. Depth is held at 8 and width is what moves, because
# that is what experiments/capacity actually measured (6 shapes x 3 seeds, 48 runs).
# experiments/ was removed from this branch; it is still there at commit dba88f8.
#
#   * n_layer 8 was the BEST downstream arm of the six at n_embd 120 -- median AUC 0.6942
#     +/- 0.0098 against 12 layers' 0.6532 +/- 0.0117. Going deeper made it worse. So the
#     2.9x more training data is absorbed by widening, not by stacking.
#   * head_dim is the one geometry knob the sweep isolated: 10 -> 20 improved 5/5 downstream
#     metrics at IDENTICAL parameter count and speed. 192/8 = 24 keeps it above 20.
#
# Do NOT rank a change to this block on validation loss. 73% of the reported val loss is
# loss_dt, which was inert across the sweep's whole 6.9x capacity range; the val-loss ranking
# and the downstream ranking disagreed outright (RESULTS.md finding 2). Score on Figure-2
# metrics with >=3 seeds.
n_layer = 8
n_head = 8
n_embd = 192

# --- timing objective gets its own parameters ---
# time_head=True adds nn.Linear(n_embd, 1) and reads the log-intensity off THAT instead of
# off logsumexp(logits). 193 parameters, 0.005% of the model. experiments/time_head ran it
# paired against this exact control shape, 3 seeds, bit-identical trunk init per pair:
#   transition-time MAE   2.962 vs 3.367 y   -0.405   3/3 seeds better
#   transition-time bias  1.821 vs 2.570 y   -0.749   3/3 seeds better  (control is +2.6 y late)
# It converges LATER, though -- best_iter 3833 vs 2533 -- which is priced into max_iters below.
time_head = True

# --- optimization ---
# max_iters 5000 -> 8000. Two independent reasons to expect a later bottom: time_head moved
# best_iter 2533 -> 3833 (x1.5) on its own, and the training set is 2.9x larger. 8000 is x1.6.
# It is NOT larger than that on purpose: experiments/capacity re-ran four shapes at 15000 with
# lr_decay_iters matched, and EVERY shape got worse (finding 4) -- stretching the cosine holds
# the LR high for longer, so schedule length is a real hyperparameter, not just a budget.
max_iters = 8000
lr_decay_iters = 8000
learning_rate = 2e-3
min_lr = 2e-4
warmup_iters = 500
beta2 = 0.99
dropout = 0.0
token_dropout = 0.0
weight_decay = 0.2
grad_clip = 1.0

# --- batching: effective batch is still 128 ---
# delphi/model.py sets self.flash = False, so attention is MATERIALISED: (B, nh, T, T) per
# layer. Going 96 -> 256 multiplies that by 7.1x -- 2.1 GB of attention maps per forward at
# batch 128, before autograd saves its copies. Flash is not an escape: its branch calls
# scaled_dot_product_attention(attn_mask=None, is_causal=True), which would silently discard
# mask_ties, the within-visit leakage guard. So split the batch instead: 32 x 4 accumulation
# is the same 128 gradients with ~0.5 GB of attention maps.
batch_size = 32
gradient_accumulation_steps = 4

eval_interval = 250
eval_iters = 200

# Single seed, by decision -- so read this run against the delivered model as a DIRECTION,
# not a measurement. experiments/capacity put the single-seed bootstrap CI on MCI->Dementia
# at +/-0.03, wide enough that any two shapes overlap; a one-seed gap smaller than that is
# not evidence. Set explicitly rather than inherited from train.py's default so out_dir's
# "s42" is self-consistent (train.py also re-seeds the batch RNG as seed + iter_num).
seed = 42

# cohort filter left at train.py's defaults (4 visits, or 2 + a staging transition).
# stage_tokens_disk stays NACCUDSD 105..108 -- the CDR split did not touch NACCUDSD.

wandb_project = "ad-projection"
