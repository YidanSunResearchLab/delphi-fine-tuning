# Delphi-AD training config (Python form -> NO pyyaml needed, zero installs on HPC).
# Run:  python training/train.py config/train_nacc.py --device=cuda    (SUPERSEDED -- see README)
dataset = "nacc-dedup-s42"        # reads data/nacc-dedup-s42/{train,val}.bin
out_dir = "out-nacc-s42"    # checkpoints -> out-nacc-s42/ckpt.pt

vocab_size = 228  # NACC vocab: 0=Padding, 1=No event, 2..227 content.
                  # 111 -> 140 (CDRSUM split into 6 CDR boxes) -> 228 (FAQ total split
                  # into 9 domains, NPI-Q total into 12 symptoms, GDS added).
                  # See data_prep/tokenize_nacc_ad.py.
block_size = 256            # 96 -> 256: at 55.4 tokens/patient (p95 100, max 250) a 96-window
                            # truncated 6.0% of patients, and select='left' drops their LATEST
                            # events -- exactly the conversions. 256 truncates nobody.

# time-to-event loss floor. MUST be > 0: it caps the predicted event-intensity at
# lambda <= 1/t_min so loss_dt (= lambda*dt - log lambda) can't blow up. t_min=0.0
# removes the cap -> logsumexp(logits) overflows in fp32 -> loss_dt = NaN (killed the
# 20k-iter run: weights went NaN between iter 10000-20000). 365.25/12 ~= 1 month, the
# original Delphi value.
t_min = 365.25 / 12

n_layer = 6
n_head = 6
n_embd = 384

max_iters = 20000
batch_size = 128
learning_rate = 0.0006
lr_decay_iters = 20000
min_lr = 0.00006
warmup_iters = 200
dropout = 0.1
weight_decay = 0.1
grad_clip = 1.0

eval_interval = 1000
eval_iters = 200
seed = 42

wandb_project = "ad-projection"
wandb_run_name = "nacc-delphi-109tok"
