# train_cohort_sweep.py -- the data-volume sweep, four cohorts in one config.
#
# WHY ONE FILE. This replaces config/train_delphi2m_strict.py, train_fu5v5t2.py,
# train_fu5v4t1.py and train_fu3v4t1.py, which were 62-line files differing in exactly three
# assignments (dataset, out_dir, wandb_run_name) plus a cohort-size comment. Keeping them
# apart meant every schedule or vocabulary change had to be applied four times by hand -- and
# the three names had to be edited in sync, with nothing checking that they were. Here the
# cohort is picked ONCE and all three are derived from the same table row, so they cannot
# disagree.
#
# THE QUESTION THE SWEEP ASKS. The strict cohort (A) overfit hard: val loss bottomed at step
# 1400 and rose for the remaining 3,600. Relaxing the rule adds patients but dilutes them with
# people who have little to predict. Where is the floor -- how much training data does this
# architecture need before it stops losing to a carry-forward baseline?
#
# RUN (from Delphi-2M/), one cohort per invocation:
#   COHORT=strict  python training/train.py config/train_cohort_sweep.py --device=cuda
#   COHORT=fu5v5t2 python training/train.py config/train_cohort_sweep.py --device=cuda
#   COHORT=fu5v4t1 python training/train.py config/train_cohort_sweep.py --device=cuda
#   COHORT=fu3v4t1 python training/train.py config/train_cohort_sweep.py --device=cuda
#   COHORT=fu5v4t1 sbatch slurm/train_delphi2m_mask_dedup.sbatch config/train_cohort_sweep.py
# COHORT is required and unset is a hard error -- a silent default would let you run cohort A
# while believing you ran D, which is the one mistake this file must not permit.
#
# BUILD THE DATASETS FIRST (data_prep/make_cohort_strict.py), each filtered OUT OF the
# nacc-dedup-s42 split, so every cohort's test set is a subset of the original one and all
# arms are scored on the very same patients.
import os as _os

# key -> (dataset, out_dir, rule, approx train / test patients as built)
# out_dir strings are verbatim from the four configs this replaces, so checkpoints and results
# written by the old files still resolve.
_COHORTS = {
    "strict":  ("nacc-strict-s42",  "out-delphi2m-strict-s42",
                "fu > 8 y, >= 6 visits, >= 2 NACCUDSD transitions", "1,483 / 437"),
    "fu5v5t2": ("nacc-fu5v5t2-s42", "out-fu5v5t2-s42",
                "fu > 5 y, >= 5 visits, >= 2 transitions",          "2,296 / 694"),
    "fu5v4t1": ("nacc-fu5v4t1-s42", "out-fu5v4t1-s42",
                "fu > 5 y, >= 4 visits, >= 1 transition",           "4,450 / 1,336"),
    "fu3v4t1": ("nacc-fu3v4t1-s42", "out-fu3v4t1-s42",
                "fu > 3 y, >= 4 visits, >= 1 transition",           "5,930 / 1,747"),
}
# for reference, the general model trains on 16,262 patients

_key = _os.environ.get("COHORT")
if _key not in _COHORTS:
    _lines = ["  COHORT=%-8s %-50s %s" % (k, v[2], v[3]) for k, v in _COHORTS.items()]
    raise SystemExit(
        "config/train_cohort_sweep.py: set COHORT to one of the four cohorts.\n"
        + ("got COHORT=%r\n" % _key if _key else "COHORT is unset\n")
        + "\n".join(_lines)
        + "\n\ne.g.  COHORT=fu5v4t1 python training/train.py config/train_cohort_sweep.py --device=cuda"
    )

dataset, out_dir, _rule, _n = _COHORTS[_key]
wandb_run_name = "delphi2m-%s-s42" % ("strict" if _key == "strict" else _key)

vocab_size = 228            # 0=Padding, 1=No event, 2..227 content. See tokenize_nacc_ad.py.
block_size = 256            # see train_itemsplit_s42.py note 2 for why 96 was not enough

# same loss mask as the delivered model -- keeps the arms directly comparable
ignore_tokens = list(range(22))

# --- architecture: 12L / 12H / 120d = 2,104,320 params ---
#
# ⚠️ THIS IS *NOT* THE DELIVERED SHAPE. The delivered model
# (config/train_delphi2m_mask_dedup.py) moved to 8L/6H/120d in commit 23c02e6, on the evidence
# of the capacity sweep; that commit did not touch the four cohort configs, so every cohort run
# that exists on disk was trained at 12/12/120. The value is kept here to MATCH THOSE
# CHECKPOINTS rather than to describe current practice.
#
# What this costs. The four configs used to claim "same shape, same schedule, one variable
# changed: the training population". That is false as long as this block reads 12/12, because
# the delivered model it is being compared against is 8/6: a cohort-vs-delivered gap mixes
# "trained on fewer patients" with "different architecture", which is exactly the confound the
# original comment warned about. Cohort-vs-cohort comparisons WITHIN the sweep are unaffected
# -- all four share this block.
#
# To re-run the sweep like-for-like against the delivered model, override on the CLI (no edit
# needed, and the old checkpoints keep their matching config):
#   COHORT=fu5v4t1 python training/train.py config/train_cohort_sweep.py \
#       --n_layer 8 --n_head 6 --out_dir out-fu5v4t1-L8H6-s42 --device=cuda
n_layer = 12
n_head = 12
n_embd = 120

t_min = 365.25 / 12

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

eval_interval = 100          # tightened from the delivered 250: these cohorts overfit fast,
eval_iters = 200             # and train.py only writes ckpt.pt when val loss improves
seed = 42

# the cohort IS the filter -- do not let train.py filter it a second time
cohort_min_visits = 1

wandb_project = "ad-projection"
