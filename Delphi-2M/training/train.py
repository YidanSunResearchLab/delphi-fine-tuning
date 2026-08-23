import os
import sys
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # Delphi-2M/
sys.path.insert(0, _ROOT)
from delphi.model import Delphi, DelphiConfig
from delphi.utils import get_p2i, get_batch


out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = False  # if True, always save a checkpoint after each eval
init_from = 'scratch'  # 'scratch' or 'resume' or 'gpt2*'
seed = 42

# wandb logging
wandb_log = False  # disabled by default
wandb_project = 'delphi'
wandb_run_name = 'run' + str(time.time())

# data
dataset = 'ukb_data'
gradient_accumulation_steps = 1  # used to simulate larger batch sizes
batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 24

# model
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.2  # for pretraining 0 is good, for finetuning try 0.1+
bias = False  # do we use bias inside LayerNorm and Linear layers?
vocab_size = 256

# adamw optimizer
learning_rate = 6e-4  # max learning rate
max_iters = 10000  # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0

# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 2000  # how many steps to warm up for
lr_decay_iters = 10000  # should be ~= max_iters per Chinchilla
min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# system
device = 'cuda'  # 'cuda', 'cuda:0', ... on the GPU cluster; 'cpu' for a local sanity check
dtype = 'float32'  # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster

# delphi training
token_dropout = 0.0
t_min = 365.25 / 12.  # ~1-month floor; MUST be > 0 (t_min=0 -> unbounded intensity -> NaN loss_dt)
mask_ties = True
# time_head=False reproduces every delivered run: one output projection, and the
# time-to-next-event intensity read off it as logsumexp(logits). True gives the timing
# objective its own scalar head (see delphi/model.py and experiments/time_head/).
time_head = False
ignore_tokens = [0]
data_fraction = 1.0
# training cohort filter: keep a subject if it has >= cohort_min_visits distinct visits,
# OR >= cohort_short_min_visits visits AND shows a NACCUDSD transition (>=2 NACCUDSD tokens
# after keep-transitions dedup). Set cohort_min_visits<=1 to disable (train on everyone).
cohort_min_visits = 4
cohort_short_min_visits = 2
no_event_token_rate = 5


# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configurator.py')) as f:
    exec(f.read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed)
torch.set_float32_matmul_precision('high')

device_type = 'cuda' if 'cuda' in device else 'cpu'  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'float64': torch.float64,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# Keep model parameters in fp32 (master weights) even for fp16/bf16 training: the low-precision
# compute is handled by autocast (ctx) + GradScaler. Setting the default to fp16/bf16 here would
# create params with no fp32 master copy and break AMP (NaNs / skipped steps).
torch.set_default_dtype(ptdtype if ptdtype in (torch.float32, torch.float64) else torch.float32)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)

train_p2i = get_p2i(train_data)
val_p2i = get_p2i(val_data)


def filter_cohort(data, p2i, min_visits, short_min_visits, udsd_disk=(105, 106, 107, 108)):
    """Keep subjects with >= min_visits distinct visits (age>0), OR >= short_min_visits
    visits that ALSO show a NACCUDSD transition (>=2 NACCUDSD tokens post keep-transitions,
    i.e. the state changed at least once). min_visits<=1 -> no filtering. Tokens here are
    disk-space (model_id - 1), so NACCUDSD Normal..Dementia = 105..108."""
    if min_visits is None or min_visits <= 1:
        return p2i
    ages = np.asarray(data[:, 1]); toks = np.asarray(data[:, 2])
    udsd = np.asarray(udsd_disk)
    keep = np.zeros(len(p2i), dtype=bool)
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        nv = np.unique(ages[s:s + n][ages[s:s + n] > 0]).size
        if nv >= min_visits:
            keep[k] = True
        elif nv >= short_min_visits:
            keep[k] = int(np.isin(toks[s:s + n], udsd).sum()) >= 2   # >=2 UDSD tokens == a transition
    return p2i[keep]


# training cohort filter (applied to train AND val so val loss tracks the same population)
if cohort_min_visits and cohort_min_visits > 1:
    n_tr0, n_va0 = len(train_p2i), len(val_p2i)
    train_p2i = filter_cohort(train_data, train_p2i, cohort_min_visits, cohort_short_min_visits)
    val_p2i = filter_cohort(val_data, val_p2i, cohort_min_visits, cohort_short_min_visits)
    print(f"cohort filter (>= {cohort_min_visits} visits OR >= {cohort_short_min_visits} + NACCUDSD transition): "
          f"train {n_tr0} -> {len(train_p2i)} | val {n_va0} -> {len(val_p2i)} patients")

# downsample the data to requested fraction
if data_fraction < 1.0:
    train_p2i = train_p2i[:int(data_fraction * len(train_p2i))]

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


print(f"found vocab_size = {vocab_size}")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout, token_dropout=token_dropout, t_min=t_min,
                  mask_ties=mask_ties, ignore_tokens=ignore_tokens,
                  time_head=time_head)  # start with model_args from command line

if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    gptconf = DelphiConfig(**model_args)
    model = Delphi(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # Also architectural: it adds a parameter tensor, so a resume that disagreed with the
    # checkpoint would fail in load_state_dict. Checkpoints written before the head existed
    # have no such key, and absent means the single-head model.
    model_args['time_head'] = checkpoint_model_args.get('time_head', False)
    # create the model
    gptconf = DelphiConfig(**model_args)
    model = Delphi(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op.
# torch>=2.4 exposes the unified torch.amp.GradScaler(device_type, ...); older
# torch (e.g. 2.2, the newest macOS-x86 wheel) only has torch.cuda.amp.GradScaler().
try:
    scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16'))
except (AttributeError, TypeError):
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# helps estimate an arbitrarily accurate loss over either split using many batches


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, 2)
        data = train_data if split == 'train' else val_data
        p2i = train_p2i if split == 'train' else val_p2i
        for k in range(eval_iters):
            ix = torch.randint(len(p2i), (batch_size,))
            X, A, Y, B = get_batch(ix, data, p2i, block_size=block_size,
                                   device=device, select='left',
                                   no_event_token_rate=no_event_token_rate, 
                                   cut_batch=True)
            with ctx:
                logits, loss, _ = model(X, A, Y, B, validation_loss_mode=True)
            losses[k] = torch.stack([loss['loss_ce'], loss['loss_dt']])
        out[split] = losses.mean(0)
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / max(1, lr_decay_iters - warmup_iters)  # guard lr_decay==warmup
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
# Re-seed the batch RNG by iter_num so a resumed run does NOT replay the exact batch
# stream a fresh run saw from iter 0 (model init above already used the base `seed`).
torch.manual_seed(seed + iter_num)
ix = torch.randint(len(train_p2i), (batch_size,))
X, A, Y, B = get_batch(ix, train_data, train_p2i, block_size=block_size, device=device,
                       padding='random', lifestyle_augmentations=True, select='left',
                       no_event_token_rate=no_event_token_rate)
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process

val_loss = None
while True:
    metrics = {"iter": iter_num}
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()
        val_loss = losses['val'].sum().item()
        print(f"step {iter_num}: train loss {losses['train'].sum().item():.4f}, val loss {val_loss:.4f}")

        metrics.update({
            "train/agg_loss": losses['train'].sum().item(),
            "val/loss":val_loss,
            "val/loss_ce": losses['val'][0].item(),
            "val/loss_dt": losses['val'][1].item()
        })

        # GUARD 2 (finite-checkpoint): never let a NaN/inf val_loss touch a checkpoint. The best-val
        # save is already NaN-safe (nan < best is always False), but the periodic ckpt_<N>.pt save
        # below is unconditional -- that is how a NaN eval overwrote ckpt_20000.pt with garbage.
        val_is_finite = math.isfinite(val_loss)
        if not val_is_finite:
            print(f"[nan-guard] val_loss is {val_loss} at iter {iter_num}; skipping ALL checkpoint writes this eval.", flush=True)

        improved = val_is_finite and (val_loss < best_val_loss)
        if improved:
            best_val_loss = val_loss
        if (always_save_checkpoint or improved) and val_is_finite:
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,   # true running best (not just this eval)
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if iter_num % 10_000 == 0 and val_is_finite:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{iter_num}.pt'))

    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss, att = model(X, A, Y, B)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        ix = torch.randint(len(train_p2i), (batch_size,))
        X, A, Y, B = get_batch(ix, train_data, train_p2i, block_size=block_size, device=device,
                               padding='random', lifestyle_augmentations=True, select='left',
                               no_event_token_rate=no_event_token_rate, cut_batch=True)

        # backward pass, with gradient scaling if training in fp16.
        # Normalize by accumulation steps so the accumulated grad is the MEAN over
        # micro-steps (not the sum) -- otherwise gradient_accumulation_steps>1 silently
        # scales the effective learning rate.
        loss_terms = loss
        loss = (loss_terms['loss_ce'] + loss_terms['loss_dt']) / gradient_accumulation_steps
        # GUARD 1 (fail-fast): abort the instant the loss is non-finite, BEFORE backward() can
        # write NaN into the weights (a silent NaN corrupts every checkpoint from here on). The
        # message names which term blew up + the max logit, so the cause is immediately obvious.
        if not torch.isfinite(loss):
            print(f"[nan-guard] non-finite loss at iter {iter_num} (micro {micro_step}): "
                  f"loss_ce={loss_terms['loss_ce'].item():.4f} loss_dt={loss_terms['loss_dt'].item():.4f} "
                  f"max|logit|={logits.detach().abs().max().item():.1f}. "
                  f"Aborting before it corrupts the weights.", flush=True)
            raise SystemExit(1)
        scaler.scale(loss).backward()
    # clip the gradient. clip_grad_norm_ returns the PRE-clip total norm; keep it for Guard 4
    # (a grad-norm spike leads a NaN by many iters, so logging it is an early-warning signal).
    grad_norm = None
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
        # GUARD 4 (leading indicators): grad-norm and max|logit| spike well before the loss goes
        # NaN -- surfacing them turns a blow-up into something you can see coming (and reach for
        # a lower LR / longer warmup) instead of discovering it 10k iters too late.
        gnf = float(grad_norm) if grad_norm is not None else float('nan')
        max_logit = logits.detach().abs().max().item()
        print(f"iter {iter_num}: loss {lossf:.4f}, gradnorm {gnf:.2f}, max|logit| {max_logit:.1f}, time {dt*1000:.2f}ms")

        metrics.update({
            "train/loss": lossf,
            "train/grad_norm": gnf,
            "train/max_logit": max_logit,
            "lr": lr,
        })

    if wandb_log and (iter_num % log_interval == 0 or "val/loss" in metrics):
        wandb.log(metrics)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
