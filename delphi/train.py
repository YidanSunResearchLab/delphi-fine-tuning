import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch

from model import Delphi, DelphiConfig
from utils import get_p2i, get_batch


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
dataset = 'ukb_simulated_data'
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
device = 'cpu'  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'float32'  # 'bfloat16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False  # use PyTorch 2.0 to compile the model to be faster

# delphi training
token_dropout = 0.0
t_min = 0.0  # 365.25/12.
mask_ties = True
# 位置编码开关。默认 False = 上游行为（wpe 注释掉，时间只通过加性绝对年龄码进入）。
# True 重新启用可学习的绝对位置嵌入 —— 见 model.py 的 DelphiConfig.pos_embedding。
pos_embedding = False
# 多任务判别头。默认 False = 上游行为。打开时需要 data/<dataset>/aux_labels_{train,val}.npy
# （delphi/make_aux_labels.py 生成，和 .bin 逐行对齐），缺文件直接报错，不静默退化。
# aux_n_targets 由那份 .npy 的列数决定，**不要**在 config 里手填；这里给 0 只是为了让
# configurator 认得这个 key。见 model.py 的 DelphiConfig.aux_head。
aux_head = False
aux_n_targets = 0
aux_lambda = 0.0
# 访视级时间模型（time_head / size_head）。默认 False = 上游行为。
# 注意：打开之后 lm_head 的 logits 不再是速率，采样侧必须同步改，见 model.py 的注释。
visit_heads = False
max_visit_size = 24
# 信息消融（figure 3 的 transformer 基线）。默认 = 上游行为。不加参数，见 model.py 改动 C。
attn_visits = 0
static_only = False
static_token_max = 33
static_first_only = False
ignore_tokens = [0]
# inclusive pre-shift id range of the background/lifestyle block whose ages get jittered
# (UKB default 3..11; ROSMAP's background block is wider)
lifestyle_token_min = 3
lifestyle_token_max = 11
data_fraction = 1.0
no_event_token_rate = 5


# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
with open('configurator.py') as f:
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

torch.set_default_dtype(ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint32, mode='r').reshape(-1, 3)

train_p2i = get_p2i(train_data)
val_p2i = get_p2i(val_data)

# downsample the data to requested fraction
if data_fraction < 1.0:
    train_p2i = train_p2i[:int(data_fraction * len(train_p2i))]

# 多任务判别头的标签。逐行对齐 train.bin / val.bin，由 make_aux_labels.py 生成。
# aux_head=False 时这两个恒为 None，get_batch 走原路径。
train_aux = val_aux = None
if aux_head:
    def _load_aux(split, n_rows):
        p = os.path.join(data_dir, f'aux_labels_{split}.npy')
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"aux_head=True 但缺 {p}。先跑：python make_aux_labels.py --dataset {dataset}")
        a = np.load(p)
        # 行数对不上就是标签接到了别人的位置上，而 BCE 照样会收敛 —— 必须在这里炸。
        assert a.shape[0] == n_rows, (
            f"{p} 有 {a.shape[0]} 行，{split}.bin 有 {n_rows} 行。两者必须逐行对齐："
            "重新分词之后要重新生成标签。")
        return np.ascontiguousarray(a, dtype=np.int8)
    train_aux = _load_aux('train', train_data.shape[0])
    val_aux = _load_aux('val', val_data.shape[0])
    assert train_aux.shape[1] == val_aux.shape[1]
    # 以文件为准覆盖 config：aux_n_targets 手填错了不会报错，只会让每一列学成别的终点。
    if aux_n_targets and aux_n_targets != train_aux.shape[1]:
        print(f"WARNING: config 里 aux_n_targets={aux_n_targets}，"
              f"以 aux_labels_train.npy 的 {train_aux.shape[1]} 列为准")
    aux_n_targets = int(train_aux.shape[1])
    print(f"aux labels: {train_aux.shape} / {val_aux.shape}, aux_lambda={aux_lambda}")

# 训练总损失的组成。aux / size 关着的时候这个列表就是 ['loss_ce', 'loss_dt']、权重全 1.0，
# 于是 estimate_loss 的 val_loss 和下面的反传目标都和改动前逐字符等价（乘 1.0 是精确的）。
loss_keys = ['loss_ce', 'loss_dt']
loss_weights = [1.0, 1.0]
if aux_head:
    loss_keys.append('loss_aux')
    loss_weights.append(aux_lambda)
if visit_heads:
    # 没有单独的 size_lambda：loss_ce / loss_dt 也都是权重 1.0，多加一个旋钮就多一个
    # 不可比的维度。要调就在 config 里调 aux_lambda 那一侧。
    loss_keys.append('loss_size')
    loss_weights.append(1.0)
loss_w = torch.tensor(loss_weights)

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9


print(f"found vocab_size = {vocab_size}")

# model init
# model_args 是硬编码的 dict：**新加的 config 不写进来就完全不生效**，而且不会有任何
# 报错（DelphiConfig 有默认值，训练照跑，只是 flag 形同虚设）。这个坑已经踩过一次。
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=vocab_size, dropout=dropout, token_dropout=token_dropout, t_min=t_min,
                  mask_ties=mask_ties, ignore_tokens=ignore_tokens,
                  pos_embedding=pos_embedding,
                  aux_head=aux_head, aux_n_targets=aux_n_targets, aux_lambda=aux_lambda,
                  visit_heads=visit_heads, max_visit_size=max_visit_size,
                  attn_visits=attn_visits, static_only=static_only,
                  static_token_max=static_token_max, static_first_only=static_first_only)  # start with model_args from command line

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
    # 这几个同样是**结构**（决定 state_dict 里有没有那几个 Linear），resume 时必须跟
    # checkpoint 走，否则 load_state_dict 会报 missing/unexpected key。用 .get 兜底，
    # 是为了让这些字段出现之前存的旧 ckpt 仍然能 resume（取当前默认值 = 关）。
    for k in ['pos_embedding', 'aux_n_targets', 'max_visit_size']:
        was = model_args[k]
        model_args[k] = checkpoint_model_args.get(k, model_args[k])
        if model_args[k] != was:
            print(f"WARNING: resume 用 checkpoint 的 {k}={model_args[k]}（config 写的是 {was}）")
    # aux_head / visit_heads 不能"跟 ckpt 走然后继续"：它们同时决定**结构**和**损失的组成**。
    # 上面的 loss_keys 是用 config 的值算的，跟 ckpt 走就会出现 loss_keys 里有 'loss_aux'
    # 而模型根本没有 aux_head（KeyError），或者反过来 —— 模型带着一个 aux_head 却没人喂它
    # 标签，那组权重整段 resume 都不更新（这一种是静默的）。所以直接拒绝。
    # 想给一个没有判别头的 ckpt 加判别头，是 init_from='scratch' 或者另写一段权重迁移，
    # 不是 resume。
    for k in ['aux_head', 'visit_heads']:
        ck_v = checkpoint_model_args.get(k, model_args[k])
        if ck_v != model_args[k]:
            raise RuntimeError(
                f"resume 的 checkpoint 是 {k}={ck_v} 训练的，但 config 写的是 "
                f"{model_args[k]}。这个 flag 同时改结构和损失的组成，resume 没法调和："
                f"要么把 config 的 {k} 改回 {ck_v}，要么 init_from='scratch' 从头训。")
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

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.amp.GradScaler(device_type, enabled=(dtype == 'float16')) 

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


def _fetch(ix, data, p2i, aux, **kw):
    """get_batch 的薄包装：aux/visit_heads 都关着时原样返回 4 元组 + 两个 None。

    存在的理由只有一个 —— 调用点不用为四种 flag 组合各写一遍解包。
    get_batch 本身仍然保持"默认 4 元组"的老契约（见它的 docstring）。
    """
    if aux is None and not visit_heads:
        X, A, Y, B = get_batch(ix, data, p2i, **kw)
        return X, A, Y, B, None, None
    return get_batch(ix, data, p2i, aux=aux, return_visit_size=visit_heads, **kw)


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, len(loss_keys))
        data = train_data if split == 'train' else val_data
        p2i = train_p2i if split == 'train' else val_p2i
        aux = train_aux if split == 'train' else val_aux
        for k in range(eval_iters):
            ix = torch.randint(len(p2i), (batch_size,))
            X, A, Y, B, AUX, KSZ = _fetch(ix, data, p2i, aux, block_size=block_size,
                                          device=device, select='left',
                                          no_event_token_rate=no_event_token_rate,
                                          cut_batch=True)
            with ctx:
                logits, loss, _ = model(X, A, Y, B, validation_loss_mode=True,
                                        aux_targets=AUX, visit_size_targets=KSZ)
            losses[k] = torch.stack([loss[n] for n in loss_keys])
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
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


# logging
if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
ix = torch.randint(len(train_p2i), (batch_size,))
X, A, Y, B, AUX, KSZ = _fetch(ix, train_data, train_p2i, train_aux,
                              block_size=block_size, device=device,
                              padding='random', lifestyle_augmentations=True, select='left',
                              lifestyle_token_range=(lifestyle_token_min, lifestyle_token_max),
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
        # 用**和训练目标同一套权重**加权，否则"选 val 最好的那一步"选的是另一个目标。
        # aux/size 都关着时 loss_w 全是 1.0，乘 1.0 是精确运算，这一行与
        # `losses['val'].sum().item()` 逐位相同。
        val_loss = (losses['val'] * loss_w).sum().item()
        print(f"step {iter_num}: train loss {(losses['train'] * loss_w).sum().item():.4f}, val loss {val_loss:.4f}")

        metrics.update({
            "train/agg_loss": (losses['train'] * loss_w).sum().item(),
            "val/loss":val_loss,
            "val/loss_ce": losses['val'][0].item(),
            "val/loss_dt": losses['val'][1].item()
        })
        # 逐项也记下来（未加权），否则调 aux_lambda 时看不出是哪一项在动
        for _j, _n in enumerate(loss_keys[2:], start=2):
            metrics[f"val/{_n}"] = losses['val'][_j].item()

        if always_save_checkpoint or best_val_loss > val_loss:
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))

        if best_val_loss > val_loss:
            best_val_loss = val_loss

        if iter_num % 10_000 == 0:
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
            logits, loss, att = model(X, A, Y, B, aux_targets=AUX, visit_size_targets=KSZ)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        ix = torch.randint(len(train_p2i), (batch_size,))
        X, A, Y, B, AUX, KSZ = _fetch(ix, train_data, train_p2i, train_aux,
                                      block_size=block_size, device=device,
                                      padding='random', lifestyle_augmentations=True, select='left',
                                      lifestyle_token_range=(lifestyle_token_min, lifestyle_token_max),
                                      no_event_token_rate=no_event_token_rate, cut_batch=True)

        # backward pass, with gradient scaling if training in fp16
        # 两个 flag 都关着时这三行就是原来的 `loss = loss['loss_ce'] + loss['loss_dt']`，
        # 一个多余的算子都不加（`+ 0.0` 也不行：那是多两次 kernel launch）。
        loss_terms = loss
        loss = loss_terms['loss_ce'] + loss_terms['loss_dt']
        if aux_head:
            loss = loss + aux_lambda * loss_terms['loss_aux']
        if visit_heads:
            loss = loss + loss_terms['loss_size']
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
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
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

        metrics.update({
            "train/loss": lossf,
            "lr": lr,
        })
        # 只在 log_interval 上取 .item()（每次都是一个 CPU-GPU 同步点），
        # 而且只在多任务打开时才有这几项
        for _n in loss_keys[2:]:
            metrics[f"train/{_n}"] = loss_terms[_n].item()

    if wandb_log and (iter_num % log_interval == 0 or "val/loss" in metrics):
        wandb.log(metrics)

    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break
