"""
Full definition of a GPT Language Model, all of it in this single file.
References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py
"""

import math
import inspect
from dataclasses import dataclass, field, fields

import torch
import torch.nn as nn
from torch.nn import functional as F

import warnings

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch nightly and still a bit scary
        self.flash = False #hasattr(torch.nn.functional, 'scaled_dot_product_attention') and self.dropout == 0.0
        if not self.flash:
            # print("WARNING: using slow attention. Flash Attention atm needs PyTorch nightly and dropout=0.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x, attn_mask):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k ,v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            #att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = att.masked_fill(attn_mask == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y, att

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask):
        y, att = self.attn(self.ln_1(x), attn_mask) 
        x = x + y
        x = x + self.mlp(self.ln_2(x))
        return x, att

class AgeEncoding(nn.Module):

    def __init__(self, config, max_dim: int = 1024):
        super().__init__()
        div_term = torch.exp(torch.arange(0, config.n_embd, 2) * (-math.log(10000.0) / config.n_embd))
        self.register_buffer('div_term', div_term)
        self.n_embd = config.n_embd
        self.linear = torch.nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        y = torch.zeros(x.shape[0], x.shape[1], self.n_embd, device=x.device)
        y[..., 0::2] = torch.sin(x / 365.25 * self.div_term) #* (1-self.div_term)
        y[..., 1::2] = torch.cos(x /365.25 * self.div_term) #* (1-self.div_term)
        y = self.linear(y)
        
        #x = self.wae[:x.size(0)]
        return y #self.dropout(x)
    
def next_visit_dt(idx, age, ignore=None):
    """The forward gap: for each position, (the next STRICTLY LATER age in the stream) - (its
    own age), in the same units as `age` (days).

    `ignore`: token ids that do NOT count as "the next event". None (default) counts every
    non-padding token, so the answer is "time to the next entry in the stream" -- which
    includes get_batch's synthetic No-event markers. Passing ignore_tokens instead gives
    "time to the next REAL event", which is what generate() is actually asked to reproduce,
    since it masks those tokens and can never emit one. Which of the two matches the true
    visit-gap distribution is an empirical question; experiments/time_head/dt_ablation.py
    (experiments/ was removed from this branch; still present at commit dba88f8)
    measures both.

    Returns (dt, ok). `ok` is False where there is no later age -- the subject's last visit --
    i.e. right-censored, no answer to score against. dt is set to 1.0 there so callers never
    see an inf; use `ok` to mask.

    This is the correct target for the time-to-next-event objective, and it is deliberately a
    masked min over the (t, t) age table rather than "the first j > i with a different age":
    that way it does not rely on the stream being perfectly sorted, and the strict `>` drops
    co-occurring (same-visit) tokens for free, so every token of a visit gets the same answer.

    Lives here, at module level, so `experiments/time_head/dt_ablation.py` can measure exactly
    what the loss uses instead of reimplementing it and drifting.
    """
    t = age.size(1)
    counts = idx > 0                                          # not padding
    if ignore is not None:
        for k in ignore:
            if k != 0:                                        # 0 is padding, already excluded
                counts = counts & (idx != k)
    later = age.unsqueeze(1)                                  # (b, 1, t) indexes j
    mine = age.unsqueeze(2)                                   # (b, t, 1) indexes i
    cand = (later > mine) & counts.unsqueeze(1)               # strictly later, and an event
    BIG = torch.finfo(age.dtype).max
    big = torch.full_like(later.expand(-1, t, -1), BIG)
    nxt = torch.where(cand, later.expand(-1, t, -1), big).min(-1).values
    ok = nxt < BIG
    dt = torch.clamp(nxt - age, min=1.0)
    return torch.where(ok, dt, torch.ones_like(dt)), ok


@dataclass
class DelphiConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    token_dropout: float = 0.0
    t_min: float = 1.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    # mask_ties=True masks same-visit tokens during training (matches the
    # paper's behavior). The training script defaults this to True; we set
    # the same default here so loading a checkpoint without overriding
    # produces the same model the checkpoint was trained with.
    mask_ties: bool = True
    ignore_tokens: list = field(default_factory=lambda: [0])
    # time_head=False -- the delivered parameterisation, and what every existing checkpoint
    # was trained with. The model then has exactly ONE output projection (`lm_head`): "which
    # event" is softmax(logits) and "when" is logsumexp(logits), a single scalar that is a
    # deterministic function of the same logits and has no parameters of its own.
    # time_head=True adds an independent scalar projection that predicts the log-intensity
    # directly. See Delphi.forward for the exact semantics, and experiments/time_head/ for
    # why (loss_dt is 73% of validation loss and moved by 0.10% of itself over a 6.9x
    # parameter range -- the capacity sweep's finding 2).
    time_head: bool = False
    # WHAT loss_dt is trained to predict.
    #
    # "gather"      -- the delivered behaviour, and what every existing checkpoint was trained
    #                  with. dt = targets_age - age, then REPLACED by the dt of the last
    #                  position the attention mask lets you see (see the gather in forward).
    #                  For the k-1 non-final tokens of a k-token visit that is the gap BEFORE
    #                  their own visit, not the gap after it -- backward-looking, and already
    #                  computable from their own input. Measured on the real split: 73.6% of
    #                  scored positions are trained on another position's dt, and that subset
    #                  runs 1.42x (median) / 1.97x (mean) longer than the true visit gaps,
    #                  while the 26.4% that keep their own dt sit at 0.95x / 1.04x.
    #
    # "next_visit"  -- the fix. dt = (the next STRICTLY LATER age in the stream) - (my own
    #                  age), i.e. the real time to the next visit, forward-looking, identical
    #                  for every token of a visit. Positions with no later age are censored
    #                  and are dropped from loss_dt (they are dropped from loss_dt ONLY --
    #                  loss_ce still scores them, since "which token comes next" is
    #                  well-defined even for a co-occurring target).
    #
    # "next_event"  -- same, but only REAL events count as "the next one": get_batch's
    #                  synthetic No-event markers are skipped over. This is what generate() is
    #                  asked to reproduce, since it masks those tokens and can never emit one.
    #
    # Default is "gather" on purpose: it keeps every delivered checkpoint bit-identical and
    # keeps the capacity sweep's numbers comparable. Flip it per-run to measure the fix.
    dt_target: str = "gather"
    # WHICH tokens the time-to-event target skips. None falls back to `ignore_tokens`, which is
    # the DELIVERED behaviour and is a bug: the two sets answer different questions.
    # `ignore_tokens` says "not a cross-entropy target" and must NOT contain No-event, because
    # the synthetic markers are the only negative evidence the model ever sees. This set says
    # "does not count as the next event", and No-event MUST be in it, because generate() masks
    # the marker and can never emit one -- so the intensity has to be trained on the gap it
    # will actually be asked to reproduce. Measured on the delivered build, 41.3% of dt targets
    # pointed at a synthetic marker, compressing the target mean from 1.65 y to 0.89 y and
    # turning 8.5% of censored positions into fabricated short answers.
    dt_ignore_tokens: list = None
    # MODEL-space id of the AD-diagnosis token, or 0 to disable the missing-label mask.
    #
    # RADC records no age_first_ad_dx for participants who were already demented at the
    # baseline cycle, so those subjects carry no AD token and the stream cannot tell them
    # apart from genuine non-converters. Passing `ad_label_missing` to forward() blanks this
    # column out of their cross-entropy, which asks "which of the OTHER tokens comes next"
    # instead of asserting that the diagnosis did not happen. 0 (Padding) is the off switch
    # because Padding can never be a real endpoint; no id is hard-coded here or in train.py.
    ad_dx_token: int = 0


def load_checkpoint(ckpt_path, device="cpu", expect_vocab_size=None,
                    expect_block_size=None):
    """Load a Delphi checkpoint and return (model, model_args).

    Performs a consistency check between the checkpoint's recorded
    model_args and any expected values supplied by the caller. This catches
    the failure mode where the committed config.yaml claims vocab_size=128
    but the actual checkpoint was trained with vocab_size=200 — the model
    will silently load but downstream code that indexes by token ID will
    be off the end of the logits.

    Args:
        ckpt_path: path to .pt checkpoint file.
        device: torch.device string.
        expect_vocab_size: if provided, raise if the checkpoint's vocab_size
            differs. Pass None to disable the check.
        expect_block_size: if provided, raise if the checkpoint's block_size
            differs.

    Returns:
        (model, model_args dict from the checkpoint).
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["model_args"]
    if expect_vocab_size is not None and args["vocab_size"] != expect_vocab_size:
        raise RuntimeError(
            f"Checkpoint vocab_size mismatch: ckpt has {args['vocab_size']}, "
            f"caller expected {expect_vocab_size}. The committed YAML config "
            f"may not match the checkpoint that was actually trained."
        )
    if expect_block_size is not None and args["block_size"] != expect_block_size:
        raise RuntimeError(
            f"Checkpoint block_size mismatch: ckpt has {args['block_size']}, "
            f"caller expected {expect_block_size}."
        )
    valid_keys = {f.name for f in fields(DelphiConfig)}
    filtered = {k: v for k, v in args.items() if k in valid_keys}
    model = Delphi(DelphiConfig(**filtered))
    state_dict = ckpt["model"]
    # Strip the torch.compile prefix if present.
    prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(prefix):
            state_dict[k[len(prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    return model, args

class Delphi(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            #wpe = nn.Embedding(config.block_size, config.n_embd),
            #wae = nn.Linear(1, config.n_embd, bias=True), ##nn.Embedding(config.block_size, config.n_embd),
            wae = AgeEncoding(config),
            #mlp = MLP(config),
            token_drop = nn.Dropout(config.token_dropout),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # The independent log-intensity head, absent unless asked for.
        #
        # Built and initialised AFTER the block above, and that ordering is load-bearing.
        # nn.Linear.__init__ draws from the global RNG (kaiming_uniform_) and those draws are
        # then thrown away by self.apply(_init_weights), so what actually sets every weight in
        # this model is the RNG STATE AT THE START OF apply(). Creating this head any earlier
        # shifts that state, and a time_head=True model would share no weight at all with its
        # time_head=False control at the same seed -- measured: every tensor differed. Built
        # here instead, the two models are bit-identical on every shared parameter, and
        # train.py re-seeds the batch RNG before the training loop, so they also see the same
        # batches. That is what makes the pair a controlled comparison of one added head
        # rather than two unrelated runs. verify.py checks it.
        #
        # The weight init is the plain one, normal_(0, 0.02). The BIAS is warm-started to
        # log(vocab_size), and that is not cosmetic -- it is what puts the two arms in the same
        # floor regime at step 0.
        #
        # loss_dt reaches the trunk through the floored log-intensity, whose derivative
        # w.r.t. the raw one is exp(-lse) / (exp(-lse) + t_min). The control starts at
        # lse = logsumexp of vocab_size near-zero logits = log(vocab_size), so exp(-lse) ~ 1/111,
        # tiny next to t_min = 365.25/12 = 30.44: it starts DEEPLY saturated and loss_dt is
        # nearly invisible to it (this is the saturation grad_balance.py measures). A zero bias
        # would put log lambda ~ 0, i.e. exp(-lse) = 1, which is ~111x further from the floor.
        # Measured at the arm config (8L/6H/120d, vocab 111, t_min = 365.25/12), at init:
        #
        #   arm                 floor-attn   loss_dt    ||g_dt(trunk)|| / ||g_ce(trunk)||
        #   control             2.79e-04     11.9101    0.0034     <- dt contributes ~nothing
        #   time_head, bias=0   3.71e-02     11.6347    3.8149     <- dt DOMINATES
        #   time_head, bias=logV 3.47e-04    11.9096    0.0373
        #
        # With a zero bias the arms are not running the same optimisation at step 0 -- one is
        # effectively CE-only, the other dt-dominated, a ~1100x gap in the timing gradient --
        # and any loss_dt win would be confounded with simply having started outside the floor.
        # log(vocab_size) starts the head at the control's own intensity, so loss_dt starts at
        # the same value in the same floor regime. The residual ~10x is structural: the
        # control's dt gradient reaches x as a softmax-weighted average over vocab_size lm_head
        # rows, which largely cancels, while the head's goes through a single row. That
        # difference IS the treatment, and it is what the arm exists to measure.
        #
        # The bias fill draws no RNG, so the ordering argument above is unaffected.
        if config.time_head:
            self.time_head = nn.Linear(config.n_embd, 1, bias=True)
            self._init_weights(self.time_head)
            with torch.no_grad():
                self.time_head.bias.fill_(math.log(config.vocab_size))
        else:
            self.time_head = None

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        #if non_embedding:
        #    n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, age, targets=None, targets_age=None, validation_loss_mode=False,
                ad_label_missing=None, soft_in=None, soft_target=None):
        device = idx.device
        b, t = idx.size()
        #assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        # forward the GPT model itself
        # Token embedding. With soft labels this is a weighted MIXTURE of the levels of one
        # ordinal scale rather than a single lookup -- w @ W[level_ids] instead of onehot @ W.
        # Same position, same vector, same sequence length; only the weights differ. Where w is
        # one-hot (every non-scale token) it reduces to the lookup bit for bit.
        if soft_in is not None:
            from .softlabel import mix_embeddings
            tok_emb = mix_embeddings(self.transformer.wte, soft_in[0], soft_in[1])
        else:
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        #pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        age_emb = self.transformer.wae(age.unsqueeze(-1)) # age embeddings of shape (b, t, n_embd)
        #age_emb = self.transformer.mlp(age_emb)
        x = self.transformer.token_drop(tok_emb) * (1-self.config.token_dropout) 
        x = x + age_emb
        x = self.transformer.drop(x)
        
        attn_mask = (idx>0).view(idx.size(0), 1, 1, idx.size(1)) * (idx>0).view(idx.size(0),1,idx.size(1),1)  # Do not attend to padded positions
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0 #self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0
        if targets is not None and self.config.mask_ties:
            attn_mask *= ((age.view(idx.size(0),1,1,idx.size(1)) != targets_age.view(idx.size(0),1,idx.size(1),1))) # Mask co-occuring tokens
            attn_mask += (attn_mask.sum(-1, keepdim=True)==0) * torch.diag(torch.ones(idx.size(1), device=device)) > 0
        attn_mask = attn_mask + (idx==0).view(idx.size(0), 1, 1, idx.size(1)) * torch.diag(torch.ones(idx.size(1), device=device)) > 0 # Except for padding
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0 #self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0

        
        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        # ---- the output head(s) -------------------------------------------------------
        # time_head=False: unchanged. `logits` is the raw lm_head projection and the loss
        # below reads the intensity off it as logsumexp(logits).
        #
        # time_head=True: `logits` becomes the PER-TOKEN LOG-RATES of the same
        # competing-exponential model, log_softmax(lm_head(x)) + log_lambda. Two exact
        # identities make that a drop-in replacement for every consumer:
        #   softmax(logits)   is unchanged -- softmax ignores a per-position constant -- so
        #                     loss_ce, ad_engine.next_event_probs, the panels' state
        #                     probabilities and generate()'s sampled token are all identical
        #                     functions of lm_head as before;
        #   logsumexp(logits) == log_lambda exactly, because logsumexp(log_softmax(.)) == 0,
        #                     so generate()'s inverse-CDF sampling picks up the new head with
        #                     no code change and with the right semantics: rate_k = lambda*p_k,
        #                     total rate lambda, min over k distributed Exp(lambda).
        # What changes is only WHERE the intensity comes from -- its own projection, trained
        # by loss_dt alone, instead of the sum of the token logits.
        if self.time_head is not None:
            log_lambda = self.time_head(x).squeeze(-1)                          # (b, t)
            logits = F.log_softmax(self.lm_head(x), dim=-1) + log_lambda.unsqueeze(-1)
        else:
            log_lambda = None
            logits = self.lm_head(x)

        if targets is not None:
            # if we are given some desired targets also calculate the loss
            ignored_tokens = self.config.ignore_tokens.copy()
            if validation_loss_mode:
                ignored_tokens += [1]
                logits[...,ignored_tokens] = -torch.inf
            targets = targets.reshape(-1)
            pass_tokens = targets != -1 
            for k in ignored_tokens: # and gender
                pass_tokens *= targets != k
            
            #age_min = age.gather(1,(((idx >=4) * (idx <=12)) + 0).argmax(1)[:,None])
            #logits[...,-1][age <= age_min] = -100. #-float('Inf') ## Death can only occur after age_min
            
            # ---- subjects whose AD label is MISSING, not negative -----------------------
            # `ad_label_missing` is a (b,) bool over the ROWS of this batch. Blanking the AD
            # column renormalises the softmax over the remaining ids, so no gradient pushes
            # P(AD) down at any of their positions -- "unknown", not "did not happen".
            #
            # ON A COPY, and both reasons matter. (1) The returned `logits` feed train.py's
            # guards 1 and 4, which report max|logit|; an in-place -inf makes that read inf
            # from the first iteration and blinds the blow-up detector. (2) `lse` below must
            # see the UNMASKED logits: under time_head=False the intensity is logsumexp of
            # this same tensor, and deflating it by the AD mass would quietly retarget the
            # timing head on these rows. The label is a claim about WHICH token, not WHEN.
            #
            # A masked row can never have AD as a target -- these subjects have no AD token by
            # construction -- but if one ever did, CE would be -log(0). train.py asserts that
            # at startup, once, rather than paying for the check on every batch.
            loss_logits = logits
            if ad_label_missing is not None and self.config.ad_dx_token:
                loss_logits = logits.clone()
                loss_logits[ad_label_missing, :, self.config.ad_dx_token] = -torch.inf

            loss_ce = F.cross_entropy(loss_logits.reshape(-1, loss_logits.size(-1))[pass_tokens], targets[pass_tokens], ignore_index=-1)
            
            # time to next event loss, padding masked
            #
            # With time_head the intensity is read straight off the head, NOT as
            # logsumexp of the rate-logits above. The two differ by exactly one thing:
            # validation_loss_mode has just written -inf into the ignored columns, so
            # logsumexp would return log(lambda * P(content)) -- lambda deflated by the
            # content probability mass, a quantity the head never sees in training (training
            # mode applies no such mask). Reading log_lambda directly means the head is
            # scored on the same definition it is trained on, in train and val alike.
            # The single-head path keeps logsumexp exactly as delivered.
            lse = log_lambda if log_lambda is not None else torch.logsumexp(logits,-1) ## More forgiving than using torch.max() for the most likely next event
            # GUARD 3 (numerical hygiene): bound the log-intensity before exp(). Prevents both
            # exp(-lse) underflowing to 0 (-> log(0) = -inf -> NaN loss_dt, the t_min=0 failure)
            # and exp(lse)*dt overflowing fp32. Never active in healthy training (lse stays O(1));
            # a pure safety net independent of t_min.
            lse = torch.clamp(lse, min=-60.0, max=60.0)
            lse = - torch.log(torch.exp(-lse) + self.config.t_min)
            # ---- WHAT the timing objective is asked to predict --------------------------
            # dt_target="next_visit": the real forward gap. get_batch emits the stream sorted
            # by age, so the next STRICTLY LATER age is the next visit, and every token of a
            # visit gets the same, correct answer. Positions with no later age are censored --
            # we do not know when that subject's next visit would have been -- so they are
            # dropped from loss_dt (only; loss_ce keeps them).
            #
            # Written as a masked min over the (t, t) age table rather than as "the first j > i
            # with a different age" so it does not depend on the stream being perfectly sorted;
            # the strict > also excludes the co-occurring tokens for free. Same cost as the
            # attention mask that is already built above.
            dt_ok = None
            if self.config.dt_target == "next_visit":
                dt, dt_ok = next_visit_dt(idx, age)
            elif self.config.dt_target == "next_event":
                dt_ign = self.config.dt_ignore_tokens or self.config.ignore_tokens
                dt, dt_ok = next_visit_dt(idx, age, ignore=dt_ign)
            elif self.config.dt_target != "gather":
                raise ValueError(f"dt_target must be 'gather', 'next_visit' or 'next_event', "
                                 f"got {self.config.dt_target!r}")
            else:
                dt = torch.clamp(targets_age - age, min=1.0)
            if self.config.dt_target == "gather" and self.config.mask_ties:
                # squeeze(1), not squeeze((1, 2)): the indices are (b, 1, t) and only the
                # head axis is meant to go. squeeze((1, 2)) also dropped the TIME axis when
                # t == 1, giving (b,) against a (b, 1) dt -- "Index tensor must have the same
                # number of dimensions as input tensor". Identical for every t > 1, so no
                # trained model is affected; it only stops a 1-token sequence from crashing
                # (which is what tests/golden.py's `single_token` case does).
                dt = torch.gather(dt, -1, (attn_mask * torch.arange(0, idx.size(1), device=device, dtype=torch.float32)
                                           .view(1, 1, 1, -1)).max(-1).indices.squeeze(1))  # Use time from last untied token
            ldt = - torch.log(dt + self.config.t_min).view(-1)
            
            loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1))) ## Exponential log-likelihood (real statistics, TM)
            # loss_ce and loss_dt share pass_tokens EXCEPT for censoring: a position in the
            # last visit has no later age, so under dt_target="next_visit" there is no answer
            # to score its timing against. Dropping it is the honest thing -- the old target
            # fed those positions the backward gap instead, which is where a large part of the
            # inflation came from. loss_ce is unaffected either way.
            dt_pass = pass_tokens if dt_ok is None else (pass_tokens & dt_ok.reshape(-1))
            loss_dt = torch.mean(loss_dt[dt_pass])
            
            # Both losses combined
            # loss = loss_ce + loss_dt
            loss = {'loss_ce': loss_ce, 'loss_dt': loss_dt}
            
            #loss += 5.0 * F.mse_loss(lse.view(-1)*(ldt != 0), ldt) ## Adds MSE for log time difference to next observed event
        else:
            # the head was already applied above, over every position (the "mini-optimization"
            # this branch used to describe -- projecting only the last position -- was never
            # actually done: the slice was x[:, :, :], the whole tensor. generate() needs the
            # last row only, but figure2's scoring reads every row, so all of them stay.)
            loss = None

        return logits, loss, att

    def crop_block_size(self, block_size):
        # Delphi uses AgeEncoding (continuous timestamp → embedding) instead of
        # a learned position embedding. There is no wpe table to crop. Block
        # size is enforced by the attention mask shape, not by a parameter.
        # Use adjust_block_size below to rebuild the attention bias buffer.
        raise NotImplementedError(
            "crop_block_size is a GPT-2 leftover and doesn't apply to Delphi "
            "(no learned position embedding). Use adjust_block_size instead."
        )
            
    def adjust_block_size(self, block_size):
        for block in self.transformer.h:
            block.attn.bias = torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size)

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('lm_head.weight')

        # Same reasoning, applied to the other arm's intensity parameters. The control's
        # intensity is logsumexp(lm_head(x)); lm_head.weight is tied to transformer.wte.weight,
        # which is blacklisted, so the control's intensity is NOT weight-decayed. Leaving
        # time_head.weight in the decay bucket would penalise the treatment's intensity and not
        # the control's -- and decay pulls that weight to 0, i.e. pulls log lambda toward a
        # constant, which is precisely the null the arm is testing against. Keep both undecayed.
        if self.time_head is not None:
            decay.discard('time_head.weight')
            no_decay.add('time_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)

        return optimizer

    @torch.no_grad()
    def generate(self, idx, age, max_new_tokens=100, max_age=85*365.25, no_repeat=True, termination_tokens=None, extra_ignore=None, visit_sizes=None, repeatable_tokens=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Selected parameters:
        --------------------

        termination_tokens: list[int] -  a list of tokens that indicate the and of the trajectory.
        Usually it is the "Death" token, but could be several tokens e.g. to indicate different
        death reasons.
        extra_ignore: list[int] - additional token ids to mask out during generation
        (e.g. [1] to suppress No-event tokens that otherwise dominate sampling).

        repeatable_tokens: list[int] - token ids EXEMPT from no_repeat. Default None keeps the
        delivered behaviour exactly.

        Why this exists. `no_repeat` was inherited from upstream Delphi, whose vocabulary is
        incident disease diagnoses: "first heart attack" can occur once, so blocking every
        token already in the sequence is correct there. This repository's vocabulary is not
        that. Its ordinal scales encode a CURRENT STATE under keep-transitions dedup, and a
        state recurs -- a subject who goes Normal -> impaired -> Normal emits the Normal bin
        twice, and keep-transitions exists precisely so those recoveries survive. `no_repeat`
        assigns every one of them probability zero.

        It also biases the *direction* of risk. Blocking every emitted token means an advanced
        subject, who has already passed through the milder bins, has fewer worse-bins left to
        sample than a mild one, so predicted worsening risk DECREASES with severity -- the
        opposite of the observed relationship. Keep-first tokens (incident diseases, first
        medication, death) have no such problem and should stay blocked, hence a list of
        exemptions rather than a global flag. radc_delphi.vocab exports the exempt set.

        visit_sizes: 1-D array of observed tokens-per-visit counts, or None (default) for the
        delivered behaviour. When given, generation emits a WHOLE VISIT per step instead of a
        single token: it samples the time to the next visit as usual, then draws k tokens at
        THAT SAME AGE, with k drawn from `visit_sizes`.

        Why this is needed: the real data records ~4 tokens at one age per visit (measured:
        mean 4.11, median 2, p95 16), but one-token-at-a-time sampling adds a strictly
        positive dt every time -- co-locating 4 tokens needs 3 consecutive draws of dt ~ 0,
        which happens 3.8% of the time. So one real visit becomes ~4 simulated steps and the
        trajectory runs 2.5x too slow (measured: 9.27 y to reach Dementia against a real
        3.67 y). See experiments/time_head/gen_steps_probe.py.

        Why the extra tokens reuse the SAME logits instead of re-running the model: with
        mask_ties, a token is trained NOT to attend to its same-visit siblings, so the model's
        own factorisation makes the tokens of one visit conditionally independent given the
        history. Drawing them i.i.d. from one forward pass is therefore what the training
        objective says to do, not an approximation of it.

        SIMPLIFICATION, stated because it is visible in the output: k is drawn once per step
        and shared by the whole batch, so rows are correlated within a step. The distribution
        of visit sizes across generated visits is still exactly `visit_sizes`; only the
        row-to-row independence is lost. Per-row k would need ragged sequence lengths.
        """
        if termination_tokens is None:
            # NO DEFAULT, DELIBERATELY. Upstream Delphi-2M defaulted to [1269] (its UK Biobank
            # Death id) and the NACC fork changed it to [110] (its own Death id). Either way it
            # is a vocabulary constant living in the architecture file, and a wrong one does not
            # raise -- it just silently never terminates, so trajectories run past death and
            # every survival number comes out too optimistic.
            #
            # This model is vocabulary-agnostic. The caller owns the vocabulary and must say
            # which tokens are absorbing; radc_delphi.vocab exports that list. Passing an empty
            # list here means "nothing terminates", which is the honest reading of a cohort with
            # no usable death token.
            termination_tokens = []

        ignore = list(self.config.ignore_tokens)
        if extra_ignore:
            ignore += list(extra_ignore)

        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=idx.device)
        mask_time = -10000

        if max_new_tokens == -1:
            max_new_tokens = 128

        # bool LUT over the vocabulary: True = exempt from no_repeat. Index 0 must stay False
        # so the "remap to 0" trick above keeps working for the padding slot.
        repeat_ok = None
        if repeatable_tokens is not None and no_repeat:
            repeat_ok = torch.zeros(self.config.vocab_size, dtype=torch.bool, device=idx.device)
            repeat_ok[torch.as_tensor(list(repeatable_tokens), dtype=torch.long,
                                      device=idx.device)] = True
            repeat_ok[0] = False

        block = self.config.block_size
        sizes = None if visit_sizes is None else torch.as_tensor(
            visit_sizes, device=idx.device, dtype=torch.long).flatten()
        n_new = 0
        for _ in range(max_new_tokens):
            # condition on at most the last block_size tokens so attention never runs at a
            # length the model was never trained on (the full idx/age are still grown for output).
            logits, _, _ = self(idx[:, -block:], age[:, -block:])
            logits = logits[:, -1, :]
            logits[:,ignore] = -torch.inf

            if no_repeat:
                fill = idx.clone()
                fill[fill == 1] = 0
                if repeat_ok is not None:
                    fill = torch.where(repeat_ok[fill], torch.zeros_like(fill), fill)
                logits = logits.scatter_(1, fill, -torch.inf)
            
            # sample from exponential distributions for each disease using the inverse CDF method, then take min
            t_next = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(), min=0, max=365*80).min(1)
            idx_next = t_next[1][:,None] # the index of the min sampled time
            age_next = age[...,[-1]] + t_next[0][:,None] # the value of the min sampled time
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            n_new += 1

            # ---- the rest of this VISIT, at the same age (visit_sizes only) ----------------
            if sizes is not None:
                k = int(sizes[torch.randint(len(sizes), (1,), device=sizes.device)].item())
                for _ in range(max(k - 1, 0)):
                    if n_new >= max_new_tokens:
                        break
                    p = torch.softmax(logits, -1)          # ignored tokens are already -inf
                    if no_repeat:                           # and nothing already emitted,
                        fill = idx.clone()                  # including earlier in THIS visit
                        fill[fill == 1] = 0
                        if repeat_ok is not None:
                            fill = torch.where(repeat_ok[fill], torch.zeros_like(fill), fill)
                        p = p.scatter(1, fill, 0.0)
                    tot = p.sum(1, keepdim=True)
                    if bool((tot <= 0).any()):              # a row has nothing left to draw
                        break
                    extra = torch.multinomial(p / tot, 1)
                    idx = torch.cat((idx, extra), dim=1)
                    age = torch.cat((age, age_next), dim=1)   # SAME age -- time does not move
                    n_new += 1

            if n_new >= max_new_tokens:
                break
            if torch.logical_or(torch.isin(idx, termination_tokens).any(-1), age_next > max_age).all():
                break
        
        pad = (torch.cumsum(torch.cumsum(torch.isin(idx, termination_tokens), 1).bool().int(), 1) > 1) + (age > max_age)

        logits, _, _ = self(idx, age)
        idx[pad] = 0
        age[pad] = mask_time

        if no_repeat:
            fill = idx + 0
            fill[fill == 1] = 0
            logits = torch.stack([logits[:,j].scatter_(1, fill[:,:j+1], -torch.inf) for j in range(fill.shape[1])]).transpose(0,1)

        return idx, age, logits