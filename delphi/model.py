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
from dataclasses import dataclass, field

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
    mask_ties: bool = False
    ignore_tokens: list = field(default_factory=lambda: [0])
    # 位置编码。上游把 wpe 注释掉了，模型对"时间"的唯一感知是 AgeEncoding —— **绝对**年龄的
    # 正弦码，加到 token 嵌入上。后果：要算"某个序数家族最近一次的值"，query 得在 n_embd 维
    # 空间里做一次跨位置的年龄大小比较，而**相对**时间 (age_q - age_k) 根本不可得。
    # 那个量恰好是单访视 logistic 基线（figure2_eval/window_baselines.py 的 W）的全部特征，
    # 也是实测里 transformer 唯一没能超过基线的地方（skill vs W 三档全为负）。
    #
    # pos_embedding=True 重新启用可学习的绝对位置嵌入。.bin 的 token 流是按年龄稳定排序的，
    # 所以**位置下标就是新近度排名**，"最近一次" = "下标最大的那个"，attention 一步可达。
    # 默认 False：保持交付版 ckpt 的可复现性。
    pos_embedding: bool = False

    # ---------------------------------------------------------------- 改动 A：多任务判别头
    # 为什么要它：figure2_eval/README.md 第 9.3 节实测，**同一个 ckpt**，只把 rollout 换成
    # "基线隐状态接 logistic"，panel a 中位 AUC 从 0.680 涨到 0.777，AD 5y 从 0.535 涨到
    # 0.818。也就是说塌陷在**输出机制**（逐 token 自回归采样 + 事后的经验访视大小补丁），
    # 不在表征。aux_head 把那个探针搬进模型里端到端训练：n_embd -> aux_n_targets 个独立
    # 的 sigmoid，标签是 figure2 的删失感知三值标签（1 / 0 / -1），由同目录的
    # make_aux_labels.py 生成，和 train.bin/val.bin **逐行对齐**。
    #
    # 默认 False，三个理由（都不是风格问题）：
    #   1) 打开就多一个 Linear，参数量和 state_dict 的 key 集合都变，交付版 ckpt 不再能
    #      字节级复现，engine.load 拿旧 ckpt 建新 config 也会多出没训过的权重；
    #   2) 它依赖一份和 .bin 逐行对齐的 aux_labels_<split>.npy，文件缺失/行数不符必须是
    #      明确报错，而不是悄悄退化成"没有监督的空头"；
    #   3) 它改变 val loss 的含义，而 train.py 用 val loss 选 checkpoint
    #      （always_save_checkpoint=False），开了之后选出来的就不是同一个 ckpt 了。
    aux_head: bool = False
    # 判别头的输出个数 = 终点数 × horizon 数（当前 3 × 5 = 15）。**不要手填**：train.py 从
    # aux_labels_<split>.npy 的列数推出来再写进 model_args。手填错了不会报错，BCE 照样
    # 收敛，只是每一列学的是别的终点 —— 典型的静默失败。
    aux_n_targets: int = 0
    # 总损失里判别头的权重（loss = loss_ce + loss_dt + aux_lambda * loss_aux）。
    # 0.0 = 头建出来但完全不参与训练，用来做"加了参数但不监督"的对照。
    # train.py 的 val loss 用**同一个**权重加权，否则选 ckpt 的目标和训练目标不是一回事。
    aux_lambda: float = 0.0

    # ---------------------------------------------------------------- 改动 B：访视级时间模型
    # 为什么要它：figure2_eval/README.md 第 3.2 节。mask_ties 下 `dt` 被 gather 换成"到上一个
    # **非同龄** token 的时间" = 访视间隔，所以时间头学到的是**访视速率**（实测
    # exp(-logsumexp) 中位 338 天 ≈ 观测访视间隔 365 天）；但生成时逐 token 采样，一次等待
    # 只发一个 token，真实访视却带 3.2–8.3 个。现在这个缺口靠 visit_sizes.npy 这个**事后
    # 经验分布补丁**填（第 7 节实测 fullvisit 均值已 8.34，补丁承担了生成过程的绝大部分）。
    #
    # visit_heads=True 把它拆成两件学出来的事：
    #   time_head  标量 log 速率 -> "下一次**访视**什么时候"（dt 的定义一个字不改，只换了
    #              速率的来源，所以和现在的 loss_dt 是同一个指数对数似然）
    #   size_head  "这一次访视发几个 token" 的分布（新的 loss_size）
    #   lm_head    继续只管"发什么"（loss_ce 不变）
    #
    # 默认 False：打开之后 lm_head 的 logits **不再被训练成速率**，上游 generate() 和
    # figure2_eval/radc_delphi/engine.simulate() 里那套 `-exp(-logits)*log(u)` 取 min 的
    # 采样就失去了含义 —— 这是本改动最危险的静默失败，采样器必须同步改（见 generate()
    # 里的 RuntimeError 和 forward() 里的一次性 warning）。
    visit_heads: bool = False
    # size_head 的类数：k ∈ {1 .. max_visit_size}，顶格那一类的含义是 ">= max_visit_size"。
    # 24 够覆盖 ROSMAP 的非基线访视（2–9 个 token）。基线那一次（statics + 首访 = 21 个）
    # 永远不会成为标签，因为它前面没有位置可问。如果换了一个访视更大的 build，这里不加大
    # 是静默的：模型学不会补发那么多 token，生成出来的访视会系统性偏小。
    max_visit_size: int = 24

    # ---------------------------------------------------------------- 改动 C：信息消融（figure 3 基线）
    # 为什么要它：figure 3 的基线必须是 **transformer 本身**，同架构、同配方，只拿掉一部分信息，
    # 这样"主模型 − 基线"才能读成"transformer 从这部分信息里挣到了多少"。两个开关都**不加参数**，
    # 只改 forward 里的输入和注意力 mask，所以 state_dict 与关着时完全相同，推理时也可以直接
    # 覆盖到一个训好的 ckpt 上（= 截断上下文的推理消融）。
    #
    # attn_visits = 1：每个位置只能注意背景块 + 当前这次访视 —— "只看现在"的 transformer。
    #   只支持 1：k>=2 的滑动窗口会跨层泄漏（见 _scope_mask）；推理时的"最近 k 次"截断改在
    #   evaluate_auc_rosmap_controls.py 里重建输入序列来做。访视按**非背景真实 token** 的年龄变化计数，背景块按 id 识别
    #   —— 不能按年龄，因为 lifestyle_augmentations 会把背景块的年龄抖到 ±20/40 年外。
    #   no-event token 不开新访视，所以访视之间的 no-event 预测点看到的是上一次访视。
    #   和 mask_ties 叠加（取交集），所以同访视内的预测仍然看不到同访视的 token。
    #   要看到完整的"现在"，得配 fullvisit 数据（每次访视发射全部状态）；在去重数据上"当前
    #   访视"只有变化的那几个量。
    # static_only = True：非背景 token 在**输入**里一律换成 no-event（目标不变），且只能注意
    #   背景块和自己。模型知道的只有性别、背景块和当前年龄 —— transformer 版的人口学先验。
    #   必须两件一起做：只改 mask 不改输入的话，残差流里仍带着自己那个 token 的身份。
    # 两个都要配 pos_embedding=False：位置下标 = 前面有多少个 token = 历史有多长。
    attn_visits: int = 0
    static_only: bool = False
    # 背景块（含性别）的 post-shift id 上界：背景 = [2, static_token_max]。ROSMAP 是 33，
    # 与 ignore_tokens = [0] + list(range(2, 34)) 同一段。
    static_token_max: int = 33
    # 只让背景 token 的**每个 id 的第一份拷贝**参与改动 C 的注意力范围。snapshot 分词
    # （tokenization/build.py --snapshot）每次访视都重发背景块，不加这一条的话，数背景拷贝
    # 的份数 = 数访视次数，历史长度就漏进了"只看现在 / 只看人口学"的基线。第一份拷贝就是
    # 其它分词里唯一的那一份，所以两种数据上的基线看到的背景信息相同。关着时与改动前逐位一致。
    static_first_only: bool = False

class Delphi(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            #wae = nn.Linear(1, config.n_embd, bias=True), ##nn.Embedding(config.block_size, config.n_embd),
            wae = AgeEncoding(config),
            #mlp = MLP(config),
            token_drop = nn.Dropout(config.token_dropout),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = LayerNorm(config.n_embd, bias=config.bias),
        ))
        if config.pos_embedding:
            self.transformer["wpe"] = nn.Embedding(config.block_size, config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # 两个改动的额外读出口。**只在 flag 打开时才建**，否则 state_dict 会多出 key，
        # 交付版 ckpt 的参数量（vocab 129 / block 144 / 6 层 / 96 维 -> 0.693M）就变了。
        if config.aux_head:
            assert config.aux_n_targets > 0, (
                "aux_head=True 但 aux_n_targets=0：判别头会是一个 0 列的 Linear，loss_aux 恒为 0 "
                "而训练照跑不误。aux_n_targets 应由 train.py 从 aux_labels_<split>.npy 的列数推出。")
            self.aux_head = nn.Linear(config.n_embd, config.aux_n_targets)
        if config.visit_heads:
            self.time_head = nn.Linear(config.n_embd, 1)
            self.size_head = nn.Linear(config.n_embd, config.max_visit_size)
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

    def _scope_mask(self, idx, age):
        """改动 C 的注意力范围，(b, 1, t, t) bool，[q, k] = query q 能否看 key k。见 DelphiConfig。

        必须对**多层叠加**封闭：第 l 层里 key 的表示已经混进了它自己第 l-1 层能看到的东西。
        所以 (1) 背景 token 作为 query 只能看背景块 —— lifestyle_augmentations 会把它们挪到
        访视之间，否则它们会吸收那次访视的信息再转交给后面的预测点；(2) 访视窗口只能是
        互不相交的块（k=1），滑动窗口 k>=2 的感受野会随层数变成 (k-1)*n_layer+1 次访视
        （test_scope_mask.py 实测 k=2 漏 4.8e-5）。"""
        smax = self.config.static_token_max
        b, t = idx.shape
        eye = torch.eye(t, dtype=torch.bool, device=idx.device)[None, None]
        is_static = (idx >= 2) & (idx <= smax)
        if self.config.static_first_only:
            # 位置 j 的 token 在 j 之前是否出现过：one-hot 沿时间做 cumsum（不含自己）
            oh = F.one_hot(idx.clamp(min=0), num_classes=self.config.vocab_size)
            seen_before = (oh.cumsum(1) - oh).gather(-1, idx.clamp(min=0).unsqueeze(-1)).squeeze(-1) > 0
            is_static = is_static & ~seen_before
        static_k = is_static.view(b, 1, 1, t)
        if self.config.static_only:
            return static_k | eye
        # 访视编号：非背景真实 token 的年龄每变一次 +1。流按年龄排序，所以"上一个真实 token 的
        # 年龄"就是到前一位为止的 cummax。no-event / padding / 背景块都不开新访视。
        real = idx > smax
        neg = torch.full_like(age, float("-inf"))
        last = torch.cummax(torch.where(real, age, neg), dim=1).values
        prev = torch.cat([neg[:, :1], last[:, :-1]], dim=1)
        vid = torch.cumsum((real & (age != prev)).long(), dim=1)
        same_visit = vid.view(b, 1, 1, t) == vid.view(b, 1, t, 1)
        # 背景 query：只看背景；其余 query：背景 + 同一次访视。
        # static_first_only 下，背景的重复拷贝既不当"背景"也不单独开访视：它们只作为所在访视的
        # 普通成员被同访视的 query 看到（real 的定义不含背景，所以不改访视编号）。
        is_static_q = ((idx >= 2) & (idx <= smax)).view(b, 1, t, 1)
        return torch.where(is_static_q, static_k | eye, static_k | same_visit)

    def forward(self, idx, age, targets=None, targets_age=None, validation_loss_mode=False,
                aux_targets=None, visit_size_targets=None):
        """新参数一律**追加在末尾**：engine.py / evaluate_auc*.py / train.py 都按位置传前四个。

        aux_targets:        (b, t, aux_n_targets)，取值 {1, 0, -1}，-1 = 该位置该终点删失/未知，
                            必须从损失里剔除；对齐到**输入**位置（utils.get_batch 负责）。
        visit_size_targets: (b, t)，下一次访视的 token 数 k >= 1，-1 = 该位置不监督。
        两个都是 None 时，这个函数和改动前逐字符等价。
        """
        device = idx.device
        b, t = idx.size()
        #assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        # pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)
        # forward the GPT model itself
        if self.config.static_only:
            # 非背景 token 在输入里抹成 no-event（id 1），年龄保留。目标 / padding mask 用原 idx。
            tok_emb = self.transformer.wte(idx.masked_fill(idx > self.config.static_token_max, 1))
        else:
            tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        age_emb = self.transformer.wae(age.unsqueeze(-1)) # age embeddings of shape (b, t, n_embd)
        #age_emb = self.transformer.mlp(age_emb)
        x = self.transformer.token_drop(tok_emb)
        x = x + age_emb
        if self.config.pos_embedding:
            # 流按年龄稳定排序 -> 位置下标 = 新近度排名。给 attention 一个能直接比较"谁更近"
            # 的坐标；加性绝对年龄码做不到这件事。见 DelphiConfig.pos_embedding。
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            x = x + self.transformer.wpe(pos)[None, :, :]
        x = self.transformer.drop(x)
        
        attn_mask = (idx>0).view(idx.size(0), 1, 1, idx.size(1)) * (idx>0).view(idx.size(0),1,idx.size(1),1)  # Do not attend to padded positions
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0 #self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0
        if targets is not None and self.config.mask_ties:
            attn_mask *= ((age.view(idx.size(0),1,1,idx.size(1)) != targets_age.view(idx.size(0),1,idx.size(1),1))) # Mask co-occuring tokens
            attn_mask += (attn_mask.sum(-1, keepdim=True)==0) * torch.diag(torch.ones(idx.size(1), device=device)) > 0
        attn_mask = attn_mask + (idx==0).view(idx.size(0), 1, 1, idx.size(1)) * torch.diag(torch.ones(idx.size(1), device=device)) > 0 # Except for padding
        attn_mask *= torch.tril(torch.ones(idx.size(1),idx.size(1), device=device))[None,None,:,:] > 0 #self.transformer.h[0].attn.bias[:,:,:idx.size(1),:idx.size(1)] > 0
        if self.config.attn_visits > 0 or self.config.static_only:
            assert self.config.attn_visits in (0, 1), \
                "attn_visits 只支持 1：k>=2 的滑动窗口会跨层泄漏，见 _scope_mask"
            attn_mask = attn_mask & self._scope_mask(idx, age)
            # 和上面 mask_ties 的处理一样：整行被遮空的位置只留自己，否则 softmax 出 NaN
            eye = torch.eye(t, dtype=torch.bool, device=device)[None, None]
            attn_mask = attn_mask | ((attn_mask.sum(-1, keepdim=True) == 0) & eye)


        att = []
        for block in self.transformer.h:
            x, a = block(x, attn_mask)
            att.append(a)
        x = self.transformer.ln_f(x)
        att = torch.stack(att)

        if targets is not None:
            # next token cross entropy loss, padding masked
            logits = self.lm_head(x)

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
            
            loss_ce = F.cross_entropy(logits.reshape(-1, logits.size(-1))[pass_tokens], targets[pass_tokens], ignore_index=-1)
            
            # time to next event loss, padding masked
            if self.config.visit_heads:
                # 访视级时间模型：速率改由**专用标量头**给，而不是从 logsumexp(lm_head) 来。
                # dt 的定义（下面几行）一个字不改，所以这是**同一个**指数对数似然，只是
                # 换了速率的来源 —— 目的是把"何时来访"和"来了发什么"解耦：解耦之前
                # logsumexp(logits) 同时充当这两件事的归一化常数，生成时一次等待只能发
                # 一个 token，缺的那 3–8 倍只能靠 visit_sizes.npy 事后补。
                lse = self.time_head(x).squeeze(-1)
            else:
                lse = torch.logsumexp(logits,-1) ## More forgiving than using torch.max() for the most likely next event
            lse = - torch.log(torch.exp(-lse) + self.config.t_min)
            dt = torch.clamp(targets_age - age, min=1.0)
            if self.config.mask_ties:
                dt = torch.gather(dt, -1, (attn_mask * torch.arange(0, idx.size(1), device=device, dtype=torch.float32)
                                           .view(1, 1, 1, -1)).max(-1).indices.squeeze((1, 2)))  # Use time from last untied token
            ldt = - torch.log(dt + self.config.t_min).view(-1)
            
            loss_dt = -(lse.reshape(-1) - torch.exp(lse.reshape(-1) - ldt.reshape(-1))) ## Exponential log-likelihood (real statistics, TM)
            loss_dt = torch.mean(loss_dt[pass_tokens]) 
            
            # Both losses combined
            # loss = loss_ce + loss_dt
            loss = {'loss_ce': loss_ce, 'loss_dt': loss_dt}

            # ------------------------------------------------ 改动 A：多任务判别头的 BCE
            # 只在 flag 打开**且**这一批带了标签时才加 key。key 集合随 flag 变，是为了让
            # train.py 那边"aux 关着的时候数值逐字符不变"这件事是结构性的，而不是靠
            # 权重恰好等于 0。
            if self.config.aux_head and aux_targets is not None:
                K = self.config.aux_n_targets
                # 形状必须硬查。`reshape(-1, K)` 在列数不等时**通常照样成功**（b*t*N 能被 K
                # 整除就行），然后每一列学的是别的终点 —— BCE 收敛、AUC 像个数、没有任何报错。
                # 这是 resume 一个旧 ckpt（它的 aux_n_targets 冻在 model_args 里）配上一份
                # 重新生成、列数变了的 aux_labels_*.npy 时的真实路径。
                assert aux_targets.shape[-1] == K, (
                    f"aux_targets 有 {aux_targets.shape[-1]} 列，aux_head 只有 {K} 个输出。"
                    "两者必须相等 —— 重新生成过 aux_labels_*.npy 就不能 resume 旧 ckpt。")
                assert aux_targets.shape[:2] == idx.shape, (
                    f"aux_targets 前两维 {tuple(aux_targets.shape[:2])} != idx {tuple(idx.shape)}："
                    "标签没有跟着 get_batch 的裁剪/位移走。")
                aux_logits = self.aux_head(x).reshape(-1, K)
                at = aux_targets.reshape(-1, K)
                # 两层掩码，缺任何一层都会静默地把损失算歪：
                #   at >= 0        ——  -1 是"删失/未知"，把它当阴性正是 labels_at_h 要避免的
                #                      系统性低估（figure2_eval/README.md 9.5 的第 (1) 条）；
                #   pass_tokens    ——  padding 和 ignore_tokens（statics 块）的位置，
                #                      loss_ce 也是这样排掉的，两边口径必须一致。
                valid = (at >= 0) & pass_tokens.unsqueeze(-1)
                bce = F.binary_cross_entropy_with_logits(
                    aux_logits, at.clamp(min=0).to(aux_logits.dtype), reduction='none')
                # 手写 masked mean 而不是先筛后算：筛出来的元素个数会随 batch 变，
                # 在 GPU 上是一次同步；乘 0 的那些位置梯度恰好是 0（目标已 clamp 过，
                # 不会出 NaN），分母 clamp(1) 保证整批全删失时得到 0 而不是 0/0。
                loss_aux = (bce * valid).sum() / valid.sum().clamp(min=1)
                loss['loss_aux'] = loss_aux

            # ------------------------------------------------ 改动 B：下一次访视的 token 数
            if self.config.visit_heads and visit_size_targets is not None:
                assert visit_size_targets.shape == idx.shape, (
                    f"visit_size_targets {tuple(visit_size_targets.shape)} != idx "
                    f"{tuple(idx.shape)}：标签没有跟着 get_batch 的裁剪/位移走。")
                M = self.config.max_visit_size
                size_logits = self.size_head(x).reshape(-1, M)
                k = visit_size_targets.reshape(-1)
                # k 截到 M，顶格那一类读作 ">= M"。get_batch 只在"下一个 token 开启一次
                # **新**访视"的位置给 k（其余 -1），这正好是生成时会去问 size_head 的
                # 位置分布：采到一个非 no-event token 才补发整次访视。
                k_cls = torch.where(k >= 1, torch.clamp(k, max=M) - 1, torch.full_like(k, -1))
                valid_k = k_cls >= 0
                ce = F.cross_entropy(size_logits, k_cls.clamp(min=0), reduction='none')
                loss_size = (ce * valid_k).sum() / valid_k.sum().clamp(min=1)
                loss['loss_size'] = loss_size


            #loss += 5.0 * F.mse_loss(lse.view(-1)*(ldt != 0), ldt) ## Adds MSE for log time difference to next observed event
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, :, :]) # note: using list [-1] to preserve the time dim
            loss = None
            if self.config.visit_heads:
                # 一次性（Python 默认 warning filter 按代码行去重）。visit_heads=True 之后
                # loss_dt 不再经过 lm_head，logits 的**尺度**就不再是速率，任何
                # `exp(-logits)` 当等待时间的采样器（本文件的 generate、
                # figure2_eval/radc_delphi/engine.simulate）都会输出垃圾而不报错。
                # 推理侧要用时间/大小，请直接调 model.time_head / model.size_head。
                warnings.warn(
                    "visit_heads=True: lm_head 的 logits 不再被训练成事件速率。"
                    "任何把 exp(-logits) 当等待时间的采样器（engine.simulate / generate）"
                    "必须改成用 time_head + size_head，否则结果是静默错的。")

        return logits, loss, att

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
            
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

        # aux_head / time_head / size_head 都是 nn.Linear：.weight 走下面的 whitelist 进
        # decay、.bias 走 endswith('bias') 进 no_decay，不需要任何特判。下面那句
        # `assert len(param_dict.keys() - union_params) == 0` 就是这件事的看门狗 ——
        # 哪天新头不是 Linear 了（比如 nn.Parameter），它会在第一步就炸，而不是静默地
        # 让那组参数完全不被优化器更新。
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
    def generate(self, idx, age, max_new_tokens=100, max_age=85*365.25, no_repeat=True, termination_tokens=None, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.

        Selected parameters:
        --------------------

        termination_tokens: list[int] -  a list of tokens that indicate the and of the trajectory.
        Usually it is the "Death" token, but could be several tokens e.g. to indicate different
        death reasons.
        top_k: None, does nothing
        """
        if self.config.visit_heads:
            # 宁可炸也不要静默出错：下面的采样把 exp(-logits) 当每个 token 的等待时间取 min，
            # 而 visit_heads=True 时速率在 time_head 里、访视大小在 size_head 里，logits 只剩
            # "发什么"。照原样跑不会报错，只会给出一个时间尺度完全错的轨迹。
            raise RuntimeError(
                "Delphi.generate() 不支持 visit_heads=True：速率在 time_head、访视大小在 "
                "size_head，这里的 `-exp(-logits)*log(u)` 取 min 已经失效。需要的采样是："
                "wait ~ Exp(time_head)，k ~ Categorical(size_head)，再从 softmax(lm_head) "
                "抽 k 个 token 放在同一个 age 上（参考 figure2_eval/radc_delphi/engine.py "
                "的 use_visit_sizes 分支，把经验分布换成 size_head）。")
        if termination_tokens is None:
            warnings.warn('When using a custem dataset, consider changing the `termination_tokens` argument.')
            termination_tokens = [1269]
        
        termination_tokens = torch.tensor(termination_tokens, dtype=torch.int64, device=idx.device)
        mask_time = -10000

        if max_new_tokens == -1:
            max_new_tokens = 128

        for _ in range(max_new_tokens):
            logits, _, _ = self(idx, age)
            logits = logits[:, -1, :]
            logits[:,self.config.ignore_tokens] = -torch.inf

            if no_repeat:
                fill = idx.clone()
                fill[fill == 1] = 0
                logits = logits.scatter_(1, fill, -torch.inf)
            
            # sample from exponential distributions for each disease using the inverse CDF method, then take min
            t_next = torch.clamp(-torch.exp(-logits) * torch.rand(logits.shape, device=idx.device).log(), min=0, max=365*80).min(1)
            idx_next = t_next[1][:,None] # the index of the min sampled time
            age_next = age[...,[-1]] + t_next[0][:,None] # the value of the min sampled time
            
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            age = torch.cat((age, age_next), dim=1)
            
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