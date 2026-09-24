import numpy as np
import torch
import re


def get_p2i(data):
    """
    Get the patient to index mapping.
    """

    px = data[:, 0].astype('int')
    p2i = []
    j = 0
    q = px[0]
    for i, p in enumerate(px):
        if p != q:
            p2i.append([j, i - j])
            q = p
            j = i
        if i == len(px) - 1:
            # add last participant
            p2i.append([j, i - j + 1])
    return np.array(p2i)


# "No event" 在模型空间里恒为 1（labels.csv 第 1 行），因为下面那句 `tokens = tokens + 1`
# 把注入的 disk id 0 移成了 1；padding 恒为 0（被 mask 掉的行填 -1，+1 之后变 0）。
# 这两个常数在本文件里本来就是写死的（见 `x.masked_fill((x == 0) * (y == 1), 0)`），
# 这里只是给它们一个名字，免得访视大小那段又硬编一遍。
_NO_EVENT_TOKEN = 1
_PADDING_TOKEN = 0


def _visit_size_targets(ages, tokens, reached_end, jitter_range=None):
    """下一次访视的 token 数 k，对齐到**输入**位置。返回 (B, L-1) int64，-1 = 不监督。

    为什么能从 ages 推：ROSMAP 的时间轴是年网格，一次访视的**每个** token 共用同一个 age
    （../tokenization/README.md 实测 83.81% 的 Δt 恰好为 0）。所以排好序之后，"一次访视"
    就是"一段同龄的连续 token"，k = 这一段的长度。

    只在**下一个 token 开启一段新的同龄 run** 的位置给标签，别的位置一律 -1。理由不是省事：
      * 那正好是生成时会去问 size_head 的位置 —— engine 采到一个非 no-event token 才补发
        整次访视，query 永远是"上一次访视的最后一个 token"；
      * mask_ties 下也只有这个位置能 attend 到自己这一次访视的全部 token（同龄的 key 被
        遮掉了），访视中间的位置连自己这次访视都看不见，让它们预测同一个 k 是把两个不同
        的条件分布混进一个头。

    四种位置额外剔除（每一条不剔都是**系统性**偏差，不是噪声）：
      * 输入是 padding；
      * 下一个 token 是注入的 no-event —— 它是 get_batch 现编的合成标记，自己独占一个
        age、run 长度恒为 1，而生成侧 no-event **不开访视**。不剔的话 no-event 会贡献
        接近一半的 k=1 样本，size_head 被直接压塌；
      * 这一段 run 碰到了窗口右边界**且**窗口没覆盖到病人最后一行 —— 那是被裁断的半次
        访视，数出来的 k 偏小。窗口覆盖到病人末尾时（select='left' + 轨迹短于 block_size
        的常见情形）最后一段是完整的，照常监督。
      * 输入或目标是**被年龄抖动过的背景块 token**（`jitter_range`，给的是位移前的 id）。
        lifestyle_augmentations 把 ~19 个 statics 打散到随机年龄上，每一个要么自成一次
        "访视"（k=1），要么并进某次真实访视把 k 抬高 1。生成时前缀里的 statics 不会被
        打散，所以这两种样本都只存在于训练分布里 —— 不剔，size_head 会被系统性拉小。
    """
    B, L = ages.shape
    idx = torch.arange(L, device=ages.device).expand(B, L)
    # 每个 token 是不是它那一段同龄 run 的最后一个（最后一列无条件算"是"）
    is_last = torch.ones(B, L, dtype=torch.bool, device=ages.device)
    is_last[:, :-1] = ages[:, 1:] != ages[:, :-1]
    # run 末列的下标：对翻转后的序列做 cummin，就是"从这一列往右第一个 run 末尾"。
    # 用 cummin/flip 而不是 python 循环，是因为 get_batch 每个训练步都要跑一次，
    # 144 次小算子的开销会直接加到 iter 时间上。
    big = torch.full_like(idx, L)
    end = torch.flip(torch.cummin(torch.flip(torch.where(is_last, idx, big), [1]), dim=1).values, [1])
    run_to_end = end - idx + 1

    opens = ages[:, 1:] != ages[:, :-1]          # 位置 i 的目标 i+1 开启了一段新 run
    k = run_to_end[:, 1:]                        # 目标所在 run 的长度 = 下一次访视的大小
    truncated = (end[:, 1:] == L - 1) & (~reached_end[:, None])
    valid = (opens & (~truncated)
             & (tokens[:, :-1] != _PADDING_TOKEN)
             & (tokens[:, 1:] != _NO_EVENT_TOKEN))
    if jitter_range is not None:
        # +1 是 get_batch 的 disk -> model 位移（`tokens = tokens + 1`），
        # lifestyle_token_range 给的是**位移前**的 id。
        lo, hi = jitter_range
        jit = (tokens >= lo + 1) & (tokens <= hi + 1)
        valid = valid & (~jit[:, :-1]) & (~jit[:, 1:])
    return torch.where(valid, k, torch.full_like(k, -1))


def get_batch(ix, data, p2i, select='random', index='patient', padding='regular',
              block_size=48, device='cpu', lifestyle_augmentations=False,
              lifestyle_token_range=(3, 11),
              no_event_token_rate=5, cut_batch=False,
              aux=None, return_visit_size=False):
    """
    Get a batch of data from the dataset. This function packs sequences in a batch and also
    inserts "no event" tokens randomly with the average rate of one every five years.

    Args:
        ix: list of indices to get data from
        data: numpy array of the dataset
        p2i: numpy array of the patient to index mapping
        select: 'left', 'right', or 'random' (window taken from the start, end, or a random offset of the trajectory)
        index: 'patient', 'random'
        padding: 'regular', 'random'
        block_size: size of the block to get
        device: 'cpu' or 'cuda'
        lifestyle_augmentations: whether to perform aurmentations of lifestyle token times
        lifestyle_token_range: (lo, hi) inclusive pre-shift token ids of the background/lifestyle
            block to jitter. Default (3, 11) is the UKB vocabulary; ROSMAP uses (3, 32).
        no_event_token_rate: average rate of "no event" tokens in years
        cut_batch: whether to cut the batch to the smallest size possible
        aux: 可选，(n_rows, n_targets) 的 int8 数组，和 data **逐行对齐**
            （delphi/make_aux_labels.py 生成）。默认 None = 完全不碰原路径。
        return_visit_size: 可选，是否额外返回"下一次访视的 token 数"。默认 False。

    Returns:
        x: input tokens
        a: input ages
        y: target tokens
        b: target ages

        aux is None 且 return_visit_size=False（默认）时返回 4 元组，和改动前逐字符一致 ——
        figure2_eval/probe_*.py、evaluate_auc*.py 等一票调用点都是 `X,A,Y,B = get_batch(...)`。
        只要请求了其中任何一个，就返回**定长 6 元组** (x, a, y, b, aux_x, visit_size)，
        没请求的那个是 None。定长是刻意的：变长返回值在调用点解包出错时报的是
        "too many values to unpack"，而不是把 aux 当成 y。
    """

    mask_time = -10000.

    x = torch.tensor(np.array([p2i[int(i)] for i in ix]))
    ix = torch.tensor(np.array(ix))

    gen = torch.Generator(device='cpu')
    gen.manual_seed(ix.sum().item())  # we want some things be random, but also deterministic

    if index == 'patient':
        if select == 'left':
            traj_start_idx = x[:, 0]
        elif select == 'right':
            traj_start_idx = torch.clamp(x[:, 0] + x[:, 1] - block_size - 1, 0, data.shape[0])
        elif select == 'random':
            traj_start_idx = x[:, 0] + (torch.randint(2**63-1, (len(ix),), generator=gen) % torch.clamp(x[:, 1] - block_size, 1))
            traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    traj_start_idx = torch.clamp(traj_start_idx, 0, data.shape[0] - block_size - 1)
    traj_start_idx = traj_start_idx.numpy()

    batch_idx = np.arange(block_size + 1)[None, :] + traj_start_idx[:, None]

    # 窗口有没有一直取到这个病人的**最后一行**。只给访视大小标签用：决定最后一段同龄 run
    # 是"完整的一次访视"还是"被右边界截断的一段"。注意 `x` 此刻还是 (B, 2) 的 p2i 行
    # (start, n)，函数末尾才被**复用**成输入 token —— 所以这一行必须留在这里。
    reached_end = torch.from_numpy(
        batch_idx[:, -1] >= (x[:, 0] + x[:, 1] - 1).numpy()) if return_visit_size else None

    # 可选：和 data 逐行对齐的辅助标签。它必须跟着下面的**每一步**走（同一个 mask、
    # 注入的 no-event 行、同一个排序 permutation、同样的两次裁剪、同样的 +1 位移对齐）。
    # 漏掉任何一步都是静默的：BCE 照样收敛，只是标签接到了别人的位置上。
    aux_b = None
    if aux is not None:
        aux_b = torch.from_numpy(np.asarray(aux)[batch_idx])   # (B, block_size+1, n_targets)

    mask = torch.from_numpy(data[:, 0][batch_idx].astype(np.int64))
    mask = mask == torch.tensor(data[p2i[ix.numpy()][:, 0], 0][:, None].astype(np.int64)).to(mask.dtype)

    tokens = torch.from_numpy(data[:, 2][batch_idx].astype(np.int64))
    ages = torch.from_numpy(data[:, 1][batch_idx].astype(np.float32))

    tokens = tokens.masked_fill(~mask, -1)
    ages = ages.masked_fill(~mask, mask_time)

    if aux_b is not None:
        # 和 tokens/ages 用**同一个** mask：窗口越过病人边界的那些行，标签是别人的。
        aux_b = aux_b.masked_fill(~mask.unsqueeze(-1), -1)

    # insert a "no event" token every 5 years on average
    if (padding.lower() == 'none' or
            padding is None or
            no_event_token_rate == 0 or
            no_event_token_rate is None):
        pad = torch.ones(len(ix), 0)
    elif padding == 'regular':
        pad = torch.arange(0, 36525, 365.25 * no_event_token_rate) * torch.ones(len(ix), 1) + 1
    elif padding == 'random':
        pad = torch.randint(1, 36525, (len(ix), int(100 / no_event_token_rate)), generator=gen)
    else:
        raise NotImplementedError

    m = ages.max(1, keepdim=True).values

    # augment lifestyle tokens to avoid immortality bias
    # NOTE: must happen AFTER computing m, otherwise augmented ages inflate m
    # and let too many no-event padding tokens survive, hurting calibration
    if lifestyle_augmentations:
        lo, hi = lifestyle_token_range
        lifestyle_idx = (tokens >= lo) * (tokens <= hi)
        if lifestyle_idx.sum():
            ages[lifestyle_idx] += torch.randint(-20*365, 365*40, (lifestyle_idx.sum(),), generator=gen).float()
        if aux_b is not None:
            # **被抖动过的行不能再带标签。** 标签是"从这一行**真实**的年龄往后 h 年"算的，
            # 抖动把这个 token 挪到了 ±20/40 年外的一个假年龄上，两者不再是同一个时刻。
            # 不剔的代价很具体：ROSMAP 的背景块有 ~19 个 token（占 nodedup 序列的 ~40%），
            # 它们会带着基线时刻的标签坐在随机年龄上监督判别头，而且不会报任何错。
            # 代价（丢掉的监督点）很小：这些位置的标签本来就和基线那次访视重复。
            aux_b = aux_b.masked_fill(lifestyle_idx.unsqueeze(-1), -1)

    # stack "no event" tokens with real tokens
    tokens = torch.hstack([tokens, torch.zeros_like(pad, dtype=torch.int)])
    ages = torch.hstack([ages, pad])

    if aux_b is not None:
        # 注入的 no-event 行一律 -1（未知）。它们的年龄是 get_batch 现编的（'regular' 是
        # 5 年一格的绝对网格），要给它算"往后 h 年会不会发生"得在这里重跑一遍生存口径，
        # 而那份口径在 make_aux_labels.py 里。填 -1 而不是 0：把删失当阴性正是
        # labels_at_h 存在的理由（figure2_eval/README.md 9.5 第 (1) 条）。
        # 代价是 no-event 位置不参与监督，按 no_event_token_rate=5 大约是 20/144 的位置。
        aux_b = torch.cat(
            [aux_b, torch.full((aux_b.shape[0], pad.shape[1], aux_b.shape[2]), -1,
                               dtype=aux_b.dtype)], dim=1)

    # mask out "no event" tokens that are too far in the future (i.e. after the last real token)
    # NOTE: 条件必须先算出来存下。下面第二行会把 ages 覆盖成 mask_time，届时 `ages > m`
    # 恒假 —— 原来的两行靠"表达式在赋值前求值"侥幸成立，加进第三个使用者就不能再靠它了。
    over_m = ages > m
    tokens = tokens.masked_fill(over_m, -1)
    if aux_b is not None:
        aux_b = aux_b.masked_fill(over_m.unsqueeze(-1), -1)
    ages = ages.masked_fill(over_m, mask_time)

    # sort everything so that things are correctly ordered about stacking
    # NOTE: stable=True matters for data on a coarse time grid. ROSMAP puts every token of a
    # visit on the same age, so ages are heavily tied; a non-stable argsort orders tied tokens
    # differently across platforms/torch versions, which silently changes which token `pred_idx`
    # picks in evaluate_auc. Stable sort keeps the emission order from the .bin file.
    s = torch.argsort(ages, stable=True, dim=1)
    tokens = torch.gather(tokens, 1, s)
    ages = torch.gather(ages, 1, s)

    if aux_b is not None:
        # **同一个** permutation，不能重新 argsort 一次：一次访视的所有 token 同龄，
        # 并列极多（83.81% 的 Δt 为 0），stable 只保证同一次调用内的顺序稳定。
        aux_b = torch.gather(aux_b, 1, s.unsqueeze(-1).expand(-1, -1, aux_b.shape[2]))

    # a technical detail: the token 0 is reserved for padding, so we shift all tokens by one
    # （标签不跟着 +1：位移换的是 token id 的编号空间，不是行的位置）
    tokens = tokens + 1

    # cut the padded tokens if possible
    if cut_batch:
        cut_margin = torch.min(torch.sum(tokens == 0, 1))
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]
        if aux_b is not None:
            aux_b = aux_b[:, cut_margin:]

    # cut to maintain the block size
    #TODO it would be better to use the strategy defined by the "select" parameter
    if tokens.shape[1] > block_size + 1:
        cut_margin = tokens.shape[1] - block_size - 1
        tokens = tokens[:, cut_margin:]
        ages = ages[:, cut_margin:]
        if aux_b is not None:
            aux_b = aux_b[:, cut_margin:]

    # 访视大小必须在**两次裁剪之后**算：run 的长度是窗口内的长度，先算再裁会把裁掉的
    # token 也数进去。reached_end 说的是右边界，两次裁剪都只从左边切，所以它仍然成立。
    ksz = _visit_size_targets(
        ages, tokens, reached_end,
        jitter_range=(lifestyle_token_range if lifestyle_augmentations else None)
    ) if return_visit_size else None

    # shift by one to generate targets
    x = tokens[:, :-1]
    a = ages[:, :-1]
    y = tokens[:, 1:]
    b = ages[:, 1:]

    # 判别头接在**输入**位置的隐状态上（和 figure2_eval/direct_head.py 那个探针取的是同一个
    # 向量），所以标签取 [:, :-1] 跟着 x，不是跟着 y。错一格 = 用下一次访视的年龄起算 h 年，
    # 少算一个访视间隔的风险，AUC 会系统性偏乐观，而且不会有任何报错。
    ax = aux_b[:, :-1, :] if aux_b is not None else None

    # if the first token is a "no event" token, mask it and the corresponding target
    x = x.masked_fill((x == 0) * (y == 1), 0)
    y = y.masked_fill(x == 0, 0)
    b = b.masked_fill(x == 0, mask_time)
    if ax is not None:
        # 冗余但留着：padding 位置本来就在上面被填成 -1 了，这一行保证"x 是 padding"
        # 和"标签未知"这两件事在任何路径下都同时成立。
        ax = ax.masked_fill((x == 0).unsqueeze(-1), -1)

    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, a, y, b = [i.pin_memory().to(device, non_blocking=True) for i in [x, a, y, b]]
        if ax is not None:
            ax = ax.pin_memory().to(device, non_blocking=True)
        if ksz is not None:
            ksz = ksz.pin_memory().to(device, non_blocking=True)
    else:
        x, a, y, b = x.to(device), a.to(device), y.to(device), b.to(device)
        if ax is not None:
            ax = ax.to(device)
        if ksz is not None:
            ksz = ksz.to(device)
    if aux is None and not return_visit_size:
        return x, a, y, b
    return x, a, y, b, ax, ksz


def shap_custom_tokenizer(s, return_offsets_mapping=True):
    """Custom tokenizers conform to a subset of the transformers API."""
    pos = 0
    offset_ranges = []
    input_ids = []
    for m in re.finditer(r"\W", s):
        start, end = m.span(0)
        offset_ranges.append((pos, start))
        input_ids.append(s[pos:start])
        pos = end
    if pos != len(s):
        offset_ranges.append((pos, len(s)))
        input_ids.append(s[pos:])
    out = {}
    out["input_ids"] = input_ids
    if return_offsets_mapping:
        out["offset_mapping"] = offset_ranges
    return out


def shap_model_creator(model, disease_ids, person_tokens_ids, person_ages, device):
    """
    Creates a pseudo model that returns only logits for specified tokens.
    Needed for SHAP values, otherwise the SHAP visualisation is too huge.
    """
    def f(ps):
        xs = []
        as_ = []

        for p in ps:
            if len(p) == 0:
                print('No tokens found??')
                raise
            p = list(map(int, p))
            new_tokens = []
            new_ages = []
            for num, (masked, value, age) in enumerate(zip(p, person_tokens_ids, person_ages)):
                if num == 0:
                    new_ages.append(age)
                    if masked == 10000:
                        new_tokens.append(2 if value == 3 else 3)
                    else:
                        new_tokens.append(value)
                else:
                    if masked != 10000 or value == 1:
                        new_ages.append(age)
                        new_tokens.append(value)

            x = (torch.tensor(new_tokens, device=device)[None, ...])
            a = (torch.tensor(new_ages, device=device)[None, ...])

            xs.append(x)
            as_.append(a)

        max_length = max([x.shape[-1] for x in xs])

        xs = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=0) for x in xs]
        as_ = [torch.nn.functional.pad(x, (max_length - x.shape[-1], 0), value=-10000) for x in as_]

        x = torch.cat(xs)
        a = torch.cat(as_)

        with torch.no_grad():
            probs = model(x, a)[0][:, -1, disease_ids].detach().cpu().numpy()
        return probs

    return f
