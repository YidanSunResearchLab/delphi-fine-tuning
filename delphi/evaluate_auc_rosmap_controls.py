"""Figure 3 panel A/B 的 AUC：上游 Delphi 口径 + transformer 基线 + 首次出现口径。

    python evaluate_auc_rosmap_controls.py \
        --run nodedup-pos-3w:42:Delphi-ROSMAP-nodedup-pos-3w/ckpt.pt:rosmap_nodedup_3w \
        --run dedup:42:Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
        --out results/fig3

`--run config:seed:ckpt:dataset` 可以给多次；同一个 config 的多个 seed 在 panel B 里算训练噪声。

**上游的核心一行都没动。** 病例/对照构造、prediction index、年龄段内每人抽一条、DeLong
全部来自 `evaluate_auc.get_calibration_auc` 的原样调用。这里只在它外面做三件事：

  1. 换打分矩阵 `p`（--scores）。同一份 `d`、同一个抽样种子喂给每一种打分，所以比的是
     **完全相同的那批观测**，差值只来自打分。所有打分都来自 transformer 本身，不用任何外部模型：
       model   训练好的 ckpt 的 logits（和上游一样存成 float16）
       iter0   同一配置、随机初始化、不加载权重。只作整体健全性检查（中位应 ≈ 0.5）；逐 token
               会偏离 0.5 不少，因为位置/时间结构本身就和病例/对照的预测点相关
       trunc1  同一个 ckpt，但每个预测点只喂"背景块 + 最近 1 次访视"
       trunc2  同上，最近 2 次访视
     trunc 是**重建输入序列**再前向，不是改 mask：mask 版的滑动窗口会跨层泄漏（见 model.py
     的 _scope_mask），而且 pos_embedding 的位置下标会泄漏历史长度。重建后对 ckpt 来说是分布外
     输入（训练时每个窗口都从轨迹开头开始），所以只作辅助证据。
     训练出来的 transformer 基线（只看现在 / 只看人口学）不是打分，是单独的 --run，
     打分用 model。见 config/train_delphi_rosmap_fig3_*.py。

  2. --first_occurrence：每人第 2 次及以后出现的目标 token 换成哨兵 REPEAT。这个人仍因首次
     出现被排除在对照外，对照定义不变。在 dedup 数据上这是恒等变换（全局首次出现去重已保证，
     val 上实测 0 个重复）；在 nodedup 上只影响 MMSE/COGN 那 6 个 token，把它们拉回同一道题。

  3. 抽样种子按 (eval_seed, token, sex) 逐个设。上游是开头设一次、一路消耗，结果依赖 token
     列表的顺序；逐个设以后，不同打分、不同 token 子集之间抽到的观测一致。--eval_seeds 个种子
     取平均，压掉"每人抽一条"带来的抽样噪声。seeding="global" 可以逐位复现上游
     （见 test_auc_controls_equiv.py）。

token 集合：train 上**首次出现人数** > --min_persons（在去重数据上等于上游的 count>100），
固定从 --token_ref 数据集算一次，所以所有 run 评的是同一组 token。
  event     Events + Death（AD_DX / *_ONSET / DEPRESSION / STROKE / Death）
  med_start 用药 *_ON
  state     Clinical measures（量表/化验的箱）
排除 *_OFF（OFF 必须先有 ON，对照组大多从没 ON 过，是语法送分）、神经病理（死后才发射）、
背景/性别/技术 token。
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from model import Delphi, DelphiConfig
from utils import get_batch, get_p2i
from evaluate_auc import get_calibration_auc
from evaluate_auc_rosmap import ROSMAP_AGE_GROUPS, chapter_of

SCORES = ("model", "iter0", "trunc1", "trunc2", "query")
SEXES = (("female", 2), ("male", 3))       # post-shift id，和 evaluate_auc_pipeline 一致
PADDING, NO_EVENT = 0, 1                   # post-shift
REPEAT = -2                                # 首次出现口径下"又一次出现"的哨兵，不等于任何 token
GROUP_OF_CHAPTER = {"Events": "event", "Death": "event",
                    "Medications": "med_start", "Clinical measures": "state"}
STATIC_MAX = 33                            # 背景块（含性别）= post-shift [2, 33]，同 ignore_tokens


# ----------------------------------------------------------------------------- 词表

def load_names(data_dir):
    return pd.read_csv(os.path.join(data_dir, "labels.csv"))["event_name"].tolist()


def select_tokens(ref_dir, min_persons):
    names = load_names(ref_dir)
    train = np.fromfile(f"{ref_dir}/train.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    first = np.unique(train[:, [0, 2]], axis=0)             # (person, pre-shift token) 去重
    ids, n_persons = np.unique(first[:, 1] + 1, return_counts=True)
    persons = dict(zip(ids.tolist(), n_persons.tolist()))
    rows = []
    for k, n in enumerate(names):
        ch = chapter_of(n)
        if ch not in GROUP_OF_CHAPTER or n.endswith("_OFF"):
            continue
        if persons.get(k, 0) <= min_persons:
            continue
        rows.append({"token": k, "name": n, "chapter": ch, "group": GROUP_OF_CHAPTER[ch],
                     "train_persons": persons[k]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- 数据

def load_val_batch(data_dir, block_size):
    val = np.fromfile(f"{data_dir}/val.bin", dtype=np.uint32).reshape(-1, 3).astype(np.int64)
    p2i = get_p2i(val)
    # 与 evaluate_auc_rosmap.py 逐参数相同
    d = get_batch(range(len(p2i)), val, p2i, select="left", block_size=block_size,
                  device="cpu", padding="random", no_event_token_rate=5)
    persons = val[p2i[:, 0], 0]
    return [t.numpy() for t in d], persons


def first_occurrence(y):
    """每人第 2 次及以后出现的目标 token -> REPEAT。padding / no-event 不动。"""
    y = y.copy()
    for b in range(y.shape[0]):
        seen = set()
        for i in range(y.shape[1]):
            k = int(y[b, i])
            if k <= NO_EVENT:
                continue
            if k in seen:
                y[b, i] = REPEAT
            else:
                seen.add(k)
    return y


def drop_pre_entry(d, offset, entry_offset=None):
    """预测点落在第一次真实访视之前的目标 -> 目标年龄设成 mask 值，上游算出 pred_idx=-1 后丢掉。

    上游 get_batch(padding="random") 会在入组前插 no-event token。在 UKB 里记录从出生附近开始，
    这些点是"健康期"；在 ROSMAP 里入组年龄中位数约 75 岁，入组前的点上只知道年龄。
    实测（交付版 ckpt，val，nogap）：去掉这些点后**按组方向相反**，全体中位几乎不动所以容易看漏：
      事件 0.764 -> 0.636（对照在入组前的点是天然易判阴性，抬高事件 AUC）
      用药起始 0.503 -> 0.614、状态 0.661 -> 0.709（入组时已在用药/已在某箱的"病例"，
      预测点落在什么都没看到的入组前，把 AUC 拉向 0.5 以下）
    根子：Delphi 为 UKB 设计，记录从出生附近开始；ROSMAP 从 ~75 岁入组，入组时已有的状态
    全被记成"首次出现"。默认仍保持上游原样以便对照，但报 ROSMAP 的数应当同时给这个口径。
    只改目标年龄、不改输入，模型打分不受影响。
    预测点早于入组 <=> 目标时间 - offset <= 入组时间。"""
    # entry_offset（默认 = offset）：用另一个 offset 的条件来筛目标。给 nogap 传 365.25，
    # nogap 和 gap365 就评的是**同一批**病例/对照，两者的 AUC 差才能读成"提前多久"的效应。
    x, a, y, b = d
    real = x > NO_EVENT
    entry = np.where(real, a, np.inf).min(1, keepdims=True)
    b = b.copy()
    eo = offset if entry_offset is None else entry_offset
    b[(b - eo <= entry) & (y > PADDING)] = -10000.0
    return [x, a, y, b]


def filter_pred_type(d, offset, kind):
    """只保留预测点是某一类位置的目标（病例和对照同样处理），其余目标年龄设成 mask 值。

    kind = "real"：预测点是真实 token（访视末尾）；"noevent"：预测点是注入的 no-event token。
    同一次访视之后，两类位置上模型掌握的信息相同，区别只在"下一个 token 是什么"：访视末尾的
    下一个是下一次访视按固定顺序发射的第一个 token，no-event 位置没有这种确定性。所以两者的 AUC
    差可以直接读成"读分位置"的效应。只改目标年龄，不改输入。"""
    if kind == "all":
        return d
    x, a, y, b = d
    pred_idx = (a[:, :, None] < b[:, None, :] - offset).sum(1) - 1
    tok_at_pred = np.take_along_axis(x, np.clip(pred_idx, 0, None), axis=1)
    want = tok_at_pred > NO_EVENT if kind == "real" else tok_at_pred == NO_EVENT
    b = b.copy()
    b[~want & (y > PADDING)] = -10000.0
    return [x, a, y, b]


# ----------------------------------------------------------------------------- 打分

def load_model(ckpt_path, device, random_init_seed=None):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    if random_init_seed is not None:
        torch.manual_seed(random_init_seed)
    model = Delphi(DelphiConfig(**ck["model_args"]))
    if random_init_seed is None:
        model.load_state_dict(ck["model"])
    return model.eval().to(device), ck


def model_scores(model, d, tokens, device, batch_size=128):
    out = []
    with torch.no_grad():
        for chunk in zip(*[torch.split(torch.from_numpy(x), batch_size) for x in d]):
            logits = model(*[c.to(device) for c in chunk])[0].cpu().numpy()
            out.append(logits[:, :, tokens].astype("float16"))    # 与上游同精度
    return np.vstack(out)


def visit_ids(x_row, a_row):
    """与 model.py 的 _scope_mask 同一个定义：非背景真实 token 的年龄每变一次 +1。"""
    vid = np.zeros(len(x_row), dtype=np.int64)
    cur, last_age = 0, None
    for i, (k, t) in enumerate(zip(x_row, a_row)):
        if k > STATIC_MAX and t != last_age:
            cur += 1
            last_age = t
        vid[i] = cur
    return vid


def truncated_scores(model, d, tokens, k, device, batch_size=512):
    """每个预测点只喂"背景块 + 最近 k 次访视"，重建输入序列后前向，取最后一位的 logits。

    上游的 pred_idx 是"最后一个早于 target-offset 的位置"，同龄 token 共享年龄，所以预测点
    一定是某段同龄 run 的最后一位（访视末尾或 no-event）。只对这些位置打分，其余位置填 NaN，
    并在 auc 之前断言用到的预测点都有分数。"""
    x, a, y, b = d
    B, L = x.shape
    p = np.full((B, L, len(tokens)), np.nan, dtype=np.float32)
    jobs = []
    for r in range(B):
        valid = x[r] != PADDING
        is_static = (x[r] >= 2) & (x[r] <= STATIC_MAX)
        vid = visit_ids(x[r], a[r])
        for i in np.where(valid)[0]:
            if i < L - 1 and valid[i + 1] and a[r, i + 1] == a[r, i]:
                continue                                         # 不是同龄 run 的最后一位
            keep = np.where(valid[: i + 1] & (is_static[: i + 1] | (vid[: i + 1] >= vid[i] - k + 1)))[0]
            xs, as_ = x[r, keep], a[r, keep]
            ys = np.r_[xs[1:], y[r, i]]
            bs = np.r_[as_[1:], b[r, i]]
            jobs.append((r, i, xs, as_, ys, bs))
    with torch.no_grad():
        for s0 in range(0, len(jobs), batch_size):
            chunk = jobs[s0: s0 + batch_size]
            T = max(len(j[2]) for j in chunk)
            X = np.zeros((len(chunk), T), dtype=np.int64); Y = np.zeros_like(X)
            A = np.full((len(chunk), T), -10000.0, dtype=np.float32); Bt = A.copy()
            for n, (_, _, xs, as_, ys, bs) in enumerate(chunk):          # 左侧 padding，同 get_batch
                X[n, T - len(xs):], A[n, T - len(xs):] = xs, as_
                Y[n, T - len(xs):], Bt[n, T - len(xs):] = ys, bs
            out = model(*[torch.from_numpy(v).to(device) for v in (X, A, Y, Bt)])[0][:, -1]
            out = out[:, tokens].cpu().numpy()
            for n, (r, i, *_) in enumerate(chunk):
                p[r, i] = out[n]
    # 目标在入组第一年内时，上游的预测点会落在左侧 padding 上；那些观测的年龄是 -10000，会被
    # 年龄分层丢掉，但分数仍会被取出来，所以填一个有限值而不是 NaN。
    p[x == PADDING] = 0.0
    return p


QUERY_DT = 1.0      # 查询 token 放在预测点之后 1 天：不和任何访视同龄，也远早于下一次访视


def query_scores(model, d, tokens, device, batch_size=256):
    """在每个预测点后面接一个 no-event 查询 token，读查询 token 上的 logits。

    为什么：在每次访视都重复发射的数据里，访视末尾的"下一个 token"几乎确定是下一次访视按固定
    顺序发射的第一个，那个位置上其它 token 的 logit 被压低，Delphi 的读分反映的是发射顺序而不是
    风险（实测：同一批目标、同样的信息，no-event 位置比访视末尾高 0.13–0.22，dedup/人口学 ≈ 0）。
    查询 token 看到的信息与预测点完全相同，只是"下一个 token"不再被顺序决定。

    输入是原行截到预测点再接查询 token，**右侧** padding：原有 token 的位置下标不变
    （pos_embedding 的模型要紧），causal mask 保证右侧 padding 不影响查询位置。
    只对同龄 run 的最后一位打分（上游的预测点只会落在这里），padding 位置填 0。"""
    x, a, y, b = d
    B, L = x.shape
    p = np.full((B, L, len(tokens)), np.nan, dtype=np.float32)
    # pos_embedding 的位置表只有 block_size 行，而 get_batch 左侧补 padding，每个人的最后一个
    # token 都在窗口末格 —— 预测点在末格时（Death 常见）查询 token 放不下。做法：丢掉最左边
    # 那一格（必须是 padding，否则报错），整行左移一位。等价于"少一格 padding 的同一条序列"，
    # 训练里不同长度的序列本来就对应不同的 padding 量，所以仍在分布内。
    limit = model.transformer.wpe.num_embeddings if model.config.pos_embedding else 10**9
    jobs = []
    n_shift = 0
    for r in range(B):
        valid = x[r] != PADDING
        for i in np.where(valid)[0]:
            if i < L - 1 and valid[i + 1] and a[r, i + 1] == a[r, i]:
                continue
            sh = max(0, i + 2 - limit)
            assert (x[r, :sh] == PADDING).all(), f"行 {r} 左移 {sh} 格会丢掉真实 token"
            n_shift += sh > 0
            jobs.append((r, i, sh))
    if n_shift:
        print(f"  query: {n_shift} / {len(jobs)} 个预测点在窗口末格，左移一格（丢的是 padding）")
    with torch.no_grad():
        for s0 in range(0, len(jobs), batch_size):
            chunk = jobs[s0: s0 + batch_size]
            T = max(i - sh for _, i, sh in chunk) + 2
            X = np.zeros((len(chunk), T), dtype=np.int64); Y = np.zeros_like(X)
            A = np.full((len(chunk), T), -10000.0, dtype=np.float32); Bt = A.copy()
            for n, (r, i, sh) in enumerate(chunk):
                qa = a[r, i] + QUERY_DT
                j = i - sh                                       # 预测点在新序列里的下标
                X[n, : j + 1], A[n, : j + 1] = x[r, sh: i + 1], a[r, sh: i + 1]
                Y[n, :j], Bt[n, :j] = y[r, sh:i], b[r, sh:i]
                Y[n, j], Bt[n, j] = NO_EVENT, qa                 # 预测点的下一个 = 查询 token
                X[n, j + 1], A[n, j + 1] = NO_EVENT, qa          # 查询 token
                Y[n, j + 1], Bt[n, j + 1] = y[r, i], b[r, i]     # 查询 token 的目标 = 原目标
            out = model(*[torch.from_numpy(v.copy()).to(device) for v in (X, A, Y, Bt)])[0]
            q = torch.tensor([i - sh + 1 for _, i, sh in chunk], device=out.device)
            out = out[torch.arange(len(chunk), device=out.device), q][:, tokens].cpu().numpy()
            for n, (r, i, _) in enumerate(chunk):
                p[r, i] = out[n]
    p[x == PADDING] = 0.0
    return p


def assert_scored(p, d, offset):
    x, a, y, b = d
    pred_idx = (a[:, :, None] < b[:, None, :] - offset).sum(1) - 1
    used = (pred_idx >= 0) & (y > PADDING) & (b > -10000)
    rows, cols = np.where(used)
    miss = np.isnan(p[rows, pred_idx[rows, cols], 0]).sum()
    assert not np.isnan(p[x == PADDING]).any()
    assert miss == 0, f"{miss} 个被上游用到的预测点没有分数"


# ----------------------------------------------------------------------------- AUC

def auc_rows(d, p, tokens, offset, eval_seed, seeding="per_token"):
    """上游 evaluate_auc_pipeline 的外层循环；核心 get_calibration_auc 原样调用。"""
    pred_idx = (d[1][:, :, None] < d[3][:, None, :] - offset).sum(1) - 1
    if seeding == "global":
        np.random.seed(eval_seed)
    rows = []
    for sex, sid in SEXES:
        m = (d[0] == sid).sum(1) > 0
        d_s = [x[m] for x in d]
        p_s, idx_s = p[m], pred_idx[m]
        for j, k in enumerate(tokens):
            if seeding == "per_token":
                np.random.seed((eval_seed * 1_000_003 + int(k) * 7 + sid) % 2**32)
            out = get_calibration_auc(j, k, d_s, p_s, offset=offset, age_groups=ROSMAP_AGE_GROUPS,
                                      precomputed_idx=idx_s, n_bootstrap=1, use_delong=True)
            for r in out or []:
                rows.append({"token": int(k), "sex": sex, "age": r["age"], "auc": r["auc"],
                             "auc_var": float(np.asarray(r["auc_variance_delong"])),
                             "n_case": r["n_diseased"], "n_ctrl": r["n_healthy"]})
    return pd.DataFrame(rows)


def pool(unpooled):
    """上游 aggregate_age_brackets_delong：各层 AUC 取平均，方差 = 和 / n²。"""
    g = unpooled.groupby("token")
    return pd.DataFrame({
        "auc": g["auc"].mean(),
        "auc_var": g["auc_var"].sum() / g.size() ** 2,
        "n_strata": g.size(),
        "n_case": g["n_case"].sum(),
        "n_ctrl": g["n_ctrl"].sum(),
    }).reset_index()


# ----------------------------------------------------------------------------- main

def parse_run(s):
    config, seed, ckpt, ds = s.split(":")
    return {"config": config, "seed": int(seed), "ckpt": ckpt, "dataset": ds}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", required=True, type=parse_run,
                    help="config:seed:ckpt:dataset，可重复")
    ap.add_argument("--data_root", default="data")
    ap.add_argument("--token_ref", default="rosmap", help="固定 token 集合用的数据集")
    ap.add_argument("--min_persons", type=int, default=100)
    ap.add_argument("--scores", default=",".join(SCORES))
    ap.add_argument("--offsets", default="0.1,365.25")
    ap.add_argument("--eval_seeds", type=int, default=10)
    ap.add_argument("--first_occurrence", type=int, default=1)
    ap.add_argument("--post_entry", type=int, default=0,
                    help="0 = 上游原样（默认）；1 = 只评入组之后做出的预测，敏感性分析（见 drop_pre_entry）")
    ap.add_argument("--pred_filter", default="all", choices=["all", "real", "noevent"],
                    help="只评预测点是真实 token / no-event token 的目标（见 filter_pred_type）")
    ap.add_argument("--entry_offset", type=float, default=None,
                    help="post_entry 的筛选用这个 offset（默认 = 各自的 offset）。365.25 = 各 offset 评同一批目标")
    ap.add_argument("--ckpt_name", default=None,
                    help="把每个 run 的 ckpt 文件名替换成它，比如 ckpt_10000.pt（固定步数，不经 val 挑选）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    scores = args.scores.split(",")
    assert set(scores) <= set(SCORES), scores
    offsets = [float(o) for o in args.offsets.split(",")]
    os.makedirs(args.out, exist_ok=True)

    tok = select_tokens(os.path.join(args.data_root, args.token_ref), args.min_persons)
    tokens = tok["token"].to_numpy()
    print(f"{len(tokens)} tokens:", tok.groupby("group")["name"].apply(list).to_dict())

    ref_names = load_names(os.path.join(args.data_root, args.token_ref))
    ref_persons = None
    all_unpooled, all_pooled = [], []

    for run in args.run:
        ckpt = run["ckpt"] if args.ckpt_name is None else os.path.join(
            os.path.dirname(run["ckpt"]), args.ckpt_name)
        data_dir = os.path.join(args.data_root, run["dataset"])
        assert load_names(data_dir) == ref_names, f"{data_dir} 的词表和 {args.token_ref} 不同"
        model, ck = load_model(ckpt, args.device)
        block_size = ck["model_args"]["block_size"]
        print(f"\n== {run['config']} s{run['seed']}: {ckpt} iter {ck.get('iter_num')} "
              f"best_val {float(ck.get('best_val_loss', float('nan'))):.4f} on {run['dataset']}")

        d, persons = load_val_batch(data_dir, block_size)
        if ref_persons is None:
            ref_persons = persons
        # panel B 的配对前提：所有 run 的 val 是同一批人、同样的顺序
        assert np.array_equal(persons, ref_persons), f"{run['dataset']} 的 val 人群和第一个 run 不同"

        # 模型的 forward 要吃原始目标（哨兵会进 loss 的索引），所以先打分、再改目标
        p_by_score = {}
        if "model" in scores:
            p_by_score["model"] = model_scores(model, d, tokens, args.device)
        if "iter0" in scores:
            m0, _ = load_model(ckpt, args.device, random_init_seed=run["seed"])
            p_by_score["iter0"] = model_scores(m0, d, tokens, args.device)
        if "query" in scores:
            p_by_score["query"] = query_scores(model, d, tokens, args.device)
        for kk in (1, 2):
            if f"trunc{kk}" in scores:
                p_by_score[f"trunc{kk}"] = truncated_scores(model, d, tokens, kk, args.device)
        if args.first_occurrence:
            d[2] = first_occurrence(d[2])

        for offset in offsets:
            for s in scores:
                if s.startswith("trunc") or s == "query":
                    assert_scored(p_by_score[s], d, offset)
            d_eval = drop_pre_entry(d, offset, args.entry_offset) if args.post_entry else d
            d_eval = filter_pred_type(d_eval, offset, args.pred_filter)
            for s in scores:
                for es in range(args.eval_seeds):
                    u = auc_rows(d_eval, p_by_score[s], tokens, offset, es)
                    meta = {"config": run["config"], "seed": run["seed"], "ckpt": ckpt,
                            "dataset": run["dataset"], "score": s, "offset": offset, "eval_seed": es,
                            "first_occurrence": args.first_occurrence, "post_entry": args.post_entry,
                            "pred_filter": args.pred_filter}
                    all_unpooled.append(u.assign(**meta))
                    all_pooled.append(pool(u).assign(**meta))
                last = all_pooled[-args.eval_seeds:]
                med = pd.concat(last).groupby("token")["auc"].mean().median()
                print(f"  offset={offset:<7} {s:9s} median AUC {med:.3f}")

    unpooled = pd.concat(all_unpooled, ignore_index=True)
    pooled = pd.concat(all_pooled, ignore_index=True).merge(tok, on="token")
    keys = ["config", "seed", "ckpt", "dataset", "score", "offset", "first_occurrence", "post_entry", "pred_filter",
            "token", "name", "group"]
    summary = pooled.groupby(keys).agg(
        auc=("auc", "mean"), auc_var=("auc_var", "mean"), auc_resample_sd=("auc", "std"),
        n_case=("n_case", "mean"), n_ctrl=("n_ctrl", "mean"), n_strata=("n_strata", "mean"),
    ).reset_index()
    summary["auc_ci95"] = 1.96 * np.sqrt(summary["auc_var"])

    unpooled.to_csv(f"{args.out}/fig3_auc_unpooled.csv", index=False)
    pooled.to_csv(f"{args.out}/fig3_auc_pooled_by_eval_seed.csv", index=False)
    summary.to_csv(f"{args.out}/fig3_auc_summary.csv", index=False)
    with open(f"{args.out}/fig3_auc_meta.json", "w") as f:
        json.dump({**{k: v for k, v in vars(args).items() if k != "run"}, "runs": args.run,
                   "tokens": tok.to_dict("records")}, f, indent=1, default=str)
    print(f"\nwrote {args.out}/fig3_auc_summary.csv ({len(summary)} rows)")


if __name__ == "__main__":
    main()
