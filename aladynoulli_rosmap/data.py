# -*- coding: utf-8 -*-
"""ROSMAP 的 Delphi .bin  ->  ALADYNOULLI 要的生存张量。

ALADYNOULLI (Urbut et al., Nature 2026) 吃的是一个稠密的 病人 x 疾病 x 年龄 网格上的
离散时间生存数据。Delphi 的 .bin 是 (projid, age_days, token_preshift) 的事件流。
这个文件做的是两者之间的转换，外加把 ALADYNOULLI 没遇到、ROSMAP 特有的三件事处理掉。

== 三个和论文不一样的地方，都是数据逼出来的 ==

1. 左截断 (LEFT TRUNCATION)。UKB 的人从 30 岁起就有 EHR，论文直接令 E_id = min(诊断, 删失) - 30，
   即所有人从 t=0 起同时进入风险集。ROSMAP 是队列研究，入组年龄中位 78.9 岁（p1=60.2），
   入组前没有观测。所以每个人的风险窗口从他自己的 idx0+1 开始，不是从网格 0 开始。
   不这么做的话，"80 岁入组的人在 65 岁没得 AD" 会被当成真实的阴性证据，而那一段根本没观测。

2. 基线患病 = 协变量，不是事件。tokenization 对每个 subject 做了全局去重（每个 token 一生最多
   出现一次），所以"首次出现"对增量事件（AD_DX 0% 在基线）是对的，对状态型 token 是错的：
   MMSE_normal 89%、ANTIHYP_ON 68%、BMI_high 75% 的首次出现就在基线访视那一天——那是入组时
   的横断面状态，不是新发。基线就带着 token d 的人直接从 d 的风险集里移出（他已经有了），
   同时把 d 写进他的基线协变量。这样留在风险集里的才是真正的转变（MMSE_normal 在基线之后
   首次出现 = 认知恢复正常，是个有意义的事件）。

3. 炎症标志物默认剔除。CRP/IL6/TNFA 的 9 个 token 只有 142/4428 = 3.2% 的人测过。没测过的人
   在生存模型里长得和"一直没发生"完全一样，但那是 missing 不是 negative。留着不会崩，只会
   让模型学出一个接近 0 的 hazard 并浪费掉谱聚类初始化的自由度。--keep-inflammation 可以留。

== 时间网格 ==

.bin 的时间轴是 age_days = round(age_bl*365.25) + k*365，k 是距基线的整年数。所以
(age_days - baseline_days) / 365 是精确整数。个人的网格下标 = idx0 + k，idx0 由基线年龄
四舍五入到整岁得到。这样保证同一个人内部的年份间隔完全无损，只有 idx0 有 ±0.5 岁的量化。

输出 (npz):
    E      (N, D) int16   事件/删失所在的网格下标
    Yobs   (N, D) int8    1=在 E 处观测到事件, 0=在 E 处删失
    atrisk (N, D) int8    0=基线即患病, 该人不进入 d 的风险集
    S      (N,)   int16   风险窗口起点下标 (= idx0 + 1)
    G      (N, P) float32 基线协变量 (含 APOE)
    split  (N,)   int8    0=train, 1=val   (沿用 Delphi 的划分)
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# RIS 上分词数据同步在 _tokenization/ 下；本地开发时在上一级的 tokenization/。
TOKDIR = os.path.normpath(os.path.join(HERE, "_tokenization", "data_rosmap"))
if not os.path.isdir(TOKDIR):
    TOKDIR = os.path.normpath(os.path.join(HERE, "..", "tokenization", "data_rosmap"))

# 网格覆盖 60-105 岁。基线 p1=60.2、末次观测 p95=98.5、max=109.1。
AGE_MIN, AGE_MAX = 60, 105

INFLAMMATION = ("CRP_", "IL6_", "TNFA_")


def _load_bin(name):
    a = np.fromfile(os.path.join(TOKDIR, f"{name}.bin"), dtype=np.uint32).reshape(-1, 3)
    return a


def build(keep_inflammation=False, verbose=True):
    meta = json.load(open(os.path.join(TOKDIR, "meta.json")))
    vocab = meta["vocab"]                       # vocab[t-1] == preshift token t 的名字
    bg_lo, bg_hi = meta["bg_range_preshift"]    # 3..32  背景块
    death_tok = meta["death_token_preshift"]    # 93
    path_lo, _ = meta["path_range_preshift"]    # 94..127 死后病理

    rows, split = [], []
    for si, name in enumerate(("train", "val")):
        a = _load_bin(name)
        rows.append(a)
        split.append(np.full(len(a), si, np.int8))
    arr = np.vstack(rows)
    rowsplit = np.concatenate(split)

    # ---- 疾病集合 D: 33..93 (事件 / 用药 / 纵向分箱 / Death)，病理 94+ 永远排除 ----
    dis_tok = [t for t in range(bg_lo + 30, death_tok + 1)]      # 33..93
    assert vocab[dis_tok[0] - 1] == "AD_DX" and vocab[dis_tok[-1] - 1] == "Death"
    if not keep_inflammation:
        dis_tok = [t for t in dis_tok
                   if not vocab[t - 1].startswith(INFLAMMATION)]
    D = len(dis_tok)
    d_of = {t: j for j, t in enumerate(dis_tok)}
    dis_names = [vocab[t - 1] for t in dis_tok]

    # ---- 协变量集合 P: 性别(1..2) + 背景块(3..32) ----
    cov_tok = list(range(1, bg_hi + 1))
    p_of = {t: j for j, t in enumerate(cov_tok)}
    cov_names = [vocab[t - 1] for t in cov_tok]

    # ---- 按人聚合 ----
    pids, inv = np.unique(arr[:, 0], return_inverse=True)
    N, T = len(pids), AGE_MAX - AGE_MIN + 1

    bl_days = np.full(N, 1 << 30, np.int64)
    np.minimum.at(bl_days, inv, arr[:, 1].astype(np.int64))
    idx0 = np.rint(bl_days / 365.25).astype(np.int64) - AGE_MIN      # 基线的网格下标

    # 每条记录相对基线的整年偏移
    off = np.rint((arr[:, 1].astype(np.int64) - bl_days[inv]) / 365.0).astype(np.int64)
    gidx = idx0[inv] + off                                           # 记录的网格下标

    last_off = np.zeros(N, np.int64)
    np.maximum.at(last_off, inv, off)
    S = idx0 + 1                                                     # 风险窗口起点
    E_cens = idx0 + last_off                                         # 删失下标

    # ---- 协变量 ----
    G = np.zeros((N, len(cov_tok)), np.float32)
    m = np.isin(arr[:, 2], cov_tok)
    G[inv[m], [p_of[t] for t in arr[m, 2]]] = 1.0

    # ---- 事件 ----
    E = np.repeat(E_cens[:, None], D, 1).astype(np.int64)
    Yobs = np.zeros((N, D), np.int8)
    atrisk = np.ones((N, D), np.int8)

    m = np.isin(arr[:, 2], dis_tok)
    ii, dd, gg, oo = inv[m], np.array([d_of[t] for t in arr[m, 2]]), gidx[m], off[m]
    prev = oo == 0                                                   # 基线即带着 = 患病
    atrisk[ii[prev], dd[prev]] = 0
    inc = ~prev
    E[ii[inc], dd[inc]] = gg[inc]
    Yobs[ii[inc], dd[inc]] = 1
    # 基线患病的 token 进协变量（在 G 后面追加，和背景块拼起来）
    Gprev = 1 - atrisk.astype(np.float32)

    # ---- 网格边界裁剪 ----
    # 少数人末次观测 > 105 岁，把 E 夹到网格内；夹掉的事件（很少）当删失处理。
    over = E > (T - 1)
    Yobs[over] = 0
    E = np.clip(E, 0, T - 1)
    S = np.clip(S, 0, T - 1)
    # 基线 < 60 岁的人（少数），起点提到 0
    S = np.maximum(S, 0)
    valid = (E >= S[:, None])                                        # 风险窗口非空
    atrisk = (atrisk.astype(bool) & valid).astype(np.int8)

    subj_split = np.zeros(N, np.int8)
    subj_split[inv] = rowsplit                                       # 每个人只属于一个 split

    out = dict(
        E=E.astype(np.int16), Yobs=Yobs, atrisk=atrisk, S=S.astype(np.int16),
        G=np.hstack([G, Gprev]).astype(np.float32),
        split=subj_split, pids=pids.astype(np.int64),
        dis_names=np.array(dis_names), cov_names=np.array(cov_names + [f"PREV_{n}" for n in dis_names]),
        age_grid=np.arange(AGE_MIN, AGE_MAX + 1),
    )

    if verbose:
        n_ev = Yobs.sum(0)
        print(f"N={N}  D={D}  T={T}  (ages {AGE_MIN}-{AGE_MAX})")
        print(f"train/val = {(subj_split==0).sum()}/{(subj_split==1).sum()}")
        print(f"idx0: min={idx0.min()} median={int(np.median(idx0))} max={idx0.max()}  "
              f"(基线<{AGE_MIN}岁的人: {(idx0<0).sum()})")
        print(f"随访年数: median={int(np.median(last_off))} p95={int(np.percentile(last_off,95))} max={last_off.max()}")
        print(f"风险集单元总数 (sum over i,d of E-S+1) = "
              f"{int(((E - S[:,None] + 1) * atrisk).sum()):,}")
        print(f"\n{'disease':<18}{'atrisk':>8}{'events':>8}{'rate':>8}")
        for j in np.argsort(-n_ev):
            ar = atrisk[:, j].sum()
            print(f"{dis_names[j]:<18}{ar:>8}{n_ev[j]:>8}{n_ev[j]/max(ar,1):>8.3f}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-inflammation", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "tensors.npz"))
    args = ap.parse_args()
    d = build(keep_inflammation=args.keep_inflammation)
    np.savez_compressed(args.out, **d)
    print(f"\n-> {args.out}")
