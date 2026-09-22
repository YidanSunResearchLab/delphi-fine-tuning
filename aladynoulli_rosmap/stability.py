# -*- coding: utf-8 -*-
"""signature 稳不稳？—— 这是决定后面所有工作值不值得做的那个问题。

论文在 40 万人上分 40 个子集重训，报 phi 的跨子集相关 r = 0.995。ROSMAP 只有 4,428 人，
signature 很可能根本不可识别。这个脚本用 10 折的 checkpoint 回答：

  1. 把每一折的 psi (K,D) 和参照折做匈牙利匹配（signature 的编号本来就是任意的，
     不匹配直接比是没有意义的）。匹配用的是 psi 行之间的相关系数。
  2. 报匹配后每个 signature 的跨折相关分布。
  3. 报"组成保留度" (composition preservation)：参照折里某 signature 的 top-m 个病，
     有多大比例还留在匹配折的 top-m 里。论文在跨队列上报的中位数是 0.80。

    python3 stability.py --k 8
"""
from __future__ import annotations

import argparse
import glob
import itertools
import json
import os

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load_psis(K):
    out = {}
    for f in sorted(glob.glob(os.path.join(RES, f"fold*_K{K}.pt"))):
        fold = int(os.path.basename(f).split("_")[0][4:])
        ck = torch.load(f, weights_only=False, map_location="cpu")
        out[fold] = dict(psi=ck["state"]["psi"].numpy(),
                         Gamma=ck["state"]["Gamma"].numpy(),
                         names=ck["dis_names"], cov=ck["cov_names"])
    return out


def match(a, b):
    """把 b 的 signature 重排到和 a 对齐。返回 (b 的行序, 匹配后的逐 signature 相关)。"""
    A = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-9)
    B = (b - b.mean(1, keepdims=True)) / (b.std(1, keepdims=True) + 1e-9)
    C = A @ B.T / a.shape[1]                      # (K,K) 相关矩阵
    ri, ci = linear_sum_assignment(-C)
    return ci, C[ri, ci]


def top_overlap(a, b, m=8):
    ta = set(np.argsort(-a)[:m]); tb = set(np.argsort(-b)[:m])
    return len(ta & tb) / m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--topm", type=int, default=8)
    args = ap.parse_args()

    P = load_psis(args.k)
    if len(P) < 2:
        raise SystemExit(f"results/ 下 K={args.k} 的 fold checkpoint 少于 2 个")
    folds = sorted(P)
    print(f"K={args.k}  折数={len(folds)}  {folds}")

    # 所有折两两比较，避免"参照折碰巧很好/很差"带来的偏差
    cors, overs = [], []
    for i, j in itertools.combinations(folds, 2):
        a, b = P[i]["psi"], P[j]["psi"]
        ci, c = match(a, b)
        cors.append(c)
        overs.append([top_overlap(a[k], b[ci[k]], args.topm) for k in range(a.shape[0])])
    cors = np.array(cors); overs = np.array(overs)

    print(f"\n== psi 的跨折相关（匈牙利匹配后，{len(cors)} 个折对 x {args.k} 个 signature）==")
    print(f"  中位 {np.median(cors):.3f}   IQR [{np.percentile(cors,25):.3f}, "
          f"{np.percentile(cors,75):.3f}]   min {cors.min():.3f}")
    print(f"  论文在 UKB 40 个子集上报的是 0.995（n=400k）")
    print(f"\n== top-{args.topm} 疾病组成保留度 ==")
    print(f"  中位 {np.median(overs):.3f}   IQR [{np.percentile(overs,25):.3f}, "
          f"{np.percentile(overs,75):.3f}]")
    print(f"  论文跨队列 (UKB vs MGB/AoU) 报的中位数是 0.80")

    # 每个 signature 各自稳不稳（按参照折 0 对齐后看）
    ref = folds[0]
    per = np.zeros((args.k, 0)).tolist()
    per = [[] for _ in range(args.k)]
    for j in folds[1:]:
        ci, c = match(P[ref]["psi"], P[j]["psi"])
        for k in range(args.k):
            per[k].append(c[k])
    print(f"\n== 逐 signature（对齐到 fold{ref}）==")
    names = list(P[ref]["names"])
    for k in range(args.k):
        top = ", ".join(names[i] for i in np.argsort(-P[ref]["psi"][k])[:5])
        print(f"  sig{k}  r_median={np.median(per[k]):+.3f}  top5: {top}")

    json.dump(dict(k=args.k, n_folds=len(folds),
                   psi_cor_median=float(np.median(cors)),
                   psi_cor_iqr=[float(np.percentile(cors, 25)), float(np.percentile(cors, 75))],
                   psi_cor_min=float(cors.min()),
                   top_overlap_median=float(np.median(overs)),
                   per_signature_r=[float(np.median(v)) for v in per]),
              open(os.path.join(RES, f"stability_K{args.k}.json"), "w"), indent=2)
    print(f"\n-> {os.path.join(RES, f'stability_K{args.k}.json')}")


if __name__ == "__main__":
    main()
