# -*- coding: utf-8 -*-
"""逐个 signature 的体检表：它稳不稳、是什么、对不对得上 APOE 和尸检病理。

为什么不能只看"跨折相关的中位数"：匈牙利匹配是**挑最佳配对**的，所以就算两组完全随机的
ψ 拿来配，也会得到正的相关。K 越大，可挑的配对越多，这个虚高越严重。所以这里额外算一个
置换零分布：把每折 ψ 的疾病标签独立打乱（保留每个 signature 的权重分布，只破坏疾病对应
关系），再走同一套匈牙利匹配，看"纯靠挑配对"能拿到多少相关。真实的 r 必须显著高过它。

    python3 per_signature.py --k 21
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from evaluate import fold_split
from fit import load
from interpret import pathology_table
from stability import load_psis, match

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def null_r(psis, folds, n_perm, rng):
    """置换零分布：打乱疾病标签后再匈牙利匹配。"""
    out = []
    for _ in range(n_perm):
        i, j = rng.choice(folds, 2, replace=False)
        a = psis[i]["psi"][:, rng.permutation(psis[i]["psi"].shape[1])]
        b = psis[j]["psi"][:, rng.permutation(psis[j]["psi"].shape[1])]
        out.append(match(a, b)[1])
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=21)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    d = load()
    names = list(d["dis_names"]); cov = list(d["cov_names"])
    Pk = load_psis(args.k)
    folds = sorted(Pk)
    psi0 = Pk[0]["psi"]
    K = psi0.shape[0]

    # 零分布
    nul = null_r(Pk, folds, args.n_perm, rng)
    thr = float(np.percentile(nul, 95))
    print(f"置换零分布 ({args.n_perm} 对): 中位 {np.median(nul):+.3f}  "
          f"95 分位 {thr:+.3f}   <-- 真实 r 必须高过这条线才算有信号")

    # 逐 signature 跨折 r + top-m 组成保留
    r_per = [[] for _ in range(K)]
    ov_per = [[] for _ in range(K)]
    for j in folds[1:]:
        ci, c = match(psi0, Pk[j]["psi"])
        for k in range(K):
            r_per[k].append(c[k])
            ta = set(np.argsort(-psi0[k])[:8])
            tb = set(np.argsort(-Pk[j]["psi"][ci[k]])[:8])
            ov_per[k].append(len(ta & tb) / 8)

    # APOE 与病理
    ck = torch.load(os.path.join(RES, f"fold0_K{args.k}.pt"), weights_only=False, map_location="cpu")
    Gam = ck["state"]["Gamma"].numpy()
    aex = F.softmax(ck["state"]["lam"], dim=1).numpy().sum(2)          # (n_tr, K)
    fs = fold_split(len(d["split"]), 10, 0)
    tr0 = np.concatenate([fs[j] for j in range(10) if j != 0])
    P, pvars = pathology_table(d["pids"][tr0])
    br, tg = pvars.index("BRAAK"), pvars.index("TANG")
    okb = np.isfinite(P[:, br]); okt = np.isfinite(P[:, tg])
    e1, e2 = cov.index("APOE_e4_1"), cov.index("APOE_e4_2")

    rows = []
    for k in range(K):
        rows.append(dict(
            sig=k, r=float(np.median(r_per[k])),
            rlo=float(np.min(r_per[k])), ov=float(np.median(ov_per[k])),
            g1=float(Gam[k, e1]), g2=float(Gam[k, e2]),
            rb=float(spearmanr(aex[okb, k], P[okb, br]).statistic),
            rt=float(spearmanr(aex[okt, k], P[okt, tg]).statistic),
            top=[names[i] for i in np.argsort(-psi0[k])[:args.top]]))
    rows.sort(key=lambda r: -r["r"])

    print(f"\n{'='*112}")
    print(f"K={args.k} 的 21 个 signature，按跨折稳定性排序（对齐到 fold0，9 个折对）")
    print(f"{'='*112}")
    hdr = (f"{'':>2} {'sig':>4} {'r中位':>7} {'r最差':>7} {'top8保留':>9} "
           f"{'ε4=1':>7} {'ε4=2':>7} {'ρBraak':>8} {'ρTau':>7}  组成")
    print(hdr); print("-" * 112)
    for r in rows:
        ok = "OK" if r["r"] > thr and r["rlo"] > 0.3 else ("~" if r["r"] > thr else "XX")
        print(f"{ok:>2} {('sig'+str(r['sig'])):>4} {r['r']:>7.3f} {r['rlo']:>7.3f} "
              f"{r['ov']:>9.3f} {r['g1']:>+7.3f} {r['g2']:>+7.3f} "
              f"{r['rb']:>+8.3f} {r['rt']:>+7.3f}  {', '.join(r['top'])}")
    n_ok = sum(1 for r in rows if r["r"] > thr and r["rlo"] > 0.3)
    print("-" * 112)
    print(f"OK = r中位 > 零分布95分位({thr:.3f}) 且 最差折也 > 0.30    -> {n_ok}/{K} 个通过")
    print(f"~  = 超过零分布但某些折上塌掉;  XX = 和随机配对没区别")
    json.dump(dict(k=args.k, null_p95=thr, n_ok=n_ok, rows=rows),
              open(os.path.join(RES, f"per_signature_K{args.k}.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
