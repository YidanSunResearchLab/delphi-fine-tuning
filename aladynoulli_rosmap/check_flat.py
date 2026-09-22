# -*- coding: utf-8 -*-
"""扁平头的完整性检查：h 到底在不在起作用，有没有残留泄漏。

delphiflat 的池化 AUC 比 Delphi 自己的 rollout 高 0.12，这个跳幅必须先证伪泄漏再报。

三个打分变体，模型参数完全不动，只改喂进去的 h：
    real    正常的 causal_h                        <- 正式结果
    zero    h 全部置零（只剩 mu_d(t) + g^T gamma）   <- 应当掉到协变量水平
    shuf    h 在人之间随机打乱（保留时间结构）        <- 应当掉到和 zero 差不多

如果 zero/shuf 掉不下来，说明 AUC 不是 h 带来的，管线有问题。
如果 real 远高于 zero 且 shuf ~ zero，那 h 是真的在起作用。

    python3 check_flat.py --fold 0
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from evaluate import fold_split, landmark_sets, surv_score
from fit import causal_h, load, load_dstates
from flat_head import FlatHead

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--H", type=int, default=5)
ap.add_argument("--landmark", type=int, default=80)
a = ap.parse_args()

d = load(); T = len(d["age_grid"]); A0 = int(d["age_grid"][0]); D = d["E"].shape[1]
names = list(d["dis_names"])
fs = fold_split(len(d["split"]), 10, 0)
te = fs[a.fold]; tr = np.concatenate([fs[j] for j in range(10) if j != a.fold])
t0 = a.landmark - A0

# 重训一遍（和 flat_head.py 同参数），拿到模型
import subprocess, sys
Htr = causal_h(load_dstates(a.fold, tr))
Hte = load_dstates(a.fold, te)
Gte = torch.as_tensor(d["G"][te])
m = FlatHead(D, T, Htr.shape[2], d["G"].shape[1])
sd = os.path.join(RES, f"flat_f{a.fold}.pt")
if not os.path.exists(sd):
    raise SystemExit(f"缺 {sd}；flat_head.py 需要先存 state_dict（见下方补丁）")
m.load_state_dict(torch.load(sd, map_location="cpu")); m.eval()

rng = np.random.default_rng(0)
variants = {
    "real": causal_h(Hte, t0),
    "zero": torch.zeros_like(causal_h(Hte, t0)),
    "shuf": causal_h(Hte[rng.permutation(len(te))], t0),
}
risk, lab = landmark_sets(d["E"][te], d["S"][te], d["atrisk"][te], d["Yobs"][te], t0, a.H, T)
print(f"fold{a.fold}  H={a.H}y  landmark {a.landmark}岁   (单折, n={len(te)})")
print(f"{'variant':<8}{'中位AUC':>10}{'n_dis':>7}")
for nm, Hv in variants.items():
    with torch.no_grad():
        p = m.pi(Hv, Gte).numpy()
    sc = surv_score(p, t0, a.H)
    aucs = []
    for j in range(D):
        mm = risk[:, j]; y = lab[mm, j].astype(int)
        if mm.sum() < 30 or y.sum() < 5 or y.sum() == mm.sum(): continue
        aucs.append(roc_auc_score(y, rankdata(sc[mm, j])))
    print(f"{nm:<8}{np.median(aucs):>10.3f}{len(aucs):>7}")
print("\nreal >> zero ~ shuf  => h 真的在起作用；否则管线有问题。")
