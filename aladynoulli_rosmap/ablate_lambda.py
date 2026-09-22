# -*- coding: utf-8 -*-
"""消融：去掉 stage-2 的逐人 lambda 拟合，lambda 固定成先验均值 r + Gamma^T g。

问题：ALADYNOULLI 比 Delphi 好的那一点，是 signature 结构的功劳，还是"测试时对每个人
单独跑几百步梯度下降"的功劳？这两个答案指向完全不同的改造方案：

  结构的功劳      -> 值得把低秩瓶颈共训进 Delphi
  测试时优化的功劳 -> 该做的是给 Delphi 加测试时自适应，瓶颈只剩可解释性价值

去掉逐人拟合之后，theta_ik 在时间上是常数（先验均值不随 t 变），pi_idt 只通过 phi_kd(t)
随年龄变化。也就是"人群年龄曲线 x 个体固定配方"，正好隔离出测试时自适应的贡献。

    python3 ablate_lambda.py --k 21 --fold 0
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch
from aladyn_model import Aladynoulli
from evaluate import fold_split, landmark_sets, surv_score
from fit import load

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")

ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=21)
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--horizons", default="1,3,5")
ap.add_argument("--landmarks", default="75,80,85,90")
a = ap.parse_args()

d = load(); T = len(d["age_grid"]); A0 = int(d["age_grid"][0]); D = d["E"].shape[1]
fs = fold_split(len(d["split"]), 10, 0); te = fs[a.fold]
ck = torch.load(os.path.join(RES, f"fold{a.fold}_K{a.k}.pt"), weights_only=False, map_location="cpu")
m = Aladynoulli(ck["state"]["lam"].shape[0], D, T, d["G"].shape[1], K=ck["K"])
m.load_state_dict(ck["state"]); m.eval()

with torch.no_grad():
    G = torch.as_tensor(d["G"][te])
    lam = (m.r[None, :, None] + (G @ m.Gamma.T)[:, :, None]).repeat(1, 1, T)  # 先验均值，不拟合
    p = m.pi(lam=lam).numpy()

out = {}
for H in [int(x) for x in a.horizons.split(",")]:
    for t0 in [int(x) - A0 for x in a.landmarks.split(",")]:
        risk, lab = landmark_sets(d["E"][te], d["S"][te], d["atrisk"][te], d["Yobs"][te], t0, H, T)
        if risk is None: continue
        out[f"{H}_{t0}_noref"] = surv_score(p, t0, H)
        out[f"{H}_{t0}_risk"] = risk; out[f"{H}_{t0}_label"] = lab
pth = os.path.join(RES, f"cvraw_f{a.fold}_noref.npz")
np.savez_compressed(pth, **out, meta=np.array(json.dumps(dict(n_test=len(te)))))
print(f"-> {pth}")
