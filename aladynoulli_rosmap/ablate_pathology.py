# -*- coding: utf-8 -*-
"""sig12 的病理相关到底来自哪？

AEX 对 Braak 的 rho=0.29 是用**拟合出的** lambda 算的。但 lambda 的先验均值里就含 APOE，
而 APOE 本身就预测 Braak。所以必须分离三个来源：

  (a) 拟合的 AEX        逐人 lambda 拟合后的 sum_t theta_ikt   <- 我之前报的那个
  (b) 先验均值 AEX      lambda = r + Gamma^T g，不做逐人拟合
  (c) APOE e4 剂量单独   0/1/2

如果 (a) >> (c)，说明 signature 轨迹带了 APOE 之外的真实病理信息，可解释性价值成立。
如果 (a) ~ (c)，那 signature 只是 APOE 的一个复杂包装。

另外报 (d)：把 APOE 从回归里 partial 掉之后，拟合 AEX 对 Braak 的偏相关。

    python3 ablate_pathology.py --k 21 --fold 0
"""
from __future__ import annotations
import argparse, json, os
import numpy as np, torch, torch.nn.functional as F
from scipy.stats import spearmanr
from aladyn_model import Aladynoulli
from evaluate import fold_split
from fit import load
from interpret import pathology_table

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
ap = argparse.ArgumentParser()
ap.add_argument("--k", type=int, default=21); ap.add_argument("--fold", type=int, default=0)
ap.add_argument("--sig", type=int, default=None)
a = ap.parse_args()

d = load(); T = len(d["age_grid"]); D = d["E"].shape[1]
names = list(d["dis_names"]); cov = list(d["cov_names"])
ck = torch.load(os.path.join(RES, f"fold{a.fold}_K{a.k}.pt"), weights_only=False, map_location="cpu")
m = Aladynoulli(ck["state"]["lam"].shape[0], D, T, d["G"].shape[1], K=ck["K"])
m.load_state_dict(ck["state"]); m.eval()
fs = fold_split(len(d["split"]), 10, 0)
tr = np.concatenate([fs[j] for j in range(10) if j != a.fold])

COG = ["AD_DX", "MMSE_impaired", "MMSE_borderline", "COGN_low", "COGN_mid"]
psi = m.psi.detach().numpy()
k = a.sig if a.sig is not None else int(np.argmax(psi[:, [names.index(c) for c in COG]].sum(1)))
print(f"认知 signature = sig{k}  组成: {', '.join(names[i] for i in np.argsort(-psi[k])[:5])}\n")

with torch.no_grad():
    aex_fit = F.softmax(m.lam, dim=1).numpy().sum(2)[:, k]
    G = torch.as_tensor(d["G"][tr])
    lam_pr = (m.r[None, :, None] + (G @ m.Gamma.T)[:, :, None]).repeat(1, 1, T)
    aex_pri = F.softmax(lam_pr, dim=1).numpy().sum(2)[:, k]
e4 = d["G"][tr][:, cov.index("APOE_e4_1")] + 2 * d["G"][tr][:, cov.index("APOE_e4_2")]

P, pvars = pathology_table(d["pids"][tr])
print(f"{'pathology':<10}{'n':>6}{'(a)拟合AEX':>12}{'(b)先验AEX':>12}{'(c)APOE剂量':>12}{'(d)偏相关':>11}")
print("-" * 63)
rows = {}
for j, v in enumerate(pvars):
    ok = np.isfinite(P[:, j])
    if ok.sum() < 100: continue
    y = P[ok, j]
    ra = spearmanr(aex_fit[ok], y).statistic
    rb = spearmanr(aex_pri[ok], y).statistic
    rc = spearmanr(e4[ok], y).statistic
    # (d) 在 rank 空间里把 APOE 剂量线性回归掉，再看残差相关
    from scipy.stats import rankdata
    X = np.c_[np.ones(ok.sum()), rankdata(e4[ok])]
    res_a = rankdata(aex_fit[ok]) - X @ np.linalg.lstsq(X, rankdata(aex_fit[ok]), rcond=None)[0]
    res_y = rankdata(y) - X @ np.linalg.lstsq(X, rankdata(y), rcond=None)[0]
    rd = np.corrcoef(res_a, res_y)[0, 1]
    rows[v] = dict(fit=float(ra), prior=float(rb), apoe=float(rc), partial=float(rd),
                   n=int(ok.sum()))
    print(f"{v:<10}{ok.sum():>6}{ra:>+12.3f}{rb:>+12.3f}{rc:>+12.3f}{rd:>+11.3f}")
print("\n(d) = 把 APOE e4 剂量 partial 掉之后，拟合 AEX 与病理的偏相关（rank 空间）。")
print("    如果 (d) 仍显著，signature 轨迹带了 APOE 之外的病理信息。")
json.dump(rows, open(os.path.join(RES, f"ablate_pathology_K{a.k}_sig{k}.json"), "w"), indent=2)
