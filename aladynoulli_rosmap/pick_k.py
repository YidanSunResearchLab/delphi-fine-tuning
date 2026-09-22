# -*- coding: utf-8 -*-
"""选 K：不看平均 AUC，看"那个认知/AD 的 signature 在每个 K 下长什么样、稳不稳、
对不对得上 APOE 和尸检病理"。

平均 AUC 会随 K 单调上升，但那部分增益有多少来自"population 级的 signature 结构"、
多少来自"stage-2 给每个人单独拟合 lambda 时自由度更大"，平均值分不清。
而我们真正要的是一个能用的 AD signature，所以直接按这个目标选 K。

每个 K 报五件事：
  1. 认知 signature 的编号和组成（ψ 在认知 token 上权重最大的那个）
  2. 它的跨折稳定性（这一个 signature 自己的 r，不是全体中位数）
  3. APOE ε4 的剂量响应 Γ(e4=2)/Γ(e4=1)，真实的等位基因剂量效应应该接近 2
  4. AEX 对 Braak / 缠结 的 Spearman ρ
  5. AD_DX 和 COGN_low 的 held-out AUC

    python3 pick_k.py
"""
from __future__ import annotations

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
KS = [3, 5, 8, 12, 16, 21]
COG = ["AD_DX", "MMSE_impaired", "MMSE_borderline", "COGN_low", "COGN_mid"]


def cognitive_sig(psi, names):
    """ψ 在认知 token 上的权重之和最大的那个 signature。"""
    idx = [names.index(c) for c in COG if c in names]
    return int(np.argmax(psi[:, idx].sum(1)))


def main():
    d = load()
    names = list(d["dis_names"]); cov = list(d["cov_names"])
    T = len(d["age_grid"]); D = len(names)
    fs = fold_split(len(d["split"]), 10, 0)
    tr0 = np.concatenate([fs[j] for j in range(10) if j != 0])
    P, pvars = pathology_table(d["pids"][tr0])
    br, tg = pvars.index("BRAAK"), pvars.index("TANG")

    print("=" * 100)
    print("按「能不能给出一个可用的 AD signature」来选 K")
    print("=" * 100)
    hdr = (f"{'K':>3} {'认知sig':>7} {'跨折r':>7} {'全体r中位':>9} "
           f"{'e4=1':>7} {'e4=2':>7} {'剂量比':>7} {'ρBraak':>8} {'ρTau':>7} "
           f"{'AD_DX':>7} {'COGNlow':>8}")
    print(hdr); print("-" * len(hdr))

    rows = []
    for K in KS:
        Pk = load_psis(K)
        if len(Pk) < 10:
            continue
        psi0 = Pk[0]["psi"]
        k = cognitive_sig(psi0, names)

        # 这一个 signature 自己的跨折稳定性
        rs = []
        for j in sorted(Pk):
            if j == 0:
                continue
            ci, c = match(psi0, Pk[j]["psi"])
            rs.append(c[k])
        r_self = float(np.median(rs))
        allr = json.load(open(os.path.join(RES, f"stability_K{K}.json")))["psi_cor_median"]

        # Γ 的 APOE 剂量响应
        ck = torch.load(os.path.join(RES, f"fold0_K{K}.pt"), weights_only=False, map_location="cpu")
        Gam = ck["state"]["Gamma"].numpy()
        g1 = Gam[k, cov.index("APOE_e4_1")]; g2 = Gam[k, cov.index("APOE_e4_2")]
        ratio = g2 / g1 if abs(g1) > 1e-6 else np.nan

        # AEX 对病理
        lam = ck["state"]["lam"]
        aex = F.softmax(lam, dim=1).numpy().sum(2)[:, k]
        ok = np.isfinite(P[:, br])
        rb = spearmanr(aex[ok], P[ok, br]).statistic
        ok = np.isfinite(P[:, tg])
        rt = spearmanr(aex[ok], P[ok, tg]).statistic

        # held-out AUC
        auc = json.load(open(os.path.join(RES, f"cv_auc_K{K}.json")))
        def get(dis, key="H5_age75"):
            hit = next((x for x in auc.get(key, []) if x["disease"] == dis), None)
            return hit["aladyn"] if hit else np.nan

        print(f"{K:>3} {('sig'+str(k)):>7} {r_self:>7.3f} {allr:>9.3f} "
              f"{g1:>+7.3f} {g2:>+7.3f} {ratio:>7.2f} {rb:>+8.3f} {rt:>+7.3f} "
              f"{get('AD_DX'):>7.3f} {get('COGN_low'):>8.3f}")
        rows.append(dict(K=K, sig=k, r_self=r_self, r_all=allr, g1=float(g1), g2=float(g2),
                         ratio=float(ratio), rho_braak=float(rb), rho_tau=float(rt),
                         auc_ad=float(get("AD_DX")), auc_cogn=float(get("COGN_low")),
                         top=[names[i] for i in np.argsort(-psi0[k])[:6]]))

    print("\n各 K 下认知 signature 的组成（fold0）:")
    for r in rows:
        print(f"  K={r['K']:>2}  sig{r['sig']:<2} -> {', '.join(r['top'])}")
    print("\n注: 跨折r 是这一个 signature 自己的中位相关（对齐到 fold0），"
          "全体r中位 是该 K 下所有 signature 的中位。")
    print("剂量比 = Γ(ε4=2)/Γ(ε4=1)，真实的加性等位基因剂量效应应该接近 2.0。")
    json.dump(rows, open(os.path.join(RES, "pick_k.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
