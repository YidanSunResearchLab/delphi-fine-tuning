# -*- coding: utf-8 -*-
"""signature 是什么、遗传怎么进去、能不能预测尸检病理。

三块：

1. 组成       psi (K,D) 的 top 疾病 —— 每个 signature 到底是什么。
2. 遗传       Gamma (K,P) 里 APOE e4 剂量 / e2 / 性别 / 教育 的载荷。
              论文在 40 万人上用 36 个 PRS 做 GWAS；ROSMAP n=4,428 做 GWAS 没有功效，
              所以这里问的是另一个问题："APOE 的效应主要加载在哪个 signature 上"。
3. 病理验证   AEX_ik = sum_t theta_ikt （论文的 "lifetime signature exposure"，
              这里是离散网格上的求和）对上死后的 Braak / CERAD / 淀粉样 / Tau 等。
              这是 ROSMAP 有而论文没有的外部效度证据：论文只能拿 GWAS 位点佐证 signature，
              我们可以拿神经病理金标准佐证。

              注意 AEX 用的是**全窗口**的 theta，含死亡前的全部随访，所以这是描述性的
              关联而不是预测；要做预测得把 theta 截到某个 landmark 之前。

    python3 interpret.py --k 8 --fold 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from fit import load
from aladyn_model import Aladynoulli

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
TOKDIR = os.path.normpath(os.path.join(HERE, "_tokenization", "data_rosmap"))
if not os.path.isdir(TOKDIR):
    TOKDIR = os.path.normpath(os.path.join(HERE, "..", "tokenization", "data_rosmap"))

# 病理变量 -> 有序等级（越大越重）。CERAD 的 high=最重，见 tokenization/spec.py 的注释。
PATH_ORD = {
    "BRAAK":  ["BRAAK_low", "BRAAK_mid", "BRAAK_high"],
    "CERAD":  ["CERAD_low", "CERAD_mid", "CERAD_high"],
    "AMYL":   ["AMYL_q1", "AMYL_q2", "AMYL_q3"],
    "TANG":   ["TANG_q1", "TANG_q2", "TANG_q3"],
    "GPATH":  ["GPATH_q1", "GPATH_q2", "GPATH_q3"],
    "TDP":    ["TDP_none", "TDP_mild", "TDP_severe"],
    "LEWY":   ["LEWY_none", "LEWY_mild", "LEWY_severe"],
    "ARTSCL": ["ARTSCL_none", "ARTSCL_mild", "ARTSCL_severe"],
    "CAA":    ["CAA_none", "CAA_mild", "CAA_severe"],
    "CVDA":   ["CVDA_none", "CVDA_mild", "CVDA_severe"],
    "INFARCT": ["INFARCT_no", "INFARCT_yes"],
    "MICROINF": ["MICROINF_no", "MICROINF_yes"],
}


def pathology_table(pids):
    """从 .bin 抽每个人的病理等级。返回 (n, n_var) 的 float 数组，缺失为 nan。"""
    meta = json.load(open(os.path.join(TOKDIR, "meta.json")))
    vocab = meta["vocab"]
    tok = {name: i + 1 for i, name in enumerate(vocab)}
    arr = np.vstack([np.fromfile(os.path.join(TOKDIR, f"{f}.bin"), dtype=np.uint32).reshape(-1, 3)
                     for f in ("train", "val")])
    pos = {p: i for i, p in enumerate(pids)}
    out = np.full((len(pids), len(PATH_ORD)), np.nan)
    lut = {}
    for j, (var, levels) in enumerate(PATH_ORD.items()):
        for lv, nm in enumerate(levels):
            if nm in tok:
                lut[tok[nm]] = (j, float(lv))
    for p, _, t in arr:
        hit = lut.get(int(t))
        if hit is not None and p in pos:
            out[pos[p], hit[0]] = hit[1]
    return out, list(PATH_ORD)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    d = load()
    T = len(d["age_grid"]); D = d["E"].shape[1]
    path = args.ckpt or os.path.join(RES, f"fold{args.fold}_K{args.k}.pt")
    ck = torch.load(path, weights_only=False, map_location="cpu")
    K = ck["K"]
    m = Aladynoulli(ck["state"]["lam"].shape[0], D, T, d["G"].shape[1], K=K)
    m.load_state_dict(ck["state"]); m.eval()
    names = list(d["dis_names"]); cov = list(d["cov_names"])

    # ---------- 1. 组成 ----------
    psi = m.psi.detach().numpy()
    print(f"=== signature 组成 (psi, K={K}, ckpt={os.path.basename(path)}) ===")
    for k in range(K):
        idx = np.argsort(-psi[k])[:args.top]
        print(f"\nsig{k}:")
        for i in idx:
            print(f"    {psi[k, i]:+6.2f}  {names[i]}")

    # ---------- 2. 遗传 / 背景 ----------
    Gam = m.Gamma.detach().numpy()
    show = ["APOE_e4_1", "APOE_e4_2", "APOE_e2carrier", "Female",
            "EDU_lt12", "EDU_ge20", "SMOKING_current", "HTN_PREVALENT", "DM_PREVALENT"]
    show = [c for c in show if c in cov]
    print(f"\n\n=== Gamma: 协变量 -> signature 载荷 ===")
    print(f"{'covariate':<18}" + "".join(f"{'sig'+str(k):>9}" for k in range(K)))
    for c in show:
        j = cov.index(c)
        print(f"{c:<18}" + "".join(f"{Gam[k, j]:>+9.3f}" for k in range(K)))
    print("\n注: Gamma 是 lambda 的 GP 先验均值的系数，softmax 之前。绝对值不可直接当 log-OR 读，"
          "跨 signature 的相对大小才是有意义的。")

    # ---------- 3. AEX vs 病理 ----------
    # AEX 要每个人的 theta。stage-1 的 lambda 只覆盖训练那批人，这里按 checkpoint 的顺序取。
    from evaluate import fold_split
    fs = fold_split(len(d["split"]), 10, 0)
    tr = np.concatenate([fs[j] for j in range(10) if j != args.fold])
    lam = m.lam.detach()
    theta = F.softmax(lam, dim=1).numpy()                   # (n_tr, K, T)
    aex = theta.sum(2)                                      # (n_tr, K)

    P, pvars = pathology_table(d["pids"][tr])
    print(f"\n\n=== AEX (lifetime signature exposure) vs 尸检病理 ===")
    print(f"训练集 {len(tr)} 人中有病理数据的人数: "
          + ", ".join(f"{v}={int(np.isfinite(P[:,j]).sum())}" for j, v in enumerate(pvars)))
    print(f"\n{'pathology':<10}" + "".join(f"{'sig'+str(k):>9}" for k in range(K)))
    rows = {}
    for j, v in enumerate(pvars):
        ok = np.isfinite(P[:, j])
        if ok.sum() < 100:
            continue
        rs = [spearmanr(aex[ok, k], P[ok, j]).statistic for k in range(K)]
        rows[v] = rs
        print(f"{v:<10}" + "".join(f"{r:>+9.3f}" for r in rs))
    print(f"\nSpearman rho。n 见上。|rho|>0.05 在 n~2000 时 p<0.05，但这是 {K}x{len(rows)} 次比较，"
          f"Bonferroni 阈值约 |rho|>{1.96*np.sqrt(1/2000)*np.sqrt(np.log(K*max(len(rows),1))):.3f}。")

    json.dump(dict(K=K, fold=args.fold,
                   psi_top={f"sig{k}": [names[i] for i in np.argsort(-psi[k])[:args.top]]
                            for k in range(K)},
                   gamma={c: [float(Gam[k, cov.index(c)]) for k in range(K)] for c in show},
                   aex_pathology_rho=rows),
              open(os.path.join(RES, f"interpret_K{K}_f{args.fold}.json"), "w"), indent=2)
    print(f"\n-> {os.path.join(RES, f'interpret_K{K}_f{args.fold}.json')}")


if __name__ == "__main__":
    main()
