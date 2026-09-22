# -*- coding: utf-8 -*-
"""把 cv_auc_K*.json 和 stability_K*.json 压成一张总表。

    python3 summarize.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KS = [3, 5, 8, 12, 16, 21]
FOCUS = ["AD_DX", "Death", "STROKE", "DEPRESSION", "MMSE_impaired", "COGN_low"]


def main():
    print("=" * 74)
    print("AUC 总表 (10 折 CV, held-out n=4428, 块内秩归一后池化)")
    print("=" * 74)
    hdr = f"{'K':>3} {'H':>2} {'landmark':>10} {'nDis':>5} {'ALADYN':>7} {'logit':>7} {'pop':>6} {'>pop':>8} {'>logit':>8}"
    print(hdr)
    print("-" * len(hdr))
    for K in KS:
        f = os.path.join(RES, f"cv_auc_K{K}.json")
        if not os.path.exists(f):
            continue
        r = json.load(open(f))
        for H in (1, 3, 5):
            for key in sorted(k for k in r if k.startswith(f"H{H}_")):
                rows = r[key]
                if not rows:
                    continue
                A = np.array([x["aladyn"] for x in rows])
                L = np.array([x["logit"] for x in rows])
                P = np.array([x["pop"] for x in rows])
                lab = key.split("_", 1)[1]
                print(f"{K:>3} {H:>2} {lab:>10} {len(A):>5} {np.median(A):>7.3f} "
                      f"{np.median(L):>7.3f} {np.median(P):>6.3f} "
                      f"{str(int((A>P).sum()))+'/'+str(len(A)):>8} "
                      f"{str(int((A>L).sum()))+'/'+str(len(L)):>8}")
        print()

    print("=" * 74)
    print(f"重点结局的逐病 AUC (H=5y)")
    print("=" * 74)
    hdr2 = f"{'disease':<16}{'landmark':>9}" + "".join(f"{'K='+str(K):>8}" for K in KS) + f"{'logit':>8}{'pop':>7}"
    print(hdr2); print("-" * len(hdr2))
    ref = json.load(open(os.path.join(RES, f"cv_auc_K{KS[0]}.json")))
    lm_keys = sorted(k for k in ref if k.startswith("H5_") and "pooled" not in k)
    for dis in FOCUS:
        for key in lm_keys:
            vals, lg, pp, npos = [], None, None, None
            for K in KS:
                rr = json.load(open(os.path.join(RES, f"cv_auc_K{K}.json"))).get(key, [])
                hit = next((x for x in rr if x["disease"] == dis), None)
                vals.append(hit["aladyn"] if hit else np.nan)
                if hit:
                    lg, pp, npos = hit["logit"], hit["pop"], hit["n_pos"]
            if all(np.isnan(v) for v in vals):
                continue
            lab = key.split("_", 1)[1] + f"(n={npos})"
            print(f"{dis:<16}{lab:>9}" + "".join(f"{v:>8.3f}" if np.isfinite(v) else f"{'-':>8}" for v in vals)
                  + f"{lg:>8.3f}{pp:>7.3f}")
        print()

    print("=" * 74)
    print("signature 稳定性 (10 折两两比较, 匈牙利匹配后)")
    print("=" * 74)
    print(f"{'K':>3} {'psi跨折r中位':>14} {'IQR':>18} {'最差':>7} {'top8组成保留':>14}")
    for K in KS:
        f = os.path.join(RES, f"stability_K{K}.json")
        if not os.path.exists(f):
            continue
        s = json.load(open(f))
        iqr = f"[{s['psi_cor_iqr'][0]:.2f}, {s['psi_cor_iqr'][1]:.2f}]"
        print(f"{K:>3} {s['psi_cor_median']:>14.3f} {iqr:>18} {s['psi_cor_min']:>7.2f} "
              f"{s['top_overlap_median']:>14.3f}")
    print("\n论文在 UKB (n=400k, 40 个子集) 上报 r=0.995; 跨队列组成保留中位 0.80")


if __name__ == "__main__":
    main()
