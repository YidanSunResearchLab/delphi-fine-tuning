# -*- coding: utf-8 -*-
"""ALADYNOULLI vs Delphi 头对头汇总表。

两个模型的风险集、标签、删失规则、池化方式逐格一致（pool_cv 里有断言核对），
唯一的差别是分数怎么算出来的：
    aladyn  解析式    1 - prod(1 - pi_idt)
    delphi  蒙特卡洛  500 次 rollout 数窗口内出现频率

    python3 h2h.py --k 21
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
M = ["aladyndh", "aladyn", "delphiflat", "delphi", "noref", "pop"]
FOCUS = ["AD_DX", "COGN_low", "MMSE_impaired", "MMSE_borderline", "COGN_mid",
         "Death", "STROKE", "DEPRESSION", "HTN_ONSET", "DM_ONSET", "CHF_ONSET"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=21)
    args = ap.parse_args()
    r = json.load(open(os.path.join(RES, f"cv_auc_K{args.k}.json")))

    print("=" * 84)
    print(f"ALADYNOULLI (K={args.k}) vs Delphi   10 折 CV, held-out n=4428")
    print("=" * 84)
    hdr = (f"{'H':>2} {'landmark':>10} {'nDis':>5} " + "".join(f"{m:>8}" for m in M)
           + f" {'A-D':>7} {'A>D':>8}")
    print(hdr); print("-" * len(hdr))
    for H in (1, 3, 5):
        for key in sorted(k for k in r if k.startswith(f"H{H}_")):
            rows = [x for x in r[key] if all(m in x for m in M)]
            if not rows:
                continue
            A = {m: np.array([x[m] for x in rows]) for m in M}
            lab = key.split("_", 1)[1]
            print(f"{H:>2} {lab:>10} {len(rows):>5} "
                  + "".join(f"{np.median(A[m]):>8.3f}" for m in M)
                  + f" {np.median(A['aladyn']-A['delphi']):>+7.3f}"
                  + f" {str(int((A['aladyn']>A['delphi']).sum()))+'/'+str(len(rows)):>8}")
        print()

    print("=" * 84)
    print("重点结局逐病 (H=5y)")
    print("=" * 84)
    hdr2 = f"{'disease':<17}{'lm':>4}{'n_pos':>7}" + "".join(f"{m:>9}" for m in M) + f"{'A-D':>8}"
    print(hdr2); print("-" * len(hdr2))
    for dis in FOCUS:
        any_row = False
        for key in sorted(k for k in r if k.startswith("H5_") and "pooled" not in k):
            hit = next((x for x in r[key] if x["disease"] == dis and all(m in x for m in M)), None)
            if not hit:
                continue
            any_row = True
            lm = key.split("age")[1]
            print(f"{dis if lm=='75' else '':<17}{lm:>4}{hit['n_pos']:>7}"
                  + "".join(f"{hit[m]:>9.3f}" for m in M)
                  + f"{hit['aladyn']-hit['delphi']:>+8.3f}")
        if any_row:
            print()


if __name__ == "__main__":
    main()
