# -*- coding: utf-8 -*-
"""把 10 折的 held-out 打分池化成一张表。每个人恰好被留出一次，所以池化后 n=4428。

    python3 pool_cv.py --k 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np

from evaluate import auc_table, print_table

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(RES, f"cvraw_f*_K{args.k}.npz")))
    if not files:
        raise SystemExit(f"results/ 下没有 cvraw_f*_K{args.k}.npz")
    zs = [np.load(f, allow_pickle=True) for f in files]
    # 如果同一折的 Delphi 打分也在，就并进来一起比。两边的 risk/label 必须逐格一致——
    # 它们都是 evaluate.landmark_sets 在同一份 tensors.npz 上算的，这里断言一下，
    # 因为一旦不一致，AUC 表里两列就不是在同一道题上比了。
    # 任何 cvraw_f{f}_<tag>.npz (tag != K{k}) 都会被并进来当成额外一列，
    # 这样 Delphi 的打分和各种消融用同一条路径进表。
    dl = {}
    for f in files:
        fold = os.path.basename(f).split("_")[1]
        for g in sorted(glob.glob(os.path.join(RES, f"cvraw_{fold}_*.npz"))):
            tag = os.path.basename(g)[:-4].split("_", 2)[2]
            if tag == f"K{args.k}":
                continue
            dl.setdefault(f, {})[tag] = np.load(g, allow_pickle=True)
    if dl:
        tags = sorted({t for v in dl.values() for t in v})
        print(f"并入额外方法 {tags}: {len(dl)}/{len(files)} 折")
    meta = json.loads(str(zs[0]["meta"]))
    names, Hs, lms, A0, D = meta["names"], meta["Hs"], meta["lms"], meta["A0"], len(meta["names"])
    n_tot = sum(json.loads(str(z["meta"]))["n_test"] for z in zs)
    print(f"{len(files)} 折,  held-out 总人数 = {n_tot},  K = {meta['K']}")

    def blocks(H, t0=None):
        out = []
        for f, z in zip(files, zs):
            for t in ([t0] if t0 is not None else lms):
                key = f"{H}_{t}_risk"
                if key not in z:
                    continue
                b = {k: z[f"{H}_{t}_{k}"] for k in ("aladyn", "pop", "logit", "risk", "label")}
                for tag, zd in dl.get(f, {}).items():
                    if f"{H}_{t}_{tag}" not in zd:
                        continue
                    assert np.array_equal(zd[f"{H}_{t}_risk"], b["risk"]), "风险集不一致"
                    assert np.array_equal(zd[f"{H}_{t}_label"], b["label"]), "标签不一致"
                    b[tag] = zd[f"{H}_{t}_{tag}"]
                out.append(b)
        return out

    report = {}
    for H in Hs:
        for t0 in lms:
            b = blocks(H, t0)
            if b:
                res = auc_table(b, D, names)
                print_table(res, f"[CV n={n_tot}] H={H}y  landmark age {t0+A0}")
                report[f"H{H}_age{t0+A0}"] = list(res.values())
        b = blocks(H)
        if b:
            res = auc_table(b, D, names)
            print_table(res, f"[CV n={n_tot}] H={H}y  池化全部 landmark (含年龄混杂)")
            report[f"H{H}_pooled"] = list(res.values())

    out = args.out or os.path.join(RES, f"cv_auc_K{args.k}.json")
    json.dump(report, open(out, "w"), indent=2)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
