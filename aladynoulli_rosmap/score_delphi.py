# -*- coding: utf-8 -*-
"""用和 ALADYNOULLI 完全相同的 landmark 协议给 Delphi 打分。

两个模型唯一不同的是「分数怎么来的」：

    ALADYNOULLI   分析式      1 - prod_{t=t0+1}^{t0+H} (1 - pi_idt)
    Delphi        蒙特卡洛    从 t0 处的前缀 rollout n_mc 次，数 d 在窗口内出现的比例

风险集、标签、删失剔除规则全部读同一个 tensors.npz，逐格一致；池化和 AUC 走同一份
evaluate.auc_table。所以两条曲线之间的差就是模型的差。

== 时间轴对齐 ==

网格下标和 age_days 的换算必须和 data.py 逐位一致，否则"窗口"在两个模型里不是同一段时间：

    bl_days = 这个人最小的 age_days
    idx0    = round(bl_days / 365.25) - 60
    某条记录的网格下标 = idx0 + round((age_days - bl_days) / 365)

于是 landmark t0 对应这个人的第 (t0 - idx0) 个随访年：

    前缀截止日 = bl_days + (t0 - idx0) * 365
    窗口结束日 = bl_days + (t0 + H - idx0) * 365

== 一次 rollout 覆盖三个 horizon ==

rollout 到最长的 H 为止，然后在同一批轨迹上数三个窗口。省 3 倍算力，而且三个 horizon
用的是同一组随机轨迹，彼此之间的差异只来自窗口长度。

== 蒙特卡洛的粒度问题 ==

出现频率的分辨率是 1/n_mc。罕见事件在 n_mc 次里可能一次都没出现，一大批人并列 0 分，
AUC 会被并列拖向 0.5 —— 这是 MC 打分相对解析式打分天然的劣势，不是 Delphi 的问题。
脚本会报每个 (病, landmark) 的并列比例，读结果时必须看这个数。

    python3 score_delphi.py --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
for p in (os.path.join(HERE, "_figure2_eval"), os.path.join(HERE, "_delphi")):
    if p not in sys.path:
        sys.path.insert(0, p)

from evaluate import fold_split, landmark_sets          # noqa: E402
from fit import load                                     # noqa: E402

DAYS_PER_YEAR = 365.25


def tok_dir():
    d = os.path.join(HERE, "_tokenization", "data_rosmap")
    return d if os.path.isdir(d) else os.path.normpath(
        os.path.join(HERE, "..", "tokenization", "data_rosmap"))


# radc_delphi/vocab.py 在 import 时就读 labels.csv，默认路径是包内相对的，在这里解析不到。
# 它留了 ROSMAP_LABELS 环境变量，指向本项目的分词表即可（各折的 labels.csv 是同一份拷贝）。
os.environ.setdefault("ROSMAP_LABELS", os.path.join(tok_dir(), "labels.csv"))


def model_ids(dis_names):
    """疾病名 -> Delphi 的 model token id（= labels.csv 的行号）。"""
    lab = [l.strip() for l in open(os.path.join(tok_dir(), "labels.csv"))][1:]
    idx = {n: i for i, n in enumerate(lab)}
    return np.array([idx[n] for n in dis_names], dtype=np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-mc", type=int, default=500)
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--landmarks", default="75,80,85,90")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    args = ap.parse_args()

    from radc_delphi.engine import load as eload

    d = load()
    T = len(d["age_grid"]); A0 = int(d["age_grid"][0]); D = d["E"].shape[1]
    names = list(d["dis_names"])
    mids = model_ids(names)

    fs = fold_split(len(d["split"]), args.folds, args.seed)
    te = fs[args.fold]
    pids_te = d["pids"][te]

    # 原始事件流，按人索引
    arr = np.vstack([np.fromfile(os.path.join(tok_dir(), f"{s}.bin"), dtype=np.uint32).reshape(-1, 3)
                     for s in ("train", "val")])
    by_pid = {}
    order = np.argsort(arr[:, 0], kind="stable")
    arr = arr[order]
    bounds = np.searchsorted(arr[:, 0], np.unique(arr[:, 0]), side="left").tolist() + [len(arr)]
    upid = np.unique(arr[:, 0])
    for i, p in enumerate(upid):
        by_pid[int(p)] = arr[bounds[i]:bounds[i + 1]]

    dd = os.path.join(HERE, "_delphi", "data", f"rosmap_f{args.fold}")
    ck = os.path.join(HERE, "_delphi", f"Delphi-fold{args.fold}", "ckpt.pt")
    eng = eload(ck, data_dir=dd, device=args.device)

    Hs = [int(x) for x in args.horizons.split(",")]
    Hmax = max(Hs)
    lms = [int(x) - A0 for x in args.landmarks.split(",")]

    out, ties = {}, {}
    for t0 in lms:
        risk, _ = landmark_sets(d["E"][te], d["S"][te], d["atrisk"][te], d["Yobs"][te], t0, Hmax, T)
        if risk is None:
            continue
        sc = {H: np.zeros((len(te), D)) for H in Hs}
        for n, (gi, pid) in enumerate(zip(te, pids_te)):
            rows = by_pid[int(pid)]
            bl = int(rows[:, 1].min())
            idx0 = int(round(bl / DAYS_PER_YEAR)) - A0
            cut = bl + (t0 - idx0) * 365                       # 前缀截止日
            end = bl + (t0 + Hmax - idx0) * 365                # rollout 终点
            pre = rows[rows[:, 1] <= cut]
            if len(pre) == 0:
                continue
            o = np.argsort(pre[:, 1], kind="stable")
            st, sa = eng.simulate(pre[o, 2].astype(np.int64) + 1, pre[o, 1].astype(np.float64),
                                  n_mc=args.n_mc, until_age_years=end / DAYS_PER_YEAR,
                                  max_new_tokens=args.max_new_tokens,
                                  seed=(args.fold * 100003 + n * 97 + t0))
            st = np.asarray(st); sa = np.asarray(sa)
            fresh = sa > cut                                    # 只看 t0 之后新生成的
            for H in Hs:
                win = fresh & (sa <= bl + (t0 + H - idx0) * 365)
                for j, mid in enumerate(mids):
                    sc[H][n, j] = ((st == mid) & win).any(1).mean()
            if n % 100 == 0:
                print(f"  landmark {t0+A0}: {n}/{len(te)}", flush=True)

        for H in Hs:
            risk_H, lab_H = landmark_sets(d["E"][te], d["S"][te], d["atrisk"][te],
                                          d["Yobs"][te], t0, H, T)
            if risk_H is None:
                continue
            out[f"{H}_{t0}_delphi"] = sc[H]
            out[f"{H}_{t0}_risk"] = risk_H
            out[f"{H}_{t0}_label"] = lab_H
            tf = [float((sc[H][risk_H[:, j], j] == 0).mean()) for j in range(D)]
            ties[f"H{H}_lm{t0+A0}"] = dict(zip(names, np.round(tf, 3).tolist()))

    p = os.path.join(RES, f"cvraw_f{args.fold}_delphi.npz")
    np.savez_compressed(p, **out, meta=np.array(json.dumps(
        dict(Hs=Hs, lms=lms, A0=A0, names=names, n_test=len(te), n_mc=args.n_mc))),
        ties=np.array(json.dumps(ties)))
    print(f"\n-> {p}")
    for k, v in ties.items():
        z = [n for n, f in v.items() if f > 0.9]
        print(f"  {k}: 打分全 0 的人超过 90% 的病 {len(z)}/{len(names)}"
              + (f"  例: {', '.join(z[:5])}" if z else ""))


if __name__ == "__main__":
    main()
