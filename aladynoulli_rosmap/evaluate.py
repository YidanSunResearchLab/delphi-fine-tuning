# -*- coding: utf-8 -*-
"""ALADYNOULLI-ROSMAP 的 landmark 评估，产出可池化的原始打分。

== landmark 设计（对应论文的 "dynamic 1-year predictions"）==

在若干个绝对年龄 t0 上：
  风险集   : atrisk_id=1 且 S_i <= t0 且 E_id > t0     （到 t0 还没发生、还在观测中）
  标签     : Yobs_id=1 且 t0 < E_id <= t0+H
  剔除     : Yobs_id=0 且 E_id < t0+H                  （窗口结束前就删失了，阳性阴性都判不了）
  打分     : 1 - prod_{t=t0+1}^{t0+H} (1 - pi_idt)
  lambda   : 用 horizon=t0 重新拟合，只吃 t0 之前的信息（论文的 prospective 协议）

== 为什么 AUC 必须分 landmark 报 ==

在单个 landmark 内，所有人年龄完全相同，所以"只看年龄"的 pop 基线在这里的 AUC 恰好是 0.5。
超过 0.5 的部分才是模型真正从个体身上学到的东西。把多个 landmark 池化会混入年龄差异，
把所有方法的 AUC 一起抬上去——那个数字好看但不回答"signature 有没有用"。
两个数都报，主结论看 per-landmark。

跨折池化时必须先做块内秩归一，见 auc_table 的 rank_within_block。不做的话留一法会制造
一个纯人为的负信号，实测能把 pop 从 0.500 压到 0.23。

== 两个对照 ==

pop   : 只用训练集的经验年龄别 hazard h_d(t)，完全不看这个人是谁。
logit : 每个病一个逻辑回归，协变量 = 随访时长 + 性别 + APOE e4 剂量 + e2 + 教育。
两个对照用的风险集、标签、剔除规则和 ALADYNOULLI 逐格一致。

== 两种运行方式 ==

    python3 evaluate.py --k 8                      # 用 Delphi 原本的 443 人 val split
    python3 evaluate.py --k 8 --fold 3 --folds 10  # 10 折 CV 的第 3 折，存原始打分
                                                   # 十折跑完用 pool_cv.py 池化 -> n=4428
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from fit import load, hazard_counts, fit_lambda, load_dstates, causal_h
from aladyn_model import Aladynoulli

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
COVS = ["Female", "APOE_e4_1", "APOE_e4_2", "APOE_e2carrier",
        "EDU_lt12", "EDU_16to19", "EDU_ge20"]


def fold_split(N, folds, seed):
    perm = np.random.default_rng(seed).permutation(N)
    return np.array_split(perm, folds)


def surv_score(p, t0, H):
    """(n,D,T) hazard -> (n,D) 的 (t0, t0+H] 累积发病概率。"""
    return 1.0 - np.prod(1.0 - p[:, :, t0 + 1: t0 + 1 + H], axis=2)


def landmark_sets(E, S, atrisk, Yobs, t0, H, T):
    E = E.astype(np.int64); S = S.astype(np.int64)
    if t0 + H > T - 1:
        return None, None
    risk = (atrisk == 1) & (S[:, None] <= t0) & (E > t0)
    pos = risk & (Yobs == 1) & (E <= t0 + H)
    drop = risk & (Yobs == 0) & (E < t0 + H)
    return risk & ~drop, pos


def baseline_pop(h_train, n, t0, H):
    s = 1.0 - np.prod(1.0 - h_train[:, t0 + 1: t0 + 1 + H], axis=1)
    return np.repeat(s[None, :], n, 0)


def baseline_logit(d, tr, te, t0, H, T, cov_idx):
    def X(ix):
        return np.hstack([d["G"][ix][:, cov_idx],
                          (t0 - d["S"][ix].astype(float))[:, None]])
    Xtr, Xte = X(tr), X(te)
    rtr, ltr = landmark_sets(d["E"][tr], d["S"][tr], d["atrisk"][tr], d["Yobs"][tr], t0, H, T)
    S = np.zeros((len(te), d["E"].shape[1]))
    for j in range(d["E"].shape[1]):
        m = rtr[:, j]; y = ltr[m, j].astype(int)
        if m.sum() < 50 or y.sum() < 10 or y.sum() == m.sum():
            continue
        S[:, j] = LogisticRegression(max_iter=2000).fit(Xtr[m], y).predict_proba(Xte)[:, 1]
    return S


def score_all(d, m, tr, te, lms, Hs, h_train, cov_idx, T, lam_steps, device, dstates=None):
    """对每个 (H, landmark) 产出各方法的 (score, risk, label)。

    dstates 不为 None 时，先验均值里带 Delphi 的隐状态。**每个 landmark 都要过 causal_h**：
    既要右移一格（不让事件当年的 h 看见事件自己），又要把 t > t0 的换成 h(t0)
    （不让 landmark 之后的信息进来）。见 fit.causal_h。
    """
    out = {}
    lam_cache = {}
    for t0 in lms:
        Hc = causal_h(dstates, t0) if dstates is not None else None
        lam_cache[t0] = fit_lambda(m, d["G"][te], d["E"][te], d["S"][te], d["atrisk"][te],
                                   d["Yobs"][te], T, horizon=t0, steps=lam_steps, device=device,
                                   Hs=Hc)
    for H in Hs:
        for t0 in lms:
            risk, label = landmark_sets(d["E"][te], d["S"][te], d["atrisk"][te],
                                        d["Yobs"][te], t0, H, T)
            if risk is None or risk.sum() == 0:
                continue
            with torch.no_grad():
                p = m.pi(lam=lam_cache[t0]).cpu().numpy()
            out[(H, t0)] = dict(
                **{("aladyndh" if dstates is not None else "aladyn"): surv_score(p, t0, H)},
                pop=baseline_pop(h_train, len(te), t0, H),
                logit=baseline_logit(d, tr, te, t0, H, T, cov_idx),
                risk=risk, label=label)
    return out


def _rank01(v):
    """块内秩归一到 [0,1]，并列取平均秩。单调变换，不改变块内 AUC。"""
    from scipy.stats import rankdata
    return rankdata(v) / (len(v) + 1.0)


def auc_table(blocks, D, names, min_pos=10, rank_within_block=True, methods=None):
    """blocks 是若干 dict（一折 x 一个 landmark 为一块）。逐病算每个方法的 AUC。

    methods: 要算哪几列。None 时从 block 的键里推断（除 risk/label 外的全部）。
    这样同一个函数既能处理 ALADYNOULLI 的三列，也能在合并了 Delphi 打分之后处理四列。

    rank_within_block: 池化多块之前先在**块内**把分数转成秩。这是必须的——
    留一法交叉验证会在"某折的预测水平"和"该折自己的事件率"之间制造负相关
    （某折事件多 => 用其余 9 折算出的 hazard 偏低），实测 r = -0.86 ~ -1.00。
    直接池化原始分数会把这个假信号算进 AUC：pop 基线本该是 0.500，池化后变成 0.23-0.38。
    秩归一只去掉块间的水平差，块内的排序信息（也就是 AUC 唯一依赖的东西）完全保留。
    """
    if methods is None:
        methods = [k for k in blocks[0] if k not in ("risk", "label")]
    res = {}
    for d_ in range(D):
        ys, ss = [], {k: [] for k in methods}
        for b in blocks:
            mm = b["risk"][:, d_]
            if mm.sum() == 0:
                continue
            ys.append(b["label"][mm, d_].astype(int))
            for k in methods:
                v = b[k][mm, d_]
                ss[k].append(_rank01(v) if rank_within_block else v)
        if not ys:
            continue
        y = np.concatenate(ys)
        if y.sum() < min_pos or y.sum() == len(y):
            continue
        res[d_] = dict(disease=names[d_], n_pos=int(y.sum()), n_risk=int(len(y)),
                       **{k: float(roc_auc_score(y, np.concatenate(v))) for k, v in ss.items()})
    return res


def print_table(res, title, methods=None, ref="pop"):
    if not res:
        print(f"\n{title}: 没有满足 n_pos>=10 的病"); return
    if methods is None:
        methods = [k for k in next(iter(res.values()))
                   if k not in ("disease", "n_pos", "n_risk")]
    main_m = methods[0]
    rows = sorted(res.values(), key=lambda r: -r[main_m])
    print(f"\n===== {title} =====")
    print(f"{'disease':<18}{'n_pos':>7}{'n_risk':>8}" + "".join(f"{m:>9}" for m in methods))
    for r in rows:
        print(f"{r['disease']:<18}{r['n_pos']:>7}{r['n_risk']:>8}"
              + "".join(f"{r[m]:>9.3f}" for m in methods))
    A = {m: np.array([r[m] for r in rows]) for m in methods}
    print(f"{'-- median --':<18}{'':>15}" + "".join(f"{np.median(A[m]):>9.3f}" for m in methods))
    for m in methods:
        if m == ref or ref not in A:
            continue
        print(f"  {m} > {ref}: {(A[m]>A[ref]).sum()}/{len(rows)}", end="")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fold", type=int, default=-1)
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--landmarks", default="75,80,85,90")
    ap.add_argument("--lam-steps", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--use-delphi", action="store_true")
    args = ap.parse_args()

    d = load()
    T = len(d["age_grid"]); A0 = int(d["age_grid"][0]); D = d["E"].shape[1]
    names = list(d["dis_names"])

    if args.fold >= 0:
        fs = fold_split(len(d["split"]), args.folds, args.seed)
        te = fs[args.fold]
        tr = np.concatenate([fs[j] for j in range(args.folds) if j != args.fold])
        sfx = "_dh" if args.use_delphi else ""
        ckpt = args.ckpt or os.path.join(RES, f"fold{args.fold}_K{args.k}{sfx}.pt")
        # 文件名里的 tag 必须和方法列名一致，pool_cv 靠它把额外的列并进主表
        tag = f"f{args.fold}_aladyndh" if args.use_delphi else f"f{args.fold}_K{args.k}"
    else:
        tr = np.nonzero(d["split"] == 0)[0]; te = np.nonzero(d["split"] == 1)[0]
        ckpt = args.ckpt or os.path.join(RES, f"fit_K{args.k}.pt")
        tag = f"val_K{args.k}"

    ck = torch.load(ckpt, weights_only=False)
    K = ck["K"]
    n_h = ck["state"]["W"].shape[1] if "W" in ck["state"] else 0
    m = Aladynoulli(ck["state"]["lam"].shape[0], D, T, d["G"].shape[1], K=K,
                    n_h=n_h, device=args.device)
    m.load_state_dict(ck["state"]); m.eval()
    dstates = load_dstates(args.fold, te, args.device) if n_h else None

    ev, ar = hazard_counts(d["E"][tr], d["S"][tr].astype(np.int64), d["atrisk"][tr],
                           d["Yobs"][tr], T)
    h_train = np.clip(ev / np.maximum(ar, 1.0), 1e-6, 0.9)
    cov_idx = [list(d["cov_names"]).index(x) for x in COVS]

    lms = [int(x) - A0 for x in args.landmarks.split(",")]
    Hs = [int(x) for x in args.horizons.split(",")]
    blocks = score_all(d, m, tr, te, lms, Hs, h_train, cov_idx, T, args.lam_steps, args.device,
                       dstates=dstates)

    # 存原始打分，供 pool_cv.py 跨折池化
    flat = {f"{H}_{t0}_{k}": v for (H, t0), b in blocks.items() for k, v in b.items()}
    np.savez_compressed(os.path.join(RES, f"cvraw_{tag}.npz"), **flat,
                        meta=np.array(json.dumps(dict(K=K, Hs=Hs, lms=lms, A0=A0,
                                                      names=names, n_test=len(te)))))
    for H in Hs:
        for t0 in lms:
            if (H, t0) in blocks:
                print_table(auc_table([blocks[(H, t0)]], D, names),
                            f"H={H}y  landmark age {t0+A0}  [{tag}]")
        print_table(auc_table([b for (h, _), b in blocks.items() if h == H], D, names),
                    f"H={H}y  池化全部 landmark  [{tag}]  (注意: 含年龄混杂)")
    print(f"\n-> {os.path.join(RES, f'cvraw_{tag}.npz')}")


if __name__ == "__main__":
    main()
