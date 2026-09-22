# -*- coding: utf-8 -*-
"""拟合 ALADYNOULLI。MAP + Adam，全批量，按论文。

两阶段，对应论文的 prospective 协议：
  stage 1  在 train 的人身上联合估 {lambda_train, phi, psi, r, Gamma, kappa}
  stage 2  冻住除 lambda 外的全部参数，只给 held-out 的人重新估 lambda
           （论文: "Fixed phi-bar, refit only individual lambda-hat, using data available
             up to each prediction timepoint"）。不这么做的话 held-out 的人根本没有 lambda，
           或者用训练时的 lambda = 泄漏。

用法
    python3 fit.py --k 8 --steps 3000                  # 单次拟合
    python3 fit.py --k 8 --folds 10                    # 10 折，用来看 signature 稳不稳
    python3 fit.py --sweep-k 3,5,8,12,16               # 选 K
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from aladyn_model import Aladynoulli, build_mask

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(path=None):
    z = np.load(path or os.path.join(HERE, "tensors.npz"), allow_pickle=True)
    return {k: z[k] for k in z.files}


def cooccurrence(Yobs, atrisk):
    """疾病共现的余弦相似度，用来做谱聚类初始化。只在两病都 at-risk 的人里算。"""
    Y = (Yobs * atrisk).astype(np.float64)
    C = Y.T @ Y
    n = np.sqrt(np.diag(C))
    C = C / np.maximum(n[:, None] * n[None, :], 1e-9)
    np.fill_diagonal(C, 1.0)
    return np.clip(C, 0, 1)


def hazard_counts(E, S, atrisk, Yobs, T):
    """(D,T) 的事件数与暴露人年，给 mu_d(t) 用。"""
    N, D = E.shape
    ev = np.zeros((D, T)); ar = np.zeros((D, T))
    t = np.arange(T)
    for d in range(D):
        m = atrisk[:, d] == 1
        lo, hi = S[m], E[m, d]
        inwin = (t[None, :] >= lo[:, None]) & (t[None, :] <= hi[:, None])
        ar[d] = inwin.sum(0)
        e = hi[Yobs[m, d] == 1]
        ev[d] = np.bincount(e, minlength=T)
    return ev, ar


def _opt(params, lr):
    return torch.optim.Adam(params, lr=lr)


def load_dstates(fold, idx=None, device="cpu"):
    """Delphi 隐状态 (n, T, n_embd)。必须用同一折的 checkpoint 抽的，见 delphi_states.py。"""
    z = np.load(os.path.join(RES, f"dstates_f{fold}.npz"), allow_pickle=True)
    H = z["H"].astype(np.float32)
    if idx is not None:
        H = H[idx]
    return torch.as_tensor(H, device=device)


def causal_h(H_incl, t0=None):
    """把"含 t 的最后一个 token"的隐状态，转成预测**第 t 年 hazard 时允许使用**的隐状态。

    dstates 里存的 H_incl[:, t, :] 是网格下标 <= t 的最后一个 token 处的表示。直接拿它去
    预测第 t 年会不会发病是泄漏：事件发生那一年 t=E 也在风险集 mask 里，而 H_incl[E] 已经
    看过 token d 本身，模型会学到"h 说 d 刚发生了 -> pi_d≈1"这条捷径。打分时 h 是截断的，
    这条捷径不存在，于是学出来的 W 完全没用。实测扁平头的 NLL/cell 会从 0.14 掉到 0.03，
    比事件基础率 0.037 对应的熵下限 0.158 还低一个量级 —— 这是发现它的那个信号。

    所以训练时整体右移一格：预测第 t 年只用到第 t-1 年末。

    t0 不为 None（打分）时，t > t0 的位置改用 H_incl[t0]：站在 landmark 上往前看，t0 当年
    为止的信息是允许用的，之后的不允许。
    """
    z = torch.zeros_like(H_incl[:, :1, :])
    out = torch.cat([z, H_incl[:, :-1, :]], dim=1)
    if t0 is not None:
        out[:, t0 + 1:, :] = H_incl[:, t0:t0 + 1, :]
    return out


def fit_stage1(d, idx, K, steps, lr, alpha_lam, alpha_phi, seed, device, verbose=True,
               es_frac=0.1, es_every=100, patience=8, Hs=None, w_ridge=1.0):
    """es_frac 的比例从 *训练集内部* 再切一块做早停。held-out fold 全程不参与，
    所以早停不会把评估用的那批人的信息漏进来。"""
    T = len(d["age_grid"])
    G = d["G"][idx]
    mask, y = build_mask(d["E"][idx], d["S"][idx], d["atrisk"][idx], d["Yobs"][idx], T)
    ev, ar = hazard_counts(d["E"][idx], d["S"][idx].astype(np.int64),
                           d["atrisk"][idx], d["Yobs"][idx], T)

    m = Aladynoulli(len(idx), d["E"].shape[1], T, G.shape[1], K=K,
                    alpha_lambda=alpha_lam, alpha_phi=alpha_phi,
                    n_h=(0 if Hs is None else Hs.shape[2]), device=device).to(device)
    m.set_mu(ev, ar)
    lab = m.init_from_clusters(cooccurrence(d["Yobs"][idx], d["atrisk"][idx]), G, seed=seed)

    tG = torch.as_tensor(G, device=device)
    tm = torch.as_tensor(mask, device=device)
    ty = torch.as_tensor(y, device=device)

    # 早停集：训练集内部随机 es_frac。它的 lambda 和其他人一样在 stage1 里被估，
    # 但它的 NLL 只用来决定"什么时候停"，不参与梯度。
    rng = np.random.default_rng(seed)
    es = np.zeros(len(idx), bool)
    es[rng.choice(len(idx), max(1, int(es_frac * len(idx))), replace=False)] = True
    tes = torch.as_tensor(es, device=device)
    fit_m = tm * (~tes).float()[:, None, None]      # 进梯度的部分
    es_m = tm * tes.float()[:, None, None]          # 只看不动的部分
    ncell = float(fit_m.sum()); n_es = float(es_m.sum())

    opt = _opt(m.parameters(), lr)
    t0 = time.time()
    best, best_it, bad, best_state = float("inf"), 0, 0, None
    for it in range(steps):
        opt.zero_grad(set_to_none=True)
        p = m.pi()
        nll = m.nll(p, ty, fit_m)
        reg = w_ridge * (m.W ** 2).sum() if m.W is not None else 0.0
        loss = (nll + m.gp_lambda(tG, Hs=Hs) + m.gp_phi() + reg) / ncell
        loss.backward()
        opt.step()
        if (it + 1) % es_every == 0 or it == steps - 1:
            with torch.no_grad():
                v = m.nll(m.pi(), ty, es_m).item() / n_es
            if v < best - 1e-6:
                best, best_it, bad = v, it, 0
                best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
            else:
                bad += 1
            if verbose:
                print(f"  it {it+1:5d}  obj {loss.item():.6f}  fit/cell {nll.item()/ncell:.6f}"
                      f"  es/cell {v:.6f}{'  *' if bad == 0 else ''}  [{time.time()-t0:.0f}s]")
            if bad >= patience:
                print(f"  早停于 it={it+1}，最好在 it={best_it+1} (es/cell={best:.6f})")
                break
    if best_state is not None:
        m.load_state_dict(best_state)
    return m, lab


def fit_lambda(m, G, E, S, atrisk, Yobs, T, horizon=None, steps=600, lr=0.05, device="cpu",
               Hs=None):
    """stage 2: 冻住全局参数，只估 held-out 的 lambda。horizon 之后的信息不参与。"""
    for p in m.parameters():
        p.requires_grad_(False)
    n = len(G)
    tG = torch.as_tensor(G, device=device)
    lam = m.prior_mean(tG, Hs).detach().clone()
    lam.requires_grad_(True)
    mask, y = build_mask(E, S, atrisk, Yobs, T, horizon=horizon)
    tm = torch.as_tensor(mask, device=device); ty = torch.as_tensor(y, device=device)
    ncell = float(tm.sum())
    opt = _opt([lam], lr)
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        p = m.pi(lam=lam)
        loss = (m.nll(p, ty, tm) + m.gp_lambda(tG, lam=lam, Hs=Hs)) / ncell
        loss.backward()
        opt.step()
    for p in m.parameters():
        p.requires_grad_(True)
    return lam.detach()


def run_once(d, K, args, seed=0, tr=None, va=None, tag="main", fold=None):
    tr = np.nonzero(d["split"] == 0)[0] if tr is None else tr
    va = np.nonzero(d["split"] == 1)[0] if va is None else va
    T = len(d["age_grid"])
    Htr = Hva = None
    if getattr(args, "use_delphi", False):
        if fold is None:
            raise SystemExit("--use-delphi 只在 --folds 模式下可用（h 必须来自同一折的 checkpoint）")
        Htr = causal_h(load_dstates(fold, tr, args.device))
        Hva = causal_h(load_dstates(fold, va, args.device))
        print(f"[{tag}] 接入 Delphi 隐状态 n_h={Htr.shape[2]}")
    print(f"[{tag}] K={K}  train={len(tr)}  val={len(va)}")
    m, lab = fit_stage1(d, tr, K, args.steps, args.lr, args.alpha_lambda, args.alpha_phi,
                        seed, args.device, Hs=Htr, w_ridge=args.w_ridge)
    lam_va = fit_lambda(m, d["G"][va], d["E"][va], d["S"][va], d["atrisk"][va],
                        d["Yobs"][va], T, horizon=None, steps=args.lam_steps,
                        lr=args.lr_lambda, device=args.device, Hs=Hva)
    with torch.no_grad():
        mk, yv = build_mask(d["E"][va], d["S"][va], d["atrisk"][va], d["Yobs"][va], T)
        tmk = torch.as_tensor(mk, device=args.device)
        nll_va = m.nll(m.pi(lam=lam_va), torch.as_tensor(yv, device=args.device),
                       tmk).item() / float(tmk.sum())
    print(f"[{tag}] val NLL (lambda refit, 全窗口) = {nll_va:.6f}")
    return m, lam_va, lab, nll_va


def save(m, lam_va, d, path):
    torch.save(dict(state=m.state_dict(), lam_va=lam_va.cpu(), K=m.K,
                    dis_names=list(d["dis_names"]), cov_names=list(d["cov_names"]),
                    age_grid=d["age_grid"]), path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lam-steps", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--lr-lambda", type=float, default=0.05)
    ap.add_argument("--alpha-lambda", type=float, default=1.0)
    ap.add_argument("--alpha-phi", type=float, default=1.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=0)
    ap.add_argument("--only-fold", type=int, default=-1,
                    help="只跑这一折（Slurm 数组任务用）。折的划分由 --seed 决定，与是否只跑一折无关。")
    ap.add_argument("--sweep-k", default="")
    ap.add_argument("--use-delphi", action="store_true",
                    help="把 Delphi 的隐状态加进 lambda 的 GP 先验均值（需先跑 delphi_states.py）")
    ap.add_argument("--w-ridge", type=float, default=1.0, help="W 的 L2 惩罚")
    args = ap.parse_args()
    os.makedirs(RES, exist_ok=True)
    torch.manual_seed(args.seed)
    d = load()

    if args.sweep_k:
        out = {}
        for K in [int(x) for x in args.sweep_k.split(",")]:
            m, lam, lab, nll = run_once(d, K, args, tag=f"sweepK{K}")
            out[K] = nll
            save(m, lam, d, os.path.join(RES, f"fit_K{K}.pt"))
        json.dump(out, open(os.path.join(RES, "sweep_k.json"), "w"), indent=2)
        print("\nK -> val NLL:", json.dumps(out, indent=2))
        return

    if args.folds:
        N = len(d["split"])
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(N)
        fold = np.array_split(perm, args.folds)
        todo = range(args.folds) if args.only_fold < 0 else [args.only_fold]
        for f in todo:
            va = fold[f]
            tr = np.concatenate([fold[j] for j in range(args.folds) if j != f])
            m, lam, lab, nll = run_once(d, args.k, args, seed=args.seed + f,
                                        tr=tr, va=va, tag=f"fold{f}", fold=f)
            sfx = "_dh" if args.use_delphi else ""
            save(m, lam, d, os.path.join(RES, f"fold{f}_K{args.k}{sfx}.pt"))
        return

    m, lam, lab, nll = run_once(d, args.k, args)
    save(m, lam, d, os.path.join(RES, f"fit_K{args.k}.pt"))


if __name__ == "__main__":
    main()
