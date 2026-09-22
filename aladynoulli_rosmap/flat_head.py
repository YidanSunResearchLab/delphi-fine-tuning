# -*- coding: utf-8 -*-
"""对照：同一个冻住的 Delphi 隐状态上挂一个**扁平头**，没有 signature 结构。

    logit(pi_idt) = mu_d(t) + h_i(t)^T beta_d + g_i^T gamma_d

和 aladyndh 的差别**只有一处**：没有低秩瓶颈、没有 GP 平滑、没有 softmax 混合。
输入（Delphi 隐状态 h + 84 个基线协变量 g）、损失（同一个离散时间生存 NLL 和同一套 mask）、
mu_d(t)（同一份经验年龄别 hazard）、打分（同一个 surv_score）、折划分，全部一致。

为什么论文需要这个对照：贝叶斯层同时改了两件事——加了 signature 结构，**并且**把输出从
Delphi 的 MC rollout 换成了逐年解析 hazard。如果不控制后者，"贝叶斯层有用"这个结论
可能只是"换了输出头有用"，而后者是个便宜得多、不需要任何贝叶斯机制的改动。

三列一起读：
    delphi        Delphi + MC rollout            骨架自己
    delphiflat    Delphi + 逐年 hazard 扁平头     换了输出头，没有结构
    aladyndh      Delphi + 逐年 hazard + signature 结构

    delphiflat - delphi    = 换输出头的贡献
    aladyndh  - delphiflat = signature 结构的贡献   <- 论文真正要 claim 的那个

    python3 flat_head.py --fold 0
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn

from aladyn_model import build_mask
from evaluate import fold_split, landmark_sets, surv_score
from fit import causal_h, hazard_counts, load, load_dstates

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
EPS = 1e-7


class FlatHead(nn.Module):
    def __init__(self, D, T, n_h, P):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros(D, n_h))
        self.gamma = nn.Parameter(torch.zeros(D, P))
        self.register_buffer("mu", torch.zeros(D, T))

    def set_mu(self, ev, ar):
        h = np.clip(
            np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, "same"), 1, ev)
            / np.maximum(np.apply_along_axis(
                lambda v: np.convolve(v, np.ones(5) / 5, "same"), 1, ar), 1.0), 1e-4, 0.5)
        self.mu.copy_(torch.as_tensor(np.log(h / (1 - h)), dtype=self.mu.dtype))

    def pi(self, Hs, G):
        z = (self.mu[None]                                        # (1,D,T)
             + torch.einsum("ntp,dp->ndt", Hs, self.beta)         # (n,D,T)
             + (G @ self.gamma.T)[:, :, None])                    # (n,D,1)
        return torch.sigmoid(z).clamp(EPS, 1 - EPS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=0.02)
    ap.add_argument("--l2", type=float, default=1.0)
    ap.add_argument("--es-every", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--horizons", default="1,3,5")
    ap.add_argument("--landmarks", default="75,80,85,90")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    d = load()
    T = len(d["age_grid"]); A0 = int(d["age_grid"][0]); D = d["E"].shape[1]
    fs = fold_split(len(d["split"]), 10, 0)
    te = fs[a.fold]; tr = np.concatenate([fs[j] for j in range(10) if j != a.fold])

    Htr = causal_h(load_dstates(a.fold, tr, a.device))      # 训练：右移一格，见 fit.causal_h
    Hte_raw = load_dstates(a.fold, te, a.device)             # 打分时按 landmark 逐个处理
    Gtr = torch.as_tensor(d["G"][tr], device=a.device)
    Gte = torch.as_tensor(d["G"][te], device=a.device)

    mask, y = build_mask(d["E"][tr], d["S"][tr], d["atrisk"][tr], d["Yobs"][tr], T)
    ev, ar = hazard_counts(d["E"][tr], d["S"][tr].astype(np.int64), d["atrisk"][tr],
                           d["Yobs"][tr], T)
    m = FlatHead(D, T, Htr.shape[2], Gtr.shape[1]).to(a.device)
    m.set_mu(ev, ar)

    tm = torch.as_tensor(mask, device=a.device); ty = torch.as_tensor(y, device=a.device)
    # 和 fit_stage1 同样的内部早停切分
    rng = np.random.default_rng(a.fold)
    es = np.zeros(len(tr), bool)
    es[rng.choice(len(tr), max(1, int(0.1 * len(tr))), replace=False)] = True
    tes = torch.as_tensor(es, device=a.device).float()[:, None, None]
    fit_m, es_m = tm * (1 - tes), tm * tes
    ncell, n_es = float(fit_m.sum()), float(es_m.sum())

    def nll(p, msk):
        return -((ty * torch.log(p) + (1 - ty) * torch.log1p(-p)) * msk).sum()

    opt = torch.optim.Adam(m.parameters(), lr=a.lr)
    best, best_it, bad, best_sd = float("inf"), 0, 0, None
    for it in range(a.steps):
        opt.zero_grad(set_to_none=True)
        p = m.pi(Htr, Gtr)
        loss = (nll(p, fit_m) + a.l2 * ((m.beta ** 2).sum() + (m.gamma ** 2).sum())) / ncell
        loss.backward(); opt.step()
        if (it + 1) % a.es_every == 0 or it == a.steps - 1:
            with torch.no_grad():
                v = nll(m.pi(Htr, Gtr), es_m).item() / n_es
            if v < best - 1e-6:
                best, best_it, bad = v, it, 0
                best_sd = {k: t.detach().clone() for k, t in m.state_dict().items()}
            else:
                bad += 1
            print(f"  it {it+1:5d}  obj {loss.item():.6f}  es/cell {v:.6f}"
                  f"{'  *' if bad == 0 else ''}", flush=True)
            if bad >= a.patience:
                print(f"  早停于 it={it+1}，最好在 it={best_it+1} (es/cell={best:.6f})")
                break
    if best_sd is not None:
        m.load_state_dict(best_sd)
    m.eval()

    out = {}
    with torch.no_grad():
        for t0 in [int(x) - A0 for x in a.landmarks.split(",")]:
            # 和 aladyndh 完全一样的 causal_h 处理
            p = m.pi(causal_h(Hte_raw, t0), Gte).cpu().numpy()
            for H in [int(x) for x in a.horizons.split(",")]:
                risk, lab = landmark_sets(d["E"][te], d["S"][te], d["atrisk"][te],
                                          d["Yobs"][te], t0, H, T)
                if risk is None:
                    continue
                out[f"{H}_{t0}_delphiflat"] = surv_score(p, t0, H)
                out[f"{H}_{t0}_risk"] = risk
                out[f"{H}_{t0}_label"] = lab

    torch.save(m.state_dict(), os.path.join(RES, f"flat_f{a.fold}.pt"))
    pth = os.path.join(RES, f"cvraw_f{a.fold}_delphiflat.npz")
    np.savez_compressed(pth, **out, meta=np.array(json.dumps(
        dict(n_test=len(te), l2=a.l2, best_it=best_it + 1, es=best))))
    print(f"-> {pth}")


if __name__ == "__main__":
    main()
