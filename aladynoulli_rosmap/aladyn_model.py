# -*- coding: utf-8 -*-
"""ALADYNOULLI (Urbut et al., Nature 2026) 的 PyTorch 实现，按论文方程写。

    pi_idt = kappa * sum_k theta_ikt * sigmoid(phi_kdt)
    theta_ikt = softmax_k(lambda_ikt)
    lambda_ik ~ GP(r_k + Gamma_k^T g_i,  Omega_lambda)
    phi_kd    ~ GP(mu_d + psi_kd,        Omega_phi)
    Omega(t,t') = alpha^2 exp(-(t-t')^2 / (2 l^2))

    NLL = -sum_i sum_d [ sum_{t<E_id} log(1-pi_idt)
                         + Y_id log pi_id(E) + (1-Y_id) log(1-pi_id(E)) ]
    L   = NLL + GP_lambda + GP_phi          (MAP, 梯度下降)

论文用的是论文自己的 UKB 设置；这里有三处按 ROSMAP 改了，改的理由写在 data.py 顶部：
风险窗口从每个人自己的 S_i 开始（左截断）、基线患病者移出风险集、炎症标志物剔除。

代码里只有一处是论文没写死、我自己定的：mu_d 取经验的年龄别 hazard 的 logit（论文说
"logit of population prevalence"，但没说是否随年龄变；ROSMAP 60-105 岁跨度上 hazard 随
年龄变化极大，取常数会让 phi 去承担全部年龄趋势，GP 先验会把它压平）。set_mu 里用
--mu-mode 可以切回常数。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-7


def rbf(T, alpha, ell, device, jitter=1e-4):
    t = torch.arange(T, dtype=torch.float64, device=device)
    d2 = (t[:, None] - t[None, :]) ** 2
    K = (alpha ** 2) * torch.exp(-d2 / (2.0 * ell ** 2))
    K = K + torch.eye(T, dtype=torch.float64, device=device) * (jitter * alpha ** 2)
    return K


class Aladynoulli(nn.Module):
    def __init__(self, N, D, T, P, K=8, *,
                 alpha_lambda=1.0, ell_lambda=None,
                 alpha_phi=1.0, ell_phi=None, n_h=0,
                 device="cpu", dtype=torch.float32):
        super().__init__()
        self.N, self.D, self.T, self.P, self.K = N, D, T, P, K
        self.n_h = n_h
        ell_lambda = ell_lambda if ell_lambda is not None else T / 4.0   # 论文 l_lambda = T/4
        ell_phi = ell_phi if ell_phi is not None else T / 3.0            # 论文 l_phi    = T/3

        for nm, (a, l) in dict(lam=(alpha_lambda, ell_lambda), phi=(alpha_phi, ell_phi)).items():
            Kmat = rbf(T, a, l, device)
            self.register_buffer(f"Kinv_{nm}", torch.linalg.inv(Kmat).to(dtype))

        self.lam = nn.Parameter(torch.zeros(N, K, T, dtype=dtype))
        self.phi = nn.Parameter(torch.zeros(K, D, T, dtype=dtype))
        self.psi = nn.Parameter(torch.zeros(K, D, dtype=dtype))
        self.r = nn.Parameter(torch.zeros(K, dtype=dtype))
        self.Gamma = nn.Parameter(torch.zeros(K, P, dtype=dtype))
        self.kappa_raw = nn.Parameter(torch.tensor(3.0, dtype=dtype))     # sigmoid -> 0.95
        self.register_buffer("mu", torch.zeros(D, T, dtype=dtype))
        # W: Delphi 隐状态 -> lambda 先验均值。n_h=0 时不存在，模型退回论文原式。
        self.W = nn.Parameter(torch.zeros(K, n_h, dtype=dtype)) if n_h else None

    # ---------- 初始化 ----------
    def set_mu(self, events, atrisk_t, mode="age"):
        """mu_d(t) = logit( 经验年龄别 hazard )。events/atrisk_t 是 (D,T) 的计数。"""
        if mode == "const":
            h = events.sum(1) / np.maximum(atrisk_t.sum(1), 1.0)
            h = np.clip(h, 1e-4, 0.5)[:, None].repeat(self.T, 1)
        else:
            # 年龄别 hazard，先按 5 年窗做移动平均再 logit（原始计数在网格两端很薄）
            ev = np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, "same"), 1, events)
            ar = np.apply_along_axis(lambda v: np.convolve(v, np.ones(5) / 5, "same"), 1, atrisk_t)
            h = np.clip(ev / np.maximum(ar, 1.0), 1e-4, 0.5)
        self.mu.copy_(torch.as_tensor(np.log(h / (1 - h)), dtype=self.mu.dtype))

    @torch.no_grad()
    def init_from_clusters(self, cooc, G, seed=0, in_val=1.0, out_val=-2.0, jitter=0.1):
        """论文的初始化：疾病共现矩阵谱聚类 -> psi 的 in/out cluster 赋值；lam/phi 用
        降幅的 GP 样本。"""
        from sklearn.cluster import SpectralClustering
        g = torch.Generator().manual_seed(seed)
        lab = SpectralClustering(n_clusters=self.K, affinity="precomputed",
                                 random_state=seed, assign_labels="kmeans"
                                 ).fit_predict(cooc)
        psi = torch.full((self.K, self.D), out_val)
        psi[torch.as_tensor(lab), torch.arange(self.D)] = in_val
        self.psi.copy_(psi.to(self.psi.dtype) + jitter * torch.randn(self.K, self.D, generator=g))

        self.Gamma.copy_(0.01 * torch.randn(self.K, self.P, generator=g))
        self.r.zero_()
        # lam 初值 = 先验均值 + 小扰动；phi 同理
        if self.W is not None:
            self.W.copy_(0.01 * torch.randn(self.K, self.n_h, generator=g))
        Gt = torch.as_tensor(G, dtype=self.lam.dtype, device=self.lam.device)
        m_lam = self.r[None, :, None] + (Gt @ self.Gamma.T)[:, :, None]
        self.lam.copy_(m_lam + jitter * torch.randn(self.N, self.K, self.T, generator=g))
        m_phi = self.mu[None] + self.psi[:, :, None]
        self.phi.copy_(m_phi + jitter * torch.randn(self.K, self.D, self.T, generator=g))
        return lab

    def prior_mean(self, G, Hs=None):
        """lambda 的 GP 先验均值 (n,K,T)。

        论文: r_k + Gamma_k^T g_i           —— 不随 t 变
        加上 Delphi: + W_k^T h_i(t)          —— 随 t 变，这是历史进来的唯一通道
        """
        m = (self.r[None, :, None] + (G @ self.Gamma.T)[:, :, None]).expand(-1, -1, self.T)
        if self.W is not None and Hs is not None:
            m = m + torch.einsum("ntp,kp->nkt", Hs, self.W)
        return m

    # ---------- 前向 ----------
    def pi(self, idx=None, lam=None):
        lam = self.lam if lam is None else lam
        if idx is not None and lam is self.lam:
            lam = lam[idx]
        theta = F.softmax(lam, dim=1)                       # (n,K,T)
        sig = torch.sigmoid(self.phi)                       # (K,D,T)
        kappa = torch.sigmoid(self.kappa_raw)
        p = kappa * torch.einsum("nkt,kdt->ndt", theta, sig)
        return p.clamp(EPS, 1.0 - EPS)

    # ---------- 损失 ----------
    # 论文的目标是 L = NLL_sum + GP_lambda_sum + GP_phi_sum（全部是 sum，不是 mean）。
    # 下面三个函数都返回 sum；调用方统一除以 mask.sum() 只为让学习率和 K/N 无关，
    # 那是对整个目标的等比缩放，不改变 MAP 解。GP 的相对强度完全由 alpha 控制，和论文一致。
    @staticmethod
    def nll(p, y, mask):
        """masked Bernoulli。mask=风险集指示 (S_i<=t<=E_id 且 atrisk)，y 只在 t=E 且事件时为 1。"""
        ll = y * torch.log(p) + (1 - y) * torch.log1p(-p)
        return -(ll * mask).sum()

    def gp_lambda(self, G, lam=None, Hs=None):
        lam = self.lam if lam is None else lam
        r_ = lam - self.prior_mean(G, Hs)                   # (n,K,T)
        return 0.5 * torch.einsum("nkt,ts,nks->", r_, self.Kinv_lam, r_)

    def gp_phi(self):
        r_ = self.phi - (self.mu[None] + self.psi[:, :, None])
        return 0.5 * torch.einsum("kdt,ts,kds->", r_, self.Kinv_phi, r_)


def build_mask(E, S, atrisk, Yobs, T, horizon=None):
    """(N,D,T) 的风险集 mask 和事件指示 y。

    mask_idt = 1  当  S_i <= t <= E_id  且  atrisk_id = 1
    y_idt    = 1  当  t = E_id  且  Yobs_id = 1
    horizon 不为 None 时，只保留 t <= horizon 的部分（用于 landmark 预测时的 lambda 重拟合，
    保证只用到预测时点之前的信息）。
    """
    E = E.astype(np.int64); S = S.astype(np.int64)
    N, D = E.shape
    t = np.arange(T)[None, None, :]
    hi = E[:, :, None] if horizon is None else np.minimum(E[:, :, None], horizon)
    mask = ((t >= S[:, None, None]) & (t <= hi) & (atrisk[:, :, None] == 1)).astype(np.float32)
    y = np.zeros((N, D, T), np.float32)
    ii, dd = np.nonzero(Yobs == 1)
    y[ii, dd, E[ii, dd]] = 1.0
    y *= mask                                                # horizon 之外的事件不算
    return mask, y
