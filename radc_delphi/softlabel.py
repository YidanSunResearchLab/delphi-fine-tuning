"""
softlabel.py -- turn raw scale readings into soft weight vectors, vectorised over a batch.

One call site in training (batching.get_batch) and one in the tokenizer, both reading the same
tables out of radc_delphi.vocab, so the representation the model is trained on and the
representation the .bin records can never drift apart.

WHAT A SOFT WEIGHT IS. For an ordinal scale with edges e_1 < ... < e_{K-1} and a measurement
whose noise SD at that reading is sigma,

    w_k = Phi((e_k - v) / sigma) - Phi((e_{k-1} - v) / sigma)

is P(the TRUE value lies in bin k | this noisy observation), under a flat prior. It is a
posterior, not a heuristic softening, and that is what makes the training target meaningful:
minimising -sum_k w_k log p_k drives the model's softmax to E[w | history], which by the
tower property equals P(true value in bin k | history). The model therefore learns the
predictive distribution over the TRUE value with measurement noise deconvolved out, rather
than being asked to predict the next noisy reading.

WHY IT IS COMPUTED HERE AND NOT STORED. Storing K floats per event would bake sigma and the
edges into the .bin, so changing either would mean rebuilding the dataset. Storing the single
raw value instead keeps the .bin a record of what was measured, and lets the binning be a
modelling decision that can be changed without touching the data.

Everything is torch so the erf is a vectorised kernel; a python-level math.erf over a
128 x 64 batch is ~50k calls and shows up in the profile.
"""
import math

import numpy as np
import torch

from . import vocab as V

_SQRT2 = math.sqrt(2.0)


class SoftBinner:
    """Precomputed tables + the batch call. Build once, reuse for every batch."""

    def __init__(self, device="cpu"):
        tok2scale, edges, ids, sig_x, sig_y = V.scale_tables()
        self.tok2scale_np = tok2scale
        self.sig_x, self.sig_y = sig_x, sig_y
        self.tok2scale = torch.as_tensor(tok2scale, device=device)
        # sigma anchors as tensors so the interpolation stays on-device. The numpy path forced
        # a .cpu() sync on every batch, which is free on CPU and costs a full pipeline stall
        # per step on a GPU.
        self.sig_xt = torch.as_tensor(sig_x, dtype=torch.float32, device=device)
        self.sig_yt = torch.as_tensor(sig_y, dtype=torch.float32, device=device)
        self.edges = torch.as_tensor(edges, dtype=torch.float32, device=device)
        self.ids = torch.as_tensor(ids, device=device)
        self.K = V.MAX_LEVELS
        self.device = device

    def sigma_t(self, scale_sel, v):
        """Measured sigma per position, piecewise-linear in the reading, entirely on-device.

        Equivalent to np.interp with flat extrapolation: the anchor arrays are padded by
        repeating their last point, and the interpolation weight is clamped to [0, 1], so a
        reading outside the measured range takes the nearest anchor's sigma rather than an
        extrapolated one.
        """
        xs = self.sig_xt.to(v.device)[scale_sel]              # (n, A)
        ys = self.sig_yt.to(v.device)[scale_sel]
        A = xs.shape[1]
        j = torch.searchsorted(xs.contiguous(), v[:, None].contiguous()).clamp(1, A - 1)
        x0 = xs.gather(1, j - 1).squeeze(1); x1 = xs.gather(1, j).squeeze(1)
        y0 = ys.gather(1, j - 1).squeeze(1); y1 = ys.gather(1, j).squeeze(1)
        t = ((v - x0) / (x1 - x0).clamp(min=1e-9)).clamp(0.0, 1.0)
        return y0 + t * (y1 - y0)

    def __call__(self, tokens, values):
        """tokens (B, T) MODEL-space long; values (B, T) float with NaN off-scale.

        Returns (W, I), both (B, T, K):
            W  weights summing to 1 along the last axis
            I  the MODEL ids those weights apply to, -1 where padded

        A non-scale position gets W = [1, 0, ...] and I = [its own token, -1, ...], so the
        caller needs no branch: the same gather-and-weight expression reproduces a plain
        embedding lookup exactly. Padding (token 0) is included in that and stays a no-op.
        """
        B, T = tokens.shape
        dev = tokens.device
        flat_t = tokens.reshape(-1)
        si = self.tok2scale.to(dev)[flat_t]                       # (BT,) -1 off-scale
        is_scale = si >= 0

        W = torch.zeros(B * T, self.K, device=dev, dtype=torch.float32)
        I = torch.full((B * T, self.K), -1, device=dev, dtype=torch.long)
        # default: one-hot on the token itself
        W[:, 0] = 1.0
        I[:, 0] = flat_t

        if bool(is_scale.any()):
            idx = is_scale.nonzero(as_tuple=True)[0]
            s_sel = si[idx]
            v = values.reshape(-1)[idx]
            # a scale token with no recorded value falls back to one-hot, which is what a
            # dataset built before soft labels produces
            good = torch.isfinite(v)
            idx, s_sel, v = idx[good], s_sel[good], v[good]
            if idx.numel():
                sig = self.sigma_t(s_sel, v)
                e = self.edges.to(dev)[s_sel]                     # (n, K-1)
                z = (e - v[:, None]) / sig[:, None].clamp(min=1e-6)
                cdf = 0.5 * (1.0 + torch.erf(z / _SQRT2))         # (n, K-1); +inf edges -> 1
                ones = torch.ones(cdf.shape[0], 1, device=dev)
                zeros = torch.zeros(cdf.shape[0], 1, device=dev)
                w = torch.diff(torch.cat([zeros, cdf, ones], dim=1), dim=1)   # (n, K)
                W[idx] = w
                I[idx] = self.ids.to(dev)[s_sel]
        return W.view(B, T, self.K), I.view(B, T, self.K)


def mix_embeddings(wte, W, I):
    """Soft input embedding: sum_k W_k * wte(I_k).

    Reduces EXACTLY to wte(token) wherever W is one-hot, because the padded slots carry
    weight 0 and their id is clamped to 0 before the lookup.
    """
    safe = I.clamp(min=0)
    emb = wte(safe)                                   # (B, T, K, D)
    return (W.unsqueeze(-1) * emb).sum(-2)


def scatter_targets(W, I, vocab_size):
    """Soft cross-entropy target: a (B, T, vocab_size) distribution from the (B, T, K) pair."""
    B, T, K = W.shape
    out = torch.zeros(B, T, vocab_size, device=W.device, dtype=W.dtype)
    out.scatter_add_(2, I.clamp(min=0), W * (I >= 0).to(W.dtype))
    return out
