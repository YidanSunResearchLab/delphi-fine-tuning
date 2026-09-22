# -*- coding: utf-8 -*-
"""抽取 Delphi 的隐状态 h_i(t)，喂给 ALADYNOULLI 的 lambda 先验均值。

    lambda_ik(t) ~ GP( r_k + Gamma_k^T g_i + W_k^T h_i(t),  Omega_lambda )

h_i(t) = Delphi 在这个人**网格下标 <= t 的最后一个 token** 处、最后一层 ln_f 之后的隐状态
（n_embd=96 维）。Delphi 全程冻住，只当特征提取器。

== 为什么一次 forward 就能拿到所有时点的 h ==

Delphi 的注意力是因果的（model.py 里 attn_mask 乘了 tril），所以第 j 个位置的隐状态只依赖
0..j。把一个人的完整序列喂进去一次，就同时得到了他在每一个前缀下的表示，不存在未来信息
泄漏。不需要为每个 landmark 重跑一次。

== 折的对应必须严格 ==

第 f 折的 h 必须用**第 f 折的 Delphi checkpoint**（它训练时没见过第 f 折的人）。用全量
checkpoint 会把留出折的信息通过 h 漏进 ALADYNOULLI。

输出: results/dstates_f{f}.npz  ->  H (N, T, n_embd) float16, 按 tensors.npz 的 pids 顺序
      网格下标 < 该人基线的位置填 0（那段没有观测，lambda 的先验均值退回 r + Gamma^T g）

    python3 delphi_states.py --fold 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
for p in (os.path.join(HERE, "_delphi"),):
    if p not in sys.path:
        sys.path.insert(0, p)

from fit import load                                      # noqa: E402

DAYS_PER_YEAR = 365.25


def tok_dir():
    d = os.path.join(HERE, "_tokenization", "data_rosmap")
    return d if os.path.isdir(d) else os.path.normpath(
        os.path.join(HERE, "..", "tokenization", "data_rosmap"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from model import Delphi, DelphiConfig                 # _delphi/model.py

    d = load()
    T = len(d["age_grid"]); A0 = int(d["age_grid"][0])
    pids = d["pids"]

    ck = torch.load(os.path.join(HERE, "_delphi", f"Delphi-fold{a.fold}", "ckpt.pt"),
                    map_location=a.device, weights_only=False)
    args = dict(ck["model_args"])
    valid = {f for f in DelphiConfig.__dataclass_fields__}
    model = Delphi(DelphiConfig(**{k: v for k, v in args.items() if k in valid}))
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in ck["model"].items()})
    model.to(a.device).eval()
    nemb = args["n_embd"]; B = args["block_size"]

    # ln_f 之后的隐状态，用 hook 取，不改上游 model.py
    box = {}
    model.transformer.ln_f.register_forward_hook(lambda m, i, o: box.__setitem__("h", o))

    arr = np.vstack([np.fromfile(os.path.join(tok_dir(), f"{s}.bin"), dtype=np.uint32).reshape(-1, 3)
                     for s in ("train", "val")])
    order = np.argsort(arr[:, 0], kind="stable"); arr = arr[order]
    upid = np.unique(arr[:, 0])
    bnd = np.searchsorted(arr[:, 0], upid, side="left").tolist() + [len(arr)]
    rows_of = {int(p): arr[bnd[i]:bnd[i + 1]] for i, p in enumerate(upid)}

    H = np.zeros((len(pids), T, nemb), np.float16)
    with torch.no_grad():
        for n, pid in enumerate(pids):
            r = rows_of[int(pid)]
            o = np.argsort(r[:, 1], kind="stable")
            toks = r[o, 2].astype(np.int64) + 1           # disk -> model
            ages = r[o, 1].astype(np.float64)
            toks, ages = toks[:B], ages[:B]
            idx = torch.as_tensor(toks[None], dtype=torch.long, device=a.device)
            age = torch.as_tensor(ages[None], dtype=torch.float32, device=a.device)
            model(idx, age)
            h = box["h"][0].float().cpu().numpy()          # (L, n_embd)

            bl = int(r[:, 1].min())
            idx0 = int(round(bl / DAYS_PER_YEAR)) - A0
            gi = idx0 + np.rint((ages - bl) / 365.0).astype(np.int64)   # 每个 token 的网格下标
            # 对每个 t，取 gi <= t 的最后一个位置
            last = np.searchsorted(gi, np.arange(T), side="right") - 1
            ok = last >= 0
            H[n, ok] = h[last[ok]].astype(np.float16)
            if n % 500 == 0:
                print(f"  {n}/{len(pids)}", flush=True)

    p = os.path.join(RES, f"dstates_f{a.fold}.npz")
    np.savez_compressed(p, H=H, meta=np.array(json.dumps(
        dict(fold=a.fold, n_embd=nemb, T=T, A0=A0))))
    nz = (np.abs(H).sum(2) > 0).mean()
    print(f"\n-> {p}   非零(有观测)的 (人,年) 格子占比 {nz:.3f}")


if __name__ == "__main__":
    main()
