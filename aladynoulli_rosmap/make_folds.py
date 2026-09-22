# -*- coding: utf-8 -*-
"""给 Delphi 造 10 折数据集，折的划分和 ALADYNOULLI 的完全一致。

头对头比较只有在两个模型看到的训练人群、留出人群逐人一致时才成立。ALADYNOULLI 的折来自
evaluate.fold_split(N, 10, seed=0)，这里用同一个函数，所以两边的第 f 折是同一批 projid。

每折写出 `_delphi/data/rosmap_f{f}/{train,val}.bin`：

    train.bin  = 9 个训练折里的 90%
    val.bin    = 9 个训练折里的 10%     <- 只用于 Delphi 的 checkpoint 选择
    留出的第 f 折 = 两个文件都不进       <- 打分时才第一次被看到

最后这条是必须的：train.py 里 `always_save_checkpoint=False`，它按 val loss 选 checkpoint。
把留出折当 val，就等于在测试集上早停，比较立刻失效。

    python3 make_folds.py
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import numpy as np

from evaluate import fold_split

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = os.path.join(HERE, "_tokenization", "data_rosmap")
if not os.path.isdir(TOK):
    TOK = os.path.normpath(os.path.join(HERE, "..", "tokenization", "data_rosmap"))
OUT = os.path.join(HERE, "_delphi", "data")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-val", type=float, default=0.1)
    args = ap.parse_args()

    arr = np.vstack([np.fromfile(os.path.join(TOK, f"{s}.bin"), dtype=np.uint32).reshape(-1, 3)
                     for s in ("train", "val")])
    pids = np.unique(arr[:, 0])
    fs = fold_split(len(pids), args.folds, args.seed)
    rng = np.random.default_rng(args.seed + 1000)

    print(f"总人数 {len(pids)},  {args.folds} 折")
    for f in range(args.folds):
        te = pids[fs[f]]
        tr_all = pids[np.concatenate([fs[j] for j in range(args.folds) if j != f])]
        perm = rng.permutation(len(tr_all))
        n_val = int(args.inner_val * len(tr_all))
        va, tr = tr_all[perm[:n_val]], tr_all[perm[n_val:]]
        assert not (set(te) & set(tr)) and not (set(te) & set(va))

        d = os.path.join(OUT, f"rosmap_f{f}")
        os.makedirs(d, exist_ok=True)
        for name, sel in (("train", tr), ("val", va)):
            rows = arr[np.isin(arr[:, 0], sel)]
            rows.astype(np.uint32).tofile(os.path.join(d, f"{name}.bin"))
        for extra in ("labels.csv", "meta.json"):
            shutil.copy(os.path.join(TOK, extra), os.path.join(d, extra))
        print(f"  fold{f}: delphi-train {len(tr)} 人 / delphi-val {len(va)} 人 / "
              f"留出 {len(te)} 人（不在任何 .bin 里）")
    print(f"\n-> {OUT}/rosmap_f*/")


if __name__ == "__main__":
    main()
