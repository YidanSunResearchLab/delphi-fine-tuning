# -*- coding: utf-8 -*-
"""消融：把 logit 对照补强到和 ALADYNOULLI 同样的输入。

上一轮的 logit 只给了 7 个协变量，而 ALADYNOULLI 的先验均值 r + Gamma^T g 吃的是全部 84 个
（性别 + 30 个背景 token + 52 个"基线即患病"指示）。而且消融已经证明 ALADYNOULLI 的预测力
基本全来自这个先验均值（去掉逐人 lambda 拟合后 AUC 不变）。所以公平的对照是：

    同样的 84 个输入，每个病一个独立的 L2 逻辑回归，不共享任何结构。

如果它追平 noref，说明低秩 signature 结构对**预测**没有额外贡献，价值全在可解释性；
如果追不平，那 52 个病之间的信息共享 + GP 年龄曲线是真的在起作用。

    python3 ablate_logit_all.py --fold 0
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from evaluate import fold_split, landmark_sets
from fit import load

HERE = os.path.dirname(os.path.abspath(__file__)); RES = os.path.join(HERE, "results")
ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True)
ap.add_argument("--horizons", default="1,3,5")
ap.add_argument("--landmarks", default="75,80,85,90")
ap.add_argument("--C", type=float, default=0.1)
a = ap.parse_args()

d = load(); T = len(d["age_grid"]); A0 = int(d["age_grid"][0]); D = d["E"].shape[1]
fs = fold_split(len(d["split"]), 10, 0)
te = fs[a.fold]; tr = np.concatenate([fs[j] for j in range(10) if j != a.fold])
out = {}
for H in [int(x) for x in a.horizons.split(",")]:
    for t0 in [int(x) - A0 for x in a.landmarks.split(",")]:
        rte, lte = landmark_sets(d["E"][te], d["S"][te], d["atrisk"][te], d["Yobs"][te], t0, H, T)
        if rte is None: continue
        rtr, ltr = landmark_sets(d["E"][tr], d["S"][tr], d["atrisk"][tr], d["Yobs"][tr], t0, H, T)
        # 全部 84 个协变量 + 该 landmark 处的随访时长（和 ALADYNOULLI 的 g 同一组，外加时长）
        X = lambda ix: np.hstack([d["G"][ix], (t0 - d["S"][ix].astype(float))[:, None]])
        Xtr, Xte = X(tr), X(te)
        S = np.zeros((len(te), D))
        for j in range(D):
            mm = rtr[:, j]; y = ltr[mm, j].astype(int)
            if mm.sum() < 50 or y.sum() < 10 or y.sum() == mm.sum(): continue
            mdl = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=5000, C=a.C))
            mdl.fit(Xtr[mm], y)
            S[:, j] = mdl.predict_proba(Xte)[:, 1]
        out[f"{H}_{t0}_logitall"] = S
        out[f"{H}_{t0}_risk"] = rte; out[f"{H}_{t0}_label"] = lte
p = os.path.join(RES, f"cvraw_f{a.fold}_logitall.npz")
np.savez_compressed(p, **out, meta=np.array(json.dumps(dict(n_test=len(te), C=a.C))))
print(f"-> {p}")
