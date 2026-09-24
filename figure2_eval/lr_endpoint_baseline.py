"""
lr_endpoint_baseline.py -- figure2 panel a/b 的终点上，逻辑回归 vs transformer。

    python lr_endpoint_baseline.py --datasets rosmap rosmap_nodedup rosmap_fullvisit

window_baselines.py 只证明了在**下一次访视的认知分箱**上，一个单访视 logistic 打平/打赢
transformer。但 figure2 报的是**5 年 AD / 死亡 / 到达 ≥Borderline**，那上面从来没有过
逻辑回归对照。这个脚本补上。它决定"把线性部分放进架构的哪里"：
如果 transformer 在长程事件上赢，就只给状态转移加线性头；如果 LR 也赢，要重谈的是整个架构。

--------------------------------------------------------------------------------------
公平性：LR 拿到的输入 = transformer rollout 的起始前缀，一个 token 不多不少

`figure2_core._baseline()` 定义 prompt = 年龄 <= 首次访视的全部 token（statics + 整个基线访视），
rollout 就是从它开始采的。这里 LR 的特征就是**同一批 token 的多热向量** + 基线年龄。
所以比较的是模型，不是信息量 —— 不存在"LR 偷看了更多东西"。

**评估口径一字不改**，直接复用 figure2 自己的函数：
  `_at_risk`（谁能被评分）、`labels_at_h`（竞争风险 + 右删失感知的二元标签，-1 丢弃）、
  `aalen_johansen`（观测累积发生率）、`boot_auc`（AUC + bootstrap CI，一行一人）。
唯一被替换的是 `ep["risk"][h]` —— 从 MC 轨迹的比例换成 LR 的预测概率。

LR 在 **train** split 上拟合（同一个 horizon、同一个 at-risk 规则、同样丢掉 y=-1），
在 val 上评。三份分词各拟合一套，因为各自的 frame（可评估人、观测首达时间）略有不同。
"""
import os
import sys
import json
import argparse
import warnings

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "figure2"))
from sklearn.linear_model import LogisticRegression          # noqa: E402
from sklearn.exceptions import ConvergenceWarning            # noqa: E402
from radc_delphi import vocab as V, batching as Bt           # noqa: E402
import figure2.radc_states as S                              # noqa: E402
import figure2.perdomain as PD                               # noqa: E402
import figure2.figure2_core as F2                            # noqa: E402
import figure2.figure2_panels as FP                          # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
D = 365.25


def bind(labels_csv):
    """把 radc_states / figure2_core 绑到这份词表上 —— 就是 F2._bind 去掉 engine 的部分。

    不加载 checkpoint：这个脚本不需要模型，只需要 token 表和观测结果。
    """
    res = V.resolve_csv(labels_csv)
    S.configure(res.NAMES)
    PD.configure()
    F2.STATE_NAMES, F2.ALL_NAMES = S.GRID_NAMES, S.ALL_NAMES
    F2.DEATH, F2.AD_DX, F2.GRID_TOKENS = S.DEATH, S.AD_DX, S.GRID_TOKENS
    F2.NSTATE, F2.NSLOT = S.NSTATE, S.NSLOT
    F2.DEATH_IDX, F2.AD_IDX = S.DEATH_IDX, S.AD_IDX
    F2._VISIT_IGNORE = set(res.IGNORE_TOKENS) | {res.PADDING}
    return res


def frame_lite(data_dir, split, res):
    """每个可评估病人的观测量 + 基线特征。**不跑任何 MC**。

    结构与 figure2_core._process 的非 MC 部分逐行一致（baseline / fp_obs / followup /
    baseline_<scale>），所以 labels_at_h / cr_times / _at_risk 拿到的东西和 panel a 一样。
    """
    d = np.fromfile(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
    p2i = Bt.get_p2i(d)
    rows, feats = [], []
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        ages, toks = Bt.patient_stream(d, s, n)
        bp = F2._baseline(ages, toks)
        if bp is None:
            continue
        base, pmask, b = bp
        a = np.asarray(ages, float); t = np.asarray(toks)
        last_obs = float(a[a > 0].max())
        r = {"baseline_state": int(b), "baseline_age": base / D,
             "followup": (last_obs - base) / D}
        for i in range(F2.NSLOT):
            m = (S.slot_of(t) == i) & (a > base)
            r[f"obs_t_{F2.ALL_NAMES[i]}"] = ((float(a[m].min()) - base) / D
                                             if m.any() else np.inf)
        for sc, pairs in S.AUX_OF_SCALE.items():
            ids = [tok for _sl, tok in pairs]
            mm = np.isin(t, ids) & (a <= base) & (a > F2.NEG)
            r[f"baseline_{sc}"] = (int(S.TOK2SLOT[int(t[mm][np.argmax(a[mm])])])
                                   if mm.any() else -1)
        rows.append(r)
        v = np.zeros(res.VOCAB_SIZE + 1, dtype=np.float32)
        for tt in t[pmask]:                       # 与 rollout 的起始前缀完全相同的 token 集合
            v[int(tt)] = 1.0
        v[-1] = (base / D) / 100.0                # 基线年龄
        feats.append(v)
    return pd.DataFrame(rows), np.array(feats, dtype=np.float32)


def ep_obs(df, states):
    """composite() 里**不依赖 MC** 的那三项，其余 figure2 的评估函数只用这三项。"""
    obs = np.minimum.reduce([df[f"obs_t_{F2.ALL_NAMES[s]}"].to_numpy(float) for s in states])
    return dict(obs_time=obs, d_obs=df["obs_t_Death"].to_numpy(float),
                fu=df["followup"].to_numpy(float))


def fit_predict(Xtr, ytr, Xva):
    keep = Xtr.std(0) > 0
    lr = LogisticRegression(max_iter=4000, C=1.0)
    lr.fit(Xtr[:, keep], ytr)
    return lr.predict_proba(Xva[:, keep])[:, 1]


def run(dataset, model_metrics):
    ddir = os.path.join(HERE, "data", dataset)
    res = bind(os.path.join(ddir, "labels.csv"))
    dtr, Xtr = frame_lite(ddir, "train", res)
    dva, Xva = frame_lite(ddir, "val", res)
    print(f"\n{'='*104}\ndataset={dataset}   train {len(dtr)} 人 / val {len(dva)} 人  "
          f"（可评估，与 panel a 同一套 _baseline 规则）\n{'='*104}")

    out = {"panel_a": [], "panel_b": []}
    # ---------------- panel a：8 行，horizon = 5 ----------------
    H = F2.H if hasattr(F2, "H") else 5
    print(f"\n【panel a】horizon={H}y   —— LR 与 transformer 用同一套 at-risk / 标签 / CI")
    hdr = (f"{'行':<26s}{'at risk':>9s}{'events':>8s}{'AUC LR':>20s}"
           f"{'AUC 模型':>10s}{'Δ':>8s}{'实测CIF':>9s}{'LR预测':>8s}")
    print(hdr); print("-" * len(hdr))
    for slot in FP.state_targets():
        name = F2.ALL_NAMES[slot]
        eptr, epva = ep_obs(dtr, [slot]), ep_obs(dva, [slot])
        artr, arva = FP._at_risk(dtr, slot), FP._at_risk(dva, slot)
        ytr_all, yva_all = F2.labels_at_h(eptr, H), F2.labels_at_h(epva, H)
        mtr, mva = artr & (ytr_all >= 0), arva & (yva_all >= 0)
        if len(np.unique(ytr_all[mtr])) < 2 or int(yva_all[mva].sum()) < FP.MIN_POS:
            continue
        s_all = np.full(len(dva), np.nan)
        s_all[arva] = fit_predict(Xtr[mtr], ytr_all[mtr], Xva[arva])
        y, s = yva_all[mva].astype(int), s_all[mva]
        if len(y) - int(y.sum()) < FP.MIN_POS:
            continue
        auc, lo, hi = FP.boot_auc(y, s)
        tt, et = F2.cr_times(epva, arva)
        cif = F2.aalen_johansen(tt, et, H)
        mm = model_metrics.get(name, {})
        mauc = mm.get("auc", float("nan"))
        print(f"{name:<26s}{int(arva.sum()):>9d}{int(y.sum()):>8d}"
              f"{auc:>10.3f} [{lo:.2f},{hi:.2f}]{mauc:>10.3f}{auc-mauc:>+8.3f}"
              f"{cif:>9.3f}{float(np.nanmean(s_all[arva])):>8.3f}")
        out["panel_a"].append(dict(label=name, n_at_risk=int(arva.sum()),
                                   n_events=int(y.sum()), auc_lr=auc, ci=[lo, hi],
                                   auc_model=mauc, obs_cif=cif,
                                   pred_lr=float(np.nanmean(s_all[arva]))))
    med_lr = np.median([r["auc_lr"] for r in out["panel_a"]])
    med_md = np.median([r["auc_model"] for r in out["panel_a"]
                        if np.isfinite(r["auc_model"])])
    print(f"  中位 AUC:  LR {med_lr:.3f}   模型 {med_md:.3f}")

    # ---------------- panel b：3 个终点 × 多个 horizon ----------------
    print(f"\n【panel b】AUC by horizon")
    hdr = f"{'终点':<32s}{'source':<8s}" + "".join(f"{str(h)+'y':>9s}" for h in F2.HORIZONS)
    print(hdr); print("-" * len(hdr))
    bmod = model_metrics.get("_b_auc_by_horizon", {})
    for lab, slots, _pre in S.endpoints():
        eptr, epva = ep_obs(dtr, slots), ep_obs(dva, slots)
        line_lr, rec = f"{lab:<32s}{'LR':<8s}", {}
        for h in F2.HORIZONS:
            ytr_all, yva_all = F2.labels_at_h(eptr, h), F2.labels_at_h(epva, h)
            mtr, mva = ytr_all >= 0, yva_all >= 0
            if len(np.unique(ytr_all[mtr])) < 2 or len(np.unique(yva_all[mva])) < 2:
                line_lr += f"{'—':>9s}"; continue
            s = fit_predict(Xtr[mtr], ytr_all[mtr], Xva[mva])
            auc, lo, hi = FP.boot_auc(yva_all[mva].astype(int), s)
            rec[str(h)] = auc
            line_lr += f"{auc:>9.3f}"
        print(line_lr)
        print(f"{'':<32s}{'模型':<8s}"
              + "".join(f"{bmod.get(lab, {}).get(str(h), float('nan')):>9.3f}"
                        for h in F2.HORIZONS))
        out["panel_b"].append(dict(label=lab, auc_lr=rec,
                                   auc_model=bmod.get(lab, {})))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["rosmap", "rosmap_nodedup", "rosmap_fullvisit"])
    ap.add_argument("--tags", nargs="+",
                    default=["gpubase", "nodedup", "fullvisit"],
                    help="各 dataset 对应的 results/figure2/<tag>，用来读 transformer 的数字")
    ap.add_argument("--cohort", default="matched")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    report = {}
    for ds, tag in zip(a.datasets, a.tags):
        mp = os.path.join(HERE, "results", "figure2", tag, f"metrics_{a.cohort}.json")
        mm = {}
        if os.path.exists(mp):
            j = json.load(open(mp))
            mm = dict(j["a"]["per_state"])
            mm["_b_auc_by_horizon"] = j["b"]["auc_by_horizon"]
        else:
            print(f"（没有 {mp}，模型列会是 nan）")
        report[ds] = run(ds, mm)
    print("\n怎么判：")
    print("  Δ > 0  = LR 赢。如果 5 年 AD / 死亡上 LR 也赢，那不是'给状态转移加线性头'能解决的，")
    print("          要重谈的是在这个语料规模下 transformer 值不值。")
    print("  LR 的输入 = rollout 的起始前缀，一个 token 不多不少，所以差异只能归给模型。")
    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1, default=str)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
