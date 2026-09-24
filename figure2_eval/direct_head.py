"""
direct_head.py -- 方案 2 的第一步：砍掉 rollout，直接在 transformer 表征上接判别头。

    python direct_head.py --runs dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
                                 nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
                                 fullvisit:../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt:rosmap_fullvisit

背景：lr_endpoint_baseline.py 量到，在 figure2 自己的终点上，一个用**基线前缀**做特征的
逻辑回归几乎全面胜过走 rollout 的 transformer（中位 AUC 0.789 vs 0.713，8 行里赢 7 行）。
那些塌陷全部出在**输出机制**（逐 token 自回归），不在表征。所以换个输出机制。

这一步**冻结** encoder，只比三组特征，代价是几分钟而不是一次重训：

    raw       基线前缀 token 的多热 + 基线年龄            —— 就是 lr_endpoint_baseline 的 LR
    emb       基线前缀最后一个位置的隐状态（post ln_f，96 维）
    raw+emb   两者拼接

判据：
  * emb >= raw  -> 表征里有 raw 没有的东西，"直接判别头"是对的架构，值得做端到端微调。
  * emb <<  raw -> 表征本身就弱，加个头救不了，方案 2 得连 encoder 一起训。
  * raw+emb > 两者 -> 互补，残差结构（LR 主干 + transformer 残差）有直接证据。

**这不是端到端的方案 2，是它的可行性探针。** 端到端要把 head 和 encoder 一起训，
而且要和 LM 损失联合训练才能保住扰动引擎需要的生成能力 —— 那是下一步。

评估口径与 lr_endpoint_baseline.py / figure2 panel a 完全一致（同一套 at-risk、
竞争风险 + 删失感知的标签、同一个 boot_auc），唯一变的是特征。
"""
import os
import sys
import json
import argparse
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "figure2"))
import torch                                                  # noqa: E402
from sklearn.linear_model import LogisticRegression           # noqa: E402
from sklearn.pipeline import make_pipeline                    # noqa: E402
from sklearn.preprocessing import StandardScaler              # noqa: E402
from sklearn.exceptions import ConvergenceWarning             # noqa: E402
import pandas as pd                                           # noqa: E402
from radc_delphi import engine as EN, batching as Bt          # noqa: E402
import figure2.radc_states as S                               # noqa: E402
import figure2.figure2_core as F2                             # noqa: E402
import figure2.figure2_panels as FP                           # noqa: E402
from lr_endpoint_baseline import bind, ep_obs                 # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)
D = 365.25


def frame_with_emb(eng, data_dir, split, res):
    """观测量 + 两组特征。emb 用 F2._hidden，取基线前缀最后一个位置（= rollout 的条件向量）。"""
    d = np.fromfile(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
    p2i = Bt.get_p2i(d)
    rows, raw, emb = [], [], []
    for k in range(len(p2i)):
        s, n = int(p2i[k, 0]), int(p2i[k, 1])
        ages, toks = Bt.patient_stream(d, s, n)
        bp = F2._baseline(ages, toks)
        if bp is None:
            continue
        base, pmask, b = bp
        a = np.asarray(ages, float); t = np.asarray(toks)
        last_obs = float(a[a > 0].max())
        r = {"baseline_state": int(b), "followup": (last_obs - base) / D}
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
        for tt in t[pmask]:
            v[int(tt)] = 1.0
        v[-1] = (base / D) / 100.0
        raw.append(v)
        with torch.no_grad():
            emb.append(F2._hidden(eng, t[pmask], a[pmask])[-1].astype(np.float32))
    return (pd.DataFrame(rows), np.array(raw, dtype=np.float32),
            np.array(emb, dtype=np.float32))


def fit_predict(Xtr, ytr, Xva, scale):
    keep = Xtr.std(0) > 0
    clf = (make_pipeline(StandardScaler(), LogisticRegression(max_iter=4000, C=1.0))
           if scale else LogisticRegression(max_iter=4000, C=1.0))
    clf.fit(Xtr[:, keep], ytr)
    return clf.predict_proba(Xva[:, keep])[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="name:ckpt:dataset")
    ap.add_argument("--tags", nargs="+", default=None,
                    help="各 run 对应的 results/figure2/<tag>，用来读 rollout 的数字")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--eval-split", default="val",
                    help="在哪个 split 上报数。三分 split（build.py --split 0.8,0.1,0.1）下应给 "
                         "test —— val 参与了 checkpoint 选择，见 README 3.3。LR/判别头仍在 "
                         "train 上拟合，所以 test 上的数字是真正没被任何选择碰过的。")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tags = a.tags or [s.split(":")[0] for s in a.runs]

    report = {}
    for spec, tag in zip(a.runs, tags):
        nm, ckpt, ds = spec.split(":")
        ddir = os.path.join(HERE, "data", ds)
        res = bind(os.path.join(ddir, "labels.csv"))
        eng = EN.load(os.path.join(HERE, ckpt), data_dir=ddir, device=a.device)
        dtr, Rtr, Etr = frame_with_emb(eng, ddir, "train", res)
        dva, Rva, Eva = frame_with_emb(eng, ddir, a.eval_split, res)
        FEATS = {"raw": (Rtr, Rva, False), "emb": (Etr, Eva, True),
                 "raw+emb": (np.hstack([Rtr, Etr]), np.hstack([Rva, Eva]), True)}

        mm = {}
        mp = os.path.join(HERE, "results", "figure2", tag, "metrics_matched.json")
        if os.path.exists(mp):
            j = json.load(open(mp))
            mm = dict(j["a"]["per_state"])
            mm["_b"] = j["b"]["auc_by_horizon"]

        print(f"\n{'='*112}\n{nm}  dataset={ds}  ckpt={os.path.basename(os.path.dirname(ckpt))}"
              f"   train {len(dtr)} / {a.eval_split} {len(dva)} 人   emb dim={Etr.shape[1]}\n{'='*112}")
        H = F2.H if hasattr(F2, "H") else 5
        hdr = (f"{'行（5y）':<26s}{'events':>7s}" + "".join(f"{'AUC ' + k:>13s}" for k in FEATS)
               + f"{'AUC rollout':>13s}")
        print(hdr); print("-" * len(hdr))
        panel_a = []
        for slot in FP.state_targets():
            name = F2.ALL_NAMES[slot]
            eptr, epva = ep_obs(dtr, [slot]), ep_obs(dva, [slot])
            artr, arva = FP._at_risk(dtr, slot), FP._at_risk(dva, slot)
            ytr_all, yva_all = F2.labels_at_h(eptr, H), F2.labels_at_h(epva, H)
            mtr, mva = artr & (ytr_all >= 0), arva & (yva_all >= 0)
            y = yva_all[mva].astype(int)
            if len(np.unique(ytr_all[mtr])) < 2 or y.sum() < FP.MIN_POS \
                    or len(y) - y.sum() < FP.MIN_POS:
                continue
            rec = {"label": name, "n_events": int(y.sum())}
            line = f"{name:<26s}{int(y.sum()):>7d}"
            for kf, (Xtr, Xva, sc) in FEATS.items():
                s_all = np.full(len(dva), np.nan)
                s_all[arva] = fit_predict(Xtr[mtr], ytr_all[mtr], Xva[arva], sc)
                auc, lo, hi = FP.boot_auc(y, s_all[mva])
                rec[f"auc_{kf}"] = auc
                line += f"{auc:>13.3f}"
            rec["auc_rollout"] = mm.get(name, {}).get("auc", float("nan"))
            print(line + f"{rec['auc_rollout']:>13.3f}")
            panel_a.append(rec)
        print(f"{'中位':<26s}{'':>7s}"
              + "".join(f"{np.median([r['auc_' + k] for r in panel_a]):>13.3f}" for k in FEATS)
              + f"{np.median([r['auc_rollout'] for r in panel_a if np.isfinite(r['auc_rollout'])]):>13.3f}")

        print(f"\npanel b — AUC by horizon")
        hdr = f"{'终点':<30s}{'source':<10s}" + "".join(f"{str(h)+'y':>9s}" for h in F2.HORIZONS)
        print(hdr); print("-" * len(hdr))
        panel_b = []
        for lab, slots, _p in S.endpoints():
            eptr, epva = ep_obs(dtr, slots), ep_obs(dva, slots)
            rec = {"label": lab}
            for kf, (Xtr, Xva, sc) in FEATS.items():
                line = f"{lab:<30s}{kf:<10s}"; vals = {}
                for h in F2.HORIZONS:
                    ytr_all, yva_all = F2.labels_at_h(eptr, h), F2.labels_at_h(epva, h)
                    mtr, mva = ytr_all >= 0, yva_all >= 0
                    if len(np.unique(ytr_all[mtr])) < 2 or len(np.unique(yva_all[mva])) < 2:
                        line += f"{'—':>9s}"; continue
                    s = fit_predict(Xtr[mtr], ytr_all[mtr], Xva[mva], sc)
                    auc, _lo, _hi = FP.boot_auc(yva_all[mva].astype(int), s)
                    vals[str(h)] = auc; line += f"{auc:>9.3f}"
                print(line); rec[kf] = vals
            rec["rollout"] = mm.get("_b", {}).get(lab, {})
            print(f"{'':<30s}{'rollout':<10s}"
                  + "".join(f"{rec['rollout'].get(str(h), float('nan')):>9.3f}"
                            for h in F2.HORIZONS))
            panel_b.append(rec)
        report[nm] = dict(panel_a=panel_a, panel_b=panel_b)

    print("\n怎么判：")
    print("  emb >= raw      -> 表征里有 raw 没有的东西，直接判别头是对的架构，值得端到端微调")
    print("  emb <<  raw     -> 表征本身弱，加头救不了，得连 encoder 一起训")
    print("  raw+emb > 两者  -> 互补，残差结构（LR 主干 + transformer 残差）有直接证据")
    print("  三列都远超 rollout -> 塌陷确实在输出机制，不在表征")
    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1, default=str)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
