"""
window_baselines.py -- 两个"不会学习/视野极窄"的基线，和三个模型在**同一张表**上对比。

    python window_baselines.py --runs dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
                                      nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
                                      fullvisit:../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt:rosmap_fullvisit

change_skill.py 只在"会不会变"那个问题上报了基线，**没有**在被持续性吹高的那张**边际 AUC**
表上报 —— 而那才是要害：如果"永远不变"这个基线在边际 AUC 上也能拿 0.88，那模型的 0.88
就一文不值。这里补上，并且把 Markov 换成视野更宽的版本。

基线（全部只在 **train** 上拟合）：

  N  **完全不转变** —— P(下一箱 = 当前箱) = 1，其余为 0。你能想到的最笨的预测器。
     它在边际 AUC 上的分数 = "光靠读出当前状态能拿多少"，是模型必须显著超过的地板。
     （硬 0/1 会让对数损失发散，所以 CE 那一栏用 train 经验持续率平滑过的版本 N'。）

  W  **单访视窗口模型** —— 多项 logistic 回归，输入只有**当前这一次访视的 token 多热向量**
     （+ 当前年龄），**看不到任何历史**。比 change_skill.py 的一阶 Markov 宽得多：
     Markov 只用当前那一个箱，W 用当前访视的全部 token（血压/血糖/用药/其它量表都在内）。
     模型（看整条历史的 transformer）如果超不过 W，就说明**历史没带来任何信息**。

  W+ 同 W，再加上 statics（性别 + 背景块：教育/种族/APOE/吸烟/饮酒/基线已患病）。
     这是"不看纵向历史、但看得到所有基线信息"的上限。

观测集合、真值、CI 口径与 next_visit_auc.py / change_skill.py 完全一致，可逐格对读。
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sklearn.linear_model import LogisticRegression          # noqa: E402
from next_visit_auc import truth_table, score_run, auc_and_ci, N_BOOT  # noqa: E402

EPS = 1e-12


def pairs_with_features(truth, res, fam, static_ids, vocab_size):
    """(pid, cur, nxt, feat) —— feat = 当前访视 token 的多热 + 年龄。"""
    ids = list(res.SCALES[fam])
    pos = {t: i for i, t in enumerate(ids)}
    out = []
    for pid, visits in truth.items():
        # 这个人的 statics（只在基线那一格出现），W+ 用
        st = np.zeros(vocab_size, dtype=np.float32)
        for a, toks in visits:
            for t in toks:
                if t in static_ids:
                    st[t] = 1.0
        for i in range(len(visits) - 1):
            a_t, tset = visits[i]
            c = [t for t in ids if t in tset]
            n = [t for t in ids if t in visits[i + 1][1]]
            if len(c) != 1 or len(n) != 1:
                continue
            v = np.zeros(vocab_size, dtype=np.float32)
            for t in tset:
                if t not in static_ids:
                    v[t] = 1.0                     # 只有**这一次访视**的临床 token
            out.append((pid, pos[c[0]], pos[n[0]], v, st, a_t / 365.25))
    return out


def design(rows, with_static):
    X = [np.concatenate([r[3], r[4], [r[5] / 100.0]]) if with_static
         else np.concatenate([r[3], [r[5] / 100.0]]) for r in rows]
    X = np.array(X, dtype=np.float32)
    keep = X.std(0) > 0                            # 去掉恒为 0 的列，否则 LR 收敛警告刷屏
    return X[:, keep], keep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--truth-dataset", default="rosmap_nodedup")
    ap.add_argument("--mask-ties", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    tdir = os.path.join(HERE, "data", a.truth_dataset)
    truth_val, res = truth_table(tdir, "val")
    truth_tr, _ = truth_table(tdir, "train")
    static_ids = set(res.STATIC_IDS)
    print(f"真值 data/{a.truth_dataset}: train {len(truth_tr)} 人 / val {len(truth_val)} 人")

    runs = []
    for spec in a.runs:
        nm, ckpt, ds = spec.split(":")
        recs, _ = score_run(ckpt, ds, truth_val, res, bool(a.mask_ties), a.device)
        runs.append((nm, recs))
        print(f"  {nm:<10s} {len(recs)} 个观测")

    report = {}
    for fam, famlab in (("MMSE", "MMSE"), ("COG", "cogn_global")):
        ids = list(res.SCALES[fam])
        k = len(ids)
        tr = pairs_with_features(truth_tr, res, fam, static_ids, res.VOCAB_SIZE)
        va = pairs_with_features(truth_val, res, fam, static_ids, res.VOCAB_SIZE)
        pid = np.array([r[0] for r in va])
        cur = np.array([r[1] for r in va])
        nxt = np.array([r[2] for r in va])
        ychg = (cur != nxt).astype(np.int8)
        p_stay = float(np.mean([r[1] == r[2] for r in tr]))

        P = {}
        # --- 模型 -------------------------------------------------------------
        for nm, recs in runs:
            M = []
            for r in recs:
                f = r[fam]
                if f["cur"] is None or f["truth"] is None:
                    continue
                M.append([f["p"][t] for t in ids])
            M = np.array(M, dtype=np.float64)
            assert len(M) == len(va), f"{nm}/{fam} 观测集合没对齐 ({len(M)} vs {len(va)})"
            P[nm] = M
        # --- N 完全不转变（硬 0/1）与 N' 平滑版 --------------------------------
        N = np.zeros((len(va), k)); N[np.arange(len(va)), cur] = 1.0
        P["N 完全不转变"] = N
        Ns = np.full((len(va), k), (1 - p_stay) / (k - 1)); Ns[np.arange(len(va)), cur] = p_stay
        P["N' 平滑持续性"] = Ns
        # --- W / W+ 单访视窗口 logistic ---------------------------------------
        ytr = np.array([r[2] for r in tr])
        for lab, ws in (("W 单访视窗口", False), ("W+ 窗口+statics", True)):
            Xtr, keep = design(tr, ws)
            Xva = np.array([np.concatenate([r[3], r[4], [r[5] / 100.0]]) if ws
                            else np.concatenate([r[3], [r[5] / 100.0]]) for r in va],
                           dtype=np.float32)[:, keep]
            lr = LogisticRegression(max_iter=3000, C=1.0, multi_class="multinomial")
            lr.fit(Xtr, ytr)
            Q = np.zeros((len(va), k))
            Q[:, lr.classes_] = lr.predict_proba(Xva)
            P[lab] = np.clip(Q, EPS, 1)
            P[lab] /= P[lab].sum(1, keepdims=True)

        order = [n for n, _ in runs] + ["N 完全不转变", "N' 平滑持续性",
                                        "W 单访视窗口", "W+ 窗口+statics"]

        print(f"\n{'='*104}")
        print(f"{famlab}  —  {len(va)} 个访视对（{len(set(pid.tolist()))} 人）  "
              f"实测变化率 {float(ychg.mean()):.3f}   train 持续率 {p_stay:.3f}")
        print(f"{'='*104}")

        # ---- (1) 边际 per-token AUC：被持续性吹高的那张表 --------------------
        print("\n【1】边际 per-token AUC —— 「下一次访视是不是这个箱」。"
              "这是被持续性吹高的那张表。")
        hdr = f"{'':<18s}" + "".join(f"{res.NAMES[t]:>20s}" for t in ids) + f"{'macro':>10s}"
        print(hdr); print("-" * len(hdr))
        marg = {}
        for n in order:
            aucs = []
            line = f"{n:<18s}"
            for j, t in enumerate(ids):
                y = (nxt == j).astype(np.int8)
                auc, ci, _, _ = auc_and_ci(y, P[n][:, j], pid, N_BOOT)
                aucs.append(auc)
                line += f"{auc:>8.3f} [{ci[0]:.2f},{ci[1]:.2f}]"
            marg[n] = float(np.mean(aucs))
            print(line + f"{marg[n]:>10.3f}")
        print(f"  --> 模型必须显著超过「N 完全不转变」的 macro {marg['N 完全不转变']:.3f}，"
              f"否则那张表里的高分只是在读当前状态。")

        # ---- (2) 会不会变 / 条件方向 / CE -----------------------------------
        print("\n【2】复读机刷不高的三个量")
        hdr2 = (f"{'':<18s}{'AUC(会不会变)':>18s}{'条件方向top1':>14s}{'CE':>10s}"
                f"{'skill vs W':>12s}{'预测变化率':>12s}")
        print(hdr2); print("-" * len(hdr2))
        ce = {n: float(-np.log(np.clip(P[n][np.arange(len(va)), nxt], EPS, 1)).mean())
              for n in order}
        m = ychg == 1
        rows_out = {}
        for n in order:
            pchg = 1.0 - P[n][np.arange(len(va)), cur]
            auc, ci, _, _ = auc_and_ci(ychg, pchg, pid, N_BOOT)
            Q = P[n][m].copy(); Q[np.arange(int(m.sum())), cur[m]] = 0.0
            Q /= np.clip(Q.sum(1, keepdims=True), EPS, None)
            top1 = float((Q.argmax(1) == nxt[m]).mean())
            sk = 1 - ce[n] / ce["W 单访视窗口"]
            print(f"{n:<18s}{auc:>10.3f} [{ci[0]:.2f},{ci[1]:.2f}]{top1:>14.3f}"
                  f"{ce[n]:>10.3f}{sk:>+12.3f}{pchg.mean():>12.3f}")
            rows_out[n] = dict(marginal_macro_auc=marg[n], auc_change=auc,
                               auc_change_ci=list(ci), cond_top1=top1, ce=ce[n],
                               skill_vs_window=sk, pred_change_rate=float(pchg.mean()))
        print(f"  条件方向的随机水平 = {1/(k-1):.3f}；"
              f"「N 完全不转变」的 CE 是 inf（硬 0/1），表里用不到它那一格。")
        report[famlab] = dict(n=len(va), obs_change_rate=float(ychg.mean()),
                              train_p_stay=p_stay, per_run=rows_out)

    print("\n怎么判：")
    print("  1) 边际 macro AUC 没有显著超过「N 完全不转变」 -> 那张表只是在读当前状态。")
    print("  2) skill vs W <= 0 -> 整条历史没带来任何信息，一个只看当次访视的 logistic 就够。")
    print("  3) AUC(会不会变)：N 结构上 = 0.500。W 不是 0.5，那部分不算模型的功劳。")
    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
