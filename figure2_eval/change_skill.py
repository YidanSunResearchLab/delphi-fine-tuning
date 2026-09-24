"""
change_skill.py -- 区分"模型一直复读当前状态"和"模型真的学到了变化"。

    python change_skill.py --runs dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
                                  nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
                                  fullvisit:../delphi/Delphi-ROSMAP-fullvisit/ckpt.pt:rosmap_fullvisit

为什么需要它：边际的 next-token / next-visit AUC 被**持续性**主导。`MMSE_normal` 的 AUC 很大
一部分在测"能不能读出这个人现在是 normal"，而那是从上下文直接抄的，不是预测。一个纯复读机
（永远输出当前箱）在那些表上分数很高，却对进展一无所知。

所以这里换成四个**复读机刷不高**的量，并且和两个不会学习的基线并排放：

  基线 A  持续性 —— 永远预测当前箱。P(stay) 用 train 上的经验持续率，剩下的均分给另两箱。
  基线 B  一阶 Markov —— train 上数出来的 3x3 转移矩阵，**没有任何协变量**。
          这是"只知道当前状态"的最优预测器，是硬门槛：模型打不过它，就说明年龄、APOE、
          病史这些它全没用上，学到的只是边际转移率。

  1. AUC(会不会变)     二分类"下一箱 != 当前箱"。复读机给所有位置同一个分数 -> 结构上 0.5。
  2. 对数损失 skill    1 - CE_model/CE_baseline。严格恰当评分，猜不变会在真变的样本上被重罚。
  3. 条件方向准确率    给定确实变了，**排除当前箱后**重整概率再问"往哪变"。
                       这条把"猜不变"的退路彻底封死。
  4. P(变化) 的定标    预测的平均变化率 vs 实测变化率。复读机会预测 ≈ 0。

口径与 next_visit_auc.py 完全一致（同样 3,384 个访视对，真值同样取自最不去重那份 .bin 的
逐访视记录），所以两边的数字可以逐格对读。CI 一律按**人** bootstrap。
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from next_visit_auc import truth_table, score_run, auc_and_ci, N_BOOT  # noqa: E402

EPS = 1e-12


def transitions(truth, res, fam):
    """(cur, nxt) 序号对，从真值表里抽某个家族。序号 = 该家族在 labels.csv 里的位置。"""
    ids = list(res.SCALES[fam])
    pos = {t: i for i, t in enumerate(ids)}
    out = []
    for pid, visits in truth.items():
        for i in range(len(visits) - 1):
            c = [t for t in ids if t in visits[i][1]]
            n = [t for t in ids if t in visits[i + 1][1]]
            if len(c) == 1 and len(n) == 1:
                out.append((pid, pos[c[0]], pos[n[0]]))
    return out


def fit_baselines(tr, k):
    """在 train 上拟合：持续率 p、以及 k x k 的一阶 Markov 转移矩阵（Laplace 平滑）。"""
    cur = np.array([c for _p, c, _n in tr])
    nxt = np.array([n for _p, _c, n in tr])
    p_stay = float((cur == nxt).mean())
    M = np.ones((k, k))                      # Laplace: 每格 +1，避免零概率炸对数损失
    for c, n in zip(cur, nxt):
        M[c, n] += 1
    M /= M.sum(1, keepdims=True)
    return p_stay, M


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
    print(f"真值 data/{a.truth_dataset}: train {len(truth_tr)} 人 / val {len(truth_val)} 人")

    runs = []
    for spec in a.runs:
        nm, ckpt, ds = spec.split(":")
        recs, _ = score_run(ckpt, ds, truth_val, res, bool(a.mask_ties), a.device)
        runs.append((nm, recs))
        print(f"  {nm:<10s} {len(recs)} 个观测")
    names = [n for n, _ in runs]

    report = {}
    for fam, famlab in (("MMSE", "MMSE"), ("COG", "cogn_global")):
        ids = list(res.SCALES[fam])
        k = len(ids)
        tr = transitions(truth_tr, res, fam)
        p_stay, M = fit_baselines(tr, k)
        pos = {t: i for i, t in enumerate(ids)}

        # 对齐到一个共同的观测集合：cur/truth 都有、且三档都打了分
        rows = []
        for r in runs[0][1]:
            f = r[fam]
            if f["cur"] is None or f["truth"] is None:
                continue
            rows.append((r["pid"], pos[f["cur"]], pos[f["truth"]]))
        pid = np.array([x[0] for x in rows])
        cur = np.array([x[1] for x in rows])
        nxt = np.array([x[2] for x in rows])
        ychg = (cur != nxt).astype(np.int8)
        obs_chg = float(ychg.mean())

        # 每个 run 的 k 维概率矩阵
        Pm = {}
        for nm, recs in runs:
            P = []
            for r in recs:
                f = r[fam]
                if f["cur"] is None or f["truth"] is None:
                    continue
                P.append([f["p"][t] for t in ids])
            Pm[nm] = np.array(P, dtype=np.float64)
            assert len(Pm[nm]) == len(rows), f"{nm}/{fam} 观测集合没对齐"
        # 基线 A：持续性。P(stay)=p_stay，其余均分
        Pa = np.full((len(rows), k), (1 - p_stay) / (k - 1))
        Pa[np.arange(len(rows)), cur] = p_stay
        Pm["基线A 持续性"] = Pa
        # 基线 B：train 上的一阶 Markov
        Pm["基线B Markov"] = M[cur]

        allnames = names + ["基线A 持续性", "基线B Markov"]
        ce = {n: float(-np.log(np.clip(Pm[n][np.arange(len(rows)), nxt], EPS, 1)).mean())
              for n in allnames}
        pchg = {n: 1.0 - Pm[n][np.arange(len(rows)), cur] for n in allnames}

        print(f"\n{'='*100}")
        print(f"{famlab}  —  {len(rows)} 个访视对（{len(set(pid.tolist()))} 人）  "
              f"实测变化率 {obs_chg:.3f}  train 持续率 {p_stay:.3f}")
        print(f"{'='*100}")
        hdr = (f"{'':<16s}{'CE(对数损失)':>14s}{'skill vs A':>12s}{'skill vs B':>12s}"
               f"{'AUC(会不会变)':>18s}{'预测变化率':>12s}")
        print(hdr); print("-" * len(hdr))
        res_f = {}
        for n in allnames:
            sk_a = 1 - ce[n] / ce["基线A 持续性"]
            sk_b = 1 - ce[n] / ce["基线B Markov"]
            auc, ci, _np_, _no = auc_and_ci(ychg, pchg[n], pid, N_BOOT)
            print(f"{n:<16s}{ce[n]:>14.4f}{sk_a:>+12.3f}{sk_b:>+12.3f}"
                  f"{auc:>10.3f} [{ci[0]:.2f},{ci[1]:.2f}]{pchg[n].mean():>12.3f}")
            res_f[n] = dict(ce=ce[n], skill_vs_persist=sk_a, skill_vs_markov=sk_b,
                            auc_change=auc, auc_change_ci=list(ci),
                            pred_change_rate=float(pchg[n].mean()))

        # 条件方向：给定确实变了，排除当前箱后重整
        m = ychg == 1
        print(f"\n给定确实变了（n={int(m.sum())}），**排除当前箱后**重整，问往哪变：")
        hdr2 = f"{'':<16s}{'top-1':>10s}{'随机':>10s}{'CE':>10s}{'skill vs B':>12s}"
        print(hdr2); print("-" * len(hdr2))
        for n in allnames:
            P = Pm[n][m].copy()
            P[np.arange(int(m.sum())), cur[m]] = 0.0
            P /= np.clip(P.sum(1, keepdims=True), EPS, None)
            top1 = float((P.argmax(1) == nxt[m]).mean())
            cec = float(-np.log(np.clip(P[np.arange(int(m.sum())), nxt[m]], EPS, 1)).mean())
            res_f[n].update(cond_top1=top1, cond_ce=cec)
        base_cec = res_f["基线B Markov"]["cond_ce"]
        for n in allnames:
            print(f"{n:<16s}{res_f[n]['cond_top1']:>10.3f}{1/(k-1):>10.3f}"
                  f"{res_f[n]['cond_ce']:>10.4f}{1 - res_f[n]['cond_ce']/base_cec:>+12.3f}")
        report[famlab] = dict(n=len(rows), obs_change_rate=obs_chg,
                              train_p_stay=p_stay, per_run=res_f)

    print("\n怎么读：")
    print("  * skill > 0 = 比那个基线好。**skill vs B（Markov）<= 0 就是决定性的坏消息**：")
    print("    说明模型没用上当前状态以外的任何信息，等价于一张在训练集上数出来的转移表。")
    print("  * AUC(会不会变)：基线 A 结构上恒为 0.500（对所有位置给同一个分数）。")
    print("    基线 B 不是 0.5，因为不同当前箱的变化率不同 —— 那部分信号也不算模型的功劳。")
    print("  * 条件方向 top-1 的随机水平是 1/(k-1)=0.500；这一栏封死了「猜不变」的退路。")
    if a.out:
        json.dump(report, open(a.out, "w"), ensure_ascii=False, indent=1)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
