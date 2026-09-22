"""
next_visit_auc.py -- 不用 rollout 的判别力：给定到本次访视为止的全部历史，预测**下一次访视**。

    python next_visit_auc.py \
        --runs dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
               nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup

figure2 的 panel a/b 问的是"5 年内会不会到达某状态"，答案来自 100 条 Monte-Carlo 轨迹。
README 6.5 节发现 AD 的塌陷只出现在 rollout 里，所以需要一个**完全不做 rollout** 的口径：
每个 (人, 访视) 一次 teacher-forced forward，取最后一个位置的 logits，问下一次访视。

--------------------------------------------------------------------------------------
三个设计决定，每一个做错都会让比较变成不公平的

1) **真值一律取自不去重那份 .bin，两个模型共用。**
   交付版分词把"认知分箱没变"的 token 整条删掉了，所以在它自己的 .bin 里根本问不出
   "下一次访视的 MMSE 是什么"——真值不存在。不去重那份每次访视都发射，是**忠实记录**，
   所以拿它当两边共同的 ground truth。两份分词的人和 train/val 划分完全相同（同一个 RNG
   抽取序列），projid 可以直接对齐。

2) **打分位置按年龄对齐，不按 token 下标对齐。**
   两份 .bin 的 token 数不同，同一次访视在两边的下标不一样。所以对每个访视年龄 a_t，
   各自取"自己的 stream 里 age <= a_t 的全部 token"当前缀——即"到 a_t 为止这个模型知道的
   一切"。交付版在某些访视上什么都没发射，它的前缀就停在更早的位置，这是那份分词的**性质**，
   不是这里的取舍。

3) **`mask_ties=True`。**
   见 engine._logits 的 docstring：本分词一次访视的 token 同龄，训练时每个位置都看不见自己
   访视的兄弟。而**最后一个位置**的 targets_age 是哨兵 1e9，不屏蔽任何 key —— 正好就是训练时
   "预测下一次访视第一个 token"的那个注意力配置。所以这里用 True 才是训练口径，
   不是 rollout 里那个"打开反而更差"的场景（那里上下文是模型自己生成的离群病史）。
   `--mask-ties 0` 可以跑另一侧做敏感性。

--------------------------------------------------------------------------------------
两张表，第二张才是决定性的

  **全部访视对**：交付版**结构上**发不出重复 token，所以"这次 normal、下次还 normal"这类
  样本它必输。这张表回答"端到端谁更好用"，但它把分词的表达能力和模型的预测能力混在一起。

  **只看真实发生了变化的访视对**：变化在两份分词里都表达得出来，交付版不吃结构性亏。
  这张表回答"给定要预测一次真实的状态改变，谁更准"。

分数按**家族内 softmax** 算（MMSE 三箱互斥、COGN 三箱互斥），这正是"下一次访视落哪个箱"
这个多分类问题；AD_DX / Death 是二元事件，用在全部 content token 上重整的概率。

置信区间是**按人 bootstrap**，不是按观测：一个人贡献多次访视，观测之间相关，
按观测做 bootstrap 或 DeLong 会把区间算窄。
"""
import os
import sys
import json
import argparse
import collections

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch                                              # noqa: E402
from radc_delphi import engine as EN, vocab as V          # noqa: E402

N_BOOT = 2000
RNG = np.random.default_rng(0)


def auc_and_ci(y, s, groups, n_boot=N_BOOT):
    """AUC + 按 group（人）bootstrap 的 95% CI。y=0/1, s=score, groups=人的 id。"""
    y = np.asarray(y, dtype=np.int8)
    s = np.asarray(s, dtype=np.float64)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan"), (float("nan"), float("nan")), int(y.sum()), len(y)
    a = roc_auc_score(y, s)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_of = {u: np.where(g == u)[0] for u in uniq}
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        ii = np.concatenate([idx_of[u] for u in pick])
        if y[ii].sum() in (0, len(ii)):
            continue
        boots.append(roc_auc_score(y[ii], s[ii]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(a), (float(lo), float(hi)), int(y.sum()), int(len(y))


def truth_table(data_dir):
    """从**不去重**那份 val.bin 读真值：{projid: [(age, {token ids at that age}), ...]}，只留临床 token。"""
    res = V.resolve_csv(os.path.join(data_dir, "labels.csv"))
    skip = set(res.IGNORE_TOKENS) | {res.NO_EVENT}
    d = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint32).reshape(-1, 3)
    out = {}
    for pid in np.unique(d[:, 0]):
        rows = d[d[:, 0] == pid]
        by = collections.defaultdict(set)
        for _p, a, t in rows:
            m = int(t) + 1                      # disk -> model space
            if m in skip:
                continue
            by[float(a)].add(m)
        if by:
            out[int(pid)] = [(a, by[a]) for a in sorted(by)]
    return out, res


def score_run(ckpt, dataset, truth, res_truth, mask_ties, device="cpu"):
    """每个 (人, 访视对) 一次 forward，返回逐观测的 (分数, 真值) 记录。"""
    data_dir = os.path.join(HERE, "data", dataset)
    eng = EN.load(os.path.join(HERE, ckpt), data_dir=data_dir, device=device)
    d = np.fromfile(os.path.join(data_dir, "val.bin"), dtype=np.uint32).reshape(-1, 3)

    R = eng.res
    MM = list(R.SCALES["MMSE"])                 # 已按 labels.csv 顺序 = worst-first
    CG = list(R.SCALES["COG"])
    content = torch.tensor(eng.content_ids, dtype=torch.long)
    cmap = {int(t): j for j, t in enumerate(eng.content_ids)}
    recs = []
    for pid, visits in truth.items():
        rows = d[d[:, 0] == pid]
        if len(rows) == 0:
            continue
        ages = rows[:, 1].astype(np.float64)
        toks = rows[:, 2].astype(np.int64) + 1  # disk -> model
        o = np.argsort(ages, kind="stable")
        ages, toks = ages[o], toks[o]
        for i in range(len(visits) - 1):
            a_t, _ = visits[i]
            a_next, tset = visits[i + 1]
            keep = ages <= a_t
            if not keep.any():
                continue
            px = torch.as_tensor(toks[keep][-eng.block_size:], dtype=torch.long)[None, :]
            pa = torch.as_tensor(ages[keep][-eng.block_size:], dtype=torch.float32)[None, :]
            with torch.no_grad():
                lg = eng._logits(px.to(device), pa.to(device), mask_ties)[0].cpu()
            p_content = torch.softmax(lg[content], 0).numpy()
            rec = {"pid": int(pid), "a_t": a_t, "a_next": a_next}
            for fam, ids in (("MMSE", MM), ("COG", CG)):
                pf = torch.softmax(lg[torch.tensor(ids)], 0).numpy()
                true = [t for t in ids if t in tset]
                cur = None
                for t in ids:                      # 本次访视的箱（真值表里 visits[i] 那一格）
                    if t in visits[i][1]:
                        cur = t
                rec[fam] = dict(p=dict(zip(ids, pf.tolist())),
                                truth=(true[0] if len(true) == 1 else None), cur=cur)
            for nm, t in (("AD_DX", R.AD_DX), ("Death", R.DEATH)):
                rec[nm] = dict(p=float(p_content[cmap[t]]), truth=int(t in tset))
            recs.append(rec)
    return recs, eng


def report(name_recs, res, mask_ties):
    names = [n for n, _ in name_recs]
    print(f"\n{'='*94}")
    print(f"下一次访视的判别力（teacher-forced，无 rollout；mask_ties={mask_ties}；"
          f"CI = 按人 bootstrap {N_BOOT} 次）")
    print(f"{'='*94}")

    for subset in ("all", "changed"):
        title = ("全部访视对" if subset == "all" else
                 "只看真实发生了状态改变的访视对（交付版分词表达得出来的那些）")
        print(f"\n--- {title} " + "-" * max(0, 70 - len(title)))
        hdr = f"{'目标':<26s}" + "".join(f"{n + ' AUC':>18s}" for n in names) + f"{'n pos/obs':>16s}"
        print(hdr)
        print("-" * len(hdr))
        for fam, famname in (("MMSE", "MMSE"), ("COG", "cogn_global")):
            ids = list(res.SCALES[fam])
            for t in ids:
                line = f"{res.NAMES[t]:<26s}"
                npos = nobs = 0
                for n, recs in name_recs:
                    y, s, g = [], [], []
                    for r in recs:
                        f = r[fam]
                        if f["truth"] is None or f["cur"] is None:
                            continue
                        if subset == "changed" and f["truth"] == f["cur"]:
                            continue
                        y.append(1 if f["truth"] == t else 0)
                        s.append(f["p"][t])
                        g.append(r["pid"])
                    a, ci, npos, nobs = auc_and_ci(y, s, g)
                    line += f"{a:>8.3f} [{ci[0]:.2f},{ci[1]:.2f}]"
                print(line + f"{f'{npos}/{nobs}':>16s}")
            # 家族的 macro AUC / top-1 准确率
            line = f"{'  -> ' + famname + ' macro/top1':<26s}"
            for n, recs in name_recs:
                aucs, hit, tot = [], 0, 0
                for t in ids:
                    y, s, g = [], [], []
                    for r in recs:
                        f = r[fam]
                        if f["truth"] is None or f["cur"] is None:
                            continue
                        if subset == "changed" and f["truth"] == f["cur"]:
                            continue
                        y.append(1 if f["truth"] == t else 0)
                        s.append(f["p"][t])
                        g.append(r["pid"])
                    if y and 0 < sum(y) < len(y):
                        aucs.append(roc_auc_score(y, s))
                for r in recs:
                    f = r[fam]
                    if f["truth"] is None or f["cur"] is None:
                        continue
                    if subset == "changed" and f["truth"] == f["cur"]:
                        continue
                    tot += 1
                    hit += int(max(f["p"], key=f["p"].get) == f["truth"])
                line += f"{np.mean(aucs):>8.3f} / {hit/max(tot,1):<6.3f}"
            print(line + f"{tot:>16d}")
        if subset == "all":
            for nm in ("AD_DX", "Death"):
                line = f"{nm:<26s}"
                npos = nobs = 0
                for n, recs in name_recs:
                    y = [r[nm]["truth"] for r in recs]
                    s = [r[nm]["p"] for r in recs]
                    g = [r["pid"] for r in recs]
                    a, ci, npos, nobs = auc_and_ci(y, s, g)
                    line += f"{a:>8.3f} [{ci[0]:.2f},{ci[1]:.2f}]"
                print(line + f"{f'{npos}/{nobs}':>16s}")
            print("  （AD_DX / Death 是一次性 token，两份分词里出现次数完全相同，"
                  "所以这两行不受表达能力差异影响，是最干净的对照）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True,
                    help="name:ckpt:dataset，例 dedup:../delphi/X/ckpt.pt:rosmap")
    ap.add_argument("--truth-dataset", default="rosmap_nodedup",
                    help="真值取自哪份 .bin —— 必须是**不去重**那份，它才是忠实记录")
    ap.add_argument("--mask-ties", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    truth, res = truth_table(os.path.join(HERE, "data", a.truth_dataset))
    print(f"真值来自 data/{a.truth_dataset}/val.bin: {len(truth)} 人, "
          f"{sum(len(v) - 1 for v in truth.values())} 个访视对")

    name_recs = []
    for spec in a.runs:
        nm, ckpt, ds = spec.split(":")
        recs, _ = score_run(ckpt, ds, truth, res, bool(a.mask_ties), a.device)
        print(f"  {nm:<10s} dataset={ds:<16s} 打了 {len(recs)} 个观测")
        name_recs.append((nm, recs))
    report(name_recs, res, bool(a.mask_ties))

    if a.out:
        with open(a.out, "w") as fh:
            json.dump({n: [{k: (v if not isinstance(v, dict) else
                               {kk: vv for kk, vv in v.items()})
                            for k, v in r.items()} for r in recs]
                       for n, recs in name_recs}, fh, default=str)
        print(f"\n逐观测记录 -> {a.out}")


if __name__ == "__main__":
    main()
