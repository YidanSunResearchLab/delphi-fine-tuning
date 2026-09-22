"""
probe_normal.py -- 模型到底有没有学会发 `MMSE_normal`？

问题背景。`MMSE_normal` 是语料里最常见的 MMSE token（3,508 个 / 50.3%），模型却几乎不预测它
（rollout 里 P(到达 Normal) 比实测低 39 倍）。一个自然的反问是：样本这么多，模型没道理学不会。

这个脚本把「学没学会」拆成**两个不同的位置**分别测，因为那 3,508 个样本并不是同一种题目：

  位置 A —— 基线那一访的首个 MMSE token（3,137 个，占 89.4%）
      上下文 = statics（性别/队列/教育/APOE/吸烟/既往病史…）+ 注入的 no-event 标记。
      题目是"这个新入组的人，MMSE 大概在哪个箱"。79% 的人答案是 normal。
      **rollout 从不问这道题**：基线那一访是当作前缀喂进去的。

  位置 B —— 基线之后的转移（371 个，占 10.6%）
      上下文 = 完整病史。题目是"这个已经掉到 B/I 的人，会不会回到 normal"。
      **rollout 只问这道题**。

如果模型在 A 上答得好、在 B 上答得差，那结论就不是"样本不够所以没学会"，而是
**"学会的是另一道题"**——而评估只考 B。这两种情况的修法完全不同：前者要改评估口径或
生成方式，后者要加样本/重加权。

    python probe_normal.py            # 全量 val
    python probe_normal.py --limit 60
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from radc_delphi import engine as EN, batching as Bt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../delphi/Delphi-ROSMAP/ckpt.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    eng = EN.load(os.path.join(HERE, a.ckpt), data_dir=os.path.join(HERE, "data", "rosmap"),
                  device="cpu")
    MM = list(eng.res.SCALES["MMSE"])
    NORM = eng.res.ID["MMSE_normal"]
    nm = [eng.res.NAMES[t].split("_")[1] for t in MM]
    data, p2i, _ = eng.load_split(a.split)
    N = len(p2i) if not a.limit else min(len(p2i), a.limit)

    # ---- 位置 A：预测基线那一访的首个 MMSE token
    A_pred, A_true = [], []
    # ---- 位置 B：预测基线之后的 MMSE 转移（只在 normal 仍可用时）
    B_pred, B_true = [], []
    for k in range(N):
        ages, toks = Bt.patient_stream(data, *[int(x) for x in p2i[k]])
        pos = [j for j, t in enumerate(toks) if int(t) in MM]
        if not pos:
            continue
        # A：在首个 MMSE token 之前打分。前缀非空才有意义。
        j0 = pos[0]
        if j0 > 0:
            p = eng.next_token_probs(toks[:j0], ages[:j0])
            q = p[MM] / max(p[MM].sum(), 1e-30)
            A_pred.append(q)
            A_true.append(MM.index(int(toks[j0])))
        # B：后续每一次 MMSE 转移
        emitted = {int(toks[j0])}
        for j_prev, j_next in zip(pos, pos[1:]):
            if NORM not in emitted:                       # normal 仍可用才算数
                p = eng.next_token_probs(toks[:j_next], ages[:j_next])
                q = p[MM] / max(p[MM].sum(), 1e-30)
                B_pred.append(q)
                B_true.append(MM.index(int(toks[j_next])))
            emitted.add(int(toks[j_next]))

    def report(pred, true, title, note):
        if not pred:
            print(f"\n{title}: 无样本"); return
        P = np.stack(pred); T = np.array(true)
        emp = np.array([(T == i).mean() for i in range(3)])
        mod = P.mean(0)
        print(f"\n{title}   n={len(T)}   {note}")
        print(f"   {'bin':12s}{'实测占比':>10s}{'模型预测':>10s}{'比值':>9s}")
        for i in range(3):
            print(f"   {nm[i]:12s}{emp[i]:10.3f}{mod[i]:10.3f}{mod[i]/max(emp[i],1e-9):8.2f}x")
        # 逐样本的判别力：模型给"真答案是 normal"的那些样本更高的 normal 分数吗
        y = (T == MM.index(NORM)).astype(int)
        s = P[:, MM.index(NORM)]
        if 0 < y.sum() < len(y):
            from sklearn.metrics import roc_auc_score
            print(f"   AUC(模型的 normal 分数 vs 真答案是否为 normal) = {roc_auc_score(y, s):.3f}")

    print(f"split={a.split}  受试者 {N}")
    report(A_pred, A_true, "位置 A — 基线首个 MMSE（rollout 从不问这道题）",
           "上下文只有 statics")
    report(B_pred, B_true, "位置 B — 基线之后的转移（rollout 只问这道题）",
           "上下文是完整病史，且 normal 仍可用")
    print("\n读法：若 A 的 normal 比值接近 1 而 B 远小于 1，说明模型学会的是"
          "\n'新入组的人多半正常'，而不是'谁会恢复'——评估只考后者。")


if __name__ == "__main__":
    main()
