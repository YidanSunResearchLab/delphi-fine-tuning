"""
probe_shift.py -- 为什么模型几乎从不发 `MMSE_normal`？两个候选解释的判别实验。

probe_normal.py 测出：在"上下文只有 statics、问新入组者的 MMSE 箱"这道最容易的题上
（3,137 个训练样本），实测 normal 占 80.8%，模型只给 0.9%，而给了 borderline 82.2%。
AUC 0.508 = 瞎猜。模型的分布 (0.169, 0.822, 0.009) 形状上很像实测 (0.070, 0.122, 0.808)
**整体往下错了一位**。两个候选：

  H1  TOKEN ID 差一位。disk→model 的 +1 位移在某处不一致，于是模型在"应该发 normal"时
      发的是它前一个 id。判据：如果成立，**每个**序数家族都会呈现同样的位移，而不只是 MMSE。
      这是代码 bug，影响一切。

  H2  前缀分布不匹配。训练时 get_batch 会注入 no-event 标记，其年龄是**绝对**的
      （1 天、5 年、10 年 … 直到该人最后一次访视），排序后它们绝大多数落在 statics **之前**。
      所以训练时模型在预测基线 MMSE 时，上下文里有 ~15 个 no-event 标记；而 figure2 /
      predict_vs_observe 的前缀是从 .bin 直接读的，**一个标记都没有**。
      判据：把标记按 get_batch 的规则补进前缀后重测，分布是否回到实测附近。
      这不是 bug，是评估与训练的口径差，但同样影响一切。

两条都排除的话，才轮到"样本不够/正则太强"那类解释。

    python probe_shift.py --limit 200
"""
import argparse
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from radc_delphi import engine as EN, batching as Bt  # noqa: E402

DAY = EN.DAYS_PER_YEAR


def with_markers(toks, ages, rate_years=5):
    """按 get_batch 的规则把 no-event 标记插进前缀。

    utils.get_batch 的 'regular' padding：`pad = arange(0, 36525, 365.25*rate) + 1`，
    即**绝对**年龄 1 天、5 年、10 年 …，再剔除晚于该人最后一个真实 token 的那些，最后按
    年龄稳定排序。这里照搬，因为差别正是要测的量。
    """
    m = np.arange(0, 36525, DAY * rate_years) + 1.0
    m = m[m <= float(np.max(ages))]
    t2 = np.concatenate([toks, np.full(len(m), 1, dtype=toks.dtype)])   # 1 = No event
    a2 = np.concatenate([ages, m])
    o = np.argsort(a2, kind="stable")
    return t2[o], a2[o]


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
    res = eng.res
    data, p2i, _ = eng.load_split(a.split)
    N = len(p2i) if not a.limit else min(len(p2i), a.limit)

    # ---------------------------------------------------------------- H1：所有家族一起看
    # 对每个序数家族，在"该家族的首个 token"之前打分，比较家族内分布。
    fams = [k for k, v in res.SCALES.items() if len(v) >= 3]
    print(f"H1 — 逐家族的家族内分布（在该家族首个 token 之前打分），split={a.split}, n<={N}\n")
    print(f"{'家族':8s}{'n':>5s}  {'实测 (低->高 id)':38s}{'模型 (低->高 id)':38s}{'判读':>6s}")
    shifted = 0
    for fam in fams:
        ids = list(res.SCALES[fam])
        emp = np.zeros(len(ids)); mod = np.zeros(len(ids)); n = 0
        for k in range(N):
            ages, toks = Bt.patient_stream(data, *[int(x) for x in p2i[k]])
            pos = [j for j, t in enumerate(toks) if int(t) in ids]
            if not pos or pos[0] == 0:
                continue
            j0 = pos[0]
            p = eng.next_token_probs(toks[:j0], ages[:j0])
            q = p[ids] / max(p[ids].sum(), 1e-30)
            mod += q
            emp[ids.index(int(toks[j0]))] += 1
            n += 1
        if n == 0:
            continue
        emp /= n; mod /= n
        # 位移判据：把实测往低 id 方向挪一位，和模型的相关性是否高于不挪
        corr0 = float(np.corrcoef(emp, mod)[0, 1]) if emp.std() > 0 and mod.std() > 0 else np.nan
        e_sh = np.concatenate([emp[1:], [0.0]])          # model[i] ?= emp[i+1]
        corr1 = float(np.corrcoef(e_sh, mod)[0, 1]) if e_sh.std() > 0 and mod.std() > 0 else np.nan
        tag = "位移!" if (np.isfinite(corr1) and np.isfinite(corr0) and corr1 > corr0 + 0.3) else ""
        shifted += bool(tag)
        f = lambda v: " ".join(f"{x:.2f}" for x in v)
        print(f"{fam:8s}{n:5d}  {f(emp):38s}{f(mod):38s}{tag:>6s}")
    print(f"\n  呈现'整体位移'的家族: {shifted}/{len(fams)}")
    print("  -> 若只有 MMSE/COG 位移而其它家族正常，则不是 id bug（H1 被否）")

    # ---------------------------------------------------------------- H2：补上 no-event 标记
    MM = list(res.SCALES["MMSE"])
    nm = [res.NAMES[t].split("_")[1] for t in MM]
    emp = np.zeros(3); raw = np.zeros(3); mk = np.zeros(3); n = 0
    for k in range(N):
        ages, toks = Bt.patient_stream(data, *[int(x) for x in p2i[k]])
        pos = [j for j, t in enumerate(toks) if int(t) in MM]
        if not pos or pos[0] == 0:
            continue
        j0 = pos[0]
        p1 = eng.next_token_probs(toks[:j0], ages[:j0])
        t2, a2 = with_markers(toks[:j0], ages[:j0])
        p2 = eng.next_token_probs(t2, a2)
        raw += p1[MM] / max(p1[MM].sum(), 1e-30)
        mk += p2[MM] / max(p2[MM].sum(), 1e-30)
        emp[MM.index(int(toks[j0]))] += 1
        n += 1
    emp /= n; raw /= n; mk /= n
    print(f"\nH2 — 前缀里补上 no-event 标记（按 get_batch 的绝对年龄网格），n={n}\n")
    print(f"   {'bin':12s}{'实测':>9s}{'无标记':>9s}{'有标记':>9s}")
    for i in range(3):
        print(f"   {nm[i]:12s}{emp[i]:9.3f}{raw[i]:9.3f}{mk[i]:9.3f}")
    print("\n  -> 若'有标记'一列明显靠近实测，说明是评估前缀缺标记造成的口径差（H2 成立），"
          "\n     figure2 / predict_vs_observe 的全部 rollout 都要按同样方式补标记后重跑。")


if __name__ == "__main__":
    main()
