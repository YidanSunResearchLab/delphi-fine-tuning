"""
probe_statics.py -- 评估时 background token 的年龄放错了吗？

训练时 `get_batch(lifestyle_augmentations=True, lifestyle_token_range=(3,32))` 把背景块
（model id 4..33：队列/教育/种族/APOE/吸烟/饮酒/既往病史）的年龄**随机偏移 −20 ~ +40 年**，
用来消除 immortality bias。性别（model 2,3）不在范围内，不偏移。

而 figure2 / predict_vs_observe 的前缀是直接从 .bin 读的，背景块**全部落在基线年龄上**——
在我们的分词里 statics 和基线那一访同龄（实测同一天）。这是训练分布里几乎不出现的配置。

这个脚本在"该受试者第一个 MMSE token 之前"这个位置，比较四种前缀下模型的 MMSE 家族内分布：
  as-is      背景块在基线年龄（= 现在的评估做法）
  jittered   背景块按训练规则随机偏移（多次抽样取平均）
  no-statics 干脆去掉背景块
  shift-20y  背景块统一提前 20 年（固定偏移，用来区分"随机性"和"位置"哪个重要）

若 jittered 明显把 normal 拉回实测的 0.80 附近，则现在所有 rollout 的**种子前缀都是离群的**，
figure2 与 predict_vs_observe 全部要按训练口径重跑。
"""
import argparse, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from radc_delphi import engine as EN, batching as Bt  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../delphi/Delphi-ROSMAP/ckpt.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=250)
    ap.add_argument("--draws", type=int, default=5, help="jittered 变体的重复抽样次数")
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    eng = EN.load(os.path.join(HERE, a.ckpt), data_dir=os.path.join(HERE,"data","rosmap"), device="cpu")
    res = eng.res
    MM = list(res.SCALES["MMSE"]); nm=[res.NAMES[t].split("_")[1] for t in MM]
    BG = np.arange(4, 34)                    # model 空间的背景块 = disk 3..32
    data, p2i, _ = eng.load_split(a.split)
    N = len(p2i) if not a.limit else min(len(p2i), a.limit)
    rng = np.random.default_rng(0)

    def dist(toks, ages):
        p = eng.next_token_probs(toks, ages); return p[MM]/max(p[MM].sum(),1e-30)

    emp=np.zeros(3); acc={k:np.zeros(3) for k in ("as-is","jittered","no-statics","shift-20y")}; n=0
    for k in range(N):
        ages, toks = Bt.patient_stream(data, *[int(x) for x in p2i[k]])
        pos=[j for j,t in enumerate(toks) if int(t) in MM]
        if not pos or pos[0]==0: continue
        j0=pos[0]; t0=toks[:j0].copy(); a0=ages[:j0].copy()
        isbg=np.isin(t0, BG)
        emp[MM.index(int(toks[j0]))]+=1; n+=1
        acc["as-is"] += dist(t0,a0)
        keep=~isbg
        if keep.sum()>0: acc["no-statics"] += dist(t0[keep],a0[keep])
        a2=a0.copy(); a2[isbg]-=20*365.25
        o=np.argsort(a2,kind="stable"); acc["shift-20y"] += dist(t0[o],a2[o])
        s=np.zeros(3)
        for _ in range(a.draws):
            a3=a0.copy()
            a3[isbg]+=rng.integers(-20*365, 40*365, size=int(isbg.sum())).astype(float)
            o=np.argsort(a3,kind="stable"); s += dist(t0[o],a3[o])
        acc["jittered"] += s/a.draws
    emp/=n
    print(f"split={a.split}  n={n}  背景块 token 数/人 ≈ {int(isbg.sum())}\n")
    print(f"   {'bin':12s}{'实测':>9s}" + "".join(f"{k:>12s}" for k in acc))
    for i in range(3):
        print(f"   {nm[i]:12s}{emp[i]:9.3f}" + "".join(f"{acc[k][i]/n:12.3f}" for k in acc))
    print("\n  as-is = 现在的评估做法；jittered = 训练时的口径")

if __name__ == "__main__":
    main()
