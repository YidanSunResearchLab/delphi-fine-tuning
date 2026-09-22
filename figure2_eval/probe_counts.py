"""
probe_counts.py -- MMSE 三个箱的**原始次数**：rollout 吐了多少次 vs 实际发生多少次。

前面都在比"率"，这里直接报次数，并按两件事分层，因为混在一起会看不出问题在哪：

  * 前缀里 normal **是否可用**。79% 的人基线就是 normal，而 build.py:204 的全局去重保证
    每个 token 每人最多一次，所以对这些人 rollout **结构上不可能**再吐 normal —— 把他们
    混进分母，"低估"里有一部分是数据定义造成的，不是模型的错。
  * 窗口：基线后 5 年 vs 全部随访。

"预测次数"的定义：每个受试者 100 条轨迹里发射该 token 的**比例**，按人加总 = 期望人数；
同时报轨迹级的原始计数（受试者数 x 100 条）。
"""
import argparse, os, sys
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from radc_delphi import engine as EN  # noqa: E402
from figure2 import figure2_core as F2  # noqa: E402

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../delphi/Delphi-ROSMAP/ckpt.pt")
    ap.add_argument("--n-mc", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a=ap.parse_args(); torch.set_num_threads(a.threads)
    eng=EN.load(os.path.join(HERE,a.ckpt), data_dir=os.path.join(HERE,"data","rosmap"), device="cpu")
    D=EN.DAYS_PER_YEAR
    MM=list(eng.res.SCALES["MMSE"]); nm=[eng.res.NAMES[t].split("_")[1] for t in MM]
    d,p2i,_=eng.load_split("val"); N=len(p2i) if not a.limit else min(a.limit,len(p2i))

    # [窗口][可用性][token] -> dict
    box={}
    for w in ("5y","全随访"):
        for av in ("normal 可用","normal 已封"):
            box[(w,av)]={'n_subj':0,'obs':np.zeros(3),'exp':np.zeros(3),
                         'traj_hit':np.zeros(3),'n_traj':0}
    for k in range(N):
        ages,toks,_=eng.stream(d,p2i,k)
        bp=F2._baseline(ages,toks)
        if bp is None: continue
        base,pmask,_b=bp
        a_=np.asarray(ages,float); t_=np.asarray(toks)
        last=float(a_[a_>0].max()); fu=(last-base)/D
        avail = MM[2] not in set(int(x) for x in t_[pmask])   # 前缀里有没有 normal
        key_av = 'normal 可用' if avail else 'normal 已封'
        T,A=eng.simulate(t_[pmask],a_[pmask],n_mc=a.n_mc,seed=42+k,
                         until_age_years=max(105.,base/D+16))
        real=A>EN.PAD_AGE+1
        for w,hd in (("5y",base+5*D),("全随访",base+fu*D)):
            if w=="5y" and not (fu>=5-1e-6 or (t_[(a_>base)&(a_<=hd)]==eng.res.DEATH).any()):
                continue                                      # 5 年窗口的删失规则
            b=box[(w,key_av)]; b['n_subj']+=1; b['n_traj']+=a.n_mc
            ow=(a_>base)&(a_<=hd); sw=real&(A>base)&(A<=hd)
            for i,tok in enumerate(MM):
                b['obs'][i]+= float((t_[ow]==tok).any())      # 该人有没有出现（0/1）
                hit=((T==tok)&sw).any(1)
                b['exp'][i]+= float(hit.mean())               # 期望人数
                b['traj_hit'][i]+= float(hit.sum())           # 轨迹级命中数

    for (w,av),b in box.items():
        if b['n_subj']==0: continue
        print(f"\n【{w} 窗口 · {av}】 {b['n_subj']} 人 · {b['n_traj']:,} 条轨迹")
        print(f"   {'bin':12s}{'实际人数':>10s}{'预测期望人数':>14s}{'比值':>8s}"
              f"{'轨迹命中/总数':>20s}")
        for i in range(3):
            r = b['obs'][i]/b['exp'][i] if b['exp'][i]>0.5 else float('inf')
            rs = f"{r:.1f}x" if np.isfinite(r) else "inf"
            print(f"   {nm[i]:12s}{int(b['obs'][i]):10d}{b['exp'][i]:14.1f}{rs:>8s}"
                  f"{int(b['traj_hit'][i]):>12,}/{b['n_traj']:,}")

if __name__ == "__main__":
    main()
