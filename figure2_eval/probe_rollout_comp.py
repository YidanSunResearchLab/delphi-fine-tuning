"""
probe_rollout_comp.py -- rollout 实际发出的 MMSE 组成，对上哪一种口径？

probe_masktie 在"后续转移"位置上测到：训练口径 normal=0.151（真值 0.118，基本对），
生成口径 normal=0.016（低估 7.3x）。而理论上 visit_sizes 开启后，rollout 生成新访视的
**第一个** token 时上下文里没有同龄 token，应当等价于训练口径。

理论和 figure2 的结果（MMSE Normal 仍低估 39x）对不上，所以直接量 rollout 自己发出来的
MMSE token 里 normal 占多少，看它落在 0.151 还是 0.016 附近 —— 这一个数就能判定 rollout
到底处在哪个口径，不必再推理。

同时按"normal 是否仍可用"分层：79% 的人基线就是 normal，全局不重复会永久封掉它，
混在一起算会被这部分主导。
"""
import argparse, os, sys, collections
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from radc_delphi import engine as EN  # noqa: E402

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../delphi/Delphi-ROSMAP/ckpt.pt")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--n-mc", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    a=ap.parse_args(); torch.set_num_threads(a.threads)
    eng=EN.load(os.path.join(HERE,a.ckpt), data_dir=os.path.join(HERE,"data","rosmap"), device="cpu")
    D=EN.DAYS_PER_YEAR; MM=list(eng.res.SCALES["MMSE"]); NORM=eng.res.ID["MMSE_normal"]
    nm=[eng.res.NAMES[t].split("_")[1] for t in MM]
    d,p2i,_=eng.load_split("val")
    N=min(a.limit,len(p2i))
    for uvs,mt in ((True,True),(True,False),(False,True)):
        cnt={'可用':np.zeros(3),'已封':np.zeros(3)}
        for k in range(N):
            ages,toks,_=eng.stream(d,p2i,k)
            va=np.unique(ages[ages>0])
            if len(va)<2: continue
            base=va[0]; m=ages<=base; npre=int(m.sum())
            avail = NORM not in set(int(x) for x in toks[m])   # 前缀里有没有 normal
            key='可用' if avail else '已封'
            T,A=eng.simulate(toks[m],ages[m],n_mc=a.n_mc,seed=k,
                             until_age_years=max(105.,base/D+16),use_visit_sizes=uvs,mask_ties=mt)
            real=A>EN.PAD_AGE+1
            new=real.copy(); new[:,:npre]=False
            for i,t in enumerate(MM):
                cnt[key][i]+= float(((T==t)&new).sum())
        print(f"use_visit_sizes={uvs}  mask_ties={mt}")
        for key in ('可用','已封'):
            s=cnt[key].sum()
            if s==0: continue
            c=cnt[key]/s
            print(f"   前缀里 normal {key}: " + "  ".join(f"{nm[i]}={c[i]:.3f}" for i in range(3))
                  + f"   (共 {int(s)} 个 MMSE token)")
        print()
    print("参照（probe_masktie 的'后续转移'位置）：真值 normal=0.118  训练口径 0.151  生成口径 0.016")

if __name__ == "__main__":
    main()
