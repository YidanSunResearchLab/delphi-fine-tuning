"""
probe_masktie.py -- mask_ties 是不是训练路径与生成路径分歧的唯一来源？

probe_loss 测出：走训练路径（get_batch + targets）时模型的边际几乎完美（中位 1.01x），
走推理路径（next_token_probs，无 targets）时 MMSE_normal 塌 17 倍。两条路径唯一的
**模型内部**差别是 model.forward 里这一段：

    if targets is not None and self.config.mask_ties:
        attn_mask *= (age != targets_age)      # 屏蔽与目标同龄的 token

传 targets 才生效。而本分词里一次访视的所有 token **同龄**，所以训练时预测某个 token 会屏蔽掉
它的同访视兄弟（含 statics），推理时全部可见 —— 这是模型从未训练过的配置。

做法：同一批数据、同一批位置，只切换 targets 传不传，比较在"目标是 MMSE token"的位置上，
模型给 MMSE 家族内的分布。两者若差异巨大，则 rollout 的每一步都用错了模型。
"""
import argparse, os, sys
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from radc_delphi import engine as EN, vocab as V  # noqa: E402
_D = os.path.abspath(os.path.join(os.path.dirname(HERE), "delphi")); sys.path.insert(0, _D)
from utils import get_batch, get_p2i  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../delphi/Delphi-ROSMAP/ckpt.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--batches", type=int, default=30)
    ap.add_argument("--threads", type=int, default=8)
    a=ap.parse_args(); torch.set_num_threads(a.threads)
    eng=EN.load(os.path.join(HERE,a.ckpt), data_dir=os.path.join(HERE,"data","rosmap"), device="cpu")
    MM=list(V.SCALES["MMSE"]); nm=[V.NAMES[t].split("_")[1] for t in MM]
    d=np.fromfile(os.path.join(HERE,"data","rosmap",f"{a.split}.bin"),dtype=np.uint32).reshape(-1,3)
    p2i=get_p2i(d)
    # 分两类位置：基线那一访的首个 MMSE（rollout 不生成，它在前缀里）
    # 与后续转移（rollout 只生成这一类）。混在一起会被前者主导。
    acc={k:{'emp':np.zeros(3),'on':np.zeros(3),'off':np.zeros(3),'n':0}
         for k in ('首个 MMSE','后续转移')}
    for b in range(a.batches):
        ix=np.random.default_rng(b).integers(0,len(p2i),64)
        X,A,Y,B=get_batch(ix,d,p2i,block_size=96,device="cpu",select="left",padding="random",
                          lifestyle_augmentations=True,lifestyle_token_range=(3,32),
                          no_event_token_rate=5)
        with torch.no_grad():
            lg_on,_,_  = eng.model(X,A,Y,B)     # targets 传入 -> mask_ties 生效（训练口径）
            lg_off,_,_ = eng.model(X,A)         # 不传 -> mask_ties 失效（生成口径）
        Yr=Y.reshape(-1)
        P_on =torch.softmax(lg_on.reshape(-1,lg_on.size(-1)),-1)[:,MM]
        P_off=torch.softmax(lg_off.reshape(-1,lg_off.size(-1)),-1)[:,MM]
        P_on =P_on /P_on.sum(1,keepdim=True).clamp(min=1e-30)
        P_off=P_off/P_off.sum(1,keepdim=True).clamp(min=1e-30)
        Yb=Y.cpu().numpy()
        for r in range(Yb.shape[0]):
            seen=False
            for c in range(Yb.shape[1]):
                t=int(Yb[r,c])
                if t not in MM: continue
                key='后续转移' if seen else '首个 MMSE'
                seen=True
                k_=r*Yb.shape[1]+c
                acc[key]['emp'][MM.index(t)]+=1
                acc[key]['on'] +=P_on[k_].numpy()
                acc[key]['off']+=P_off[k_].numpy()
                acc[key]['n']+=1
    for key,d_ in acc.items():
        n=d_['n']
        if not n: continue
        emp,on,off=d_['emp']/n, d_['on']/n, d_['off']/n
        tag = "（rollout 不生成，在前缀里）" if key=='首个 MMSE' else "（rollout 只生成这一类）"
        print(f"{key}  n={n}  {tag}")
        print(f"   {'bin':12s}{'真实':>9s}{'训练口径':>12s}{'生成口径':>12s}")
        for i in range(3):
            print(f"   {nm[i]:12s}{emp[i]:9.3f}{on[i]:12.3f}{off[i]:12.3f}")
        print(f"   normal 低估:  训练口径 {emp[2]/max(on[2],1e-9):.1f}x   "
              f"生成口径 {emp[2]/max(off[2],1e-9):.1f}x\n")

if __name__ == "__main__":
    main()
