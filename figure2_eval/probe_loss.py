"""
probe_loss.py -- 损失到底在什么水平？模型有没有在学？

线索：ckpt 的 best_val_loss = 9.82，而 129 个 token 的**均匀分布**交叉熵才 ln(129)=4.86。
如果 loss_ce 本身就在均匀分布附近或更差，那"模型没学会某个 token"就不是那个 token 的问题，
而是整个分类头没训起来。train.py 的 best_val_loss 是 ce+dt 之和，两项必须拆开看。

同时按**同一口径**比较模型的平均预测分布与训练目标的经验分布（都含 No event，不做任何
重归一化），列出偏差最大的 token —— 这是判断"个别 token 异常"还是"整体没拟合"的直接依据。
"""
import argparse, os, sys, collections
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from radc_delphi import engine as EN, vocab as V  # noqa: E402
_D = os.path.abspath(os.path.join(os.path.dirname(HERE), "delphi")); sys.path.insert(0, _D)
from utils import get_batch, get_p2i  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../delphi/Delphi-ROSMAP/ckpt.pt")
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    eng = EN.load(os.path.join(HERE,a.ckpt), data_dir=os.path.join(HERE,"data","rosmap"), device="cpu")
    ck = eng.ckpt
    print(f"checkpoint: iter={ck.get('iter_num')}  best_val_loss={ck.get('best_val_loss'):.4f}")
    IGN=set(V.IGNORE_TOKENS); NAMES=V.NAMES
    n_target = len([t for t in range(V.VOCAB_SIZE) if t not in IGN])
    print(f"参考：{n_target} 个可作目标的 token，均匀分布的 CE = ln({n_target}) = {np.log(n_target):.3f}\n")

    for split in ("train","val"):
        d=np.fromfile(os.path.join(HERE,"data","rosmap",f"{split}.bin"),dtype=np.uint32).reshape(-1,3)
        p2i=get_p2i(d)
        ce=[]; dt=[]
        pred=np.zeros(V.VOCAB_SIZE); tgt=collections.Counter(); npos=0
        for b in range(a.batches):
            ix=np.random.default_rng(b).integers(0,len(p2i),64)
            X,A,Y,B = get_batch(ix,d,p2i,block_size=96,device="cpu",select="left",
                                padding="random",lifestyle_augmentations=True,
                                lifestyle_token_range=(3,32),no_event_token_rate=5)
            with torch.no_grad():
                logits,loss,_ = eng.model(X,A,Y,B)
            ce.append(float(loss["loss_ce"])); dt.append(float(loss["loss_dt"]))
            # 同口径的边际比较
            y=Y.reshape(-1); keep=(y!=-1)
            for k in IGN: keep &= (y!=k)
            p=torch.softmax(logits.reshape(-1,logits.size(-1))[keep],-1)
            pred+=p.sum(0).numpy(); npos+=int(keep.sum())
            for t in y[keep].tolist(): tgt[t]+=1
        pred/=max(npos,1)
        print(f"[{split}]  loss_ce={np.mean(ce):.4f}   loss_dt={np.mean(dt):.4f}   "
              f"合计={np.mean(ce)+np.mean(dt):.4f}")
        if split=="train":
            emp=np.array([tgt[t]/npos for t in range(V.VOCAB_SIZE)])
            rat=np.where(pred>1e-12, emp/np.maximum(pred,1e-12), np.nan)
            cand=[t for t in range(V.VOCAB_SIZE) if t not in IGN and emp[t]>0.002]
            cand.sort(key=lambda t: -rat[t])
            print(f"\n  同口径边际对比（训练目标 vs 模型预测，均含 No event，频率>0.2% 的 {len(cand)} 个）")
            print(f"  {'token':22s}{'目标频率':>10s}{'模型预测':>10s}{'比值':>8s}")
            for t in cand[:6]: print(f"  {NAMES[t]:22s}{emp[t]:10.4f}{pred[t]:10.4f}{rat[t]:7.2f}x")
            print("   ...")
            for t in cand[-4:]: print(f"  {NAMES[t]:22s}{emp[t]:10.4f}{pred[t]:10.4f}{rat[t]:7.2f}x")
            print(f"\n  比值的分布: 中位 {np.nanmedian([rat[t] for t in cand]):.2f}x  "
                  f"最大 {np.nanmax([rat[t] for t in cand]):.1f}x  最小 {np.nanmin([rat[t] for t in cand]):.2f}x")
        print()

if __name__ == "__main__":
    main()
