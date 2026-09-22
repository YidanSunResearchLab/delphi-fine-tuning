"""
probe_batch.py -- 训练时到底喂了什么？直接数 get_batch 产出的**目标** token。

前面全部是从推理行为反推，这个脚本改为直接检查训练输入本身，回答两个具体质疑：

  Q1  token 有没有排错顺序 / 差 ±1？
      做法：按 train.py 的原样调 get_batch，取一个受试者，把 (x, y) 逐位解码成名字，
      和 .bin 里的原始流逐一对照。y 必须恰好是 x 右移一位，且两者都等于 .bin 的 token+1。

  Q2  MMSE_normal 有没有被排除在损失之外？
      做法：复现 model.forward 里 `pass_tokens` 的算法（targets != -1 且不在 ignore_tokens），
      统计每个 token 作为**被计入损失的目标**出现的频率，和语料边际比。
      若 MMSE_normal 的目标频率远低于语料频率，说明它在训练里根本没被充分监督——那就不是
      "学不会"，而是"没喂到"。
"""
import argparse, os, sys, collections
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from radc_delphi import vocab as V  # noqa: E402
_D = os.path.abspath(os.path.join(os.path.dirname(HERE), "delphi"))
sys.path.insert(0, _D)
from utils import get_batch, get_p2i  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    NAMES = V.NAMES
    IGN = set(V.IGNORE_TOKENS)                    # [0] + 2..33，和 ckpt 的 ignore_tokens 一致

    data = np.fromfile(os.path.join(HERE,"data","rosmap","train.bin"), dtype=np.uint32).reshape(-1,3)
    p2i = get_p2i(data)

    # ---------------- Q1：对齐核查
    ix = [0]
    X,A,Y,B = get_batch(ix, data, p2i, block_size=96, device="cpu", select="left",
                        padding="random", lifestyle_augmentations=True,
                        lifestyle_token_range=(3,32), no_event_token_rate=5)
    s,n = int(p2i[0,0]), int(p2i[0,1])
    raw = [(int(data[s+i,1]), int(data[s+i,2])+1) for i in range(n)]
    print("Q1 — 对齐核查（受试者 0）")
    print(f"   .bin 原始流前 6 个 (age_days, model_id, 名字):")
    for ag,t in raw[:6]: print(f"      {ag:6d}  {t:3d}  {NAMES[t]}")
    x0,y0 = X[0].tolist(), Y[0].tolist()
    print(f"   get_batch 的 x[:8]: {[NAMES[t] if t>0 else '<pad>' for t in x0[:8]]}")
    print(f"   get_batch 的 y[:8]: {[NAMES[t] if t>0 else '<pad>' for t in y0[:8]]}")
    shifted_ok = all(y0[i]==x0[i+1] for i in range(len(x0)-1))
    print(f"   y 恰好是 x 右移一位: {shifted_ok}")
    inbin = set(t for _a,t in raw)
    got = set(t for t in x0 if t>0)
    print(f"   x 里出现但 .bin 里没有的 token: {sorted(got-inbin-{1})}   （1=No event 是注入的）")

    # ---------------- Q2：目标频率
    tgt = collections.Counter(); npos=0
    for b in range(a.batches):
        ix = np.random.default_rng(b).integers(0, len(p2i), a.batch_size)
        _X,_A,Y,_B = get_batch(ix, data, p2i, block_size=96, device="cpu", select="left",
                               padding="random", lifestyle_augmentations=True,
                               lifestyle_token_range=(3,32), no_event_token_rate=5)
        y = Y.reshape(-1).tolist()
        for t in y:
            if t == -1 or t in IGN: continue      # 复现 pass_tokens
            tgt[t]+=1; npos+=1
    corpus = collections.Counter((data[:,2]+1).tolist())
    ctot = sum(v for k,v in corpus.items() if k not in IGN)

    print(f"\nQ2 — 作为【被计入损失的目标】的频率   共 {npos:,} 个有效目标位置\n")
    print(f"{'token':24s}{'语料频率':>10s}{'目标频率':>10s}{'比值':>8s}")
    show = [V.ID[n] for n in ("MMSE_normal","MMSE_borderline","MMSE_impaired",
                              "COGN_high","COGN_mid","COGN_low","Death","AD_DX","STROKE")]
    for t in show:
        c = corpus[t]/ctot; g = tgt[t]/max(npos,1)
        print(f"{NAMES[t]:24s}{c:10.4f}{g:10.4f}{g/max(c,1e-9):7.2f}x")
    print(f"\n{'No event (注入的)':24s}{'—':>10s}{tgt[1]/max(npos,1):10.4f}")
    print("\n  比值≈1 表示喂进损失的比例和语料一致；远小于 1 表示这个 token 在训练里被稀释/丢掉了。")

if __name__ == "__main__":
    main()
