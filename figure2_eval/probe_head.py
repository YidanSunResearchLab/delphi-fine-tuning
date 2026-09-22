"""
probe_head.py -- 直接看输出头：模型对某些 token 的抑制是写在权重里的吗？

前面把上下文侧的解释全排除了（id 位移、no-event 标记、statics 年龄、mask_ties、采样器），
模型在**任何**上下文下都几乎不发 `MMSE_normal`。那就该看参数本身。

Delphi 的 lm_head 无 bias（DelphiConfig.bias=False），所以 logit_k = w_k · h。
若某个 token 的 ||w_k|| 远小于同伴，它的 logit 就被钉在 0 附近、竞赛里永远输——这是"抑制写在
权重里"的直接证据，且与上下文无关，正好解释我们看到的现象。

同时报出每个 token 的经验边际频率与模型平均预测频率，按低估倍数排序，看 MMSE_normal 是不是
离群点还是一批 token 的共性。
"""
import argparse, os, sys, collections
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from radc_delphi import engine as EN, batching as Bt  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../delphi/Delphi-ROSMAP/ckpt.pt")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args(); torch.set_num_threads(a.threads)
    eng = EN.load(os.path.join(HERE,a.ckpt), data_dir=os.path.join(HERE,"data","rosmap"), device="cpu")
    res = eng.res
    W = eng.model.lm_head.weight.detach()          # (vocab, n_embd)
    nrm = W.norm(dim=1).numpy()
    content = [t for t in range(res.VOCAB_SIZE)
               if t not in set(res.IGNORE_TOKENS) and t != res.NO_EVENT
               and t not in set(res.PATHOLOGY_IDS)]

    # 训练语料里的经验边际（token 作为"下一个 token"出现的频率）
    tr = np.fromfile(os.path.join(HERE,"data","rosmap","train.bin"), dtype=np.uint32).reshape(-1,3)
    cnt = collections.Counter((tr[:,2]+1).tolist())
    tot = sum(cnt[t] for t in content)

    # 模型的平均预测频率（真实前缀上）
    data,p2i,_ = eng.load_split("val")
    N = min(a.limit, len(p2i)); acc=np.zeros(res.VOCAB_SIZE); n=0
    for k in range(N):
        ages,toks = Bt.patient_stream(data, *[int(x) for x in p2i[k]])
        for j in range(5, len(toks)-1, 4):
            p = eng.next_token_probs(toks[:j+1], ages[:j+1]); acc += p; n+=1
    mod = acc/max(n,1)
    modc = mod[content]/max(mod[content].sum(),1e-30)
    empc = np.array([cnt[t] for t in content], float); empc/=empc.sum()

    print(f"lm_head 无 bias: {eng.model.lm_head.bias is None}   打分位置 n={n}\n")
    print(f"{'token':24s}{'经验边际':>10s}{'模型边际':>10s}{'低估':>9s}{'||w||':>9s}{'w 排名':>8s}")
    order = np.argsort(-(empc/np.maximum(modc,1e-12)))
    rank = {t: r for r, t in enumerate(sorted(content, key=lambda x: -nrm[x]))}
    for i in order[:8]:
        t = content[i]
        print(f"{res.display(t):24s}{empc[i]:10.4f}{modc[i]:10.4f}"
              f"{empc[i]/max(modc[i],1e-12):8.1f}x{nrm[t]:9.3f}{rank[t]:5d}/{len(content)}")
    print("  ...")
    for i in order[-3:]:
        t = content[i]
        print(f"{res.display(t):24s}{empc[i]:10.4f}{modc[i]:10.4f}"
              f"{empc[i]/max(modc[i],1e-12):8.1f}x{nrm[t]:9.3f}{rank[t]:5d}/{len(content)}")
    print(f"\n||w|| 统计（content token）: 中位 {np.median(nrm[content]):.3f}  "
          f"最小 {nrm[content].min():.3f}  最大 {nrm[content].max():.3f}")
    print("\n各量表家族的 ||w||（看最好的箱是否被钉住）:")
    for fam in ("MMSE","COG","BMI","SBP"):
        ids = res.SCALES.get(fam, ())
        print(f"  {fam:6s}" + "  ".join(f"{res.NAMES[t].split('_')[1]}={nrm[t]:.3f}" for t in ids))

if __name__ == "__main__":
    main()
