"""
probe_perturb.py -- 扰动引擎的第一个合理性检验：改心血管相关 token，AD 速率往哪边动？

设计上的四个决定，每个都有理由：

1. **单步，不做 rollout。** 模型的一步预测已验证可信（loss_ce=1.30 对均匀分布 4.56；按训练
   口径的边际比值中位 1.01x）。而 rollout 有自回归漂移（生成的 `MMSE_impaired` 占 0.875，
   真值 0.390），且漂移大小依赖生成出来的历史，所以"扰动前后相减"并不能把它抵消。见 README 3.1。

2. **效应量用速率比，不用概率差。** 模型的 logit 就是 log 速率（`t ~ Exp(exp(logit))`），所以
   `exp(logit_扰动 - logit_原始)` 是一个**速率比**，和流行病学的 hazard ratio 同构，能直接和
   文献对照。用 softmax 概率差会把效应和"其它 token 的速率变了多少"混在一起。

3. **打分点 = 基线那一访的末尾**，和 figure2 的前缀一致。扰动的是基线访视内的 token，
   问的是"入组时如果这个人是另一种状态，模型给的 AD 速率差多少"。

4. **家族内替换**，而不是随便塞 token。`.bin` 的硬不变量是"一次访视内每个量表家族最多一个
   token"（25,055 次访视 0 次违规），所以把 `SBP_high` 换成 `SBP_normal` 是合法的，
   而两个都塞进去会造出数据里不存在的输入。既往病史（*_PREVALENT）是二元的，用增/删。

CONTROL。表里有一行 no-op（把 token 换成它自己），速率比必须**恰好 1.000**。它不是摆设：
若它不等于 1，说明扰动管线本身有副作用（比如改动了排序或年龄），后面所有数字都不可信。

CAVEAT。这里**没有干净的阴性对照** —— 心血管面板里挑不出一个"与 AD 无关"的变量，
所以只能看效应的**方向和相对大小**，不能声称"显著"。
"""
import argparse, os, sys
import numpy as np, torch
HERE=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,HERE)
from radc_delphi import engine as EN  # noqa: E402
from figure2 import figure2_core as F2  # noqa: E402

# (标签, 类型, 参数, 期望方向)  期望方向：'↓' = 常识认为 AD 风险应下降
PERTURB = [
    ("no-op 对照（SBP_normal→SBP_normal）", "swap", ("SBP_normal","SBP_normal"), "=1"),
    ("SBP_high → SBP_normal（降压达标）",   "swap", ("SBP_high","SBP_normal"),   "↓"),
    ("SBP_normal → SBP_high（血压恶化）",   "swap", ("SBP_normal","SBP_high"),   "↑"),
    ("GLU_diabetic → GLU_normal（控糖）",   "swap", ("GLU_diabetic","GLU_normal"),"↓"),
    ("HBA1C_diabetic → HBA1C_normal",       "swap", ("HBA1C_diabetic","HBA1C_normal"),"↓"),
    ("HDL_low → HDL_high（升高好胆固醇）",  "swap", ("HDL_low","HDL_high"),      "↓"),
    ("LDL_high → LDL_optimal（降脂）",      "swap", ("LDL_high","LDL_optimal"),  "↓"),
    ("BMI_high → BMI_mid（减重）",          "swap", ("BMI_high","BMI_mid"),      "↓?"),
    ("加 HTN_PREVALENT（本来没高血压）",    "add",  ("HTN_PREVALENT",),          "↑"),
    ("加 DM_PREVALENT（本来没糖尿病）",     "add",  ("DM_PREVALENT",),           "↑"),
    ("加 HEART_PREVALENT（本来没心脏病）",  "add",  ("HEART_PREVALENT",),        "↑"),
    ("加 STROKE（入组时中风）",             "add",  ("STROKE",),                 "↑↑"),
    ("去掉 HTN_PREVALENT（本来有高血压）",  "del",  ("HTN_PREVALENT",),          "↓"),
    # 位置对照：同一个扰动，故意插到访视末尾（错的位置）。若它和上面"加 STROKE"差很多，
    # 就证明 add 类扰动对插入位置敏感，必须按 spec 顺序放。
    ("加 STROKE【故意放末尾·位置对照】",    "add_tail", ("STROKE",),             "对照"),
    # ---- APOE：AD 最强的遗传风险因子。文献上 ε4 杂合约 3x、纯合约 8-12x（白人队列，
    #      本队列 RACE_white 占多数）。这是扰动引擎最该能回答的靶点，所以单独一组。
    ("APOE ε4: 0 → 1 拷贝",                "apoe", ("APOE_e4_0","APOE_e4_1"),  "↑↑"),
    ("APOE ε4: 1 → 2 拷贝",                "apoe", ("APOE_e4_1","APOE_e4_2"),  "↑↑↑"),
    ("APOE ε4: 0 → 2 拷贝",                "apoe", ("APOE_e4_0","APOE_e4_2"),  "↑↑↑"),
    ("APOE ε4: 1 → 0 拷贝（去掉风险等位）",  "apoe", ("APOE_e4_1","APOE_e4_0"),  "↓↓"),
    ("APOE ε4: 2 → 0 拷贝",                "apoe", ("APOE_e4_2","APOE_e4_0"),  "↓↓↓"),
    ("加 APOE_e2carrier（保护性等位）",      "add",  ("APOE_e2carrier",),        "↓"),
]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../delphi/Delphi-ROSMAP/ckpt.pt")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--threads", type=int, default=8)
    a=ap.parse_args(); torch.set_num_threads(a.threads)
    eng=EN.load(os.path.join(HERE,a.ckpt), data_dir=os.path.join(HERE,"data","rosmap"), device="cpu")
    res=eng.res; ID=res.ID
    AD, DEATH = res.AD_DX, res.DEATH
    ign=torch.tensor(eng.ignore_tokens)

    def rates(tk, ag):
        idx=torch.as_tensor(np.asarray(tk)[None],dtype=torch.long)
        age=torch.as_tensor(np.asarray(ag)[None],dtype=torch.float32)
        with torch.no_grad(): lg,_,_=eng.model(idx,age)
        lg=lg[0,-1].double().clone(); lg[ign]=-torch.inf
        r=torch.exp(lg)
        return float(r[AD]), float(r[DEATH])

    d,p2i,_=eng.load_split(a.split); N=len(p2i) if not a.limit else min(a.limit,len(p2i))
    subs=[]
    for k in range(N):
        ages,toks,_=eng.stream(d,p2i,k)
        bp=F2._baseline(ages,toks)
        if bp is None: continue
        base,pmask,_b=bp
        subs.append((np.asarray(toks)[pmask].copy(), np.asarray(ages,float)[pmask].copy(), base))
    print(f"split={a.split}  可评估 {len(subs)} 人   打分点 = 基线那一访末尾")
    import collections as _c
    cnt=_c.Counter()
    for tk,_ag,_b in subs:
        st=set(int(x) for x in tk)
        dose=next((n for n in ("APOE_e4_0","APOE_e4_1","APOE_e4_2","APOE_unk") if ID[n] in st), "无")
        cnt[(dose, "ε2+" if ID["APOE_e2carrier"] in st else "ε2−")]+=1
    print("  APOE 基线分布: " + "  ".join(f"{k[0]}/{k[1]}={v}" for k,v in sorted(cnt.items())))
    print()
    print(f"{'扰动':36s}{'n':>5s}{'期望':>6s}{'AD 速率比':>12s}{'95% CI':>18s}{'死亡速率比':>12s}")
    print("-"*92)
    rng=np.random.default_rng(0)
    for lab, kind, args, exp in PERTURB:
        la=[]; ld=[]; n_e2_dropped=[0]
        for tk, ag, base in subs:
            t2, a2 = tk.copy(), ag.copy()
            if kind=="swap":
                src,dst=ID[args[0]],ID[args[1]]
                pos=np.where(t2==src)[0]
                if len(pos)==0: continue          # 这个人本来不是这个状态，跳过
                t2[pos[-1]]=dst
            elif kind=="add":
                tok=ID[args[0]]
                if tok in set(int(x) for x in t2): continue   # 已经有了，不能重复（全局去重不变量）
                # 必须插在**spec 顺序里它该在的位置**，不能 append 到末尾。
                # 基线访视的所有 token 同龄，稳定排序会保留输入顺序，所以 append 会让新 token
                # 成为**最后一个**，打分位置就从"MMSE 之后"变成"这个 static 之后" —— 第一版
                # 就是这么写的，四个 add 全部给出 0.56-0.71 的反向结果，那是位置假象不是效应。
                # 真实数据里 *_PREVALENT 排在 statics 中间（约第 9 位），所以插到最后一个
                # static 之后。
                st=set(res.STATIC_IDS)
                idxs=[i for i,x in enumerate(t2) if int(x) in st]
                at = idxs[-1]+1 if idxs else 0
                t2=np.insert(t2,at,tok); a2=np.insert(a2,at,base)
            elif kind=="apoe":
                src,dst=ID[args[0]],ID[args[1]]
                pos=np.where(t2==src)[0]
                if len(pos)==0: continue
                t2[pos[-1]]=dst
                # meta.json 的 apoe_encoding 明写：(dose=2, e2=1) 非法 —— ε4/ε4 不可能同时带 ε2。
                # 扰到纯合就必须把 e2carrier 去掉，否则生成的是一个基因型上不存在的病人。
                if args[1]=="APOE_e4_2":
                    e2=ID["APOE_e2carrier"]
                    q=np.where(t2==e2)[0]
                    if len(q):
                        n_e2_dropped[0]+=1
                        t2=np.delete(t2,q); a2=np.delete(a2,q)
            elif kind=="add_tail":
                tok=ID[args[0]]
                if tok in set(int(x) for x in t2): continue
                t2=np.append(t2,tok); a2=np.append(a2,base)
            elif kind=="del":
                tok=ID[args[0]]
                pos=np.where(t2==tok)[0]
                if len(pos)==0: continue
                t2=np.delete(t2,pos); a2=np.delete(a2,pos)
            # 稳定排序：同龄 token 保留插入顺序，所以上面精心选的位置不会被打乱
            o=np.argsort(a2,kind="stable"); t2,a2=t2[o],a2[o]
            ad0,dt0=rates(tk,ag); ad1,dt1=rates(t2,a2)
            if ad0>0 and dt0>0: la.append(ad1/ad0); ld.append(dt1/dt0)
        if not la:
            print(f"{lab:36s}{0:5d}{exp:>6s}{'无可扰动的人':>12s}"); continue
        la=np.array(la); ld=np.array(ld)
        g=float(np.exp(np.mean(np.log(la))))      # 几何均值，比值的正确平均方式
        bs=[float(np.exp(np.mean(np.log(la[rng.integers(0,len(la),len(la))])))) for _ in range(500)]
        lo,hi=np.percentile(bs,[2.5,97.5])
        gd=float(np.exp(np.mean(np.log(ld))))
        note = f"  (校验: 去掉 e2carrier {n_e2_dropped[0]} 人)" if n_e2_dropped[0] else ""
        print(f"{lab:36s}{len(la):5d}{exp:>6s}{g:12.3f}  [{lo:.3f},{hi:.3f}]{gd:12.3f}{note}")
    print("-"*92)
    print("速率比 >1 = 扰动后 AD 速率上升。几何均值 + 受试者级 bootstrap（500 次）。")

if __name__=="__main__":
    main()
