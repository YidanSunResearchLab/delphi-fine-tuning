"""
perturbation_eval.py -- 扰动引擎的评估套件：量的是**反事实可不可信**，不是判别力。

    python perturbation_eval.py \
        --runs nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
        --n-mc 100 --workers 30 --out results/perturbation_eval_default.json
    python perturbation_eval.py --runs ... --limit 20        # 冒烟，几分钟

在 RIS（compute2 / Slurm）上跑。**本文件不改 ris/figure2.sbatch**，所以要么用 --wrap，
要么在那个 case 里加一行 `pe) python -u perturbation_eval.py ... ;;`：

    sbatch --job-name=rosmap-pe --partition=general-cpu --account=compute2-mdan \
           --cpus-per-task=32 --mem=64G --time=06:00:00 --exclude=c2-node-011 \
           --output=logs/pe_%j.out --error=logs/pe_%j.err --wrap \
      'source $AD_CONDA/etc/profile.d/conda.sh && conda activate ad-projection && \
       export OMP_NUM_THREADS=1 MKL_THREADING_LAYER=GNU && cd $AD_BASE/figure2_eval && \
       python -u perturbation_eval.py --runs \
         dedup:../delphi/Delphi-ROSMAP-gpubase/ckpt.pt:rosmap \
         nodedup:../delphi/Delphi-ROSMAP-nodedup/ckpt.pt:rosmap_nodedup \
         --n-mc 100 --workers 30 --out results/perturbation_eval_default.json'

`OMP_NUM_THREADS=1` 必须由 sbatch 设（每个 MC worker 单线程，否则 30 个进程各开 32 线程
严重超订，wall time 反而变长）。成本量级：每人 7 条 rollout，比 figure2 的 MC pass 贵 7 倍，
但 --cap-years 默认只跑到基线 +11 年（figure2 跑到 105 岁）把它拉回来一半左右。

这个项目叫 "perturbation engine"，但到 2026-09-23 为止**扰动能力一个数都没量过**。
判别力已经被量到烂了（README 第 8、9 节：边际 AUC、下一次访视、逻辑回归对照、直接判别头），
而第 9.4 节的结论正是"transformer 在判别上不比一个基线前缀的 LR 强，它的正当性得落在
LR 做不到的事上：反事实模拟"。**这个文件就是去检验那句话有没有资格说出口。**

判别力和反事实可信度是两件事，而且可以完全脱钩：一个把所有人都排对序的模型，完全可能
对"如果这个人 APOE 是 ε4/ε4 会怎样"给出一个和"如果这个人从 ROS 队列换到 MAP 队列会怎样"
一样大的反应 —— 那样的模型 AUC 再高也不能用来做干预模拟，因为它把**任何**输入扰动都翻译
成了输出扰动。AUC 测不出这件事，只有下面这四条能。

--------------------------------------------------------------------------------------
四条测试，以及**每一条失败分别意味着什么**

  0. MC 噪声地板（`ref` vs `ref2`：同一个未扰动前缀，两个独立 seed）
     先算、先打印，后面每一个 delta 都要给出"是噪声地板的几倍"。
     失败（地板和效应同量级）= 报出来的所有"效应"都可能只是采样噪声，要么加 n_mc，
     要么这个 readout 根本不可用。**没有这一条，下面三条一个数都不能引用。**

  1. 剂量反应：APOE ε4 拷贝数 0 / 1 / 2 -> 5 年 AD 风险，应当**单调上升**
     （文献：ε4 杂合约 3x、纯合约 8-12x，白人队列；本队列 RACE_white 占多数）。
     失败（非单调、或 2 拷贝不高于 0 拷贝）= 引擎连教科书级、且在训练数据里被反复见过的
     因果方向都复现不出来，任何"新"的反事实结论都没有可信度。

  2. 零扰动对照：`STUDY_ROS` <-> `STUDY_MAP` 互换。**这是最重要的一条。**
     队列标签是"这个人是从哪个研究招进来的"，不是生物学变量，对 5 年 AD 风险不该有
     因果效应。delta 应当 ~= 0（CI 含 0，且量级远小于 APOE）。
     失败（null 的 delta 和 APOE 的 delta 同量级）= **整个引擎不可用**。它说明模型把
     "输入变了"本身当成了信号 —— 换任何一个 token 都会晃动输出，于是"扰动 X 使风险升高
     0.03"这句话里的 X 是谁根本不重要。这种失败在判别力指标上完全看不见。

  3. 方向已知的干预：往基线访视里注入 `ANTIHYP_ON`（开始吃降压药），
     **并配一个注入对照** `ANTIHYP_OFF`（id 43，紧挨着 ON 的 42，落在同一个插入位置、
     同样只多一个 token，临床方向相反；本 split 无人基线已带 OFF，所以逐人配对）。
     第 2 条的零扰动是**原地替换**，长度和位置都不变，管不到"多插了一个 token 本身就让
     输出动"这条通路 —— 在打开 wpe 的 checkpoint 上尤其要命。只有 Δ_ON - Δ_OFF 才把它减掉。
     它应当主要压低 **SBP_/DBP_** 的 5 年分箱（靶点），而**不应当**显著改变
     `BRAAK_`/`CERAD_` 这类死后病理（脱靶），也不该大幅改 AD。报靶点/脱靶比值。
     失败（比值 ~ 1）= 扰动的影响在 token 空间里是弥散的，引擎没有学到"哪些变量受这个
     干预支配"，反事实轨迹里除靶点外的一切都是噪声。

--------------------------------------------------------------------------------------
三个必须写下来的口径决定（改错了不会报错，只会给出好看的假数）

A. **各臂用互相独立的 seed，故意不用 common random numbers。**
   CRN（扰动臂和参照臂共用随机数）能把 MC 噪声消掉一大半，是反事实模拟的标准做法。
   这里**不用**，因为一旦用了，第 0 条的噪声地板就不再是 delta 的零假设分布（CRN 下同一
   前缀跑两次 delta 恒等于 0），而"和噪声地板比"是这套评估唯一的定量锚。代价是每个 delta
   都带着 sqrt(2/n_mc) 量级的噪声，得靠按人平均和 bootstrap 压下去。

B. **`× 噪声地板` 这个数依赖 n_mc，`mean Δ [CI]` 不依赖。**
   `mean_i|Δ_i| / mean_i|Δ_noise,i|` 是逐人幅度比，n_mc 越大分母越小、这个比值越大，
   所以**引用它必须带上 n_mc**。真正的推断统计量是 `mean Δ` 加按人 bootstrap 的 CI：
   它对 MC 噪声是无偏的（噪声均值为 0），n_mc 只影响它的宽度。两个都报，别只看比值。

C. **死后病理（BRAAK_/CERAD_）的脱靶效应只能从 logits 读，不能从 rollout 读。**
   vocab.py 把 34 个病理 token 从采样里封掉了（README 3.5：Death 终止 rollout，采样出来的
   病理 token 只可能落在训练数据里不存在的位置）。所以在轨迹里它们的 5 年发生率**恒等于 0**，
   两臂相减恒等于 0 —— 一个永远"通过"的假测试。这里改成读**单步速率比**
   `exp(logit_扰动 - logit_参照)`（probe_perturb.py 的口径，模型的 logit 就是 log 速率），
   它是确定性的、**没有 MC 噪声**，所以那一列不配噪声地板，理想值恰好 1.000。

--------------------------------------------------------------------------------------
其它跟着已有代码走的地方

  * 前缀 = `figure2_core._baseline()` 定义的那个（statics + 整个基线访视），也就是 rollout
    的起点，和 lr_endpoint_baseline.py / direct_head.py 的特征**一个 token 不多不少**。
  * 词表绑定复用 `lr_endpoint_baseline.bind()`（S.configure + F2 的别名刷新），不重复实现。
  * 所有 CI 按**人** bootstrap（一行一人），不是按观测、更不是按 MC 轨迹 —— 按轨迹会把
    n 虚报 n_mc 倍，CI 窄到毫无意义。
  * APOE 的 `(dose=2, e2=1)` 是 meta.json `apoe_encoding` 明写的非法组合（ε4/ε4 不可能同时
    带 ε2）。扰到 2 拷贝时必须删掉 `APOE_e2carrier`，否则生成的是一个基因型上不存在的病人。
    但那一删会让 2 拷贝臂比 0/1 拷贝臂**多改一个 token**，剂量反应就不再是单轴操作 ——
    所以**主分析限定在非 ε2 携带者**（val 416 人里 374 人），ε2 携带者单列并标注混杂。
  * 注入类扰动（`ANTIHYP_ON`）必须插在**基线访视内按 token id 升序该在的位置**，不能 append
    到末尾。probe_perturb.py 的注释记了这个坑：append 会把打分位置从"MMSE 之后"变成
    "新 token 之后"，四个 add 扰动全部给出反向结果，那是位置假象不是效应。本 val split 的
    416 个基线访视 token id **全部严格升序**（实测 0 违规），所以升序插入是无歧义的。

--------------------------------------------------------------------------------------
这个文件**不改任何已有文件**，也不新增任何默认打开的行为：它是一个独立入口，
不跑它就完全不影响现有的 .bin / ckpt / figure2 缓存。
"""
import os
import sys
import json
import time
import argparse
import logging

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "figure2"))
import torch                                                   # noqa: E402
from radc_delphi import engine as EN, batching as Bt           # noqa: E402
import figure2.figure2_core as F2                              # noqa: E402
from lr_endpoint_baseline import bind                          # noqa: E402

log = logging.getLogger("perturb")
D = EN.DAYS_PER_YEAR
NEG = -1e4 + 1                        # 轨迹 padding 的 age 是 -10000，任何读 age 的地方都要排除

HORIZONS = (1, 3, 5, 10)
PRIMARY_H = 5                         # 全部"靶点/脱靶/剂量反应"都在 5 年上报

# 5 年分箱 readout 里，谁算靶点、谁算脱靶。ANTIHYP_ON 是降压药：
#   靶点 = 血压两个家族；
#   脱靶 = 其余**同样是序数分箱、同样可采样**的家族 —— 选它们而不是选 AD/Death，是为了让
#          靶点和脱靶是**同一个单位**（bin 数），比值才有意义。AD/Death 是概率，单列。
# 注意 PSQI/APNEA/CRP/IL6/TNFA 这些家族在很多人身上基线就没记录（readout 记 NaN 被排除），
# 所以它们进不进脱靶集合对结果影响很小，但留着能暴露"扰动影响弥散"的情况。
TARGET_FAMS = ("SBP", "DBP")
OFFTARGET_FAMS = ("GLU", "HBA1C", "HDL", "LDL", "BMI", "GFR", "MMSE", "COG")

# 读单步速率的 token 组。名字 -> 该组内所有 token 的速率之和（速率可加：指数竞赛里
# "这一组里任意一个先发生"的速率就是各自速率之和）。
RATE_GROUPS = {
    "AD_DX": ("AD_DX",),
    "Death": ("Death",),
    "BRAAK": ("BRAAK_",),            # 前缀形式，见 _rate_ids
    "CERAD": ("CERAD_",),
    "PATHOLOGY": ("__PATH__",),      # 特殊：整个 res.PATHOLOGY_IDS
    "SBP_high": ("SBP_high",),
    "SBP_normal": ("SBP_normal",),
    "DBP_high": ("DBP_high",),
}

# 臂的顺序就是打印顺序。`ref` 必须是第一个：所有 delta 都是"某臂 - ref"。
ARMS = ("ref", "ref2", "apoe0", "apoe1", "apoe2", "null_study", "antihyp", "antihyp_off")
ARM_DESC = {
    "ref":        "参照（未扰动）",
    "ref2":       "未扰动·第二个 seed（= 噪声地板）",
    "apoe0":      "APOE ε4 = 0 拷贝",
    "apoe1":      "APOE ε4 = 1 拷贝",
    "apoe2":      "APOE ε4 = 2 拷贝",
    "null_study": "零扰动：STUDY_ROS <-> STUDY_MAP",
    "antihyp":    "注入 ANTIHYP_ON（开始降压药）",
    "antihyp_off": "注入对照：ANTIHYP_OFF（停降压药，反方向）",
}


# =====================================================================  前缀扰动
# 三条规则，违反任何一条都会静默地把"位置假象"报成"效应"：
#   (1) 替换类扰动**原地改 token id**，位置和年龄都不动；
#   (2) 注入类扰动按 token id 升序插进基线访视；
#   (3) 删除类（只有 ε2 那一处）会改变序列长度，所以主分析把它排除掉。
def _apoe_arm(res, t, a, dose):
    """把前缀的 APOE 剂量轴 token 换成指定拷贝数。返回 (toks, ages, 是否删了 e2) 或 None。

    剂量轴是 {APOE_e4_0, APOE_e4_1, APOE_e4_2, APOE_unk} 四选一，实测 val 里每人**恰好一个**。
    不是恰好一个就返回 None 而不是猜：猜错会造出一个带两个剂量 token 的病人，
    模型照样给分，而那个分数没有任何意义。
    """
    ID = res.ID
    axis = [ID["APOE_e4_0"], ID["APOE_e4_1"], ID["APOE_e4_2"]]
    pool = axis + [ID["APOE_unk"]]
    pos = np.where(np.isin(t, pool))[0]
    if len(pos) != 1:
        return None
    t2, a2 = t.copy(), a.copy()
    t2[pos[0]] = axis[dose]
    dropped = False
    if dose == 2:
        # meta.json apoe_encoding: (2,1) 非法。不删 e2 就是在模拟一个不存在的基因型。
        q = np.where(t2 == ID["APOE_e2carrier"])[0]
        if len(q):
            t2 = np.delete(t2, q); a2 = np.delete(a2, q); dropped = True
    return t2, a2, dropped


def _null_arm(res, t, a):
    """STUDY_ROS <-> STUDY_MAP 原地互换。STUDY_LATC 的人没有对手可换 -> None（被排除）。"""
    ID = res.ID
    ros, mp = ID["STUDY_ROS"], ID["STUDY_MAP"]
    p_ros, p_map = np.where(t == ros)[0], np.where(t == mp)[0]
    t2 = t.copy()
    if len(p_ros) == 1 and len(p_map) == 0:
        t2[p_ros[0]] = mp
    elif len(p_map) == 1 and len(p_ros) == 0:
        t2[p_map[0]] = ros
    else:
        return None
    return t2, a.copy()


def _inject_arm(res, t, a, name, base):
    """把 `name` 注入基线访视，按 token id 升序放在它该在的位置。

    已经有这个 token 的人直接跳过：分词的硬不变量是每个 token 每人最多一次
    （`--nodedup` 只豁免了 MMSE/COGN），塞第二个会造出 .bin 里不存在的流，
    而生成侧的 no-repeat 又会把它封掉，结果是一个既不像数据也不像模型的输入。
    """
    tok = res.ID[name]
    if tok in set(int(x) for x in t):
        return None
    # 这份分词把 statics 放在**基线访视那一天**（实测 val 416/416 人 min(age) == base，
    # 不是 figure2_core 文档里写的 base-1 天）。所以 vis 其实是"整个前缀"，而整个前缀的
    # token id 实测严格升序（416/416 无违规），升序插入因此仍然是无歧义的。
    vis = np.where(a == base)[0]
    if len(vis) == 0:
        return None
    gt = [int(i) for i in vis if int(t[i]) > tok]
    at = gt[0] if gt else int(vis[-1]) + 1
    return np.insert(t, at, tok), np.insert(a, at, float(base))


# =====================================================================  readout
def _first_day(T, A, ids, base):
    """每条轨迹首次发出 `ids` 中任一 token 的绝对天数；inf = 从未。

    `A > NEG` 不能省：padding 位置的 age 是 -10000，不排除的话它会赢下 min。
    """
    hit = np.isin(T, list(ids)) & (A > NEG) & (A > base)
    return np.where(hit, A, np.inf).min(1)


def _levels_at(T, A, fam_ids, at_day, lvlmap):
    """每条轨迹在 `at_day` 时该量表家族的分箱序号（最后一次发射前向填充）；-1 = 从未记录。

    **死后按最后一次观测前向填充**，不是"只在还活着的轨迹里算"。后者会引入各臂不同的
    选择效应：扰动改变死亡率 -> 参与平均的轨迹集合变了 -> 分箱均值的差里混进了生存偏差。
    前向填充没有这个问题，代价是"死人也有一个血压分箱"，而我们只比较两臂之差，那个代价抵消。
    """
    m = np.isin(T, list(fam_ids)) & (A > NEG) & (A <= at_day)
    ev = np.where(m, A, -np.inf)
    best = ev.max(1)
    j = ev.argmax(1)
    got = lvlmap[T[np.arange(T.shape[0]), j]]
    return np.where(best > -np.inf, got, -1)


def _readout(res, T, A, base, base_lvl, lvlmaps):
    """一条臂的全部 rollout 标量。所有臂用**完全相同**的函数，否则 delta 无意义。"""
    out = {}
    d_sim = _first_day(T, A, [res.DEATH], base)
    ad = _first_day(T, A, [res.AD_DX], base)
    for h in HORIZONS:
        hd = base + h * D
        # 与 figure2_core._process 同一口径：事件要在 horizon 内**且早于本条轨迹自己的死亡**
        out[f"ad_{h}y"] = float(np.mean((ad <= hd) & (ad <= d_sim)))
        out[f"death_{h}y"] = float(np.mean(d_sim <= hd))
    hd = base + PRIMARY_H * D
    for fam, ids in res.SCALES.items():
        bl = base_lvl.get(fam, -1)
        if bl < 0:
            # 基线没记录这个量表 -> 不评它。和 figure2 的 `baseline_<scale> == -1` 同一条规则：
            # 在 TV 门控的发射下，量过就一定至少发过一次，所以 -1 真的是"没量过"。
            out[f"lvl_{fam}"] = float("nan")
            out[f"chg_{fam}"] = float("nan")
            continue
        lv = _levels_at(T, A, ids, hd, lvlmaps[fam])
        lv = np.where(lv < 0, bl, lv)
        out[f"lvl_{fam}"] = float(lv.mean())
        out[f"chg_{fam}"] = float(np.mean(lv != bl))
    return out


def _rate_ids(res, spec):
    if spec == ("__PATH__",):
        return list(res.PATHOLOGY_IDS)
    ids = []
    for s in spec:
        if s.endswith("_"):
            ids += [i for i, n in enumerate(res.NAMES) if n.startswith(s)]
        elif s in res.ID:
            ids.append(res.ID[s])
    return ids


@torch.no_grad()
def _log_rates(eng, groups, t, a):
    """前缀末端的 log 速率（每组求和后取 log）。**确定性**，没有 MC 噪声。

    模型的 logit 就是 log 速率（采样是 t_k ~ Exp(exp(logit_k))），所以两臂相减再 exp
    就是一个**速率比**，和流行病学的 hazard ratio 同构。用 softmax 概率差会把效应和
    "别的 token 速率变了多少"混在一起 —— 那正是零扰动对照要能分辨的东西。
    statics 列先置 -inf：它们本来就不是候选，留在分母里只会稀释份额。
    """
    B = eng.block_size
    idx = torch.as_tensor(np.asarray(t[-B:])[None], dtype=torch.long, device=eng.device)
    age = torch.as_tensor(np.asarray(a[-B:])[None], dtype=torch.float32, device=eng.device)
    lg, _, _ = eng.model(idx, age)
    lg = lg[0, -1].double().clone()
    lg[torch.tensor(eng.ignore_tokens, device=lg.device)] = -torch.inf
    r = torch.exp(lg)
    out = {}
    for nm, ids in groups.items():
        s = float(r[torch.tensor(ids, device=r.device)].sum()) if ids else 0.0
        out[nm] = float(np.log(s)) if s > 0 else float("nan")
    return out


# =====================================================================  worker
_G = {}


def _init_worker(ckpt, dataset, split, device):
    """每个进程各自 bind + load。`bind()` 改的是 S/F2 的模块全局，所以必须在**这个进程里**跑过，
    否则 `F2._baseline` 会用上一次绑定（甚至 import 时的 live 表）的 STAGE_TOKENS —— 不报错，
    只是把人按错的量表分期，然后整套数据静默地评在错的前缀上。"""
    torch.set_num_threads(1)
    ddir = os.path.join(HERE, "data", dataset)
    res = bind(os.path.join(ddir, "labels.csv"))
    eng = EN.load(os.path.join(HERE, ckpt), data_dir=ddir, device=device,
                  vocab_labels=os.path.join(ddir, "labels.csv"))
    d = np.fromfile(os.path.join(ddir, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
    lvlmaps = {}
    for fam, ids in eng.res.SCALES.items():
        mp = np.full(eng.res.VOCAB_SIZE, -1, dtype=np.int64)
        mp[list(ids)] = np.arange(len(ids))
        lvlmaps[fam] = mp
    groups = {nm: _rate_ids(eng.res, spec) for nm, spec in RATE_GROUPS.items()}
    _G.update(eng=eng, res=res, data=d, p2i=Bt.get_p2i(d), lvlmaps=lvlmaps, groups=groups)


def _process(job):
    """一个病人的全部臂。返回 None = 这个人不可评估（`_baseline` 的规则，和 figure2 一致）。"""
    k, n_mc, seed, cap_years, arms = job
    eng, data, p2i = _G["eng"], _G["data"], _G["p2i"]
    res = eng.res
    lvlmaps, groups = _G["lvlmaps"], _G["groups"]

    s, n = int(p2i[k, 0]), int(p2i[k, 1])
    ages, toks = Bt.patient_stream(data, s, n)
    bp = F2._baseline(ages, toks)
    if bp is None:
        return None
    base, pmask, _b = bp
    t0 = np.asarray(toks)[pmask].copy()
    a0 = np.asarray(ages, float)[pmask].copy()
    base_y = base / D
    until = base_y + float(cap_years)

    # 基线分箱从**参照前缀**读，所有臂共用。基线状态是这个人的性质，不是臂的性质；
    # 各臂各读一遍只会在"扰动恰好改了某个量表 token"时偷偷改变分母。
    base_lvl = {}
    for fam, ids in res.SCALES.items():
        m = np.isin(t0, list(ids))
        base_lvl[fam] = int(lvlmaps[fam][int(t0[m][np.argmax(a0[m])])]) if m.any() else -1

    st = set(int(x) for x in t0)
    dose_name = next((nm for nm in ("APOE_e4_0", "APOE_e4_1", "APOE_e4_2", "APOE_unk")
                      if res.ID[nm] in st), "none")
    study_name = next((nm for nm in ("STUDY_ROS", "STUDY_MAP", "STUDY_LATC")
                       if res.ID[nm] in st), "none")
    meta = dict(pid=int(k), projid=int(data[s, 0]), baseline_age=float(base_y),
                apoe_dose=dose_name, e2carrier=bool(res.ID["APOE_e2carrier"] in st),
                study=study_name, n_prefix=int(len(t0)))

    # 各臂的前缀
    pre = {"ref": (t0, a0), "ref2": (t0, a0)}
    for dose in (0, 1, 2):
        r = _apoe_arm(res, t0, a0, dose)
        if r is not None:
            pre[f"apoe{dose}"] = (r[0], r[1])
            if dose == 2:
                meta["apoe2_dropped_e2"] = bool(r[2])
    r = _null_arm(res, t0, a0)
    if r is not None:
        pre["null_study"] = r
    r = _inject_arm(res, t0, a0, "ANTIHYP_ON", base)
    if r is not None:
        pre["antihyp"] = r
    # 注入类扰动的对照臂。【2】的零扰动只覆盖**替换**（token id 换一个，长度和位置都不动）；
    # 注入多一个 token、把后面每个位置往后推一格，这两件事本身就能让输出动 —— 在打开 wpe
    # （DelphiConfig.pos_embedding）的 checkpoint 上尤其如此。ANTIHYP_OFF 的 id 43 紧挨着
    # ANTIHYP_ON 的 42，落在**同一个插入位置**、带来**同样的长度变化**，临床含义却相反，
    # 所以 Δ_ON - Δ_OFF 把"插了一个 token"减掉，剩下的才是"插的是哪一个"。
    # **只给 antihyp 臂也成立的人跑**。OFF 在本 split 的基线前缀里一个都没有，所以不加这个
    # 门控的话 OFF 臂会覆盖全部 416 人、ON 臂只有 217 人，两列的 n 不同 -> 又变成拿两批人
    # 相减。门控之后两臂逐人配对，而且省掉一半的 rollout。
    if "antihyp" in pre:
        r = _inject_arm(res, t0, a0, "ANTIHYP_OFF", base)
        if r is not None:
            pre["antihyp_off"] = r

    out = {"meta": meta, "arms": {}, "rates": {}}
    ref_lr = None
    for ai, arm in enumerate(ARMS):
        if arm not in arms or arm not in pre:
            continue
        tt, aa = pre[arm]
        # 各臂 seed 互相独立（见模块 docstring 的口径决定 A）。乘上大质数是为了让
        # "臂"和"病人"两个维度不会撞在一起 —— 撞了就等于偷偷用了 CRN，噪声地板会假性变小。
        sd = int(seed) + ai * 1000003 + k * 97
        T, A = eng.simulate(tt, aa, n_mc=n_mc, seed=sd, until_age_years=until)
        out["arms"][arm] = _readout(res, T, A, base, base_lvl, lvlmaps)
        lr = _log_rates(eng, groups, tt, aa)
        if arm == "ref":
            ref_lr = lr
        if ref_lr is not None:
            out["rates"][arm] = {nm: float(lr[nm] - ref_lr[nm]) for nm in lr}   # log 速率比
    return out


# =====================================================================  统计
def _boot_mean(x, n_boot, rng):
    """(mean, lo, hi)，按**人** bootstrap。x 一行一人。"""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(x.mean())
    if x.size < 2:
        return m, float("nan"), float("nan"), int(x.size)
    bs = x[rng.integers(0, x.size, (n_boot, x.size))].mean(1)
    return m, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), int(x.size)


def _spearman(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = pd.Series(x[ok]).rank().to_numpy()
    ry = pd.Series(y[ok]).rank().to_numpy()
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _delta(rows, arm, key, cohort=None):
    """每人一个 delta = 该臂 readout - ref readout。缺臂的人给 NaN（被 _boot_mean 丢掉）。"""
    out = []
    for i, r in enumerate(rows):
        if cohort is not None and not cohort[i]:
            out.append(np.nan); continue
        a, b = r["arms"].get(arm), r["arms"].get("ref")
        if a is None or b is None or key not in a or key not in b:
            out.append(np.nan); continue
        out.append(float(a[key]) - float(b[key]))
    return np.asarray(out, float)


def _floor(rows, key, cohort=None):
    """噪声地板：同一未扰动前缀、两个独立 seed 的 delta 分布。

    `cohort` 不是可选的装饰：地板必须在**和效应同一批人**上算。antihyp 只能注入到 217 个
    基线还没吃降压药的人身上，这批人系统性更健康、事件更少、MC 方差更小；拿全 416 人的地板
    当分母，`× 地板` 和 `z` 就是两个人群的混合比，不是同一批人的信噪比。
    """
    d = _delta(rows, "ref2", key, cohort)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return dict(n=0, mean_abs=float("nan"), sd=float("nan"), mean=float("nan"))
    return dict(n=int(d.size), mean_abs=float(np.abs(d).mean()), sd=float(d.std(ddof=1))
                if d.size > 1 else float("nan"), mean=float(d.mean()))


def _effect(rows, arm, key, n_boot, rng, cohort=None):
    """一个 (臂, readout) 的完整效应描述，包含与噪声地板的两种对比。

    地板**就地在这个效应实际用到的那批人上重算**（`used`），而不是取全队列那张表。
    臂的可用性本身就是一个隐式的人群筛选（antihyp 217/416、null_study 388/416），
    拿全队列地板当分母会把两个人群的差异算进信噪比里。
    """
    d = _delta(rows, arm, key, cohort)
    used = np.isfinite(d)
    m, lo, hi, n = _boot_mean(d, n_boot, rng)
    dd = d[used]
    fl = _floor(rows, key, used)
    fa, fsd = fl.get("mean_abs", np.nan), fl.get("sd", np.nan)
    mean_abs = float(np.abs(dd).mean()) if dd.size else float("nan")
    # 逐人幅度比。~1.0 = 和"同一个人跑两遍"没区别。**依赖 n_mc**，见 docstring 口径决定 B。
    x_floor = mean_abs / fa if (np.isfinite(fa) and fa > 0) else float("nan")
    # 均值相对噪声标准误的 z。不依赖 n_mc 的那一个（噪声均值为 0）。
    z = (abs(m) / (fsd / np.sqrt(max(n, 1)))
         if (np.isfinite(fsd) and fsd > 0 and n > 0) else float("nan"))
    # 扣掉 MC 方差之后，效应在人和人之间还剩多少异质性。负数截到 0。
    v = float(dd.var(ddof=1)) if dd.size > 1 else np.nan
    sd_true = float(np.sqrt(max(0.0, v - (fsd ** 2 if np.isfinite(fsd) else 0.0)))) \
        if np.isfinite(v) else float("nan")
    return dict(arm=arm, key=key, n=n, mean=m, ci=[lo, hi], mean_abs=mean_abs,
                x_floor=x_floor, z_vs_floor=z, sd_true=sd_true, floor=fl,
                ci_excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi)
                                      and (lo > 0 or hi < 0)))


def _contrast(rows, arm_a, arm_b, key, n_boot, rng):
    """逐人 (臂 A 的 readout - 臂 B 的 readout)，两臂都可用的人才算。

    ref 在相减里自己消掉了，所以这等价于 Δ_A - Δ_B。噪声地板照样在**这批人**上重算，
    但要注意它是单个 delta 的地板，而这个对比里有两条独立 seed 的轨迹，
    所以它的零假设幅度比地板再大 sqrt(2) 倍左右 —— 判 "有没有差别" 看 CI，不看 × 地板。
    """
    v = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        a, b = r["arms"].get(arm_a), r["arms"].get(arm_b)
        if a is not None and b is not None and key in a and key in b:
            v[i] = float(a[key]) - float(b[key])
    m, lo, hi, n = _boot_mean(v, n_boot, rng)
    used = np.isfinite(v)
    fl = _floor(rows, key, used)
    vv = v[used]
    return dict(arm=f"{arm_a}-{arm_b}", key=key, n=n, mean=m, ci=[lo, hi],
                mean_abs=float(np.abs(vv).mean()) if vv.size else float("nan"),
                floor_mean_abs=fl.get("mean_abs", float("nan")),
                ci_excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi)
                                      and (lo > 0 or hi < 0)))


def _rate_effect(rows, arm, key, n_boot, rng, cohort=None):
    """速率比的几何均值 + 按人 bootstrap。存的是 log 比值，所以直接算均值再 exp。

    这一列**没有 MC 噪声**（读的是 logits 不是轨迹），所以它的"地板"恰好是 1.000，
    不配噪声地板列。任何偏离 1.000 都是模型真的这么认为，不是采样波动。
    """
    v = []
    for i, r in enumerate(rows):
        if cohort is not None and not cohort[i]:
            v.append(np.nan); continue
        a = r["rates"].get(arm)
        v.append(float(a[key]) if (a is not None and key in a) else np.nan)
    m, lo, hi, n = _boot_mean(np.asarray(v, float), n_boot, rng)
    ex = (lambda z: float(np.exp(z)) if np.isfinite(z) else float("nan"))
    return dict(arm=arm, key=key, n=n, gm=ex(m), ci=[ex(lo), ex(hi)])


# =====================================================================  报告
def _fmt_ci(e, w=7, p=4):
    return f"{e['mean']:+{w}.{p}f} [{e['ci'][0]:+.{p}f},{e['ci'][1]:+.{p}f}]"


def _dose_block(rows, key, cohort, n_boot, rng):
    """APOE 剂量反应的全部统计，外加"三臂都可用"的人群掩码。

    `ok` 要求**同一个人**三个剂量臂都算出来了，这样 dose0/1/2 的均值是配对的。
    不配对的话（各臂各自取自己那批人的均值）剂量效应会和人群构成混在一起 —— 这份数据里
    ε4 剂量和年龄/教育都相关，混起来足以把单调性翻过来。
    """
    lvls = []
    for d in (0, 1, 2):
        v = np.array([(r["arms"][f"apoe{d}"][key] if f"apoe{d}" in r["arms"] else np.nan)
                      for r in rows], dtype=float)
        lvls.append(np.where(cohort, v, np.nan))
    ok = np.all([np.isfinite(v) for v in lvls], axis=0)
    xs, ys = [], []
    for d in (0, 1, 2):
        xs += [d] * int(ok.sum()); ys += list(lvls[d][ok])
    sp = _spearman(np.asarray(xs, float), np.asarray(ys, float))
    # 单调比例有三个版本，**必须一起看**。`frac_monotone` 用的是非严格 <=，而 ad_5y 在
    # n_mc=100 下是 k/100 且绝大多数人三臂全是 0.00 —— 三个 0 满足 <=，于是它被"完全没反应
    # 的人"顶上去，可以在效应恒等于 0 的模型上读到 0.9+。分母换成"三臂里至少有一个不相等
    # 的人"（frac_monotone_varying）才是对"有反应的人里方向对不对"的回答；frac_strict 是
    # 严格上升的比例，提供另一端的界。
    lo3, mid3, hi3 = lvls[0][ok], lvls[1][ok], lvls[2][ok]
    mono = (float(np.mean((lo3 <= mid3) & (mid3 <= hi3))) if ok.any() else float("nan"))
    varying = (lo3 != mid3) | (mid3 != hi3)
    mono_var = (float(np.mean(((lo3 <= mid3) & (mid3 <= hi3))[varying]))
                if varying.any() else float("nan"))
    strict = (float(np.mean((lo3 < mid3) & (mid3 < hi3))) if ok.any() else float("nan"))
    means = [float(v[ok].mean()) if ok.any() else float("nan") for v in lvls]
    fa = _floor(rows, key, ok).get("mean_abs", np.nan)
    pairs = {}
    for lo_d, hi_d in ((0, 1), (1, 2), (0, 2)):
        dd = np.where(ok, lvls[hi_d] - lvls[lo_d], np.nan)
        m, lo, hi, n = _boot_mean(dd, n_boot, rng)
        ddf = dd[np.isfinite(dd)]
        ma = float(np.abs(ddf).mean()) if ddf.size else float("nan")
        pairs[f"{hi_d}-{lo_d}"] = dict(
            mean=m, ci=[lo, hi], n=n, mean_abs=ma,
            x_floor=(ma / fa if (np.isfinite(ma) and np.isfinite(fa) and fa > 0)
                     else float("nan")),
            ci_excludes_zero=bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)))
    return dict(key=key, n=int(ok.sum()), mean_by_dose=means, spearman=sp,
                frac_monotone=mono, frac_monotone_varying=mono_var,
                n_varying=int(varying.sum()), frac_strict=strict, pairs=pairs), ok


def _report(name, rows, n_mc, n_boot, seed):
    rng = np.random.default_rng(seed)
    rep = {"run": name, "n_subjects": len(rows), "n_mc": n_mc, "n_boot": n_boot}
    keys_prob = [f"ad_{h}y" for h in HORIZONS] + [f"death_{h}y" for h in HORIZONS]
    fams = [f for f in (TARGET_FAMS + OFFTARGET_FAMS)]
    keys_lvl = [f"lvl_{f}" for f in fams]
    keys = keys_prob + keys_lvl + [f"chg_{f}" for f in fams]
    floors = {k: _floor(rows, k) for k in keys}
    rep["floors"] = floors

    e2 = np.array([r["meta"]["e2carrier"] for r in rows], bool)
    no_e2 = ~e2
    counts = {
        "evaluable": len(rows),
        "e2carrier": int(e2.sum()),
        "apoe_dose_baseline": pd.Series([r["meta"]["apoe_dose"] for r in rows])
                                .value_counts().to_dict(),
        "study_baseline": pd.Series([r["meta"]["study"] for r in rows])
                            .value_counts().to_dict(),
        "arm_n": {a: int(sum(1 for r in rows if a in r["arms"])) for a in ARMS},
    }
    rep["counts"] = counts

    W = 108
    print(f"\n{'='*W}\nrun={name}   可评估 {len(rows)} 人   n_mc={n_mc}   "
          f"bootstrap {n_boot} 次（按人）\n{'='*W}")
    print("  基线 APOE 剂量: " + "  ".join(f"{k}={v}" for k, v in
                                           sorted(counts["apoe_dose_baseline"].items()))
          + f"   ε2 携带 {int(e2.sum())}")
    print("  基线队列: " + "  ".join(f"{k}={v}" for k, v in
                                     sorted(counts["study_baseline"].items())))
    print("  各臂可用人数: " + "  ".join(f"{a}={counts['arm_n'][a]}" for a in ARMS))

    # ---------------------------------------------------------------- 0. 噪声地板
    print(f"\n【0】MC 噪声地板 —— 同一个未扰动前缀，两个独立 seed（n_mc={n_mc}）")
    print("  下面每一个 delta 都要和这张表比。`mean|Δ|` 是逐人幅度，`sd` 用来算 z。")
    hdr = f"{'readout':<16s}{'n':>5s}{'mean|Δ|':>11s}{'sd(Δ)':>11s}{'mean Δ':>11s}"
    print(hdr); print("-" * len(hdr))
    for k in keys_prob[:4] + [f"lvl_{f}" for f in TARGET_FAMS] + \
            [f"lvl_{f}" for f in ("MMSE", "COG")]:
        f = floors[k]
        print(f"{k:<16s}{f['n']:>5d}{f['mean_abs']:>11.4f}{f['sd']:>11.4f}{f['mean']:>+11.4f}")
    print(f"  地板随 n_mc 以 1/sqrt(n_mc) 收缩 —— 引用 `× 地板` 必须带 n_mc={n_mc}。")

    # ---------------------------------------------------------------- 1. 剂量反应
    print(f"\n【1】剂量反应：APOE ε4 拷贝数 -> {PRIMARY_H} 年 AD 风险（应单调上升）")
    print("  主分析 = 非 ε2 携带者（(dose=2,e2=1) 非法，删 e2 会让 2 拷贝臂多改一个 token）")
    key = f"ad_{PRIMARY_H}y"
    dose_rep = {}
    dose_ok = {}
    for cname, coh in (("noe2", no_e2), ("all", np.ones(len(rows), bool))):
        blk, ok = _dose_block(rows, key, coh, n_boot, rng)
        dose_rep[cname] = blk
        dose_ok[cname] = ok
        means, pairs = blk["mean_by_dose"], blk["pairs"]
        print(f"\n  cohort={cname}  n={blk['n']}")
        print(f"    平均 {key}:  dose0 {means[0]:.4f}   dose1 {means[1]:.4f}   "
              f"dose2 {means[2]:.4f}")
        print(f"    Spearman(dose, risk) 在 3n 个点上 = {blk['spearman']:+.3f}    "
              f"逐人单调比例 = {blk['frac_monotone']:.3f}"
              f"（三臂有差异的 {blk['n_varying']} 人里 {blk['frac_monotone_varying']:.3f}，"
              f"严格上升 {blk['frac_strict']:.3f}）")
        h2 = f"    {'配对 Δ':<10s}{'n':>5s}{'mean [95% CI]':>28s}{'mean|Δ|':>10s}{'× 地板':>9s}"
        print(h2); print("    " + "-" * (len(h2) - 4))
        for pk in ("1-0", "2-1", "2-0"):
            p = pairs[pk]
            print(f"    {pk:<10s}{p['n']:>5d}"
                  f"{p['mean']:>+14.4f} [{p['ci'][0]:+.4f},{p['ci'][1]:+.4f}]"
                  f"{p['mean_abs']:>10.4f}{p['x_floor']:>9.2f}")
    # 次要 readout 只进 JSON 不打印：剂量反应的判据是 AD，其余是"有没有把整条轨迹都晃动"的旁证
    dose_rep["secondary"] = {
        k: _dose_block(rows, k, no_e2, n_boot, rng)[0]
        for k in (f"ad_{max(HORIZONS)}y", f"death_{PRIMARY_H}y", "lvl_MMSE", "lvl_COG")}
    rep["dose_response"] = dose_rep
    # 自剂量对照：本来就是这个剂量的人，该臂前缀与 ref 逐字节相同，只有 seed 不同 -> 应等于地板
    self_d, self_m = [], np.zeros(len(rows), bool)
    for i, r in enumerate(rows):
        nm = r["meta"]["apoe_dose"]
        if nm.startswith("APOE_e4_") and f"apoe{nm[-1]}" in r["arms"]:
            self_d.append(r["arms"][f"apoe{nm[-1]}"][key] - r["arms"]["ref"][key])
            self_m[i] = True
    if self_d:
        m, lo, hi, n = _boot_mean(np.asarray(self_d, float), n_boot, rng)
        _sfa = _floor(rows, key, self_m)["mean_abs"]
        xf = (float(np.abs(self_d).mean()) / _sfa
              if (np.isfinite(_sfa) and _sfa > 0) else float("nan"))
        rep["dose_self_control"] = dict(n=n, mean=m, ci=[lo, hi], x_floor=xf)
        print(f"\n    自剂量对照（扰到自己本来的剂量，前缀不变、只换 seed）: n={n}  "
              f"Δ={m:+.4f} [{lo:+.4f},{hi:+.4f}]  × 地板 {xf:.2f}  ← 应 ≈ 1.00")

    # ---------------------------------------------------------------- 2. 零扰动对照
    print(f"\n【2】零扰动对照：STUDY_ROS <-> STUDY_MAP（队列标签，不该有因果效应）")
    null_rep = {}
    hdr = (f"  {'readout':<14s}{'n':>5s}{'mean Δ [95% CI]':>30s}{'mean|Δ|':>10s}"
           f"{'× 地板':>9s}{'z':>8s}{'CI含0':>7s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    for k in [f"ad_{PRIMARY_H}y", f"death_{PRIMARY_H}y", "lvl_MMSE", "lvl_COG",
              "lvl_SBP", "chg_MMSE"]:
        e = _effect(rows, "null_study", k, n_boot, rng)
        null_rep[k] = e
        print(f"  {k:<14s}{e['n']:>5d}{_fmt_ci(e):>30s}{e['mean_abs']:>10.4f}"
              f"{e['x_floor']:>9.2f}{e['z_vs_floor']:>8.2f}"
              f"{('是' if not e['ci_excludes_zero'] else '否'):>7s}")
    rep["null"] = null_rep
    # 量级对比：这是第 2 条的判决书。
    # **人群必须对齐，而且是两边一起对齐**：APOE 那一格算在"非 ε2 且三臂齐全"（noe2 里
    # 每个人都有剂量 token，所以是全部 374 人）上，null 那一格少了 STUDY_LATC 的 28 人
    # （没有对手可换 -> `_null_arm` 返回 None -> delta 是 NaN 被丢掉）。只把 null 限制到
    # dose_ok 上是**不够的**：null 掉到 346 人而 APOE 还是 374 人，比值里仍然混着那 28 个人
    # 的人群差异。所以这里取**交集**，两边都在交集上重算。
    k = f"ad_{PRIMARY_H}y"
    nul = null_rep[k]
    has_null = np.array([("null_study" in r["arms"]) for r in rows], bool)
    matched = dose_ok["noe2"] & has_null
    blk_m, _okm = _dose_block(rows, k, matched, n_boot, rng)
    apoe20 = dose_rep["noe2"]["pairs"]["2-0"]      # 判据 dose_2v0_significant 仍用全 noe2
    apoe20_m = blk_m["pairs"]["2-0"]               # 比值用交集人群
    nul_m = _effect(rows, "null_study", k, n_boot, rng, matched)
    rep["null_matched_cohort"] = nul_m
    rep["apoe_matched_cohort"] = blk_m

    def _ratio(a, b):
        return float(abs(a) / abs(b)) if (np.isfinite(a) and np.isfinite(b)
                                          and abs(b) > 1e-12) else float("nan")
    ratio_mean = _ratio(nul_m["mean"], apoe20_m["mean"])
    ratio_abs = _ratio(nul_m["mean_abs"], apoe20_m["mean_abs"])
    rep["null_vs_apoe"] = dict(key=k, cohort="noe2 ∩ 三臂齐全 ∩ null 臂可用",
                               n_null=nul_m["n"], n_apoe=apoe20_m["n"],
                               ratio_of_means=ratio_mean, ratio_of_mean_abs=ratio_abs,
                               null=nul_m, apoe_2_minus_0=apoe20_m)
    print(f"\n  量级对比（{k}，交集人群：null n={nul_m['n']} / APOE n={apoe20_m['n']}"
          f"，非 ε2、三个剂量臂齐全、且 null 臂可用）:")
    print(f"    null  Δ = {nul_m['mean']:+.4f} [{nul_m['ci'][0]:+.4f},{nul_m['ci'][1]:+.4f}]"
          f"   mean|Δ| = {nul_m['mean_abs']:.4f}")
    print(f"    APOE(2-0) Δ = {apoe20_m['mean']:+.4f} "
          f"[{apoe20_m['ci'][0]:+.4f},{apoe20_m['ci'][1]:+.4f}]"
          f"   mean|Δ| = {apoe20_m['mean_abs']:.4f}")
    print(f"    |mean Δ_null| / |mean Δ_APOE(2-0)| = {ratio_mean:.3f}")
    print(f"    mean|Δ_null| / mean|Δ_APOE(2-0)|   = {ratio_abs:.3f}")
    print("  判据：两个比值都 << 1 且 null 的 CI 含 0 -> 通过。若 ~1 -> **整个引擎不可用**，")
    print("        模型把'输入被改过'本身当成了信号，任何反事实结论里的自变量是谁都不重要。")

    # ---------------------------------------------------------------- 3. 方向已知
    print(f"\n【3】方向已知的干预：注入 ANTIHYP_ON（{PRIMARY_H} 年平均分箱；靶点应下降）")
    anti = {}
    hdr = (f"  {'家族':<10s}{'角色':<6s}{'n':>5s}{'mean Δ bin [95% CI]':>32s}"
           f"{'mean|Δ|':>10s}{'× 地板':>9s}{'z':>8s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    tgt_abs, off_abs = [], []
    for fam in TARGET_FAMS + OFFTARGET_FAMS:
        role = "靶点" if fam in TARGET_FAMS else "脱靶"
        e = _effect(rows, "antihyp", f"lvl_{fam}", n_boot, rng)
        anti[f"lvl_{fam}"] = dict(role=role, **e)
        (tgt_abs if fam in TARGET_FAMS else off_abs).append(e["mean_abs"])
        print(f"  {fam:<10s}{role:<6s}{e['n']:>5d}{_fmt_ci(e):>32s}"
              f"{e['mean_abs']:>10.4f}{e['x_floor']:>9.2f}{e['z_vs_floor']:>8.2f}")
    for k in (f"ad_{PRIMARY_H}y", f"death_{PRIMARY_H}y"):
        e = _effect(rows, "antihyp", k, n_boot, rng)
        anti[k] = dict(role="脱靶(概率)", **e)
        print(f"  {k:<10s}{'脱靶':<6s}{e['n']:>5d}{_fmt_ci(e):>32s}"
              f"{e['mean_abs']:>10.4f}{e['x_floor']:>9.2f}{e['z_vs_floor']:>8.2f}")
    t_m = float(np.nanmean(tgt_abs)) if tgt_abs else float("nan")
    o_m = float(np.nanmean(off_abs)) if off_abs else float("nan")
    sel = t_m / o_m if (np.isfinite(o_m) and o_m > 0) else float("nan")
    anti["selectivity"] = dict(target_mean_abs=t_m, offtarget_mean_abs=o_m, ratio=sel)
    print(f"\n  靶点/脱靶 = {t_m:.4f} / {o_m:.4f} = {sel:.2f}   "
          f"（同一单位：bin 数。~1 = 影响弥散，引擎没学到谁受这个干预支配）")

    # -------- 3b. 注入对照：ANTIHYP_OFF。【2】的零扰动管不到注入 --------------------
    # 上面那张表里的每一个 Δ 都混着两件事：(i) 插进去的是 ANTIHYP_ON，(ii) 前缀多了一个
    # token、后面每个位置往后推一格。(ii) 单独就能让输出动，打开 wpe 的 checkpoint 上更明显，
    # 而【2】的 STUDY 互换是**原地替换**，长度和位置都没变，所以它证明不了 (ii) 无害。
    # ANTIHYP_OFF（id 43）紧挨着 ANTIHYP_ON（id 42），落在同一个插入位置、同样只多一个
    # token，临床方向相反；本 split 里没有任何人的基线前缀已经带 ANTIHYP_OFF，所以两个臂是
    # **逐人配对**的。Δ_ON - Δ_OFF 把 (ii) 减掉，剩下的才是 (i)。
    print(f"\n  注入对照（ANTIHYP_OFF，同一插入位、同样 +1 token、方向相反；逐人配对）:")
    hdr = (f"  {'家族':<10s}{'角色':<6s}{'n':>5s}{'Δ_ON':>10s}{'Δ_OFF':>10s}"
           f"{'Δ_ON-Δ_OFF [95% CI]':>32s}{'CI含0':>7s}")
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    anti_off, contr = {}, {}
    # 两臂都可用的人（_process 已经把 OFF 门控成 ON 的子集，这里再显式取一次交集，
    # 免得将来有人放开那个门控之后这三列又落在不同的 n 上）。
    paired = np.array([("antihyp" in r["arms"]) and ("antihyp_off" in r["arms"])
                       for r in rows], bool)
    for fam in TARGET_FAMS + OFFTARGET_FAMS:
        key_f = f"lvl_{fam}"
        eo = _effect(rows, "antihyp_off", key_f, n_boot, rng, paired)
        c = _contrast(rows, "antihyp", "antihyp_off", key_f, n_boot, rng)
        anti_off[key_f] = eo; contr[key_f] = c
        print(f"  {fam:<10s}{('靶点' if fam in TARGET_FAMS else '脱靶'):<6s}{c['n']:>5d}"
              f"{anti[key_f]['mean']:>+10.4f}{eo['mean']:>+10.4f}"
              f"{c['mean']:>+18.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]"
              f"{('是' if not c['ci_excludes_zero'] else '否'):>7s}")
    for kk in (f"ad_{PRIMARY_H}y", f"death_{PRIMARY_H}y"):
        eo = _effect(rows, "antihyp_off", kk, n_boot, rng, paired)
        c = _contrast(rows, "antihyp", "antihyp_off", kk, n_boot, rng)
        anti_off[kk] = eo; contr[kk] = c
        print(f"  {kk:<10s}{'脱靶':<6s}{c['n']:>5d}"
              f"{anti[kk]['mean']:>+10.4f}{eo['mean']:>+10.4f}"
              f"{c['mean']:>+18.4f} [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}]"
              f"{('是' if not c['ci_excludes_zero'] else '否'):>7s}")
    anti["n_paired"] = int(paired.sum())
    anti["off_arm"] = anti_off
    anti["on_minus_off"] = contr
    print("  读法：Δ_ON 和 Δ_OFF 若同号同量级，那一行测到的是\"插了个 token\"，不是药。")
    print("        只有 Δ_ON-Δ_OFF 的 CI 不含 0 且方向对（SBP/DBP 为负），靶点效应才成立。")

    # 病理：只能读速率，见 docstring 口径决定 C
    print(f"\n  死后病理的脱靶效应（**速率比**，不是轨迹 —— 病理 token 被禁止采样，"
          f"在 rollout 里恒为 0）")
    hdr = f"  {'组':<12s}{'n':>5s}{'速率比（几何均值）':>22s}{'95% CI':>22s}"
    print(hdr); print("  " + "-" * (len(hdr) - 2))
    rates = {}
    for g in ("BRAAK", "CERAD", "PATHOLOGY", "AD_DX", "Death", "SBP_high", "SBP_normal",
              "DBP_high"):
        e = _rate_effect(rows, "antihyp", g, n_boot, rng)
        rates[g] = e
        ci = "[%.4f,%.4f]" % (e["ci"][0], e["ci"][1])
        print(f"  {g:<12s}{e['n']:>5d}{e['gm']:>22.4f}{ci:>22s}")
    anti["rates"] = rates
    rep["antihyp"] = anti

    # 顺带把 APOE / null 的速率比也报一遍：确定性、无 MC 噪声，是 rollout 数字的独立佐证
    print(f"\n  对照：各臂在基线前缀上的单步 AD 速率比（确定性，无 MC 噪声，理想的零扰动 = 1.000）")
    rate_arms = {}
    for arm in ("apoe0", "apoe1", "apoe2", "null_study", "antihyp", "antihyp_off"):
        coh = no_e2 if arm.startswith("apoe") else None
        e = _rate_effect(rows, arm, "AD_DX", n_boot, rng, coh)
        rate_arms[arm] = e
        print(f"    {arm:<12s}{ARM_DESC[arm]:<34s}n={e['n']:<5d}"
              f"{e['gm']:>8.4f}  [{e['ci'][0]:.4f},{e['ci'][1]:.4f}]")
    rep["rate_AD_by_arm"] = rate_arms
    # 这一格是免费的（确定性，不用 rollout），但它是注入类扰动最直接的证伪：
    # 若 antihyp 与 antihyp_off 的 AD 速率比几乎相等，那个"降压药把 AD 速率压低 x%"
    # 就只是"前缀多了一个 token"。
    _ron, _rof = rate_arms.get("antihyp", {}), rate_arms.get("antihyp_off", {})
    if np.isfinite(_ron.get("gm", np.nan)) and np.isfinite(_rof.get("gm", np.nan)) \
            and _rof["gm"] > 0:
        _r = float(_ron["gm"] / _rof["gm"])
        rep["rate_AD_on_over_off"] = _r
        print(f"    -> ON / OFF 的 AD 速率比之比 = {_r:.4f}"
              f"（~1.000 = 上面那个 {_ron['gm']:.4f} 是插入假象，不是药效）")

    # ---------------------------------------------------------------- 判决
    print(f"\n【判决】（阈值写死在代码里，改阈值请连同这段一起改）")
    v = {}
    v["dose_monotone"] = bool(dose_rep["noe2"]["mean_by_dose"][0]
                              <= dose_rep["noe2"]["mean_by_dose"][1]
                              <= dose_rep["noe2"]["mean_by_dose"][2])
    v["dose_2v0_significant"] = bool(apoe20["ci_excludes_zero"] and apoe20["mean"] > 0)
    v["null_ci_contains_zero"] = bool(not nul["ci_excludes_zero"])
    v["null_much_smaller_than_apoe"] = bool(np.isfinite(ratio_abs) and ratio_abs < 0.33)
    v["null_near_floor"] = bool(np.isfinite(nul["x_floor"]) and nul["x_floor"] < 1.3)
    v["null_rate_near_one"] = bool(
        np.isfinite(rate_arms["null_study"]["ci"][0])
        and rate_arms["null_study"]["ci"][0] < 1.0 < rate_arms["null_study"]["ci"][1])
    v["antihyp_target_direction"] = bool(
        all(anti[f"lvl_{f}"]["mean"] < 0 for f in TARGET_FAMS))
    v["antihyp_selective"] = bool(np.isfinite(sel) and sel > 1.5)
    # 注入假象的证伪：靶点上 ON 和 OFF 必须**分得开**，且 ON-OFF 的方向是负的。
    # 只要这一条不过，上面所有 antihyp 的数字都只是"前缀多了一个 token"。
    v["antihyp_vs_off_separates"] = bool(all(
        contr[f"lvl_{f}"]["ci_excludes_zero"] and contr[f"lvl_{f}"]["mean"] < 0
        for f in TARGET_FAMS))
    v["pathology_untouched"] = bool(all(
        np.isfinite(rates[g]["ci"][0]) and rates[g]["ci"][0] < 1.0 < rates[g]["ci"][1]
        for g in ("BRAAK", "CERAD")))
    # 臂被 --arms 排除掉时，它的判据是 NaN 驱动的 False。区分"未跑"和"未通过"：
    # 把没跑过的臂印成 False 会让人以为引擎坏了，而真相只是这次没测。
    ran = {a: int(sum(1 for r in rows if a in r["arms"])) for a in ARMS}
    dep = {"dose_monotone": "apoe0", "dose_2v0_significant": "apoe0",
           "null_ci_contains_zero": "null_study", "null_much_smaller_than_apoe": "null_study",
           "null_near_floor": "null_study", "null_rate_near_one": "null_study",
           "antihyp_target_direction": "antihyp", "antihyp_selective": "antihyp",
           "pathology_untouched": "antihyp", "antihyp_vs_off_separates": "antihyp_off"}
    for kk, vv in v.items():
        if ran.get(dep.get(kk, "ref"), 0) == 0:
            print(f"  未跑    {kk}   （--arms 排除了 {dep[kk]}）")
            v[kk] = None
        else:
            print(f"  {'通过' if vv else '未通过'}  {kk}")
    if ran["null_study"] and not (v["null_ci_contains_zero"]
                                  and v["null_much_smaller_than_apoe"]):
        print("  >>> 零扰动对照未过：在修好之前，本文件其余所有数字都不能引用。")
    if ran["antihyp_off"] and v.get("antihyp_vs_off_separates") is False:
        print("  >>> 注入对照未过：ANTIHYP_ON 和 ANTIHYP_OFF 在靶点上分不开，"
              "【3】测到的是插入假象，靶点/脱靶比值不能引用。")
    rep["verdict"] = v
    return rep


# =====================================================================  驱动
def run(name, ckpt, dataset, split, device, n_mc, limit, workers, seed, cap_years,
        arms, n_boot):
    ddir = os.path.join(HERE, "data", dataset)
    res = bind(os.path.join(ddir, "labels.csv"))                 # 主进程也要绑（给 _baseline 用）
    d = np.fromfile(os.path.join(ddir, f"{split}.bin"), dtype=np.uint32).reshape(-1, 3)
    p2i = Bt.get_p2i(d)
    N = len(p2i) if not limit else min(len(p2i), limit)
    payload = [(k, n_mc, seed, cap_years, arms) for k in range(N)]
    log.info("run=%s dataset=%s split=%s N=%d n_mc=%d arms=%s workers=%s",
             name, dataset, split, N, n_mc, ",".join(arms), workers)

    rows, t0 = [], time.time()
    if workers and workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        with ctx.Pool(workers, initializer=_init_worker,
                      initargs=(ckpt, dataset, split, device)) as pool:
            for i, r in enumerate(pool.imap(_process, payload, chunksize=2)):
                if r is not None:
                    rows.append(r)
                if (i + 1) % 25 == 0:
                    el = time.time() - t0
                    log.info("  %d/%d (%d kept) %.0fs, ~%.0fs left",
                             i + 1, N, len(rows), el, el / (i + 1) * (N - i - 1))
    else:
        _init_worker(ckpt, dataset, split, device)
        for i, job in enumerate(payload):
            r = _process(job)
            if r is not None:
                rows.append(r)
            if (i + 1) % 25 == 0:
                log.info("  %d/%d (%d kept) %.0fs", i + 1, N, len(rows), time.time() - t0)
    log.info("rollout 完成 %.1f 分钟（%d 人 × %d 臂 × n_mc=%d）",
             (time.time() - t0) / 60, len(rows), len(arms), n_mc)
    if not rows:
        raise RuntimeError(f"{dataset}/{split} 上没有可评估的人 —— 检查 --limit 和 _baseline 规则")
    rep = _report(name, rows, n_mc, n_boot, seed)
    rep["per_subject_meta"] = [r["meta"] for r in rows]
    rep["config"] = dict(ckpt=ckpt, dataset=dataset, split=split, n_mc=n_mc, limit=limit,
                         seed=seed, cap_years=cap_years, arms=list(arms), n_boot=n_boot)
    return rep


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="扰动引擎评估套件（反事实可信度，不是判别力）")
    ap.add_argument("--runs", nargs="+", required=True, help="name:ckpt:dataset")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n-mc", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="0 = 全部；冒烟用 20")
    ap.add_argument("--workers", type=int, default=0, help="0/1 = 单进程；RIS 上给 CPUS-2")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--cap-years", type=float, default=float(max(HORIZONS)) + 1.0,
                    help="rollout 最多跑到 基线年龄 + 这个值。默认 max(HORIZONS)+1，"
                         "比 figure2 的 105 岁快一倍多，且对 <= max(HORIZONS) 的读数完全等价"
                         "（simulate 只是把超过的位置置 PAD）。要报更长的 horizon 必须同时调大它，"
                         "否则风险会被静默截断成偏低。")
    ap.add_argument("--arms", default="all",
                    help="逗号分隔的臂子集，默认 all。ref / ref2 永远包含"
                         "（参照 + 噪声地板，缺了整套判据会静默变成 NaN）。")
    ap.add_argument("--out", default=None, help="写 JSON 到这里")
    a = ap.parse_args()

    if a.cap_years < max(HORIZONS) + 1e-9:
        ap.error(f"--cap-years {a.cap_years} 不足以覆盖 max(HORIZONS)={max(HORIZONS)}"
                 f"（要求严格大于，留一年余量）；否则 {max(HORIZONS)}y 的风险会被静默"
                 f"截断成偏低值")
    # ref / ref2 强制保留：前者是所有 delta 的参照，后者是噪声地板，缺了任何一个整套判据失效
    # （× 地板 / z 全变 NaN），而那不会报错，只会安静地给出一张没有分母的表。
    arms = tuple(ARMS) if a.arms == "all" else \
        tuple(x for x in ARMS
              if x in set(y.strip() for y in a.arms.split(",")) | {"ref", "ref2"})

    report = {}
    for spec in a.runs:
        parts = spec.split(":")
        if len(parts) != 3:
            ap.error(f"--runs 的格式是 name:ckpt:dataset，收到 {spec!r}")
        nm, ckpt, ds = parts
        report[nm] = run(nm, ckpt, ds, a.split, a.device, a.n_mc, a.limit,
                         a.workers, a.seed, a.cap_years, arms, a.n_boot)

    print("\n怎么读这套数（按重要性）：")
    print("  1. 先看【2】零扰动。null 的 delta 若和 APOE 同量级，后面什么都别看了。")
    print("  2. 再看【0】地板。任何 `× 地板` ~ 1.0 的 delta 都是采样噪声，不是效应。")
    print("  3. 【1】剂量反应给的是'引擎能不能复现已知因果方向'，不是'AD 风险预测准不准'。")
    print("  4. 【3】靶点/脱靶比值 ~ 1 = 干预的影响在 token 空间弥散，反事实轨迹只有靶点可读。")
    print("  * 这套全部测的是**反事实可信度**。判别力见 README 第 8、9 节，两者可以完全脱钩。")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=1, default=str)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
