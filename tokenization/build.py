# -*- coding: utf-8 -*-
"""ROSMAP → Delphi .bin 格式 tokenizer。纯硬分箱，临床切点优先。"""
import sys, os, json, pickle, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec

HERE = os.path.dirname(os.path.abspath(__file__))
# 原始 RADC 表格所在目录。默认是仓库上一层（Mac 上的布局）；RIS 上用 ROSMAP_DATA_ROOT 指过去。
DATA = os.environ.get("ROSMAP_DATA_ROOT", os.path.dirname(HERE))

# ---------------------------------------------------------------- 去重开关
# 交付版对每个纵向连续量施加**两级**去重：(1) run-length —— 只在箱变化时发射；
# (2) 全局 —— 每个 token id 每人最多一次（下面 `seen` 那段，Delphi 的 first-occurrence 语义）。
# 两级合起来让"26 分附近来回跳"的真实轨迹 N-B-N-B 在 .bin 里只剩 N-B。
#
# `--nodedup MMSE,COGN` 把列出的前缀从**两级**去重里都豁免掉：这些变量改成**每次随访发射
# 一个 token**，忠实保留往复。代价有两条，都不是免费的：
#   * 序列变长（block_size 要跟着调，本脚本按 p99 自动算）；
#   * 下游 rollout 的 no-repeat 规则必须同步放开，否则生成侧永远发不出第二个 MMSE token。
#     所以豁免前缀会写进 meta.json 的 `repeatable_token_prefixes`，由
#     ../figure2_eval/radc_delphi/vocab.py 读取 —— 规则跟着数据走，不靠人记。
_ap = argparse.ArgumentParser()
_ap.add_argument("--out", default=os.path.join(HERE, "out"))
_ap.add_argument("--data-root", default=None,
                 help="原始 RADC 表格目录（也可用 ROSMAP_DATA_ROOT 环境变量）")
_ap.add_argument("--dataset", default="rosmap",
                 help="生成的训练配置里的 dataset 名（决定 delphi/data/<name> 与 out_dir）")
_ap.add_argument("--nodedup", default="",
                 help="逗号分隔的 token 前缀，这些变量每次随访都发射且不去重。"
                      "例: MMSE,COGN。`all` = spec.CONT 的全部 15 个家族")
_ap.add_argument("--per-visit-events", action="store_true",
                 help="STROKE / DEPRESSION 改成**每个被置位的访视**都发射，而不是只发首次。"
                      "默认 off —— 打开会改变 token 的语义（从'首次发病'变成'本次访视有记录'），"
                      "而且已交付的 rosmap_fullvisit 是在 off 下建的。见 README 的已知偏差一节。")
_ap.add_argument("--no-global-dedup", action="store_true",
                 help="连第 2 级（全局 first-occurrence）也整体关掉，不只是对 --nodedup 的前缀。"
                      "影响的是 CONT 之外还会往复的东西：用药 ON/OFF 的第二次开关。")
_ap.add_argument("--split", default="0.9,0.1,0",
                 help="人级别划分比例 train,val,test。默认 0.9,0.1,0 = 交付版的两分行为："
                      "**不写 test.bin**，train.bin/val.bin 与 data_rosmap/ 里那两份字节级一致。"
                      "默认必须是两分，因为已交付的 Delphi-ROSMAP/ckpt.pt 和 figure2_eval 的全部"
                      "结果都长在这份数据上；换成三分就是换了一份训练集，两边的数字不能对读。"
                      "例: --split 0.8,0.1,0.1（见 README 的三分 split 一节）")
_A = _ap.parse_args()
OUT = _A.out
DATASET = _A.dataset
_cont_pre = [pre for _, pre, _, _ in spec.CONT]
if _A.nodedup.strip() == "all":
    NODEDUP = tuple(_cont_pre)                 # 15 个纵向量全部每次随访发射
else:
    NODEDUP = tuple(x.strip() for x in _A.nodedup.split(",") if x.strip())
    _bad = [p for p in NODEDUP if p not in set(_cont_pre)]
    if _bad:
        raise SystemExit(f"--nodedup 里的前缀 {_bad} 不在 spec.CONT 中（可选: {sorted(_cont_pre)}）")
NO_GLOBAL_DEDUP = bool(_A.no_global_dedup)
PER_VISIT_EVENTS = bool(_A.per_visit_events)
if PER_VISIT_EVENTS and not NO_GLOBAL_DEDUP:
    raise SystemExit("--per-visit-events 必须配 --no-global-dedup，否则第 2 级去重会把"
                     "重复的 STROKE/DEPRESSION 又删掉，等于什么都没做")
# ---------------------------------------------------------------- 三分 split
# 只解析和校验，真正的切分在文件末尾的"划分 / 写出"段，那里有为什么是 [train|test|val] 的论证。
_sp = [x.strip() for x in _A.split.split(",") if x.strip()]
if len(_sp) != 3:
    raise SystemExit(f"--split 要三个比例 train,val,test（逗号分隔），收到 {_A.split!r}")
try:
    F_TRAIN, F_VAL, F_TEST = (float(x) for x in _sp)
except ValueError:
    raise SystemExit(f"--split 的三个值必须是数字，收到 {_A.split!r}")
if min(F_TRAIN, F_VAL, F_TEST) < 0:
    raise SystemExit(f"--split 的比例不能为负: {_A.split!r}")
if abs(F_TRAIN + F_VAL + F_TEST - 1.0) > 1e-9:
    # 不允许"随便给三个数自动归一化"：0.8,0.1,0.2 多半是手滑，归一化会安静地给出
    # 一份和预期不同的 train/val，而 .bin 里没有任何东西能事后看出来。
    raise SystemExit(f"--split 三个比例之和必须是 1，收到 {_A.split!r} (和 = {F_TRAIN+F_VAL+F_TEST})")
if F_VAL <= 0:
    # delphi/train.py 无条件 memmap val.bin 再 reshape(-1,3)，空 val 会在那里才炸，
    # 而那时训练任务已经排队排了半天。宁可在分词这一步就拒绝。
    raise SystemExit("--split 的 val 比例必须 > 0：delphi/train.py 直接 memmap val.bin")
if F_TRAIN <= 0:
    raise SystemExit("--split 的 train 比例必须 > 0")
if _A.data_root:
    DATA = _A.data_root
for _f in ("cross-sectional-data-gk.xlsx", "longitudinal_data_gk.xlsx", "ROSMAP_clinical.csv"):
    if not os.path.exists(os.path.join(DATA, _f)):
        raise SystemExit(f"在 {DATA} 下找不到 {_f}；用 --data-root 或 ROSMAP_DATA_ROOT 指定")
os.makedirs(OUT, exist_ok=True)
print(f"原始数据 <- {DATA}")
print(f"输出 -> {OUT}   dataset={DATASET}")
print(f"  run-length 去重豁免: {list(NODEDUP) or '无（交付版行为）'}")
print(f"  全局 first-occurrence 去重: {'整体关闭' if NO_GLOBAL_DEDUP else '仅对上面的前缀豁免'}")
print(f"  STROKE/DEPRESSION: {'每个被置位的访视都发射' if PER_VISIT_EVENTS else '只发首次（默认）'}")
print(f"  split(train,val,test) = {F_TRAIN},{F_VAL},{F_TEST}"
      f"{'   两分（交付版行为，不写 test.bin）' if F_TEST <= 0 else '   三分'}")
RNG  = np.random.default_rng(42)
DPY  = 365.25

# ============================ 载入 ============================
cs = pd.read_excel(f"{DATA}/cross-sectional-data-gk.xlsx")
lo = pd.read_excel(f"{DATA}/longitudinal_data_gk.xlsx").sort_values(["projid", "fu_year"])
cl = pd.read_csv(f"{DATA}/ROSMAP_clinical.csv", dtype=str)
cl["pid"] = pd.to_numeric(cl.projid, errors="coerce").astype("Int64")
cs["study"] = cs.study.str.strip(); lo["study"] = lo.study.str.strip()
lo.loc[lo.hba1c > 20, "hba1c"] = np.nan                       # 剔除 505 录入错误
cs = cs.set_index("projid")
race = pd.to_numeric(cl.set_index("pid").race, errors="coerce").to_dict()
span = pd.to_numeric(cl.set_index("pid").spanish, errors="coerce").to_dict()
adx  = pd.to_numeric(cl.set_index("pid").age_death, errors="coerce").to_dict()
cens = cl.set_index("pid").age_death.eq("90+").to_dict()

# ============================ 分箱器 ============================
def make_binner(kind, rule, series=None):
    if kind == "cut":
        edges, labs = rule
        return lambda v: None if pd.isna(v) else (
            labs[min(int(np.searchsorted(edges, v, side="right")) - 1, len(labs) - 1)]
            if v >= edges[0] else labs[0])
    if kind == "map":
        def _m(v):
            if v is None or (not isinstance(v, str) and pd.isna(v)): return None
            if isinstance(v, str): return rule.get(v.strip())
            try:
                f = float(v)
                return rule.get(int(f) if f.is_integer() else f)
            except (TypeError, ValueError):
                return rule.get(v)
        return _m
    if kind == "qcut":
        qs = np.nanquantile(series.astype(float), np.linspace(0, 1, rule + 1))
        qs[0], qs[-1] = -np.inf, np.inf
        labs = [f"q{i+1}" for i in range(rule)]
        return lambda v: None if pd.isna(v) else labs[min(int(np.searchsorted(qs, v, side="right")) - 1, rule - 1)]
    raise ValueError(kind)

BIN, VOCAB = {}, []
def reg(name):
    VOCAB.append(name); return len(VOCAB)          # pre-shift id, 从 1 开始
TOK = {}

# --- 1. 性别（与 Delphi 同位置）---
TOK["FEMALE"] = reg("Female"); TOK["MALE"] = reg("Male")
BG_START = len(VOCAB) + 1

# --- 2. 背景 / 生活方式块 ---
for var, pre, kind, rule in spec.BACKGROUND:
    src = cs[var] if var in cs.columns else (lo[var] if var in lo.columns else None)
    BIN[var] = make_binner(kind, rule, src)
    labs = rule[1] if kind == "cut" else (list(dict.fromkeys(rule.values())) if kind == "map" else [f"q{i+1}" for i in range(rule)])
    for L in labs: TOK[f"{pre}_{L}"] = reg(f"{pre}_{L}")
    # 只为人数足够的缺失类保留 _unk token；smoking/alcohol 的缺失由"不发射"隐式编码
    if var in ("race", "spanish", "apoe_genotype"):
        TOK[f"{pre}_unk"] = reg(f"{pre}_unk")
    if var == "apoe_genotype":
        TOK["APOE_e2carrier"] = reg("APOE_e2carrier")      # ε2 保护轴，与剂量轴正交
for var, pre in spec.PREVALENT:
    TOK[f"{pre}_PREVALENT"] = reg(f"{pre}_PREVALENT")
BG_END = len(VOCAB)                                # 背景块 pre-shift 区间 [BG_START, BG_END]

# --- 3. 事件 ---
for e in spec.EVENTS: TOK[e] = reg(e)
for var, pre in spec.ONSET: TOK[f"{pre}_ONSET"] = reg(f"{pre}_ONSET")
for var, pre in spec.MEDS:
    TOK[f"{pre}_ON"] = reg(f"{pre}_ON"); TOK[f"{pre}_OFF"] = reg(f"{pre}_OFF")

# --- 4. 纵向连续量 ---
for var, pre, kind, rule in spec.CONT:
    BIN[var] = make_binner(kind, rule, lo[var])
    labs = rule[1] if kind == "cut" else (list(dict.fromkeys(rule.values())) if kind == "map" else [f"q{i+1}" for i in range(rule)])
    for L in labs: TOK[f"{pre}_{L}"] = reg(f"{pre}_{L}")

# --- 5. 死亡（Delphi 把 Death 放最后附近）---
TOK["DEATH"] = reg("Death")

# --- 6. 死后病理 ---
PATH_START = len(VOCAB) + 1
for var, pre, kind, rule in spec.PATHOLOGY:
    BIN[var] = make_binner(kind, rule, cs[var])
    labs = rule[1] if kind == "cut" else (list(dict.fromkeys(rule.values())) if kind == "map" else [f"q{i+1}" for i in range(rule)])
    for L in labs: TOK[f"{pre}_{L}"] = reg(f"{pre}_{L}")
PATH_END = len(VOCAB)

# 豁免去重的 token（pre-shift id）。按 "<前缀>_" 匹配名字，不按 id 段，因为词表里
# STUDY_ROS / HTN_ONSET 这类名字同样含下划线，按段划会顺手带上别的家族。
REPEATABLE_PREFIXES = [p + "_" for p in NODEDUP]
REPEATABLE_IDS = {i for n, i in TOK.items() if n.startswith(tuple(REPEATABLE_PREFIXES))} \
    if REPEATABLE_PREFIXES else set()
if PER_VISIT_EVENTS:
    REPEATABLE_IDS |= {TOK["STROKE"], TOK["DEPRESSION"]}

print(f"词表: {len(VOCAB)} 个 token (pre-shift 1..{len(VOCAB)})")
print(f"  背景块 [{BG_START}..{BG_END}]  病理块 [{PATH_START}..{PATH_END}]  Death={TOK['DEATH']}")

# ============================ 死亡时间：整数年网格 ============================
lastfu = lo.groupby("projid").fu_year.max()
_alv   = cs.age_bl + lastfu                                   # 末次访视年龄
_AGEB = [0, 80, 85, 88, 200]                                  # 末次访视年龄分层
_ALAB = ["lt80", "80-85", "85-88", "ge88"]
_gapd = {L: {} for L in _ALAB}
def _band(a): return _ALAB[max(0, int(np.searchsorted(_AGEB, a, side="right")) - 2 + 1)]
for pid in cs.index[cs.died == 1]:
    ex = adx.get(pid)
    if ex is not None and not pd.isna(ex):
        g = int(np.ceil(max(ex - _alv[pid], 0.001)))
        b = _band(float(_alv[pid])); _gapd[b][g] = _gapd[b].get(g, 0) + 1
_GK, _GP = {}, {}
for L in _ALAB:
    k = np.array(sorted(_gapd[L])); pv = np.array([_gapd[L][x] for x in k], float)
    _GK[L], _GP[L] = k, pv / pv.sum()
    print(f"  {L:>6s} n={int(pv.sum()):4d}  均值gap {float((k*pv).sum()/pv.sum()):.2f}y  "
          f"分布 {dict(zip(k.tolist()[:4], np.round(pv/pv.sum(),3).tolist()[:4]))}")
MAX_AGE = 110.0                                               # 生理上限

def death_age(pid):
    """返回死亡年龄(年)。整数年网格：末次访视 + 整数年。"""
    alv = float(_alv[pid]); ex = adx.get(pid)
    if ex is not None and not pd.isna(ex):                    # ① 有精确值
        return alv + max(int(np.ceil(max(ex - alv, 0.001))), 1)
    L = _band(alv)                                            # ②③ 按年龄分层抽样
    k = int(RNG.choice(_GK[L], p=_GP[L]))
    a = alv + k
    if cens.get(pid, False):                                  # '90+' 硬下界
        while a < 90: k += 1; a = alv + k
    return min(a, max(alv + 1, MAX_AGE))                      # 生理上限兜底

# ============================ 逐人生成序列 ============================
GRID = 365                                                     # 强制 365 天网格
def D(age_years): return int(round(age_years * DPY))           # 年 → 天（仅用于基线）
def G(ab, k):                                                  # 基线真实天数 + k 个整年×365
    return int(round(ab * DPY)) + int(round(k)) * GRID

rows, stats = [], {"dedup": 0, "bg": 0, "event": 0, "med": 0, "cont": 0,
                   "cont_nodedup": 0, "death": 0, "path": 0}
lo_g = {p: g for p, g in lo.groupby("projid")}
for pid in cs.index:
    r = cs.loc[pid]; g = lo_g.get(pid)
    if g is None: continue
    ab = float(r.age_bl); ev = []
    def add(tok, k, cat):                                      # k = 距基线的整年数
        ev.append((G(ab, k), TOK[tok])); stats[cat] += 1

    # --- 背景（全部放在 age_bl；训练时会被 lifestyle_augmentations 随机偏移）---
    add("MALE" if r.msex == 1 else "FEMALE", 0, "bg")
    for var, pre, kind, rule in spec.BACKGROUND:
        v = {"race": race.get(pid), "spanish": span.get(pid)}.get(var, r.get(var))
        lab = BIN[var](v) if v is not None else None
        key = f"{pre}_{lab}" if lab is not None else f"{pre}_unk"
        if key in TOK: add(key, 0, "bg")
        if var == "apoe_genotype" and pd.notna(v) and int(v) in spec.E2_CARRIER:
            add("APOE_e2carrier", 0, "bg")                 # 额外发射，不替代剂量 token
    b0 = g[g.fu_year == 0]
    for var, pre in spec.PREVALENT:
        if len(b0) and b0[var].iloc[0] == 1: add(f"{pre}_PREVALENT", 0, "bg")

    # --- 事件 ---
    ad_k = None
    if pd.notna(r.age_first_ad_dx):
        ad_k = max(int(round(float(r.age_first_ad_dx) - ab)), 0)
    # r_stroke / r_depres 是**逐访视**标志（411 / 474 人在多次访视上被置位，最多 11 / 18 次），
    # 不是像 *_cum 那样的累积量。默认仍只发首次，保持 token 语义是"首次发病"；
    # --per-visit-events 才逐访视发射。这是"全部不去重"唯一没有被两个 dedup 开关覆盖到的地方，
    # 因为限制写在**发射逻辑**里而不在去重那一步。
    for col, tok in [("r_stroke", "STROKE"), ("r_depres", "DEPRESSION")]:
        h = g[g[col].isin([1, 2, 3])]
        if not len(h):
            continue
        if PER_VISIT_EVENTS:
            for y in h.fu_year.tolist():
                add(tok, float(y), "event")
        else:
            add(tok, float(h.fu_year.iloc[0]), "event")
    for var, pre in spec.ONSET:
        s = g[["fu_year", var]].dropna()
        if len(s) and s[var].iloc[0] == 0:
            j = s[s[var] == 1]
            if len(j): add(f"{pre}_ONSET", float(j.fu_year.iloc[0]), "event")

    # --- 用药：只在"开/关"发生时发射（首次为1也算开）---
    for var, pre in spec.MEDS:
        s = g[["fu_year", var]].dropna(); prev = 0
        for _, x in s.iterrows():
            v = int(x[var])
            if v != prev: add(f"{pre}_{'ON' if v else 'OFF'}", float(x.fu_year), "med")
            prev = v

    # --- 连续量：硬分箱 + 去重（箱变化才发射，首次必发）---
    # --nodedup 列出的前缀走另一条分支：**每次随访都发射**，箱没变也发。
    for var, pre, kind, rule in spec.CONT:
        s = g[["fu_year", var]].dropna(); prev = None
        nd = pre in NODEDUP
        for _, x in s.iterrows():
            lab = BIN[var](x[var])
            if lab is None:
                continue
            if nd or lab != prev:
                add(f"{pre}_{lab}", float(x.fu_year), "cont_nodedup" if nd else "cont")
            prev = lab

    # --- 死亡 + 死后病理 ---
    death_k = None
    if r.died == 1:
        death_k = max(int(round(death_age(pid) - ab)), 1)
    if ad_k is not None:                                       # AD_DX 吸附后再发射
        if death_k is not None: ad_k = min(ad_k, death_k)
        add("AD_DX", ad_k, "event")
    if death_k is not None:
        add("DEATH", death_k, "death")
        for var, pre, kind, rule in spec.PATHOLOGY:
            lab = BIN[var](r.get(var))
            if lab is not None: add(f"{pre}_{lab}", death_k, "path")        # 与 DEATH 同一天

    ev.sort(key=lambda t: t[0])
    seen = set()                                               # Delphi 语义: first occurrence only
    for age_d, tid in ev:
        if not NO_GLOBAL_DEDUP and tid not in REPEATABLE_IDS:  # 豁免的前缀跳过全局去重
            if tid in seen: stats["dedup"] += 1; continue
            seen.add(tid)
        rows.append((int(pid), age_d, tid))

arr = np.array(rows, dtype=np.uint32)
print(f"\n总 token 行数 {len(arr):,}  涉及 {len(np.unique(arr[:,0])):,} 人")
print("构成:", {k: f"{v:,}" for k, v in stats.items()})

# ============================ 划分 / 写出 ============================
# **人级别**划分：同一个人的所有行只进一份，否则 test 上的"泛化"其实是在背这个人的前半段。
#
# RNG 只 shuffle 一次，三份都是这同一个排列上的连续段。加 test **不能多抽一次随机数**：
# RNG 是全局共享的（上面 death_age 用的是同一个对象），多一次调用就是另一份数据集，而两份
# 数据集的 .bin 看起来一模一样，只有 md5 会不同 —— 典型的静默漂移。
#
# 段的顺序刻意定成 [train | test | val]，不是直观的 [train | val | test]：
#   * val 的起点只由 F_VAL 决定：va_start = int((1 - F_VAL) * N)。只要 F_VAL 不变，
#     0.9/0.1/0 和 0.8/0.1/0.1 切出来的 val 是**同一批人、同样的行、同样的字节**；
#   * train 永远从头切，所以 0.8 的 train 是 0.9 那份 train 的**前缀子集**；
#   * test 吃掉中间那一段 —— 它是从原 train 里划出来的，和 val 无关。
# 这条性质是三分版能和已有结果对读的全部依据：val 没换人，两个数据集上报的 val 指标直接可比，
# train 的差异只能归到"少了 F_TEST 那批人"，而不是"换了一份数据"。
# 反过来，如果把 test 切在末尾（[train|val|test]），val 会整体平移到另一批人身上，现有的每一
# 张图、每一个 AUC 都失去参照物，**而且不会有任何报错**。所以下面有硬断言把这条钉住。
pids = np.unique(arr[:, 0]); RNG.shuffle(pids)
_N = len(pids)
ntr = int(F_TRAIN * _N)                       # 默认 F_TRAIN=0.9 时与两分版的 int(0.9*len(pids)) 逐字相同
va_start = int((1.0 - F_VAL) * _N)            # val 恒为末尾段；两分版这里也是 int(0.9*N)
if F_TEST <= 0:
    # 两分请求：val 之前的人**全部**归 train。int(F_TRAIN*N) 和 int((1-F_VAL)*N) 在数学上
    # 相等（sum==1 已校验），但浮点取整可以差 1：那 1 个人会落进一个 F_TEST=0 却非空的
    # "test" 段，被写进一份 meta 里 split_fractions.test=0.0 的 test.bin，同时从 train 里
    # 消失 —— 两头都是静默的。默认 0.9,0.1,0 时这一行是 no-op（两边都是 3985，md5 已验）。
    ntr = va_start
if ntr > va_start:
    raise SystemExit(f"train({ntr}) 越过了 val 起点({va_start})，--split 有问题")
tr_ids, te_ids, va_ids = pids[:ntr], pids[ntr:va_start], pids[va_start:]
if F_TEST > 0 and len(te_ids) == 0:
    # 要了 test 却因为取整拿到空段。绝不能静默退回两分：下面会把 test.bin 删掉、日志还会
    # 印"test 比例为 0"，于是一份**以为是三分**的数据集被当成三分用下去。
    raise SystemExit(f"--split 要了 test={F_TEST} 但取整后 test 段是空的"
                     f"（int(f_tr*N)={ntr} == va_start={va_start}, N={_N}）——"
                     "把 test 比例调大，或确认人数")
if len(tr_ids) == 0:
    # F_TRAIN > 0 只约束比例，不约束 int(F_TRAIN*N)。空 train.bin 要到 delphi/train.py
    # 的 get_batch 里才炸，而那时任务已经排过队了。
    raise SystemExit(f"--split 的 train 段取整后是空的（int({F_TRAIN}*{_N})=0）")

tr_set = set(tr_ids.tolist())
mask = np.isin(arr[:, 0], list(tr_set))
va_mask = np.isin(arr[:, 0], list(set(va_ids.tolist())))
# test 取**补集**而不是 isin(te_ids)：补集保证没有任何一行被悄悄丢掉（三份行数之和必须等于总行数）。
# 下一行再断言补集确实等于 te_ids 的 isin，两边对不上说明 pids 不是 arr 的全部人。
te_mask = ~(mask | va_mask)
assert np.array_equal(te_mask, np.isin(arr[:, 0], list(set(te_ids.tolist())))), \
    "test 的补集与 te_ids 不一致 —— 三段没有覆盖 arr 里的全部 projid"
train, val, test = arr[mask], arr[va_mask], arr[te_mask]
assert len(train) + len(val) + len(test) == len(arr), "三份行数之和 != 总行数，有行被丢了"
# 人级别无交集（不是行级别）。行级别不交叉是它的推论，反过来不成立。
_str, _sva, _ste = tr_set, set(va_ids.tolist()), set(te_ids.tolist())
assert not (_str & _sva) and not (_str & _ste) and not (_sva & _ste), "三份 projid 有交集"
assert len(_str | _sva | _ste) == _N, "三份 projid 并集 != 全部人"

# 与两分基线的可对读性。这不是装饰：它是能在**生成时**抓住"段顺序被改回 [train|val|test]"、
# "F_VAL 被顺手改了"、"RNG 被多抽了一次"的唯一一组检查 —— 这三种错误产出的 .bin 都完全合法，
# 只是和现有结果不可比。下面三个量各管一段，口径不同，见各自的注释。
_legacy = int(0.9 * _N)
# 注意这两个量的**限界**：_sva 和 pids[_legacy:] 取自同一次运行的同一个排列，所以它们等价于
# "va_start == int(0.9*N)"，_is_prefix 等价于 "ntr <= int(0.9*N)"（tr_ids 按构造就是
# pids[:ntr]，和自己比永远相等）。也就是说它们只能抓住**切点/段顺序**被改，抓不住
# **排列本身**变了 —— 在 RNG.shuffle 之前多抽一次随机数就会这样，而那恰好是最贵的静默漂移：
# 两份 .bin 肉眼一样、这两个布尔量还都是 True。所以下面再和**落盘的基线 val.bin** 比一次人。
_same_val = (_sva == set(pids[_legacy:].tolist()))
_is_prefix = (ntr <= _legacy)
# 真正能抓住排列漂移的检查：和一份已经存在的两分 val.bin 逐人对比。找不到基线就如实记 None，
# 不伪造 True。projid 的划分与去重开关无关（同一个排列），所以 data_rosmap/val.bin 对
# nodedup / fullvisit / 三分 各版都是合法基线。
_ref_val_bin = os.environ.get("ROSMAP_BASELINE_VAL_BIN",
                              os.path.join(HERE, "data_rosmap", "val.bin"))
_ref_match = None
# 注意：这里读的是**写出之前**的那份文件，所以 --out data_rosmap 原地重建时比的是上一轮的
# val.bin（也就是交付版），这正是最需要被检查的一次。
if os.path.exists(_ref_val_bin):
    try:
        _ref_rows = np.fromfile(_ref_val_bin, dtype=np.uint32).reshape(-1, 3)
    except ValueError:                       # 指错文件了，别用 numpy 的 traceback 吓人
        raise SystemExit(f"基线 {_ref_val_bin} 不是 (projid, age, token) 的 uint32 .bin")
    _ref_pids = set(_ref_rows[:, 0].tolist())
    _ref_match = (_ref_pids == _sva)
    print(f"  基线 val.bin <- {_ref_val_bin}  ({len(_ref_pids)} 人)  与本次 val 同一批人 = {_ref_match}")
    # 人数相同却不是同一批人 = 语料没变而排列/切点变了 = 静默漂移，当场停。
    # （人数不同多半是原始表更新了，那时 val 换人是不可避免的，只警告。）
    if not _ref_match:
        if len(_ref_pids) == len(_sva) and abs(F_VAL - 0.1) <= 1e-12:
            raise SystemExit(
                f"本次 val 的 {len(_sva)} 人与基线 {_ref_val_bin} 的 {len(_ref_pids)} 人"
                "数量相同但不是同一批 —— pids 的排列或切点变了（多半是 RNG 被多抽了一次）。"
                "这份 .bin 完全合法，但和现有的每一个 val 指标都不可比。"
                "确认是故意的话用 ROSMAP_BASELINE_VAL_BIN 指到新基线，或把老基线挪走。")
        print("  警告: val 与基线不是同一批人（人数也不同）——"
              "原始表大概更新过，本次结果不能和现有 val 指标并列引用")
else:
    print(f"  基线 val.bin 未找到（{_ref_val_bin}）—— 跳过'与基线同一批人'的实测检查")
# 上面 _same_val/_is_prefix 只是**记录**。硬断言是这一条 —— f_val 还是基线的 0.1 时，
# val 必须还是 pids[int(0.9N):] 那段。段顺序一旦被改回 [train|val|test]，这里当场炸。
# （f_val 被故意调开时这条不适用，那时 val 换人是**要求**，由上面那行警告如实说明。）
assert _same_val or abs(F_VAL - 0.1) > 1e-12, (
    "f_val 仍是 0.1，但 val 已经不是两分基线的那 int(0.1*N) 个人了 —— 段顺序被改了"
    "（必须是 [train | test | val]）。现有的每一个 val 指标都会失去参照物。")

# .bin 要求同一病人的行连续
train = train[np.argsort(train[:, 0], kind="stable")]
val   = val[np.argsort(val[:, 0], kind="stable")]
test  = test[np.argsort(test[:, 0], kind="stable")]
train.astype(np.uint32).tofile(f"{OUT}/train.bin")
val.astype(np.uint32).tofile(f"{OUT}/val.bin")
_test_path = f"{OUT}/test.bin"
if len(te_ids):
    test.astype(np.uint32).tofile(_test_path)
elif os.path.exists(_test_path):
    # 两分模式跑进一个跑过三分的目录：留着上一轮的 test.bin 就是静默出错的种子 —— 下游会把一份
    # 和本次 train **有交集**的旧 test 当 held-out 用，而文件名对、格式对、没人会去查时间戳。
    os.remove(_test_path)
    print(f"  已删除上一轮残留的 {_test_path}（本次是两分 split，不该有 test.bin）")
pd.DataFrame({"event_name": ["Padding", "No event"] + VOCAB}).to_csv(f"{OUT}/labels.csv", index=False)

# 实测哪些 token 真的在某个人身上出现过多次（post-shift id）。这是下游 no-repeat 规则的
# 唯一权威来源：声明的前缀可能漏（--no-global-dedup 让用药也重复），也可能多（某个前缀
# 在数据里恰好从未重复）。
_rep_cnt = pd.DataFrame(arr, columns=["p", "a", "t"]).groupby(["p", "t"]).size()
MEASURED_REPEATABLE = sorted({int(t) for _p, t in _rep_cnt[_rep_cnt > 1].index})
MEASURED_REPEATABLE_POSTSHIFT = [t + 1 for t in MEASURED_REPEATABLE]
print(f"\n实测可重复 token: {len(MEASURED_REPEATABLE)} 个 -> "
      f"{[VOCAB[t-1] for t in MEASURED_REPEATABLE]}")

L = pd.Series(arr[:, 0]).value_counts()
ignore = [0, 1 + TOK["FEMALE"], 1 + TOK["MALE"]] + list(range(1 + BG_START, 1 + BG_END + 1))
_suf = DATASET.replace("rosmap", "").strip("_")
_OUTDIR = "Delphi-ROSMAP" + (f"-{_suf}" if _suf else "")
cfg = f'''import time
# ROSMAP / Delphi-2M 配置（由 rosmap_tok/build.py 自动生成）
out_dir = '{_OUTDIR}'
eval_interval = 250
eval_iters = 25
log_interval = 25
seed = 42
always_save_checkpoint = False
wandb_log = False
wandb_project = 'delphi-rosmap'
wandb_run_name = 'rosmap' + str(time.time())

dataset = '{DATASET}'
batch_size = 128
block_size = {int(L.quantile(0.99)) + 8}          # 覆盖 p99 序列长度
data_fraction = 1.0

# 语料仅 {len(arr):,} token / {len(pids):,} 人，远小于 UKB(40万人)，模型必须缩小以防过拟合
n_layer = 6
n_head = 6
n_embd = 96
dropout = 0.1
weight_decay = 2e-1
vocab_size = {len(VOCAB) + 2}

learning_rate = 6e-4
max_iters = 20000
lr_decay_iters = 20000
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 500

# padding + 性别 + 背景/生活方式块（post-shift id）
ignore_tokens = {ignore}
# 背景/生活方式块的**位移前** id 区间（delphi/utils.py:get_batch 的 lifestyle_token_range 用它
# 给 statics 的年龄加抖动）。跟着本次分词的 BG 块走。**漏了不会报错**：train.py 的默认值是
# UKB 的 (3, 11)，于是 12..32 这些 ROSMAP 独有的背景 token 不再被抖动 —— 训练照跑，只是和
# 交付版不是同一个配方。
lifestyle_token_min = {BG_START}
lifestyle_token_max = {BG_END}
t_min = 0.1
token_dropout = 0.1        # 稀疏 token 的正则化（见 README 稀疏性一节）
no_event_token_rate = 5

# 这份配置是**按数据自动生成的起点**，不是任何一次已交付训练的复刻。它只知道数据侧的东西
# （dataset / vocab_size / block_size / ignore_tokens / lifestyle 区间），不知道配方侧的旋钮：
# pos_embedding / aux_head / aux_lambda / visit_heads 一概没写，train.py 会用默认值（全 False），
# 而且不会有任何提示。max_iters / token_dropout 也可能和你要对读的那次训练不同。
# **用之前先 diff 一遍 delphi/config/ 里你要复现的那份配方。**
'''
open(f"{OUT}/train_delphi_rosmap.py", "w").write(cfg)

meta = {"vocab": VOCAB, "tok": TOK, "bg_range_preshift": [BG_START, BG_END],
        "path_range_preshift": [PATH_START, PATH_END], "death_token_preshift": TOK["DEATH"],
        "ignore_tokens_postshift": ignore, "vocab_size": len(VOCAB) + 2,
        "excluded": spec.EXCLUDED,
        # 去重豁免。下游（figure2_eval/radc_delphi/vocab.py）据此决定 rollout 的 no-repeat
        # 规则；空 = 每 token 每人最多一次，Delphi 的 first-occurrence 语义。
        "nodedup_prefixes": list(NODEDUP),
        "no_global_dedup": NO_GLOBAL_DEDUP,
        "per_visit_events": PER_VISIT_EVENTS,
        "repeatable_token_prefixes": REPEATABLE_PREFIXES,
        # **实测**，不是声明：真正在这份 .bin 里对某个人出现过 >1 次的 token id。
        # 下游（figure2_eval/radc_delphi/vocab.py）优先读这一项，因为 rollout 的 no-repeat
        # 规则要跟 .bin 的事实一致，而不是跟命令行参数一致 —— `--no-global-dedup` 会让
        # 用药 ON/OFF 也重复，那不在任何 --nodedup 前缀里，靠前缀推会漏掉。
        "repeatable_tokens_postshift": MEASURED_REPEATABLE_POSTSHIFT,
        # 人级别划分。`test` 为 0 时没有 test.bin（交付版的两分行为）。
        # layout 那句必须跟着代码走：下游如果要论证"我的 val 和你的 val 是同一批人"，
        # 靠的就是这句 + split_fractions.val 相同。
        "split_fractions": {"train": F_TRAIN, "val": F_VAL, "test": F_TEST},
        "split_layout": "shuffle(unique projid) 一次，按 [train | test | val] 切；"
                        "val 恒为末尾 int((1-f_val)*N): 之后那段，故同一 f_val 下 val 与两分版完全相同",
        "split_counts": {
            "train": {"patients": int(len(tr_ids)), "rows": int(len(train))},
            "val":   {"patients": int(len(va_ids)), "rows": int(len(val))},
            "test":  {"patients": int(len(te_ids)), "rows": int(len(test))}},
        # val_identical / train_is_prefix 是**同一次运行内的索引关系**（va_start 是否等于
        # int(0.9N)、ntr 是否 <= int(0.9N)），它们抓不到"排列变了"。
        # val_projids_match_reference_bin 才是实测：和落盘的基线 val.bin 逐人比。
        # null = 当时找不到基线文件，没查过 —— 不要当成 True 读。
        "split_matches_twoway_baseline": {"val_identical": bool(_same_val),
                                          "train_is_prefix": bool(_is_prefix),
                                          "val_projids_match_reference_bin": _ref_match,
                                          "reference_val_bin": _ref_val_bin if _ref_match is not None else None},
        "apoe_encoding": {"axis_dose": ["APOE_e4_0", "APOE_e4_1", "APOE_e4_2"],
                          "axis_e2": "APOE_e2carrier", "missing": "APOE_unk",
                          "valid_combos_dose_e2": spec.APOE_VALID_COMBOS,
                          "note": "(2,1) 非法: ε4/ε4 不可能同时携带 ε2。扰动时须校验。"}, "gap_dist_by_band": {L: {int(a): float(b) for a, b in zip(_GK[L], _GP[L])} for L in _ALAB}}
json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=1)

print(f"\n{'='*66}\n写出到 {OUT}\n{'='*66}")
print(f"  train.bin  {len(train):,} 行  ({len(tr_ids):,} 人)   pids[0:{ntr}]")
if len(te_ids):
    print(f"  test.bin   {len(test):,} 行  ({len(te_ids):,} 人)   pids[{ntr}:{va_start}]")
else:
    print(f"  test.bin   未写出（--split 的 test 比例为 0，交付版的两分行为）")
print(f"  val.bin    {len(val):,} 行  ({len(va_ids):,} 人)   pids[{va_start}:{_N}]")
print(f"  labels.csv {len(VOCAB)+2} 行   vocab_size={len(VOCAB)+2}")
print(f"  人级别划分自检: 三份 projid 两两无交集 ✓  并集 {len(_str|_sva|_ste):,} 人 = 全部 {_N:,} 人 ✓  "
      f"三份行数之和 {len(train)+len(val)+len(test):,} = 全部 {len(arr):,} ✓")
print(f"  与两分基线（切点 int(0.9*N)={_legacy}）对读: "
      f"切点 val 段相同 = {_same_val}；train 是前缀 = {_is_prefix}；"
      f"与落盘基线 val.bin 同一批人 = {_ref_match if _ref_match is not None else '未查（无基线文件）'}"
      f"{'' if _same_val else '   ← val 换人了，本次结果不能和现有的 val 指标并列引用'}")
print(f"\n序列长度: 中位 {int(L.median())}  p90 {int(L.quantile(.9))}  p95 {int(L.quantile(.95))}  "
      f"p99 {int(L.quantile(.99))}  max {int(L.max())}  → block_size={int(L.quantile(0.99))+8}")
print(f"ignore_tokens: {len(ignore)} 个 (padding + 性别 + 背景块)")

# 重复检查。没关全局去重时，非豁免 token **一次都不该**重复；关了就只报告，不断言。
_df = pd.DataFrame(arr, columns=["p", "a", "t"])
_rep = _df.groupby(["p", "t"]).size()
_rep = _rep[_rep > 1]
_is_rep = _rep.index.get_level_values("t").isin(list(REPEATABLE_IDS))
print(f"\n重复 token：豁免家族 {int(_is_rep.sum())} 个 (人, token) 组合重复；"
      f"其余 {int((~_is_rep).sum())} 个"
      f"{'（全局去重已关，允许）' if NO_GLOBAL_DEDUP else '（必须是 0）'}")
if not NO_GLOBAL_DEDUP:
    assert int((~_is_rep).sum()) == 0, "非豁免 token 出现重复 —— 全局去重被绕过了"
else:
    _extra = sorted({int(t) for t in _rep.index.get_level_values("t")} - set(REPEATABLE_IDS))
    print(f"  CONT 之外还重复的: {[VOCAB[t-1] for t in _extra]}")
if REPEATABLE_IDS:
    _n = _df.t.isin(list(REPEATABLE_IDS))
    print(f"  豁免家族共 {int(_n.sum()):,} 个 token（占全部 {100*_n.mean():.1f}%），"
          f"人均 {_n.sum()/_df.p.nunique():.1f} 个")
    # 往复：同一家族内 bin 回到曾经离开过的值，交付版结构上不可能发生
    _pre2ids = {p: sorted(i for n, i in TOK.items() if n.startswith(p)) for p in REPEATABLE_PREFIXES}
    for _p, _ids in _pre2ids.items():
        _sub = _df[_df.t.isin(_ids)].sort_values(["p", "a"], kind="stable")
        _rt = 0
        for _pid, _gg in _sub.groupby("p"):
            _seq = [t for t in _gg.t.tolist()]
            _seen = set(); _left = set()
            for _i, _t in enumerate(_seq):
                if _i and _t != _seq[_i-1] and _t in _seen:
                    _rt += 1
                if _i and _t != _seq[_i-1]:
                    _left.add(_seq[_i-1])
                _seen.add(_t)
        print(f"  {_p:<8s} 往复（回到曾经离开过的箱）{_rt} 次")

# Δt 诊断：死亡是否还是独立尖峰
import collections
dt = []
for pid, grp in pd.DataFrame(arr, columns=["p", "a", "t"]).groupby("p"):
    a = np.sort(grp.a.values.astype(float))
    dt.extend(np.diff(a).tolist())
dt = np.array(dt); dt = dt[dt > 0]
c = collections.Counter(np.round(dt / DPY, 2))
print(f"\nΔt(年) 分布 前8: " + ", ".join(f"{v}({100*n/len(dt):.1f}%)" for v, n in c.most_common(8)))
print(f"  唯一 Δt 取值数 {len(c)}  |  Δt 恰为整数年的占比 {100*np.mean(np.abs(dt/DPY - np.round(dt/DPY))<0.01):.1f}%")
