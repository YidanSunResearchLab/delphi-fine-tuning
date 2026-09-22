# -*- coding: utf-8 -*-
"""ROSMAP → Delphi .bin 格式 tokenizer。纯硬分箱，临床切点优先。"""
import sys, os, json, pickle, argparse
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import spec

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/Users/yihuang/cc_test/ADPROJECTION/0912_newstart_with_perturbation_engine"

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
_ap.add_argument("--dataset", default="rosmap",
                 help="生成的训练配置里的 dataset 名（决定 delphi/data/<name> 与 out_dir）")
_ap.add_argument("--nodedup", default="",
                 help="逗号分隔的 token 前缀，这些变量每次随访都发射且不去重。例: MMSE,COGN")
_A = _ap.parse_args()
OUT = _A.out
DATASET = _A.dataset
NODEDUP = tuple(x.strip() for x in _A.nodedup.split(",") if x.strip())
_cont_pre = {pre for _, pre, _, _ in spec.CONT}
_bad = [p for p in NODEDUP if p not in _cont_pre]
if _bad:
    raise SystemExit(f"--nodedup 里的前缀 {_bad} 不在 spec.CONT 中（可选: {sorted(_cont_pre)}）")
os.makedirs(OUT, exist_ok=True)
print(f"输出 -> {OUT}   dataset={DATASET}   不去重的前缀: {list(NODEDUP) or '无（交付版行为）'}")
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
    for col, tok in [("r_stroke", "STROKE"), ("r_depres", "DEPRESSION")]:
        h = g[g[col].isin([1, 2, 3])]
        if len(h): add(tok, float(h.fu_year.iloc[0]), "event")
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
        if tid not in REPEATABLE_IDS:                          # 豁免的前缀跳过全局去重
            if tid in seen: stats["dedup"] += 1; continue
            seen.add(tid)
        rows.append((int(pid), age_d, tid))

arr = np.array(rows, dtype=np.uint32)
print(f"\n总 token 行数 {len(arr):,}  涉及 {len(np.unique(arr[:,0])):,} 人")
print("构成:", {k: f"{v:,}" for k, v in stats.items()})

# ============================ 划分 / 写出 ============================
pids = np.unique(arr[:, 0]); RNG.shuffle(pids)
ntr = int(0.9 * len(pids)); tr_set = set(pids[:ntr].tolist())
mask = np.isin(arr[:, 0], list(tr_set))
train, val = arr[mask], arr[~mask]
# .bin 要求同一病人的行连续
train = train[np.argsort(train[:, 0], kind="stable")]
val   = val[np.argsort(val[:, 0], kind="stable")]
train.astype(np.uint32).tofile(f"{OUT}/train.bin")
val.astype(np.uint32).tofile(f"{OUT}/val.bin")
pd.DataFrame({"event_name": ["Padding", "No event"] + VOCAB}).to_csv(f"{OUT}/labels.csv", index=False)

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
t_min = 0.1
token_dropout = 0.1        # 稀疏 token 的正则化（见 README 稀疏性一节）
no_event_token_rate = 5
'''
open(f"{OUT}/train_delphi_rosmap.py", "w").write(cfg)

meta = {"vocab": VOCAB, "tok": TOK, "bg_range_preshift": [BG_START, BG_END],
        "path_range_preshift": [PATH_START, PATH_END], "death_token_preshift": TOK["DEATH"],
        "ignore_tokens_postshift": ignore, "vocab_size": len(VOCAB) + 2,
        "excluded": spec.EXCLUDED,
        # 去重豁免。下游（figure2_eval/radc_delphi/vocab.py）据此决定 rollout 的 no-repeat
        # 规则；空 = 每 token 每人最多一次，Delphi 的 first-occurrence 语义。
        "nodedup_prefixes": list(NODEDUP),
        "repeatable_token_prefixes": REPEATABLE_PREFIXES,
        "repeatable_tokens_postshift": sorted(int(t) + 1 for t in REPEATABLE_IDS),
        "apoe_encoding": {"axis_dose": ["APOE_e4_0", "APOE_e4_1", "APOE_e4_2"],
                          "axis_e2": "APOE_e2carrier", "missing": "APOE_unk",
                          "valid_combos_dose_e2": spec.APOE_VALID_COMBOS,
                          "note": "(2,1) 非法: ε4/ε4 不可能同时携带 ε2。扰动时须校验。"}, "gap_dist_by_band": {L: {int(a): float(b) for a, b in zip(_GK[L], _GP[L])} for L in _ALAB}}
json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=1)

print(f"\n{'='*66}\n写出到 {OUT}\n{'='*66}")
print(f"  train.bin  {len(train):,} 行  ({ntr:,} 人)")
print(f"  val.bin    {len(val):,} 行  ({len(pids)-ntr:,} 人)")
print(f"  labels.csv {len(VOCAB)+2} 行   vocab_size={len(VOCAB)+2}")
print(f"\n序列长度: 中位 {int(L.median())}  p90 {int(L.quantile(.9))}  p95 {int(L.quantile(.95))}  "
      f"p99 {int(L.quantile(.99))}  max {int(L.max())}  → block_size={int(L.quantile(0.99))+8}")
print(f"ignore_tokens: {len(ignore)} 个 (padding + 性别 + 背景块)")

# 重复检查：豁免的 token 应当重复，其余的**一次都不该**重复
_df = pd.DataFrame(arr, columns=["p", "a", "t"])
_rep = _df.groupby(["p", "t"]).size()
_rep = _rep[_rep > 1]
_is_rep = _rep.index.get_level_values("t").isin(list(REPEATABLE_IDS))
print(f"\n重复 token：豁免家族 {int(_is_rep.sum())} 个 (人, token) 组合重复；"
      f"其余家族 {int((~_is_rep).sum())} 个（必须是 0）")
assert int((~_is_rep).sum()) == 0, "非豁免 token 出现重复 —— 全局去重被绕过了"
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
