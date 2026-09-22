# -*- coding: utf-8 -*-
"""ROSMAP → Delphi 词表与分箱规格。全部硬分箱，优先用临床切点。"""

# ---------- 背景 / 静态 token（进 ignore_tokens，且参与 lifestyle age 随机化）----------
# 形式: (变量, token前缀, 分箱规则)
#   'cut'  : (切点list, 标签list)   左闭右开, 末箱含上界
#   'map'  : {原值: 标签}
BACKGROUND = [
    ("study",            "STUDY",    "map", {"ROS": "ROS", "MAP": "MAP", "LATC": "LATC"}),
    ("educ",             "EDU",      "cut", ([0, 12, 16, 20, 99], ["lt12", "12to15", "16to19", "ge20"])),
    ("race",             "RACE",     "cut", ([0.5, 1.5, 2.5, 7.5], ["white", "black", "other"])),
    ("spanish",          "HISP",     "map", {1: "yes", 2: "no"}),          # 1=是 2=否（反向编码）
    # APOE 二维编码：这一行只管 ε4 剂量轴；ε2 保护轴由 build.py 额外发射 APOE_e2carrier
    ("apoe_genotype",    "APOE",     "map", {22: "e4_0", 23: "e4_0", 33: "e4_0",
                                             24: "e4_1", 34: "e4_1", 44: "e4_2"}),
    ("smoking_bl",       "SMOKING",  "map", {0: "never", 1: "former", 2: "current"}),
    # Delphi 的 alcohol 是 3 级；ldai_bl 是每日杯数，按 0 / <1 / >=1 切
    ("ldai_bl",          "ALCOHOL",  "cut", ([-0.001, 0.001, 1.0, 99], ["none", "light", "heavy"])),
]
# 基线即患病（入组前发生，无时间戳）→ 背景 token
PREVALENT = [("hypertension_cum", "HTN"), ("dm_cum", "DM"), ("chf_cum", "CHF"),
             ("claudication_cum", "CLAUD"), ("heart_cum", "HEART")]

# ---------- 纵向连续量：硬分箱 + 去重 ----------
# 临床切点优先；无公认切点的用全样本分位数（在 build 时计算）
CONT = [
    ("bmi",         "BMI",    "cut", ([0, 22, 28, 999], ["low", "mid", "high"])),          # 同 Delphi
    ("sbp_avg",     "SBP",    "cut", ([0, 120, 140, 999], ["normal", "elevated", "high"])),
    ("dbp_avg",     "DBP",    "cut", ([0, 80, 90, 999], ["normal", "elevated", "high"])),
    ("glucose",     "GLU",    "cut", ([0, 100, 126, 9999], ["normal", "pre", "diabetic"])),
    ("hba1c",       "HBA1C",  "cut", ([0, 5.7, 6.5, 99], ["normal", "pre", "diabetic"])),
    ("hdlchlstrl",  "HDL",    "cut", ([0, 40, 60, 999], ["low", "mid", "high"])),
    ("ldlchlstrl",  "LDL",    "cut", ([0, 100, 160, 999], ["optimal", "borderline", "high"])),
    ("gfr_mdrs",    "GFR",    "cut", ([0, 60, 90, 999], ["ckd", "mild", "normal"])),       # CKD 分期
    ("cts_estmmse30","MMSE",  "cut", ([-0.1, 24, 27, 30.1], ["impaired", "borderline", "normal"])),
    ("cogn_global", "COGN",   "cut", ([-99, -1.0, 0.0, 99], ["low", "mid", "high"])),      # z 分数
    ("psqi_sum",    "PSQI",   "cut", ([-0.1, 5.5, 10.5, 99], ["good", "poor", "verypoor"])),  # >5 为睡眠差
    ("berlin_risk_class","APNEA","map", {0: "low", 1: "high"}),
    ("log_hcrp",    "CRP",    "qcut", 3),    # 无临床切点 → 三分位
    ("log_hil6",    "IL6",    "qcut", 3),
    ("log_htnfa",   "TNFA",   "qcut", 3),
]

# ---------- 事件 token ----------
EVENTS = ["AD_DX", "STROKE", "DEPRESSION"]
ONSET  = [("hypertension_cum", "HTN"), ("dm_cum", "DM"), ("chf_cum", "CHF"),
          ("claudication_cum", "CLAUD"), ("heart_cum", "HEART")]
MEDS   = [("antihyp_rx", "ANTIHYP"), ("statin_rx", "STATIN"),
          ("diabetes_rx", "DIABRX"), ("ad_rx", "ADRX")]

# ---------- 死后病理（放在 [DEATH] 之后）----------
PATHOLOGY = [
    ("braaksc",           "BRAAK",  "cut", ([-0.1, 2.5, 4.5, 6.1], ["low", "mid", "high"])),
    ("ceradsc",           "CERAD",  "cut", ([0.5, 1.5, 2.5, 4.5], ["high", "mid", "low"])),      # 1=最重→high
    ("gpath",             "GPATH",  "qcut", 3),
    ("amylsqrt_est_8reg", "AMYL",   "qcut", 3),
    ("tangsqrt_est_8reg", "TANG",   "qcut", 3),
    ("tdp_st4",           "TDP",    "cut", ([-0.5, 0.5, 1.5, 3.5], ["none", "mild", "severe"])), # RADC建议切在1|2
    ("lewydx_st4",        "LEWY",   "cut", ([-0.5, 0.5, 2.5, 3.5], ["none", "mild", "severe"])),
    ("arteriol_scler",    "ARTSCL", "cut", ([-0.5, 0.5, 1.5, 3.5], ["none", "mild", "severe"])),
    ("caa_4gp",           "CAA",    "cut", ([-0.5, 0.5, 1.5, 3.5], ["none", "mild", "severe"])),
    ("cvda_4gp2",         "CVDA",   "cut", ([-0.5, 0.5, 1.5, 3.5], ["none", "mild", "severe"])),
    ("ci_num2_mct",       "MICROINF","map", {0: "no", 1: "yes"}),
    ("ci_num2_tct",       "INFARCT","map", {0: "no", 1: "yes"}),
]

# 明确排除的变量（与前面分析一致）
# ε2 携带者（额外发射一个 token，与 ε4 剂量正交）
E2_CARRIER = {22, 23, 24}
# (ε4剂量, ε2携带) 的合法组合；(2,1) 生物学上不可能——一个人只有两个等位基因
APOE_VALID_COMBOS = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)]

EXCLUDED = {
    "cogng_demog_slope": "严重泄漏：由含未来的全部 cogn_global 拟合得到 (r=0.83)",
    "cogdx":             "研究结束时的汇总诊断 = 泄漏",
    "dcfdx_lv":          "最后一次评估的诊断 = 泄漏",
    "pmi":               "尸检排期，非病人属性，会制造虚假扰动靶点",
    "cts_mmse30_lv":     "汇总量，逐访 cts_estmmse30 已覆盖",
    "cts_mmse30_first_ad_dx": "汇总量，冗余",
    "age_at_visit_max":  "CSV 版删截；由 age_bl+fu_year 重建",
    "age_death":         "仅用于确定 [DEATH] 位置，不作为 token",
    "projid":            "标识符",
    "individualID":      "标识符（保留用于接组学）",
    "age_bl":            "时间轴原点",
    "fu_year":           "时间轴",
    "died":              "→ [DEATH] / [CENSORED] 终止符",
}
