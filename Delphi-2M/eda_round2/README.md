# RADC EDA — Round 2

第二轮 EDA，按《RADC 数据集 EDA 规格书》逐节执行。**第一轮（`../eda/eda_radc.py` →
`../out-radc-eda/`）没有任何改动**，本目录是与之并列的独立一轮。

## 跑法

```bash
cd Delphi-2M/eda_round2
python run_all.py          # 约 3–4 分钟，产出全部落在 eda_out/
```

单独跑某一节：`python -c "import s4_timeline; s4_timeline.run()"`（注意 `run_all.py`
会先清空 `eda_out/REPORT.md` 与 `summary.json`，单节运行是**追加**）。

## 产出

```
eda_out/
  REPORT.md        逐节的「算了什么 / 关键数值 / 决策门判定」，末尾 §10 是七个问题的答案 ← 先读这个
  summary.json     所有关键数值的机读汇总
  schema_map.json  字段角色映射 + 每个变量的 codebook 原文注释
  figs/            每张图配一份同名 _data.csv（source data）
  tables/          全部统计表
```

## 模块

| 文件 | 规格书章节 | 回答什么 |
|---|---|---|
| `common.py` | — | 加载三张表、codebook 语义表、调色板、报告/摘要的记账 |
| `labels.py` | — | 两个代理结局（本 release 没有逐次访视诊断，必须重建） |
| `s2_inventory.py` | §2 | 盘点、逐列画像、join 完整性、`schema_map.json` |
| `s3_cohort.py` | §3 | 队列与序列规模（风险 R2） |
| `s4_timeline.py` | §4 | 年龄轴重建的正确性、Δage、年龄顶编（风险 R1） |
| `s5_missingness.py` | §5 | 三类缺失的区分、脱落、终末期下降 |
| `s6_tokenization.py` | §6 | ICC、离散化损失、变化 token、同时刻 token 数（风险 R3） |
| `s7_outcomes.py` | §7 | 状态转移与逆转率、AD 进展、临床 vs 病理 |
| `s8_leakage.py` | §8 | 泄漏黑名单 + 定量演示 |
| `s9_splits_baselines.py` | §9 | 5 折划分、基线、跨度分析 |
| `s10_report.py` | §10 | 七个问题的答案 |

## 三条必须知道的数据事实

1. **这个 release 没有逐次访视的年龄，也没有逐次访视的诊断。** 年龄只能重建为
   `age_bl + fu_year`（§4.1 验证了年度网格成立），诊断只能用 person-level 的
   `age_first_ad_dx` 还原成事件。规格书里假定存在的 `age_at_visit`、`dcfdx`、
   5 个认知域分、`cts_*` 单项测验、访视日期，全都不在。
2. **`ROSMAP_clinical.csv` 的三个 age 列是字符串**，90 岁以上顶编成 `'90+'`。
   两个 `*-gk.xlsx` 里的年龄没有顶编——**位置编码用 gk 表的，不要用 ROSMAP_clinical 的**。
3. **反向编码**：`r_stroke` / `r_depres` 的 `4 = 没有该诊断`；`ceradsc` 的 `1 = definite AD`、
   `4 = no AD`。这三个最容易在建模时搞反。

## 输出为什么不进版本库

`eda_out/` 在 `.gitignore` 里。图表的 source-data CSV 与 `tables/cv_folds.csv` 含
participant-level 的 `projid` 与测量值，按仓库既有的 RADC/NACC 数据政策不入库；
全部可由 `python run_all.py` 复现。
