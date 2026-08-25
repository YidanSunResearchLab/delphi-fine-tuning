# 序数量表变量专项 EDA — Round 3

第三轮 EDA，按《序数量表变量专项 EDA 规格书》逐节执行。前两轮
（`../eda/eda_radc.py` → `../out-radc-eda/`，`../eda_round2/` → 自带 `eda_out/`）
**都没有任何改动**，本目录是与之并列的独立一轮。

这一轮和前两轮的目的不同：前两轮描述数据长什么样，**这一轮只检验一个假设——
「这个变量是有序标度」在数据上成立吗？** 序数平滑正则和锚点插值都建立在它之上，
假设不成立而强行施加，会主动损害模型。

## 跑法

```bash
cd Delphi-2M/eda_round3
python run_all.py          # 冷启动约 80 秒（含读 512MB NACC 导出并建缓存），之后约 65 秒
```

单独跑某一节：`python -c "import s3_specific; s3_specific.run()"`。
`run_all.py` 会先清空 `eda_out/scales/REPORT.md` 与 `summary.json`，单节运行是**追加**。

### NACC 数据位置

规格书 §1 的变量清单横跨 NACC 与 RADC，其中 CDR 六域、MoCA、FAQ、GDS、NACCUDSD
**只有 NACC 有**。原始 NACC 导出是 PHI，**不在本仓库内**
（`.gitignore` 拦截 `**/investigator_nacc*.csv`）。`common.nacc_path()` 的解析顺序：

1. 环境变量 `NACC_CSV`
2. `data/NACC/investigator_nacc*.csv`（仓库内，若有人放进来）
3. `~/cc_test/EDA_NACC/investigator_nacc*.csv`

载入时 `NACCID` 立刻 factorize 成整数 `pid` 并丢弃，所以 `.cache/` 与全部 source-data CSV
都不含 NACC 标识符。**找不到 NACC 时不会报错**：NACC 变量在 §1 记为
`panel_unavailable`，RADC 部分照常跑完。

## 产出

```
eda_out/scales/
  REPORT.md                    逐节的「算了什么 / 关键数值 / 决策门判定」，末尾 §6 是十个问题的答案 ← 先读这个
  SCALE_SUMMARY.csv            每变量一行，列名与规格书 §5 表格一一对应
  summary.json                 所有关键数值的机读汇总
  figs/                        跨变量的图（相关热图、泄漏矩阵），每张配同名 _data.csv
  tables/                      跨变量的表（variable_detection、leakage matrix、incremental value …）
  <dataset>_<var>/figs/        逐变量的 §2.1/§2.2/§2.4/§2.6 图
  <dataset>_<var>/tables/      逐变量的取值清点、相邻档可分性、专项表
  .cache/                      NACC 子集 parquet + RADC pickle（加速重跑）
```

## 模块

| 文件 | 规格书章节 | 回答什么 |
|---|---|---|
| `common.py` | §1 | 路径解析、变量注册表（含 codebook 原文注释）、三个 panel、参照结局、统计小工具、报告记账 |
| `ordinal.py` | §2 | **通用组件**：`analyze_ordinal()` 一个函数跑完 §2.1–§2.7，每个变量共用 |
| `ml.py` | §3.6 / §4.3 | 按人分组的 OOF AUC 与前向选择（填补/标准化只在训练折上估计） |
| `s1_detect.py` | §1 | 存在性与编码探测；哨兵码 vs 非法值逐一判定；与规格书不符之处 |
| `s2_generic.py` | §2 | 对全部通过 §1 的变量跑通用组件，按小节横向对比 + 决策门 |
| `s3_specific.py` | §3 | 九个专项：MMSE / MoCA / CDRSUM / CDRGLOB / FAQ / GDS / NACCUDSD / dcfdx / 病理 |
| `s4_cross.py` | §4 | 冗余与共线、同时刻泄漏矩阵、增量价值排序 |
| `s5_summary.py` | §5 | `SCALE_SUMMARY.csv` |
| `s6_answers.py` | §6 | 十个问题的答案 |

## 五条必须知道的数据事实

1. **`niareagansc` 在 RADC 这个 release 里根本不存在**，codebook 与三张表都没有。
   规格书 §1.3 关于它「与 Braak 反向」的预期无法核对。病理结局只剩 Braak 与 CERAD。
2. **`cts_mmse30` 的实际列是 `cts_estmmse30`，而且不是原始 MMSE。** codebook p.230：
   该列是「MMSE 分数，**或由 MoCA 换算出的**完整 MMSE 估计分」，1.55% 的取值不是整数。
   规格书表格里「0–30 整数」这个预期是错的。
3. **RADC 没有逐次访视的诊断。** 只有人级 `dcfdx_lv`（最后一次有效访视）。
   §3.8 的序数性检验因此改在人级做，参照结局换成尸检 Braak ≥ 4。
4. **`age_first_ad_dx` 对基线已痴呆者不记录。** 不排除这些人，他们会以「永远 at-risk 且
   永远标 0」的身份留在样本里，而且集中在各认知量表的最低端，把整条剂量-反应曲线压平
   （实测：`cts_estmmse30` 的 incident |ρ| 从 0.03 恢复到 0.90）。本轮按 round 2 的同一规则剔除。
5. **NACC 的 FAQ 分项里 `8` 是「从不做该活动」，不是 0。** 记成 0 会系统性低估总分，
   而且偏差与性别、独居状况相关。本 release 没有 FAQ 总分列，本轮自行合成两个口径。

## 一个方法学决定

规格书 §2.6 的参照结局字面定义是 `P(下次访视为痴呆 | 当前取值 = v)`，**不排除 t 时刻已经
痴呆的行**。对严重度量表来说，那部分测的是「痴呆会不会持续」而不是「会不会发生」，
于是 CDRGLOB、五个 CDR box、NACCUDSD 的 |ρ| 全都恰好等于 1.000。
本轮**两个口径都算**：字面口径用于判定（规格书要求的就是它），
at-risk-only 的 incident 口径并列报告，两者不一致处在 §2.6 末尾与 §6 Q1 明确指出。

## 输出为什么不进版本库

`eda_out/` 在 `.gitignore` 里。图的 source-data CSV 与 `.cache/` 是 participant-level 的
测量值，按仓库既有的 NACC/RADC 数据政策不入库；全部可由 `python run_all.py` 复现。
