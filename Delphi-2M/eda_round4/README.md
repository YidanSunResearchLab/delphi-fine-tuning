# Round 4 — 五个变量的覆盖率

只做一件事：CDRSUM / MOCATOTS / FAQTOTAL / NACCUDSD / Death 在 **NACC 原始导出文件**里的
覆盖率。分母是全部访视行，**没有做任何 deduplicate**。

```bash
cd Delphi-2M/eda_round4
python coverage.py                       # 自动找 NACC 导出文件，约 25 秒
python coverage.py --nacc /path/to/file  # 或者指定
```

产出全在 `eda_out/`（已 gitignore，含 participant-level 数值）：

```
COVERAGE.md            ← 先读这个，六节
COVERAGE_SUMMARY.csv   第 1 节的表
tables/                六张明细表
figs/                  两张图 + 各自的 source-data CSV
```

## 关于数据源

这四个量表变量**只存在于 NACC**，RADC/ROSMAP 三张表里一个都没有（round 3 §1 已核对）。
本机上 NACC 只有一份 CSV 导出（`~/cc_test/EDA_NACC/investigator_nacc72.csv`，512 MB，
1024 列），**没有 xlsx 版本**。若你指的是另一个文件，用 `--nacc` 指过去即可，
脚本对 csv 与 xlsx 都能读。

原始文件 207,455 行，丢弃 1 行末尾空行后 **207,454 行 / 55,268 人**。
已核对 `(参与者, NACCVNUM)` 无重复——这个文件本身就是每次访视一行，不存在需要去重的情况。

`NACCID` 在载入时立刻 factorize 成整数并丢弃，所以 `eda_out/` 里不含 NACC 标识符。

## 三个不能合并成一个数字的区分

1. **缺失的原因**。`-4` = 该表单版本没这道题（结构性），`88` = 有题没施测，
   `95–98` = 参与者无法/拒绝完成。MoCA 63.7% 的缺失里，62.1pp 是 `-4`，只有 1.6pp 是真没测。
2. **时间**。MoCA 2015 年 UDS v3 才启用，把它平均进 2005–2014 的访视得到的数字
   不描述任何真实队列。第 5 节按年份和表单版本分列。
3. **FAQTOTAL 的合成规则**。NACC 不发布 FAQ 总分，只有 10 个分项。
   要求十项全有 = 74.4%，接受 ≥8 项 = 92.3%。第 1 节用严格口径。

## Death 为什么单独一节

`NACCDIED` 是人级标记复制到每一行，行覆盖率恒为 100%，这个数字没有信息量。
真正该问的是死亡率（26.8%）与死亡日期的记录率（100%）。
而且它是**右删失的生存结局**，不是协变量：`NACCDIED = 0` 意思是
「截至最后一次随访尚未记录死亡」，当二分类标签训练会把随访时长学成死亡风险。
