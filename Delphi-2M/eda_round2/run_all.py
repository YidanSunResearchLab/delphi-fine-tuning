"""Round-2 RADC EDA -- run every section in order into eda_out/.

  python run_all.py

Round 1 (../eda/eda_radc.py -> ../out-radc-eda/) is a separate, untouched pass.
"""
import time

import common
import s2_inventory, s3_cohort, s4_timeline, s5_missingness
import s6_tokenization, s7_outcomes, s8_leakage, s9_splits_baselines, s10_report

SECTIONS = [s2_inventory, s3_cohort, s4_timeline, s5_missingness,
            s6_tokenization, s7_outcomes, s8_leakage, s9_splits_baselines]


def main():
    common.reset_report()
    common.report("""# RADC 数据集 EDA — Round 2

针对 `data/RADC/` 的三张表，按《RADC 数据集 EDA 规格书》逐节执行的第二轮分析。
**目标：为 age-encoded transformer（Delphi-2M 风格）建模做前置评估。**

- 第一轮（`Delphi-2M/eda/eda_radc.py` → `Delphi-2M/out-radc-eda/`）**未改动**，本轮是并列的独立目录。
- 随机种子固定 42。图在 `figs/`，表在 `tables/`，数值摘要在 `summary.json`，字段映射在 `schema_map.json`。
- 变量语义全部以 RADC codebook 为准（见 `schema_map.json` 的 `codebook_note`），**没有任何字段的含义是从名字推断的**。
- 所有交叉验证均按 `projid` 分组，同一人的全部访视不跨折；分箱切点、填补统计量只在训练折上估计。
- 每节末尾的「决策门」标 `[PASS]`（可按规格书原方案继续）或 `[ACTION]`（必须改方案）。

**先读 §10** —— 七个问题的答案都在那里。
""")
    t0 = time.time()
    for m in SECTIONS:
        t = time.time()
        m.run()
        print(f"    {m.__name__}: {time.time() - t:.1f}s")
    s10_report.run()
    print(f"total {time.time() - t0:.1f}s -> {common.REPORT}")


if __name__ == "__main__":
    main()
