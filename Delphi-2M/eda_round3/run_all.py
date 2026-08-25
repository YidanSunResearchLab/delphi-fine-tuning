"""Round-3 ordinal-scale audit -- run every section in order into eda_out/scales/.

  python run_all.py            # ~2 min once the NACC subset is cached

Rounds 1 (../eda/eda_radc.py) and 2 (../eda_round2/) are separate, untouched passes.
"""
import time

import common
import s1_detect, s2_generic, s3_specific, s4_cross, s5_summary, s6_answers

SECTIONS = [s1_detect, s2_generic, s3_specific, s4_cross, s5_summary, s6_answers]

HEADER = """# 序数量表变量专项 EDA — Round 3

按《序数量表变量专项 EDA 规格书》逐节执行。**这一轮只回答一个问题：
「这个变量是有序标度」这个假设在数据上成立吗？** 序数平滑正则与锚点插值都建立在它之上，
假设不成立而强行施加，会主动损害模型。

- 前两轮（`Delphi-2M/eda/` → `out-radc-eda/`，`Delphi-2M/eda_round2/` → 自带 `eda_out/`）
  **均未改动**，本目录是并列的第三轮。
- 随机种子 42，matplotlib 用 Agg 后端。图在 `figs/` 与 `<dataset>_<var>/figs/`，
  表在 `tables/` 与 `<dataset>_<var>/tables/`，机读汇总在 `summary.json`，
  逐变量一行的汇总在 `SCALE_SUMMARY.csv`。
- 每张图都配一份同名 `_data.csv`（source data）。
- 每节末尾的「决策门」标 `[PASS]`（规格书原方案可照原样继续）或 `[ACTION]`（必须改方案）。

**两个数据集**，因为规格书 §1 的变量清单横跨两者，而它们没有共同的行：

| panel | 来源 | 规模 |
|---|---|---|
| `nacc_visit` | NACC UDS investigator 导出（不在本仓库，见 §1） | 207,454 次访视 / 55,268 人 / 2005–2025 |
| `radc_visit` | `data/RADC/longitudinal_data_gk.xlsx` | 36,138 次访视 / 4,428 人 |
| `radc_person` | `data/RADC/cross-sectional-data-gk.xlsx` + `ROSMAP_clinical.csv` | 4,428 人 |

**先读 §6** —— 规格书要求回答的十个问题，答案都在那里。
"""


def main():
    common.reset_report()
    common.report(HEADER)
    t0 = time.time()
    for m in SECTIONS:
        t = time.time()
        m.run()
        print(f"    {m.__name__}: {time.time() - t:.1f}s", flush=True)
    print(f"total {time.time() - t0:.1f}s -> {common.REPORT}")


if __name__ == "__main__":
    main()
