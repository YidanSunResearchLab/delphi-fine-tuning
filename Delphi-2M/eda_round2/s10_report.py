"""Section 10 -- the synthesis: the seven questions that gate writing any model code."""
import json

import pandas as pd

from common import SUMMARY, TABLES, report, summary_put
import os


def run():
    with open(SUMMARY) as f:
        s = json.load(f)
    bal = pd.read_csv(os.path.join(TABLES, "cv_fold_balance.csv"))
    bl = pd.read_csv(os.path.join(TABLES, "leakage_blacklist.csv"))
    excl = bl[bl["action"].str.startswith("EXCLUDE")]["column"].tolist()
    ld = pd.read_csv(os.path.join(TABLES, "level_plus_delta_sweep.csv"))
    ld = ld[ld.variable == "cogn_global"].set_index(["target", "n_level_bins"])["auc_loss_vs_raw"]

    def loss(target, n):
        return float(ld.loc[(target, n)])

    ready = {
        "1 有效样本量够不够": "否（小样本口径）",
        "2 time-to-next-event 头": "删除",
        "3 tokenizer": "离散分箱，10 档水平 + 4 档变化",
        "4 年龄截断": "gk 表未截断，可直接用",
        "5 attention mask": "同一 (projid, fu_year) 内互相屏蔽",
        "6 泄漏黑名单": f"{len(excl)} 个字段",
        "7 基线下限": "已确定（§9.5）",
    }
    summary_put(seven_questions_answered=ready)

    report(f"""## 10. 最终产出：七个问题的答案

规格书要求「在这七个问题全部有答案之前，不要开始写模型代码」。以下是答案。

### Q1 有效样本量是多少，够不够训 transformer

**{s['n_participants']:,} 人 / {s['n_visits']:,} 次访视，中位每人 {s['median_visits']:.0f} 次，
{s['pct_single_visit']:.1f}% 的人只有基线一次。两条阈值（5,000 人 / 100,000 访视）都不满足 → 小样本场景。**

后果（全部按 §3 决策门执行）：模型规模上限 1–3M 参数且应比 Delphi-2M 的 2.2M 更保守；
**必须 5 折 CV，不能单次 holdout**（单次 test 只有 {s['test_set_participants_if_80_10_10']} 人、{s['test_set_ad_cases_if_80_10_10']} 例 AD）；
必须与线性混合效应 / GRU 基线对照。
中位序列长度 {s['median_visits']:.0f}（勉强高于规格书 5 的下限），但
{100 - 100 * s['n_with_ge5_visits'] / s['n_participants']:.0f}% 的人不足 5 次访视——
用 transformer 的理由不能是「长程依赖」，只能是「统一表示 + 生成式联合采样」。

### Q2 time-to-next-event 头保留还是删除

**删除。**

Δage 的 CV = {s['delta_age_cv']:.3f}，看似过了 0.2 的线，但 **{s['delta_age_pct_exactly_1']:.1f}% 的间隔恰好等于 1.0 年**，
全部变异来自「漏了几次年度访视」——release 里**没有任何日期字段**，真实间隔已被抹除。
用健康状态回归 Δage 的 out-of-fold R² = **{s['delta_age_predictability_r2']:.4f}**（< 0.05）。
这个头会消耗参数、稀释主损失，且目标变量语义是错的。

**替代方案**（有价值、必须写进方法学）：改成**信息性缺访/脱落**的二元头——
「下次是否漏访」AUC {s['delta_age_gap_auc']:.3f}、「这次是否为末次访视」AUC {s['last_visit_auc_informative_dropout']:.3f}。
后者说明脱落确实与认知状态相关（§5），是真实可学的信号。

### Q3 tokenizer 用离散分箱还是连续嵌入，词表多大

**离散分箱，但规格书预设的「约 4 档」在这份数据上站不住，必须改成 ~10 档水平 + 4 档变化。**

- 4 档比 20 档少 **{s['discretization_gap_4_vs_20']['y_ad_next']['gap_4_vs_20']:.3f} / {s['discretization_gap_4_vs_20']['y_imp_next']['gap_4_vs_20']:.3f}** AUC，是 0.01 无差异带的数倍。
- 加 4 档变化 token 后，10 档水平的损失降到 **{loss('y_ad_next', 10):.4f} / {loss('y_imp_next', 10):.4f}**；再加到 20 档只多买回 {loss('y_ad_next', 10) - loss('y_ad_next', 20):.4f} / {loss('y_imp_next', 10) - loss('y_imp_next', 20):.4f}，不值得为此把 value embedding 翻倍。
- **分箱方式：等频（分位数），切点在训练折上确定后冻结写入配置**，验证/测试折复用同一组切点。
- 词表构成：`~19 个变量 ID × 10 档水平`（核心认知量另加 `× 4 档变化`）+ 静态协变量 token
  + `<死亡>`、`<delta_unknown>`、`<pad>`。量级约 300–400 个 token，对 1–3M 参数的模型完全可承受。
- **降级为静态协变量**：ICC ≥ 0.85 的变量（`bmi`）；
  **降级或排除**：只在个别 cycle 采集的子研究变量（`log_hcrp`/`log_hil6`/`log_htnfa`/`berlin_risk_class`/
  `psqi_sum`/`hba1c`/血脂）；**改成事件型 token**（只在首次转 1 时发射）：全部 `*_cum` 共病与 `*_rx` 用药。

### Q4 年龄是否被截断，如何处理

**主表没有被截断，可以直接用；但要注意两件事。**

- `cross-sectional-data-gk.xlsx` 的 `age_bl` 与 `age_first_ad_dx` 都是**未截断的浮点数**（重建的 `age_at_visit` 只有 {s['pct_age_capped_gk_tables']:.3f}% 的访视取到最大值，远低于 5% 的顶编判定线）→ 位置编码直接用 `age_bl + fu_year`。
- `ROSMAP_clinical.csv` 的 `age_at_visit_max` / `age_death` 有 **{s['pct_age_capped_rosmap_clinical']:.1f}% / {s['pct_age_death_capped_rosmap_clinical']:.1f}%** 是字符串 `'90+'`。
  **必须先解析再用**，否则会静默变 NaN 或按字典序错排。本轮所有死亡时点分析只用未顶编子集（n={s['n_death_age_usable']}）。
- 真正的年龄问题不是顶编而是**左截断**：只有 {s['pct_baseline_before_65']:.1f}% 在 65 岁前入组，
  中年期完全不可见，位置编码的有效区间约 70–100 岁。

### Q5 attention mask 需要屏蔽哪些同时刻 token

**同一 `(projid, fu_year)` 的所有 token 必须彼此完全不可见，只能看到严格更早 `fu_year` 的 token。**

一次访视同时产生 median **{s['n_measurements_per_visit_median']:.0f}** 个测量（mean {s['n_measurements_per_visit_mean']:.1f}）。
同一次访视内 `cogn_global` 与 `cts_estmmse30` 的相关系数 **r = {s['same_visit_cogn_mmse_r']:.3f}**——
不屏蔽就等于把答案放在输入里。这是本项目最容易踩、也最难事后发现的泄漏。
实现上：同一次访视的所有 token 共享同一个位置（年龄），block-diagonal 屏蔽。

### Q6 泄漏黑名单包含哪些字段

**{len(excl)} 个字段硬排除**（完整表：`tables/leakage_blacklist.csv`，建模代码里应硬编码为常量并断言）：

`{'`, `'.join(excl)}`

另有 {s['n_masked_not_excluded']} 个字段**不删除但必须靠 mask 处理**（同一访视内的共时测量，见 Q5）。

三条需要单独说明的：
1. `*_cum` 共病**不是泄漏**（审计确认全部单调非减，只用过去信息），但一旦置 1 永不回退 → 改事件型 token。
2. `cogn_global` 的 z 分基准用的是**全队列基线** cycle 的均值/标准差——不是未来信息，但跨了 split，
   严格做法是在训练折重算，实际影响极小，**必须在方法学写明**。
3. **构念循环**：`age_first_ad_dx` 由 `dcfdx` 定义，而临床医生给 `dcfdx` 时看的正是构成 `cogn_global` 的那批测验。
   这不是能靠黑名单解决的问题，只能在论文里明确声明（§8.5）。

### Q7 transformer 需要超过的基线指标是多少

5 折按人 CV，B3 = 年龄 + 性别 + 教育 + 上次 `cogn_global` + 变化量 + 上次 MMSE：

| 任务 | 指标 | 基线下限 |
|---|---|---|
| 下次访视 AD 诊断（正类 3.3%） | ROC-AUC / PR-AUC | **{s['baseline_b3_full_auc']:.4f} / {s['baseline_b3_full_pr_auc']:.4f}** |
| 5 年内 AD 诊断（正类 21.9%） | ROC-AUC / PR-AUC | **{s['baseline_auc_5y']:.4f} / {s['baseline_pr_auc_5y']:.4f}** |
| 下次 `cogn_global` | R² | **{s['best_next_cognition_r2']:.4f}**（LOCF 什么都不做已有 0.805） |

**并且必须正视这一条**：「上次认知分」**单变量**就有 AUC **{s['baseline_b0_last_cognition_auc']:.3f}**（§9.2 的「下一次访视」口径；§9.4 的 1 年 landmark 口径是 0.921，5 年 0.871），
远超规格书 0.85 的「没有空间了」判定线；把跨度拉到 5 年，最强基线也只掉约 0.04。
→ **本项目不应把「提升单点预测 AUC」当作目标**，那个目标在这份数据上大概率达不到。
可辩护的定位是：(a) 多变量**联合轨迹生成**（评估用校准、多步生成的分布保真度、生存曲线）；
(b) 条件模拟 / 反事实采样（逻辑回归给不出的东西）；(c) 明确写成**架构可行性验证**。

---

## 结论：可以开始写模型代码，但方案要按上面七条改

三个前置风险的判定：

| # | 风险 | 判定 |
|---|---|---|
| R1 | 访视间隔由方案决定，dt 头无信息 | **证实** → 删掉 dt 头，改成信息性缺访头 |
| R2 | 样本量不足以训 transformer | **证实** → 1–3M 参数上限 + 5 折 CV + 强基线对照，且要准备接受「数据量不支持」这个结论 |
| R3 | 连续测量无法直接 tokenize | **部分排除** → 离散分箱可行，但档数要从 4 提到 ~10（+4 档变化），不是规格书预设的那样 |

最需要注意的一条不在原始风险清单里：**基线太强 + 结局与主特征构念循环**，
使得「单点预测 AUC」不是一个值得追的目标。这一点应当在写第一行模型代码之前先和项目目标对齐。
""")
    print("s10 done")


if __name__ == "__main__":
    run()
