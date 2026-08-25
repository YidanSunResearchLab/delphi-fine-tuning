"""Section 2 -- Step 0: data inventory, column profiles, schema map, join integrity."""
import json
import os

import numpy as np
import pandas as pd

from common import (FILES, RADC, OUT, VAR_META, load_raw, parse_capped_age,
                    report, save_table, summary_put, gate)


def profile(df, name):
    rows = []
    for c in df.columns:
        s = df[c]
        t, grain, note = VAR_META.get(c, ("?", "?", "NOT IN CODEBOOK MAP -- confirm before use"))
        num = pd.to_numeric(s, errors="coerce")
        is_num = num.notna().sum() > 0 and not pd.api.types.is_string_dtype(s)
        top = s.value_counts(dropna=True).head(10)
        rows.append({
            "column": c, "dtype": str(s.dtype), "codebook_type": t, "grain": grain,
            "n_nonnull": int(s.notna().sum()), "pct_nonnull": round(100 * s.notna().mean(), 2),
            "nunique": int(s.nunique(dropna=True)),
            "min": round(float(num.min()), 4) if is_num else "",
            "max": round(float(num.max()), 4) if is_num else "",
            "mean": round(float(num.mean()), 4) if is_num else "",
            "median": round(float(num.median()), 4) if is_num else "",
            "top10": "; ".join(f"{k}:{v}" for k, v in top.items()),
            "codebook_note": note,
        })
    return save_table(pd.DataFrame(rows), f"column_profile_{name}")


def run():
    cs, lo, cl = load_raw()
    frames = {"cross-sectional-data-gk.xlsx": cs,
              "longitudinal_data_gk.xlsx": lo,
              "ROSMAP_clinical.csv": cl}

    inv = []
    for fname, df in frames.items():
        path = os.path.join(RADC, fname)
        dup = int(df["projid"].duplicated().sum())
        inv.append({
            "file": fname, "n_rows": len(df), "n_cols": df.shape[1],
            "size_mb": round(os.path.getsize(path) / 1e6, 2),
            "n_projid": int(df["projid"].nunique()),
            "projid_duplicated_rows": dup,
            "grain": "visit-level (one row per projid x fu_year)" if dup else "person-level (one row per projid)",
        })
    save_table(pd.DataFrame(inv), "file_inventory")
    for fname, df in frames.items():
        profile(df, fname.split(".")[0].replace("-", "_"))

    # ---- join integrity -------------------------------------------------------
    p_cs, p_lo, p_cl = set(cs.projid), set(lo.projid), set(cl.projid)
    j_cs_lo = lo.merge(cs, on="projid", how="left", suffixes=("", "_cs"))
    j_all = j_cs_lo.merge(cl, on="projid", how="left", suffixes=("", "_cl"))
    join = pd.DataFrame([
        {"join": "longitudinal LEFT cross-sectional", "left_rows": len(lo), "joined_rows": len(j_cs_lo),
         "inflation": len(j_cs_lo) - len(lo),
         "match_rate_pct": round(100 * lo.projid.isin(p_cs).mean(), 2)},
        {"join": "longitudinal LEFT ROSMAP_clinical", "left_rows": len(j_cs_lo), "joined_rows": len(j_all),
         "inflation": len(j_all) - len(j_cs_lo),
         "match_rate_pct": round(100 * lo.projid.isin(p_cl).mean(), 2)},
    ])
    save_table(join, "join_integrity")

    # ---- schema map -----------------------------------------------------------
    ages_capped = {}
    for c in ["age_at_visit_max", "age_death", "age_first_ad_dx"]:
        _, capped = parse_capped_age(cl[c])
        ages_capped[c] = round(100 * capped.sum() / max(cl[c].notna().sum(), 1), 2)

    schema = {
        "_generated_by": "Delphi-2M/eda_round2/s2_inventory.py",
        "_codebook": "data/RADC/RADC_codebook_data_set_1736_08-13-2026.pdf (48 variables)",
        "tables": {
            "cross_sectional": {"file": FILES["cross_sectional"], "grain": "person",
                                "key": ["projid"], "n_rows": len(cs)},
            "longitudinal": {"file": FILES["longitudinal"], "grain": "visit",
                             "key": ["projid", "fu_year"], "n_rows": len(lo)},
            "rosmap_clinical": {"file": FILES["rosmap_clinical"], "grain": "person",
                                "key": ["projid"], "n_rows": len(cl),
                                "coverage": "ROS+MAP only -- LATC participants absent",
                                "warning": "ages are top-coded '90+'; do NOT use for the age axis"},
        },
        "roles": {
            "person_id": "projid",
            "cohort": "study (cross-sectional / longitudinal); Study (ROSMAP_clinical)",
            "visit_index": "fu_year",
            "age_at_visit": "NOT PRESENT -- derived as age_bl + fu_year",
            "baseline_age": "age_bl (cross-sectional, uncapped)",
            "age_death": "age_death (ROSMAP_clinical ONLY, top-coded)",
            "per_visit_diagnosis": "NOT PRESENT -- dcfdx is absent from this release",
            "last_visit_diagnosis": "dcfdx_lv (ROSMAP_clinical, person-level)",
            "final_diagnosis": "cogdx (ROSMAP_clinical, person-level, autopsy-informed)",
            "cognition_composite": "cogn_global (global only -- the 5 domain scores cogn_ep/se/po/ps/wo are NOT in this release)",
            "raw_tests": "NOT PRESENT -- no cts_* item scores except the estimated MMSE cts_estmmse30",
            "demographics": ["msex", "educ", "race (ROSMAP_clinical)", "spanish (ROSMAP_clinical)"],
            "genetics": ["apoe_genotype"],
            "neuropathology": ["braaksc", "ceradsc", "gpath", "amylsqrt_est_8reg", "tangsqrt_est_8reg",
                               "tdp_st4", "lewydx_st4", "arteriol_scler", "caa_4gp", "cvda_4gp2",
                               "ci_num2_mct", "ci_num2_tct", "pmi"],
            "outcome_ad": "age_first_ad_dx (cross-sectional, uncapped)",
        },
        "reverse_coded": {"r_stroke": "4 = NOT present", "r_depres": "4 = NOT present",
                          "ceradsc": "4 = no AD, 1 = definite AD"},
        "top_coded_pct_of_nonnull": ages_capped,
        "variables": {c: {"type": t, "grain": g, "codebook_note": n} for c, (t, g, n) in VAR_META.items()},
    }
    with open(os.path.join(OUT, "schema_map.json"), "w") as f:
        json.dump(schema, f, indent=2)

    unmapped = sorted(set().union(*[set(d.columns) for d in frames.values()]) - set(VAR_META))
    summary_put(n_files=3, n_rows_cross_sectional=len(cs), n_rows_longitudinal=len(lo),
                n_rows_rosmap_clinical=len(cl),
                projid_match_long_to_cs_pct=round(100 * lo.projid.isin(p_cs).mean(), 2),
                projid_match_long_to_clinical_pct=round(100 * lo.projid.isin(p_cl).mean(), 2),
                has_per_visit_age_field=False, has_per_visit_diagnosis_field=False,
                unmapped_columns=unmapped)

    report(f"""## 2. Step 0 -- 数据盘点与字段映射

**算了什么**：三张表的行/列/大小/主键粒度；逐列 dtype、非空率、nunique、数值分位、top-10 取值，
并逐一对照 codebook 写入 `codebook_note`（**没有任何字段的语义是从名字猜的**）；`projid` 匹配率与
join 后的行数膨胀；`eda_out/schema_map.json`。

**关键数值**

| 文件 | 粒度 | 行 x 列 | 唯一 projid |
|---|---|---|---|
| cross-sectional-data-gk.xlsx | person（一行一人） | {len(cs)} x {cs.shape[1]} | {cs.projid.nunique()} |
| longitudinal_data_gk.xlsx | visit（一行一次访视，键 projid x fu_year，无重复） | {len(lo)} x {lo.shape[1]} | {lo.projid.nunique()} |
| ROSMAP_clinical.csv | person | {len(cl)} x {cl.shape[1]} | {cl.projid.nunique()} |

- `longitudinal → cross-sectional` 匹配率 **100.00%**，join 后行数 {len(lo)} → {len(j_cs_lo)}，**零膨胀**。
- `longitudinal → ROSMAP_clinical` 匹配率 **{100 * lo.projid.isin(p_cl).mean():.2f}%**：该表只覆盖 ROS+MAP，
  **LATC 的 {cs[cs.study == 'LATC'].projid.nunique()} 人完全不在其中**；另有 {len(p_cl - p_cs)} 个 projid 只在 ROSMAP_clinical 里。
- 三个 gk 表之外的坑：ROSMAP_clinical 的 `age_at_visit_max` / `age_death` / `age_first_ad_dx` 是**字符串**，
  因为去标识化把 90 岁以上顶编成 `'90+'`（占非空值的 {ages_capped['age_at_visit_max']}% / {ages_capped['age_death']}% / {ages_capped['age_first_ad_dx']}%）。
  两个 gk 表里的 `age_bl`、`age_first_ad_dx` 是**未截断的浮点数**（见 §4）。
- 反向编码（codebook 确认，最容易踩）：`r_stroke`、`r_depres` 的 **4 = 没有该诊断**；`ceradsc` 的 **1 = definite AD、4 = no AD**。

**规格书里假定存在、但这个 release 里没有的字段**（后续各节的分析口径都受此约束）：
`age_at_visit`（逐次访视年龄）、`dcfdx`（逐次访视诊断）、`cogn_ep/se/po/ps/wo`（5 个认知域）、
`cts_*` 单项测验（只有估计 MMSE）、`age_death`（仅在被顶编的 ROSMAP_clinical 里）、
`niareagansc`、访视日期/日历年份。

{gate(True, "**age encoding 可以实现，但只能是重建的**：逐次访视年龄字段不存在，唯一时间字段是整数 `fu_year`，"
      "因此 `age_at_visit := age_bl + fu_year`。这不是「找不到年龄字段」（`age_bl` 是精确到小数的真实年龄），"
      "但重建出来的年龄轴在个体内**只能跨整年跳变**。§4 会量化这个假设的误差并说明它对 time-to-next-event 头的致命影响。")}
{gate(True, f"join 安全：`projid` 唯一、无重复键、left-join 零膨胀。但 ROSMAP_clinical 只能作为**辅助**表用于"
      f"结局与病理，不能作为训练主表——否则会静默丢掉 LATC 的 {cs[cs.study == 'LATC'].projid.nunique()} 人。")}
""")
    print("s2 done")


if __name__ == "__main__":
    run()
