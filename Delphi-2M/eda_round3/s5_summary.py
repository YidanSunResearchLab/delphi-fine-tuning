"""Section 5 -- assemble SCALE_SUMMARY.csv, exactly the columns the spec's §5 table lists
(plus a few that the verdicts depend on, so a reader can audit a verdict without opening
another file). Variables that failed §1 get a row too, with the failure in `status`."""
import numpy as np
import pandas as pd

import s1_detect
import s2_generic
from common import SCALE_SUMMARY, VARS, md_table, report, summary_put

# spec §5 name  <-  internal name
SPEC_COLS = [
    ("varname", "varname"), ("dataset", "dataset"), ("n_obs", "n_obs"),
    ("pct_missing", "pct_missing"), ("n_distinct_values", "n_distinct_values"),
    ("n_rare_50", "n_rare_50"), ("n_rare_100", "n_rare_100"), ("n_rare_500", "n_rare_500"),
    ("normalized_entropy", "normalized_entropy"), ("max_category_pct", "max_category_pct"),
    ("pct_at_ceiling", "pct_at_ceiling"), ("pct_at_floor", "pct_at_floor"),
    ("icc", "icc"), ("rank_transition_rate", "rank_transition_rate"),
    ("delta_sd", "delta_sd"), ("noise_floor_sd", "noise_floor_sd"), ("snr", "snr"),
    ("ordinality_spearman", "rho_used_for_verdict"),
    ("isotonic_residual_ratio", "ord_isotonic_residual_ratio"),
    ("n_indistinguishable_adjacent_pairs", "ord_n_indistinguishable_adjacent_pairs"),
    ("direction", "direction_measured"), ("smoothing_verdict", "smoothing_verdict"),
    ("role_verdict", "role_verdict"),
]
EXTRA_COLS = [
    ("column_used", "column_used"), ("grain", "grain"), ("group", "group"),
    ("median_obs_per_person", "median_obs_per_person"),
    ("pct_persons_with_ge3_obs", "pct_persons_with_ge3_obs"),
    ("n_persons_with_obs", "n_persons_with_obs"),
    ("expected_range", "expected_range"), ("level_transform", "level_transform"),
    ("pct_offgrid_snapped", "pct_snapped"),
    ("ordinality_rho_source", "rho_source"),
    ("ordinality_spearman_literal", "ord_spearman"),
    ("ordinality_spearman_incident", "ord_incident_spearman"),
    ("isotonic_residual_ratio_incident", "ord_incident_isotonic_residual_ratio"),
    ("n_adjacent_pairs_tested", "ord_n_adjacent_pairs_tested"),
    ("n_indistinguishable_holm", "ord_n_indistinguishable_holm"),
    ("resolution_excess", "resolution_excess"),
    ("extreme_level_pile_pct", "extreme_level_pile_pct"),
    ("ceiling_test_vacuous", "ceiling_test_vacuous"),
    ("noise_floor_circular", "noise_floor_circular"),
    ("noise_floor_practice_shift", "noise_floor_practice_shift"),
    ("pct_delta_zero", "pct_delta_zero"),
    ("direction_expected", "direction_expected"),
    ("direction_matches_spec", "direction_matches_spec"),
    ("reference_outcome", "reference_outcome"),
    ("verdict_reasons", "verdict_reasons"),
]


def run():
    df, _ = s2_generic.rows_frame()
    out = pd.DataFrame(index=df.index)
    for spec_name, internal in SPEC_COLS + EXTRA_COLS:
        out[spec_name] = df[internal] if internal in df.columns else np.nan
    out["status"] = "ok"

    failed = [v for v in VARS if v.status != "ok"]
    if failed:
        rows = [{"varname": v.spec_name, "dataset": v.dataset, "grain": v.grain,
                 "group": v.group, "expected_range": v.expected_str(),
                 "column_used": v.column or "", "status": v.status,
                 "smoothing_verdict": "n/a", "role_verdict": "unavailable",
                 "verdict_reasons": "failed the §1 existence/encoding probe"}
                for v in failed]
        out = pd.concat([out, pd.DataFrame(rows)], ignore_index=True)

    out = out.sort_values(["dataset", "group", "varname"]).reset_index(drop=True)
    out.to_csv(SCALE_SUMMARY, index=False)

    report("## §5 汇总表\n")
    report(f"`SCALE_SUMMARY.csv`（{len(out)} 行 × {out.shape[1]} 列）已写出，"
           "列名与规格书 §5 表格一一对应，另附若干支撑列（`ordinality_rho_source`、"
           "`extreme_level_pile_pct`、`verdict_reasons` 等），"
           "让任何一条判定都能在不打开别的文件的情况下复核。")
    report("### 规格书 §5 要求的列（完整表在 CSV）\n")
    show = out[[c for c, _ in SPEC_COLS] + ["status"]]
    report(md_table(show, floatfmt="{:.3f}"))
    report("""几列的读法，避免误读：

- `ordinality_spearman` 取的是 **incident 口径**（若可算），列 `ordinality_rho_source` 标明来源。
  字面口径的值在 `ordinality_spearman_literal`。
- `pct_at_ceiling` / `pct_at_floor` 是**最高档 / 最低档**的占比，不是最大档的占比
  （后者是 `max_category_pct`）。
- `cogn_global` 的 `normalized_entropy` = 1.000、两端各 10%，是**十分位分箱的必然结果**，
  不是这个变量的性质；该行 `ceiling_test_vacuous` = True，§2.2/§2.3 的门对它不适用。
- `noise_floor_circular` = True 表示噪声底子集是**由该变量本身**定义的（`NACCUDSD`），
  Δ 恒为 0，SNR 未定义而非"很低"。""")

    counts = out["role_verdict"].value_counts()
    report("### 判定分布\n")
    report(md_table(pd.DataFrame({"role_verdict": counts.index, "n": counts.to_numpy()})))
    sm = out["smoothing_verdict"].value_counts()
    report(md_table(pd.DataFrame({"smoothing_verdict": sm.index, "n": sm.to_numpy()})))
    summary_put(s5_role_counts=counts.to_dict(), s5_smoothing_counts=sm.to_dict(),
                s5_rows=len(out))
    return out
