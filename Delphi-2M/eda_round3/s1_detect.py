"""Section 1 -- existence and encoding probe. NOTHING downstream may run on a variable
that fails here.

The spec's §1.4 table wants one row per variable. Two columns are split out of its
`illegal_values` because the judgement it demands ("sentinel code, or a real value that
proves the expected range wrong?") is exactly the distinction worth making explicit:

  sentinel_values_found   values we DECLARED as missing-data codes (common.Var.sentinels),
                          with counts -- these are removed, not treated as data
  illegal_values          values that survive sentinel removal yet sit off the declared
                          grid -- these are unexplained and gate the variable
"""
import numpy as np
import pandas as pd

from common import (CONTINUOUS, VARS, gate, md_table, nacc_panel, nacc_path,
                    radc_panels, report, save_table, summary_put, _num)


def probe(v, panel):
    """Locate v in a panel and describe what is actually there."""
    row = {"spec_name": v.spec_name, "dataset": v.dataset, "grain_expected": v.grain,
           "expected_range": v.expected_str(), "expected_direction": v.direction}
    if panel is None:
        row.update(varname="", status="panel_unavailable", range_matches=False,
                   n_nonmissing=0, pct_missing=np.nan, observed_min=np.nan,
                   observed_max=np.nan, observed_values="", illegal_values="",
                   sentinel_values_found="", name_matches_spec=False,
                   pct_offgrid=np.nan, offgrid_reason="")
        return row
    d = panel.df
    col = next((c for c in v.candidates if c in d.columns), None)
    if col is None:
        row.update(varname="", status="not_found", range_matches=False, n_nonmissing=0,
                   pct_missing=100.0, observed_min=np.nan, observed_max=np.nan,
                   observed_values="", illegal_values="", sentinel_values_found="",
                   name_matches_spec=False, pct_offgrid=np.nan, offgrid_reason="")
        return row

    raw = pd.to_numeric(d[col], errors="coerce")
    sent = raw[raw.isin(list(v.sentinels))] if v.sentinels else pd.Series(dtype=float)
    x = v.clean(d[col]).dropna()
    ill = v.illegal(d[col])
    vals = np.sort(x.unique())
    if len(vals) > 40:
        vshow = f"{len(vals)} distinct values, {_num(vals[0])}…{_num(vals[-1])}"
    else:
        vshow = ", ".join(_num(t) for t in vals)
    ill_counts = ill.value_counts().sort_index()
    ill_show = ", ".join(f"{_num(k)}×{int(n)}" for k, n in ill_counts.items()[:12]) \
        if len(ill_counts) else "none"
    if len(ill_counts) > 12:
        ill_show += f", … ({len(ill_counts)} distinct illegal values in all)"
    row.update(
        varname=col,
        name_matches_spec=bool(col == v.spec_name),
        observed_min=float(vals[0]) if len(vals) else np.nan,
        observed_max=float(vals[-1]) if len(vals) else np.nan,
        observed_values=vshow,
        n_distinct=int(len(vals)),
        sentinel_values_found=", ".join(f"{_num(k)}×{int(n)}" for k, n in
                                        sent.value_counts().sort_index().items()) or "none",
        illegal_values=ill_show,
        pct_offgrid=v.pct_offgrid(d[col]),
        offgrid_reason=(v.offgrid or ""),
        n_nonmissing=int(len(x)),
        pct_missing=float(100.0 * (1 - len(x) / len(d))),
    )
    if v.expected is CONTINUOUS:
        row["range_matches"] = True
    else:
        row["range_matches"] = bool(ill_counts.empty)
    row["status"] = "ok" if row["range_matches"] and len(x) > 0 else (
        "empty_after_sentinel_removal" if len(x) == 0 else "range_mismatch")
    return row


def get_panels():
    return {"nacc_visit": nacc_panel(), "radc_visit": radc_panels()[0],
            "radc_person": radc_panels()[1]}


def detect_only(panels=None):
    """Populate Var.column / Var.status without writing to the report.

    Lets any single section be run on its own; run_all() calls run() which supersedes this.
    """
    panels = panels or get_panels()
    for v in VARS:
        if v.status == "unprobed":
            r = probe(v, panels[v.panel])
            v.column = r["varname"] or None
            v.status = r["status"]
    return panels


def run():
    nacc = nacc_panel()
    rv, rp = radc_panels()
    panels = {"nacc_visit": nacc, "radc_visit": rv, "radc_person": rp}

    rows = []
    for v in VARS:
        p = panels[v.panel]
        r = probe(v, p)
        rows.append(r)
        v.column = r["varname"] or None
        v.status = r["status"]
    det = pd.DataFrame(rows)
    cols = ["spec_name", "varname", "dataset", "grain_expected", "expected_range",
            "expected_direction", "observed_min", "observed_max", "n_distinct",
            "observed_values", "sentinel_values_found", "illegal_values", "pct_offgrid",
            "offgrid_reason", "n_nonmissing", "pct_missing", "range_matches",
            "name_matches_spec", "status"]
    det = det.reindex(columns=cols)
    save_table(det, "variable_detection")

    ok = det[det["status"] == "ok"]
    bad = det[det["status"] != "ok"]
    renamed = det[(det["status"] == "ok") & (~det["name_matches_spec"].astype(bool))]

    report("## §1 变量清单与探测\n")
    report(f"NACC 源文件：`{nacc_path() or '未找到'}`"
           + (f"（{len(nacc.df):,} 次访视 / {nacc.df['pid'].nunique():,} 人，"
              f"{int(nacc.df['VISITYR'].min())}–{int(nacc.df['VISITYR'].max())}）" if nacc else "")
           + "\nRADC 源文件：`data/RADC/` 三张表"
           + f"（访视级 {len(rv.df):,} × {rv.df['projid'].nunique():,} 人；人级 {len(rp.df):,}）")
    report("**原始 NACC 导出文件不在本仓库内**（`.gitignore` 拦截 `**/investigator_nacc*.csv`）。"
           "载入时 `NACCID` 立刻被 factorize 成整数 `pid` 并丢弃，因此本轮所有缓存与 source-data "
           "CSV 都不含 NACC 标识符。路径可用环境变量 `NACC_CSV` 覆盖。")
    report(md_table(det[["spec_name", "varname", "dataset", "expected_range", "observed_min",
                        "observed_max", "n_distinct", "n_nonmissing", "pct_missing",
                        "status"]], floatfmt="{:.2f}"))

    report("### 哨兵码与非法值逐一判定\n")
    report(md_table(det.loc[det["varname"] != "",
                            ["spec_name", "varname", "sentinel_values_found",
                             "illegal_values", "pct_offgrid", "range_matches"]],
                    floatfmt="{:.2f}"))
    report("""判定结论（每个哨兵码都按 NACC Data Element Dictionary / RADC codebook 归因，不是猜的）：

- `NACCMMSE` 的 `-4 / 88 / 95 / 96 / 97 / 98`：-4 = 该表单版本无此题，88 = 未施测，
  95–98 = 因躯体问题 / 认知行为问题 / 其他问题 / 口头拒答而未完成。**全部是哨兵码，置为缺失。**
  注意 95–98 不是"分数很低"，把它们当数值会在量表顶端凭空造出一批伪高分。
- `MOCATOTS` 的 `-4 / 88`：同上（-4 = UDS v3 之前不存在此题）。**哨兵码。**
- `CDRSUM` / `CDRGLOB` / 六个 box 分的 `99`：未知/未评。**哨兵码**，各 19 次，量级可忽略。
- `NACCGDS` 的 `-4 / 88`：88 = 缺项过多无法计算总分。**哨兵码。**
- FAQ 十个分项的 `8 / 9 / -4`：8 = 从不做该活动（not applicable），9 = 未知。
  **两者都不是 0**——把"从不做"记成"没有困难"会系统性低估 FAQ 总分。
- 去掉哨兵码后，**没有任何变量残留无法解释的非法值**（`illegal_values` 全为 none）。
- **一个例外必须单独判定**：`cts_estmmse30` 有 1.55% 的取值落在申报的整数格之间
  （`pct_offgrid` 列）。按规格书 §1.4 的要求逐一判定后，这些**不是哨兵码，是真实值**——
  codebook p.230 说该列可以是由 MoCA 换算出的估计分，换算结果本来就不是整数。
  所以**是规格书表格里「0–30 整数」这个预期错了**，需要更新文档；本轮把它记为
  `offgrid_reason` 并放行，档位分析时就近吸附并单独报吸附比例（§2、§3.1）。""")

    report("### 与规格书 §1 表格不符之处\n")
    report("""| 规格书写的 | 数据里实际的 | 性质 | 处理 |
|---|---|---|---|
| `cts_mmse30`（RADC，0–30 整数） | `cts_estmmse30` | **列名不同，且语义不同**：codebook p.230 明确说这是"MMSE 分数，或由 MoCA 换算出的估计 MMSE 分数"，1.55% 的取值不是整数 | 继续分析，但 §3.1 专门量化插补比例；任何"这是原始 MMSE"的表述都是错的 |
| `dcfdx`（RADC，逐次访视 1–6） | `dcfdx_lv` | **粒度不同**：本 release 没有逐次访视诊断，只有"最后一次有效访视"的人级诊断 | §3.8 的序数性检验在**人级**做，参照结局换成尸检 Braak（见下）；`dcfdx` 的逐次访视版本无法分析 |
| `niareagansc`（RADC，1–4） | — | **完全不存在**：codebook 里没有这个变量，三张表里也没有 | 剔除。病理结局只剩 Braak 与 CERAD |
| `NACCUDSD` 另查取值 9 | 只有 1/2/3/4 | 本 release（v72）没有 9 | 无需处理，但 tokenizer 不应为 9 预留档位 |
| FAQ 总分（NACC） | 没有总分列 | 只有 10 个分项 | 本轮**自行合成** `FAQTOTAL`（十项全有才计总分）与 `FAQTOTAL_PRORATED`（≥8 项按比例放大），§3.5 报两者覆盖率差 |
| `cogn_global 等`（RADC 五个认知域分） | 只有 `cogn_global` | 本 release 不含 `cogn_ep/se/wo/ps` 五个域分与 `cts_*` 单项测验 | 只分析 `cogn_global` |""")

    report(f"""### 参照结局（§2.6 用什么当"下一次访视是否为痴呆"）

三个 panel 的参照结局必须分开说明，因为它们不是同一个东西：

| panel | 参照结局 | 说明 |
|---|---|---|
| `nacc_visit` | `NACCUDSD = 4` at the next visit | 规格书要求的定义，直接可得 |
| `radc_visit` | 下一个观测周期已达 AD 痴呆 | 本 release 无逐次访视诊断，由人级 `age_first_ad_dx` 还原（与 round 2 同一代理） |
| `radc_person` | 尸检 `braaksc >= 4` | `dcfdx_lv`/`cogdx` 是人级"最后一次"诊断，没有"下一次"。用病理当参照**同时避免了循环**：临床诊断用病理检验，病理反过来用临床诊断（`dcfdx_lv >= 4`）检验 |

同时刻还额外算了一个 **incident 变体**（只保留 t 时刻尚未痴呆的行）。理由：规格书的字面定义
包含了 t 时刻已经痴呆的行，对严重度量表来说那一半测的是"痴呆会不会持续"而不是"会不会发生"，
两者在量表顶端会给出完全不同的曲线。**字面版本用于判定**（规格书要求的就是它），incident
版本并列报告；§6 会指出两者不一致的地方。""")

    report(gate(len(bad) == 0,
                f"{len(det)} 个变量中 {len(ok)} 个 status=ok；"
                + (f"**{len(bad)} 个不合格：**"
                   + "、".join(f"`{r.spec_name}`({r.status})" for r in bad.itertuples())
                   + "，这些变量不进入 §2 之后的分析。"
                   if len(bad) else "全部通过。")))
    report(gate(len(renamed) == 0,
                f"{len(renamed)} 个变量的实际列名与规格书不同"
                + (f"（{'、'.join('`%s`→`%s`' % (r.spec_name, r.varname) for r in renamed.itertuples())}）"
                   "，其中 `cts_mmse30→cts_estmmse30` 与 `dcfdx→dcfdx_lv` 不只是改名，"
                   "语义/粒度都变了，已在上表登记。" if len(renamed) else "。")))

    summary_put(s1_n_vars=len(det), s1_n_ok=len(ok),
                s1_failed={r.spec_name: r.status for r in bad.itertuples()},
                s1_renamed={r.spec_name: r.varname for r in renamed.itertuples()},
                s1_nacc_visits=int(len(nacc.df)) if nacc else 0,
                s1_nacc_persons=int(nacc.df["pid"].nunique()) if nacc else 0,
                s1_radc_visits=int(len(rv.df)), s1_radc_persons=int(rp.df["projid"].nunique()))
    return det, panels
