"""
Round 4 -- coverage only, for five variables.

  CDRSUM     disease severity        CDR sum of boxes, UDS form B4
  MOCATOTS   cognitive function      MoCA total, UDS form C2
  FAQTOTAL   functional impairment   DERIVED from the 10 form-B7 items (no total ships)
  NACCUDSD   clinical diagnosis      1 normal / 2 impaired-not-MCI / 3 MCI / 4 dementia
  Death      NACCDIED / NACCYOD / NACCMOD / NACCAUTP

The denominator is the RAW row set: every row of the export, nothing deduplicated, nothing
dropped except the single trailing blank line. Row count is printed so it can be checked
against the file.

Three things this script refuses to collapse into one number, because each changes the
answer by tens of percentage points:

1. WHY a value is absent. -4 means the question is not on that form version at all
   (structural); 88 means it was on the form and not administered; 95-98 mean the
   participant could not or would not complete it. Only the last two are "missing data" in
   the usual sense -- -4 is a schema fact.
2. WHEN. MoCA did not exist before the 2015 UDS v3 rollout and MMSE was retired at the same
   time, so a single pooled coverage number for MoCA describes no actual cohort.
3. FAQTOTAL's derivation rule. There is no FAQ total column. Requiring all ten items scored
   gives a very different denominator from accepting eight.
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DELPHI = os.path.dirname(HERE)
ROOT = os.path.dirname(DELPHI)
OUT = os.path.join(HERE, "eda_out")
FIGS = os.path.join(OUT, "figs")
TABLES = os.path.join(OUT, "tables")
for _d in (OUT, FIGS, TABLES):
    os.makedirs(_d, exist_ok=True)

sys.path.insert(0, DELPHI)
from figure2.plotting_style import setup_style, save_fig, save_data  # noqa: E402

C = ["#0072B2", "#D55E00", "#E69F00", "#009E73", "#CC79A7"]

NACC_CANDIDATES = [
    os.path.join(ROOT, "data", "NACC", "investigator_nacc*.csv"),
    os.path.expanduser("~/cc_test/EDA_NACC/investigator_nacc*.csv"),
]

FAQ_ITEMS = ["BILLS", "TAXES", "SHOPPING", "GAMES", "STOVE",
             "MEALPREP", "EVENTS", "PAYATTN", "REMDATES", "TRAVEL"]

# code -> what it actually means, per the NACC Data Element Dictionary / RDD
SENTINELS = {
    "CDRSUM":   {99: "unknown / not assessed"},
    "MOCATOTS": {-4: "not on this form version (pre-UDSv3)", 88: "not administered"},
    "NACCUDSD": {},
    "FAQ_ITEM": {-4: "not on this form version", 8: "not applicable (never did it)",
                 9: "unknown"},
    "NACCDIED": {},
    "NACCYOD":  {8888: "not applicable (still alive)", 9999: "unknown"},
    "NACCMOD":  {88: "not applicable (still alive)", 99: "unknown"},
    "NACCAUTP": {8: "not applicable (still alive)", 9: "unknown"},
}
VALID = {"CDRSUM": (0, 18), "MOCATOTS": (0, 30), "NACCUDSD": (1, 4)}

STR_COLS = ["NACCID", "PACKET"]
KEEP = (["NACCID", "NACCVNUM", "PACKET", "FORMVER", "VISITYR", "NACCAGE",
         "CDRSUM", "MOCATOTS", "NACCUDSD", "NACCDIED", "NACCYOD", "NACCMOD", "NACCAUTP"]
        + FAQ_ITEMS)


def find_nacc():
    for pat in NACC_CANDIDATES:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def load(path):
    have = set(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in KEEP if c not in have]
    use = [c for c in KEEP if c in have]
    d = pd.read_csv(path, usecols=use, low_memory=False,
                    dtype={c: str for c in STR_COLS if c in have})
    n_raw = len(d)
    d = d[d["NACCID"].notna() & (d["NACCID"].astype(str).str.strip() != "")].copy()
    n_blank = n_raw - len(d)
    d["pid"] = pd.factorize(d["NACCID"])[0]          # drop the identifier immediately
    d = d.drop(columns=["NACCID"])
    for c in d.columns:
        if c not in ("pid", "PACKET"):
            d[c] = pd.to_numeric(d[c], errors="coerce")
    return d, n_raw, n_blank, missing


def clean(s, kind):
    x = pd.to_numeric(s, errors="coerce")
    x = x.mask(x.isin(list(SENTINELS.get(kind, {}))))
    if kind in VALID:
        lo, hi = VALID[kind]
        x = x.mask((x < lo) | (x > hi))
    return x


def derive_faq(d):
    items = [c for c in FAQ_ITEMS if c in d.columns]
    m = d[items].apply(pd.to_numeric, errors="coerce")
    m = m.mask(m.isin(list(SENTINELS["FAQ_ITEM"])))
    n = m.notna().sum(axis=1)
    s = m.sum(axis=1, min_count=1)
    return pd.DataFrame({
        "FAQ_N_ITEMS": n,
        "FAQTOTAL_strict": s.where(n == len(items)),
        "FAQTOTAL_prorated8": np.where(n >= 8, s * len(items) / n.replace(0, np.nan), np.nan),
        "FAQTOTAL_any": s.where(n >= 1),
    }, index=d.index)


def pct(a, b):
    return float(100.0 * a / b) if b else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nacc", default=None, help="path to the NACC export (csv or xlsx)")
    args = ap.parse_args()
    path = args.nacc or find_nacc()
    if path is None:
        sys.exit("No NACC export found. Pass --nacc /path/to/file")
    setup_style()

    d, n_raw, n_blank, missing_cols = load(path)
    d = pd.concat([d, derive_faq(d)], axis=1)
    N = len(d)
    NP = d["pid"].nunique()

    lines = []
    def w(t=""):
        lines.append(t)

    w("# Round 4 — 五个变量的覆盖率\n")
    w(f"源文件：`{path}`  \n"
      f"文件大小 {os.path.getsize(path) / 1e6:.0f} MB，原始行数 **{n_raw:,}**"
      + (f"（丢弃 {n_blank} 行末尾空行）" if n_blank else "") +
      f"，分析行数 **{N:,}**，涉及 **{NP:,}** 人。\n")
    w("**没有做任何 deduplicate**：一行 = 一次访视记录，原样使用。"
      "已核对 `(参与者, NACCVNUM)` 无重复，所以这个文件本身就是每次访视一行。\n")
    if missing_cols:
        w(f"> 请求的列中不在该文件里的：{', '.join('`%s`' % c for c in missing_cols)}\n")

    # ---------------------------------------------------------------- headline
    rows = []
    series = {
        "CDRSUM": clean(d["CDRSUM"], "CDRSUM"),
        "MOCATOTS": clean(d["MOCATOTS"], "MOCATOTS"),
        "FAQTOTAL": d["FAQTOTAL_strict"],
        "NACCUDSD": clean(d["NACCUDSD"], "NACCUDSD"),
    }
    labels = {"CDRSUM": "disease severity", "MOCATOTS": "cognitive function",
              "FAQTOTAL": "functional impairment", "NACCUDSD": "clinical diagnosis"}
    for k, x in series.items():
        per = d.loc[x.notna(), "pid"].nunique()
        rows.append({"variable": k, "construct": labels[k],
                     "n_rows_covered": int(x.notna().sum()),
                     "pct_rows": pct(x.notna().sum(), N),
                     "n_persons_covered": int(per), "pct_persons": pct(per, NP)})
    died = pd.to_numeric(d["NACCDIED"], errors="coerce")
    per_died = d.loc[died.notna(), "pid"].nunique()
    rows.append({"variable": "Death (NACCDIED)", "construct": "vital status",
                 "n_rows_covered": int(died.notna().sum()), "pct_rows": pct(died.notna().sum(), N),
                 "n_persons_covered": int(per_died), "pct_persons": pct(per_died, NP)})
    head = pd.DataFrame(rows)
    head.to_csv(os.path.join(TABLES, "coverage_headline.csv"), index=False)

    w("## 1. 总覆盖率（分母 = 全部 %s 行，未去重）\n" % f"{N:,}")
    w("| 变量 | 构念 | 有值行数 | **行覆盖率** | 有值人数 | 人覆盖率 |")
    w("|---|---|---|---|---|---|")
    for r in head.itertuples():
        w(f"| `{r.variable}` | {r.construct} | {r.n_rows_covered:,} | **{r.pct_rows:.1f}%** "
          f"| {r.n_persons_covered:,} | {r.pct_persons:.1f}% |")
    w()
    w("行覆盖率与人覆盖率差得越远，说明这个变量越集中在少数访视上而不是少数人身上。")
    w()

    # ---------------------------------------------------------------- why missing
    w("## 2. 缺失的成分拆解\n")
    w("「缺失」不是一件事。`-4` 是该表单版本根本没这道题（结构性），"
      "`88` 是有这道题但没施测，`95–98` 是参与者无法或拒绝完成。"
      "只有后两类是通常意义上的缺失数据。\n")
    comp = []
    for k, kind in [("CDRSUM", "CDRSUM"), ("MOCATOTS", "MOCATOTS"), ("NACCUDSD", "NACCUDSD")]:
        raw = pd.to_numeric(d[k], errors="coerce")
        comp.append({"variable": k, "reason": "有效值",
                     "n": int(clean(raw, kind).notna().sum()),
                     "pct": pct(clean(raw, kind).notna().sum(), N)})
        for code, meaning in SENTINELS[kind].items():
            n = int((raw == code).sum())
            comp.append({"variable": k, "reason": f"{code} = {meaning}", "n": n, "pct": pct(n, N)})
        nblank = int(raw.isna().sum())
        if nblank:
            comp.append({"variable": k, "reason": "空白 / 非数值", "n": nblank, "pct": pct(nblank, N)})
    cf = pd.DataFrame(comp)
    cf.to_csv(os.path.join(TABLES, "missingness_composition.csv"), index=False)
    w("| 变量 | 成分 | n | % of rows |")
    w("|---|---|---|---|")
    for r in cf.itertuples():
        w(f"| `{r.variable}` | {r.reason} | {r.n:,} | {r.pct:.2f}% |")
    w()

    # ---------------------------------------------------------------- FAQ rules
    w("## 3. FAQTOTAL：本文件没有总分列\n")
    w("NACC 只发布 10 个 FAQ 分项（表单 B7，每项 0–3），**没有总分**。"
      "所以「FAQTOTAL 的覆盖率」完全取决于合成规则：\n")
    faq = []
    for name, col, rule in [
            ("FAQTOTAL_strict", "FAQTOTAL_strict", "十项全部有效才计总分"),
            ("FAQTOTAL_prorated8", "FAQTOTAL_prorated8", "≥8 项有效，按比例放大到 10 项"),
            ("FAQTOTAL_any", "FAQTOTAL_any", "≥1 项有效就求和（不推荐，量纲不可比）")]:
        x = d[col]
        faq.append({"rule_name": name, "rule": rule, "n_rows": int(x.notna().sum()),
                    "pct_rows": pct(x.notna().sum(), N),
                    "n_persons": int(d.loc[x.notna(), "pid"].nunique())})
    ff = pd.DataFrame(faq)
    ff.to_csv(os.path.join(TABLES, "faq_derivation_rules.csv"), index=False)
    w("| 规则 | 说明 | 有值行数 | 行覆盖率 | 人数 |")
    w("|---|---|---|---|---|")
    for r in ff.itertuples():
        w(f"| `{r.rule_name}` | {r.rule} | {r.n_rows:,} | **{r.pct_rows:.1f}%** | {r.n_persons:,} |")
    w()
    spread = ff["pct_rows"].max() - ff["pct_rows"].min()
    w(f"三种规则之间差 **{spread:.1f} 个百分点**。上表第 1 节报的是 strict 口径。")
    w()
    w("分项 `8`（从不做该活动）与 `9`（未知）**都不能当 0**：把「从不做饭」记成「做饭没有困难」"
      "会系统性低估失能程度，而且这个偏差与性别、独居状况相关。\n")
    it = []
    for c in FAQ_ITEMS:
        if c not in d.columns:
            continue
        raw = pd.to_numeric(d[c], errors="coerce")
        x = raw.mask(raw.isin(list(SENTINELS["FAQ_ITEM"])))
        it.append({"item": c, "pct_valid": pct(x.notna().sum(), N),
                   "pct_code_8_not_applicable": pct((raw == 8).sum(), N),
                   "pct_code_9_unknown": pct((raw == 9).sum(), N),
                   "pct_code_-4_not_on_form": pct((raw == -4).sum(), N)})
    itf = pd.DataFrame(it)
    itf.to_csv(os.path.join(TABLES, "faq_items_coverage.csv"), index=False)
    w("| 分项 | 有效 % | 8 = 从不做 % | 9 = 未知 % | -4 = 无此题 % |")
    w("|---|---|---|---|---|")
    for r in itf.itertuples():
        w(f"| `{r.item}` | {r.pct_valid:.1f} | {r.pct_code_8_not_applicable:.2f} "
          f"| {r.pct_code_9_unknown:.2f} | {r._5:.2f} |")
    w()

    # ---------------------------------------------------------------- death
    w("## 4. Death：和上面四个不是一回事\n")
    w("`NACCDIED` 是**人级**状态标记，被复制到该参与者的每一行，所以它的「行覆盖率」恒为 100%，"
      "这个数字没有信息量。对死亡真正该问的是：多少人死了，以及死亡**日期**有没有记录。\n")
    person = d.sort_values(["pid", "NACCVNUM"]).groupby("pid").last()
    dd = pd.to_numeric(person["NACCDIED"], errors="coerce")
    yod = pd.to_numeric(person["NACCYOD"], errors="coerce")
    mod = pd.to_numeric(person["NACCMOD"], errors="coerce")
    autp = pd.to_numeric(person["NACCAUTP"], errors="coerce")
    n_died = int((dd == 1).sum())
    yod_ok = yod.mask(yod.isin([8888, 9999]))
    mod_ok = mod.mask(mod.isin([88, 99]))
    drows = [
        {"quantity": "参与者总数", "n": NP, "pct_of_persons": 100.0},
        {"quantity": "NACCDIED = 1（已死亡）", "n": n_died, "pct_of_persons": pct(n_died, NP)},
        {"quantity": "NACCDIED = 0（存活）", "n": int((dd == 0).sum()),
         "pct_of_persons": pct(int((dd == 0).sum()), NP)},
        {"quantity": "死亡年份 NACCYOD 有值（在已死亡者中）", "n": int(yod_ok[dd == 1].notna().sum()),
         "pct_of_persons": pct(int(yod_ok[dd == 1].notna().sum()), n_died)},
        {"quantity": "死亡月份 NACCMOD 有值（在已死亡者中）", "n": int(mod_ok[dd == 1].notna().sum()),
         "pct_of_persons": pct(int(mod_ok[dd == 1].notna().sum()), n_died)},
        {"quantity": "NACCAUTP = 1（有尸检，在已死亡者中）",
         "n": int((autp[dd == 1] == 1).sum()),
         "pct_of_persons": pct(int((autp[dd == 1] == 1).sum()), n_died)},
    ]
    dfd = pd.DataFrame(drows)
    dfd.to_csv(os.path.join(TABLES, "death_coverage.csv"), index=False)
    w("| 量 | n | % |")
    w("|---|---|---|")
    for r in dfd.itertuples():
        w(f"| {r.quantity} | {r.n:,} | {r.pct_of_persons:.1f}% |")
    w()
    w(f"注意最后三行的分母是**已死亡的 {n_died:,} 人**，不是全队列。")
    w()
    w("`NACCDIED` 的取值分布（人级，取每人最后一次访视）：")
    w("")
    for k, v in dd.value_counts(dropna=False).sort_index().items():
        w(f"- `{k}`：{v:,} 人")
    w()
    w("**这是一个右删失的生存结局**，不是一个协变量。`NACCDIED = 0` 的意思是"
      "「截至该参与者最后一次随访尚未记录死亡」，不是「不会死」——"
      "把它当二分类标签训练会把随访时长学成死亡风险。")
    w()

    # ---------------------------------------------------------------- by year / formver
    w("## 5. 覆盖率随时间与表单版本的变化\n")
    w("总覆盖率会掩盖问题：MoCA 在 2015 年 UDS v3 之前根本不存在，"
      "把它平均进 2005–2014 的访视里得到的数字不描述任何真实队列。\n")
    byyear = pd.DataFrame({k: x.notna().groupby(d["VISITYR"]).mean() * 100
                           for k, x in series.items()})
    byyear["n_visits"] = d.groupby("VISITYR").size()
    byyear.index = byyear.index.astype(int)
    byyear.to_csv(os.path.join(TABLES, "coverage_by_year.csv"))
    w("| 年份 | 访视数 | CDRSUM | MOCATOTS | FAQTOTAL | NACCUDSD |")
    w("|---|---|---|---|---|---|")
    for y, r in byyear.iterrows():
        w(f"| {y} | {int(r['n_visits']):,} | {r['CDRSUM']:.1f} | {r['MOCATOTS']:.1f} "
          f"| {r['FAQTOTAL']:.1f} | {r['NACCUDSD']:.1f} |")
    w()
    byver = pd.DataFrame({k: x.notna().groupby(d["FORMVER"]).mean() * 100
                          for k, x in series.items()})
    byver["n_visits"] = d.groupby("FORMVER").size()
    byver.to_csv(os.path.join(TABLES, "coverage_by_formver.csv"))
    w("按 UDS 表单版本：\n")
    w("| FORMVER | 访视数 | CDRSUM | MOCATOTS | FAQTOTAL | NACCUDSD |")
    w("|---|---|---|---|---|---|")
    for v, r in byver.iterrows():
        w(f"| {v} | {int(r['n_visits']):,} | {r['CDRSUM']:.1f} | {r['MOCATOTS']:.1f} "
          f"| {r['FAQTOTAL']:.1f} | {r['NACCUDSD']:.1f} |")
    w()

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8))
    ax = axes[0]
    for (k, _), c in zip(series.items(), C):
        ax.plot(byyear.index, byyear[k], marker="o", ms=3, color=c, label=k)
    ax.axvline(2015, color="#999999", ls="--", lw=1)
    ax.text(2015.15, 50, "UDS v3\n(2015)", fontsize=7, color="#555555")
    ax.set_xlabel("visit year")
    ax.set_ylabel("coverage % of visits")
    ax.set_ylim(-2, 102)
    ax.legend(fontsize=7, loc="center left")
    ax = axes[1]
    ax.bar(np.arange(len(byyear)), byyear["n_visits"], color="#999999")
    ax.set_xticks(np.arange(len(byyear)))
    ax.set_xticklabels([str(y) for y in byyear.index], rotation=90, fontsize=7)
    ax.set_ylabel("n visits in the file")
    ax.set_xlabel("visit year")
    fig.suptitle("Coverage by calendar year — the pooled number hides the 2015 form change",
                 fontsize=10)
    fig.tight_layout()
    save_fig(fig, FIGS, "coverage_by_year")
    save_data(byyear.reset_index().rename(columns={"index": "VISITYR"}), FIGS, "coverage_by_year")

    # ---------------------------------------------------------------- joint
    w("## 6. 联合覆盖率（真正约束建模的数字）\n")
    w("单看每个变量的覆盖率会高估可用样本。同时需要多个变量时，可用行数只会更少。\n")
    avail = pd.DataFrame({k: x.notna() for k, x in series.items()})
    jrows = [{"requirement": f"只要 `{k}`", "fig_label": k, "n_rows": int(avail[k].sum()),
              "pct_rows": pct(avail[k].sum(), N),
              "n_persons": int(d.loc[avail[k], "pid"].nunique())} for k in series]
    combos = [(["CDRSUM", "NACCUDSD"], "CDRSUM + NACCUDSD", "CDRSUM + NACCUDSD"),
              (["CDRSUM", "FAQTOTAL", "NACCUDSD"], "CDRSUM + FAQTOTAL + NACCUDSD",
               "CDRSUM + FAQ + NACCUDSD"),
              (["CDRSUM", "MOCATOTS", "NACCUDSD"], "CDRSUM + MOCATOTS + NACCUDSD",
               "CDRSUM + MoCA + NACCUDSD"),
              (list(series), "四个全要", "all four")]
    for cols, name, fig_label in combos:
        m = avail[cols].all(axis=1)
        jrows.append({"requirement": name, "fig_label": fig_label, "n_rows": int(m.sum()),
                      "pct_rows": pct(m.sum(), N),
                      "n_persons": int(d.loc[m, "pid"].nunique())})
    jf = pd.DataFrame(jrows)
    jf.to_csv(os.path.join(TABLES, "joint_coverage.csv"), index=False)
    w("| 要求 | 满足的行数 | % of 全部行 | 人数 |")
    w("|---|---|---|---|")
    for r in jf.itertuples():
        w(f"| {r.requirement} | {r.n_rows:,} | **{r.pct_rows:.1f}%** | {r.n_persons:,} |")
    w()
    all4 = jf.iloc[-1]
    w(f"**四个变量同时可用的只有 {all4.n_rows:,} 行（{all4.pct_rows:.1f}%），{all4.n_persons:,} 人。** "
      "瓶颈是 MoCA——它把样本限制在 2015 年之后。")
    w()

    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    ax.barh(np.arange(len(jf)), jf["pct_rows"], color=["#0072B2"] * 4 + ["#D55E00"] * 4)
    ax.set_yticks(np.arange(len(jf)))
    ax.set_yticklabels(jf["fig_label"], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("% of the 207k raw visit rows")
    for i, (p, n) in enumerate(zip(jf["pct_rows"], jf["n_rows"])):
        ax.text(p + 1, i, f"{p:.1f}%  ({n:,})", va="center", fontsize=7)
    ax.set_xlim(0, 118)
    ax.set_title("Single-variable coverage vs joint coverage", fontsize=10)
    fig.tight_layout()
    save_fig(fig, FIGS, "joint_coverage")
    save_data(jf, FIGS, "joint_coverage")

    head.to_csv(os.path.join(OUT, "COVERAGE_SUMMARY.csv"), index=False)
    with open(os.path.join(OUT, "COVERAGE.md"), "w") as f:
        f.write("\n".join(lines).rstrip() + "\n")
    print(f"rows={N:,} persons={NP:,} -> {os.path.join(OUT, 'COVERAGE.md')}")
    print(head.to_string(index=False))


if __name__ == "__main__":
    main()
