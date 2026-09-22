"""
compare_runs.py -- 把两次 figure2 run 的 panel a / panel b 指标并排列出来。

    python compare_runs.py results/figure2/gpubase results/figure2/nodedup \
        --names dedup nodedup --cohort matched

为什么需要它：`FIG2_TAG` 让多次 run 的输出各自成目录，但 metrics_*.json 是嵌套的，肉眼对读
八行 AUC × 两组 CI 很容易串行。这个脚本只做**对齐和排版**，不重算任何统计量。

两个必须一起读的限制：
  * **行集合可能不同。** panel a 的 at-risk 规则是"基线分期 != 该行"，而 `_baseline()` 要求
    至少两次不同访视 + 基线有认知分期。不去重那份分词让**每次**随访都发射认知 token，所以
    可评估人数和每行的 at-risk/events 都会变。脚本把 n 一起打出来，差异大的行标 `!`。
  * **单个 seed。** 两边都只有一次训练，AUC 的 CI 宽到 ±0.06~0.11，所以逐行差异小于 CI 宽度时
    不要当成效应。
"""
import os
import sys
import json
import argparse


def load(d, cohort):
    p = os.path.join(d, f"metrics_{cohort}.json")
    if not os.path.exists(p):
        sys.exit(f"没有 {p}")
    with open(p) as fh:
        return json.load(fh)


def fmt(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs=2, help="两个 run 的输出目录（results/figure2/<tag>）")
    ap.add_argument("--names", nargs=2, default=["A", "B"])
    ap.add_argument("--cohort", default="matched", choices=["matched", "allcohort"])
    a = ap.parse_args()
    (n1, n2), (m1, m2) = a.names, [load(d, a.cohort) for d in a.dirs]

    print(f"# figure2 对照  cohort={a.cohort}")
    print(f"  {n1:<10s} {a.dirs[0]}")
    print(f"  {n2:<10s} {a.dirs[1]}\n")

    # ---------------------------------------------------------------- panel a
    A1, A2 = m1["a"], m2["a"]
    print("## panel a — 5 年内到达各状态")
    print(f"  可评估人数（panel c 的 n_patients 代理）: {n1} {m1['c']['n_patients']} | "
          f"{n2} {m2['c']['n_patients']}")
    print(f"  median AUC          {n1} {fmt(A1['median_auc'])}   {n2} {fmt(A2['median_auc'])}")
    print(f"  mean calib error    {n1} {fmt(A1['mean_calibration_error'],4)}   "
          f"{n2} {fmt(A2['mean_calibration_error'],4)}")
    t1, t2 = A1.get("transition_time", {}), A2.get("transition_time", {})
    print(f"  转移时间 R²/MAE/bias {n1} {fmt(t1.get('r2'))}/{fmt(t1.get('mae'),2)}/"
          f"{fmt(t1.get('bias'),2)}   {n2} {fmt(t2.get('r2'))}/{fmt(t2.get('mae'),2)}/"
          f"{fmt(t2.get('bias'),2)}\n")

    rows = list(A1["per_state"]) + [k for k in A2["per_state"] if k not in A1["per_state"]]
    hdr = (f"{'状态':<26s} {'n@risk(' + n1 + '/' + n2 + ')':>16s} "
           f"{'events':>12s} {'AUC ' + n1:>12s} {'AUC ' + n2:>12s} {'Δ':>7s}  "
           f"{'实测/预测 ' + n1:>16s} {'实测/预测 ' + n2:>16s}")
    print(hdr)
    print("-" * len(hdr))
    for k in rows:
        p, q = A1["per_state"].get(k), A2["per_state"].get(k)
        if p is None or q is None:
            print(f"{k:<26s} {'只在一侧出现':>16s}")
            continue
        d = q["auc"] - p["auc"]
        # CI 宽度的一半作为"噪声尺度"；差异没超过它就不值得解读
        half = max(p["ci"][1] - p["ci"][0], q["ci"][1] - q["ci"][0]) / 2
        mark = "*" if abs(d) > half else " "
        nmark = "!" if abs(q["n_at_risk"] - p["n_at_risk"]) > 0.1 * p["n_at_risk"] else " "
        print(f"{k:<26s} {p['n_at_risk']:>7d}/{q['n_at_risk']:<7d}{nmark}"
              f"{p['n_events']:>5d}/{q['n_events']:<6d}"
              f"{p['auc']:>12.3f}{q['auc']:>13.3f}{d:>+8.3f}{mark} "
              f"{p['observed_cif']:>7.3f}/{p['predicted']:<8.3f}"
              f"{q['observed_cif']:>7.3f}/{q['predicted']:<8.3f}")
    print("  * = |ΔAUC| 超过两边 CI 半宽的较大者   ! = at-risk 人数差 >10%（行不同人，别直接对读）\n")

    # ---------------------------------------------------------------- panel b
    print("## panel b — 按 horizon 的 AUC")
    B1, B2 = m1["b"]["auc_by_horizon"], m2["b"]["auc_by_horizon"]
    hz = sorted({h for d in (B1, B2) for v in d.values() for h in v}, key=float)
    print(f"{'终点':<34s} {'run':<10s} " + "".join(f"{h + 'y':>8s}" for h in hz))
    for k in list(B1) + [x for x in B2 if x not in B1]:
        for nm, B in ((n1, B1), (n2, B2)):
            v = B.get(k, {})
            print(f"{k:<34s} {nm:<10s} " + "".join(f"{v.get(h, float('nan')):>8.3f}" for h in hz))
        print()

    print("## panel b — timing error (MAE 年 / bias 年，按真实首达时间分箱)")
    T1, T2 = m1["b"]["timing_error"], m2["b"]["timing_error"]
    for k in list(T1) + [x for x in T2 if x not in T1]:
        print(f"  {k}")
        bins = list(T1.get(k, {})) + [b for b in T2.get(k, {}) if b not in T1.get(k, {})]
        for b in bins:
            p, q = T1.get(k, {}).get(b), T2.get(k, {}).get(b)
            g = lambda x, f: "—" if x is None else f"{x[f]:.2f}"
            gn = lambda x: "—" if x is None else str(x["n"])
            print(f"    {b:<8s} n {gn(p):>4s}/{gn(q):<4s}  "
                  f"MAE {g(p,'mae'):>6s}/{g(q,'mae'):<6s}  "
                  f"bias {g(p,'bias'):>7s}/{g(q,'bias'):<7s}  "
                  f"(naive MAE {g(p,'naive_mae'):>6s}/{g(q,'naive_mae'):<6s})")
        print()


if __name__ == "__main__":
    main()
